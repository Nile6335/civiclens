"""The /voice WebSocket: push-to-talk audio in, cited spoken answer out.

Turn flow (see voice/protocol.py for the wire format):
  audio frames → rolling partial transcripts → finalize on end_of_speech →
  the existing LangGraph pipeline streams tokens → sentences are flushed to Piper as
  they complete → WAV frames stream back before the full answer exists.

Every turn logs asr_final/ttft/ttfa/total (ms from end_of_speech) to
evals/results/voice_turns.jsonl, refreshes voice_latency.json, and traces to Langfuse.
"""

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from starlette.websockets import WebSocketState

from voice import protocol
from voice.metrics import VoiceTurn, aggregate, record_turn, trace_turn

logger = logging.getLogger(__name__)

router = APIRouter()

VOICE_PAGE = Path(__file__).resolve().parent / "static" / "voice.html"
PARTIAL_MIN_INTERVAL_S = 1.0


@router.get("/voice-demo")
def voice_demo() -> FileResponse:
    return FileResponse(VOICE_PAGE, media_type="text/html")


async def _send_json(ws: WebSocket, payload: dict) -> None:
    if ws.client_state == WebSocketState.CONNECTED:
        await ws.send_text(json.dumps(payload, default=str))


def _build_filters(opts: dict):
    from retrieval.search import SearchFilters

    return SearchFilters(city=opts.get("city") or None, source_type=opts.get("source_type") or None)


@router.websocket("/voice")
async def voice_ws(ws: WebSocket) -> None:
    await ws.accept()
    from voice.asr_stream import StreamingTranscriber

    transcriber = StreamingTranscriber()
    opts: dict = {}
    partial_task: asyncio.Task | None = None
    try:
        # ---- receive phase: audio frames until end_of_speech
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                return
            if (data := message.get("bytes")) is not None:
                transcriber.accept_audio(data)
                if partial_task is None or partial_task.done():
                    partial_task = asyncio.create_task(_emit_partial(ws, transcriber))
                continue
            if (text := message.get("text")) is None:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                continue
            if event.get("type") == protocol.CLIENT_START:
                opts = event
            elif event.get("type") == protocol.CLIENT_END_OF_SPEECH:
                break

        if partial_task and not partial_task.done():
            partial_task.cancel()

        turn = VoiceTurn()
        turn.start_clock()

        transcript = (await asyncio.to_thread(transcriber.finalize)).strip()
        turn.mark("asr_final_ms")
        turn.transcript = transcript
        await _send_json(ws, {"type": protocol.SERVER_FINAL_TRANSCRIPT, "text": transcript})
        if not transcript:
            await _send_json(
                ws,
                {"type": protocol.SERVER_ERROR, "message": "no speech detected"},
            )
            return

        await _run_answer_phase(ws, transcript, opts, turn)

        turn.mark("total_ms")
        await _send_json(ws, {"type": protocol.SERVER_METRICS, **turn.to_event()})
        record_turn(turn)
        aggregate()
        trace_turn(turn)
    except WebSocketDisconnect:
        logger.info("voice client disconnected")
    except Exception as exc:
        logger.exception("voice turn failed")
        await _send_json(
            ws, {"type": protocol.SERVER_ERROR, "message": f"voice pipeline error: {exc}"}
        )


async def _emit_partial(ws: WebSocket, transcriber) -> None:
    """Cheap rolling partial transcript between frames (throttled, best-effort)."""
    try:
        if not transcriber.has_new_audio_since_partial():
            return
        text = await asyncio.to_thread(transcriber.partial)
        if text:
            await _send_json(ws, {"type": protocol.SERVER_PARTIAL_TRANSCRIPT, "text": text})
        await asyncio.sleep(PARTIAL_MIN_INTERVAL_S)
    except Exception as exc:  # partials are advisory; never kill the turn
        logger.debug("partial transcript skipped: %s", exc)


async def _run_answer_phase(ws: WebSocket, transcript: str, opts: dict, turn: VoiceTurn) -> None:
    """Stream the agent pipeline; pipeline sentences into TTS as they complete."""
    from agents.graph import ask_stream
    from voice.tts import SentenceStreamer, synthesize_wav_bytes

    streamer = SentenceStreamer()

    async def speak(sentence: str) -> None:
        wav = await asyncio.to_thread(synthesize_wav_bytes, sentence)
        if not wav:
            return
        turn.mark("ttfa_ms")
        turn.n_sentences += 1
        turn.n_audio_bytes += len(wav)
        await _send_json(ws, {"type": protocol.SERVER_SENTENCE, "text": sentence})
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.send_bytes(wav)

    # fast_route: voice turns skip the serial LLM router (latency optimization)
    async for event in ask_stream(transcript, _build_filters(opts), fast_route=True):
        kind = event.get("type")
        if kind == "status":
            await _send_json(ws, {**event, "type": protocol.SERVER_STATUS})
        elif kind == "token":
            turn.mark("ttft_ms")
            await _send_json(ws, {"type": protocol.SERVER_TOKEN, "text": event.get("text", "")})
            for sentence in streamer.feed(str(event.get("text", ""))):
                await speak(sentence)
        elif kind == "result":
            for sentence in streamer.flush():
                await speak(sentence)
            data = event.get("data", {})
            if turn.n_sentences == 0 and data.get("answer"):
                # model produced no streamable tokens (e.g. extractive fallback):
                # speak the final answer so the turn always has audio
                from voice.tts import split_sentences

                for sentence in split_sentences(str(data["answer"])):
                    await speak(sentence)
            await _send_json(ws, {"type": protocol.SERVER_RESULT, "data": data})
