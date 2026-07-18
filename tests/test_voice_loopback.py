"""Phase 7 acceptance: a Piper-spoken question through the full /voice WebSocket.

No human speech needed: Piper synthesizes the test question, faster-whisper transcribes
it back, the agent pipeline answers, and Piper speaks the answer. Requires the stack
(Postgres + Ollama) and downloads the whisper/piper models on first run — marked slow.
"""

import json
import re
import time

import httpx
import pytest

from common.settings import get_settings

pytestmark = [pytest.mark.slow]

QUESTION = "Which council member was excused from the April sixth Mesa city council meeting?"
EXPECTED_TOKENS = {"council", "member", "excused", "april", "mesa", "meeting"}
TURN_TIMEOUT_S = 300


@pytest.fixture(scope="module")
def voice_ready(db_conn):
    settings = get_settings()
    try:
        httpx.get(f"{settings.ollama_base_url}/api/version", timeout=3)
    except httpx.HTTPError:
        pytest.skip("ollama is not reachable")
    from retrieval.index import embed_pending_chunks

    embed_pending_chunks()


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def test_voice_loopback(voice_ready) -> None:
    from fastapi.testclient import TestClient

    from agents.evidence import CITATION_RE, NOT_FOUND
    from api.main import app
    from voice.metrics import LATENCY_PATH, METRIC_KEYS, TURNS_PATH
    from voice.tts import synthesize_wav_bytes

    wav = synthesize_wav_bytes(QUESTION)
    assert wav.startswith(b"RIFF"), "piper produced invalid audio"

    turns_before = TURNS_PATH.read_text().count("\n") if TURNS_PATH.exists() else 0

    events: list[dict] = []
    audio_frames: list[bytes] = []
    client = TestClient(app)
    with client.websocket_connect("/voice") as ws:
        ws.send_text(json.dumps({"type": "start", "format": "wav", "city": "mesa"}))
        third = len(wav) // 3
        ws.send_bytes(wav[:third])
        ws.send_bytes(wav[third : 2 * third])
        ws.send_bytes(wav[2 * third :])
        ws.send_text(json.dumps({"type": "end_of_speech"}))

        deadline = time.monotonic() + TURN_TIMEOUT_S
        got_metrics = False
        while time.monotonic() < deadline and not got_metrics:
            message = ws.receive()
            if message.get("bytes") is not None:
                audio_frames.append(message["bytes"])
                continue
            if message.get("text") is None:
                continue
            event = json.loads(message["text"])
            events.append(event)
            if event.get("type") == "error":
                pytest.fail(f"pipeline error: {event.get('message')}")
            if event.get("type") == "metrics":
                got_metrics = True

    by_type = {e["type"]: e for e in events}

    # 1) transcript accuracy above threshold
    assert "final_transcript" in by_type, f"no final transcript; events: {by_type.keys()}"
    transcript_tokens = _tokens(by_type["final_transcript"]["text"])
    overlap = len(EXPECTED_TOKENS & transcript_tokens) / len(EXPECTED_TOKENS)
    assert overlap >= 0.6, f"transcript too lossy ({overlap:.2f}): {by_type['final_transcript']}"

    # 2) a cited answer is produced
    assert "result" in by_type, "no result event"
    answer = by_type["result"]["data"].get("answer", "")
    assert answer.strip()
    assert CITATION_RE.search(answer) or answer.strip() == NOT_FOUND

    # 3) audio: valid WAVs when TTS works on this platform; otherwise the turn must
    #    degrade cleanly (a text answer + an explicit unavailable signal, never a crash)
    from voice.tts import tts_available

    if tts_available():
        assert audio_frames, "no audio frames streamed back"
        assert all(frame.startswith(b"RIFF") for frame in audio_frames)
    else:
        statuses = [e for e in events if e.get("type") == "status"]
        assert any(s.get("tts_available") is False for s in statuses), (
            "TTS unavailable but the client was not told"
        )

    # 4) all latency metrics recorded (ttfa is null when no audio was produced)
    metrics = by_type["metrics"]
    required = METRIC_KEYS if tts_available() else ("asr_final_ms", "ttft_ms", "total_ms")
    for key in required:
        assert isinstance(metrics.get(key), int | float), f"metric {key} missing: {metrics}"

    # 5) persisted: a new turn row and an aggregated percentile file
    assert TURNS_PATH.exists()
    assert TURNS_PATH.read_text().count("\n") == turns_before + 1
    latency = json.loads(LATENCY_PATH.read_text())
    gate_metric = "ttfa_ms" if tts_available() else "total_ms"
    assert latency["metrics"].get(gate_metric, {}).get("n", 0) >= 1
