"""Voice latency benchmark: warm the models, then measure N steady-state turns.

    python -m voice.bench [--turns 5]

Drives the real /voice WebSocket in-process with Piper-synthesized questions (no human
speech, no browser). Discards the first (cold) turn from percentiles and reports p50/p95
for ASR-final, time-to-first-token, time-to-first-audio, and total turn time to
evals/results/voice_latency.json. The target is TTFA < 3s on CPU for cached-index
queries; on a small CPU box this is not met (local-LLM synthesis dominates) — the point
is the honest, instrumented measurement.
"""

import argparse
import json
import logging

from voice.metrics import LATENCY_PATH, TURNS_PATH, aggregate

logger = logging.getLogger(__name__)

BENCH_QUESTIONS = [
    "Which council member was excused from the April sixth Mesa city council meeting?",
    "What awards were announced at the Mesa city council meeting?",
    "How many agenda items were on the Mesa council agenda?",
    "What was discussed about sustainability at the Mesa meeting?",
    "Who gave the invocation at the Mesa city council meeting?",
]


def _reset_loop_bound_caches() -> None:
    """TestClient opens a fresh event loop per connection; the cached async Ollama/
    Langfuse HTTP clients are bound to the previous loop and raise 'Event loop is
    closed' on reuse. Clearing the caches forces a fresh client on the current loop.
    (The real uvicorn server runs one persistent loop, so this is a bench-harness
    concern, not a product bug.)"""
    from common.llm import get_chat_model

    get_chat_model.cache_clear()
    try:
        from agents.graph import _langfuse_handler

        _langfuse_handler.cache_clear()
    except Exception:
        pass


def _run_turn(client, wav: bytes) -> dict:
    events: dict = {}
    _reset_loop_bound_caches()
    with client.websocket_connect("/voice") as ws:
        ws.send_text(json.dumps({"type": "start", "format": "wav", "city": "mesa"}))
        ws.send_bytes(wav)
        ws.send_text(json.dumps({"type": "end_of_speech"}))
        while True:
            message = ws.receive()
            if message.get("bytes") is not None:
                continue
            if message.get("text") is None:
                continue
            event = json.loads(message["text"])
            if event.get("type") == "metrics":
                events = event
                break
            if event.get("type") == "error":
                logger.warning("turn error: %s", event.get("message"))
                break
    return events


def run_benchmark(turns: int = 5) -> dict:
    from fastapi.testclient import TestClient

    from agents.graph import warmup
    from api.main import app
    from voice.tts import synthesize_wav_bytes

    # start each benchmark from a clean turn log so percentiles reflect THIS run
    if TURNS_PATH.exists():
        TURNS_PATH.unlink()

    logger.info("warming models...")
    warmup()

    client = TestClient(app)
    # one discarded cold turn (first request still pays some lazy init)
    logger.info("cold turn (discarded)...")
    _run_turn(client, synthesize_wav_bytes(BENCH_QUESTIONS[0]))
    if TURNS_PATH.exists():
        TURNS_PATH.unlink()

    for i in range(turns):
        question = BENCH_QUESTIONS[i % len(BENCH_QUESTIONS)]
        logger.info("turn %d/%d: %s", i + 1, turns, question[:50])
        _run_turn(client, synthesize_wav_bytes(question))

    summary = aggregate()
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=5)
    args = parser.parse_args()
    summary = run_benchmark(args.turns)
    print(
        f"\nvoice latency over {summary['n_turns']} warmed turns "
        f"(target TTFA {summary['target_ttfa_ms']}ms):"
    )
    for key, stats in summary["metrics"].items():
        print(f"  {key:<14} p50={stats['p50']:>8.0f}ms  p95={stats['p95']:>8.0f}ms")
    print(f"written to {LATENCY_PATH}")


if __name__ == "__main__":
    main()
