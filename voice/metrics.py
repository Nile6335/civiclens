"""Voice-turn latency instrumentation: per-turn records, percentiles, Langfuse spans.

The measurement discipline is the point: every voice turn logs ASR finalization lag,
time-to-first-token, time-to-first-audio, and total turn time; `aggregate()` reduces
the log to p50/p95 for evals/results/voice_latency.json.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "evals" / "results"
TURNS_PATH = RESULTS_DIR / "voice_turns.jsonl"
LATENCY_PATH = RESULTS_DIR / "voice_latency.json"

METRIC_KEYS = ("asr_final_ms", "ttft_ms", "ttfa_ms", "total_ms")


@dataclass
class VoiceTurn:
    """One spoken question → spoken answer turn. Times are ms from end_of_speech."""

    transcript: str = ""
    asr_final_ms: float | None = None
    ttft_ms: float | None = None
    ttfa_ms: float | None = None
    total_ms: float | None = None
    n_sentences: int = 0
    n_audio_bytes: int = 0
    _t0: float = field(default=0.0, repr=False)

    def start_clock(self) -> None:
        self._t0 = time.monotonic()

    def mark(self, key: str) -> float:
        """Record elapsed-ms for one METRIC_KEYS entry (first call wins); returns it."""
        elapsed = (time.monotonic() - self._t0) * 1000.0
        if getattr(self, key) is None:
            setattr(self, key, round(elapsed, 1))
        return elapsed

    def to_event(self) -> dict:
        return {k: getattr(self, k) for k in METRIC_KEYS}


def record_turn(turn: VoiceTurn, path: Path = TURNS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in asdict(turn).items() if not k.startswith("_")}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(int(round(pct / 100 * (len(ordered) - 1))), len(ordered) - 1)
    return round(ordered[idx], 1)


def aggregate(turns_path: Path = TURNS_PATH, out_path: Path = LATENCY_PATH) -> dict:
    """Reduce the turn log to per-metric p50/p95 (+n); write voice_latency.json."""
    rows = []
    if turns_path.exists():
        for line in turns_path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    summary: dict = {"n_turns": len(rows), "target_ttfa_ms": 3000, "metrics": {}}
    for key in METRIC_KEYS:
        values = [r[key] for r in rows if isinstance(r.get(key), int | float)]
        if values:
            summary["metrics"][key] = {
                "p50": _percentile(values, 50),
                "p95": _percentile(values, 95),
                "n": len(values),
            }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def trace_turn(turn: VoiceTurn) -> None:
    """Best-effort Langfuse trace for one voice turn (no-op when unconfigured)."""
    try:
        from agents.graph import _langfuse_handler

        handler = _langfuse_handler()
        if handler is None:
            return
        client = handler.langfuse
        trace = client.trace(name="voice-turn", input=turn.transcript)
        for key in METRIC_KEYS:
            value = getattr(turn, key)
            if value is not None:
                trace.event(name=key, metadata={"ms": value})
    except Exception as exc:
        logger.debug("langfuse voice trace skipped: %s", exc)
