"""CivicLens FastAPI app: /health, /ask (plain JSON or SSE token streaming), /examples."""

import json
import logging
from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path
from typing import Any

import psycopg
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agents import graph
from common.llm import llm_description
from common.settings import get_settings
from retrieval.search import SearchFilters

logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).resolve().parents[1] / "evals" / "golden_dataset.json"
MAX_EXAMPLES = 8

# Demo questions about the Mesa 2026-04-06 sample meeting (used when no golden dataset).
FALLBACK_EXAMPLES: tuple[str, ...] = (
    "Who was excused from the Mesa city council meeting on April 6, 2026?",
    "What items are on the consent agenda for the April 6 Mesa meeting?",
    "How many agenda items were there in the Mesa 2026-04-06 meeting?",
    "What awards were announced at the Mesa city council meeting?",
    "When is the next Mesa city council meeting?",
)

app = FastAPI(title="CivicLens", version="0.1.0")

from api.voice_ws import router as voice_router  # noqa: E402

app.include_router(voice_router)
app.add_middleware(  # local demo: UI is served from another origin
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    """Body for POST /ask; the filter fields mirror retrieval.search.SearchFilters."""

    question: str = Field(min_length=3)
    city: str | None = None
    source_type: str | None = None
    topic: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    stream: bool = True


def build_filters(request: AskRequest) -> SearchFilters:
    """Build retrieval SearchFilters from the request's metadata fields."""
    return SearchFilters(
        city=request.city,
        source_type=request.source_type,
        topic=request.topic,
        date_from=request.date_from,
        date_to=request.date_to,
    )


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    db_ok = False
    pgvector_version: str | None = None
    try:
        with psycopg.connect(settings.database_url, connect_timeout=2) as conn:
            row = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            pgvector_version = row[0] if row else None
            db_ok = True
    except psycopg.Error as exc:  # db down → degrade, never crash
        logger.warning("health check: database unreachable (%s)", exc)
    return {
        "status": "ok" if db_ok else "degraded",
        "db": db_ok,
        "pgvector": pgvector_version,
        "llm_backend": llm_description(),
    }


@app.post("/ask")
async def ask(request: AskRequest) -> Response:
    """Run the multi-agent pipeline; SSE token stream by default, plain JSON on demand."""
    filters = build_filters(request)
    if request.stream:
        return EventSourceResponse(_sse_events(request.question, filters))
    try:
        result = await run_in_threadpool(graph.ask, request.question, filters)
    except Exception as exc:
        logger.exception("ask pipeline failed")
        return JSONResponse({"type": "error", "message": str(exc)}, status_code=500)
    return JSONResponse(result.to_dict())


async def _sse_events(question: str, filters: SearchFilters) -> AsyncIterator[dict[str, str]]:
    """Adapt agents.graph.ask_stream events to SSE frames; errors become a clean event."""
    try:
        async for event in graph.ask_stream(question, filters):
            yield {
                "event": str(event.get("type", "message")),
                "data": json.dumps(event, default=str),
            }
    except Exception as exc:  # no tracebacks to clients
        logger.exception("streaming ask pipeline failed")
        yield {"event": "error", "data": json.dumps({"type": "error", "message": str(exc)})}


@app.get("/examples")
def examples() -> dict:
    """Example questions: stratified sample of the golden dataset, else a canned list."""
    questions = _golden_questions(GOLDEN_DATASET_PATH) or list(FALLBACK_EXAMPLES)
    return {"examples": [{"question": q} for q in questions]}


def _golden_questions(path: Path) -> list[str]:
    """Questions from evals/golden_dataset.json, or [] when absent/malformed."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("could not load golden dataset %s: %s", path, exc)
        return []
    if not isinstance(data, list):
        return []
    rows = [row for row in data if isinstance(row, dict)]
    # every example shown must be answerable from what is actually ingested: a fresh
    # clone has only the bundled sample meeting, so restrict to sample questions
    # unless the live corpus is present (sample-first ordering either way)
    if not _live_corpus_present():
        rows = [row for row in rows if row.get("sample", False)]
    rows = sorted(rows, key=lambda row: not row.get("sample", False))
    return _stratified_questions(rows, limit=MAX_EXAMPLES)


def _live_corpus_present() -> bool:
    """True when non-sample sources are ingested; False on any doubt (DB down)."""
    try:
        with psycopg.connect(get_settings().database_url, connect_timeout=2) as conn:
            row = conn.execute(
                "SELECT count(*) FROM sources WHERE metadata->>'sample' IS DISTINCT FROM 'true'"
            ).fetchone()
            return bool(row and row[0])
    except psycopg.Error:
        return False


def _stratified_questions(rows: list[Any], limit: int) -> list[str]:
    """Up to `limit` questions, round-robin across the source_type values present."""
    groups: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            continue
        source_type = row.get("source_type")
        key = source_type if isinstance(source_type, str) else "unknown"
        groups.setdefault(key, []).append(question)
    picked: list[str] = []
    while len(picked) < limit and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(picked) < limit:
                picked.append(groups[key].pop(0))
    return picked
