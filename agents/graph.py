"""LangGraph multi-agent pipeline: supervisor → {transcript, document, tabular} → synthesis.

Design notes for small local models:
- The supervisor asks the LLM for a JSON route list but falls back to (and sanity-checks
  against) a keyword heuristic — an invalid/empty LLM answer never breaks routing.
- Synthesis cites evidence via [E#] markers that are deterministically resolved to
  canonical citations (agents/evidence.py); if the model produces uncited claims, we fall
  back to an extractive answer built from the evidence itself. "Not found in the record."
  is returned when there is no usable evidence.
"""

import json
import logging
import re
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.evidence import NOT_FOUND, Evidence, resolve_markers, sentences_without_citation
from agents.state import ROUTES, AskResult, AskState, result_from_state
from common.llm import get_chat_model
from common.settings import get_settings
from retrieval.search import SearchFilters, hybrid_search

logger = logging.getLogger(__name__)

MAX_EVIDENCE = 8
EVIDENCE_CHAR_CAP = 600

# ---------------------------------------------------------------- supervisor

_TABULAR_CUES = re.compile(
    r"\b(how many|count|number of|total|average|sum|list all|which items|items were|table)\b",
    re.IGNORECASE,
)
_DOCUMENT_CUES = re.compile(
    r"\b(agenda|document|packet|page|pdf|report|resolution|ordinance|minutes|written)\b",
    re.IGNORECASE,
)
_TRANSCRIPT_CUES = re.compile(
    r"\b(say|said|discuss|discussed|mention|spoke|speak|comment|meeting video|talk|"
    r"who was|excused|announce)\b",
    re.IGNORECASE,
)

_ROUTER_PROMPT = """You route questions about city-council meetings to data sources.
Sources:
- "transcript": what was said in the meeting video
- "document": agenda/packet PDFs (written record)
- "tabular": structured tables (agenda items, budgets, votes) for counting/aggregation

Question: {question}

Reply with ONLY a JSON array of the sources needed, e.g. ["transcript","document"]."""


def route_heuristic(question: str) -> list[str]:
    routes = []
    if _TRANSCRIPT_CUES.search(question):
        routes.append("transcript")
    if _DOCUMENT_CUES.search(question):
        routes.append("document")
    if _TABULAR_CUES.search(question):
        routes.append("tabular")
    return routes or list(ROUTES)


def _parse_routes(raw: str) -> list[str] | None:
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    routes = [r for r in parsed if isinstance(r, str) and r in ROUTES]
    return routes or None


def supervisor_node(state: AskState) -> dict:
    """Route via the LLM, unioned with the keyword heuristic.

    The union matters: small routers under-select (measured: a "how many items"
    question routed to document-only and produced a wrong count), and extra fan-out
    only adds evidence — synthesis ranks it anyway.
    """
    question = state["question"]
    llm_routes: list[str] = []
    source = "heuristic"
    try:
        llm = get_chat_model()
        raw = llm.invoke(_ROUTER_PROMPT.format(question=question)).content
        llm_routes = _parse_routes(str(raw)) or []
        if llm_routes:
            source = "llm+heuristic"
    except Exception as exc:  # LLM down → heuristic keeps the pipeline alive
        logger.warning("supervisor LLM routing failed: %s", exc)
    routes = [r for r in ROUTES if r in set(llm_routes) | set(route_heuristic(question))]
    logger.info("routes=%s (%s)", routes, source)
    return {"routes": routes, "route_source": source}


# ------------------------------------------------------------- retrieval agents


def _with_source_type(filters: SearchFilters | None, source_type: str) -> SearchFilters:
    base = filters or SearchFilters()
    return SearchFilters(
        city=base.city,
        source_type=source_type,
        topic=base.topic,
        date_from=base.date_from,
        date_to=base.date_to,
    )


def transcript_node(state: AskState) -> dict:
    results = hybrid_search(
        state["question"],
        filters=_with_source_type(state.get("filters"), "transcript"),
        k=4,
        mode="hybrid_rerank",
    )
    evidence = [
        Evidence.from_video(
            text=r.text,
            url=r.url or "",
            t_start=r.t_start or 0.0,
            score=r.score,
            title=r.title,
            city=r.city,
            meeting_date=str(r.meeting_date),
        )
        for r in results
        if r.url
    ]
    return {"evidence": evidence}


def document_node(state: AskState) -> dict:
    results = hybrid_search(
        state["question"],
        filters=_with_source_type(state.get("filters"), "pdf"),
        k=4,
        mode="hybrid_rerank",
    )
    evidence = [
        Evidence.from_doc(
            text=r.text,
            page_no=r.page_no or 1,
            score=r.score,
            title=r.title,
            city=r.city,
            url=r.url,
            meeting_date=str(r.meeting_date),
        )
        for r in results
    ]
    return {"evidence": evidence}


def tabular_node(state: AskState) -> dict:
    try:
        from agents.tabular import run_tabular_agent  # separate module (guardrails live there)

        return run_tabular_agent(state["question"], state.get("filters"))
    except Exception:
        logger.exception("tabular agent unavailable")
        return {"evidence": [], "sql": []}


# ---------------------------------------------------------------- synthesis

_SYNTHESIS_PROMPT = """You answer questions about city-council meetings using ONLY the \
evidence below.

Rules:
- Every sentence MUST end with the marker of the evidence supporting it, like [E1].
- Use only facts stated in the evidence. Do not use outside knowledge.
- If the evidence does not answer the question, reply exactly: {not_found}
- Be concise (1-4 sentences).

Example:
Evidence: [E1] (video) The mayor announced that Main Street will close on Monday.
Question: When will Main Street close?
Answer: Main Street will close on Monday [E1].

Evidence:
{evidence_block}

Question: {question}

Answer:"""


def _evidence_block(evidence: list[Evidence]) -> str:
    lines = []
    for i, ev in enumerate(evidence, start=1):
        snippet = ev.text[:EVIDENCE_CHAR_CAP]
        lines.append(f"[E{i}] ({ev.kind}) {snippet}")
    return "\n\n".join(lines)


def _top_evidence(evidence: list[Evidence]) -> list[Evidence]:
    ranked = sorted(evidence, key=lambda e: e.score, reverse=True)
    # keep at least one of each kind present, then fill by score
    kept: list[Evidence] = []
    for kind in ("video", "doc", "table"):
        first = next((e for e in ranked if e.kind == kind), None)
        if first:
            kept.append(first)
    for ev in ranked:
        if ev not in kept and len(kept) < MAX_EVIDENCE:
            kept.append(ev)
    return kept[:MAX_EVIDENCE]


def extractive_answer(evidence: list[Evidence]) -> str:
    """Deterministic fallback: quote the best evidence with its citation."""
    if not evidence:
        return NOT_FOUND
    parts = []
    for ev in evidence[:2]:
        quote = ev.text[:220].rstrip()
        parts.append(f"The record shows: “{quote}…” {ev.citation}")
    return "\n\n".join(parts)


def synthesize(question: str, evidence: list[Evidence]) -> str:
    """Non-streaming synthesis with citation enforcement."""
    if not evidence:
        return NOT_FOUND
    top = _top_evidence(evidence)
    prompt = _SYNTHESIS_PROMPT.format(
        evidence_block=_evidence_block(top), question=question, not_found=NOT_FOUND
    )
    try:
        raw = str(get_chat_model().invoke(prompt).content).strip()
    except Exception as exc:
        logger.warning("synthesis LLM failed (%s); extractive fallback", exc)
        return extractive_answer(top)
    return _postprocess_answer(raw, top)


def _postprocess_answer(raw: str, evidence: list[Evidence]) -> str:
    if not raw or NOT_FOUND.lower().rstrip(".") in raw.lower():
        return NOT_FOUND
    resolved = resolve_markers(raw, evidence)
    if resolved == raw and not re.search(r"\[E\d+\]", raw):
        # model ignored the marker protocol entirely
        logger.info("no evidence markers in answer; extractive fallback")
        return extractive_answer(evidence)
    if sentences_without_citation(resolved):
        logger.info("uncited claims in answer; extractive fallback")
        return extractive_answer(evidence)
    return resolved


def synthesis_node(state: AskState) -> dict:
    return {"answer": synthesize(state["question"], state.get("evidence", []))}


# ------------------------------------------------------------------- graph

_NODE_BY_ROUTE = {
    "transcript": transcript_node,
    "document": document_node,
    "tabular": tabular_node,
}


@lru_cache(maxsize=1)
def build_graph():
    graph = StateGraph(AskState)
    graph.add_node("supervisor", supervisor_node)
    for name, fn in _NODE_BY_ROUTE.items():
        graph.add_node(name, fn)
    graph.add_node("synthesis", synthesis_node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", lambda s: s["routes"], {r: r for r in ROUTES})
    for r in ROUTES:
        graph.add_edge(r, "synthesis")
    graph.add_edge("synthesis", END)
    return graph.compile()


@lru_cache(maxsize=1)
def _langfuse_handler():
    """Langfuse callback handler, or None when unconfigured/unreachable."""
    settings = get_settings()
    if not (settings.langfuse_host and settings.langfuse_public_key):
        return None
    try:
        import httpx
        from langfuse.callback import CallbackHandler

        httpx.get(f"{settings.langfuse_host}/api/public/health", timeout=2)
        return CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception as exc:
        logger.warning("langfuse disabled: %s", exc)
        return None


def _invoke_config() -> dict:
    handler = _langfuse_handler()
    return {"callbacks": [handler]} if handler else {}


def ask(question: str, filters: SearchFilters | None = None) -> AskResult:
    """Synchronous end-to-end ask; every node traced to Langfuse when configured."""
    state = build_graph().invoke(
        {"question": question, "filters": filters}, config=_invoke_config()
    )
    return result_from_state(state)


async def ask_stream(
    question: str, filters: SearchFilters | None = None
) -> AsyncIterator[dict[str, Any]]:
    """Streaming ask: status events per node, token events during synthesis, final result.

    Retrieval/routing runs first (traced); then the synthesis LLM call is streamed
    token-by-token; the final event carries the fully post-processed AskResult (marker
    citations resolved — the streamed text is a preview, the result is authoritative).
    """
    import asyncio

    graph = build_graph()
    config = _invoke_config()

    yield {"type": "status", "node": "supervisor"}
    retrieval_state: AskState = await asyncio.to_thread(
        _run_retrieval_phase, graph, question, filters, config
    )
    yield {
        "type": "status",
        "node": "retrieval",
        "routes": retrieval_state.get("routes", []),
        "evidence_count": len(retrieval_state.get("evidence", [])),
    }

    evidence = _top_evidence(retrieval_state.get("evidence", []))
    answer: str
    if not evidence:
        answer = NOT_FOUND
        yield {"type": "token", "text": answer}
    else:
        prompt = _SYNTHESIS_PROMPT.format(
            evidence_block=_evidence_block(evidence), question=question, not_found=NOT_FOUND
        )
        raw_parts: list[str] = []
        try:
            async for part in get_chat_model().astream(prompt, config=config):
                text = str(part.content)
                if text:
                    raw_parts.append(text)
                    yield {"type": "token", "text": text}
        except Exception as exc:
            logger.warning("streaming synthesis failed (%s); extractive fallback", exc)
        answer = _postprocess_answer("".join(raw_parts), evidence)

    final = retrieval_state.copy()
    final["evidence"] = evidence
    final["answer"] = answer
    yield {"type": "result", "data": result_from_state(final).to_dict()}


def _run_retrieval_phase(
    graph, question: str, filters: SearchFilters | None, config: dict
) -> AskState:
    """Run supervisor + specialists, stopping before synthesis (which streams separately)."""
    state: AskState = {"question": question, "filters": filters}
    updates = supervisor_node(state)
    state.update(updates)  # type: ignore[typeddict-item]
    _ = graph  # full graph kept for the sync path; streaming drives nodes directly
    evidence: list[Evidence] = []
    sql: list[str] = []
    for route in state.get("routes", []):
        try:
            out = _NODE_BY_ROUTE[route](state)
        except Exception:
            logger.exception("route %s failed", route)
            continue
        evidence.extend(out.get("evidence", []))
        sql.extend(out.get("sql", []))
    state["evidence"] = evidence
    state["sql"] = sql
    return state
