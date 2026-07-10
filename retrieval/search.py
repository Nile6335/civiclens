"""Hybrid retrieval: dense (pgvector cosine) + Postgres full-text search, fused with
Reciprocal Rank Fusion and optionally reranked by a cross-encoder.

`hybrid_search` is the single entry point used by the agents (Phase 3) and the eval
ablations (Phase 4); the lower-level functions are exposed for testing and ablations.
"""

import logging
import re
from dataclasses import dataclass, replace
from datetime import date
from functools import lru_cache

import psycopg

from common.db import get_connection
from common.settings import get_settings
from retrieval.embeddings import get_embedder

logger = logging.getLogger(__name__)

SEARCH_MODES = ("dense", "keyword", "hybrid", "hybrid_rerank")


@dataclass(frozen=True)
class SearchFilters:
    """Metadata filters applied to every retrieval mode."""

    city: str | None = None
    source_type: str | None = None
    topic: str | None = None
    date_from: date | None = None
    date_to: date | None = None


@dataclass
class SearchResult:
    """One retrieved chunk with its source metadata and a mode-dependent score."""

    chunk_id: int
    source_id: int
    text: str
    score: float
    source_type: str
    city: str
    title: str
    url: str | None
    meeting_date: date | None
    t_start: float | None
    t_end: float | None
    page_no: int | None
    topic: str | None


_SELECT = (
    "SELECT c.id, c.source_id, c.text, {score_expr} AS score,"
    " s.source_type, s.city, s.title, s.url, s.meeting_date,"
    " c.t_start, c.t_end, c.page_no, c.topic"
    " FROM chunks c JOIN sources s ON s.id = c.source_id"
)


def _filter_conditions(filters: SearchFilters | None) -> tuple[list[str], list[object]]:
    """Build WHERE-clause fragments + params from SearchFilters (aliases: c=chunks, s=sources)."""
    conditions: list[str] = []
    params: list[object] = []
    if filters is None:
        return conditions, params
    if filters.city is not None:
        conditions.append("s.city = %s")
        params.append(filters.city)
    if filters.source_type is not None:
        conditions.append("s.source_type = %s")
        params.append(filters.source_type)
    if filters.topic is not None:
        conditions.append("c.topic = %s")
        params.append(filters.topic)
    if filters.date_from is not None:
        conditions.append("s.meeting_date >= %s")
        params.append(filters.date_from)
    if filters.date_to is not None:
        conditions.append("s.meeting_date <= %s")
        params.append(filters.date_to)
    return conditions, params


def _rows_to_results(rows: list[tuple]) -> list[SearchResult]:
    return [
        SearchResult(
            chunk_id=row[0],
            source_id=row[1],
            text=row[2],
            score=float(row[3]),
            source_type=row[4],
            city=row[5],
            title=row[6],
            url=row[7],
            meeting_date=row[8],
            t_start=row[9],
            t_end=row[10],
            page_no=row[11],
            topic=row[12],
        )
        for row in rows
    ]


def dense_search(
    conn: psycopg.Connection,
    query_vec: list[float],
    filters: SearchFilters | None = None,
    k: int = 8,
) -> list[SearchResult]:
    """Top-k chunks by pgvector cosine similarity; score = 1 - cosine distance."""
    conditions, params = _filter_conditions(filters)
    conditions.insert(0, "c.embedding IS NOT NULL")
    vec = str(query_vec)
    sql = (
        _SELECT.format(score_expr="1 - (c.embedding <=> %s::vector)")
        + " WHERE "
        + " AND ".join(conditions)
        + " ORDER BY c.embedding <=> %s::vector ASC LIMIT %s"
    )
    rows = conn.execute(sql, [vec, *params, vec, k]).fetchall()  # type: ignore[arg-type]
    return _rows_to_results(rows)


_TSQUERY_WORD_RE = re.compile(r"[a-zA-Z0-9]{3,}")

# Ubiquitous in this corpus: OR-matching them floods the ranking with junk.
_CORPUS_STOPWORDS = frozenset({"city", "council", "meeting", "meetings", "member", "members"})


def _or_tsquery(query: str) -> str:
    """OR-of-discriminative-words websearch query: 'a OR b OR c'."""
    words = dict.fromkeys(
        w.lower() for w in _TSQUERY_WORD_RE.findall(query) if w.lower() not in _CORPUS_STOPWORDS
    )
    return " OR ".join(words)


def keyword_search(
    conn: psycopg.Connection,
    query: str,
    filters: SearchFilters | None = None,
    k: int = 8,
) -> list[SearchResult]:
    """Top-k chunks by full-text match; score = ts_rank_cd against websearch_to_tsquery.

    websearch_to_tsquery ANDs all terms, so full-sentence questions usually match
    nothing; when the strict query comes back empty we retry with an OR of the query's
    content words (ts_rank_cd still favors chunks matching many of them).
    """
    conditions, params = _filter_conditions(filters)
    conditions.insert(0, "c.tsv @@ websearch_to_tsquery('english', %s)")
    sql = (
        _SELECT.format(score_expr="ts_rank_cd(c.tsv, websearch_to_tsquery('english', %s))")
        + " WHERE "
        + " AND ".join(conditions)
        + " ORDER BY score DESC, c.id ASC LIMIT %s"
    )
    rows = conn.execute(sql, [query, query, *params, k]).fetchall()  # type: ignore[arg-type]
    if not rows:
        relaxed = _or_tsquery(query)
        if relaxed:
            rows = conn.execute(  # type: ignore[arg-type]
                sql, [relaxed, relaxed, *params, k]
            ).fetchall()
    return _rows_to_results(rows)


def rrf_fuse(result_lists: list[list[SearchResult]], k: int = 60) -> list[SearchResult]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank_i(d)), 1-based ranks.

    Documents are keyed by chunk_id; the first-seen payload is kept (with the score field
    replaced by the RRF score). Ties preserve first-seen order.
    """
    scores: dict[int, float] = {}
    payloads: dict[int, SearchResult] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank)
            payloads.setdefault(result.chunk_id, result)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [replace(payloads[chunk_id], score=score) for chunk_id, score in ordered]


@lru_cache(maxsize=1)
def _get_cross_encoder():  # noqa: ANN202 - sentence_transformers is a lazy import
    from sentence_transformers import CrossEncoder  # lazy: 100MB+ import chain

    model_name = get_settings().reranker_model
    logger.info("loading reranker model %s", model_name)
    return CrossEncoder(model_name, device="cpu")


def rerank(query: str, results: list[SearchResult], top_n: int = 8) -> list[SearchResult]:
    """Rescore results with the cross-encoder and return the top_n (score = CE score)."""
    if not results:
        return []
    cross_encoder = _get_cross_encoder()
    ce_scores = cross_encoder.predict([(query, r.text) for r in results])
    if any(s != s for s in ce_scores):  # NaN scores: sorted() would silently no-op
        logger.warning(
            "cross-encoder returned NaN scores (model %s broken on this stack); "
            "keeping fused order",
            get_settings().reranker_model,
        )
        return results[:top_n]
    scored = sorted(
        zip(results, ce_scores, strict=True), key=lambda pair: float(pair[1]), reverse=True
    )
    return [replace(r, score=float(s)) for r, s in scored[:top_n]]


def hybrid_search(
    query: str,
    filters: SearchFilters | None = None,
    k: int = 8,
    mode: str = "hybrid_rerank",
    conn: psycopg.Connection | None = None,
    fuse_k: int = 60,
    candidates: int = 20,
) -> list[SearchResult]:
    """Single retrieval entry point for the agents and eval ablations.

    Modes:
      - "dense": embed the query, dense top-k.
      - "keyword": full-text top-k.
      - "hybrid": dense top-candidates + keyword top-candidates -> RRF -> top-k.
      - "hybrid_rerank": RRF top-candidates -> cross-encoder -> top-k.
    """
    if mode not in SEARCH_MODES:
        raise ValueError(f"unknown search mode {mode!r}; expected one of {SEARCH_MODES}")
    owns_conn = conn is None
    if conn is None:
        conn = get_connection()
    try:
        if mode == "keyword":
            return keyword_search(conn, query, filters, k)
        query_vec = get_embedder().encode_query(query)
        if mode == "dense":
            return dense_search(conn, query_vec, filters, k)
        dense_results = dense_search(conn, query_vec, filters, candidates)
        keyword_results = keyword_search(conn, query, filters, candidates)
        fused = rrf_fuse([dense_results, keyword_results], k=fuse_k)
        if mode == "hybrid":
            return fused[:k]
        # rerank over both full candidate lists (fusion interleaves dense needles with
        # keyword noise; truncating at `candidates` would drop tail dense candidates)
        return rerank(query, fused[: candidates * 2], top_n=k)
    finally:
        if owns_conn:
            conn.close()
