"""LLM-free retrieval metrics (hit@k, MRR) over the golden dataset.

For each golden item with spans, the spans are resolved to current chunks.id values
(natural keys survive re-ingestion) and the question is retrieved with NO filters —
retrieval must find the needle in the whole corpus.
"""

from collections.abc import Callable

import psycopg

from common.db import get_connection
from evals import dataset
from evals.dataset import GoldenItem
from retrieval.search import SearchResult, hybrid_search

# Indirection so tests can inject a fake retriever (no pgvector needed for the math).
search_fn: Callable[..., list[SearchResult]] = hybrid_search


def retrieval_metrics(
    items: list[GoldenItem],
    mode: str,
    k: int = 5,
    conn: psycopg.Connection | None = None,
) -> dict:
    """Compute hit@k and MRR for one retrieval mode over every span-backed item.

    Items with no spans (e.g. table-derived items) are skipped outright; items whose
    spans resolve to no current chunk ids are counted as "unresolved" and excluded from
    the averages. A connection is opened lazily via get_connection() when conn is None.
    """
    per_item: list[dict] = []
    unresolved = 0
    owns_conn = False
    try:
        for item in items:
            if not item.spans:
                continue
            if conn is None:
                conn = get_connection()
                owns_conn = True
            relevant = dataset.expanded_relevant_ids(conn, item)
            if not relevant:
                unresolved += 1
                continue
            results = search_fn(item.question, k=k, mode=mode)
            rr = 0.0
            for rank, result in enumerate(results, start=1):
                if result.chunk_id in relevant:
                    rr = 1.0 / rank
                    break
            per_item.append({"id": item.id, "hit": rr > 0.0, "rr": rr})
    finally:
        if owns_conn and conn is not None:
            conn.close()
    n = len(per_item)
    return {
        "mode": mode,
        "k": k,
        "n": n,
        "unresolved": unresolved,
        "hit_rate": (sum(1 for row in per_item if row["hit"]) / n) if n else 0.0,
        "mrr": (sum(row["rr"] for row in per_item) / n) if n else 0.0,
        "per_item": per_item,
    }
