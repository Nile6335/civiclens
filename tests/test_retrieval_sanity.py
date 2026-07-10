"""Phase 2 acceptance: the scripted sanity query over sample data hits the right chunk.

These tests load the real (lean-profile) embedding/reranker models — marked slow.
They self-provision: if the sample corpus is missing or unembedded, they ingest/embed.
"""

import pytest

pytestmark = [pytest.mark.slow]


@pytest.fixture(scope="module")
def embedded_corpus(db_conn):
    row = db_conn.execute(
        "SELECT count(*) FROM chunks c JOIN sources s ON s.id = c.source_id"
        " WHERE s.metadata->>'sample' = 'true'"
    ).fetchone()
    if row[0] == 0:
        from ingestion.cli import ingest_samples

        ingest_samples()
    from retrieval.index import embed_pending_chunks

    embed_pending_chunks()
    return db_conn


def test_sanity_query_top3(embedded_corpus) -> None:
    """The known fact from the sample meeting is retrieved in the top-3."""
    from retrieval.search import SearchFilters, hybrid_search

    results = hybrid_search(
        "Which council member was excused from the Mesa city council meeting?",
        filters=SearchFilters(city="mesa", source_type="transcript"),
        k=3,
        mode="hybrid_rerank",
    )
    assert results, "no results returned"
    assert any("GoForth" in r.text for r in results[:3]), [r.text[:80] for r in results]


def test_sanity_query_pdf(embedded_corpus) -> None:
    from retrieval.search import SearchFilters, hybrid_search

    results = hybrid_search(
        "items on the consent agenda for approval",
        filters=SearchFilters(source_type="pdf", city="mesa"),
        k=3,
        mode="hybrid",
    )
    assert results
    assert all(r.source_type == "pdf" for r in results)
    assert any("consent" in r.text.lower() for r in results[:3])


def test_dense_vs_hybrid_modes_run(embedded_corpus) -> None:
    from retrieval.search import hybrid_search

    for mode in ("dense", "keyword", "hybrid", "hybrid_rerank"):
        results = hybrid_search("public comment at the council meeting", k=3, mode=mode)
        assert results, f"mode {mode} returned nothing"
