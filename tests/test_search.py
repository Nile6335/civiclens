"""Tests for retrieval.search: RRF math, filter building, hybrid retrieval, reranking.

No network, no model downloads: the embedder and cross-encoder are always faked.
Integration tests use the db_conn fixture (auto-skips when Postgres is down).
"""

from datetime import date

import pytest

from common.settings import get_settings
from retrieval import search
from retrieval.search import (
    SearchFilters,
    SearchResult,
    dense_search,
    hybrid_search,
    keyword_search,
    rerank,
    rrf_fuse,
)


def _result(chunk_id: int, text: str = "text", score: float = 0.5) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        source_id=1,
        text=text,
        score=score,
        source_type="pdf",
        city="mesa",
        title="title",
        url=None,
        meeting_date=None,
        t_start=None,
        t_end=None,
        page_no=None,
        topic=None,
    )


# --- rrf_fuse -------------------------------------------------------------------------


def test_rrf_fuse_exact_scores_and_ordering() -> None:
    list_a = [_result(1), _result(2), _result(3)]
    list_b = [_result(2), _result(4)]
    fused = rrf_fuse([list_a, list_b], k=60)

    assert [r.chunk_id for r in fused] == [2, 1, 4, 3]
    by_id = {r.chunk_id: r.score for r in fused}
    assert by_id[1] == pytest.approx(1 / 61)
    assert by_id[2] == pytest.approx(1 / 62 + 1 / 61)
    assert by_id[3] == pytest.approx(1 / 63)
    assert by_id[4] == pytest.approx(1 / 62)


def test_rrf_fuse_dedups_and_keeps_first_seen_payload() -> None:
    list_a = [_result(7, text="first payload")]
    list_b = [_result(7, text="second payload"), _result(8)]
    fused = rrf_fuse([list_a, list_b], k=60)

    assert [r.chunk_id for r in fused] == [7, 8]
    assert fused[0].text == "first payload"
    assert fused[0].score == pytest.approx(2 / 61)


def test_rrf_fuse_does_not_mutate_inputs_and_respects_k() -> None:
    original = _result(1, score=0.123)
    fused = rrf_fuse([[original]], k=10)
    assert original.score == 0.123  # inputs untouched; score replaced on a copy
    assert fused[0].score == pytest.approx(1 / 11)


def test_rrf_fuse_empty() -> None:
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


# --- filter builder -------------------------------------------------------------------


def test_filter_conditions_empty() -> None:
    assert search._filter_conditions(None) == ([], [])
    assert search._filter_conditions(SearchFilters()) == ([], [])


@pytest.mark.parametrize(
    ("filters", "condition", "param"),
    [
        (SearchFilters(city="mesa"), "s.city = %s", "mesa"),
        (SearchFilters(source_type="pdf"), "s.source_type = %s", "pdf"),
        (SearchFilters(topic="budget"), "c.topic = %s", "budget"),
        (SearchFilters(date_from=date(2026, 1, 1)), "s.meeting_date >= %s", date(2026, 1, 1)),
        (SearchFilters(date_to=date(2026, 12, 31)), "s.meeting_date <= %s", date(2026, 12, 31)),
    ],
)
def test_filter_conditions_single(filters: SearchFilters, condition: str, param: object) -> None:
    assert search._filter_conditions(filters) == ([condition], [param])


def test_filter_conditions_combined() -> None:
    filters = SearchFilters(
        city="mesa",
        source_type="pdf",
        topic="budget",
        date_from=date(2026, 1, 1),
        date_to=date(2026, 12, 31),
    )
    conditions, params = search._filter_conditions(filters)
    assert conditions == [
        "s.city = %s",
        "s.source_type = %s",
        "c.topic = %s",
        "s.meeting_date >= %s",
        "s.meeting_date <= %s",
    ]
    assert params == ["mesa", "pdf", "budget", date(2026, 1, 1), date(2026, 12, 31)]


# --- keyword_search (integration) -----------------------------------------------------


def test_keyword_search_sample_corpus(db_conn) -> None:
    results = keyword_search(db_conn, "consent agenda", None, k=10)
    assert results, "sample corpus should match 'consent agenda'"
    assert len(results) <= 10
    assert {r.source_type for r in results} <= {"transcript", "pdf"}
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    for r in results:
        assert isinstance(r.chunk_id, int) and isinstance(r.source_id, int)
        assert "consent" in r.text.lower()
        assert r.score > 0
        assert r.city and r.title
        if r.source_type == "pdf":
            assert r.page_no is not None and r.t_start is None
        else:  # transcript
            assert r.t_start is not None and r.t_end is not None and r.page_no is None


def test_keyword_search_source_type_filter(db_conn) -> None:
    results = keyword_search(db_conn, "consent agenda", SearchFilters(source_type="pdf"), k=10)
    assert results
    assert all(r.source_type == "pdf" for r in results)


def test_keyword_search_date_filter_excludes_all(db_conn) -> None:
    filters = SearchFilters(date_from=date(1900, 1, 1), date_to=date(1900, 12, 31))
    assert keyword_search(db_conn, "consent agenda", filters, k=10) == []


# --- dense_search (integration, hand-built embeddings) --------------------------------


def _axis_vector(dim: int, axis: int, value: float = 1.0) -> list[float]:
    vec = [0.0] * dim
    vec[axis] = value
    return vec


def test_dense_search_ordering_with_known_embeddings(db_conn) -> None:
    dim = get_settings().embedding_dim
    try:
        (source_id,) = db_conn.execute(
            "INSERT INTO sources (city, source_type, title, meeting_id)"
            " VALUES ('densetestville', 'transcript', 'dense search test', 'dense-test')"
            " RETURNING id"
        ).fetchone()
        vectors = [_axis_vector(dim, 0), _axis_vector(dim, 1), _axis_vector(dim, 2)]
        for i, vec in enumerate(vectors):
            db_conn.execute(
                "INSERT INTO chunks (source_id, chunk_index, text, embedding)"
                " VALUES (%s, %s, %s, %s::vector)",
                (source_id, i, f"dense test chunk {i}", str(vec)),
            )
        # A fourth chunk with NULL embedding must never appear in dense results.
        db_conn.execute(
            "INSERT INTO chunks (source_id, chunk_index, text) VALUES (%s, 3, %s)",
            (source_id, "dense test chunk without embedding"),
        )

        query_vec = _axis_vector(dim, 0, 0.9)
        query_vec[1] = 0.1  # closest to axis 0, then axis 1, then axis 2
        filters = SearchFilters(city="densetestville")
        results = dense_search(db_conn, query_vec, filters, k=10)

        assert [r.text for r in results] == [
            "dense test chunk 0",
            "dense test chunk 1",
            "dense test chunk 2",
        ]
        norm = (0.9**2 + 0.1**2) ** 0.5
        assert results[0].score == pytest.approx(0.9 / norm, abs=1e-4)
        assert results[1].score == pytest.approx(0.1 / norm, abs=1e-4)
        assert results[2].score == pytest.approx(0.0, abs=1e-4)
        assert all(r.source_id == source_id for r in results)
        assert all(r.city == "densetestville" and r.source_type == "transcript" for r in results)
    finally:
        db_conn.rollback()  # discard the fake source + chunks


# --- rerank ---------------------------------------------------------------------------


class _FakeCrossEncoder:
    """Deterministic stand-in: scores looked up from the pair's document text."""

    def __init__(self, scores: dict[str, float]) -> None:
        self._scores = scores
        self.seen_pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.seen_pairs.extend(pairs)
        return [self._scores[text] for _, text in pairs]


def test_rerank_reorders_and_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeCrossEncoder({"low": 0.1, "high": 0.9, "mid": 0.5})
    monkeypatch.setattr(search, "_get_cross_encoder", lambda: fake)

    results = [_result(1, text="low"), _result(2, text="high"), _result(3, text="mid")]
    reranked = rerank("some query", results, top_n=2)

    assert [r.chunk_id for r in reranked] == [2, 3]
    assert [r.score for r in reranked] == [pytest.approx(0.9), pytest.approx(0.5)]
    assert fake.seen_pairs == [("some query", "low"), ("some query", "high"), ("some query", "mid")]
    assert results[0].score == 0.5  # inputs not mutated


def test_rerank_empty_does_not_load_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> None:
        raise AssertionError("cross-encoder must not be loaded for empty input")

    monkeypatch.setattr(search, "_get_cross_encoder", _boom)
    assert rerank("q", [], top_n=5) == []


# --- hybrid_search --------------------------------------------------------------------


class _FakeEmbedder:
    """Query encoder returning a fixed unit vector; never touches the network."""

    def __init__(self, dim: int) -> None:
        self._vec = _axis_vector(dim, 0)

    def encode_query(self, text: str) -> list[float]:
        return list(self._vec)


def test_hybrid_search_end_to_end(db_conn, monkeypatch: pytest.MonkeyPatch) -> None:
    dim = get_settings().embedding_dim
    monkeypatch.setattr(search, "get_embedder", lambda: _FakeEmbedder(dim))

    results = hybrid_search("consent agenda", k=5, mode="hybrid", conn=db_conn, candidates=10)
    assert results and len(results) <= 5
    # RRF scores: descending, each bounded by n_lists / (fuse_k + 1).
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    assert all(0.0 < r.score <= 2 / 61 for r in results)
    assert len({r.chunk_id for r in results}) == len(results)
    # The keyword leg's rank-1 doc scores >= 1/61, so RRF must keep at least one keyword hit
    # in the top-k regardless of how much of the corpus has embeddings.
    assert any("consent" in r.text.lower() for r in results)


def test_hybrid_search_dense_and_keyword_modes(db_conn, monkeypatch: pytest.MonkeyPatch) -> None:
    dim = get_settings().embedding_dim
    monkeypatch.setattr(search, "get_embedder", lambda: _FakeEmbedder(dim))

    keyword_results = hybrid_search("consent agenda", k=3, mode="keyword", conn=db_conn)
    assert keyword_results and all(r.score > 0 for r in keyword_results)

    dense_results = hybrid_search("consent agenda", k=3, mode="dense", conn=db_conn)
    assert isinstance(dense_results, list)  # sample corpus may have no embedded chunks


def test_hybrid_search_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unknown search mode"):
        hybrid_search("anything", mode="bm25")
