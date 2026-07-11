"""Tests for evals.metrics: hit@k / MRR math on crafted items.

No database, no network, no LLM: resolve_span_chunk_ids and the retriever are always
faked, so the metric math is exercised in isolation.
"""

import pytest

from evals import dataset, metrics
from evals.dataset import GoldenItem, Span
from retrieval.search import SearchResult

_CONN_SENTINEL = object()


def _span(meeting_id: str) -> Span:
    return Span(
        city="mesa",
        source_type="transcript",
        meeting_id=meeting_id,
        chunk_index=0,
        text_snippet="snippet",
    )


def _item(item_id: str, spans: list[Span]) -> GoldenItem:
    return GoldenItem(
        id=item_id,
        question=f"question {item_id}",
        answer="answer",
        difficulty="easy",
        source_type="transcript",
        city="mesa",
        spans=spans,
    )


def _result(chunk_id: int) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        source_id=1,
        text="text",
        score=0.5,
        source_type="transcript",
        city="mesa",
        title="title",
        url=None,
        meeting_date=None,
        t_start=None,
        t_end=None,
        page_no=None,
        topic=None,
    )


class _FakeSearch:
    """Retriever stand-in: returns fixed chunk-id sequences keyed by question."""

    def __init__(self, ids_by_question: dict[str, list[int]]) -> None:
        self._ids_by_question = ids_by_question
        self.calls: list[dict] = []

    def __call__(self, question: str, *, k: int, mode: str) -> list[SearchResult]:
        self.calls.append({"question": question, "k": k, "mode": mode})
        return [_result(cid) for cid in self._ids_by_question[question]]


# Span meeting_id -> resolved chunk ids ("unresolved" resolves to nothing).
_RELEVANT = {"m-first": {10}, "m-third": {20, 21}, "m-miss": {30}, "m-unresolved": set()}


def _fake_resolve(conn: object, item: GoldenItem) -> set[int]:
    assert conn is _CONN_SENTINEL  # the provided conn must be passed through
    return set(_RELEVANT[item.spans[0].meeting_id])


@pytest.fixture
def fake_search(monkeypatch: pytest.MonkeyPatch) -> _FakeSearch:
    fake = _FakeSearch(
        {
            "question first": [10, 11, 12, 13, 14],  # relevant at rank 1 -> rr = 1
            "question third": [98, 99, 21, 3, 4],  # relevant at rank 3 -> rr = 1/3
            "question miss": [1, 2, 3, 4, 5],  # no relevant -> rr = 0
        }
    )
    monkeypatch.setattr(metrics, "search_fn", fake)
    monkeypatch.setattr(dataset, "expanded_relevant_ids", _fake_resolve)
    return fake


def _crafted_items() -> list[GoldenItem]:
    return [
        _item("first", [_span("m-first")]),
        _item("third", [_span("m-third")]),
        _item("miss", [_span("m-miss")]),
        _item("no-spans", []),  # table item: skipped, not counted anywhere
        _item("unresolved", [_span("m-unresolved")]),  # spans resolve to nothing
    ]


def test_retrieval_metrics_exact_values(fake_search: _FakeSearch) -> None:
    report = metrics.retrieval_metrics(_crafted_items(), mode="hybrid", k=5, conn=_CONN_SENTINEL)

    assert report["mode"] == "hybrid"
    assert report["k"] == 5
    assert report["n"] == 3  # first, third, miss; no-spans skipped, unresolved excluded
    assert report["unresolved"] == 1
    assert report["hit_rate"] == pytest.approx(2 / 3)
    assert report["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert report["per_item"] == [
        {"id": "first", "hit": True, "rr": pytest.approx(1.0)},
        {"id": "third", "hit": True, "rr": pytest.approx(1 / 3)},
        {"id": "miss", "hit": False, "rr": 0.0},
    ]


def test_retrieval_metrics_mode_and_k_pass_through(fake_search: _FakeSearch) -> None:
    metrics.retrieval_metrics(_crafted_items(), mode="keyword", k=7, conn=_CONN_SENTINEL)

    # One search per evaluated item — none for no-spans (skipped) or unresolved items.
    assert [c["question"] for c in fake_search.calls] == [
        "question first",
        "question third",
        "question miss",
    ]
    assert all(c["mode"] == "keyword" and c["k"] == 7 for c in fake_search.calls)


def test_retrieval_metrics_empty_and_span_free_items(
    monkeypatch: pytest.MonkeyPatch, fake_search: _FakeSearch
) -> None:
    def _no_db() -> None:
        raise AssertionError("get_connection must not be called")

    monkeypatch.setattr(metrics, "get_connection", _no_db)

    for items in ([], [_item("no-spans", [])]):
        report = metrics.retrieval_metrics(items, mode="dense", k=5, conn=None)
        assert report["n"] == 0
        assert report["unresolved"] == 0
        assert report["hit_rate"] == 0.0
        assert report["mrr"] == 0.0
        assert report["per_item"] == []
    assert fake_search.calls == []


def test_retrieval_metrics_all_unresolved(fake_search: _FakeSearch) -> None:
    items = [_item("u1", [_span("m-unresolved")]), _item("u2", [_span("m-unresolved")])]
    report = metrics.retrieval_metrics(items, mode="hybrid_rerank", k=5, conn=_CONN_SENTINEL)

    assert report["n"] == 0
    assert report["unresolved"] == 2
    assert report["hit_rate"] == 0.0
    assert report["mrr"] == 0.0
    assert fake_search.calls == []


def test_retrieval_metrics_opens_and_closes_own_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeConn:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    fake_conn = _FakeConn()
    monkeypatch.setattr(metrics, "get_connection", lambda: fake_conn)
    monkeypatch.setattr(dataset, "expanded_relevant_ids", lambda conn, item: {10})
    fake = _FakeSearch({"question first": [10]})
    monkeypatch.setattr(metrics, "search_fn", fake)

    report = metrics.retrieval_metrics([_item("first", [_span("m-first")])], mode="dense", k=3)

    assert report["n"] == 1
    assert report["hit_rate"] == pytest.approx(1.0)
    assert fake_conn.closed  # lazily opened connection is closed on the way out
