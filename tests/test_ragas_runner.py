"""Deterministic tests for the RAGAS runner: no network, no models, no live LLM calls."""

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest
import ragas

from evals.dataset import GoldenItem
from evals.ragas_runner import LocalEmbeddings, collect_answers, ragas_scores

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FakeAskResult:
    answer: str
    evidence: list[dict] = field(default_factory=list)


def _item(i: int) -> GoldenItem:
    return GoldenItem(
        id=f"g{i}",
        question=f"What happened in meeting {i}?",
        answer=f"Ground truth {i}.",
        difficulty="easy",
        source_type="transcript",
        city="springfield",
    )


# ---------------------------------------------------------------- collect_answers


def test_collect_answers_builds_records() -> None:
    items = [_item(1), _item(2)]

    def ask_fn(question: str) -> FakeAskResult:
        return FakeAskResult(
            answer=f"Answer to: {question}",
            evidence=[{"text": "ctx one", "score": 0.9}, {"text": "ctx two"}],
        )

    records = collect_answers(items, ask_fn=ask_fn)
    assert len(records) == 2
    assert records[0] == {
        "id": "g1",
        "question": "What happened in meeting 1?",
        "answer": "Answer to: What happened in meeting 1?",
        "contexts": ["ctx one", "ctx two"],
        "ground_truth": "Ground truth 1.",
    }
    assert records[1]["id"] == "g2"


def test_collect_answers_empty_evidence_gets_placeholder_context() -> None:
    records = collect_answers([_item(1)], ask_fn=lambda q: FakeAskResult(answer="a", evidence=[]))
    assert records[0]["contexts"] == [""]


def test_collect_answers_survives_ask_fn_exception() -> None:
    items = [_item(1), _item(2), _item(3)]

    def ask_fn(question: str) -> FakeAskResult:
        if "meeting 2" in question:
            raise RuntimeError("pipeline exploded")
        return FakeAskResult(answer="fine", evidence=[{"text": "ctx"}])

    records = collect_answers(items, ask_fn=ask_fn)
    assert len(records) == 3
    assert records[0]["answer"] == "fine"
    assert records[1] == {
        "id": "g2",
        "question": "What happened in meeting 2?",
        "answer": "",
        "contexts": [""],
        "ground_truth": "Ground truth 2.",
    }
    assert records[2]["answer"] == "fine"


def test_collect_answers_respects_limit() -> None:
    items = [_item(i) for i in range(5)]
    calls: list[str] = []

    def ask_fn(question: str) -> FakeAskResult:
        calls.append(question)
        return FakeAskResult(answer="a", evidence=[{"text": "t"}])

    records = collect_answers(items, ask_fn=ask_fn, limit=2)
    assert [r["id"] for r in records] == ["g0", "g1"]
    assert len(calls) == 2
    assert len(collect_answers(items, ask_fn=ask_fn, limit=None)) == 5


# ---------------------------------------------------------------- LocalEmbeddings


class FakeEmbedder:
    def __init__(self) -> None:
        self.passage_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        self.passage_calls.append(texts)
        return [[0.1, 0.2] for _ in texts]

    def encode_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [0.3, 0.4]


def test_local_embeddings_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeEmbedder()
    monkeypatch.setattr("retrieval.embeddings.get_embedder", lambda: fake)
    emb = LocalEmbeddings()
    assert emb.embed_documents(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert emb.embed_query("q") == [0.3, 0.4]
    assert fake.passage_calls == [["a", "b"]]
    assert fake.query_calls == ["q"]


def test_local_embeddings_construction_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> None:
        raise AssertionError("embedder must not load at construction time")

    monkeypatch.setattr("retrieval.embeddings.get_embedder", boom)
    LocalEmbeddings()  # must not raise


# ---------------------------------------------------------------- ragas_scores


class FakeEvalResult:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def to_pandas(self) -> pd.DataFrame:
        return self._frame


def _records() -> list[dict]:
    return [
        {"id": "g1", "question": "q1", "answer": "a1", "contexts": ["c1"], "ground_truth": "t1"},
        {"id": "g2", "question": "q2", "answer": "a2", "contexts": [], "ground_truth": "t2"},
        {"id": "g3", "question": "q3", "answer": "", "contexts": [""], "ground_truth": "t3"},
    ]


def test_ragas_scores_nanmean_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        {
            "faithfulness": [1.0, 0.5, float("nan")],
            "answer_relevancy": [0.9, 0.6, 0.3],
            "context_precision": [float("nan"), 1.0, 0.0],
            "context_recall": [1.0, 1.0, 0.0],
        }
    )
    captured: dict = {}

    def fake_evaluate(dataset, **kwargs) -> FakeEvalResult:
        captured["dataset"] = dataset
        captured["kwargs"] = kwargs
        return FakeEvalResult(frame)

    monkeypatch.setattr(ragas, "evaluate", fake_evaluate)
    monkeypatch.setattr("common.llm.get_chat_model", lambda: object())

    scores = ragas_scores(_records(), max_workers=3)

    assert scores["faithfulness"] == pytest.approx(0.75)
    assert scores["answer_relevancy"] == pytest.approx(0.6)
    assert scores["context_precision"] == pytest.approx(0.5)
    assert scores["context_recall"] == pytest.approx(2 / 3)
    assert scores["n"] == 3
    assert scores["per_metric_n"] == {
        "faithfulness": 2,
        "answer_relevancy": 3,
        "context_precision": 2,
        "context_recall": 3,
    }

    dataset = captured["dataset"]
    assert set(dataset.column_names) == {"question", "answer", "contexts", "ground_truth"}
    assert len(dataset) == 3
    assert dataset[1]["contexts"] == [""]  # empty contexts must not crash the build
    assert len(captured["kwargs"]["metrics"]) == 4
    assert captured["kwargs"]["run_config"].max_workers == 3
    assert captured["kwargs"]["run_config"].timeout == 120


def test_ragas_scores_failure_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_evaluate(dataset, **kwargs) -> None:
        raise ValueError("judge fell over")

    monkeypatch.setattr(ragas, "evaluate", fake_evaluate)
    monkeypatch.setattr("common.llm.get_chat_model", lambda: object())
    with pytest.raises(RuntimeError, match="judge fell over"):
        ragas_scores(_records())


def test_ragas_scores_empty_records_raises() -> None:
    with pytest.raises(RuntimeError):
        ragas_scores([])


# ---------------------------------------------------------------- import-time safety


def test_import_is_lazy() -> None:
    """Importing evals.ragas_runner must not pull in ragas/datasets/models or the DB."""
    code = (
        "import sys; import evals.ragas_runner; "
        "banned = {'ragas', 'datasets', 'agents.graph', 'sentence_transformers', 'psycopg_pool'}; "
        "loaded = sorted(banned & set(sys.modules)); "
        "assert not loaded, f'eagerly imported: {loaded}'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
