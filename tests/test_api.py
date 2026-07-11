"""API tests: /ask (JSON + SSE), /examples, /health. No db or LLM required.

agents.graph.ask / ask_stream are monkeypatched on the module object api.main imported,
so the endpoints exercise the real request→filters→pipeline wiring against fakes.
"""

import json
from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from agents.state import AskResult
from retrieval.search import SearchFilters


@pytest.fixture
def client() -> TestClient:
    return TestClient(api_main.app)


def _parse_sse(lines: list[str]) -> list[tuple[str, dict]]:
    """(event, parsed-json-data) pairs from raw SSE lines; ping comments are skipped."""
    events: list[tuple[str, dict]] = []
    current = "message"
    for line in lines:
        if line.startswith("event:"):
            current = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            events.append((current, json.loads(line.split(":", 1)[1].strip())))
    return events


# ------------------------------------------------------------------ POST /ask (JSON)


def test_ask_json_returns_result_and_passes_filters(client: TestClient, monkeypatch) -> None:
    captured: dict = {}

    def fake_ask(question: str, filters: SearchFilters | None = None) -> AskResult:
        captured["question"] = question
        captured["filters"] = filters
        return AskResult(
            question=question,
            answer="There were 42 agenda items. [doc, p.3]",
            routes=["tabular"],
            route_source="llm",
            citations=["[doc, p.3]"],
        )

    monkeypatch.setattr(api_main.graph, "ask", fake_ask)
    response = client.post(
        "/ask",
        json={
            "question": "How many agenda items were there?",
            "city": "mesa",
            "source_type": "pdf",
            "topic": "budget",
            "date_from": "2026-04-01",
            "date_to": "2026-04-30",
            "stream": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "There were 42 agenda items. [doc, p.3]"
    assert body["routes"] == ["tabular"]
    assert body["citations"] == ["[doc, p.3]"]
    assert captured["question"] == "How many agenda items were there?"
    assert captured["filters"] == SearchFilters(
        city="mesa",
        source_type="pdf",
        topic="budget",
        date_from=date(2026, 4, 1),
        date_to=date(2026, 4, 30),
    )


def test_ask_json_omitted_filters_are_none(client: TestClient, monkeypatch) -> None:
    captured: dict = {}

    def fake_ask(question: str, filters: SearchFilters | None = None) -> AskResult:
        captured["filters"] = filters
        return AskResult(question=question, answer="ok")

    monkeypatch.setattr(api_main.graph, "ask", fake_ask)
    response = client.post("/ask", json={"question": "What happened?", "stream": False})
    assert response.status_code == 200
    assert captured["filters"] == SearchFilters()


def test_ask_json_pipeline_error_is_clean(client: TestClient, monkeypatch) -> None:
    def boom(question: str, filters: SearchFilters | None = None) -> AskResult:
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr(api_main.graph, "ask", boom)
    response = client.post("/ask", json={"question": "What happened?", "stream": False})
    assert response.status_code == 500
    body = response.json()
    assert body["type"] == "error"
    assert body["message"] == "pipeline exploded"
    assert "Traceback" not in body["message"]


def test_ask_question_too_short_is_422(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "hi", "stream": False})
    assert response.status_code == 422


# ------------------------------------------------------------------- POST /ask (SSE)


def test_ask_stream_yields_tokens_then_result(client: TestClient, monkeypatch) -> None:
    captured: dict = {}

    async def fake_ask_stream(
        question: str, filters: SearchFilters | None = None
    ) -> AsyncIterator[dict]:
        captured["question"] = question
        captured["filters"] = filters
        yield {"type": "status", "node": "supervisor"}
        yield {"type": "token", "text": "Hello"}
        yield {"type": "token", "text": " world"}
        yield {"type": "result", "data": {"question": question, "answer": "Hello world"}}

    monkeypatch.setattr(api_main.graph, "ask_stream", fake_ask_stream)
    with client.stream(
        "POST", "/ask", json={"question": "What happened?", "city": "mesa", "stream": True}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(list(response.iter_lines()))

    tokens = [data for name, data in events if name == "token"]
    assert tokens, "expected at least one token event"
    assert tokens[0] == {"type": "token", "text": "Hello"}

    assert events[-1][0] == "result"
    result = events[-1][1]
    assert result["type"] == "result"
    assert result["data"]["answer"] == "Hello world"

    assert captured["question"] == "What happened?"
    assert captured["filters"] == SearchFilters(city="mesa")


def test_ask_stream_error_becomes_error_event(client: TestClient, monkeypatch) -> None:
    async def broken_stream(
        question: str, filters: SearchFilters | None = None
    ) -> AsyncIterator[dict]:
        yield {"type": "status", "node": "supervisor"}
        raise RuntimeError("stream broke")

    monkeypatch.setattr(api_main.graph, "ask_stream", broken_stream)
    with client.stream("POST", "/ask", json={"question": "What happened?"}) as response:
        assert response.status_code == 200
        events = _parse_sse(list(response.iter_lines()))

    assert events[-1] == ("error", {"type": "error", "message": "stream broke"})


# --------------------------------------------------------------------- GET /examples


def test_examples_fallback_without_golden_dataset(
    client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(api_main, "GOLDEN_DATASET_PATH", tmp_path / "golden_dataset.json")
    response = client.get("/examples")
    assert response.status_code == 200
    examples = response.json()["examples"]
    assert len(examples) == 5
    assert all(isinstance(e["question"], str) and e["question"] for e in examples)


def test_examples_fallback_on_malformed_golden_dataset(
    client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "golden_dataset.json"
    path.write_text("not json at all", encoding="utf-8")
    monkeypatch.setattr(api_main, "GOLDEN_DATASET_PATH", path)
    response = client.get("/examples")
    assert response.status_code == 200
    assert len(response.json()["examples"]) == 5


def test_examples_stratified_from_golden_dataset(
    client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    rows: list[dict] = [{"bogus": "no question key"}, "not even a dict"]  # type: ignore[list-item]
    for source_type, n in (("transcript", 5), ("pdf", 5), ("table", 2)):
        rows.extend(
            {"question": f"{source_type} question {i}", "source_type": source_type}
            for i in range(n)
        )
    path = tmp_path / "golden_dataset.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(api_main, "GOLDEN_DATASET_PATH", path)
    monkeypatch.setattr(api_main, "_live_corpus_present", lambda: True)

    response = client.get("/examples")
    assert response.status_code == 200
    questions = [e["question"] for e in response.json()["examples"]]
    assert len(questions) == 8
    for source_type in ("transcript", "pdf", "table"):
        assert any(q.startswith(source_type) for q in questions)


def test_examples_sample_only_on_fresh_corpus(
    client: TestClient, monkeypatch, tmp_path: Path
) -> None:
    """A clean clone (sample corpus only) must never show unanswerable examples."""
    rows = [
        {"question": "sample q1", "source_type": "table", "sample": True},
        {"question": "live q1", "source_type": "transcript", "sample": False},
        {"question": "sample q2", "source_type": "transcript", "sample": True},
    ]
    path = tmp_path / "golden_dataset.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    monkeypatch.setattr(api_main, "GOLDEN_DATASET_PATH", path)
    monkeypatch.setattr(api_main, "_live_corpus_present", lambda: False)

    questions = [e["question"] for e in client.get("/examples").json()["examples"]]
    assert questions and all(q.startswith("sample") for q in questions)


# ------------------------------------------------------------------------ GET /health


def test_health_degrades_instead_of_crashing(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert isinstance(body["db"], bool)
    assert body["llm_backend"].startswith(("ollama:", "anthropic:"))
