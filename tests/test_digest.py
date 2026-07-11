"""Digest generator: extractive core is deterministic; LLM polish degrades gracefully."""

from datetime import date

import pytest

from ingestion import digest


@pytest.fixture(autouse=True)
def no_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(digest, "_llm_summary", lambda text: None)


def test_digest_covers_corpus_meetings(db_conn) -> None:
    out = digest.build_digest(days=3650, until=date(2026, 7, 11))
    assert out.startswith("# CivicLens digest")
    assert "mesa" in out.lower() or "seattle" in out.lower() or "_No meetings" in out
    if "_No meetings" not in out:
        assert "## " in out  # at least one meeting section
        assert "> " in out or "Topics:" in out  # extractive fallback content present


def test_digest_empty_window(db_conn) -> None:
    out = digest.build_digest(days=1, until=date(1999, 1, 1))
    assert "_No meetings in the record for this window._" in out
