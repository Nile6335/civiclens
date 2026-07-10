"""Unit tests for retrieval/topics.py (no network, no model downloads)."""

import sys
import types
from types import SimpleNamespace

import pytest

from ingestion.models import TOPIC_LABELS, ChunkRecord
from retrieval import topics
from retrieval.topics import KEYWORD_MAP, tag_chunks, tag_text, tag_text_keyword


@pytest.fixture(autouse=True)
def _keyword_settings(monkeypatch):
    """Pin the dispatcher to the keyword backend and keep the pipeline cache clean."""
    monkeypatch.setattr(topics, "get_settings", lambda: SimpleNamespace(topic_tagger="keyword"))
    topics._get_zeroshot_pipeline.cache_clear()
    yield
    topics._get_zeroshot_pipeline.cache_clear()


def test_keyword_map_covers_all_non_other_labels() -> None:
    assert set(KEYWORD_MAP) == set(TOPIC_LABELS) - {"other"}
    for label, cues in KEYWORD_MAP.items():
        assert 8 <= len(cues) <= 15, label
        assert all(cue == cue.lower() for cue in cues), label
        assert len(set(cues)) == len(cues), label


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The commission approved a rezone and a zoning variance.", "zoning"),
        ("Staff presented the plat and annexation request under the land use plan.", "zoning"),
        ("The fiscal year budget adds an appropriation from the general fund revenue.", "budget"),
        ("A bond levy will fund the capital improvement expenditures this year.", "budget"),
        ("Police and the fire department responded after a 911 call.", "public safety"),
        ("Crime is down and a new ambulance joins the public safety fleet.", "public safety"),
        ("Light rail and bus riders asked for a bike lane and a sidewalk.", "transportation"),
        ("Traffic calming and pavement work in the street improvement plan.", "transportation"),
        ("Affordable housing for homeless residents; tenants face eviction.", "housing"),
        ("The shelter and new dwelling units help families avoid eviction.", "housing"),
        ("Members discussed the holiday parade and the library book sale.", "other"),
        ("", "other"),
    ],
)
def test_tag_text_keyword_labels(text: str, expected: str) -> None:
    assert tag_text_keyword(text) == expected


def test_tag_text_keyword_tie_breaks_by_topic_labels_order() -> None:
    # One zoning cue vs one budget cue: zoning precedes budget in TOPIC_LABELS.
    assert tag_text_keyword("The zoning item follows the budget item.") == "zoning"
    # One transportation cue vs one housing cue: transportation comes first.
    assert tag_text_keyword("The bus stops near the shelter.") == "transportation"


def test_tag_text_keyword_highest_count_wins() -> None:
    text = "The budget appropriation for the fiscal year grew; zoning came up once."
    assert tag_text_keyword(text) == "budget"


def test_tag_text_keyword_word_boundaries() -> None:
    assert tag_text_keyword("Renters attended the meeting.") == "other"  # not 'rent'
    assert tag_text_keyword("A busy agenda for the evening.") == "other"  # not 'bus'
    assert tag_text_keyword("Plates were served at the reception.") == "other"  # not 'plat'
    # Plurals and hyphenated cues still hit.
    assert tag_text_keyword("Tenants asked about rents.") == "housing"
    assert tag_text_keyword("Acquisition of right-of-way along Main Street.") == "transportation"


def test_tag_text_keyword_is_case_insensitive() -> None:
    assert tag_text_keyword("ZONING VARIANCE public hearing") == "zoning"


def test_tag_chunks_sets_topics_and_preserves_existing() -> None:
    chunks = [
        ChunkRecord(chunk_index=0, text="A zoning variance and setback request."),
        ChunkRecord(chunk_index=1, text="General fund budget appropriation.", topic="housing"),
        ChunkRecord(chunk_index=2, text="Nothing topical here at all."),
    ]
    result = tag_chunks(chunks)
    assert result is chunks  # tagged in place
    assert chunks[0].topic == "zoning"
    assert chunks[1].topic == "housing"  # pre-set topic untouched
    assert chunks[2].topic == "other"


def _install_fake_transformers(monkeypatch, calls: dict) -> None:
    """Put a fake transformers module in sys.modules so no real model is loaded."""

    def fake_pipeline(task: str, model: str | None = None, device: int | None = None):
        calls["constructed"] = calls.get("constructed", 0) + 1
        calls["task"] = task
        calls["model"] = model
        calls["device"] = device

        def classify(text: str, labels: list[str], multi_label: bool | None = None) -> dict:
            calls["text"] = text
            calls["labels"] = labels
            calls["multi_label"] = multi_label
            return {"labels": ["transportation", "other"], "scores": [0.9, 0.1]}

        return classify

    fake_module = types.ModuleType("transformers")
    fake_module.pipeline = fake_pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_module)


def test_tag_text_zeroshot_uses_pipeline_and_truncates(monkeypatch) -> None:
    calls: dict = {}
    _install_fake_transformers(monkeypatch, calls)
    monkeypatch.setattr(
        topics,
        "get_settings",
        lambda: SimpleNamespace(topic_tagger="zeroshot", zero_shot_model="fake/zero-shot"),
    )

    long_text = "traffic and transit updates " * 100  # well over 1000 chars
    assert topics.tag_text_zeroshot(long_text) == "transportation"

    assert calls["task"] == "zero-shot-classification"
    assert calls["model"] == "fake/zero-shot"
    assert calls["device"] == -1
    assert calls["labels"] == TOPIC_LABELS
    assert calls["multi_label"] is False
    assert len(calls["text"]) <= 1000
    assert calls["text"] == long_text[:1000]

    # The pipeline is lazily built once and cached.
    assert topics.tag_text_zeroshot("short text") == "transportation"
    assert calls["constructed"] == 1


def test_tag_text_dispatches_on_settings(monkeypatch) -> None:
    settings = SimpleNamespace(topic_tagger="zeroshot", zero_shot_model="fake/zero-shot")
    monkeypatch.setattr(topics, "get_settings", lambda: settings)
    monkeypatch.setattr(topics, "tag_text_zeroshot", lambda text: "housing")

    assert tag_text("anything at all") == "housing"

    settings.topic_tagger = "keyword"
    assert tag_text("tenant eviction and rent relief") == "housing"
    assert tag_text("nothing topical") == "other"
