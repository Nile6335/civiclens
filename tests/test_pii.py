"""Unit tests for safety/pii.py (no NER model, no network, no downloads).

Person detection is exercised only through a fake transformers module injected into
sys.modules; the real model is never loaded and importing safety.pii must not pull
transformers in.
"""

import importlib
import sys
import types
from dataclasses import asdict

import pytest

import safety.pii as pii
from ingestion.models import ChunkRecord
from safety.pii import (
    PiiSpan,
    build_seeded_pii_testset,
    detect_persons,
    detect_regex,
    merge_spans,
    redact,
    redact_chunk_records,
    redact_text,
    score_pii_detection,
)

# ---------------------------------------------------------------------------
# import hygiene
# ---------------------------------------------------------------------------


def test_import_does_not_import_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    monkeypatch.delitem(sys.modules, "safety.pii", raising=False)
    importlib.import_module("safety.pii")
    assert "transformers" not in sys.modules


# ---------------------------------------------------------------------------
# regex detection: exact counts + redaction placeholders
# ---------------------------------------------------------------------------

_PARAGRAPH = (
    "Call me at (602) 555-1234 or 602-555-1234, and after hours 602.555.1234. "
    "Email jane.doe@example.com or council@civic.example.org. "
    "I live at 123 East Main Street near 45 N 2nd Ave."
)


def test_detect_regex_exact_counts() -> None:
    spans = detect_regex(_PARAGRAPH)
    by_type: dict[str, int] = {}
    for span in spans:
        by_type[span.type] = by_type.get(span.type, 0) + 1
    assert by_type == {"phone": 3, "email": 2, "address": 2}
    # sorted by start ascending
    assert spans == sorted(spans, key=lambda s: s.start)


def test_detect_regex_spans_are_accurate() -> None:
    # Every span's recorded text matches what the offsets slice out.
    for span in detect_regex(_PARAGRAPH):
        assert _PARAGRAPH[span.start : span.end] == span.text


def test_redact_replaces_with_typed_placeholders_and_leaves_text_intact() -> None:
    spans = detect_regex(_PARAGRAPH)
    redacted = redact(_PARAGRAPH, spans)
    assert redacted.count("[PHONE]") == 3
    assert redacted.count("[EMAIL]") == 2
    assert redacted.count("[ADDRESS]") == 2
    # Surrounding prose survives.
    assert redacted.startswith("Call me at [PHONE]")
    assert "after hours [PHONE]." in redacted
    # The trailing "." is part of "45 N 2nd Ave." and gets absorbed into [ADDRESS].
    assert redacted.endswith("near [ADDRESS]")
    # No raw PII remains.
    assert "@" not in redacted
    assert "555" not in redacted
    assert "Main Street" not in redacted


def test_redacted_text_has_no_raw_offsets_left() -> None:
    redacted, _ = redact_text(_PARAGRAPH, use_ner=False)
    # No 4+ digit runs (phone/street numbers all gone).
    import re

    assert not re.search(r"\d{4}", redacted)


# ---------------------------------------------------------------------------
# word-boundary sanity (false positives we must NOT flag)
# ---------------------------------------------------------------------------


def test_street_smarts_is_not_an_address() -> None:
    spans = detect_regex("She showed some Street smarts during the 12 hour debate.")
    assert [s for s in spans if s.type == "address"] == []


def test_meet_me_at_5_is_not_a_phone() -> None:
    spans = detect_regex("Let's meet me at 5 tomorrow, the vote was 4 to 3 in 2024.")
    assert [s for s in spans if s.type == "phone"] == []


def test_year_and_short_numbers_are_not_phones() -> None:
    spans = detect_regex("Ordinance 2024 passed; item 7 on page 12 of the 300 page report.")
    assert [s for s in spans if s.type == "phone"] == []


def test_bare_ten_digit_run_is_a_phone() -> None:
    spans = detect_regex("Reach the office line 6025550164 anytime.")
    phones = [s for s in spans if s.type == "phone"]
    assert len(phones) == 1
    assert phones[0].text == "6025550164"


# ---------------------------------------------------------------------------
# merge_spans: overlapping phone-inside-address keeps the longer
# ---------------------------------------------------------------------------


def test_merge_spans_keeps_longer_on_overlap() -> None:
    # A phone-like span nested inside a longer address span.
    address = PiiSpan(type="address", start=0, end=20, text="123 East Main Street")
    phone = PiiSpan(type="phone", start=0, end=3, text="123")
    merged = merge_spans([phone, address])
    assert merged == [address]


def test_merge_spans_keeps_disjoint_spans() -> None:
    a = PiiSpan(type="phone", start=0, end=10, text="a" * 10)
    b = PiiSpan(type="email", start=15, end=25, text="b" * 10)
    merged = merge_spans([b, a])
    assert merged == [a, b]


def test_merge_spans_empty() -> None:
    assert merge_spans([]) == []


# ---------------------------------------------------------------------------
# redact_text with use_ner=False is deterministic
# ---------------------------------------------------------------------------


def test_redact_text_no_ner_is_deterministic() -> None:
    first = redact_text(_PARAGRAPH, use_ner=False)
    second = redact_text(_PARAGRAPH, use_ner=False)
    assert first == second
    redacted, spans = first
    assert len(spans) == 7  # 3 phone + 2 email + 2 address
    assert "[PHONE]" in redacted and "[EMAIL]" in redacted and "[ADDRESS]" in redacted


def test_redact_text_no_ner_does_not_call_detect_persons(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(text: str) -> list[PiiSpan]:
        raise AssertionError("detect_persons must not run when use_ner=False")

    monkeypatch.setattr(pii, "detect_persons", boom)
    redact_text(_PARAGRAPH, use_ner=False)


# ---------------------------------------------------------------------------
# detect_persons with a fake transformers pipeline (no real model)
# ---------------------------------------------------------------------------


def _install_fake_transformers(monkeypatch: pytest.MonkeyPatch, entities: list[dict]) -> dict:
    """Inject a fake transformers module whose pipeline returns fixed entities."""
    calls: dict = {}

    def fake_pipeline(task: str, model: str | None = None, aggregation_strategy: str | None = None):
        calls["task"] = task
        calls["model"] = model
        calls["aggregation_strategy"] = aggregation_strategy

        def run(text: str) -> list[dict]:
            calls["text"] = text
            return entities

        return run

    fake_module = types.ModuleType("transformers")
    fake_module.pipeline = fake_pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    pii._get_ner_pipeline.cache_clear()
    return calls


@pytest.fixture(autouse=True)
def _clear_ner_cache() -> None:
    pii._get_ner_pipeline.cache_clear()
    yield
    pii._get_ner_pipeline.cache_clear()


def _entity(text: str, phrase: str, group: str, score: float) -> dict:
    start = text.index(phrase)
    return {"entity_group": group, "score": score, "start": start, "end": start + len(phrase)}


def test_detect_persons_maps_per_and_applies_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Councilmember Maria Lopez thanked Bob Smith for the report."
    entities = [
        _entity(text, "Maria Lopez", "PER", 0.99),
        _entity(text, "Bob Smith", "PER", 0.80),
        _entity(text, "Councilmember", "ORG", 0.99),
    ]
    calls = _install_fake_transformers(monkeypatch, entities)

    spans = detect_persons(text)

    # Below-threshold PER dropped; ORG dropped; only high-confidence PER survives.
    assert len(spans) == 1
    assert spans[0].type == "person"
    assert spans[0].text == "Maria Lopez"
    # Pipeline was constructed with the spec's arguments (model from settings).
    assert calls["task"] == "token-classification"
    assert calls["aggregation_strategy"] == "simple"
    assert calls["model"] == pii.get_settings().ner_model


def test_detect_persons_returns_empty_on_pipeline_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_pipeline(*args: object, **kwargs: object):
        raise RuntimeError("model missing / offline")

    fake_module = types.ModuleType("transformers")
    fake_module.pipeline = fake_pipeline  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_module)
    pii._get_ner_pipeline.cache_clear()

    # Regex PII must still work; persons just degrade to [].
    assert detect_persons("Maria Lopez spoke.") == []


def test_redact_text_with_ner_merges_person_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Maria Lopez lives at 123 East Main Street, call 602-555-1234."
    entities = [_entity(text, "Maria Lopez", "PER", 0.97)]
    _install_fake_transformers(monkeypatch, entities)

    redacted, spans = redact_text(text, use_ner=True)
    types_seen = {s.type for s in spans}
    assert types_seen == {"person", "address", "phone"}
    assert redacted.startswith("[PERSON] lives at [ADDRESS], call [PHONE].")


# ---------------------------------------------------------------------------
# redact_chunk_records: copies untouched originals + quarantine only for PII chunks
# ---------------------------------------------------------------------------


def test_redact_chunk_records_copies_and_quarantines() -> None:
    chunks = [
        ChunkRecord(chunk_index=0, text="Call me at 602-555-1234.", t_start=1.0, t_end=2.0),
        ChunkRecord(chunk_index=1, text="The council approved the agenda.", page_no=3),
        ChunkRecord(chunk_index=2, text="Email jane.doe@example.com please.", topic="budget"),
    ]
    redacted, quarantine = redact_chunk_records(chunks, use_ner=False)

    # Originals untouched.
    assert chunks[0].text == "Call me at 602-555-1234."
    assert chunks[2].text == "Email jane.doe@example.com please."

    # New objects, not aliases.
    assert redacted[0] is not chunks[0]
    assert redacted[0].text == "Call me at [PHONE]."
    assert redacted[1].text == "The council approved the agenda."  # untouched, no PII
    assert redacted[2].text == "Email [EMAIL] please."

    # Metadata carried over onto the copies.
    assert (redacted[0].t_start, redacted[0].t_end) == (1.0, 2.0)
    assert redacted[1].page_no == 3
    assert redacted[2].topic == "budget"

    # Quarantine only for chunks 0 and 2; chunk_index preserved.
    assert [q["chunk_index"] for q in quarantine] == [0, 2]
    q0 = quarantine[0]
    assert q0["original_text"] == "Call me at 602-555-1234."
    assert q0["redacted_text"] == "Call me at [PHONE]."
    assert q0["spans"] == [{"type": "phone", "start": 11, "end": 23, "text": "602-555-1234"}]


def test_redact_chunk_records_empty() -> None:
    redacted, quarantine = redact_chunk_records([], use_ner=False)
    assert redacted == []
    assert quarantine == []


# ---------------------------------------------------------------------------
# seeded eval + scorer (deterministic, regex-only)
# ---------------------------------------------------------------------------


def test_build_seeded_pii_testset_injects_at_known_offsets() -> None:
    base = ["The council heard from residents at the podium during public comment."]
    # default (regex-only): 3 types (phone/email/address) x 3 each, no person
    testset = build_seeded_pii_testset(base, n_per_type=3)
    assert len(testset) == 9
    for item in testset:
        assert len(item["gold_spans"]) == 1
        gold = item["gold_spans"][0]
        # Gold offsets slice out exactly the injected token.
        assert item["text"][gold["start"] : gold["end"]] == gold["text"]
        assert gold["type"] in {"phone", "email", "address"}
        assert item["seed_type"] == gold["type"]


def test_build_seeded_pii_testset_person_only_with_ner_flag() -> None:
    base = ["The council heard from residents at the podium during public comment."]
    with_person = build_seeded_pii_testset(base, n_per_type=3, include_person=True)
    # 4 types x 3 each.
    assert len(with_person) == 12
    person_items = [it for it in with_person if it["seed_type"] == "person"]
    # person items use neutral carriers (no real-name confound), not the base text
    assert len(person_items) == 3
    assert all("podium" not in it["text"] for it in person_items)


def test_score_pii_detection_perfect_on_seeded_regex() -> None:
    base = ["Residents lined up to speak on the proposed ordinance this evening."]
    testset = build_seeded_pii_testset(base, n_per_type=10)
    scores = score_pii_detection(testset, use_ner=False)
    for pii_type in ("phone", "email", "address"):
        assert scores[pii_type]["recall"] == 1.0, pii_type
        assert scores[pii_type]["fn"] == 0, pii_type


def test_score_pii_detection_math_hand_built() -> None:
    # Two items with known gold; predictions are computed by the real regex detectors.
    testset = [
        # 1 phone gold, detector finds it -> tp.
        {
            "text": "Reach me at 602-555-1234 tonight.",
            "gold_spans": [asdict(PiiSpan("phone", 12, 24, "602-555-1234"))],
        },
        # 1 email gold, but we mislabel the gold as 'address' -> the email prediction
        # is an FP for 'email' and the gold is an FN for 'address'.
        {
            "text": "Write to clerk@example.com anytime.",
            "gold_spans": [asdict(PiiSpan("address", 9, 26, "clerk@example.com"))],
        },
    ]
    scores = score_pii_detection(testset, use_ner=False)
    # phone: one true positive.
    assert scores["phone"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "tp": 1,
        "fp": 0,
        "fn": 0,
    }
    # email: the detected email has no same-type gold -> false positive.
    assert scores["email"]["tp"] == 0
    assert scores["email"]["fp"] == 1
    assert scores["email"]["precision"] == 0.0
    # address: gold never matched -> false negative.
    assert scores["address"]["fn"] == 1
    assert scores["address"]["recall"] == 0.0
    # overall aggregates: tp=1, fp=1, fn=1.
    assert scores["overall"]["tp"] == 1
    assert scores["overall"]["fp"] == 1
    assert scores["overall"]["fn"] == 1
    assert scores["overall"]["precision"] == 0.5
    assert scores["overall"]["recall"] == 0.5


def test_score_pii_detection_overlap_counts_as_hit() -> None:
    # Prediction partially overlaps gold of the same type -> still a hit.
    testset = [
        {
            "text": "Call 602-555-1234 now.",
            "gold_spans": [asdict(PiiSpan("phone", 5, 13, "602-555-"))],
        }
    ]
    scores = score_pii_detection(testset, use_ner=False)
    assert scores["phone"]["tp"] == 1
    assert scores["phone"]["fn"] == 0
