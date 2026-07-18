"""PII redaction for transcript chunks (Phase 8).

Residents recite home addresses, phone numbers, and emails during public comment.
We redact those spans at ingest, replacing each with a typed placeholder
([PHONE]/[EMAIL]/[ADDRESS]/[PERSON]), and quarantine the originals in the
``pii_quarantine`` table (owner-only; the read-only tabular role can never see them).

Design bias: **recall over precision**. Leaking a resident's phone number is far worse
than over-redacting a street name, so the regexes are deliberately generous and the NER
threshold is the only precision lever. The seeded-eval helpers below measure the
resulting per-type precision/recall so the tradeoff is documented, not guessed. Persons
detection is optional (transformers NER); it degrades gracefully to regex-only PII when
the model is missing or offline — the regex detectors never depend on it.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import psycopg

from common.settings import get_settings
from ingestion.models import ChunkRecord

logger = logging.getLogger(__name__)

PiiType = str  # one of: 'phone' | 'email' | 'address' | 'person'

_PLACEHOLDER = {
    "phone": "[PHONE]",
    "email": "[EMAIL]",
    "address": "[ADDRESS]",
    "person": "[PERSON]",
}

# Minimum NER confidence to accept a PER entity. Public-comment transcripts are noisy;
# 0.9 trims low-confidence false positives while keeping recall high.
_PERSON_SCORE_THRESHOLD = 0.9


@dataclass
class PiiSpan:
    """A detected PII span: half-open [start, end) offsets into the source text."""

    type: PiiType
    start: int
    end: int
    text: str


# ---------------------------------------------------------------------------
# regex detectors (compiled once)
# ---------------------------------------------------------------------------

# Phone: US formats, generous. Covers (602) 555-1234, 602-555-1234, 602.555.1234,
# +1 602 555 1234, and bare 10-digit runs 5555551234. A leading country code and the
# area-code parentheses are optional; separators may be space, dot, or hyphen. Word
# boundaries keep "meet me at 5" and years like 2024 from matching.
_PHONE_RE = re.compile(
    r"""
    (?<!\d)                       # not mid-number
    (?:\+?1[\s.\-]*)?             # optional US country code
    (?:
        \(\d{3}\)[\s.\-]*         # (602) with optional trailing separator
      | \d{3}[\s.\-]              # 602- / 602. / 602<space>
    )
    \d{3}[\s.\-]\d{4}             # 555-1234 / 555.1234 / 555 1234
    (?!\d)
    |
    (?<!\d)\d{10}(?!\d)           # bare 10-digit run
    """,
    re.VERBOSE,
)

# Email: pragmatic standard address form.
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
)

# Street-type suffixes (long form + common abbreviation). Trailing dot on the
# abbreviation is optional (matched by the surrounding pattern).
_STREET_SUFFIX = (
    r"Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|"
    r"Way|Place|Pl|Circle|Cir|Terrace|Ter"
)
# Optional compass direction between the number and the street name.
_DIRECTION = r"(?:N|S|E|W|NE|NW|SE|SW|North|South|East|West)"

# Address: number, optional direction, one-or-more name words (ordinals/words), then a
# street-type suffix. Case-insensitive. The trailing \b (plus optional '.') stops
# "Street smarts" from matching — the suffix must end a word, not start one.
_ADDRESS_RE = re.compile(
    rf"""
    \b\d{{1,6}}                             # house number
    (?:\s+{_DIRECTION})?                     # optional direction
    (?:\s+[A-Za-z0-9][A-Za-z0-9'.\-]*){{1,4}} # 1-4 name words (incl. 2nd, Martin L.)
    \s+(?:{_STREET_SUFFIX})\b\.?             # street-type suffix, ends a word
    """,
    re.VERBOSE | re.IGNORECASE,
)

_REGEX_DETECTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", _EMAIL_RE),
    ("phone", _PHONE_RE),
    ("address", _ADDRESS_RE),
)


def detect_regex(text: str) -> list[PiiSpan]:
    """Detect phone/email/address spans with the compiled regexes, sorted by start.

    On exact-offset ties, the longer span sorts first so ``merge_spans`` keeps it.
    """
    spans: list[PiiSpan] = []
    for pii_type, pattern in _REGEX_DETECTORS:
        for match in pattern.finditer(text):
            spans.append(
                PiiSpan(type=pii_type, start=match.start(), end=match.end(), text=match.group(0))
            )
    spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
    return spans


# ---------------------------------------------------------------------------
# NER person detector (lazy, optional)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_ner_pipeline() -> Any:
    """Lazily build and cache the transformers token-classification pipeline.

    Imported here, never at module import time, so tests (and regex-only ingest paths)
    never pull in transformers or download a model.
    """
    from transformers import pipeline  # lazy: heavy import chain + model download

    settings = get_settings()
    logger.info("loading PII NER model %s", settings.ner_model)
    return pipeline(
        "token-classification",
        model=settings.ner_model,
        aggregation_strategy="simple",
    )


def detect_persons(text: str) -> list[PiiSpan]:
    """Detect person-name spans via the NER pipeline; [] on any failure.

    Keeps only PER entities scoring >= 0.9. If the model is missing or the backend is
    offline we log a warning and return [] — regex PII must still work without NER.
    """
    if not text:
        return []
    try:
        pipe = _get_ner_pipeline()
        entities = pipe(text)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never block ingest
        logger.warning("PII NER unavailable (%s); skipping person detection", exc)
        return []

    spans: list[PiiSpan] = []
    for ent in entities:
        group = str(ent.get("entity_group") or ent.get("entity") or "")
        if "PER" not in group.upper():
            continue
        if float(ent.get("score", 0.0)) < _PERSON_SCORE_THRESHOLD:
            continue
        start = int(ent["start"])
        end = int(ent["end"])
        spans.append(PiiSpan(type="person", start=start, end=end, text=text[start:end]))
    spans.sort(key=lambda s: (s.start, -(s.end - s.start)))
    return spans


# ---------------------------------------------------------------------------
# merge + redact
# ---------------------------------------------------------------------------


def merge_spans(spans: list[PiiSpan]) -> list[PiiSpan]:
    """Sort spans and drop overlaps, keeping the longer of any overlapping pair."""
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    merged: list[PiiSpan] = []
    for span in ordered:
        if merged and span.start < merged[-1].end:
            # Overlap with the last kept span: keep whichever is longer.
            if (span.end - span.start) > (merged[-1].end - merged[-1].start):
                merged[-1] = span
            continue
        merged.append(span)
    return merged


def redact(text: str, spans: list[PiiSpan]) -> str:
    """Replace each span with its typed placeholder, right-to-left so offsets stay valid."""
    result = text
    for span in sorted(spans, key=lambda s: s.start, reverse=True):
        placeholder = _PLACEHOLDER.get(span.type, "[REDACTED]")
        result = result[: span.start] + placeholder + result[span.end :]
    return result


def redact_text(text: str, use_ner: bool = True) -> tuple[str, list[PiiSpan]]:
    """Detect (regex, plus NER persons when ``use_ner``), merge, and redact.

    Returns ``(redacted_text, merged_spans)``.
    """
    spans = detect_regex(text)
    if use_ner:
        spans += detect_persons(text)
    merged = merge_spans(spans)
    return redact(text, merged), merged


def redact_chunk_records(
    chunks: list[ChunkRecord], use_ner: bool = True
) -> tuple[list[ChunkRecord], list[dict]]:
    """Redact every chunk's text into NEW ChunkRecords; originals are left untouched.

    Returns ``(redacted_chunks, quarantine_items)``. Quarantine items are produced only
    for chunks that actually contained PII, each shaped as::

        {chunk_index, original_text, redacted_text, spans: [asdict(span), ...]}

    Only transcript chunks carry public-comment PII in practice, but this redacts
    whatever it is given — the caller decides which source types to pass.
    """
    redacted_chunks: list[ChunkRecord] = []
    quarantine: list[dict] = []
    for chunk in chunks:
        redacted_text, spans = redact_text(chunk.text, use_ner=use_ner)
        new_chunk = ChunkRecord(
            chunk_index=chunk.chunk_index,
            text=redacted_text,
            t_start=chunk.t_start,
            t_end=chunk.t_end,
            page_no=chunk.page_no,
            topic=chunk.topic,
        )
        redacted_chunks.append(new_chunk)
        if spans:
            quarantine.append(
                {
                    "chunk_index": chunk.chunk_index,
                    "original_text": chunk.text,
                    "redacted_text": redacted_text,
                    "spans": [asdict(span) for span in spans],
                }
            )
    return redacted_chunks, quarantine


def store_quarantine(
    conn: psycopg.Connection,
    city: str,
    source_type: str,
    meeting_id: str | None,
    items: list[dict],
) -> int:
    """Upsert quarantine items into ``pii_quarantine``; returns rows written.

    Must use the OWNER connection — the read-only role is revoked on this table. The
    natural key (city, source_type, meeting_id, chunk_index) makes re-ingestion
    idempotent (NULLS NOT DISTINCT so a NULL meeting_id still de-dupes).
    """
    if not items:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO pii_quarantine
                (city, source_type, meeting_id, chunk_index, original_text,
                 redacted_text, spans)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (city, source_type, meeting_id, chunk_index)
            DO UPDATE SET original_text = EXCLUDED.original_text,
                          redacted_text = EXCLUDED.redacted_text,
                          spans = EXCLUDED.spans,
                          created_at = now()
            """,
            [
                (
                    city,
                    source_type,
                    meeting_id,
                    item["chunk_index"],
                    item["original_text"],
                    item["redacted_text"],
                    json.dumps(item["spans"]),
                )
                for item in items
            ],
        )
    return len(items)


# ---------------------------------------------------------------------------
# seeded evaluation (shared with the redteam runner)
# ---------------------------------------------------------------------------

# Synthetic PII strings injected into real transcript text at known offsets. Kept small
# and unambiguous so gold offsets are exact and the scorer is deterministic.
_SEED_TOKENS: dict[str, list[str]] = {
    "phone": [
        "(602) 555-0143",
        "480-555-0198",
        "602.555.0172",
        "+1 928 555 0110",
        "5205550164",
    ],
    "email": [
        "jane.doe@example.com",
        "resident_42@mail.example.org",
        "council.watch@civic.example.net",
    ],
    "address": [
        "123 East Main Street",
        "45 N 2nd Ave",
        "1600 Pennsylvania Ave",
        "78 South Cactus Road",
    ],
    "person": [
        "Marcus Delgado",
        "Priya Ramaswamy",
        "Eleanor Whitfield",
        "Tomás Okonkwo",
        "Rebecca Al-Amin",
    ],
}

# Neutral carrier sentences (no pre-existing names) used ONLY for person seeding, so the
# NER detector is scored against the injected synthetic name alone. Real transcript text is
# full of legitimately-named public officials that NER would (correctly) flag but that are
# not private PII — measuring person precision on that text confounds detector quality with
# the deliberate decision not to redact public figures (see docs/DECISIONS.md).
_PERSON_CARRIERS: list[str] = [
    "A member of the public rose during the comment period and stated concerns.",
    "The speaker asked the council to reconsider the proposed change.",
    "During public comment a resident described the impact on their block.",
    "The next commenter urged the council to delay the vote until spring.",
]


def build_seeded_pii_testset(
    base_texts: list[str],
    n_per_type: int = 50,
    seed_tokens: dict[str, list[str]] | None = None,
    include_person: bool = False,
) -> list[dict]:
    """Inject synthetic PII into copies of real transcript text at known offsets.

    For each PII type we produce ``n_per_type`` items by cycling the seed tokens and the
    base texts, inserting the token at a fixed anchor inside a copy of the base text.
    Each item is ``{text, gold_spans}`` where every gold span records the exact
    ``{type, start, end, text}`` of the injected token — ground truth for scoring.

    Regex types (phone/email/address) are seeded into real transcript text. Person names
    are seeded into neutral carrier sentences (``_PERSON_CARRIERS``) so the NER detector is
    scored against the injected synthetic name alone, without the confound of real
    public-official names that appear throughout genuine transcript prose.

    Person items are only produced when ``include_person`` is set — otherwise the person
    type would appear with gold spans that a regex-only detector can never match, which
    would (correctly but uselessly) report person recall 0. Callers that score with NER
    pass ``include_person=True``; the regex-only CI gate leaves it off.
    """
    tokens = seed_tokens or _SEED_TOKENS
    if not base_texts:
        base_texts = ["The council heard public comment on the item."]

    testset: list[dict] = []
    for pii_type, samples in tokens.items():
        if not samples or (pii_type == "person" and not include_person):
            continue
        carriers = _PERSON_CARRIERS if pii_type == "person" else base_texts
        for i in range(n_per_type):
            token = samples[i % len(samples)]
            base = carriers[i % len(carriers)]
            # Anchor near the middle of the base text (on a word boundary), so the token
            # sits amid real prose rather than at an edge.
            anchor = _mid_word_boundary(base)
            text = f"{base[:anchor]} {token} {base[anchor:]}".strip()
            start = text.index(token)
            gold = PiiSpan(type=pii_type, start=start, end=start + len(token), text=token)
            testset.append({"text": text, "gold_spans": [asdict(gold)], "seed_type": pii_type})
    return testset


def _mid_word_boundary(text: str) -> int:
    """Return an offset near the middle of ``text`` that falls on a space (or 0/len)."""
    if not text:
        return 0
    mid = len(text) // 2
    space = text.find(" ", mid)
    return space if space != -1 else len(text)


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """True if the half-open intervals [a) and [b) overlap."""
    return a_start < b_end and b_start < a_end


def score_pii_detection(testset: list[dict], use_ner: bool = False) -> dict:
    """Score detection against seeded gold spans, per type: precision/recall/f1.

    A predicted span counts as a hit when it overlaps a gold span **of the same type**.
    Each gold span may be matched at most once (no double-counting), and each prediction
    is a true positive only if it matches an as-yet-unmatched gold span. The returned
    dict is ``{type: {precision, recall, f1, tp, fp, fn}}`` plus an ``"overall"`` block.

    Each type is scored only on the items seeded for it (``item['seed_type']``): a person
    detection on a phone-seeded item — which uses real transcript text full of real
    public-official names — is not a person-detection failure, it is the deliberate
    non-redaction of public figures (see docs/DECISIONS.md). Scoping each detector to its
    own inputs measures detector quality without that confound.
    """
    types = ["phone", "email", "address", "person"]
    counts = {t: {"tp": 0, "fp": 0, "fn": 0} for t in types}

    for item in testset:
        gold = [PiiSpan(**g) if isinstance(g, dict) else g for g in item["gold_spans"]]
        _, predicted = redact_text(item["text"], use_ner=use_ner)
        # scope to the item's seeded type when present (falls back to all-types)
        only = item.get("seed_type")
        scoped_pred = [p for p in predicted if only is None or p.type == only]
        _tally(gold, scoped_pred, counts)

    result: dict[str, dict] = {}
    total = {"tp": 0, "fp": 0, "fn": 0}
    for t in types:
        c = counts[t]
        result[t] = _prf(c["tp"], c["fp"], c["fn"])
        for k in total:
            total[k] += c[k]
    result["overall"] = _prf(total["tp"], total["fp"], total["fn"])
    return result


def _tally(gold: list[PiiSpan], predicted: list[PiiSpan], counts: dict[str, dict]) -> None:
    """Greedily match predictions to gold spans of the same type; update tp/fp/fn."""
    matched_gold: set[int] = set()
    for pred in predicted:
        hit = False
        for gi, g in enumerate(gold):
            if gi in matched_gold or g.type != pred.type:
                continue
            if _overlaps(pred.start, pred.end, g.start, g.end):
                matched_gold.add(gi)
                hit = True
                break
        counts[pred.type]["tp" if hit else "fp"] += 1
    for gi, g in enumerate(gold):
        if gi not in matched_gold:
            counts[g.type]["fn"] += 1


def _prf(tp: int, fp: int, fn: int) -> dict:
    """Precision/recall/f1 from tp/fp/fn (0.0 when undefined)."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
