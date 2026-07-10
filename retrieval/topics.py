"""Topic tagging: label chunks with one of TOPIC_LABELS at ingest time.

Two backends, selected via TOPIC_TAGGER: "keyword" (default; deterministic, no
downloads) counts cue hits per label, and "zeroshot" runs an HF zero-shot
classification pipeline over ZERO_SHOT_MODEL (lazy-loaded and cached).
"""

import logging
import re
from functools import cache, lru_cache
from typing import Any

from common.settings import get_settings
from ingestion.models import TOPIC_LABELS, ChunkRecord

logger = logging.getLogger(__name__)

# Strong lowercase cues per non-'other' label. Single words match on word boundaries
# (with an optional plural "s"); multi-word phrases match as substrings.
KEYWORD_MAP: dict[str, list[str]] = {
    "zoning": [
        "rezone",
        "rezoning",
        "zoning",
        "variance",
        "land use",
        "setback",
        "planned development",
        "plat",
        "annexation",
        "easement",
        "conditional use",
        "site plan",
    ],
    "budget": [
        "budget",
        "appropriation",
        "fiscal year",
        "general fund",
        "expenditure",
        "revenue",
        "capital improvement",
        "bond",
        "levy",
        "grant",
        "audit",
        "deficit",
    ],
    "public safety": [
        "police",
        "fire department",
        "emergency",
        "911",
        "crime",
        "public safety",
        "ambulance",
        "dispatch",
        "firefighter",
        "law enforcement",
        "paramedic",
        "patrol",
    ],
    "transportation": [
        "transit",
        "bus",
        "light rail",
        "traffic",
        "street improvement",
        "sidewalk",
        "bike lane",
        "pavement",
        "right-of-way",
        "transportation",
        "intersection",
        "crosswalk",
    ],
    "housing": [
        "housing",
        "affordable housing",
        "homeless",
        "shelter",
        "tenant",
        "rent",
        "eviction",
        "dwelling",
        "landlord",
        "apartment",
        "rental",
        "homelessness",
    ],
}

# Zero-shot models cap out around 512 tokens anyway; keep the input cheap.
_MAX_ZEROSHOT_CHARS = 1000


@cache
def _cue_pattern(cue: str) -> re.Pattern[str]:
    """Compile a cue matcher: word-boundary (plus optional plural) for single words,
    plain substring for phrases."""
    if " " in cue:
        return re.compile(re.escape(cue))
    return re.compile(rf"\b{re.escape(cue)}s?\b")


def tag_text_keyword(text: str) -> str:
    """Return the label whose cues hit most often; 'other' when nothing hits.

    Ties break deterministically by TOPIC_LABELS order.
    """
    lowered = text.lower()
    best_label = "other"
    best_count = 0
    for label in TOPIC_LABELS:
        cues = KEYWORD_MAP.get(label)
        if not cues:
            continue
        count = sum(len(_cue_pattern(cue).findall(lowered)) for cue in cues)
        if count > best_count:
            best_label = label
            best_count = count
    return best_label


@lru_cache(maxsize=1)
def _get_zeroshot_pipeline() -> Any:
    from transformers import pipeline  # lazy: heavy import chain + model download

    settings = get_settings()
    logger.info("loading zero-shot topic model %s", settings.zero_shot_model)
    return pipeline("zero-shot-classification", model=settings.zero_shot_model, device=-1)


def tag_text_zeroshot(text: str) -> str:
    """Classify text against TOPIC_LABELS with the zero-shot pipeline; return top label."""
    classifier = _get_zeroshot_pipeline()
    result = classifier(text[:_MAX_ZEROSHOT_CHARS], TOPIC_LABELS, multi_label=False)
    return str(result["labels"][0])


def tag_text(text: str) -> str:
    """Tag one text with the backend selected by settings.topic_tagger."""
    if get_settings().topic_tagger == "zeroshot":
        return tag_text_zeroshot(text)
    return tag_text_keyword(text)


def tag_chunks(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    """Fill .topic in place on each chunk that doesn't have one yet; return the list."""
    for chunk in chunks:
        if chunk.topic is None:
            chunk.topic = tag_text(chunk.text)
    return chunks
