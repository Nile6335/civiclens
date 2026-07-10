"""Evidence and citation formatting shared by all specialist agents.

Canonical citation forms (the integration tests regex against these):
  video:  [video @ mm:ss](url_with_t_param)
  doc:    [doc, p.N]
  table:  [table: civic_tbl_name]

Synthesis works with evidence markers ([E1], [E2], ...) which are deterministically
rewritten into the canonical forms — well-formed citations never depend on the LLM
reproducing URLs correctly.
"""

import re
from dataclasses import dataclass, field
from typing import Literal

EvidenceKind = Literal["video", "doc", "table"]


def fmt_timestamp(seconds: float) -> str:
    """125 -> '02:05'; 3725 -> '1:02:05'."""
    s = int(seconds)
    if s >= 3600:
        return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"
    return f"{s // 60:02d}:{s % 60:02d}"


def video_url_at(url: str, seconds: float) -> str:
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}t={int(seconds)}s"


@dataclass
class Evidence:
    kind: EvidenceKind
    text: str
    citation: str  # canonical citation string, prebuilt
    score: float = 0.0
    meta: dict = field(default_factory=dict)  # title, city, url, t_start, page_no, table...

    @staticmethod
    def from_video(text: str, url: str, t_start: float, score: float, **meta) -> "Evidence":
        return Evidence(
            kind="video",
            text=text,
            citation=f"[video @ {fmt_timestamp(t_start)}]({video_url_at(url, t_start)})",
            score=score,
            meta={"url": url, "t_start": t_start, **meta},
        )

    @staticmethod
    def from_doc(text: str, page_no: int, score: float, **meta) -> "Evidence":
        return Evidence(
            kind="doc",
            text=text,
            citation=f"[doc, p.{page_no}]",
            score=score,
            meta={"page_no": page_no, **meta},
        )

    @staticmethod
    def from_table(text: str, table_name: str, **meta) -> "Evidence":
        return Evidence(
            kind="table",
            text=text,
            citation=f"[table: {table_name}]",
            meta={"table": table_name, **meta},
        )


_MARKER_RE = re.compile(r"\[E(\d+)\]")

CITATION_RE = re.compile(
    r"\[video @ \d{1,2}:\d{2}(?::\d{2})?\]\([^)]+\)|\[doc, p\.\d+\]|\[table: [a-z0-9_]+\]"
)

NOT_FOUND = "Not found in the record."


def resolve_markers(answer: str, evidence: list[Evidence]) -> str:
    """Rewrite [E#] markers into canonical citations; drop out-of-range markers."""

    def _sub(match: re.Match) -> str:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(evidence):
            return evidence[idx].citation
        return ""

    return _MARKER_RE.sub(_sub, answer)


_QUOTED_RE = re.compile(r"[“\"][^”\"]*[”\"]")


def sentences_without_citation(answer: str) -> list[str]:
    """Sentences making claims with neither a canonical citation nor an [E#] marker.

    Used by tests ('no uncited claims') and by synthesis to decide whether to fall back
    to NOT_FOUND. Short connective/refusal sentences are exempt. Quoted spans are
    treated as atomic — a verbatim quote is covered by the citation that follows it,
    and its internal punctuation must not create phantom sentence breaks.
    """
    flattened = _QUOTED_RE.sub("“…”", answer.strip())
    offenders: list[str] = []
    for raw in re.split(r"(?<=[.!?])\s+", flattened):
        sentence = raw.strip()
        if not sentence or len(sentence) < 40:
            continue
        if NOT_FOUND.lower().rstrip(".") in sentence.lower():
            continue
        if CITATION_RE.search(sentence) or _MARKER_RE.search(sentence):
            continue
        offenders.append(sentence)
    return offenders
