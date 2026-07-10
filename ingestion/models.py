"""Shared data contracts for the ingestion pipelines.

Every pipeline (captions, ASR, PDF, tables) produces these records; store.py is the
single writer that persists them. Keep this module dependency-free (stdlib only).
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

SourceType = Literal["transcript", "pdf", "table"]

TOPIC_LABELS = ["zoning", "budget", "public safety", "transportation", "housing", "other"]


@dataclass
class SourceRecord:
    """One ingested artifact: a meeting transcript, an agenda PDF, or a data table."""

    city: str
    source_type: SourceType
    title: str
    meeting_id: str | None = None
    url: str | None = None
    meeting_date: date | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ChunkRecord:
    """A retrievable unit of text.

    Transcript chunks carry t_start/t_end (seconds into the video); PDF chunks carry
    page_no. topic is filled by the tagger at ingest time.
    """

    chunk_index: int
    text: str
    t_start: float | None = None
    t_end: float | None = None
    page_no: int | None = None
    topic: str | None = None


@dataclass
class Cue:
    """A single timed caption/ASR segment before windowing."""

    start: float
    end: float
    text: str


@dataclass
class RawTable:
    """A table as extracted from a PDF page (or CSV), before normalization."""

    header: list[str]
    rows: list[list[str]]
    page_no: int | None = None
    caption: str | None = None


@dataclass
class ColumnSpec:
    """A column in a normalized table. sql_type is a small allowlisted set."""

    name: str  # snake_case identifier
    sql_type: Literal["text", "numeric", "integer", "date"]


@dataclass
class NormalizedTable:
    """A budget/vote table normalized for text-to-SQL. Becomes civic_tbl_{slug}."""

    slug: str  # lowercase [a-z0-9_]+
    description: str  # human description used by the tabular agent's schema prompt
    columns: list[ColumnSpec]
    rows: list[list]  # row-major values, aligned with columns
