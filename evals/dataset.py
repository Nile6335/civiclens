"""Golden dataset schema, IO, and span resolution.

Items reference supporting spans by NATURAL KEY (city, source_type, meeting_id,
chunk_index) — never by chunks.id — so the dataset survives re-ingestion. Items whose
spans come from the bundled sample corpus are marked sample=true; CI evaluates only
those (it never has the live corpus).
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import psycopg

DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class Span:
    city: str
    source_type: str
    meeting_id: str
    chunk_index: int
    text_snippet: str  # first ~200 chars, for human review and drift detection


@dataclass
class GoldenItem:
    id: str
    question: str
    answer: str
    difficulty: str  # "easy" | "multi-hop"
    source_type: str  # dominant evidence type: transcript | pdf | table
    city: str
    spans: list[Span] = field(default_factory=list)
    table_name: str | None = None  # for table-derived items
    sample: bool = False  # answerable from the bundled sample corpus alone
    validated: bool = False  # passed the LLM-as-judge pass
    judge_notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def save_dataset(items: list[GoldenItem], path: Path = DATASET_PATH) -> None:
    path.write_text(json.dumps([i.to_dict() for i in items], indent=1), encoding="utf-8")


def load_dataset(path: Path = DATASET_PATH) -> list[GoldenItem]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for row in raw:
        spans = [Span(**s) for s in row.pop("spans", [])]
        items.append(GoldenItem(spans=spans, **row))
    return items


def resolve_span_chunk_ids(conn: psycopg.Connection, spans: list[Span]) -> set[int]:
    """Map natural-key spans to current chunks.id values (empty when not ingested)."""
    ids: set[int] = set()
    for span in spans:
        row = conn.execute(
            """
            SELECT c.id FROM chunks c JOIN sources s ON s.id = c.source_id
            WHERE s.city = %s AND s.source_type = %s AND s.meeting_id = %s
              AND c.chunk_index = %s
            """,
            (span.city, span.source_type, span.meeting_id, span.chunk_index),
        ).fetchone()
        if row:
            ids.add(row[0])
    return ids


def _answer_words(answer: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", answer.lower()) if len(w) >= 3]
    return [w for w in words if not w.isdigit()]


def expanded_relevant_ids(conn: psycopg.Connection, item: GoldenItem) -> set[int]:
    """Answer-bearing relevance, the open-domain-QA convention.

    Councils repeat names and topics across a meeting (and the agenda PDF restates what
    the transcript says), so judging only the exact generation-time span produces false
    negatives for retrievals that genuinely contain the answer. Relevant =
    span chunks ∪ same-source ±1 window neighbours ∪ corpus-wide chunks containing the
    answer text (only for answers with ≥2 non-numeric content words — short/numeric
    answers like "2009" would match everywhere).
    """
    ids = resolve_span_chunk_ids(conn, item.spans)
    for span in item.spans:
        rows = conn.execute(
            """
            SELECT c.id FROM chunks c JOIN sources s ON s.id = c.source_id
            WHERE s.city = %s AND s.source_type = %s AND s.meeting_id = %s
              AND c.chunk_index BETWEEN %s AND %s
            """,
            (
                span.city,
                span.source_type,
                span.meeting_id,
                span.chunk_index - 1,
                span.chunk_index + 1,
            ),
        ).fetchall()
        ids.update(r[0] for r in rows)
    words = _answer_words(item.answer)
    if len(words) >= 2:
        conditions = " AND ".join(["c.text ILIKE %s"] * len(words))
        rows = conn.execute(
            f"SELECT c.id FROM chunks c WHERE {conditions}",  # noqa: S608
            tuple(f"%{w}%" for w in words),
        ).fetchall()
        ids.update(r[0] for r in rows)
    return ids
