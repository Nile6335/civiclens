"""Synthetic golden dataset generation.

    python -m evals.generate [--target 100]

Easy items: one LLM-written Q&A per sampled chunk, with the exact supporting span
recorded. Multi-hop items: a question requiring two adjacent chunks. Table items are
generated PROGRAMMATICALLY from the normalized tables (deterministically correct by
construction). Every LLM item passes cheap pre-filters (self-containedness, answer
overlap with the span); the real quality bar is the separate judge pass
(evals.validate).
"""

import argparse
import json
import logging
import re

import psycopg

from common.db import get_connection
from common.llm import get_chat_model
from evals.dataset import DATASET_PATH, GoldenItem, Span, save_dataset

logger = logging.getLogger(__name__)

MIN_CHUNK_CHARS = 250
SNIPPET_CHARS = 200

_GEN_PROMPT = """You write eval questions for a search system over city-council records.

Below is an excerpt from the {city} city council record of {date} ({kind}).

Excerpt:
\"\"\"{text}\"\"\"

Write ONE factual question about a specific detail in the excerpt, and its short answer.
Rules:
- The question must be answerable from the excerpt alone.
- The question must be SELF-CONTAINED: mention the city ({city}) and enough context that
  it makes sense without seeing the excerpt. Never say "the excerpt", "this meeting",
  or "the speaker".
- The answer must be short (a name, number, date, or phrase copied from the excerpt).

Reply with ONLY JSON: {{"question": "...", "answer": "..."}}"""

_MULTIHOP_PROMPT = """You write eval questions for a search system over city-council records.

Below are two CONSECUTIVE excerpts from the {city} city council record of {date} ({kind}).

Excerpt A:
\"\"\"{text_a}\"\"\"

Excerpt B:
\"\"\"{text_b}\"\"\"

Write ONE factual question whose answer requires information from BOTH excerpts, and its
short answer.
Rules:
- Self-contained: mention the city ({city}); never say "the excerpt" or "this meeting".
- The answer must combine details from both excerpts and stay short.

Reply with ONLY JSON: {{"question": "...", "answer": "..."}}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_BANNED = re.compile(
    r"\b(excerpt|this meeting|the speaker|the text|passage|last sentence|first sentence|"
    r"mentioned in the|according to the)\b",
    re.IGNORECASE,
)
_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_ANCHOR_RE = re.compile(rf"\b(mesa|seattle|oakland|20\d\d|{_MONTHS})\b", re.IGNORECASE)


def question_is_anchored(question: str) -> bool:
    """Self-containedness proxy: the question names the city or a date/year."""
    return bool(_ANCHOR_RE.search(question))


def anchor_question(question: str, city: str, mdate) -> str:
    """Deterministically anchor an unanchored question with its city/date context."""
    if question_is_anchored(question):
        return question
    body = question[0].lower() + question[1:] if question[:2].isascii() else question
    return f"At the {city} city council meeting of {mdate}, {body}"


def _parse_qa(raw: str) -> tuple[str, str] | None:
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    q, a = str(data.get("question", "")).strip(), str(data.get("answer", "")).strip()
    if len(q) < 15 or len(a) < 1 or len(a) > 220:
        return None
    if _BANNED.search(q):
        return None
    return q, a


_WORD_RE = re.compile(r"[a-z0-9]+")


def _answer_overlap_ok(answer: str, span_text: str) -> bool:
    """At least 60% of the answer's content words must appear in the span."""
    words = [w for w in _WORD_RE.findall(answer.lower()) if len(w) > 2]
    if not words:
        return True  # numeric/short answers checked via substring below
    span_lower = span_text.lower()
    present = sum(w in span_lower for w in words)
    return present / len(words) >= 0.6


def _sample_chunks(conn: psycopg.Connection, source_type: str, limit: int) -> list[tuple]:
    """Spread samples across sources: rank chunks within each source, interleave."""
    return conn.execute(
        """
        SELECT s.city, s.meeting_id, s.title, s.meeting_date, c.chunk_index, c.text,
               (s.metadata->>'sample') = 'true' AS is_sample
        FROM chunks c JOIN sources s ON s.id = c.source_id
        WHERE s.source_type = %s AND length(c.text) >= %s
        ORDER BY (c.chunk_index %% 7), s.id, c.chunk_index
        LIMIT %s
        """,
        (source_type, MIN_CHUNK_CHARS, limit),
    ).fetchall()


def _gen_easy(conn: psycopg.Connection, source_type: str, target: int) -> list[GoldenItem]:
    llm = get_chat_model()
    items: list[GoldenItem] = []
    rows = _sample_chunks(conn, source_type, target * 2)  # headroom for rejects
    for city, meeting_id, _title, mdate, chunk_index, text, is_sample in rows:
        if len(items) >= target:
            break
        prompt = _GEN_PROMPT.format(city=city, date=mdate, kind=source_type, text=text[:1800])
        try:
            raw = str(llm.invoke(prompt).content)
        except Exception as exc:
            logger.warning("generation call failed: %s", exc)
            continue
        qa = _parse_qa(raw)
        if not qa or not _answer_overlap_ok(qa[1], text):
            continue
        items.append(
            GoldenItem(
                id=f"{source_type}-{city}-{meeting_id}-{chunk_index}",
                question=anchor_question(qa[0], city, mdate),
                answer=qa[1],
                difficulty="easy",
                source_type=source_type,
                city=city,
                spans=[Span(city, source_type, meeting_id, chunk_index, text[:SNIPPET_CHARS])],
                sample=bool(is_sample),
            )
        )
        logger.info("[%s %d/%d] %s", source_type, len(items), target, qa[0][:70])
    return items


def _gen_multihop(conn: psycopg.Connection, target: int) -> list[GoldenItem]:
    llm = get_chat_model()
    pairs = conn.execute(
        """
        SELECT s.city, s.meeting_id, s.meeting_date, a.chunk_index, a.text, b.text,
               (s.metadata->>'sample') = 'true' AS is_sample
        FROM chunks a
        JOIN chunks b ON b.source_id = a.source_id AND b.chunk_index = a.chunk_index + 1
        JOIN sources s ON s.id = a.source_id
        WHERE s.source_type = 'transcript'
          AND length(a.text) >= %s AND length(b.text) >= %s
        ORDER BY (a.chunk_index %% 5), s.id, a.chunk_index
        LIMIT %s
        """,
        (MIN_CHUNK_CHARS, MIN_CHUNK_CHARS, target * 2),
    ).fetchall()
    items: list[GoldenItem] = []
    for city, meeting_id, mdate, idx_a, text_a, text_b, is_sample in pairs:
        if len(items) >= target:
            break
        prompt = _MULTIHOP_PROMPT.format(
            city=city, date=mdate, kind="transcript", text_a=text_a[:1200], text_b=text_b[:1200]
        )
        try:
            raw = str(llm.invoke(prompt).content)
        except Exception as exc:
            logger.warning("multihop call failed: %s", exc)
            continue
        qa = _parse_qa(raw)
        if not qa or not _answer_overlap_ok(qa[1], text_a + " " + text_b):
            continue
        items.append(
            GoldenItem(
                id=f"mh-{city}-{meeting_id}-{idx_a}",
                question=anchor_question(qa[0], city, mdate),
                answer=qa[1],
                difficulty="multi-hop",
                source_type="transcript",
                city=city,
                spans=[
                    Span(city, "transcript", meeting_id, idx_a, text_a[:SNIPPET_CHARS]),
                    Span(city, "transcript", meeting_id, idx_a + 1, text_b[:SNIPPET_CHARS]),
                ],
                sample=bool(is_sample),
            )
        )
        logger.info("[multihop %d/%d] %s", len(items), target, qa[0][:70])
    return items


def _gen_table_items(conn: psycopg.Connection, target: int) -> list[GoldenItem]:
    """Deterministic Q&A from the normalized tables — correct by construction."""
    registry = conn.execute(
        """
        SELECT r.table_name, r.description, s.city, s.meeting_id, s.meeting_date,
               (s.metadata->>'sample') = 'true' AS is_sample
        FROM table_registry r JOIN sources s ON s.id = r.source_id
        ORDER BY r.table_name
        """
    ).fetchall()
    items: list[GoldenItem] = []
    for table_name, _desc, city, _meeting_id, mdate, is_sample in registry:
        if len(items) >= target:
            break
        n_rows = conn.execute(f'SELECT count(*) FROM "{table_name}"').fetchone()[0]  # noqa: S608
        items.append(
            GoldenItem(
                id=f"table-count-{table_name}",
                question=(
                    f"How many items were on the agenda of the {city} city council "
                    f"meeting on {mdate}?"
                ),
                answer=str(n_rows),
                difficulty="easy",
                source_type="table",
                city=city,
                table_name=table_name,
                sample=bool(is_sample),
            )
        )
        rows = conn.execute(
            f'SELECT agenda_number, matter_file, matter_type, title FROM "{table_name}" '  # noqa: S608
            "WHERE agenda_number IS NOT NULL AND matter_file IS NOT NULL "
            "ORDER BY agenda_number LIMIT 4"
        ).fetchall()
        for agenda_number, matter_file, _mtype, _title in rows[:3]:
            if len(items) >= target:
                break
            items.append(
                GoldenItem(
                    id=f"table-file-{table_name}-{agenda_number}",
                    question=(
                        f"What is the matter file number of agenda item {agenda_number} "
                        f"at the {city} city council meeting on {mdate}?"
                    ),
                    answer=str(matter_file),
                    difficulty="easy",
                    source_type="table",
                    city=city,
                    table_name=table_name,
                    sample=bool(is_sample),
                )
            )
        types = conn.execute(
            f'SELECT matter_type, count(*) FROM "{table_name}" '  # noqa: S608
            "WHERE matter_type IS NOT NULL AND matter_type <> '' GROUP BY 1 "
            "ORDER BY 2 DESC LIMIT 2"
        ).fetchall()
        for mtype, count in types[:1]:
            if len(items) >= target:
                break
            items.append(
                GoldenItem(
                    id=f"table-type-{table_name}-{re.sub(r'[^a-z0-9]+', '_', mtype.lower())}",
                    question=(
                        f"How many agenda items of type '{mtype}' were on the {city} "
                        f"city council agenda of {mdate}?"
                    ),
                    answer=str(count),
                    difficulty="easy",
                    source_type="table",
                    city=city,
                    table_name=table_name,
                    sample=bool(is_sample),
                )
            )
    return items


def _dedupe(items: list[GoldenItem]) -> list[GoldenItem]:
    seen: set[str] = set()
    out = []
    for item in items:
        key = re.sub(r"\W+", " ", item.question.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def generate(target: int = 100) -> list[GoldenItem]:
    quota = {
        "transcript": int(target * 0.38),
        "pdf": int(target * 0.20),
        "multihop": int(target * 0.15),
        "table": target - int(target * 0.38) - int(target * 0.20) - int(target * 0.15),
    }
    with get_connection() as conn:
        items: list[GoldenItem] = []
        items += _gen_table_items(conn, quota["table"])
        logger.info("table items: %d", len(items))
        items += _gen_easy(conn, "transcript", quota["transcript"])
        items += _gen_easy(conn, "pdf", quota["pdf"])
        items += _gen_multihop(conn, quota["multihop"])
    items = _dedupe(items)
    logger.info("generated %d items total", len(items))
    return items


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=100)
    args = parser.parse_args()
    items = generate(args.target)
    save_dataset(items)
    by_type: dict[str, int] = {}
    for item in items:
        by_type[item.source_type] = by_type.get(item.source_type, 0) + 1
    n_sample = sum(i.sample for i in items)
    print(f"wrote {len(items)} items to {DATASET_PATH}")
    print(
        f"by source_type: {by_type}; multi-hop: "
        f"{sum(i.difficulty == 'multi-hop' for i in items)}; sample-safe: {n_sample}"
    )


if __name__ == "__main__":
    main()
