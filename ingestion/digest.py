"""Weekly digest generator: a markdown briefing of recent meetings, written to file.

    python -m ingestion.digest [--days 7] [--out digests/]

No email credentials required (stretch-phase spec): output is a file. Summaries are
extractive-first (deterministic) with an optional LLM polish that degrades gracefully.
"""

import argparse
import logging
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from common.db import get_connection

logger = logging.getLogger(__name__)


def _llm_summary(text: str) -> str | None:
    """One-paragraph LLM summary; None when the LLM is unavailable or rambles."""
    try:
        from common.llm import get_chat_model

        prompt = (
            "Summarize this city-council meeting excerpt in 2 concise sentences, "
            "plain facts only:\n\n" + text[:2500] + "\n\nSummary:"
        )
        out = str(get_chat_model().invoke(prompt).content).strip()
        return out if 20 < len(out) < 600 else None
    except Exception as exc:
        logger.warning("LLM summary unavailable: %s", exc)
        return None


def build_digest(days: int = 7, until: date | None = None) -> str:
    """Markdown digest of meetings whose date falls in the trailing window."""
    until = until or date.today()
    since = until - timedelta(days=days)
    lines = [f"# CivicLens digest — meetings {since} to {until}", ""]
    with get_connection() as conn:
        meetings = conn.execute(
            """
            SELECT DISTINCT s.city, s.meeting_id, s.meeting_date
            FROM sources s
            WHERE s.meeting_date > %s AND s.meeting_date <= %s
            ORDER BY s.meeting_date DESC, s.city
            """,
            (since, until),
        ).fetchall()
        if not meetings:
            lines.append("_No meetings in the record for this window._")
        for city, meeting_id, mdate in meetings:
            row = conn.execute(
                """
                SELECT title, url FROM sources
                WHERE city = %s AND meeting_id = %s AND source_type = 'transcript' LIMIT 1
                """,
                (city, meeting_id),
            ).fetchone()
            title = row[0] if row else f"{city} meeting {meeting_id}"
            url = row[1] if row else None
            lines.append(f"## {title} ({city}, {mdate})")
            if url:
                lines.append(f"[watch]({url})")
            topics = conn.execute(
                """
                SELECT c.topic FROM chunks c JOIN sources s ON s.id = c.source_id
                WHERE s.city = %s AND s.meeting_id = %s AND c.topic IS NOT NULL
                  AND c.topic <> 'other'
                """,
                (city, meeting_id),
            ).fetchall()
            top = Counter(t[0] for t in topics).most_common(3)
            if top:
                lines.append("Topics: " + ", ".join(f"{t} ({n})" for t, n in top))
            n_items = conn.execute(
                """
                SELECT count(*) FROM table_registry r JOIN sources s ON s.id = r.source_id
                WHERE s.city = %s AND s.meeting_id = %s
                """,
                (city, meeting_id),
            ).fetchone()[0]
            if n_items:
                lines.append(f"Agenda-item tables on record: {n_items}")
            first_chunk = conn.execute(
                """
                SELECT c.text FROM chunks c JOIN sources s ON s.id = c.source_id
                WHERE s.city = %s AND s.meeting_id = %s AND s.source_type = 'transcript'
                ORDER BY c.chunk_index LIMIT 1
                """,
                (city, meeting_id),
            ).fetchone()
            if first_chunk:
                summary = _llm_summary(first_chunk[0])
                lines.append(summary if summary else f"> {first_chunk[0][:280]}…")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--until", type=str, default=None, help="YYYY-MM-DD window end")
    parser.add_argument("--out", type=Path, default=Path("digests"))
    args = parser.parse_args()
    until = date.fromisoformat(args.until) if args.until else date.today()
    digest = build_digest(args.days, until)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"civiclens-digest-{until.isoformat()}.md"
    path.write_text(digest, encoding="utf-8")
    print(f"wrote {path} ({len(digest)} chars)")


if __name__ == "__main__":
    main()
