"""Ingestion CLI.

python -m ingestion.cli samples            # bundled corpus (no network)
python -m ingestion.cli live               # real meetings via Legistar + YouTube
"""

import argparse
import logging
import tempfile
from datetime import date, timedelta
from pathlib import Path

from common.db import get_connection, run_migrations
from ingestion.captions import fetch_youtube_captions, transcript_chunks_from_vtt
from ingestion.legistar import LegistarClient, MeetingInfo, pick_responsive_clients
from ingestion.models import ChunkRecord, SourceRecord
from ingestion.pdf import extract_pdf_blocks, extract_pdf_tables
from ingestion.store import (
    corpus_stats,
    create_normalized_table,
    replace_chunks,
    upsert_source,
)
from ingestion.tables import load_csv_table, normalize_raw_table
from ingestion.video_match import find_youtube_video

logger = logging.getLogger("ingestion")

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
SAMPLE_MEETING = {
    "city": "mesa",
    "meeting_id": "4474",
    "date": date(2026, 4, 6),
    "video_url": "https://www.youtube.com/watch?v=4Ey7MKj_n7Y",
    "agenda_url": (
        "https://legistar1.granicus.com/Mesa/meetings/2026/4/"
        "4474_A_City_Council_26-04-06_Meeting_Agenda.pdf"
    ),
}


def _maybe_tag(chunks: list[ChunkRecord]) -> list[ChunkRecord]:
    """Topic-tag chunks at ingest time (Phase 2 tagger; soft dependency)."""
    try:
        from retrieval.topics import tag_chunks
    except ImportError:
        return chunks
    return tag_chunks(chunks)


def ingest_samples() -> dict:
    """Ingest the bundled Mesa 2026-04-06 meeting: captions + agenda PDF + items CSV."""
    run_migrations()
    m = SAMPLE_MEETING
    with get_connection() as conn:
        # transcript
        vtt = SAMPLES / "mesa_council_2026-04-06.en.vtt"
        chunks = _maybe_tag(transcript_chunks_from_vtt(vtt))
        sid = upsert_source(
            conn,
            SourceRecord(
                city=m["city"],
                source_type="transcript",
                title="City Council Meeting 2026-04-06",
                meeting_id=m["meeting_id"],
                url=m["video_url"],
                meeting_date=m["date"],
                metadata={"captions": "youtube-auto", "sample": True},
            ),
        )
        n_t = replace_chunks(conn, sid, chunks)
        logger.info("transcript: %d chunks", n_t)

        # agenda PDF
        pdf_path = SAMPLES / "mesa_council_2026-04-06_agenda.pdf"
        blocks = _maybe_tag(extract_pdf_blocks(pdf_path))
        sid_pdf = upsert_source(
            conn,
            SourceRecord(
                city=m["city"],
                source_type="pdf",
                title="City Council Meeting Agenda 2026-04-06",
                meeting_id=m["meeting_id"],
                url=m["agenda_url"],
                meeting_date=m["date"],
                metadata={"pages": max((b.page_no or 1) for b in blocks), "sample": True},
            ),
        )
        n_p = replace_chunks(conn, sid_pdf, blocks)
        logger.info("pdf: %d chunks", n_p)

        # tables from the PDF (if any well-formed ones) + the agenda-items CSV
        for i, raw in enumerate(extract_pdf_tables(pdf_path)):
            try:
                table = normalize_raw_table(
                    raw,
                    slug=f"mesa_agenda_pdf_{m['meeting_id']}_t{i}",
                    description=f"Table {i} from Mesa council agenda PDF of {m['date']} "
                    f"(page {raw.page_no})",
                )
                create_normalized_table(conn, sid_pdf, table)
            except ValueError as exc:
                logger.warning("skipping malformed pdf table %d: %s", i, exc)

        csv_path = SAMPLES / "mesa_council_2026-04-06_agenda_items.csv"
        sid_tbl = upsert_source(
            conn,
            SourceRecord(
                city=m["city"],
                source_type="table",
                title="Agenda items 2026-04-06",
                meeting_id=m["meeting_id"],
                url=f"https://webapi.legistar.com/v1/mesa/events/{m['meeting_id']}/eventitems",
                meeting_date=m["date"],
                metadata={"sample": True},
            ),
        )
        table = load_csv_table(
            csv_path,
            slug=f"mesa_agenda_items_{m['meeting_id']}",
            description="Agenda items of the Mesa City Council meeting on 2026-04-06: "
            "agenda_number, agenda_sequence, matter_file, matter_type, title",
        )
        create_normalized_table(conn, sid_tbl, table)

        conn.commit()
        stats = corpus_stats(conn)
    logger.info("corpus stats: %s", stats)
    return stats


def _ingest_meeting(
    conn, client: LegistarClient, meeting: MeetingInfo, workdir: Path, asr_budget: list[int]
) -> bool:
    """Ingest one live meeting: captions (or capped ASR) + agenda PDF + agenda items."""
    assert meeting.date is not None
    video_url = meeting.video_url or find_youtube_video(meeting.client, meeting.date)
    got_anything = False

    if video_url:
        vtt = fetch_youtube_captions(video_url, workdir)
        chunks: list[ChunkRecord] = []
        if vtt is not None:
            chunks = transcript_chunks_from_vtt(vtt)
        elif asr_budget[0] > 0:
            from ingestion.asr import download_audio, transcript_chunks_from_audio

            logger.info("no captions; ASR fallback for %s", video_url)
            asr_budget[0] -= 1
            chunks = transcript_chunks_from_audio(download_audio(video_url, workdir))
        if chunks:
            sid = upsert_source(
                conn,
                SourceRecord(
                    city=meeting.client,
                    source_type="transcript",
                    title=f"{meeting.body_name} {meeting.date}",
                    meeting_id=str(meeting.event_id),
                    url=video_url,
                    meeting_date=meeting.date,
                    metadata={"captions": vtt is not None},
                ),
            )
            replace_chunks(conn, sid, _maybe_tag(chunks))
            got_anything = True

    if meeting.agenda_url:
        import httpx

        pdf_path = workdir / f"{meeting.client}_{meeting.event_id}_agenda.pdf"
        resp = httpx.get(meeting.agenda_url, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        pdf_path.write_bytes(resp.content)
        blocks = _maybe_tag(extract_pdf_blocks(pdf_path))
        if blocks:
            sid = upsert_source(
                conn,
                SourceRecord(
                    city=meeting.client,
                    source_type="pdf",
                    title=f"{meeting.body_name} Agenda {meeting.date}",
                    meeting_id=str(meeting.event_id),
                    url=meeting.agenda_url,
                    meeting_date=meeting.date,
                    metadata={"pages": max((b.page_no or 1) for b in blocks)},
                ),
            )
            replace_chunks(conn, sid, blocks)
            got_anything = True

    items = client.event_items(meeting.client, meeting.event_id)
    rows = [
        [
            it.get("EventItemAgendaNumber") or "",
            str(it.get("EventItemAgendaSequence") or ""),
            it.get("EventItemMatterFile") or "",
            it.get("EventItemMatterType") or "",
            " ".join((it.get("EventItemTitle") or "").split())[:500],
        ]
        for it in items
        if (it.get("EventItemTitle") or "").strip()
    ]
    if rows:
        from ingestion.models import RawTable

        raw = RawTable(
            header=["agenda_number", "agenda_sequence", "matter_file", "matter_type", "title"],
            rows=rows,
        )
        sid = upsert_source(
            conn,
            SourceRecord(
                city=meeting.client,
                source_type="table",
                title=f"Agenda items {meeting.date} ({meeting.body_name})",
                meeting_id=str(meeting.event_id),
                url=f"https://webapi.legistar.com/v1/{meeting.client}/events/{meeting.event_id}/eventitems",
                meeting_date=meeting.date,
                metadata={},
            ),
        )
        table = normalize_raw_table(
            raw,
            slug=f"{meeting.client}_agenda_items_{meeting.event_id}",
            description=f"Agenda items of {meeting.body_name} ({meeting.client}) on "
            f"{meeting.date}: agenda_number, agenda_sequence, matter_file, matter_type, title",
        )
        create_normalized_table(conn, sid, table)
        got_anything = True
    return got_anything


def ingest_live(
    cities: list[str] | None = None, per_city: int = 3, asr_cap: int = 2, total_cap: int = 6
) -> dict:
    """Ingest recent council meetings (with agendas) from two responsive Legistar clients."""
    run_migrations()
    client = LegistarClient()
    cities = cities or pick_responsive_clients(["seattle", "mesa", "oakland"], client, need=2)
    logger.info("ingesting from cities: %s", cities)
    asr_budget = [asr_cap]
    ingested = 0
    with get_connection() as conn, tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for city in cities:
            events = client.recent_events(city, top=80, before=date.today() + timedelta(days=1))
            council = [
                e
                for e in events
                if e.date
                and e.date <= date.today()
                and e.agenda_url
                and "council" in e.body_name.lower()
                and "study" not in e.body_name.lower()
                and "cancel" not in (e.agenda_status or "").lower()
            ][:per_city]
            for meeting in council:
                if ingested >= total_cap:
                    break
                logger.info("meeting: %s %s (%s)", city, meeting.date, meeting.event_id)
                try:
                    if _ingest_meeting(conn, client, meeting, workdir, asr_budget):
                        ingested += 1
                        conn.commit()
                except Exception:
                    conn.rollback()
                    logger.exception("failed to ingest %s event %s", city, meeting.event_id)
        stats = corpus_stats(conn)
    logger.info("ingested %d meetings; corpus stats: %s", ingested, stats)
    return stats


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="CivicLens ingestion")
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("samples", help="ingest the bundled sample corpus (no network)")
    live = sub.add_parser("live", help="ingest real meetings from Legistar + YouTube")
    live.add_argument("--cities", nargs="*", default=None)
    live.add_argument("--per-city", type=int, default=3)
    live.add_argument("--asr-cap", type=int, default=2)
    live.add_argument("--total-cap", type=int, default=6)
    args = parser.parse_args()
    if args.mode == "samples":
        stats = ingest_samples()
    else:
        stats = ingest_live(args.cities, args.per_city, args.asr_cap, args.total_cap)
    print(f"done. corpus: {stats}")


if __name__ == "__main__":
    main()
