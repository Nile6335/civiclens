"""Match a Legistar meeting to its YouTube recording by searching the city's channel."""

import logging
from datetime import date

logger = logging.getLogger(__name__)


def _date_variants(d: date) -> list[str]:
    """Title date forms seen on city channels: 4/6/2026, 04/06/2026, 4/6/26."""
    return [
        f"{d.month}/{d.day}/{d.year}",
        f"{d.month:02d}/{d.day:02d}/{d.year}",
        f"{d.month}/{d.day}/{d.year % 100}",
    ]


def find_youtube_video(
    city: str, meeting_date: date, extra_terms: str = "city council"
) -> str | None:
    """Search YouTube for the city's council meeting on a given date; return URL or None.

    Only accepts results whose title contains the meeting date, to avoid grabbing an
    unrelated meeting.
    """
    import yt_dlp  # local import: keep module import cheap

    d = meeting_date
    query = f"ytsearch10:{city} {extra_terms} meeting {d.month}/{d.day}/{d.year}"
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(query, download=False)
    except Exception as exc:  # network hiccups: caller falls back
        logger.warning("youtube search failed for %s %s: %s", city, meeting_date, exc)
        return None
    variants = _date_variants(meeting_date)
    for entry in (info or {}).get("entries", []):
        title = entry.get("title") or ""
        channel = (entry.get("channel") or "").lower()
        if any(v in title for v in variants) and (
            city.lower() in channel or "channel" in channel or city.lower() in title.lower()
        ):
            url = entry.get("url") or f"https://www.youtube.com/watch?v={entry.get('id')}"
            logger.info("matched %s %s -> %s (%s)", city, meeting_date, title, url)
            return url
    logger.info("no dated youtube match for %s %s", city, meeting_date)
    return None
