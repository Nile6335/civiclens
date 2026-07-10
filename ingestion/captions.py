"""YouTube caption (WebVTT) ingestion: fetch and parse captions into transcript chunks.

parse_vtt cleans the YouTube auto-caption artifacts (inline <00:00:01.234><c> tags,
rolling duplicate lines, empty cues); merge_cues_into_windows does the windowing.
"""

import html
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import webvtt
import yt_dlp

from ingestion.models import ChunkRecord, Cue
from ingestion.windows import DEFAULT_WINDOW_SECONDS, merge_cues_into_windows

_TAG_RE = re.compile(r"<[^>]*>")
_PATH_PREFIXES = ("live", "embed", "shorts", "v")


def _timestamp_to_seconds(timestamp: str) -> float:
    """Convert a VTT timestamp ('HH:MM:SS.mmm' or 'MM:SS.mmm') to seconds."""
    seconds = 0.0
    for part in timestamp.split(":"):
        seconds = seconds * 60.0 + float(part)
    return seconds


def _clean_lines(lines: list[str], prev_cue_lines: set[str]) -> str:
    """Strip inline tags/entities; drop lines repeated within the cue or held over.

    YouTube auto-captions roll: each cue re-displays the previous cue's line(s) above
    the newly spoken line (and emits "hold" cues repeating a line verbatim), so any line
    that already appeared in the immediately preceding cue is display carry-over, not new
    speech. prev_cue_lines is mutated to hold ALL of this cue's cleaned lines — including
    dropped ones — so a line can ride through an arbitrarily long run of hold cues.
    """
    cue_lines = [" ".join(html.unescape(_TAG_RE.sub(" ", line)).split()) for line in lines]
    cue_lines = [line for line in cue_lines if line]
    emitted: list[str] = []
    seen: set[str] = set()
    for line in cue_lines:
        if line in seen or line in prev_cue_lines:
            continue
        seen.add(line)
        emitted.append(line)
    prev_cue_lines.clear()
    prev_cue_lines.update(cue_lines)
    return " ".join(emitted)


def parse_vtt(path_or_text: Path | str) -> list[Cue]:
    """Parse a WebVTT file (or raw VTT text) into cleaned, timestamped Cues.

    Handles YouTube auto-caption artifacts: inline <00:00:01.234><c> tags and HTML
    entities are stripped, multi-line payloads are joined with spaces, lines repeated
    within a cue or carried over from the previous cue (the rolling-display pattern)
    are dropped, and empty cues are skipped.
    """
    if isinstance(path_or_text, str) and path_or_text.lstrip("\ufeff \t\r\n").startswith("WEBVTT"):
        parsed = webvtt.from_string(path_or_text)
    else:
        parsed = webvtt.read(str(path_or_text))
    cues: list[Cue] = []
    prev_cue_lines: set[str] = set()
    for caption in parsed:
        text = _clean_lines(caption.lines, prev_cue_lines)
        if not text:
            continue
        cues.append(
            Cue(
                start=_timestamp_to_seconds(caption.start),
                end=_timestamp_to_seconds(caption.end),
                text=text,
            )
        )
    return cues


def transcript_chunks_from_vtt(
    path_or_text: Path | str, window_seconds: float = DEFAULT_WINDOW_SECONDS
) -> list[ChunkRecord]:
    """Parse a VTT source and pack its cues into ~window_seconds transcript chunks."""
    return merge_cues_into_windows(parse_vtt(path_or_text), window_seconds=window_seconds)


def fetch_youtube_captions(video_url: str, workdir: Path, lang: str = "en") -> Path | None:
    """Download a video's captions as .vtt into workdir; None if it has no captions.

    Human subtitles are preferred over automatic captions (yt-dlp resolves per-language
    when both write flags are set). Network-dependent; not exercised by unit tests.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    options = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang, f"{lang}-.*"],
        "subtitlesformat": "vtt",
        "outtmpl": str(workdir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(video_url, download=True)
    if not info:
        return None
    for sub in (info.get("requested_subtitles") or {}).values():
        filepath = (sub or {}).get("filepath")
        if filepath and filepath.endswith(".vtt") and Path(filepath).exists():
            return Path(filepath)
    video_id = info.get("id")
    candidates = sorted(workdir.glob(f"{video_id}*.vtt")) if video_id else []
    return candidates[0] if candidates else None


def video_id_from_url(url: str) -> str | None:
    """Extract the YouTube video id from watch?v=, youtu.be/, /live/ (and similar) URLs."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    parts = [p for p in parsed.path.split("/") if p]
    if host == "youtu.be":
        return parts[0] if parts else None
    if host == "youtube.com" or host.endswith(".youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if len(parts) >= 2 and parts[0] in _PATH_PREFIXES:
            return parts[1]
    return None
