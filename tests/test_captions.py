"""Unit tests for ingestion/captions.py (no network, no model downloads)."""

from pathlib import Path

import pytest

from ingestion import captions
from ingestion.captions import (
    fetch_youtube_captions,
    parse_vtt,
    transcript_chunks_from_vtt,
    video_id_from_url,
)

VTT_FIXTURE = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:10.000
good evening everyone

00:00:10.000 --> 00:00:20.000
we<00:00:11.000><c> will</c><00:00:12.000><c> now</c> begin
we will now begin

00:00:20.000 --> 00:00:30.000
the council meeting

00:00:30.000 --> 00:00:40.000
the council meeting

00:00:40.000 --> 00:00:50.000


00:00:50.000 --> 00:01:00.000
first item on the agenda

00:01:00.000 --> 00:01:10.000
is the zoning variance

00:01:10.000 --> 00:01:20.000
for elm street

00:01:20.000 --> 00:01:30.000
public comment is open

00:01:30.000 --> 00:01:40.000
please state your name
"""


def test_parse_vtt_strips_tags_and_duplicate_lines() -> None:
    cues = parse_vtt(VTT_FIXTURE)
    tagged = cues[1]
    assert tagged.text == "we will now begin"
    assert tagged.start == pytest.approx(10.0)
    assert tagged.end == pytest.approx(20.0)


def test_parse_vtt_drops_empty_cues_and_keeps_order() -> None:
    cues = parse_vtt(VTT_FIXTURE)
    # 10 cues in the fixture: one is empty, and the rolling-display repeat of
    # "the council meeting" at 30s is carry-over, not new speech.
    assert len(cues) == 8
    assert all(c.text for c in cues)
    assert not any(c.start == pytest.approx(40.0) for c in cues)
    assert sum("the council meeting" in c.text for c in cues) == 1
    starts = [c.start for c in cues]
    assert starts == sorted(starts)


def test_parse_vtt_accepts_file_path(tmp_path: Path) -> None:
    vtt_file = tmp_path / "meeting.en.vtt"
    vtt_file.write_text(VTT_FIXTURE, encoding="utf-8")
    assert parse_vtt(vtt_file) == parse_vtt(VTT_FIXTURE)
    assert parse_vtt(str(vtt_file)) == parse_vtt(VTT_FIXTURE)


def test_transcript_chunks_windowing() -> None:
    chunks = transcript_chunks_from_vtt(VTT_FIXTURE, window_seconds=45.0)
    assert len(chunks) >= 2
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    cues = parse_vtt(VTT_FIXTURE)
    cue_starts = {c.start for c in cues}
    cue_ends = {c.end for c in cues}
    for chunk in chunks:
        assert chunk.t_start in cue_starts  # chunks align to cue boundaries
        assert chunk.t_end in cue_ends

    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert prev.t_start is not None and prev.t_end is not None
        assert nxt.t_start is not None and nxt.t_end is not None
        assert prev.t_start < prev.t_end
        assert prev.t_end <= nxt.t_start
    for chunk in chunks[:-1]:
        assert chunk.t_end - chunk.t_start >= 45.0  # only the remainder may be shorter

    joined = " ".join(c.text for c in chunks)
    assert joined.count("the council meeting") == 1  # consecutive duplicate cues collapsed
    assert "we will now begin" in joined


def test_fetch_youtube_captions_builds_options(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    class FakeYDL:
        def __init__(self, options: dict) -> None:
            captured.update(options)

        def __enter__(self) -> "FakeYDL":
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def extract_info(self, url: str, download: bool = True) -> dict:
            out = tmp_path / "vid123.en.vtt"
            out.write_text("WEBVTT\n", encoding="utf-8")
            return {"id": "vid123", "requested_subtitles": {"en": {"filepath": str(out)}}}

    monkeypatch.setattr(captions.yt_dlp, "YoutubeDL", FakeYDL)
    result = fetch_youtube_captions("https://youtu.be/vid123", tmp_path, lang="en")
    assert result == tmp_path / "vid123.en.vtt"
    assert captured["skip_download"] is True
    assert captured["writesubtitles"] is True
    assert captured["writeautomaticsub"] is True
    assert captured["subtitlesformat"] == "vtt"
    assert "en" in captured["subtitleslangs"]
    assert any(lang.startswith("en") and lang != "en" for lang in captured["subtitleslangs"])


def test_fetch_youtube_captions_none_when_no_subs(monkeypatch, tmp_path: Path) -> None:
    class FakeYDL:
        def __init__(self, options: dict) -> None:
            pass

        def __enter__(self) -> "FakeYDL":
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

        def extract_info(self, url: str, download: bool = True) -> dict:
            return {"id": "vid123", "requested_subtitles": None}

    monkeypatch.setattr(captions.yt_dlp, "YoutubeDL", FakeYDL)
    assert fetch_youtube_captions("https://youtu.be/vid123", tmp_path) is None


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=5", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/live/AbC_dEf1234?feature=share", "AbC_dEf1234"),
        ("https://youtube.com/live/AbC_dEf1234", "AbC_dEf1234"),
        ("https://www.youtube.com/watch", None),
        ("https://example.com/watch?v=dQw4w9WgXcQ", None),
        ("not a url", None),
    ],
)
def test_video_id_from_url(url: str, expected: str | None) -> None:
    assert video_id_from_url(url) == expected
