"""Unit tests for the ASR fallback: a fake faster_whisper is injected before the lazy import."""

import importlib
import sys
import types
from pathlib import Path

import pytest

from ingestion.models import ChunkRecord, Cue


def _fake_segments() -> list[types.SimpleNamespace]:
    """Ten 9.5s segments spanning 0..99.5s (more than two 45s windows)."""
    return [
        types.SimpleNamespace(start=i * 10.0, end=i * 10.0 + 9.5, text=f" segment {i} ")
        for i in range(10)
    ]


@pytest.fixture
def whisper_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Install a fake faster_whisper module; returns the WhisperModel constructor calls."""
    calls: list[dict] = []

    class FakeWhisperModel:
        def __init__(self, model_size: str, device: str = "auto", compute_type: str = "default"):
            calls.append({"model_size": model_size, "device": device, "compute_type": compute_type})

        def transcribe(self, audio_path: str) -> tuple:
            return iter(_fake_segments()), types.SimpleNamespace(language="en")

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    return calls


def test_import_does_not_import_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)
    monkeypatch.delitem(sys.modules, "ingestion.asr", raising=False)
    importlib.import_module("ingestion.asr")
    assert "faster_whisper" not in sys.modules


def test_transcribe_audio_converts_segments_to_cues(whisper_calls: list[dict]) -> None:
    from ingestion.asr import transcribe_audio

    cues = transcribe_audio(Path("meeting.m4a"))
    assert len(cues) == 10
    assert all(isinstance(c, Cue) for c in cues)
    assert cues[0] == Cue(start=0.0, end=9.5, text="segment 0")
    assert cues[-1] == Cue(start=90.0, end=99.5, text="segment 9")


def test_whisper_model_constructed_for_cpu_int8(whisper_calls: list[dict]) -> None:
    from ingestion.asr import transcribe_audio

    transcribe_audio(Path("meeting.m4a"))
    assert whisper_calls == [{"model_size": "small", "device": "cpu", "compute_type": "int8"}]


def test_transcribe_audio_honors_model_size(whisper_calls: list[dict]) -> None:
    from ingestion.asr import transcribe_audio

    transcribe_audio(Path("meeting.m4a"), model_size="tiny")
    assert whisper_calls[0]["model_size"] == "tiny"


def test_transcript_chunks_from_audio_windows(whisper_calls: list[dict]) -> None:
    from ingestion.asr import transcript_chunks_from_audio

    chunks = transcript_chunks_from_audio(Path("meeting.m4a"))
    assert len(chunks) >= 2
    assert all(isinstance(c, ChunkRecord) for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].t_start == pytest.approx(0.0)
    assert chunks[0].t_end == pytest.approx(49.5)
    assert chunks[0].t_end - chunks[0].t_start >= 45.0
    assert chunks[1].t_start == pytest.approx(50.0)
    assert chunks[1].t_end == pytest.approx(99.5)
    assert "segment 0" in chunks[0].text and "segment 4" in chunks[0].text
    assert "segment 5" in chunks[1].text and "segment 9" in chunks[1].text


def test_download_audio_option_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import ingestion.asr as asr

    captured: dict = {}

    class FakeYoutubeDL:
        def __init__(self, opts: dict):
            captured["opts"] = opts

        def __enter__(self) -> "FakeYoutubeDL":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def extract_info(self, url: str, download: bool = False) -> dict:
            captured["url"] = url
            captured["download"] = download
            return {"id": "abc123", "ext": "m4a"}

        def prepare_filename(self, info: dict) -> str:
            return str(tmp_path / f"{info['id']}.{info['ext']}")

    monkeypatch.setattr(asr.yt_dlp, "YoutubeDL", FakeYoutubeDL)
    path = asr.download_audio("https://youtube.com/watch?v=abc123", tmp_path)
    assert path == tmp_path / "abc123.m4a"
    assert captured["url"] == "https://youtube.com/watch?v=abc123"
    assert captured["download"] is True
    opts = captured["opts"]
    assert opts["format"] == "bestaudio[ext=m4a]/bestaudio"
    assert opts["outtmpl"] == str(tmp_path / "%(id)s.%(ext)s")
