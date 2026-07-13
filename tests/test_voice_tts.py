"""Unit tests for voice/tts.py: no piper loading, no network — fakes are injected."""

import importlib
import io
import subprocess
import sys
import types
import wave
from pathlib import Path

import pytest

import voice.tts as tts

# ---------------------------------------------------------------------------
# import hygiene
# ---------------------------------------------------------------------------


def test_import_does_not_import_piper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "piper", raising=False)
    monkeypatch.delitem(sys.modules, "voice.tts", raising=False)
    importlib.import_module("voice.tts")
    assert "piper" not in sys.modules


# ---------------------------------------------------------------------------
# strip_citations / split_sentences
# ---------------------------------------------------------------------------


def test_split_sentences_multi_sentence() -> None:
    text = "The council met Tuesday. The budget passed! Was anyone opposed? Two members."
    assert tts.split_sentences(text) == [
        "The council met Tuesday.",
        "The budget passed!",
        "Was anyone opposed?",
        "Two members.",
    ]


def test_split_sentences_video_citation_not_split_and_not_spoken() -> None:
    text = (
        "The mayor spoke [video @ 12:34](https://youtube.com/watch?v=abc&t=754s). The vote passed."
    )
    sentences = tts.split_sentences(text)
    assert sentences == ["The mayor spoke.", "The vote passed."]
    assert not any("http" in s or "[video" in s for s in sentences)


def test_split_sentences_doc_page_dots_do_not_split() -> None:
    text = "Fees rise in July [doc, p.4]. See the fee schedule [doc, p.12]. Done deal."
    assert tts.split_sentences(text) == [
        "Fees rise in July.",
        "See the fee schedule.",
        "Done deal.",
    ]


def test_strip_citations_removes_all_patterns() -> None:
    text = (
        "Revenue grew [table: budget_2026]. See [doc, p.4] and "
        "[video @ 1:02:03](https://x.io/v?t=3723s) [E2]."
    )
    assert tts.strip_citations(text) == "Revenue grew. See and."


def test_split_sentences_drops_tiny_fragments_and_empty_text() -> None:
    assert tts.split_sentences("Yes. A") == ["Yes."]
    assert tts.split_sentences("   \n ") == []
    assert tts.split_sentences("[doc, p.4]") == []


# ---------------------------------------------------------------------------
# SentenceStreamer
# ---------------------------------------------------------------------------


def test_streamer_trailing_fragment_kept_for_flush_not_feed() -> None:
    streamer = tts.SentenceStreamer()
    assert streamer.feed("Hello there. Bye") == ["Hello there."]
    assert streamer.flush() == ["Bye"]
    assert streamer.flush() == []


def test_streamer_emits_first_sentence_once_terminator_and_next_token_arrive() -> None:
    streamer = tts.SentenceStreamer()
    assert streamer.feed("First ") == []
    assert streamer.feed("point ") == []
    assert streamer.feed("[E1]") == []
    # Terminator at the very end of the buffer: the tail may still grow.
    assert streamer.feed(".") == []
    assert streamer.feed(" Second") == ["First point."]
    assert streamer.feed(" point.") == []
    assert streamer.flush() == ["Second point."]


def test_streamer_strips_citations_and_never_splits_inside_them() -> None:
    text = (
        "Watch the debate [video @ 12:34](https://youtu.be/a?t=754s). "
        "Fees are set [doc, p.4]. Final answer."
    )
    streamer = tts.SentenceStreamer()
    got: list[str] = []
    for i in range(0, len(text), 3):  # small pieces, citation split across many tokens
        got += streamer.feed(text[i : i + 3])
    got += streamer.flush()
    assert got == ["Watch the debate.", "Fees are set.", "Final answer."]


def test_streamer_citation_only_tail_is_silent() -> None:
    streamer = tts.SentenceStreamer()
    assert streamer.feed("Answer done. ") == ["Answer done."]
    assert streamer.feed("[doc, p.9]") == []
    assert streamer.flush() == []


# ---------------------------------------------------------------------------
# synthesize_wav_bytes
# ---------------------------------------------------------------------------


def test_synthesize_wav_bytes_empty_returns_empty_without_loading_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom() -> None:
        raise AssertionError("get_voice must not be called for empty text")

    monkeypatch.setattr(tts, "get_voice", boom)
    assert tts.synthesize_wav_bytes("") == b""
    assert tts.synthesize_wav_bytes("   \n\t") == b""


def test_synthesize_wav_bytes_returns_complete_wav(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeVoice:
        def synthesize_wav(self, text: str, wav: wave.Wave_write) -> None:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(22050)
            wav.writeframes(b"\x01\x02" * 128)

    monkeypatch.setattr(tts, "get_voice", lambda: FakeVoice())
    data = tts.synthesize_wav_bytes("Hello there.")
    assert data.startswith(b"RIFF") and data[8:12] == b"WAVE"
    with wave.open(io.BytesIO(data), "rb") as wav:
        assert wav.getnframes() == 128
        assert wav.getframerate() == 22050


# ---------------------------------------------------------------------------
# get_voice (fake piper module + fake subprocess; nothing real loads)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> types.SimpleNamespace:
    """Point tts at a tmp_path piper_data_dir and reset the lru_cache around the test."""
    settings = types.SimpleNamespace(
        piper_voice="en_US-lessac-medium", piper_data_dir=str(tmp_path / "piper")
    )
    monkeypatch.setattr(tts, "get_settings", lambda: settings)
    tts.get_voice.cache_clear()
    yield settings
    tts.get_voice.cache_clear()


@pytest.fixture
def piper_loads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Install a fake piper module; returns the paths passed to PiperVoice.load."""
    loads: list[str] = []

    class FakePiperVoice:
        @classmethod
        def load(cls, path: str) -> "FakePiperVoice":
            loads.append(path)
            return cls()

    fake = types.ModuleType("piper")
    fake.PiperVoice = FakePiperVoice  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "piper", fake)
    return loads


def test_get_voice_downloads_when_onnx_missing(
    fake_settings: types.SimpleNamespace,
    piper_loads: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool = False, **kwargs: object) -> object:
        commands.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(tts.subprocess, "run", fake_run)
    voice = tts.get_voice()
    assert commands == [
        [
            sys.executable,
            "-m",
            "piper.download_voices",
            "en_US-lessac-medium",
            "--data-dir",
            fake_settings.piper_data_dir,
        ]
    ]
    assert Path(fake_settings.piper_data_dir).is_dir()  # created before the download runs
    expected_onnx = Path(fake_settings.piper_data_dir) / "en_US-lessac-medium.onnx"
    assert piper_loads == [str(expected_onnx)]
    assert voice is not None


def test_get_voice_skips_download_when_onnx_present(
    fake_settings: types.SimpleNamespace,
    piper_loads: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = Path(fake_settings.piper_data_dir)
    data_dir.mkdir(parents=True)
    onnx = data_dir / "en_US-lessac-medium.onnx"
    onnx.write_bytes(b"onnx")

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("download must not run when the onnx already exists")

    monkeypatch.setattr(tts.subprocess, "run", boom)
    first = tts.get_voice()
    second = tts.get_voice()
    assert piper_loads == [str(onnx)]  # lru_cache: loaded exactly once
    assert first is second
