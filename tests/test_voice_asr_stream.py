"""Unit tests for voice.asr_stream: a fake faster_whisper is injected before the lazy import."""

import importlib
import sys
import types
from pathlib import Path

import pytest

from common.settings import get_settings


class FakeRecorder:
    """Constructor calls and transcribe calls captured from the fake faster_whisper module."""

    def __init__(self) -> None:
        self.ctor_calls: list[dict] = []
        self.transcribe_calls: list[dict] = []


@pytest.fixture
def whisper(monkeypatch: pytest.MonkeyPatch) -> FakeRecorder:
    """Install a fake faster_whisper module and reset the module-level model cache."""
    recorder = FakeRecorder()

    class FakeWhisperModel:
        def __init__(self, model_size: str, device: str = "auto", compute_type: str = "default"):
            recorder.ctor_calls.append(
                {"model_size": model_size, "device": device, "compute_type": compute_type}
            )

        def transcribe(self, audio_path: str, **kwargs: object) -> tuple:
            recorder.transcribe_calls.append({"audio": Path(audio_path).read_bytes(), **kwargs})
            segments = [
                types.SimpleNamespace(text=" hello "),
                types.SimpleNamespace(text=" world "),
            ]
            return iter(segments), types.SimpleNamespace(language="en")

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)

    from voice.asr_stream import get_model

    get_model.cache_clear()
    yield recorder
    get_model.cache_clear()


def test_import_does_not_import_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "faster_whisper", raising=False)
    monkeypatch.delitem(sys.modules, "voice.asr_stream", raising=False)
    importlib.import_module("voice.asr_stream")
    assert "faster_whisper" not in sys.modules


def test_get_model_constructs_cpu_int8_with_settings_default(whisper: FakeRecorder) -> None:
    from voice.asr_stream import StreamingTranscriber

    t = StreamingTranscriber()
    t.accept_audio(b"RIFF....some audio bytes....")
    t.partial()
    assert whisper.ctor_calls == [
        {
            "model_size": get_settings().whisper_model,
            "device": "cpu",
            "compute_type": "int8",
        }
    ]


def test_get_model_honors_explicit_model_size(whisper: FakeRecorder) -> None:
    from voice.asr_stream import StreamingTranscriber

    t = StreamingTranscriber(model_size="tiny")
    t.accept_audio(b"RIFF....some audio bytes....")
    t.partial()
    assert whisper.ctor_calls[0]["model_size"] == "tiny"


def test_accept_audio_accumulates_and_partial_uses_fast_settings(whisper: FakeRecorder) -> None:
    from voice.asr_stream import StreamingTranscriber

    t = StreamingTranscriber()
    t.accept_audio(b"RIFFchunk1")
    t.accept_audio(b"chunk2")
    assert t.partial() == "hello world"
    call = whisper.transcribe_calls[0]
    assert call["audio"] == b"RIFFchunk1chunk2"
    assert call["beam_size"] == 1
    assert call["vad_filter"] is True
    assert call["condition_on_previous_text"] is False


def test_empty_buffer_short_circuits_before_model(whisper: FakeRecorder) -> None:
    from voice.asr_stream import StreamingTranscriber

    t = StreamingTranscriber()
    assert t.finalize() == ""
    assert t.partial() == ""
    assert whisper.transcribe_calls == []  # model never called on empty audio


def test_undecodable_audio_degrades_to_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    # A model whose transcribe raises (browsers send the occasional malformed WAV):
    # the turn must degrade to "" so the endpoint reports "no speech detected".
    import voice.asr_stream as asr

    class RaisingModel:
        def transcribe(self, *_a: object, **_k: object):
            raise ValueError("Invalid data found when processing input")

    monkeypatch.setattr(asr, "get_model", lambda _size: RaisingModel())
    t = asr.StreamingTranscriber()
    t.accept_audio(b"not really a wav file")
    assert t.finalize() == ""
    assert t.partial() == ""


def test_partial_is_cached_until_new_audio(whisper: FakeRecorder) -> None:
    from voice.asr_stream import StreamingTranscriber

    t = StreamingTranscriber()
    t.accept_audio(b"RIFFchunk1")
    assert t.has_new_audio_since_partial()
    assert t.partial() == "hello world"
    assert not t.has_new_audio_since_partial()
    assert t.partial() == "hello world"  # cache hit: no second transcription
    assert len(whisper.transcribe_calls) == 1

    t.accept_audio(b"chunk2")
    assert t.has_new_audio_since_partial()
    assert t.partial() == "hello world"
    assert len(whisper.transcribe_calls) == 2
    assert whisper.transcribe_calls[1]["audio"] == b"RIFFchunk1chunk2"


def test_finalize_uses_full_beam_over_whole_buffer(whisper: FakeRecorder) -> None:
    from voice.asr_stream import StreamingTranscriber

    t = StreamingTranscriber()
    t.accept_audio(b"RIFFchunk1")
    t.accept_audio(b"chunk2")
    assert t.finalize() == "hello world"
    call = whisper.transcribe_calls[0]
    assert call["audio"] == b"RIFFchunk1chunk2"
    assert call["beam_size"] == 5
    assert call["vad_filter"] is True


def test_reset_clears_buffer_and_cache(whisper: FakeRecorder) -> None:
    from voice.asr_stream import StreamingTranscriber

    t = StreamingTranscriber()
    t.accept_audio(b"RIFFchunk1")
    t.partial()
    t.reset()
    assert not t.has_new_audio_since_partial()

    t.accept_audio(b"RIFFfresh")
    assert t.has_new_audio_since_partial()
    t.partial()
    assert whisper.transcribe_calls[1]["audio"] == b"RIFFfresh"


def test_model_is_loaded_once_per_size(whisper: FakeRecorder) -> None:
    from voice.asr_stream import StreamingTranscriber

    t = StreamingTranscriber()
    t.accept_audio(b"RIFFchunk1")
    t.partial()
    t.accept_audio(b"chunk2")
    t.partial()
    t.finalize()
    assert len(whisper.ctor_calls) == 1
