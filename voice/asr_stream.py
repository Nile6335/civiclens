"""Chunked-buffer streaming transcription for the /voice WebSocket.

The browser sends push-to-talk WAVs as ordered binary frames that concatenate into one
complete WAV file. StreamingTranscriber accumulates those bytes and offers two passes:
`partial()` — a cheap greedy transcription of the current buffer, called by the endpoint
between frames and cached until new audio arrives — and `finalize()` — a full-quality
beam-search pass over the whole utterance after end_of_speech.

This is honest chunked-buffer streaming, not word-level streaming ASR: nothing here does
VAD-chunking of a live mic stream, and each `partial()` re-transcribes the buffer from
the start rather than emitting incremental word hypotheses.

faster_whisper is imported lazily inside `get_model` so that importing this module never
pulls in the package (or triggers its model download) at import or test time — mirroring
ingestion/asr.py.
"""

import logging
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from common.settings import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def get_model(model_size: str) -> Any:
    """Lazily build (and cache) a CPU int8 faster-whisper model for model_size."""
    from faster_whisper import WhisperModel  # lazy: the model load must not happen at import

    logger.info("loading faster-whisper model %s (cpu/int8)", model_size)
    return WhisperModel(model_size, device="cpu", compute_type="int8")


class StreamingTranscriber:
    """Accumulate WAV bytes for one utterance; emit rolling partials and a final pass."""

    def __init__(self, model_size: str | None = None) -> None:
        self._model_size = model_size or get_settings().whisper_model
        self._buffer = bytearray()
        self._partial_text: str | None = None
        self._bytes_at_last_partial = 0

    def accept_audio(self, data: bytes) -> None:
        """Append one binary frame (frames arrive in order and form one complete WAV)."""
        self._buffer.extend(data)

    def has_new_audio_since_partial(self) -> bool:
        return len(self._buffer) != self._bytes_at_last_partial

    def partial(self) -> str:
        """Fast greedy transcription of the current buffer; cached until new audio arrives."""
        if not self.has_new_audio_since_partial() and self._partial_text is not None:
            return self._partial_text
        text = self._transcribe(beam_size=1, condition_on_previous_text=False)
        self._partial_text = text
        self._bytes_at_last_partial = len(self._buffer)
        return text

    def finalize(self) -> str:
        """Full-quality beam-search pass over the whole buffered utterance."""
        return self._transcribe(beam_size=5)

    def reset(self) -> None:
        """Clear the audio buffer and the partial-transcript cache for the next utterance."""
        self._buffer.clear()
        self._partial_text = None
        self._bytes_at_last_partial = 0

    def _transcribe(self, **transcribe_kwargs: Any) -> str:
        """Write the buffer to a temp .wav and transcribe it; returns joined segment text."""
        model = get_model(self._model_size)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(bytes(self._buffer))
            tmp_path = Path(tmp.name)
        try:
            segments, _info = model.transcribe(str(tmp_path), vad_filter=True, **transcribe_kwargs)
            return " ".join(s.text.strip() for s in segments).strip()
        finally:
            tmp_path.unlink(missing_ok=True)
