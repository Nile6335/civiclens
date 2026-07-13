"""Piper text-to-speech with sentence-level streaming.

Sentences are synthesized one at a time so playback can start before the full answer is
generated. Citations ([video @ mm:ss](url), [doc, p.N], [table: name]) and [E#] evidence
markers are stripped BEFORE segmentation: dots inside "p.4" or URLs must never split a
sentence, and the spoken answer must not read URLs aloud. piper is imported lazily inside
get_voice so importing this module never loads (or downloads) the model.
"""

import io
import re
import subprocess
import sys
import wave
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from agents.evidence import CITATION_RE
from common.settings import get_settings

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from piper import PiperVoice

_EVIDENCE_MARKER_RE = re.compile(r"\[E\d+\]")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([.!?,;:])")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
# A sentence ends at [.!?] followed by whitespace; end-of-text closes the last one.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_COMPLETE_BOUNDARY_RE = re.compile(r"[.!?]\s+")

MIN_FRAGMENT_CHARS = 2


@lru_cache(maxsize=1)
def get_voice() -> "PiperVoice":
    """Load (and cache) the configured Piper voice, auto-downloading the model if missing."""
    settings = get_settings()
    data_dir = Path(settings.piper_data_dir)
    model_path = data_dir / f"{settings.piper_voice}.onnx"
    if not model_path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "piper.download_voices",
                settings.piper_voice,
                "--data-dir",
                settings.piper_data_dir,
            ],
            check=True,
        )
    from piper import PiperVoice  # lazy: the model must never load at import time

    return PiperVoice.load(str(model_path))


def synthesize_wav_bytes(text: str) -> bytes:
    """Return complete, valid WAV bytes for one utterance; empty/whitespace text -> b""."""
    if not text.strip():
        return b""
    voice = get_voice()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        voice.synthesize_wav(text, wav)
    return buf.getvalue()


def strip_citations(text: str) -> str:
    """Remove citations and [E#] markers so the spoken answer never reads URLs aloud."""
    cleaned = CITATION_RE.sub("", text)
    cleaned = _EVIDENCE_MARKER_RE.sub("", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    return _MULTI_SPACE_RE.sub(" ", cleaned).strip()


def split_sentences(text: str) -> list[str]:
    """Segment text into speakable sentences: citations stripped first, delimiters kept.

    Citations are removed before splitting, so the dot in "[doc, p.4]" or inside a
    "[video @ 12:34](https://...)" URL can never break a sentence. Fragments shorter
    than MIN_FRAGMENT_CHARS are dropped.
    """
    cleaned = strip_citations(text)
    if not cleaned:
        return []
    parts = (part.strip() for part in _SENTENCE_SPLIT_RE.split(cleaned))
    return [part for part in parts if len(part) >= MIN_FRAGMENT_CHARS]


class SentenceStreamer:
    """Incremental token accumulator that emits complete, citation-stripped sentences.

    A sentence is complete only when its terminator is followed by further text — a
    terminator at the very end of the buffer may still grow (e.g. "p." + "4"), so the
    tail is only released by flush(). Boundary detection runs on the raw buffer, which
    is safe because no citation pattern contains a terminator followed by whitespace.
    """

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, token: str) -> list[str]:
        """Append one token; return any sentences newly completed by it."""
        self._buffer += token
        cut = 0
        for match in _COMPLETE_BOUNDARY_RE.finditer(self._buffer):
            cut = match.end()
        if cut == 0:
            return []
        complete, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return split_sentences(complete)

    def flush(self) -> list[str]:
        """Return the remaining tail as final sentence(s) (if non-empty) and reset."""
        tail, self._buffer = self._buffer, ""
        return split_sentences(tail)
