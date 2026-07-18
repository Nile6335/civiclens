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


def _model_path() -> Path:
    settings = get_settings()
    return Path(settings.piper_data_dir) / f"{settings.piper_voice}.onnx"


def _ensure_model() -> Path:
    """Download the configured Piper voice model if missing; return its path.

    Downloading only fetches files (no native synthesis), so it cannot abort the
    process — unlike synthesis on a broken espeak build, which is why availability is
    probed separately in a subprocess below.
    """
    settings = get_settings()
    model_path = _model_path()
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
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
    return model_path


# A tiny synthesis probe run in a child process. Piper's bundled espeak-ng aborts()
# (SIGABRT, not a catchable exception) on some prebuilt wheels — notably macOS arm64 —
# which would take down the whole API server if run in-process. Running it once in a
# subprocess contains the crash: the child dies, the parent reads a non-zero exit and
# marks TTS unavailable, and the voice turn degrades to a text-only answer.
_PROBE = (
    "import io,wave;from piper import PiperVoice;"
    "v=PiperVoice.load(__import__('sys').argv[1]);b=io.BytesIO();"
    "w=wave.open(b,'wb');v.synthesize_wav('ok',w);w.close();"
    "print(len(b.getvalue()))"
)


@lru_cache(maxsize=1)
def tts_available() -> bool:
    """True if Piper can actually synthesize audio on this platform (probed once).

    Never raises; a broken native build returns False rather than crashing the caller.
    """
    try:
        model_path = _ensure_model()
    except Exception:
        return False
    try:
        result = subprocess.run(
            [sys.executable, "-c", _PROBE, str(model_path)],
            capture_output=True,
            timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    ok = result.returncode == 0 and result.stdout.strip().isdigit()
    if not ok:
        import logging

        logging.getLogger(__name__).warning(
            "Piper TTS unavailable on this platform (rc=%s); voice answers will be "
            "text-only. On macOS this often means the prebuilt piper-tts wheel's espeak "
            "build is broken — see the README voice notes.",
            result.returncode,
        )
    return ok


@lru_cache(maxsize=1)
def get_voice() -> "PiperVoice":
    """Load (and cache) the configured Piper voice, auto-downloading the model if missing."""
    from piper import PiperVoice  # lazy: the model must never load at import time

    return PiperVoice.load(str(_ensure_model()))


def synthesize_wav_bytes(text: str) -> bytes:
    """Return WAV bytes for one utterance; b"" for empty text or when TTS is unavailable.

    The subprocess-probed ``tts_available`` gate means in-process synthesis is only ever
    attempted on a platform where it has been proven not to abort — so a broken Piper
    build degrades to a (silent) text answer instead of crashing the server.
    """
    if not text.strip() or not tts_available():
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
