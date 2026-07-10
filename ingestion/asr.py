"""Fallback ASR pipeline: download meeting audio and transcribe it with faster-whisper.

Used when a meeting video exposes no captions. faster_whisper is imported lazily inside
transcribe_audio so that importing this module never pulls in the package (or triggers
its ~250MB model download) at import or test time.
"""

import logging
from pathlib import Path

import yt_dlp

from ingestion.models import ChunkRecord, Cue
from ingestion.windows import DEFAULT_WINDOW_SECONDS, merge_cues_into_windows

logger = logging.getLogger(__name__)


def download_audio(video_url: str, workdir: Path) -> Path:
    """Download the best audio-only stream for video_url into workdir; returns the file path."""
    opts = {
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": str(workdir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        audio_path = Path(ydl.prepare_filename(info))
    logger.info("downloaded audio for %s -> %s", video_url, audio_path)
    return audio_path


def transcribe_audio(audio_path: Path, model_size: str = "small") -> list[Cue]:
    """Transcribe an audio file into timed cues with faster-whisper on CPU (int8)."""
    from faster_whisper import WhisperModel  # lazy: the model download must not happen at import

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(audio_path))
    cues = [Cue(start=float(s.start), end=float(s.end), text=s.text.strip()) for s in segments]
    logger.info("transcribed %s into %d cues", audio_path, len(cues))
    return cues


def transcript_chunks_from_audio(
    audio_path: Path, window_seconds: float = DEFAULT_WINDOW_SECONDS
) -> list[ChunkRecord]:
    """Full ASR fallback: transcribe audio, then pack the cues into ~window_seconds chunks."""
    return merge_cues_into_windows(transcribe_audio(audio_path), window_seconds=window_seconds)
