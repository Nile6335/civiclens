"""Merge timed cues into ~fixed-duration transcript windows aligned to cue boundaries."""

from ingestion.models import ChunkRecord, Cue

DEFAULT_WINDOW_SECONDS = 45.0


def merge_cues_into_windows(
    cues: list[Cue], window_seconds: float = DEFAULT_WINDOW_SECONDS
) -> list[ChunkRecord]:
    """Greedily pack consecutive cues into windows of ~window_seconds.

    A window closes once it spans at least window_seconds (measured from the first
    cue's start to the current cue's end), so chunks always end on a cue boundary and
    never split a cue. Empty-text cues are dropped. Consecutive duplicate cue texts
    (a YouTube auto-caption artifact) are collapsed.
    """
    chunks: list[ChunkRecord] = []
    buf: list[Cue] = []

    def flush() -> None:
        if not buf:
            return
        text = " ".join(c.text.strip() for c in buf if c.text.strip())
        if text:
            chunks.append(
                ChunkRecord(
                    chunk_index=len(chunks),
                    text=text,
                    t_start=buf[0].start,
                    t_end=buf[-1].end,
                )
            )
        buf.clear()

    prev_text: str | None = None
    for cue in cues:
        stripped = cue.text.strip()
        if not stripped:
            continue
        if stripped == prev_text:
            # extend the previous cue's span instead of repeating its text
            if buf:
                buf[-1] = Cue(start=buf[-1].start, end=cue.end, text=buf[-1].text)
            continue
        prev_text = stripped
        buf.append(cue)
        if cue.end - buf[0].start >= window_seconds:
            flush()
    flush()
    return chunks
