"""Phase 1 acceptance: pipelines validated against the bundled real-world sample corpus."""

from pathlib import Path

from ingestion.captions import parse_vtt, transcript_chunks_from_vtt
from ingestion.pdf import extract_pdf_blocks, pdf_is_image_only
from ingestion.tables import load_csv_table

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
VTT = SAMPLES / "mesa_council_2026-04-06.en.vtt"
PDF = SAMPLES / "mesa_council_2026-04-06_agenda.pdf"
CSV = SAMPLES / "mesa_council_2026-04-06_agenda_items.csv"


def test_sample_vtt_chunk_alignment() -> None:
    chunks = transcript_chunks_from_vtt(VTT)
    assert 20 <= len(chunks) <= 40  # 23-minute meeting at ~45s windows
    assert chunks[0].t_start is not None and chunks[0].t_start < 10
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        assert prev.t_end is not None and cur.t_start is not None
        assert prev.t_start is not None and prev.t_start < prev.t_end
        assert cur.t_start >= prev.t_end - 0.01  # non-overlapping, monotonic
    # windows span roughly >=45s except the last
    for c in chunks[:-1]:
        assert c.t_end - c.t_start >= 40.0, f"short window: {c}"
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_sample_vtt_rolling_duplicates_removed() -> None:
    """The YouTube rolling-display artifact must not double the text."""
    chunks = transcript_chunks_from_vtt(VTT)
    text = " ".join(c.text for c in chunks)
    assert "Welcome to the Mesa City Thank you" not in text  # doubled-line signature
    assert text.count("Welcome to the Mesa City Council meeting for April 6th") == 1
    assert "&gt;" not in text and "&amp;" not in text  # entities unescaped
    # keep it real speech: no leftover inline tags
    assert "<c>" not in text and "</c>" not in text


def test_sample_vtt_cue_parsing() -> None:
    cues = parse_vtt(VTT)
    assert len(cues) > 100
    assert all(cue.end >= cue.start for cue in cues)
    joined = " ".join(c.text for c in cues)
    assert "Mesa City" in joined and "April 6th" in joined


def test_sample_pdf_page_numbers_preserved() -> None:
    assert not pdf_is_image_only(PDF)
    blocks = extract_pdf_blocks(PDF)
    assert blocks, "agenda PDF produced no chunks"
    pages = {b.page_no for b in blocks}
    assert 1 in pages and max(pages) >= 4  # 6-page agenda; content spread over pages
    assert all(b.page_no is not None and b.page_no >= 1 for b in blocks)
    joined = " ".join(b.text for b in blocks)
    assert "Mayor" in joined and "Council" in joined


def test_sample_csv_table_extraction() -> None:
    table = load_csv_table(CSV, slug="pytest_sample_items", description="test")
    names = [c.name for c in table.columns]
    assert names == ["agenda_number", "agenda_sequence", "matter_file", "matter_type", "title"]
    assert len(table.rows) == 33
    seq_col = names.index("agenda_sequence")
    title_col = names.index("title")
    assert any("consent agenda" in str(r[title_col]).lower() for r in table.rows)
    assert all(r[seq_col] is not None for r in table.rows)
