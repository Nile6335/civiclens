"""Unit tests for ingestion.pdf using fixture PDFs generated at test time with fpdf2."""

import re
from importlib.util import find_spec
from pathlib import Path

import pytest
from fpdf import FPDF

from ingestion.pdf import extract_pdf_blocks, extract_pdf_tables, ocr_pdf, pdf_is_image_only

TARGET_CHARS = 300

PAGE1_PARAGRAPHS = [
    "The city council convened at seven in the evening to discuss the proposed budget "
    "amendments for the coming fiscal year, including allocations for road maintenance, "
    "library services, and the new community center on Oak Street.",
    "Public comment opened with remarks about the crosswalk near the elementary school "
    "and a request for additional lighting along the riverfront trail.",
    "Staff presented an update on the stormwater improvement project, noting that the "
    "second phase remains on schedule and within the approved budget envelope.",
]
PAGE2_PARAGRAPHS = [
    "Ordinance 2042 rezoning parcel 88 from light industrial to mixed use was adopted "
    "by a vote of six to one after a brief discussion of traffic impacts.",
    "The meeting adjourned at nine fifteen with the next session scheduled for the "
    "second Tuesday of the following month at city hall.",
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _new_pdf() -> FPDF:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    return pdf


@pytest.fixture()
def text_pdf(tmp_path: Path) -> Path:
    pdf = _new_pdf()
    for paragraphs in (PAGE1_PARAGRAPHS, PAGE2_PARAGRAPHS):
        pdf.add_page()
        for paragraph in paragraphs:
            pdf.multi_cell(0, 6, paragraph)
            pdf.ln(4)
    path = tmp_path / "agenda.pdf"
    pdf.output(str(path))
    return path


@pytest.fixture()
def table_pdf(tmp_path: Path) -> Path:
    pdf = _new_pdf()
    pdf.add_page()
    grid = [
        ["Department", "Amount", "Year"],
        ["Police", "1200", "2024"],
        ["Parks", "300", "2024"],
    ]
    for row in grid:
        for cell in row:
            pdf.cell(50, 10, cell, border=1)
        pdf.ln(10)
    pdf.add_page()  # a single-column grid: under 2 cols, must be dropped
    for cell in ["Only", "Column"]:
        pdf.cell(50, 10, cell, border=1)
        pdf.ln(10)
    path = tmp_path / "budget.pdf"
    pdf.output(str(path))
    return path


@pytest.fixture()
def image_only_pdf(tmp_path: Path) -> Path:
    pdf = FPDF()
    pdf.add_page()
    pdf.rect(20, 20, 100, 50)
    path = tmp_path / "scan.pdf"
    pdf.output(str(path))
    return path


def test_blocks_page_no_and_sequential_index(text_pdf: Path) -> None:
    chunks = extract_pdf_blocks(text_pdf, target_chars=TARGET_CHARS)
    assert len(chunks) >= 2
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    page1 = [c for c in chunks if "stormwater improvement project" in c.text]
    page2 = [c for c in chunks if "Ordinance 2042" in c.text]
    assert page1 and all(c.page_no == 1 for c in page1)
    assert page2 and all(c.page_no == 2 for c in page2)


def test_blocks_respect_size_cap(text_pdf: Path) -> None:
    chunks = extract_pdf_blocks(text_pdf, target_chars=TARGET_CHARS)
    assert all(len(c.text) <= 2 * TARGET_CHARS + 50 for c in chunks)


def test_blocks_preserve_all_source_text(text_pdf: Path) -> None:
    chunks = extract_pdf_blocks(text_pdf, target_chars=TARGET_CHARS)
    combined = _norm(" ".join(c.text for c in chunks))
    for paragraph in PAGE1_PARAGRAPHS + PAGE2_PARAGRAPHS:
        assert _norm(paragraph) in combined


def test_extract_tables_header_rows_and_page(table_pdf: Path) -> None:
    tables = extract_pdf_tables(table_pdf)
    assert len(tables) == 1
    table = tables[0]
    assert table.header == ["Department", "Amount", "Year"]
    assert table.rows == [["Police", "1200", "2024"], ["Parks", "300", "2024"]]
    assert table.page_no == 1
    assert table.caption is None


def test_pdf_is_image_only(text_pdf: Path, image_only_pdf: Path) -> None:
    assert not pdf_is_image_only(text_pdf)
    assert pdf_is_image_only(image_only_pdf)


def test_ocr_pdf_raises_without_ocr_libs(image_only_pdf: Path) -> None:
    if find_spec("pytesseract") is not None and find_spec("pdf2image") is not None:
        pytest.skip("OCR toolchain installed; the RuntimeError branch is not reachable")
    with pytest.raises(RuntimeError, match="pytesseract"):
        ocr_pdf(image_only_pdf)
