"""Agenda-packet PDF ingestion: page-aware text chunks and raw tables via pdfplumber."""

import re
from itertools import pairwise
from pathlib import Path

import pdfplumber
from pdfplumber.page import Page

from ingestion.models import ChunkRecord, RawTable

DEFAULT_TARGET_CHARS = 1000
_PARAGRAPH_GAP_RATIO = 1.5  # line advance above 1.5x the tightest advance starts a new paragraph
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _page_paragraphs(page: Page) -> list[str]:
    """Group a page's text lines into paragraphs using vertical line-gap heuristics."""
    lines = sorted(page.extract_text_lines(), key=lambda line: line["top"])
    if not lines:
        text = _normalize_ws(page.extract_text() or "")
        return [text] if text else []
    deltas = [b["top"] - a["top"] for a, b in pairwise(lines)]
    advances = sorted(d for d in deltas if d > 0)
    threshold = advances[0] * _PARAGRAPH_GAP_RATIO if advances else float("inf")
    paragraphs: list[list[str]] = [[lines[0]["text"]]]
    for delta, line in zip(deltas, lines[1:], strict=True):
        if delta > threshold:
            paragraphs.append([])
        paragraphs[-1].append(line["text"])
    normalized = (_normalize_ws(" ".join(parts)) for parts in paragraphs)
    return [p for p in normalized if p]


def _split_long_paragraph(paragraph: str, target_chars: int) -> list[str]:
    """Hard-split an oversized paragraph on sentence-ish boundaries."""
    parts: list[str] = []
    buf = ""
    for sentence in _SENTENCE_BOUNDARY.split(paragraph):
        if buf and len(buf) + len(sentence) + 1 > target_chars:
            parts.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}" if buf else sentence
    if buf:
        parts.append(buf)
    return parts


def _pack_paragraphs(paragraphs: list[str], target_chars: int) -> list[str]:
    """Greedily pack paragraphs to ~target_chars; only paragraphs beyond 2x get split."""
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) > 2 * target_chars:
            pieces.extend(_split_long_paragraph(paragraph, target_chars))
        else:
            pieces.append(paragraph)
    packed: list[str] = []
    buf = ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + 1 > target_chars:
            packed.append(buf)
            buf = piece
        else:
            buf = f"{buf}\n\n{piece}" if buf else piece
    if buf:
        packed.append(buf)
    return packed


def extract_pdf_blocks(path: Path, target_chars: int = DEFAULT_TARGET_CHARS) -> list[ChunkRecord]:
    """Extract page-aware text chunks of roughly target_chars from a PDF.

    Chunks never cross page boundaries, so page_no is the page each chunk starts on;
    chunk_index runs sequentially across the whole document. Empty pages are skipped.
    """
    chunks: list[ChunkRecord] = []
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for text in _pack_paragraphs(_page_paragraphs(page), target_chars):
                chunks.append(ChunkRecord(chunk_index=len(chunks), text=text, page_no=page_no))
    return chunks


def extract_pdf_tables(path: Path) -> list[RawTable]:
    """Extract drawn tables per page; tables under 2 rows or 2 columns are dropped."""
    tables: list[RawTable] = []
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for grid in page.extract_tables():
                if len(grid) < 2 or len(grid[0]) < 2:
                    continue
                header = ["" if cell is None else str(cell) for cell in grid[0]]
                rows = [["" if cell is None else str(cell) for cell in row] for row in grid[1:]]
                tables.append(RawTable(header=header, rows=rows, page_no=page_no))
    return tables


def pdf_is_image_only(path: Path) -> bool:
    """True when no page yields extractable text (scan-only PDFs needing OCR)."""
    with pdfplumber.open(path) as pdf:
        return not any((page.extract_text() or "").strip() for page in pdf.pages)


def ocr_pdf(path: Path) -> str:
    """OCR fallback for image-only PDFs. Requires the optional OCR toolchain."""
    try:
        import pdf2image
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "OCR fallback needs pytesseract and pdf2image: "
            "`pip install pytesseract pdf2image` plus the tesseract and poppler binaries"
        ) from exc
    images = pdf2image.convert_from_path(str(path))
    return "\n\n".join(pytesseract.image_to_string(image) for image in images).strip()
