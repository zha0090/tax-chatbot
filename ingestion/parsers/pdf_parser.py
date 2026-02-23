from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

from .csv_parser import ParsedChunk

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def parse_pdf(
    file_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    max_pages: int | None = None,
) -> list[ParsedChunk]:
    """Parse a PDF file into overlapping text chunks.

    Extracts text page-by-page, then splits into chunks of approximately
    `chunk_size` characters with `chunk_overlap` overlap for context continuity.
    Set `max_pages` to limit how many pages are processed (useful for very large PDFs).
    """
    file_path = Path(file_path)
    pages = _extract_pages(file_path, max_pages=max_pages)
    chunks = _chunk_pages(pages, file_path.name, chunk_size, chunk_overlap)
    return chunks


def _extract_pages(file_path: Path, max_pages: int | None = None) -> list[dict]:
    """Extract text and tables from each PDF page."""
    pages = []
    with pdfplumber.open(file_path) as pdf:
        page_list = pdf.pages[:max_pages] if max_pages else pdf.pages
        for page_num, page in enumerate(page_list, start=1):
            text = page.extract_text() or ""
            text = _clean_text(text)

            tables = page.extract_tables() or []
            table_texts = []
            for table in tables:
                table_texts.append(_table_to_text(table))

            full_text = text
            if table_texts:
                full_text += "\n\n" + "\n\n".join(table_texts)

            if full_text.strip():
                pages.append({"page_num": page_num, "text": full_text.strip()})

    return pages


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"^-+ \d+ of \d+ -+$", "", text, flags=re.MULTILINE)
    return text.strip()


def _table_to_text(table: list[list]) -> str:
    """Convert a pdfplumber table into readable text."""
    if not table:
        return ""
    rows = []
    for row in table:
        cells = [str(cell).strip() if cell else "" for cell in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def _chunk_pages(
    pages: list[dict],
    source_file: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ParsedChunk]:
    """Split page texts into overlapping chunks, tracking page boundaries."""
    all_text = ""
    page_boundaries: list[tuple[int, int, int]] = []

    for page in pages:
        start = len(all_text)
        all_text += page["text"] + "\n\n"
        end = len(all_text)
        page_boundaries.append((start, end, page["page_num"]))

    chunks: list[ParsedChunk] = []
    pos = 0
    chunk_idx = 0

    while pos < len(all_text):
        end = min(pos + chunk_size, len(all_text))
        chunk_text = all_text[pos:end].strip()

        if not chunk_text:
            break

        page_nums = _get_page_nums(pos, end, page_boundaries)

        chunks.append(
            ParsedChunk(
                text=chunk_text,
                metadata={"pages": page_nums, "char_offset": pos},
                source_file=source_file,
                source_type="pdf",
                chunk_index=chunk_idx,
            )
        )
        chunk_idx += 1

        step = chunk_size - chunk_overlap
        if step <= 0:
            step = chunk_size // 2
        pos += step

    return chunks


def _get_page_nums(
    start: int, end: int, boundaries: list[tuple[int, int, int]]
) -> list[int]:
    pages = []
    for b_start, b_end, page_num in boundaries:
        if b_start < end and b_end > start:
            pages.append(page_num)
    return pages
