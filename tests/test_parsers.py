"""Tests for ingestion parsers (CSV, PDF, PPT).

PDF tests use max_pages to keep the suite fast (<60s).
"""

from pathlib import Path

import pandas as pd
import pytest

from ingestion.parsers.csv_parser import ParsedChunk, get_dataframe, parse_csv
from ingestion.parsers.pdf_parser import parse_pdf
from ingestion.parsers.ppt_parser import parse_ppt

REFERS_DIR = Path(__file__).resolve().parent.parent / "refers"
CSV_PATH = REFERS_DIR / "tax_data.csv"
PDF_1040_PATH = REFERS_DIR / "i1040gi.pdf"
PDF_IRC_PATH = REFERS_DIR / "usc26@118-78.pdf"
PPT_PATH = REFERS_DIR / "MIC_3e_Ch11.ppt"

PDF_TEST_MAX_PAGES = 15


# ── CSV Parser Tests ──────────────────────────────────────────────────


class TestCsvParser:
    @pytest.fixture(autouse=True)
    def _skip_if_no_csv(self):
        if not CSV_PATH.exists():
            pytest.skip("CSV test data not found")

    @pytest.fixture()
    def chunks(self):
        return parse_csv(CSV_PATH)

    def test_returns_chunks(self, chunks):
        assert len(chunks) > 0
        assert all(isinstance(c, ParsedChunk) for c in chunks)

    def test_chunk_fields(self, chunks):
        row_chunk = chunks[0]
        assert row_chunk.source_type == "csv"
        assert row_chunk.source_file == "tax_data.csv"
        assert row_chunk.chunk_index == 0
        assert len(row_chunk.text) > 10

    def test_row_chunks_have_metadata_columns(self, chunks):
        expected_cols = {
            "Taxpayer Type",
            "Tax Year",
            "State",
            "Income",
            "Tax Owed",
        }
        assert expected_cols.issubset(set(chunks[0].metadata.keys()))

    def test_row_count_matches_csv(self, chunks):
        df = pd.read_csv(CSV_PATH)
        row_chunks = [c for c in chunks if c.metadata.get("summary_type") is None]
        assert len(row_chunks) == len(df)

    def test_summary_chunks_exist(self, chunks):
        summaries = [c for c in chunks if c.metadata.get("summary_type") is not None]
        assert len(summaries) > 0
        overview = [
            c for c in summaries if c.metadata["summary_type"] == "dataset_overview"
        ]
        assert len(overview) == 1

    def test_group_summary_chunks(self, chunks):
        group_summaries = [
            c for c in chunks if c.metadata.get("summary_type") == "group_summary"
        ]
        assert len(group_summaries) > 0
        assert any(c.metadata.get("group_column") == "State" for c in group_summaries)

    def test_get_dataframe(self):
        df = get_dataframe(CSV_PATH)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "Income" in df.columns

    def test_text_is_human_readable(self, chunks):
        text = chunks[0].text
        assert "Taxpayer Type:" in text
        assert "State:" in text


# ── PDF Parser Tests ──────────────────────────────────────────────────


class TestPdfParser1040:
    @pytest.fixture(autouse=True)
    def _skip_if_no_pdf(self):
        if not PDF_1040_PATH.exists():
            pytest.skip("1040 PDF test data not found")

    @pytest.fixture()
    def chunks(self):
        return parse_pdf(PDF_1040_PATH, max_pages=PDF_TEST_MAX_PAGES)

    def test_returns_chunks(self, chunks):
        assert len(chunks) > 0
        assert all(isinstance(c, ParsedChunk) for c in chunks)

    def test_chunk_fields(self, chunks):
        chunk = chunks[0]
        assert chunk.source_type == "pdf"
        assert chunk.source_file == "i1040gi.pdf"
        assert "pages" in chunk.metadata

    def test_chunks_have_page_metadata(self, chunks):
        for chunk in chunks:
            assert isinstance(chunk.metadata["pages"], list)
            assert all(isinstance(p, int) for p in chunk.metadata["pages"])

    def test_contains_tax_content(self, chunks):
        all_text = " ".join(c.text for c in chunks)
        assert "1040" in all_text
        assert "income" in all_text.lower()

    def test_chunk_size_reasonable(self):
        chunks = parse_pdf(PDF_1040_PATH, chunk_size=500, max_pages=5)
        for chunk in chunks:
            assert len(chunk.text) <= 600

    def test_overlap_provides_continuity(self):
        chunks = parse_pdf(
            PDF_1040_PATH, chunk_size=500, chunk_overlap=100, max_pages=5
        )
        if len(chunks) >= 2:
            tail = chunks[0].text[-100:]
            head = chunks[1].text[:200]
            assert tail[:50] in head or len(chunks[0].text) < 500


class TestPdfParserIRC:
    """Test IRC PDF with a small page sample to keep tests fast."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_pdf(self):
        if not PDF_IRC_PATH.exists():
            pytest.skip("IRC PDF test data not found")

    @pytest.fixture()
    def chunks(self):
        return parse_pdf(PDF_IRC_PATH, chunk_size=2000, max_pages=PDF_TEST_MAX_PAGES)

    def test_returns_chunks(self, chunks):
        assert len(chunks) > 0

    def test_contains_legal_content(self, chunks):
        all_text = " ".join(c.text[:500] for c in chunks[:20])
        assert "internal revenue" in all_text.lower()


# ── PPT Parser Tests ─────────────────────────────────────────────────


class TestPptParser:
    @pytest.fixture(autouse=True)
    def _skip_if_no_ppt(self):
        if not PPT_PATH.exists():
            pytest.skip("PPT test data not found")

    @pytest.fixture()
    def chunks(self):
        return parse_ppt(PPT_PATH)

    def test_returns_chunks(self, chunks):
        assert len(chunks) > 0
        assert all(isinstance(c, ParsedChunk) for c in chunks)

    def test_chunk_fields(self, chunks):
        chunk = chunks[0]
        assert chunk.source_type == "ppt"
        assert chunk.source_file == "MIC_3e_Ch11.ppt"
        assert "slide_num" in chunk.metadata

    def test_contains_tax_content(self, chunks):
        all_text = " ".join(c.text for c in chunks).lower()
        assert "tax" in all_text

    def test_filters_template_text(self, chunks):
        for chunk in chunks:
            assert "Click to edit Master" not in chunk.text

    def test_chunk_count_reasonable(self, chunks):
        assert len(chunks) >= 5


# ── Cross-Parser Tests ────────────────────────────────────────────────


class TestParsedChunkContract:
    """Verify all parsers produce consistent ParsedChunk objects."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_data(self):
        if not CSV_PATH.exists():
            pytest.skip("Test data not found")

    def test_all_parsers_produce_same_type(self):
        csv_chunks = parse_csv(CSV_PATH)
        assert all(isinstance(c, ParsedChunk) for c in csv_chunks)

        if PDF_1040_PATH.exists():
            pdf_chunks = parse_pdf(PDF_1040_PATH, max_pages=5)
            assert all(isinstance(c, ParsedChunk) for c in pdf_chunks)

        if PPT_PATH.exists():
            ppt_chunks = parse_ppt(PPT_PATH)
            assert all(isinstance(c, ParsedChunk) for c in ppt_chunks)

    def test_all_chunks_have_required_fields(self):
        all_chunks = parse_csv(CSV_PATH)
        if PPT_PATH.exists():
            all_chunks += parse_ppt(PPT_PATH)

        for chunk in all_chunks:
            assert isinstance(chunk.text, str) and len(chunk.text) > 0
            assert isinstance(chunk.metadata, dict)
            assert isinstance(chunk.source_file, str) and len(chunk.source_file) > 0
            assert chunk.source_type in ("csv", "pdf", "ppt")
            assert isinstance(chunk.chunk_index, int)
