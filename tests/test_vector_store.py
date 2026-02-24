"""Tests for the embedding + FAISS vector store pipeline.

Uses a temporary directory for the store to avoid polluting real data.
Requires OPENAI_API_KEY in .env for embedding tests.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ingestion.parsers.csv_parser import ParsedChunk

SKIP_REASON = "OPENAI_API_KEY not set"


def _has_api_key() -> bool:
    from django.conf import settings

    return bool(
        settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "sk-your-key-here"
    )


def _make_chunks(n: int = 5) -> list[ParsedChunk]:
    return [
        ParsedChunk(
            text=f"Test chunk number {i} about tax deductions in state {chr(65 + i)}.",
            metadata={"test_field": f"value_{i}"},
            source_file="test.csv",
            source_type="csv",
            chunk_index=i,
        )
        for i in range(n)
    ]


@pytest.fixture()
def temp_store():
    """Create a VectorStore in a temporary directory."""
    from retrieval.vector_search import VectorStore

    with tempfile.TemporaryDirectory() as tmpdir:
        yield VectorStore(persist_dir=tmpdir)


# ── Embedder Tests ────────────────────────────────────────────────────


class TestEmbedder:
    @pytest.fixture(autouse=True)
    def _skip_if_no_key(self):
        if not _has_api_key():
            pytest.skip(SKIP_REASON)

    def test_embed_single(self):
        from ingestion.embedder import embed_single

        vec = embed_single("What is the tax rate in California?")
        assert isinstance(vec, list)
        assert len(vec) > 100
        assert all(isinstance(v, float) for v in vec)

    def test_embed_texts_batch(self):
        from ingestion.embedder import embed_texts

        texts = ["Tax deduction", "Income tax rate", "IRS Form 1040"]
        vecs = embed_texts(texts)
        assert len(vecs) == 3
        assert all(len(v) == len(vecs[0]) for v in vecs)

    def test_embed_texts_empty(self):
        from ingestion.embedder import embed_texts

        vecs = embed_texts([])
        assert vecs == []

    def test_embedding_dimensions_consistent(self):
        from ingestion.embedder import embed_texts

        vecs = embed_texts(["hello", "world"])
        assert len(vecs[0]) == len(vecs[1])


# ── Vector Store Indexing Tests ───────────────────────────────────────


class TestVectorStoreIndexing:
    @pytest.fixture(autouse=True)
    def _skip_if_no_key(self):
        if not _has_api_key():
            pytest.skip(SKIP_REASON)

    def test_index_and_count(self, temp_store):
        from ingestion.embedder import embed_texts

        chunks = _make_chunks(3)
        embeddings = embed_texts([c.text for c in chunks])
        count = temp_store.index_chunks(chunks, embeddings)
        assert count == 3
        assert temp_store.count() == 3

    def test_upsert_does_not_duplicate(self, temp_store):
        from ingestion.embedder import embed_texts

        chunks = _make_chunks(2)
        embeddings = embed_texts([c.text for c in chunks])
        temp_store.index_chunks(chunks, embeddings)
        temp_store.index_chunks(chunks, embeddings)
        assert temp_store.count() == 2

    def test_metadata_stored(self, temp_store):
        from ingestion.embedder import embed_texts

        chunks = _make_chunks(1)
        embeddings = embed_texts([c.text for c in chunks])
        temp_store.index_chunks(chunks, embeddings)

        results = temp_store.search("tax deductions", n_results=1)
        assert len(results) == 1
        assert results[0]["metadata"]["source_type"] == "csv"
        assert results[0]["metadata"]["source_file"] == "test.csv"

    def test_clear(self, temp_store):
        from ingestion.embedder import embed_texts

        chunks = _make_chunks(3)
        embeddings = embed_texts([c.text for c in chunks])
        temp_store.index_chunks(chunks, embeddings)
        assert temp_store.count() == 3

        temp_store.clear()
        assert temp_store.count() == 0

    def test_persistence(self):
        from ingestion.embedder import embed_texts
        from retrieval.vector_search import VectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store1 = VectorStore(persist_dir=tmpdir)
            chunks = _make_chunks(3)
            embeddings = embed_texts([c.text for c in chunks])
            store1.index_chunks(chunks, embeddings)

            store2 = VectorStore(persist_dir=tmpdir)
            assert store2.count() == 3


# ── Vector Store Search Tests ─────────────────────────────────────────


class TestVectorStoreSearch:
    @pytest.fixture(autouse=True)
    def _skip_if_no_key(self):
        if not _has_api_key():
            pytest.skip(SKIP_REASON)

    @pytest.fixture()
    def populated_store(self, temp_store):
        from ingestion.embedder import embed_texts

        chunks = [
            ParsedChunk(
                text="The corporate tax rate in California is approximately 8.84 percent.",
                metadata={"topic": "corporate_tax"},
                source_file="test.csv",
                source_type="csv",
                chunk_index=0,
            ),
            ParsedChunk(
                text="IRS Form 1040 is used to file individual income tax returns.",
                metadata={"topic": "irs_form"},
                source_file="test.pdf",
                source_type="pdf",
                chunk_index=1,
            ),
            ParsedChunk(
                text="An excise tax on a product results in a higher price.",
                metadata={"topic": "excise_tax"},
                source_file="test.ppt",
                source_type="ppt",
                chunk_index=2,
            ),
        ]
        embeddings = embed_texts([c.text for c in chunks])
        temp_store.index_chunks(chunks, embeddings)
        return temp_store

    def test_search_returns_results(self, populated_store):
        results = populated_store.search("What is the corporate tax rate?", n_results=2)
        assert len(results) == 2

    def test_search_relevance(self, populated_store):
        results = populated_store.search("California corporate tax rate", n_results=1)
        top = results[0]
        assert "California" in top["text"] or "corporate" in top["text"].lower()

    def test_search_with_source_filter(self, populated_store):
        results = populated_store.search(
            "tax", n_results=3, source_type_filter="pdf"
        )
        for hit in results:
            assert hit["metadata"]["source_type"] == "pdf"

    def test_search_empty_store(self, temp_store):
        results = temp_store.search("anything")
        assert results == []
