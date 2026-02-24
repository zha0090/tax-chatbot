"""Vector store backed by FAISS for fast similarity search.

Stores embeddings in a FAISS index on disk, with a sidecar JSON file
for document texts and metadata. Supports filtering by source_type
after retrieval.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from django.conf import settings

from ingestion.embedder import embed_single
from ingestion.parsers.csv_parser import ParsedChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-backed vector store with metadata support."""

    def __init__(self, persist_dir: str | Path | None = None):
        self.persist_dir = Path(
            persist_dir or settings.VECTOR_STORE_DIR
        )
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.persist_dir / "faiss.index"
        self._meta_path = self.persist_dir / "metadata.json"
        self._index: faiss.Index | None = None
        self._documents: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._ids: list[str] = []
        self._load()

    def _load(self) -> None:
        if self._index_path.exists() and self._meta_path.exists():
            try:
                self._index = faiss.read_index(str(self._index_path))
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._documents = data["documents"]
                self._metadatas = data["metadatas"]
                self._ids = data["ids"]
                logger.info("Loaded vector store: %d vectors", self._index.ntotal)
            except (json.JSONDecodeError, KeyError, OSError):
                logger.exception("Corrupted vector store files, starting fresh")
                self._reset()
        else:
            self._reset()

    def _reset(self) -> None:
        self._index = None
        self._documents = []
        self._metadatas = []
        self._ids = []

    def _save(self) -> None:
        try:
            if self._index is not None:
                faiss.write_index(self._index, str(self._index_path))
            with open(self._meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "documents": self._documents,
                        "metadatas": self._metadatas,
                        "ids": self._ids,
                    },
                    f,
                )
            logger.info("Saved vector store: %d vectors", len(self._ids))
        except OSError:
            logger.exception("Failed to save vector store")

    def index_chunks(
        self,
        chunks: list[ParsedChunk],
        embeddings: list[list[float]],
    ) -> int:
        """Add parsed chunks and their embeddings to the store.

        Uses upsert semantics: chunks with existing IDs are replaced.
        """
        if not chunks:
            return 0

        dim = len(embeddings[0])
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)

        id_to_idx = {id_: i for i, id_ in enumerate(self._ids)}

        new_ids = []
        new_docs = []
        new_metas = []
        new_vecs = []

        for chunk, emb in zip(chunks, embeddings):
            chunk_id = f"{chunk.source_type}_{chunk.source_file}_{chunk.chunk_index}"
            if chunk_id in id_to_idx:
                existing_idx = id_to_idx[chunk_id]
                self._documents[existing_idx] = chunk.text
                self._metadatas[existing_idx] = _prepare_metadata(chunk)
            else:
                new_ids.append(chunk_id)
                new_docs.append(chunk.text)
                new_metas.append(_prepare_metadata(chunk))
                new_vecs.append(emb)

        if new_vecs:
            vectors = np.array(new_vecs, dtype=np.float32)
            faiss.normalize_L2(vectors)
            self._index.add(vectors)
            self._ids.extend(new_ids)
            self._documents.extend(new_docs)
            self._metadatas.extend(new_metas)

        self._save()
        return len(chunks)

    def search(
        self,
        query: str,
        n_results: int = 5,
        source_type_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search. Returns top results sorted by similarity."""
        if self._index is None or self._index.ntotal == 0:
            return []

        query_vec = np.array([embed_single(query)], dtype=np.float32)
        faiss.normalize_L2(query_vec)

        fetch_k = n_results * 3 if source_type_filter else n_results
        fetch_k = min(fetch_k, self._index.ntotal)

        scores, indices = self._index.search(query_vec, fetch_k)

        hits: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadatas[idx]
            if source_type_filter and meta.get("source_type") != source_type_filter:
                continue
            hits.append(
                {
                    "id": self._ids[idx],
                    "text": self._documents[idx],
                    "metadata": meta,
                    "score": float(score),
                }
            )
            if len(hits) >= n_results:
                break

        return hits

    def count(self) -> int:
        return self._index.ntotal if self._index else 0

    def clear(self) -> None:
        """Remove all data from the store."""
        self._index = None
        self._documents = []
        self._metadatas = []
        self._ids = []
        if self._index_path.exists():
            self._index_path.unlink()
        if self._meta_path.exists():
            self._meta_path.unlink()
        logger.info("Vector store cleared")


def get_store(persist_dir: str | Path | None = None) -> VectorStore:
    """Get a VectorStore instance (convenience function)."""
    return VectorStore(persist_dir=persist_dir)


def _prepare_metadata(chunk: ParsedChunk) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "source_file": chunk.source_file,
        "source_type": chunk.source_type,
        "chunk_index": chunk.chunk_index,
    }
    for key, val in chunk.metadata.items():
        if isinstance(val, (str, int, float, bool)):
            meta[key] = val
        elif isinstance(val, list) and all(isinstance(v, int) for v in val):
            meta[key] = ",".join(str(v) for v in val)
    return meta
