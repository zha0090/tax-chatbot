"""Management command to ingest all data sources into the vector store.

Usage:
    python manage.py ingest                  # ingest all sources
    python manage.py ingest --source csv     # ingest CSV only
    python manage.py ingest --clear          # clear store before ingesting
"""

from __future__ import annotations

import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

import pandas as pd

from ingestion.embedder import embed_texts
from ingestion.parsers.csv_parser import parse_csv
from ingestion.parsers.pdf_parser import parse_pdf
from ingestion.parsers.ppt_parser import parse_ppt
from retrieval.graph_builder import build_graph, save_graph
from retrieval.vector_search import VectorStore

REFERS_DIR = Path(settings.BASE_DIR) / "refers"

SOURCE_FILES = {
    "csv": REFERS_DIR / "tax_data.csv",
    "pdf_1040": REFERS_DIR / "i1040gi.pdf",
    "pdf_irc": REFERS_DIR / "usc26@118-78.pdf",
    "ppt": REFERS_DIR / "MIC_3e_Ch11.ppt",
}

PDF_IRC_MAX_PAGES = 200


class Command(BaseCommand):
    help = "Ingest datasets into the ChromaDB vector store"

    def add_arguments(self, parser):
        parser.add_argument(
            "--source",
            type=str,
            choices=["csv", "pdf_1040", "pdf_irc", "ppt", "all"],
            default="all",
            help="Which source to ingest (default: all)",
        )
        parser.add_argument(
            "--clear",
            action="store_true",
            help="Clear the vector store before ingesting",
        )
        parser.add_argument(
            "--irc-max-pages",
            type=int,
            default=PDF_IRC_MAX_PAGES,
            help=f"Max pages to parse from the IRC PDF (default: {PDF_IRC_MAX_PAGES})",
        )

    def handle(self, *args, **options):
        source = options["source"]
        irc_max_pages = options["irc_max_pages"]

        store = VectorStore()

        if options["clear"]:
            self.stdout.write("Clearing vector store...")
            store.clear()

        sources = (
            list(SOURCE_FILES.keys()) if source == "all" else [source]
        )

        total_indexed = 0
        for src in sources:
            file_path = SOURCE_FILES[src]
            if not file_path.exists():
                self.stderr.write(f"  File not found: {file_path}, skipping.")
                continue

            self.stdout.write(f"\n{'='*60}")
            self.stdout.write(f"Ingesting: {src} ({file_path.name})")
            self.stdout.write(f"{'='*60}")

            t0 = time.time()
            chunks = self._parse_source(src, file_path, irc_max_pages)
            parse_time = time.time() - t0
            self.stdout.write(
                f"  Parsed {len(chunks)} chunks in {parse_time:.1f}s"
            )

            if not chunks:
                continue

            t0 = time.time()
            texts = [c.text for c in chunks]
            embeddings = embed_texts(texts)
            embed_time = time.time() - t0
            self.stdout.write(
                f"  Embedded {len(embeddings)} chunks in {embed_time:.1f}s"
            )

            t0 = time.time()
            indexed = store.index_chunks(chunks, embeddings)
            index_time = time.time() - t0
            self.stdout.write(
                f"  Indexed {indexed} chunks in {index_time:.1f}s"
            )

            total_indexed += indexed

        csv_path = SOURCE_FILES.get("csv")
        if csv_path and csv_path.exists():
            self.stdout.write("\nBuilding knowledge graph from CSV...")
            t0 = time.time()
            df = pd.read_csv(csv_path)
            G = build_graph(df)
            save_graph(G)
            self.stdout.write(
                f"  Graph: {G.number_of_nodes()} nodes, "
                f"{G.number_of_edges()} edges in {time.time() - t0:.1f}s"
            )

        self.stdout.write(
            self.style.SUCCESS(
                f"\nDone. Indexed {total_indexed} new chunks. "
                f"Total in store: {store.count()}."
            )
        )

    def _parse_source(self, source_key, file_path, irc_max_pages):
        if source_key == "csv":
            return parse_csv(file_path)
        elif source_key == "pdf_1040":
            return parse_pdf(file_path, chunk_size=1000, chunk_overlap=200)
        elif source_key == "pdf_irc":
            return parse_pdf(
                file_path,
                chunk_size=1500,
                chunk_overlap=300,
                max_pages=irc_max_pages,
            )
        elif source_key == "ppt":
            return parse_ppt(file_path)
        return []
