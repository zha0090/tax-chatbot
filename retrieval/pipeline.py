"""Hybrid retrieval pipeline: orchestrates vector, graph, and structured search.

This is the central module that:
1. Classifies the user query via the router
2. Fetches context from the appropriate lanes
3. Merges results into a unified context string
4. Sends context + query to the LLM for answer generation
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from django.conf import settings
from openai import OpenAI

from retrieval.graph_builder import load_graph
from retrieval.graph_search import graph_context_for_query
from retrieval.router import classify_query
from retrieval.vector_search import VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are TaxGPT, a knowledgeable financial and tax assistant. Answer the user's question using ONLY the provided context. Follow these rules:

1. Base your answer strictly on the context provided. Do not make up information.
2. If the context contains exact numbers, use them precisely.
3. If the context is insufficient, say so clearly rather than guessing.
4. For tax rule questions, cite the relevant section or form when available.
5. For data questions, mention the number of records or time period when relevant.
6. Be concise but thorough. Use bullet points for comparisons or lists.
7. When showing financial figures, format them clearly with $ signs and commas."""


class ChatPipeline:
    """Orchestrates the full query-to-answer pipeline.

    Thread-safe: lazy-loaded resources are guarded by a lock so the pipeline
    can be shared across Django's request threads via a module-level singleton.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._vector_store: VectorStore | None = None
        self._graph: nx.DiGraph | None = None
        self._df: pd.DataFrame | None = None
        self._openai_client: OpenAI | None = None

    @property
    def openai_client(self) -> OpenAI:
        if self._openai_client is None:
            with self._lock:
                if self._openai_client is None:
                    self._openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._openai_client

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            with self._lock:
                if self._vector_store is None:
                    self._vector_store = VectorStore()
        return self._vector_store

    @property
    def graph(self) -> nx.DiGraph | None:
        if self._graph is None:
            with self._lock:
                if self._graph is None:
                    graph_path = Path(settings.GRAPH_PERSIST_PATH)
                    if graph_path.exists():
                        try:
                            self._graph = load_graph(graph_path)
                        except Exception:
                            logger.exception("Failed to load knowledge graph")
                    else:
                        logger.warning("Knowledge graph not found at %s", graph_path)
        return self._graph

    @property
    def df(self) -> pd.DataFrame | None:
        if self._df is None:
            with self._lock:
                if self._df is None:
                    csv_path = Path(settings.CSV_DATA_PATH)
                    if csv_path.exists():
                        try:
                            self._df = pd.read_csv(csv_path)
                            self._df.columns = self._df.columns.str.strip()
                        except Exception:
                            logger.exception("Failed to load CSV data")
                    else:
                        logger.warning("CSV data not found at %s", csv_path)
        return self._df

    def answer(self, query: str) -> dict[str, Any]:
        """Process a user query through the full hybrid retrieval pipeline."""
        routing = classify_query(query, client=self.openai_client)
        lanes = routing["lanes"]
        entities = routing["entities"]
        search_query = routing.get("rewritten_query", query)

        logger.info("Routing: lanes=%s, entities=%s", lanes, entities)

        if "chitchat" in lanes:
            return self._handle_chitchat(query, lanes)

        context_parts: list[tuple[str, str]] = []
        sources: list[str] = []

        if "vector" in lanes:
            vec_ctx, vec_sources = self._vector_search(search_query)
            if vec_ctx:
                context_parts.append(("DOCUMENT SEARCH RESULTS", vec_ctx))
                sources.extend(vec_sources)

        if "graph" in lanes and self.graph is not None:
            graph_ctx = graph_context_for_query(self.graph, entities)
            if graph_ctx:
                context_parts.append(("KNOWLEDGE GRAPH DATA", graph_ctx))
                sources.append("knowledge_graph")

        if "structured" in lanes and self.df is not None:
            struct_ctx = self._structured_search(entities)
            if struct_ctx:
                context_parts.append(("STRUCTURED DATA ANALYSIS", struct_ctx))
                sources.append("tax_data.csv")

        if not context_parts:
            context_parts.append(
                ("NOTE", "No specific data was found for this query.")
            )

        context = self._format_context(context_parts)
        answer = self._generate_answer(query, context)

        return {
            "answer": answer,
            "sources": list(set(sources)),
            "routing_info": {"lanes": lanes, "entities": entities},
        }

    def _handle_chitchat(self, query: str, lanes: list[str]) -> dict[str, Any]:
        """Respond to greetings and capability questions without hitting retrieval."""
        return {
            "answer": (
                "Hello! I'm TaxGPT, a financial and tax assistant. I can help you with:\n\n"
                "- **Tax data analysis**: Average tax rates, totals, comparisons across "
                "states, taxpayer types, and years (5,000 records)\n"
                "- **IRS Form 1040 instructions**: Filing requirements, deductions, credits, deadlines\n"
                "- **US Tax Code (Title 26)**: Legal provisions and cross-references\n"
                "- **Tax economics**: Concepts like excise taxes, elasticity, and welfare analysis\n\n"
                "Try asking something specific like:\n"
                '- "What is the average tax rate for corporations in California?"\n'
                '- "What are the standard deduction amounts for 2023?"\n'
                '- "Compare tax rates between partnerships and individuals"'
            ),
            "sources": [],
            "routing_info": {"lanes": lanes, "entities": {}},
        }

    def _vector_search(self, query: str) -> tuple[str, list[str]]:
        """Retrieve semantically similar chunks from the vector store."""
        if self.vector_store.count() == 0:
            return "", []

        results = self.vector_store.search(
            query, n_results=settings.RETRIEVAL_TOP_K
        )
        if not results:
            return "", []

        parts = []
        sources = []
        for i, hit in enumerate(results, 1):
            src = hit["metadata"].get("source_file", "unknown")
            parts.append(f"[Result {i} from {src}]:\n{hit['text']}")
            sources.append(src)

        return "\n\n".join(parts), sources

    def _structured_search(self, entities: dict) -> str:
        """Run pandas queries on the CSV data for exact numerical answers."""
        df = self.df
        if df is None:
            return ""

        filters = []
        if entities.get("taxpayer_type"):
            filters.append(df["Taxpayer Type"] == entities["taxpayer_type"])
        if entities.get("state"):
            filters.append(df["State"] == entities["state"])
        if entities.get("tax_year"):
            try:
                filters.append(df["Tax Year"] == int(entities["tax_year"]))
            except (ValueError, TypeError):
                pass
        if entities.get("income_source"):
            filters.append(df["Income Source"] == entities["income_source"])
        if entities.get("deduction_type"):
            filters.append(df["Deduction Type"] == entities["deduction_type"])

        if not filters:
            subset = df
        else:
            mask = filters[0]
            for f in filters[1:]:
                mask = mask & f
            subset = df[mask]

        if subset.empty:
            return "No matching records found for the given criteria."

        parts = [f"Matched {len(subset)} records from the tax dataset:"]
        parts.append(f"  Average Income: ${subset['Income'].mean():,.2f}")
        parts.append(f"  Total Income: ${subset['Income'].sum():,.2f}")
        parts.append(f"  Average Deductions: ${subset['Deductions'].mean():,.2f}")
        parts.append(f"  Total Deductions: ${subset['Deductions'].sum():,.2f}")
        parts.append(
            f"  Average Taxable Income: ${subset['Taxable Income'].mean():,.2f}"
        )
        parts.append(f"  Average Tax Rate: {subset['Tax Rate'].mean():.2%}")
        parts.append(f"  Min Tax Rate: {subset['Tax Rate'].min():.2%}")
        parts.append(f"  Max Tax Rate: {subset['Tax Rate'].max():.2%}")
        parts.append(f"  Total Tax Owed: ${subset['Tax Owed'].sum():,.2f}")
        parts.append(f"  Average Tax Owed: ${subset['Tax Owed'].mean():,.2f}")

        if not entities.get("state"):
            by_state = (
                subset.groupby("State")["Tax Rate"]
                .mean()
                .sort_values(ascending=False)
            )
            parts.append("\n  Breakdown by State (avg tax rate):")
            for st, rate in by_state.items():
                count = len(subset[subset["State"] == st])
                parts.append(f"    {st}: {rate:.2%} ({count} records)")

        if not entities.get("taxpayer_type"):
            by_type = (
                subset.groupby("Taxpayer Type")["Tax Rate"]
                .mean()
                .sort_values(ascending=False)
            )
            parts.append("\n  Breakdown by Taxpayer Type (avg tax rate):")
            for tp, rate in by_type.items():
                count = len(subset[subset["Taxpayer Type"] == tp])
                parts.append(f"    {tp}: {rate:.2%} ({count} records)")

        return "\n".join(parts)

    def _format_context(self, parts: list[tuple[str, str]]) -> str:
        return "\n\n".join(f"=== {title} ===\n{content}" for title, content in parts)

    def _generate_answer(self, query: str, context: str) -> str:
        """Send retrieved context + user query to the LLM for answer generation."""
        try:
            response = self.openai_client.chat.completions.create(
                model=settings.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}",
                    },
                ],
                temperature=settings.OPENAI_CHAT_TEMPERATURE,
                max_tokens=settings.OPENAI_CHAT_MAX_TOKENS,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            logger.exception("LLM generation failed")
            return "Sorry, I encountered an error generating the answer. Please try again."


_pipeline_lock = threading.Lock()
_pipeline: ChatPipeline | None = None


def get_pipeline() -> ChatPipeline:
    """Get or create a singleton ChatPipeline instance (thread-safe)."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = ChatPipeline()
    return _pipeline
