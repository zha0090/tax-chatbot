"""Hybrid retrieval pipeline: orchestrates vector, graph, and structured search.

This is the central module that:
1. Classifies the user query via the router
2. Fetches context from the appropriate lanes
3. Merges results into a unified context string
4. Sends context + query to GPT-4o-mini for answer generation
"""

from __future__ import annotations

import logging
from pathlib import Path

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
    """Orchestrates the full query-to-answer pipeline."""

    def __init__(self):
        self._vector_store: VectorStore | None = None
        self._graph = None
        self._df: pd.DataFrame | None = None

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore()
        return self._vector_store

    @property
    def graph(self):
        if self._graph is None:
            graph_path = Path(settings.GRAPH_PERSIST_PATH)
            if graph_path.exists():
                self._graph = load_graph(graph_path)
            else:
                logger.warning("Knowledge graph not found at %s", graph_path)
        return self._graph

    @property
    def df(self) -> pd.DataFrame | None:
        if self._df is None:
            csv_path = Path(settings.BASE_DIR) / "refers" / "tax_data.csv"
            if csv_path.exists():
                self._df = pd.read_csv(csv_path)
                self._df.columns = self._df.columns.str.strip()
        return self._df

    def answer(self, query: str) -> dict:
        """Process a user query through the full pipeline.

        Returns dict with: answer, sources, routing_info.
        """
        routing = classify_query(query)
        lanes = routing["lanes"]
        entities = routing["entities"]
        search_query = routing.get("rewritten_query", query)

        logger.info("Routing: lanes=%s, entities=%s", lanes, entities)

        context_parts = []
        sources = []

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
            "routing_info": {
                "lanes": lanes,
                "entities": entities,
            },
        }

    def _vector_search(
        self, query: str, n_results: int = 5
    ) -> tuple[str, list[str]]:
        if self.vector_store.count() == 0:
            return "", []

        results = self.vector_store.search(query, n_results=n_results)
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
            return ""

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

        if entities.get("taxpayer_type") and not entities.get("state"):
            by_state = (
                subset.groupby("State")["Tax Rate"]
                .mean()
                .sort_values(ascending=False)
            )
            parts.append("\n  Breakdown by State:")
            for state, rate in by_state.items():
                count = len(subset[subset["State"] == state])
                parts.append(f"    {state}: {rate:.2%} ({count} records)")

        if entities.get("state") and not entities.get("taxpayer_type"):
            by_type = (
                subset.groupby("Taxpayer Type")["Tax Rate"]
                .mean()
                .sort_values(ascending=False)
            )
            parts.append("\n  Breakdown by Taxpayer Type:")
            for tp, rate in by_type.items():
                count = len(subset[subset["Taxpayer Type"] == tp])
                parts.append(f"    {tp}: {rate:.2%} ({count} records)")

        return "\n".join(parts)

    def _format_context(self, parts: list[tuple[str, str]]) -> str:
        sections = []
        for title, content in parts:
            sections.append(f"=== {title} ===\n{content}")
        return "\n\n".join(sections)

    def _generate_answer(self, query: str, context: str) -> str:
        try:
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=settings.OPENAI_CHAT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {query}",
                    },
                ],
                temperature=0.1,
                max_tokens=800,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("LLM generation failed: %s", e)
            return f"Sorry, I encountered an error generating the answer: {e}"


_pipeline: ChatPipeline | None = None


def get_pipeline() -> ChatPipeline:
    """Get or create a singleton ChatPipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ChatPipeline()
    return _pipeline
