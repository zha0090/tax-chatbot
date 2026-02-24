"""Tests for the knowledge graph builder and search module."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from retrieval.graph_builder import build_graph, load_graph, save_graph
from retrieval.graph_search import (
    compare_across,
    find_related,
    get_edge_info,
    get_global_stats,
    get_node_info,
    graph_context_for_query,
    query_by_type_and_state,
    rank_by_metric,
)

REFERS_DIR = Path(__file__).resolve().parent.parent / "refers"
CSV_PATH = REFERS_DIR / "tax_data.csv"


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Small synthetic DataFrame for fast, deterministic tests."""
    return pd.DataFrame(
        {
            "Taxpayer Type": ["Corporation", "Corporation", "Individual", "Individual", "Trust"],
            "Tax Year": [2022, 2023, 2022, 2023, 2022],
            "Transaction Date": ["2022-01-01", "2023-01-01", "2022-06-01", "2023-06-01", "2022-03-01"],
            "Income Source": ["Salary", "Salary", "Capital Gains", "Salary", "Rental"],
            "Deduction Type": ["Mortgage Interest", "Business Expenses", "Education Expenses", "Mortgage Interest", "Business Expenses"],
            "State": ["CA", "CA", "TX", "TX", "NY"],
            "Income": [100000.0, 150000.0, 80000.0, 90000.0, 200000.0],
            "Deductions": [10000.0, 20000.0, 5000.0, 8000.0, 30000.0],
            "Taxable Income": [90000.0, 130000.0, 75000.0, 82000.0, 170000.0],
            "Tax Rate": [0.25, 0.28, 0.22, 0.24, 0.30],
            "Tax Owed": [22500.0, 36400.0, 16500.0, 19680.0, 51000.0],
        }
    )


@pytest.fixture()
def graph(sample_df) -> nx.DiGraph:
    return build_graph(sample_df)


@pytest.fixture()
def real_df():
    if not CSV_PATH.exists():
        pytest.skip("CSV test data not found")
    return pd.read_csv(CSV_PATH)


@pytest.fixture()
def real_graph(real_df) -> nx.DiGraph:
    return build_graph(real_df)


# ── Graph Builder Tests ───────────────────────────────────────────────


class TestGraphBuilder:
    def test_returns_digraph(self, graph):
        assert isinstance(graph, nx.DiGraph)

    def test_has_nodes(self, graph):
        assert graph.number_of_nodes() > 0

    def test_has_edges(self, graph):
        assert graph.number_of_edges() > 0

    def test_node_types_present(self, graph):
        node_types = {d["node_type"] for _, d in graph.nodes(data=True)}
        assert "TaxpayerType" in node_types
        assert "State" in node_types
        assert "TaxYear" in node_types
        assert "IncomeSource" in node_types
        assert "DeductionType" in node_types

    def test_nodes_have_metrics(self, graph):
        corp = graph.nodes.get("TaxpayerType:Corporation")
        assert corp is not None
        assert corp["record_count"] == 2
        assert corp["total_income"] == 250000.0
        assert 0.2 < corp["avg_tax_rate"] < 0.3

    def test_edges_have_metrics(self, graph):
        edge = graph.edges.get(("TaxpayerType:Corporation", "State:CA"))
        assert edge is not None
        assert edge["count"] == 2
        assert edge["relation"] == "FILED_IN"
        assert edge["total_income"] == 250000.0

    def test_global_stats(self, graph):
        stats = graph.graph
        assert stats["total_records"] == 5
        assert "CA" in stats["states"]
        assert 2022 in stats["years"]

    def test_save_and_load(self, graph, tmp_path):
        path = tmp_path / "test_graph.gpickle"
        save_graph(graph, path)
        loaded = load_graph(path)
        assert loaded.number_of_nodes() == graph.number_of_nodes()
        assert loaded.number_of_edges() == graph.number_of_edges()


class TestGraphBuilderRealData:
    def test_real_csv_builds(self, real_graph):
        assert real_graph.number_of_nodes() > 20
        assert real_graph.number_of_edges() > 50

    def test_real_all_states_present(self, real_graph):
        states = [
            n for n, d in real_graph.nodes(data=True) if d["node_type"] == "State"
        ]
        assert len(states) == 10

    def test_real_all_taxpayer_types(self, real_graph):
        types = [
            n for n, d in real_graph.nodes(data=True) if d["node_type"] == "TaxpayerType"
        ]
        assert len(types) == 5

    def test_real_record_counts_sum(self, real_graph, real_df):
        total_from_graph = sum(
            d["record_count"]
            for _, d in real_graph.nodes(data=True)
            if d["node_type"] == "TaxpayerType"
        )
        assert total_from_graph == len(real_df)


# ── Graph Search Tests ────────────────────────────────────────────────


class TestGraphSearch:
    def test_get_node_info(self, graph):
        info = get_node_info(graph, "TaxpayerType:Corporation")
        assert info is not None
        assert info["label"] == "Corporation"
        assert info["record_count"] == 2

    def test_get_node_info_missing(self, graph):
        assert get_node_info(graph, "TaxpayerType:Nonexistent") is None

    def test_get_edge_info(self, graph):
        info = get_edge_info(graph, "TaxpayerType:Corporation", "State:CA")
        assert info is not None
        assert info["relation"] == "FILED_IN"
        assert info["count"] == 2

    def test_query_by_type_and_state(self, graph):
        result = query_by_type_and_state(graph, "Corporation", "CA")
        assert result is not None
        assert result["count"] == 2
        assert result["avg_tax_rate"] == pytest.approx(0.265, abs=0.01)

    def test_find_related_outgoing(self, graph):
        related = find_related(
            graph, "TaxpayerType:Corporation", relation="FILED_IN"
        )
        assert len(related) == 1
        assert related[0]["node"] == "State:CA"

    def test_rank_by_metric(self, graph):
        ranked = rank_by_metric(graph, "State", "avg_tax_rate", top_n=3)
        assert len(ranked) > 0
        assert ranked[0]["avg_tax_rate"] >= ranked[-1]["avg_tax_rate"]

    def test_compare_across(self, graph):
        results = compare_across(
            graph, "TaxpayerType:Individual", "FILED_IN", "avg_tax_rate"
        )
        assert len(results) == 1
        assert results[0]["label"] == "TX"

    def test_global_stats(self, graph):
        stats = get_global_stats(graph)
        assert stats["total_records"] == 5

    def test_graph_context_with_type_and_state(self, graph):
        ctx = graph_context_for_query(
            graph, {"taxpayer_type": "Corporation", "state": "CA"}
        )
        assert "Corporation" in ctx
        assert "CA" in ctx
        assert "avg tax rate" in ctx

    def test_graph_context_type_only(self, graph):
        ctx = graph_context_for_query(graph, {"taxpayer_type": "Corporation"})
        assert "Corporation" in ctx
        assert "total income" in ctx

    def test_graph_context_state_only(self, graph):
        ctx = graph_context_for_query(graph, {"state": "CA"})
        assert "CA" in ctx

    def test_graph_context_empty_entities(self, graph):
        ctx = graph_context_for_query(graph, {})
        assert "Dataset totals" in ctx
