"""Builds a NetworkX knowledge graph from the structured CSV tax data.

Graph structure:
  Node types: TaxpayerType, State, IncomeSource, DeductionType, TaxYear
  Edges carry aggregated financial metrics (count, totals, averages)
  enabling relationship queries that vector search cannot answer.

Example edges:
  (TaxpayerType:Corporation) --[FILED_IN {count, avg_rate, ...}]--> (State:CA)
  (State:TX) --[HAS_SOURCE {count, total_income, ...}]--> (IncomeSource:Royalties)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd
from django.conf import settings

logger = logging.getLogger(__name__)

CATEGORICAL_COLS = [
    "Taxpayer Type",
    "Tax Year",
    "Income Source",
    "Deduction Type",
    "State",
]

EDGE_PAIRS = [
    ("Taxpayer Type", "State", "FILED_IN"),
    ("Taxpayer Type", "Income Source", "EARNED_FROM"),
    ("Taxpayer Type", "Deduction Type", "CLAIMED"),
    ("Taxpayer Type", "Tax Year", "IN_YEAR"),
    ("State", "Income Source", "HAS_SOURCE"),
    ("State", "Deduction Type", "HAS_DEDUCTION"),
    ("State", "Tax Year", "STATE_YEAR"),
]


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """Build a knowledge graph from the tax CSV DataFrame."""
    G = nx.DiGraph()
    _add_nodes(G, df)
    _add_edges(G, df)
    _add_global_stats(G, df)
    logger.info(
        "Built graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
    )
    return G


def save_graph(G: nx.DiGraph, path: str | Path | None = None) -> Path:
    path = Path(path or settings.GRAPH_PERSIST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)
    logger.info("Saved graph to %s", path)
    return path


def load_graph(path: str | Path | None = None) -> nx.DiGraph:
    path = Path(path or settings.GRAPH_PERSIST_PATH)
    with open(path, "rb") as f:
        G = pickle.load(f)
    logger.info("Loaded graph from %s (%d nodes)", path, G.number_of_nodes())
    return G


def _add_nodes(G: nx.DiGraph, df: pd.DataFrame) -> None:
    for col in CATEGORICAL_COLS:
        node_type = col.replace(" ", "")
        for val in df[col].dropna().unique():
            node_id = f"{node_type}:{val}"
            subset = df[df[col] == val]
            G.add_node(
                node_id,
                node_type=node_type,
                label=str(val),
                record_count=len(subset),
                total_income=float(subset["Income"].sum()),
                total_deductions=float(subset["Deductions"].sum()),
                total_taxable=float(subset["Taxable Income"].sum()),
                total_tax_owed=float(subset["Tax Owed"].sum()),
                avg_tax_rate=float(subset["Tax Rate"].mean()),
                avg_income=float(subset["Income"].mean()),
            )


def _add_edges(G: nx.DiGraph, df: pd.DataFrame) -> None:
    for col_a, col_b, rel_type in EDGE_PAIRS:
        type_a = col_a.replace(" ", "")
        type_b = col_b.replace(" ", "")

        grouped = df.groupby([col_a, col_b])
        for (val_a, val_b), subset in grouped:
            src = f"{type_a}:{val_a}"
            dst = f"{type_b}:{val_b}"
            G.add_edge(
                src,
                dst,
                relation=rel_type,
                count=len(subset),
                total_income=float(subset["Income"].sum()),
                total_deductions=float(subset["Deductions"].sum()),
                total_taxable=float(subset["Taxable Income"].sum()),
                total_tax_owed=float(subset["Tax Owed"].sum()),
                avg_tax_rate=float(subset["Tax Rate"].mean()),
                avg_income=float(subset["Income"].mean()),
                min_tax_rate=float(subset["Tax Rate"].min()),
                max_tax_rate=float(subset["Tax Rate"].max()),
            )


def _add_global_stats(G: nx.DiGraph, df: pd.DataFrame) -> None:
    """Store dataset-wide statistics as a graph attribute."""
    G.graph["total_records"] = len(df)
    G.graph["years"] = sorted(df["Tax Year"].unique().tolist())
    G.graph["states"] = sorted(df["State"].unique().tolist())
    G.graph["taxpayer_types"] = sorted(df["Taxpayer Type"].unique().tolist())
    G.graph["income_sources"] = sorted(df["Income Source"].unique().tolist())
    G.graph["deduction_types"] = sorted(df["Deduction Type"].unique().tolist())
    G.graph["total_income"] = float(df["Income"].sum())
    G.graph["total_tax_owed"] = float(df["Tax Owed"].sum())
    G.graph["avg_tax_rate"] = float(df["Tax Rate"].mean())
