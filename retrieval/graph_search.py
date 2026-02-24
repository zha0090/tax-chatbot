"""Query interface for the knowledge graph.

Provides structured lookups that return natural-language answers
from the graph's aggregated data -- things vector search is bad at,
like comparisons, rankings, and exact numerical aggregations.
"""

from __future__ import annotations

from typing import Any

import networkx as nx


def get_node_info(G: nx.DiGraph, node_id: str) -> dict[str, Any] | None:
    """Get full attributes for a node."""
    if node_id not in G:
        return None
    return {"id": node_id, **G.nodes[node_id]}


def get_edge_info(
    G: nx.DiGraph, src: str, dst: str
) -> dict[str, Any] | None:
    """Get edge attributes between two nodes."""
    if not G.has_edge(src, dst):
        return None
    return {"src": src, "dst": dst, **G.edges[src, dst]}


def find_related(
    G: nx.DiGraph,
    node_id: str,
    relation: str | None = None,
    direction: str = "outgoing",
) -> list[dict[str, Any]]:
    """Find nodes connected to a given node, optionally filtered by relation type."""
    if node_id not in G:
        return []

    results = []

    if direction in ("outgoing", "both"):
        for _, dst, data in G.out_edges(node_id, data=True):
            if relation and data.get("relation") != relation:
                continue
            results.append({"node": dst, "direction": "outgoing", **data})

    if direction in ("incoming", "both"):
        for src, _, data in G.in_edges(node_id, data=True):
            if relation and data.get("relation") != relation:
                continue
            results.append({"node": src, "direction": "incoming", **data})

    return results


def query_by_type_and_state(
    G: nx.DiGraph, taxpayer_type: str, state: str
) -> dict[str, Any] | None:
    """Get aggregated tax data for a specific taxpayer type in a state."""
    src = f"TaxpayerType:{taxpayer_type}"
    dst = f"State:{state}"
    return get_edge_info(G, src, dst)


def rank_by_metric(
    G: nx.DiGraph,
    node_type: str,
    metric: str,
    top_n: int = 5,
    ascending: bool = False,
) -> list[dict[str, Any]]:
    """Rank nodes of a given type by a metric (e.g., avg_tax_rate)."""
    nodes = [
        (nid, attrs)
        for nid, attrs in G.nodes(data=True)
        if attrs.get("node_type") == node_type and metric in attrs
    ]
    nodes.sort(key=lambda x: x[1][metric], reverse=not ascending)
    return [
        {"id": nid, "label": attrs["label"], metric: attrs[metric]}
        for nid, attrs in nodes[:top_n]
    ]


def compare_across(
    G: nx.DiGraph,
    source_node: str,
    relation: str,
    metric: str,
) -> list[dict[str, Any]]:
    """Compare a metric across all outgoing edges of a given relation.

    E.g., compare avg_tax_rate for Corporation across all states:
      compare_across(G, "TaxpayerType:Corporation", "FILED_IN", "avg_tax_rate")
    """
    results = []
    for _, dst, data in G.out_edges(source_node, data=True):
        if data.get("relation") != relation:
            continue
        if metric in data:
            target_label = G.nodes[dst].get("label", dst)
            results.append(
                {
                    "target": dst,
                    "label": target_label,
                    metric: data[metric],
                    "count": data.get("count", 0),
                }
            )
    results.sort(key=lambda x: x[metric], reverse=True)
    return results


def get_global_stats(G: nx.DiGraph) -> dict[str, Any]:
    """Return dataset-wide statistics stored in the graph."""
    return dict(G.graph)


def graph_context_for_query(G: nx.DiGraph, entities: dict[str, str]) -> str:
    """Build a natural-language context string from graph data given extracted entities.

    `entities` maps entity types to values, e.g.:
      {"taxpayer_type": "Corporation", "state": "CA"}
    """
    parts = []
    tp = entities.get("taxpayer_type")
    state = entities.get("state")
    income_src = entities.get("income_source")
    year = entities.get("tax_year")

    if tp and state:
        edge = query_by_type_and_state(G, tp, state)
        if edge:
            parts.append(
                f"{tp} taxpayers in {state}: "
                f"{edge['count']} records, "
                f"avg tax rate {edge['avg_tax_rate']:.2%}, "
                f"total income ${edge['total_income']:,.0f}, "
                f"total tax owed ${edge['total_tax_owed']:,.0f}."
            )

    if tp and not state:
        node = get_node_info(G, f"TaxpayerType:{tp}")
        if node:
            parts.append(
                f"{tp} taxpayers overall: "
                f"{node['record_count']} records, "
                f"avg tax rate {node['avg_tax_rate']:.2%}, "
                f"total income ${node['total_income']:,.0f}, "
                f"total tax owed ${node['total_tax_owed']:,.0f}."
            )
            state_data = compare_across(
                G, f"TaxpayerType:{tp}", "FILED_IN", "avg_tax_rate"
            )
            if state_data:
                top3 = state_data[:3]
                lines = [
                    f"  {d['label']}: {d['avg_tax_rate']:.2%} ({d['count']} records)"
                    for d in top3
                ]
                parts.append(f"Top states by avg tax rate for {tp}:\n" + "\n".join(lines))

    if state and not tp:
        node = get_node_info(G, f"State:{state}")
        if node:
            parts.append(
                f"State {state} overall: "
                f"{node['record_count']} records, "
                f"avg tax rate {node['avg_tax_rate']:.2%}, "
                f"total income ${node['total_income']:,.0f}."
            )

    if tp and income_src:
        edge = get_edge_info(
            G, f"TaxpayerType:{tp}", f"IncomeSource:{income_src}"
        )
        if edge:
            parts.append(
                f"{tp} with {income_src}: "
                f"{edge['count']} records, avg income ${edge['avg_income']:,.0f}."
            )

    if year:
        node = get_node_info(G, f"TaxYear:{year}")
        if node:
            parts.append(
                f"Tax year {year}: "
                f"{node['record_count']} records, "
                f"avg tax rate {node['avg_tax_rate']:.2%}, "
                f"total tax owed ${node['total_tax_owed']:,.0f}."
            )

    stats = get_global_stats(G)
    if stats:
        parts.append(
            f"Dataset totals: {stats.get('total_records', 0)} records, "
            f"overall avg tax rate {stats.get('avg_tax_rate', 0):.2%}."
        )

    return "\n\n".join(parts) if parts else ""
