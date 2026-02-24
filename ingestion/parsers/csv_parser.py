from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ParsedChunk:
    """A single chunk of parsed content with metadata."""

    text: str
    metadata: dict[str, Any]
    source_file: str
    source_type: str  # "csv", "pdf", "ppt"
    chunk_index: int


def parse_csv(file_path: str | Path) -> list[ParsedChunk]:
    """Parse a CSV file into chunks suitable for embedding and retrieval.

    Each row becomes a natural-language chunk with all column values.
    Returns both per-row chunks and aggregate summary chunks.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        pd.errors.ParserError: If the CSV is malformed.
    """
    file_path = Path(file_path)
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    columns = df.columns.tolist()
    chunks: list[ParsedChunk] = []

    for row in df.itertuples(index=True, name=None):
        idx = row[0]
        row_dict = {col: row[i + 1] for i, col in enumerate(columns)}
        text = _row_dict_to_text(row_dict)
        metadata = {col: _serialize(val) for col, val in row_dict.items()}
        metadata["row_index"] = int(idx)
        chunks.append(
            ParsedChunk(
                text=text,
                metadata=metadata,
                source_file=file_path.name,
                source_type="csv",
                chunk_index=int(idx),
            )
        )

    summaries = _build_summary_chunks(df, file_path.name, len(chunks))
    chunks.extend(summaries)
    return chunks


def get_dataframe(file_path: str | Path) -> pd.DataFrame:
    """Load the CSV into a DataFrame for structured queries."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df


def _row_dict_to_text(row_dict: dict[str, Any]) -> str:
    parts = []
    for col, val in row_dict.items():
        if val is not None and not (isinstance(val, float) and pd.isna(val)):
            parts.append(f"{col}: {val}")
    return ". ".join(parts) + "."


def _serialize(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if isinstance(val, (int, float, str, bool)):
        return val
    return str(val)


def _build_summary_chunks(
    df: pd.DataFrame, source_file: str, start_index: int
) -> list[ParsedChunk]:
    """Generate aggregate summary chunks for high-level queries."""
    summaries: list[ParsedChunk] = []
    chunk_idx = start_index

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    overview_parts = [
        f"Dataset '{source_file}' contains {len(df)} records.",
        f"Columns: {', '.join(df.columns)}.",
        f"Numeric columns: {', '.join(numeric_cols)}.",
        f"Categorical columns: {', '.join(categorical_cols)}.",
    ]

    for col in categorical_cols:
        values = df[col].dropna().unique()
        if len(values) <= 20:
            overview_parts.append(
                f"Unique values in '{col}': {', '.join(str(v) for v in sorted(values))}."
            )

    summaries.append(
        ParsedChunk(
            text=" ".join(overview_parts),
            metadata={"summary_type": "dataset_overview", "row_count": len(df)},
            source_file=source_file,
            source_type="csv",
            chunk_index=chunk_idx,
        )
    )
    chunk_idx += 1

    for col in categorical_cols:
        for val in df[col].dropna().unique():
            subset = df[df[col] == val]
            parts = [f"For {col} = '{val}' ({len(subset)} records):"]
            for nc in numeric_cols:
                mean = subset[nc].mean()
                total = subset[nc].sum()
                parts.append(f"  {nc}: avg={mean:,.2f}, total={total:,.2f}.")
            summaries.append(
                ParsedChunk(
                    text=" ".join(parts),
                    metadata={
                        "summary_type": "group_summary",
                        "group_column": col,
                        "group_value": str(val),
                    },
                    source_file=source_file,
                    source_type="csv",
                    chunk_index=chunk_idx,
                )
            )
            chunk_idx += 1

    return summaries
