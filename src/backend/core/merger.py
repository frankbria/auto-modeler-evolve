"""Multi-dataset merge utilities.

Provides join-key suggestions and merge execution for combining two CSVs
within the same project.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def suggest_join_keys(df1: pd.DataFrame, df2: pd.DataFrame) -> list[dict[str, Any]]:
    """Return candidate join columns that appear in both DataFrames.

    For each common column, returns:
    - name: column name
    - dtype_left / dtype_right: pandas dtype strings
    - unique_left / unique_right: number of distinct values
    - uniqueness_left / uniqueness_right: distinct / total (0–1 fraction)
    - recommended: True when both sides have high uniqueness (good join key)

    Sorted by combined uniqueness score descending, so the best candidate is first.
    """
    common = set(df1.columns) & set(df2.columns)
    suggestions: list[dict[str, Any]] = []

    for col in sorted(common):
        ul = df1[col].nunique() / max(len(df1), 1)
        ur = df2[col].nunique() / max(len(df2), 1)
        suggestions.append(
            {
                "name": col,
                "dtype_left": str(df1[col].dtype),
                "dtype_right": str(df2[col].dtype),
                "unique_left": int(df1[col].nunique()),
                "unique_right": int(df2[col].nunique()),
                "uniqueness_left": round(ul, 3),
                "uniqueness_right": round(ur, 3),
                # Recommended when at least one side is >50% unique values
                "recommended": ul > 0.5 or ur > 0.5,
            }
        )

    # Best candidates first
    suggestions.sort(
        key=lambda s: s["uniqueness_left"] + s["uniqueness_right"], reverse=True
    )
    return suggestions


def merge_datasets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    join_key: str,
    how: str = "inner",
    suffix_left: str = "_left",
    suffix_right: str = "_right",
) -> dict[str, Any]:
    """Merge two DataFrames on a shared column.

    Parameters
    ----------
    df1, df2:
        DataFrames to join.
    join_key:
        Column name present in both DataFrames.
    how:
        Pandas merge type: 'inner', 'left', 'right', or 'outer'.
    suffix_left, suffix_right:
        Appended to conflicting non-key column names.

    Returns
    -------
    dict with keys:
        merged_df   – the merged DataFrame
        row_count   – number of rows in merged result
        column_count – number of columns
        columns     – list of column names
        conflict_columns – list of column names that had the suffix applied
        preview_rows – first 10 rows as list[dict] (NaN/inf sanitized)
    """
    if join_key not in df1.columns:
        raise ValueError(f"Join key '{join_key}' not found in left dataset")
    if join_key not in df2.columns:
        raise ValueError(f"Join key '{join_key}' not found in right dataset")

    valid_hows = {"inner", "left", "right", "outer"}
    if how not in valid_hows:
        raise ValueError(f"how must be one of {sorted(valid_hows)}, got '{how}'")

    # Identify non-key columns that collide → will get suffixes
    non_key_left = set(df1.columns) - {join_key}
    non_key_right = set(df2.columns) - {join_key}
    conflict_cols = sorted(non_key_left & non_key_right)

    merged = df1.merge(df2, on=join_key, how=how, suffixes=(suffix_left, suffix_right))

    preview_rows = _sanitize_rows(merged.head(10).to_dict(orient="records"))

    return {
        "merged_df": merged,
        "row_count": len(merged),
        "column_count": len(merged.columns),
        "columns": list(merged.columns),
        "conflict_columns": conflict_cols,
        "preview_rows": preview_rows,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sanitize_rows(rows: list[dict]) -> list[dict[str, Any]]:
    """Replace NaN/inf floats with None for safe JSON serialization."""
    import math

    clean = []
    for row in rows:
        safe: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                safe[k] = None
            else:
                safe[k] = v
        clean.append(safe)
    return clean
