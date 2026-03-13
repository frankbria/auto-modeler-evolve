"""chart_builder.py

Generates Recharts-compatible chart configuration objects from pandas data.
The frontend receives these as JSON and passes them directly to Recharts components.

Chart spec shape (all charts):
  {
    "chart_type": "bar" | "line" | "histogram" | "scatter" | "pie" | "heatmap",
    "title": str,
    "data": [...],           # Recharts data array
    "x_key": str,            # key used for x-axis
    "y_keys": [str, ...],    # keys used for y-axis series
    "x_label": str,
    "y_label": str,
  }

For "heatmap" chart type (correlation matrix):
  "data": [{"row": "col_a", "col_a": 1.0, "col_b": 0.85, ...}, ...]
  "x_key": "row"
  "y_keys": [list of column names]
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_bar_chart(
    data: dict[str, float] | pd.Series,
    title: str,
    x_label: str = "",
    y_label: str = "Value",
    limit: int = 20,
) -> dict[str, Any]:
    """Vertical bar chart from a label→value mapping or Series."""
    if isinstance(data, pd.Series):
        items = list(zip(data.index.astype(str).tolist(), data.values.tolist()))
    else:
        items = [(str(k), v) for k, v in data.items()]

    # Sort descending, cap at limit
    items.sort(key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)
    items = items[:limit]

    chart_data = [{"label": k, "value": _jsonify(v)} for k, v in items]

    return {
        "chart_type": "bar",
        "title": title,
        "data": chart_data,
        "x_key": "label",
        "y_keys": ["value"],
        "x_label": x_label,
        "y_label": y_label,
    }


def build_line_chart(
    x_values: list,
    y_series: dict[str, list],
    title: str,
    x_label: str = "",
    y_label: str = "Value",
) -> dict[str, Any]:
    """Multi-series line chart.

    x_values: list of x-axis labels (strings or numbers)
    y_series: {"Series Name": [v1, v2, ...], ...}
    """
    chart_data = []
    for i, x in enumerate(x_values):
        point: dict[str, Any] = {"x": _jsonify(x)}
        for series_name, values in y_series.items():
            if i < len(values):
                point[series_name] = _jsonify(values[i])
        chart_data.append(point)

    return {
        "chart_type": "line",
        "title": title,
        "data": chart_data,
        "x_key": "x",
        "y_keys": list(y_series.keys()),
        "x_label": x_label,
        "y_label": y_label,
    }


def build_histogram(
    bins: list[float],
    counts: list[int],
    title: str,
    x_label: str = "Value",
    y_label: str = "Count",
) -> dict[str, Any]:
    """Histogram from pre-computed bins and counts (matches numpy.histogram output)."""
    chart_data = [
        {"bin": round(b, 4), "count": c} for b, c in zip(bins, counts)
    ]
    return {
        "chart_type": "histogram",
        "title": title,
        "data": chart_data,
        "x_key": "bin",
        "y_keys": ["count"],
        "x_label": x_label,
        "y_label": y_label,
    }


def build_scatter_chart(
    x_values: list,
    y_values: list,
    title: str,
    x_label: str = "X",
    y_label: str = "Y",
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Scatter plot from paired x/y value lists."""
    chart_data = []
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        point: dict[str, Any] = {"x": _jsonify(x), "y": _jsonify(y)}
        if labels and i < len(labels):
            point["label"] = str(labels[i])
        chart_data.append(point)

    return {
        "chart_type": "scatter",
        "title": title,
        "data": chart_data,
        "x_key": "x",
        "y_keys": ["y"],
        "x_label": x_label,
        "y_label": y_label,
    }


def build_pie_chart(
    data: dict[str, float] | pd.Series,
    title: str,
    limit: int = 10,
) -> dict[str, Any]:
    """Pie / donut chart from a label→value mapping."""
    if isinstance(data, pd.Series):
        items = list(zip(data.index.astype(str).tolist(), data.values.tolist()))
    else:
        items = [(str(k), v) for k, v in data.items()]

    items.sort(key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)

    # Roll up tail into "Other"
    if len(items) > limit:
        top = items[:limit]
        other_total = sum(v for _, v in items[limit:] if isinstance(v, (int, float)))
        if other_total > 0:
            top.append(("Other", other_total))
        items = top

    chart_data = [{"name": k, "value": _jsonify(v)} for k, v in items]
    return {
        "chart_type": "pie",
        "title": title,
        "data": chart_data,
        "x_key": "name",
        "y_keys": ["value"],
        "x_label": "",
        "y_label": "",
    }


def chart_from_query_result(
    result_df: pd.DataFrame | pd.Series,
    question: str,
    x_col: str | None = None,
    y_col: str | None = None,
) -> dict[str, Any] | None:
    """Auto-select the best chart type for a query result.

    Heuristics:
    - 1-column Series or single-column DF with string index → bar chart
    - 2-column DF where one is numeric → bar or line chart
    - 1 scalar → no chart (text-only answer is fine)
    """
    if isinstance(result_df, pd.Series):
        result_df = result_df.reset_index()
        result_df.columns = ["label", "value"]

    if result_df.empty:
        return None

    if len(result_df.columns) == 1:
        # Single numeric column — histogram
        col = result_df.columns[0]
        if pd.api.types.is_numeric_dtype(result_df[col]):
            counts, bins = np.histogram(result_df[col].dropna(), bins=min(15, result_df[col].nunique()))
            return build_histogram(
                bins=[round(float(b), 4) for b in bins[:-1]],
                counts=[int(c) for c in counts],
                title=f"Distribution of {col}",
                x_label=col,
            )
        return None

    if len(result_df.columns) == 2:
        col_a, col_b = result_df.columns[0], result_df.columns[1]
        # Determine which is categorical, which is numeric
        if x_col is None:
            x_col = col_a if not pd.api.types.is_numeric_dtype(result_df[col_a]) else col_b
        if y_col is None:
            y_col = col_b if x_col == col_a else col_a

        if pd.api.types.is_numeric_dtype(result_df[y_col]):
            data = dict(zip(result_df[x_col].astype(str), result_df[y_col]))
            return build_bar_chart(data, title=question[:60], x_label=str(x_col), y_label=str(y_col))

    # Multi-column with a clear x-axis
    if x_col and len(result_df.columns) > 2:
        numeric_cols = [c for c in result_df.columns if c != x_col and pd.api.types.is_numeric_dtype(result_df[c])]
        if numeric_cols:
            y_series = {c: result_df[c].tolist() for c in numeric_cols[:3]}
            return build_line_chart(
                x_values=result_df[x_col].tolist(),
                y_series=y_series,
                title=question[:60],
                x_label=str(x_col),
            )

    return None


def build_correlation_heatmap(
    corr_matrix: list[dict],
    columns: list[str],
    title: str = "Correlation Matrix",
) -> dict[str, Any]:
    """Build a heatmap chart spec from a pre-computed correlation matrix.

    corr_matrix: row-oriented list from _corr_matrix_dict, each entry has
      {"column": col_name, col1: val, col2: val, ...}
    columns: ordered list of column names in the matrix

    Returns a chart spec with chart_type="heatmap". The frontend renders each
    cell with a color based on the correlation value (-1 → red, 0 → white, 1 → blue).
    """
    # Normalise: rename "column" key to "row" for clarity in the frontend
    data = []
    for row in corr_matrix:
        entry: dict[str, Any] = {"row": row.get("column", "")}
        for col in columns:
            val = row.get(col)
            entry[col] = round(float(val), 3) if val is not None else None
        data.append(entry)

    return {
        "chart_type": "heatmap",
        "title": title,
        "data": data,
        "x_key": "row",
        "y_keys": columns,
        "x_label": "",
        "y_label": "",
    }


def _jsonify(value: Any) -> Any:
    """Convert numpy scalars and NaN to JSON-safe Python types."""
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
