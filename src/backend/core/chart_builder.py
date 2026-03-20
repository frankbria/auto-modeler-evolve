"""chart_builder.py

Generates Recharts-compatible chart configuration objects from pandas data.
The frontend receives these as JSON and passes them directly to Recharts components.

Chart spec shape (all charts):
  {
    "chart_type": "bar" | "line" | "histogram" | "scatter" | "pie" | "heatmap" | "boxplot",
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

For "boxplot" chart type (distribution comparison):
  "data": [{"group": str, "min": f, "q1": f, "median": f, "q3": f, "max": f, "mean": f}, ...]
  "x_key": "group"
  "y_keys": ["min", "q1", "median", "q3", "max"]
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


def build_model_comparison_radar(
    models: list[dict],
    problem_type: str,
) -> dict[str, Any] | None:
    """Build a radar chart spec for comparing trained models side-by-side.

    Each model becomes a colored polygon on the radar. Metrics are normalized
    to [0, 1] so all spokes are comparable (higher always means better).

    Normalization rules:
      - r2, accuracy, f1, precision, recall  → already 0-1, clip at 0
      - mae, rmse                             → inverted: 1 - value/max_value
                                                (worst model → 0, best → 1)

    Args:
        models:       list of {"algorithm": str, "metrics": dict, "run_id": str}
        problem_type: "regression" or "classification"

    Returns None if fewer than 2 done models are provided (radar needs comparison).
    """
    done = [m for m in models if m.get("metrics") and m.get("algorithm")]
    if len(done) < 2:
        return None

    if problem_type == "regression":
        lower_is_better = ["mae", "rmse"]
        metric_labels = {"r2": "R²", "mae": "MAE Score", "rmse": "RMSE Score"}
    else:
        lower_is_better = []
        metric_labels = {
            "accuracy": "Accuracy",
            "f1": "F1 Score",
            "precision": "Precision",
            "recall": "Recall",
        }

    all_metrics = list(metric_labels.keys())

    # Gather raw values per metric across all models
    raw: dict[str, list[float]] = {m: [] for m in all_metrics}
    for model in done:
        metrics = model["metrics"]
        for key in all_metrics:
            raw[key].append(float(metrics.get(key, 0.0) or 0.0))

    # Normalize to [0, 1]
    def _normalize(key: str, values: list[float]) -> list[float]:
        if key in lower_is_better:
            mx = max(values) if values else 1.0
            if mx == 0:
                return [1.0] * len(values)
            return [max(0.0, 1.0 - v / mx) for v in values]
        else:
            return [max(0.0, min(1.0, v)) for v in values]

    normalized: dict[str, list[float]] = {
        key: _normalize(key, raw[key]) for key in all_metrics
    }

    # Build radar data: one dict per metric-spoke
    data = []
    for metric_key in all_metrics:
        entry: dict[str, Any] = {"metric": metric_labels[metric_key]}
        for i, model in enumerate(done):
            algo_name = model["algorithm"].replace("_", " ").title()
            entry[algo_name] = round(normalized[metric_key][i], 3)
        data.append(entry)

    algo_names = [m["algorithm"].replace("_", " ").title() for m in done]

    return {
        "chart_type": "radar",
        "title": "Model Comparison",
        "data": data,
        "x_key": "metric",
        "y_keys": algo_names,
        "x_label": "Metric",
        "y_label": "Normalized Score (higher = better)",
    }


def build_timeseries_chart(
    dates: list,
    values: list[float],
    column_name: str,
    window: int = 7,
) -> dict[str, Any]:
    """Build a time-series line chart with original values and rolling average.

    Args:
        dates:       List of date labels (strings) for the x-axis.
        values:      Numeric values aligned with dates.
        column_name: The name of the value column (shown as y-label).
        window:      Rolling average window size (default 7). Auto-adjusted
                     to min(window, len//3) so it works on short series.

    Returns a chart spec with three series:
        - "{column_name}": raw values
        - "{window}-period average": rolling mean
        - "Trend": linear trend line
    The frontend renders this as a multi-series line chart.
    """
    n = len(values)
    if n == 0:
        return build_line_chart([], {}, f"{column_name} over time", x_label="Date", y_label=column_name)

    # Adjust window to dataset length
    effective_window = max(2, min(window, n // 3)) if n >= 6 else 1

    # Rolling average
    rolling_avg: list[Any] = []
    for i in range(n):
        start = max(0, i - effective_window + 1)
        chunk = [v for v in values[start:i + 1] if v is not None and not (isinstance(v, float) and np.isnan(v))]
        rolling_avg.append(round(sum(chunk) / len(chunk), 4) if chunk else None)

    # Linear trend (OLS via numpy)
    finite_indices = [i for i, v in enumerate(values) if v is not None and not (isinstance(v, float) and np.isnan(v))]
    finite_values = [values[i] for i in finite_indices]
    trend: list[Any] = [None] * n
    if len(finite_indices) >= 2:
        x_arr = np.array(finite_indices, dtype=float)
        y_arr = np.array(finite_values, dtype=float)
        # Fit y = m*x + b
        m, b = np.polyfit(x_arr, y_arr, 1)
        for i in range(n):
            trend[i] = round(float(m * i + b), 4)

    avg_label = f"{effective_window}-period avg"
    y_series = {
        column_name: [_jsonify(v) for v in values],
        avg_label: rolling_avg,
        "Trend": trend,
    }

    return build_line_chart(
        x_values=[str(d) for d in dates],
        y_series=y_series,
        title=f"{column_name} over time",
        x_label="Date",
        y_label=column_name,
    )


def build_boxplot(
    df: pd.DataFrame,
    value_col: str,
    group_col: str | None = None,
    title: str = "",
    limit: int = 15,
) -> dict[str, Any]:
    """Box-and-whisker chart for distribution comparison.

    When group_col is None: a single box for the entire value column.
    When group_col is provided: one box per category, sorted by median descending.

    Each box encodes:
      min, Q1 (25th pct), median, Q3 (75th pct), max, mean
    Whiskers use the Tukey fence (1.5 × IQR), capped at actual data range
    so the chart shows the non-outlier spread naturally.

    Returns chart_type="boxplot" which the frontend renders as SVG boxes.
    """
    col_data = df[value_col].dropna()
    if col_data.empty:
        return {
            "chart_type": "boxplot",
            "title": title or f"Distribution of {value_col}",
            "data": [],
            "x_key": "group",
            "y_keys": ["min", "q1", "median", "q3", "max"],
            "x_label": group_col or "",
            "y_label": value_col,
        }

    def _box_stats(series: pd.Series) -> dict[str, Any]:
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        fence_lo = q1 - 1.5 * iqr
        fence_hi = q3 + 1.5 * iqr
        non_outliers = series[(series >= fence_lo) & (series <= fence_hi)]
        return {
            "min": round(float(non_outliers.min()) if not non_outliers.empty else series.min(), 4),
            "q1": round(q1, 4),
            "median": round(float(series.median()), 4),
            "q3": round(q3, 4),
            "max": round(float(non_outliers.max()) if not non_outliers.empty else series.max(), 4),
            "mean": round(float(series.mean()), 4),
            "count": int(series.count()),
        }

    if group_col is None or group_col not in df.columns:
        stats = _box_stats(col_data)
        data = [{"group": value_col, **stats}]
    else:
        groups = df[group_col].dropna().unique()
        rows = []
        for grp in groups:
            subset = df.loc[df[group_col] == grp, value_col].dropna()
            if subset.empty:
                continue
            stats = _box_stats(subset)
            rows.append({"group": str(grp), **stats})
        # Sort by median descending, cap at limit
        rows.sort(key=lambda r: r["median"], reverse=True)
        data = rows[:limit]

    return {
        "chart_type": "boxplot",
        "title": title or (
            f"Distribution of {value_col} by {group_col}"
            if group_col
            else f"Distribution of {value_col}"
        ),
        "data": data,
        "x_key": "group",
        "y_keys": ["min", "q1", "median", "q3", "max"],
        "x_label": group_col or value_col,
        "y_label": value_col,
    }


def _jsonify(value: Any) -> Any:
    """Convert numpy scalars and NaN to JSON-safe Python types."""
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
