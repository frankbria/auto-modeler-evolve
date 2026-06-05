from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Basic column statistics (called on every upload)
# ---------------------------------------------------------------------------


def analyze_dataframe(df: pd.DataFrame) -> dict:
    """Return column statistics for a dataframe.

    For each column produces: dtype, non_null_count, null_count, null_pct,
    unique_count, and 5 sample_values. Numeric columns also get min, max,
    mean, and std.
    """
    columns = []

    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        null_count = int(series.isna().sum())
        total = len(series)
        null_pct = round(null_count / total * 100, 2) if total > 0 else 0.0

        sample_values = (
            series.dropna()
            .head(5)
            .apply(lambda v: v.item() if isinstance(v, np.generic) else v)
            .tolist()
        )

        stat: dict = {
            "name": col,
            "dtype": str(series.dtype),
            "non_null_count": non_null,
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_count": int(series.nunique()),
            "sample_values": sample_values,
        }

        if pd.api.types.is_numeric_dtype(series):
            stat["min"] = _safe_scalar(series.min())
            stat["max"] = _safe_scalar(series.max())
            stat["mean"] = _safe_scalar(series.mean())
            stat["std"] = _safe_scalar(series.std())

        columns.append(stat)

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns,
    }


# ---------------------------------------------------------------------------
# Full profile (called on upload, cached in DB)
# ---------------------------------------------------------------------------


def compute_full_profile(df: pd.DataFrame) -> dict:
    """Generate a comprehensive data profile including distributions, correlations,
    outliers, and actionable pattern insights.

    The result is stored in Dataset.profile and surfaced through /api/data/{id}/profile.
    """
    base = analyze_dataframe(df)

    # Enrich each column with distribution data
    for col_stat in base["columns"]:
        col = col_stat["name"]
        series = df[col].dropna()
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stat["distribution"] = _numeric_distribution(series)
            col_stat["outliers"] = _detect_outliers(series)
        else:
            col_stat["distribution"] = _categorical_distribution(series)

    # Correlation matrix (numeric columns only)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    correlations: dict[str, Any] = {}
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        # Convert to a flat list of significant pairs for easy rendering
        pairs = []
        cols = corr_matrix.columns.tolist()
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1 :]:
                val = corr_matrix.loc[c1, c2]
                if not np.isnan(val):
                    pairs.append(
                        {"col_a": c1, "col_b": c2, "correlation": round(float(val), 3)}
                    )
        pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
        correlations = {
            "pairs": pairs,
            "columns": cols,
            "matrix": _corr_matrix_dict(corr_matrix),
        }

    # Auto-generated pattern insights
    insights = _detect_patterns(df, base["columns"], correlations.get("pairs", []))

    return {
        **base,
        "correlations": correlations,
        "insights": insights,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _numeric_distribution(series: pd.Series) -> dict:
    """Histogram bins for a numeric column (up to 20 bins)."""
    if series.empty:
        return {"bins": [], "counts": []}
    # Drop inf values so np.histogram doesn't crash on unbounded range
    finite_series = series[np.isfinite(series)]
    if finite_series.empty:
        return {"bins": [], "counts": []}
    counts, bin_edges = np.histogram(
        finite_series, bins=min(20, finite_series.nunique())
    )
    bins = [round(float(e), 4) for e in bin_edges[:-1]]
    return {"bins": bins, "counts": [int(c) for c in counts]}


def _categorical_distribution(series: pd.Series) -> dict:
    """Top-20 value counts for a categorical column."""
    counts = series.value_counts(dropna=True).head(20)
    return {
        "labels": counts.index.astype(str).tolist(),
        "counts": [int(v) for v in counts.values],
    }


def _detect_outliers(series: pd.Series) -> dict:
    """IQR-based outlier detection. Returns count and threshold values."""
    if series.empty or len(series) < 4:
        return {"count": 0, "lower_fence": None, "upper_fence": None}
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count = int(((series < lower) | (series > upper)).sum())
    return {
        "count": outlier_count,
        "lower_fence": round(lower, 4),
        "upper_fence": round(upper, 4),
        "pct": round(outlier_count / len(series) * 100, 2),
    }


def _corr_matrix_dict(corr_df: pd.DataFrame) -> list[dict]:
    """Convert correlation matrix to row-oriented list for Recharts heatmap."""
    rows = []
    for col in corr_df.columns:
        row: dict = {"column": col}
        for other in corr_df.columns:
            val = corr_df.loc[col, other]
            row[other] = round(float(val), 3) if not np.isnan(val) else None
        rows.append(row)
    return rows


def _detect_patterns(
    df: pd.DataFrame, column_stats: list[dict], corr_pairs: list[dict]
) -> list[dict]:
    """Generate plain-English insights from the dataset profile.

    Each insight has: type, severity ('info'|'warning'|'critical'), title, detail.
    """
    insights: list[dict] = []

    # High missing values
    for col in column_stats:
        if col["null_pct"] >= 30:
            insights.append(
                {
                    "type": "missing_values",
                    "severity": "warning",
                    "title": f"High missing rate in '{col['name']}'",
                    "detail": (
                        f"{col['null_pct']:.1f}% of values are missing. "
                        "Consider filling with median/mode or dropping the column."
                    ),
                }
            )
        elif col["null_pct"] >= 5:
            insights.append(
                {
                    "type": "missing_values",
                    "severity": "info",
                    "title": f"Some missing values in '{col['name']}'",
                    "detail": f"{col['null_pct']:.1f}% of values are missing.",
                }
            )

    # High cardinality (likely ID columns)
    _numeric_dtypes = {
        "float64",
        "int64",
        "float32",
        "int32",
        "float16",
        "int16",
        "int8",
    }
    total_rows = len(df)
    for col in column_stats:
        if col["unique_count"] == total_rows and col["dtype"] not in _numeric_dtypes:
            insights.append(
                {
                    "type": "high_cardinality",
                    "severity": "info",
                    "title": f"'{col['name']}' looks like a unique identifier",
                    "detail": "Every value is unique — this column probably won't help prediction.",
                }
            )

    # Strong correlations
    for pair in corr_pairs[:3]:
        if abs(pair["correlation"]) >= 0.8:
            direction = "positively" if pair["correlation"] > 0 else "negatively"
            insights.append(
                {
                    "type": "correlation",
                    "severity": "info",
                    "title": f"Strong relationship: '{pair['col_a']}' and '{pair['col_b']}'",
                    "detail": (
                        f"These columns are strongly {direction} correlated "
                        f"(r={pair['correlation']}). They carry similar information."
                    ),
                }
            )

    # Outliers
    for col in column_stats:
        if "outliers" in col and col["outliers"]["count"] > 0:
            pct = col["outliers"]["pct"]
            if pct >= 5:
                insights.append(
                    {
                        "type": "outliers",
                        "severity": "warning",
                        "title": f"Outliers detected in '{col['name']}'",
                        "detail": (
                            f"{col['outliers']['count']} values ({pct:.1f}%) fall outside "
                            f"the expected range "
                            f"[{col['outliers']['lower_fence']} – {col['outliers']['upper_fence']}]."
                        ),
                    }
                )

    # Duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        insights.append(
            {
                "type": "duplicates",
                "severity": "warning",
                "title": f"{dup_count} duplicate row{'s' if dup_count > 1 else ''} found",
                "detail": "Duplicate rows can inflate model performance. Consider removing them.",
            }
        )

    # Possible date columns (string dtype with date-like values)
    # dtype may be "object" (pandas < 3) or "str" (pandas >= 3 with StringDtype)
    for col in column_stats:
        if col["dtype"] in ("object", "str", "string") and col["sample_values"]:
            sample = str(col["sample_values"][0])
            if _looks_like_date(sample):
                insights.append(
                    {
                        "type": "date_column",
                        "severity": "info",
                        "title": f"'{col['name']}' looks like a date column",
                        "detail": (
                            "Converting it to datetime could unlock time-based features "
                            "like month, day-of-week, or trend analysis."
                        ),
                    }
                )

    return insights


def detect_time_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of column names that look like date/time series.

    Heuristic: tries pd.to_datetime on the first 10 non-null values.
    Returns columns where at least 80% of those samples parse successfully.
    """
    time_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            time_cols.append(col)
            continue
        # Only check string/object columns
        if str(df[col].dtype) not in ("object", "str", "string"):
            continue
        sample = df[col].dropna().head(10)
        if sample.empty:
            continue
        successes = 0
        for val in sample:
            try:
                pd.to_datetime(str(val))
                successes += 1
            except (ValueError, TypeError):
                pass
        if successes / len(sample) >= 0.8:
            time_cols.append(col)
    return time_cols


def _looks_like_date(value: str) -> bool:
    """Quick heuristic: does the value look like a date string?"""
    import re

    date_pattern = re.compile(
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}"
    )
    return bool(date_pattern.match(value.strip()))


def compare_segments(df: pd.DataFrame, group_col: str, val1: str, val2: str) -> dict:
    """Compare two segments of a dataframe on all numeric columns.

    For each numeric column, computes mean/std/count/median for each group and
    a Cohen's-d-style effect size: (mean1 - mean2) / pooled_std.

    Returns a dict with:
    - group_col, val1, val2, count1, count2
    - columns: list of per-numeric-column stats dicts
    - notable_diffs: columns where abs(effect_size) > 0.5, sorted by magnitude
    - summary: plain-English description of the key differences
    """
    g1 = df[df[group_col].astype(str).str.strip().str.lower() == val1.strip().lower()]
    g2 = df[df[group_col].astype(str).str.strip().str.lower() == val2.strip().lower()]

    numeric_cols = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != group_col
    ]

    col_stats = []
    notable = []

    for col in numeric_cols:
        s1 = g1[col].dropna()
        s2 = g2[col].dropna()
        mean1 = _safe_scalar(s1.mean()) if len(s1) > 0 else None
        mean2 = _safe_scalar(s2.mean()) if len(s2) > 0 else None
        std1 = _safe_scalar(s1.std()) if len(s1) > 1 else None
        std2 = _safe_scalar(s2.std()) if len(s2) > 1 else None
        med1 = _safe_scalar(s1.median()) if len(s1) > 0 else None
        med2 = _safe_scalar(s2.median()) if len(s2) > 0 else None

        effect_size = None
        direction = None
        if mean1 is not None and mean2 is not None:
            pooled_std = None
            n1, n2 = len(s1), len(s2)
            if std1 is not None and std2 is not None and (n1 + n2 > 2):
                pooled_var = ((n1 - 1) * (std1**2) + (n2 - 1) * (std2**2)) / (
                    n1 + n2 - 2
                )
                pooled_std = pooled_var**0.5 if pooled_var > 0 else None
            if pooled_std and pooled_std > 0:
                effect_size = round((mean1 - mean2) / pooled_std, 3)
            elif mean2 != 0:
                effect_size = (
                    round((mean1 - mean2) / abs(mean2), 3) if mean2 != 0 else None
                )

            if effect_size is not None:
                direction = "higher_in_val1" if effect_size > 0 else "higher_in_val2"

        stat = {
            "name": col,
            "mean1": mean1,
            "std1": std1,
            "median1": med1,
            "count1": int(len(s1)),
            "mean2": mean2,
            "std2": std2,
            "median2": med2,
            "count2": int(len(s2)),
            "effect_size": effect_size,
            "direction": direction,
        }
        col_stats.append(stat)

        if effect_size is not None and abs(effect_size) > 0.5:
            notable.append(
                {"name": col, "effect_size": effect_size, "direction": direction}
            )

    notable.sort(key=lambda x: abs(x["effect_size"]), reverse=True)

    # Build plain-English summary
    summary_parts = []
    label1 = val1.title()
    label2 = val2.title()
    summary_parts.append(
        f"Comparing {label1} ({len(g1)} rows) vs {label2} ({len(g2)} rows)."
    )
    if notable:
        top = notable[:3]
        diff_descs = []
        for n in top:
            col_name = n["name"].replace("_", " ")
            if n["direction"] == "higher_in_val1":
                diff_descs.append(f"{col_name} is higher in {label1}")
            else:
                diff_descs.append(f"{col_name} is higher in {label2}")
        summary_parts.append(f"Notable differences: {'; '.join(diff_descs)}.")
    else:
        summary_parts.append("No strong differences found between the two groups.")

    return {
        "group_col": group_col,
        "val1": val1,
        "val2": val2,
        "count1": int(len(g1)),
        "count2": int(len(g2)),
        "columns": col_stats,
        "notable_diffs": notable,
        "summary": " ".join(summary_parts),
    }


def _safe_scalar(value):
    """Convert numpy scalars to native Python types for JSON serialization."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def compute_top_n(
    df: pd.DataFrame,
    sort_col: str,
    n: int = 10,
    ascending: bool = False,
    display_cols: list[str] | None = None,
) -> dict:
    """Return the top (or bottom) N rows ranked by a numeric column.

    Parameters
    ----------
    df          : source DataFrame
    sort_col    : column to rank by (must exist in df)
    n           : number of rows to return (capped at 50)
    ascending   : False = top/highest first; True = bottom/lowest first
    display_cols: columns to include in output (defaults to all, capped at 8)

    Returns a dict with:
      sort_col, ascending, n_returned, total_rows, rows (list of dicts),
      summary (plain-English), direction ("top" or "bottom")
    """
    if sort_col not in df.columns:
        return {"error": f"Column '{sort_col}' not found in dataset."}

    if not pd.api.types.is_numeric_dtype(df[sort_col]):
        return {"error": f"Column '{sort_col}' is not numeric and cannot be ranked."}

    n = max(1, min(n, 50))
    direction = "bottom" if ascending else "top"

    # Drop NaN in sort column for ranking; keep other columns as-is
    ranked = df.dropna(subset=[sort_col])
    if ascending:
        ranked = ranked.nsmallest(n, sort_col, keep="first")
    else:
        ranked = ranked.nlargest(n, sort_col, keep="first")

    # Select display columns
    if display_cols:
        valid_display = [c for c in display_cols if c in df.columns]
    else:
        valid_display = list(df.columns)

    # Cap at 8 columns; always include sort_col
    if sort_col in valid_display:
        other_cols = [c for c in valid_display if c != sort_col][:7]
        show_cols = [sort_col] + other_cols
    else:
        show_cols = [sort_col] + valid_display[:7]

    rows = []
    for rank_i, (_, row) in enumerate(ranked.iterrows(), start=1):
        row_dict: dict[str, Any] = {"_rank": rank_i}
        for col in show_cols:
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                row_dict[col] = None
            elif isinstance(val, np.generic):
                row_dict[col] = val.item()
            else:
                row_dict[col] = val
        rows.append(row_dict)

    n_returned = len(rows)
    total_rows = len(df)

    # Plain-English summary
    col_label = sort_col.replace("_", " ")
    if n_returned == 0:
        summary = f"No rows found with valid values in {col_label}."
    else:
        top_val = rows[0][sort_col]
        bottom_val = rows[-1][sort_col] if n_returned > 1 else top_val
        if isinstance(top_val, float):
            val_str = f"{top_val:,.2f}"
            bot_str = f"{bottom_val:,.2f}"
        else:
            val_str = str(top_val)
            bot_str = str(bottom_val)

        if direction == "top":
            summary = (
                f"Top {n_returned} records by {col_label} "
                f"(highest: {val_str}, lowest in this list: {bot_str}). "
                f"Showing {n_returned} of {total_rows} total rows."
            )
        else:
            summary = (
                f"Bottom {n_returned} records by {col_label} "
                f"(lowest: {val_str}, highest in this list: {bot_str}). "
                f"Showing {n_returned} of {total_rows} total rows."
            )

    return {
        "sort_col": sort_col,
        "direction": direction,
        "ascending": ascending,
        "n_requested": n,
        "n_returned": n_returned,
        "total_rows": total_rows,
        "display_cols": show_cols,
        "rows": rows,
        "summary": summary,
    }


def _correlation_strength(r: float) -> str:
    abs_r = abs(r)
    if abs_r >= 0.8:
        return "very strong"
    if abs_r >= 0.6:
        return "strong"
    if abs_r >= 0.4:
        return "moderate"
    if abs_r >= 0.2:
        return "weak"
    return "negligible"


def analyze_target_correlations(
    df: pd.DataFrame, target_col: str, top_n: int = 10
) -> dict:
    """Compute Pearson correlations between a target column and all other numeric columns.

    Returns a ranked list of correlations sorted by absolute value (strongest first).

    Args:
        df: Input DataFrame
        target_col: The column to compute correlations against
        top_n: Maximum number of columns to return

    Returns:
        Dict with:
        - target_col: name of the column analysed
        - correlations: list of {column, correlation, strength, direction} sorted by |r|
        - summary: plain-English description of the strongest relationships
        - error: set only when target_col is not numeric or not found
    """
    if target_col not in df.columns:
        return {
            "target_col": target_col,
            "correlations": [],
            "summary": f"Column '{target_col}' not found in the dataset.",
            "error": "column_not_found",
        }

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        return {
            "target_col": target_col,
            "correlations": [],
            "summary": f"Column '{target_col}' is not numeric — correlation analysis requires a numeric target.",
            "error": "not_numeric",
        }

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns if c != target_col
    ]

    if not numeric_cols:
        return {
            "target_col": target_col,
            "correlations": [],
            "summary": "No other numeric columns found to correlate against.",
            "error": "no_numeric_columns",
        }

    entries = []
    for col in numeric_cols:
        paired = df[[target_col, col]].dropna()
        if len(paired) < 3:
            continue
        r = paired[target_col].corr(paired[col])
        if pd.isna(r):
            continue
        r_val = round(float(r), 4)
        entries.append(
            {
                "column": col,
                "correlation": r_val,
                "strength": _correlation_strength(r_val),
                "direction": "positive" if r_val >= 0 else "negative",
            }
        )

    entries.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    top_entries = entries[:top_n]

    # Build plain-English summary
    if not top_entries:
        summary = f"No meaningful correlations found with {target_col}."
    else:
        best = top_entries[0]
        direction_word = (
            "positively" if best["direction"] == "positive" else "negatively"
        )
        col_name = best["column"].replace("_", " ")
        target_name = target_col.replace("_", " ")
        summary = (
            f"The strongest relationship with {target_name} is {col_name} "
            f"(r = {best['correlation']:+.2f}, {best['strength']} {direction_word} correlated)."
        )
        if len(top_entries) > 1:
            second = top_entries[1]
            second_dir = (
                "positively" if second["direction"] == "positive" else "negatively"
            )
            second_name = second["column"].replace("_", " ")
            summary += (
                f" {second_name.capitalize()} is also {second['strength']} {second_dir} "
                f"correlated (r = {second['correlation']:+.2f})."
            )
        strong = [e for e in top_entries if e["strength"] in ("strong", "very strong")]
        if strong:
            summary += (
                f" {len(strong)} column(s) show strong or very strong correlation."
            )

    return {
        "target_col": target_col,
        "correlations": top_entries,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Group-by statistics (aggregate metrics grouped by a categorical column)
# ---------------------------------------------------------------------------

_VALID_AGGS = {"sum", "mean", "count", "min", "max", "median"}
_MAX_GROUPS = 30  # cap to prevent giant tables


def compute_group_stats(
    df: pd.DataFrame,
    group_col: str,
    value_cols: list[str] | None = None,
    agg: str = "sum",
) -> dict:
    """Aggregate *value_cols* of *df* grouped by *group_col*.

    Parameters
    ----------
    df        : The source DataFrame.
    group_col : The categorical column to group by.
    value_cols: Which numeric columns to aggregate.  ``None`` → all numeric
                columns except *group_col*.
    agg       : Aggregation function — one of sum / mean / count / min / max /
                median.  Defaults to "sum".

    Returns a dict with keys:
      group_col, value_col, agg, rows (sorted descending by value),
      total, summary, error (if something went wrong).
    """
    if group_col not in df.columns:
        return {"error": f"Column '{group_col}' not found in dataset."}

    agg_fn = agg.lower()
    if agg_fn not in _VALID_AGGS:
        agg_fn = "sum"

    # Pick numeric columns to aggregate
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if value_cols:
        # Validate supplied columns
        missing = [c for c in value_cols if c not in df.columns]
        if missing:
            return {"error": f"Columns not found: {', '.join(missing)}"}
        # Only keep numeric ones
        value_cols = [c for c in value_cols if c in numeric_cols]
    else:
        value_cols = [c for c in numeric_cols if c != group_col]

    if not value_cols:
        # Fall back to row count if no numeric columns available
        agg_fn = "count"

    # Apply aggregation
    try:
        if agg_fn == "count":
            grouped = (
                df.groupby(group_col, dropna=False).size().reset_index(name="count")
            )
            value_col_name = "count"
            grouped_sorted = grouped.sort_values("count", ascending=False)
            rows = [
                {
                    "group": str(r[group_col]),
                    "count": _safe_scalar(r["count"]),
                }
                for _, r in grouped_sorted.iterrows()
            ][:_MAX_GROUPS]
            total = int(grouped["count"].sum())
        else:
            # Multi-column aggregation — produce one entry per value column
            # Use the first value column as the primary sort key
            primary = value_cols[0]
            agg_dict = {c: agg_fn for c in value_cols}
            grouped = df.groupby(group_col, dropna=False).agg(agg_dict).reset_index()
            grouped_sorted = grouped.sort_values(primary, ascending=False)
            value_col_name = primary  # used for summary/label

            rows = []
            for _, r in grouped_sorted.iterrows():
                row: dict[str, Any] = {"group": str(r[group_col])}
                for vc in value_cols:
                    row[vc] = _safe_scalar(r[vc])
                rows.append(row)
            rows = rows[:_MAX_GROUPS]

            total_series = df[primary].dropna()
            if agg_fn == "sum":
                total = _safe_scalar(total_series.sum())
            elif agg_fn == "mean":
                total = _safe_scalar(total_series.mean())
            elif agg_fn == "min":
                total = _safe_scalar(total_series.min())
            elif agg_fn == "max":
                total = _safe_scalar(total_series.max())
            elif agg_fn == "median":
                total = _safe_scalar(total_series.median())
            else:
                total = len(total_series)

    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}

    if not rows:
        return {"error": "No groups found after aggregation."}

    # Build plain-English summary
    group_label = group_col.replace("_", " ")
    value_label = value_col_name.replace("_", " ")
    top = rows[0]
    top_group = top["group"]
    top_val = top.get(value_col_name, top.get("value"))

    n_groups = len(rows)
    summary = (
        f"Grouped {value_label} by {group_label} ({agg_fn}) — "
        f"{n_groups} group{'s' if n_groups != 1 else ''}. "
        f"Highest: {top_group}"
    )
    if top_val is not None:
        try:
            summary += (
                f" ({top_val:,.2f})"
                if isinstance(top_val, float)
                else f" ({top_val:,})"
            )
        except (TypeError, ValueError):
            summary += f" ({top_val})"
    summary += "."

    # Add share-of-total if total makes sense (sum only)
    if agg_fn == "sum" and total and total != 0:
        try:
            pct = (top_val / total) * 100
            summary += f" Top group is {pct:.1f}% of the total."
        except (TypeError, ZeroDivisionError):
            pass

    return {
        "group_col": group_col,
        "value_col": value_col_name,
        "value_cols": value_cols if agg_fn != "count" else ["count"],
        "agg": agg_fn,
        "rows": rows,
        "total": total,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Column profile deep-dive (rich per-column analytics)
# ---------------------------------------------------------------------------


def compute_column_profile(df: pd.DataFrame, col_name: str) -> dict:
    """Return a rich profile for a single column: stats, distribution, issues, summary.

    Supports numeric, categorical, and date-like columns.
    Result is designed for inline chat card rendering.
    """
    if col_name not in df.columns:
        return {"error": f"Column '{col_name}' not found"}

    series = df[col_name]
    n_total = len(series)
    n_null = int(series.isna().sum())
    null_pct = round(n_null / n_total * 100, 1) if n_total > 0 else 0.0
    n_unique = int(series.nunique(dropna=True))
    non_null = series.dropna()

    issues: list[dict] = []
    stats: dict = {
        "total_rows": n_total,
        "null_count": n_null,
        "null_pct": null_pct,
        "unique_count": n_unique,
    }

    # Detect column type
    if pd.api.types.is_numeric_dtype(series):
        col_type = "numeric"
        _populate_numeric_stats(series, non_null, stats, issues, n_total, n_unique)
    elif _is_date_column(series, col_name):
        col_type = "date"
        _populate_date_stats(series, non_null, stats, issues)
    else:
        col_type = "categorical"
        _populate_categorical_stats(series, non_null, stats, issues, n_total, n_unique)

    # Common issue: high null rate
    if null_pct > 20:
        severity = "critical" if null_pct > 50 else "warning"
        issues.append(
            {
                "type": "high_null_rate",
                "severity": severity,
                "message": f"{null_pct:.0f}% of values are missing — consider filling or dropping",
            }
        )

    # Build distribution for chart rendering
    distribution = _build_column_distribution(series, col_type, non_null, n_unique)

    # Build plain-English summary
    summary = _build_column_summary(col_name, col_type, stats, issues, n_total)

    return {
        "col_name": col_name,
        "col_type": col_type,
        "stats": stats,
        "distribution": distribution,
        "issues": issues,
        "summary": summary,
    }


def _populate_numeric_stats(
    series: pd.Series,
    non_null: pd.Series,
    stats: dict,
    issues: list,
    n_total: int,
    n_unique: int,
) -> None:
    """Add numeric-specific stats and detect issues."""
    if non_null.empty:
        return
    finite = non_null[np.isfinite(non_null)]
    if finite.empty:
        return

    stats["min"] = _safe_scalar(finite.min())
    stats["max"] = _safe_scalar(finite.max())
    stats["mean"] = round(_safe_scalar(finite.mean()), 4)
    stats["median"] = round(_safe_scalar(finite.median()), 4)
    stats["std"] = round(_safe_scalar(finite.std()), 4) if len(finite) > 1 else 0.0
    stats["p25"] = round(_safe_scalar(finite.quantile(0.25)), 4)
    stats["p75"] = round(_safe_scalar(finite.quantile(0.75)), 4)

    if len(finite) > 2:
        try:
            skew_val = float(finite.skew())
            stats["skewness"] = round(skew_val, 3)
            if abs(skew_val) > 2:
                direction = "right" if skew_val > 0 else "left"
                issues.append(
                    {
                        "type": "skewed",
                        "severity": "info",
                        "message": f"Distribution is {direction}-skewed (skewness {skew_val:.2f}) — a log transform might help",
                    }
                )
        except Exception:  # noqa: BLE001
            pass

    # Constant value check
    if n_unique == 1:
        issues.append(
            {
                "type": "constant_value",
                "severity": "warning",
                "message": "All non-null values are identical — this column has no predictive power",
            }
        )

    # Potential ID column (near-unique numeric)
    if n_unique >= n_total * 0.95 and n_total > 10:
        issues.append(
            {
                "type": "potential_id",
                "severity": "info",
                "message": "Nearly every value is unique — this may be an ID column that should be excluded from modeling",
            }
        )


def _populate_categorical_stats(
    series: pd.Series,
    non_null: pd.Series,
    stats: dict,
    issues: list,
    n_total: int,
    n_unique: int,
) -> None:
    """Add categorical-specific stats and detect issues."""
    if non_null.empty:
        return

    value_counts = non_null.value_counts(dropna=True)
    most_common = str(value_counts.index[0]) if not value_counts.empty else None
    most_common_pct = (
        round(float(value_counts.iloc[0]) / len(non_null) * 100, 1)
        if not value_counts.empty
        else 0.0
    )

    stats["most_common"] = most_common
    stats["most_common_pct"] = most_common_pct
    stats["top_categories"] = [
        {"label": str(idx), "count": int(cnt)}
        for idx, cnt in value_counts.head(10).items()
    ]

    # Constant value
    if n_unique == 1:
        issues.append(
            {
                "type": "constant_value",
                "severity": "warning",
                "message": "Only one unique value — this column adds no variation",
            }
        )

    # High cardinality
    if n_unique > 50:
        issues.append(
            {
                "type": "high_cardinality",
                "severity": "warning",
                "message": f"{n_unique} unique values — too many for direct use; consider grouping or encoding",
            }
        )
    elif n_unique >= n_total * 0.8 and n_total > 10:
        issues.append(
            {
                "type": "near_unique",
                "severity": "info",
                "message": "Most values are unique — likely a free-text or ID column, not suitable as a category",
            }
        )

    # Dominant value
    if most_common_pct > 90 and n_unique > 1:
        issues.append(
            {
                "type": "dominant_value",
                "severity": "info",
                "message": f"'{most_common}' appears in {most_common_pct:.0f}% of rows — low variation",
            }
        )


def _populate_date_stats(
    series: pd.Series, non_null: pd.Series, stats: dict, issues: list
) -> None:
    """Add date-specific stats."""
    try:
        parsed = pd.to_datetime(non_null, errors="coerce").dropna()
        if parsed.empty:
            return
        stats["min_date"] = str(parsed.min().date())
        stats["max_date"] = str(parsed.max().date())
        stats["date_range_days"] = int((parsed.max() - parsed.min()).days)
        # Estimate frequency
        if len(parsed) > 2:
            sorted_dates = parsed.sort_values()
            median_gap = (sorted_dates.diff().dropna().median()).days
            if median_gap is not None:
                if median_gap <= 1:
                    stats["estimated_frequency"] = "daily"
                elif median_gap <= 8:
                    stats["estimated_frequency"] = "weekly"
                elif median_gap <= 35:
                    stats["estimated_frequency"] = "monthly"
                elif median_gap <= 100:
                    stats["estimated_frequency"] = "quarterly"
                else:
                    stats["estimated_frequency"] = "annual"
    except Exception:  # noqa: BLE001
        pass


def _is_date_column(series: pd.Series, col_name: str) -> bool:
    """Heuristic: is this a date-like column?"""
    name_lower = col_name.lower()
    if any(kw in name_lower for kw in ("date", "time", "at", "day", "month", "year")):
        # Only treat as date if it's a string/object column
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            try:
                sample = series.dropna().head(5)
                pd.to_datetime(sample, errors="raise")
                return True
            except Exception:  # noqa: BLE001
                pass
    return False


def _build_column_distribution(
    series: pd.Series, col_type: str, non_null: pd.Series, n_unique: int
) -> dict:
    """Build chart-ready distribution data."""
    if col_type == "numeric":
        finite = non_null[np.isfinite(non_null)] if not non_null.empty else non_null
        if finite.empty:
            return {"type": "histogram", "bins": [], "counts": []}
        n_bins = min(10, n_unique)
        if n_bins < 2:
            return {
                "type": "histogram",
                "bins": [_safe_scalar(finite.iloc[0])],
                "counts": [len(finite)],
            }
        counts, bin_edges = np.histogram(finite, bins=n_bins)
        return {
            "type": "histogram",
            "bins": [round(float(e), 4) for e in bin_edges[:-1]],
            "counts": [int(c) for c in counts],
        }
    elif col_type == "categorical":
        vc = non_null.value_counts(dropna=True).head(10)
        return {
            "type": "bar",
            "labels": vc.index.astype(str).tolist(),
            "counts": [int(c) for c in vc.values],
        }
    elif col_type == "date":
        return {"type": "date", "bins": [], "counts": []}
    return {"type": "unknown", "bins": [], "counts": []}


def _build_column_summary(
    col_name: str, col_type: str, stats: dict, issues: list, n_total: int
) -> str:
    """Generate a plain-English one-sentence summary for the column."""
    null_pct = stats.get("null_pct", 0)
    n_unique = stats.get("unique_count", 0)
    parts = []

    if col_type == "numeric":
        mean_val = stats.get("mean")
        min_val = stats.get("min")
        max_val = stats.get("max")
        if mean_val is not None:
            parts.append(
                f"Numeric column ranging from {min_val:g} to {max_val:g} with a mean of {mean_val:g}"
            )
    elif col_type == "categorical":
        most_common = stats.get("most_common")
        most_common_pct = stats.get("most_common_pct", 0)
        parts.append(
            f"Categorical column with {n_unique} unique value{'s' if n_unique != 1 else ''}"
        )
        if most_common:
            parts.append(f"; most common is '{most_common}' ({most_common_pct:.0f}%)")
    elif col_type == "date":
        min_d = stats.get("min_date", "")
        max_d = stats.get("max_date", "")
        freq = stats.get("estimated_frequency", "")
        if min_d and max_d:
            parts.append(f"Date column from {min_d} to {max_d}")
            if freq:
                parts.append(f" ({freq} data)")

    if null_pct > 0:
        parts.append(f"; {null_pct:.0f}% missing")
    elif n_total > 0:
        parts.append("; no missing values")

    if issues:
        critical = [i for i in issues if i["severity"] == "critical"]
        warnings = [i for i in issues if i["severity"] == "warning"]
        if critical:
            parts.append(f". ⚠️ {critical[0]['message']}")
        elif warnings:
            parts.append(f". Note: {warnings[0]['message']}")

    return (
        "".join(parts) + "." if parts else f"Column '{col_name}' with {n_total} rows."
    )


# ---------------------------------------------------------------------------
# K-means clustering
# ---------------------------------------------------------------------------

_MIN_ROWS_FOR_CLUSTERING = 10
_MAX_K = 8


def compute_clusters(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    n_clusters: int | None = None,
) -> dict:
    """Cluster numeric columns using K-means.

    Returns a dict with n_clusters, features_used, auto_k, clusters (list of
    ClusterProfile dicts), and a plain-English summary.  Each ClusterProfile
    contains: cluster_id, size, size_pct, centroid, distinguishing, description.

    Distinguishing features are those whose cluster mean deviates from the
    global mean by more than 0.5 standard deviations, sorted by magnitude.
    """
    # --- select feature columns ---
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if feature_cols:
        # keep only valid numeric cols from the requested list
        feature_cols = [c for c in feature_cols if c in numeric_cols]
    if not feature_cols:
        feature_cols = numeric_cols

    if len(feature_cols) < 1:
        return {"error": "No numeric columns available for clustering."}

    n_rows = len(df)
    if n_rows < _MIN_ROWS_FOR_CLUSTERING:
        return {
            "error": f"Need at least {_MIN_ROWS_FOR_CLUSTERING} rows for clustering (got {n_rows})."
        }

    # Drop rows with any NaN in the selected features
    data = df[feature_cols].dropna()
    if len(data) < _MIN_ROWS_FOR_CLUSTERING:
        return {
            "error": "Not enough non-null rows for clustering after dropping missing values."
        }

    # --- scale ---
    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    # --- choose k ---
    auto_k = n_clusters is None
    if auto_k:
        max_k = min(_MAX_K, len(data) - 1)
        best_k, best_score = 2, -1.0
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = float(silhouette_score(X, labels))
            if score > best_score:
                best_score, best_k = score, k
        n_clusters = best_k
    else:
        n_clusters = max(2, min(int(n_clusters), min(_MAX_K, len(data) - 1)))

    # --- fit final model ---
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # --- global stats for distinguishing features ---
    global_means = {c: float(data[c].mean()) for c in feature_cols}
    global_stds = {c: float(data[c].std()) for c in feature_cols}

    clusters = []
    total_rows = len(data)
    for cid in range(n_clusters):
        mask = labels == cid
        cluster_data = data[mask]
        size = int(mask.sum())

        centroid = {c: round(float(cluster_data[c].mean()), 4) for c in feature_cols}

        # Distinguishing: |cluster_mean - global_mean| / std > 0.5
        distinguishing = []
        for c in feature_cols:
            gstd = global_stds[c]
            if gstd < 1e-9:
                continue
            cmean = centroid[c]
            gmean = global_means[c]
            deviation = (cmean - gmean) / gstd
            if abs(deviation) >= 0.5:
                distinguishing.append(
                    {
                        "feature": c,
                        "cluster_mean": centroid[c],
                        "global_mean": round(gmean, 4),
                        "direction": "above" if deviation > 0 else "below",
                        "magnitude": round(abs(deviation), 2),
                    }
                )
        distinguishing.sort(key=lambda x: x["magnitude"], reverse=True)

        description = _build_cluster_description(cid, size, total_rows, distinguishing)

        clusters.append(
            {
                "cluster_id": cid,
                "size": size,
                "size_pct": round(size / total_rows * 100, 1),
                "centroid": centroid,
                "distinguishing": distinguishing[:5],  # top 5
                "description": description,
            }
        )

    # Sort clusters by size descending
    clusters.sort(key=lambda c: c["size"], reverse=True)
    # Re-label 0-based after sort
    for i, c in enumerate(clusters):
        c["cluster_id"] = i

    summary = _build_cluster_summary(clusters, feature_cols, n_rows, total_rows)

    return {
        "n_clusters": n_clusters,
        "features_used": feature_cols,
        "auto_k": auto_k,
        "rows_clustered": total_rows,
        "clusters": clusters,
        "summary": summary,
    }


def _build_cluster_description(
    cluster_id: int, size: int, total: int, distinguishing: list[dict]
) -> str:
    """Generate a plain-English one-sentence description for a cluster."""
    pct = round(size / total * 100)
    if not distinguishing:
        return f"Group {cluster_id + 1}: {size} records ({pct}%) with no strongly distinguishing features."

    top = distinguishing[:3]
    parts = []
    for d in top:
        feat = d["feature"].replace("_", " ")
        direction = "high" if d["direction"] == "above" else "low"
        parts.append(f"{direction} {feat}")

    feature_desc = ", ".join(parts)
    return f"Group {cluster_id + 1} ({pct}% of data): tends toward {feature_desc}."


def _build_cluster_summary(
    clusters: list[dict], feature_cols: list[str], n_rows: int, n_clustered: int
) -> str:
    """Generate a plain-English overview of all clusters."""
    k = len(clusters)
    skipped = n_rows - n_clustered
    skip_note = f" ({skipped} rows with missing values excluded)" if skipped > 0 else ""
    intro = f"Found {k} natural groups in {n_clustered} rows{skip_note} using {len(feature_cols)} feature{'s' if len(feature_cols) != 1 else ''}."
    largest = clusters[0]
    smallest = clusters[-1]
    size_note = (
        f" Largest group has {largest['size']} records ({largest['size_pct']}%),"
        f" smallest has {smallest['size']} ({smallest['size_pct']}%)."
    )
    return intro + size_note


# ---------------------------------------------------------------------------
# Time-period comparison (compare metrics across two date ranges)
# ---------------------------------------------------------------------------


def compare_time_windows(
    df: pd.DataFrame,
    date_col: str,
    period1_name: str,
    period1_start: str,
    period1_end: str,
    period2_name: str,
    period2_start: str,
    period2_end: str,
) -> dict:
    """Compare numeric column means between two date ranges.

    Parameters
    ----------
    df           : Source DataFrame (must contain *date_col*).
    date_col     : Name of the date/datetime column to filter on.
    period1_name : Display label for the first period (e.g. "2023", "Q1").
    period1_start: ISO date string for period 1 start (inclusive).
    period1_end  : ISO date string for period 1 end (inclusive).
    period2_name : Display label for the second period.
    period2_start: ISO date string for period 2 start (inclusive).
    period2_end  : ISO date string for period 2 end (inclusive).

    Returns a dict with:
      date_col, period1 {name, start, end, row_count},
      period2 {name, start, end, row_count},
      columns [{column, p1_mean, p2_mean, pct_change, direction, notable}],
      notable_changes [column names with abs(pct_change) > 20],
      summary (plain English), error (if something went wrong).
    """
    if date_col not in df.columns:
        return {"error": f"Column '{date_col}' not found in dataset."}

    # Parse dates
    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
    except Exception:  # noqa: BLE001
        return {"error": f"Could not parse '{date_col}' as dates."}

    if df.empty:
        return {"error": "No valid dates found in the date column after parsing."}

    try:
        p1s = pd.Timestamp(period1_start)
        p1e = pd.Timestamp(period1_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        p2s = pd.Timestamp(period2_start)
        p2e = pd.Timestamp(period2_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    except Exception:  # noqa: BLE001
        return {"error": "Could not parse the provided date range boundaries."}

    df1 = df[(df[date_col] >= p1s) & (df[date_col] <= p1e)]
    df2 = df[(df[date_col] >= p2s) & (df[date_col] <= p2e)]

    if df1.empty:
        return {
            "error": f"No rows found for period '{period1_name}' ({period1_start} – {period1_end})."
        }
    if df2.empty:
        return {
            "error": f"No rows found for period '{period2_name}' ({period2_start} – {period2_end})."
        }

    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns if c != date_col
    ]
    if not numeric_cols:
        return {"error": "No numeric columns available for comparison."}

    columns = []
    notable_changes: list[str] = []

    for col in numeric_cols:
        p1_mean = float(df1[col].mean()) if not df1[col].dropna().empty else None
        p2_mean = float(df2[col].mean()) if not df2[col].dropna().empty else None

        if p1_mean is None or p2_mean is None:
            continue

        # Round to 4 significant figures
        p1_mean = round(p1_mean, 4)
        p2_mean = round(p2_mean, 4)

        if abs(p1_mean) < 1e-10:
            pct_change = 0.0
        else:
            pct_change = round((p2_mean - p1_mean) / abs(p1_mean) * 100, 1)

        direction = (
            "flat" if abs(pct_change) < 1.0 else ("up" if pct_change > 0 else "down")
        )
        notable = abs(pct_change) >= 20.0

        if notable:
            notable_changes.append(col)

        columns.append(
            {
                "column": col,
                "p1_mean": p1_mean,
                "p2_mean": p2_mean,
                "pct_change": pct_change,
                "direction": direction,
                "notable": notable,
            }
        )

    if not columns:
        return {"error": "No numeric columns had data in both periods."}

    # Build plain-English summary
    summary = _build_timewindow_summary(
        period1_name, period2_name, len(df1), len(df2), columns, notable_changes
    )

    return {
        "date_col": date_col,
        "period1": {
            "name": period1_name,
            "start": period1_start,
            "end": period1_end,
            "row_count": len(df1),
        },
        "period2": {
            "name": period2_name,
            "start": period2_start,
            "end": period2_end,
            "row_count": len(df2),
        },
        "columns": columns,
        "notable_changes": notable_changes,
        "summary": summary,
    }


def _build_timewindow_summary(
    p1_name: str,
    p2_name: str,
    p1_rows: int,
    p2_rows: int,
    columns: list[dict],
    notable_changes: list[str],
) -> str:
    """Generate a plain-English summary for a time-window comparison."""
    total_cols = len(columns)
    up = [c for c in columns if c["direction"] == "up"]
    down = [c for c in columns if c["direction"] == "down"]

    summary = f"Comparing {p1_name} ({p1_rows} rows) vs {p2_name} ({p2_rows} rows) across {total_cols} metric{'s' if total_cols != 1 else ''}."

    if not notable_changes:
        summary += f" {p2_name} is broadly similar to {p1_name} — no metrics changed by more than 20%."
        return summary

    # Summarise the biggest mover
    biggest = max(columns, key=lambda c: abs(c["pct_change"]))
    col_label = biggest["column"].replace("_", " ")
    direction_word = "increased" if biggest["direction"] == "up" else "decreased"
    summary += (
        f" Biggest change: {col_label} {direction_word} by {abs(biggest['pct_change']):.0f}%"
        f" ({biggest['p1_mean']:,.2f} → {biggest['p2_mean']:,.2f})."
    )

    if len(up) > 0 and len(down) > 0:
        summary += f" Overall, {len(up)} metric{'s' if len(up) != 1 else ''} went up and {len(down)} went down."
    elif len(up) > 0:
        summary += f" All tracked metrics improved in {p2_name}."
    elif len(down) > 0:
        summary += f" All tracked metrics declined in {p2_name}."

    return summary


# ---------------------------------------------------------------------------
# Record table viewer (show me the data / peek at rows)
# ---------------------------------------------------------------------------


def sample_records(
    df: pd.DataFrame,
    n: int = 20,
    conditions: list[dict] | None = None,
    offset: int = 0,
) -> dict:
    """Return a sample of rows from the DataFrame, with optional filtering.

    Parameters
    ----------
    df         : source DataFrame
    n          : rows to return (capped at 50)
    conditions : list of FilterCondition dicts (column/operator/value)
                 applied via boolean AND logic
    offset     : starting row index (for paging, default 0)

    Returns a dict with:
      columns, rows (list of serialisable dicts), total_rows, shown_rows,
      filtered (bool), condition_summary (plain-English), summary
    """
    from core.filter_view import apply_active_filter  # avoid circular

    n = max(1, min(n, 50))
    offset = max(0, offset)
    total_rows = len(df)
    filtered = bool(conditions)

    working = df
    if conditions:
        working = apply_active_filter(df, conditions)

    filtered_rows = len(working)
    page = working.iloc[offset : offset + n]
    shown = len(page)

    # Cap display columns at 8
    display_cols = list(df.columns[:8])

    rows = []
    for _, row in page.iterrows():
        row_dict: dict = {}
        for col in display_cols:
            val = row.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                row_dict[col] = None
            elif isinstance(val, np.generic):
                row_dict[col] = val.item()
            else:
                row_dict[col] = val
        rows.append(row_dict)

    condition_summary = ""
    if conditions:
        parts = []
        for c in conditions:
            op_labels = {
                "eq": "=",
                "ne": "≠",
                "gt": ">",
                "lt": "<",
                "gte": "≥",
                "lte": "≤",
                "contains": "contains",
                "not_contains": "does not contain",
            }
            op = op_labels.get(c.get("operator", "eq"), c.get("operator", "="))
            parts.append(f"{c['column']} {op} {c['value']}")
        condition_summary = " AND ".join(parts)

    if filtered:
        if filtered_rows == 0:
            summary = f"No rows match: {condition_summary}."
        else:
            pct = round(filtered_rows / total_rows * 100) if total_rows else 0
            summary = (
                f"Found {filtered_rows:,} matching rows ({pct}% of {total_rows:,} total). "
                f"Showing {shown}."
            )
    else:
        summary = (
            f"Showing {shown} of {total_rows:,} rows"
            f"{' (starting from row ' + str(offset + 1) + ')' if offset > 0 else ''}."
        )

    return {
        "columns": display_cols,
        "rows": rows,
        "total_rows": total_rows,
        "filtered_rows": filtered_rows if filtered else total_rows,
        "shown_rows": shown,
        "filtered": filtered,
        "condition_summary": condition_summary,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Summary statistics (describe() equivalent for all columns)
# ---------------------------------------------------------------------------


def compute_summary_stats(df: pd.DataFrame) -> dict:
    """Return describe()-style statistics for all columns in the DataFrame.

    Numeric columns: count, mean, std, min, Q25, median, Q75, max, null_count.
    Categorical columns: count, unique, top (most common value), freq, null_count.

    Returns a dict with total_rows, total_cols, numeric_stats, categorical_stats,
    and a plain-English summary.
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    def _safe_round(val: Any, ndigits: int = 4) -> float | None:
        try:
            f = float(val)
            if np.isnan(f) or np.isinf(f):
                return None
            return round(f, ndigits)
        except (TypeError, ValueError):
            return None

    numeric_stats = []
    for col in numeric_cols:
        s = df[col].dropna()
        n = len(s)
        numeric_stats.append(
            {
                "column": col,
                "count": n,
                "mean": _safe_round(s.mean()) if n > 0 else None,
                "std": _safe_round(s.std()) if n > 1 else None,
                "min": _safe_round(s.min()) if n > 0 else None,
                "q25": _safe_round(s.quantile(0.25)) if n > 0 else None,
                "median": _safe_round(s.median()) if n > 0 else None,
                "q75": _safe_round(s.quantile(0.75)) if n > 0 else None,
                "max": _safe_round(s.max()) if n > 0 else None,
                "null_count": int(df[col].isna().sum()),
            }
        )

    categorical_stats = []
    for col in categorical_cols:
        s = df[col].dropna().astype(str)
        n = len(s)
        top_val: str | None = None
        top_freq = 0
        if n > 0:
            vc = s.value_counts()
            top_val = str(vc.index[0])
            top_freq = int(vc.iloc[0])
        categorical_stats.append(
            {
                "column": col,
                "count": n,
                "unique": int(df[col].nunique()),
                "top": top_val,
                "freq": top_freq,
                "null_count": int(df[col].isna().sum()),
            }
        )

    total_rows = len(df)
    total_cols = len(df.columns)
    summary = (
        f"{total_rows:,} rows × {total_cols} columns "
        f"({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)."
    )

    return {
        "total_rows": total_rows,
        "total_cols": total_cols,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Category value counts (frequency table for a single categorical column)
# ---------------------------------------------------------------------------


def compute_value_counts(df: pd.DataFrame, col: str, n: int = 20) -> dict:
    """Return the top-N value frequencies for a single column.

    Returns a dict with column, total_rows, unique_count, rows (list of
    {value, count, pct}), has_more (bool), and a plain-English summary.
    """
    n = max(1, min(n, 50))
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset.")

    series = df[col].dropna().astype(str)
    total_rows = len(df)
    non_null = len(series)
    null_count = total_rows - non_null
    unique_count = int(series.nunique())

    vc = series.value_counts().head(n)
    rows = [
        {
            "value": str(val),
            "count": int(cnt),
            "pct": round(int(cnt) / non_null * 100, 1) if non_null > 0 else 0.0,
        }
        for val, cnt in vc.items()
    ]

    has_more = unique_count > n
    top_val = rows[0]["value"] if rows else None
    top_pct = rows[0]["pct"] if rows else 0.0

    summary_parts = [
        f"'{col}' has {unique_count} unique value{'s' if unique_count != 1 else ''}."
    ]
    if top_val:
        summary_parts.append(f"Most common: '{top_val}' ({top_pct}% of non-null rows).")
    if null_count > 0:
        summary_parts.append(
            f"{null_count} null value{'s' if null_count != 1 else ''}."
        )
    if has_more:
        summary_parts.append(f"Showing top {n} of {unique_count}.")

    return {
        "column": col,
        "total_rows": total_rows,
        "non_null": non_null,
        "null_count": null_count,
        "unique_count": unique_count,
        "rows": rows,
        "has_more": has_more,
        "summary": " ".join(summary_parts),
    }


def compute_pair_correlation(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """Compute Pearson correlation between two numeric columns.

    Returns r, p_value, n, strength label, direction, significance, and summary.
    """
    from scipy import stats

    if col1 not in df.columns:
        raise ValueError(f"Column '{col1}' not found in dataset.")
    if col2 not in df.columns:
        raise ValueError(f"Column '{col2}' not found in dataset.")

    valid_mask = df[[col1, col2]].notna().all(axis=1)
    aligned = df.loc[valid_mask, [col1, col2]]
    n = len(aligned)

    if n < 3:
        return {
            "col1": col1,
            "col2": col2,
            "r": None,
            "p_value": None,
            "n": n,
            "strength": "insufficient data",
            "direction": "unknown",
            "significant": "insufficient data for correlation",
            "summary": f"Need at least 3 paired observations (only {n} found).",
        }

    try:
        r_val, p_val = stats.pearsonr(aligned[col1].values, aligned[col2].values)
        r_val = float(r_val)
        p_val = float(p_val)
    except Exception:  # noqa: BLE001
        return {
            "col1": col1,
            "col2": col2,
            "r": None,
            "p_value": None,
            "n": n,
            "strength": "error",
            "direction": "unknown",
            "significant": "could not compute",
            "summary": f"Could not compute correlation between '{col1}' and '{col2}'.",
        }

    abs_r = abs(r_val)
    if abs_r >= 0.9:
        strength = "very strong"
    elif abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.5:
        strength = "moderate"
    elif abs_r >= 0.3:
        strength = "weak"
    else:
        strength = "negligible"

    direction = "positive" if r_val >= 0 else "negative"

    if p_val < 0.001:
        sig = "highly significant (p < 0.001)"
    elif p_val < 0.01:
        sig = "significant (p < 0.01)"
    elif p_val < 0.05:
        sig = "significant (p < 0.05)"
    else:
        sig = "not statistically significant (p ≥ 0.05)"

    if abs_r >= 0.7:
        interp = f"When '{col1}' increases, '{col2}' tends to {'increase' if r_val > 0 else 'decrease'} strongly."
    elif abs_r >= 0.5:
        interp = f"There is a moderate {'positive' if r_val > 0 else 'negative'} relationship between these columns."
    elif abs_r >= 0.3:
        interp = f"There is a weak {'positive' if r_val > 0 else 'negative'} relationship — other factors are likely involved."
    else:
        interp = "These two columns show little to no linear relationship."

    summary = (
        f"'{col1}' and '{col2}' have a {strength} {direction} correlation "
        f"(r = {r_val:.3f}, n = {n}). {interp} The relationship is {sig}."
    )

    return {
        "col1": col1,
        "col2": col2,
        "r": round(r_val, 4),
        "p_value": round(p_val, 6),
        "n": n,
        "strength": strength,
        "direction": direction,
        "significant": sig,
        "interpretation": interp,
        "summary": summary,
    }


def compute_group_trends(
    df: pd.DataFrame,
    date_col: str,
    group_col: str,
    value_col: str,
) -> dict:
    """Compute per-group trends over time using OLS slope.

    For each unique value in *group_col*, fits a linear regression of
    *value_col* over time (converted to a numeric index) and returns
    slope, total % change (first→last non-null period), direction, and rank.

    Parameters
    ----------
    df        : Source DataFrame.
    date_col  : Name of the date/time column.
    group_col : Categorical column to group by (≤50 unique values).
    value_col : Numeric column whose trend to measure.

    Returns a dict with keys:
      date_col, group_col, value_col, groups (list, ranked fastest→slowest),
      rising, falling, flat, summary, error.
    """
    if date_col not in df.columns:
        return {"error": f"Date column '{date_col}' not found."}
    if group_col not in df.columns:
        return {"error": f"Group column '{group_col}' not found."}
    if value_col not in df.columns:
        return {"error": f"Value column '{value_col}' not found."}

    n_unique = df[group_col].nunique()
    if n_unique > 50:
        return {
            "error": (
                f"'{group_col}' has {n_unique} unique values — too many to compute "
                "group trends. Choose a column with 50 or fewer categories."
            )
        }

    # Parse dates
    dates = pd.to_datetime(df[date_col], errors="coerce")
    values = pd.to_numeric(df[value_col], errors="coerce")
    groups_series = df[group_col]

    # Drop rows with missing date or value
    mask = dates.notna() & values.notna() & groups_series.notna()
    dates = dates[mask]
    values = values[mask]
    groups_series = groups_series[mask]

    if len(dates) == 0:
        return {"error": "No valid rows after dropping missing date/value entries."}

    # Convert dates to numeric index (days since min date)
    min_date = dates.min()
    date_index = (dates - min_date).dt.days.astype(float)

    group_results = []
    for grp_val in groups_series.unique():
        sel = groups_series == grp_val
        x = date_index[sel].values
        y = values[sel].values

        if len(x) < 2:
            continue

        # Sort by date
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        # OLS slope: b = cov(x,y) / var(x)
        var_x = float(np.var(x))
        if var_x == 0:
            slope = 0.0
        else:
            slope = float(np.cov(x, y, ddof=0)[0, 1] / var_x)

        # % change first → last
        first_val = float(y[0])
        last_val = float(y[-1])
        if first_val != 0:
            pct_change = (last_val - first_val) / abs(first_val) * 100
        else:
            pct_change = 0.0

        if slope > 0.001:
            direction = "up"
        elif slope < -0.001:
            direction = "down"
        else:
            direction = "flat"

        group_results.append(
            {
                "group": str(grp_val),
                "slope": round(slope, 4),
                "pct_change": round(pct_change, 1),
                "direction": direction,
                "first_value": round(first_val, 2),
                "last_value": round(last_val, 2),
                "n_periods": int(len(x)),
            }
        )

    if not group_results:
        return {"error": "No groups had enough data points to compute a trend."}

    # Rank by slope descending (fastest growers first)
    group_results.sort(key=lambda r: r["slope"], reverse=True)
    for i, r in enumerate(group_results):
        r["rank"] = i + 1

    rising = [r for r in group_results if r["direction"] == "up"]
    falling = [r for r in group_results if r["direction"] == "down"]
    flat = [r for r in group_results if r["direction"] == "flat"]

    # Build plain-English summary
    if rising:
        top = rising[0]["group"]
        summary = (
            f"'{top}' is growing fastest in '{value_col}' "
            f"(+{rising[0]['pct_change']:.1f}% over the period)."
        )
        if len(rising) > 1:
            summary += f" {len(rising)} group(s) are trending up"
        if falling:
            bottom = falling[-1]["group"]
            summary += (
                f", while '{bottom}' is declining most "
                f"({falling[-1]['pct_change']:.1f}%)."
            )
        else:
            summary += "."
    elif falling:
        bottom = falling[-1]["group"]
        summary = (
            f"All groups are declining. '{bottom}' is falling fastest "
            f"({falling[-1]['pct_change']:.1f}% over the period)."
        )
    else:
        summary = f"All groups show flat trends in '{value_col}'."

    return {
        "date_col": date_col,
        "group_col": group_col,
        "value_col": value_col,
        "groups": group_results,
        "rising": len(rising),
        "falling": len(falling),
        "flat": len(flat),
        "summary": summary,
    }


def compute_stat_query(
    df: pd.DataFrame,
    agg: str,
    col: str | None = None,
) -> dict:
    """Compute a single aggregate statistic for a column.

    agg: one of count, sum, mean, median, max, min, std
    col: column name (required unless agg == 'count')

    Returns: agg, col, value, n_rows, formatted_value, summary.
    """
    agg = agg.lower().strip()
    valid_aggs = ("count", "sum", "mean", "median", "max", "min", "std")
    if agg not in valid_aggs:
        raise ValueError(
            f"Unknown aggregation '{agg}'. Choose from: {', '.join(valid_aggs)}."
        )

    n_rows = len(df)

    if agg == "count":
        if col and col in df.columns:
            value = int(df[col].notna().sum())
            formatted = f"{value:,}"
            summary = f"There are {value:,} non-null values in '{col}' (out of {n_rows:,} total rows)."
            return {
                "agg": "count",
                "col": col,
                "value": value,
                "n_rows": n_rows,
                "formatted_value": formatted,
                "summary": summary,
            }
        else:
            value = n_rows
            formatted = f"{value:,}"
            summary = f"The dataset has {value:,} rows."
            return {
                "agg": "count",
                "col": None,
                "value": value,
                "n_rows": n_rows,
                "formatted_value": formatted,
                "summary": summary,
            }

    if not col:
        raise ValueError("Column name is required for aggregations other than count.")
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in dataset.")

    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(series) == 0:
        raise ValueError(f"Column '{col}' has no numeric values.")

    if agg == "sum":
        value = float(series.sum())
    elif agg == "mean":
        value = float(series.mean())
    elif agg == "median":
        value = float(series.median())
    elif agg == "max":
        value = float(series.max())
    elif agg == "min":
        value = float(series.min())
    elif agg == "std":
        value = float(series.std())
    else:
        raise ValueError(f"Unknown aggregation '{agg}'.")

    # Format value nicely
    if abs(value) >= 1_000_000:
        formatted = f"{value / 1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        formatted = f"{value / 1_000:.2f}k"
    else:
        formatted = f"{value:,.2f}" if value != int(value) else f"{int(value):,}"

    agg_labels = {
        "sum": "total",
        "mean": "average",
        "median": "median",
        "max": "maximum",
        "min": "minimum",
        "std": "standard deviation",
    }
    label = agg_labels.get(agg, agg)
    summary = (
        f"The {label} of '{col}' is {formatted} "
        f"(based on {len(series):,} non-null values out of {n_rows:,} rows)."
    )

    return {
        "agg": agg,
        "col": col,
        "value": value,
        "n_rows": n_rows,
        "n_valid": len(series),
        "formatted_value": formatted,
        "label": label,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Project health summary (proactive model drift alerts)
# ---------------------------------------------------------------------------

_ALGO_SHORT: dict[str, str] = {
    "linear_regression": "Linear Regression",
    "ridge": "Ridge",
    "logistic_regression": "Logistic Regression",
    "random_forest_regressor": "Random Forest",
    "random_forest_classifier": "Random Forest",
    "gradient_boosting_regressor": "Gradient Boosting",
    "gradient_boosting_classifier": "Gradient Boosting",
    "decision_tree_regressor": "Decision Tree",
    "decision_tree_classifier": "Decision Tree",
    "xgboost_regressor": "XGBoost",
    "xgboost_classifier": "XGBoost",
    "lightgbm_regressor": "LightGBM",
    "lightgbm_classifier": "LightGBM",
    "mlp_regressor": "Neural Network",
    "mlp_classifier": "Neural Network",
}


def _deployment_age_score(created_at: datetime, now: datetime) -> int:
    """Return 0-100 age score — higher means fresher model."""
    try:
        age_days = (now - created_at.replace(tzinfo=None)).days
    except Exception:  # noqa: BLE001
        return 100  # unknown age = assume fresh
    if age_days < 30:
        return 100
    if age_days < 60:
        return 80
    if age_days < 90:
        return 60
    if age_days < 180:
        return 40
    return 20


def _deployment_usage_score(
    request_count: int, last_predicted_at: datetime | None, now: datetime
) -> int:
    """Return 0-100 usage score — higher means actively used."""
    if request_count == 0:
        return 30  # never used — note but don't flag harshly
    if last_predicted_at is None:
        return 70
    try:
        idle_days = (now - last_predicted_at.replace(tzinfo=None)).days
    except Exception:  # noqa: BLE001
        return 70
    if idle_days < 7:
        return 100
    if idle_days < 30:
        return 80
    if idle_days < 90:
        return 60
    return 40


def compute_deployment_health_item(
    deployment_id: str,
    algorithm: str | None,
    target_column: str | None,
    created_at: datetime,
    request_count: int,
    last_predicted_at: datetime | None,
    environment: str,
    now: datetime | None = None,
) -> dict:
    """Return a health item dict for one deployment.

    Pure function — no database or filesystem access.

    Args:
        deployment_id: UUID of the deployment.
        algorithm: sklearn algorithm key (e.g. "random_forest_regressor").
        target_column: Column name being predicted.
        created_at: When the deployment was created.
        request_count: Total number of predictions served.
        last_predicted_at: Timestamp of the most recent prediction (or None).
        environment: "staging" or "production".
        now: Reference timestamp (defaults to UTC now).

    Returns:
        Dict with keys: deployment_id, name, algorithm_plain, target_column,
        environment, health_score, status, top_issue, recommendation,
        age_score, usage_score.
    """
    if now is None:
        now = datetime.now(UTC).replace(tzinfo=None)

    age_score = _deployment_age_score(created_at, now)
    usage_score = _deployment_usage_score(request_count, last_predicted_at, now)
    health_score = int(age_score * 0.55 + usage_score * 0.45)

    # Determine status
    if health_score >= 75:
        status = "healthy"
    elif health_score >= 50:
        status = "warning"
    else:
        status = "critical"

    # Determine top issue and recommendation
    algo_plain = _ALGO_SHORT.get(algorithm or "", algorithm or "Model")
    target_label = target_column or "target"
    name = f"{algo_plain} → {target_label}"

    try:
        age_days = (now - created_at.replace(tzinfo=None)).days
    except Exception:  # noqa: BLE001
        age_days = 0

    top_issue: str | None = None
    recommendation: str | None = None

    if age_days >= 90:
        top_issue = (
            f"Model is {age_days} days old — patterns in your data may have changed."
        )
        recommendation = (
            "Retrain with your most recent data to keep predictions accurate."
        )
    elif age_days >= 30 and request_count == 0:
        top_issue = "Model has not received any predictions yet."
        recommendation = (
            "Share the prediction dashboard link or API endpoint with your team."
        )
    elif last_predicted_at is not None:
        try:
            idle_days = (now - last_predicted_at.replace(tzinfo=None)).days
        except Exception:  # noqa: BLE001
            idle_days = 0
        if idle_days >= 30:
            top_issue = f"No predictions in the last {idle_days} days."
            recommendation = (
                "Check if the prediction URL is still being used by your team."
            )

    return {
        "deployment_id": deployment_id,
        "name": name,
        "algorithm_plain": algo_plain,
        "target_column": target_label,
        "environment": environment,
        "health_score": health_score,
        "status": status,
        "top_issue": top_issue,
        "recommendation": recommendation,
        "age_score": age_score,
        "usage_score": usage_score,
    }


def compute_project_health_summary(
    deployment_dicts: list[dict],
    now: datetime | None = None,
) -> dict:
    """Aggregate health items for all active deployments in a project.

    Args:
        deployment_dicts: List of dicts, each with the same keys as
            compute_deployment_health_item's parameters.
        now: Reference timestamp for age/usage calculations.

    Returns:
        Dict with keys: total, healthy, warning, critical, alerts (only
        warning/critical items), overall_status, summary.
    """
    if now is None:
        now = datetime.now(UTC).replace(tzinfo=None)

    items = [
        compute_deployment_health_item(
            deployment_id=d["deployment_id"],
            algorithm=d.get("algorithm"),
            target_column=d.get("target_column"),
            created_at=d["created_at"],
            request_count=d.get("request_count", 0),
            last_predicted_at=d.get("last_predicted_at"),
            environment=d.get("environment", "staging"),
            now=now,
        )
        for d in deployment_dicts
    ]

    healthy = [i for i in items if i["status"] == "healthy"]
    warning = [i for i in items if i["status"] == "warning"]
    critical = [i for i in items if i["status"] == "critical"]

    # Overall project status: worst single deployment wins
    if critical:
        overall_status = "critical"
    elif warning:
        overall_status = "warning"
    else:
        overall_status = "healthy"

    # Build plain-English project summary
    total = len(items)
    if total == 0:
        summary = "No active deployments found for this project."
    elif overall_status == "healthy":
        summary = (
            f"All {total} deployed model{'s' if total > 1 else ''} "
            f"{'are' if total > 1 else 'is'} healthy."
        )
    else:
        n_issues = len(warning) + len(critical)
        summary = (
            f"{n_issues} of {total} deployed model{'s' if total > 1 else ''} "
            f"{'need' if n_issues > 1 else 'needs'} attention."
        )

    return {
        "total": total,
        "healthy": len(healthy),
        "warning": len(warning),
        "critical": len(critical),
        "alerts": warning + critical,  # non-healthy items only
        "all_items": items,
        "overall_status": overall_status,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Prediction opportunity discovery
# ---------------------------------------------------------------------------

# Column name patterns that suggest high business value as a prediction target
_HIGH_VALUE_NAMES = re.compile(
    r"(?i)\b(revenue|sales|profit|churn|conversion|return|target|outcome|label|"
    r"y|loss|gain|win|default|fraud|cancellation|renewal|subscribe|purchase|buy)\b"
)
_MEDIUM_VALUE_NAMES = re.compile(
    r"(?i)\b(price|cost|quantity|volume|count|rate|score|demand|margin|spend|"
    r"clicks|visits|duration|tenure|amount|value|total|gross|net|units)\b"
)

# Column name patterns that indicate poor prediction targets (IDs, timestamps)
# Matches: standalone "id", columns ending with "_id"/"_key"/etc., or starting with "id_"/"pk_"
_ID_LIKE_NAMES = re.compile(
    r"(?i)(\bid\b|_id$|^id_|_uuid|_guid|_key$|^pk$|^pk_|_hash$|_token$|_index$|_ref$)"
)


def _business_value(col_name: str) -> str:
    """Classify business value of predicting a column as 'high', 'medium', or 'low'."""
    if _HIGH_VALUE_NAMES.search(col_name):
        return "high"
    if _MEDIUM_VALUE_NAMES.search(col_name):
        return "medium"
    return "low"


def _example_question(col_name: str, problem_type: str, business_value: str) -> str:
    """Generate a plain-English prediction question for this target."""
    col_display = col_name.replace("_", " ").title()
    if problem_type == "regression":
        if business_value == "high":
            return (
                f"Can you predict the {col_display} for each record in my next dataset?"
            )
        return f"What will the {col_display} be for new records?"
    else:  # classification
        if business_value == "high":
            return f"Which records are most likely to have a specific {col_display} outcome?"
        return f"Can you classify each record by {col_display}?"


def compute_prediction_opportunities(
    col_stats: list[dict],
    row_count: int,
) -> list[dict]:
    """Analyze dataset columns and return ranked prediction opportunities.

    Each opportunity represents a column that could serve as a good prediction
    target, with a feasibility score, business value rating, and an example
    question the analyst could answer with this model.

    Args:
        col_stats: List of column stat dicts from analyze_dataframe().
            Each dict must have: name, dtype, null_pct, unique_count,
            and optionally min/max/mean/std for numeric columns.
        row_count: Total number of rows in the dataset.

    Returns:
        List of opportunity dicts, ranked by feasibility_score descending.
        Each dict has: target_col, problem_type, feasibility_score,
        reason, business_value, example_question, predictor_count.
    """
    if not col_stats or row_count < 10:
        return []

    opportunities = []

    for stat in col_stats:
        col = stat["name"]
        dtype = stat.get("dtype", "object")
        null_pct = stat.get("null_pct", 0.0)
        unique_count = stat.get("unique_count", 0)

        is_numeric = "int" in dtype.lower() or "float" in dtype.lower()

        # Skip ID-like columns: name pattern or high uniqueness for categoricals
        # (numeric columns naturally have many unique values — don't filter them)
        if not is_numeric and row_count > 0 and unique_count / row_count > 0.8:
            continue
        if _ID_LIKE_NAMES.search(col):
            continue

        # Skip columns with too much missing data
        if null_pct > 30:
            continue

        # Determine problem type
        n_unique = unique_count

        if is_numeric:
            mean_val = stat.get("mean", 0) or 0
            std_val = stat.get("std", 0) or 0
            # Skip constant columns (no variation)
            if mean_val != 0 and std_val / abs(mean_val) < 0.001:
                continue
            if std_val == 0:
                continue
            problem_type = "regression"
        elif n_unique <= 20 and n_unique >= 2:
            problem_type = "classification"
        else:
            # Too many categories → bad target
            continue

        # Count predictor columns (other non-target columns)
        predictor_count = sum(
            1 for s in col_stats if s["name"] != col and s.get("null_pct", 0) <= 50
        )

        # Feasibility score (0-100)
        score = 55  # base

        # Reward: low missing data
        if null_pct < 5:
            score += 20
        elif null_pct < 15:
            score += 10

        # Reward: enough predictors
        if predictor_count >= 5:
            score += 15
        elif predictor_count >= 3:
            score += 8

        # Reward: named like a good target
        bv = _business_value(col)
        if bv == "high":
            score += 10
        elif bv == "medium":
            score += 5

        # Penalize: near-unique categorical (poor grouping)
        if not is_numeric and row_count > 0 and n_unique / row_count > 0.4:
            score -= 20

        score = max(0, min(100, score))

        # Build plain-English reason
        if problem_type == "regression":
            reason = (
                f"'{col}' is a numeric column with {100 - null_pct:.0f}% "
                f"complete data and real variation — a natural regression target."
            )
        else:
            reason = (
                f"'{col}' has {n_unique} distinct categories and "
                f"{100 - null_pct:.0f}% complete data — a good classification target."
            )

        opportunities.append(
            {
                "target_col": col,
                "problem_type": problem_type,
                "feasibility_score": score,
                "reason": reason,
                "business_value": bv,
                "example_question": _example_question(col, problem_type, bv),
                "predictor_count": predictor_count,
            }
        )

    # Sort by feasibility descending, then business value as tiebreaker
    _bv_rank = {"high": 2, "medium": 1, "low": 0}
    opportunities.sort(
        key=lambda o: (o["feasibility_score"], _bv_rank[o["business_value"]]),
        reverse=True,
    )

    return opportunities[:5]


# ---------------------------------------------------------------------------
# Dataset distribution comparison (detect meaningful changes between uploads)
# ---------------------------------------------------------------------------

_SHIFT_HIGH = 0.30  # >30% mean shift → high severity
_SHIFT_MEDIUM = 0.10  # 10–30% mean shift → medium severity


def _col_drift_severity(pct_change: float) -> str:
    """Classify percentage change into low / medium / high severity."""
    abs_change = abs(pct_change)
    if abs_change >= _SHIFT_HIGH:
        return "high"
    if abs_change >= _SHIFT_MEDIUM:
        return "medium"
    return "low"


def compute_dataset_comparison(
    old_df: "pd.DataFrame",
    new_df: "pd.DataFrame",
) -> dict:
    """Compare two DataFrames and return a structured distribution drift report.

    Pure function — no database access, no side effects.

    Args:
        old_df: The baseline (training / previously uploaded) DataFrame.
        new_df: The new (recently uploaded) DataFrame to compare against.

    Returns:
        {
            row_count_old, row_count_new, row_count_change_pct,
            col_count_old, col_count_new,
            new_columns: [str, ...],
            dropped_columns: [str, ...],
            numeric_drifts: [{col, old_mean, new_mean, pct_change, severity}, ...],
            categorical_drifts: [{col, new_categories, dropped_categories,
                                   top_shift_pct, severity}, ...],
            drift_score: 0-100,
            summary: str,
        }
    """
    old_cols = set(old_df.columns)
    new_cols = set(new_df.columns)
    new_columns = sorted(new_cols - old_cols)
    dropped_columns = sorted(old_cols - new_cols)
    common_cols = old_cols & new_cols

    # ---- Row count change ----
    n_old = len(old_df)
    n_new = len(new_df)
    if n_old > 0:
        row_count_change_pct = round((n_new - n_old) / n_old * 100, 1)
    else:
        row_count_change_pct = 0.0

    # ---- Per-column comparisons ----
    numeric_drifts = []
    categorical_drifts = []

    for col in sorted(common_cols):
        old_ser = old_df[col].dropna()
        new_ser = new_df[col].dropna()

        if len(old_ser) == 0 or len(new_ser) == 0:
            continue

        if pd.api.types.is_numeric_dtype(old_ser):
            old_mean = float(old_ser.mean())
            new_mean = float(new_ser.mean())
            old_std = float(old_ser.std())
            new_std = float(new_ser.std())

            if abs(old_mean) > 1e-9:
                pct_change = (new_mean - old_mean) / abs(old_mean)
            else:
                pct_change = 0.0

            severity = _col_drift_severity(pct_change)

            # Only report columns that actually shifted meaningfully
            if severity != "low" or (
                new_std > 0 and abs(new_std - old_std) / (old_std + 1e-9) > 0.20
            ):
                numeric_drifts.append(
                    {
                        "col": col,
                        "old_mean": round(old_mean, 4),
                        "new_mean": round(new_mean, 4),
                        "old_std": round(old_std, 4),
                        "new_std": round(new_std, 4),
                        "pct_change": round(pct_change * 100, 1),
                        "severity": severity,
                    }
                )
        else:
            # Categorical column
            old_cats = set(old_ser.astype(str).unique())
            new_cats = set(new_ser.astype(str).unique())
            added_cats = sorted(new_cats - old_cats)[:10]
            removed_cats = sorted(old_cats - new_cats)[:10]

            # Frequency shift: compare top category share
            old_freq = old_ser.astype(str).value_counts(normalize=True)
            new_freq = new_ser.astype(str).value_counts(normalize=True)
            common_cats = set(old_freq.index) & set(new_freq.index)

            if common_cats:
                max_shift = max(
                    abs(new_freq.get(c, 0) - old_freq.get(c, 0)) for c in common_cats
                )
                top_shift_pct = round(float(max_shift) * 100, 1)
            else:
                top_shift_pct = 0.0

            if added_cats or removed_cats or top_shift_pct >= 10.0:
                severity = (
                    "high"
                    if (
                        len(added_cats) > 2
                        or len(removed_cats) > 2
                        or top_shift_pct >= 20
                    )
                    else "medium"
                )
                categorical_drifts.append(
                    {
                        "col": col,
                        "new_categories": added_cats,
                        "dropped_categories": removed_cats,
                        "top_shift_pct": top_shift_pct,
                        "severity": severity,
                    }
                )

    # ---- Overall drift score (0-100) ----
    # Higher score = more drift = more caution warranted
    drift_components: list[float] = []

    # Row count component (up to 15 points)
    drift_components.append(min(15.0, abs(row_count_change_pct) / 2))

    # Schema change component (up to 15 points)
    schema_changes = len(new_columns) + len(dropped_columns)
    drift_components.append(min(15.0, schema_changes * 5.0))

    # Numeric drift component (up to 40 points)
    high_numeric = sum(1 for d in numeric_drifts if d["severity"] == "high")
    med_numeric = sum(1 for d in numeric_drifts if d["severity"] == "medium")
    drift_components.append(min(40.0, high_numeric * 15.0 + med_numeric * 6.0))

    # Categorical drift component (up to 30 points)
    high_cat = sum(1 for d in categorical_drifts if d["severity"] == "high")
    med_cat = sum(1 for d in categorical_drifts if d["severity"] == "medium")
    drift_components.append(min(30.0, high_cat * 15.0 + med_cat * 6.0))

    drift_score = int(min(100, sum(drift_components)))

    # ---- Plain-English summary ----
    total_issues = (
        len(numeric_drifts)
        + len(categorical_drifts)
        + len(new_columns)
        + len(dropped_columns)
    )

    if drift_score == 0 and total_issues == 0:
        summary = "The new dataset looks very similar to the original — distributions match closely."
    elif drift_score < 15:
        summary = (
            f"Minor differences detected ({total_issues} column{'s' if total_issues != 1 else ''}). "
            "The datasets are broadly compatible."
        )
    elif drift_score < 35:
        summary = (
            f"Moderate changes detected in {total_issues} column{'s' if total_issues != 1 else ''}. "
            "Review highlighted columns before retraining."
        )
    else:
        summary = (
            f"Significant distribution shifts detected ({total_issues} column{'s' if total_issues != 1 else ''} affected). "
            "Consider whether the model needs retraining to reflect the new data patterns."
        )

    return {
        "row_count_old": n_old,
        "row_count_new": n_new,
        "row_count_change_pct": row_count_change_pct,
        "col_count_old": len(old_cols),
        "col_count_new": len(new_cols),
        "new_columns": new_columns,
        "dropped_columns": dropped_columns,
        "numeric_drifts": numeric_drifts,
        "categorical_drifts": categorical_drifts,
        "drift_score": drift_score,
        "summary": summary,
    }


def compute_version_history(
    datasets: list[dict],
    dataframes: list["pd.DataFrame"],
) -> dict:
    """Build a version history timeline from multiple dataset uploads.

    For each consecutive pair, computes drift using compute_dataset_comparison().
    Returns a timeline with per-version metadata and drift info between versions.

    Pure function — no database access, no side effects.

    Args:
        datasets: list of dicts with {id, filename, row_count, column_count,
                  uploaded_at, size_bytes}, sorted by uploaded_at ascending.
        dataframes: corresponding DataFrames loaded from disk (same order).

    Returns:
        {
            version_count: int,
            versions: [{version, dataset_id, filename, row_count, column_count,
                        uploaded_at, size_bytes,
                        drift_from_previous: {...} | None}, ...],
            overall_stability: "stable" | "moderate" | "high",
            summary: str,
        }
    """
    if not datasets or not dataframes:
        return {
            "version_count": 0,
            "versions": [],
            "overall_stability": "stable",
            "summary": "No dataset uploads found for this project.",
        }

    versions = []
    max_drift = 0

    for i, (ds, df) in enumerate(zip(datasets, dataframes)):
        version_entry: dict = {
            "version": i + 1,
            "dataset_id": ds.get("id", ""),
            "filename": ds.get("filename", ""),
            "row_count": ds.get("row_count", 0),
            "column_count": ds.get("column_count", 0),
            "uploaded_at": ds.get("uploaded_at", ""),
            "size_bytes": ds.get("size_bytes", 0),
            "drift_from_previous": None,
        }

        if i > 0:
            prev_df = dataframes[i - 1]
            try:
                comparison = compute_dataset_comparison(prev_df, df)
                drift_score = comparison["drift_score"]
                max_drift = max(max_drift, drift_score)
                n_changed = len(comparison["numeric_drifts"]) + len(
                    comparison["categorical_drifts"]
                )
                version_entry["drift_from_previous"] = {
                    "drift_score": drift_score,
                    "summary": comparison["summary"],
                    "changed_columns": n_changed,
                    "new_columns": comparison["new_columns"],
                    "dropped_columns": comparison["dropped_columns"],
                    "row_count_change_pct": comparison["row_count_change_pct"],
                }
            except Exception:  # noqa: BLE001
                version_entry["drift_from_previous"] = None

        versions.append(version_entry)

    # Overall stability from max drift seen across all transitions
    if max_drift >= 50:
        overall_stability = "high"
        stability_label = "significant"
    elif max_drift >= 20:
        overall_stability = "moderate"
        stability_label = "moderate"
    else:
        overall_stability = "stable"
        stability_label = "minimal"

    n = len(datasets)
    if n == 1:
        summary = "One dataset upload on record. No drift comparison available yet."
    else:
        summary = (
            f"{n} dataset versions uploaded. "
            f"Distribution changes across versions are {stability_label}. "
            + (
                "Data appears stable — retraining may not be necessary."
                if overall_stability == "stable"
                else "Consider retraining your model on the latest data."
            )
        )

    return {
        "version_count": n,
        "versions": versions,
        "overall_stability": overall_stability,
        "summary": summary,
    }


def compute_portfolio_summary(project_summaries: list[dict]) -> dict:
    """Aggregate all projects into a cross-project portfolio overview.

    Args:
        project_summaries: List of dicts with keys:
            project_id, name, dataset_filename, row_count,
            model_count, best_algorithm, best_metric_name,
            best_metric_value, best_problem_type, best_target_column,
            has_deployment, prediction_count, last_activity_at.

    Returns:
        Dict with: total_projects, active_deployments, total_predictions,
        best_performer (or None), projects (list), summary (plain English).
    """
    total = len(project_summaries)
    if total == 0:
        return {
            "total_projects": 0,
            "active_deployments": 0,
            "total_predictions": 0,
            "best_performer": None,
            "projects": [],
            "summary": "No projects found. Create a project and upload some data to get started.",
        }

    active_deployments = sum(1 for p in project_summaries if p.get("has_deployment"))
    total_predictions = sum(p.get("prediction_count", 0) for p in project_summaries)

    # Find best performer: project with highest metric value that has a model
    modeled = [
        p
        for p in project_summaries
        if p.get("best_metric_value") is not None and p.get("model_count", 0) > 0
    ]
    best_performer = None
    if modeled:
        # For R² (regression) higher is better; for accuracy/f1 higher is better too.
        # Sort by metric value descending.
        modeled_sorted = sorted(
            modeled, key=lambda p: p.get("best_metric_value", 0), reverse=True
        )
        bp = modeled_sorted[0]
        best_performer = {
            "project_id": bp["project_id"],
            "name": bp["name"],
            "metric_name": bp.get("best_metric_name", "score"),
            "metric_value": bp.get("best_metric_value"),
            "algorithm": bp.get("best_algorithm", ""),
            "problem_type": bp.get("best_problem_type", ""),
            "target_column": bp.get("best_target_column", ""),
        }

    # Build plain-English summary
    parts = []
    parts.append(f"You have {total} project{'s' if total > 1 else ''}")
    if active_deployments > 0:
        parts.append(
            f"{active_deployments} live prediction "
            f"API{'s' if active_deployments > 1 else ''}"
        )
    if total_predictions > 0:
        parts.append(
            f"{total_predictions:,} total prediction{'s' if total_predictions != 1 else ''} made"
        )
    if best_performer:
        metric_pct = (
            int(best_performer["metric_value"] * 100)
            if best_performer["metric_value"] is not None
            else 0
        )
        parts.append(
            f"best model: {best_performer['name']} "
            f"({best_performer['algorithm'].replace('_', ' ').title()}, "
            f"{metric_pct}% {best_performer['metric_name']})"
        )
    summary = ". ".join(parts) + "."

    return {
        "total_projects": total,
        "active_deployments": active_deployments,
        "total_predictions": total_predictions,
        "best_performer": best_performer,
        "projects": project_summaries,
        "summary": summary,
    }


_DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
_DAY_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def compute_usage_pattern(prediction_logs: list) -> dict:
    """Compute hour-of-day and day-of-week prediction usage patterns.

    Accepts a list of objects/dicts with a ``created_at`` datetime field.
    Returns aggregated counts, peak identifiers, quiet-hour recommendations,
    and a plain-English summary — all timezone-naive (UTC).
    """
    from datetime import datetime as _dt

    hour_counts = [0] * 24
    day_counts = [0] * 7
    total = 0

    for log in prediction_logs:
        ts = getattr(log, "created_at", None)
        if ts is None:
            continue
        if isinstance(ts, str):
            try:
                ts = _dt.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:  # noqa: BLE001
                continue
        try:
            hour_counts[ts.hour] += 1
            # weekday(): 0=Mon … 6=Sun
            day_counts[ts.weekday()] += 1
            total += 1
        except Exception:  # noqa: BLE001
            continue

    if total == 0:
        return {
            "hour_counts": hour_counts,
            "day_counts": day_counts,
            "peak_hour": None,
            "peak_hour_count": 0,
            "peak_day": None,
            "peak_day_name": None,
            "peak_day_count": 0,
            "quiet_hours": [],
            "busiest_period": None,
            "total_predictions": 0,
            "summary": "No predictions recorded yet — usage patterns will appear once the model starts receiving requests.",
        }

    peak_hour = int(hour_counts.index(max(hour_counts)))
    peak_day_idx = int(day_counts.index(max(day_counts)))
    peak_day_count = day_counts[peak_day_idx]

    # Quiet hours: hours with < 5% of peak volume (good maintenance windows)
    peak_h_count = hour_counts[peak_hour]
    quiet_threshold = max(1, peak_h_count * 0.05)
    quiet_hours = [h for h in range(24) if hour_counts[h] < quiet_threshold]

    # Plain-English busiest period (AM/PM grouping)
    am_total = sum(hour_counts[6:12])
    pm_total = sum(hour_counts[12:18])
    evening_total = sum(hour_counts[18:24])
    night_total = sum(hour_counts[0:6])
    period_map = {
        "morning (6am–12pm)": am_total,
        "afternoon (12pm–6pm)": pm_total,
        "evening (6pm–midnight)": evening_total,
        "late night (midnight–6am)": night_total,
    }
    busiest_period = max(period_map, key=lambda k: period_map[k])

    # Format peak hour as 12-hour clock
    def _fmt_hour(h: int) -> str:
        if h == 0:
            return "12am"
        if h < 12:
            return f"{h}am"
        if h == 12:
            return "12pm"
        return f"{h - 12}pm"

    summary_parts = [
        f"Peak usage is at {_fmt_hour(peak_hour)} UTC ({peak_h_count} "
        f"prediction{'s' if peak_h_count != 1 else ''}) "
        f"and on {_DAY_NAMES[peak_day_idx]}s ({peak_day_count} predictions).",
    ]
    if quiet_hours:
        quiet_str = ", ".join(_fmt_hour(h) for h in quiet_hours[:3])
        if len(quiet_hours) > 3:
            quiet_str += f" and {len(quiet_hours) - 3} more"
        summary_parts.append(
            f"Lowest usage: {quiet_str} UTC — good windows for maintenance or retraining."
        )

    return {
        "hour_counts": hour_counts,
        "day_counts": day_counts,
        "peak_hour": peak_hour,
        "peak_hour_count": peak_h_count,
        "peak_day": peak_day_idx,
        "peak_day_name": _DAY_NAMES[peak_day_idx],
        "peak_day_short": _DAY_SHORT[peak_day_idx],
        "quiet_hours": quiet_hours,
        "busiest_period": busiest_period,
        "total_predictions": total,
        "day_names": _DAY_SHORT,
        "summary": " ".join(summary_parts),
    }


def compute_covariate_drift_alert(
    all_inputs: list[dict],
    feature_ranges: dict[str, dict],
    *,
    oor_threshold_medium: float = 0.15,
    oor_threshold_high: float = 0.30,
    max_features: int = 10,
) -> dict:
    """Assess covariate drift between production inputs and training feature ranges.

    Compares each feature's production values against training min/max (numeric)
    or known_categories (categorical). Features exceeding oor_threshold_medium
    (15%) generate a "medium" alert; those exceeding oor_threshold_high (30%)
    generate a "high" alert.

    Args:
        all_inputs: Parsed input_features dicts from PredictionLog records.
        feature_ranges: {feature: {"min", "max", "p5", "p95"} or
                         {"known_categories": [...]}} from PredictionPipeline.
        oor_threshold_medium: Fraction of OOR/unseen values for "medium" severity.
        oor_threshold_high: Fraction of OOR/unseen values for "high" severity.
        max_features: Maximum number of features to analyse.

    Returns:
        Dict with has_alerts, severity, severity_label, sample_count,
        feature_count, alert_count, alerts (list), and summary.
    """
    if not all_inputs:
        return {
            "has_alerts": False,
            "severity": "low",
            "severity_label": "No predictions yet",
            "sample_count": 0,
            "feature_count": 0,
            "alert_count": 0,
            "alerts": [],
            "summary": "No predictions have been made yet.",
        }

    feat_names = list(all_inputs[0].keys())[:max_features]
    alerts: list[dict] = []

    for feat in feat_names:
        values = [
            inp[feat] for inp in all_inputs if feat in inp and inp[feat] is not None
        ]
        if not values:
            continue

        numeric: list[float] = []
        for v in values:
            try:
                numeric.append(float(v))
            except (TypeError, ValueError):
                pass

        ranges = feature_ranges.get(feat, {})

        if len(numeric) > len(values) * 0.5:
            train_min = ranges.get("min")
            train_max = ranges.get("max")
            if train_min is None or train_max is None:
                continue

            oor = sum(1 for v in numeric if v < train_min or v > train_max)
            oor_pct = oor / len(numeric)

            if oor_pct >= oor_threshold_medium:
                sev = "high" if oor_pct >= oor_threshold_high else "medium"
                alerts.append(
                    {
                        "feature": feat,
                        "feature_type": "numeric",
                        "oor_count": oor,
                        "oor_pct": round(oor_pct * 100, 1),
                        "total_count": len(numeric),
                        "train_min": train_min,
                        "train_max": train_max,
                        "severity": sev,
                        "description": (
                            f"{oor_pct:.0%} of '{feat}' values are outside the "
                            f"training range [{train_min:.3g}, {train_max:.3g}]"
                        ),
                    }
                )
        else:
            known = set(ranges.get("known_categories", []))
            if not known:
                continue

            unseen = sum(1 for v in values if str(v) not in known)
            unseen_pct = unseen / len(values)

            if unseen_pct >= oor_threshold_medium:
                sev = "high" if unseen_pct >= oor_threshold_high else "medium"
                alerts.append(
                    {
                        "feature": feat,
                        "feature_type": "categorical",
                        "unseen_count": unseen,
                        "unseen_pct": round(unseen_pct * 100, 1),
                        "total_count": len(values),
                        "severity": sev,
                        "description": (
                            f"{unseen_pct:.0%} of '{feat}' values are categories "
                            "not seen during training"
                        ),
                    }
                )

    has_high = any(a["severity"] == "high" for a in alerts)
    has_medium = any(a["severity"] == "medium" for a in alerts)

    if has_high:
        severity = "high"
        severity_label = "Significant drift detected"
    elif has_medium:
        severity = "medium"
        severity_label = "Some drift detected"
    else:
        severity = "low"
        severity_label = "No significant drift"

    has_alerts = bool(alerts)

    if not has_alerts:
        summary = (
            f"Production inputs look good — all {len(feat_names)} features are within "
            f"training ranges across {len(all_inputs)} recent "
            f"prediction{'s' if len(all_inputs) != 1 else ''}."
        )
    elif severity == "high":
        high_feats = [a["feature"] for a in alerts if a["severity"] == "high"]
        feat_list = ", ".join(f"'{f}'" for f in high_feats[:3])
        summary = (
            f"High drift detected in {len(alerts)} "
            f"feature{'s' if len(alerts) != 1 else ''} ({feat_list}). "
            "These inputs diverge significantly from training data — "
            "consider retraining the model."
        )
    else:
        med_feats = [a["feature"] for a in alerts]
        feat_list = ", ".join(f"'{f}'" for f in med_feats[:3])
        summary = (
            f"Moderate drift detected in {len(alerts)} "
            f"feature{'s' if len(alerts) != 1 else ''} ({feat_list}). "
            "Some production inputs are outside training ranges — "
            "monitor closely and consider retraining."
        )

    return {
        "has_alerts": has_alerts,
        "severity": severity,
        "severity_label": severity_label,
        "sample_count": len(all_inputs),
        "feature_count": len(feat_names),
        "alert_count": len(alerts),
        "alerts": alerts,
        "summary": summary,
    }


def compute_prediction_audit(
    logs: list,
    deployment: object,
    now_utc: "datetime | None" = None,
) -> dict:
    """Aggregate production monitoring signals into a single audit snapshot.

    Returns volume stats, confidence distribution, SLA percentiles, quota
    usage, and an overall health assessment — all from the same set of
    PredictionLog rows so callers only need one DB query.

    Args:
        logs: list of PredictionLog ORM objects (all logs for the deployment).
        deployment: Deployment ORM object (for quota, rate-limit metadata).
        now_utc: current UTC datetime for "today" calculations (injectable for tests).

    Returns dict with keys:
        total_predictions, predictions_today, predictions_7d, predictions_30d,
        confidence_high_pct, confidence_medium_pct, confidence_low_pct,
        has_confidence_data,
        p50_ms, p95_ms, avg_ms, has_latency_data, sla_alert,
        quota_used, monthly_quota, quota_pct, quota_enabled,
        overall_status ("healthy" | "warning" | "critical"),
        overall_label, summary.
    """
    from datetime import datetime, timezone

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    # ── Volume ──────────────────────────────────────────────────────────────
    cutoff_today = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff_7d = now_utc - __import__("datetime").timedelta(days=7)
    cutoff_30d = now_utc - __import__("datetime").timedelta(days=30)

    total = len(logs)
    today_count = sum(1 for lg in logs if _log_created_aware(lg) >= cutoff_today)
    count_7d = sum(1 for lg in logs if _log_created_aware(lg) >= cutoff_7d)
    count_30d = sum(1 for lg in logs if _log_created_aware(lg) >= cutoff_30d)

    # ── Confidence distribution ─────────────────────────────────────────────
    conf_values = [lg.confidence for lg in logs if lg.confidence is not None]
    has_conf = bool(conf_values)
    if has_conf:
        n_conf = len(conf_values)
        high = sum(1 for c in conf_values if c >= 0.80) / n_conf * 100
        medium = sum(1 for c in conf_values if 0.60 <= c < 0.80) / n_conf * 100
        low = sum(1 for c in conf_values if c < 0.60) / n_conf * 100
    else:
        high = medium = low = 0.0

    # ── SLA / latency ───────────────────────────────────────────────────────
    latencies = sorted(lg.response_ms for lg in logs if lg.response_ms is not None)
    has_latency = bool(latencies)
    if has_latency:
        p50 = _percentile_list(latencies, 50)
        p95 = _percentile_list(latencies, 95)
        avg_ms = round(sum(latencies) / len(latencies), 2)
        sla_alert = p95 > 500.0
    else:
        p50 = p95 = avg_ms = None
        sla_alert = False

    # ── Quota ───────────────────────────────────────────────────────────────
    monthly_quota = getattr(deployment, "monthly_quota", None)
    quota_enabled = bool(monthly_quota and monthly_quota > 0)
    quota_pct: float | None = None
    if quota_enabled:
        quota_pct = round(count_30d / monthly_quota * 100, 1)

    # ── Overall status ──────────────────────────────────────────────────────
    issues: list[str] = []
    if sla_alert:
        issues.append(f"p95 latency {p95}ms exceeds 500ms SLA")
    if quota_enabled and quota_pct is not None and quota_pct >= 90:
        issues.append(f"quota {quota_pct:.0f}% used this month")
    if has_conf and low > 30:
        issues.append(f"{low:.0f}% of predictions have low confidence (<60%)")

    warnings: list[str] = []
    if quota_enabled and quota_pct is not None and 70 <= quota_pct < 90:
        warnings.append(f"quota {quota_pct:.0f}% used")
    if has_conf and 15 <= low <= 30:
        warnings.append(f"{low:.0f}% low-confidence predictions")

    if issues:
        status = "critical"
        status_label = "Critical"
    elif warnings:
        status = "warning"
        status_label = "Needs Attention"
    else:
        status = "healthy"
        status_label = "Healthy"

    # ── Plain-English summary ────────────────────────────────────────────────
    if total == 0:
        summary = "No predictions recorded yet. Once the API receives requests, metrics will appear here."
    else:
        parts = [f"{total:,} total prediction{'s' if total != 1 else ''} served."]
        if count_7d > 0:
            parts.append(f"{count_7d:,} in the last 7 days, {today_count} today.")
        if has_latency:
            sla_note = " ⚠ above SLA target" if sla_alert else " within SLA"
            parts.append(f"Median response time {p50}ms (p95: {p95}ms{sla_note}).")
        if has_conf:
            parts.append(
                f"Confidence: {high:.0f}% high, {medium:.0f}% medium, {low:.0f}% low."
            )
        if issues:
            parts.append("Action needed: " + "; ".join(issues) + ".")
        summary = " ".join(parts)

    return {
        "total_predictions": total,
        "predictions_today": today_count,
        "predictions_7d": count_7d,
        "predictions_30d": count_30d,
        "confidence_high_pct": round(high, 1),
        "confidence_medium_pct": round(medium, 1),
        "confidence_low_pct": round(low, 1),
        "has_confidence_data": has_conf,
        "p50_ms": p50,
        "p95_ms": p95,
        "avg_ms": avg_ms,
        "has_latency_data": has_latency,
        "sla_alert": sla_alert,
        "quota_used": count_30d,
        "monthly_quota": monthly_quota,
        "quota_pct": quota_pct,
        "quota_enabled": quota_enabled,
        "overall_status": status,
        "overall_label": status_label,
        "summary": summary,
    }


def _log_created_aware(log: object) -> "datetime":
    """Return log.created_at as a timezone-aware UTC datetime."""
    from datetime import datetime, timezone

    dt = log.created_at  # type: ignore[attr-defined]
    if dt is None:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _percentile_list(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return round(sorted_vals[0], 2)
    idx = (pct / 100.0) * (n - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= n:
        return round(sorted_vals[-1], 2)
    frac = idx - lo
    return round(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]), 2)


def compute_confidence_trend(
    prediction_logs: list,
    window_days: int = 30,
    now_utc: "datetime | None" = None,
) -> dict:
    """Compute daily average confidence trend for a deployment.

    Bins PredictionLog records into daily buckets, computes per-day average
    confidence, fits a linear trend, and classifies direction as improving /
    stable / declining.

    Args:
        prediction_logs: list of PredictionLog ORM objects with ``confidence``
            (float 0–1) and ``created_at`` fields.
        window_days: how many calendar days to look back (default 30).
        now_utc: injectable UTC datetime for tests; defaults to datetime.now(UTC).

    Returns dict with keys:
        daily_stats: list of {date, avg_confidence, count} dicts (oldest first),
        overall_avg: mean confidence across all logs in the window,
        trend_direction: "improving" | "stable" | "declining",
        trend_rate_per_day: float — daily change in avg confidence (as 0–100 pct pts),
        peak_day: date string of highest-avg-confidence day (or None),
        peak_value: highest daily avg (0–100),
        low_day: date string of lowest-avg-confidence day (or None),
        low_value: lowest daily avg (0–100),
        sample_count: total logs with confidence data in window,
        has_data: bool,
        summary: plain-English description.
    """
    from collections import defaultdict
    from datetime import datetime, timezone

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    cutoff = now_utc - __import__("datetime").timedelta(days=window_days)

    # Collect confidence values per calendar day within the window
    day_buckets: dict = defaultdict(list)
    for log in prediction_logs:
        conf = getattr(log, "confidence", None)
        if conf is None:
            continue
        dt = _log_created_aware(log)
        if dt < cutoff:
            continue
        day_key = dt.strftime("%Y-%m-%d")
        day_buckets[day_key].append(float(conf))

    if not day_buckets:
        return {
            "daily_stats": [],
            "overall_avg": None,
            "trend_direction": "stable",
            "trend_rate_per_day": 0.0,
            "peak_day": None,
            "peak_value": None,
            "low_day": None,
            "low_value": None,
            "sample_count": 0,
            "has_data": False,
            "summary": "No confidence data available for this deployment.",
        }

    # Build daily_stats sorted oldest → newest
    sorted_days = sorted(day_buckets.keys())
    daily_stats = []
    for day in sorted_days:
        vals = day_buckets[day]
        avg_pct = round(sum(vals) / len(vals) * 100, 1)
        daily_stats.append({"date": day, "avg_confidence": avg_pct, "count": len(vals)})

    all_avgs = [d["avg_confidence"] for d in daily_stats]
    overall_avg = round(sum(all_avgs) / len(all_avgs), 1)
    sample_count = sum(d["count"] for d in daily_stats)

    # Linear trend via OLS slope (x = day index 0..n-1, y = avg_confidence %)
    n = len(all_avgs)
    slope = 0.0
    if n >= 2:
        xs = list(range(n))
        x_mean = sum(xs) / n
        y_mean = sum(all_avgs) / n
        num = sum((xs[i] - x_mean) * (all_avgs[i] - y_mean) for i in range(n))
        den = sum((xs[i] - x_mean) ** 2 for i in range(n))
        slope = num / den if den != 0 else 0.0

    trend_rate = round(slope, 3)
    if slope > 0.3:
        direction = "improving"
    elif slope < -0.3:
        direction = "declining"
    else:
        direction = "stable"

    # Peak and low days
    peak_stat = max(daily_stats, key=lambda d: d["avg_confidence"])
    low_stat = min(daily_stats, key=lambda d: d["avg_confidence"])

    # Plain-English summary
    direction_word = {
        "improving": "improving",
        "stable": "stable",
        "declining": "declining",
    }[direction]
    if direction == "stable":
        trend_desc = f"averaging {overall_avg:.0f}% confidence"
    elif direction == "improving":
        trend_desc = f"improving (+{abs(trend_rate):.2f}% per day), now averaging {all_avgs[-1]:.0f}%"
    else:
        trend_desc = (
            f"declining ({trend_rate:.2f}% per day), now averaging {all_avgs[-1]:.0f}%"
        )

    summary = (
        f"Over the last {window_days} days ({len(daily_stats)} active days, "
        f"{sample_count} predictions), model confidence is {direction_word} — {trend_desc}."
    )

    return {
        "daily_stats": daily_stats,
        "overall_avg": overall_avg,
        "trend_direction": direction,
        "trend_rate_per_day": trend_rate,
        "peak_day": peak_stat["date"],
        "peak_value": peak_stat["avg_confidence"],
        "low_day": low_stat["date"],
        "low_value": low_stat["avg_confidence"],
        "sample_count": sample_count,
        "has_data": True,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Feedback Accuracy Report — real-world accuracy from analyst-recorded outcomes
# ---------------------------------------------------------------------------


def _iso_week_start(dt: datetime) -> str:
    """Return ISO week start date string (Monday) for a given datetime."""
    from datetime import timedelta

    d = dt.date() - timedelta(days=dt.weekday())
    return d.isoformat()


def _trend_direction_from_values(values: list[float], higher_is_better: bool) -> str:
    """Detect improving/stable/declining across a time-ordered list of values."""
    if len(values) < 2:
        return "stable"
    n = len(values)
    half = n // 2
    first_half = values[:half] if half else values[:1]
    second_half = values[half:] if half else values[-1:]
    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)
    if avg_first == 0:
        return "stable"
    change_pct = (avg_second - avg_first) / abs(avg_first) * 100
    threshold = 5.0  # 5% change required to call it a trend
    if higher_is_better:
        if change_pct > threshold:
            return "improving"
        if change_pct < -threshold:
            return "declining"
    else:
        # lower is better (MAE)
        if change_pct < -threshold:
            return "improving"
        if change_pct > threshold:
            return "declining"
    return "stable"


def compute_feedback_accuracy_report(
    feedback_records: list,
    prediction_logs_map: dict,
    problem_type: str,
) -> dict:
    """Compute a real-world accuracy report from analyst-recorded feedback.

    Args:
        feedback_records: List of FeedbackRecord-like objects with attributes:
            prediction_log_id, actual_value, actual_label, is_correct, created_at.
        prediction_logs_map: Dict mapping prediction_log_id → PredictionLog-like
            object with attribute prediction_numeric (regression only).
        problem_type: "regression" or "classification".

    Returns a dict with status, accuracy metrics, weekly trend, verdict, and summary.
    """
    total_feedback = len(feedback_records)

    if total_feedback == 0:
        return {
            "status": "no_feedback",
            "total_feedback": 0,
            "problem_type": problem_type,
            "has_data": False,
            "summary": (
                "No feedback recorded yet. After making predictions and seeing the real "
                "outcomes, record them using the Deployment tab to track how well the "
                "model is performing in practice."
            ),
        }

    if problem_type == "regression":
        pairs = []  # (actual, predicted, created_at)
        for fb in feedback_records:
            if fb.actual_value is not None and fb.prediction_log_id:
                log_obj = prediction_logs_map.get(fb.prediction_log_id)
                if log_obj is not None:
                    pred_num = getattr(log_obj, "prediction_numeric", None)
                    if pred_num is not None:
                        pairs.append((fb.actual_value, float(pred_num), fb.created_at))

        if not pairs:
            return {
                "status": "feedback_only",
                "total_feedback": total_feedback,
                "paired_count": 0,
                "problem_type": problem_type,
                "has_data": False,
                "summary": (
                    f"{total_feedback} actual outcome(s) recorded but no paired "
                    "prediction logs found. Link prediction_log_id to feedback entries "
                    "to compute error metrics."
                ),
            }

        actual_vals = [p[0] for p in pairs]
        predicted_vals = [p[1] for p in pairs]
        mae = sum(abs(a - p) for a, p in zip(actual_vals, predicted_vals)) / len(pairs)
        avg_actual = sum(actual_vals) / len(actual_vals)
        pct_error = (mae / (abs(avg_actual) + 1e-9)) * 100

        if pct_error < 5:
            verdict = "excellent"
            verdict_msg = "Excellent — predictions are very close to actual outcomes."
        elif pct_error < 15:
            verdict = "good"
            verdict_msg = (
                "Good accuracy — predictions are reasonably close to actual outcomes."
            )
        elif pct_error < 30:
            verdict = "moderate"
            verdict_msg = "Moderate accuracy — consider adding more features or retraining with newer data."
        else:
            verdict = "poor"
            verdict_msg = (
                "Predictions are significantly off. Retraining is recommended."
            )

        # Weekly trend: group pairs by ISO-week start
        week_buckets: dict[str, list[float]] = {}
        for actual, predicted, ts in pairs:
            week = _iso_week_start(ts)
            week_buckets.setdefault(week, []).append(abs(actual - predicted))
        weekly_trend = [
            {
                "week_start": w,
                "mae": round(sum(errs) / len(errs), 4),
                "sample_count": len(errs),
            }
            for w, errs in sorted(week_buckets.items())
        ]
        trend_direction = _trend_direction_from_values(
            [wt["mae"] for wt in weekly_trend], higher_is_better=False
        )

        summary = (
            f"Based on {len(pairs)} matched prediction(s): "
            f"mean absolute error = {mae:.4f} ({pct_error:.1f}% of average actual value). "
            + verdict_msg
        )

        return {
            "status": "computed",
            "problem_type": problem_type,
            "total_feedback": total_feedback,
            "paired_count": len(pairs),
            "mae": round(mae, 4),
            "pct_error": round(pct_error, 2),
            "avg_actual": round(avg_actual, 4),
            "verdict": verdict,
            "verdict_msg": verdict_msg,
            "weekly_trend": weekly_trend,
            "trend_direction": trend_direction,
            "has_data": True,
            "summary": summary,
        }

    else:
        # Classification
        correct_count = sum(1 for fb in feedback_records if fb.is_correct is True)
        incorrect_count = sum(1 for fb in feedback_records if fb.is_correct is False)
        rated_count = correct_count + incorrect_count

        if rated_count == 0:
            return {
                "status": "feedback_only",
                "total_feedback": total_feedback,
                "rated_count": 0,
                "problem_type": problem_type,
                "has_data": False,
                "summary": (
                    f"{total_feedback} feedback record(s) found, but none have "
                    "is_correct set. Provide actual_label with feedback to enable "
                    "accuracy tracking."
                ),
            }

        accuracy = correct_count / rated_count
        accuracy_pct = round(accuracy * 100, 1)

        if accuracy >= 0.90:
            verdict = "excellent"
            verdict_msg = (
                f"Excellent — {accuracy_pct}% of real-world predictions were correct."
            )
        elif accuracy >= 0.75:
            verdict = "good"
            verdict_msg = f"Good — {accuracy_pct}% accuracy in practice."
        elif accuracy >= 0.60:
            verdict = "moderate"
            verdict_msg = f"Moderate — {accuracy_pct}% accuracy. Consider retraining with newer data."
        else:
            verdict = "poor"
            verdict_msg = (
                f"Low accuracy ({accuracy_pct}%). Retraining is strongly recommended."
            )

        # Weekly trend grouped by feedback.created_at
        week_buckets_cls: dict[str, list[bool]] = {}
        for fb in feedback_records:
            if fb.is_correct is not None:
                week = _iso_week_start(fb.created_at)
                week_buckets_cls.setdefault(week, []).append(fb.is_correct)
        weekly_trend = [
            {
                "week_start": w,
                "accuracy": round(sum(vals) / len(vals) * 100, 1),
                "sample_count": len(vals),
            }
            for w, vals in sorted(week_buckets_cls.items())
        ]
        trend_direction = _trend_direction_from_values(
            [wt["accuracy"] for wt in weekly_trend], higher_is_better=True
        )

        unknown_count = total_feedback - rated_count
        summary = (
            f"{correct_count} of {rated_count} rated prediction(s) were correct "
            f"({accuracy_pct}% real-world accuracy). "
            + (f"{unknown_count} record(s) not yet rated. " if unknown_count else "")
            + verdict_msg
        )

        return {
            "status": "computed",
            "problem_type": problem_type,
            "total_feedback": total_feedback,
            "rated_count": rated_count,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "unknown_count": total_feedback - rated_count,
            "accuracy": round(accuracy, 4),
            "accuracy_pct": accuracy_pct,
            "verdict": verdict,
            "verdict_msg": verdict_msg,
            "weekly_trend": weekly_trend,
            "trend_direction": trend_direction,
            "has_data": True,
            "summary": summary,
        }


# ---------------------------------------------------------------------------
# Batch job results analytics
# ---------------------------------------------------------------------------


def compute_batch_job_results(
    output_csv_bytes: bytes,
    problem_type: str,
    target_column: str,
) -> dict:
    """Analyse a batch prediction output CSV and return distribution stats.

    For regression: avg/median/min/max/std + 10-bin histogram.
    For classification: per-class count/pct + optional avg confidence.
    """
    import io as _io

    try:
        df = pd.read_csv(_io.BytesIO(output_csv_bytes))
    except Exception:
        return {"has_data": False, "summary": "Unable to parse batch output."}

    if df.empty:
        return {"has_data": False, "summary": "Batch output is empty."}

    # Locate the prediction column (prediction or {target}_prediction)
    pred_col: str | None = None
    candidates = [
        target_column,
        f"{target_column}_prediction",
        "prediction",
        f"predicted_{target_column}",
    ]
    for c in candidates:
        if c in df.columns:
            pred_col = c
            break
    if pred_col is None:
        pred_col = df.columns[-1]

    total_rows = len(df)

    if problem_type == "regression":
        values = pd.to_numeric(df[pred_col], errors="coerce").dropna()
        if len(values) == 0:
            return {"has_data": False, "summary": "No numeric predictions found."}

        avg_prediction = float(values.mean())
        median_prediction = float(values.median())
        min_prediction = float(values.min())
        max_prediction = float(values.max())
        std_prediction = float(values.std()) if len(values) > 1 else 0.0

        n_bins = min(10, max(3, len(values) // 5))
        counts, edges = np.histogram(values.values, bins=n_bins)
        histogram = [
            {
                "bin_start": float(edges[i]),
                "bin_end": float(edges[i + 1]),
                "count": int(counts[i]),
            }
            for i in range(len(counts))
        ]

        summary = (
            f"Batch produced {total_rows} predictions for {target_column}. "
            f"Average: {avg_prediction:.2f}, range {min_prediction:.2f}\u2013{max_prediction:.2f}."
        )

        return {
            "has_data": True,
            "problem_type": "regression",
            "target_column": target_column,
            "prediction_column": pred_col,
            "total_rows": total_rows,
            "avg_prediction": avg_prediction,
            "median_prediction": median_prediction,
            "min_prediction": min_prediction,
            "max_prediction": max_prediction,
            "std_prediction": std_prediction,
            "histogram": histogram,
            "summary": summary,
        }

    # classification
    class_series = df[pred_col].fillna("unknown").astype(str)
    class_counts = class_series.value_counts()
    total_classified = int(class_counts.sum())

    class_distribution = [
        {
            "class_name": str(cls),
            "count": int(cnt),
            "pct": (
                round(100.0 * int(cnt) / total_classified, 1)
                if total_classified > 0
                else 0.0
            ),
        }
        for cls, cnt in class_counts.items()
    ]

    top_class = class_distribution[0]["class_name"] if class_distribution else "unknown"
    top_pct = class_distribution[0]["pct"] if class_distribution else 0.0

    # Optional: average confidence column
    avg_confidence: float | None = None
    for col in df.columns:
        if "confidence" in col.lower():
            conf_vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(conf_vals) > 0:
                mean_val = float(conf_vals.mean())
                # Convert proportion (0–1) to percentage
                if mean_val <= 1.0:
                    mean_val *= 100.0
                avg_confidence = round(mean_val, 1)
            break

    summary = (
        f"Batch produced {total_rows} predictions for {target_column}. "
        f"Most common: '{top_class}' ({top_pct}% of predictions)."
    )

    return {
        "has_data": True,
        "problem_type": "classification",
        "target_column": target_column,
        "prediction_column": pred_col,
        "total_rows": total_rows,
        "top_class": top_class,
        "top_pct": top_pct,
        "class_distribution": class_distribution,
        "avg_confidence": avg_confidence,
        "summary": summary,
    }


def compute_deployments_overview(deployment_summaries: list[dict]) -> dict:
    """Aggregate all active deployments into a cross-project status overview.

    Pure function — no database or filesystem access.

    Args:
        deployment_summaries: List of dicts, one per active deployment, with keys:
            deployment_id, project_id, project_name, algorithm, algorithm_plain,
            target_column, environment, created_at_iso, request_count,
            last_predicted_at_iso, health_score, status, top_issue, recommendation,
            api_key_enabled, rate_limit_rpm, monthly_quota,
            predictions_last_7d, predictions_today.

    Returns:
        Dict with: total_deployments, production_count, staging_count,
        total_predictions, avg_health_score, healthy_count, warning_count,
        critical_count, deployments (sorted health desc then request_count desc),
        summary (plain English).
    """
    total = len(deployment_summaries)
    if total == 0:
        return {
            "total_deployments": 0,
            "production_count": 0,
            "staging_count": 0,
            "total_predictions": 0,
            "avg_health_score": 0,
            "healthy_count": 0,
            "warning_count": 0,
            "critical_count": 0,
            "deployments": [],
            "summary": (
                "No active deployments found. Deploy a trained model to create a "
                "live prediction endpoint."
            ),
        }

    production_count = sum(
        1 for d in deployment_summaries if d.get("environment") == "production"
    )
    staging_count = total - production_count
    total_predictions = sum(d.get("request_count", 0) for d in deployment_summaries)
    avg_health_score = int(
        sum(d.get("health_score", 0) for d in deployment_summaries) / total
    )

    healthy_count = sum(1 for d in deployment_summaries if d.get("status") == "healthy")
    warning_count = sum(1 for d in deployment_summaries if d.get("status") == "warning")
    critical_count = sum(
        1 for d in deployment_summaries if d.get("status") == "critical"
    )

    # Sort: production first, then by health score desc, then by request count desc
    sorted_deployments = sorted(
        deployment_summaries,
        key=lambda d: (
            0 if d.get("environment") == "production" else 1,
            -d.get("health_score", 0),
            -d.get("request_count", 0),
        ),
    )

    # Build plain-English summary
    parts = [f"You have {total} active deployment{'s' if total > 1 else ''}"]
    if production_count > 0:
        parts.append(
            f"{production_count} in production"
            + (f", {staging_count} in staging" if staging_count > 0 else "")
        )
    if total_predictions > 0:
        parts.append(
            f"{total_predictions:,} total prediction{'s' if total_predictions != 1 else ''} served"
        )
    if critical_count > 0:
        parts.append(
            f"{critical_count} deployment{'s' if critical_count > 1 else ''} need{'s' if critical_count == 1 else ''} attention"
        )
    elif warning_count > 0:
        parts.append(
            f"{warning_count} deployment{'s' if warning_count > 1 else ''} showing warnings"
        )
    else:
        parts.append("all deployments healthy")

    summary = ". ".join(parts) + "."

    return {
        "total_deployments": total,
        "production_count": production_count,
        "staging_count": staging_count,
        "total_predictions": total_predictions,
        "avg_health_score": avg_health_score,
        "healthy_count": healthy_count,
        "warning_count": warning_count,
        "critical_count": critical_count,
        "deployments": sorted_deployments,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Training vs production performance comparison
# ---------------------------------------------------------------------------


def compute_training_vs_production(
    feedback_records: list,
    prediction_logs_map: dict,
    model_run_metrics: dict,
    problem_type: str,
) -> dict:
    """Compare training metrics against live production accuracy from feedback.

    Args:
        feedback_records: List of FeedbackRecord-like objects.
        prediction_logs_map: Dict mapping prediction_log_id → PredictionLog-like.
        model_run_metrics: ModelRun.metrics dict from the deployed model run.
        problem_type: "regression" or "classification".

    Returns a dict with: training_value, live_value, metric_name, degradation_pct,
    status, weekly_timeline, and plain-English summary.
    """
    metrics = model_run_metrics or {}

    if problem_type == "regression":
        metric_name = "MAE"
        training_value: float | None = None
        raw = metrics.get("mae")
        if raw is not None:
            try:
                training_value = float(raw)
            except (TypeError, ValueError):
                pass

        # Compute live MAE from feedback pairs
        pairs = []
        for fb in feedback_records:
            if fb.actual_value is not None and fb.prediction_log_id:
                log_obj = prediction_logs_map.get(fb.prediction_log_id)
                if log_obj is not None:
                    pred_num = getattr(log_obj, "prediction_numeric", None)
                    if pred_num is not None:
                        pairs.append(
                            (float(fb.actual_value), float(pred_num), fb.created_at)
                        )

        if not pairs:
            return {
                "has_data": False,
                "problem_type": problem_type,
                "metric_name": metric_name,
                "training_value": training_value,
                "status": "no_feedback",
                "summary": (
                    "No matched feedback yet. Record actual outcomes in the Deployment "
                    "tab to compare training and production accuracy."
                ),
            }

        live_mae = sum(abs(a - p) for a, p, _ in pairs) / len(pairs)

        # Degradation: how much worse is the live MAE vs training?
        if training_value and training_value > 0:
            degradation_pct = round(
                (live_mae - training_value) / training_value * 100, 1
            )
        else:
            degradation_pct = 0.0

        if degradation_pct < 10:
            status = "stable"
        elif degradation_pct < 30:
            status = "warning"
        else:
            status = "degrading"

        # Weekly timeline (MAE per week)
        week_buckets: dict[str, list[float]] = {}
        for actual, predicted, ts in pairs:
            week = _iso_week_start(ts)
            week_buckets.setdefault(week, []).append(abs(actual - predicted))
        weekly_timeline = [
            {
                "period": w,
                "value": round(sum(errs) / len(errs), 4),
                "n": len(errs),
            }
            for w, errs in sorted(week_buckets.items())
        ]

        if status == "stable":
            status_msg = "Production performance is stable."
        elif status == "warning":
            status_msg = (
                f"Error has increased by {degradation_pct:.0f}% vs training. "
                "Consider monitoring closely."
            )
        else:
            status_msg = (
                f"Error has increased by {degradation_pct:.0f}% vs training. "
                "Retraining with recent data is recommended."
            )

        summary = (
            f"Training MAE: {training_value:.4f} | Live MAE: {live_mae:.4f} "
            f"(from {len(pairs)} feedback records). {status_msg}"
        )

        return {
            "has_data": True,
            "problem_type": problem_type,
            "metric_name": metric_name,
            "metric_direction": "lower_is_better",
            "training_value": round(training_value, 4) if training_value else None,
            "live_value": round(live_mae, 4),
            "degradation_pct": degradation_pct,
            "status": status,
            "n_feedback": len(pairs),
            "weekly_timeline": weekly_timeline,
            "summary": summary,
        }

    else:
        # Classification — compare training accuracy vs live accuracy
        metric_name = "Accuracy"
        training_value = None
        raw = metrics.get("accuracy")
        if raw is not None:
            try:
                training_value = float(raw)
            except (TypeError, ValueError):
                pass

        correct = sum(1 for fb in feedback_records if fb.is_correct is True)
        incorrect = sum(1 for fb in feedback_records if fb.is_correct is False)
        rated = correct + incorrect

        if rated == 0:
            return {
                "has_data": False,
                "problem_type": problem_type,
                "metric_name": metric_name,
                "training_value": training_value,
                "status": "no_feedback",
                "summary": (
                    "No rated feedback yet. After making predictions and seeing outcomes, "
                    "record them in the Deployment tab to track live accuracy."
                ),
            }

        live_accuracy = correct / rated

        # Degradation: how much accuracy dropped vs training
        if training_value and training_value > 0:
            degradation_pct = round(
                (training_value - live_accuracy) / training_value * 100, 1
            )
        else:
            degradation_pct = 0.0

        if degradation_pct < 5:
            status = "stable"
        elif degradation_pct < 15:
            status = "warning"
        else:
            status = "degrading"

        # Weekly timeline (accuracy % per week)
        week_buckets_cls: dict[str, list[bool]] = {}
        for fb in feedback_records:
            if fb.is_correct is not None:
                week = _iso_week_start(fb.created_at)
                week_buckets_cls.setdefault(week, []).append(fb.is_correct)
        weekly_timeline = [
            {
                "period": w,
                "value": round(sum(vals) / len(vals) * 100, 1),
                "n": len(vals),
            }
            for w, vals in sorted(week_buckets_cls.items())
        ]

        live_pct = round(live_accuracy * 100, 1)
        train_pct = round((training_value or 0) * 100, 1)

        if status == "stable":
            status_msg = "Production accuracy is stable."
        elif status == "warning":
            status_msg = (
                f"Accuracy dropped {degradation_pct:.0f}% vs training. "
                "Monitor for further drift."
            )
        else:
            status_msg = (
                f"Accuracy dropped {degradation_pct:.0f}% vs training. "
                "Retraining with recent data is recommended."
            )

        summary = (
            f"Training accuracy: {train_pct}% | Live accuracy: {live_pct}% "
            f"(from {rated} rated feedback records). {status_msg}"
        )

        return {
            "has_data": True,
            "problem_type": problem_type,
            "metric_name": metric_name,
            "metric_direction": "higher_is_better",
            "training_value": round(training_value, 4) if training_value else None,
            "training_pct": train_pct,
            "live_value": round(live_accuracy, 4),
            "live_pct": live_pct,
            "degradation_pct": degradation_pct,
            "status": status,
            "n_feedback": rated,
            "weekly_timeline": weekly_timeline,
            "summary": summary,
        }


# ---------------------------------------------------------------------------
# Auto-insight on new dataset upload (Track E — proactive data findings)
# ---------------------------------------------------------------------------

# Token-set of date-related words for column name detection.
# We split on underscores/hyphens and check individual tokens rather than
# using \b which fails on underscore-separated names like "order_date".
_DATE_TOKENS = frozenset(
    {
        "date",
        "datetime",
        "time",
        "timestamp",
        "created",
        "updated",
        "year",
        "month",
        "day",
        "week",
        "period",
        "quarter",
    }
)


def _has_date_token(col_name: str) -> bool:
    """Return True if any underscore/hyphen-separated token in col_name is a date keyword."""
    tokens = re.split(r"[_\-\s]+", col_name.lower())
    return bool(_DATE_TOKENS & set(tokens))


# Column name patterns that suggest an ID / key column
_ID_NAME_RE = re.compile(
    r"(^id$|_id$|^id_|_key$|^pk$|_pk$|^uuid$|^guid$)",
    re.IGNORECASE,
)


def compute_auto_insights(profile: dict, col_names: list[str]) -> list[dict]:
    """Return up to 3 ranked interesting findings from a dataset profile.

    Each finding is a dict with:
        insight_type:     str  (slug for the finding category)
        icon:             str  (emoji)
        finding:          str  (plain-English description of what was found)
        suggested_action: str  (one-click follow-up question for the analyst)
        priority:         int  (1=high, 2=medium, 3=low)

    Returns sorted by priority (ascending), capped at 3 entries.
    Designed to be called as a pure function — no database access.
    """
    findings: list[dict] = []
    columns: list[dict] = profile.get("columns", [])
    row_count: int = profile.get("row_count", 0)

    # 1. Strong correlation between two numeric columns
    correlations: list[dict] = profile.get("correlations", [])
    if correlations:
        # correlations is a list of {col1, col2, r} sorted by |r|
        for corr in correlations[:5]:
            r_val = float(corr.get("r", 0))
            if abs(r_val) >= 0.65:
                col1 = corr.get("col1", "")
                col2 = corr.get("col2", "")
                direction = "positively" if r_val > 0 else "negatively"
                strength = "very strongly" if abs(r_val) >= 0.85 else "strongly"
                findings.append(
                    {
                        "insight_type": "strong_correlation",
                        "icon": "🔗",
                        "finding": (
                            f"**{col1}** and **{col2}** are {strength} {direction} "
                            f"correlated (r = {r_val:+.2f}) — they tend to move together."
                        ),
                        "suggested_action": f"Show me the relationship between {col1} and {col2}",
                        "priority": 1,
                    }
                )
                break  # Only report the strongest correlation

    # 2. Date / time column detected → temporal analysis possible
    date_cols = [
        col
        for col in columns
        if _has_date_token(col.get("name", ""))
        or str(col.get("dtype", "")).startswith("datetime")
    ]
    numeric_cols = [
        col
        for col in columns
        if pd.api.types.is_numeric_dtype(pd.Series(dtype=col.get("dtype", "object")))
        and not _ID_NAME_RE.search(col.get("name", ""))
    ]
    if date_cols and numeric_cols:
        date_col_name = date_cols[0].get("name", "date")
        num_col_name = numeric_cols[0].get("name", "value")
        findings.append(
            {
                "insight_type": "date_column",
                "icon": "📅",
                "finding": (
                    f"I found a **{date_col_name}** column — this data has a time dimension. "
                    f"Temporal patterns (trends, seasonality) can reveal a lot."
                ),
                "suggested_action": f"Show me {num_col_name} trends over time",
                "priority": 1,
            }
        )

    # 3. Class imbalance in a binary-ish categorical column
    for col in columns:
        dtype = str(col.get("dtype", "object"))
        unique_count = col.get("unique_count", 0)
        if unique_count not in (2, 3):
            continue
        if "int" in dtype or "float" in dtype:
            continue
        dist: list = col.get("value_counts", [])
        if not dist or len(dist) < 2:
            continue
        # dist is list of {value, count} sorted by count desc
        total_in_dist = sum(item.get("count", 0) for item in dist)
        if total_in_dist == 0:
            continue
        top_pct = round(dist[0].get("count", 0) / total_in_dist * 100, 1)
        if top_pct >= 75:
            col_name = col.get("name", "")
            minority_pct = round(100 - top_pct, 1)
            findings.append(
                {
                    "insight_type": "class_imbalance",
                    "icon": "⚖️",
                    "finding": (
                        f"Column **{col_name}** is imbalanced: {top_pct}% "
                        f'"{dist[0].get("value", "")}" vs {minority_pct}% other. '
                        "Imbalanced targets can make models biased toward the majority class."
                    ),
                    "suggested_action": f"How do I handle class imbalance in {col_name}?",
                    "priority": 1,
                }
            )
            break  # Report only the most imbalanced column

    # 4. High missing values in an important column
    high_missing = [
        col
        for col in columns
        if col.get("null_pct", 0) >= 20 and not _ID_NAME_RE.search(col.get("name", ""))
    ]
    if high_missing:
        worst = max(high_missing, key=lambda c: c.get("null_pct", 0))
        col_name = worst.get("name", "")
        pct = worst.get("null_pct", 0)
        severity = "critical" if pct >= 50 else "notable"
        findings.append(
            {
                "insight_type": "high_missing",
                "icon": "⚠️",
                "finding": (
                    f"Column **{col_name}** is missing **{pct:.0f}%** of its values "
                    f"— that's {severity}. Missing data can reduce model accuracy."
                ),
                "suggested_action": f"How should I handle missing values in {col_name}?",
                "priority": 2,
            }
        )

    # 5. Likely ID column included in dataset
    id_cols = [
        col
        for col in columns
        if _ID_NAME_RE.search(col.get("name", ""))
        or (col.get("unique_count", 0) >= max(row_count * 0.9, 50) and row_count > 0)
    ]
    if id_cols:
        id_col_name = id_cols[0].get("name", "")
        findings.append(
            {
                "insight_type": "high_cardinality",
                "icon": "🔑",
                "finding": (
                    f"Column **{id_col_name}** looks like an ID field "
                    f"({id_cols[0].get('unique_count', 0):,} unique values) — "
                    "it should probably be excluded from model features."
                ),
                "suggested_action": f"Should I include {id_col_name} in my model?",
                "priority": 3,
            }
        )

    # 6. Highly skewed numeric column (right-tail — suggests log transform)
    for col in columns:
        mean_val = col.get("mean")
        std_val = col.get("std")
        min_val = col.get("min")
        if mean_val is None or std_val is None or min_val is None:
            continue
        if mean_val <= 0 or std_val <= 0:
            continue
        # Coefficient of variation > 2 with non-negative values → high right skew
        if std_val > 2 * mean_val and min_val >= 0:
            col_name = col.get("name", "")
            if _ID_NAME_RE.search(col_name):
                continue
            findings.append(
                {
                    "insight_type": "numeric_skew",
                    "icon": "📊",
                    "finding": (
                        f"Column **{col_name}** is heavily right-skewed "
                        f"(std {std_val:.1f} >> mean {mean_val:.1f}). "
                        "A log transform often improves model accuracy for skewed targets."
                    ),
                    "suggested_action": f"Should I apply a log transform to {col_name}?",
                    "priority": 3,
                }
            )
            break  # Report only one skewed column

    # Sort by priority (ascending = most important first), cap at 3
    findings.sort(key=lambda f: f["priority"])
    return findings[:3]


# ---------------------------------------------------------------------------
# Column type suggestion helpers
# ---------------------------------------------------------------------------

_BOOL_VALUES = frozenset({"true", "false", "yes", "no", "0", "1", "y", "n", "t", "f"})

_NUMERIC_COL_NAME_RE = re.compile(
    r"\b(price|amount|cost|revenue|salary|wage|qty|quantity|count|total|"
    r"score|rate|ratio|pct|percent|age|weight|height|distance|duration|"
    r"budget|spend|value|profit|loss|margin|balance|units|volume)\b",
    re.IGNORECASE,
)


def _sample_looks_numeric(sample_values: list) -> bool:
    """Return True if all non-empty sample values parse as float."""
    if not sample_values:
        return False
    valid = [str(v).strip() for v in sample_values if str(v).strip()]
    if not valid:
        return False
    parsed = 0
    for v in valid:
        try:
            float(v.replace(",", "").replace("$", "").replace("%", ""))
            parsed += 1
        except ValueError:
            return False
    return parsed > 0


def _sample_looks_boolean(sample_values: list, unique_count: int) -> bool:
    """Return True if all sample values are boolean-like."""
    if unique_count > 3 or not sample_values:
        return False
    lower_vals = {str(v).strip().lower() for v in sample_values if str(v).strip()}
    return bool(lower_vals) and lower_vals.issubset(_BOOL_VALUES)


def _sample_looks_datetime(col_name: str, sample_values: list) -> bool:
    """Return True if the column name has a date token and sample values look like dates."""
    if not _has_date_token(col_name):
        return False
    if not sample_values:
        return False
    try:
        import pandas as _pd_dt

        for v in sample_values[:3]:
            _pd_dt.to_datetime(str(v))  # No deprecated infer_datetime_format arg
        return True
    except Exception:
        return False


def _sample_all_whole_numbers(sample_values: list) -> bool:
    """Return True if all sample float values are whole numbers (e.g. 1.0, 2.0)."""
    if not sample_values:
        return False
    for v in sample_values:
        try:
            f = float(v)
            if f != int(f):
                return False
        except (TypeError, ValueError):
            return False
    return True


def compute_column_type_suggestions(profile: dict) -> dict:
    """Scan dataset column profiles and return type mismatch suggestions.

    Works purely from the stored profile dict — no DataFrame or file access.
    Each suggestion describes a column whose stored dtype does not match what
    the data actually looks like, with a plain-English reason and a one-click
    fix action.

    Returns::
        {
          "suggestions": list[dict],   # each has column/current_dtype/suggested_dtype/...
          "has_suggestions": bool,
          "dataset_rows": int,
          "dataset_cols": int,
        }
    """
    columns: list[dict] = profile.get("columns", [])
    row_count: int = profile.get("row_count", 0)
    suggestions: list[dict] = []

    for col in columns:
        col_name: str = col.get("name", "")
        dtype: str = str(col.get("dtype", "object"))
        unique_count: int = col.get("unique_count", 0)
        null_pct: float = col.get("null_pct", 0.0)
        sample_values: list = col.get("sample_values", [])

        # Skip columns with very high null rate — unreliable sample
        if null_pct > 90:
            continue

        # ---- Rule 1: object dtype that looks numeric ----
        if dtype == "object" and not _has_date_token(col_name):
            if _sample_looks_numeric(sample_values):
                suggestions.append(
                    {
                        "column": col_name,
                        "current_dtype": "text",
                        "suggested_dtype": "numeric",
                        "reason": (
                            f"The values in **{col_name}** look like numbers "
                            f"(e.g. {', '.join(str(v) for v in sample_values[:3])}), "
                            "but the column is stored as text. "
                            "Keeping it as text prevents AutoModeler from computing "
                            "statistics, correlations, and predictions with this column."
                        ),
                        "confidence": "high",
                        "sample_values": [str(v) for v in sample_values[:4]],
                        "suggested_action": f"Convert {col_name} to numeric",
                    }
                )
                continue  # Don't double-report this column

        # ---- Rule 2: object dtype that looks boolean ----
        if dtype == "object" and _sample_looks_boolean(sample_values, unique_count):
            suggestions.append(
                {
                    "column": col_name,
                    "current_dtype": "text",
                    "suggested_dtype": "boolean",
                    "reason": (
                        f"**{col_name}** only contains values like "
                        f"{', '.join(repr(str(v)) for v in sample_values[:3])}, "
                        "which look like True/False flags. "
                        "Converting to boolean allows AutoModeler to use this column "
                        "as a classification target or binary feature."
                    ),
                    "confidence": "high",
                    "sample_values": [str(v) for v in sample_values[:4]],
                    "suggested_action": f"Convert {col_name} to boolean",
                }
            )
            continue

        # ---- Rule 3: object dtype that looks like dates ----
        if dtype == "object" and _sample_looks_datetime(col_name, sample_values):
            suggestions.append(
                {
                    "column": col_name,
                    "current_dtype": "text",
                    "suggested_dtype": "datetime",
                    "reason": (
                        f"**{col_name}** contains date-like values "
                        f"(e.g. {', '.join(str(v) for v in sample_values[:2])}). "
                        "Parsing it as a proper date lets AutoModeler extract "
                        "day-of-week, month, and seasonal patterns automatically."
                    ),
                    "confidence": "medium",
                    "sample_values": [str(v) for v in sample_values[:4]],
                    "suggested_action": f"Parse {col_name} as a date column",
                }
            )
            continue

        # ---- Rule 4: float64 where all sample values are whole numbers ----
        if dtype in ("float64", "float32") and _sample_all_whole_numbers(sample_values):
            # Only flag if not an ID column (IDs can legitimately be floats)
            if not _ID_NAME_RE.search(col_name):
                suggestions.append(
                    {
                        "column": col_name,
                        "current_dtype": "decimal",
                        "suggested_dtype": "integer",
                        "reason": (
                            f"**{col_name}** is stored as a decimal number "
                            f"(e.g. {', '.join(str(v) for v in sample_values[:3])}), "
                            "but all values are whole numbers. "
                            "Converting to integer makes the column cleaner and "
                            "prevents unexpected .0 suffixes in predictions."
                        ),
                        "confidence": "medium",
                        "sample_values": [str(v) for v in sample_values[:4]],
                        "suggested_action": f"Convert {col_name} to integer",
                    }
                )

    return {
        "suggestions": suggestions,
        "has_suggestions": bool(suggestions),
        "dataset_rows": row_count,
        "dataset_cols": len(columns),
    }


def compute_feature_redundancy(
    df: pd.DataFrame,
    feature_names: list[str],
    threshold: float = 0.85,
) -> dict:
    """Detect pairs of features that are so highly correlated they carry duplicate information.

    Uses Pearson correlation on numeric columns. For each redundant pair (|corr| > threshold),
    recommends which feature to keep (higher variance wins; ties broken alphabetically).

    Args:
        df: DataFrame containing the feature columns.
        feature_names: List of columns to check (should exclude the target).
        threshold: Absolute correlation above which features are considered redundant (0–1).

    Returns:
        dict with keys:
            redundant_pairs    – list of {feature_a, feature_b, correlation, keep, drop}
            redundant_groups   – list of grouped feature clusters (each a list of col names)
            n_redundant        – int: total features involved in at least one redundant pair
            n_features_checked – int: numeric features examined
            threshold          – float: threshold used
            verdict            – "none" | "low" | "high"
            verdict_label      – plain-English label
            summary            – one-sentence overview
    """
    MIN_ROWS = 10

    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Dataset has only {len(df)} rows — need at least {MIN_ROWS} for "
            "feature redundancy analysis."
        )

    # Keep only numeric columns that appear in feature_names
    numeric_cols = [
        c
        for c in feature_names
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(numeric_cols) < 2:
        return {
            "redundant_pairs": [],
            "redundant_groups": [],
            "n_redundant": 0,
            "n_features_checked": len(numeric_cols),
            "threshold": threshold,
            "verdict": "none",
            "verdict_label": "No Redundancy Detected",
            "summary": (
                "Not enough numeric features to check for redundancy "
                f"(found {len(numeric_cols)} numeric column(s) — need at least 2)."
            ),
        }

    sub = df[numeric_cols].dropna(how="all")
    # Fill remaining NaN with column median so correlation can be computed
    sub = sub.fillna(sub.median(numeric_only=True))

    corr_matrix = sub.corr(method="pearson")
    variances = sub.var()

    # Collect all pairs above threshold (upper triangle only to avoid duplicates)
    redundant_pairs: list[dict] = []
    seen_as_drop: set[str] = set()

    for i, col_a in enumerate(numeric_cols):
        for col_b in numeric_cols[i + 1 :]:
            corr_val = corr_matrix.loc[col_a, col_b]
            if pd.isna(corr_val):
                continue
            if abs(corr_val) >= threshold:
                # Recommend keeping the feature with higher variance
                var_a = float(variances.get(col_a, 0.0))
                var_b = float(variances.get(col_b, 0.0))
                if var_a >= var_b:
                    keep, drop = col_a, col_b
                else:
                    keep, drop = col_b, col_a
                seen_as_drop.add(drop)
                redundant_pairs.append(
                    {
                        "feature_a": col_a,
                        "feature_b": col_b,
                        "correlation": round(float(corr_val), 4),
                        "correlation_abs": round(abs(float(corr_val)), 4),
                        "direction": "positive" if corr_val > 0 else "negative",
                        "keep": keep,
                        "drop": drop,
                        "reason": (
                            f"{col_a} and {col_b} are {abs(corr_val):.0%} correlated — "
                            f"they carry nearly identical information. "
                            f"Keeping {keep} (higher variance) is sufficient."
                        ),
                    }
                )

    # Build redundant groups using union-find
    parent: dict[str, str] = {}

    def _find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent.get(x, x)
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for pair in redundant_pairs:
        _union(pair["feature_a"], pair["feature_b"])

    groups_map: dict[str, list[str]] = {}
    all_involved = set()
    for pair in redundant_pairs:
        for col in (pair["feature_a"], pair["feature_b"]):
            all_involved.add(col)
            root = _find(col)
            groups_map.setdefault(root, [])
            if col not in groups_map[root]:
                groups_map[root].append(col)

    redundant_groups = [
        sorted(members) for members in groups_map.values() if len(members) >= 2
    ]
    n_redundant = len(all_involved)
    n_features_checked = len(numeric_cols)

    if n_redundant == 0:
        verdict = "none"
        verdict_label = "No Redundancy Detected"
        summary = (
            f"None of the {n_features_checked} numeric features exceed the "
            f"{threshold:.0%} correlation threshold — your feature set has minimal redundancy."
        )
    elif len(redundant_pairs) <= 2:
        verdict = "low"
        verdict_label = "Low Redundancy"
        summary = (
            f"Found {len(redundant_pairs)} redundant feature pair(s) out of "
            f"{n_features_checked} numeric features checked. "
            f"Consider dropping: {', '.join(sorted(seen_as_drop))}."
        )
    else:
        verdict = "high"
        verdict_label = "High Redundancy"
        summary = (
            f"Found {len(redundant_pairs)} redundant feature pair(s) involving "
            f"{n_redundant} features. "
            f"Simplify your model by dropping: {', '.join(sorted(seen_as_drop))}."
        )

    return {
        "redundant_pairs": redundant_pairs,
        "redundant_groups": redundant_groups,
        "n_redundant": n_redundant,
        "n_features_checked": n_features_checked,
        "threshold": threshold,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "summary": summary,
    }


def compute_target_leakage(
    df: pd.DataFrame,
    target_col: str,
    feature_names: list[str],
    high_threshold: float = 0.90,
    moderate_threshold: float = 0.75,
) -> dict:
    """Identify features that may be leaking information about the target.

    Computes Pearson correlation for numeric features vs a numeric target,
    and mutual information (normalized) for all features vs a categorical target.
    Flags features whose correlation exceeds the moderate or high threshold.

    Args:
        df: DataFrame with all columns.
        target_col: Name of the target column.
        feature_names: Feature columns to check (should exclude target).
        high_threshold: Correlation above this is "high risk" leakage (default 0.90).
        moderate_threshold: Correlation above this is "moderate risk" (default 0.75).

    Returns:
        dict with keys:
            leaky_features  – list of {feature, correlation, risk_level, reason}
            n_checked       – int: number of features examined
            verdict         – "none" | "warning" | "severe"
            verdict_label   – plain-English verdict label
            high_threshold  – threshold used for high-risk classification
            moderate_threshold – threshold used for moderate-risk classification
            target_col      – target column name
            summary         – one-sentence plain-English overview
    """
    MIN_ROWS = 10

    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Dataset has only {len(df)} rows — need at least {MIN_ROWS} for "
            "target leakage analysis."
        )

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    valid_features = [c for c in feature_names if c in df.columns and c != target_col]
    if not valid_features:
        return {
            "leaky_features": [],
            "n_checked": 0,
            "verdict": "none",
            "verdict_label": "No Leakage Detected",
            "high_threshold": high_threshold,
            "moderate_threshold": moderate_threshold,
            "target_col": target_col,
            "summary": "No features to check — dataset may have no feature columns.",
        }

    target_series = df[target_col].dropna()
    is_numeric_target = pd.api.types.is_numeric_dtype(df[target_col])

    leaky_features: list[dict] = []

    if is_numeric_target:
        # Pearson correlation for numeric features vs numeric target
        for feat in valid_features:
            col_series = df[feat]
            if not pd.api.types.is_numeric_dtype(col_series):
                continue
            combined = df[[feat, target_col]].dropna()
            if len(combined) < MIN_ROWS:
                continue
            try:
                corr_val = combined[feat].corr(combined[target_col])
            except Exception:  # noqa: BLE001
                continue
            if pd.isna(corr_val):
                continue
            abs_corr = abs(corr_val)
            if abs_corr >= moderate_threshold:
                if abs_corr >= high_threshold:
                    risk_level = "high"
                    risk_label = "High Risk"
                else:
                    risk_level = "moderate"
                    risk_label = "Moderate Risk"
                leaky_features.append(
                    {
                        "feature": feat,
                        "correlation": round(float(corr_val), 4),
                        "correlation_abs": round(abs_corr, 4),
                        "risk_level": risk_level,
                        "risk_label": risk_label,
                        "reason": (
                            f"'{feat}' has a {abs_corr:.0%} Pearson correlation with "
                            f"'{target_col}'. A correlation this high suggests the feature "
                            "may be a direct derivative of the target or recorded at the "
                            "same time — it could be cheating."
                        ),
                    }
                )
    else:
        # Categorical target: use mutual information (normalized by target entropy)
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.preprocessing import LabelEncoder

        le_target = LabelEncoder()
        y_enc = le_target.fit_transform(target_series.astype(str))
        # Need to align indices
        valid_idx = df[target_col].notna()

        numeric_feats = [
            f for f in valid_features if pd.api.types.is_numeric_dtype(df[f])
        ]
        if numeric_feats:
            X_num = df.loc[valid_idx, numeric_feats].fillna(0).values
            try:
                mi_scores = mutual_info_classif(
                    X_num, y_enc, discrete_features=False, random_state=42
                )
                # Normalize by log2(n_classes) to get 0-1 range
                import math

                n_classes = len(le_target.classes_)
                norm_factor = math.log2(n_classes) if n_classes > 1 else 1.0
                for feat, mi in zip(numeric_feats, mi_scores):
                    norm_mi = mi / norm_factor if norm_factor > 0 else mi
                    norm_mi = min(norm_mi, 1.0)
                    if norm_mi >= moderate_threshold:
                        if norm_mi >= high_threshold:
                            risk_level = "high"
                            risk_label = "High Risk"
                        else:
                            risk_level = "moderate"
                            risk_label = "Moderate Risk"
                        leaky_features.append(
                            {
                                "feature": feat,
                                "correlation": round(float(norm_mi), 4),
                                "correlation_abs": round(float(norm_mi), 4),
                                "risk_level": risk_level,
                                "risk_label": risk_label,
                                "reason": (
                                    f"'{feat}' shares {norm_mi:.0%} of the information "
                                    f"in '{target_col}' (normalized mutual information). "
                                    "This level of information overlap suggests the feature "
                                    "may be derived from the target."
                                ),
                            }
                        )
            except Exception:  # noqa: BLE001
                pass

    # Sort by correlation_abs descending
    leaky_features.sort(key=lambda x: x["correlation_abs"], reverse=True)

    n_checked = len(valid_features)
    n_high = sum(1 for f in leaky_features if f["risk_level"] == "high")
    n_moderate = sum(1 for f in leaky_features if f["risk_level"] == "moderate")

    if n_high > 0:
        verdict = "severe"
        verdict_label = "Likely Leakage"
    elif n_moderate > 0:
        verdict = "warning"
        verdict_label = "Possible Leakage"
    else:
        verdict = "none"
        verdict_label = "No Leakage Detected"

    if verdict == "none":
        summary = (
            f"None of the {n_checked} features show suspicious correlation "
            f"with '{target_col}' — no obvious leakage detected."
        )
    elif verdict == "warning":
        feat_names = ", ".join(f["feature"] for f in leaky_features[:3])
        summary = (
            f"{n_moderate} feature(s) show moderate correlation with '{target_col}': "
            f"{feat_names}. Review whether these columns are available at prediction time."
        )
    else:
        feat_names = ", ".join(
            f["feature"] for f in leaky_features if f["risk_level"] == "high"
        )
        summary = (
            f"{n_high} feature(s) are highly correlated with '{target_col}' "
            f"({feat_names}) — these may be leaking the answer into the training data. "
            "If these features are only known after the target is determined, remove them."
        )

    return {
        "leaky_features": leaky_features,
        "n_checked": n_checked,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "high_threshold": high_threshold,
        "moderate_threshold": moderate_threshold,
        "target_col": target_col,
        "summary": summary,
    }


def compute_sample_size_adequacy(
    n_rows: int,
    n_features: int,
    problem_type: str,
    n_classes: int = 2,
    cv_std: float | None = None,
) -> dict:
    """Assess whether the training dataset has enough rows for reliable modeling.

    Uses the 10× rule of thumb:
    - Regression: recommended = max(50, 10 * n_features)
    - Classification: recommended = max(50 * n_classes, 10 * n_features * n_classes)

    Verdicts:
    - adequate: n_rows >= recommended
    - borderline: n_rows >= 0.6 * recommended
    - insufficient: n_rows < 0.6 * recommended

    Args:
        n_rows: Number of training rows.
        n_features: Number of input features (after encoding).
        problem_type: "regression" or "classification".
        n_classes: Number of target classes (ignored for regression).
        cv_std: Cross-validation standard deviation from existing run (optional).

    Returns:
        Dict with verdict, recommended_n, shortfall, ratio, cv checks, and
        plain-English summary and recommendations.

    Raises:
        ValueError: If n_rows < 5 or n_features < 1.
    """
    if n_rows < 5:
        raise ValueError("Need at least 5 rows to assess sample size adequacy.")
    if n_features < 1:
        raise ValueError("Need at least 1 feature to assess sample size adequacy.")

    is_classification = problem_type == "classification"
    _n_classes = max(2, n_classes) if is_classification else 1

    if is_classification:
        recommended_n = max(50 * _n_classes, 10 * n_features * _n_classes)
    else:
        recommended_n = max(50, 10 * n_features)

    shortfall = max(0, recommended_n - n_rows)
    ratio = round(n_features / n_rows, 4)

    coverage = n_rows / recommended_n
    if coverage >= 1.0:
        verdict = "adequate"
        verdict_label = "Adequate"
    elif coverage >= 0.6:
        verdict = "borderline"
        verdict_label = "Borderline"
    else:
        verdict = "insufficient"
        verdict_label = "Insufficient"

    cv_stable: bool | None = None
    if cv_std is not None:
        cv_stable = cv_std <= 0.10

    recommendations: list[str] = []
    if verdict == "insufficient":
        recommendations.append(
            f"Collect at least {shortfall:,} more rows to reach the recommended {recommended_n:,}."
        )
        if is_classification:
            recommendations.append(
                "Focus on under-represented classes — imbalanced data amplifies small-sample problems."
            )
        else:
            recommendations.append(
                "With limited rows, prefer simpler models (Linear Regression, Ridge) "
                "to reduce overfitting risk."
            )
        if ratio > 0.1:
            recommendations.append(
                f"Your feature-to-sample ratio is {ratio:.2f} ({n_features} features / "
                f"{n_rows:,} rows) — consider dropping low-importance features to reduce noise."
            )
    elif verdict == "borderline":
        recommendations.append(
            f"You are {shortfall:,} rows short of the recommended {recommended_n:,}. "
            "Results may be noisy — cross-validation is especially important."
        )
        if cv_std is not None and not cv_stable:
            recommendations.append(
                f"Your CV standard deviation ({cv_std:.3f}) is high, confirming "
                "that more data would improve model stability."
            )
    else:
        recommendations.append(
            "Your dataset meets the size recommendation — training should be reliable."
        )
        if cv_std is not None and not cv_stable:
            recommendations.append(
                f"Cross-validation shows high variance (std={cv_std:.3f}) despite adequate size — "
                "consider feature engineering or a more robust algorithm."
            )

    if verdict == "adequate":
        summary = (
            f"Your {n_rows:,}-row dataset meets the recommended minimum of {recommended_n:,} rows "
            f"for a {problem_type} model with {n_features} features"
            + (f" and {_n_classes} classes" if is_classification else "")
            + ". Training should produce reliable results."
        )
    elif verdict == "borderline":
        summary = (
            f"Your {n_rows:,} rows are in the borderline range — the rule of thumb recommends "
            f"{recommended_n:,} rows for {n_features} features"
            + (f" and {_n_classes} classes" if is_classification else "")
            + ". Results may be noisier than ideal."
        )
    else:
        summary = (
            f"Your {n_rows:,} rows are below the recommended {recommended_n:,} for a "
            f"{problem_type} model with {n_features} features"
            + (f" and {_n_classes} classes" if is_classification else "")
            + f". You need approximately {shortfall:,} more rows for reliable modeling."
        )

    return {
        "n_rows": n_rows,
        "n_features": n_features,
        "n_classes": _n_classes if is_classification else None,
        "problem_type": problem_type,
        "recommended_n": recommended_n,
        "shortfall": shortfall,
        "coverage_pct": round(min(coverage, 1.0) * 100, 1),
        "ratio": ratio,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "cv_std": cv_std,
        "cv_stable": cv_stable,
        "summary": summary,
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------------
# Prediction output anomaly detection
# ---------------------------------------------------------------------------


def compute_prediction_output_anomalies(
    logs: list[dict],
    problem_type: str,
    z_score_threshold: float = 2.5,
    confidence_threshold: float = 0.55,
    max_anomalies: int = 10,
) -> dict:
    """Detect anomalous prediction output values in production logs.

    For regression: flags predictions whose z-score (distance from mean in std units)
    exceeds z_score_threshold. Helps spot extreme outlier predictions that may indicate
    unusual inputs or model extrapolation.

    For classification: flags predictions where the model's confidence (max predict_proba)
    is below confidence_threshold — indicating the model was uncertain about its output.

    Args:
        logs: List of dicts with keys: id, prediction_numeric, prediction, confidence,
              created_at, input_features (JSON string).
        problem_type: "regression" or "classification".
        z_score_threshold: For regression — min |z-score| to flag as anomalous.
        confidence_threshold: For classification — max confidence below which a prediction
              is flagged as anomalous.
        max_anomalies: Maximum number of anomalies to include in the result.

    Returns:
        Dict with n_total, n_anomalous, anomaly_rate, verdict, anomalies, stats, summary.

    Raises:
        ValueError: When fewer than 5 logs are provided.
    """
    if len(logs) < 5:
        raise ValueError("Need at least 5 prediction logs to detect output anomalies.")

    import json as _json

    anomalies: list[dict] = []
    stats: dict = {}

    if problem_type == "regression":
        values = [
            (
                log["id"],
                float(log["prediction_numeric"]),
                log["created_at"],
                log.get("input_features", "{}"),
            )
            for log in logs
            if log.get("prediction_numeric") is not None
        ]
        if len(values) < 5:
            raise ValueError(
                "Need at least 5 numeric predictions to detect output anomalies."
            )

        nums = np.array([v[1] for v in values])
        mean_val = float(np.mean(nums))
        std_val = float(np.std(nums))

        stats = {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "min": round(float(np.min(nums)), 4),
            "max": round(float(np.max(nums)), 4),
        }

        if std_val == 0:
            # All identical predictions — nothing is anomalous
            return {
                "n_total": len(values),
                "n_anomalous": 0,
                "anomaly_rate": 0.0,
                "verdict": "no_anomalies",
                "verdict_label": "No unusual predictions detected",
                "problem_type": problem_type,
                "stats": stats,
                "anomalies": [],
                "summary": f"All {len(values)} predictions are identical — the model output is perfectly consistent.",
            }

        for log_id, val, created_at, input_features in values:
            z = (val - mean_val) / std_val
            if abs(z) > z_score_threshold:
                direction = "above" if z > 0 else "below"
                try:
                    input_dict = _json.loads(input_features)
                    input_summary = {k: v for k, v in list(input_dict.items())[:3]}
                except Exception:  # noqa: BLE001
                    input_summary = {}
                anomalies.append(
                    {
                        "id": str(log_id)[:8],
                        "prediction_value": str(round(val, 4)),
                        "z_score": round(abs(z), 2),
                        "confidence": None,
                        "deviation": f"{abs(z):.1f}σ {direction} mean",
                        "reason": f"Prediction is {abs(z):.1f} standard deviations {direction} the typical value of {mean_val:.2f}",
                        "created_at": str(created_at),
                        "input_summary": input_summary,
                    }
                )

        # Sort by z-score descending
        anomalies.sort(key=lambda a: a["z_score"], reverse=True)

    else:
        # Classification: flag low-confidence predictions
        values = [
            (
                log["id"],
                log.get("confidence"),
                log["prediction"],
                log["created_at"],
                log.get("input_features", "{}"),
            )
            for log in logs
            if log.get("confidence") is not None
        ]
        if len(values) < 5:
            raise ValueError(
                "Need at least 5 predictions with confidence scores to detect output anomalies."
            )

        confs = np.array([float(v[1]) for v in values])
        mean_conf = float(np.mean(confs))
        stats = {
            "mean_confidence": round(mean_conf, 4),
            "min_confidence": round(float(np.min(confs)), 4),
            "max_confidence": round(float(np.max(confs)), 4),
        }

        for log_id, conf, pred, created_at, input_features in values:
            conf_f = float(conf)
            if conf_f < confidence_threshold:
                try:
                    input_dict = _json.loads(input_features)
                    input_summary = {k: v for k, v in list(input_dict.items())[:3]}
                except Exception:  # noqa: BLE001
                    input_summary = {}
                anomalies.append(
                    {
                        "id": str(log_id)[:8],
                        "prediction_value": str(pred),
                        "z_score": None,
                        "confidence": round(conf_f * 100, 1),
                        "deviation": f"{conf_f * 100:.0f}% confidence",
                        "reason": f"Model was only {conf_f * 100:.0f}% confident — below the {confidence_threshold * 100:.0f}% threshold for reliable predictions",
                        "created_at": str(created_at),
                        "input_summary": input_summary,
                    }
                )

        # Sort by confidence ascending (most uncertain first)
        anomalies.sort(key=lambda a: a["confidence"] or 0)

    anomalies = anomalies[:max_anomalies]
    n_total = len(logs)
    n_anomalous = len(anomalies)
    anomaly_rate = round(n_anomalous / n_total * 100, 1) if n_total > 0 else 0.0

    if n_anomalous == 0:
        verdict = "no_anomalies"
        verdict_label = "No unusual predictions detected"
    elif anomaly_rate < 10.0:
        verdict = "few_anomalies"
        verdict_label = "A few unusual predictions detected"
    else:
        verdict = "many_anomalies"
        verdict_label = "Several unusual predictions detected"

    if problem_type == "regression":
        if n_anomalous == 0:
            summary = f"All {n_total} recent predictions fall within the normal range (mean ± {z_score_threshold}σ)."
        else:
            summary = (
                f"{n_anomalous} of {n_total} recent predictions ({anomaly_rate}%) are unusually "
                f"extreme — more than {z_score_threshold} standard deviations from the typical value of {stats.get('mean', 0):.2f}."
            )
    else:
        if n_anomalous == 0:
            summary = f"All {n_total} recent predictions exceed the {confidence_threshold * 100:.0f}% confidence threshold."
        else:
            summary = (
                f"{n_anomalous} of {n_total} recent predictions ({anomaly_rate}%) have low model confidence "
                f"(below {confidence_threshold * 100:.0f}%) — the model was uncertain about these outputs."
            )

    return {
        "n_total": n_total,
        "n_anomalous": n_anomalous,
        "anomaly_rate": anomaly_rate,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "problem_type": problem_type,
        "stats": stats,
        "anomalies": anomalies,
        "summary": summary,
    }


def compute_prediction_output_distribution_shift(
    training_preds: list[float],
    production_preds: list[float],
    n_bins: int = 10,
) -> dict:
    """Compare the distribution of training-time predictions vs production predictions.

    Uses the Kolmogorov-Smirnov two-sample test to measure how differently the model
    is behaving in production compared to training. Works without labeled ground truth —
    only raw prediction values are needed.

    Distinct from:
    - DriftCard: compares *input* feature distributions (covariate drift)
    - compute_training_vs_production: requires labeled feedback to compare accuracy
    - compute_prediction_output_anomalies: finds individual anomalous production predictions

    Args:
        training_preds: Predictions from re-running the deployed model on training data.
        production_preds: Recent production prediction values from PredictionLog.
        n_bins: Number of histogram bins for visualization.

    Returns:
        Dict with verdict, KS statistics, per-distribution stats, aligned histograms,
        mean shift, and plain-English summary.

    Raises:
        ValueError: When fewer than 10 samples in either list.
    """
    if len(training_preds) < 10:
        raise ValueError(
            "Need at least 10 training predictions to compare distributions."
        )
    if len(production_preds) < 10:
        raise ValueError(
            "Need at least 10 production predictions to compare distributions."
        )

    from scipy.stats import ks_2samp

    train_arr = np.array(training_preds, dtype=float)
    prod_arr = np.array(production_preds, dtype=float)

    ks_stat, ks_pvalue = ks_2samp(train_arr, prod_arr)
    ks_stat = float(round(ks_stat, 4))
    ks_pvalue = float(round(ks_pvalue, 6))

    def _dist_stats(arr: np.ndarray) -> dict:
        return {
            "mean": float(round(np.mean(arr), 4)),
            "std": float(round(np.std(arr), 4)),
            "min": float(round(np.min(arr), 4)),
            "p25": float(round(np.percentile(arr, 25), 4)),
            "median": float(round(np.median(arr), 4)),
            "p75": float(round(np.percentile(arr, 75), 4)),
            "p95": float(round(np.percentile(arr, 95), 4)),
            "max": float(round(np.max(arr), 4)),
        }

    training_stats = _dist_stats(train_arr)
    production_stats = _dist_stats(prod_arr)

    mean_shift = float(round(production_stats["mean"] - training_stats["mean"], 4))
    mean_shift_pct = (
        float(round(mean_shift / abs(training_stats["mean"]) * 100, 1))
        if training_stats["mean"] != 0
        else 0.0
    )
    std_ratio = (
        float(round(production_stats["std"] / training_stats["std"], 3))
        if training_stats["std"] > 0
        else None
    )

    # Verdict: significant → KS p < 0.01 or |mean shift| > 30%
    #          moderate   → KS p < 0.05 or |mean shift| > 10%
    #          stable     → otherwise
    abs_shift = abs(mean_shift_pct)
    if ks_pvalue < 0.01 or abs_shift > 30:
        verdict = "significant_shift"
        verdict_label = "Significant distribution shift detected"
    elif ks_pvalue < 0.05 or abs_shift > 10:
        verdict = "moderate_shift"
        verdict_label = "Moderate distribution shift detected"
    else:
        verdict = "stable"
        verdict_label = "Distribution is stable"

    # Build aligned histograms using shared bin edges
    combined_min = float(min(train_arr.min(), prod_arr.min()))
    combined_max = float(max(train_arr.max(), prod_arr.max()))
    if combined_min == combined_max:
        combined_max = combined_min + 1.0
    bin_edges = np.linspace(combined_min, combined_max, n_bins + 1)

    def _build_histogram(arr: np.ndarray) -> list[dict]:
        counts, _ = np.histogram(arr, bins=bin_edges)
        bins = []
        for i, count in enumerate(counts):
            bins.append(
                {
                    "bin_start": float(round(bin_edges[i], 4)),
                    "bin_end": float(round(bin_edges[i + 1], 4)),
                    "count": int(count),
                    "label": f"{bin_edges[i]:.2f}–{bin_edges[i + 1]:.2f}",
                }
            )
        return bins

    training_histogram = _build_histogram(train_arr)
    production_histogram = _build_histogram(prod_arr)

    # Plain-English summary
    direction_word = "above" if mean_shift > 0 else "below"
    if verdict == "stable":
        summary = (
            f"The model's output distribution is stable. Production predictions "
            f"(n={len(production_preds)}) closely match the training-time distribution "
            f"(KS p={ks_pvalue:.3f}, mean shift {abs_shift:.1f}%)."
        )
    elif verdict == "moderate_shift":
        summary = (
            f"Moderate distribution shift detected. Production predictions are {abs_shift:.1f}% "
            f"{direction_word} the training baseline on average "
            f"(KS statistic {ks_stat:.3f}, p={ks_pvalue:.3f}). "
            f"Monitor closely — consider retraining if the trend continues."
        )
    else:
        summary = (
            f"Significant distribution shift detected. Production predictions are {abs_shift:.1f}% "
            f"{direction_word} the training baseline on average "
            f"(KS statistic {ks_stat:.3f}, p={ks_pvalue:.4f}). "
            f"The model is behaving very differently in production than during training — "
            f"retraining with recent data is strongly recommended."
        )

    return {
        "n_training": len(training_preds),
        "n_production": len(production_preds),
        "verdict": verdict,
        "verdict_label": verdict_label,
        "ks_statistic": ks_stat,
        "ks_p_value": ks_pvalue,
        "mean_shift": mean_shift,
        "mean_shift_pct": mean_shift_pct,
        "std_ratio": std_ratio,
        "training_stats": training_stats,
        "production_stats": production_stats,
        "training_histogram": training_histogram,
        "production_histogram": production_histogram,
        "summary": summary,
    }


def _psi_for_numeric(
    training_values: list[float],
    production_values: list[float],
    n_bins: int,
) -> float:
    """Compute PSI between two numeric value lists using equal-frequency training bins."""
    import numpy as _np

    train_arr = _np.array(training_values, dtype=float)
    prod_arr = _np.array(production_values, dtype=float)

    # Cap bins to ensure at least 10 production samples per bin (prevents spurious drift)
    safe_bins = max(2, min(n_bins, len(prod_arr) // 10))

    # Build bin edges from training percentiles (equal-frequency bins)
    percentiles = _np.linspace(0, 100, safe_bins + 1)
    bin_edges = _np.unique(_np.percentile(train_arr, percentiles))
    if len(bin_edges) < 2:
        return 0.0  # all identical values — no drift measurable

    # Extend edges to catch all production values
    bin_edges[0] = -_np.inf
    bin_edges[-1] = _np.inf

    train_counts, _ = _np.histogram(train_arr, bins=bin_edges)
    prod_counts, _ = _np.histogram(prod_arr, bins=bin_edges)

    n_train = len(train_arr)
    n_prod = len(prod_arr)
    if n_train == 0 or n_prod == 0:
        return 0.0

    eps = 1e-4  # avoid log(0)
    train_pct = _np.clip(train_counts / n_train, eps, None)
    prod_pct = _np.clip(prod_counts / n_prod, eps, None)

    psi = float(_np.sum((prod_pct - train_pct) * _np.log(prod_pct / train_pct)))
    return round(max(0.0, psi), 4)


def _psi_for_categorical(
    training_values: list,
    production_values: list,
) -> tuple[float, int, int]:
    """Compute PSI for a categorical feature. Returns (psi, n_new_cats, n_dropped_cats)."""
    import numpy as _np

    train_str = [str(v) for v in training_values]
    prod_str = [str(v) for v in production_values]
    n_train = len(train_str)
    n_prod = len(prod_str)
    if n_train == 0 or n_prod == 0:
        return 0.0, 0, 0

    # Gather all unique categories
    train_cats = set(train_str)
    prod_cats = set(prod_str)
    all_cats = train_cats | prod_cats
    n_new = len(prod_cats - train_cats)
    n_dropped = len(train_cats - prod_cats)

    eps = 1e-4
    psi = 0.0
    for cat in all_cats:
        t_pct = max(train_str.count(cat) / n_train, eps)
        p_pct = max(prod_str.count(cat) / n_prod, eps)
        psi += (p_pct - t_pct) * _np.log(p_pct / t_pct)

    return round(max(0.0, float(psi)), 4), n_new, n_dropped


def compute_feature_psi_ranking(
    training_df,
    production_inputs: list[dict],
    feature_names: list[str] | None = None,
    *,
    n_bins: int = 10,
    max_features: int = 15,
) -> dict:
    """Rank all input features by their PSI (Population Stability Index) between
    training data and recent production predictions.

    PSI measures distributional shift per feature:
    - PSI < 0.10: Stable (no significant change)
    - 0.10 ≤ PSI < 0.20: Watch (minor change — monitor)
    - PSI ≥ 0.20: Critical (major change — investigate)

    Args:
        training_df: pandas DataFrame of training data (features only, no target).
        production_inputs: list of dicts from PredictionLog.input_features.
        feature_names: columns to analyse; defaults to all columns in training_df.
        n_bins: number of equal-frequency bins for numeric PSI (default 10).
        max_features: cap on features analysed (default 15).

    Returns:
        Dict with features (sorted by PSI desc), counts, overall_psi, verdict, summary.

    Raises:
        ValueError: if training_df has < 10 rows or production_inputs is empty.
    """
    if training_df is None or len(training_df) < 10:
        raise ValueError("Training dataset must have at least 10 rows.")
    if not production_inputs:
        raise ValueError("production_inputs must not be empty.")

    cols = list(feature_names or training_df.columns)[:max_features]

    features_result = []
    for col in cols:
        if col not in training_df.columns:
            continue

        train_vals = training_df[col].dropna().tolist()
        prod_vals = [
            inp[col] for inp in production_inputs if col in inp and inp[col] is not None
        ]
        if len(train_vals) < 5 or len(prod_vals) < 5:
            continue

        # Determine feature type
        numeric_count = 0
        for v in prod_vals:
            try:
                float(v)
                numeric_count += 1
            except (TypeError, ValueError):
                pass
        is_numeric = (numeric_count / len(prod_vals)) > 0.5

        if is_numeric:
            try:
                t_floats = [float(v) for v in train_vals]
                p_floats = [float(v) for v in prod_vals]
            except (TypeError, ValueError):
                is_numeric = False

        if is_numeric:
            psi = _psi_for_numeric(t_floats, p_floats, n_bins)
            feature_entry = {
                "feature": col,
                "feature_type": "numeric",
                "psi": psi,
                "n_new_categories": None,
                "n_dropped_categories": None,
            }
        else:
            psi, n_new, n_dropped = _psi_for_categorical(train_vals, prod_vals)
            feature_entry = {
                "feature": col,
                "feature_type": "categorical",
                "psi": psi,
                "n_new_categories": n_new,
                "n_dropped_categories": n_dropped,
            }

        if psi >= 0.20:
            feature_entry["severity"] = "critical"
            feature_entry["psi_label"] = "Critical"
        elif psi >= 0.10:
            feature_entry["severity"] = "watch"
            feature_entry["psi_label"] = "Watch"
        else:
            feature_entry["severity"] = "stable"
            feature_entry["psi_label"] = "Stable"

        features_result.append(feature_entry)

    # Sort by PSI descending
    features_result.sort(key=lambda x: x["psi"], reverse=True)

    stable_count = sum(1 for f in features_result if f["severity"] == "stable")
    watch_count = sum(1 for f in features_result if f["severity"] == "watch")
    critical_count = sum(1 for f in features_result if f["severity"] == "critical")
    features_analyzed = len(features_result)

    overall_psi = (
        round(sum(f["psi"] for f in features_result) / features_analyzed, 4)
        if features_analyzed > 0
        else 0.0
    )

    if critical_count > 0:
        verdict = "critical"
        verdict_label = f"{critical_count} feature{'s' if critical_count > 1 else ''} with major shift"
    elif watch_count > 0:
        verdict = "watch"
        verdict_label = (
            f"{watch_count} feature{'s' if watch_count > 1 else ''} with minor shift"
        )
    else:
        verdict = "stable"
        verdict_label = "All features stable"

    top = features_result[0] if features_result else None
    if critical_count > 0 and top:
        summary = (
            f"{critical_count} of {features_analyzed} input feature(s) show major distribution "
            f"shift (PSI ≥ 0.20). The most drifted feature is '{top['feature']}' "
            f"(PSI = {top['psi']:.3f}). These features should be investigated — "
            f"data pipeline changes or concept drift may be responsible."
        )
    elif watch_count > 0 and top:
        top_watch = next((f for f in features_result if f["severity"] != "stable"), top)
        summary = (
            f"{watch_count} of {features_analyzed} input feature(s) show minor distribution "
            f"shift (PSI 0.10–0.20). The most shifted is '{top_watch['feature']}' "
            f"(PSI = {top_watch['psi']:.3f}). Monitor these features for continued change."
        )
    else:
        summary = (
            f"All {features_analyzed} analyzed feature(s) are stable "
            f"(PSI < 0.10). Production inputs closely match the training distribution."
        )

    return {
        "features": features_result,
        "features_analyzed": features_analyzed,
        "stable_count": stable_count,
        "watch_count": watch_count,
        "critical_count": critical_count,
        "overall_psi": overall_psi,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "sample_count": len(production_inputs),
        "training_count": len(training_df),
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Retraining Readiness
# ---------------------------------------------------------------------------

_RETRAIN_VERDICTS = {
    "stable": "No action needed",
    "monitor": "Worth watching",
    "retrain_soon": "Retrain recommended",
    "retrain_now": "Retrain urgently",
}


def compute_retraining_readiness(  # noqa: PLR0912, PLR0915
    age_days: int | None = None,
    anomaly_rate: float | None = None,
    confidence_trend: str | None = None,
    feedback_verdict: str | None = None,
    psi_critical_count: int | None = None,
    output_shift_verdict: str | None = None,
) -> dict:
    """Aggregate monitoring signals into a 0-100 retraining urgency score.

    Higher score means more evidence that the model should be retrained.
    Each signal contributes a weighted sub-score; available signals are
    normalized so the total always ranges 0-100.

    Returns a dict with: score, verdict, verdict_label, signals,
    signals_available, signals_firing, top_reason, recommendations, summary.
    """
    signals = []

    # --- model age ---
    age_contribution = 0
    if age_days is not None:
        if age_days > 180:
            age_contribution = 40
            age_detail = f"Trained {age_days} days ago (very old)"
        elif age_days > 90:
            age_contribution = 25
            age_detail = f"Trained {age_days} days ago (aging)"
        elif age_days > 60:
            age_contribution = 15
            age_detail = f"Trained {age_days} days ago (getting old)"
        elif age_days > 30:
            age_contribution = 5
            age_detail = f"Trained {age_days} days ago (acceptable)"
        else:
            age_contribution = 0
            age_detail = f"Trained {age_days} days ago (fresh)"
        signals.append(
            {
                "name": "model_age",
                "label": "Model Age",
                "value_str": f"{age_days} days",
                "score_contribution": age_contribution,
                "is_firing": age_contribution > 0,
                "detail": age_detail,
            }
        )

    # --- prediction anomaly rate ---
    anomaly_contribution = 0
    if anomaly_rate is not None:
        if anomaly_rate >= 0.15:
            anomaly_contribution = 30
            anom_detail = f"Anomaly rate {anomaly_rate:.0%} (high)"
        elif anomaly_rate >= 0.05:
            anomaly_contribution = 15
            anom_detail = f"Anomaly rate {anomaly_rate:.0%} (elevated)"
        else:
            anomaly_contribution = 0
            anom_detail = f"Anomaly rate {anomaly_rate:.0%} (normal)"
        signals.append(
            {
                "name": "anomaly_rate",
                "label": "Prediction Anomalies",
                "value_str": f"{anomaly_rate:.0%} anomalous",
                "score_contribution": anomaly_contribution,
                "is_firing": anomaly_contribution > 0,
                "detail": anom_detail,
            }
        )

    # --- confidence trend ---
    conf_contribution = 0
    if confidence_trend is not None:
        if confidence_trend == "declining":
            conf_contribution = 20
            conf_detail = "Model confidence is declining over time"
        else:
            conf_contribution = 0
            conf_detail = f"Confidence trend: {confidence_trend}"
        signals.append(
            {
                "name": "confidence_trend",
                "label": "Confidence Trend",
                "value_str": confidence_trend.replace("_", " ").capitalize(),
                "score_contribution": conf_contribution,
                "is_firing": conf_contribution > 0,
                "detail": conf_detail,
            }
        )

    # --- feedback accuracy ---
    fb_contribution = 0
    if feedback_verdict is not None:
        if feedback_verdict == "poor":
            fb_contribution = 30
            fb_detail = "Real-world accuracy is poor"
        elif feedback_verdict == "moderate":
            fb_contribution = 15
            fb_detail = "Real-world accuracy is moderate"
        else:
            fb_contribution = 0
            fb_detail = f"Feedback verdict: {feedback_verdict}"
        signals.append(
            {
                "name": "feedback_accuracy",
                "label": "Feedback Accuracy",
                "value_str": feedback_verdict.replace("_", " ").capitalize(),
                "score_contribution": fb_contribution,
                "is_firing": fb_contribution > 0,
                "detail": fb_detail,
            }
        )

    # --- PSI critical features ---
    psi_contribution = 0
    if psi_critical_count is not None:
        if psi_critical_count >= 2:
            psi_contribution = 25
            psi_detail = f"{psi_critical_count} input features show critical distribution shift (PSI ≥ 0.20)"
        elif psi_critical_count == 1:
            psi_contribution = 15
            psi_detail = (
                "1 input feature shows critical distribution shift (PSI ≥ 0.20)"
            )
        else:
            psi_contribution = 0
            psi_detail = "No input features show critical distribution shift"
        signals.append(
            {
                "name": "psi_drift",
                "label": "Input Feature Drift (PSI)",
                "value_str": f"{psi_critical_count} critical feature(s)",
                "score_contribution": psi_contribution,
                "is_firing": psi_contribution > 0,
                "detail": psi_detail,
            }
        )

    # --- output distribution shift ---
    shift_contribution = 0
    if output_shift_verdict is not None and output_shift_verdict != "no_data":
        if output_shift_verdict == "significant_shift":
            shift_contribution = 25
            shift_detail = (
                "Production prediction distribution significantly differs from training"
            )
        elif output_shift_verdict == "moderate_shift":
            shift_contribution = 10
            shift_detail = (
                "Production prediction distribution moderately differs from training"
            )
        else:
            shift_contribution = 0
            shift_detail = "Prediction output distribution is stable"
        signals.append(
            {
                "name": "output_shift",
                "label": "Output Distribution Shift",
                "value_str": output_shift_verdict.replace("_", " ").capitalize(),
                "score_contribution": shift_contribution,
                "is_firing": shift_contribution > 0,
                "detail": shift_detail,
            }
        )

    if not signals:
        return {
            "score": 0,
            "verdict": "stable",
            "verdict_label": _RETRAIN_VERDICTS["stable"],
            "signals": [],
            "signals_available": 0,
            "signals_firing": 0,
            "top_reason": None,
            "recommendations": [
                "Collect more production data to assess retraining need."
            ],
            "summary": "No monitoring signals available yet. Run predictions and submit feedback to build a picture of model health.",
        }

    # --- aggregate: normalize each contribution by the max possible from its signal ---
    max_by_signal = {
        "model_age": 40,
        "anomaly_rate": 30,
        "confidence_trend": 20,
        "feedback_accuracy": 30,
        "psi_drift": 25,
        "output_shift": 25,
    }
    total_possible = sum(max_by_signal[s["name"]] for s in signals)
    raw_total = sum(s["score_contribution"] for s in signals)
    score = int(round(raw_total / total_possible * 100)) if total_possible > 0 else 0
    score = min(100, max(0, score))

    # verdict
    if score >= 80:
        verdict = "retrain_now"
    elif score >= 60:
        verdict = "retrain_soon"
    elif score >= 30:
        verdict = "monitor"
    else:
        verdict = "stable"

    # top reason = highest-contributing firing signal
    firing = sorted(
        [s for s in signals if s["is_firing"]],
        key=lambda s: s["score_contribution"],
        reverse=True,
    )
    top_reason = firing[0]["detail"] if firing else None

    # recommendations
    recs: list[str] = []
    if verdict in ("retrain_now", "retrain_soon"):
        recs.append("Retrain the model on fresh data to restore accuracy.")
    if any(s["name"] == "psi_drift" and s["is_firing"] for s in signals):
        recs.append(
            "Investigate input feature distribution changes — your data pipeline may have shifted."
        )
    if any(s["name"] == "feedback_accuracy" and s["is_firing"] for s in signals):
        recs.append(
            "Review recent predictions where feedback was submitted to identify failure patterns."
        )
    if any(s["name"] == "model_age" and s["is_firing"] for s in signals):
        recs.append("Upload a fresh dataset and retrain to reflect current patterns.")
    if verdict == "monitor":
        recs.append("Continue monitoring — no immediate action needed.")
    if verdict == "stable":
        recs.append(
            "Model is performing well. Keep collecting feedback to maintain visibility."
        )

    # summary
    available = len(signals)
    firing_count = len(firing)
    if verdict == "stable":
        summary = (
            f"All {available} available monitoring signal(s) are within normal range. "
            "No retraining action is needed at this time."
        )
    elif verdict == "monitor":
        summary = (
            f"{firing_count} of {available} signal(s) are elevated. "
            f"Keep an eye on this — {top_reason.lower() if top_reason else 'trends may worsen'}."
        )
    elif verdict == "retrain_soon":
        summary = (
            f"{firing_count} of {available} signal(s) indicate model degradation. "
            f"Retraining is recommended. Key concern: {top_reason.lower() if top_reason else 'model quality has declined'}."
        )
    else:
        summary = (
            f"{firing_count} of {available} signal(s) point to significant model degradation. "
            f"Retrain urgently. Main concern: {top_reason.lower() if top_reason else 'multiple quality indicators are alarming'}."
        )

    return {
        "score": score,
        "verdict": verdict,
        "verdict_label": _RETRAIN_VERDICTS[verdict],
        "signals": signals,
        "signals_available": available,
        "signals_firing": firing_count,
        "top_reason": top_reason,
        "recommendations": recs,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Production prediction value trend
# ---------------------------------------------------------------------------


def compute_prediction_value_trend(
    logs_data: list[dict],
    period: str = "day",
    n_periods: int = 14,
) -> dict:
    """Compute how mean regression prediction values have trended over time.

    Parameters
    ----------
    logs_data:
        List of dicts with keys ``prediction_numeric`` (float | None) and
        ``created_at`` (datetime or ISO-format string).
    period:
        Grouping granularity — "day" (default) or "week".
    n_periods:
        Maximum number of recent periods to include (default 14).

    Returns
    -------
    Dict with keys: periods (list), n_total, n_periods_with_data, direction,
    direction_label, slope, slope_pct_per_period, first_period_mean,
    last_period_mean, overall_change_pct, summary.

    Raises
    ------
    ValueError
        If fewer than 2 periods have prediction data.
    """
    # Filter to numeric regression predictions only
    valid: list[tuple[datetime, float]] = []
    for row in logs_data:
        val = row.get("prediction_numeric")
        ts = row.get("created_at")
        if val is None or ts is None:
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                continue
        # Normalise to naive UTC for consistent comparison
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            ts = ts.replace(tzinfo=None)
        valid.append((ts, v))

    if not valid:
        raise ValueError("No numeric regression predictions found in logs.")

    valid.sort(key=lambda t: t[0])

    # Floor timestamps to the requested period
    def _floor(ts: datetime, p: str) -> datetime:
        if p == "week":
            # Monday of the ISO week
            return ts - pd.Timedelta(days=ts.weekday())
        return datetime(ts.year, ts.month, ts.day)

    from collections import defaultdict as _defaultdict

    buckets: dict[datetime, list[float]] = _defaultdict(list)
    for ts, val in valid:
        buckets[_floor(ts, period)].append(val)

    # Keep only the n_periods most-recent periods
    sorted_keys = sorted(buckets.keys())[-n_periods:]
    if len(sorted_keys) < 2:
        raise ValueError(
            f"Need at least 2 periods with data; only {len(sorted_keys)} found."
        )

    periods_out: list[dict] = []
    for pk in sorted_keys:
        vals = buckets[pk]
        period_mean = float(np.mean(vals))
        periods_out.append(
            {
                "period_label": (
                    pk.strftime("%b %d")
                    if period == "day"
                    else pk.strftime("Week of %b %d")
                ),
                "period_start": pk.isoformat(),
                "mean": round(period_mean, 4),
                "count": len(vals),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }
        )

    means = np.array([p["mean"] for p in periods_out], dtype=float)
    x = np.arange(len(means), dtype=float)

    coeffs = np.polyfit(x, means, 1)
    slope = float(coeffs[0])

    first_mean = float(means[0])
    last_mean = float(means[-1])
    overall_change_pct = (
        ((last_mean - first_mean) / abs(first_mean)) * 100
        if abs(first_mean) > 1e-9
        else 0.0
    )
    slope_pct_per_period = (
        (slope / abs(first_mean)) * 100 if abs(first_mean) > 1e-9 else 0.0
    )

    if overall_change_pct > 5.0:
        direction = "trending_up"
        direction_label = "Trending Up"
    elif overall_change_pct < -5.0:
        direction = "trending_down"
        direction_label = "Trending Down"
    else:
        direction = "stable"
        direction_label = "Stable"

    n_total = sum(p["count"] for p in periods_out)
    n_periods_with_data = len(periods_out)

    change_str = (
        f"{overall_change_pct:+.1f}%"
        if abs(overall_change_pct) >= 0.1
        else "no net change"
    )
    if direction == "trending_up":
        summary = (
            f"Prediction values have increased {change_str} over the last "
            f"{n_periods_with_data} {period}(s) ({n_total} predictions). "
            "Your model is consistently producing higher outputs — verify this reflects real-world change."
        )
    elif direction == "trending_down":
        summary = (
            f"Prediction values have decreased {change_str} over the last "
            f"{n_periods_with_data} {period}(s) ({n_total} predictions). "
            "Your model is producing lower outputs over time — consider whether this is expected or a sign of drift."
        )
    else:
        summary = (
            f"Prediction values are stable ({change_str}) over the last "
            f"{n_periods_with_data} {period}(s) ({n_total} predictions). "
            "No systematic upward or downward trend detected."
        )

    return {
        "periods": periods_out,
        "n_total": n_total,
        "n_periods_with_data": n_periods_with_data,
        "direction": direction,
        "direction_label": direction_label,
        "slope": round(slope, 6),
        "slope_pct_per_period": round(slope_pct_per_period, 4),
        "first_period_mean": round(first_mean, 4),
        "last_period_mean": round(last_mean, 4),
        "overall_change_pct": round(overall_change_pct, 2),
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Deployment monitoring signal digest
# ---------------------------------------------------------------------------


def compute_deployment_monitoring_digest(
    anomaly_result: dict | None = None,
    value_trend_result: dict | None = None,
    dist_shift_result: dict | None = None,
    readiness_result: dict | None = None,
    usage_7d: int = 0,
    usage_prev_7d: int = 0,
    problem_type: str = "regression",
) -> dict:
    """Aggregate all available monitoring signal verdicts into a single digest.

    Parameters
    ----------
    anomaly_result:
        Output from compute_prediction_output_anomalies (or None if unavailable).
    value_trend_result:
        Output from compute_prediction_value_trend (regression only, or None).
    dist_shift_result:
        Output from compute_prediction_output_distribution_shift (or None).
    readiness_result:
        Output from compute_retraining_readiness (or None).
    usage_7d:
        Prediction count for the last 7 days.
    usage_prev_7d:
        Prediction count for the prior 7 days (for trend comparison).
    problem_type:
        "regression" or "classification".

    Returns
    -------
    Dict with keys: signals (list), overall_health, overall_score, signals_total,
    signals_firing, priority_actions (list), summary.
    """
    signals: list[dict] = []

    # --- Signal 1: Output anomalies ---
    if anomaly_result is not None:
        _av = anomaly_result.get("verdict", "no_data")
        if _av == "no_anomalies":
            _asev, _avl = "green", "No anomalies"
            _afind = "All production predictions are within normal range."
        elif _av == "few_anomalies":
            _ar = anomaly_result.get("anomaly_rate", 0.0)
            _asev, _avl = "amber", "Some anomalies"
            _afind = (
                f"{_ar:.0%} of predictions are statistical outliers — worth monitoring."
            )
        elif _av == "many_anomalies":
            _ar = anomaly_result.get("anomaly_rate", 0.0)
            _asev, _avl = "red", "Many anomalies"
            _afind = (
                f"{_ar:.0%} of predictions are anomalous — investigate input patterns."
            )
        else:
            _asev, _avl = "gray", "No data"
            _afind = "Not enough prediction data to assess anomalies."
        signals.append(
            {
                "signal_key": "output_anomalies",
                "signal_label": "Output Anomalies",
                "verdict": _av,
                "verdict_label": _avl,
                "severity": _asev,
                "finding": _afind,
                "icon": "🔍",
                "is_available": _av != "no_data",
            }
        )

    # --- Signal 2: Prediction value trend (regression only) ---
    if problem_type == "regression":
        if value_trend_result is not None:
            _tv = value_trend_result.get("direction", "stable")
            _tc = value_trend_result.get("overall_change_pct", 0.0)
            if _tv == "stable":
                _tsev, _tvl = "green", "Stable trend"
                _tfind = f"Prediction values are stable ({_tc:+.1f}% net change)."
            elif _tv == "trending_up":
                _tsev, _tvl = "amber", "Trending up"
                _tfind = f"Prediction values increased {_tc:+.1f}% — verify this reflects real-world change."
            else:
                _tsev, _tvl = "amber", "Trending down"
                _tfind = f"Prediction values decreased {_tc:+.1f}% — may indicate model drift."
            signals.append(
                {
                    "signal_key": "value_trend",
                    "signal_label": "Prediction Value Trend",
                    "verdict": _tv,
                    "verdict_label": _tvl,
                    "severity": _tsev,
                    "finding": _tfind,
                    "icon": "📈",
                    "is_available": True,
                }
            )
        else:
            signals.append(
                {
                    "signal_key": "value_trend",
                    "signal_label": "Prediction Value Trend",
                    "verdict": "no_data",
                    "verdict_label": "No data",
                    "severity": "gray",
                    "finding": "Not enough predictions to compute trend.",
                    "icon": "📈",
                    "is_available": False,
                }
            )

    # --- Signal 3: Output distribution shift ---
    if dist_shift_result is not None:
        _dv = dist_shift_result.get("verdict", "no_data")
        if _dv == "stable":
            _dsev, _dvl = "green", "Stable"
            _dfind = "Output distribution matches training — no shift detected."
        elif _dv == "moderate_shift":
            _ds = dist_shift_result.get("mean_shift_pct", 0.0)
            _dsev, _dvl = "amber", "Moderate shift"
            _dfind = f"Prediction distribution shifted {_ds:+.1f}% from training — monitor closely."
        elif _dv == "significant_shift":
            _ds = dist_shift_result.get("mean_shift_pct", 0.0)
            _dsev, _dvl = "red", "Significant shift"
            _dfind = f"Major output distribution shift ({_ds:+.1f}%) — model behaviour has changed."
        else:
            _dsev, _dvl = "gray", "No data"
            _dfind = "Not enough production predictions to assess distribution shift."
        signals.append(
            {
                "signal_key": "dist_shift",
                "signal_label": "Output Distribution",
                "verdict": _dv,
                "verdict_label": _dvl,
                "severity": _dsev,
                "finding": _dfind,
                "icon": "📊",
                "is_available": _dv != "no_data",
            }
        )

    # --- Signal 4: Retraining readiness ---
    if readiness_result is not None:
        _rv = readiness_result.get("verdict", "stable")
        _rs = readiness_result.get("score", 0)
        if _rv == "stable":
            _rsev, _rvl = "green", "No action needed"
            _rfind = f"Retraining score {_rs}/100 — model is healthy."
        elif _rv == "monitor":
            _rsev, _rvl = "amber", "Monitor signals"
            _rfind = f"Retraining score {_rs}/100 — some signals warrant attention."
        elif _rv == "retrain_soon":
            _rsev, _rvl = "amber", "Retrain soon"
            _rfind = (
                f"Retraining score {_rs}/100 — consider retraining in the coming days."
            )
        else:
            _rsev, _rvl = "red", "Retrain now"
            _rfind = (
                f"Retraining score {_rs}/100 — multiple signals indicate degradation."
            )
        signals.append(
            {
                "signal_key": "retraining_readiness",
                "signal_label": "Retraining Readiness",
                "verdict": _rv,
                "verdict_label": _rvl,
                "severity": _rsev,
                "finding": _rfind,
                "icon": "🔄",
                "is_available": True,
            }
        )

    # --- Signal 5: Usage activity ---
    if usage_7d > 0 and usage_prev_7d > 0:
        _usage_ratio = usage_7d / usage_prev_7d
        if _usage_ratio >= 0.8:
            _usev, _uvl = "green", "Active"
            _ufind = f"{usage_7d} predictions this week (vs {usage_prev_7d} last week) — normal activity."
        elif _usage_ratio >= 0.5:
            _usev, _uvl = "amber", "Declining usage"
            _ufind = f"{usage_7d} predictions this week vs {usage_prev_7d} last week — usage dropped {(1 - _usage_ratio):.0%}."
        else:
            _usev, _uvl = "amber", "Low usage"
            _ufind = f"Only {usage_7d} predictions this week vs {usage_prev_7d} last week — significant drop."
    elif usage_7d > 0:
        _usev, _uvl = "green", "Active"
        _ufind = f"{usage_7d} predictions in the last 7 days."
    else:
        _usev, _uvl = "amber", "No recent activity"
        _ufind = (
            "No predictions in the last 7 days — verify the endpoint is being called."
        )
    signals.append(
        {
            "signal_key": "usage_activity",
            "signal_label": "Usage Activity",
            "verdict": "active" if _usev == "green" else "low",
            "verdict_label": _uvl,
            "severity": _usev,
            "finding": _ufind,
            "icon": "📡",
            "is_available": True,
        }
    )

    # --- Aggregate health ---
    red_count = sum(1 for s in signals if s["severity"] == "red")
    amber_count = sum(1 for s in signals if s["severity"] == "amber")

    if red_count >= 2:
        overall_health = "critical"
    elif red_count == 1:
        overall_health = "warning"
    elif amber_count >= 1:
        overall_health = "watching"
    else:
        overall_health = "healthy"

    overall_score = max(0, 100 - red_count * 25 - amber_count * 10)
    signals_total = sum(1 for s in signals if s["is_available"])
    signals_firing = amber_count + red_count

    # --- Priority actions ---
    priority_actions: list[str] = []
    if readiness_result and readiness_result.get("verdict") in (
        "retrain_now",
        "retrain_soon",
    ):
        recs = readiness_result.get("recommendations", [])
        for rec in recs[:2]:
            if rec not in priority_actions:
                priority_actions.append(rec)
    if any(s["signal_key"] == "dist_shift" and s["severity"] == "red" for s in signals):
        priority_actions.append(
            "Investigate significant output distribution shift — run 'output distribution shift' for details."
        )
    if any(
        s["signal_key"] == "output_anomalies" and s["severity"] == "red"
        for s in signals
    ):
        priority_actions.append(
            "Review anomalous production predictions — run 'show prediction anomalies' for details."
        )
    if (
        any(
            s["signal_key"] == "dist_shift" and s["severity"] == "amber"
            for s in signals
        )
        and len(priority_actions) < 3
    ):
        priority_actions.append(
            "Monitor output distribution — moderate shift detected since training."
        )
    if (
        any(
            s["signal_key"] == "usage_activity" and s["severity"] == "amber"
            for s in signals
        )
        and len(priority_actions) < 3
    ):
        priority_actions.append(
            "Check API endpoint availability — prediction volume has dropped."
        )
    priority_actions = priority_actions[:3]

    # --- Summary ---
    available_names = [s["signal_label"] for s in signals if s["is_available"]]
    if overall_health == "healthy":
        summary = (
            f"All {signals_total} monitoring signals are healthy — no action needed. "
            f"Signals checked: {', '.join(available_names)}."
        )
    elif overall_health in ("warning", "critical"):
        red_signals = [s["signal_label"] for s in signals if s["severity"] == "red"]
        summary = (
            f"{red_count} signal{'s' if red_count > 1 else ''} need{'s' if red_count == 1 else ''} "
            f"attention: {', '.join(red_signals)}. "
            + (
                f"Additional {amber_count} signal{'s' if amber_count > 1 else ''} "
                f"{'are' if amber_count > 1 else 'is'} worth watching."
                if amber_count > 0
                else "Address these before sharing results with stakeholders."
            )
        )
    else:
        amber_signals = [s["signal_label"] for s in signals if s["severity"] == "amber"]
        summary = (
            f"{amber_count} signal{'s' if amber_count > 1 else ''} "
            f"{'warrant' if amber_count > 1 else 'warrants'} monitoring: "
            f"{', '.join(amber_signals)}. No immediate action required."
        )

    return {
        "signals": signals,
        "overall_health": overall_health,
        "overall_score": overall_score,
        "signals_total": signals_total,
        "signals_firing": signals_firing,
        "priority_actions": priority_actions,
        "summary": summary,
        "problem_type": problem_type,
    }


# ---------------------------------------------------------------------------
# Deployment prediction distribution comparison
# ---------------------------------------------------------------------------


def compute_deployment_prediction_comparison(
    baseline_logs: list[dict],
    current_logs: list[dict],
    problem_type: str,
) -> dict:
    """Compare production prediction distributions across two deployments.

    Answers "is my retrained model predicting higher values in production than
    my previous deployment?"

    For regression: compares mean/median/std/min/max prediction_numeric values,
    computes mean shift and direction verdict.
    For classification: compares class frequency distributions, identifies
    which class changed most.

    Args:
        baseline_logs: PredictionLog dicts from the older deployment.
        current_logs:  PredictionLog dicts from the newer deployment.
        problem_type:  "regression" or "classification".

    Returns:
        Dict with baseline_stats, current_stats, verdict, mean_shift_pct (regression),
        class_shifts (classification), n_baseline, n_current, and plain-English summary.

    Raises:
        ValueError: When fewer than 5 samples in either log list.
    """
    if len(baseline_logs) < 5:
        raise ValueError(
            f"Need at least 5 baseline predictions to compare. Got {len(baseline_logs)}."
        )
    if len(current_logs) < 5:
        raise ValueError(
            f"Need at least 5 current predictions to compare. Got {len(current_logs)}."
        )

    n_baseline = len(baseline_logs)
    n_current = len(current_logs)

    if problem_type == "regression":
        baseline_vals = [
            float(lg["prediction_numeric"])
            for lg in baseline_logs
            if lg.get("prediction_numeric") is not None
        ]
        current_vals = [
            float(lg["prediction_numeric"])
            for lg in current_logs
            if lg.get("prediction_numeric") is not None
        ]

        if len(baseline_vals) < 5 or len(current_vals) < 5:
            raise ValueError(
                "Not enough numeric prediction values to compare distributions."
            )

        b_arr = np.array(baseline_vals, dtype=float)
        c_arr = np.array(current_vals, dtype=float)

        def _stats(arr: np.ndarray) -> dict:
            return {
                "mean": float(round(np.mean(arr), 4)),
                "median": float(round(np.median(arr), 4)),
                "std": float(round(np.std(arr), 4)),
                "min": float(round(np.min(arr), 4)),
                "max": float(round(np.max(arr), 4)),
                "p25": float(round(np.percentile(arr, 25), 4)),
                "p75": float(round(np.percentile(arr, 75), 4)),
                "n": len(arr),
            }

        baseline_stats = _stats(b_arr)
        current_stats = _stats(c_arr)

        b_mean = baseline_stats["mean"]
        c_mean = current_stats["mean"]
        mean_shift = float(round(c_mean - b_mean, 4))
        mean_shift_pct = (
            float(round(mean_shift / abs(b_mean) * 100, 1)) if b_mean != 0 else 0.0
        )

        if mean_shift_pct > 5:
            verdict = "current_higher"
            verdict_label = (
                f"New deployment predicts higher values (+{mean_shift_pct:.1f}%)"
            )
        elif mean_shift_pct < -5:
            verdict = "current_lower"
            verdict_label = (
                f"New deployment predicts lower values ({mean_shift_pct:.1f}%)"
            )
        else:
            verdict = "similar"
            verdict_label = "Similar prediction levels across deployments"

        summary = (
            f"Compared {n_baseline} predictions from the previous deployment to "
            f"{n_current} from the current one. "
            f"Previous mean: {b_mean:.4g}. Current mean: {c_mean:.4g}. "
            + (
                f"The new deployment is predicting {abs(mean_shift_pct):.1f}% "
                f"{'higher' if mean_shift_pct > 0 else 'lower'} on average."
                if verdict != "similar"
                else "Prediction levels are broadly similar between deployments."
            )
        )

        return {
            "problem_type": "regression",
            "verdict": verdict,
            "verdict_label": verdict_label,
            "mean_shift": mean_shift,
            "mean_shift_pct": mean_shift_pct,
            "baseline_stats": baseline_stats,
            "current_stats": current_stats,
            "n_baseline": n_baseline,
            "n_current": n_current,
            "summary": summary,
        }

    else:
        # Classification — compare class frequency distributions
        from collections import Counter

        def _parse_label(lg: dict) -> str:
            raw = lg.get("prediction", "")
            try:
                import json as _json

                return str(_json.loads(raw))
            except Exception:  # noqa: BLE001
                return str(raw)

        b_counts = Counter(_parse_label(lg) for lg in baseline_logs)
        c_counts = Counter(_parse_label(lg) for lg in current_logs)

        all_classes = sorted(set(b_counts) | set(c_counts))

        class_shifts = []
        for cls in all_classes:
            b_pct = round(b_counts.get(cls, 0) / n_baseline * 100, 1)
            c_pct = round(c_counts.get(cls, 0) / n_current * 100, 1)
            shift_pct = round(c_pct - b_pct, 1)
            class_shifts.append(
                {
                    "class_label": cls,
                    "baseline_pct": b_pct,
                    "current_pct": c_pct,
                    "shift_pct": shift_pct,
                }
            )

        # Sort by absolute shift descending
        class_shifts.sort(key=lambda x: abs(x["shift_pct"]), reverse=True)

        # Verdict: largest absolute shift
        max_shift = max((abs(s["shift_pct"]) for s in class_shifts), default=0.0)
        if max_shift >= 10:
            biggest = class_shifts[0]
            dir_word = "increased" if biggest["shift_pct"] > 0 else "decreased"
            verdict = "distribution_shifted"
            verdict_label = (
                f"Class '{biggest['class_label']}' {dir_word} "
                f"by {abs(biggest['shift_pct']):.1f} percentage points"
            )
        else:
            verdict = "similar"
            verdict_label = "Class distribution is similar across deployments"

        summary = (
            f"Compared class distributions across {n_baseline} baseline and "
            f"{n_current} current predictions. "
            + (
                f"Biggest shift: '{class_shifts[0]['class_label']}' "
                f"{'up' if class_shifts[0]['shift_pct'] > 0 else 'down'} "
                f"{abs(class_shifts[0]['shift_pct']):.1f}pp "
                f"({class_shifts[0]['baseline_pct']}% → {class_shifts[0]['current_pct']}%)."
                if class_shifts
                else "No class shifts detected."
            )
        )

        return {
            "problem_type": "classification",
            "verdict": verdict,
            "verdict_label": verdict_label,
            "class_shifts": class_shifts,
            "n_classes": len(all_classes),
            "n_baseline": n_baseline,
            "n_current": n_current,
            "summary": summary,
        }


# ---------------------------------------------------------------------------
# Deployment Comparison Scorecard
# ---------------------------------------------------------------------------


def _usage_score(request_count: int) -> int:
    """0-100 score based on log-scale prediction volume."""
    if request_count <= 0:
        return 0
    import math

    # 1 pred → 20, 10 → 40, 100 → 60, 1000 → 80, 10000+ → 100
    log_val = math.log10(max(request_count, 1))
    return min(100, int(log_val * 25))


def _sla_score(p95_ms: float | None) -> int | None:
    """0-100 score from p95 latency; None when no SLA data available."""
    if p95_ms is None:
        return None
    if p95_ms <= 100:
        return 100
    if p95_ms >= 2000:
        return 0
    # Linear decay 100ms → 2000ms
    return max(0, int(100 - (p95_ms - 100) * 100 / 1900))


def _freshness_score(age_days: int) -> int:
    """0-100 score based on deployment age."""
    if age_days < 7:
        return 100
    if age_days < 30:
        return 80
    if age_days < 60:
        return 60
    if age_days < 90:
        return 40
    if age_days < 180:
        return 20
    return 0


def compute_deployment_scorecard(entries: list[dict]) -> dict:
    """Rank project deployments by composite production performance score.

    Pure function — no database or filesystem access.

    Each entry must contain:
        deployment_id: str
        algorithm_plain: str
        target_column: str
        environment: str
        request_count: int
        feedback_accuracy: float | None   (0–1 scale from FeedbackRecord)
        p95_ms: float | None              (SLA p95 latency in ms)
        age_days: int

    Returns a dict with:
        total: int
        entries: list of scored + ranked entry dicts (score-sorted descending)
        winner_id: str | None
        summary: str
    """
    if not entries:
        return {
            "total": 0,
            "entries": [],
            "winner_id": None,
            "summary": "No active deployments found for this project.",
        }

    scored: list[dict] = []
    for entry in entries:
        usc = _usage_score(entry.get("request_count", 0))
        ssc = _sla_score(entry.get("p95_ms"))
        fsc: int | None = None
        raw_acc = entry.get("feedback_accuracy")
        if raw_acc is not None:
            fsc = int(float(raw_acc) * 100)
        fresh = _freshness_score(entry.get("age_days", 0))

        # Weighted composite — availability-aware
        weights: list[tuple[int, float]] = [(usc, 0.40), (fresh, 0.20)]
        if fsc is not None:
            weights.append((fsc, 0.30))
        if ssc is not None:
            weights.append((ssc, 0.10))

        total_w = sum(w for _, w in weights)
        composite = int(sum(v * w for v, w in weights) / total_w) if total_w > 0 else 0

        scored.append(
            {
                **entry,
                "usage_score": usc,
                "sla_score": ssc,
                "accuracy_score": fsc,
                "freshness_score": fresh,
                "composite_score": composite,
            }
        )

    # Sort by composite score descending, then by request_count as tiebreaker
    scored.sort(
        key=lambda e: (e["composite_score"], e.get("request_count", 0)), reverse=True
    )

    for i, e in enumerate(scored):
        e["rank"] = i + 1

    winner = scored[0] if scored else None
    winner_id = winner["deployment_id"] if winner else None

    # Build summary
    total = len(scored)
    if total == 1:
        summary = (
            f"1 deployment: {winner['algorithm_plain']} → {winner['target_column']} "
            f"(score {winner['composite_score']}/100, "
            f"{winner.get('request_count', 0)} predictions)."
        )
    else:
        runner = scored[1] if len(scored) > 1 else None
        gap = winner["composite_score"] - runner["composite_score"] if runner else 0
        summary = (
            f"{total} deployments ranked. Top performer: "
            f"{winner['algorithm_plain']} → {winner['target_column']} "
            f"(score {winner['composite_score']}/100). "
        )
        if gap >= 20:
            summary += f"Leads by {gap} points."
        elif gap >= 5:
            summary += f"Narrow lead of {gap} points."
        else:
            summary += "Very close competition."

    return {
        "total": total,
        "entries": scored,
        "winner_id": winner_id,
        "summary": summary,
    }


def _format_duration(seconds: float) -> str:
    """Return a human-readable duration string for a given number of seconds."""
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    if seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m} minute{'s' if m != 1 else ''}" + (f" {s}s" if s else "")
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h} hour{'s' if h != 1 else ''}" + (f" {m}m" if m else "")


def compute_deployment_throughput(log_dicts: list[dict], target_n: int = 1000) -> dict:
    """Estimate throughput capacity from measured prediction latencies.

    Pure function — no database or filesystem access.

    Args:
        log_dicts: List of dicts with ``response_ms`` key (float or None).
        target_n: Number of predictions to estimate processing time for.

    Returns:
        Dict with latency stats, throughput estimates, and plain-English summary.
        When fewer than 5 valid latency samples are available, returns
        ``{"verdict": "no_data", ...}`` with null latency fields.
    """
    valid_ms = sorted(
        float(d["response_ms"]) for d in log_dicts if d.get("response_ms") is not None
    )

    if len(valid_ms) < 5:
        return {
            "verdict": "no_data",
            "n_samples": len(valid_ms),
            "target_n": target_n,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "mean_ms": None,
            "max_rps": None,
            "serial_seconds": None,
            "serial_duration": None,
            "summary": "Not enough latency data yet — make at least 5 predictions to see throughput estimates.",
        }

    n = len(valid_ms)
    p50 = valid_ms[int(n * 0.50)]
    p95 = valid_ms[min(int(n * 0.95), n - 1)]
    p99 = valid_ms[min(int(n * 0.99), n - 1)]
    mean_ms = sum(valid_ms) / n

    # Conservative single-threaded throughput: 1 request per p95 latency
    max_rps = round(1000.0 / p95, 2) if p95 > 0 else None

    # Serial processing estimate (one at a time, p95 latency per request)
    serial_seconds = (target_n * p95 / 1000.0) if p95 > 0 else None

    if serial_seconds is None:
        verdict = "no_data"
    elif serial_seconds < 5:
        verdict = "instant"
    elif serial_seconds < 60:
        verdict = "fast"
    elif serial_seconds < 600:
        verdict = "moderate"
    elif serial_seconds < 3600:
        verdict = "slow"
    else:
        verdict = "very_slow"

    serial_duration = (
        _format_duration(serial_seconds) if serial_seconds is not None else None
    )

    rps_str = f"{max_rps:.1f}" if max_rps is not None else "unknown"
    dur_str = serial_duration or "unknown"

    summary = (
        f"Based on {n} measured predictions, your deployment's p95 latency is {p95:.0f} ms. "
        f"Single-threaded throughput: ~{rps_str} requests/second. "
        f"Processing {target_n:,} predictions serially would take ~{dur_str}."
    )

    return {
        "verdict": verdict,
        "n_samples": n,
        "target_n": target_n,
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "p99_ms": round(p99, 1),
        "mean_ms": round(mean_ms, 1),
        "max_rps": max_rps,
        "serial_seconds": (
            round(serial_seconds, 2) if serial_seconds is not None else None
        ),
        "serial_duration": serial_duration,
        "summary": summary,
    }


def compute_drift_importance_ranking(
    all_inputs: list[dict],
    feature_ranges: dict[str, dict],
    feature_importances: list[dict],
    *,
    max_features: int = 15,
) -> dict:
    """Rank input features by combined drift severity and model importance.

    Cross-references production input OOR/unseen rates with feature importances
    to identify which drifted features most affect prediction quality.

    Args:
        all_inputs: Parsed input_features dicts from PredictionLog records.
        feature_ranges: {feature: {"min", "max"} or {"known_categories": [...]}}
                        from PredictionPipeline.
        feature_importances: [{name, importance, rank, is_weak}] from
                             identify_weak_features() — importance normalized to sum=1.
        max_features: Maximum number of top-importance features to analyse.

    Returns:
        Dict with ranked_features, verdict, priority counts, n_features,
        n_samples, has_importances, and summary.
    """
    n_samples = len(all_inputs)

    if not feature_importances or all(
        fi.get("importance") is None for fi in feature_importances
    ):
        return {
            "has_importances": False,
            "ranked_features": [],
            "n_features": 0,
            "n_samples": n_samples,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "no_drift_count": 0,
            "verdict": "no_importances",
            "summary": (
                "Feature importances are not available for this model type. "
                "Try Random Forest or Gradient Boosting for built-in importance ranking."
            ),
        }

    # Cap to top max_features (already sorted importance-descending by identify_weak_features)
    top_features = [
        fi for fi in feature_importances if fi.get("importance") is not None
    ][:max_features]

    # Compute median importance for priority thresholds
    imp_values = sorted(fi["importance"] for fi in top_features)
    median_importance = imp_values[len(imp_values) // 2] if imp_values else 0.0

    # Build lookup: feature -> production values (skip None)
    feat_values: dict[str, list] = {}
    for inp in all_inputs:
        for k, v in inp.items():
            if v is not None:
                feat_values.setdefault(k, []).append(v)

    ranked: list[dict] = []
    for fi in top_features:
        feat = fi["name"]
        raw_imp = fi["importance"]
        importance_pct = round(raw_imp * 100, 2)
        rank = fi.get("rank", 0)

        drift_pct = 0.0
        feature_type = "unknown"
        drift_details = "No training range data available"

        values = feat_values.get(feat, [])
        if values:
            numeric: list[float] = []
            for v in values:
                try:
                    numeric.append(float(v))
                except (TypeError, ValueError):
                    pass

            ranges = feature_ranges.get(feat, {})

            if len(numeric) > len(values) * 0.5:
                feature_type = "numeric"
                train_min = ranges.get("min")
                train_max = ranges.get("max")
                if train_min is not None and train_max is not None:
                    oor = sum(1 for v in numeric if v < train_min or v > train_max)
                    drift_pct = oor / len(numeric)
                    if drift_pct > 0:
                        drift_details = (
                            f"{drift_pct:.0%} outside training range "
                            f"[{train_min:.3g}, {train_max:.3g}]"
                        )
                    else:
                        drift_details = "All values within training range"
                else:
                    drift_details = "No numeric range data"
            else:
                feature_type = "categorical"
                known = {str(c) for c in ranges.get("known_categories", [])}
                if known:
                    unseen = sum(1 for v in values if str(v) not in known)
                    drift_pct = unseen / len(values)
                    if drift_pct > 0:
                        drift_details = (
                            f"{drift_pct:.0%} unseen categories not in training"
                        )
                    else:
                        drift_details = "All categories seen during training"
                else:
                    drift_details = "No category data available"
        elif not all_inputs:
            drift_details = "No predictions yet"

        drift_pct_display = round(drift_pct * 100, 1)
        risk_score = round(drift_pct * raw_imp * 100, 4)

        if drift_pct >= 0.20 and raw_imp >= median_importance:
            priority = "critical"
        elif drift_pct >= 0.10 or (drift_pct >= 0.05 and raw_imp >= median_importance):
            priority = "high"
        elif drift_pct >= 0.02:
            priority = "medium"
        elif drift_pct > 0:
            priority = "low"
        else:
            priority = "no_drift"

        ranked.append(
            {
                "name": feat,
                "importance_pct": importance_pct,
                "rank": rank,
                "drift_pct": drift_pct_display,
                "risk_score": risk_score,
                "priority": priority,
                "feature_type": feature_type,
                "drift_details": drift_details,
            }
        )

    ranked.sort(key=lambda x: x["risk_score"], reverse=True)

    critical_count = sum(1 for r in ranked if r["priority"] == "critical")
    high_count = sum(1 for r in ranked if r["priority"] == "high")
    medium_count = sum(1 for r in ranked if r["priority"] == "medium")
    low_count = sum(1 for r in ranked if r["priority"] == "low")
    no_drift_count = sum(1 for r in ranked if r["priority"] == "no_drift")

    if critical_count > 0:
        verdict = "action_required"
    elif high_count > 0:
        verdict = "attention"
    elif medium_count + low_count > 0:
        verdict = "monitoring"
    else:
        verdict = "clear"

    drifting_important = [r for r in ranked if r["priority"] in ("critical", "high")]
    if not drifting_important:
        if medium_count + low_count == 0:
            summary = (
                f"No input drift detected across {len(ranked)} feature"
                f"{'s' if len(ranked) != 1 else ''} "
                f"({n_samples} recent prediction{'s' if n_samples != 1 else ''}). "
                "Your model is seeing representative production inputs."
            )
        else:
            feat_list = ", ".join(
                f"'{r['name']}'" for r in ranked if r["priority"] in ("medium", "low")
            )[:80]
            summary = (
                f"Minor drift in {medium_count + low_count} low-importance "
                f"feature{'s' if medium_count + low_count != 1 else ''} ({feat_list}). "
                "No significant impact on prediction quality."
            )
    elif critical_count > 0:
        top_names = ", ".join(
            f"'{r['name']}'" for r in ranked[:3] if r["priority"] == "critical"
        )
        summary = (
            f"{critical_count} critical feature{'s' if critical_count != 1 else ''} "
            f"({top_names}) — high drift in your most important predictors. "
            "Retraining is recommended."
        )
    else:
        top_names = ", ".join(
            f"'{r['name']}'" for r in ranked[:3] if r["priority"] == "high"
        )
        summary = (
            f"{high_count} high-priority feature{'s' if high_count != 1 else ''} "
            f"({top_names}) showing meaningful drift. "
            "Monitor closely and consider retraining."
        )

    return {
        "has_importances": True,
        "ranked_features": ranked,
        "n_features": len(ranked),
        "n_samples": n_samples,
        "critical_count": critical_count,
        "high_count": high_count,
        "medium_count": medium_count,
        "low_count": low_count,
        "no_drift_count": no_drift_count,
        "verdict": verdict,
        "summary": summary,
    }
