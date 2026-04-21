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
    direction_word = {"improving": "improving", "stable": "stable", "declining": "declining"}[
        direction
    ]
    if direction == "stable":
        trend_desc = f"averaging {overall_avg:.0f}% confidence"
    elif direction == "improving":
        trend_desc = f"improving (+{abs(trend_rate):.2f}% per day), now averaging {all_avgs[-1]:.0f}%"
    else:
        trend_desc = f"declining ({trend_rate:.2f}% per day), now averaging {all_avgs[-1]:.0f}%"

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
