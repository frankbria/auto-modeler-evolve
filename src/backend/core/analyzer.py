from typing import Any

import numpy as np
import pandas as pd

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
            return {"type": "histogram", "bins": [_safe_scalar(finite.iloc[0])], "counts": [len(finite)]}
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

    return "".join(parts) + "." if parts else f"Column '{col_name}' with {n_total} rows."
