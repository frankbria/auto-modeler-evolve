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
        correlations = {"pairs": pairs, "columns": cols, "matrix": _corr_matrix_dict(corr_matrix)}

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
    counts, bin_edges = np.histogram(finite_series, bins=min(20, finite_series.nunique()))
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
            insights.append({
                "type": "missing_values",
                "severity": "warning",
                "title": f"High missing rate in '{col['name']}'",
                "detail": (
                    f"{col['null_pct']:.1f}% of values are missing. "
                    "Consider filling with median/mode or dropping the column."
                ),
            })
        elif col["null_pct"] >= 5:
            insights.append({
                "type": "missing_values",
                "severity": "info",
                "title": f"Some missing values in '{col['name']}'",
                "detail": f"{col['null_pct']:.1f}% of values are missing.",
            })

    # High cardinality (likely ID columns)
    _numeric_dtypes = {"float64", "int64", "float32", "int32", "float16", "int16", "int8"}
    total_rows = len(df)
    for col in column_stats:
        if col["unique_count"] == total_rows and col["dtype"] not in _numeric_dtypes:
            insights.append({
                "type": "high_cardinality",
                "severity": "info",
                "title": f"'{col['name']}' looks like a unique identifier",
                "detail": "Every value is unique — this column probably won't help prediction.",
            })

    # Strong correlations
    for pair in corr_pairs[:3]:
        if abs(pair["correlation"]) >= 0.8:
            direction = "positively" if pair["correlation"] > 0 else "negatively"
            insights.append({
                "type": "correlation",
                "severity": "info",
                "title": f"Strong relationship: '{pair['col_a']}' and '{pair['col_b']}'",
                "detail": (
                    f"These columns are strongly {direction} correlated "
                    f"(r={pair['correlation']}). They carry similar information."
                ),
            })

    # Outliers
    for col in column_stats:
        if "outliers" in col and col["outliers"]["count"] > 0:
            pct = col["outliers"]["pct"]
            if pct >= 5:
                insights.append({
                    "type": "outliers",
                    "severity": "warning",
                    "title": f"Outliers detected in '{col['name']}'",
                    "detail": (
                        f"{col['outliers']['count']} values ({pct:.1f}%) fall outside "
                        f"the expected range "
                        f"[{col['outliers']['lower_fence']} – {col['outliers']['upper_fence']}]."
                    ),
                })

    # Duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        insights.append({
            "type": "duplicates",
            "severity": "warning",
            "title": f"{dup_count} duplicate row{'s' if dup_count > 1 else ''} found",
            "detail": "Duplicate rows can inflate model performance. Consider removing them.",
        })

    # Possible date columns (string dtype with date-like values)
    # dtype may be "object" (pandas < 3) or "str" (pandas >= 3 with StringDtype)
    for col in column_stats:
        if col["dtype"] in ("object", "str", "string") and col["sample_values"]:
            sample = str(col["sample_values"][0])
            if _looks_like_date(sample):
                insights.append({
                    "type": "date_column",
                    "severity": "info",
                    "title": f"'{col['name']}' looks like a date column",
                    "detail": (
                        "Converting it to datetime could unlock time-based features "
                        "like month, day-of-week, or trend analysis."
                    ),
                })

    return insights


def _looks_like_date(value: str) -> bool:
    """Quick heuristic: does the value look like a date string?"""
    import re
    date_pattern = re.compile(
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}"
    )
    return bool(date_pattern.match(value.strip()))


def _safe_scalar(value):
    """Convert numpy scalars to native Python types for JSON serialization."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
