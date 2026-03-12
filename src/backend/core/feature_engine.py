"""Feature engineering: suggestions, application, target detection, and importance.

Design principles:
- Suggestions are derived purely from statistical analysis (no LLM) for speed and
  determinism. Plain-English descriptions are written directly in the code.
- apply_transformations returns a new DataFrame without mutating the input.
- detect_problem_type uses a simple heuristic (numeric → regression, categorical
  or low-cardinality int → classification).
- compute_feature_importance uses sklearn mutual information, which works for both
  classification and regression targets and handles mixed dtypes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FeatureSuggestion:
    id: str
    column: str                   # source column (or "col_a__col_b" for interactions)
    transform_type: str           # see TRANSFORM_TYPES below
    title: str
    description: str
    preview_columns: list[str]    # names of columns that will be added
    example_values: list[Any]     # sample values from the source column


# Valid transform types
TRANSFORM_TYPES = {
    "date_decompose",
    "log_transform",
    "one_hot",
    "label_encode",
    "bin_quartile",
    "interaction",
}


# ---------------------------------------------------------------------------
# Suggest features
# ---------------------------------------------------------------------------

def suggest_features(
    df: pd.DataFrame,
    column_stats: list[dict],
) -> list[FeatureSuggestion]:
    """Generate feature transformation suggestions based on statistical analysis.

    Returns a list of FeatureSuggestion objects, ordered by expected impact.
    """
    suggestions: list[FeatureSuggestion] = []
    col_lookup = {c["name"]: c for c in column_stats}

    for col_stat in column_stats:
        col = col_stat["name"]
        dtype = col_stat.get("dtype", "")
        unique = col_stat.get("unique_count", 0)
        n_rows = len(df)

        # --- Date decomposition ---
        if dtype in ("object", "str", "string") and col_stat.get("sample_values"):
            sample = str(col_stat["sample_values"][0])
            if _looks_like_date(sample):
                suggestions.append(FeatureSuggestion(
                    id=str(uuid4()),
                    column=col,
                    transform_type="date_decompose",
                    title=f"Extract date parts from '{col}'",
                    description=(
                        f"Convert '{col}' into numeric features: year, month, "
                        "day-of-week (0=Mon…6=Sun), and is_weekend (0/1). "
                        "Time-based patterns like seasonality or weekday effects "
                        "become visible to the model."
                    ),
                    preview_columns=[
                        f"{col}_year", f"{col}_month",
                        f"{col}_dayofweek", f"{col}_is_weekend",
                    ],
                    example_values=col_stat["sample_values"][:3],
                ))
            continue  # string columns: only date_decompose, skip numeric checks

        # --- Numeric column suggestions ---
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            if series.empty:
                continue

            # Log transform for right-skewed distributions
            skew = float(series.skew()) if len(series) >= 3 else 0.0
            if skew > 1.5 and series.min() >= 0:
                suggestions.append(FeatureSuggestion(
                    id=str(uuid4()),
                    column=col,
                    transform_type="log_transform",
                    title=f"Log-transform '{col}' (skewness {skew:.1f})",
                    description=(
                        f"'{col}' is heavily right-skewed (skewness={skew:.1f}). "
                        "A log transform compresses the long tail and helps linear "
                        "models treat large and small values more evenly. "
                        "Uses log(1 + x) so zero values are handled safely."
                    ),
                    preview_columns=[f"{col}_log"],
                    example_values=col_stat["sample_values"][:3],
                ))

            # Binning for continuous numeric columns with many unique values
            if unique > 20 and pd.api.types.is_float_dtype(series):
                suggestions.append(FeatureSuggestion(
                    id=str(uuid4()),
                    column=col,
                    transform_type="bin_quartile",
                    title=f"Bin '{col}' into quartiles",
                    description=(
                        f"Group '{col}' into four equal-sized buckets: "
                        "Q1 (bottom 25%), Q2, Q3, Q4 (top 25%). "
                        "Useful when the exact numeric value matters less than "
                        "which tier a record falls into (e.g. low/mid/high revenue)."
                    ),
                    preview_columns=[f"{col}_quartile"],
                    example_values=col_stat["sample_values"][:3],
                ))

    # --- Categorical encoding suggestions ---
    for col_stat in column_stats:
        col = col_stat["name"]
        dtype = col_stat.get("dtype", "")
        unique = col_stat.get("unique_count", 0)
        n_rows = len(df)

        # Skip date-like string columns (already handled above)
        if dtype in ("object", "str", "string") and col_stat.get("sample_values"):
            sample = str(col_stat["sample_values"][0])
            if _looks_like_date(sample):
                continue

        is_categorical = (
            dtype in ("object", "str", "string", "category")
            or (pd.api.types.is_integer_dtype(df[col]) and unique <= 20)
        )
        if not is_categorical or unique < 2:
            continue

        if unique <= 15:
            suggestions.append(FeatureSuggestion(
                id=str(uuid4()),
                column=col,
                transform_type="one_hot",
                title=f"One-hot encode '{col}' ({unique} categories)",
                description=(
                    f"Create {unique} binary (0/1) columns — one for each "
                    f"value of '{col}'. Most ML algorithms can't directly use "
                    "text categories, so this converts them into numbers the "
                    "model can understand. Works best for columns with fewer "
                    "than 15 distinct values."
                ),
                preview_columns=[
                    f"{col}_{v}" for v in
                    df[col].dropna().value_counts().head(5).index.astype(str)
                ],
                example_values=col_stat["sample_values"][:3],
            ))
        elif unique <= 50:
            suggestions.append(FeatureSuggestion(
                id=str(uuid4()),
                column=col,
                transform_type="label_encode",
                title=f"Label-encode '{col}' ({unique} categories)",
                description=(
                    f"Assign each of the {unique} values in '{col}' a unique "
                    "integer (0, 1, 2…). Faster than one-hot for high-cardinality "
                    "columns, but the order of numbers is arbitrary — works best "
                    "with tree-based models (Random Forest, XGBoost) that can "
                    "ignore spurious ordering."
                ),
                preview_columns=[f"{col}_encoded"],
                example_values=col_stat["sample_values"][:3],
            ))

    # --- Interaction features for top correlated numeric pairs ---
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().abs()
        pairs_seen: set[tuple] = set()
        for c1 in numeric_cols:
            for c2 in numeric_cols:
                if c1 == c2:
                    continue
                key = tuple(sorted([c1, c2]))
                if key in pairs_seen:
                    continue
                pairs_seen.add(key)
                r = corr.loc[c1, c2]
                if pd.isna(r) or r < 0.5:
                    continue
                if len(suggestions) < 12:  # cap total suggestions
                    suggestions.append(FeatureSuggestion(
                        id=str(uuid4()),
                        column=f"{c1}__{c2}",
                        transform_type="interaction",
                        title=f"Multiply '{c1}' × '{c2}' (r={r:.2f})",
                        description=(
                            f"'{c1}' and '{c2}' are correlated (r={r:.2f}). "
                            "Their product captures joint effects — for example, "
                            "price × quantity = revenue. "
                            "Useful when the combination matters more than each "
                            "column individually."
                        ),
                        preview_columns=[f"{c1}_x_{c2}"],
                        example_values=(
                            (df[c1] * df[c2]).dropna().head(3).tolist()
                        ),
                    ))

    return suggestions


# ---------------------------------------------------------------------------
# Apply transformations
# ---------------------------------------------------------------------------

def apply_transformations(
    df: pd.DataFrame,
    transformations: list[dict],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Apply a list of approved transformations to a DataFrame.

    Each transformation dict must have: column, transform_type.
    Optional: params dict (e.g. {"n_bins": 4}).

    Returns:
        (transformed_df, column_mapping) where column_mapping maps each
        source column to the list of new columns it produced.
    """
    result = df.copy()
    column_mapping: dict[str, list[str]] = {}

    for t in transformations:
        col = t["column"]
        ttype = t["transform_type"]
        params = t.get("params", {})

        try:
            if ttype == "date_decompose":
                new_cols = _apply_date_decompose(result, col)
                column_mapping[col] = new_cols

            elif ttype == "log_transform":
                new_col = f"{col}_log"
                result[new_col] = np.log1p(result[col].clip(lower=0))
                column_mapping[col] = [new_col]

            elif ttype == "bin_quartile":
                new_col = f"{col}_quartile"
                result[new_col] = pd.qcut(
                    result[col],
                    q=params.get("n_bins", 4),
                    labels=["Q1", "Q2", "Q3", "Q4"],
                    duplicates="drop",
                ).astype(str)
                column_mapping[col] = [new_col]

            elif ttype == "one_hot":
                dummies = pd.get_dummies(
                    result[col].fillna("MISSING").astype(str),
                    prefix=col,
                    drop_first=False,
                    dtype=int,
                )
                new_cols = dummies.columns.tolist()
                result = pd.concat([result, dummies], axis=1)
                column_mapping[col] = new_cols

            elif ttype == "label_encode":
                new_col = f"{col}_encoded"
                categories = result[col].dropna().unique()
                label_map = {v: i for i, v in enumerate(sorted(categories, key=str))}
                result[new_col] = result[col].map(label_map).fillna(-1).astype(int)
                column_mapping[col] = [new_col]

            elif ttype == "interaction":
                # col is encoded as "col_a__col_b"
                parts = col.split("__", 1)
                if len(parts) == 2:
                    c1, c2 = parts
                    new_col = f"{c1}_x_{c2}"
                    result[new_col] = result[c1] * result[c2]
                    column_mapping[col] = [new_col]

        except Exception:
            # Skip failed transforms silently; caller can detect via column_mapping gaps
            continue

    return result, column_mapping


# ---------------------------------------------------------------------------
# Target variable detection
# ---------------------------------------------------------------------------

def detect_problem_type(df: pd.DataFrame, target_col: str) -> dict:
    """Determine whether a target column implies classification or regression.

    Returns a dict with: problem_type, reason, classes (for classification).
    """
    if target_col not in df.columns:
        return {"problem_type": None, "reason": f"Column '{target_col}' not found."}

    series = df[target_col].dropna()
    n_unique = series.nunique()

    if pd.api.types.is_bool_dtype(series):
        return {
            "problem_type": "classification",
            "reason": (
                f"'{target_col}' is a yes/no column — the model will learn "
                "to predict which category a new row belongs to."
            ),
            "classes": series.unique().astype(str).tolist(),
        }

    if not pd.api.types.is_numeric_dtype(series):
        classes = series.value_counts().head(20).index.astype(str).tolist()
        return {
            "problem_type": "classification",
            "reason": (
                f"'{target_col}' contains text categories — the model will "
                f"predict which of the {n_unique} categories a row belongs to."
            ),
            "classes": classes,
        }

    # Float columns are almost always continuous → regression
    if pd.api.types.is_float_dtype(series):
        return {
            "problem_type": "regression",
            "reason": (
                f"'{target_col}' is a continuous number ({series.min():.2g}–{series.max():.2g}). "
                "The model will predict an exact numeric value for new rows."
            ),
            "classes": [],
        }

    # Integer column: low cardinality relative to dataset size → classification
    cardinality_threshold = max(10, int(len(series) * 0.05))
    if n_unique <= cardinality_threshold:
        classes = sorted(series.unique().astype(str).tolist())
        return {
            "problem_type": "classification",
            "reason": (
                f"'{target_col}' is a whole number with only {n_unique} distinct values — "
                "likely a rating, score, or category. The model will predict "
                "which value a row belongs to."
            ),
            "classes": classes,
        }

    return {
        "problem_type": "regression",
        "reason": (
            f"'{target_col}' is a continuous number ({series.min():.2g}–{series.max():.2g}). "
            "The model will predict an exact numeric value for new rows."
        ),
        "classes": [],
    }


# ---------------------------------------------------------------------------
# Feature importance preview
# ---------------------------------------------------------------------------

def compute_feature_importance(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    column_stats: list[dict],
) -> list[dict]:
    """Compute mutual-information-based feature importance before training.

    Returns a list of {column, importance, rank, description} dicts,
    sorted highest → lowest importance.

    Mutual information captures non-linear relationships, unlike Pearson
    correlation, making it a better pre-training signal.
    """
    from sklearn.feature_selection import (
        mutual_info_classif,
        mutual_info_regression,
    )
    from sklearn.preprocessing import LabelEncoder

    if target_col not in df.columns:
        return []

    # Prepare feature matrix: numeric only, fill NaN
    feature_cols = [c for c in df.columns if c != target_col]
    X_parts = []
    used_cols: list[str] = []

    for col in feature_cols:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            X_parts.append(series.fillna(series.median()).values.reshape(-1, 1))
            used_cols.append(col)
        elif series.dtype in (object, "string", "str", "category"):
            le = LabelEncoder()
            encoded = le.fit_transform(series.fillna("MISSING").astype(str))
            X_parts.append(encoded.reshape(-1, 1))
            used_cols.append(col)

    if not X_parts:
        return []

    X = np.hstack(X_parts)

    # Prepare target
    y_series = df[target_col].fillna(
        df[target_col].mode()[0] if problem_type == "classification"
        else df[target_col].median()
    )
    if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y_series):
        le = LabelEncoder()
        y = le.fit_transform(y_series.astype(str))
    else:
        y = y_series.values

    mi_fn = (
        mutual_info_classif if problem_type == "classification"
        else mutual_info_regression
    )
    try:
        scores = mi_fn(X, y, random_state=42)
    except Exception:
        return []

    total = scores.sum()
    results = []
    for col, score in zip(used_cols, scores):
        pct = round(float(score / total * 100), 1) if total > 0 else 0.0
        results.append({
            "column": col,
            "importance": round(float(score), 4),
            "importance_pct": pct,
            "description": _importance_description(col, pct),
        })

    results.sort(key=lambda r: r["importance"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _apply_date_decompose(df: pd.DataFrame, col: str) -> list[str]:
    """Parse a string/object column as datetime and extract date parts in-place."""
    parsed = pd.to_datetime(df[col], errors="coerce")
    new_cols = []
    for suffix, accessor in [
        ("year", lambda s: s.dt.year),
        ("month", lambda s: s.dt.month),
        ("dayofweek", lambda s: s.dt.dayofweek),
        ("is_weekend", lambda s: (s.dt.dayofweek >= 5).astype(int)),
    ]:
        new_col = f"{col}_{suffix}"
        df[new_col] = accessor(parsed)
        new_cols.append(new_col)
    return new_cols


def _looks_like_date(value: str) -> bool:
    """Quick heuristic: does the value look like a date string?"""
    date_pattern = re.compile(
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}"
    )
    return bool(date_pattern.match(value.strip()))


def _importance_description(col: str, pct: float) -> str:
    if pct >= 20:
        return f"Very strong predictor — explains ~{pct:.0f}% of the target's variation."
    if pct >= 10:
        return f"Strong predictor — responsible for ~{pct:.0f}% of predictive signal."
    if pct >= 5:
        return f"Moderate predictor (~{pct:.0f}% signal)."
    if pct >= 1:
        return f"Weak predictor (~{pct:.0f}% signal) — may still help in combination."
    return "Very low predictive signal — consider excluding this feature."
