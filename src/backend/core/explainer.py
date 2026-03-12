"""Feature importance and individual prediction explanations.

Design:
- Global importance: sklearn permutation_importance on a held-out split
  (or all data if small). Model-agnostic — works with any sklearn Pipeline.
- Individual explanations: perturbation-based. For each feature, we replace
  its value with the column median (numeric) or mode (categorical) and measure
  how much the prediction changes. This is a simplified SHAP-like approach
  with no extra dependencies.
"""

from __future__ import annotations

import copy
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def compute_global_importance(
    model_path: str,
    df: pd.DataFrame,
    n_repeats: int = 10,
) -> dict[str, Any]:
    """Compute global feature importance via permutation importance.

    Uses the already-fitted pipeline (no re-training). Evaluates on a 25%
    held-out split when n >= 20, otherwise evaluates on all data.

    Args:
        model_path: Path to joblib artifact from trainer.train_model().
        df: DataFrame used for training (or feature-engineered version).
        n_repeats: Number of permutation repeats (higher = more stable).

    Returns:
        Dict with keys: features (sorted by importance), summary, problem_type.
    """
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    feature_names: list[str] = artifact["feature_names"]
    target_col: str = artifact["target_col"]
    problem_type: str = artifact["problem_type"]

    X = df[feature_names].copy()
    y_raw = df[target_col]
    y = _encode_y(y_raw, problem_type)

    n = len(X)
    if n >= 20:
        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        # Refit a deep copy on the training portion
        p = copy.deepcopy(pipeline)
        p.fit(X_train, y_train)
    else:
        X_eval, y_eval = X, y
        p = pipeline  # Already fitted on all data

    scoring = "accuracy" if problem_type == "classification" else "r2"
    perm = permutation_importance(
        p, X_eval, y_eval,
        n_repeats=n_repeats,
        random_state=42,
        scoring=scoring,
    )

    # Build sorted feature list
    total_positive = max(float(np.sum(np.maximum(perm.importances_mean, 0))), 1e-10)
    features = []
    for i, feat in enumerate(feature_names):
        imp = float(perm.importances_mean[i])
        features.append({
            "column": feat,
            "importance": round(max(imp, 0.0), 4),
            "importance_pct": round(max(imp, 0.0) / total_positive * 100, 1),
            "std": round(float(perm.importances_std[i]), 4),
        })

    features.sort(key=lambda x: x["importance"], reverse=True)
    for i, f in enumerate(features):
        f["rank"] = i + 1

    top3 = [f["column"] for f in features[:3] if f["importance"] > 0]
    if top3:
        summary = f"The most influential factors are: {', '.join(top3)}."
    else:
        summary = "No features had a measurable impact on predictions with this dataset size."

    return {
        "features": features,
        "summary": summary,
        "problem_type": problem_type,
        "scoring_metric": scoring,
    }


def explain_prediction(
    model_path: str,
    df: pd.DataFrame,
    row_index: int,
) -> dict[str, Any]:
    """Explain a single prediction by feature perturbation.

    For each feature, replaces its value with the column median (numeric) or
    mode (categorical), then measures the change in prediction. Features with
    large positive impact push the prediction *up*; negative impact pushes it
    *down* from the baseline.

    Args:
        model_path: Path to joblib artifact.
        df: DataFrame used for training.
        row_index: Zero-based index of the row to explain.

    Returns:
        Dict with keys: row_index, prediction, probability (classification only),
        problem_type, top_factors (sorted by |impact|).
    """
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    feature_names: list[str] = artifact["feature_names"]
    target_col: str = artifact["target_col"]
    problem_type: str = artifact["problem_type"]

    X = df[feature_names].copy()

    if row_index < 0 or row_index >= len(X):
        raise ValueError(f"row_index {row_index} out of range [0, {len(X) - 1}]")

    row = X.iloc[[row_index]]

    # Baseline
    raw_pred = pipeline.predict(row)[0]
    if problem_type == "regression":
        baseline_pred: Any = float(raw_pred)
        baseline_prob = None
    else:
        baseline_pred = str(raw_pred)
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(row)[0]
            baseline_prob = float(np.max(probs))
        else:
            baseline_prob = None

    # Neutral values per column (median for numeric, mode for categorical)
    neutral_vals = _compute_neutral_values(X)

    contributions = []
    for feat in feature_names:
        perturbed = row.copy()
        perturbed[feat] = neutral_vals[feat]

        pert_pred = pipeline.predict(perturbed)[0]

        if problem_type == "regression":
            impact = float(raw_pred) - float(pert_pred)
        else:
            if hasattr(pipeline, "predict_proba"):
                orig_prob = float(np.max(pipeline.predict_proba(row)[0]))
                pert_prob = float(np.max(pipeline.predict_proba(perturbed)[0]))
                impact = orig_prob - pert_prob
            else:
                impact = 1.0 if str(raw_pred) != str(pert_pred) else 0.0

        original_value = row[feat].iloc[0]
        contributions.append({
            "column": feat,
            "original_value": (
                float(original_value)
                if pd.api.types.is_numeric_dtype(type(original_value))
                else str(original_value)
            ),
            "impact": round(impact, 4),
            "direction": "positive" if impact > 0.001 else ("negative" if impact < -0.001 else "neutral"),
        })

    contributions.sort(key=lambda x: abs(x["impact"]), reverse=True)

    return {
        "row_index": row_index,
        "prediction": baseline_pred,
        "probability": baseline_prob,
        "problem_type": problem_type,
        "top_factors": contributions[:10],
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _encode_y(y_raw: pd.Series, problem_type: str) -> np.ndarray:
    """Encode target to numeric, consistent with trainer._prepare_xy."""
    if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y_raw):
        le = LabelEncoder()
        return le.fit_transform(y_raw.fillna("MISSING").astype(str))
    return y_raw.fillna(
        y_raw.median() if pd.api.types.is_numeric_dtype(y_raw) else 0
    ).values


def _compute_neutral_values(X: pd.DataFrame) -> dict[str, Any]:
    """Compute a neutral (baseline) value per column for perturbation."""
    neutral: dict[str, Any] = {}
    for col in X.columns:
        col_data = X[col].dropna()
        if col_data.empty:
            neutral[col] = 0
        elif pd.api.types.is_numeric_dtype(col_data):
            neutral[col] = float(col_data.median())
        else:
            mode = col_data.mode()
            neutral[col] = str(mode.iloc[0]) if not mode.empty else "missing"
    return neutral
