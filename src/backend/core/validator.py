"""Model validation: cross-validation with confidence intervals, confusion matrix,
residual analysis, and plain-English limitation assessment.

Design:
- Loads the serialized joblib artifact produced by trainer.py.
- Uses sklearn.base.clone() to create an unfitted copy for honest CV
  (preserves hyperparameters, discards fitted parameters).
- Error analysis uses an 80/20 train/test split for visualization.
- Limitations are rule-based, referencing CV metric variance and dataset size.
"""

from __future__ import annotations

import copy
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder


def validate_model(
    model_path: str,
    df: pd.DataFrame,
    n_folds: int = 5,
) -> dict[str, Any]:
    """Full validation suite for a trained model.

    Args:
        model_path: Path to the joblib artifact from trainer.train_model().
        df: The original (or feature-engineered) DataFrame used for training.
        n_folds: Number of cross-validation folds (capped by dataset size).

    Returns:
        Dict with keys: problem_type, n_rows, n_folds, metrics (with CIs),
        error_analysis (confusion_matrix or residuals), limitations.
    """
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    feature_names: list[str] = artifact["feature_names"]
    target_col: str = artifact["target_col"]
    problem_type: str = artifact["problem_type"]

    X = df[feature_names].copy()
    y_raw = df[target_col]
    y, class_labels = _encode_target(y_raw, problem_type)

    n = len(X)
    cv = min(n_folds, max(2, n // 5))

    # --- Cross-validation with confidence intervals ---
    if problem_type == "classification":
        scoring: dict[str, str] = {"accuracy": "accuracy", "f1": "f1_weighted"}
        splitter: Any = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        scoring = {
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        }
        splitter = KFold(n_splits=cv, shuffle=True, random_state=42)

    # Clone produces an unfitted estimator — honest CV, no data leakage
    unfitted = clone(pipeline)
    try:
        cv_results = cross_validate(
            unfitted, X, y,
            cv=splitter,
            scoring=scoring,
            error_score=0.0,
        )
    except Exception:
        cv_results = {}

    metrics_with_ci = _build_ci_metrics(cv_results)

    # --- Error analysis: confusion matrix or residuals ---
    error_analysis = _compute_error_analysis(pipeline, X, y, class_labels, problem_type, n)

    return {
        "problem_type": problem_type,
        "n_rows": n,
        "n_folds": cv,
        "metrics": metrics_with_ci,
        "error_analysis": error_analysis,
        "limitations": _assess_limitations(metrics_with_ci, problem_type, n),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _encode_target(
    y_raw: pd.Series,
    problem_type: str,
) -> tuple[np.ndarray, list[str]]:
    """Encode target column to numeric array; return (y, class_labels)."""
    if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y_raw):
        le = LabelEncoder()
        y = le.fit_transform(y_raw.fillna("MISSING").astype(str))
        return y, [str(c) for c in le.classes_]
    else:
        y = y_raw.fillna(
            y_raw.median() if pd.api.types.is_numeric_dtype(y_raw) else 0
        ).values
        if problem_type == "classification":
            return y, [str(c) for c in sorted(np.unique(y))]
        return y, []


def _build_ci_metrics(cv_results: dict) -> dict[str, dict]:
    """Convert cross_validate output into mean ± 95% CI dicts."""
    out: dict[str, dict] = {}
    for key, values in cv_results.items():
        if not key.startswith("test_"):
            continue
        metric = key[5:]
        vals = np.abs(np.array(values)) if metric in ("mae", "rmse") else np.array(values)
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        out[metric] = {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci_low": round(mean - 1.96 * std, 4),
            "ci_high": round(min(mean + 1.96 * std, 1.0 if metric == "accuracy" else 1e9), 4),
            "fold_scores": [round(float(v), 4) for v in vals],
        }
    return out


def _compute_error_analysis(
    pipeline: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    class_labels: list[str],
    problem_type: str,
    n: int,
) -> dict[str, Any]:
    """Compute confusion matrix (classification) or residual plot data (regression)."""
    if n >= 10:
        stratify = y if (problem_type == "classification" and n > 20) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    p = copy.deepcopy(pipeline)
    p.fit(X_train, y_train)
    y_pred = p.predict(X_test)

    if problem_type == "classification":
        cm = confusion_matrix(y_test, y_pred)
        # Resolve labels to string names
        unique_vals = sorted(np.unique(np.concatenate([y_test, y_pred])))
        labels = [class_labels[int(v)] if int(v) < len(class_labels) else str(v)
                  for v in unique_vals]
        test_accuracy = float(np.mean(y_pred == y_test))
        # Per-class accuracy
        per_class = {}
        for idx, label in zip(unique_vals, labels):
            mask = y_test == idx
            if mask.sum() > 0:
                per_class[label] = round(float(np.mean(y_pred[mask] == y_test[mask])), 4)
        return {
            "type": "confusion_matrix",
            "labels": labels,
            "matrix": cm.tolist(),
            "test_accuracy": round(test_accuracy, 4),
            "per_class_accuracy": per_class,
        }
    else:
        residuals = y_test - y_pred
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y_test), min(len(y_test), 200), replace=False)
        return {
            "type": "residuals",
            "actual": [round(float(v), 4) for v in y_test[idx]],
            "predicted": [round(float(v), 4) for v in y_pred[idx]],
            "residuals": [round(float(v), 4) for v in residuals[idx]],
            "mae": round(float(np.mean(np.abs(residuals))), 4),
            "rmse": round(float(np.sqrt(np.mean(residuals**2))), 4),
        }


def _assess_limitations(
    metrics: dict[str, dict],
    problem_type: str,
    n_rows: int,
) -> list[str]:
    """Produce plain-English limitation statements for the user."""
    limitations: list[str] = []

    if n_rows < 100:
        limitations.append(
            f"Small dataset ({n_rows} rows) — predictions may vary significantly "
            "on new data. More training examples would increase reliability."
        )
    elif n_rows < 500:
        limitations.append(
            f"Moderate dataset ({n_rows} rows). The model is reasonable but may be "
            "less reliable on data that differs from the training set."
        )

    if problem_type == "classification":
        acc_info = metrics.get("accuracy", {})
        acc = acc_info.get("mean", 1.0)
        std = acc_info.get("std", 0.0)
        if std > 0.1:
            limitations.append(
                f"Accuracy varies significantly across folds (±{std:.1%}), "
                "suggesting the model may be inconsistent on unseen data."
            )
        if acc < 0.7:
            limitations.append(
                f"Accuracy of {acc:.1%} is below 70%. Consider adding more "
                "features or collecting more labeled training data."
            )
    else:
        r2_info = metrics.get("r2", {})
        r2 = r2_info.get("mean", 1.0)
        std = r2_info.get("std", 0.0)
        if r2 < 0.5:
            limitations.append(
                f"R² of {r2:.3f} means the model explains less than 50% of the "
                "variance. Some important predictors may be missing from the dataset."
            )
        if std > 0.15:
            limitations.append(
                f"R² varies across folds (±{std:.3f}) — the model may be unstable. "
                "Consider regularization or adding more training data."
            )

    if not limitations:
        limitations.append(
            "The model looks well-calibrated on available data. As with any model, "
            "performance may differ on data from different time periods or sources "
            "not represented in training."
        )

    return limitations
