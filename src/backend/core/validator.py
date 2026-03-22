"""Validation logic: cross-validation, confusion matrix, residual analysis,
and honest confidence/limitations assessment.

Design principles:
- All functions accept sklearn-compatible X, y arrays and a fitted or unfitted model.
- No mutation of inputs.
- Plain-English summaries accompany all numeric results.
- For cross-validation we use an *unfitted* estimator (re-created from algorithm registry).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def run_cross_validation(
    model_unfitted: Any,
    X: np.ndarray,
    y: np.ndarray,
    problem_type: str,
    n_splits: int = 5,
) -> dict:
    """Run K-fold cross-validation and return mean ± std + confidence interval.

    Uses StratifiedKFold for classification (preserves class ratio per fold).
    Falls back to 2-fold if dataset is too small for 5-fold.
    """
    # Clamp n_splits to avoid "n_splits > n_samples" errors on tiny datasets
    n_splits = min(n_splits, len(y))
    if n_splits < 2:
        return {
            "metric": "n/a",
            "scores": [],
            "mean": None,
            "std": None,
            "ci_low": None,
            "ci_high": None,
            "n_splits": 0,
            "summary": "Dataset too small to run cross-validation (need at least 2 rows).",
        }

    metric = "r2" if problem_type == "regression" else "f1_weighted"

    if problem_type == "classification":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = cross_val_score(model_unfitted, X, y, cv=cv, scoring=metric, n_jobs=1)

    mean = float(np.mean(scores))
    std = float(np.std(scores))
    ci_low = float(mean - 1.96 * std)
    ci_high = float(mean + 1.96 * std)

    return {
        "metric": metric,
        "scores": [round(float(s), 4) for s in scores],
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "n_splits": n_splits,
        "summary": _cv_summary(mean, std, n_splits, problem_type),
    }


def _cv_summary(mean: float, std: float, n_splits: int, problem_type: str) -> str:
    metric_name = "R²" if problem_type == "regression" else "F1"
    consistency = "consistent" if std < 0.05 else "somewhat variable"
    if mean >= 0.9:
        quality = "excellent"
    elif mean >= 0.7:
        quality = "good"
    elif mean >= 0.5:
        quality = "moderate"
    else:
        quality = "weak"
    return (
        f"Across {n_splits} folds, {metric_name} = {mean:.3f} ± {std:.3f} "
        f"({quality}, {consistency}). "
        "This tells us the model's performance is not just a fluke on one data split."
    )


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list | None = None,
) -> dict:
    """Compute confusion matrix with plain-English annotation.

    Returns a dict with 'matrix' (list-of-lists), 'labels', and 'summary'.
    """
    cm = confusion_matrix(y_true, y_pred)
    total = len(y_true)
    correct = int(np.trace(cm))

    if class_labels is not None:
        labels = [str(lbl) for lbl in class_labels]
    else:
        unique = sorted(set(y_true.tolist()))
        labels = [str(lbl) for lbl in unique]

    return {
        "matrix": cm.tolist(),
        "labels": labels,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "summary": _confusion_summary(cm, labels),
    }


def _confusion_summary(cm: np.ndarray, labels: list[str]) -> str:
    n_classes = len(labels)
    if n_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        return (
            f"True positives: {tp}, True negatives: {tn}, "
            f"False positives: {fp} (predicted positive, actually negative), "
            f"False negatives: {fn} (predicted negative, actually positive)."
        )
    # Multi-class: find worst class by per-class recall
    row_sums = cm.sum(axis=1).clip(min=1)
    recalls = cm.diagonal() / row_sums
    worst_idx = int(np.argmin(recalls))
    worst_label = labels[worst_idx] if worst_idx < len(labels) else "unknown"
    return (
        f"The model struggles most with class '{worst_label}' "
        f"(recall = {recalls[worst_idx]:.0%}). "
        "Consider collecting more training examples for this class."
    )


# ---------------------------------------------------------------------------
# Residual analysis (regression only)
# ---------------------------------------------------------------------------


def compute_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """Residual analysis for regression error characterization.

    Returns scatter data (predicted vs residual), error stats, and a summary.
    Scatter is sub-sampled to max 200 points for UI performance.
    """
    residuals = y_true - y_pred
    abs_errors = np.abs(residuals)

    n = len(y_true)
    rng = np.random.RandomState(42)
    idx = rng.choice(n, min(n, 200), replace=False)

    scatter = [
        {
            "predicted": round(float(y_pred[i]), 4),
            "residual": round(float(residuals[i]), 4),
        }
        for i in sorted(idx)
    ]

    return {
        "scatter": scatter,
        "mae": round(float(np.mean(abs_errors)), 4),
        "bias": round(float(np.mean(residuals)), 4),
        "std": round(float(np.std(residuals)), 4),
        "percentile_75": round(float(np.percentile(abs_errors, 75)), 4),
        "percentile_90": round(float(np.percentile(abs_errors, 90)), 4),
        "summary": _residual_summary(residuals, y_true),
    }


def _residual_summary(residuals: np.ndarray, y_true: np.ndarray) -> str:
    bias = float(np.mean(residuals))
    std_y = float(np.std(y_true)) or 1.0
    bias_frac = abs(bias) / std_y

    if bias_frac < 0.02:
        bias_str = "no systematic over- or under-prediction"
    elif bias > 0:
        bias_str = f"slight under-prediction bias (mean residual = {bias:+.2f})"
    else:
        bias_str = f"slight over-prediction bias (mean residual = {bias:+.2f})"

    return (
        f"Residual analysis shows {bias_str}. "
        "Ideally residuals are randomly scattered around zero with no pattern."
    )


# ---------------------------------------------------------------------------
# Confidence & limitations
# ---------------------------------------------------------------------------


def assess_confidence_limitations(
    metrics: dict,
    problem_type: str,
    n_rows: int,
    n_features: int,
    cv_std: float | None,
) -> dict:
    """Generate an honest confidence rating and list of limitation warnings."""
    limitations: list[str] = []

    if n_rows < 100:
        limitations.append(
            f"Small dataset ({n_rows} rows) — predictions may not generalize well to new data."
        )

    if cv_std is not None and cv_std > 0.1:
        limitations.append(
            f"High cross-validation variance (std = {cv_std:.3f}) — "
            "the model's accuracy is unstable across data splits."
        )

    if problem_type == "regression":
        r2 = metrics.get("r2", 0.0)
        if r2 < 0.5:
            limitations.append(
                f"R² = {r2:.2f} — the model explains less than half the variance "
                "in your target variable. It may be missing important predictors."
            )
    else:
        acc = metrics.get("accuracy", 0.0)
        if acc < 0.7:
            limitations.append(
                f"Accuracy = {acc:.0%} — the model is wrong more than 30% of the time. "
                "Consider collecting more data or engineering better features."
            )

    if n_features > n_rows * 0.5:
        limitations.append(
            f"You have {n_features} features and only {n_rows} rows — "
            "the model may be overfitting. Try removing less-important features."
        )

    if not limitations:
        limitations.append(
            "No major concerns detected — this model appears reliable for this dataset."
        )

    confidence = _overall_confidence(metrics, problem_type, cv_std)

    return {
        "overall_confidence": confidence,
        "limitations": limitations,
        "summary": (f"Overall confidence: {confidence.upper()}. " + limitations[0]),
    }


def _overall_confidence(
    metrics: dict,
    problem_type: str,
    cv_std: float | None,
) -> str:
    if problem_type == "regression":
        score = metrics.get("r2", 0.0)
    else:
        score = metrics.get("f1", metrics.get("accuracy", 0.0))

    stable = cv_std is None or cv_std < 0.05

    if score >= 0.85 and stable:
        return "high"
    if score >= 0.65:
        return "medium"
    return "low"
