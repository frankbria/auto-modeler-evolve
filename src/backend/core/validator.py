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
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix
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

    Returns matrix, labels, per_class_metrics, most_confused_pair, and summary.
    """
    cm = confusion_matrix(y_true, y_pred)
    total = len(y_true)
    correct = int(np.trace(cm))

    if class_labels is not None:
        labels = [str(lbl) for lbl in class_labels]
    else:
        unique = sorted(set(y_true.tolist()))
        labels = [str(lbl) for lbl in unique]

    # Per-class precision, recall, f1, support
    n = len(cm)
    per_class_metrics = []
    for i in range(n):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = (
            round(2 * precision * recall / (precision + recall), 4)
            if (precision + recall) > 0
            else 0.0
        )
        support = int(cm[i, :].sum())
        lbl = labels[i] if i < len(labels) else str(i)
        per_class_metrics.append(
            {
                "label": lbl,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    # Most common misclassification (highest off-diagonal cell)
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    most_confused_pair = None
    if cm_no_diag.max() > 0:
        idx = np.unravel_index(cm_no_diag.argmax(), cm_no_diag.shape)
        actual_lbl = labels[idx[0]] if idx[0] < len(labels) else str(idx[0])
        pred_lbl = labels[idx[1]] if idx[1] < len(labels) else str(idx[1])
        most_confused_pair = {
            "actual": actual_lbl,
            "predicted": pred_lbl,
            "count": int(cm_no_diag[idx]),
        }

    return {
        "matrix": cm.tolist(),
        "labels": labels,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "per_class_metrics": per_class_metrics,
        "most_confused_pair": most_confused_pair,
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


# ---------------------------------------------------------------------------
# Segment performance breakdown
# ---------------------------------------------------------------------------


def compute_segment_performance(
    group_values: list,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    problem_type: str,
    max_groups: int = 15,
) -> dict:
    """Compute per-segment model performance metrics.

    Args:
        group_values: List of group labels aligned 1-to-1 with y_true / y_pred rows.
        y_true: Ground-truth target values.
        y_pred: Model predictions.
        problem_type: "regression" or "classification".
        max_groups: Cap number of groups shown (picks largest groups first).

    Returns:
        dict with segments list, best/worst segment, gap, and plain-English summary.
    """
    metric_name = "R²" if problem_type == "regression" else "Accuracy"
    groups: dict[str, dict] = {}
    for gval, yt, yp in zip(group_values, y_true.tolist(), y_pred.tolist()):
        key = str(gval)
        if key not in groups:
            groups[key] = {"y_true": [], "y_pred": []}
        groups[key]["y_true"].append(yt)
        groups[key]["y_pred"].append(yp)

    if not groups:
        return {
            "segments": [],
            "best_segment": None,
            "worst_segment": None,
            "gap": None,
            "metric_name": metric_name,
            "summary": "No segment data available.",
        }

    # Sort by group size descending, cap at max_groups
    sorted_keys = sorted(groups, key=lambda k: len(groups[k]["y_true"]), reverse=True)[
        :max_groups
    ]

    segments = []
    for key in sorted_keys:
        yt_arr = np.array(groups[key]["y_true"])
        yp_arr = np.array(groups[key]["y_pred"])
        n = len(yt_arr)
        metric = _segment_metric(yt_arr, yp_arr, problem_type)
        segments.append(
            {
                "name": key,
                "n": n,
                "metric": metric,
                "metric_name": metric_name,
                "status": _segment_status(metric, problem_type),
                "low_sample": n < 10,
            }
        )

    # Sort segments by metric descending for display
    segments.sort(
        key=lambda s: (s["metric"] is not None, s["metric"] or 0), reverse=True
    )

    valid_segs = [s for s in segments if s["metric"] is not None]
    best = valid_segs[0] if valid_segs else None
    worst = valid_segs[-1] if valid_segs else None
    gap = (
        round(best["metric"] - worst["metric"], 4)
        if best and worst and best is not worst
        else 0.0
    )

    summary = _segment_perf_summary(best, worst, gap, metric_name, problem_type)

    return {
        "segments": segments,
        "best_segment": best["name"] if best else None,
        "worst_segment": worst["name"] if worst else None,
        "gap": gap,
        "metric_name": metric_name,
        "summary": summary,
    }


def _segment_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    problem_type: str,
) -> float | None:
    """Compute a single numeric metric for one segment. Returns None if insufficient data."""
    if len(y_true) < 2:
        return None
    if problem_type == "regression":
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot == 0:
            return None
        r2 = 1 - ss_res / ss_tot
        return round(max(r2, -1.0), 4)  # clamp to avoid extreme negatives in display
    else:
        accuracy = float(np.mean(y_true == y_pred))
        return round(accuracy, 4)


def _segment_status(metric: float | None, problem_type: str) -> str:
    if metric is None:
        return "insufficient_data"
    if problem_type == "regression":
        if metric >= 0.85:
            return "strong"
        if metric >= 0.65:
            return "moderate"
        if metric >= 0.4:
            return "weak"
        return "poor"
    else:
        if metric >= 0.85:
            return "strong"
        if metric >= 0.70:
            return "moderate"
        if metric >= 0.50:
            return "weak"
        return "poor"


def _segment_perf_summary(
    best: dict | None,
    worst: dict | None,
    gap: float,
    metric_name: str,
    problem_type: str,
) -> str:
    if not best:
        return "Not enough data to compare segment performance."

    if best is worst or worst is None:
        m_val = best["metric"]
        m_str = f"{m_val:.2f}" if problem_type == "regression" else f"{m_val:.0%}"
        return (
            f"Only one segment found. {metric_name} = {m_str}. "
            "Collect data from more groups to compare performance across segments."
        )

    best_m = best["metric"]
    worst_m = worst["metric"]
    best_str = f"{best_m:.2f}" if problem_type == "regression" else f"{best_m:.0%}"
    worst_str = f"{worst_m:.2f}" if problem_type == "regression" else f"{worst_m:.0%}"
    gap_str = f"{gap:.2f}" if problem_type == "regression" else f"{gap:.0%}"

    summary = (
        f"Your model performs best on '{best['name']}' ({metric_name}={best_str}) "
        f"and worst on '{worst['name']}' ({metric_name}={worst_str}). "
    )
    if gap > 0.2:
        summary += (
            f"The {gap_str} performance gap is significant — "
            f"consider collecting more training data for '{worst['name']}' "
            "or training a separate model for that segment."
        )
    else:
        summary += f"The {gap_str} gap is small — the model is fairly consistent across segments."

    if worst.get("low_sample"):
        summary += (
            f" Note: '{worst['name']}' has fewer than 10 rows — "
            "its metric may not be reliable."
        )
    return summary


# ---------------------------------------------------------------------------
# Prediction error analysis (worst-case training errors)
# ---------------------------------------------------------------------------


def compute_prediction_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    problem_type: str,
    n: int = 10,
    feature_rows: list[dict] | None = None,
    target_classes: list | None = None,
) -> dict:
    """Find the top-N largest prediction errors on training data.

    For regression: rows with largest absolute residuals, sorted descending.
    For classification: rows where the prediction was wrong (first n wrong rows).

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions aligned 1-to-1 with y_true.
        problem_type: "regression" or "classification".
        n: Max error rows to return (clamped 1–50, default 10).
        feature_rows: Optional list of feature dicts (same row order as y_true).
                      When provided, each error row includes a 'features' sub-dict.
        target_classes: Class label list for classification decoding.

    Returns:
        dict with errors list, total_errors, error_rate, problem_type, summary.
    """
    n = max(1, min(50, n))
    total = len(y_true)

    if total == 0:
        return {
            "errors": [],
            "total_errors": 0,
            "error_rate": 0.0,
            "problem_type": problem_type,
            "summary": "No training data available.",
        }

    if problem_type == "regression":
        abs_errors = np.abs(y_true - y_pred)
        sorted_idx = np.argsort(abs_errors)[::-1]
        top_idx = sorted_idx[:n]

        errors = []
        for rank, i in enumerate(top_idx, 1):
            row: dict = {
                "actual": round(float(y_true[i]), 4),
                "predicted": round(float(y_pred[i]), 4),
                "error": round(float(y_true[i] - y_pred[i]), 4),
                "abs_error": round(float(abs_errors[i]), 4),
                "rank": rank,
            }
            if feature_rows and i < len(feature_rows):
                row["features"] = feature_rows[i]
            errors.append(row)

        mae = float(np.mean(abs_errors))
        y_range = float(np.max(y_true) - np.min(y_true)) if np.ptp(y_true) != 0 else 1.0
        worst_pct = round(float(abs_errors[sorted_idx[0]]) / y_range * 100, 1)

        summary = (
            f"The model's worst {len(errors)} predictions have errors up to "
            f"{worst_pct:.0f}% of the data range "
            f"(MAE across all {total} training rows = {mae:.3f}). "
            "Examine these rows for patterns — they may reveal segments where the "
            "model needs more training data or better features."
        )
        return {
            "errors": errors,
            "total_errors": total,
            "error_rate": 0.0,
            "problem_type": problem_type,
            "summary": summary,
        }

    else:
        # Classification: wrong predictions
        wrong_mask = y_true != y_pred
        wrong_idx = np.where(wrong_mask)[0]
        total_errors = int(wrong_mask.sum())
        error_rate = round(total_errors / total, 4) if total > 0 else 0.0
        top_idx = wrong_idx[:n]

        def _decode(v: float) -> str:
            if target_classes:
                idx = max(0, min(int(round(float(v))), len(target_classes) - 1))
                return str(target_classes[idx])
            return str(v)

        errors = []
        for rank, i in enumerate(top_idx, 1):
            actual_lbl = _decode(float(y_true[i]))
            pred_lbl = _decode(float(y_pred[i]))
            row = {
                "actual": actual_lbl,
                "predicted": pred_lbl,
                "error": f"predicted {pred_lbl}, actually {actual_lbl}",
                "abs_error": None,
                "rank": rank,
            }
            if feature_rows and i < len(feature_rows):
                row["features"] = feature_rows[i]
            errors.append(row)

        acc = round(1.0 - error_rate, 4)
        summary = (
            f"The model made {total_errors} incorrect predictions out of {total} "
            f"training rows ({error_rate:.0%} error rate, {acc:.0%} accuracy). "
            f"Showing {len(errors)} misclassified rows. "
            "Check if they share common feature values — that may indicate a "
            "segment where the model needs more training data."
        )
        return {
            "errors": errors,
            "total_errors": total_errors,
            "error_rate": error_rate,
            "problem_type": problem_type,
            "summary": summary,
        }


def compute_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    problem_type: str,
    n_bins: int = 10,
    target_classes: list | None = None,
) -> dict:
    """Compute the distribution of prediction errors across all training rows.

    For regression: histogram of residuals (actual - predicted), bias stats, shape.
    For classification: per-class error rates showing which classes the model struggles with.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions (same order as y_true).
        problem_type: "regression" or "classification".
        n_bins: Histogram bin count for regression (5-30, default 10).
        target_classes: Class label list for classification decoding.

    Returns:
        dict with bins, stats, problem_type, summary.
    """
    n_bins = max(5, min(30, n_bins))
    total = len(y_true)

    if total == 0:
        return {
            "problem_type": problem_type,
            "bins": [],
            "stats": {},
            "summary": "No training data available.",
        }

    if problem_type == "regression":
        residuals = y_true - y_pred
        mean_res = float(np.mean(residuals))
        std_res = float(np.std(residuals))
        mae = float(np.mean(np.abs(residuals)))
        # Adaptive bin count: fewer bins for small datasets
        actual_bins = min(n_bins, max(5, total // 10))
        counts, edges = np.histogram(residuals, bins=actual_bins)

        bins = []
        for i in range(len(counts)):
            lo = float(edges[i])
            hi = float(edges[i + 1])
            pct = round(float(counts[i]) / total * 100, 1)
            bins.append(
                {
                    "lo": round(lo, 4),
                    "hi": round(hi, 4),
                    "count": int(counts[i]),
                    "pct": pct,
                    "label": f"{lo:.2f} to {hi:.2f}",
                }
            )

        # Bias: if mean residual is far from 0, model is systematically over/under-predicting
        y_range = float(np.ptp(y_true)) if np.ptp(y_true) != 0 else 1.0
        bias_pct = abs(mean_res) / y_range * 100
        if bias_pct < 2:
            bias_label = "unbiased"
            bias_desc = (
                "The model has no systematic over- or under-prediction tendency."
            )
        elif mean_res > 0:
            bias_label = "over-predicts"
            bias_desc = (
                f"The model tends to over-predict by {bias_pct:.1f}% of the data range "
                f"on average (mean residual = +{mean_res:.3f})."
            )
        else:
            bias_label = "under-predicts"
            bias_desc = (
                f"The model tends to under-predict by {bias_pct:.1f}% of the data range "
                f"on average (mean residual = {mean_res:.3f})."
            )

        # Shape: are most errors small (tight) or spread out?
        within_1std = int(np.sum(np.abs(residuals) <= std_res))
        within_1std_pct = round(within_1std / total * 100, 1)

        summary = (
            f"Residual distribution across {total} training rows: "
            f"MAE = {mae:.3f}, std = {std_res:.3f}. "
            f"{within_1std_pct}% of errors fall within ±{std_res:.3f}. "
            f"{bias_desc}"
        )

        stats = {
            "mean": round(mean_res, 4),
            "std": round(std_res, 4),
            "mae": round(mae, 4),
            "bias_label": bias_label,
            "bias_pct": round(bias_pct, 2),
            "within_1std_pct": within_1std_pct,
            "total": total,
        }

        return {
            "problem_type": problem_type,
            "bins": bins,
            "stats": stats,
            "summary": summary,
        }

    else:
        # Classification: per-class error rates
        def _decode(v: float) -> str:
            idx = int(round(v))
            if target_classes and 0 <= idx < len(target_classes):
                return str(target_classes[idx])
            return str(idx)

        y_true_labels = [_decode(v) for v in y_true]
        y_pred_labels = [_decode(v) for v in y_pred]

        # Build per-class metrics
        unique_classes = sorted(set(y_true_labels))
        class_rows = []
        for cls in unique_classes:
            true_mask = [t == cls for t in y_true_labels]
            cls_total = sum(true_mask)
            cls_wrong = sum(
                1
                for i, is_this in enumerate(true_mask)
                if is_this and y_pred_labels[i] != cls
            )
            cls_error_rate = round(cls_wrong / cls_total, 4) if cls_total > 0 else 0.0
            class_rows.append(
                {
                    "class": cls,
                    "total": cls_total,
                    "wrong": cls_wrong,
                    "error_rate": cls_error_rate,
                    "error_pct": round(cls_error_rate * 100, 1),
                }
            )

        # Sort highest error first
        class_rows.sort(key=lambda r: r["error_rate"], reverse=True)

        total_wrong = sum(r["wrong"] for r in class_rows)
        overall_error_rate = round(total_wrong / total, 4) if total > 0 else 0.0
        overall_accuracy = round(1 - overall_error_rate, 4)

        worst = class_rows[0] if class_rows else None
        best = class_rows[-1] if class_rows else None

        if worst and best and worst["class"] != best["class"]:
            summary = (
                f"The model is {overall_accuracy:.0%} accurate across {total} training rows. "
                f"Hardest class: '{worst['class']}' ({worst['error_pct']:.0f}% errors). "
                f"Easiest class: '{best['class']}' ({best['error_pct']:.0f}% errors). "
                "Large gaps between classes may indicate data imbalance or feature gaps."
            )
        else:
            summary = (
                f"The model is {overall_accuracy:.0%} accurate across {total} training rows "
                f"({total_wrong} misclassifications)."
            )

        stats = {
            "total": total,
            "total_wrong": total_wrong,
            "overall_error_rate": overall_error_rate,
            "overall_accuracy": overall_accuracy,
            "n_classes": len(unique_classes),
        }

        return {
            "problem_type": problem_type,
            "bins": [],
            "class_breakdown": class_rows,
            "stats": stats,
            "summary": summary,
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


# ---------------------------------------------------------------------------
# Fairness / bias analysis
# ---------------------------------------------------------------------------


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_values: np.ndarray,
    problem_type: str,
) -> dict:
    """Compute fairness metrics across groups defined by a sensitive column.

    For classification:
      - Statistical Parity Difference (SPD): difference in positive prediction rates
        between the most-favoured and least-favoured group.  |SPD| < 0.1 → fair.
      - Disparate Impact Ratio (DIR): ratio of positive-prediction rates.
        0.8–1.2 → fair (the "4/5ths rule").
      - Per-group accuracy.

    For regression:
      - Per-group MAE.
      - MAE Disparity: ratio of max-group MAE to min-group MAE.
        < 1.25 → fair; 1.25–1.5 → warning; ≥ 1.5 → biased.

    Returns a dict with per_group_metrics, spd/dir (classification) or mae_disparity
    (regression), overall_status, and a plain-English summary.
    """
    groups = {}
    for val, yt, yp in zip(sensitive_values.tolist(), y_true.tolist(), y_pred.tolist()):
        key = str(val)
        groups.setdefault(key, {"y_true": [], "y_pred": []})
        groups[key]["y_true"].append(yt)
        groups[key]["y_pred"].append(yp)

    if len(groups) < 2:
        return {
            "sensitive_col": None,
            "groups": list(groups.keys()),
            "per_group_metrics": [],
            "overall_status": "insufficient_data",
            "summary": (
                "Need at least 2 distinct groups in the sensitive column to check fairness."
            ),
        }

    per_group: list[dict] = []

    if problem_type == "classification":
        # Determine positive label globally so all groups use the same reference
        all_preds = np.concatenate([np.array(d["y_pred"]) for d in groups.values()])
        _global_unique = np.unique(all_preds)
        pos_label = 1 if 1 in _global_unique else _global_unique[-1]

        for grp_name, data in sorted(groups.items()):
            yt_arr = np.array(data["y_true"])
            yp_arr = np.array(data["y_pred"])
            pos_rate = float(np.mean(yp_arr == pos_label))
            accuracy = float(np.mean(yt_arr == yp_arr))
            per_group.append(
                {
                    "group": grp_name,
                    "count": len(yt_arr),
                    "positive_rate": round(pos_rate, 4),
                    "accuracy": round(accuracy, 4),
                }
            )

        rates = [g["positive_rate"] for g in per_group]
        max_rate = max(rates)
        min_rate = min(rates)
        spd = round(max_rate - min_rate, 4)
        dir_val = round(min_rate / max_rate, 4) if max_rate > 0 else 1.0

        if abs(spd) < 0.1 and 0.8 <= dir_val <= 1.2:
            overall_status = "fair"
        elif abs(spd) < 0.2 and 0.7 <= dir_val <= 1.3:
            overall_status = "warning"
        else:
            overall_status = "biased"

        spd_label = _fairness_spd_label(spd)
        dir_label = _fairness_dir_label(dir_val)

        summary = _fairness_classification_summary(
            per_group, spd, dir_val, spd_label, dir_label, overall_status
        )

        return {
            "problem_type": problem_type,
            "groups": [g["group"] for g in per_group],
            "per_group_metrics": per_group,
            "spd": spd,
            "spd_label": spd_label,
            "dir": dir_val,
            "dir_label": dir_label,
            "overall_status": overall_status,
            "summary": summary,
        }

    else:
        # Regression: per-group MAE
        for grp_name, data in sorted(groups.items()):
            yt_arr = np.array(data["y_true"], dtype=float)
            yp_arr = np.array(data["y_pred"], dtype=float)
            mae = float(np.mean(np.abs(yt_arr - yp_arr)))
            per_group.append(
                {
                    "group": grp_name,
                    "count": len(yt_arr),
                    "mae": round(mae, 4),
                }
            )

        maes = [g["mae"] for g in per_group]
        max_mae = max(maes)
        min_mae = min(maes)
        # When all groups have zero MAE (perfect predictions), disparity = 1.0 (fair)
        mae_disparity = (
            round(max_mae / min_mae, 4)
            if min_mae > 0
            else (1.0 if max_mae == 0 else float("inf"))
        )

        if mae_disparity < 1.25:
            overall_status = "fair"
        elif mae_disparity < 1.5:
            overall_status = "warning"
        else:
            overall_status = "biased"

        summary = _fairness_regression_summary(per_group, mae_disparity, overall_status)

        return {
            "problem_type": problem_type,
            "groups": [g["group"] for g in per_group],
            "per_group_metrics": per_group,
            "mae_disparity": mae_disparity,
            "overall_status": overall_status,
            "summary": summary,
        }


def _fairness_spd_label(spd: float) -> str:
    """Plain-English label for Statistical Parity Difference."""
    if abs(spd) < 0.1:
        return "fair"
    if abs(spd) < 0.2:
        return "slight disparity"
    if abs(spd) < 0.3:
        return "moderate disparity"
    return "significant disparity"


def _fairness_dir_label(dir_val: float) -> str:
    """Plain-English label for Disparate Impact Ratio (4/5ths rule)."""
    if 0.8 <= dir_val <= 1.2:
        return "passes 4/5ths rule"
    if 0.7 <= dir_val <= 1.3:
        return "borderline"
    return "fails 4/5ths rule"


def _fairness_classification_summary(
    per_group: list[dict],
    spd: float,
    dir_val: float,
    spd_label: str,
    dir_label: str,
    overall_status: str,
) -> str:
    groups_desc = ", ".join(
        f"'{g['group']}' ({g['positive_rate']:.0%} positive rate, {g['accuracy']:.0%} accuracy)"
        for g in per_group[:5]
    )

    if overall_status == "fair":
        verdict = (
            "Your model appears fair across groups — prediction rates are similar."
        )
    elif overall_status == "warning":
        verdict = (
            "Minor disparity detected — worth monitoring, "
            "but likely within acceptable limits."
        )
    else:
        verdict = (
            "Significant disparity detected. "
            "The model may be treating groups unequally. "
            "Consider collecting more balanced training data or applying re-weighting."
        )

    return (
        f"Groups: {groups_desc}. "
        f"SPD = {spd:.3f} ({spd_label}). "
        f"DIR = {dir_val:.3f} ({dir_label}). "
        f"{verdict}"
    )


def _fairness_regression_summary(
    per_group: list[dict],
    mae_disparity: float,
    overall_status: str,
) -> str:
    worst = max(per_group, key=lambda g: g["mae"])
    best = min(per_group, key=lambda g: g["mae"])

    groups_desc = ", ".join(
        f"'{g['group']}' (MAE {g['mae']:.4f})" for g in per_group[:5]
    )

    if overall_status == "fair":
        verdict = "Error rates are consistent across groups — no significant disparity."
    elif overall_status == "warning":
        verdict = (
            f"The model is {mae_disparity:.1f}× less accurate for '{worst['group']}' "
            f"than '{best['group']}'. Monitor this as data grows."
        )
    else:
        verdict = (
            f"The model is {mae_disparity:.1f}× less accurate for '{worst['group']}' "
            f"than '{best['group']}'. "
            "This bias may disadvantage certain groups — consider re-balancing training data."
        )

    return f"MAE by group: {groups_desc}. MAE disparity ratio: {mae_disparity:.2f}. {verdict}"


# ---------------------------------------------------------------------------
# Input validation rules — pure function, no DB dependencies
# ---------------------------------------------------------------------------


def validate_prediction_inputs(
    inputs: dict,
    rules: list[dict],
) -> tuple[bool, list[dict]]:
    """Check prediction inputs against a list of validation rules.

    Args:
        inputs: dict of feature_name → value from the prediction request.
        rules: list of dicts with keys: feature_name, rule_type,
               min_val (float|None), max_val (float|None),
               allowed_values (list[str]|None).

    Returns:
        (is_valid, violations) where violations is a list of
        {feature_name, value, rule_type, message} dicts.
    """
    import json as _json

    violations: list[dict] = []

    for rule in rules:
        feature = rule["feature_name"]
        rule_type = rule["rule_type"]
        value = inputs.get(feature)

        if rule_type == "not_null":
            if value is None or (isinstance(value, str) and value.strip() == ""):
                violations.append(
                    {
                        "feature_name": feature,
                        "value": value,
                        "rule_type": rule_type,
                        "message": f"'{feature}' is required but was not provided.",
                    }
                )

        elif rule_type == "range":
            min_val = rule.get("min_val")
            max_val = rule.get("max_val")
            try:
                numeric = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                violations.append(
                    {
                        "feature_name": feature,
                        "value": value,
                        "rule_type": rule_type,
                        "message": (
                            f"'{feature}' must be a number"
                            + (
                                f" between {min_val} and {max_val}"
                                if min_val is not None and max_val is not None
                                else ""
                            )
                            + f" — received '{value}'."
                        ),
                    }
                )
                continue

            lo_ok = min_val is None or numeric >= min_val
            hi_ok = max_val is None or numeric <= max_val
            if not (lo_ok and hi_ok):
                bounds = (
                    f"between {min_val} and {max_val}"
                    if min_val is not None and max_val is not None
                    else (f"≥ {min_val}" if min_val is not None else f"≤ {max_val}")
                )
                violations.append(
                    {
                        "feature_name": feature,
                        "value": value,
                        "rule_type": rule_type,
                        "message": (
                            f"'{feature}' must be {bounds} — received {numeric}."
                        ),
                    }
                )

        elif rule_type == "one_of":
            allowed_raw = rule.get("allowed_values")
            if allowed_raw:
                allowed: list[str] = (
                    _json.loads(allowed_raw)
                    if isinstance(allowed_raw, str)
                    else list(allowed_raw)
                )
                str_value = str(value).strip() if value is not None else ""
                if str_value not in [str(a).strip() for a in allowed]:
                    allowed_display = ", ".join(f"'{a}'" for a in allowed[:6])
                    if len(allowed) > 6:
                        allowed_display += f", … ({len(allowed)} total)"
                    violations.append(
                        {
                            "feature_name": feature,
                            "value": value,
                            "rule_type": rule_type,
                            "message": (
                                f"'{feature}' must be one of: {allowed_display} "
                                f"— received '{value}'."
                            ),
                        }
                    )

    return len(violations) == 0, violations


# ---------------------------------------------------------------------------
# Classification threshold analysis
# ---------------------------------------------------------------------------


def compute_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """Sweep probability thresholds and return precision/recall/F1 trade-off data.

    y_proba should be the probability for the positive class (binary) or the
    maximum class probability (multiclass — used as a confidence proxy).

    Returns a dict with:
      sweep          — list of {threshold, precision, recall, f1, positive_rate}
      current_threshold — 0.5 (the default used by most classifiers)
      recommendations — dict with max_f1 / high_recall / high_precision options
      summary         — plain-English description of the trade-off situation
      n_positive      — count of true positive labels in y_true
      n_total         — total number of samples
      positive_class  — name of the positive class (from class_names)
    """
    if len(y_true) < 10:
        raise ValueError("Need at least 10 samples to compute threshold analysis.")

    # Determine positive label: use 1 (binary) or class with highest count if int-encoded
    unique_labels = sorted(set(y_true.tolist()))
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 classes to compute threshold analysis.")

    # For multiclass, we treat the most-common class as negative, least-common as positive
    # For binary, positive label = 1 if present, else the higher label
    if 1 in unique_labels:
        positive_label = 1
    else:
        # Use the second label (assumes binary after encoding: 0/1 or 0/2 etc.)
        positive_label = unique_labels[-1]

    pos_class_name = (
        class_names[positive_label]
        if class_names and positive_label < len(class_names)
        else str(positive_label)
    )

    y_bin = (np.array(y_true) == positive_label).astype(int)
    n_positive = int(y_bin.sum())
    n_total = len(y_bin)
    prevalence = n_positive / n_total

    thresholds = [round(t * 0.05, 2) for t in range(1, 20)]  # 0.05 … 0.95

    sweep: list[dict] = []
    for thr in thresholds:
        y_pred_bin = (np.array(y_proba) >= thr).astype(int)
        tp = int(((y_pred_bin == 1) & (y_bin == 1)).sum())
        fp = int(((y_pred_bin == 1) & (y_bin == 0)).sum())
        fn = int(((y_pred_bin == 0) & (y_bin == 1)).sum())
        pred_pos = tp + fp
        actual_pos = tp + fn

        prec = tp / pred_pos if pred_pos > 0 else 0.0
        rec = tp / actual_pos if actual_pos > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        pos_rate = pred_pos / n_total

        sweep.append(
            {
                "threshold": thr,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "positive_rate": round(pos_rate, 4),
            }
        )

    # Find key thresholds
    best_f1 = max(sweep, key=lambda p: p["f1"])
    high_recall = next(
        (p for p in sweep if p["recall"] >= 0.80),
        sweep[0],
    )
    high_precision = next(
        (p for p in reversed(sweep) if p["precision"] >= 0.80),
        sweep[-1],
    )

    # Current threshold (0.50) metrics
    current_point = next((p for p in sweep if p["threshold"] == 0.50), sweep[9])

    recommendations = {
        "max_f1": {
            "threshold": best_f1["threshold"],
            "precision": best_f1["precision"],
            "recall": best_f1["recall"],
            "f1": best_f1["f1"],
            "label": "Balanced (Max F1)",
            "description": (
                f"At {int(best_f1['threshold'] * 100)}%, precision and recall are most balanced. "
                "Best when false positives and false negatives are equally costly."
            ),
        },
        "high_recall": {
            "threshold": high_recall["threshold"],
            "precision": high_recall["precision"],
            "recall": high_recall["recall"],
            "f1": high_recall["f1"],
            "label": "High Recall (Catch More)",
            "description": (
                f"At {int(high_recall['threshold'] * 100)}%, you catch "
                f"{int(high_recall['recall'] * 100)}% of actual positives. "
                "Best when missing a positive case is expensive (e.g., fraud, churn)."
            ),
        },
        "high_precision": {
            "threshold": high_precision["threshold"],
            "precision": high_precision["precision"],
            "recall": high_precision["recall"],
            "f1": high_precision["f1"],
            "label": "High Precision (Fewer False Alarms)",
            "description": (
                f"At {int(high_precision['threshold'] * 100)}%, "
                f"{int(high_precision['precision'] * 100)}% of flagged cases are truly positive. "
                "Best when acting on a false alarm is expensive (e.g., costly interventions)."
            ),
        },
    }

    # Current-threshold plain-English
    current_prec_pct = int(current_point["precision"] * 100)
    current_rec_pct = int(current_point["recall"] * 100)
    best_thr_pct = int(best_f1["threshold"] * 100)

    if abs(best_f1["threshold"] - 0.50) < 0.001:
        thr_advice = "the default 50% threshold is already optimal for this dataset."
    elif best_f1["threshold"] < 0.50:
        thr_advice = f"consider lowering your threshold to {best_thr_pct}% to improve overall performance."
    else:
        thr_advice = f"consider raising your threshold to {best_thr_pct}% to improve overall performance."

    summary = (
        f"At the default 50% threshold: {current_prec_pct}% precision, "
        f"{current_rec_pct}% recall. "
        f"The best F1 score is at {best_thr_pct}% — {thr_advice} "
        f"The dataset has {prevalence:.0%} true positives ({n_positive} of {n_total} rows)."
    )

    return {
        "sweep": sweep,
        "current_threshold": 0.50,
        "current_metrics": current_point,
        "recommendations": recommendations,
        "n_positive": n_positive,
        "n_total": n_total,
        "prevalence": round(prevalence, 4),
        "positive_class": pos_class_name,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Confidence distribution analysis
# ---------------------------------------------------------------------------


def compute_confidence_distribution(
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    n_bins: int = 10,
) -> dict:
    """Compute the distribution of model prediction confidences across training data.

    y_proba: max-class probability per row (shape [n_samples]).
    y_pred:  predicted class labels (encoded integers).
    class_names: optional list of decoded class labels.

    Returns a dict with:
      bins           — list of {lo, hi, count, pct, label} histogram bars
      n_total        — total number of samples
      mean_confidence  — overall mean of max-class probabilities
      median_confidence
      pct_high       — % of predictions with confidence >= 0.80 (decisive)
      pct_medium     — % of predictions with confidence 0.50..0.80
      pct_low        — % of predictions with confidence < 0.50
      decisiveness   — "decisive" / "moderate" / "uncertain"
      per_class_mean — list of {class_name, mean_confidence, count}
      summary        — plain-English description
    """
    if len(y_proba) < 5:
        raise ValueError("Need at least 5 samples to compute confidence distribution.")

    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = np.asarray(y_pred)
    n_total = len(y_proba)
    n_bins = max(5, min(20, n_bins))

    # Histogram from 0.0 to 1.0
    counts, edges = np.histogram(y_proba, bins=n_bins, range=(0.0, 1.0))
    bins: list[dict] = []
    for i in range(n_bins):
        lo = round(float(edges[i]), 4)
        hi = round(float(edges[i + 1]), 4)
        count = int(counts[i])
        pct = round(count / n_total * 100, 1)
        label = f"{int(lo * 100)}–{int(hi * 100)}%"
        bins.append({"lo": lo, "hi": hi, "count": count, "pct": pct, "label": label})

    mean_conf = round(float(np.mean(y_proba)), 4)
    median_conf = round(float(np.median(y_proba)), 4)

    n_high = int(np.sum(y_proba >= 0.80))
    n_medium = int(np.sum((y_proba >= 0.50) & (y_proba < 0.80)))
    n_low = int(np.sum(y_proba < 0.50))

    pct_high = round(n_high / n_total * 100, 1)
    pct_medium = round(n_medium / n_total * 100, 1)
    pct_low = round(n_low / n_total * 100, 1)

    # Decisiveness verdict
    if pct_high >= 60:
        decisiveness = "decisive"
        decisiveness_label = "Decisive"
        decisiveness_desc = (
            f"{pct_high:.0f}% of predictions have confidence ≥ 80% — "
            "the model is confident and decisive in most cases."
        )
    elif pct_high >= 30:
        decisiveness = "moderate"
        decisiveness_label = "Moderate"
        decisiveness_desc = (
            f"{pct_high:.0f}% of predictions have confidence ≥ 80%. "
            "The model has moderate decisiveness — "
            "many predictions fall in the uncertain middle range."
        )
    else:
        decisiveness = "uncertain"
        decisiveness_label = "Uncertain"
        decisiveness_desc = (
            f"Only {pct_high:.0f}% of predictions have confidence ≥ 80%. "
            "The model is often uncertain — predictions tend to cluster near 50%, "
            "which may indicate class overlap in the training data."
        )

    # Per-class mean confidence
    unique_classes = sorted(set(y_pred.tolist()))
    per_class_mean: list[dict] = []
    for cls in unique_classes:
        mask = y_pred == cls
        cls_proba = y_proba[mask]
        cls_name = (
            class_names[int(cls)]
            if class_names and int(cls) < len(class_names)
            else str(cls)
        )
        per_class_mean.append(
            {
                "class_name": cls_name,
                "mean_confidence": (
                    round(float(np.mean(cls_proba)), 4) if len(cls_proba) > 0 else 0.0
                ),
                "count": int(mask.sum()),
            }
        )

    summary = (
        f"The model has mean confidence of {mean_conf:.0%} across {n_total} training predictions. "
        f"{decisiveness_desc} "
        f"{pct_high:.0f}% high-confidence (≥80%), "
        f"{pct_medium:.0f}% medium (50–80%), "
        f"{pct_low:.0f}% low (<50%)."
    )

    return {
        "bins": bins,
        "n_total": n_total,
        "n_bins": n_bins,
        "mean_confidence": mean_conf,
        "median_confidence": median_conf,
        "pct_high": pct_high,
        "pct_medium": pct_medium,
        "pct_low": pct_low,
        "n_high": n_high,
        "n_medium": n_medium,
        "n_low": n_low,
        "decisiveness": decisiveness,
        "decisiveness_label": decisiveness_label,
        "per_class_mean": per_class_mean,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Calibration check
# ---------------------------------------------------------------------------


def compute_calibration_check(
    y_true: np.ndarray,
    y_proba_matrix: np.ndarray,
    class_names: list[str] | None = None,
    n_bins: int = 10,
) -> dict:
    """Check how well the model's predicted probabilities match actual outcome rates.

    A well-calibrated model that says "70% probability" is correct ~70% of the time.

    y_true:         true labels (integer-encoded).
    y_proba_matrix: full predict_proba output, shape [n_samples, n_classes].
    class_names:    optional decoded class labels.
    n_bins:         number of calibration bins (5–20).

    Returns a dict with:
      curve_points   — list of {mean_prob, frac_positive, count, bin_label} per bin
      ece            — Expected Calibration Error (0 = perfect, closer to 0 = better)
      brier_score    — mean squared error of predicted probabilities vs actual labels
      verdict        — "well_calibrated" / "moderate" / "poorly_calibrated"
      verdict_label  — human-readable label
      per_class      — list of {class_name, brier_score, ece, n_positive} (multiclass)
      n_classes      — number of classes
      n_total        — number of samples
      summary        — plain-English description
    """
    y_true = np.asarray(y_true)
    y_proba_matrix = np.asarray(y_proba_matrix, dtype=float)

    if len(y_true) < 10:
        raise ValueError("Need at least 10 samples to compute calibration check.")

    n_classes = y_proba_matrix.shape[1] if y_proba_matrix.ndim == 2 else 1
    n_total = len(y_true)
    n_bins = max(5, min(20, n_bins))

    classes = sorted(set(y_true.tolist()))
    if class_names is None or len(class_names) != n_classes:
        class_names = [str(c) for c in classes]

    # For binary: use positive-class probability column
    # For multiclass: use max-class probability (one-vs-rest approximation for ECE)
    if n_classes == 2:
        pos_class_idx = 1
        y_prob_pos = y_proba_matrix[:, pos_class_idx]
        y_binary = (y_true == classes[pos_class_idx]).astype(int)

        frac_pos, mean_prob = calibration_curve(
            y_binary, y_prob_pos, n_bins=n_bins, strategy="uniform"
        )
        # Count samples in each bin
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(y_prob_pos, bin_edges[1:-1])
        curve_points: list[dict] = []
        for i, (fp, mp) in enumerate(zip(frac_pos, mean_prob)):
            count = int(np.sum(bin_indices == i))
            lo = round(bin_edges[i], 2)
            hi = round(bin_edges[i + 1], 2)
            curve_points.append(
                {
                    "mean_prob": round(float(mp), 4),
                    "frac_positive": round(float(fp), 4),
                    "count": count,
                    "bin_label": f"{int(lo * 100)}–{int(hi * 100)}%",
                }
            )

        # ECE = weighted average of |predicted_prob - actual_fraction|
        bin_weights = np.array([p["count"] for p in curve_points], dtype=float)
        bin_weights /= bin_weights.sum() if bin_weights.sum() > 0 else 1
        ece = float(
            np.sum(
                bin_weights
                * np.abs(
                    np.array([p["mean_prob"] for p in curve_points])
                    - np.array([p["frac_positive"] for p in curve_points])
                )
            )
        )
        overall_brier = float(brier_score_loss(y_binary, y_prob_pos))

        per_class = [
            {
                "class_name": class_names[pos_class_idx],
                "brier_score": round(overall_brier, 4),
                "ece": round(ece, 4),
                "n_positive": int(np.sum(y_binary)),
            }
        ]
    else:
        # Multiclass: aggregate calibration via one-vs-rest per class
        all_ece: list[float] = []
        all_brier: list[float] = []
        curve_points = []
        per_class = []

        # Build curve points using the "confidence" approach: max-class probability
        y_conf = y_proba_matrix.max(axis=1)
        y_correct = (
            np.array(
                [
                    1 if y_true[i] == classes[np.argmax(y_proba_matrix[i])] else 0
                    for i in range(n_total)
                ]
            )
        )
        try:
            frac_pos_mc, mean_prob_mc = calibration_curve(
                y_correct, y_conf, n_bins=n_bins, strategy="uniform"
            )
        except ValueError:
            frac_pos_mc = np.array([float(np.mean(y_correct))])
            mean_prob_mc = np.array([float(np.mean(y_conf))])

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(y_conf, bin_edges[1:-1])
        for i, (fp, mp) in enumerate(zip(frac_pos_mc, mean_prob_mc)):
            count = int(np.sum(bin_indices == i))
            lo = round(bin_edges[i], 2)
            hi = round(bin_edges[i + 1], 2)
            curve_points.append(
                {
                    "mean_prob": round(float(mp), 4),
                    "frac_positive": round(float(fp), 4),
                    "count": count,
                    "bin_label": f"{int(lo * 100)}–{int(hi * 100)}%",
                }
            )

        # Per-class one-vs-rest brier and ECE
        for k, cls in enumerate(classes):
            cls_name = class_names[k] if k < len(class_names) else str(cls)
            y_bin = (y_true == cls).astype(int)
            y_prob_k = y_proba_matrix[:, k]
            b = float(brier_score_loss(y_bin, y_prob_k))
            try:
                fp_k, mp_k = calibration_curve(y_bin, y_prob_k, n_bins=n_bins, strategy="uniform")
                bw = np.ones(len(fp_k)) / len(fp_k)
                e_k = float(np.sum(bw * np.abs(mp_k - fp_k)))
            except ValueError:
                e_k = 0.0
            all_ece.append(e_k)
            all_brier.append(b)
            per_class.append(
                {
                    "class_name": cls_name,
                    "brier_score": round(b, 4),
                    "ece": round(e_k, 4),
                    "n_positive": int(np.sum(y_bin)),
                }
            )

        # Weighted ECE and aggregate brier
        class_counts = np.array([p["n_positive"] for p in per_class], dtype=float)
        weights = class_counts / class_counts.sum() if class_counts.sum() > 0 else np.ones(len(per_class)) / len(per_class)
        ece = float(np.sum(weights * np.array(all_ece)))
        overall_brier = float(np.mean(all_brier))

    ece_r = round(ece, 4)
    brier_r = round(overall_brier, 4)

    if ece_r < 0.05:
        verdict = "well_calibrated"
        verdict_label = "Well-Calibrated"
        verdict_desc = (
            f"ECE {ece_r:.3f} — the model's probability scores closely match actual outcome rates. "
            "When it says 80%, it's correct about 80% of the time."
        )
    elif ece_r < 0.15:
        verdict = "moderate"
        verdict_label = "Moderately Calibrated"
        verdict_desc = (
            f"ECE {ece_r:.3f} — the model's probabilities are roughly reliable but noticeably off in some ranges. "
            "Use probability scores as directional guidance, not precise estimates."
        )
    else:
        verdict = "poorly_calibrated"
        verdict_label = "Poorly Calibrated"
        verdict_desc = (
            f"ECE {ece_r:.3f} — the model's probability scores do not reliably match actual outcome rates. "
            "Avoid using the raw probability values for business decisions. "
            "Consider probability calibration (Platt scaling or isotonic regression)."
        )

    brier_guidance = (
        "closer to 0 is better"
        if brier_r < 0.25
        else "a score above 0.25 suggests poor probability estimates"
    )
    summary = (
        f"{verdict_label}. {verdict_desc} "
        f"Brier score: {brier_r:.3f} ({brier_guidance}). "
        f"Computed on {n_total} training samples across {n_classes} {'class' if n_classes == 1 else 'classes'}."
    )

    return {
        "curve_points": curve_points,
        "ece": ece_r,
        "brier_score": brier_r,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "per_class": per_class,
        "n_classes": n_classes,
        "n_total": n_total,
        "summary": summary,
    }
