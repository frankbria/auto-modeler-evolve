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
# Per-class threshold analysis (multiclass one-vs-rest sweep)
# ---------------------------------------------------------------------------


def compute_per_class_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """Independently sweep confidence thresholds for each class using one-vs-rest.

    y_true  — integer-encoded true labels (shape: n_samples)
    y_proba — probability matrix (shape: n_samples × n_classes) from predict_proba
    class_names — optional human-readable names indexed by class integer

    Returns a dict with:
      classes    — list of per-class result dicts (see below)
      n_classes  — number of classes analysed
      n_samples  — total sample count
      summary    — plain-English summary of actionable threshold changes
    """
    if len(y_true) < 10:
        raise ValueError("Need at least 10 samples for per-class threshold analysis.")

    y_true_arr = np.array(y_true)
    y_proba_arr = np.array(y_proba)

    unique_classes = sorted(set(y_true_arr.tolist()))
    n_classes = len(unique_classes)
    if n_classes < 2:
        raise ValueError("Need at least 2 classes.")
    if y_proba_arr.ndim != 2 or y_proba_arr.shape[1] != n_classes:
        raise ValueError(
            f"y_proba must be (n_samples, {n_classes}); got shape {y_proba_arr.shape}."
        )

    thresholds = [round(t * 0.05, 2) for t in range(1, 20)]  # 0.05 … 0.95

    class_results: list[dict] = []
    n_actionable = 0

    for idx, cls_int in enumerate(unique_classes):
        cls_name = (
            class_names[cls_int]
            if class_names and cls_int < len(class_names)
            else str(cls_int)
        )
        y_bin = (y_true_arr == cls_int).astype(int)
        n_positive = int(y_bin.sum())
        prevalence = n_positive / len(y_bin)

        # Use the class column's probability for one-vs-rest
        col_idx = cls_int if cls_int < y_proba_arr.shape[1] else idx
        col_idx = min(col_idx, y_proba_arr.shape[1] - 1)
        scores = y_proba_arr[:, col_idx]

        sweep: list[dict] = []
        for thr in thresholds:
            y_pred_bin = (scores >= thr).astype(int)
            tp = int(((y_pred_bin == 1) & (y_bin == 1)).sum())
            fp = int(((y_pred_bin == 1) & (y_bin == 0)).sum())
            fn = int(((y_pred_bin == 0) & (y_bin == 1)).sum())
            pred_pos = tp + fp
            actual_pos = tp + fn

            prec = tp / pred_pos if pred_pos > 0 else 0.0
            rec = tp / actual_pos if actual_pos > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            pos_rate = pred_pos / len(y_bin)

            sweep.append(
                {
                    "threshold": thr,
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1": round(f1, 4),
                    "positive_rate": round(pos_rate, 4),
                }
            )

        best = max(sweep, key=lambda p: p["f1"])
        default_pt = next((p for p in sweep if p["threshold"] == 0.50), sweep[9])

        f1_gain = round(best["f1"] - default_pt["f1"], 4)
        is_actionable = abs(best["threshold"] - 0.50) >= 0.10 and f1_gain > 0.02

        if best["threshold"] > 0.50:
            direction = "raise"
            recommendation = (
                f"Raise to {int(best['threshold'] * 100)}%: "
                f"fewer false '{cls_name}' predictions, "
                f"+{int(f1_gain * 100)}pp F1 gain."
            )
        elif best["threshold"] < 0.50:
            direction = "lower"
            recommendation = (
                f"Lower to {int(best['threshold'] * 100)}%: "
                f"catch more '{cls_name}' cases, "
                f"+{int(f1_gain * 100)}pp F1 gain."
            )
        else:
            direction = "keep"
            recommendation = f"Default 50% is already optimal for '{cls_name}'."

        if is_actionable:
            n_actionable += 1

        class_results.append(
            {
                "class_name": cls_name,
                "class_index": cls_int,
                "n_positive": n_positive,
                "prevalence": round(prevalence, 4),
                "sweep": sweep,
                "optimal_threshold": best["threshold"],
                "optimal_f1": best["f1"],
                "optimal_precision": best["precision"],
                "optimal_recall": best["recall"],
                "default_f1": default_pt["f1"],
                "f1_gain": f1_gain,
                "direction": direction,
                "recommendation": recommendation,
                "is_actionable": is_actionable,
            }
        )

    # Summary
    actionable_classes = [c for c in class_results if c["is_actionable"]]
    if not actionable_classes:
        summary = (
            f"All {n_classes} classes perform near-optimally at the default 50% "
            "confidence threshold — no changes recommended."
        )
    else:
        names = ", ".join(f"'{c['class_name']}'" for c in actionable_classes[:3])
        summary = (
            f"{n_actionable} of {n_classes} classes benefit from threshold tuning: "
            f"{names}. "
            "Adjusting per-class thresholds can meaningfully improve precision or recall "
            "for those specific outcomes."
        )

    return {
        "classes": class_results,
        "n_classes": n_classes,
        "n_samples": len(y_true_arr),
        "n_actionable": n_actionable,
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
        y_correct = np.array(
            [
                1 if y_true[i] == classes[np.argmax(y_proba_matrix[i])] else 0
                for i in range(n_total)
            ]
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
                fp_k, mp_k = calibration_curve(
                    y_bin, y_prob_k, n_bins=n_bins, strategy="uniform"
                )
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
        weights = (
            class_counts / class_counts.sum()
            if class_counts.sum() > 0
            else np.ones(len(per_class)) / len(per_class)
        )
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


def compute_prediction_error_correlation(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
    problem_type: str,
) -> dict:
    """Identify which input features most correlate with prediction errors.

    For regression: error vector = |y_pred - y_true| (absolute residuals).
    For classification: error vector = (y_pred != y_true).astype(int).

    Returns a sorted list of features with Pearson correlation to the error
    signal, plus a plain-English verdict and recommendation.

    Returns:
        dict with keys: features (list of dicts with feature/correlation/
        correlation_abs/direction/rank), error_type, n_total, n_errors,
        error_rate, verdict, verdict_label, top_driver, summary.

    Raises:
        ValueError: if fewer than 10 samples.
    """
    MIN_ROWS = 10
    if len(X) < MIN_ROWS:
        raise ValueError(
            f"Dataset has only {len(X)} rows — need at least {MIN_ROWS} for "
            "error correlation analysis."
        )

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    if problem_type == "regression":
        error_vec = np.abs(y_pred_arr.astype(float) - y_true_arr.astype(float))
        error_type = "absolute_residual"
        n_errors = int(np.sum(error_vec > 0))
        error_rate = None
    else:
        # Encode both as ints to handle string labels
        y_true_enc = (y_true_arr != y_true_arr).astype(float)  # init zeros
        y_pred_enc = y_true_enc.copy()
        for i, (yt, yp) in enumerate(zip(y_true_arr, y_pred_arr)):
            y_pred_enc[i] = 0.0 if str(yt) == str(yp) else 1.0
        error_vec = y_pred_enc
        error_type = "misclassification"
        n_errors = int(np.sum(error_vec))
        error_rate = round(float(n_errors / len(error_vec)), 4)

    # Correlate each numeric feature column with the error vector
    results = []
    for i, fname in enumerate(feature_names):
        col = X[:, i].astype(float)
        col_std = float(np.std(col))
        err_std = float(np.std(error_vec))
        if col_std < 1e-9 or err_std < 1e-9:
            corr = 0.0
        else:
            try:
                corr = float(np.corrcoef(col, error_vec)[0, 1])
                if np.isnan(corr):
                    corr = 0.0
            except Exception:  # noqa: BLE001
                corr = 0.0

        direction = (
            "positive" if corr > 0.05 else ("negative" if corr < -0.05 else "neutral")
        )
        results.append(
            {
                "feature": fname,
                "correlation": round(corr, 4),
                "correlation_abs": round(abs(corr), 4),
                "direction": direction,
            }
        )

    # Sort by absolute correlation descending, cap at top 10
    results.sort(key=lambda x: x["correlation_abs"], reverse=True)
    results = results[:10]
    for rank_idx, r in enumerate(results, start=1):
        r["rank"] = rank_idx

    # Verdict based on max absolute correlation
    max_abs = results[0]["correlation_abs"] if results else 0.0
    if max_abs >= 0.30:
        verdict = "clear_drivers"
        verdict_label = "Clear Error Drivers Found"
    elif max_abs >= 0.10:
        verdict = "weak_drivers"
        verdict_label = "Weak Error Drivers"
    else:
        verdict = "none"
        verdict_label = "No Clear Error Drivers"

    top_driver = results[0]["feature"] if results else None
    top_corr = results[0]["correlation"] if results else 0.0

    if verdict == "clear_drivers" and top_driver:
        direction_phrase = (
            f"higher {top_driver} values are associated with larger errors"
            if top_corr > 0
            else f"lower {top_driver} values are associated with larger errors"
        )
        summary = (
            f"Feature '{top_driver}' most strongly correlates with prediction errors "
            f"(r={top_corr:.2f}). {direction_phrase.capitalize()}. "
            f"Consider adding more training samples for {top_driver} extremes or engineering "
            f"interaction features."
        )
    elif verdict == "weak_drivers" and top_driver:
        summary = (
            f"Weak error correlation found. '{top_driver}' shows the strongest link "
            f"(r={top_corr:.2f}) — insufficient to explain systematic errors."
        )
    else:
        summary = (
            "No input feature strongly correlates with prediction errors. "
            "Errors appear random rather than feature-driven — the model may benefit "
            "from more training data rather than feature engineering."
        )

    return {
        "features": results,
        "error_type": error_type,
        "n_total": len(y_true_arr),
        "n_errors": n_errors,
        "error_rate": error_rate,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "top_driver": top_driver,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# 13. Minimum viable feature set (greedy backward elimination)
# ---------------------------------------------------------------------------


def compute_min_viable_feature_set(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_class: Any,
    model_params: dict,
    problem_type: str,
    tolerance: float = 0.02,
    max_rows: int = 2000,
    cv: int = 3,
) -> dict:
    """Find the smallest subset of features that preserves model accuracy.

    Uses greedy backward elimination: removes the least-important feature
    at each step (by the trained model's importances) only if accuracy loss
    remains within ``tolerance``.  Sub-samples to ``max_rows`` for speed.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Target vector.
        feature_names: Column names for X.
        model_class: Unfitted sklearn estimator class.
        model_params: Constructor kwargs for model_class.
        problem_type: "regression" or "classification".
        tolerance: Maximum allowed accuracy drop (e.g., 0.02 = 2%).
        max_rows: Sub-sample threshold for speed.
        cv: Number of cross-validation folds.

    Returns:
        dict with keys: features (list of dicts), n_original, n_minimal,
        features_dropped, features_retained, features_dropped_list,
        baseline_score, minimal_score, score_loss, can_simplify,
        reduction_pct, metric, summary.

    Raises:
        ValueError: if fewer than 10 samples or fewer than 2 features.
    """
    MIN_ROWS = 10
    MIN_FEATURES = 2
    if len(X) < MIN_ROWS:
        raise ValueError(
            f"Dataset has only {len(X)} rows — need at least {MIN_ROWS} rows."
        )
    if X.shape[1] < MIN_FEATURES:
        raise ValueError(
            f"Need at least {MIN_FEATURES} features for elimination analysis."
        )

    # Sub-sample for speed
    n = min(len(X), max_rows)
    if n < len(X):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=n, replace=False)
        X_use = X[idx]
        y_use = y[idx]
    else:
        X_use = X
        y_use = y

    metric = "r2" if problem_type == "regression" else "accuracy"
    scoring = "r2" if problem_type == "regression" else "accuracy"

    def _score(X_sub: np.ndarray) -> float:
        m = model_class(**model_params)
        scores = cross_val_score(m, X_sub, y_use, cv=cv, scoring=scoring)
        return float(np.mean(scores))

    # Baseline: all features
    baseline_score = _score(X_use)

    # Get feature importances from a freshly-trained model (not CV)
    fitted = model_class(**model_params)
    fitted.fit(X_use, y_use)
    if hasattr(fitted, "feature_importances_"):
        importances = np.array(fitted.feature_importances_, dtype=float)
    elif hasattr(fitted, "coef_"):
        coef = np.array(fitted.coef_)
        importances = np.abs(coef.ravel())[: X_use.shape[1]]
    else:
        importances = np.ones(X_use.shape[1])

    # Build initial feature index list sorted by importance ascending
    n_feat = X_use.shape[1]
    feature_names_list = list(feature_names)
    remaining_idx = list(range(n_feat))

    # Greedy elimination: try removing the least-important remaining feature
    dropped_idx: list[int] = []
    current_score = baseline_score

    for _ in range(n_feat - 1):  # always keep at least 1 feature
        if len(remaining_idx) <= 1:
            break
        # Find least-important remaining feature
        imp_remaining = [(idx, importances[idx]) for idx in remaining_idx]
        imp_remaining.sort(key=lambda t: t[1])
        candidate_idx, _ = imp_remaining[0]

        # Trial score without the candidate
        trial_idx = [i for i in remaining_idx if i != candidate_idx]
        trial_score = _score(X_use[:, trial_idx])

        score_loss = baseline_score - trial_score
        if score_loss <= tolerance:
            dropped_idx.append(candidate_idx)
            remaining_idx = trial_idx
            current_score = trial_score
        else:
            break  # dropping this feature costs too much — stop

    # Build per-feature result list (original order)
    features_retained = [feature_names_list[i] for i in remaining_idx]
    features_dropped_list = [feature_names_list[i] for i in dropped_idx]

    feature_entries: list[dict] = []
    for rank_i, fname in enumerate(feature_names_list, start=1):
        imp_val = round(float(importances[rank_i - 1]), 6)
        kept = fname in features_retained
        feature_entries.append(
            {
                "feature": fname,
                "importance": imp_val,
                "kept": kept,
                "rank": rank_i,
            }
        )

    n_original = n_feat
    n_minimal = len(features_retained)
    n_dropped = len(features_dropped_list)
    score_loss_final = round(baseline_score - current_score, 4)
    reduction_pct = round(n_dropped / n_original * 100, 1) if n_original > 0 else 0.0
    can_simplify = n_dropped > 0

    # Plain-English summary
    metric_label = "R²" if metric == "r2" else "accuracy"
    bs_disp = f"{baseline_score:.3f}" if metric == "r2" else f"{baseline_score:.1%}"
    ms_disp = f"{current_score:.3f}" if metric == "r2" else f"{current_score:.1%}"

    if can_simplify:
        summary = (
            f"Your model can be simplified from {n_original} to {n_minimal} features "
            f"(dropping {n_dropped}, a {reduction_pct:.0f}% reduction) while keeping "
            f"{metric_label} at {ms_disp} (vs {bs_disp} with all features, "
            f"loss of {score_loss_final:.3f}). "
            f"Dropped: {', '.join(features_dropped_list[:5])}"
            + (f" and {len(features_dropped_list) - 5} more." if n_dropped > 5 else ".")
        )
    else:
        summary = (
            f"All {n_original} features are needed — removing any one of them causes "
            f"{metric_label} to drop by more than the {tolerance:.0%} tolerance. "
            f"Baseline {metric_label}: {bs_disp}."
        )

    return {
        "features": feature_entries,
        "n_original": n_original,
        "n_minimal": n_minimal,
        "features_dropped": n_dropped,
        "features_retained": features_retained,
        "features_dropped_list": features_dropped_list,
        "baseline_score": round(baseline_score, 4),
        "minimal_score": round(current_score, 4),
        "score_loss": score_loss_final,
        "can_simplify": can_simplify,
        "reduction_pct": reduction_pct,
        "metric": metric,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Production feedback threshold optimizer
# ---------------------------------------------------------------------------


def compute_production_threshold_optimizer(
    feedback_pairs: list[dict],
) -> dict:
    """Find the optimal confidence threshold using real production feedback outcomes.

    feedback_pairs: list of dicts with keys:
        - confidence: float (0.0–1.0, model confidence for this prediction)
        - predicted_label: str (what the model predicted)
        - actual_label: str (ground truth recorded by the analyst)

    Sweeps confidence thresholds from 0.05 to 0.95 and computes:
        - precision: accuracy among predictions served at that threshold
        - recall: fraction of all correct predictions that are served
        - f1: harmonic mean of precision and recall
        - coverage: fraction of total predictions that are served

    Returns a dict with sweep data and the optimal threshold (max F1).
    Raises ValueError when fewer than 5 feedback pairs are provided.
    """
    if len(feedback_pairs) < 5:
        raise ValueError(
            "Need at least 5 feedback pairs with confidence scores to "
            "compute production threshold optimization."
        )

    n_total = len(feedback_pairs)

    # Total correct predictions in the feedback set (denominator for recall)
    n_correct_total = sum(
        1
        for p in feedback_pairs
        if str(p.get("predicted_label", "")) == str(p.get("actual_label", ""))
    )

    thresholds = [round(t * 0.05, 2) for t in range(1, 20)]  # 0.05 … 0.95

    sweep: list[dict] = []
    for thr in thresholds:
        served = [p for p in feedback_pairs if (p.get("confidence") or 0.0) >= thr]
        n_served = len(served)
        n_correct_served = sum(
            1
            for p in served
            if str(p.get("predicted_label", "")) == str(p.get("actual_label", ""))
        )

        precision = n_correct_served / n_served if n_served > 0 else 0.0
        recall = n_correct_served / n_correct_total if n_correct_total > 0 else 0.0
        coverage = n_served / n_total
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        sweep.append(
            {
                "threshold": thr,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "coverage": round(coverage, 4),
                "n_served": n_served,
            }
        )

    best_point = max(sweep, key=lambda p: p["f1"])
    current_point = next(
        (p for p in sweep if abs(p["threshold"] - 0.50) < 0.001), sweep[9]
    )

    overall_accuracy = n_correct_total / n_total if n_total > 0 else 0.0

    if best_point["f1"] > current_point["f1"] + 0.01:
        verdict = "improved"
        verdict_label = "Optimal ≠ Default"
        if best_point["threshold"] > 0.50:
            thr_advice = (
                f"Raising the threshold from 50% to {int(best_point['threshold'] * 100)}% "
                f"improves accuracy by filtering out low-confidence predictions "
                f"(coverage: {int(best_point['coverage'] * 100)}% of predictions served)."
            )
        else:
            thr_advice = (
                f"Lowering the threshold to {int(best_point['threshold'] * 100)}% "
                f"increases coverage without sacrificing accuracy "
                f"({int(best_point['coverage'] * 100)}% of predictions served)."
            )
    else:
        verdict = "same"
        verdict_label = "Default Is Optimal"
        thr_advice = "The default 50% threshold is already near-optimal for your production data."

    summary = (
        f"Based on {n_total} real-world outcomes "
        f"(overall accuracy: {overall_accuracy:.0%}): "
        f"at threshold {int(best_point['threshold'] * 100)}%, "
        f"your model achieves F1={best_point['f1']:.2f} with "
        f"{int(best_point['coverage'] * 100)}% coverage. "
        f"{thr_advice}"
    )

    return {
        "sweep": sweep,
        "optimal_threshold": best_point["threshold"],
        "optimal_f1": best_point["f1"],
        "optimal_precision": best_point["precision"],
        "optimal_recall": best_point["recall"],
        "optimal_coverage": best_point["coverage"],
        "current_threshold": 0.50,
        "current_f1": current_point["f1"],
        "current_precision": current_point["precision"],
        "current_recall": current_point["recall"],
        "current_coverage": current_point["coverage"],
        "n_feedback": n_total,
        "n_correct_total": n_correct_total,
        "overall_accuracy": round(overall_accuracy, 4),
        "verdict": verdict,
        "verdict_label": verdict_label,
        "summary": summary,
    }
