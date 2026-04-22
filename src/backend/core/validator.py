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
            round(max_mae / min_mae, 4) if min_mae > 0 else (1.0 if max_mae == 0 else float("inf"))
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
        verdict = "Your model appears fair across groups — prediction rates are similar."
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
