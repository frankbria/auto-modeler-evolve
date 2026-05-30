"""Feature importance and individual prediction explanations.

Uses sklearn's built-in feature importances (no external SHAP dependency):
- Tree models (RandomForest, GradientBoosting): model.feature_importances_
- Linear models (LinearRegression, LogisticRegression): model.coef_

For individual predictions we compute a simple contribution score:
  contribution_i = feature_importance_i * (x_i - mean_i) / std_i

This is a linear approximation (not SHAP), but it's fast, interpretable,
and works without adding heavy dependencies.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Global feature importance
# ---------------------------------------------------------------------------


def compute_feature_importance(
    model,  # fitted sklearn estimator
    feature_names: list[str],
) -> list[dict]:
    """Return a ranked list of feature importances.

    Works with: RandomForestRegressor, RandomForestClassifier,
                GradientBoostingRegressor, GradientBoostingClassifier,
                LinearRegression, LogisticRegression.

    Returns:
        List of {feature, importance, rank} sorted descending by importance.
    """
    importances = _extract_importances(model, len(feature_names))

    if importances is None:
        # Fallback: equal importances
        importances = np.ones(len(feature_names)) / max(len(feature_names), 1)

    # Normalise to [0, 1]
    total = np.sum(np.abs(importances))
    if total > 0:
        importances = np.abs(importances) / total

    results = []
    for i, (name, imp) in enumerate(zip(feature_names, importances)):
        results.append(
            {
                "feature": name,
                "importance": round(float(imp), 6),
                "rank": 0,  # filled in below
            }
        )

    # Sort descending
    results.sort(key=lambda x: x["importance"], reverse=True)
    for rank, item in enumerate(results, start=1):
        item["rank"] = rank

    return results


def _extract_importances(model, n_features: int) -> np.ndarray | None:
    """Extract raw importance values from common sklearn model types."""
    # Tree-based: direct feature_importances_
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        return np.array(fi[:n_features])

    # Linear models: coef_
    if hasattr(model, "coef_"):
        coef = np.array(model.coef_)
        # LogisticRegression multiclass → coef_ shape (n_classes, n_features)
        if coef.ndim == 2:
            coef = np.mean(np.abs(coef), axis=0)
        return coef[:n_features]

    return None


# ---------------------------------------------------------------------------
# Individual prediction explanation
# ---------------------------------------------------------------------------


def explain_single_prediction(
    model,  # fitted sklearn estimator
    x_row: np.ndarray,  # shape (n_features,)
    X_train: np.ndarray,  # full training set, shape (n_samples, n_features)
    feature_names: list[str],
    problem_type: str,
    target_name: str = "target",
) -> dict:
    """Explain one prediction using feature contributions.

    Contribution formula (simple local linear attribution):
        contribution_i = importance_i * (x_i - mean_i) / (std_i + ε)

    Returns:
        {
          prediction: float | int,
          contributions: [{feature, value, contribution, direction}],
          summary: str,
        }
    """
    importances_list = compute_feature_importance(model, feature_names)
    imp_map = {item["feature"]: item["importance"] for item in importances_list}

    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0)

    # Make prediction
    x_2d = x_row.reshape(1, -1)
    if hasattr(model, "predict_proba") and problem_type == "classification":
        proba = model.predict_proba(x_2d)[0]
        prediction_val = float(np.max(proba))
        predicted_class = int(model.predict(x_2d)[0])
    else:
        prediction_val = float(model.predict(x_2d)[0])
        predicted_class = None

    # Compute contributions
    contributions = []
    for i, name in enumerate(feature_names):
        imp = imp_map.get(name, 0.0)
        std_i = float(stds[i]) if float(stds[i]) > 1e-10 else 1.0
        deviation = float(x_row[i] - means[i]) / std_i
        contrib = imp * deviation

        contributions.append(
            {
                "feature": name,
                "value": round(float(x_row[i]), 4),
                "mean_value": round(float(means[i]), 4),
                "contribution": round(float(contrib), 6),
                "direction": "positive" if contrib >= 0 else "negative",
            }
        )

    # Sort by absolute contribution
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

    summary = _prediction_summary(
        contributions[:3],
        prediction_val,
        predicted_class,
        problem_type,
        target_name,
    )

    return {
        "prediction": (
            predicted_class if predicted_class is not None else round(prediction_val, 4)
        ),
        "prediction_value": round(prediction_val, 4),
        "contributions": contributions,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Partial Dependence Plot (PDP)
# ---------------------------------------------------------------------------


def compute_partial_dependence(
    model,  # fitted sklearn estimator
    X_train: np.ndarray,  # shape (n_samples, n_features)
    feature_idx: int,  # column index in X_train to sweep
    grid_values: np.ndarray,  # 1-D array of values to sweep
    problem_type: str = "regression",
    class_names: list[str] | None = None,
) -> dict:
    """Compute partial dependence of the model output on one feature.

    Unlike sensitivity analysis (which holds all other features at training means),
    PDP averages over the *actual* training distribution for the other features —
    giving a more statistically accurate marginal effect estimate.

    For regression: returns average model prediction at each grid value.
    For binary classification: returns average probability of the positive class.
    For multiclass classification: returns average probability per class.

    Returns:
        {
          grid_values: [float, ...],
          mean_predictions: [float, ...],   # averaged over all training rows
          std_predictions: [float, ...],    # std dev across training rows
          class_curves: {class_name: [float, ...]} | None  (multiclass only)
          problem_type: str,
          n_training_rows: int,
          summary: str,
        }
    """
    grid = np.array(grid_values, dtype=float)
    n_rows = len(X_train)

    mean_preds: list[float] = []
    std_preds: list[float] = []
    # Per-class curves for multiclass classification
    class_sums: list[list[float]] | None = None

    is_multiclass = False
    if problem_type == "classification" and hasattr(model, "predict_proba"):
        try:
            probe = model.predict_proba(X_train[:1])
            if probe.shape[1] > 2:
                is_multiclass = True
                class_sums = [[] for _ in range(probe.shape[1])]
        except Exception:  # noqa: BLE001
            pass

    for val in grid:
        X_mod = X_train.copy()
        X_mod[:, feature_idx] = val

        if problem_type == "classification" and hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_mod)  # (n_rows, n_classes)
                if is_multiclass and class_sums is not None:
                    for ci in range(len(class_sums)):
                        class_sums[ci].append(float(np.mean(proba[:, ci])))
                    # mean_prediction = average predicted class index (not very meaningful)
                    # Use max-class average probability instead
                    avg_proba = np.mean(proba, axis=0)
                    mean_preds.append(float(np.max(avg_proba)))
                    std_preds.append(0.0)
                else:
                    # Binary: use positive class (index 1)
                    pos_proba = proba[:, 1]
                    mean_preds.append(float(np.mean(pos_proba)))
                    std_preds.append(float(np.std(pos_proba)))
            except Exception:  # noqa: BLE001
                # Fallback to label prediction
                preds = model.predict(X_mod).astype(float)
                mean_preds.append(float(np.mean(preds)))
                std_preds.append(float(np.std(preds)))
        else:
            preds = model.predict(X_mod).astype(float)
            mean_preds.append(float(np.mean(preds)))
            std_preds.append(float(np.std(preds)))

    # Build class_curves dict if multiclass
    class_curves: dict[str, list[float]] | None = None
    if is_multiclass and class_sums is not None:
        if class_names and len(class_names) == len(class_sums):
            class_curves = {
                str(class_names[i]): class_sums[i] for i in range(len(class_sums))
            }
        else:
            class_curves = {f"class_{i}": class_sums[i] for i in range(len(class_sums))}

    # Build a plain-English summary
    if len(mean_preds) >= 2:
        first_val = round(float(grid[0]), 4)
        last_val = round(float(grid[-1]), 4)
        first_pred = round(mean_preds[0], 4)
        last_pred = round(mean_preds[-1], 4)
        change = last_pred - first_pred
        direction = (
            "increases" if change > 0 else "decreases" if change < 0 else "stays flat"
        )
        if problem_type == "classification":
            summary = (
                f"As the feature varies from {first_val} to {last_val}, "
                f"the average predicted probability {direction} "
                f"({first_pred:.3f} → {last_pred:.3f}) across {n_rows} training records."
            )
        else:
            summary = (
                f"As the feature varies from {first_val} to {last_val}, "
                f"the average prediction {direction} "
                f"({first_pred:.4g} → {last_pred:.4g}) across {n_rows} training records."
            )
    else:
        summary = "Partial dependence computed."

    return {
        "grid_values": [round(float(v), 6) for v in grid],
        "mean_predictions": [round(v, 6) for v in mean_preds],
        "std_predictions": [round(v, 6) for v in std_preds],
        "class_curves": class_curves,
        "problem_type": problem_type,
        "n_training_rows": n_rows,
        "summary": summary,
    }


def compute_class_conditional_importance(
    model,  # fitted sklearn classifier with predict() and feature_importances_ or coef_
    X: np.ndarray,  # shape (n_samples, n_features) — training data
    y_pred: np.ndarray,  # predicted class labels (int or str)
    feature_names: list[str],
    class_names: list[str] | None = None,
) -> dict:
    """Per-class feature importance breakdown for classification models.

    For each predicted class, filters rows where the model predicts that class,
    then computes which features deviate most (in importance-weighted terms) for
    that cohort vs. the overall training distribution.

    This answers "what makes the model predict class X?" — showing which features
    are systematically different for each predicted outcome.

    Returns:
        {
          classes: [
            {
              class_label: str,
              sample_count: int,
              sample_pct: float,
              features: [
                {
                  feature: str,
                  global_importance: float,
                  mean_for_class: float,
                  global_mean: float,
                  deviation_pct: float,   # (class_mean - global_mean) / (global_mean + eps) * 100
                  weighted_deviation: float,  # importance * |deviation_pct| / 100
                  direction: "above_average" | "below_average" | "similar",
                },
                ...
              ],  # top 8 by weighted_deviation
              top_feature: str,
              summary: str,
            }
          ],
          n_classes: int,
          n_samples: int,
          feature_names: [str, ...],
          summary: str,
        }

    Raises:
        ValueError: if fewer than 2 distinct classes in y_pred, or fewer than 10 rows.
    """
    if len(X) < 10:
        raise ValueError("Need at least 10 rows for class-conditional analysis.")

    unique_preds = np.unique(y_pred)
    if len(unique_preds) < 2:
        raise ValueError("Need at least 2 distinct predicted classes.")

    importances_list = compute_feature_importance(model, feature_names)
    imp_map = {item["feature"]: item["importance"] for item in importances_list}

    global_means = np.mean(X, axis=0)
    n_samples = len(X)

    class_results = []
    for cls_idx, cls_val in enumerate(unique_preds):
        mask = y_pred == cls_val
        X_cls = X[mask]
        count = int(np.sum(mask))

        if count == 0:
            continue

        cls_means = np.mean(X_cls, axis=0) if count > 0 else global_means.copy()
        pct = round(float(count / n_samples * 100), 1)

        features = []
        for i, fname in enumerate(feature_names):
            g_mean = float(global_means[i])
            c_mean = float(cls_means[i])
            imp = imp_map.get(fname, 0.0)

            eps = max(abs(g_mean) * 0.01, 1e-6)
            dev_pct = (c_mean - g_mean) / (abs(g_mean) + eps) * 100.0
            weighted_dev = imp * abs(dev_pct) / 100.0

            direction: str
            if abs(dev_pct) < 5.0:
                direction = "similar"
            elif dev_pct > 0:
                direction = "above_average"
            else:
                direction = "below_average"

            features.append(
                {
                    "feature": fname,
                    "global_importance": round(imp, 6),
                    "mean_for_class": round(c_mean, 4),
                    "global_mean": round(float(g_mean), 4),
                    "deviation_pct": round(dev_pct, 1),
                    "weighted_deviation": round(weighted_dev, 6),
                    "direction": direction,
                }
            )

        features.sort(key=lambda f: f["weighted_deviation"], reverse=True)
        top_8 = features[:8]

        top_feat = (
            top_8[0]["feature"]
            if top_8
            else (feature_names[0] if feature_names else "")
        )

        # Resolve class label
        if class_names and cls_idx < len(class_names):
            cls_label = str(class_names[cls_idx])
        else:
            cls_label = str(cls_val)

        # Build summary sentence
        if top_8:
            t = top_8[0]
            dir_phrase = (
                "higher than average"
                if t["direction"] == "above_average"
                else (
                    "lower than average"
                    if t["direction"] == "below_average"
                    else "average"
                )
            )
            summary = (
                f"For predictions of '{cls_label}' ({count:,} of {n_samples:,} records): "
                f"'{top_feat}' is the most distinguishing feature ({dir_phrase} by "
                f"{abs(t['deviation_pct']):.0f}%)."
            )
        else:
            summary = f"Class '{cls_label}' has {count:,} predicted records."

        class_results.append(
            {
                "class_label": cls_label,
                "sample_count": count,
                "sample_pct": pct,
                "features": top_8,
                "top_feature": top_feat,
                "summary": summary,
            }
        )

    # Overall summary
    if len(class_results) >= 2:
        top_classes = sorted(
            class_results, key=lambda c: c["sample_count"], reverse=True
        )
        main_cls = top_classes[0]
        if main_cls["features"]:
            top_feat_main = main_cls["features"][0]["feature"]
            overall_summary = (
                f"Across {len(class_results)} classes, '{top_feat_main}' is the top "
                f"distinguishing feature for the most common prediction "
                f"('{main_cls['class_label']}', {main_cls['sample_pct']:.0f}% of records)."
            )
        else:
            overall_summary = (
                f"Class-conditional analysis across {len(class_results)} classes."
            )
    elif len(class_results) == 1:
        overall_summary = class_results[0]["summary"]
    else:
        overall_summary = "No class-conditional data available."

    return {
        "classes": class_results,
        "n_classes": len(class_results),
        "n_samples": n_samples,
        "feature_names": feature_names,
        "summary": overall_summary,
    }


def _prediction_summary(
    top_contributions: list[dict],
    prediction_val: float,
    predicted_class: int | None,
    problem_type: str,
    target_name: str,
) -> str:
    if not top_contributions:
        return "No contribution data available."

    top = top_contributions[0]
    direction = "increased" if top["direction"] == "positive" else "decreased"

    if problem_type == "classification":
        pred_str = (
            f"class {predicted_class}"
            if predicted_class is not None
            else str(round(prediction_val, 2))
        )
        return (
            f"Predicted {target_name} = {pred_str}. "
            f"The strongest driver was '{top['feature']}' (value = {top['value']:.2f}), "
            f"which {direction} the prediction."
        )
    else:
        return (
            f"Predicted {target_name} = {prediction_val:.4f}. "
            f"The strongest driver was '{top['feature']}' (value = {top['value']:.2f}), "
            f"which {direction} the prediction relative to the average."
        )
