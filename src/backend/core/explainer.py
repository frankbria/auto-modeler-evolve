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
