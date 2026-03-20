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
        "prediction": predicted_class
        if predicted_class is not None
        else round(prediction_val, 4),
        "prediction_value": round(prediction_val, 4),
        "contributions": contributions,
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
