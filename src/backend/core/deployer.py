"""Model deployment: packaging, prediction, and batch inference.

Design principles:
- PredictionPipeline captures all preprocessing needed to transform new data
  identically to how training data was prepared (same encoders, same fill values).
- predict_single accepts a plain dict {feature: value} — no pandas required on the caller.
- predict_batch accepts CSV bytes and returns CSV bytes with a prediction column added.
- No mutation of the incoming data.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# PredictionPipeline — everything needed to transform new inputs
# ---------------------------------------------------------------------------


@dataclass
class PredictionPipeline:
    """Encapsulates all preprocessing logic learned from training data."""

    feature_names: list[str]
    column_types: dict[str, str]  # 'numeric' | 'categorical'
    label_encoders: dict[str, LabelEncoder] = field(default_factory=dict)
    medians: dict[str, float] = field(default_factory=dict)
    target_column: str = ""
    problem_type: str = "regression"
    target_encoder: Optional[LabelEncoder] = None
    target_classes: Optional[list] = None
    # Statistics for explanation (stored at build time)
    feature_means: dict[str, float] = field(default_factory=dict)
    feature_stds: dict[str, float] = field(default_factory=dict)
    # Residual std for regression prediction intervals (stored at deploy time)
    residual_std: float = 0.0

    def transform(self, input_dict: dict) -> np.ndarray:
        """Transform a single row dict into a feature vector for prediction."""
        row = []
        for col in self.feature_names:
            val = input_dict.get(col)
            if self.column_types.get(col) == "numeric":
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = self.medians.get(col, 0.0)
                row.append(float(val))
            else:
                # Categorical
                str_val = str(val) if val is not None else "MISSING"
                le = self.label_encoders.get(col)
                if le is not None:
                    if str_val in le.classes_:
                        row.append(float(le.transform([str_val])[0]))
                    else:
                        # Unseen category — use 0 (best-effort)
                        row.append(0.0)
                else:
                    row.append(0.0)
        return np.array(row, dtype=float).reshape(1, -1)

    def transform_df(self, df: pd.DataFrame) -> np.ndarray:
        """Transform a DataFrame of rows into a feature matrix."""
        rows = []
        for _, row in df.iterrows():
            rows.append(self.transform(row.to_dict()).flatten())
        return np.array(rows, dtype=float)

    def decode_prediction(self, raw_pred) -> str | float:
        """Decode a raw model prediction to a human-readable value."""
        if self.problem_type == "classification" and self.target_encoder is not None:
            idx = int(round(float(raw_pred)))
            idx = max(0, min(idx, len(self.target_encoder.classes_) - 1))
            return str(self.target_encoder.classes_[idx])
        return round(float(raw_pred), 4)


# ---------------------------------------------------------------------------
# Build pipeline from training data
# ---------------------------------------------------------------------------


def build_prediction_pipeline(
    df: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
    problem_type: str,
) -> PredictionPipeline:
    """Fit a PredictionPipeline from the training DataFrame.

    Mirrors the preprocessing in trainer.prepare_features so that predictions
    on new data use the exact same encoding/fill logic.
    """
    pipeline = PredictionPipeline(
        feature_names=feature_names,
        column_types={},
        target_column=target_col,
        problem_type=problem_type,
    )

    # Drop rows with missing target to match training
    df_clean = (
        df[feature_names + [target_col]]
        .dropna(subset=[target_col])
        .reset_index(drop=True)
    )

    for col in feature_names:
        series = df_clean[col]
        if pd.api.types.is_numeric_dtype(series):
            pipeline.column_types[col] = "numeric"
            pipeline.medians[col] = float(series.median())
            pipeline.feature_means[col] = float(series.mean())
            pipeline.feature_stds[col] = float(series.std()) if len(series) > 1 else 1.0
        else:
            pipeline.column_types[col] = "categorical"
            le = LabelEncoder()
            le.fit(series.fillna("MISSING").astype(str))
            pipeline.label_encoders[col] = le

    # Target encoder (classification with string labels)
    y_series = df_clean[target_col]
    if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y_series):
        le_target = LabelEncoder()
        le_target.fit(y_series.astype(str))
        pipeline.target_encoder = le_target
        pipeline.target_classes = list(le_target.classes_)

    return pipeline


# ---------------------------------------------------------------------------
# Save / load pipeline
# ---------------------------------------------------------------------------


def save_pipeline(pipeline: PredictionPipeline, path: Path) -> None:
    joblib.dump(pipeline, path)


def load_pipeline(path: str | Path) -> PredictionPipeline:
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------


def predict_single(
    pipeline_path: str,
    model_path: str,
    input_data: dict,
) -> dict:
    """Make a single prediction from a dict of feature values.

    Returns:
        {prediction, decoded_prediction, problem_type, feature_names}
    """
    pipeline = load_pipeline(pipeline_path)
    model = joblib.load(model_path)

    X = pipeline.transform(input_data)
    raw = model.predict(X)[0]
    decoded = pipeline.decode_prediction(raw)

    result: dict = {
        "prediction": decoded,
        "problem_type": pipeline.problem_type,
        "target_column": pipeline.target_column,
        "feature_names": pipeline.feature_names,
    }

    # For classification, also return class probabilities if available
    if pipeline.problem_type == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = (
            [str(c) for c in pipeline.target_encoder.classes_]
            if pipeline.target_encoder is not None
            else [str(i) for i in range(len(proba))]
        )
        result["probabilities"] = {
            cls: round(float(p), 4) for cls, p in zip(classes, proba)
        }
        result["confidence"] = round(float(proba.max()), 4)

    # For regression, return a 95% prediction interval using stored residual std
    if pipeline.problem_type == "regression":
        residual_std = getattr(pipeline, "residual_std", 0.0)
        if residual_std > 0:
            z = 1.96  # 95% prediction interval
            pred_value = float(decoded)
            result["confidence_interval"] = {
                "lower": round(pred_value - z * residual_std, 4),
                "upper": round(pred_value + z * residual_std, 4),
                "level": 0.95,
                "label": "95% prediction interval",
            }

    return result


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------


def predict_batch(
    pipeline_path: str,
    model_path: str,
    csv_bytes: bytes,
) -> bytes:
    """Make predictions for all rows in a CSV.

    Returns CSV bytes with a 'prediction' column appended.
    """
    pipeline = load_pipeline(pipeline_path)
    model = joblib.load(model_path)

    df = pd.read_csv(io.BytesIO(csv_bytes))

    # Predict row by row so missing columns get default-filled gracefully
    X = pipeline.transform_df(df)
    raw_preds = model.predict(X)

    decoded = [pipeline.decode_prediction(p) for p in raw_preds]
    df["prediction"] = decoded

    # For classification add confidence if possible
    if pipeline.problem_type == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        df["confidence"] = [round(float(p.max()), 4) for p in proba]

    out = io.BytesIO()
    df.to_csv(out, index=False)
    return out.getvalue()


# ---------------------------------------------------------------------------
# Build feature schema for the prediction form
# ---------------------------------------------------------------------------


def get_feature_schema(pipeline_path: str) -> list[dict]:
    """Return a JSON-serialisable schema describing each feature for the UI form."""
    pipeline = load_pipeline(pipeline_path)
    schema = []
    for col in pipeline.feature_names:
        col_type = pipeline.column_types.get(col, "numeric")
        entry: dict = {
            "name": col,
            "type": col_type,
        }
        if col_type == "categorical":
            le = pipeline.label_encoders.get(col)
            entry["options"] = list(le.classes_) if le is not None else []
        else:
            entry["median"] = pipeline.medians.get(col, 0.0)
            entry["mean"] = pipeline.feature_means.get(col, None)
            entry["std"] = pipeline.feature_stds.get(col, None)
        schema.append(entry)
    return schema


# ---------------------------------------------------------------------------
# Prediction explanation
# ---------------------------------------------------------------------------


def explain_prediction(
    pipeline_path: str,
    model_path: str,
    input_data: dict,
) -> dict:
    """Explain a single prediction using feature contributions.

    Uses global feature importance multiplied by normalised deviation from mean.
    Works with any sklearn estimator that has feature_importances_ or coef_.

    Returns:
        {
          prediction: decoded prediction value,
          contributions: [{feature, value, mean_value, contribution, direction}],
          summary: str,
          top_drivers: [str],   # plain-English top-3 driver names
        }
    """
    from core.explainer import compute_feature_importance

    pipeline = load_pipeline(pipeline_path)
    model = joblib.load(model_path)

    x_vec = pipeline.transform(input_data).flatten()  # shape (n_features,)

    # Get global feature importance
    feature_names = pipeline.feature_names
    importance_list = compute_feature_importance(model, feature_names)
    imp_map = {item["feature"]: item["importance"] for item in importance_list}

    # Use stored means/stds (fall back to 0/1 for old pipelines without them)
    contributions = []
    for i, col in enumerate(feature_names):
        imp = imp_map.get(col, 0.0)
        mean_val = getattr(pipeline, "feature_means", {}).get(col, float(x_vec[i]))
        std_val = getattr(pipeline, "feature_stds", {}).get(col, 1.0)
        if std_val < 1e-10:
            std_val = 1.0

        deviation = (float(x_vec[i]) - mean_val) / std_val
        contrib = imp * deviation

        contributions.append(
            {
                "feature": col,
                "value": round(float(x_vec[i]), 4),
                "mean_value": round(mean_val, 4),
                "contribution": round(float(contrib), 6),
                "direction": "positive" if contrib >= 0 else "negative",
            }
        )

    # Sort by absolute contribution (highest first)
    contributions.sort(key=lambda c: abs(c["contribution"]), reverse=True)

    # Make prediction for display
    raw = model.predict(x_vec.reshape(1, -1))[0]
    decoded = pipeline.decode_prediction(raw)

    # Build plain-English summary
    top = contributions[:3] if contributions else []
    drivers = [c["feature"] for c in top if abs(c["contribution"]) > 1e-8]
    top_names = (
        " and ".join(f"'{d}'" for d in drivers[:2]) if drivers else "the input features"
    )

    if pipeline.problem_type == "classification":
        summary = (
            f"Predicted {pipeline.target_column} = {decoded}. "
            f"The prediction was primarily driven by {top_names}."
        )
    else:
        summary = (
            f"Predicted {pipeline.target_column} = {decoded}. "
            f"The main factors were {top_names}."
        )

    return {
        "prediction": decoded,
        "target_column": pipeline.target_column,
        "problem_type": pipeline.problem_type,
        "contributions": contributions,
        "summary": summary,
        "top_drivers": drivers[:3],
    }


def run_sensitivity_analysis(
    pipeline_path: str,
    model_path: str,
    feature_name: str,
    sweep_values: list[float],
    base_features: dict,
) -> dict:
    """Sweep a single feature across a range of values and collect predictions.

    All other features are held at their training-data means (base_features).

    Returns:
        {
          feature, target_column, problem_type,
          values: [float],
          predictions: [float | str],   # numeric for regression, top class for classification
          confidences: [float | None],   # max class probability for classification
          min_pred, max_pred, change_pct,
          summary
        }
    """
    import joblib as _jl

    pipeline = load_pipeline(pipeline_path)
    model = _jl.load(model_path)

    if feature_name not in pipeline.feature_names:
        raise ValueError(f"Feature '{feature_name}' not found in model.")

    results: list = []
    for v in sweep_values:
        inputs = {**base_features, feature_name: float(v)}
        x = pipeline.transform(inputs)
        raw = model.predict(x)[0]
        decoded = pipeline.decode_prediction(raw)
        results.append(decoded)

    # Confidences (classification only)
    confidences: list = []
    if pipeline.problem_type == "classification" and hasattr(model, "predict_proba"):
        for v in sweep_values:
            inputs = {**base_features, feature_name: float(v)}
            x = pipeline.transform(inputs)
            proba = model.predict_proba(x)[0]
            confidences.append(round(float(proba.max()), 4))
    else:
        confidences = [None] * len(sweep_values)

    # Stats (regression only — predictions are numeric)
    min_pred: float | None = None
    max_pred: float | None = None
    change_pct: float | None = None
    if pipeline.problem_type == "regression":
        try:
            numeric_preds = [float(r) for r in results]
            min_pred = round(min(numeric_preds), 4)
            max_pred = round(max(numeric_preds), 4)
            first = numeric_preds[0]
            last = numeric_preds[-1]
            if first != 0:
                change_pct = round((last - first) / abs(first) * 100, 1)
        except (TypeError, ValueError):
            pass

    # Plain-English summary
    feat_display = feature_name.replace("_", " ")
    target_display = (
        pipeline.target_column.replace("_", " ")
        if pipeline.target_column
        else "prediction"
    )
    n = len(sweep_values)
    if pipeline.problem_type == "regression" and change_pct is not None:
        direction = "increases" if change_pct > 0 else "decreases"
        summary = (
            f"As {feat_display} varies from {sweep_values[0]:g} to {sweep_values[-1]:g} "
            f"across {n} steps, {target_display} {direction} by {abs(change_pct):.1f}% "
            f"(from {min_pred:,.4g} to {max_pred:,.4g})."
        )
    elif pipeline.problem_type == "classification":
        unique_classes = list(dict.fromkeys(str(r) for r in results))
        if len(unique_classes) == 1:
            summary = (
                f"As {feat_display} varies from {sweep_values[0]:g} to {sweep_values[-1]:g}, "
                f"the predicted class remains '{unique_classes[0]}' across all {n} steps."
            )
        else:
            summary = (
                f"As {feat_display} varies from {sweep_values[0]:g} to {sweep_values[-1]:g}, "
                f"the predicted class switches between: {', '.join(unique_classes[:4])}."
            )
    else:
        summary = (
            f"Sensitivity sweep of {feat_display} from {sweep_values[0]:g} "
            f"to {sweep_values[-1]:g} ({n} steps) complete."
        )

    return {
        "feature": feature_name,
        "target_column": pipeline.target_column,
        "problem_type": pipeline.problem_type,
        "values": [float(v) for v in sweep_values],
        "predictions": results,
        "confidences": confidences,
        "min_pred": min_pred,
        "max_pred": max_pred,
        "change_pct": change_pct,
        "summary": summary,
    }


def run_feature_interaction(
    pipeline_path: str,
    model_path: str,
    feature1: str,
    feature2: str,
    base_features: dict,
    n_steps: int = 7,
) -> dict:
    """Build a 2-D prediction grid by jointly sweeping two features.

    For numeric features: sweep linearly over [mean ± 2*std] (n_steps points).
    For categorical features: use every known class from the label encoder
    (capped at n_steps).

    All other features are held at their training-data means (base_features).

    Returns:
        {
          feature1, feature2, target_column, problem_type,
          row_labels: [str],          # display values for feature1 axis
          col_labels: [str],          # display values for feature2 axis
          values: [[float|str]],      # 2-D grid: values[i][j] = prediction
          min_val: float | None,      # numeric min (regression)
          max_val: float | None,      # numeric max (regression)
          summary: str
        }
    """
    import joblib as _jl
    import numpy as _np

    pipeline = load_pipeline(pipeline_path)
    model = _jl.load(model_path)

    for feat in (feature1, feature2):
        if feat not in pipeline.feature_names:
            raise ValueError(f"Feature '{feat}' not found in model.")

    def _sweep_values(feat: str) -> tuple[list, list[str]]:
        """Return (raw_values, display_labels) for a feature."""
        if pipeline.column_types.get(feat) == "categorical":
            le = pipeline.label_encoders.get(feat)
            if le is not None and len(le.classes_) > 0:
                classes = list(le.classes_[:n_steps])
                return classes, [str(c) for c in classes]
            return [], []
        else:
            mean = base_features.get(feat, 0.0)
            std = pipeline.feature_stds.get(feat, 0.0)
            half = max(std * 2, abs(mean) * 0.5, 1.0)
            lo = mean - half
            hi = mean + half
            vals = list(_np.linspace(lo, hi, n_steps))
            labels = [f"{v:g}" for v in vals]
            return vals, labels

    row_raw, row_labels = _sweep_values(feature1)
    col_raw, col_labels = _sweep_values(feature2)

    if not row_raw or not col_raw:
        raise ValueError("Could not generate sweep values for one or both features.")

    # Build prediction grid
    grid: list[list] = []
    for r_val in row_raw:
        row: list = []
        for c_val in col_raw:
            inputs = {**base_features, feature1: r_val, feature2: c_val}
            x = pipeline.transform(inputs)
            raw = model.predict(x)[0]
            decoded = pipeline.decode_prediction(raw)
            row.append(decoded)
        grid.append(row)

    # Stats for regression
    min_val: float | None = None
    max_val: float | None = None
    if pipeline.problem_type == "regression":
        try:
            flat = [float(v) for row in grid for v in row]
            min_val = round(min(flat), 4)
            max_val = round(max(flat), 4)
        except (TypeError, ValueError):
            pass

    # Plain-English summary
    target = pipeline.target_column
    f1_display = feature1.replace("_", " ")
    f2_display = feature2.replace("_", " ")
    if pipeline.problem_type == "regression" and min_val is not None and max_val is not None:
        spread = max_val - min_val
        pct = round(spread / max(abs(min_val), 1e-9) * 100, 1) if min_val != 0 else 0
        summary = (
            f"Across all combinations of {f1_display} and {f2_display}, "
            f"{target} ranges from {min_val:g} to {max_val:g} "
            f"(a {pct}% spread). "
            f"Look for cells with the highest values to find the best-performing combination."
        )
    else:
        # Classification — count distinct predicted classes
        all_preds = [v for row in grid for v in row]
        unique_classes = sorted(set(str(p) for p in all_preds))
        if len(unique_classes) == 1:
            summary = (
                f"For all combinations of {f1_display} and {f2_display}, "
                f"the model always predicts '{unique_classes[0]}'."
            )
        else:
            summary = (
                f"Across combinations of {f1_display} and {f2_display}, "
                f"the model predicts {len(unique_classes)} different classes: "
                f"{', '.join(unique_classes)}."
            )

    return {
        "feature1": feature1,
        "feature2": feature2,
        "target_column": target,
        "problem_type": pipeline.problem_type,
        "row_labels": row_labels,
        "col_labels": col_labels,
        "values": grid,
        "min_val": min_val,
        "max_val": max_val,
        "summary": summary,
    }
