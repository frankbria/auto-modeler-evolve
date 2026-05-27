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
    # Per-feature ranges for input validation (stored at build time)
    # numeric: {"p5": float, "p95": float, "min": float, "max": float}
    # categorical: {"known_categories": list[str]}
    feature_ranges: dict[str, dict] = field(default_factory=dict)

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
            s_clean = series.dropna()
            if len(s_clean) >= 2:
                pipeline.feature_ranges[col] = {
                    "p5": float(s_clean.quantile(0.05)),
                    "p95": float(s_clean.quantile(0.95)),
                    "min": float(s_clean.min()),
                    "max": float(s_clean.max()),
                }
        else:
            pipeline.column_types[col] = "categorical"
            le = LabelEncoder()
            le.fit(series.fillna("MISSING").astype(str))
            pipeline.label_encoders[col] = le
            known = [c for c in series.dropna().astype(str).unique() if c != "MISSING"]
            pipeline.feature_ranges[col] = {"known_categories": sorted(known)}

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
# Input validation
# ---------------------------------------------------------------------------


def validate_prediction_inputs(
    provided_features: dict,
    pipeline: PredictionPipeline,
) -> list[dict]:
    """Check provided feature values against training-data ranges.

    Returns a list of warning dicts for out-of-range or unknown-category inputs.
    Only features present in ``provided_features`` are checked (defaults are not
    validated — they are by definition in-range from training data).

    Each warning dict contains:
        feature, provided_value, severity, message
    Numeric warnings also include: expected_min, expected_max, training_min, training_max
    Categorical warnings also include: known_categories (up to 10 examples)
    """
    warnings: list[dict] = []
    feature_ranges = getattr(pipeline, "feature_ranges", {})
    if not feature_ranges:
        return warnings

    for feat, value in provided_features.items():
        if feat not in feature_ranges or value is None:
            continue
        ranges = feature_ranges[feat]
        col_type = pipeline.column_types.get(feat, "numeric")

        if col_type == "numeric":
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            p5 = ranges.get("p5")
            p95 = ranges.get("p95")
            vmin = ranges.get("min")
            vmax = ranges.get("max")

            if vmin is not None and vmax is not None and (v < vmin or v > vmax):
                severity = "extreme_outlier"
                msg = (
                    f"{feat}: {v:.4g} is outside the training range "
                    f"({vmin:.4g}–{vmax:.4g}); predictions may be unreliable"
                )
            elif p5 is not None and p95 is not None and (v < p5 or v > p95):
                severity = "out_of_range"
                msg = (
                    f"{feat}: {v:.4g} is outside the typical range "
                    f"({p5:.4g}–{p95:.4g}); confidence may be lower"
                )
            else:
                continue

            warnings.append(
                {
                    "feature": feat,
                    "provided_value": v,
                    "severity": severity,
                    "expected_min": p5,
                    "expected_max": p95,
                    "training_min": vmin,
                    "training_max": vmax,
                    "message": msg,
                }
            )

        else:  # categorical
            known = ranges.get("known_categories", [])
            if not known:
                continue
            str_val = str(value)
            if str_val not in known:
                warnings.append(
                    {
                        "feature": feat,
                        "provided_value": str_val,
                        "severity": "unknown_category",
                        "known_categories": known[:10],
                        "message": (
                            f"{feat}: '{str_val}' was not seen during training; "
                            f"model will use a fallback encoding"
                        ),
                    }
                )

    return warnings


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------


def predict_single(
    pipeline_path: str,
    model_path: str,
    input_data: dict,
    provided_features: dict | None = None,
) -> dict:
    """Make a single prediction from a dict of feature values.

    Args:
        pipeline_path: Path to the saved PredictionPipeline joblib file.
        model_path: Path to the saved sklearn model joblib file.
        input_data: Full feature dict (user inputs + any defaults already filled).
        provided_features: Optional subset of input_data that came from the user
            (not filled-in defaults). Used to validate only user-supplied inputs
            against training ranges. When None, no guard-rail warnings are emitted.

    Returns:
        dict with prediction, problem_type, target_column, feature_names, and
        optionally: probabilities, confidence, confidence_interval, guard_rail_warnings.
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

    # Guard-rail input validation: only check user-provided values (not defaults)
    if provided_features is not None:
        warnings = validate_prediction_inputs(provided_features, pipeline)
        if warnings:
            result["guard_rail_warnings"] = warnings

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
            ranges = getattr(pipeline, "feature_ranges", {}).get(col, {})
            if ranges:
                entry["min"] = ranges.get("min")
                entry["max"] = ranges.get("max")
                entry["p5"] = ranges.get("p5")
                entry["p95"] = ranges.get("p95")
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

    Also computes a baseline prediction (all features at training-data means) to
    show how the user's specific inputs shift the outcome vs a "typical" case.

    Returns:
        {
          prediction: decoded prediction value,
          contributions: [{feature, value, mean_value, contribution, direction}],
          summary: str,
          top_drivers: [str],          # plain-English top-3 driver names
          baseline_prediction: decoded, # prediction when all features = training mean
          delta: float | None,          # prediction − baseline (regression only)
          pct_change: float | None,     # % change from baseline (regression only)
          direction: str,               # "above_baseline" | "below_baseline" | "at_baseline"
                                        # | "class_changed" | "same_class" (classification)
          current_confidence: float | None,  # max predict_proba for user's inputs
          baseline_confidence: float | None, # max predict_proba for baseline inputs
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

    # Make prediction for user's inputs
    raw = model.predict(x_vec.reshape(1, -1))[0]
    decoded = pipeline.decode_prediction(raw)

    # -----------------------------------------------------------------------
    # Baseline prediction (all features at encoded training-data means)
    # feature_means holds the mean of each feature in its *encoded* form
    # (integers for categorical columns after LabelEncoding).
    # -----------------------------------------------------------------------
    means = getattr(pipeline, "feature_means", {})
    baseline_x = np.array(
        [means.get(col, 0.0) for col in feature_names], dtype=float
    ).reshape(1, -1)
    baseline_raw = model.predict(baseline_x)[0]
    baseline_decoded = pipeline.decode_prediction(baseline_raw)

    # Compute delta / direction
    delta: float | None = None
    pct_change: float | None = None
    current_confidence: float | None = None
    baseline_confidence: float | None = None

    if pipeline.problem_type == "regression":
        try:
            _delta = float(raw) - float(baseline_raw)
            _base = float(baseline_raw)
            delta = round(_delta, 4)
            pct_change = (
                round(_delta / abs(_base) * 100, 1) if abs(_base) > 1e-10 else None
            )
            if abs(_delta) < 1e-6:
                direction = "at_baseline"
            elif _delta > 0:
                direction = "above_baseline"
            else:
                direction = "below_baseline"
        except (ValueError, TypeError):
            direction = "at_baseline"
    else:
        # Classification — compare predicted classes; optionally compare confidence
        direction = (
            "class_changed" if str(decoded) != str(baseline_decoded) else "same_class"
        )
        if hasattr(model, "predict_proba"):
            try:
                curr_proba = model.predict_proba(x_vec.reshape(1, -1))[0]
                base_proba = model.predict_proba(baseline_x)[0]
                current_confidence = round(float(curr_proba.max()), 4)
                baseline_confidence = round(float(base_proba.max()), 4)
            except Exception:  # noqa: BLE001
                pass

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
        "baseline_prediction": baseline_decoded,
        "delta": delta,
        "pct_change": pct_change,
        "direction": direction,
        "current_confidence": current_confidence,
        "baseline_confidence": baseline_confidence,
    }


def compute_aggregate_explanations(
    pipeline_path: str,
    model_path: str,
    input_data_list: list[dict],
) -> dict:
    """Aggregate feature contribution explanations across multiple production predictions.

    Loads the model/pipeline once and processes all inputs in a single pass.

    Returns:
        {
          features: [{feature, avg_abs_contribution, positive_pct, direction_label,
                      top_driver_pct, sample_count}],
          sample_count: int,
          summary: str,
        }
    """
    from core.explainer import compute_feature_importance

    pipeline = load_pipeline(pipeline_path)
    model = joblib.load(model_path)

    feature_names = pipeline.feature_names
    importance_list = compute_feature_importance(model, feature_names)
    imp_map = {item["feature"]: item["importance"] for item in importance_list}

    # Per-feature accumulators: list of signed contributions per sample
    feature_contribs: dict[str, list[float]] = {f: [] for f in feature_names}
    # Per-sample top-3 driver counts
    top3_counts: dict[str, int] = {f: 0 for f in feature_names}
    valid_count = 0

    for input_data in input_data_list:
        try:
            x_vec = pipeline.transform(input_data).flatten()
        except Exception:  # noqa: BLE001
            continue

        sample_abs: list[tuple[str, float]] = []
        for i, col in enumerate(feature_names):
            imp = imp_map.get(col, 0.0)
            mean_val = getattr(pipeline, "feature_means", {}).get(col, float(x_vec[i]))
            std_val = getattr(pipeline, "feature_stds", {}).get(col, 1.0)
            if std_val < 1e-10:
                std_val = 1.0
            deviation = (float(x_vec[i]) - mean_val) / std_val
            contrib = imp * deviation
            feature_contribs[col].append(contrib)
            sample_abs.append((col, abs(contrib)))

        # Record top-3 contributors for this sample
        sample_abs.sort(key=lambda t: t[1], reverse=True)
        for col, _ in sample_abs[:3]:
            top3_counts[col] += 1
        valid_count += 1

    if valid_count == 0:
        return {
            "features": [],
            "sample_count": 0,
            "summary": "No predictions available to analyze.",
        }

    result_features = []
    for col in feature_names:
        contribs = feature_contribs[col]
        if not contribs:
            continue
        avg_abs = sum(abs(c) for c in contribs) / len(contribs)
        positive_count = sum(1 for c in contribs if c > 0)
        positive_pct = round(positive_count / len(contribs) * 100, 1)
        top_driver_pct = round(top3_counts.get(col, 0) / valid_count * 100, 1)

        if positive_pct >= 70:
            direction_label = "mostly positive"
        elif positive_pct <= 30:
            direction_label = "mostly negative"
        else:
            direction_label = "mixed"

        result_features.append(
            {
                "feature": col,
                "avg_abs_contribution": round(avg_abs, 6),
                "positive_pct": positive_pct,
                "direction_label": direction_label,
                "top_driver_pct": top_driver_pct,
                "sample_count": len(contribs),
            }
        )

    result_features.sort(key=lambda f: f["avg_abs_contribution"], reverse=True)

    top = result_features[0] if result_features else None
    if top:
        summary = (
            f"Across {valid_count} production predictions, '{top['feature']}' "
            f"had the strongest average influence ({top['direction_label']}, "
            f"top driver in {top['top_driver_pct']}% of predictions)."
        )
    else:
        summary = f"No feature contributions found across {valid_count} predictions."

    return {
        "features": result_features,
        "sample_count": valid_count,
        "summary": summary,
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
    if (
        pipeline.problem_type == "regression"
        and min_val is not None
        and max_val is not None
    ):
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


# ---------------------------------------------------------------------------
# Dataset ranking — apply model to all rows, return top N ranked
# ---------------------------------------------------------------------------


def run_dataset_ranking(
    pipeline_path: str,
    model_path: str,
    df: pd.DataFrame,
    n: int = 20,
    direction: str = "highest",
) -> dict:
    """Apply model to every row of df and return the top N ranked by prediction.

    For regression: ranks by predicted value (direction "highest" or "lowest").
    For classification: ranks by model confidence (max predicted probability).

    The df should be the working dataset (with target column present — it is
    ignored during prediction; only pipeline.feature_names are used).

    Returns:
        {problem_type, target_column, direction, n, total_scored, rows, summary,
         class_names}
    """
    pipeline = load_pipeline(pipeline_path)
    model = joblib.load(model_path)

    if len(df) == 0:
        raise ValueError("Dataset is empty — cannot rank predictions.")

    # Score every row
    X = pipeline.transform_df(df)
    raw_preds = model.predict(X)

    proba_arr: np.ndarray | None = None
    class_names: list[str] | None = None

    if pipeline.problem_type == "classification" and hasattr(model, "predict_proba"):
        proba_arr = model.predict_proba(X)
        if pipeline.target_encoder is not None:
            class_names = [str(c) for c in pipeline.target_encoder.classes_]
        else:
            class_names = [str(i) for i in range(proba_arr.shape[1])]
        # Rank by max class probability (overall confidence)
        scores = [float(proba_arr[i].max()) for i in range(len(proba_arr))]
    else:
        scores = [float(pipeline.decode_prediction(p)) for p in raw_preds]

    # Rank and select top N
    n_capped = min(max(1, n), len(df))
    reverse = direction == "highest"
    sorted_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=reverse
    )
    top_indices = sorted_indices[:n_capped]

    # Build result rows — include up to 4 feature values for display
    display_cols = pipeline.feature_names[:4]
    rows = []
    for rank, row_idx in enumerate(top_indices, 1):
        row_data = df.iloc[row_idx]
        feature_values: dict = {}
        for col in display_cols:
            val = row_data.get(col) if col in row_data.index else None
            feature_values[col] = (
                None
                if (val is not None and isinstance(val, float) and pd.isna(val))
                else (
                    val
                    if val is None
                    else (
                        float(val)
                        if isinstance(val, (int, float, np.floating))
                        else str(val)
                    )
                )
            )

        entry: dict = {
            "rank": rank,
            "row_index": int(row_idx),
            "score": round(scores[row_idx], 4),
            "feature_values": feature_values,
        }

        if pipeline.problem_type == "regression":
            entry["prediction"] = round(scores[row_idx], 4)
        else:
            pred_class = str(pipeline.decode_prediction(raw_preds[row_idx]))
            entry["predicted_class"] = pred_class
            entry["confidence"] = round(scores[row_idx], 4)
            if proba_arr is not None and class_names is not None:
                entry["probabilities"] = {
                    cls: round(float(proba_arr[row_idx][i]), 4)
                    for i, cls in enumerate(class_names)
                }

        rows.append(entry)

    total = len(df)
    target = pipeline.target_column
    dir_word = "highest" if direction == "highest" else "lowest"

    if pipeline.problem_type == "regression":
        top_val = rows[0]["prediction"] if rows else 0
        summary = (
            f"Scored all {total:,} rows and ranked by predicted {target}. "
            f"The {dir_word} predicted value is {top_val:g}."
        )
    else:
        top_class = rows[0].get("predicted_class", "") if rows else ""
        top_conf = rows[0].get("confidence", 0.0) if rows else 0.0
        summary = (
            f"Scored all {total:,} rows by model confidence. "
            f"Top result: '{top_class}' at {round(top_conf * 100, 1)}% confidence."
        )

    return {
        "problem_type": pipeline.problem_type,
        "target_column": target,
        "direction": direction,
        "n": n_capped,
        "total_scored": total,
        "rows": rows,
        "summary": summary,
        "class_names": class_names,
    }


def compute_prediction_cohort(
    pipeline_path: str,
    model_path: str,
    df: pd.DataFrame,
    n: int = 20,
    direction: str = "highest",
) -> dict:
    """Profile the top-N predictions as a cohort vs the full dataset.

    Re-scores the dataset (same as run_dataset_ranking), then compares the
    top-N rows against the overall population across categorical and numeric
    features to answer "who are the highest-risk predictions?"

    Returns:
        {target_column, problem_type, n, direction, total_scored,
         categorical_profile, numeric_profile, characterization}
    """
    ranking = run_dataset_ranking(
        pipeline_path, model_path, df, n=n, direction=direction
    )
    top_indices = [row["row_index"] for row in ranking["rows"]]
    top_df = df.iloc[top_indices]
    target_col = ranking["target_column"]

    # --- Categorical profile ---
    categorical_profile: list[dict] = []
    for col in df.select_dtypes(include=["str", "object", "category"]).columns:
        if col == target_col:
            continue
        if df[col].nunique() > 20:
            continue  # skip high-cardinality columns
        top_counts = top_df[col].value_counts(normalize=True)
        overall_counts = df[col].value_counts(normalize=True)

        categories = []
        for val in top_counts.index[:5]:
            top_pct = float(top_counts.get(val, 0.0)) * 100
            overall_pct = float(overall_counts.get(val, 0.0)) * 100
            ratio = top_pct / overall_pct if overall_pct > 0 else 0.0
            categories.append(
                {
                    "value": str(val),
                    "top_pct": round(top_pct, 1),
                    "overall_pct": round(overall_pct, 1),
                    "ratio": round(ratio, 2),
                }
            )

        if not categories:
            continue
        categorical_profile.append(
            {
                "column": col,
                "categories": categories,
                "dominant": categories[0]["value"],
                "dominant_top_pct": categories[0]["top_pct"],
            }
        )

    # --- Numeric profile ---
    numeric_profile: list[dict] = []
    for col in df.select_dtypes(include=["number"]).columns:
        if col == target_col:
            continue
        top_mean = float(top_df[col].mean()) if len(top_df) > 0 else 0.0
        overall_mean = float(df[col].mean())
        if pd.isna(top_mean) or pd.isna(overall_mean):
            continue
        ratio: float | None = (
            round(top_mean / overall_mean, 2) if overall_mean != 0 else None
        )
        direction_label = (
            "higher"
            if top_mean > overall_mean
            else "lower"
            if top_mean < overall_mean
            else "similar"
        )
        numeric_profile.append(
            {
                "column": col,
                "top_mean": round(top_mean, 3),
                "overall_mean": round(overall_mean, 3),
                "ratio": ratio,
                "direction": direction_label,
            }
        )

    # --- Plain-English characterization ---
    char_parts: list[str] = []
    # Top 2 most prominent categorical differences
    for cp in categorical_profile[:2]:
        if cp["dominant_top_pct"] > 25:
            char_parts.append(
                f"{round(cp['dominant_top_pct'])}% have {cp['column'].replace('_', ' ')} = '{cp['dominant']}'"
            )
    # Top 2 most extreme numeric differences
    numeric_sorted = sorted(
        [r for r in numeric_profile if r["ratio"] is not None],
        key=lambda r: abs(r["ratio"] - 1),  # type: ignore[operator]
        reverse=True,
    )
    for nr in numeric_sorted[:2]:
        r = nr["ratio"]
        if r is not None and abs(r - 1) > 0.1:
            pct_diff = round(abs(r - 1) * 100)
            char_parts.append(
                f"{nr['column'].replace('_', ' ')} is {pct_diff}% {nr['direction']} on average"
            )

    dir_word = "highest" if direction == "highest" else "lowest"
    base = f"The {n} {dir_word}-scoring {target_col} predictions"
    characterization = (
        (base + ": " + "; ".join(char_parts) + ".") if char_parts else base + "."
    )

    return {
        "target_column": target_col,
        "problem_type": ranking["problem_type"],
        "n": ranking["n"],
        "direction": direction,
        "total_scored": ranking["total_scored"],
        "categorical_profile": categorical_profile,
        "numeric_profile": numeric_profile,
        "characterization": characterization,
    }


def compute_cohort_evolution(
    pipeline_path: str,
    model_path: str,
    dataframes: list[tuple[pd.DataFrame, str, str]],
    n: int = 20,
    direction: str = "highest",
) -> dict:
    """Track how the top-N prediction cohort profile evolves across multiple dataset uploads.

    Args:
        pipeline_path: Path to the serialised PredictionPipeline joblib file.
        model_path: Path to the serialised model joblib file.
        dataframes: List of (DataFrame, dataset_id, period_label) tuples, ordered
                    oldest-first.  Each DataFrame is the raw (unfiltered) dataset for
                    that period.  Maximum 6 periods used to keep the result manageable.
        n: How many top predictions to include in each period's cohort.
        direction: "highest" (top-N by prediction value) or "lowest".

    Returns a dict::

        {
            n, direction, target_column, problem_type,
            n_periods,
            periods: [
                {
                    period_label, dataset_id, total_rows,
                    categorical_profile,   # same shape as compute_prediction_cohort
                    numeric_profile,
                },
                ...
            ],
            shifts: [
                {
                    from_period, to_period,
                    categorical_shifts: [
                        {column, category, from_pct, to_pct, change, direction},
                        ...
                    ],
                    summary,
                },
                ...
            ],
            summary,
        }
    """
    if len(dataframes) < 2:
        raise ValueError(
            "At least 2 datasets are required for cohort evolution analysis."
        )

    # Limit to most-recent 6 periods
    data_slice = dataframes[-6:]

    periods: list[dict] = []
    pipeline = load_pipeline(pipeline_path)
    target_col = pipeline.target_column
    problem_type = pipeline.problem_type

    for df, ds_id, label in data_slice:
        if len(df) == 0:
            continue
        try:
            cohort = compute_prediction_cohort(
                pipeline_path, model_path, df, n=n, direction=direction
            )
            periods.append(
                {
                    "period_label": label,
                    "dataset_id": ds_id,
                    "total_rows": len(df),
                    "categorical_profile": cohort["categorical_profile"],
                    "numeric_profile": cohort["numeric_profile"],
                }
            )
        except Exception:  # noqa: BLE001
            # Skip a period that can't be scored (e.g. schema mismatch)
            continue

    if len(periods) < 2:
        raise ValueError("Not enough scoreable datasets for cohort evolution analysis.")

    # --- Build period-over-period shifts ---
    shifts: list[dict] = []
    for i in range(len(periods) - 1):
        prev = periods[i]
        curr = periods[i + 1]

        # Build lookup: column → category → pct for each period
        def _build_cat_lookup(cat_profile: list[dict]) -> dict:
            lookup: dict[str, dict[str, float]] = {}
            for cp in cat_profile:
                col = cp["column"]
                lookup[col] = {}
                for cat_entry in cp.get("categories", []):
                    lookup[col][cat_entry["value"]] = cat_entry["top_pct"]
            return lookup

        prev_lookup = _build_cat_lookup(prev["categorical_profile"])
        curr_lookup = _build_cat_lookup(curr["categorical_profile"])

        categorical_shifts: list[dict] = []
        all_cols = set(prev_lookup.keys()) | set(curr_lookup.keys())
        for col in all_cols:
            prev_cats = prev_lookup.get(col, {})
            curr_cats = curr_lookup.get(col, {})
            all_categories = set(prev_cats.keys()) | set(curr_cats.keys())
            for cat_val in all_categories:
                from_pct = prev_cats.get(cat_val, 0.0)
                to_pct = curr_cats.get(cat_val, 0.0)
                change = round(to_pct - from_pct, 1)
                if abs(change) < 5.0:
                    continue  # Only surface meaningful shifts (≥5 pp)
                shift_dir = "rising" if change > 0 else "falling"
                categorical_shifts.append(
                    {
                        "column": col,
                        "category": cat_val,
                        "from_pct": round(from_pct, 1),
                        "to_pct": round(to_pct, 1),
                        "change": change,
                        "direction": shift_dir,
                    }
                )

        # Sort by absolute change descending, cap at 5 per shift
        categorical_shifts.sort(key=lambda x: abs(x["change"]), reverse=True)
        top_shifts = categorical_shifts[:5]

        # Plain-English shift summary
        if top_shifts:
            s = top_shifts[0]
            col_plain = s["column"].replace("_", " ")
            dir_word = "grew" if s["direction"] == "rising" else "fell"
            shift_summary = (
                f"{col_plain} = '{s['category']}' {dir_word} from "
                f"{s['from_pct']}% to {s['to_pct']}% of the top-{n} cohort."
            )
        else:
            shift_summary = (
                f"The top-{n} cohort composition was stable between "
                f"{prev['period_label']} and {curr['period_label']}."
            )

        shifts.append(
            {
                "from_period": prev["period_label"],
                "to_period": curr["period_label"],
                "categorical_shifts": top_shifts,
                "summary": shift_summary,
            }
        )

    # --- Overall summary ---
    all_shifts_flat = [s for shift in shifts for s in shift["categorical_shifts"]]
    if all_shifts_flat:
        # Find the single biggest mover across all periods
        biggest = max(all_shifts_flat, key=lambda x: abs(x["change"]))
        col_plain = biggest["column"].replace("_", " ")
        dir_word = "increasing" if biggest["direction"] == "rising" else "decreasing"
        dir_word2 = "growing" if biggest["direction"] == "rising" else "shrinking"
        overall_summary = (
            f"Over {len(periods)} periods, the top-{n} {direction} {target_col} "
            f"predictions show {col_plain} = '{biggest['category']}' "
            f"{dir_word} ({biggest['from_pct']}% → {biggest['to_pct']}%) — "
            f"this segment's share of the highest-scoring records is {dir_word2}."
        )
    else:
        overall_summary = (
            f"The top-{n} {direction} {target_col} predictions had a stable "
            f"composition across {len(periods)} periods with no segments "
            f"shifting more than 5 percentage points."
        )

    return {
        "n": n,
        "direction": direction,
        "target_column": target_col,
        "problem_type": problem_type,
        "n_periods": len(periods),
        "periods": periods,
        "shifts": shifts,
        "summary": overall_summary,
    }


# ---------------------------------------------------------------------------
# Goal seek — reverse prediction: find inputs that produce a target output
# ---------------------------------------------------------------------------

# Plain-English algorithm names (mirrors trainer.py)
_ALGO_PLAIN_GS = {
    "linear_regression": "Linear Regression",
    "ridge_regression": "Ridge Regression",
    "random_forest_regressor": "Random Forest",
    "gradient_boosting_regressor": "Gradient Boosting",
    "xgboost_regressor": "XGBoost",
    "lightgbm_regressor": "LightGBM",
    "mlp_regressor": "Neural Network",
    "logistic_regression": "Logistic Regression",
    "random_forest_classifier": "Random Forest",
    "gradient_boosting_classifier": "Gradient Boosting",
    "xgboost_classifier": "XGBoost",
    "lightgbm_classifier": "LightGBM",
    "mlp_classifier": "Neural Network",
    "decision_tree_classifier": "Decision Tree",
    "voting_regressor": "Voting Ensemble",
    "voting_classifier": "Voting Ensemble",
    "stacking_regressor": "Stacking Ensemble",
    "stacking_classifier": "Stacking Ensemble",
}


def run_goal_seek(
    pipeline_path: str,
    model_path: str,
    target_value: float | str,
    fixed_features: dict | None = None,
    algorithm: str = "",
    max_iter: int = 300,
) -> dict:
    """Find feature values that produce a desired prediction target.

    For regression: minimise |predict(x) - target_value|.
    For classification: maximise predict_proba for the target class.

    Only numeric features are optimized; categorical features are held at
    their training-data median (encoded mean from pipeline.feature_means).
    Features in fixed_features are held at the provided values.

    Returns:
        {
          target_column, problem_type, algorithm_plain,
          target_value,          # desired output (number or class string)
          achieved_value,        # best prediction found
          achieved,              # True when within 5% for regression / exact class for classification
          gap_pct,               # |target - achieved| / |target| * 100 (regression); None for classification
          suggestions: [         # numeric features that changed from their means
            {feature, current_mean, suggested_value, direction, change_pct}
          ],
          fixed_features,        # features explicitly held fixed by caller
          categorical_features,  # categorical features set to baseline
          n_optimized,           # number of numeric features actually optimized
          feasible,              # True if a solution was found (optimizer converged)
          summary,               # plain-English sentence
        }
    """
    from scipy.optimize import minimize as _sp_minimize  # optional import

    import joblib as _jl

    pipeline = load_pipeline(pipeline_path)
    model = _jl.load(model_path)
    fixed = fixed_features or {}

    # Separate features into: optimize (free numeric), fixed (explicit), categorical (baseline)
    free_numeric: list[str] = []
    cat_baseline: dict[str, float] = {}

    for f in pipeline.feature_names:
        if f in fixed:
            continue
        if pipeline.column_types.get(f) == "numeric":
            free_numeric.append(f)
        else:
            # Categorical: keep at encoded mean (best proxy for mode)
            cat_baseline[f] = float(pipeline.feature_means.get(f, 0.0))

    # For classification, resolve target_class index
    is_classification = pipeline.problem_type == "classification"
    target_class_idx: int | None = None
    target_class_str: str = ""
    if is_classification:
        target_class_str = str(target_value)
        if pipeline.target_classes:
            class_strs = [str(c) for c in pipeline.target_classes]
            if target_class_str in class_strs:
                target_class_idx = class_strs.index(target_class_str)
            else:
                # Default to class index 1 (positive class) if not found
                target_class_idx = 1 if len(class_strs) > 1 else 0
        else:
            target_class_idx = 1
    else:
        # Regression: ensure target_value is numeric
        try:
            target_value = float(target_value)
        except (TypeError, ValueError):
            target_value = 0.0

    # Build base feature dict (fixed + categorical)
    def _build_row(x_vals: list[float]) -> dict:
        row: dict = {}
        for fn in pipeline.feature_names:
            if fn in fixed:
                row[fn] = float(fixed[fn])
            elif fn in cat_baseline:
                row[fn] = cat_baseline[fn]
        for fn, v in zip(free_numeric, x_vals):
            row[fn] = v
        return row

    # Initial guess: training means for free numeric features
    x0 = [float(pipeline.feature_means.get(f, 0.0)) for f in free_numeric]

    # Bounds: p5 - span .. p95 + span (extrapolate ±100% range for flexibility)
    bounds: list[tuple[float | None, float | None]] = []
    for f in free_numeric:
        fr = pipeline.feature_ranges.get(f, {})
        lo = fr.get("p5", fr.get("min", None))
        hi = fr.get("p95", fr.get("max", None))
        if lo is not None and hi is not None:
            span = max(hi - lo, abs(hi) * 0.1, 1.0)
            bounds.append((lo - span, hi + span))
        else:
            bounds.append((None, None))

    # Objective function
    def _objective(x_vals: list[float]) -> float:
        row = _build_row(list(x_vals))
        x_arr = pipeline.transform(row)
        if is_classification:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x_arr)[0]
                if target_class_idx is not None and target_class_idx < len(proba):
                    return float(-proba[target_class_idx])
                return float(-proba.max())
            # No predict_proba — approximate via prediction
            pred = model.predict(x_arr)[0]
            return (
                0.0
                if str(pipeline.decode_prediction(pred)) == target_class_str
                else 1.0
            )
        else:
            pred = float(model.predict(x_arr)[0])
            return float((pred - target_value) ** 2)  # type: ignore[operator]

    feasible = True
    if free_numeric:
        try:
            opt_result = _sp_minimize(
                _objective,
                x0,
                method="L-BFGS-B",
                bounds=bounds if any(b != (None, None) for b in bounds) else None,
                options={"maxiter": max_iter, "ftol": 1e-12},
            )
            best_x = list(opt_result.x)
            feasible = opt_result.success or opt_result.fun < 1e6
        except Exception:  # noqa: BLE001
            best_x = x0
            feasible = False
    else:
        best_x = x0

    # Compute achieved prediction
    best_row = _build_row(best_x)
    best_x_arr = pipeline.transform(best_row)
    raw_pred = model.predict(best_x_arr)[0]
    achieved_decoded = pipeline.decode_prediction(raw_pred)

    # Determine if goal was achieved
    achieved = False
    gap_pct: float | None = None
    if is_classification:
        achieved = str(achieved_decoded) == target_class_str
    else:
        achieved_val = float(achieved_decoded)
        tgt_val = float(target_value)  # type: ignore[arg-type]
        if abs(tgt_val) > 1e-9:
            gap_pct = round(abs(tgt_val - achieved_val) / abs(tgt_val) * 100, 1)
            achieved = gap_pct <= 5.0
        else:
            gap_pct = 0.0
            achieved = abs(achieved_val) < 0.1

    # Build suggestions (features that changed meaningfully from their means)
    suggestions: list[dict] = []
    for f, suggested in zip(free_numeric, best_x):
        mean_val = float(pipeline.feature_means.get(f, 0.0))
        if abs(mean_val) > 1e-9:
            change_pct = round((suggested - mean_val) / abs(mean_val) * 100, 1)
        else:
            change_pct = 0.0
        direction = (
            "increase"
            if suggested > mean_val + 1e-9
            else ("decrease" if suggested < mean_val - 1e-9 else "no_change")
        )
        # Only include features that changed by at least 1%
        if abs(change_pct) >= 1.0:
            suggestions.append(
                {
                    "feature": f,
                    "current_mean": round(mean_val, 4),
                    "suggested_value": round(float(suggested), 4),
                    "direction": direction,
                    "change_pct": change_pct,
                }
            )
    # Sort by absolute change, descending
    suggestions.sort(key=lambda s: abs(s["change_pct"]), reverse=True)
    # Cap at 8 most impactful
    suggestions = suggestions[:8]

    # Plain-English summary
    algo_plain = _ALGO_PLAIN_GS.get(algorithm, algorithm.replace("_", " ").title())
    target_col = pipeline.target_column or "output"

    if is_classification:
        if achieved:
            summary = f"Goal achieved: the model predicts '{target_class_str}' with the suggested inputs."
        else:
            best_proba_str = ""
            if hasattr(model, "predict_proba") and target_class_idx is not None:
                proba = model.predict_proba(best_x_arr)[0]
                if target_class_idx < len(proba):
                    best_proba_str = (
                        f" Best confidence: {round(proba[target_class_idx] * 100, 1)}%."
                    )
            summary = (
                f"Could not achieve class '{target_class_str}' — the model predicts "
                f"'{achieved_decoded}' even with optimized inputs.{best_proba_str}"
            )
    else:
        if achieved:
            summary = (
                f"Goal achieved: the model predicts "
                f"{round(float(achieved_decoded), 2):,} for {target_col} with the suggested inputs."
            )
        else:
            if len(suggestions) > 0:
                top = suggestions[0]
                direction_word = (
                    "increasing" if top["direction"] == "increase" else "decreasing"
                )
                summary = (
                    f"Best effort: predicted {round(float(achieved_decoded), 2):,} vs target "
                    f"{round(float(target_value), 2):,} "  # type: ignore[arg-type]
                    f"({gap_pct}% gap). "
                    f"The biggest lever is {direction_word} {top['feature'].replace('_', ' ')}."
                )
            else:
                summary = (
                    f"Best effort: predicted {round(float(achieved_decoded), 2):,} vs target "
                    f"{round(float(target_value), 2):,} ({gap_pct}% gap). "  # type: ignore[arg-type]
                    f"No free numeric features to optimize."
                )

    return {
        "target_column": target_col,
        "problem_type": pipeline.problem_type,
        "algorithm_plain": algo_plain,
        "target_value": target_value if not is_classification else target_class_str,
        "achieved_value": achieved_decoded,
        "achieved": achieved,
        "gap_pct": gap_pct,
        "suggestions": suggestions,
        "fixed_features": {k: v for k, v in fixed.items()},
        "categorical_features": {k: v for k, v in cat_baseline.items()},
        "n_optimized": len(free_numeric),
        "feasible": feasible,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Prediction delta — "why did my prediction change after retraining?"
# ---------------------------------------------------------------------------


def compute_prediction_delta(
    pipeline_path_new: str,
    model_path_new: str,
    pipeline_path_old: str,
    model_path_old: str,
    input_features: dict,
) -> dict:
    """Compare predictions between two model versions for the same inputs.

    Explains how and why the prediction changed after retraining by comparing
    per-feature contributions between the old and new model versions.

    Args:
        pipeline_path_new: Path to the new (current) PredictionPipeline joblib.
        model_path_new: Path to the new sklearn model joblib.
        pipeline_path_old: Path to the previous PredictionPipeline joblib.
        model_path_old: Path to the previous sklearn model joblib.
        input_features: Feature dict to run through both models.

    Returns:
        {
            old_prediction, new_prediction, delta, pct_change, direction,
            problem_type, target_column, provided_features,
            feature_delta: [{feature, old_contribution, new_contribution,
                             contribution_delta, direction}],
            top_drivers: [str],  # plain-English driver phrases
            summary: str,
        }
    """
    from core.explainer import compute_feature_importance  # noqa: PLC0415

    pipeline_new = load_pipeline(pipeline_path_new)
    model_new = joblib.load(model_path_new)
    pipeline_old = load_pipeline(pipeline_path_old)
    model_old = joblib.load(model_path_old)

    target_col = pipeline_new.target_column
    problem_type = pipeline_new.problem_type

    # ── Predictions ──────────────────────────────────────────────────────────
    x_new = pipeline_new.transform(input_features).flatten()
    raw_new = model_new.predict(x_new.reshape(1, -1))[0]
    pred_new = pipeline_new.decode_prediction(raw_new)

    x_old = pipeline_old.transform(input_features).flatten()
    raw_old = model_old.predict(x_old.reshape(1, -1))[0]
    pred_old = pipeline_old.decode_prediction(raw_old)

    # ── Numeric delta ─────────────────────────────────────────────────────────
    try:
        num_new = float(pred_new)
        num_old = float(pred_old)
        delta = round(num_new - num_old, 4)
        pct_change = round((delta / abs(num_old) * 100) if num_old != 0 else 0.0, 1)
        direction = "up" if delta > 1e-9 else ("down" if delta < -1e-9 else "unchanged")
    except (TypeError, ValueError):
        # Classification — use top-class confidence delta
        delta = 0.0
        pct_change = 0.0
        direction = "unchanged" if str(pred_new) == str(pred_old) else "changed"

    # ── Feature contribution delta ────────────────────────────────────────────
    def _contributions(pipeline, model, x_vec: "np.ndarray") -> dict[str, float]:  # type: ignore[name-defined]
        feat_names = pipeline.feature_names
        imp_list = compute_feature_importance(model, feat_names)
        imp_map = {item["feature"]: item["importance"] for item in imp_list}
        contribs: dict[str, float] = {}
        for i, col in enumerate(feat_names):
            imp = imp_map.get(col, 0.0)
            mean_val = getattr(pipeline, "feature_means", {}).get(col, float(x_vec[i]))
            std_val = getattr(pipeline, "feature_stds", {}).get(col, 1.0)
            if std_val < 1e-10:
                std_val = 1.0
            deviation = (float(x_vec[i]) - mean_val) / std_val
            contribs[col] = float(imp * deviation)
        return contribs

    new_contribs = _contributions(pipeline_new, model_new, x_new)
    old_contribs = _contributions(pipeline_old, model_old, x_old)

    # Merge feature sets (use the new model's features as reference)
    feature_delta: list[dict] = []
    for feat in pipeline_new.feature_names:
        c_new = new_contribs.get(feat, 0.0)
        c_old = old_contribs.get(feat, 0.0)
        c_delta = c_new - c_old
        feature_delta.append(
            {
                "feature": feat,
                "old_contribution": round(c_old, 6),
                "new_contribution": round(c_new, 6),
                "contribution_delta": round(c_delta, 6),
                "direction": (
                    "increased"
                    if c_delta > 1e-9
                    else ("decreased" if c_delta < -1e-9 else "stable")
                ),
            }
        )

    # Sort by absolute contribution delta (largest driver of change first)
    feature_delta.sort(key=lambda f: abs(f["contribution_delta"]), reverse=True)

    # ── Plain-English summary ─────────────────────────────────────────────────
    top_drivers: list[str] = []
    for f in feature_delta[:3]:
        if abs(f["contribution_delta"]) < 1e-9:
            continue
        word = "↑" if f["direction"] == "increased" else "↓"
        top_drivers.append(f"{f['feature'].replace('_', ' ')} {word}")

    if problem_type == "regression":
        if direction == "unchanged":
            summary = (
                f"The {target_col.replace('_', ' ')} prediction remained unchanged "
                f"at {pred_new} after retraining."
            )
        else:
            dir_word = "increased" if direction == "up" else "decreased"
            summary = (
                f"The {target_col.replace('_', ' ')} prediction {dir_word} from "
                f"{pred_old} to {pred_new} "
                f"({'+' if delta >= 0 else ''}{delta:,}, "
                f"{'+' if pct_change >= 0 else ''}{pct_change}%)."
            )
            if top_drivers:
                summary += f" Top drivers of change: {', '.join(top_drivers)}."
    else:
        if str(pred_new) == str(pred_old):
            summary = f"The predicted class remained '{pred_new}' after retraining."
        else:
            summary = (
                f"The predicted class changed from '{pred_old}' to '{pred_new}' "
                f"after retraining."
            )
            if top_drivers:
                summary += f" Top feature shifts: {', '.join(top_drivers)}."

    return {
        "old_prediction": pred_old,
        "new_prediction": pred_new,
        "delta": delta,
        "pct_change": pct_change,
        "direction": direction,
        "problem_type": problem_type,
        "target_column": target_col,
        "provided_features": input_features,
        "feature_delta": feature_delta,
        "top_drivers": top_drivers,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Counterfactual Explanation
# ---------------------------------------------------------------------------


def compute_counterfactual(
    pipeline_path: str,
    model_path: str,
    original_features: dict,
    target_class: str | None = None,
    max_steps: int = 15,
    step_fraction: float = 0.1,
) -> dict:
    """Find the minimal feature change to flip a classification prediction.

    Uses greedy gradient-based search: at each step perturbs the numeric
    feature whose finite-difference gradient most moves the prediction toward
    the desired class.

    Only numeric features are perturbed; categorical features stay fixed.

    Args:
        pipeline_path: Path to PredictionPipeline joblib.
        model_path: Path to sklearn model joblib.
        original_features: Feature dict for the specific instance (raw values).
        target_class: Desired class to flip to.  None → flip to any other class.
        max_steps: Maximum greedy steps before giving up.
        step_fraction: Each step nudges a feature by this fraction of its
            training std (from pipeline.feature_stds).

    Returns:
        {
          problem_type, target_column,
          original_prediction, original_class, original_confidence,
          counterfactual_prediction, target_class,
          counterfactual_confidence,
          flipped,              # True when class actually changed
          n_steps,
          changed_features: [  # only features that were perturbed
            {name, original_value, counterfactual_value, change_pct, direction}
          ],
          summary,
        }

    Raises:
        ValueError: if the model is not a classifier or has no predict_proba.
    """
    import joblib as _jl

    pipeline = load_pipeline(pipeline_path)
    model = _jl.load(model_path)

    if pipeline.problem_type != "classification":
        raise ValueError(
            "Counterfactual explanations are only supported for classification models. "
            "For regression, use goal_seek to find inputs that meet a target value."
        )
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Counterfactual requires a probabilistic classifier (predict_proba). "
            "This model does not support probability estimates."
        )

    # ── Decode class names ────────────────────────────────────────────────────
    classes: list[str] = []
    if pipeline.target_encoder is not None:
        classes = [str(c) for c in pipeline.target_encoder.classes_]
    else:
        # Probe with the original features to get class count
        x_probe = pipeline.transform(original_features)
        proba_probe = model.predict_proba(x_probe)[0]
        classes = [str(i) for i in range(len(proba_probe))]

    if len(classes) < 2:
        raise ValueError("Counterfactual requires at least 2 classes.")

    # ── Original prediction ───────────────────────────────────────────────────
    x_orig = pipeline.transform(original_features)
    proba_orig = model.predict_proba(x_orig)[0]
    orig_class_idx = int(np.argmax(proba_orig))
    orig_class = classes[orig_class_idx]
    orig_confidence = round(float(proba_orig[orig_class_idx]), 4)

    # Resolve target class index
    if target_class is not None:
        if target_class not in classes:
            raise ValueError(
                f"Target class '{target_class}' not found. Available classes: {classes}"
            )
        target_class_idx = classes.index(target_class)
    else:
        # Flip to the class with second-highest probability
        sorted_idxs = np.argsort(proba_orig)[::-1]
        target_class_idx = int(sorted_idxs[1])
        target_class = classes[target_class_idx]

    if target_class_idx == orig_class_idx:
        # Already at the target class — nothing to flip
        return {
            "problem_type": "classification",
            "target_column": pipeline.target_column,
            "original_prediction": orig_class,
            "original_class": orig_class,
            "original_confidence": orig_confidence,
            "counterfactual_prediction": orig_class,
            "target_class": target_class,
            "counterfactual_confidence": orig_confidence,
            "flipped": True,
            "n_steps": 0,
            "changed_features": [],
            "summary": (
                f"The prediction is already '{orig_class}' — no change needed."
            ),
        }

    # ── Identify perturbable numeric features ────────────────────────────────
    numeric_features = [
        f for f in pipeline.feature_names if pipeline.column_types.get(f) == "numeric"
    ]
    if not numeric_features:
        raise ValueError(
            "No numeric features found — counterfactual requires at least one "
            "numeric feature to perturb."
        )

    # Step sizes: step_fraction × training std (floor at small epsilon)
    feature_stds = getattr(pipeline, "feature_stds", {})
    step_sizes: dict[str, float] = {}
    for f in numeric_features:
        std = float(feature_stds.get(f, 1.0))
        step_sizes[f] = max(std * step_fraction, 1e-6)

    # ── Greedy search ─────────────────────────────────────────────────────────
    current_features = dict(original_features)
    n_steps = 0
    flipped = False
    current_class_idx = orig_class_idx

    for _step in range(max_steps):
        # Compute finite-difference gradient of target class probability
        # for each numeric feature
        x_curr = pipeline.transform(current_features)
        proba_curr = model.predict_proba(x_curr)[0]
        current_class_idx = int(np.argmax(proba_curr))

        if current_class_idx == target_class_idx:
            flipped = True
            break

        target_prob_curr = float(proba_curr[target_class_idx])

        best_feature: str | None = None
        best_gain: float = -1.0
        best_direction: float = 0.0

        for feat in numeric_features:
            step = step_sizes[feat]
            curr_val = float(current_features.get(feat, 0.0))

            # Probe + direction
            for direction_sign in (+1.0, -1.0):
                probe = dict(current_features)
                probe[feat] = curr_val + direction_sign * step
                x_probe = pipeline.transform(probe)
                proba_probe = model.predict_proba(x_probe)[0]
                gain = float(proba_probe[target_class_idx]) - target_prob_curr
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feat
                    best_direction = direction_sign

        if best_feature is None or best_gain <= 0:
            # No numeric feature can improve — stop early
            break

        # Apply the best step
        curr_val = float(current_features.get(best_feature, 0.0))
        current_features[best_feature] = (
            curr_val + best_direction * step_sizes[best_feature]
        )
        n_steps += 1

    # Final prediction after search
    x_final = pipeline.transform(current_features)
    proba_final = model.predict_proba(x_final)[0]
    final_class_idx = int(np.argmax(proba_final))
    final_class = classes[final_class_idx]
    cf_confidence = round(float(proba_final[target_class_idx]), 4)
    flipped = final_class_idx == target_class_idx

    # ── Build changed-features list ───────────────────────────────────────────
    changed_features: list[dict] = []
    for feat in numeric_features:
        orig_val = float(original_features.get(feat, 0.0))
        cf_val = float(current_features.get(feat, 0.0))
        if abs(cf_val - orig_val) < 1e-9:
            continue
        change = cf_val - orig_val
        pct = round((change / abs(orig_val) * 100) if orig_val != 0 else 0.0, 1)
        changed_features.append(
            {
                "name": feat,
                "original_value": round(orig_val, 4),
                "counterfactual_value": round(cf_val, 4),
                "change_pct": pct,
                "direction": "increase" if change > 0 else "decrease",
            }
        )

    # Sort by absolute change magnitude
    changed_features.sort(
        key=lambda f: abs(f["counterfactual_value"] - f["original_value"]),
        reverse=True,
    )

    # ── Plain-English summary ─────────────────────────────────────────────────
    target_col = pipeline.target_column.replace("_", " ")
    if flipped:
        if changed_features:
            top = changed_features[0]
            dir_word = "increase" if top["direction"] == "increase" else "decrease"
            feat_name = top["name"].replace("_", " ")
            summary = (
                f"By {dir_word}ing {feat_name} from "
                f"{top['original_value']:,} to {top['counterfactual_value']:,}, "
                f"the {target_col} prediction flips from '{orig_class}' to '{final_class}'. "
            )
            if len(changed_features) > 1:
                extra = [f["name"].replace("_", " ") for f in changed_features[1:3]]
                summary += f"Other helpful changes: {', '.join(extra)}."
        else:
            summary = (
                f"The prediction is already '{final_class}' with no changes needed."
            )
    else:
        if changed_features:
            summary = (
                f"Even after {n_steps} adjustments, the model still predicts '{orig_class}'. "
                f"This instance is firmly in the '{orig_class}' category — "
                f"larger changes may be required to reach '{target_class}'."
            )
        else:
            summary = (
                f"No numeric features could be adjusted to flip the prediction "
                f"from '{orig_class}' to '{target_class}'."
            )

    return {
        "problem_type": "classification",
        "target_column": pipeline.target_column,
        "original_prediction": orig_class,
        "original_class": orig_class,
        "original_confidence": orig_confidence,
        "counterfactual_prediction": final_class,
        "target_class": target_class,
        "counterfactual_confidence": cf_confidence,
        "flipped": flipped,
        "n_steps": n_steps,
        "changed_features": changed_features,
        "summary": summary,
    }


def compute_population_counterfactual(
    pipeline_path: str,
    model_path: str,
    input_features_list: list[dict],
    target_class: str | None = None,
    max_rows: int = 20,
    max_steps: int = 15,
    step_fraction: float = 0.1,
) -> dict:
    """Find the single feature intervention that flips the most predictions in a cohort.

    Runs the greedy counterfactual for each row (up to max_rows), then aggregates
    which feature + direction combination was the dominant driver of prediction flips.
    Returns the most impactful population-level intervention.

    Only works for classification models.

    Args:
        pipeline_path: Path to PredictionPipeline joblib.
        model_path: Path to sklearn model joblib.
        input_features_list: List of feature dicts (one per row).
        target_class: Desired class to flip to.  None → flip away from current prediction.
        max_rows: Cap on rows to analyse (default 20; performance guard).
        max_steps: Max greedy steps per row.
        step_fraction: Step size as fraction of training std.

    Returns:
        {
          problem_type, target_column,
          total_rows,               # rows analysed
          flipped_count,            # rows where flip was found
          flip_rate,                # flipped_count / total_rows (0–1)
          dominant_feature,         # feature name with most flips as top change
          dominant_direction,       # "increase" | "decrease"
          dominant_flip_count,      # rows flipped with this as primary feature
          dominant_avg_change_pct,  # average % change needed for those rows
          feature_summary: [        # per-feature aggregated stats
            {feature, flip_count, flip_pct, avg_change_pct, direction}
          ],
          summary,                  # plain-English narrative
        }

    Raises:
        ValueError: if model is not a classifier, or fewer than 2 input rows.
    """
    import joblib as _jl
    from collections import defaultdict as _dd

    if len(input_features_list) < 2:
        raise ValueError("Population counterfactual requires at least 2 rows.")

    pipeline = load_pipeline(pipeline_path)
    model = _jl.load(model_path)

    if pipeline.problem_type != "classification":
        raise ValueError(
            "Population counterfactual is only supported for classification models. "
            "Use goal_seek for regression."
        )
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Population counterfactual requires a probabilistic classifier (predict_proba)."
        )

    # ── Decode class names ────────────────────────────────────────────────────
    classes: list[str] = []
    if pipeline.target_encoder is not None:
        classes = [str(c) for c in pipeline.target_encoder.classes_]
    else:
        x_probe = pipeline.transform(input_features_list[0])
        proba_probe = model.predict_proba(x_probe)[0]
        classes = [str(i) for i in range(len(proba_probe))]

    if len(classes) < 2:
        raise ValueError("Population counterfactual requires at least 2 classes.")

    # Numeric features and step sizes
    numeric_features = [
        f for f in pipeline.feature_names if pipeline.column_types.get(f) == "numeric"
    ]
    if not numeric_features:
        raise ValueError(
            "Population counterfactual requires at least one numeric feature."
        )
    feature_stds = getattr(pipeline, "feature_stds", {})
    step_sizes: dict[str, float] = {}
    for f in numeric_features:
        std = float(feature_stds.get(f, 1.0))
        step_sizes[f] = max(std * step_fraction, 1e-6)

    # ── Run counterfactual for each row ───────────────────────────────────────
    rows = input_features_list[:max_rows]
    total_rows = len(rows)

    # Aggregation: {feature -> {direction -> [change_pcts]}}
    feature_direction_changes: dict = _dd(lambda: {"increase": [], "decrease": []})
    flipped_count = 0

    for orig_feats in rows:
        try:
            x_orig = pipeline.transform(orig_feats)
            proba_orig = model.predict_proba(x_orig)[0]
            orig_class_idx = int(np.argmax(proba_orig))

            # Determine target class for this row
            if target_class is not None and target_class in classes:
                t_class_idx = classes.index(target_class)
            else:
                sorted_idxs = np.argsort(proba_orig)[::-1]
                t_class_idx = int(sorted_idxs[1])

            if t_class_idx == orig_class_idx:
                # Already at target — still counts as flipped
                flipped_count += 1
                continue

            # Run greedy search
            current_features = dict(orig_feats)
            row_flipped = False

            for _step in range(max_steps):
                x_curr = pipeline.transform(current_features)
                proba_curr = model.predict_proba(x_curr)[0]
                current_class_idx = int(np.argmax(proba_curr))

                if current_class_idx == t_class_idx:
                    row_flipped = True
                    break

                target_prob_curr = float(proba_curr[t_class_idx])
                best_feature: str | None = None
                best_gain: float = -1.0
                best_direction: float = 0.0

                for feat in numeric_features:
                    step = step_sizes[feat]
                    curr_val = float(current_features.get(feat, 0.0))
                    for direction_sign in (+1.0, -1.0):
                        probe = dict(current_features)
                        probe[feat] = curr_val + direction_sign * step
                        x_probe = pipeline.transform(probe)
                        proba_probe = model.predict_proba(x_probe)[0]
                        gain = float(proba_probe[t_class_idx]) - target_prob_curr
                        if gain > best_gain:
                            best_gain = gain
                            best_feature = feat
                            best_direction = direction_sign

                if best_feature is None or best_gain <= 0:
                    break

                curr_val = float(current_features.get(best_feature, 0.0))
                current_features[best_feature] = (
                    curr_val + best_direction * step_sizes[best_feature]
                )

            if row_flipped:
                flipped_count += 1
                # Record the primary changed feature (largest absolute change)
                primary_feat: str | None = None
                primary_dir = "increase"
                max_abs_change = 0.0
                for feat in numeric_features:
                    orig_val = float(orig_feats.get(feat, 0.0))
                    cf_val = float(current_features.get(feat, 0.0))
                    abs_change = abs(cf_val - orig_val)
                    if abs_change > max_abs_change:
                        max_abs_change = abs_change
                        primary_feat = feat
                        primary_dir = "increase" if cf_val > orig_val else "decrease"
                        orig_v = orig_val

                if primary_feat is not None:
                    pct = (
                        round((max_abs_change / abs(orig_v) * 100), 1)
                        if orig_v != 0
                        else 0.0
                    )
                    feature_direction_changes[primary_feat][primary_dir].append(pct)

        except Exception:  # noqa: BLE001
            continue  # Skip rows that error; don't crash the whole analysis

    # ── Aggregate by feature + direction ─────────────────────────────────────
    # Find (feature, direction) pair with most associated flips
    best_feat: str | None = None
    best_dir = "increase"
    best_flip_ct = 0
    best_avg_pct = 0.0

    all_feature_rows: list[dict] = []
    for feat, dirs in feature_direction_changes.items():
        for direction, pct_list in dirs.items():
            if not pct_list:
                continue
            flip_ct = len(pct_list)
            avg_pct = round(sum(pct_list) / len(pct_list), 1)
            all_feature_rows.append(
                {
                    "feature": feat,
                    "flip_count": flip_ct,
                    "flip_pct": round(flip_ct / total_rows * 100, 1),
                    "avg_change_pct": avg_pct,
                    "direction": direction,
                }
            )
            if flip_ct > best_flip_ct or (
                flip_ct == best_flip_ct and avg_pct < best_avg_pct
            ):
                best_flip_ct = flip_ct
                best_feat = feat
                best_dir = direction
                best_avg_pct = avg_pct

    # Sort feature_summary by flip_count desc
    all_feature_rows.sort(key=lambda r: r["flip_count"], reverse=True)

    flip_rate = round(flipped_count / total_rows, 3) if total_rows > 0 else 0.0
    target_col = pipeline.target_column.replace("_", " ")

    # ── Plain-English summary ─────────────────────────────────────────────────
    if flipped_count == 0:
        summary = (
            f"None of the {total_rows} analysed records could be flipped within the "
            f"step budget. This cohort appears firmly in its current prediction — "
            f"larger changes than typical may be required."
        )
    elif best_feat is None:
        summary = (
            f"{flipped_count} of {total_rows} records could be flipped, "
            f"but no single dominant feature emerged as the primary driver."
        )
    else:
        dir_phrase = (
            f"increasing '{best_feat.replace('_', ' ')}'"
            if best_dir == "increase"
            else f"decreasing '{best_feat.replace('_', ' ')}'"
        )
        flip_pct = round(best_flip_ct / total_rows * 100)
        summary = (
            f"For {best_flip_ct} of {total_rows} records ({flip_pct}%), "
            f"{dir_phrase} by ~{best_avg_pct}% was the primary change needed "
            f"to flip the {target_col} prediction. "
            f"This is the single most impactful lever across the cohort."
        )

    return {
        "problem_type": "classification",
        "target_column": pipeline.target_column,
        "total_rows": total_rows,
        "flipped_count": flipped_count,
        "flip_rate": flip_rate,
        "dominant_feature": best_feat,
        "dominant_direction": best_dir,
        "dominant_flip_count": best_flip_ct,
        "dominant_avg_change_pct": best_avg_pct,
        "feature_summary": all_feature_rows,
        "summary": summary,
    }
