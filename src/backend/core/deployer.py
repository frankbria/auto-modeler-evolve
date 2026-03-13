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
import json
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
    column_types: dict[str, str]         # 'numeric' | 'categorical'
    label_encoders: dict[str, LabelEncoder] = field(default_factory=dict)
    medians: dict[str, float] = field(default_factory=dict)
    target_column: str = ""
    problem_type: str = "regression"
    target_encoder: Optional[LabelEncoder] = None
    target_classes: Optional[list] = None

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
    df_clean = df[feature_names + [target_col]].dropna(subset=[target_col]).reset_index(drop=True)

    for col in feature_names:
        series = df_clean[col]
        if pd.api.types.is_numeric_dtype(series):
            pipeline.column_types[col] = "numeric"
            pipeline.medians[col] = float(series.median())
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
        result["probabilities"] = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

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
        schema.append(entry)
    return schema
