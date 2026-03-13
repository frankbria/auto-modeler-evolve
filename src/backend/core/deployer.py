"""Deployment and prediction logic.

Design:
- Each trained model is persisted as a joblib artifact containing a sklearn
  Pipeline plus metadata (feature_names, target_col, problem_type, algorithm).
- `get_feature_schema` extracts column metadata for frontend form generation.
- `predict_single` / `predict_batch` wrap the pipeline's predict call with
  DataFrame construction, type coercion, and plain-English output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_artifact(model_path: str) -> dict:
    """Load a joblib artifact.  Returns dict with keys:
        pipeline, feature_names, target_col, problem_type, algorithm
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

def get_feature_schema(model_path: str) -> list[dict]:
    """Return a list of feature descriptors for form generation.

    Each entry:
      {"name": str, "dtype": "numeric" | "categorical", "sample_values": list}
    """
    artifact = load_artifact(model_path)
    feature_names: list[str] = artifact["feature_names"]

    # Reconstruct dtype info from the pipeline's ColumnTransformer
    preprocessor = artifact["pipeline"].named_steps.get("preprocessor")
    numeric_cols: set[str] = set()
    categorical_cols: set[str] = set()

    if hasattr(preprocessor, "transformers_") or hasattr(preprocessor, "transformers"):
        transformers = (
            preprocessor.transformers_
            if hasattr(preprocessor, "transformers_")
            else preprocessor.transformers
        )
        for name, _, cols in transformers:
            if isinstance(cols, list):
                if name == "num":
                    numeric_cols.update(cols)
                elif name == "cat":
                    categorical_cols.update(cols)

    schema = []
    for col in feature_names:
        if col in numeric_cols:
            dtype = "numeric"
        elif col in categorical_cols:
            dtype = "categorical"
        else:
            dtype = "numeric"  # default

        schema.append({"name": col, "dtype": dtype, "sample_values": []})

    return schema


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _build_input_df(feature_names: list[str], row: dict) -> pd.DataFrame:
    """Build a single-row DataFrame from a user-supplied dict."""
    data: dict[str, Any] = {}
    for col in feature_names:
        val = row.get(col)
        data[col] = [val]
    return pd.DataFrame(data)


def _format_prediction(prediction: Any, problem_type: str, probabilities: np.ndarray | None) -> dict:
    """Format a raw sklearn prediction into a user-friendly response."""
    if problem_type == "classification":
        label = str(prediction[0]) if hasattr(prediction, "__len__") else str(prediction)
        prob = None
        if probabilities is not None and len(probabilities) > 0:
            probs = probabilities[0]
            # Confidence = max class probability
            prob = float(np.max(probs))
        return {
            "prediction": label,
            "probability": prob,
            "interpretation": f"Predicted class: **{label}**"
            + (f" (confidence: {prob:.1%})" if prob is not None else ""),
        }
    else:
        val = float(prediction[0]) if hasattr(prediction, "__len__") else float(prediction)
        return {
            "prediction": round(val, 4),
            "probability": None,
            "interpretation": f"Predicted value: **{val:,.4f}**",
        }


# ---------------------------------------------------------------------------
# Public prediction API
# ---------------------------------------------------------------------------

def predict_single(model_path: str, row: dict) -> dict:
    """Make a single prediction.

    Args:
        model_path: path to the joblib artifact
        row: dict mapping feature column names to values

    Returns:
        dict with prediction, probability, interpretation
    """
    artifact = load_artifact(model_path)
    pipeline = artifact["pipeline"]
    feature_names: list[str] = artifact["feature_names"]
    problem_type: str = artifact["problem_type"]

    X = _build_input_df(feature_names, row)
    prediction = pipeline.predict(X)

    probabilities = None
    if problem_type == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(X)
        except Exception:
            pass

    result = _format_prediction(prediction, problem_type, probabilities)
    result["target_column"] = artifact.get("target_col")
    result["algorithm"] = artifact.get("algorithm")
    result["problem_type"] = problem_type
    return result


def predict_batch(model_path: str, df: pd.DataFrame) -> dict:
    """Make predictions for a DataFrame of input rows.

    Returns:
        dict with:
          - predictions: list of raw prediction values
          - probabilities: list of confidence scores (or None for regression)
          - output_df: DataFrame with original columns + prediction + confidence
    """
    artifact = load_artifact(model_path)
    pipeline = artifact["pipeline"]
    feature_names: list[str] = artifact["feature_names"]
    problem_type: str = artifact["problem_type"]
    target_col: str = artifact.get("target_col", "prediction")

    # Keep only known feature columns; fill missing with None
    X = pd.DataFrame({col: df.get(col) for col in feature_names})
    predictions = pipeline.predict(X)

    probabilities = None
    if problem_type == "classification" and hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)
            probabilities = np.max(proba, axis=1)
        except Exception:
            pass

    output = df.copy()
    pred_col = f"predicted_{target_col}"
    output[pred_col] = predictions

    if probabilities is not None:
        output["confidence"] = [round(float(p), 4) for p in probabilities]

    return {
        "predictions": [
            str(p) if problem_type == "classification" else round(float(p), 4)
            for p in predictions
        ],
        "probabilities": (
            [round(float(p), 4) for p in probabilities]
            if probabilities is not None
            else None
        ),
        "output_df": output,
        "target_column": target_col,
        "problem_type": problem_type,
        "row_count": len(df),
    }
