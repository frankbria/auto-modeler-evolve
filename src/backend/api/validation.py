"""Model validation & explainability API endpoints.

Routes:
  GET /api/validate/{model_run_id}/metrics      — cross-val, confusion matrix / residuals
  GET /api/validate/{model_run_id}/explain      — global feature importance
  GET /api/validate/{model_run_id}/explain/{row_index} — single-row explanation
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

import db as _db
from core.explainer import compute_feature_importance, explain_single_prediction
from core.feature_engine import apply_transformations
from core.trainer import (
    CLASSIFICATION_ALGORITHMS,
    REGRESSION_ALGORITHMS,
    prepare_features,
)
from core.validator import (
    assess_confidence_limitations,
    compute_confusion_matrix,
    compute_residuals,
    run_cross_validation,
)
from db import get_session
from models.dataset import Dataset
from models.feature_set import FeatureSet
from models.model_run import ModelRun

router = APIRouter(tags=["validation"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_run_context(model_run_id: str, session: Session):
    """Load ModelRun + FeatureSet + Dataset from DB."""
    run = session.get(ModelRun, model_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found")

    if run.status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Model run status is '{run.status}'. Validation requires a completed run.",
        )

    if not run.model_path or not Path(run.model_path).exists():
        raise HTTPException(status_code=404, detail="Serialized model file not found on disk")

    feature_set = session.get(FeatureSet, run.feature_set_id)
    if not feature_set:
        raise HTTPException(status_code=404, detail="Feature set not found")

    dataset = session.exec(
        select(Dataset).where(Dataset.id == feature_set.dataset_id)
    ).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    return run, feature_set, dataset, file_path


def _build_Xy(
    file_path: Path,
    feature_set: FeatureSet,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load CSV, apply transforms, return (X, y, feature_names)."""
    df = pd.read_csv(file_path)
    transforms = json.loads(feature_set.transformations or "[]")
    if transforms:
        df, _ = apply_transformations(df, transforms)

    target_col = feature_set.target_column
    feature_cols = [c for c in df.columns if c != target_col]
    problem_type = feature_set.problem_type or "regression"

    X, y, _ = prepare_features(df, feature_cols, target_col, problem_type)
    return X, y, feature_cols


def _get_unfitted_model(algorithm: str, problem_type: str):
    """Return a fresh unfitted estimator for cross-validation."""
    registry = (
        REGRESSION_ALGORITHMS if problem_type == "regression"
        else CLASSIFICATION_ALGORITHMS
    )
    if algorithm not in registry:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown algorithm '{algorithm}'",
        )
    info = registry[algorithm]
    return info["class"](**info["params"])


# ---------------------------------------------------------------------------
# 1. Validation metrics
# ---------------------------------------------------------------------------


@router.get("/api/validate/{model_run_id}/metrics")
def get_validation_metrics(
    model_run_id: str,
    session: Session = Depends(get_session),
):
    """Cross-validation + confusion matrix (classification) or residuals (regression)."""
    run, feature_set, _dataset, file_path = _load_run_context(model_run_id, session)

    problem_type = feature_set.problem_type or "regression"
    metrics = json.loads(run.metrics) if run.metrics else {}

    X, y, feature_cols = _build_Xy(file_path, feature_set)

    # Cross-validation
    unfitted = _get_unfitted_model(run.algorithm, problem_type)
    cv_result = run_cross_validation(unfitted, X, y, problem_type)

    # Confusion matrix or residuals
    fitted_model = joblib.load(run.model_path)
    y_pred = fitted_model.predict(X)

    if problem_type == "classification":
        cm_result = compute_confusion_matrix(y, y_pred)
        error_analysis = {"type": "confusion_matrix", **cm_result}
    else:
        residual_result = compute_residuals(y, y_pred)
        error_analysis = {"type": "residuals", **residual_result}

    # Confidence assessment
    cv_std = cv_result.get("std")
    confidence = assess_confidence_limitations(
        metrics=metrics,
        problem_type=problem_type,
        n_rows=len(X),
        n_features=len(feature_cols),
        cv_std=cv_std,
    )

    return {
        "model_run_id": model_run_id,
        "algorithm": run.algorithm,
        "problem_type": problem_type,
        "held_out_metrics": metrics,
        "cross_validation": cv_result,
        "error_analysis": error_analysis,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# 2. Global feature importance
# ---------------------------------------------------------------------------


@router.get("/api/validate/{model_run_id}/explain")
def get_global_explanation(
    model_run_id: str,
    session: Session = Depends(get_session),
):
    """Return global feature importance for the trained model."""
    run, feature_set, _dataset, file_path = _load_run_context(model_run_id, session)

    problem_type = feature_set.problem_type or "regression"
    X, y, feature_cols = _build_Xy(file_path, feature_set)

    fitted_model = joblib.load(run.model_path)
    importance = compute_feature_importance(fitted_model, feature_cols)

    top_3 = [item["feature"] for item in importance[:3]]
    summary = (
        f"The top factors driving predictions are: {', '.join(top_3)}. "
        "Higher bars mean a feature has more influence on the model's output."
    )

    return {
        "model_run_id": model_run_id,
        "algorithm": run.algorithm,
        "problem_type": problem_type,
        "feature_importance": importance,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# 3. Individual prediction explanation
# ---------------------------------------------------------------------------


@router.get("/api/validate/{model_run_id}/explain/{row_index}")
def get_row_explanation(
    model_run_id: str,
    row_index: int,
    session: Session = Depends(get_session),
):
    """Explain a single prediction using feature contributions."""
    run, feature_set, _dataset, file_path = _load_run_context(model_run_id, session)

    problem_type = feature_set.problem_type or "regression"
    target_col = feature_set.target_column or "target"
    X, y, feature_cols = _build_Xy(file_path, feature_set)

    if row_index < 0 or row_index >= len(X):
        raise HTTPException(
            status_code=400,
            detail=f"row_index {row_index} out of range (dataset has {len(X)} rows).",
        )

    fitted_model = joblib.load(run.model_path)
    x_row = X[row_index]

    explanation = explain_single_prediction(
        model=fitted_model,
        x_row=x_row,
        X_train=X,
        feature_names=feature_cols,
        problem_type=problem_type,
        target_name=target_col,
    )

    return {
        "model_run_id": model_run_id,
        "row_index": row_index,
        "actual_value": round(float(y[row_index]), 4) if y is not None else None,
        **explanation,
    }
