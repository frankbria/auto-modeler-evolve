"""Model validation and explainability API endpoints.

Endpoints:
1. GET  /api/validate/{model_run_id}/metrics
        → Cross-validation with CI, confusion matrix or residual analysis,
          limitations assessment
2. GET  /api/validate/{model_run_id}/explain
        → Global feature importance (permutation-based)
3. GET  /api/validate/{model_run_id}/explain/{row_index}
        → Per-prediction explanation (perturbation-based)
"""

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from core.explainer import compute_global_importance, explain_prediction
from core.feature_engine import apply_transformations
from core.validator import validate_model
from db import get_session
from models.dataset import Dataset
from models.feature_set import FeatureSet
from models.model_run import ModelRun

router = APIRouter(prefix="/api/validate", tags=["validation"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model_and_df(model_run_id: str, session: Session) -> tuple[ModelRun, pd.DataFrame]:
    """Load a ModelRun and the associated DataFrame."""
    run = session.get(ModelRun, model_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="ModelRun not found")
    if run.status != "done" or not run.model_path:
        raise HTTPException(status_code=400, detail="ModelRun has no trained model")

    dataset = session.get(Dataset, run.dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    df = pd.read_csv(file_path)

    # Apply feature transformations if the run was based on a feature set
    if run.feature_set_id:
        fs = session.get(FeatureSet, run.feature_set_id)
        if fs and fs.transformations:
            transforms = json.loads(fs.transformations)
            df, _ = apply_transformations(df, transforms)

    return run, df


# ---------------------------------------------------------------------------
# 1. Validation metrics
# ---------------------------------------------------------------------------

@router.get("/{model_run_id}/metrics")
def get_validation_metrics(
    model_run_id: str,
    session: Session = Depends(get_session),
):
    """Cross-validation results with confidence intervals, error analysis, limitations."""
    run, df = _load_model_and_df(model_run_id, session)

    try:
        result = validate_model(run.model_path, df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Validation failed: {exc}") from exc

    return {
        "model_run_id": model_run_id,
        "algorithm": run.algorithm,
        "display_name": run.display_name,
        "target_column": run.target_column,
        **result,
    }


# ---------------------------------------------------------------------------
# 2. Global feature importance
# ---------------------------------------------------------------------------

@router.get("/{model_run_id}/explain")
def get_global_importance(
    model_run_id: str,
    session: Session = Depends(get_session),
):
    """Global feature importance via permutation importance."""
    run, df = _load_model_and_df(model_run_id, session)

    try:
        result = compute_global_importance(run.model_path, df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}") from exc

    return {
        "model_run_id": model_run_id,
        "algorithm": run.algorithm,
        "display_name": run.display_name,
        "target_column": run.target_column,
        **result,
    }


# ---------------------------------------------------------------------------
# 3. Individual prediction explanation
# ---------------------------------------------------------------------------

@router.get("/{model_run_id}/explain/{row_index}")
def get_prediction_explanation(
    model_run_id: str,
    row_index: int,
    session: Session = Depends(get_session),
):
    """Explain a single prediction: what factors pushed it up or down."""
    run, df = _load_model_and_df(model_run_id, session)

    try:
        result = explain_prediction(run.model_path, df, row_index)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {exc}") from exc

    return {
        "model_run_id": model_run_id,
        "algorithm": run.algorithm,
        "display_name": run.display_name,
        "target_column": run.target_column,
        **result,
    }
