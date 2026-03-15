"""Model deployment & prediction API endpoints.

Routes:
  POST   /api/deploy/{model_run_id}                — deploy a trained model
  GET    /api/deployments                           — list all active deployments
  GET    /api/deploy/{deployment_id}                — deployment detail + feature schema
  DELETE /api/deploy/{deployment_id}                — undeploy (soft delete)
  POST   /api/predict/{deployment_id}               — single prediction (JSON → JSON)
  POST   /api/predict/{deployment_id}/batch         — batch prediction (CSV → CSV)
  GET    /api/deploy/{deployment_id}/analytics      — prediction usage analytics
  GET    /api/deploy/{deployment_id}/logs           — paginated prediction log
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response
from sqlmodel import Session, select

import db as _db
from core.deployer import (
    build_prediction_pipeline,
    get_feature_schema,
    predict_batch,
    predict_single,
    save_pipeline,
)
from core.feature_engine import apply_transformations
from db import get_session
from models.dataset import Dataset
from models.deployment import Deployment
from models.feature_set import FeatureSet
from models.model_run import ModelRun
from models.prediction_log import PredictionLog

router = APIRouter(tags=["deployment"])

DEPLOY_DIR = Path(__file__).parent.parent / "data" / "deployments"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_deploy_context(model_run_id: str, session: Session):
    """Load ModelRun + FeatureSet + Dataset for deployment."""
    run = session.get(ModelRun, model_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found")
    if run.status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Model run status is '{run.status}'. Only completed runs can be deployed.",
        )
    if not run.model_path or not Path(run.model_path).exists():
        raise HTTPException(status_code=404, detail="Serialized model file not found")

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


# ---------------------------------------------------------------------------
# 1. Deploy model
# ---------------------------------------------------------------------------


@router.post("/api/deploy/{model_run_id}", status_code=201)
def deploy_model(
    model_run_id: str,
    session: Session = Depends(get_session),
):
    """Package a trained model for deployment.

    - Builds & serializes a PredictionPipeline from the training data
    - Creates a Deployment record with endpoint/dashboard URLs
    - Marks the ModelRun as deployed
    """
    # Check if already deployed
    existing = session.exec(
        select(Deployment).where(
            Deployment.model_run_id == model_run_id,
            Deployment.is_active == True,  # noqa: E712
        )
    ).first()
    if existing:
        return _deployment_response(existing)

    run, feature_set, dataset, file_path = _load_deploy_context(model_run_id, session)

    # Build prediction pipeline from training data
    df = pd.read_csv(file_path)
    transforms = json.loads(feature_set.transformations or "[]")
    if transforms:
        df, _ = apply_transformations(df, transforms)

    target_col = feature_set.target_column or ""
    feature_names = [c for c in df.columns if c != target_col]
    problem_type = feature_set.problem_type or "regression"

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = build_prediction_pipeline(df, feature_names, target_col, problem_type)

    # Persist pipeline
    pipeline_path = DEPLOY_DIR / f"{model_run_id}_pipeline.joblib"
    save_pipeline(pipeline, pipeline_path)

    # Create Deployment record
    deployment = Deployment(
        model_run_id=model_run_id,
        project_id=run.project_id,
        endpoint_path=f"/api/predict/{{id}}",  # filled at serve time
        dashboard_url=f"/predict/{{id}}",
        pipeline_path=str(pipeline_path),
        algorithm=run.algorithm,
        problem_type=problem_type,
        feature_names=json.dumps(feature_names),
        target_column=target_col,
        metrics=run.metrics,
    )
    session.add(deployment)

    # Fix URLs now that we have the ID
    deployment.endpoint_path = f"/api/predict/{deployment.id}"
    deployment.dashboard_url = f"/predict/{deployment.id}"

    # Mark model run as deployed
    run.is_deployed = True
    session.add(run)

    session.commit()
    session.refresh(deployment)

    return _deployment_response(deployment)


# ---------------------------------------------------------------------------
# 2. List deployments
# ---------------------------------------------------------------------------


@router.get("/api/deployments")
def list_deployments(session: Session = Depends(get_session)):
    """Return all active deployments."""
    deployments = session.exec(
        select(Deployment).where(Deployment.is_active == True)  # noqa: E712
    ).all()
    return [_deployment_response(d) for d in deployments]


# ---------------------------------------------------------------------------
# 3. Deployment detail + feature schema
# ---------------------------------------------------------------------------


@router.get("/api/deploy/{deployment_id}")
def get_deployment(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """Return deployment info + feature schema for the prediction form."""
    deployment = session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    schema = []
    if deployment.pipeline_path and Path(deployment.pipeline_path).exists():
        schema = get_feature_schema(deployment.pipeline_path)

    return {
        **_deployment_response(deployment),
        "feature_schema": schema,
    }


# ---------------------------------------------------------------------------
# 4. Undeploy
# ---------------------------------------------------------------------------


@router.delete("/api/deploy/{deployment_id}", status_code=204)
def undeploy_model(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """Soft-delete a deployment (sets is_active=False)."""
    deployment = session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    deployment.is_active = False

    # Update model run
    run = session.get(ModelRun, deployment.model_run_id)
    if run:
        run.is_deployed = False
        session.add(run)

    session.add(deployment)
    session.commit()
    return None


# ---------------------------------------------------------------------------
# 5. Single prediction
# ---------------------------------------------------------------------------


@router.post("/api/predict/{deployment_id}")
def make_prediction(
    deployment_id: str,
    input_data: dict,
    session: Session = Depends(get_session),
):
    """Make a single prediction.

    Request body: JSON object mapping feature names to values.
    Response: prediction + optional class probabilities.
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    if not deployment.pipeline_path or not Path(deployment.pipeline_path).exists():
        raise HTTPException(status_code=500, detail="Prediction pipeline not found on disk")

    run = session.get(ModelRun, deployment.model_run_id)
    if not run or not run.model_path or not Path(run.model_path).exists():
        raise HTTPException(status_code=500, detail="Model file not found on disk")

    result = predict_single(deployment.pipeline_path, run.model_path, input_data)

    # Update usage stats
    deployment.request_count += 1
    deployment.last_predicted_at = datetime.now(UTC).replace(tzinfo=None)
    session.add(deployment)

    # Log prediction for analytics
    prediction_value = result.get("prediction")
    numeric_value: float | None = None
    try:
        numeric_value = float(prediction_value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        pass

    log_entry = PredictionLog(
        deployment_id=deployment_id,
        input_features=json.dumps(input_data),
        prediction=json.dumps(prediction_value),
        prediction_numeric=numeric_value,
        confidence=result.get("confidence"),
    )
    session.add(log_entry)
    session.commit()

    return {
        "deployment_id": deployment_id,
        **result,
    }


# ---------------------------------------------------------------------------
# 6. Batch prediction
# ---------------------------------------------------------------------------


@router.post("/api/predict/{deployment_id}/batch")
def batch_prediction(
    deployment_id: str,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    """Batch prediction: upload CSV, get back CSV with predictions added.

    Returns the enriched CSV as a file download.
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    if not deployment.pipeline_path or not Path(deployment.pipeline_path).exists():
        raise HTTPException(status_code=500, detail="Prediction pipeline not found on disk")

    run = session.get(ModelRun, deployment.model_run_id)
    if not run or not run.model_path or not Path(run.model_path).exists():
        raise HTTPException(status_code=500, detail="Model file not found on disk")

    csv_bytes = file.file.read()
    result_csv = predict_batch(deployment.pipeline_path, run.model_path, csv_bytes)

    # Update usage stats
    deployment.request_count += 1
    deployment.last_predicted_at = datetime.now(UTC).replace(tzinfo=None)
    session.add(deployment)
    session.commit()

    return Response(
        content=result_csv,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"},
    )


# ---------------------------------------------------------------------------
# 7. Prediction analytics
# ---------------------------------------------------------------------------


@router.get("/api/deploy/{deployment_id}/analytics")
def get_deployment_analytics(
    deployment_id: str,
    days: int = Query(7, ge=1, le=90),
    session: Session = Depends(get_session),
):
    """Return usage analytics for a deployment.

    Returns:
    - predictions_by_day: list of {date, count} for the last `days` days
    - total_predictions: total all-time count
    - prediction_distribution: histogram buckets for numeric predictions
    - recent_avg: mean prediction value (regression only)
    - class_counts: {label: count} for classification models
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    logs = session.exec(
        select(PredictionLog).where(PredictionLog.deployment_id == deployment_id)
    ).all()

    # --- Predictions by day (last N days) ---
    day_counts: dict[str, int] = defaultdict(int)
    for log in logs:
        day_key = log.created_at.strftime("%Y-%m-%d")
        day_counts[day_key] += 1

    predictions_by_day = [
        {"date": date, "count": count}
        for date, count in sorted(day_counts.items())
    ]

    # --- Prediction distribution ---
    numeric_vals = [log.prediction_numeric for log in logs if log.prediction_numeric is not None]
    prediction_distribution: list[dict] = []
    recent_avg: float | None = None

    if numeric_vals:
        recent_avg = round(sum(numeric_vals) / len(numeric_vals), 4)
        # Build histogram with up to 10 buckets
        min_val, max_val = min(numeric_vals), max(numeric_vals)
        if min_val == max_val:
            prediction_distribution = [{"bucket": str(round(min_val, 3)), "count": len(numeric_vals)}]
        else:
            bucket_size = (max_val - min_val) / 10
            buckets: dict[int, int] = defaultdict(int)
            for v in numeric_vals:
                idx = min(int((v - min_val) / bucket_size), 9)
                buckets[idx] += 1
            prediction_distribution = [
                {
                    "bucket": f"{round(min_val + i * bucket_size, 2)}–{round(min_val + (i+1) * bucket_size, 2)}",
                    "count": buckets[i],
                }
                for i in range(10)
                if buckets[i] > 0
            ]

    # --- Classification class counts ---
    class_counts: dict[str, int] = defaultdict(int)
    for log in logs:
        if log.prediction_numeric is None:
            try:
                label = str(json.loads(log.prediction))
                class_counts[label] += 1
            except (json.JSONDecodeError, TypeError):
                pass

    return {
        "deployment_id": deployment_id,
        "total_predictions": deployment.request_count,
        "predictions_by_day": predictions_by_day,
        "prediction_distribution": prediction_distribution,
        "recent_avg": recent_avg,
        "class_counts": dict(class_counts) if class_counts else None,
        "problem_type": deployment.problem_type,
    }


# ---------------------------------------------------------------------------
# 8. Prediction log (paginated)
# ---------------------------------------------------------------------------


@router.get("/api/deploy/{deployment_id}/logs")
def get_prediction_logs(
    deployment_id: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: Session = Depends(get_session),
):
    """Return a paginated list of individual prediction records."""
    deployment = session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    all_logs = session.exec(
        select(PredictionLog).where(PredictionLog.deployment_id == deployment_id)
    ).all()

    # Sort by most recent first
    sorted_logs = sorted(all_logs, key=lambda l: l.created_at, reverse=True)
    page = sorted_logs[offset : offset + limit]

    return {
        "deployment_id": deployment_id,
        "total": len(all_logs),
        "offset": offset,
        "limit": limit,
        "logs": [
            {
                "id": log.id,
                "input_features": json.loads(log.input_features),
                "prediction": json.loads(log.prediction),
                "confidence": log.confidence,
                "created_at": log.created_at.isoformat(),
            }
            for log in page
        ],
    }


# ---------------------------------------------------------------------------
# Helper: serialize deployment
# ---------------------------------------------------------------------------


def _deployment_response(d: Deployment) -> dict:
    return {
        "id": d.id,
        "model_run_id": d.model_run_id,
        "project_id": d.project_id,
        "endpoint_path": d.endpoint_path,
        "dashboard_url": d.dashboard_url,
        "is_active": d.is_active,
        "request_count": d.request_count,
        "algorithm": d.algorithm,
        "problem_type": d.problem_type,
        "feature_names": json.loads(d.feature_names) if d.feature_names else [],
        "target_column": d.target_column,
        "metrics": json.loads(d.metrics) if d.metrics else {},
        "created_at": d.created_at.isoformat() if d.created_at else None,
        "last_predicted_at": (
            d.last_predicted_at.isoformat() if d.last_predicted_at else None
        ),
    }
