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
from pydantic import BaseModel
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
# 9. Drift detection
# ---------------------------------------------------------------------------


@router.get("/api/deploy/{deployment_id}/drift")
def get_prediction_drift(
    deployment_id: str,
    window: int = Query(20, ge=5, le=200, description="Predictions per comparison window"),
    session: Session = Depends(get_session),
):
    """Detect if the model's prediction distribution has shifted over time.

    Compares the first `window` predictions (baseline) against the most recent
    `window` predictions to detect distribution shift without requiring training
    data — uses only the PredictionLog.

    Returns:
    - status: "insufficient_data" | "stable" | "mild_drift" | "significant_drift"
    - drift_score: 0–100 (0 = no drift, 100 = extreme drift)
    - explanation: plain-English description
    - baseline_stats / recent_stats: mean, std, count for numeric models
    - baseline_dist / recent_dist: class proportions for classification models
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    all_logs = session.exec(
        select(PredictionLog)
        .where(PredictionLog.deployment_id == deployment_id)
    ).all()
    logs_sorted = sorted(all_logs, key=lambda l: l.created_at)

    min_required = window * 2
    if len(logs_sorted) < min_required:
        return {
            "deployment_id": deployment_id,
            "status": "insufficient_data",
            "drift_score": None,
            "explanation": (
                f"Not enough predictions yet to detect drift. "
                f"Need at least {min_required} predictions (currently {len(logs_sorted)})."
            ),
            "baseline_stats": None,
            "recent_stats": None,
            "baseline_dist": None,
            "recent_dist": None,
            "problem_type": deployment.problem_type,
        }

    baseline_logs = logs_sorted[:window]
    recent_logs = logs_sorted[-window:]

    problem_type = deployment.problem_type or "regression"

    if problem_type == "regression":
        baseline_vals = [l.prediction_numeric for l in baseline_logs if l.prediction_numeric is not None]
        recent_vals = [l.prediction_numeric for l in recent_logs if l.prediction_numeric is not None]

        if not baseline_vals or not recent_vals:
            return {
                "deployment_id": deployment_id,
                "status": "insufficient_data",
                "drift_score": None,
                "explanation": "No numeric prediction values found to compare.",
                "baseline_stats": None,
                "recent_stats": None,
                "baseline_dist": None,
                "recent_dist": None,
                "problem_type": problem_type,
            }

        b_mean = sum(baseline_vals) / len(baseline_vals)
        r_mean = sum(recent_vals) / len(recent_vals)
        b_std = (sum((v - b_mean) ** 2 for v in baseline_vals) / len(baseline_vals)) ** 0.5

        # Z-score of recent mean vs baseline distribution
        z = abs(r_mean - b_mean) / (b_std + 1e-9)
        drift_score = min(100, int(z * 33))  # z=3 → 99

        if z < 1.0:
            status = "stable"
            explanation = (
                f"Prediction values are stable. Recent average ({r_mean:.3f}) is "
                f"close to the baseline average ({b_mean:.3f})."
            )
        elif z < 2.0:
            status = "mild_drift"
            explanation = (
                f"Mild drift detected. Recent average ({r_mean:.3f}) has shifted "
                f"from the baseline ({b_mean:.3f}) — a {abs(r_mean - b_mean):.3f} difference. "
                "This may reflect normal variation or a gradual change in your data."
            )
        else:
            status = "significant_drift"
            explanation = (
                f"Significant drift detected. Recent predictions average {r_mean:.3f} "
                f"vs baseline {b_mean:.3f} — a {abs(r_mean - b_mean):.3f} shift "
                f"({z:.1f} standard deviations). Consider retraining with newer data."
            )

        r_std = (sum((v - r_mean) ** 2 for v in recent_vals) / len(recent_vals)) ** 0.5
        baseline_stats = {"mean": round(b_mean, 4), "std": round(b_std, 4), "count": len(baseline_vals)}
        recent_stats = {"mean": round(r_mean, 4), "std": round(r_std, 4), "count": len(recent_vals)}

        return {
            "deployment_id": deployment_id,
            "status": status,
            "drift_score": drift_score,
            "explanation": explanation,
            "baseline_stats": baseline_stats,
            "recent_stats": recent_stats,
            "baseline_dist": None,
            "recent_dist": None,
            "problem_type": problem_type,
        }

    else:
        # Classification: compare class distribution proportions
        def _class_dist(logs: list) -> dict[str, float]:
            counts: dict[str, int] = {}
            for l in logs:
                try:
                    label = str(json.loads(l.prediction))
                except (json.JSONDecodeError, TypeError):
                    label = "unknown"
                counts[label] = counts.get(label, 0) + 1
            total = sum(counts.values()) or 1
            return {k: round(v / total, 4) for k, v in counts.items()}

        baseline_dist = _class_dist(baseline_logs)
        recent_dist = _class_dist(recent_logs)
        all_classes = set(baseline_dist) | set(recent_dist)

        # Total variation distance (max class-proportion difference)
        tvd = sum(
            abs(recent_dist.get(c, 0) - baseline_dist.get(c, 0))
            for c in all_classes
        ) / 2

        drift_score = min(100, int(tvd * 200))  # TVD 0.5 → 100

        if tvd < 0.1:
            status = "stable"
            explanation = "Class distribution is stable — predictions are consistent with baseline patterns."
        elif tvd < 0.25:
            status = "mild_drift"
            explanation = (
                f"Mild class distribution shift (TVD={tvd:.2f}). "
                "Some class proportions have changed since deployment."
            )
        else:
            status = "significant_drift"
            explanation = (
                f"Significant class distribution shift (TVD={tvd:.2f}). "
                "The mix of predicted classes has changed substantially — this may indicate "
                "data drift or a shift in your user base. Consider retraining."
            )

        return {
            "deployment_id": deployment_id,
            "status": status,
            "drift_score": drift_score,
            "explanation": explanation,
            "baseline_stats": None,
            "recent_stats": None,
            "baseline_dist": baseline_dist,
            "recent_dist": recent_dist,
            "problem_type": problem_type,
        }


# ---------------------------------------------------------------------------
# 10. What-if analysis
# ---------------------------------------------------------------------------


class WhatIfRequest(BaseModel):
    base: dict
    overrides: dict


@router.post("/api/predict/{deployment_id}/whatif")
def whatif_prediction(
    deployment_id: str,
    body: WhatIfRequest,
    session: Session = Depends(get_session),
):
    """Compare predictions with and without feature overrides.

    Accepts a base feature dict and a set of override values. Returns the
    original prediction, the modified prediction, and the delta — in plain
    language so users understand the effect of changing a specific feature.

    Example: "What would the revenue be if region was 'West' instead of 'East'?"
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    if not deployment.pipeline_path or not Path(deployment.pipeline_path).exists():
        raise HTTPException(status_code=500, detail="Prediction pipeline not found on disk")

    run = session.get(ModelRun, deployment.model_run_id)
    if not run or not run.model_path or not Path(run.model_path).exists():
        raise HTTPException(status_code=500, detail="Model file not found on disk")

    # Original prediction
    original_result = predict_single(deployment.pipeline_path, run.model_path, body.base)

    # Modified prediction (base + overrides)
    modified_input = {**body.base, **body.overrides}
    modified_result = predict_single(deployment.pipeline_path, run.model_path, modified_input)

    original_pred = original_result["prediction"]
    modified_pred = modified_result["prediction"]

    # Compute delta for numeric predictions
    delta: float | None = None
    percent_change: float | None = None
    direction: str | None = None
    try:
        orig_num = float(original_pred)  # type: ignore[arg-type]
        mod_num = float(modified_pred)  # type: ignore[arg-type]
        delta = round(mod_num - orig_num, 4)
        percent_change = round((delta / (orig_num + 1e-9)) * 100, 2) if orig_num != 0 else None
        direction = "increase" if delta > 0 else ("decrease" if delta < 0 else "no change")
    except (TypeError, ValueError):
        pass

    # Plain-English summary
    changed_features = list(body.overrides.keys())
    changes_text = ", ".join(
        f"{k}: {body.base.get(k, '?')} → {v}" for k, v in body.overrides.items()
    )
    if delta is not None and direction != "no change":
        summary = (
            f"Changing {changes_text} would {direction} the prediction "
            f"from {original_pred} to {modified_pred} (Δ {delta:+.4f}"
            + (f", {percent_change:+.1f}%" if percent_change is not None else "")
            + ")."
        )
    elif delta == 0:
        summary = f"Changing {changes_text} has no effect on the prediction ({original_pred})."
    else:
        # Classification
        if original_pred == modified_pred:
            summary = f"Changing {changes_text} does not change the predicted class ({original_pred})."
        else:
            summary = (
                f"Changing {changes_text} changes the prediction "
                f"from '{original_pred}' to '{modified_pred}'."
            )

    return {
        "deployment_id": deployment_id,
        "original_prediction": original_pred,
        "modified_prediction": modified_pred,
        "changed_features": changed_features,
        "delta": delta,
        "percent_change": percent_change,
        "direction": direction,
        "summary": summary,
        "problem_type": deployment.problem_type,
        "target_column": deployment.target_column,
        "original_probabilities": original_result.get("probabilities"),
        "modified_probabilities": modified_result.get("probabilities"),
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
