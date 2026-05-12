"""Model deployment & prediction API endpoints.

Routes:
  POST   /api/deploy/{model_run_id}                — deploy a trained model
  GET    /api/deployments                           — list all active deployments
  GET    /api/deploy/{deployment_id}                — deployment detail + feature schema
  DELETE /api/deploy/{deployment_id}                — undeploy (soft delete)
  POST   /api/predict/{deployment_id}               — single prediction (JSON → JSON)
  POST   /api/predict/{deployment_id}/batch         — batch prediction (CSV → CSV)
  POST   /api/predict/{deployment_id}/scenarios     — bulk scenario comparison (N what-ifs in one call)
  POST   /api/predict/compare                       — cross-deployment model comparison (same input, N models)
  GET    /api/deploy/{deployment_id}/analytics      — prediction usage analytics
  GET    /api/deploy/{deployment_id}/logs           — paginated prediction log
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Header, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from core.analyzer import (
    compute_batch_job_results,
    compute_covariate_drift_alert,
    compute_confidence_trend,
    compute_deployment_health_item,
    compute_deployments_overview,
    compute_prediction_audit,
    compute_usage_pattern,
)
from core.deployer import (
    build_prediction_pipeline,
    explain_prediction,
    get_feature_schema,
    predict_batch,
    predict_single,
    save_pipeline,
)
from core.feature_engine import apply_transformations
from db import get_session
from models.dataset import Dataset
from models.deployment import Deployment
from models.deployment_version import DeploymentVersion
from models.feature_set import FeatureSet
from models.model_run import ModelRun
from models.ab_test import ABTest
from models.deployment_preset import DeploymentPreset
from models.feedback_record import FeedbackRecord
from models.prediction_log import PredictionLog
from models.webhook_config import WebhookConfig
from models.prediction_alert_rule import PredictionAlertRule

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


def _build_pipeline_for_run(
    run: "ModelRun",
    feature_set: "FeatureSet",
    file_path,
    session: "Session",
) -> tuple:
    """Build a PredictionPipeline for a model run. Returns (pipeline, pipeline_path)."""
    import pandas as _pd

    df = _pd.read_csv(file_path)
    transforms = json.loads(feature_set.transformations or "[]")
    if transforms:
        df, _ = apply_transformations(df, transforms)

    target_col = feature_set.target_column or ""
    feature_names = [c for c in df.columns if c != target_col]
    problem_type = feature_set.problem_type or "regression"

    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    pipeline = build_prediction_pipeline(df, feature_names, target_col, problem_type)

    # Compute residual std for regression prediction intervals
    if problem_type == "regression" and target_col in df.columns:
        try:
            import joblib as _joblib
            import numpy as _np

            model = _joblib.load(run.model_path)
            df_clean = df[feature_names + [target_col]].dropna(subset=[target_col])
            X_train = pipeline.transform_df(df_clean[feature_names])
            y_true = df_clean[target_col].values.astype(float)
            y_pred = model.predict(X_train)
            pipeline.residual_std = float(_np.std(y_true - y_pred))
        except Exception:
            pipeline.residual_std = 0.0

    pipeline_path = DEPLOY_DIR / f"{run.id}_pipeline.joblib"
    save_pipeline(pipeline, pipeline_path)
    return pipeline, pipeline_path, feature_names, target_col, problem_type


def _archive_current_version(deployment, session) -> None:
    """Mark all current versions of a deployment as not-current (archive them).

    Called before updating the deployment to a new model. The new current version
    record is created by the caller after this function returns.
    """
    current_versions = session.exec(
        select(DeploymentVersion).where(
            DeploymentVersion.deployment_id == deployment.id,
            DeploymentVersion.is_current == True,  # noqa: E712
        )
    ).all()
    for v in current_versions:
        v.is_current = False
        session.add(v)


def execute_deployment(model_run_id: str, session: Session) -> dict:
    """Build and register a deployment for a completed model run.

    Returns the deployment response dict. On first deploy, creates a new Deployment
    record and saves DeploymentVersion v1. On re-deploy (same project, different run),
    archives the current version and updates the Deployment in-place so the endpoint
    URL stays stable. Idempotent — returns the existing deployment if same run is
    already active.
    """
    run, feature_set, dataset, file_path = _load_deploy_context(model_run_id, session)

    # Check if this exact run is already deployed (idempotent path)
    existing_same_run = session.exec(
        select(Deployment).where(
            Deployment.model_run_id == model_run_id,
            Deployment.is_active == True,  # noqa: E712
        )
    ).first()
    if existing_same_run:
        return _deployment_response(existing_same_run)

    _, pipeline_path, feature_names, target_col, problem_type = _build_pipeline_for_run(
        run, feature_set, file_path, session
    )

    # Check if there's already an active deployment for this project
    existing_for_project = session.exec(
        select(Deployment).where(
            Deployment.project_id == run.project_id,
            Deployment.is_active == True,  # noqa: E712
        )
    ).first()

    if existing_for_project:
        # Re-deploy: archive current state, increment version, update deployment in-place
        _archive_current_version(existing_for_project, session)

        new_version_number = (
            getattr(existing_for_project, "current_version_number", 1) + 1
        )

        # Update deployment to point at new model
        existing_for_project.model_run_id = model_run_id
        existing_for_project.pipeline_path = str(pipeline_path)
        existing_for_project.algorithm = run.algorithm
        existing_for_project.problem_type = problem_type
        existing_for_project.feature_names = json.dumps(feature_names)
        existing_for_project.target_column = target_col
        existing_for_project.metrics = run.metrics
        existing_for_project.current_version_number = new_version_number
        session.add(existing_for_project)

        # Save the new (current) version snapshot
        new_version = DeploymentVersion(
            deployment_id=existing_for_project.id,
            version_number=new_version_number,
            model_run_id=model_run_id,
            algorithm=run.algorithm,
            problem_type=problem_type,
            target_column=target_col,
            metrics=run.metrics,
            pipeline_path=str(pipeline_path),
            is_current=True,
        )
        session.add(new_version)

        # Mark old run as not deployed, new run as deployed
        old_run = session.get(ModelRun, existing_for_project.model_run_id)
        if old_run and old_run.id != model_run_id:
            old_run.is_deployed = False
            session.add(old_run)
        run.is_deployed = True
        session.add(run)

        session.commit()
        session.refresh(existing_for_project)
        return _deployment_response(existing_for_project)

    # First deployment for this project
    deployment = Deployment(
        model_run_id=model_run_id,
        project_id=run.project_id,
        endpoint_path="/api/predict/{id}",  # filled below
        dashboard_url="/predict/{id}",
        pipeline_path=str(pipeline_path),
        algorithm=run.algorithm,
        problem_type=problem_type,
        feature_names=json.dumps(feature_names),
        target_column=target_col,
        metrics=run.metrics,
        current_version_number=1,
    )
    session.add(deployment)
    # Flush to get the deployment.id
    session.flush()

    # Fix URLs now that we have the ID
    deployment.endpoint_path = f"/api/predict/{deployment.id}"
    deployment.dashboard_url = f"/predict/{deployment.id}"

    # Save initial version snapshot (v1)
    v1 = DeploymentVersion(
        deployment_id=deployment.id,
        version_number=1,
        model_run_id=model_run_id,
        algorithm=run.algorithm,
        problem_type=problem_type,
        target_column=target_col,
        metrics=run.metrics,
        pipeline_path=str(pipeline_path),
        is_current=True,
    )
    session.add(v1)

    # Mark model run as deployed
    run.is_deployed = True
    session.add(run)

    session.commit()
    session.refresh(deployment)

    return _deployment_response(deployment)


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
    return execute_deployment(model_run_id, session)


# ---------------------------------------------------------------------------
# 2. List deployments
# ---------------------------------------------------------------------------


@router.get("/api/deployments")
def list_deployments(
    project_id: str | None = Query(None),
    session: Session = Depends(get_session),
):
    """Return all active deployments, optionally filtered by project."""
    q = select(Deployment).where(Deployment.is_active == True)  # noqa: E712
    if project_id:
        q = q.where(Deployment.project_id == project_id)
    deployments = session.exec(q).all()
    return [_deployment_response(d) for d in deployments]


# ---------------------------------------------------------------------------
# 3. Deployments overview (must be before /api/deploy/{deployment_id})
# ---------------------------------------------------------------------------


@router.get("/api/deploy/overview")
def deployments_overview(session: Session = Depends(get_session)):
    """Return a cross-project status overview for all active deployments.

    Aggregates health, request volume, environment, and configuration data
    for every active Deployment row across all projects.
    """
    from datetime import UTC, datetime

    now = datetime.now(UTC).replace(tzinfo=None)
    active_deployments = list(
        session.exec(
            select(Deployment).where(Deployment.is_active == True)  # noqa: E712
        ).all()
    )

    deployment_summaries: list[dict] = []
    for dep in active_deployments:
        cutoff_7d = now - timedelta(days=7)
        cutoff_today = now.replace(hour=0, minute=0, second=0, microsecond=0)

        all_logs = list(
            session.exec(
                select(PredictionLog).where(
                    PredictionLog.deployment_id == dep.id,
                    PredictionLog.created_at >= cutoff_7d,
                )
            ).all()
        )
        predictions_last_7d = len(all_logs)
        predictions_today = sum(1 for lg in all_logs if lg.created_at >= cutoff_today)

        from models.project import Project as _Project

        proj = session.get(_Project, dep.project_id)
        project_name = proj.name if proj else "Unknown"

        health_item = compute_deployment_health_item(
            deployment_id=dep.id,
            algorithm=dep.algorithm,
            target_column=dep.target_column,
            created_at=dep.created_at,
            request_count=dep.request_count,
            last_predicted_at=dep.last_predicted_at,
            environment=dep.environment,
            now=now,
        )

        deployment_summaries.append(
            {
                **health_item,
                "project_id": dep.project_id,
                "project_name": project_name,
                "algorithm": dep.algorithm,
                "created_at_iso": dep.created_at.isoformat()
                if dep.created_at
                else None,
                "last_predicted_at_iso": (
                    dep.last_predicted_at.isoformat() if dep.last_predicted_at else None
                ),
                "api_key_enabled": dep.api_key_enabled,
                "rate_limit_rpm": dep.rate_limit_rpm,
                "monthly_quota": dep.monthly_quota,
                "predictions_last_7d": predictions_last_7d,
                "predictions_today": predictions_today,
                "dashboard_url": dep.dashboard_url,
                "endpoint_path": dep.endpoint_path,
            }
        )

    return compute_deployments_overview(deployment_summaries)


# ---------------------------------------------------------------------------
# 4. Deployment detail + feature schema
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
# 4b. API key management
# ---------------------------------------------------------------------------


@router.post("/api/deploy/{deployment_id}/api-key", status_code=201)
def generate_api_key(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """Generate (or regenerate) an API key for a deployment.

    The plaintext key is returned ONCE in this response.
    Only its hash is stored — it cannot be retrieved again.
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    key = secrets.token_urlsafe(32)
    salt = secrets.token_hex(16)
    key_hash = hashlib.sha256(f"{salt}:{key}".encode()).hexdigest()

    deployment.api_key_enabled = True
    deployment.api_key_hash = key_hash
    deployment.api_key_salt = salt
    session.add(deployment)
    session.commit()

    return {
        "deployment_id": deployment_id,
        "api_key": key,  # shown only once
        "message": "API key generated. Store this key — it cannot be retrieved again.",
    }


@router.delete("/api/deploy/{deployment_id}/api-key", status_code=204)
def disable_api_key(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """Remove API key protection from a deployment (makes it publicly accessible again)."""
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    deployment.api_key_enabled = False
    deployment.api_key_hash = None
    deployment.api_key_salt = None
    session.add(deployment)
    session.commit()
    return None


# ---------------------------------------------------------------------------
# 4c. Rate limiting & monthly quota management
# ---------------------------------------------------------------------------


class RateLimitRequest(BaseModel):
    rate_limit_rpm: int | None = None  # None = remove limit; 0 = remove limit; >0 = set
    monthly_quota: int | None = None  # None = remove quota; 0 = remove; >0 = set


@router.put("/api/deploy/{deployment_id}/rate-limit")
def set_rate_limit(
    deployment_id: str,
    body: RateLimitRequest,
    session: Session = Depends(get_session),
):
    """Set or remove per-minute rate limit and/or monthly prediction quota.

    Pass rate_limit_rpm=N to enforce N requests/minute (sliding window).
    Pass rate_limit_rpm=0 or null to remove the limit.
    Pass monthly_quota=N to enforce N predictions per rolling 30 days.
    Pass monthly_quota=0 or null to remove the quota.
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    if body.rate_limit_rpm is not None and body.rate_limit_rpm < 0:
        raise HTTPException(
            status_code=422, detail="rate_limit_rpm must be >= 0 (0 = remove limit)"
        )
    if body.monthly_quota is not None and body.monthly_quota < 0:
        raise HTTPException(
            status_code=422, detail="monthly_quota must be >= 0 (0 = remove quota)"
        )

    # 0 means "remove" — store as None
    deployment.rate_limit_rpm = (
        body.rate_limit_rpm if (body.rate_limit_rpm or 0) > 0 else None
    )
    deployment.monthly_quota = (
        body.monthly_quota if (body.monthly_quota or 0) > 0 else None
    )
    session.add(deployment)
    session.commit()

    return {
        "deployment_id": deployment_id,
        "rate_limit_rpm": deployment.rate_limit_rpm,
        "monthly_quota": deployment.monthly_quota,
        "message": _rate_limit_summary(
            deployment.rate_limit_rpm, deployment.monthly_quota
        ),
    }


def _rate_limit_summary(rpm: int | None, quota: int | None) -> str:
    parts = []
    if rpm:
        parts.append(f"{rpm} requests/minute")
    if quota:
        parts.append(f"{quota} predictions/month")
    if parts:
        return "Rate limits active: " + " and ".join(parts) + "."
    return "No rate limits configured — endpoint is open access."


@router.get("/api/deploy/{deployment_id}/quota-status")
def get_quota_status(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """Return current monthly quota usage for a deployment.

    Returns:
        quota_enabled: whether a monthly_quota is configured
        monthly_quota: the configured limit (or None)
        used_this_month: number of predictions in the last 30 days
        remaining: quota - used (None if no quota)
        pct_used: percentage of quota used (None if no quota)
        rate_limit_rpm: configured per-minute limit (or None)
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    quota = getattr(deployment, "monthly_quota", None)
    rpm = getattr(deployment, "rate_limit_rpm", None)

    # Count predictions in last 30 days
    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=30)
    from sqlmodel import func as _func

    used = session.exec(
        select(_func.count(PredictionLog.id)).where(
            PredictionLog.deployment_id == deployment_id,
            PredictionLog.created_at >= cutoff,
        )
    ).one()

    remaining = (quota - used) if quota else None
    pct_used = round((used / quota * 100), 1) if quota else None

    alert_pct = getattr(deployment, "quota_alert_threshold_pct", None)

    return {
        "deployment_id": deployment_id,
        "quota_enabled": quota is not None,
        "monthly_quota": quota,
        "used_this_month": used,
        "remaining": remaining,
        "pct_used": pct_used,
        "rate_limit_rpm": rpm,
        "rate_limit_enabled": rpm is not None,
        "quota_alert_threshold_pct": alert_pct,
        "quota_alert_enabled": alert_pct is not None,
        "summary": _rate_limit_summary(rpm, quota),
    }


# ---------------------------------------------------------------------------
# 4d. Quota alert threshold configuration
# ---------------------------------------------------------------------------


class QuotaAlertRequest(BaseModel):
    threshold_pct: int | None = None  # 1-99 = set; 0 or null = remove


@router.put("/api/deploy/{deployment_id}/quota-alert")
def set_quota_alert(
    deployment_id: str,
    body: QuotaAlertRequest,
    session: Session = Depends(get_session),
):
    """Configure or remove a quota usage alert threshold.

    When monthly_quota is set and usage reaches this percentage of the quota,
    registered webhooks with the 'quota_alert' event type are dispatched.
    The alert fires exactly once when usage first crosses each threshold.

    Pass threshold_pct=N (1-99) to enable alerts at N% of the monthly quota.
    Pass threshold_pct=0 or null to disable quota alerts.
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    pct = body.threshold_pct
    if pct is not None and pct < 0:
        raise HTTPException(
            status_code=422, detail="threshold_pct must be >= 0 (0 = remove alert)"
        )
    if pct is not None and pct > 99:
        raise HTTPException(
            status_code=422,
            detail="threshold_pct must be <= 99 (100% means quota is already exhausted)",
        )

    deployment.quota_alert_threshold_pct = pct if pct and pct > 0 else None
    session.add(deployment)
    session.commit()

    quota = getattr(deployment, "monthly_quota", None)
    return {
        "deployment_id": deployment_id,
        "quota_alert_enabled": deployment.quota_alert_threshold_pct is not None,
        "quota_alert_threshold_pct": deployment.quota_alert_threshold_pct,
        "monthly_quota": quota,
        "message": (
            f"Quota alerts enabled at {deployment.quota_alert_threshold_pct}% of {quota} predictions."
            if deployment.quota_alert_threshold_pct and quota
            else (
                "Quota alerts enabled at "
                f"{deployment.quota_alert_threshold_pct}% — set a monthly_quota to activate."
                if deployment.quota_alert_threshold_pct
                else "Quota alerts disabled."
            )
        ),
    }


# ---------------------------------------------------------------------------
# 5a. Cross-deployment model comparison (must appear before parameterised routes)
# ---------------------------------------------------------------------------


class CompareRequest(BaseModel):
    deployment_ids: list[str]  # 2-4 deployment IDs to compare
    features: dict  # input feature values (same for all deployments)


@router.post("/api/predict/compare")
def compare_deployments(
    body: CompareRequest,
    session: Session = Depends(get_session),
):
    """Compare predictions from multiple deployed model versions on the same input.

    Accepts 2-4 deployment IDs and a feature dict. Returns a prediction from each
    deployment so analysts can verify whether a retrained model actually improved.

    Returns:
        {"results": [{deployment_id, algorithm, trained_at, prediction,
                       confidence_interval, confidence, probabilities, error}]}
    """
    if len(body.deployment_ids) < 2:
        raise HTTPException(
            status_code=400, detail="At least 2 deployment IDs required for comparison"
        )
    if len(body.deployment_ids) > 4:
        raise HTTPException(
            status_code=400, detail="Maximum 4 deployments can be compared at once"
        )

    results = []
    for dep_id in body.deployment_ids:
        dep = session.get(Deployment, dep_id)
        if not dep or not dep.is_active:
            results.append(
                {"deployment_id": dep_id, "error": "Deployment not found or inactive"}
            )
            continue

        run = session.get(ModelRun, dep.model_run_id)
        model_path = run.model_path if run else None

        if not (
            dep.pipeline_path
            and model_path
            and Path(dep.pipeline_path).exists()
            and Path(model_path).exists()
        ):
            results.append(
                {
                    "deployment_id": dep_id,
                    "algorithm": dep.algorithm,
                    "trained_at": (
                        dep.created_at.isoformat() if dep.created_at else None
                    ),
                    "error": "Model file not found",
                }
            )
            continue

        try:
            pred = predict_single(dep.pipeline_path, model_path, body.features)
            entry: dict = {
                "deployment_id": dep_id,
                "algorithm": dep.algorithm,
                "trained_at": dep.created_at.isoformat() if dep.created_at else None,
                "prediction": pred["prediction"],
                "problem_type": pred.get("problem_type"),
                "target_column": pred.get("target_column"),
                "error": None,
            }
            if "confidence_interval" in pred:
                entry["confidence_interval"] = pred["confidence_interval"]
            if "confidence" in pred:
                entry["confidence"] = pred["confidence"]
            if "probabilities" in pred:
                entry["probabilities"] = pred["probabilities"]
            results.append(entry)
        except Exception as exc:
            results.append(
                {
                    "deployment_id": dep_id,
                    "algorithm": dep.algorithm,
                    "trained_at": (
                        dep.created_at.isoformat() if dep.created_at else None
                    ),
                    "error": str(exc),
                }
            )

    return {"results": results}


# ---------------------------------------------------------------------------
# 5b. Single prediction
# ---------------------------------------------------------------------------


@router.post("/api/predict/{deployment_id}")
def make_prediction(
    deployment_id: str,
    input_data: dict,
    authorization: str | None = Header(default=None),
    session: Session = Depends(get_session),
):
    """Make a single prediction.

    Request body: JSON object mapping feature names to values.
    Response: prediction + optional class probabilities.
    If the deployment has API key protection, include Authorization: Bearer <key>.
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    _verify_api_key(deployment, authorization)

    # Rate limiting — per-minute sliding window
    rpm = getattr(deployment, "rate_limit_rpm", None)
    if rpm:
        if not _check_rate_limit(deployment_id, rpm):
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded. This deployment allows {rpm} "
                    "requests per minute. Please slow down or contact the model owner."
                ),
            )

    # Monthly quota — rolling 30-day cap
    allowed, used, quota = _check_monthly_quota(deployment, session)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Monthly prediction quota exceeded ({used}/{quota} predictions used "
                "in the last 30 days). Contact the model owner to increase the limit."
            ),
        )

    if not deployment.pipeline_path or not Path(deployment.pipeline_path).exists():
        raise HTTPException(
            status_code=500, detail="Prediction pipeline not found on disk"
        )

    run = session.get(ModelRun, deployment.model_run_id)
    if not run or not run.model_path or not Path(run.model_path).exists():
        raise HTTPException(status_code=500, detail="Model file not found on disk")

    # A/B test routing: check if this deployment has an active champion-challenger test
    ab_variant: str | None = None
    serving_pipeline = deployment.pipeline_path
    serving_model = run.model_path

    active_test = session.exec(
        select(ABTest).where(
            ABTest.champion_id == deployment_id,
            ABTest.is_active == True,  # noqa: E712
        )
    ).first()
    if active_test:
        ab_variant = "champion"
        if random.random() >= active_test.champion_split_pct / 100:
            challenger = session.get(Deployment, active_test.challenger_id)
            if challenger and challenger.is_active and challenger.pipeline_path:
                challenger_run = session.get(ModelRun, challenger.model_run_id)
                if (
                    challenger_run
                    and challenger_run.model_path
                    and Path(challenger.pipeline_path).exists()
                    and Path(challenger_run.model_path).exists()
                ):
                    serving_pipeline = challenger.pipeline_path
                    serving_model = challenger_run.model_path
                    ab_variant = "challenger"

    _t0 = time.monotonic()
    result = predict_single(
        serving_pipeline, serving_model, input_data, provided_features=input_data
    )
    response_ms = round((time.monotonic() - _t0) * 1000, 2)

    # Update usage stats (always on champion endpoint — that's the URL the analyst shared)
    deployment.request_count += 1
    deployment.last_predicted_at = datetime.now(UTC).replace(tzinfo=None)
    session.add(deployment)

    # Log prediction for analytics (deployment_id = champion; ab_variant tags which model served)
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
        response_ms=response_ms,
        ab_variant=ab_variant,
    )
    session.add(log_entry)
    session.commit()

    # Quota alert: fire webhooks if usage just crossed the configured threshold
    _quota_alert_pct = getattr(deployment, "quota_alert_threshold_pct", None)
    _quota_cap = getattr(deployment, "monthly_quota", None)
    if _quota_alert_pct and _quota_cap:
        # Re-count including the prediction we just committed
        _cutoff_alert = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=30)
        from sqlmodel import func as _func_alert

        _used_now = session.exec(
            select(_func_alert.count(PredictionLog.id)).where(
                PredictionLog.deployment_id == deployment_id,
                PredictionLog.created_at >= _cutoff_alert,
            )
        ).one()
        import threading as _threading

        _t = _threading.Thread(
            target=_check_and_fire_quota_alert,
            args=(deployment_id, _used_now, _quota_cap, _quota_alert_pct),
            daemon=True,
        )
        _t.start()

    # Prediction alert rules: check business-rule-based conditions in background
    import threading as _threading2

    _predicted_class: str | None = None
    if result.get("prediction") is not None and numeric_value is None:
        _predicted_class = str(result["prediction"])

    _t2 = _threading2.Thread(
        target=_fire_alert_rules,
        args=(deployment_id, numeric_value, result.get("confidence"), _predicted_class),
        daemon=True,
    )
    _t2.start()

    return {
        "deployment_id": deployment_id,
        "ab_variant": ab_variant,
        **result,
    }


# ---------------------------------------------------------------------------
# 6. Batch prediction
# ---------------------------------------------------------------------------


@router.post("/api/predict/{deployment_id}/batch")
def batch_prediction(
    deployment_id: str,
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
    session: Session = Depends(get_session),
):
    """Batch prediction: upload CSV, get back CSV with predictions added.

    Returns the enriched CSV as a file download.
    If the deployment has API key protection, include Authorization: Bearer <key>.
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    _verify_api_key(deployment, authorization)

    if not deployment.pipeline_path or not Path(deployment.pipeline_path).exists():
        raise HTTPException(
            status_code=500, detail="Prediction pipeline not found on disk"
        )

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
# 7. Prediction explanation (live — explain why the model predicted X)
# ---------------------------------------------------------------------------


@router.post("/api/predict/{deployment_id}/explain")
def explain_single_prediction(
    deployment_id: str,
    input_data: dict,
    authorization: str | None = Header(default=None),
    session: Session = Depends(get_session),
):
    """Explain a single prediction using feature contributions.

    Takes the same feature dict as /api/predict/{deployment_id}.
    Returns prediction + per-feature contributions so the dashboard can
    show "why did the model predict this?" in plain language.

    Returns:
        {prediction, target_column, problem_type, contributions, summary, top_drivers}
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    _verify_api_key(deployment, authorization)

    if not deployment.pipeline_path or not Path(deployment.pipeline_path).exists():
        raise HTTPException(
            status_code=500, detail="Prediction pipeline not found on disk"
        )

    run = session.get(ModelRun, deployment.model_run_id)
    if not run or not run.model_path or not Path(run.model_path).exists():
        raise HTTPException(status_code=500, detail="Model file not found on disk")

    result = explain_prediction(deployment.pipeline_path, run.model_path, input_data)
    return result


# ---------------------------------------------------------------------------
# 7b. Production prediction explanation (chat-triggered — most recent PredictionLog)
# ---------------------------------------------------------------------------


@router.get("/api/deploy/{deployment_id}/explain-prediction")
def explain_production_prediction(
    deployment_id: str,
    prediction_id: str | None = Query(default=None),
    session: Session = Depends(get_session),
):
    """Return feature-contribution explanation for a production prediction.

    Without prediction_id, explains the most recent PredictionLog record.
    With prediction_id, explains that specific log entry.
    Returns 404 when no PredictionLog records exist or deployment is inactive.
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    if prediction_id:
        log_entry = session.get(PredictionLog, prediction_id)
        if not log_entry or log_entry.deployment_id != deployment_id:
            raise HTTPException(status_code=404, detail="Prediction log not found")
    else:
        log_entry = session.exec(
            select(PredictionLog)
            .where(PredictionLog.deployment_id == deployment_id)
            .order_by(PredictionLog.created_at.desc())
        ).first()
        if not log_entry:
            raise HTTPException(
                status_code=404, detail="No prediction logs found for this deployment"
            )

    import json as _json_ep

    input_data = log_entry.input_features
    if isinstance(input_data, str):
        input_data = _json_ep.loads(input_data)

    run = session.get(ModelRun, deployment.model_run_id)
    if not run or not run.model_path or not Path(run.model_path).exists():
        raise HTTPException(status_code=500, detail="Model file not found on disk")

    if not deployment.pipeline_path or not Path(deployment.pipeline_path).exists():
        raise HTTPException(
            status_code=500, detail="Prediction pipeline not found on disk"
        )

    result = explain_prediction(deployment.pipeline_path, run.model_path, input_data)
    result["prediction_log_id"] = log_entry.id
    result["created_at"] = (
        log_entry.created_at.isoformat() if log_entry.created_at else None
    )
    result["confidence"] = log_entry.confidence
    result["algorithm"] = getattr(run, "algorithm", None)
    result["deployment_id"] = deployment_id
    return result


# ---------------------------------------------------------------------------
# 7c. Aggregate production explanation (patterns across many predictions)
# ---------------------------------------------------------------------------


@router.get("/api/deploy/{deployment_id}/aggregate-explanations")
def get_aggregate_explanations(
    deployment_id: str,
    n: int = Query(50, ge=5, le=200),
    session: Session = Depends(get_session),
):
    """Return aggregated feature contribution statistics across recent production predictions.

    Loads the last `n` PredictionLog entries and runs a single-pass explanation
    aggregation — which features consistently drive predictions up or down.
    Returns 404 when the deployment is inactive or no prediction logs exist.
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    logs = session.exec(
        select(PredictionLog)
        .where(PredictionLog.deployment_id == deployment_id)
        .order_by(PredictionLog.created_at.desc())
        .limit(n)
    ).all()

    if not logs:
        raise HTTPException(
            status_code=404, detail="No prediction logs found for this deployment"
        )

    run = session.get(ModelRun, deployment.model_run_id)
    if not run or not run.model_path or not Path(run.model_path).exists():
        raise HTTPException(status_code=500, detail="Model file not found on disk")
    if not deployment.pipeline_path or not Path(deployment.pipeline_path).exists():
        raise HTTPException(
            status_code=500, detail="Prediction pipeline not found on disk"
        )

    import json as _json_ae

    input_data_list = []
    for log in logs:
        try:
            data = log.input_features
            if isinstance(data, str):
                data = _json_ae.loads(data)
            if isinstance(data, dict):
                input_data_list.append(data)
        except Exception:  # noqa: BLE001
            pass

    from core.deployer import compute_aggregate_explanations

    result = compute_aggregate_explanations(
        deployment.pipeline_path, run.model_path, input_data_list
    )
    result["deployment_id"] = deployment_id
    return result


# ---------------------------------------------------------------------------
# 8. Prediction analytics
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
        {"date": date, "count": count} for date, count in sorted(day_counts.items())
    ]

    # --- Prediction distribution ---
    numeric_vals = [
        log.prediction_numeric for log in logs if log.prediction_numeric is not None
    ]
    prediction_distribution: list[dict] = []
    recent_avg: float | None = None

    if numeric_vals:
        recent_avg = round(sum(numeric_vals) / len(numeric_vals), 4)
        # Build histogram with up to 10 buckets
        min_val, max_val = min(numeric_vals), max(numeric_vals)
        if min_val == max_val:
            prediction_distribution = [
                {"bucket": str(round(min_val, 3)), "count": len(numeric_vals)}
            ]
        else:
            bucket_size = (max_val - min_val) / 10
            buckets: dict[int, int] = defaultdict(int)
            for v in numeric_vals:
                idx = min(int((v - min_val) / bucket_size), 9)
                buckets[idx] += 1
            prediction_distribution = [
                {
                    "bucket": f"{round(min_val + i * bucket_size, 2)}–{round(min_val + (i + 1) * bucket_size, 2)}",
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
    sorted_logs = sorted(all_logs, key=lambda log: log.created_at, reverse=True)
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


@router.get("/api/deploy/{deployment_id}/recent-predictions")
def get_recent_predictions(
    deployment_id: str,
    n: int = Query(10, ge=1, le=50),
    session: Session = Depends(get_session),
):
    """Return the last N predictions in a lightweight format for the chat card."""
    deployment = session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    from sqlalchemy import func as _sql_func

    total = session.exec(
        select(_sql_func.count(PredictionLog.id)).where(
            PredictionLog.deployment_id == deployment_id
        )
    ).one()

    logs = session.exec(
        select(PredictionLog)
        .where(PredictionLog.deployment_id == deployment_id)
        .order_by(PredictionLog.created_at.desc())  # type: ignore[arg-type]
        .limit(n)
    ).all()

    rows = []
    for log in logs:
        try:
            feats = json.loads(log.input_features) if log.input_features else {}
        except Exception:  # noqa: BLE001
            feats = {}
        input_summary = [
            {"key": str(k), "value": str(v)} for k, v in list(feats.items())[:3]
        ]
        try:
            pred_raw = json.loads(log.prediction)
        except Exception:  # noqa: BLE001
            pred_raw = log.prediction
        rows.append(
            {
                "id": log.id[:8],
                "created_at": log.created_at.isoformat(),
                "prediction": str(pred_raw),
                "confidence": round(log.confidence * 100, 1)
                if log.confidence is not None
                else None,
                "response_ms": round(log.response_ms, 1)
                if log.response_ms is not None
                else None,
                "input_summary": input_summary,
                "ab_variant": log.ab_variant,
            }
        )

    if total > 0:
        summary = (
            f"Showing the {len(rows)} most recent of {total:,} "
            f"total prediction{'s' if total != 1 else ''}."
        )
    else:
        summary = "No predictions have been made yet."

    return {
        "deployment_id": deployment_id,
        "n_shown": len(rows),
        "total_all_time": total,
        "predictions": rows,
        "export_url": f"/api/deploy/{deployment_id}/prediction-logs/export",
        "summary": summary,
    }


@router.get("/api/deploy/{deployment_id}/prediction-logs/export")
def export_prediction_logs(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """Download all prediction logs for a deployment as a CSV file.

    Columns: id, created_at, prediction, confidence, response_ms, and one column
    per input feature (flattened from the JSON blob stored at log time).
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    logs = session.exec(
        select(PredictionLog)
        .where(PredictionLog.deployment_id == deployment_id)
        .order_by(PredictionLog.created_at)  # type: ignore[arg-type]
    ).all()

    # Collect all feature key names (union across all logs)
    feature_keys: list[str] = []
    parsed_features: list[dict] = []
    for log in logs:
        try:
            feats = json.loads(log.input_features)
        except (json.JSONDecodeError, TypeError):
            feats = {}
        parsed_features.append(feats)
        for k in feats:
            if k not in feature_keys:
                feature_keys.append(k)

    base_cols = ["id", "created_at", "prediction", "confidence", "response_ms"]
    all_cols = base_cols + feature_keys

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=all_cols, extrasaction="ignore")
    writer.writeheader()
    for log, feats in zip(logs, parsed_features):
        try:
            pred_raw = json.loads(log.prediction)
            pred_val = (
                pred_raw
                if not isinstance(pred_raw, (list, dict))
                else json.dumps(pred_raw)
            )
        except (json.JSONDecodeError, TypeError):
            pred_val = log.prediction
        row: dict = {
            "id": log.id,
            "created_at": log.created_at.isoformat(),
            "prediction": pred_val,
            "confidence": log.confidence if log.confidence is not None else "",
            "response_ms": log.response_ms if log.response_ms is not None else "",
        }
        row.update(feats)
        writer.writerow(row)

    csv_bytes = output.getvalue().encode("utf-8")
    filename = f"predictions_{deployment_id[:8]}.csv"
    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ---------------------------------------------------------------------------
# 9. Drift detection
# ---------------------------------------------------------------------------


@router.get("/api/deploy/{deployment_id}/drift")
def get_prediction_drift(
    deployment_id: str,
    window: int = Query(
        20, ge=5, le=200, description="Predictions per comparison window"
    ),
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
        select(PredictionLog).where(PredictionLog.deployment_id == deployment_id)
    ).all()
    logs_sorted = sorted(all_logs, key=lambda log: log.created_at)

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
        baseline_vals = [
            log.prediction_numeric
            for log in baseline_logs
            if log.prediction_numeric is not None
        ]
        recent_vals = [
            log.prediction_numeric
            for log in recent_logs
            if log.prediction_numeric is not None
        ]

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
        b_std = (
            sum((v - b_mean) ** 2 for v in baseline_vals) / len(baseline_vals)
        ) ** 0.5

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
        baseline_stats = {
            "mean": round(b_mean, 4),
            "std": round(b_std, 4),
            "count": len(baseline_vals),
        }
        recent_stats = {
            "mean": round(r_mean, 4),
            "std": round(r_std, 4),
            "count": len(recent_vals),
        }

        drift_result = {
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
        if drift_score >= 50:
            try:
                from core.webhook import EVENT_DRIFT_DETECTED, dispatch_webhooks

                dispatch_webhooks(
                    deployment_id,
                    EVENT_DRIFT_DETECTED,
                    {
                        "drift_score": drift_score,
                        "status": status,
                        "explanation": explanation,
                    },
                )
            except Exception:
                pass
        return drift_result

    else:
        # Classification: compare class distribution proportions
        def _class_dist(logs: list) -> dict[str, float]:
            counts: dict[str, int] = {}
            for log in logs:
                try:
                    label = str(json.loads(log.prediction))
                except (json.JSONDecodeError, TypeError):
                    label = "unknown"
                counts[label] = counts.get(label, 0) + 1
            total = sum(counts.values()) or 1
            return {k: round(v / total, 4) for k, v in counts.items()}

        baseline_dist = _class_dist(baseline_logs)
        recent_dist = _class_dist(recent_logs)
        all_classes = set(baseline_dist) | set(recent_dist)

        # Total variation distance (max class-proportion difference)
        tvd = (
            sum(
                abs(recent_dist.get(c, 0) - baseline_dist.get(c, 0))
                for c in all_classes
            )
            / 2
        )

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

        drift_result = {
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
        if drift_score >= 50:
            try:
                from core.webhook import EVENT_DRIFT_DETECTED, dispatch_webhooks

                dispatch_webhooks(
                    deployment_id,
                    EVENT_DRIFT_DETECTED,
                    {
                        "drift_score": drift_score,
                        "status": status,
                        "explanation": explanation,
                    },
                )
            except Exception:
                pass
        return drift_result


# ---------------------------------------------------------------------------
# 10. SLA monitoring
# ---------------------------------------------------------------------------


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Return the pct-th percentile of a pre-sorted list (0-100 scale)."""
    if not sorted_vals:
        return 0.0
    idx = (pct / 100) * (len(sorted_vals) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    return round(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo), 2)


@router.get("/api/deploy/{deployment_id}/sla")
def get_sla_metrics(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """Return prediction latency percentiles (p50/p95/p99) and per-day averages.

    Uses response_ms stored on PredictionLog entries. Returns:
    - p50_ms, p95_ms, p99_ms: latency percentiles across all recorded predictions
    - avg_ms: mean latency
    - sample_count: number of predictions with latency data
    - alert: True when p95 > 500ms
    - alert_message: plain-English description when alert is True
    - latency_by_day: [{"date": "YYYY-MM-DD", "avg_ms": N}] for sparkline
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    logs = session.exec(
        select(PredictionLog).where(PredictionLog.deployment_id == deployment_id)
    ).all()

    timed_logs = [log for log in logs if log.response_ms is not None]

    if not timed_logs:
        return {
            "deployment_id": deployment_id,
            "sample_count": 0,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "avg_ms": None,
            "alert": False,
            "alert_message": None,
            "latency_by_day": [],
        }

    latencies = sorted(log.response_ms for log in timed_logs)  # type: ignore[misc]
    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    p99 = _percentile(latencies, 99)
    avg_ms = round(sum(latencies) / len(latencies), 2)

    alert = p95 > 500.0
    alert_message = (
        f"p95 latency is {p95}ms — above the 500ms target. "
        "Consider retraining with fewer features or switching to a simpler algorithm."
        if alert
        else None
    )

    # Per-day averages for sparkline (last 30 days)
    day_totals: dict[str, list[float]] = defaultdict(list)
    for log in timed_logs:
        day_key = log.created_at.strftime("%Y-%m-%d")
        day_totals[day_key].append(log.response_ms)  # type: ignore[arg-type]

    latency_by_day = [
        {"date": date, "avg_ms": round(sum(ms) / len(ms), 2)}
        for date, ms in sorted(day_totals.items())
    ]

    return {
        "deployment_id": deployment_id,
        "sample_count": len(timed_logs),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "avg_ms": avg_ms,
        "alert": alert,
        "alert_message": alert_message,
        "latency_by_day": latency_by_day,
    }


# ---------------------------------------------------------------------------
# 11. What-if analysis
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
        raise HTTPException(
            status_code=500, detail="Prediction pipeline not found on disk"
        )

    run = session.get(ModelRun, deployment.model_run_id)
    if not run or not run.model_path or not Path(run.model_path).exists():
        raise HTTPException(status_code=500, detail="Model file not found on disk")

    # Original prediction
    original_result = predict_single(
        deployment.pipeline_path, run.model_path, body.base
    )

    # Modified prediction (base + overrides)
    modified_input = {**body.base, **body.overrides}
    modified_result = predict_single(
        deployment.pipeline_path, run.model_path, modified_input
    )

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
        percent_change = (
            round((delta / (orig_num + 1e-9)) * 100, 2) if orig_num != 0 else None
        )
        direction = (
            "increase" if delta > 0 else ("decrease" if delta < 0 else "no change")
        )
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
# 11. Bulk scenario comparison — run multiple what-if inputs side by side
# ---------------------------------------------------------------------------


class ScenarioItem(BaseModel):
    label: str  # Human-readable label for this scenario (e.g. "High Marketing")
    overrides: (
        dict  # Feature overrides on top of base (same as WhatIfRequest.overrides)
    )


class ScenarioRequest(BaseModel):
    base: dict  # Base feature values (used as the baseline prediction)
    scenarios: list[ScenarioItem]  # Up to 10 scenarios to compare


@router.post("/api/predict/{deployment_id}/scenarios")
def compare_scenarios(
    deployment_id: str,
    body: ScenarioRequest,
    session: Session = Depends(get_session),
):
    """Run multiple prediction scenarios against a shared base case.

    Accepts a base feature dict and up to 10 labelled override sets.
    Returns the base prediction plus a result for each scenario including
    delta, percent change, and direction vs the base.  Ideal for "what if
    revenue if region = X vs Y vs Z?" analysis.

    Request body:
        {
          "base": {feature: value, ...},
          "scenarios": [
            {"label": "Scenario A", "overrides": {feature: new_value, ...}},
            ...
          ]
        }

    Returns:
        {
          "deployment_id": str,
          "base_prediction": decoded value,
          "problem_type": str,
          "target_column": str,
          "scenarios": [
            {
              "label": str,
              "overrides": dict,
              "prediction": decoded value,
              "delta": float | None,
              "percent_change": float | None,
              "direction": str | None,   # "increase" | "decrease" | "no change"
              "probabilities": dict | None,  # classification only
            }
          ],
          "summary": str,  # plain-English comparison of all scenarios
        }
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    if not deployment.pipeline_path or not Path(deployment.pipeline_path).exists():
        raise HTTPException(
            status_code=500, detail="Prediction pipeline not found on disk"
        )

    run = session.get(ModelRun, deployment.model_run_id)
    if not run or not run.model_path or not Path(run.model_path).exists():
        raise HTTPException(status_code=500, detail="Model file not found on disk")

    if len(body.scenarios) > 10:
        raise HTTPException(
            status_code=400, detail="Maximum 10 scenarios allowed per request"
        )

    if not body.scenarios:
        raise HTTPException(status_code=400, detail="At least one scenario is required")

    # Base prediction
    base_result = predict_single(deployment.pipeline_path, run.model_path, body.base)
    base_pred = base_result["prediction"]

    # Numeric base value for delta calculations
    base_numeric: float | None = None
    try:
        base_numeric = float(base_pred)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        pass

    # Run each scenario
    scenario_results = []
    for scenario in body.scenarios:
        modified_input = {**body.base, **scenario.overrides}
        result = predict_single(
            deployment.pipeline_path, run.model_path, modified_input
        )
        pred = result["prediction"]

        delta: float | None = None
        percent_change: float | None = None
        direction: str | None = None

        if base_numeric is not None:
            try:
                mod_num = float(pred)  # type: ignore[arg-type]
                delta = round(mod_num - base_numeric, 4)
                if base_numeric != 0:
                    percent_change = round((delta / abs(base_numeric)) * 100, 2)
                direction = (
                    "increase"
                    if delta > 0
                    else ("decrease" if delta < 0 else "no change")
                )
            except (TypeError, ValueError):
                pass
        else:
            # Classification: direction based on class change
            direction = "same" if pred == base_pred else "changed"

        scenario_results.append(
            {
                "label": scenario.label,
                "overrides": scenario.overrides,
                "prediction": pred,
                "delta": delta,
                "percent_change": percent_change,
                "direction": direction,
                "probabilities": result.get("probabilities"),
            }
        )

    # Build a plain-English summary
    target_col = deployment.target_column or "the target"
    if base_numeric is not None:
        # Regression: find best and worst scenarios
        sorted_by_delta = sorted(
            [s for s in scenario_results if s["delta"] is not None],
            key=lambda s: s["delta"],  # type: ignore[arg-type]
            reverse=True,
        )
        if sorted_by_delta:
            best = sorted_by_delta[0]
            worst = sorted_by_delta[-1]
            if best["label"] == worst["label"]:
                summary = (
                    f"Base {target_col} = {base_pred}. "
                    f"Scenario '{best['label']}' predicts {best['prediction']} "
                    f"(Δ {best['delta']:+.4f})."
                )
            else:
                summary = (
                    f"Base {target_col} = {base_pred}. "
                    f"Best outcome: '{best['label']}' → {best['prediction']} "
                    f"({best['percent_change']:+.1f}%). "
                    f"Worst outcome: '{worst['label']}' → {worst['prediction']} "
                    f"({worst['percent_change']:+.1f}%)."
                    if (
                        best["percent_change"] is not None
                        and worst["percent_change"] is not None
                    )
                    else f"Base = {base_pred}. Range: {worst['prediction']} to {best['prediction']}."
                )
        else:
            summary = f"Base {target_col} = {base_pred}. {len(scenario_results)} scenario(s) computed."
    else:
        # Classification: count class changes
        changed = sum(1 for s in scenario_results if s["direction"] == "changed")
        summary = (
            f"Base predicted class: '{base_pred}'. "
            f"{changed} of {len(scenario_results)} scenario(s) predict a different class."
        )

    return {
        "deployment_id": deployment_id,
        "base_prediction": base_pred,
        "base_probabilities": base_result.get("probabilities"),
        "problem_type": deployment.problem_type,
        "target_column": deployment.target_column,
        "scenarios": scenario_results,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# 12. Prediction feedback — record actual outcomes
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    prediction_log_id: str | None = None
    actual_value: float | None = None  # For regression: true numeric outcome
    actual_label: str | None = None  # For classification: true class label
    is_correct: bool | None = None  # For classification: optional override
    comment: str | None = None  # Free-text note from the user


@router.post("/api/predict/{deployment_id}/feedback", status_code=201)
def submit_feedback(
    deployment_id: str,
    body: FeedbackRequest,
    session: Session = Depends(get_session),
):
    """Record the actual outcome for a past prediction.

    This closes the feedback loop: after a prediction is made and the real
    outcome is known, the user can record it here. The system tracks how often
    the model was right over time.

    For regression models: provide ``actual_value`` (the true numeric outcome).
    For classification models: provide ``actual_label`` and/or ``is_correct``.

    If ``prediction_log_id`` is provided, we can auto-compute ``is_correct``
    for classification by comparing the stored prediction to the actual label.
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    if body.actual_value is None and body.actual_label is None:
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: actual_value (regression) or actual_label (classification).",
        )

    # Auto-compute is_correct for classification when we have the prediction log
    is_correct = body.is_correct
    if is_correct is None and body.prediction_log_id and body.actual_label:
        log_entry = session.get(PredictionLog, body.prediction_log_id)
        if log_entry:
            try:
                stored_pred = str(json.loads(log_entry.prediction))
                is_correct = stored_pred == body.actual_label
            except (json.JSONDecodeError, TypeError):
                pass

    record = FeedbackRecord(
        deployment_id=deployment_id,
        prediction_log_id=body.prediction_log_id,
        actual_value=body.actual_value,
        actual_label=body.actual_label,
        is_correct=is_correct,
        comment=body.comment,
    )
    session.add(record)
    session.commit()
    session.refresh(record)

    return {
        "id": record.id,
        "deployment_id": deployment_id,
        "prediction_log_id": record.prediction_log_id,
        "actual_value": record.actual_value,
        "actual_label": record.actual_label,
        "is_correct": record.is_correct,
        "comment": record.comment,
        "created_at": record.created_at.isoformat(),
        "message": "Feedback recorded. Thank you — this helps improve the model over time.",
    }


# ---------------------------------------------------------------------------
# 12. Feedback accuracy stats
# ---------------------------------------------------------------------------


@router.get("/api/deploy/{deployment_id}/feedback-accuracy")
def get_feedback_accuracy(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """Compute real-world prediction accuracy from user feedback.

    For regression: shows mean absolute error between predicted and actual values,
    and a ±% error rate.

    For classification: shows how often the model predicted the right class
    (based on user-provided is_correct flags or computed comparisons).

    Returns ``status: "no_feedback"`` when no records exist yet.
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    feedback_records = session.exec(
        select(FeedbackRecord).where(FeedbackRecord.deployment_id == deployment_id)
    ).all()

    total_feedback = len(feedback_records)

    if total_feedback == 0:
        return {
            "deployment_id": deployment_id,
            "status": "no_feedback",
            "total_feedback": 0,
            "message": (
                "No feedback recorded yet. After making predictions and seeing the "
                "real outcomes, record them using POST /api/predict/{id}/feedback to "
                "track how well the model is performing in practice."
            ),
            "problem_type": deployment.problem_type,
        }

    problem_type = deployment.problem_type or "regression"

    if problem_type == "regression":
        # Compute MAE between actual and predicted values
        pairs = []
        for fb in feedback_records:
            if fb.actual_value is not None and fb.prediction_log_id:
                log_entry = session.get(PredictionLog, fb.prediction_log_id)
                if log_entry and log_entry.prediction_numeric is not None:
                    pairs.append((fb.actual_value, log_entry.prediction_numeric))

        if not pairs:
            # We have feedback but no paired prediction logs — just report count
            actual_values = [
                fb.actual_value
                for fb in feedback_records
                if fb.actual_value is not None
            ]
            return {
                "deployment_id": deployment_id,
                "status": "feedback_only",
                "total_feedback": total_feedback,
                "actual_values_recorded": len(actual_values),
                "message": (
                    f"{total_feedback} actual outcome(s) recorded. "
                    "Link prediction_log_id to feedback entries to compute error metrics."
                ),
                "problem_type": problem_type,
            }

        actual_vals, predicted_vals = zip(*pairs, strict=False)
        mae = sum(abs(a - p) for a, p in pairs) / len(pairs)
        avg_actual = sum(actual_vals) / len(actual_vals)
        pct_error = (mae / (abs(avg_actual) + 1e-9)) * 100

        if pct_error < 5:
            verdict = "excellent"
            verdict_msg = "Excellent — predictions are very close to actual outcomes."
        elif pct_error < 15:
            verdict = "good"
            verdict_msg = (
                "Good accuracy — predictions are reasonably close to actual outcomes."
            )
        elif pct_error < 30:
            verdict = "moderate"
            verdict_msg = "Moderate accuracy — consider adding more features or retraining with newer data."
        else:
            verdict = "poor"
            verdict_msg = (
                "Predictions are significantly off. Retraining is recommended."
            )

        return {
            "deployment_id": deployment_id,
            "status": "computed",
            "total_feedback": total_feedback,
            "paired_count": len(pairs),
            "mae": round(mae, 4),
            "pct_error": round(pct_error, 2),
            "avg_actual": round(avg_actual, 4),
            "verdict": verdict,
            "message": (
                f"Based on {len(pairs)} matched prediction(s): "
                f"mean absolute error = {mae:.4f} ({pct_error:.1f}% of average actual value). "
                + verdict_msg
            ),
            "problem_type": problem_type,
        }

    else:
        # Classification: count correct vs incorrect
        correct_records = [fb for fb in feedback_records if fb.is_correct is True]
        incorrect_records = [fb for fb in feedback_records if fb.is_correct is False]
        unknown_records = [fb for fb in feedback_records if fb.is_correct is None]

        correct_count = len(correct_records)
        incorrect_count = len(incorrect_records)
        rated_count = correct_count + incorrect_count

        if rated_count == 0:
            return {
                "deployment_id": deployment_id,
                "status": "feedback_only",
                "total_feedback": total_feedback,
                "message": (
                    f"{total_feedback} feedback record(s) found, but none have is_correct set. "
                    "Provide actual_label with feedback to enable accuracy tracking."
                ),
                "problem_type": problem_type,
            }

        accuracy = correct_count / rated_count

        if accuracy >= 0.9:
            verdict = "excellent"
            verdict_msg = (
                "Excellent real-world accuracy — the model is performing as expected."
            )
        elif accuracy >= 0.75:
            verdict = "good"
            verdict_msg = "Good real-world accuracy — the model is mostly reliable."
        elif accuracy >= 0.6:
            verdict = "moderate"
            verdict_msg = (
                "Moderate accuracy. Consider reviewing feature inputs or retraining."
            )
        else:
            verdict = "poor"
            verdict_msg = (
                "Below-expected accuracy in practice. Retraining is recommended."
            )

        return {
            "deployment_id": deployment_id,
            "status": "computed",
            "total_feedback": total_feedback,
            "rated_count": rated_count,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "unknown_count": len(unknown_records),
            "accuracy_from_feedback": round(accuracy, 4),
            "verdict": verdict,
            "message": (
                f"Based on {rated_count} rated prediction(s): "
                f"the model was correct {correct_count} time(s) — "
                f"{accuracy:.1%} real-world accuracy. " + verdict_msg
            ),
            "problem_type": problem_type,
        }


# ---------------------------------------------------------------------------
# 13. Unified model health score
# ---------------------------------------------------------------------------


@router.get("/api/deploy/{deployment_id}/health")
def get_model_health(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """Return a unified health score for a deployed model.

    Combines three signals into a single 0-100 health score:
    - Model age: how long since training (freshness degrades over time)
    - Feedback accuracy: real-world performance from user-recorded outcomes
    - Prediction drift: distribution shift detected from PredictionLog

    Returns:
    - health_score: 0–100 (100 = perfectly healthy)
    - status: "healthy" | "warning" | "critical"
    - component_scores: individual scores for each signal
    - recommendations: plain-English actions to take
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    run = session.get(ModelRun, deployment.model_run_id)

    # ----- Component 1: Model age score -----
    age_days = 0
    age_score = 100
    age_note = "Model is fresh."
    if run and run.created_at:
        now = datetime.now(UTC).replace(tzinfo=None)
        age_days = max(0, (now - run.created_at).days)
        if age_days <= 30:
            age_score = 100
            age_note = f"Model is {age_days} day(s) old — still fresh."
        elif age_days <= 60:
            age_score = 75
            age_note = f"Model is {age_days} day(s) old — consider retraining monthly."
        elif age_days <= 90:
            age_score = 50
            age_note = f"Model is {age_days} day(s) old — retraining is recommended."
        else:
            age_score = 25
            age_note = f"Model is {age_days} day(s) old — likely stale. Retrain with recent data."

    # ----- Component 2: Feedback accuracy score -----
    feedback_score = 100
    feedback_note = "No feedback recorded yet."
    feedback_records = session.exec(
        select(FeedbackRecord).where(FeedbackRecord.deployment_id == deployment_id)
    ).all()
    if feedback_records:
        problem_type = deployment.problem_type or "regression"
        if problem_type == "regression":
            pairs = []
            for fb in feedback_records:
                if fb.actual_value is not None and fb.prediction_log_id:
                    log_entry = session.get(PredictionLog, fb.prediction_log_id)
                    if log_entry and log_entry.prediction_numeric is not None:
                        pairs.append((fb.actual_value, log_entry.prediction_numeric))
            if pairs:
                avg_actual = sum(a for a, _ in pairs) / len(pairs)
                mae = sum(abs(a - p) for a, p in pairs) / len(pairs)
                pct_error = (mae / (abs(avg_actual) + 1e-9)) * 100
                if pct_error < 5:
                    feedback_score = 100
                    feedback_note = f"Excellent accuracy ({pct_error:.1f}% error)."
                elif pct_error < 15:
                    feedback_score = 75
                    feedback_note = f"Good accuracy ({pct_error:.1f}% error)."
                elif pct_error < 30:
                    feedback_score = 50
                    feedback_note = f"Moderate accuracy ({pct_error:.1f}% error). Consider retraining."
                else:
                    feedback_score = 20
                    feedback_note = f"Poor accuracy ({pct_error:.1f}% error). Retraining strongly recommended."
        else:
            # Classification: use is_correct ratios
            rated = [fb for fb in feedback_records if fb.is_correct is not None]
            if rated:
                accuracy = sum(1 for fb in rated if fb.is_correct) / len(rated)
                if accuracy >= 0.9:
                    feedback_score = 100
                    feedback_note = f"Excellent accuracy ({accuracy:.1%} correct)."
                elif accuracy >= 0.75:
                    feedback_score = 75
                    feedback_note = f"Good accuracy ({accuracy:.1%} correct)."
                elif accuracy >= 0.6:
                    feedback_score = 50
                    feedback_note = f"Moderate accuracy ({accuracy:.1%} correct). Consider retraining."
                else:
                    feedback_score = 20
                    feedback_note = f"Poor accuracy ({accuracy:.1%} correct). Retraining strongly recommended."

    # ----- Component 3: Drift score -----
    drift_health_score = 100
    drift_note = "Not enough predictions to assess drift."
    all_logs = session.exec(
        select(PredictionLog).where(PredictionLog.deployment_id == deployment_id)
    ).all()
    if len(all_logs) >= 40:  # minimum for drift comparison
        logs_sorted = sorted(all_logs, key=lambda log: log.created_at)
        window = 20
        baseline_logs = logs_sorted[:window]
        recent_logs = logs_sorted[-window:]
        problem_type = deployment.problem_type or "regression"

        if problem_type == "regression":
            baseline_vals = [
                log.prediction_numeric
                for log in baseline_logs
                if log.prediction_numeric is not None
            ]
            recent_vals = [
                log.prediction_numeric
                for log in recent_logs
                if log.prediction_numeric is not None
            ]
            if baseline_vals and recent_vals:
                b_mean = sum(baseline_vals) / len(baseline_vals)
                r_mean = sum(recent_vals) / len(recent_vals)
                b_std = (
                    sum((v - b_mean) ** 2 for v in baseline_vals) / len(baseline_vals)
                ) ** 0.5
                z = abs(r_mean - b_mean) / (b_std + 1e-9)
                if z < 1.0:
                    drift_health_score = 100
                    drift_note = "Prediction distribution is stable."
                elif z < 2.0:
                    drift_health_score = 60
                    drift_note = "Mild drift detected — monitor closely."
                else:
                    drift_health_score = 25
                    drift_note = "Significant drift detected — retraining is strongly recommended."
        else:
            # Classification: total variation distance
            baseline_preds = [
                str(json.loads(log.prediction))
                for log in baseline_logs
                if log.prediction
            ]
            recent_preds = [
                str(json.loads(log.prediction)) for log in recent_logs if log.prediction
            ]
            all_classes = set(baseline_preds + recent_preds)
            if all_classes and baseline_preds and recent_preds:
                b_n, r_n = len(baseline_preds), len(recent_preds)
                tvd = 0.5 * sum(
                    abs(baseline_preds.count(c) / b_n - recent_preds.count(c) / r_n)
                    for c in all_classes
                )
                if tvd < 0.1:
                    drift_health_score = 100
                    drift_note = "Prediction class distribution is stable."
                elif tvd < 0.25:
                    drift_health_score = 60
                    drift_note = "Mild class distribution shift — monitor closely."
                else:
                    drift_health_score = 25
                    drift_note = "Significant class shift detected — retraining is strongly recommended."

    # ----- Composite score: weighted average -----
    # Weights: feedback matters most (if available), then drift, then age
    has_feedback = len(feedback_records) > 0
    has_drift_data = len(all_logs) >= 40

    if has_feedback and has_drift_data:
        health_score = int(
            feedback_score * 0.4 + drift_health_score * 0.35 + age_score * 0.25
        )
    elif has_feedback:
        health_score = int(feedback_score * 0.55 + age_score * 0.45)
    elif has_drift_data:
        health_score = int(drift_health_score * 0.6 + age_score * 0.4)
    else:
        health_score = age_score

    # ----- Status label -----
    if health_score >= 75:
        status = "healthy"
    elif health_score >= 50:
        status = "warning"
    else:
        status = "critical"

    # ----- Recommendations -----
    recommendations = []
    if age_score < 75:
        recommendations.append(
            "Retrain the model with more recent data to improve freshness."
        )
    if feedback_score < 75 and has_feedback:
        recommendations.append(
            "Real-world accuracy is declining — consider retraining or adding features."
        )
    if drift_health_score < 75 and has_drift_data:
        recommendations.append(
            "Prediction drift detected — the input data may have changed; retraining is advised."
        )
    if not has_feedback:
        recommendations.append(
            "Record actual outcomes using the feedback form to enable real-world accuracy tracking."
        )
    if not recommendations:
        recommendations.append(
            "Model health is good. Continue monitoring for drift and feedback accuracy."
        )

    # Fire "health_degraded" webhook when score drops below 60
    if health_score < 60:
        try:
            from core.webhook import EVENT_HEALTH_DEGRADED, dispatch_webhooks

            dispatch_webhooks(
                deployment_id,
                EVENT_HEALTH_DEGRADED,
                {
                    "health_score": health_score,
                    "status": status,
                    "recommendations": recommendations,
                },
            )
        except Exception:
            pass

    return {
        "deployment_id": deployment_id,
        "health_score": health_score,
        "status": status,
        "model_age_days": age_days,
        "component_scores": {
            "age": age_score,
            "feedback": feedback_score if has_feedback else None,
            "drift": drift_health_score if has_drift_data else None,
        },
        "component_notes": {
            "age": age_note,
            "feedback": feedback_note,
            "drift": drift_note,
        },
        "recommendations": recommendations,
        "has_feedback_data": has_feedback,
        "has_drift_data": has_drift_data,
        "algorithm": deployment.algorithm,
        "problem_type": deployment.problem_type,
    }


# ---------------------------------------------------------------------------
# 14. Developer integration snippets
# ---------------------------------------------------------------------------


@router.get("/api/deploy/{deployment_id}/integration")
def get_integration_snippets(
    deployment_id: str,
    base_url: str = Query(
        "http://localhost:8000", description="Base URL of the API server"
    ),
    session: Session = Depends(get_session),
):
    """Return copy-pasteable code snippets for calling the prediction API.

    Generates curl, Python (requests), and JavaScript (fetch) examples
    automatically from the deployment's feature schema. Designed for the
    developer handoff use case: an analyst deploys a model and sends the
    generated code to their developer to integrate into reporting tools.

    Returns:
        {endpoint_url, example_input, curl, python, javascript, openapi_url}
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    feature_names = (
        json.loads(deployment.feature_names) if deployment.feature_names else []
    )

    # Build a realistic example input using feature schema when available
    example_input: dict = {}
    if deployment.pipeline_path and Path(deployment.pipeline_path).exists():
        schema = get_feature_schema(deployment.pipeline_path)
        for field_info in schema:
            fname = field_info.get("name", "")
            ftype = field_info.get("type", "numeric")
            if ftype == "numeric":
                example_input[fname] = 1.0
            else:
                example_input[fname] = "value"
    else:
        # Fallback: use feature names with generic defaults
        for fname in feature_names:
            example_input[fname] = 1.0

    endpoint_url = f"{base_url}/api/predict/{deployment_id}"
    example_json = json.dumps(example_input, indent=2)
    example_json_compact = json.dumps(example_input)

    # --- curl ---
    curl_snippet = (
        f"curl -X POST \\\n"
        f"  '{endpoint_url}' \\\n"
        f"  -H 'Content-Type: application/json' \\\n"
        f"  -d '{example_json_compact}'"
    )

    # --- Python ---
    python_snippet = (
        f"import requests\n\n"
        f'url = "{endpoint_url}"\n'
        f"data = {example_json}\n\n"
        f"response = requests.post(url, json=data)\n"
        f"result = response.json()\n\n"
        f"print(f\"Prediction: {{result['prediction']}}\")\n"
    )
    if deployment.problem_type == "classification":
        python_snippet += "print(f\"Confidence: {result.get('confidence', 'N/A')}\")\n"
    else:
        python_snippet += (
            "if 'confidence_interval' in result:\n"
            "    ci = result['confidence_interval']\n"
            "    print(f\"95% interval: {ci['lower']:.2f} – {ci['upper']:.2f}\")\n"
        )

    # --- JavaScript ---
    js_snippet = (
        f"const response = await fetch('{endpoint_url}', {{\n"
        f"  method: 'POST',\n"
        f"  headers: {{ 'Content-Type': 'application/json' }},\n"
        f"  body: JSON.stringify({example_json_compact}),\n"
        f"}});\n\n"
        f"const result = await response.json();\n"
        f"console.log('Prediction:', result.prediction);\n"
    )
    if deployment.problem_type == "classification":
        js_snippet += "console.log('Confidence:', result.confidence);\n"
    else:
        js_snippet += (
            "if (result.confidence_interval) {\n"
            "  const { lower, upper } = result.confidence_interval;\n"
            "  console.log(`95% interval: ${lower.toFixed(2)} – ${upper.toFixed(2)}`);\n"
            "}\n"
        )

    return {
        "deployment_id": deployment_id,
        "endpoint_url": endpoint_url,
        "problem_type": deployment.problem_type,
        "target_column": deployment.target_column,
        "algorithm": deployment.algorithm,
        "example_input": example_input,
        "curl": curl_snippet,
        "python": python_snippet,
        "javascript": js_snippet,
        "openapi_url": f"{base_url}/docs",
        "batch_url": f"{base_url}/api/predict/{deployment_id}/batch",
        "batch_note": (
            "For bulk predictions, POST a CSV file to the batch endpoint: "
            f"curl -X POST '{base_url}/api/predict/{deployment_id}/batch' "
            "-F 'file=@your_data.csv'"
        ),
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
        "api_key_enabled": bool(getattr(d, "api_key_enabled", False)),
        "environment": getattr(d, "environment", "staging"),
        "rate_limit_rpm": getattr(d, "rate_limit_rpm", None),
        "monthly_quota": getattr(d, "monthly_quota", None),
    }


def _verify_api_key(deployment: Deployment, authorization: str | None) -> None:
    """Raise 401 if the deployment is key-protected and the key is missing/wrong."""
    if not getattr(deployment, "api_key_enabled", False):
        return  # No protection — open access

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="This prediction endpoint requires an API key. "
            "Include 'Authorization: Bearer <key>' in your request.",
        )

    provided_key = authorization.removeprefix("Bearer ").strip()
    salt = getattr(deployment, "api_key_salt", "") or ""
    expected_hash = hashlib.sha256(f"{salt}:{provided_key}".encode()).hexdigest()
    stored_hash = getattr(deployment, "api_key_hash", "") or ""

    if not secrets.compare_digest(expected_hash, stored_hash):
        raise HTTPException(status_code=401, detail="Invalid API key.")


# ---------------------------------------------------------------------------
# Rate-limit helpers (in-memory sliding window)
# ---------------------------------------------------------------------------

_rate_windows: dict[str, deque] = {}  # deployment_id → deque of timestamps (monotonic)
_rate_windows_lock = threading.Lock()


def _check_rate_limit(deployment_id: str, rpm: int) -> bool:
    """Sliding-window rate limit check.  Returns True when the request is allowed.

    Maintains a per-deployment deque of request timestamps.  On each call we
    evict entries older than 60 seconds and compare the remaining count to rpm.
    Thread-safe via a module-level lock.
    """
    now = time.monotonic()
    cutoff = now - 60.0
    with _rate_windows_lock:
        if deployment_id not in _rate_windows:
            _rate_windows[deployment_id] = deque()
        window = _rate_windows[deployment_id]
        # Evict stale entries
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= rpm:
            return False  # quota exceeded
        window.append(now)
        return True


def _check_monthly_quota(
    deployment: Deployment, session: Session
) -> tuple[bool, int, int]:
    """Check whether the rolling-30-day prediction count is under the monthly quota.

    Returns (allowed, used_this_month, quota).  When quota is None/0 always allows.
    """
    quota = getattr(deployment, "monthly_quota", None)
    if not quota:
        return True, 0, 0
    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=30)
    from sqlmodel import func as _func

    used = session.exec(
        select(_func.count(PredictionLog.id)).where(
            PredictionLog.deployment_id == deployment.id,
            PredictionLog.created_at >= cutoff,
        )
    ).one()
    return used < quota, used, quota


def _check_and_fire_quota_alert(
    deployment_id: str,
    used: int,
    quota: int,
    threshold_pct: int,
) -> None:
    """Fire quota_alert webhooks exactly once when usage crosses the threshold.

    The alert fires only when the current prediction was the one that crossed
    the threshold — i.e., used >= ceil(quota * threshold_pct / 100) but
    (used - 1) was still below it. This prevents repeated alerts on every
    subsequent prediction after the threshold is hit.

    Dispatched in a daemon thread — never blocks the prediction response.
    """
    import math

    crossing_count = math.ceil(quota * threshold_pct / 100)
    if used != crossing_count:
        return  # either already past threshold or not there yet

    try:
        from core.webhook import EVENT_QUOTA_ALERT, dispatch_webhooks

        pct_used = round(used / quota * 100, 1)
        remaining = quota - used
        payload = {
            "deployment_id": deployment_id,
            "used_this_month": used,
            "monthly_quota": quota,
            "pct_used": pct_used,
            "remaining": remaining,
            "threshold_pct": threshold_pct,
            "message": (
                f"Quota alert: {used}/{quota} predictions used "
                f"({pct_used}% — {remaining} remaining). "
                f"Your configured alert threshold of {threshold_pct}% has been reached."
            ),
        }
        dispatch_webhooks(deployment_id, EVENT_QUOTA_ALERT, payload)
    except Exception:
        pass  # Best-effort; never crash the prediction path


# ---------------------------------------------------------------------------
# Batch schedule endpoints
# ---------------------------------------------------------------------------

from core.scheduler import compute_next_run  # noqa: E402
from models.batch_schedule import BatchJobRun, BatchSchedule  # noqa: E402


def _schedule_response(s: BatchSchedule) -> dict:
    return {
        "id": s.id,
        "deployment_id": s.deployment_id,
        "frequency": s.frequency,
        "run_hour": s.run_hour,
        "run_minute": s.run_minute,
        "day_of_week": s.day_of_week,
        "day_of_month": s.day_of_month,
        "is_active": s.is_active,
        "last_run": s.last_run.isoformat() if s.last_run else None,
        "next_run": s.next_run.isoformat() if s.next_run else None,
        "last_output_path": s.last_output_path,
        "last_row_count": s.last_row_count,
        "last_error": s.last_error,
        "created_at": s.created_at.isoformat() if s.created_at else None,
    }


def _job_run_response(r: BatchJobRun) -> dict:
    return {
        "id": r.id,
        "schedule_id": r.schedule_id,
        "deployment_id": r.deployment_id,
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "completed_at": r.completed_at.isoformat() if r.completed_at else None,
        "status": r.status,
        "row_count": r.row_count,
        "error": r.error,
        "download_url": (
            f"/api/deploy/batch-outputs/{Path(r.output_path).name}"
            if r.output_path and r.status == "success"
            else None
        ),
    }


class ScheduleCreate(BaseModel):
    frequency: str = "daily"  # daily | weekly | monthly
    run_hour: int = 9
    run_minute: int = 0
    day_of_week: int | None = None
    day_of_month: int | None = None


@router.post("/api/deploy/{deployment_id}/schedules", status_code=201)
def create_schedule(
    deployment_id: str,
    body: ScheduleCreate,
    session: Session = Depends(get_session),
):
    """Create a recurring batch prediction schedule."""
    dep = session.get(Deployment, deployment_id)
    if not dep or not dep.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found")

    if body.frequency not in ("daily", "weekly", "monthly"):
        raise HTTPException(
            status_code=400, detail="frequency must be daily, weekly, or monthly"
        )
    if not (0 <= body.run_hour <= 23):
        raise HTTPException(status_code=400, detail="run_hour must be 0-23")
    if not (0 <= body.run_minute <= 59):
        raise HTTPException(status_code=400, detail="run_minute must be 0-59")

    schedule = BatchSchedule(
        deployment_id=deployment_id,
        frequency=body.frequency,
        run_hour=body.run_hour,
        run_minute=body.run_minute,
        day_of_week=body.day_of_week,
        day_of_month=body.day_of_month,
    )
    schedule.next_run = compute_next_run(
        body.frequency,
        body.run_hour,
        body.run_minute,
        body.day_of_week,
        body.day_of_month,
    )
    session.add(schedule)
    session.commit()
    session.refresh(schedule)
    return _schedule_response(schedule)


@router.get("/api/deploy/{deployment_id}/schedules")
def list_schedules(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """List all schedules for a deployment."""
    dep = session.get(Deployment, deployment_id)
    if not dep:
        raise HTTPException(status_code=404, detail="Deployment not found")

    schedules = session.exec(
        select(BatchSchedule).where(BatchSchedule.deployment_id == deployment_id)
    ).all()
    return [_schedule_response(s) for s in schedules]


@router.delete("/api/deploy/{deployment_id}/schedules/{schedule_id}", status_code=204)
def delete_schedule(
    deployment_id: str,
    schedule_id: str,
    session: Session = Depends(get_session),
):
    """Delete (deactivate) a schedule."""
    schedule = session.get(BatchSchedule, schedule_id)
    if not schedule or schedule.deployment_id != deployment_id:
        raise HTTPException(status_code=404, detail="Schedule not found")

    schedule.is_active = False
    session.add(schedule)
    session.commit()


@router.post("/api/deploy/{deployment_id}/schedules/{schedule_id}/run")
def trigger_schedule_run(
    deployment_id: str,
    schedule_id: str,
    session: Session = Depends(get_session),
):
    """Trigger an immediate run of a schedule (outside its normal schedule)."""
    schedule = session.get(BatchSchedule, schedule_id)
    if not schedule or schedule.deployment_id != deployment_id:
        raise HTTPException(status_code=404, detail="Schedule not found")

    # Run in a background thread so the request returns immediately
    import threading
    from core.scheduler import _run_job

    thread = threading.Thread(target=_run_job, args=(schedule_id,), daemon=True)
    thread.start()
    return {"status": "running", "schedule_id": schedule_id}


@router.get("/api/deploy/{deployment_id}/schedules/{schedule_id}/runs")
def list_schedule_runs(
    deployment_id: str,
    schedule_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_session),
):
    """Return recent job runs for a schedule."""
    schedule = session.get(BatchSchedule, schedule_id)
    if not schedule or schedule.deployment_id != deployment_id:
        raise HTTPException(status_code=404, detail="Schedule not found")

    runs = session.exec(
        select(BatchJobRun)
        .where(BatchJobRun.schedule_id == schedule_id)
        .order_by(BatchJobRun.started_at.desc())  # type: ignore[union-attr]
        .limit(limit)
    ).all()
    return [_job_run_response(r) for r in runs]


@router.get("/api/deploy/batch-outputs/{filename}")
def download_batch_output(filename: str):
    """Download a completed batch prediction CSV."""
    from fastapi.responses import FileResponse

    # Security: filename must be alphanumeric + underscores/dashes + .csv
    import re

    if not re.match(r"^[\w\-]+\.csv$", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")

    from core.scheduler import BATCH_OUTPUT_DIR

    path = BATCH_OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        path=str(path),
        media_type="text/csv",
        filename=filename,
    )


# ---------------------------------------------------------------------------
# Deployment versioning endpoints
# ---------------------------------------------------------------------------


def _version_response(v: DeploymentVersion) -> dict:
    return {
        "id": v.id,
        "deployment_id": v.deployment_id,
        "version_number": v.version_number,
        "model_run_id": v.model_run_id,
        "algorithm": v.algorithm,
        "problem_type": v.problem_type,
        "target_column": v.target_column,
        "metrics": json.loads(v.metrics) if v.metrics else {},
        "pipeline_path": v.pipeline_path,
        "deployed_at": v.deployed_at.isoformat() if v.deployed_at else None,
        "is_current": v.is_current,
    }


@router.get("/api/deploy/{deployment_id}/versions")
def list_deployment_versions(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """Return all versions of a deployment, ordered newest-first.

    Each entry represents a point-in-time snapshot of what algorithm/model was
    actively serving predictions. Useful for auditing retraining history.
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    versions = session.exec(
        select(DeploymentVersion)
        .where(DeploymentVersion.deployment_id == deployment_id)
        .order_by(DeploymentVersion.version_number.desc())
    ).all()

    return {
        "deployment_id": deployment_id,
        "current_version_number": getattr(deployment, "current_version_number", 1),
        "versions": [_version_response(v) for v in versions],
    }


@router.post("/api/deploy/{deployment_id}/rollback/{version_number}", status_code=200)
def rollback_deployment(
    deployment_id: str,
    version_number: int,
    session: Session = Depends(get_session),
):
    """Restore a deployment to a previous version.

    Finds the specified version snapshot, restores the Deployment record's
    model/pipeline data from that snapshot, archives the previous state, and
    saves a new version entry (so the history is always append-only).
    Returns the updated deployment response.
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    # Find the target version
    target = session.exec(
        select(DeploymentVersion).where(
            DeploymentVersion.deployment_id == deployment_id,
            DeploymentVersion.version_number == version_number,
        )
    ).first()
    if not target:
        raise HTTPException(
            status_code=404,
            detail=f"Version {version_number} not found for this deployment",
        )

    # Validate the pipeline file still exists on disk
    if not target.pipeline_path or not Path(target.pipeline_path).exists():
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline file for version {version_number} is no longer available on disk",
        )

    # Validate the model run still has its model file
    target_run = session.get(ModelRun, target.model_run_id)
    if (
        not target_run
        or not target_run.model_path
        or not Path(target_run.model_path).exists()
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Model file for version {version_number} is no longer available on disk",
        )

    # Archive current state (mark all current versions as not current)
    current_versions = session.exec(
        select(DeploymentVersion).where(
            DeploymentVersion.deployment_id == deployment_id,
            DeploymentVersion.is_current == True,  # noqa: E712
        )
    ).all()
    for v in current_versions:
        v.is_current = False
        session.add(v)

    new_version_number = getattr(deployment, "current_version_number", 1) + 1

    # Restore deployment from the target version snapshot
    deployment.model_run_id = target.model_run_id
    deployment.pipeline_path = target.pipeline_path
    deployment.algorithm = target.algorithm
    deployment.problem_type = target.problem_type
    deployment.target_column = target.target_column
    deployment.metrics = target.metrics
    deployment.current_version_number = new_version_number
    session.add(deployment)

    # Save new version entry for the rollback (append-only history)
    rollback_version = DeploymentVersion(
        deployment_id=deployment_id,
        version_number=new_version_number,
        model_run_id=target.model_run_id,
        algorithm=target.algorithm,
        problem_type=target.problem_type,
        target_column=target.target_column,
        metrics=target.metrics,
        pipeline_path=target.pipeline_path,
        is_current=True,
    )
    session.add(rollback_version)

    session.commit()
    session.refresh(deployment)

    return {
        **_deployment_response(deployment),
        "rolled_back_to_version": version_number,
        "new_version_number": new_version_number,
    }


# ---------------------------------------------------------------------------
# Export deployment as self-contained prediction service (ZIP download)
# ---------------------------------------------------------------------------

_SERVER_PY_TEMPLATE = '''\
"""Self-contained AutoModeler prediction service.

Generated by AutoModeler. Runs a FastAPI server that serves predictions
using the exported model pipeline.

Usage:
    pip install -r requirements.txt
    uvicorn server:app --host 0.0.0.0 --port 8000

Then call predictions:
    curl -X POST http://localhost:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{EXAMPLE_PAYLOAD}'
"""
from __future__ import annotations

import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="{model_title} Prediction Service",
    description="Auto-generated by AutoModeler. Predicts: {target_column}",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and pipeline once at startup
_pipeline = joblib.load("model_pipeline.joblib")
_model = joblib.load("model.joblib")


class PredictRequest(BaseModel):
    features: dict


class PredictResponse(BaseModel):
    prediction: object
    target_column: str
    problem_type: str
    feature_names: list


@app.get("/")
def root():
    return {{
        "service": "{model_title} Prediction Service",
        "target_column": _pipeline.target_column,
        "problem_type": _pipeline.problem_type,
        "feature_names": _pipeline.feature_names,
        "docs": "/docs",
    }}


@app.get("/health")
def health():
    return {{"status": "ok"}}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    """Make a single prediction.

    Pass a JSON dict of feature names → values.
    Missing features will be filled with training-set medians (numeric)
    or treated as unseen categories (categorical).
    """
    X = _pipeline.transform(body.features)
    raw = _model.predict(X)[0]
    decoded = _pipeline.decode_prediction(raw)

    result: dict = {{
        "prediction": decoded,
        "target_column": _pipeline.target_column,
        "problem_type": _pipeline.problem_type,
        "feature_names": _pipeline.feature_names,
    }}

    if _pipeline.problem_type == "classification" and hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(X)[0]
        classes = (
            [str(c) for c in _pipeline.target_encoder.classes_]
            if _pipeline.target_encoder is not None
            else [str(i) for i in range(len(proba))]
        )
        result["probabilities"] = {{cls: round(float(p), 4) for cls, p in zip(classes, proba)}}
        result["confidence"] = round(float(proba.max()), 4)

    if _pipeline.problem_type == "regression":
        residual_std = getattr(_pipeline, "residual_std", 0.0)
        if residual_std > 0:
            pred_value = float(decoded)
            result["confidence_interval"] = {{
                "lower": round(pred_value - 1.96 * residual_std, 4),
                "upper": round(pred_value + 1.96 * residual_std, 4),
                "level": 0.95,
            }}

    return result
'''

_REQUIREMENTS_TXT = """\
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
pydantic>=2.0.0
scikit-learn>=1.4.0
numpy>=1.26.0
pandas>=2.0.0
joblib>=1.3.0
"""

_README_TEMPLATE = """\
# {model_title} — Self-Contained Prediction Service

Exported from AutoModeler on {export_date}.

## What this is

A standalone FastAPI service that serves predictions from your trained model.
No AutoModeler installation required — just Python and the packages below.

**Target:** `{target_column}`
**Algorithm:** `{algorithm}`
**Problem type:** `{problem_type}`
**Features:** `{feature_list}`

## Quick start

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Start the server
uvicorn server:app --host 0.0.0.0 --port 8000

# 3. Make a prediction
curl -X POST http://localhost:8000/predict \\
     -H "Content-Type: application/json" \\
     -d '{{"features": {example_payload}}}'
```

## API reference

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Service info + feature list |
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/docs` | GET | Interactive Swagger UI |

### Prediction request

```json
{{
  "features": {{
{feature_schema_comment}
  }}
}}
```

### Prediction response

```json
{{
  "prediction": <value>,
  "target_column": "{target_column}",
  "problem_type": "{problem_type}",
  "feature_names": {feature_list_json}
}}
```

## Files

| File | Description |
|------|-------------|
| `server.py` | FastAPI application (the server code) |
| `model_pipeline.joblib` | Preprocessing pipeline (encoders, medians, feature config) |
| `model.joblib` | Trained {algorithm} model |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |
"""


@router.get("/api/deploy/{deployment_id}/export")
def export_deployment(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """Export a deployed model as a self-contained prediction service ZIP.

    The ZIP contains:
    - server.py         — minimal FastAPI prediction server
    - model_pipeline.joblib — preprocessing pipeline
    - model.joblib      — trained model
    - requirements.txt  — Python dependencies
    - README.md         — quick-start instructions

    The developer can unzip, install requirements, and run:
        uvicorn server:app --host 0.0.0.0 --port 8000
    """
    import io
    import zipfile
    from datetime import UTC, datetime

    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    run = session.get(ModelRun, deployment.model_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found")

    pipeline_path = deployment.pipeline_path
    model_path = run.model_path if run else None

    if not pipeline_path or not Path(pipeline_path).exists():
        raise HTTPException(status_code=404, detail="Pipeline file not found on disk")
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="Model file not found on disk")

    algorithm = deployment.algorithm or "model"
    target_column = deployment.target_column or "target"
    problem_type = deployment.problem_type or "regression"
    feature_names: list[str] = (
        json.loads(deployment.feature_names) if deployment.feature_names else []
    )
    export_date = datetime.now(UTC).strftime("%Y-%m-%d")

    # Build example payload (feature medians / first category value)
    try:
        from core.deployer import load_pipeline as _load_pipeline

        _pl = _load_pipeline(pipeline_path)
        example: dict = {}
        for fname in feature_names:
            if _pl.column_types.get(fname) == "numeric":
                example[fname] = round(_pl.medians.get(fname, 0.0), 4)
            else:
                le = _pl.label_encoders.get(fname)
                example[fname] = (
                    str(le.classes_[0])
                    if (le is not None and len(le.classes_))
                    else "value"
                )
    except Exception:
        example = {f: 0 for f in feature_names}

    example_json = json.dumps(example)
    feature_list_str = ", ".join(feature_names) if feature_names else "—"
    feature_list_json = json.dumps(feature_names)

    # Build README feature schema comment
    feat_lines = []
    for fname in feature_names:
        sample_val = example.get(fname, 0)
        if isinstance(sample_val, str):
            feat_lines.append(f'    "{fname}": "{sample_val}"  // categorical')
        else:
            feat_lines.append(f'    "{fname}": {sample_val}  // numeric')
    feature_schema_comment = ",\n".join(feat_lines) or '    "feature": value'

    # Render server.py (title uses algorithm + target)
    model_title = f"{algorithm.replace('_', ' ').title()} — {target_column}"
    server_py = _SERVER_PY_TEMPLATE.format(
        model_title=model_title,
        target_column=target_column,
        EXAMPLE_PAYLOAD=example_json,
    )
    readme_md = _README_TEMPLATE.format(
        model_title=model_title,
        export_date=export_date,
        target_column=target_column,
        algorithm=algorithm,
        problem_type=problem_type,
        feature_list=feature_list_str,
        feature_list_json=feature_list_json,
        example_payload=example_json,
        feature_schema_comment=feature_schema_comment,
    )

    # Build in-memory ZIP
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("server.py", server_py)
        zf.writestr("requirements.txt", _REQUIREMENTS_TXT)
        zf.writestr("README.md", readme_md)
        zf.write(pipeline_path, "model_pipeline.joblib")
        zf.write(model_path, "model.joblib")
    buf.seek(0)

    safe_name = f"automodeler_{target_column}_{algorithm}".replace(" ", "_")[:60]
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.zip"',
        },
    )


# ---------------------------------------------------------------------------
# SDK generation — downloadable Python / JavaScript client library
# ---------------------------------------------------------------------------


def _sdk_class_name(target_column: str, algorithm: str) -> str:
    """Derive a PascalCase class name from target column + algorithm."""
    parts = (target_column + "_predictor").replace("-", "_").split("_")
    return "".join(p.capitalize() for p in parts if p)


def _generate_python_sdk(
    deployment_id: str,
    target_column: str,
    algorithm: str,
    problem_type: str,
    feature_schema: list[dict],
    base_url: str,
    export_date: str,
) -> str:
    """Generate a standalone Python SDK module for a deployed model.

    The module contains a single class with predict() and predict_batch()
    methods, typed signatures, and embedded docstrings.
    """
    class_name = _sdk_class_name(target_column, algorithm)

    # Build typed parameter signatures and docstrings
    param_lines = []
    arg_doc_lines = []
    payload_lines = []
    for field in feature_schema:
        fname = field.get("name", "feature")
        ftype = field.get("type", "numeric")
        py_type = "float" if ftype == "numeric" else "str"
        default = "0.0" if ftype == "numeric" else '"value"'
        param_lines.append(f"        {fname}: {py_type} = {default},")
        arg_doc_lines.append(
            f"            {fname}: {'Numeric value' if ftype == 'numeric' else 'Category string'}"
        )
        payload_lines.append(f'            "{fname}": {fname},')

    params_block = "\n".join(param_lines) if param_lines else "        **features,"
    args_block = (
        "\n".join(arg_doc_lines)
        if arg_doc_lines
        else "            **features: Feature values"
    )
    payload_block = (
        "\n".join(payload_lines) if payload_lines else "            **features,"
    )

    if problem_type == "classification":
        returns_block = (
            "- prediction: predicted class label\n"
            "                - confidence: model confidence (0–1)\n"
            "                - probabilities: per-class probabilities dict"
        )
    else:
        returns_block = (
            "- prediction: numeric predicted value\n"
            "                - confidence_interval: dict with 'lower' and 'upper' (95% CI)"
        )

    algorithm_plain = algorithm.replace("_", " ").title()

    sdk = f'''\
"""
AutoModeler SDK — {target_column} predictor
Generated: {export_date}
Algorithm: {algorithm_plain}
Problem type: {problem_type}

Usage:
    from {class_name.lower()}_sdk import {class_name}

    predictor = {class_name}(base_url="http://localhost:8000")
    result = predictor.predict(
{_indent_lines(params_block, 8)}
    )
    print(result["prediction"])
"""

from __future__ import annotations

from typing import Optional

import requests


class {class_name}:
    """Client for the AutoModeler {target_column} prediction model.

    Algorithm : {algorithm_plain}
    Target    : {target_column}
    Problem   : {problem_type}
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
    ) -> None:
        self._endpoint = f"{{base_url}}/api/predict/{deployment_id}"
        self._api_key = api_key
        self._session = requests.Session()
        self._session.headers.update({{"Content-Type": "application/json"}})
        if api_key:
            self._session.headers.update({{"Authorization": f"Bearer {{api_key}}"}})

    def predict(
        self,
{params_block}
    ) -> dict:
        """Run a single prediction.

        Args:
{args_block}

        Returns:
            dict with keys:
                {returns_block}

        Raises:
            requests.HTTPError: if the prediction endpoint returns an error.
        """
        payload = {{
{payload_block}
        }}
        response = self._session.post(self._endpoint, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def predict_batch(self, rows: list[dict]) -> list[dict]:
        """Run predictions for a list of feature dicts.

        Args:
            rows: list of dicts, each mapping feature name to value.

        Returns:
            list of prediction result dicts in the same order.
        """
        return [self.predict(**row) for row in rows]
'''
    return sdk


def _indent_lines(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.splitlines())


def _generate_javascript_sdk(
    deployment_id: str,
    target_column: str,
    algorithm: str,
    problem_type: str,
    feature_schema: list[dict],
    base_url: str,
    export_date: str,
) -> str:
    """Generate a standalone JavaScript/TypeScript-compatible SDK module."""
    class_name = _sdk_class_name(target_column, algorithm)
    algorithm_plain = algorithm.replace("_", " ").title()

    # Build JSDoc @param lines
    param_doc_lines = []
    param_names = []
    for field in feature_schema:
        fname = field.get("name", "feature")
        ftype = field.get("type", "numeric")
        js_type = "number" if ftype == "numeric" else "string"
        param_doc_lines.append(f" * @param {{{js_type}}} features.{fname}")
        param_names.append(fname)

    params_doc = (
        "\n".join(param_doc_lines) or " * @param {Object} features - Feature values"
    )
    feature_list_comment = ", ".join(param_names) or "feature"

    if problem_type == "classification":
        return_doc = "{{prediction: string, confidence: number, probabilities: Object}}"
    else:
        return_doc = "{{prediction: number, confidence_interval: {{lower: number, upper: number}}}}"

    sdk = f"""\
/**
 * AutoModeler SDK — {target_column} predictor
 * Generated: {export_date}
 * Algorithm: {algorithm_plain}
 * Problem type: {problem_type}
 *
 * @example
 * import {{ {class_name} }} from './{class_name.lower()}_sdk.js';
 *
 * const predictor = new {class_name}({{ baseUrl: 'http://localhost:8000' }});
 * const result = await predictor.predict({{ {feature_list_comment}: ... }});
 * console.log(result.prediction);
 */
export class {class_name} {{
  /**
   * @param {{Object}} [options]
   * @param {{string}} [options.baseUrl='http://localhost:8000'] - API server base URL
   * @param {{string}} [options.apiKey] - Optional API key for authenticated deployments
   */
  constructor(options = {{}}) {{
    this._endpoint = (options.baseUrl || '{base_url}') + '/api/predict/{deployment_id}';
    this._apiKey = options.apiKey || null;
  }}

  /**
   * Run a single prediction.
   *
{params_doc}
   * @returns {{Promise<{return_doc}>}}
   */
  async predict(features = {{}}) {{
    const headers = {{ 'Content-Type': 'application/json' }};
    if (this._apiKey) {{
      headers['Authorization'] = `Bearer ${{this._apiKey}}`;
    }}

    const response = await fetch(this._endpoint, {{
      method: 'POST',
      headers,
      body: JSON.stringify(features),
    }});

    if (!response.ok) {{
      const text = await response.text().catch(() => response.statusText);
      throw new Error(`Prediction failed (${{response.status}}): ${{text}}`);
    }}

    return response.json();
  }}

  /**
   * Run predictions for multiple rows.
   *
   * @param {{Object[]}} rows - Array of feature dicts
   * @returns {{Promise<Object[]>}}
   */
  async predictBatch(rows) {{
    return Promise.all(rows.map(row => this.predict(row)));
  }}
}}
"""
    return sdk


@router.get("/api/deploy/{deployment_id}/sdk")
def get_sdk(
    deployment_id: str,
    language: str = Query("python", description="SDK language: python or javascript"),
    base_url: str = Query(
        "http://localhost:8000", description="Base URL of the prediction API server"
    ),
    session: Session = Depends(get_session),
):
    """Generate and download a client SDK for a deployed model.

    Supports Python (requests-based class) and JavaScript (ES module class).
    The generated SDK wraps the prediction API into a typed, documented class
    so developers can integrate it without manually constructing HTTP requests.

    Returns a downloadable source file (.py or .js).
    """
    from datetime import UTC, datetime

    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    if language not in ("python", "javascript"):
        raise HTTPException(
            status_code=400,
            detail="Invalid language. Supported values: python, javascript",
        )

    target_column = deployment.target_column or "target"
    algorithm = deployment.algorithm or "model"
    problem_type = deployment.problem_type or "regression"
    export_date = datetime.now(UTC).strftime("%Y-%m-%d")

    # Load feature schema from pipeline (type info: numeric vs categorical)
    feature_schema: list[dict] = []
    if deployment.pipeline_path and Path(deployment.pipeline_path).exists():
        try:
            feature_schema = get_feature_schema(deployment.pipeline_path)
        except Exception:
            feature_names = (
                json.loads(deployment.feature_names) if deployment.feature_names else []
            )
            feature_schema = [{"name": f, "type": "numeric"} for f in feature_names]
    else:
        feature_names = (
            json.loads(deployment.feature_names) if deployment.feature_names else []
        )
        feature_schema = [{"name": f, "type": "numeric"} for f in feature_names]

    class_name = _sdk_class_name(target_column, algorithm)

    if language == "python":
        content = _generate_python_sdk(
            deployment_id=deployment_id,
            target_column=target_column,
            algorithm=algorithm,
            problem_type=problem_type,
            feature_schema=feature_schema,
            base_url=base_url,
            export_date=export_date,
        )
        filename = f"{class_name.lower()}_sdk.py"
        media_type = "text/x-python"
    else:
        content = _generate_javascript_sdk(
            deployment_id=deployment_id,
            target_column=target_column,
            algorithm=algorithm,
            problem_type=problem_type,
            feature_schema=feature_schema,
            base_url=base_url,
            export_date=export_date,
        )
        filename = f"{class_name.lower()}_sdk.js"
        media_type = "text/javascript"

    return Response(
        content=content.encode("utf-8"),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Webhook notification endpoints
# ---------------------------------------------------------------------------


class WebhookCreateBody(BaseModel):
    url: str
    event_types: list[str] = ["batch_complete", "drift_detected", "health_degraded"]


def _webhook_response(wh: WebhookConfig, include_secret: bool = False) -> dict:
    data = {
        "id": wh.id,
        "deployment_id": wh.deployment_id,
        "url": wh.url,
        "event_types": json.loads(wh.event_types or "[]"),
        "is_active": wh.is_active,
        "created_at": wh.created_at.isoformat() if wh.created_at else None,
        "last_fired_at": wh.last_fired_at.isoformat() if wh.last_fired_at else None,
        "last_status_code": wh.last_status_code,
    }
    if include_secret:
        data["secret"] = wh.secret
    return data


@router.post("/api/deploy/{deployment_id}/webhooks", status_code=201)
def create_webhook(
    deployment_id: str,
    body: WebhookCreateBody,
    session: Session = Depends(get_session),
):
    """Register a webhook URL for a deployment.

    The secret is returned **once** in the response. Store it securely — it is
    used to verify the ``X-AutoModeler-Signature`` header on every dispatch.

    Supported event_types:
    - "batch_complete"  — a scheduled batch job finished (success or failure)
    - "drift_detected"  — prediction drift score >= 50
    - "health_degraded" — model health score < 60
    """
    from core.webhook import ALL_EVENTS

    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    if not body.url.startswith(("http://", "https://")):
        raise HTTPException(
            status_code=400, detail="URL must start with http:// or https://"
        )

    unknown = set(body.event_types) - ALL_EVENTS
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown event types: {sorted(unknown)}. Valid: {sorted(ALL_EVENTS)}",
        )

    wh = WebhookConfig(
        deployment_id=deployment_id,
        url=body.url,
        event_types=json.dumps(body.event_types),
    )
    session.add(wh)
    session.commit()
    session.refresh(wh)
    return _webhook_response(wh, include_secret=True)


@router.get("/api/deploy/{deployment_id}/webhooks")
def list_webhooks(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """List all registered webhooks for a deployment (secrets excluded)."""
    deployment = session.get(Deployment, deployment_id)
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")

    hooks = session.exec(
        select(WebhookConfig).where(
            WebhookConfig.deployment_id == deployment_id,
            WebhookConfig.is_active == True,  # noqa: E712
        )
    ).all()
    return [_webhook_response(h) for h in hooks]


@router.delete("/api/deploy/{deployment_id}/webhooks/{webhook_id}", status_code=204)
def delete_webhook(
    deployment_id: str,
    webhook_id: str,
    session: Session = Depends(get_session),
):
    """Remove (soft-delete) a webhook registration."""
    wh = session.get(WebhookConfig, webhook_id)
    if not wh or wh.deployment_id != deployment_id:
        raise HTTPException(status_code=404, detail="Webhook not found")

    wh.is_active = False
    session.add(wh)
    session.commit()
    return None


@router.post("/api/deploy/{deployment_id}/webhooks/{webhook_id}/test")
def test_webhook(
    deployment_id: str,
    webhook_id: str,
    session: Session = Depends(get_session),
):
    """Send a test payload to the webhook URL immediately.

    Returns the HTTP status code received from the target server (or 0 on network error).
    Useful for verifying the URL and signature verification are working.
    """
    wh = session.get(WebhookConfig, webhook_id)
    if not wh or wh.deployment_id != deployment_id:
        raise HTTPException(status_code=404, detail="Webhook not found")

    from core.webhook import _do_dispatch

    test_payload = {
        "deployment_id": deployment_id,
        "event_type": "test",
        "fired_at": datetime.now(UTC).isoformat(),
        "message": "AutoModeler webhook test — if you received this, your webhook is working correctly.",
    }
    status_code = _do_dispatch(wh.id, wh.url, wh.secret, test_payload)

    # Update stats
    wh.last_fired_at = datetime.now(UTC).replace(tzinfo=None)
    wh.last_status_code = status_code
    session.add(wh)
    session.commit()

    return {
        "webhook_id": webhook_id,
        "url": wh.url,
        "status_code": status_code,
        "success": 200 <= status_code < 300,
    }


# ---------------------------------------------------------------------------
# Webhook event history
# ---------------------------------------------------------------------------


@router.get("/api/deploy/{deployment_id}/webhook-history")
def webhook_event_history(
    deployment_id: str,
    limit: int = 20,
    session: Session = Depends(get_session),
):
    """Return the most recent webhook dispatch events for this deployment.

    Events are ordered most-recent-first.  ``status_code`` of 0 means the
    HTTP request itself failed (network error / timeout); None means the event
    was created but status not yet recorded (shouldn't happen in practice).
    """
    dep = session.get(Deployment, deployment_id)
    if not dep:
        raise HTTPException(status_code=404, detail="Deployment not found")

    from models.webhook_event import WebhookEvent

    limit = max(1, min(limit, 100))
    events = session.exec(
        select(WebhookEvent)
        .where(WebhookEvent.deployment_id == deployment_id)
        .order_by(WebhookEvent.fired_at.desc())
        .limit(limit)
    ).all()

    # Pull webhook URLs once so we can surface them per-event
    hooks = session.exec(
        select(WebhookConfig).where(WebhookConfig.deployment_id == deployment_id)
    ).all()
    url_by_id = {h.id: h.url for h in hooks}

    event_list = [
        {
            "id": e.id,
            "webhook_id": e.webhook_id,
            "webhook_url": url_by_id.get(e.webhook_id, "(deleted)"),
            "event_type": e.event_type,
            "fired_at": e.fired_at.isoformat() if e.fired_at else None,
            "status_code": e.status_code,
            "success": (200 <= (e.status_code or 0) < 300) if e.status_code else False,
        }
        for e in events
    ]

    # Plain-English summary
    total = len(event_list)
    if total == 0:
        summary = "No webhook events recorded yet for this deployment."
    else:
        successes = sum(1 for e in event_list if e["success"])
        types_seen = list(dict.fromkeys(e["event_type"] for e in event_list))
        type_str = ", ".join(types_seen[:3])
        summary = (
            f"{total} recent event{'s' if total != 1 else ''} "
            f"({successes} successful). "
            f"Event types: {type_str}."
        )

    return {
        "deployment_id": deployment_id,
        "total": total,
        "events": event_list,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Champion-challenger A/B testing
# ---------------------------------------------------------------------------


def _percentile_sorted(sorted_vals: list[float], pct: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list (0 < pct < 100)."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    idx = (pct / 100) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def _ab_variant_metrics(champion_id: str, variant: str, session: Session) -> dict:
    """Return aggregate metrics for one A/B variant (all logs keyed to champion endpoint)."""
    logs = session.exec(
        select(PredictionLog).where(
            PredictionLog.deployment_id == champion_id,
            PredictionLog.ab_variant == variant,
        )
    ).all()

    request_count = len(logs)
    confidences = [lg.confidence for lg in logs if lg.confidence is not None]
    avg_confidence = (
        round(sum(confidences) / len(confidences), 4) if confidences else None
    )
    latencies = sorted([lg.response_ms for lg in logs if lg.response_ms is not None])
    p95_ms = round(_percentile_sorted(latencies, 95), 2) if latencies else None
    predictions = [
        lg.prediction_numeric for lg in logs if lg.prediction_numeric is not None
    ]
    avg_prediction = (
        round(sum(predictions) / len(predictions), 4) if predictions else None
    )

    return {
        "request_count": request_count,
        "avg_confidence": avg_confidence,
        "p95_ms": p95_ms,
        "avg_prediction": avg_prediction,
    }


def _ab_significance(champion_id: str, session: Session) -> dict:
    """Mann-Whitney U significance test on champion vs challenger numeric predictions."""
    champ_preds = [
        lg.prediction_numeric
        for lg in session.exec(
            select(PredictionLog).where(
                PredictionLog.deployment_id == champion_id,
                PredictionLog.ab_variant == "champion",
            )
        ).all()
        if lg.prediction_numeric is not None
    ]
    chall_preds = [
        lg.prediction_numeric
        for lg in session.exec(
            select(PredictionLog).where(
                PredictionLog.deployment_id == champion_id,
                PredictionLog.ab_variant == "challenger",
            )
        ).all()
        if lg.prediction_numeric is not None
    ]

    min_samples = 5
    if len(champ_preds) < min_samples or len(chall_preds) < min_samples:
        needed = min_samples - min(len(champ_preds), len(chall_preds))
        return {
            "significant": False,
            "p_value": None,
            "note": f"Need {needed} more samples per variant (minimum {min_samples})",
        }

    try:
        from scipy.stats import mannwhitneyu

        _, p = mannwhitneyu(champ_preds, chall_preds, alternative="two-sided")
        return {
            "significant": bool(p < 0.05),
            "p_value": round(float(p), 4),
            "note": "Mann-Whitney U test (α=0.05)",
        }
    except Exception:  # noqa: BLE001
        return {
            "significant": False,
            "p_value": None,
            "note": "Statistical test unavailable",
        }


def _ab_test_response(test: "ABTest", session: Session) -> dict:
    champion = session.get(Deployment, test.champion_id)
    challenger = session.get(Deployment, test.challenger_id)
    return {
        "id": test.id,
        "champion_id": test.champion_id,
        "challenger_id": test.challenger_id,
        "champion_algorithm": champion.algorithm if champion else None,
        "challenger_algorithm": challenger.algorithm if challenger else None,
        "champion_split_pct": test.champion_split_pct,
        "challenger_split_pct": 100 - test.champion_split_pct,
        "is_active": test.is_active,
        "auto_promote": test.auto_promote,
        "created_at": test.created_at.isoformat() if test.created_at else None,
        "ended_at": test.ended_at.isoformat() if test.ended_at else None,
        "winner": test.winner,
        "champion_metrics": _ab_variant_metrics(test.champion_id, "champion", session),
        "challenger_metrics": _ab_variant_metrics(
            test.champion_id, "challenger", session
        ),
        "significance": _ab_significance(test.champion_id, session),
    }


class ABTestCreate(BaseModel):
    challenger_id: str
    champion_split_pct: int = 80  # 1-99
    auto_promote: bool = False


@router.post("/api/deploy/{deployment_id}/ab-test", status_code=201)
def create_ab_test(
    deployment_id: str,
    body: ABTestCreate,
    session: Session = Depends(get_session),
):
    """Start a champion-challenger A/B test.

    Routes (100 - champion_split_pct)% of live prediction traffic to the challenger
    so analysts can measure real-world performance differences before committing.
    """
    champion = session.get(Deployment, deployment_id)
    if not champion or not champion.is_active:
        raise HTTPException(
            status_code=404, detail="Champion deployment not found or inactive"
        )

    if body.challenger_id == deployment_id:
        raise HTTPException(
            status_code=400,
            detail="Champion and challenger must be different deployments",
        )

    challenger = session.get(Deployment, body.challenger_id)
    if not challenger or not challenger.is_active:
        raise HTTPException(
            status_code=404, detail="Challenger deployment not found or inactive"
        )

    if not (1 <= body.champion_split_pct <= 99):
        raise HTTPException(
            status_code=400, detail="champion_split_pct must be between 1 and 99"
        )

    # Deactivate any existing A/B test for this champion
    existing = session.exec(
        select(ABTest).where(
            ABTest.champion_id == deployment_id,
            ABTest.is_active == True,  # noqa: E712
        )
    ).first()
    if existing:
        existing.is_active = False
        existing.ended_at = datetime.now(UTC).replace(tzinfo=None)
        session.add(existing)

    ab_test = ABTest(
        champion_id=deployment_id,
        challenger_id=body.challenger_id,
        champion_split_pct=body.champion_split_pct,
        auto_promote=body.auto_promote,
    )
    session.add(ab_test)
    session.commit()
    session.refresh(ab_test)
    return _ab_test_response(ab_test, session)


@router.get("/api/deploy/{deployment_id}/ab-test")
def get_ab_test(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """Return the active A/B test status and per-variant metrics."""
    ab_test = session.exec(
        select(ABTest).where(
            ABTest.champion_id == deployment_id,
            ABTest.is_active == True,  # noqa: E712
        )
    ).first()
    if not ab_test:
        raise HTTPException(
            status_code=404, detail="No active A/B test for this deployment"
        )
    return _ab_test_response(ab_test, session)


@router.delete("/api/deploy/{deployment_id}/ab-test", status_code=204)
def end_ab_test(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """End the current A/B test without promoting the challenger."""
    ab_test = session.exec(
        select(ABTest).where(
            ABTest.champion_id == deployment_id,
            ABTest.is_active == True,  # noqa: E712
        )
    ).first()
    if not ab_test:
        raise HTTPException(
            status_code=404, detail="No active A/B test for this deployment"
        )

    ab_test.is_active = False
    ab_test.ended_at = datetime.now(UTC).replace(tzinfo=None)
    session.add(ab_test)
    session.commit()
    return None


@router.post("/api/deploy/{deployment_id}/ab-test/promote")
def promote_challenger(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """Promote the challenger to champion.

    Copies the challenger's model artifacts into the champion deployment so the
    prediction endpoint URL stays stable (the link the analyst shared with their VP
    keeps working).  Archives the current champion as a versioned snapshot.
    """
    ab_test = session.exec(
        select(ABTest).where(
            ABTest.champion_id == deployment_id,
            ABTest.is_active == True,  # noqa: E712
        )
    ).first()
    if not ab_test:
        raise HTTPException(
            status_code=404, detail="No active A/B test for this deployment"
        )

    champion = session.get(Deployment, deployment_id)
    challenger = session.get(Deployment, ab_test.challenger_id)
    if not champion or not challenger:
        raise HTTPException(status_code=404, detail="Deployment not found")

    if not (challenger.pipeline_path and Path(challenger.pipeline_path).exists()):
        raise HTTPException(
            status_code=400,
            detail="Challenger pipeline file not found — cannot promote",
        )

    # Archive current champion model as a versioned snapshot
    _archive_current_version(champion, session)
    new_version_number = getattr(champion, "current_version_number", 1) + 1

    # Copy challenger's model info into the champion deployment
    champion.model_run_id = challenger.model_run_id
    champion.pipeline_path = challenger.pipeline_path
    champion.algorithm = challenger.algorithm
    champion.problem_type = challenger.problem_type
    champion.feature_names = challenger.feature_names
    champion.target_column = challenger.target_column
    champion.metrics = challenger.metrics
    champion.current_version_number = new_version_number
    session.add(champion)

    # Record promoted version in history
    session.add(
        DeploymentVersion(
            deployment_id=deployment_id,
            version_number=new_version_number,
            model_run_id=challenger.model_run_id,
            algorithm=challenger.algorithm,
            problem_type=challenger.problem_type,
            target_column=challenger.target_column,
            metrics=challenger.metrics,
            pipeline_path=challenger.pipeline_path,
            is_current=True,
        )
    )

    # End the A/B test and record winner
    ab_test.is_active = False
    ab_test.ended_at = datetime.now(UTC).replace(tzinfo=None)
    ab_test.winner = "challenger"
    session.add(ab_test)

    session.commit()
    session.refresh(champion)
    return {
        "message": "Challenger promoted to champion. Prediction endpoint URL is unchanged.",
        "deployment": _deployment_response(champion),
    }


# ---------------------------------------------------------------------------
# 20. Deployment environment promotion (staging → production)
# ---------------------------------------------------------------------------


@router.post("/api/deploy/{deployment_id}/promote-to-production")
def promote_to_production(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """Promote a staging deployment to production.

    - Marks this deployment's environment as "production"
    - Demotes any existing production deployment for the same project back to "staging"
      (so it can still be used as a test endpoint)
    - The staging URL is preserved — analysts can still share/test the staging link
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    if getattr(deployment, "environment", "staging") == "production":
        # Already production — idempotent, return as-is
        return {
            "message": "Deployment is already tagged as production.",
            "deployment": _deployment_response(deployment),
        }

    # Demote any existing production deployment for this project
    existing_production = session.exec(
        select(Deployment).where(
            Deployment.project_id == deployment.project_id,
            Deployment.is_active == True,  # noqa: E712
        )
    ).all()
    for dep in existing_production:
        if (
            dep.id != deployment_id
            and getattr(dep, "environment", "staging") == "production"
        ):
            dep.environment = "staging"
            session.add(dep)

    # Promote this deployment to production
    deployment.environment = "production"
    session.add(deployment)
    session.commit()
    session.refresh(deployment)

    return {
        "message": "Deployment promoted to production. Staging URL is preserved for testing.",
        "deployment": _deployment_response(deployment),
    }


@router.post("/api/deploy/{deployment_id}/demote-to-staging")
def demote_to_staging(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """Demote a production deployment back to staging."""
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    deployment.environment = "staging"
    session.add(deployment)
    session.commit()
    session.refresh(deployment)

    return {
        "message": "Deployment demoted back to staging.",
        "deployment": _deployment_response(deployment),
    }


# ---------------------------------------------------------------------------
# Prediction presets — named scenarios for the VP dashboard
# ---------------------------------------------------------------------------


class PresetBody(BaseModel):
    name: str
    feature_values: dict


@router.get("/api/deploy/{deployment_id}/presets")
def list_presets(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """List all prediction presets for a deployment."""
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    presets = session.exec(
        select(DeploymentPreset)
        .where(DeploymentPreset.deployment_id == deployment_id)
        .order_by(DeploymentPreset.created_at)
    ).all()

    return [
        {
            "id": p.id,
            "deployment_id": p.deployment_id,
            "name": p.name,
            "feature_values": json.loads(p.feature_values),
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in presets
    ]


@router.post("/api/deploy/{deployment_id}/presets", status_code=201)
def create_preset(
    deployment_id: str,
    body: PresetBody,
    session: Session = Depends(get_session),
):
    """Create a named prediction preset for a deployment."""
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Preset name cannot be empty")
    if not body.feature_values:
        raise HTTPException(
            status_code=422, detail="Preset must include at least one feature value"
        )

    preset = DeploymentPreset(
        deployment_id=deployment_id,
        name=name,
        feature_values=json.dumps(body.feature_values),
    )
    session.add(preset)
    session.commit()
    session.refresh(preset)

    return {
        "id": preset.id,
        "deployment_id": preset.deployment_id,
        "name": preset.name,
        "feature_values": body.feature_values,
        "created_at": preset.created_at.isoformat() if preset.created_at else None,
    }


@router.delete("/api/deploy/{deployment_id}/presets/{preset_id}", status_code=204)
def delete_preset(
    deployment_id: str,
    preset_id: str,
    session: Session = Depends(get_session),
):
    """Delete a prediction preset."""
    preset = session.get(DeploymentPreset, preset_id)
    if not preset or preset.deployment_id != deployment_id:
        raise HTTPException(status_code=404, detail="Preset not found")

    session.delete(preset)
    session.commit()
    return None


@router.get("/api/deploy/{deployment_id}/covariate-drift")
def get_covariate_drift(
    deployment_id: str,
    sample_size: int = Query(default=500, ge=10, le=2000),
    session: Session = Depends(get_session),
):
    """Return a covariate drift assessment for a deployment.

    Compares recent production inputs against training feature ranges stored
    in the PredictionPipeline. Returns severity (low/medium/high), per-feature
    alerts, and a plain-English summary.
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    logs = session.exec(
        select(PredictionLog)
        .where(PredictionLog.deployment_id == deployment_id)
        .order_by(PredictionLog.created_at.desc())
        .limit(sample_size)
    ).all()

    all_inputs: list[dict] = []
    for log in logs:
        try:
            parsed = json.loads(log.input_features)
            if isinstance(parsed, dict):
                all_inputs.append(parsed)
        except Exception:  # noqa: BLE001
            pass

    feature_ranges: dict = {}
    if deployment.pipeline_path:
        try:
            from core.deployer import load_pipeline as _load_pipeline

            pipeline = _load_pipeline(deployment.pipeline_path)
            feature_ranges = getattr(pipeline, "feature_ranges", {})
        except Exception:  # noqa: BLE001
            pass

    result = compute_covariate_drift_alert(all_inputs, feature_ranges)
    result["deployment_id"] = deployment_id
    return result


@router.get("/api/deploy/{deployment_id}/usage-pattern")
def get_usage_pattern(
    deployment_id: str,
    sample_size: int = Query(default=1000, ge=10, le=5000),
    session: Session = Depends(get_session),
):
    """Return hour-of-day and day-of-week prediction usage patterns for a deployment."""
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    logs = session.exec(
        select(PredictionLog)
        .where(PredictionLog.deployment_id == deployment_id)
        .order_by(PredictionLog.created_at.desc())
        .limit(sample_size)
    ).all()

    result = compute_usage_pattern(logs)
    result["deployment_id"] = deployment_id
    return result


@router.get("/api/deploy/{deployment_id}/prediction-audit")
def get_prediction_audit(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """Return a comprehensive production monitoring snapshot for a deployment.

    Combines volume (today/7d/30d), confidence distribution, SLA percentiles,
    and quota usage into one call. Useful for chat-triggered "how is my
    deployment doing?" queries that would otherwise require four separate calls.
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    logs = session.exec(
        select(PredictionLog).where(PredictionLog.deployment_id == deployment_id)
    ).all()

    result = compute_prediction_audit(logs, deployment)
    result["deployment_id"] = deployment_id
    return result


@router.get("/api/deploy/{deployment_id}/confidence-trend")
def get_confidence_trend(
    deployment_id: str,
    window: int = Query(default=30, ge=1, le=365),
    session: Session = Depends(get_session),
):
    """Return daily confidence trend for a deployment over the last N days.

    Bins PredictionLog records by calendar day, computes average daily
    confidence, and fits a linear trend to classify direction as improving /
    stable / declining.
    """
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    logs = session.exec(
        select(PredictionLog).where(PredictionLog.deployment_id == deployment_id)
    ).all()

    result = compute_confidence_trend(list(logs), window_days=window)
    result["deployment_id"] = deployment_id
    return result


@router.get("/api/deploy/{deployment_id}/batch-results")
def get_batch_results(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """Return analytics for the most recent successful batch prediction job.

    Reads the batch output CSV from disk and computes prediction distribution
    stats (regression: avg/min/max/histogram; classification: class counts/pcts).
    Returns has_results=False with guidance when no completed jobs exist.
    """
    from models.batch_schedule import BatchJobRun

    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    job_run = session.exec(
        select(BatchJobRun)
        .where(
            BatchJobRun.deployment_id == deployment_id,
            BatchJobRun.status == "success",
        )
        .order_by(BatchJobRun.completed_at.desc())
    ).first()

    if not job_run or not job_run.output_path:
        return {
            "deployment_id": deployment_id,
            "has_results": False,
            "summary": "No completed batch jobs found. Schedule a batch prediction first.",
        }

    output_path = Path(job_run.output_path)
    if not output_path.exists():
        return {
            "deployment_id": deployment_id,
            "has_results": False,
            "summary": "Batch output file not found on disk.",
        }

    csv_bytes = output_path.read_bytes()
    problem_type = deployment.problem_type or "regression"
    target_column = deployment.target_column or "prediction"

    result = compute_batch_job_results(csv_bytes, problem_type, target_column)
    result["deployment_id"] = deployment_id
    result["has_results"] = result.get("has_data", False)
    result["job_run_id"] = job_run.id
    result["completed_at"] = (
        job_run.completed_at.isoformat() if job_run.completed_at else None
    )
    result["row_count"] = job_run.row_count
    result["schedule_id"] = job_run.schedule_id
    return result


# ---------------------------------------------------------------------------
# Prediction alert rule endpoints
# ---------------------------------------------------------------------------

_OP_SYMBOLS = {"lt": "<", "gt": ">", "lte": "≤", "gte": "≥", "eq": "="}


def _alert_rule_description(rule: PredictionAlertRule) -> str:
    """Return a plain-English description of the rule condition."""
    if rule.condition_type == "predicted_class":
        return f"Fires when predicted class = '{rule.condition_class}'"
    op = _OP_SYMBOLS.get(rule.condition_op, rule.condition_op)
    field = (
        "Prediction value"
        if rule.condition_type == "prediction_value"
        else "Confidence"
    )
    suffix = "%" if rule.condition_type == "confidence" else ""
    val = rule.condition_value if rule.condition_value is not None else "?"
    return f"Fires when {field} {op} {val}{suffix}"


def _alert_rule_response(rule: PredictionAlertRule) -> dict:
    return {
        "id": rule.id,
        "deployment_id": rule.deployment_id,
        "name": rule.name,
        "condition_type": rule.condition_type,
        "condition_op": rule.condition_op,
        "condition_value": rule.condition_value,
        "condition_class": rule.condition_class,
        "is_active": rule.is_active,
        "created_at": rule.created_at.isoformat() if rule.created_at else None,
        "last_triggered_at": (
            rule.last_triggered_at.isoformat() if rule.last_triggered_at else None
        ),
        "trigger_count": rule.trigger_count,
        "description": _alert_rule_description(rule),
    }


def _evaluate_alert_rule(
    rule: PredictionAlertRule,
    prediction_numeric: float | None,
    confidence: float | None,
    predicted_class: str | None,
) -> bool:
    """Return True if the rule condition is met for this prediction result."""
    if rule.condition_type == "predicted_class":
        return (
            predicted_class is not None
            and rule.condition_class is not None
            and str(predicted_class).lower() == str(rule.condition_class).lower()
        )

    val = (
        prediction_numeric if rule.condition_type == "prediction_value" else confidence
    )
    if val is None or rule.condition_value is None:
        return False

    op = rule.condition_op
    thresh = rule.condition_value
    if op == "lt":
        return val < thresh
    if op == "gt":
        return val > thresh
    if op == "lte":
        return val <= thresh
    if op == "gte":
        return val >= thresh
    if op == "eq":
        return abs(val - thresh) < 1e-9
    return False


def _fire_alert_rules(
    deployment_id: str,
    prediction_numeric: float | None,
    confidence: float | None,
    predicted_class: str | None,
) -> None:
    """Check all active alert rules and dispatch webhooks for triggered ones.

    Runs in a daemon thread — never blocks or crashes the prediction path.
    """
    try:
        from core.webhook import EVENT_PREDICTION_ALERT, dispatch_webhooks
        from db import engine
        from sqlmodel import Session

        with Session(engine) as session:
            rules = session.exec(
                select(PredictionAlertRule).where(
                    PredictionAlertRule.deployment_id == deployment_id,
                    PredictionAlertRule.is_active == True,  # noqa: E712
                )
            ).all()

            for rule in rules:
                if not _evaluate_alert_rule(
                    rule, prediction_numeric, confidence, predicted_class
                ):
                    continue

                # Update rule stats
                rule.last_triggered_at = datetime.now(UTC).replace(tzinfo=None)
                rule.trigger_count += 1
                session.add(rule)
                session.commit()

                dispatch_webhooks(
                    deployment_id,
                    EVENT_PREDICTION_ALERT,
                    {
                        "alert_rule_id": rule.id,
                        "alert_rule_name": rule.name,
                        "condition_description": _alert_rule_description(rule),
                        "prediction_value": prediction_numeric,
                        "confidence": confidence,
                        "predicted_class": predicted_class,
                    },
                )
    except Exception:
        pass  # Best-effort; never crash the prediction path


class AlertRuleRequest(BaseModel):
    name: str
    condition_type: str  # "prediction_value" | "confidence" | "predicted_class"
    condition_op: str = "lt"  # "lt" | "gt" | "lte" | "gte" | "eq"
    condition_value: float | None = None
    condition_class: str | None = None


@router.post("/api/deploy/{deployment_id}/alert-rules")
def create_alert_rule(
    deployment_id: str,
    req: AlertRuleRequest,
    session: Session = Depends(get_session),
):
    """Create a new prediction alert rule for a deployment."""
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    if not req.name.strip():
        raise HTTPException(status_code=422, detail="Rule name cannot be empty")

    valid_types = {"prediction_value", "confidence", "predicted_class"}
    if req.condition_type not in valid_types:
        raise HTTPException(
            status_code=422,
            detail=f"condition_type must be one of {sorted(valid_types)}",
        )

    valid_ops = {"lt", "gt", "lte", "gte", "eq"}
    if req.condition_type != "predicted_class" and req.condition_op not in valid_ops:
        raise HTTPException(
            status_code=422,
            detail=f"condition_op must be one of {sorted(valid_ops)}",
        )

    if (
        req.condition_type in {"prediction_value", "confidence"}
        and req.condition_value is None
    ):
        raise HTTPException(
            status_code=422,
            detail="condition_value is required for numeric condition types",
        )

    if req.condition_type == "predicted_class" and not req.condition_class:
        raise HTTPException(
            status_code=422,
            detail="condition_class is required for predicted_class condition type",
        )

    rule = PredictionAlertRule(
        deployment_id=deployment_id,
        name=req.name.strip(),
        condition_type=req.condition_type,
        condition_op=req.condition_op,
        condition_value=req.condition_value,
        condition_class=req.condition_class,
    )
    session.add(rule)
    session.commit()
    session.refresh(rule)

    return _alert_rule_response(rule)


@router.get("/api/deploy/{deployment_id}/alert-rules")
def list_alert_rules(
    deployment_id: str,
    session: Session = Depends(get_session),
):
    """List all active prediction alert rules for a deployment."""
    deployment = session.get(Deployment, deployment_id)
    if not deployment or not deployment.is_active:
        raise HTTPException(status_code=404, detail="Deployment not found or inactive")

    rules = session.exec(
        select(PredictionAlertRule).where(
            PredictionAlertRule.deployment_id == deployment_id,
            PredictionAlertRule.is_active == True,  # noqa: E712
        )
    ).all()

    return {
        "deployment_id": deployment_id,
        "count": len(rules),
        "rules": [_alert_rule_response(r) for r in rules],
    }


@router.delete("/api/deploy/{deployment_id}/alert-rules/{rule_id}")
def delete_alert_rule(
    deployment_id: str,
    rule_id: str,
    session: Session = Depends(get_session),
):
    """Soft-delete (deactivate) a prediction alert rule."""
    rule = session.get(PredictionAlertRule, rule_id)
    if not rule or rule.deployment_id != deployment_id:
        raise HTTPException(status_code=404, detail="Alert rule not found")

    rule.is_active = False
    session.add(rule)
    session.commit()

    return {"deleted": True, "rule_id": rule_id}
