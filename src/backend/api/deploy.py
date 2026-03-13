"""Deployment and prediction API endpoints.

Endpoints:
  POST   /api/deploy/{model_run_id}          — Deploy model, create Deployment record
  GET    /api/deployments                    — List all active deployments
  GET    /api/deployments/{deployment_id}    — Get deployment details + feature schema
  DELETE /api/deploy/{deployment_id}         — Undeploy (soft delete)

  POST   /api/predict/{deployment_id}        — Single prediction (JSON in → JSON out)
  POST   /api/predict/{deployment_id}/batch  — Batch prediction (CSV upload → CSV download)
"""

import csv
import io
import json
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from core.deployer import get_feature_schema, predict_batch, predict_single
from db import get_session
from models.deployment import Deployment
from models.model_run import ModelRun
from models.project import Project

deploy_router = APIRouter(prefix="/api/deploy", tags=["deploy"])
predict_router = APIRouter(prefix="/api/predict", tags=["predict"])
deployments_router = APIRouter(prefix="/api/deployments", tags=["deploy"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deployment_to_dict(dep: Deployment, run: ModelRun | None = None) -> dict:
    d = {
        "id": dep.id,
        "model_run_id": dep.model_run_id,
        "project_id": dep.project_id,
        "is_active": dep.is_active,
        "request_count": dep.request_count,
        "created_at": dep.created_at.isoformat(),
        "last_predicted_at": (
            dep.last_predicted_at.isoformat() if dep.last_predicted_at else None
        ),
        "feature_schema": json.loads(dep.feature_schema) if dep.feature_schema else [],
    }
    if run:
        d["algorithm"] = run.algorithm
        d["display_name"] = run.display_name
        d["target_column"] = run.target_column
        d["problem_type"] = run.problem_type
        d["metrics"] = json.loads(run.metrics) if run.metrics else None
    return d


def _get_active_deployment(deployment_id: str, session: Session) -> tuple[Deployment, ModelRun]:
    dep = session.get(Deployment, deployment_id)
    if not dep:
        raise HTTPException(status_code=404, detail="Deployment not found")
    if not dep.is_active:
        raise HTTPException(status_code=410, detail="Deployment has been deactivated")
    run = session.get(ModelRun, dep.model_run_id)
    if not run or not run.model_path:
        raise HTTPException(status_code=500, detail="Model artifact missing")
    return dep, run


# ---------------------------------------------------------------------------
# Deploy lifecycle
# ---------------------------------------------------------------------------

@deploy_router.post("/{model_run_id}", status_code=201)
def deploy_model(model_run_id: str, session: Session = Depends(get_session)):
    """Deploy a selected model run.  Creates a Deployment record."""
    run = session.get(ModelRun, model_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="ModelRun not found")
    if not run.model_path:
        raise HTTPException(status_code=400, detail="Model has no artifact — train it first")

    # Build feature schema from the saved joblib
    try:
        schema = get_feature_schema(run.model_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read model: {exc}") from exc

    # Deactivate any existing deployments for this model run
    existing = session.exec(
        select(Deployment).where(
            Deployment.model_run_id == model_run_id,
            Deployment.is_active == True,  # noqa: E712
        )
    ).all()
    for old in existing:
        old.is_active = False
        session.add(old)

    dep = Deployment(
        model_run_id=model_run_id,
        project_id=run.project_id,
        feature_schema=json.dumps(schema),
    )
    session.add(dep)

    # Mark the model run as deployed
    run.is_deployed = True
    session.add(run)

    session.commit()
    session.refresh(dep)

    return _deployment_to_dict(dep, run)


@deployments_router.get("")
def list_deployments(
    project_id: str | None = None,
    session: Session = Depends(get_session),
):
    """Return all active deployments, optionally filtered by project."""
    stmt = select(Deployment).where(Deployment.is_active == True)  # noqa: E712
    if project_id:
        stmt = stmt.where(Deployment.project_id == project_id)
    deployments = session.exec(stmt.order_by(Deployment.created_at.desc())).all()  # type: ignore[arg-type]

    result = []
    for dep in deployments:
        run = session.get(ModelRun, dep.model_run_id)
        result.append(_deployment_to_dict(dep, run))
    return {"deployments": result}


@deployments_router.get("/{deployment_id}")
def get_deployment(deployment_id: str, session: Session = Depends(get_session)):
    """Return deployment details including feature schema for form generation."""
    dep = session.get(Deployment, deployment_id)
    if not dep:
        raise HTTPException(status_code=404, detail="Deployment not found")
    run = session.get(ModelRun, dep.model_run_id)
    return _deployment_to_dict(dep, run)


@deploy_router.delete("/{deployment_id}", status_code=204)
def undeploy_model(deployment_id: str, session: Session = Depends(get_session)):
    """Soft-delete a deployment (marks is_active=False)."""
    dep = session.get(Deployment, deployment_id)
    if not dep:
        raise HTTPException(status_code=404, detail="Deployment not found")

    dep.is_active = False
    session.add(dep)

    # Mark model run as not deployed if this was the only active deployment
    other_active = session.exec(
        select(Deployment).where(
            Deployment.model_run_id == dep.model_run_id,
            Deployment.id != deployment_id,
            Deployment.is_active == True,  # noqa: E712
        )
    ).first()
    if not other_active:
        run = session.get(ModelRun, dep.model_run_id)
        if run:
            run.is_deployed = False
            session.add(run)

    session.commit()


# ---------------------------------------------------------------------------
# Prediction endpoints
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    inputs: dict  # {column_name: value, ...}


@predict_router.post("/{deployment_id}")
def predict(
    deployment_id: str,
    body: PredictRequest,
    session: Session = Depends(get_session),
):
    """Make a single prediction.

    Returns:
      prediction, probability (if classification), interpretation string,
      target_column, algorithm, problem_type
    """
    dep, run = _get_active_deployment(deployment_id, session)

    try:
        result = predict_single(run.model_path, body.inputs)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc

    # Update usage stats
    dep.request_count = (dep.request_count or 0) + 1
    dep.last_predicted_at = datetime.utcnow()
    session.add(dep)
    session.commit()

    return {
        "deployment_id": deployment_id,
        **result,
    }


@predict_router.post("/{deployment_id}/batch")
async def predict_batch_endpoint(
    deployment_id: str,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    """Batch prediction: upload a CSV, download predictions as CSV.

    The input CSV should have columns matching the model's feature columns.
    Extra columns are preserved; missing feature columns are treated as NaN.

    Returns a CSV with original columns + predicted_{target} + confidence (if classification).
    """
    dep, run = _get_active_deployment(deployment_id, session)

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}") from exc

    if len(df) == 0:
        raise HTTPException(status_code=422, detail="CSV file is empty")

    try:
        result = predict_batch(run.model_path, df)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Batch prediction failed: {exc}") from exc

    # Update usage stats
    dep.request_count = (dep.request_count or 0) + len(df)
    dep.last_predicted_at = datetime.utcnow()
    session.add(dep)
    session.commit()

    # Stream CSV response
    output = io.StringIO()
    result["output_df"].to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="predictions_{deployment_id[:8]}.csv"'
        },
    )
