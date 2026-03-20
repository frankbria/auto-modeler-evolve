import json
import os
from datetime import UTC, datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from db import get_session
from models.dataset import Dataset
from models.deployment import Deployment
from models.feature_set import FeatureSet
from models.feedback_record import FeedbackRecord
from models.model_run import ModelRun
from models.prediction_log import PredictionLog
from models.project import Project

router = APIRouter(prefix="/api/projects", tags=["projects"])


class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    created_at: datetime
    updated_at: datetime
    status: str


class ProjectWithStats(ProjectResponse):
    dataset_id: Optional[str] = None
    dataset_filename: Optional[str] = None
    dataset_rows: Optional[int] = None
    model_count: int = 0
    has_deployment: bool = False


@router.post("", response_model=ProjectResponse, status_code=201)
def create_project(body: ProjectCreate, session: Session = Depends(get_session)):
    project = Project(name=body.name, description=body.description)
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


@router.get("", response_model=list[ProjectWithStats])
def list_projects(session: Session = Depends(get_session)):
    projects = session.exec(select(Project)).all()
    result = []
    for project in projects:
        dataset = session.exec(
            select(Dataset).where(Dataset.project_id == project.id)
        ).first()
        model_count = len(
            session.exec(
                select(ModelRun).where(
                    ModelRun.project_id == project.id,
                    ModelRun.status == "done",
                )
            ).all()
        )
        has_deployment = any(
            mr.is_deployed
            for mr in session.exec(
                select(ModelRun).where(ModelRun.project_id == project.id)
            ).all()
        )
        result.append(
            ProjectWithStats(
                id=project.id,
                name=project.name,
                description=project.description,
                created_at=project.created_at,
                updated_at=project.updated_at,
                status=project.status,
                dataset_id=dataset.id if dataset else None,
                dataset_filename=dataset.filename if dataset else None,
                dataset_rows=dataset.row_count if dataset else None,
                model_count=model_count,
                has_deployment=has_deployment,
            )
        )
    return result


@router.get("/{project_id}", response_model=ProjectWithStats)
def get_project(project_id: str, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    dataset = session.exec(
        select(Dataset).where(Dataset.project_id == project_id)
    ).first()
    model_count = len(
        session.exec(
            select(ModelRun).where(
                ModelRun.project_id == project_id,
                ModelRun.status == "done",
            )
        ).all()
    )
    has_deployment = any(
        mr.is_deployed
        for mr in session.exec(
            select(ModelRun).where(ModelRun.project_id == project_id)
        ).all()
    )
    return ProjectWithStats(
        id=project.id,
        name=project.name,
        description=project.description,
        created_at=project.created_at,
        updated_at=project.updated_at,
        status=project.status,
        dataset_id=dataset.id if dataset else None,
        dataset_filename=dataset.filename if dataset else None,
        dataset_rows=dataset.row_count if dataset else None,
        model_count=model_count,
        has_deployment=has_deployment,
    )


@router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: str,
    body: ProjectUpdate,
    session: Session = Depends(get_session),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if body.name is not None:
        project.name = body.name
    if body.description is not None:
        project.description = body.description
    project.updated_at = datetime.now(UTC).replace(tzinfo=None)
    session.add(project)
    session.commit()
    session.refresh(project)
    return project


@router.post("/{project_id}/duplicate", response_model=ProjectResponse, status_code=201)
def duplicate_project(project_id: str, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    new_project = Project(
        name=f"{project.name} (copy)",
        description=project.description,
        status="exploring",
    )
    session.add(new_project)
    session.commit()
    session.refresh(new_project)
    return new_project


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, session: Session = Depends(get_session)):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    session.delete(project)
    session.commit()


# ---------------------------------------------------------------------------
# AI Project Narrative
# ---------------------------------------------------------------------------


@router.post("/{project_id}/narrative")
def generate_project_narrative(
    project_id: str,
    session: Session = Depends(get_session),
):
    """Generate a plain-English executive summary of the project.

    Gathers all available project artifacts — dataset stats, engineered features,
    trained models, deployment status, prediction analytics — and synthesises them
    into a coherent narrative that a business analyst can share with stakeholders.

    Uses Claude when an Anthropic auth token is available; falls back to a structured
    static summary otherwise.
    """
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # ---- Gather context -------------------------------------------------------

    dataset = session.exec(
        select(Dataset).where(Dataset.project_id == project_id)
    ).first()

    feature_set = None
    if dataset:
        feature_set = session.exec(
            select(FeatureSet).where(
                FeatureSet.dataset_id == dataset.id,
                FeatureSet.is_active == True,  # noqa: E712
            )
        ).first()

    completed_runs = session.exec(
        select(ModelRun).where(
            ModelRun.project_id == project_id,
            ModelRun.status == "done",
        )
    ).all()

    selected_run = next((r for r in completed_runs if r.is_selected), None)
    best_run = selected_run or (completed_runs[0] if completed_runs else None)

    deployment = None
    prediction_count = 0
    if best_run and best_run.is_deployed:
        deployment = session.exec(
            select(Deployment).where(
                Deployment.model_run_id == best_run.id,
                Deployment.is_active == True,  # noqa: E712
            )
        ).first()
        if deployment:
            prediction_count = session.exec(
                select(PredictionLog).where(
                    PredictionLog.deployment_id == deployment.id
                )
            ).__class__  # just count
            logs = session.exec(
                select(PredictionLog).where(
                    PredictionLog.deployment_id == deployment.id
                )
            ).all()
            prediction_count = len(logs)

    # Build context dict for narrative generation
    ctx: dict = {
        "project_name": project.name,
        "project_description": project.description or "",
        "created_at": project.created_at.strftime("%B %d, %Y") if project.created_at else "recently",
    }

    if dataset:
        profile = json.loads(dataset.profile or "{}") if dataset.profile else {}
        ctx["dataset"] = {
            "filename": dataset.filename,
            "rows": dataset.row_count,
            "columns": dataset.column_count,
            "missing_pct": profile.get("missing_percentage", 0),
            "has_outliers": bool(profile.get("outlier_columns")),
        }

    if feature_set:
        transforms = json.loads(feature_set.transformations or "[]")
        column_mapping = json.loads(feature_set.column_mapping or "{}")
        ctx["features"] = {
            "target_column": feature_set.target_column,
            "problem_type": feature_set.problem_type,
            "n_transforms": len(transforms),
            "n_engineered_features": len(column_mapping),
            "transform_types": list({t.get("type", "") for t in transforms if isinstance(t, dict)}),
        }

    if best_run:
        metrics = json.loads(best_run.metrics) if best_run.metrics else {}
        ctx["model"] = {
            "algorithm": best_run.algorithm,
            "metrics": metrics,
            "summary": best_run.summary or "",
            "is_selected": best_run.is_selected,
            "n_models_compared": len(completed_runs),
        }

    if deployment:
        ctx["deployment"] = {
            "is_live": True,
            "endpoint": deployment.endpoint_path,
            "dashboard_url": deployment.dashboard_url,
            "prediction_count": prediction_count,
            "created_at": deployment.created_at.strftime("%B %d, %Y") if deployment.created_at else "recently",
        }
    else:
        ctx["deployment"] = {"is_live": False}

    # ---- Generate narrative ---------------------------------------------------

    narrative = _generate_narrative(ctx)

    return {
        "project_id": project_id,
        "project_name": project.name,
        "narrative": narrative,
        "context": ctx,
        "generated_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
    }


def _generate_narrative(ctx: dict) -> str:
    """Generate narrative via Claude API or static fallback."""
    auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    if auth_token and not auth_token.startswith("sk-placeholder"):
        try:
            return _call_claude_narrative(ctx)
        except Exception:  # noqa: BLE001
            pass  # Fall through to static narrative

    return _static_narrative(ctx)


def _call_claude_narrative(ctx: dict) -> str:
    """Call Claude to generate the narrative."""
    import anthropic

    dataset_section = ""
    if "dataset" in ctx:
        d = ctx["dataset"]
        dataset_section = (
            f"Dataset: {d['filename']} ({d['rows']:,} rows, {d['columns']} columns, "
            f"{d['missing_pct']:.1f}% missing values)"
        )

    features_section = ""
    if "features" in ctx:
        f = ctx["features"]
        features_section = (
            f"Feature engineering: target column '{f['target_column']}', "
            f"problem type: {f['problem_type']}, "
            f"{f['n_transforms']} transformations applied, "
            f"{f['n_engineered_features']} features engineered"
        )

    model_section = ""
    if "model" in ctx:
        m = ctx["model"]
        metrics_str = ", ".join(f"{k}={v}" for k, v in m["metrics"].items() if k not in ("train_size", "test_size"))
        model_section = (
            f"Best model: {m['algorithm']} ({metrics_str}), "
            f"compared against {m['n_models_compared']} total model(s)"
        )

    deploy_section = ""
    if ctx.get("deployment", {}).get("is_live"):
        dep = ctx["deployment"]
        deploy_section = (
            f"Deployment: live since {dep['created_at']}, "
            f"{dep['prediction_count']} predictions made so far"
        )

    prompt = f"""You are summarising an AutoModeler project for a business stakeholder.
Write a concise, plain-English executive summary (3-5 paragraphs) for the project named "{ctx['project_name']}".
Focus on: what was analysed, what was discovered, the model performance in simple terms, and the business value.
Avoid technical jargon. Do not use bullet points. Write in a warm, confident tone as if written by a data analyst colleague.

Project context:
{dataset_section}
{features_section}
{model_section}
{deploy_section}
"""

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


def _static_narrative(ctx: dict) -> str:
    """Build a structured narrative without an LLM."""
    parts = []

    project_name = ctx["project_name"]
    created = ctx.get("created_at", "recently")
    parts.append(f"**{project_name}** — AutoModeler Analysis Report\n")
    parts.append(f"Created {created}.\n")

    if "dataset" in ctx:
        d = ctx["dataset"]
        quality = "clean" if d["missing_pct"] < 5 else "mostly complete" if d["missing_pct"] < 20 else "with some gaps"
        parts.append(
            f"The analysis is based on **{d['filename']}** — "
            f"{d['rows']:,} rows and {d['columns']} columns of data, {quality} "
            f"({d['missing_pct']:.1f}% missing values). "
        )

    if "features" in ctx:
        f = ctx["features"]
        parts.append(
            f"The goal is to predict **{f['target_column']}** "
            f"({f['problem_type']}). "
        )
        if f["n_transforms"] > 0:
            parts.append(
                f"{f['n_transforms']} feature transformations were applied to prepare "
                f"the data for modelling, yielding {f['n_engineered_features']} input features. "
            )

    parts.append("\n")

    if "model" in ctx:
        m = ctx["model"]
        metrics = m["metrics"]
        if "r2" in metrics:
            r2 = metrics["r2"]
            quality = "excellent" if r2 >= 0.9 else "good" if r2 >= 0.7 else "moderate"
            parts.append(
                f"The best model — **{m['algorithm'].replace('_', ' ').title()}** — "
                f"achieves an R² of {r2:.2f} ({quality} fit). "
                f"On average, predictions are within {metrics.get('mae', 0):.2f} units of the actual value. "
            )
        elif "accuracy" in metrics:
            acc = metrics["accuracy"]
            parts.append(
                f"The best model — **{m['algorithm'].replace('_', ' ').title()}** — "
                f"correctly classifies {acc:.1%} of cases. "
                f"F1 score: {metrics.get('f1', 0):.2f}. "
            )
        if m["n_models_compared"] > 1:
            parts.append(f"{m['n_models_compared']} algorithms were trained and compared. ")

    parts.append("\n")

    dep = ctx.get("deployment", {})
    if dep.get("is_live"):
        parts.append(
            f"The model is **live and accepting predictions** (deployed {dep.get('created_at', 'recently')}). "
            f"It has processed {dep.get('prediction_count', 0):,} prediction request(s) so far. "
            f"The prediction dashboard is available at `{dep.get('dashboard_url', '')}`. "
        )
    else:
        parts.append(
            "The model has not yet been deployed. Once deployed, it will be accessible "
            "via a shareable prediction dashboard and a REST API endpoint."
        )

    return "".join(parts)


# ---------------------------------------------------------------------------
# Model Monitoring Alerts
# ---------------------------------------------------------------------------


@router.get("/{project_id}/alerts")
def get_project_alerts(
    project_id: str,
    session: Session = Depends(get_session),
):
    """Scan all active deployments for actionable health alerts.

    For each active deployment in the project, checks:
    - Model age (stale_model): deployed model > 60 days old
    - Prediction drift (drift_detected): distribution shift from PredictionLog
    - Real-world accuracy (poor_feedback): feedback accuracy < 70%
    - Usage gap (no_predictions): active deployment with zero usage after 1 day

    Returns alerts sorted by severity (critical → warning).
    """
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    deployments = session.exec(
        select(Deployment).where(
            Deployment.project_id == project_id,
            Deployment.is_active == True,  # noqa: E712
        )
    ).all()

    alerts: list[dict] = []
    now = datetime.now(UTC).replace(tzinfo=None)

    for dep in deployments:
        run = session.get(ModelRun, dep.model_run_id)
        algorithm_label = dep.algorithm or "model"

        # --- Alert: stale model (age > 60 days) ---
        age_days = 0
        if run and run.created_at:
            age_days = max(0, (now - run.created_at).days)
        if age_days > 90:
            alerts.append({
                "deployment_id": dep.id,
                "algorithm": algorithm_label,
                "severity": "critical",
                "type": "stale_model",
                "message": f"Model '{algorithm_label}' is {age_days} days old — predictions may be unreliable.",
                "recommendation": "Retrain the model with current data using the Retrain button.",
            })
        elif age_days > 60:
            alerts.append({
                "deployment_id": dep.id,
                "algorithm": algorithm_label,
                "severity": "warning",
                "type": "stale_model",
                "message": f"Model '{algorithm_label}' is {age_days} days old — consider refreshing.",
                "recommendation": "Review recent predictions and consider retraining if accuracy has changed.",
            })

        # --- Alert: no predictions (deployed > 1 day with zero usage) ---
        if dep.request_count == 0:
            deployed_days = 0
            if dep.created_at:
                deployed_days = max(0, (now - dep.created_at).days)
            if deployed_days >= 1:
                alerts.append({
                    "deployment_id": dep.id,
                    "algorithm": algorithm_label,
                    "severity": "warning",
                    "type": "no_predictions",
                    "message": f"'{algorithm_label}' has been deployed for {deployed_days} day(s) but received no predictions.",
                    "recommendation": "Share the prediction dashboard link, or use the API to make predictions.",
                })

        # --- Alert: drift detection (requires 40+ prediction logs) ---
        logs = session.exec(
            select(PredictionLog).where(PredictionLog.deployment_id == dep.id)
        ).all()
        if len(logs) >= 40:
            logs_sorted = sorted(logs, key=lambda log: log.created_at)
            window = 20
            baseline = logs_sorted[:window]
            recent = logs_sorted[-window:]
            problem_type = dep.problem_type or "regression"

            drift_score: int | None = None
            if problem_type == "regression":
                b_vals = [log.prediction_numeric for log in baseline if log.prediction_numeric is not None]
                r_vals = [log.prediction_numeric for log in recent if log.prediction_numeric is not None]
                if b_vals and r_vals:
                    b_mean = sum(b_vals) / len(b_vals)
                    r_mean = sum(r_vals) / len(r_vals)
                    b_std = (sum((v - b_mean) ** 2 for v in b_vals) / len(b_vals)) ** 0.5
                    z = abs(r_mean - b_mean) / (b_std + 1e-9)
                    drift_score = min(100, int(z * 33))
            else:
                b_preds = [log.prediction for log in baseline if log.prediction]
                r_preds = [log.prediction for log in recent if log.prediction]
                all_labels = set(b_preds + r_preds)
                if all_labels:
                    bn, rn = len(b_preds) or 1, len(r_preds) or 1
                    tvd = 0.5 * sum(
                        abs(b_preds.count(c) / bn - r_preds.count(c) / rn)
                        for c in all_labels
                    )
                    drift_score = min(100, int(tvd * 200))

            if drift_score is not None and drift_score >= 60:
                severity = "critical" if drift_score >= 80 else "warning"
                alerts.append({
                    "deployment_id": dep.id,
                    "algorithm": algorithm_label,
                    "severity": severity,
                    "type": "drift_detected",
                    "message": f"Prediction drift detected for '{algorithm_label}' (drift score: {drift_score}/100).",
                    "recommendation": "Review your input data for changes and consider retraining with more recent examples.",
                })

        # --- Alert: poor real-world accuracy (feedback accuracy < 70%) ---
        feedback_records = session.exec(
            select(FeedbackRecord).where(FeedbackRecord.deployment_id == dep.id)
        ).all()
        problem_type = dep.problem_type or "regression"
        if problem_type == "classification":
            rated = [fb for fb in feedback_records if fb.is_correct is not None]
            if len(rated) >= 3:
                accuracy = sum(1 for fb in rated if fb.is_correct) / len(rated)
                if accuracy < 0.7:
                    severity = "critical" if accuracy < 0.5 else "warning"
                    alerts.append({
                        "deployment_id": dep.id,
                        "algorithm": algorithm_label,
                        "severity": severity,
                        "type": "poor_feedback",
                        "message": (
                            f"Real-world accuracy for '{algorithm_label}' is {accuracy:.0%} "
                            f"(based on {len(rated)} feedback records)."
                        ),
                        "recommendation": "Retrain with updated labelled examples or add more features.",
                    })
        else:
            paired = [fb for fb in feedback_records if fb.actual_value is not None]
            if len(paired) >= 3:
                preds = []
                for fb in paired:
                    if fb.prediction_log_id:
                        log = session.get(PredictionLog, fb.prediction_log_id)
                        if log and log.prediction_numeric is not None:
                            preds.append((log.prediction_numeric, fb.actual_value))
                if preds:
                    mae = sum(abs(p - a) for p, a in preds) / len(preds)
                    avg_actual = sum(a for _, a in preds) / len(preds)
                    pct_err = abs(mae / (avg_actual + 1e-9))
                    if pct_err > 0.30:
                        severity = "critical" if pct_err > 0.50 else "warning"
                        alerts.append({
                            "deployment_id": dep.id,
                            "algorithm": algorithm_label,
                            "severity": severity,
                            "type": "poor_feedback",
                            "message": (
                                f"Real-world predictions for '{algorithm_label}' are off by "
                                f"{pct_err:.0%} on average (MAE: {mae:.3f})."
                            ),
                            "recommendation": "Gather more training data that reflects current patterns and retrain.",
                        })

    # Sort: critical first, then warning; then by deployment_id for stability
    severity_order = {"critical": 0, "warning": 1}
    alerts.sort(key=lambda a: (severity_order.get(a["severity"], 2), a["deployment_id"]))

    return {
        "project_id": project_id,
        "alert_count": len(alerts),
        "critical_count": sum(1 for a in alerts if a["severity"] == "critical"),
        "warning_count": sum(1 for a in alerts if a["severity"] == "warning"),
        "alerts": alerts,
    }
