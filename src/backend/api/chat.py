import json
import re
from datetime import UTC, datetime
from pathlib import Path

import anthropic
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from chat.orchestrator import build_system_prompt
from core.query_engine import generate_chart_for_message
from db import get_session
from models.conversation import Conversation
from models.dataset import Dataset
from models.deployment import Deployment
from models.feature_set import FeatureSet
from models.model_run import ModelRun
from models.prediction_log import PredictionLog
from models.project import Project

router = APIRouter(prefix="/api/chat", tags=["chat"])

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024

# Keywords that trigger an inline model readiness assessment
_READINESS_PATTERNS = re.compile(
    r"\b(ready|readiness|production.ready|deploy|is.*(model|it).*ready|can.*deploy|"
    r"should.*deploy|good enough|production|go.live|ship it|launch)\b",
    re.IGNORECASE,
)

# Keywords that trigger an inline drift assessment
_DRIFT_PATTERNS = re.compile(
    r"\b(drift|drifting|shifted|predictions.*(off|wrong|different|changed)|"
    r"still accurate|performance.*(drop|degraded|worse)|data.*(changed|shifted|stale)|"
    r"retrain|re.train|model.*stale|stale.*model|distribution.*changed)\b",
    re.IGNORECASE,
)

# Keywords that trigger a hyperparameter tuning suggestion
_TUNE_PATTERNS = re.compile(
    r"\b(tune|tuning|optimize|optimise|improve.*model|better.*model|model.*better|"
    r"increase accuracy|boost performance|hyperparameter|grid search|random search|"
    r"can.*do better|make.*better|improve.*accuracy|improve.*performance|"
    r"best.*hyperparameter|find.*best.*param)\b",
    re.IGNORECASE,
)


class ChatMessage(BaseModel):
    message: str


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _compute_readiness(run: ModelRun, dataset: Dataset | None, feature_set: FeatureSet | None) -> dict:
    """Inline model readiness calculation (same logic as /api/models/{id}/readiness)."""
    metrics = json.loads(run.metrics) if run.metrics else {}
    problem_type = (feature_set.problem_type if feature_set else None) or "regression"
    row_count = dataset.row_count if dataset else 0
    feature_count = len(json.loads(feature_set.column_mapping or "{}")) if feature_set else 0

    checks: list[dict] = []
    total_points = 0
    earned_points = 0

    # Training complete
    total_points += 10
    earned_points += 10
    checks.append({"id": "training_complete", "label": "Training completed", "passed": True, "weight": 10})

    # Sufficient data
    total_points += 20
    data_ok = row_count >= 100
    earned_points += 20 if data_ok else (10 if row_count >= 50 else 0)
    checks.append({"id": "sufficient_data", "label": f"Sufficient data ({row_count} rows)", "passed": data_ok, "weight": 20})

    # Accuracy
    total_points += 30
    if problem_type == "regression":
        r2 = metrics.get("r2", 0.0)
        perf_ok = r2 >= 0.7
        earned_points += 30 if perf_ok else (15 if r2 >= 0.5 else 0)
        checks.append({"id": "accuracy", "label": f"R² = {r2:.3f} (threshold: 0.70)", "passed": perf_ok, "weight": 30})
    else:
        acc = metrics.get("accuracy", 0.0)
        perf_ok = acc >= 0.8
        earned_points += 30 if perf_ok else (15 if acc >= 0.65 else 0)
        checks.append({"id": "accuracy", "label": f"Accuracy = {acc:.1%} (threshold: 80%)", "passed": perf_ok, "weight": 30})

    # Features
    total_points += 15
    has_features = feature_count > 1
    earned_points += 15 if has_features else 5
    checks.append({"id": "features", "label": f"{feature_count} features used", "passed": has_features, "weight": 15})

    # Data quality
    total_points += 15
    profile = json.loads(dataset.profile or "{}") if dataset else {}
    missing_pct = profile.get("missing_percentage", 0.0)
    dq_ok = missing_pct < 10.0
    earned_points += 15 if dq_ok else (8 if missing_pct < 30.0 else 0)
    checks.append({"id": "data_quality", "label": f"Data quality ({missing_pct:.1f}% missing)", "passed": dq_ok, "weight": 15})

    # Selected
    total_points += 10
    earned_points += 10 if run.is_selected else 0
    checks.append({"id": "selected", "label": "Marked as preferred model", "passed": run.is_selected, "weight": 10})

    score = round((earned_points / total_points) * 100) if total_points > 0 else 0
    verdict = "ready" if score >= 85 else ("needs_attention" if score >= 60 else "not_ready")

    return {
        "model_run_id": run.id,
        "algorithm": run.algorithm,
        "score": score,
        "verdict": verdict,
        "checks": checks,
        "problem_type": problem_type,
    }


def _compute_drift(deployment: Deployment, logs: list) -> dict:
    """Compute prediction drift inline (same logic as GET /api/deploy/{id}/drift)."""
    WINDOW = 10
    logs_sorted = sorted(logs, key=lambda l: l.created_at)

    if len(logs_sorted) < WINDOW * 2:
        return {
            "deployment_id": deployment.id,
            "status": "insufficient_data",
            "drift_score": None,
            "explanation": (
                f"Need at least {WINDOW * 2} predictions to detect drift "
                f"(currently {len(logs_sorted)})."
            ),
            "problem_type": deployment.problem_type,
        }

    baseline = logs_sorted[:WINDOW]
    recent = logs_sorted[-WINDOW:]
    problem_type = deployment.problem_type or "regression"

    if problem_type == "regression":
        b_vals = [l.prediction_numeric for l in baseline if l.prediction_numeric is not None]
        r_vals = [l.prediction_numeric for l in recent if l.prediction_numeric is not None]
        if not b_vals or not r_vals:
            return {"deployment_id": deployment.id, "status": "insufficient_data",
                    "drift_score": None, "explanation": "No numeric values.", "problem_type": problem_type}
        b_mean = sum(b_vals) / len(b_vals)
        r_mean = sum(r_vals) / len(r_vals)
        b_std = (sum((v - b_mean) ** 2 for v in b_vals) / len(b_vals)) ** 0.5
        z = abs(r_mean - b_mean) / (b_std + 1e-9)
        drift_score = min(100, int(z * 33))
        status = "stable" if z < 1.0 else ("mild_drift" if z < 2.0 else "significant_drift")
        explanation = (
            f"Prediction mean shifted from {b_mean:.3f} to {r_mean:.3f} "
            f"(z={z:.1f}). Status: {status.replace('_', ' ')}."
        )
    else:
        def _dist(ls: list) -> dict[str, float]:
            counts: dict[str, int] = {}
            for l in ls:
                try:
                    label = str(json.loads(l.prediction))
                except (json.JSONDecodeError, TypeError):
                    label = "unknown"
                counts[label] = counts.get(label, 0) + 1
            total = sum(counts.values()) or 1
            return {k: v / total for k, v in counts.items()}
        b_dist = _dist(baseline)
        r_dist = _dist(recent)
        all_classes = set(b_dist) | set(r_dist)
        tvd = sum(abs(r_dist.get(c, 0) - b_dist.get(c, 0)) for c in all_classes) / 2
        drift_score = min(100, int(tvd * 200))
        status = "stable" if tvd < 0.1 else ("mild_drift" if tvd < 0.25 else "significant_drift")
        explanation = f"Class distribution TVD={tvd:.2f}. Status: {status.replace('_', ' ')}."

    return {
        "deployment_id": deployment.id,
        "status": status,
        "drift_score": drift_score,
        "explanation": explanation,
        "problem_type": problem_type,
    }


def _load_project_context(project_id: str, session: Session) -> dict:
    """Load the full project context needed for the state-aware system prompt."""
    dataset = session.exec(
        select(Dataset).where(Dataset.project_id == project_id)
    ).first()

    # Latest active feature set (most recently created)
    feature_set = None
    if dataset:
        feature_set = session.exec(
            select(FeatureSet)
            .where(FeatureSet.dataset_id == dataset.id, FeatureSet.is_active == True)  # noqa: E712
            .order_by(FeatureSet.created_at.desc())  # type: ignore[arg-type]
        ).first()

    model_runs = list(
        session.exec(
            select(ModelRun).where(ModelRun.project_id == project_id)
        ).all()
    )

    # Latest active deployment
    deployment = session.exec(
        select(Deployment)
        .where(Deployment.project_id == project_id, Deployment.is_active == True)  # noqa: E712
        .order_by(Deployment.created_at.desc())  # type: ignore[arg-type]
    ).first()

    return {
        "dataset": dataset,
        "feature_set": feature_set,
        "model_runs": model_runs,
        "deployment": deployment,
    }


@router.post("/{project_id}")
def send_message(
    project_id: str,
    body: ChatMessage,
    session: Session = Depends(get_session),
):
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get or create conversation
    statement = select(Conversation).where(Conversation.project_id == project_id)
    conversation = session.exec(statement).first()
    if not conversation:
        conversation = Conversation(project_id=project_id)
        session.add(conversation)
        session.commit()
        session.refresh(conversation)

    messages = json.loads(conversation.messages)

    messages.append(
        {
            "role": "user",
            "content": body.message,
            "timestamp": _utcnow().isoformat(),
        }
    )

    # Load full project context for state-aware prompt
    ctx = _load_project_context(project_id, session)

    # Pass recent conversation messages for multi-turn context
    # Exclude the just-appended user message (last item) — Claude already gets
    # the full message list; we only want the preceding turns for the system prompt
    recent_for_context = messages[:-1][-6:]  # up to 3 exchanges before this message
    system_prompt = build_system_prompt(
        project,
        dataset=ctx["dataset"],
        feature_set=ctx["feature_set"],
        model_runs=ctx["model_runs"],
        deployment=ctx["deployment"],
        recent_messages=recent_for_context if recent_for_context else None,
    )

    api_messages = [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]

    client = anthropic.Anthropic()

    # Capture dataset info for post-stream chart generation
    dataset = ctx["dataset"]
    dataset_file_path: str | None = dataset.file_path if dataset else None
    column_info: list = json.loads(dataset.columns) if (dataset and dataset.columns) else []

    # Check if this is a readiness-related question
    readiness_data: dict | None = None
    if _READINESS_PATTERNS.search(body.message):
        completed_runs = [mr for mr in ctx["model_runs"] if mr.status == "done"]
        selected_run = next((mr for mr in completed_runs if mr.is_selected), None)
        target_run = selected_run or (completed_runs[-1] if completed_runs else None)
        if target_run:
            try:
                readiness_data = _compute_readiness(target_run, ctx["dataset"], ctx["feature_set"])
                # Inject readiness summary into system prompt so Claude can incorporate it
                score = readiness_data["score"]
                verdict = readiness_data["verdict"]
                passed = sum(1 for c in readiness_data["checks"] if c["passed"])
                total = len(readiness_data["checks"])
                system_prompt += (
                    f"\n\n## Model Readiness Check (just computed)\n"
                    f"Algorithm: {target_run.algorithm} | Score: {score}/100 | "
                    f"Verdict: {verdict.upper()} | Checks passed: {passed}/{total}\n"
                    "Reference this assessment in your response. Be specific about the score "
                    "and what the user should do next."
                )
            except Exception:  # noqa: BLE001
                pass  # Readiness check is nice-to-have; never crash chat

    # Check if this is a tune/optimize request
    tune_data: dict | None = None
    if _TUNE_PATTERNS.search(body.message):
        completed_runs = [mr for mr in ctx["model_runs"] if mr.status == "done"]
        selected_run = next((mr for mr in completed_runs if mr.is_selected), None)
        target_run = selected_run or (completed_runs[-1] if completed_runs else None)
        if target_run:
            from core.tuner import is_tunable as _is_tunable
            if _is_tunable(target_run.algorithm):
                tune_data = {
                    "model_run_id": target_run.id,
                    "algorithm": target_run.algorithm,
                    "metrics": json.loads(target_run.metrics) if target_run.metrics else {},
                }
                system_prompt += (
                    f"\n\n## Hyperparameter Tuning Available\n"
                    f"The user is asking about improving model performance. "
                    f"Their current best model is {target_run.algorithm} "
                    f"(metrics: {tune_data['metrics']}). "
                    "Inform them that you can automatically tune the hyperparameters using "
                    "RandomizedSearchCV to find better settings — no technical knowledge needed. "
                    "Tell them the Tune button is now available in the Models tab, or they can "
                    "say 'go ahead and tune it' to start immediately."
                )

    # Check if this is a drift-related question
    drift_data: dict | None = None
    if _DRIFT_PATTERNS.search(body.message) and ctx["deployment"]:
        try:
            deployment = ctx["deployment"]
            logs = list(
                session.exec(
                    select(PredictionLog).where(
                        PredictionLog.deployment_id == deployment.id
                    )
                ).all()
            )
            drift_data = _compute_drift(deployment, logs)
            system_prompt += (
                f"\n\n## Prediction Drift Check (just computed)\n"
                f"Status: {drift_data['status']} | "
                f"Drift score: {drift_data['drift_score'] if drift_data['drift_score'] is not None else 'N/A'}/100\n"
                f"{drift_data['explanation']}\n"
                "Reference this drift analysis in your response. Help the user understand "
                "what drift means and whether they need to take action."
            )
        except Exception:  # noqa: BLE001
            pass  # Drift check is nice-to-have; never crash chat

    def stream_response():
        full_response = ""
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=api_messages,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    chunk = json.dumps({"type": "token", "content": text})
                    yield f"data: {chunk}\n\n"

        finally:
            # Save assistant response
            messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": _utcnow().isoformat(),
                }
            )
            from db import engine

            with Session(engine) as save_session:
                conv = save_session.get(Conversation, conversation.id)
                if conv:
                    conv.messages = json.dumps(messages)
                    conv.updated_at = _utcnow()
                    save_session.add(conv)
                    save_session.commit()

        # Emit readiness card if computed
        if readiness_data:
            yield f"data: {json.dumps({'type': 'readiness', 'readiness': readiness_data})}\n\n"

        # Emit drift card if computed
        if drift_data:
            yield f"data: {json.dumps({'type': 'drift', 'drift': drift_data})}\n\n"

        # Emit tune suggestion if detected
        if tune_data:
            yield f"data: {json.dumps({'type': 'tune', 'tune': tune_data})}\n\n"

        # After text stream, opportunistically generate a chart if the
        # message is about data and we have a dataset loaded
        if dataset_file_path:
            try:
                fp = Path(dataset_file_path)
                if fp.exists() and column_info:
                    df = pd.read_csv(fp)
                    chart = generate_chart_for_message(
                        body.message, df, column_info, full_response
                    )
                    if chart:
                        yield f"data: {json.dumps({'type': 'chart', 'chart': chart})}\n\n"
            except Exception:  # noqa: BLE001
                pass  # Charts are nice-to-have; never crash the chat

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/{project_id}/history")
def get_history(
    project_id: str,
    session: Session = Depends(get_session),
):
    statement = select(Conversation).where(Conversation.project_id == project_id)
    conversation = session.exec(statement).first()
    if not conversation:
        return {"messages": []}

    return {"messages": json.loads(conversation.messages)}
