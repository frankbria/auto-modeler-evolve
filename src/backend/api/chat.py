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

from chat.orchestrator import build_system_prompt, detect_state, generate_suggestions
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

# Keywords that trigger a model health + retraining guidance
_HEALTH_PATTERNS = re.compile(
    r"\b(model.health|health.*model|how.*model.*doing|model.*status|"
    r"should.*retrain|time.*retrain|need.*retrain|when.*retrain|"
    r"train.*again|update.*model|refresh.*model|model.*up.?to.?date|"
    r"model.*current|model.*stale|stale.*model|model.*fresh|"
    r"is.*model.*still.*good|check.*model.*health)\b",
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

# Keywords that trigger a cross-deployment alerts scan
# Note: no trailing \b — patterns use .* wildcards so plurals ("alerts") work fine
_ALERTS_PATTERNS = re.compile(
    r"\b(any.*alert|alert.*model|monitor|check.*model|model.*issue|problem.*model|"
    r"system.*status|health.*check|model.*health.*all|all.*model.*health|"
    r"\bissues\b|anything.*wrong|something.*wrong|model.*ok|models.*ok|"
    r"status.*update|how.*model.*doing|all.*deployment)",
    re.IGNORECASE,
)

# Keywords that trigger the model version history card
_HISTORY_PATTERNS = re.compile(
    r"\b(version.*histor|model.*histor|show.*histor|past.*run|previous.*run|"
    r"training.*histor|model.*over.*time|how.*model.*improv|model.*progress|"
    r"show.*improvement|histor.*model|how.*improv|trend.*model|"
    r"model.*trend|improving.*over|getting.*better)",
    re.IGNORECASE,
)

# Keywords that trigger the prediction analytics card
_ANALYTICS_PATTERNS = re.compile(
    r"\b(prediction.*analytic|analytic.*prediction|how.*many.*prediction|"
    r"prediction.*count|usage.*stat|stat.*usage|prediction.*volume|"
    r"prediction.*log|log.*prediction|how.*often.*predict|prediction.*traffic|"
    r"show.*analytic|prediction.*usage|usage.*dashboard)",
    re.IGNORECASE,
)

# Keywords that suggest a data cleaning operation
# Note: these suggest the operation; actual application requires user confirmation via button.
_CLEAN_PATTERNS = re.compile(
    r"\b(clean|fix.*missing|fill.*missing|fill.*null|fill.*empty|"
    r"remove.*duplicat|drop.*duplicat|deduplic|dedup|"
    r"remove.*rows.*where|drop.*rows.*where|filter.*out|exclude.*rows|"
    r"cap.*outlier|remove.*outlier|handle.*outlier|clip.*outlier|"
    r"drop.*column|remove.*column|delete.*column|"
    r"fix.*data|clean.*data|clean.*up|data.*quality|improve.*data|"
    r"missing.*value|null.*value|handle.*null|handle.*missing)",
    re.IGNORECASE,
)

_FILL_COL_PATTERN = re.compile(
    r"\bfill\s+(?:missing\s+)?(?:values?\s+in\s+)?[\"']?(\w+)[\"']?\s+"
    r"(?:column\s+)?with\s+(mean|median|mode|zero|[\d.]+)",
    re.IGNORECASE,
)
_FILTER_COL_PATTERN = re.compile(
    r"\b(?:remove|drop|filter|exclude)\s+rows?\s+where\s+[\"']?(\w+)[\"']?\s*"
    r"(>|<|>=|<=|==|!=|is|equals?|greater than|less than|not)\s*([\d.]+|\w+)",
    re.IGNORECASE,
)
_CAP_COL_PATTERN = re.compile(
    r"\bcap\s+(?:outliers?\s+in\s+)?[\"']?(\w+)[\"']?(?:\s+at\s+([\d.]+)\s*%?)?",
    re.IGNORECASE,
)
_DROP_COL_PATTERN = re.compile(
    r"\b(?:drop|remove|delete)\s+(?:the\s+)?(?:column\s+)?[\"']?(\w+)[\"']?\s+column",
    re.IGNORECASE,
)

_OP_MAP = {"is": "eq", "equals": "eq", "equal": "eq", ">": "gt", "<": "lt",
           ">=": "gte", "<=": "lte", "!=": "ne", "not": "ne",
           "greater than": "gt", "less than": "lt"}


def _detect_clean_op(message: str, columns: list[str]) -> dict | None:
    """Try to extract a specific cleaning operation from the user's message.

    Returns a dict matching CleanRequest fields, or None if only general intent
    detected (caller emits a generic suggestion).
    """
    col_set = set(c.lower() for c in columns)

    # fill_missing: "fill missing age with median" / "fill age column with 0"
    m = _FILL_COL_PATTERN.search(message)
    if m:
        col_raw, strat = m.group(1), m.group(2).lower()
        col = next((c for c in columns if c.lower() == col_raw.lower()), col_raw)
        try:
            val = float(strat)
            return {"operation": "fill_missing", "column": col, "strategy": "value", "fill_value": val}
        except ValueError:
            pass
        if strat in ("mean", "median", "mode", "zero"):
            return {"operation": "fill_missing", "column": col, "strategy": strat}

    # remove_duplicates: "remove duplicates"
    if re.search(r"\b(duplicate|dedup)\b", message, re.IGNORECASE):
        return {"operation": "remove_duplicates"}

    # filter_rows: "drop rows where quantity < 0"
    m = _FILTER_COL_PATTERN.search(message)
    if m:
        col_raw, op_raw, val_raw = m.group(1), m.group(2).strip().lower(), m.group(3)
        col = next((c for c in columns if c.lower() == col_raw.lower()), col_raw)
        operator = _OP_MAP.get(op_raw, "eq")
        try:
            val: float | str = float(val_raw)
        except ValueError:
            val = val_raw
        return {"operation": "filter_rows", "column": col, "operator": operator, "value": val}

    # cap_outliers: "cap outliers in sales" / "cap revenue outliers at 99%"
    m = _CAP_COL_PATTERN.search(message)
    if m:
        col_raw = m.group(1)
        pct_raw = m.group(2)
        col = next((c for c in columns if c.lower() == col_raw.lower()), None)
        if col or col_raw.lower() in col_set:
            resolved = col or col_raw
            pct = float(pct_raw) if pct_raw else 99.0
            return {"operation": "cap_outliers", "column": resolved, "percentile": pct}

    # drop_column: "drop the sales column" / "remove region column"
    m = _DROP_COL_PATTERN.search(message)
    if m:
        col_raw = m.group(1)
        col = next((c for c in columns if c.lower() == col_raw.lower()), None)
        if col:
            return {"operation": "drop_column", "column": col}

    return None  # general cleaning intent — no specific op detected


# Keywords that trigger anomaly detection on the current dataset
_ANOMALY_PATTERNS = re.compile(
    r"\b(anomal|unusual.*record|outlier|strange.*data|weird.*record|"
    r"suspicious|find.*weird|anything.*odd|odd.*row|odd.*record|"
    r"which.*record.*unusual|unusual.*data|data.*unusual|"
    r"detect.*anomal|spot.*anomal|identify.*anomal|show.*anomal)",
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


def _compute_health(deployment: Deployment, run: ModelRun, feedback_records: list, all_logs: list) -> dict:
    """Compute model health inline (same logic as GET /api/deploy/{id}/health).

    Returns health_score 0-100, status, and a human-readable summary suitable
    for injecting into the system prompt.
    """
    from datetime import UTC, datetime

    # Age
    age_days = 0
    age_score = 100
    if run and run.created_at:
        now = datetime.now(UTC).replace(tzinfo=None)
        age_days = max(0, (now - run.created_at).days)
        age_score = 100 if age_days <= 30 else (75 if age_days <= 60 else (50 if age_days <= 90 else 25))

    # Feedback
    feedback_score = 100
    feedback_note = "no feedback yet"
    has_feedback = bool(feedback_records)
    if has_feedback:
        problem_type = deployment.problem_type or "regression"
        if problem_type == "regression":
            pairs = [
                (fb.actual_value, fb.prediction_log_id)
                for fb in feedback_records
                if fb.actual_value is not None and fb.prediction_log_id
            ]
            # Simplified: just use count-based heuristic if we can't get pairs
            if pairs:
                feedback_score = 75  # Moderate — we have data but can't compute inline without session
            else:
                feedback_score = 80
        else:
            rated = [fb for fb in feedback_records if fb.is_correct is not None]
            if rated:
                accuracy = sum(1 for fb in rated if fb.is_correct) / len(rated)
                feedback_score = 100 if accuracy >= 0.9 else (75 if accuracy >= 0.75 else (50 if accuracy >= 0.6 else 20))
                feedback_note = f"{accuracy:.1%} real-world accuracy from {len(rated)} feedback records"

    # Drift
    drift_health_score = 100
    has_drift_data = len(all_logs) >= 40
    if has_drift_data:
        logs_sorted = sorted(all_logs, key=lambda l: l.created_at)
        window = 20
        baseline_logs = logs_sorted[:window]
        recent_logs = logs_sorted[-window:]
        problem_type = deployment.problem_type or "regression"
        if problem_type == "regression":
            b_vals = [l.prediction_numeric for l in baseline_logs if l.prediction_numeric is not None]
            r_vals = [l.prediction_numeric for l in recent_logs if l.prediction_numeric is not None]
            if b_vals and r_vals:
                b_mean = sum(b_vals) / len(b_vals)
                r_mean = sum(r_vals) / len(r_vals)
                b_std = (sum((v - b_mean) ** 2 for v in b_vals) / len(b_vals)) ** 0.5
                z = abs(r_mean - b_mean) / (b_std + 1e-9)
                drift_health_score = 100 if z < 1.0 else (60 if z < 2.0 else 25)
        else:
            b_preds = [str(json.loads(l.prediction)) for l in baseline_logs if l.prediction]
            r_preds = [str(json.loads(l.prediction)) for l in recent_logs if l.prediction]
            all_classes = set(b_preds + r_preds)
            if all_classes:
                b_n, r_n = len(b_preds) or 1, len(r_preds) or 1
                tvd = 0.5 * sum(abs(b_preds.count(c) / b_n - r_preds.count(c) / r_n) for c in all_classes)
                drift_health_score = 100 if tvd < 0.1 else (60 if tvd < 0.25 else 25)

    # Composite
    if has_feedback and has_drift_data:
        health_score = int(feedback_score * 0.4 + drift_health_score * 0.35 + age_score * 0.25)
    elif has_feedback:
        health_score = int(feedback_score * 0.55 + age_score * 0.45)
    elif has_drift_data:
        health_score = int(drift_health_score * 0.6 + age_score * 0.4)
    else:
        health_score = age_score

    status = "healthy" if health_score >= 75 else ("warning" if health_score >= 50 else "critical")

    return {
        "deployment_id": deployment.id,
        "health_score": health_score,
        "status": status,
        "model_age_days": age_days,
        "algorithm": deployment.algorithm,
        "has_feedback_data": has_feedback,
        "has_drift_data": has_drift_data,
        "feedback_note": feedback_note,
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

    # Check if this is a model health / retraining question
    health_data: dict | None = None
    if _HEALTH_PATTERNS.search(body.message) and ctx["deployment"]:
        try:
            deployment = ctx["deployment"]
            run_for_health = next(
                (mr for mr in ctx["model_runs"] if mr.id == deployment.model_run_id), None
            )
            if run_for_health:
                from models.feedback_record import FeedbackRecord
                fb_records = list(
                    session.exec(
                        select(FeedbackRecord).where(
                            FeedbackRecord.deployment_id == deployment.id
                        )
                    ).all()
                )
                logs_for_health = list(
                    session.exec(
                        select(PredictionLog).where(
                            PredictionLog.deployment_id == deployment.id
                        )
                    ).all()
                )
                health_data = _compute_health(deployment, run_for_health, fb_records, logs_for_health)
                score = health_data["health_score"]
                health_status = health_data["status"]
                system_prompt += (
                    f"\n\n## Model Health Check (just computed)\n"
                    f"Algorithm: {deployment.algorithm} | Health score: {score}/100 | "
                    f"Status: {health_status.upper()} | Age: {health_data['model_age_days']} day(s)\n"
                    f"Has feedback data: {health_data['has_feedback_data']} | "
                    f"Has drift data: {health_data['has_drift_data']}\n"
                    "Reference this health check in your response. Explain what the score means "
                    "and whether the user should consider retraining. If the model is healthy, "
                    "reassure them. If it's warning or critical, guide them to retrain using the "
                    "'Retrain' button in the Models tab or by clicking the health card."
                )
        except Exception:  # noqa: BLE001
            pass  # Health check is nice-to-have; never crash chat

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

    # Check for cross-deployment alerts request
    alerts_data: dict | None = None
    if _ALERTS_PATTERNS.search(body.message):
        try:

            active_deployments = list(
                session.exec(
                    select(Deployment).where(
                        Deployment.project_id == project_id,
                        Deployment.is_active == True,  # noqa: E712
                    )
                ).all()
            )
            alert_list: list[dict] = []
            now_ts = datetime.now(UTC).replace(tzinfo=None)

            for dep in active_deployments:
                run_a = session.get(ModelRun, dep.model_run_id)
                alg = dep.algorithm or "model"
                age_d = 0
                if run_a and run_a.created_at:
                    age_d = max(0, (now_ts - run_a.created_at).days)
                if age_d > 60:
                    alert_list.append({
                        "deployment_id": dep.id, "algorithm": alg,
                        "severity": "critical" if age_d > 90 else "warning",
                        "type": "stale_model",
                        "message": f"'{alg}' is {age_d} days old.",
                        "recommendation": "Consider retraining with more recent data.",
                    })
                if dep.request_count == 0 and dep.created_at:
                    dep_age = max(0, (now_ts - dep.created_at).days)
                    if dep_age >= 1:
                        alert_list.append({
                            "deployment_id": dep.id, "algorithm": alg,
                            "severity": "warning", "type": "no_predictions",
                            "message": f"'{alg}' has been deployed {dep_age} day(s) with no predictions.",
                            "recommendation": "Share the dashboard link to start receiving predictions.",
                        })

            alerts_data = {
                "project_id": project_id,
                "alert_count": len(alert_list),
                "critical_count": sum(1 for a in alert_list if a["severity"] == "critical"),
                "warning_count": sum(1 for a in alert_list if a["severity"] == "warning"),
                "alerts": alert_list,
            }
            alert_summary = (
                f"{len(alert_list)} alert(s) found: "
                f"{alerts_data['critical_count']} critical, {alerts_data['warning_count']} warning."
                if alert_list else "No active alerts — all deployments look healthy."
            )
            system_prompt += (
                f"\n\n## Deployment Alerts (just scanned)\n{alert_summary}\n"
                "Summarise the alert status for the user. If there are critical alerts, "
                "guide them on what to do. If everything is healthy, reassure them."
            )
        except Exception:  # noqa: BLE001
            pass  # Alerts are nice-to-have; never crash chat

    # Check for model version history request
    history_event: dict | None = None
    if _HISTORY_PATTERNS.search(body.message) and ctx["model_runs"]:
        completed = [mr for mr in ctx["model_runs"] if mr.status == "done"]
        if len(completed) >= 2:
            history_event = {"project_id": project_id}
            system_prompt += (
                "\n\n## Model Version History\n"
                f"The project has {len(completed)} completed training run(s). "
                "The Version History card is now visible in the Models tab — it shows "
                "a timeline of model performance and trend direction. "
                "Tell the user their model history is available in the Models tab."
            )

    # Check for prediction analytics request
    analytics_event: dict | None = None
    if _ANALYTICS_PATTERNS.search(body.message) and ctx["deployment"]:
        dep_for_analytics = ctx["deployment"]
        count = dep_for_analytics.request_count
        analytics_event = {
            "deployment_id": dep_for_analytics.id,
            "total_predictions": count,
        }
        system_prompt += (
            f"\n\n## Prediction Analytics\n"
            f"The active deployment has logged {count} prediction(s) total. "
            "The Analytics card is visible in the Deployment tab with a usage chart. "
            "Reference the prediction count in your response and mention the Analytics card."
        )

    # Check for data cleaning suggestion request
    # Vision: "Explain before executing" — we suggest the operation, user confirms via button.
    cleaning_suggestion: dict | None = None
    if _CLEAN_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = pd.read_csv(_file_path)
                _cols = list(_df.columns)
                _op = _detect_clean_op(body.message, _cols)
                # Build a quality summary for context
                _null_counts = {col: int(_df[col].isna().sum()) for col in _cols if _df[col].isna().any()}
                _dup_count = int(_df.duplicated().sum())
                _context_parts = []
                if _dup_count > 0:
                    _context_parts.append(f"{_dup_count} duplicate row(s)")
                if _null_counts:
                    _top = sorted(_null_counts.items(), key=lambda x: -x[1])[:3]
                    _context_parts.append(
                        "missing values in: " + ", ".join(f"'{k}' ({v})" for k, v in _top)
                    )
                cleaning_suggestion = {
                    "dataset_id": _ds.id,
                    "suggested_operation": _op,
                    "quality_summary": {
                        "duplicate_rows": _dup_count,
                        "missing_value_columns": _null_counts,
                        "total_rows": len(_df),
                    },
                }
                _ctx_text = "; ".join(_context_parts) if _context_parts else "no obvious issues detected"
                system_prompt += (
                    f"\n\n## Data Cleaning Context\n"
                    f"Dataset quality: {_ctx_text}. "
                    + (
                        f"The user seems to want: {_op['operation'].replace('_', ' ')} "
                        + (f"on column '{_op.get('column')}'" if _op and _op.get('column') else "")
                        + ". A cleaning suggestion card is shown — explain what it will do and ask the user to confirm before applying."
                        if _op
                        else "Describe the available cleaning operations (remove duplicates, fill missing values, filter rows, cap outliers, drop columns) and let the user choose."
                    )
                )
        except Exception:  # noqa: BLE001
            pass  # Cleaning suggestion is nice-to-have; never crash chat

    # Check for anomaly detection request
    anomaly_event: dict | None = None
    if _ANOMALY_PATTERNS.search(body.message) and ctx["dataset"]:
        try:
            from core.anomaly import detect_anomalies as _detect
            _ds = ctx["dataset"]
            _file_path = Path(_ds.file_path)
            if _file_path.exists():
                _df = pd.read_csv(_file_path)
                _numeric_cols = _df.select_dtypes(include="number").columns.tolist()[:10]
                if _numeric_cols:
                    _result = _detect(_df, features=_numeric_cols, contamination=0.05, n_top=10)
                    anomaly_event = {"dataset_id": _ds.id, **_result}
                    system_prompt += (
                        f"\n\n## Anomaly Detection Results\n"
                        f"{_result['summary']}\n"
                        f"Features analysed: {', '.join(_result['features_used'])}.\n"
                        "The Anomaly Detection card is now visible in the Data tab. "
                        "Tell the user what you found and suggest they examine the top anomalous rows."
                    )
        except Exception:  # noqa: BLE001
            pass  # Anomaly detection is nice-to-have; never crash chat

    # Pre-compute follow-up suggestions (based on state + current message)
    current_state = detect_state(
        ctx["dataset"], ctx["feature_set"], ctx["model_runs"], ctx["deployment"]
    )
    suggestions_list = generate_suggestions(
        state=current_state,
        dataset=ctx["dataset"],
        feature_set=ctx["feature_set"],
        model_runs=ctx["model_runs"],
        deployment=ctx["deployment"],
        last_user_message=body.message,
    )

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

        # Emit model health card if computed
        if health_data:
            yield f"data: {json.dumps({'type': 'health', 'health': health_data})}\n\n"

        # Emit deployment alerts if scanned
        if alerts_data:
            yield f"data: {json.dumps({'type': 'alerts', 'alerts': alerts_data})}\n\n"

        # Emit model history trigger if detected
        if history_event:
            yield f"data: {json.dumps({'type': 'history', 'history': history_event})}\n\n"

        # Emit analytics trigger if detected
        if analytics_event:
            yield f"data: {json.dumps({'type': 'analytics', 'analytics': analytics_event})}\n\n"

        # Emit anomaly detection results if computed
        if anomaly_event:
            yield f"data: {json.dumps({'type': 'anomalies', 'anomalies': anomaly_event})}\n\n"

        # Emit cleaning suggestion (user must click to apply — "explain before executing")
        if cleaning_suggestion:
            yield f"data: {json.dumps({'type': 'cleaning_suggestion', 'cleaning': cleaning_suggestion})}\n\n"

        # Emit follow-up suggestion chips (always, if we have any)
        if suggestions_list:
            yield f"data: {json.dumps({'type': 'suggestions', 'suggestions': suggestions_list})}\n\n"

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
