"""Model training API endpoints.

Workflow:
1. GET  /api/models/{project_id}/recommendations  → algorithm suggestions
2. POST /api/models/{project_id}/train            → start training (background)
3. GET  /api/models/{project_id}/runs             → list runs + status
4. GET  /api/models/{project_id}/compare          → side-by-side metrics
5. POST /api/models/{model_run_id}/select         → choose a model
"""

import json
import queue
import threading
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

import db as _db
from chat.narration import (
    append_bot_message_to_conversation,
    narrate_training_with_ai,
)
from core.feature_engine import apply_transformations
from core.report_generator import generate_model_report
from core.chart_builder import build_model_comparison_radar
from core.trainer import (
    CLASSIFICATION_ALGORITHMS,
    REGRESSION_ALGORITHMS,
    get_tuning_grid,
    pick_best_model,
    prepare_features,
    recommend_models,
    train_single_model,
    tune_model,
)
from db import get_session
from models.dataset import Dataset
from models.feature_set import FeatureSet
from models.model_run import ModelRun
from models.project import Project

router = APIRouter(tags=["models"])

# ---------------------------------------------------------------------------
# In-process event bus for SSE training progress
# ---------------------------------------------------------------------------
_lock = threading.Lock()
# project_id → Queue of dicts (or None sentinel for end-of-stream)
_training_queues: dict[str, queue.Queue] = {}
# project_id → count of still-running training threads
_training_counters: dict[str, int] = {}

MODELS_DIR = Path(__file__).parent.parent / "data" / "models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_project_context(
    project_id: str, session: Session
) -> tuple[Project, Dataset, FeatureSet]:
    """Return project, its latest dataset, and active feature set."""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    dataset = session.exec(
        select(Dataset).where(Dataset.project_id == project_id)
    ).first()
    if not dataset:
        raise HTTPException(
            status_code=400,
            detail="No dataset found for this project. Upload a CSV first.",
        )

    feature_set = session.exec(
        select(FeatureSet).where(
            FeatureSet.dataset_id == dataset.id,
            FeatureSet.is_active == True,  # noqa: E712
        )
    ).first()
    if not feature_set:
        raise HTTPException(
            status_code=400,
            detail=(
                "No active feature set found. "
                "Go to the Features tab and apply transformations first."
            ),
        )

    if not feature_set.target_column:
        raise HTTPException(
            status_code=400,
            detail=(
                "Target column not set. "
                "Go to the Features tab and set the column you want to predict."
            ),
        )

    return project, dataset, feature_set


def _run_to_dict(run: ModelRun) -> dict:
    return {
        "id": run.id,
        "algorithm": run.algorithm,
        "status": run.status,
        "is_selected": run.is_selected,
        "is_deployed": run.is_deployed,
        "metrics": json.loads(run.metrics) if run.metrics else None,
        "summary": run.summary,
        "training_duration_ms": run.training_duration_ms,
        "error_message": run.error_message,
        "created_at": run.created_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# Background training worker
# ---------------------------------------------------------------------------


def _push_event(project_id: str, event: dict) -> None:
    """Thread-safe: post an event to the project's SSE queue if one exists."""
    with _lock:
        q = _training_queues.get(project_id)
    if q is not None:
        q.put(event)


def _finish_training_thread(project_id: str) -> None:
    """Decrement the counter; post sentinel when all threads are done."""
    with _lock:
        _training_counters[project_id] = _training_counters.get(project_id, 1) - 1
        remaining = _training_counters[project_id]
        q = _training_queues.get(project_id)
    if remaining <= 0 and q is not None:
        q.put(None)  # sentinel: close SSE stream
        # Inject training-complete narration into the project conversation
        try:
            _narrate_training_done(project_id)
        except Exception:  # noqa: BLE001
            pass  # Narration is nice-to-have; never crash the training thread


def _narrate_training_done(project_id: str) -> None:
    """Load completed runs and append a training-complete bot message."""
    with Session(_db.engine) as session:
        runs = list(
            session.exec(
                select(ModelRun).where(
                    ModelRun.project_id == project_id,
                    ModelRun.status.in_(["done", "failed"]),  # type: ignore[attr-defined]
                )
            ).all()
        )
        if not runs:
            return

        # Determine problem type from active feature set
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

        problem_type = (
            feature_set.problem_type if feature_set else None
        ) or "regression"
        target_col = (feature_set.target_column if feature_set else None) or "target"

        runs_dicts = [
            {
                "algorithm": r.algorithm,
                "status": r.status,
                "metrics": json.loads(r.metrics) if r.metrics else {},
                "summary": r.summary or "",
            }
            for r in runs
        ]

        # Try Claude-powered AI narration first; fall back to static ranking summary
        message = narrate_training_with_ai(runs_dicts, problem_type, target_col)
        append_bot_message_to_conversation(project_id, message, session)


def _train_in_background(
    model_run_id: str,
    project_id: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    algorithm: str,
    problem_type: str,
    model_dir: Path,
) -> None:
    """Runs in a daemon thread. Updates ModelRun status in DB and pushes SSE events."""
    # Mark as training
    with Session(_db.engine) as session:
        run = session.get(ModelRun, model_run_id)
        if not run:
            _finish_training_thread(project_id)
            return
        run.status = "training"
        session.add(run)
        session.commit()

    _push_event(
        project_id,
        {
            "type": "status",
            "run_id": model_run_id,
            "status": "training",
            "algorithm": algorithm,
        },
    )

    try:
        X, y, _ = prepare_features(df, feature_cols, target_col, problem_type)
        result = train_single_model(
            X, y, algorithm, problem_type, model_dir, model_run_id
        )

        with Session(_db.engine) as session:
            run = session.get(ModelRun, model_run_id)
            if run:
                run.status = "done"
                run.metrics = json.dumps(result["metrics"])
                run.model_path = result["model_path"]
                run.training_duration_ms = result["training_duration_ms"]
                run.summary = result["summary"]
                session.add(run)
                session.commit()

        _push_event(
            project_id,
            {
                "type": "done",
                "run_id": model_run_id,
                "status": "done",
                "algorithm": algorithm,
                "metrics": result["metrics"],
                "summary": result["summary"],
                "training_duration_ms": result["training_duration_ms"],
            },
        )

    except Exception as exc:  # noqa: BLE001
        with Session(_db.engine) as session:
            run = session.get(ModelRun, model_run_id)
            if run:
                run.status = "failed"
                run.error_message = str(exc)[:500]
                session.add(run)
                session.commit()

        _push_event(
            project_id,
            {
                "type": "failed",
                "run_id": model_run_id,
                "status": "failed",
                "algorithm": algorithm,
                "error": str(exc)[:200],
            },
        )

    finally:
        _finish_training_thread(project_id)


# ---------------------------------------------------------------------------
# 1. Recommendations
# ---------------------------------------------------------------------------


@router.get("/api/models/{project_id}/recommendations")
def get_recommendations(project_id: str, session: Session = Depends(get_session)):
    """Return algorithm recommendations tailored to this dataset."""
    _, dataset, feature_set = _get_project_context(project_id, session)

    problem_type = feature_set.problem_type or "regression"
    recs = recommend_models(problem_type, dataset.row_count, dataset.column_count)

    return {
        "project_id": project_id,
        "problem_type": problem_type,
        "target_column": feature_set.target_column,
        "n_rows": dataset.row_count,
        "n_features": dataset.column_count,
        "recommendations": recs,
    }


# ---------------------------------------------------------------------------
# 2. Start training
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    algorithms: list[str]


@router.post("/api/models/{project_id}/train", status_code=202)
def start_training(
    project_id: str,
    body: TrainRequest,
    session: Session = Depends(get_session),
):
    """Start background training for the requested algorithms.

    Returns immediately with the list of model_run_ids. Poll
    GET /api/models/{project_id}/runs for status updates.
    """
    _, dataset, feature_set = _get_project_context(project_id, session)

    problem_type = feature_set.problem_type or "regression"
    target_col = feature_set.target_column  # type: ignore[assignment]

    valid_algos = (
        set(REGRESSION_ALGORITHMS)
        if problem_type == "regression"
        else set(CLASSIFICATION_ALGORITHMS)
    )
    invalid = [a for a in body.algorithms if a not in valid_algos]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown algorithms: {invalid}. Valid choices: {sorted(valid_algos)}",
        )

    # Load and transform dataset
    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    df = pd.read_csv(file_path)
    transforms = json.loads(feature_set.transformations or "[]")
    if transforms:
        df, _ = apply_transformations(df, transforms)

    feature_cols = [c for c in df.columns if c != target_col]
    model_dir = MODELS_DIR / project_id

    # Set up SSE event queue for this training batch
    with _lock:
        _training_queues[project_id] = queue.Queue()
        _training_counters[project_id] = len(body.algorithms)

    run_ids = []
    for algo in body.algorithms:
        run = ModelRun(
            project_id=project_id,
            feature_set_id=feature_set.id,
            algorithm=algo,
            hyperparameters=json.dumps({}),
            status="pending",
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_ids.append(run.id)

        t = threading.Thread(
            target=_train_in_background,
            args=(
                run.id,
                project_id,
                df.copy(),
                feature_cols,
                target_col,
                algo,
                problem_type,
                model_dir,
            ),
            daemon=True,
        )
        t.start()

    return {
        "project_id": project_id,
        "model_run_ids": run_ids,
        "algorithms": body.algorithms,
        "status": "training_started",
        "message": (
            f"Training {len(run_ids)} model(s) in the background. "
            "Poll GET /api/models/{project_id}/runs for status updates."
        ),
    }


# ---------------------------------------------------------------------------
# 3. List runs / poll status
# ---------------------------------------------------------------------------


@router.get("/api/models/{project_id}/runs")
def list_runs(project_id: str, session: Session = Depends(get_session)):
    """List all model runs for a project with current status."""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    runs = session.exec(select(ModelRun).where(ModelRun.project_id == project_id)).all()

    return {
        "project_id": project_id,
        "runs": [_run_to_dict(r) for r in runs],
    }


# ---------------------------------------------------------------------------
# 3b. Model version history (timeline of all training runs)
# ---------------------------------------------------------------------------


@router.get("/api/models/{project_id}/history")
def model_history(project_id: str, session: Session = Depends(get_session)):
    """Return the full training history for a project, sorted oldest-first.

    Includes all runs (done, failed, pending, training) with their metrics,
    a performance trend computed over completed runs, and the primary metric
    key/label for the frontend to render the timeline chart.

    Response shape:
        {
          project_id: str,
          problem_type: str,
          primary_metric: str,           # "r2" | "accuracy"
          primary_metric_label: str,     # "R²" | "Accuracy"
          runs: [...],                   # all runs, oldest first
          trend: "improving" | "declining" | "stable" | "insufficient_data",
          trend_summary: str,            # plain-English trend description
          best_metric: float | None,
          latest_metric: float | None,
        }
    """
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Determine problem_type from active feature set
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

    problem_type = (feature_set.problem_type if feature_set else None) or "regression"
    primary_metric = "r2" if problem_type == "regression" else "accuracy"
    primary_metric_label = "R²" if problem_type == "regression" else "Accuracy"

    all_runs = session.exec(
        select(ModelRun)
        .where(ModelRun.project_id == project_id)
        .order_by(ModelRun.created_at)  # type: ignore[attr-defined]
    ).all()

    runs_list = [_run_to_dict(r) for r in all_runs]

    # Extract primary metric values from completed runs (oldest first)
    completed_metrics = [
        r["metrics"][primary_metric]
        for r in runs_list
        if r["status"] == "done"
        and r["metrics"]
        and primary_metric in r["metrics"]
        and r["metrics"][primary_metric] is not None
    ]

    best_metric = max(completed_metrics) if completed_metrics else None
    latest_metric = completed_metrics[-1] if completed_metrics else None

    # Trend: compare first half vs second half of completed runs
    trend, trend_summary = _compute_trend(completed_metrics, primary_metric_label)

    return {
        "project_id": project_id,
        "problem_type": problem_type,
        "primary_metric": primary_metric,
        "primary_metric_label": primary_metric_label,
        "runs": runs_list,
        "trend": trend,
        "trend_summary": trend_summary,
        "best_metric": round(best_metric, 4) if best_metric is not None else None,
        "latest_metric": round(latest_metric, 4) if latest_metric is not None else None,
    }


def _compute_trend(
    metrics: list[float],
    metric_label: str,
) -> tuple[str, str]:
    """Compute a trend direction and plain-English description from a time series.

    Uses linear regression slope over the sequence to determine trend direction.
    Requires at least 2 data points; returns "insufficient_data" otherwise.
    """
    import numpy as np

    n = len(metrics)
    if n < 2:
        return "insufficient_data", "Not enough training runs to determine a trend yet."

    x = np.arange(n, dtype=float)
    y = np.array(metrics, dtype=float)
    # Linear regression slope via least squares
    x_mean = x.mean()
    y_mean = y.mean()
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2))

    # Use the larger of (1% of range) and (2% of mean) as the stable threshold.
    # The 2%-of-mean floor prevents noise in tight ranges from being flagged as trends.
    metric_range = max(abs(y.max() - y.min()), 1e-9)
    threshold = max(0.01 * metric_range, abs(float(y_mean)) * 0.02, 1e-6)

    best = float(y.max())
    latest = float(y[-1])
    delta = latest - float(y[0])

    if slope > threshold:
        trend = "improving"
        pct = abs(delta / (float(y[0]) + 1e-9)) * 100
        summary = (
            f"{metric_label} has improved by {pct:.1f}% over {n} training runs. "
            "Keep going — the model is getting better with each iteration."
        )
    elif slope < -threshold:
        trend = "declining"
        pct = abs(delta / (float(y[0]) + 1e-9)) * 100
        summary = (
            f"{metric_label} has declined by {pct:.1f}% over {n} training runs. "
            "Consider adjusting your features or trying a different algorithm."
        )
    else:
        trend = "stable"
        summary = (
            f"{metric_label} is stable at {best:.3f} across {n} training runs. "
            "The model performance is consistent — try hyperparameter tuning to improve further."
        )

    return trend, summary


# ---------------------------------------------------------------------------
# 4. Compare
# ---------------------------------------------------------------------------


@router.get("/api/models/{project_id}/compare")
def compare_models(project_id: str, session: Session = Depends(get_session)):
    """Return side-by-side metric comparison for all completed model runs."""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    completed_runs = session.exec(
        select(ModelRun).where(
            ModelRun.project_id == project_id,
            ModelRun.status == "done",
        )
    ).all()

    if not completed_runs:
        return {"project_id": project_id, "models": [], "recommendation": None}

    # Determine problem type from the active feature set
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

    problem_type = (feature_set.problem_type if feature_set else None) or "regression"

    models = [_run_to_dict(r) for r in completed_runs]
    recommendation = pick_best_model(models, problem_type)

    return {
        "project_id": project_id,
        "problem_type": problem_type,
        "models": models,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# 4b. Model comparison radar chart
# ---------------------------------------------------------------------------


@router.get("/api/models/{project_id}/comparison-radar")
def comparison_radar(project_id: str, session: Session = Depends(get_session)):
    """Return a radar chart spec comparing all completed model runs.

    Returns 204 No Content when fewer than 2 models are done (radar needs 2+).
    """
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    completed_runs = session.exec(
        select(ModelRun).where(
            ModelRun.project_id == project_id,
            ModelRun.status == "done",
        )
    ).all()

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

    problem_type = (feature_set.problem_type if feature_set else None) or "regression"

    models_data = [
        {
            "algorithm": r.algorithm,
            "run_id": r.id,
            "metrics": json.loads(r.metrics) if r.metrics else {},
        }
        for r in completed_runs
    ]

    chart = build_model_comparison_radar(models_data, problem_type)
    if chart is None:
        return Response(status_code=204)

    return {"project_id": project_id, "problem_type": problem_type, "chart": chart}


# ---------------------------------------------------------------------------
# 5. Select a model
# ---------------------------------------------------------------------------


@router.post("/api/models/{model_run_id}/select")
def select_model(model_run_id: str, session: Session = Depends(get_session)):
    """Mark a completed model run as the selected model for this project."""
    run = session.get(ModelRun, model_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found")

    if run.status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot select a model with status '{run.status}'. Wait for training to complete.",
        )

    # Deselect all other runs in the same project
    other_runs = session.exec(
        select(ModelRun).where(
            ModelRun.project_id == run.project_id,
            ModelRun.is_selected == True,  # noqa: E712
        )
    ).all()
    for r in other_runs:
        r.is_selected = False
        session.add(r)

    run.is_selected = True
    session.add(run)
    session.commit()
    session.refresh(run)

    return _run_to_dict(run)


# ---------------------------------------------------------------------------
# 6. Training progress SSE stream
# ---------------------------------------------------------------------------


@router.get("/api/models/{project_id}/training-stream")
def training_stream(project_id: str):
    """Server-Sent Events stream for real-time training progress.

    Subscribe immediately after POST /api/models/{project_id}/train.
    The stream closes automatically when all runs complete or fail.
    If no training queue exists (already finished), returns a single done event.
    """
    with _lock:
        q = _training_queues.get(project_id)

    def event_generator():
        # If no queue, training already completed — emit done and close
        if q is None:
            yield f"data: {json.dumps({'type': 'all_done'})}\n\n"
            return

        while True:
            try:
                event = q.get(timeout=60)  # 60s max wait per event
            except queue.Empty:
                # Heartbeat keep-alive so connection doesn't time out
                yield ": keep-alive\n\n"
                continue

            if event is None:
                # Sentinel: all training threads finished
                with _lock:
                    _training_queues.pop(project_id, None)
                    _training_counters.pop(project_id, None)
                yield f"data: {json.dumps({'type': 'all_done'})}\n\n"
                return

            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# 7. Download model pickle
# ---------------------------------------------------------------------------


@router.get("/api/models/{model_run_id}/download")
def download_model(model_run_id: str, session: Session = Depends(get_session)):
    """Download the serialized model pipeline as a joblib pickle file."""
    run = session.get(ModelRun, model_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found")
    if run.status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Model is not ready (status: {run.status}). Wait for training to complete.",
        )
    if not run.model_path:
        raise HTTPException(status_code=404, detail="Model file not found")

    model_path = Path(run.model_path)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file missing from disk")

    filename = f"automodeler_{run.algorithm}_{run.id[:8]}.joblib"
    return FileResponse(
        path=str(model_path),
        filename=filename,
        media_type="application/octet-stream",
    )


# ---------------------------------------------------------------------------
# 8. PDF Report
# ---------------------------------------------------------------------------


@router.get("/api/models/{model_run_id}/report")
def download_report(model_run_id: str, session: Session = Depends(get_session)):
    """Generate and download a PDF model report for the given run.

    Includes project metadata, dataset overview, training metrics, feature
    importance, and a confidence/limitations assessment.
    """
    run = session.get(ModelRun, model_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found")
    if run.status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Model is not ready (status: {run.status}). Wait for training to complete.",
        )

    # Gather project and dataset metadata
    project = session.get(Project, run.project_id)
    project_name = project.name if project else "Unknown Project"

    dataset = session.exec(
        select(Dataset).where(Dataset.project_id == run.project_id)
    ).first()
    dataset_filename = dataset.filename if dataset else "unknown.csv"
    dataset_rows = dataset.row_count if dataset else 0
    dataset_columns = dataset.column_count if dataset else 0

    # Determine problem type from feature set
    feature_set = None
    if run.feature_set_id:
        feature_set = session.get(FeatureSet, run.feature_set_id)
    problem_type = (feature_set.problem_type if feature_set else None) or "regression"

    metrics = json.loads(run.metrics) if run.metrics else {}

    # Build feature importances by loading model (best-effort)
    feature_importances = None
    if run.model_path and Path(run.model_path).exists() and feature_set:
        try:
            from core.explainer import compute_feature_importance
            from core.trainer import prepare_features

            df = pd.read_csv(dataset.file_path)
            transforms = json.loads(feature_set.transformations or "[]")
            if transforms:
                df, _ = apply_transformations(df, transforms)
            target_col = feature_set.target_column or ""
            feature_cols = [c for c in df.columns if c != target_col]
            X, y, _ = prepare_features(df, feature_cols, target_col, problem_type)
            pipeline = joblib.load(run.model_path)
            model = pipeline.get("model") if isinstance(pipeline, dict) else pipeline
            fi_result = compute_feature_importance(model, feature_cols)
            feature_importances = fi_result.get("features", [])
        except Exception:  # noqa: BLE001
            feature_importances = None

    # Confidence assessment (best-effort)
    confidence_assessment = None
    if metrics and feature_set:
        try:
            from core.validator import assess_confidence_limitations

            dataset_info = {
                "row_count": dataset_rows,
                "column_count": dataset_columns,
            }
            confidence_assessment = assess_confidence_limitations(
                metrics, problem_type, dataset_info
            )
        except Exception:  # noqa: BLE001
            confidence_assessment = None

    pdf_bytes = generate_model_report(
        project_name=project_name,
        dataset_filename=dataset_filename,
        dataset_rows=dataset_rows,
        dataset_columns=dataset_columns,
        algorithm=run.algorithm,
        problem_type=problem_type,
        metrics=metrics,
        summary=run.summary,
        training_duration_ms=run.training_duration_ms,
        feature_importances=feature_importances,
        confidence_assessment=confidence_assessment,
        created_at=run.created_at,
    )

    safe_algo = run.algorithm.lower().replace(" ", "_")
    filename = f"automodeler_report_{safe_algo}_{run.id[:8]}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Model readiness assessment
# ---------------------------------------------------------------------------


@router.get("/api/models/{model_run_id}/readiness")
def get_model_readiness(
    model_run_id: str,
    session: Session = Depends(get_session),
):
    """Evaluate whether a model is production-ready.

    Returns a readiness score (0–100) plus a plain-English checklist of
    what passed, what needs attention, and what to do next.
    """
    run = session.get(ModelRun, model_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found")
    if run.status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Model is not ready (status: {run.status}). Wait for training to complete.",
        )

    metrics = json.loads(run.metrics) if run.metrics else {}

    # Gather context
    dataset = session.exec(
        select(Dataset).where(Dataset.project_id == run.project_id)
    ).first()
    feature_set = None
    if run.feature_set_id:
        feature_set = session.get(FeatureSet, run.feature_set_id)

    problem_type = (feature_set.problem_type if feature_set else None) or "regression"
    row_count = dataset.row_count if dataset else 0
    feature_count = (
        len(json.loads(feature_set.column_mapping or "{}")) if feature_set else 0
    )

    # --- Checklist evaluation ---
    checks: list[dict] = []
    total_points = 0
    earned_points = 0

    # 1. Training completed successfully
    total_points += 10
    earned_points += 10
    checks.append(
        {
            "id": "training_complete",
            "label": "Training completed successfully",
            "passed": True,
            "detail": f"Model trained in {run.training_duration_ms or 0}ms using {run.algorithm}.",
            "weight": 10,
        }
    )

    # 2. Sufficient training data
    total_points += 20
    min_rows = 100
    data_ok = row_count >= min_rows
    earned_points += 20 if data_ok else (10 if row_count >= 50 else 0)
    checks.append(
        {
            "id": "sufficient_data",
            "label": "Sufficient training data",
            "passed": data_ok,
            "detail": (
                f"{row_count} rows used for training. "
                + (
                    "Great — models generally improve with more data."
                    if data_ok
                    else f"Consider collecting more data (recommended: {min_rows}+ rows)."
                )
            ),
            "weight": 20,
        }
    )

    # 3. Accuracy / performance threshold
    total_points += 30
    if problem_type == "regression":
        r2 = metrics.get("r2", 0.0)
        perf_ok = r2 >= 0.7
        perf_partial = r2 >= 0.5
        earned_points += 30 if perf_ok else (15 if perf_partial else 0)
        checks.append(
            {
                "id": "model_accuracy",
                "label": "Meets accuracy threshold",
                "passed": perf_ok,
                "detail": (
                    f"R² = {r2:.3f} "
                    + (
                        "— excellent fit. Model explains most of the variance in your data."
                        if perf_ok
                        else "— moderate fit. Consider more features or a different algorithm."
                        if perf_partial
                        else "— poor fit. The model may not be reliable for production use."
                    )
                ),
                "weight": 30,
            }
        )
    else:
        acc = metrics.get("accuracy", 0.0)
        perf_ok = acc >= 0.8
        perf_partial = acc >= 0.65
        earned_points += 30 if perf_ok else (15 if perf_partial else 0)
        checks.append(
            {
                "id": "model_accuracy",
                "label": "Meets accuracy threshold",
                "passed": perf_ok,
                "detail": (
                    f"Accuracy = {acc:.1%} "
                    + (
                        "— strong performance. Model is predicting reliably."
                        if perf_ok
                        else "— moderate performance. Consider feature improvements."
                        if perf_partial
                        else "— below threshold. Review your features and target column."
                    )
                ),
                "weight": 30,
            }
        )

    # 4. Features are meaningful (more than 1 feature)
    total_points += 15
    has_features = feature_count > 1
    earned_points += 15 if has_features else 5
    checks.append(
        {
            "id": "feature_quality",
            "label": "Multiple features used",
            "passed": has_features,
            "detail": (
                f"{feature_count} features in the model. "
                + (
                    "Good diversity of input signals."
                    if has_features
                    else "Only 1 feature — consider adding more input columns to improve predictions."
                )
            ),
            "weight": 15,
        }
    )

    # 5. No data quality warnings (missing values handled)
    total_points += 15
    profile = json.loads(dataset.profile or "{}") if dataset else {}
    missing_pct = profile.get("missing_percentage", 0.0)
    data_quality_ok = missing_pct < 10.0
    earned_points += 15 if data_quality_ok else (8 if missing_pct < 30.0 else 0)
    checks.append(
        {
            "id": "data_quality",
            "label": "Data quality is acceptable",
            "passed": data_quality_ok,
            "detail": (
                f"{missing_pct:.1f}% missing values across all columns. "
                + (
                    "Data looks clean."
                    if data_quality_ok
                    else "High missing data rate — review your dataset for completeness."
                )
            ),
            "weight": 15,
        }
    )

    # 6. Model is selected for deployment
    total_points += 10
    is_selected = run.is_selected
    earned_points += 10 if is_selected else 0
    checks.append(
        {
            "id": "model_selected",
            "label": "Marked as the preferred model",
            "passed": is_selected,
            "detail": (
                "This model is selected as the best for this project."
                if is_selected
                else "Mark this model as selected in the Models tab to confirm it's your preferred choice."
            ),
            "weight": 10,
        }
    )

    score = round((earned_points / total_points) * 100) if total_points > 0 else 0

    # Build verdict
    if score >= 85:
        verdict = "ready"
        summary = (
            f"Your {run.algorithm} model scores {score}/100 on the readiness checklist. "
            "It's well-trained, has good accuracy, and uses quality data. "
            "You're ready to deploy."
        )
    elif score >= 60:
        verdict = "needs_attention"
        failing = [c["label"] for c in checks if not c["passed"]]
        summary = (
            f"Your model scores {score}/100. It could go live, but a few things could be improved: "
            + ", ".join(failing)
            + ". See details below."
        )
    else:
        verdict = "not_ready"
        failing = [c["label"] for c in checks if not c["passed"]]
        summary = (
            f"Your model scores {score}/100 and isn't quite ready for production. "
            "Key issues: " + ", ".join(failing) + ". Address these before deploying."
        )

    return {
        "model_run_id": model_run_id,
        "algorithm": run.algorithm,
        "score": score,
        "verdict": verdict,
        "summary": summary,
        "checks": checks,
        "problem_type": problem_type,
    }


# ---------------------------------------------------------------------------
# 9. Hyperparameter auto-tuning
# ---------------------------------------------------------------------------


@router.post("/api/models/{model_run_id}/tune", status_code=201)
def tune_model_endpoint(
    model_run_id: str,
    session: Session = Depends(get_session),
):
    """Auto-tune hyperparameters for a completed model run.

    Runs RandomizedSearchCV on the algorithm used in the original run,
    creates a new ModelRun with the tuned model, and returns a comparison
    of before vs. after metrics.

    - For algorithms with no tunable hyperparameters (e.g. Linear Regression),
      returns tunable=false with an explanation.
    - The new run is NOT automatically selected; the user can compare and choose.
    """
    run = session.get(ModelRun, model_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Model run not found")
    if run.status != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Can only tune completed runs (status is '{run.status}').",
        )

    # Gather feature set + dataset
    feature_set = (
        session.get(FeatureSet, run.feature_set_id) if run.feature_set_id else None
    )
    if not feature_set:
        raise HTTPException(
            status_code=400, detail="Feature set not found for this run."
        )

    dataset = session.exec(
        select(Dataset).where(Dataset.id == feature_set.dataset_id)
    ).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk.")

    problem_type = feature_set.problem_type or "regression"
    target_col = feature_set.target_column or ""

    # Check if algorithm is tunable before loading data
    param_grid = get_tuning_grid(run.algorithm)
    if param_grid is None:
        original_metrics = json.loads(run.metrics) if run.metrics else {}
        return {
            "original_model_run_id": model_run_id,
            "tuned_model_run_id": None,
            "algorithm": run.algorithm,
            "tunable": False,
            "original_metrics": original_metrics,
            "tuned_metrics": None,
            "best_params": None,
            "improved": False,
            "improvement_pct": None,
            "summary": (
                f"{run.algorithm.replace('_', ' ').title()} has no hyperparameters "
                "to tune — it's already using the optimal configuration."
            ),
        }

    # Load + transform dataset
    df = pd.read_csv(file_path)
    transforms = json.loads(feature_set.transformations or "[]")
    if transforms:
        from core.feature_engine import apply_transformations

        df, _ = apply_transformations(df, transforms)

    feature_cols = [c for c in df.columns if c != target_col]
    model_dir = MODELS_DIR / run.project_id

    X, y, _ = prepare_features(df, feature_cols, target_col, problem_type)

    # Create placeholder ModelRun for tuned model
    tuned_run = ModelRun(
        project_id=run.project_id,
        feature_set_id=run.feature_set_id,
        algorithm=run.algorithm,
        hyperparameters=json.dumps({"tuned": True, "original_run_id": model_run_id}),
        status="training",
    )
    session.add(tuned_run)
    session.commit()
    session.refresh(tuned_run)

    try:
        result = tune_model(X, y, run.algorithm, problem_type, model_dir, tuned_run.id)
    except Exception as exc:
        tuned_run.status = "failed"
        tuned_run.error_message = str(exc)[:500]
        session.add(tuned_run)
        session.commit()
        raise HTTPException(status_code=500, detail=f"Tuning failed: {exc}") from exc

    original_metrics = json.loads(run.metrics) if run.metrics else {}
    tuned_metrics = result["metrics"]

    # Determine improvement
    improved = False
    improvement_pct: float | None = None
    if tuned_metrics and original_metrics:
        primary = "r2" if problem_type == "regression" else "accuracy"
        orig_score = original_metrics.get(primary, 0.0)
        tuned_score = tuned_metrics.get(primary, 0.0)
        if orig_score and orig_score != 0:
            improvement_pct = round(
                ((tuned_score - orig_score) / abs(orig_score)) * 100, 2
            )
        improved = bool(tuned_score > orig_score)

    # Save tuned run
    tuned_run.status = "done"
    tuned_run.metrics = json.dumps(tuned_metrics)
    tuned_run.model_path = result["model_path"]
    tuned_run.training_duration_ms = result["training_duration_ms"]
    tuned_run.summary = result["summary"]
    tuned_run.hyperparameters = json.dumps(
        {
            "tuned": True,
            "original_run_id": model_run_id,
            "best_params": result["best_params"],
        }
    )
    session.add(tuned_run)
    session.commit()
    session.refresh(tuned_run)

    # Build human-readable comparison summary
    if improved and improvement_pct is not None:
        primary = "R²" if problem_type == "regression" else "accuracy"
        orig_val = original_metrics.get(
            "r2" if problem_type == "regression" else "accuracy", 0
        )
        tuned_val = tuned_metrics.get(
            "r2" if problem_type == "regression" else "accuracy", 0
        )
        comparison_summary = (
            f"Tuning improved {primary} from {orig_val:.3f} to {tuned_val:.3f} "
            f"(+{improvement_pct:.1f}%). Best settings: "
            + ", ".join(f"{k}={v}" for k, v in result["best_params"].items())
            + "."
        )
    elif improvement_pct is not None and improvement_pct < 0:
        comparison_summary = (
            "Tuning did not improve this model — the default settings are already "
            "well-suited for your data. The original model remains the best choice."
        )
    else:
        comparison_summary = (
            "Tuning complete. Metrics are equivalent to the default — "
            "the original hyperparameters were already close to optimal."
        )

    return {
        "original_model_run_id": model_run_id,
        "tuned_model_run_id": tuned_run.id,
        "algorithm": run.algorithm,
        "tunable": True,
        "original_metrics": original_metrics,
        "tuned_metrics": tuned_metrics,
        "best_params": result["best_params"],
        "tuned_cv_score": result["tuned_cv_score"],
        "improved": improved,
        "improvement_pct": improvement_pct,
        "summary": comparison_summary,
        "tuned_run": _run_to_dict(tuned_run),
    }


# ---------------------------------------------------------------------------
# 10. Smart retrain — reuse existing feature set + selected algorithm
# ---------------------------------------------------------------------------


@router.post("/api/models/{project_id}/retrain", status_code=202)
def retrain_model(
    project_id: str,
    session: Session = Depends(get_session),
):
    """Start a new training run using the project's current configuration.

    Finds the most recently selected (or most recent completed) model run to
    determine which algorithm to use. Reuses the existing active FeatureSet.
    This is the "one-click refresh" endpoint — the user doesn't need to pick
    an algorithm again.

    Returns the same shape as POST /api/models/{project_id}/train.
    """
    _, dataset, feature_set = _get_project_context(project_id, session)

    # Find the algorithm to reuse from the most recent selected or completed run
    all_done_runs = session.exec(
        select(ModelRun).where(
            ModelRun.project_id == project_id,
            ModelRun.status == "done",  # type: ignore[arg-type]
        )
    ).all()

    if not all_done_runs:
        raise HTTPException(
            status_code=400,
            detail=(
                "No completed model runs found. "
                "Train a model first before using retrain."
            ),
        )

    # Prefer the selected run; fall back to the most recently created done run
    selected = next((r for r in all_done_runs if r.is_selected), None)
    source_run = selected or sorted(all_done_runs, key=lambda r: r.created_at)[-1]
    algorithm = source_run.algorithm

    # Strip "_tuned" suffix if present — retrain from the base algorithm
    base_algorithm = algorithm.replace("_tuned", "")
    problem_type = feature_set.problem_type or "regression"
    valid_algos = (
        set(REGRESSION_ALGORITHMS)
        if problem_type == "regression"
        else set(CLASSIFICATION_ALGORITHMS)
    )
    if base_algorithm not in valid_algos:
        # Fall back to a recommended algorithm if the stored one is unrecognised
        recs = recommend_models(problem_type, dataset.row_count, dataset.column_count)
        base_algorithm = recs[0]["algorithm"] if recs else list(valid_algos)[0]

    target_col = feature_set.target_column  # type: ignore[assignment]

    # Load and transform dataset
    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    df = pd.read_csv(file_path)
    transforms = json.loads(feature_set.transformations or "[]")
    if transforms:
        df, _ = apply_transformations(df, transforms)

    feature_cols = [c for c in df.columns if c != target_col]
    model_dir = MODELS_DIR / project_id

    # Set up SSE queue for this training batch
    with _lock:
        _training_queues[project_id] = queue.Queue()
        _training_counters[project_id] = 1

    run = ModelRun(
        project_id=project_id,
        feature_set_id=feature_set.id,
        algorithm=base_algorithm,
        hyperparameters=json.dumps({"retrain": True, "source_run_id": source_run.id}),
        status="pending",
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    t = threading.Thread(
        target=_train_in_background,
        args=(
            run.id,
            project_id,
            df.copy(),
            feature_cols,
            target_col,
            base_algorithm,
            problem_type,
            model_dir,
        ),
        daemon=True,
    )
    t.start()

    return {
        "project_id": project_id,
        "model_run_ids": [run.id],
        "algorithms": [base_algorithm],
        "status": "training_started",
        "source_run_id": source_run.id,
        "message": (
            f"Retraining {base_algorithm} with your current data and features. "
            "Poll GET /api/models/{project_id}/runs for status updates."
        ),
    }
