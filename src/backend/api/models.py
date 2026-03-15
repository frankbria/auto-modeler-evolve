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
    narrate_training_complete,
    narrate_training_with_ai,
)
from core.feature_engine import apply_transformations
from core.report_generator import generate_model_report
from core.chart_builder import build_model_comparison_radar
from core.trainer import (
    CLASSIFICATION_ALGORITHMS,
    REGRESSION_ALGORITHMS,
    pick_best_model,
    prepare_features,
    recommend_models,
    train_single_model,
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
        runs = list(session.exec(
            select(ModelRun).where(
                ModelRun.project_id == project_id,
                ModelRun.status.in_(["done", "failed"]),  # type: ignore[attr-defined]
            )
        ).all())
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

        problem_type = (feature_set.problem_type if feature_set else None) or "regression"
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

    _push_event(project_id, {"type": "status", "run_id": model_run_id, "status": "training", "algorithm": algorithm})

    try:
        X, y, _ = prepare_features(df, feature_cols, target_col, problem_type)
        result = train_single_model(X, y, algorithm, problem_type, model_dir, model_run_id)

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

        _push_event(project_id, {
            "type": "done",
            "run_id": model_run_id,
            "status": "done",
            "algorithm": algorithm,
            "metrics": result["metrics"],
            "summary": result["summary"],
            "training_duration_ms": result["training_duration_ms"],
        })

    except Exception as exc:  # noqa: BLE001
        with Session(_db.engine) as session:
            run = session.get(ModelRun, model_run_id)
            if run:
                run.status = "failed"
                run.error_message = str(exc)[:500]
                session.add(run)
                session.commit()

        _push_event(project_id, {"type": "failed", "run_id": model_run_id, "status": "failed", "algorithm": algorithm, "error": str(exc)[:200]})

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

    runs = session.exec(
        select(ModelRun).where(ModelRun.project_id == project_id)
    ).all()

    return {
        "project_id": project_id,
        "runs": [_run_to_dict(r) for r in runs],
    }


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
    feature_count = len(json.loads(feature_set.column_mapping or "{}")) if feature_set else 0

    # --- Checklist evaluation ---
    checks: list[dict] = []
    total_points = 0
    earned_points = 0

    # 1. Training completed successfully
    total_points += 10
    earned_points += 10
    checks.append({
        "id": "training_complete",
        "label": "Training completed successfully",
        "passed": True,
        "detail": f"Model trained in {run.training_duration_ms or 0}ms using {run.algorithm}.",
        "weight": 10,
    })

    # 2. Sufficient training data
    total_points += 20
    min_rows = 100
    data_ok = row_count >= min_rows
    earned_points += 20 if data_ok else (10 if row_count >= 50 else 0)
    checks.append({
        "id": "sufficient_data",
        "label": "Sufficient training data",
        "passed": data_ok,
        "detail": (
            f"{row_count} rows used for training. "
            + ("Great — models generally improve with more data." if data_ok
               else f"Consider collecting more data (recommended: {min_rows}+ rows).")
        ),
        "weight": 20,
    })

    # 3. Accuracy / performance threshold
    total_points += 30
    if problem_type == "regression":
        r2 = metrics.get("r2", 0.0)
        perf_ok = r2 >= 0.7
        perf_partial = r2 >= 0.5
        earned_points += 30 if perf_ok else (15 if perf_partial else 0)
        checks.append({
            "id": "model_accuracy",
            "label": "Meets accuracy threshold",
            "passed": perf_ok,
            "detail": (
                f"R² = {r2:.3f} "
                + ("— excellent fit. Model explains most of the variance in your data." if perf_ok
                   else "— moderate fit. Consider more features or a different algorithm." if perf_partial
                   else "— poor fit. The model may not be reliable for production use.")
            ),
            "weight": 30,
        })
    else:
        acc = metrics.get("accuracy", 0.0)
        perf_ok = acc >= 0.8
        perf_partial = acc >= 0.65
        earned_points += 30 if perf_ok else (15 if perf_partial else 0)
        checks.append({
            "id": "model_accuracy",
            "label": "Meets accuracy threshold",
            "passed": perf_ok,
            "detail": (
                f"Accuracy = {acc:.1%} "
                + ("— strong performance. Model is predicting reliably." if perf_ok
                   else "— moderate performance. Consider feature improvements." if perf_partial
                   else "— below threshold. Review your features and target column.")
            ),
            "weight": 30,
        })

    # 4. Features are meaningful (more than 1 feature)
    total_points += 15
    has_features = feature_count > 1
    earned_points += 15 if has_features else 5
    checks.append({
        "id": "feature_quality",
        "label": "Multiple features used",
        "passed": has_features,
        "detail": (
            f"{feature_count} features in the model. "
            + ("Good diversity of input signals." if has_features
               else "Only 1 feature — consider adding more input columns to improve predictions.")
        ),
        "weight": 15,
    })

    # 5. No data quality warnings (missing values handled)
    total_points += 15
    profile = json.loads(dataset.profile or "{}") if dataset else {}
    missing_pct = profile.get("missing_percentage", 0.0)
    data_quality_ok = missing_pct < 10.0
    earned_points += 15 if data_quality_ok else (8 if missing_pct < 30.0 else 0)
    checks.append({
        "id": "data_quality",
        "label": "Data quality is acceptable",
        "passed": data_quality_ok,
        "detail": (
            f"{missing_pct:.1f}% missing values across all columns. "
            + ("Data looks clean." if data_quality_ok
               else "High missing data rate — review your dataset for completeness.")
        ),
        "weight": 15,
    })

    # 6. Model is selected for deployment
    total_points += 10
    is_selected = run.is_selected
    earned_points += 10 if is_selected else 0
    checks.append({
        "id": "model_selected",
        "label": "Marked as the preferred model",
        "passed": is_selected,
        "detail": (
            "This model is selected as the best for this project."
            if is_selected
            else "Mark this model as selected in the Models tab to confirm it's your preferred choice."
        ),
        "weight": 10,
    })

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
            + ", ".join(failing) + ". See details below."
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
