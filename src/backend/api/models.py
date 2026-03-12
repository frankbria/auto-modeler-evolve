"""Model training API endpoints.

Workflow:
1. GET  /api/models/{project_id}/recommendations  → algorithm suggestions
2. POST /api/models/{project_id}/train            → start training (background)
3. GET  /api/models/{project_id}/runs             → list runs + status
4. GET  /api/models/{project_id}/compare          → side-by-side metrics
5. POST /api/models/{model_run_id}/select         → choose a model
"""

import json
import threading
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

import db as _db
from core.feature_engine import apply_transformations
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

def _train_in_background(
    model_run_id: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    algorithm: str,
    problem_type: str,
    model_dir: Path,
) -> None:
    """Runs in a daemon thread. Updates ModelRun status in DB."""
    # Mark as training
    with Session(_db.engine) as session:
        run = session.get(ModelRun, model_run_id)
        if not run:
            return
        run.status = "training"
        session.add(run)
        session.commit()

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

    except Exception as exc:  # noqa: BLE001
        with Session(_db.engine) as session:
            run = session.get(ModelRun, model_run_id)
            if run:
                run.status = "failed"
                run.error_message = str(exc)[:500]
                session.add(run)
                session.commit()


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
