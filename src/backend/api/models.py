"""Model training and comparison API endpoints.

Workflow:
1. GET  /api/models/{project_id}/recommend?dataset_id=X&target_column=Y
        → algorithm recommendations with plain-English explanations
2. POST /api/models/{project_id}/train
        → train one or more algorithms, persist ModelRun records
3. GET  /api/models/{project_id}/runs
        → list all ModelRun records for a project
4. GET  /api/models/{project_id}/compare
        → ranked comparison with summary
5. POST /api/models/{model_run_id}/select
        → mark a model as selected (deselects others in same project)
"""

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from core.feature_engine import apply_transformations
from core.trainer import compare_models, recommend_models, train_model
from db import get_session
from models.dataset import Dataset
from models.feature_set import FeatureSet
from models.model_run import ModelRun

router = APIRouter(prefix="/api/models", tags=["models"])

MODELS_DIR = Path(__file__).parent.parent / "data" / "models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_df(dataset_id: str, feature_set_id: str | None, session: Session) -> tuple[Dataset, pd.DataFrame]:
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")
    df = pd.read_csv(file_path)

    if feature_set_id:
        fs = session.get(FeatureSet, feature_set_id)
        if fs and fs.transformations:
            transforms = json.loads(fs.transformations)
            df, _ = apply_transformations(df, transforms)

    return dataset, df


def _run_to_dict(run: ModelRun) -> dict:
    return {
        "id": run.id,
        "project_id": run.project_id,
        "dataset_id": run.dataset_id,
        "feature_set_id": run.feature_set_id,
        "algorithm": run.algorithm,
        "display_name": run.display_name,
        "target_column": run.target_column,
        "problem_type": run.problem_type,
        "metrics": json.loads(run.metrics) if run.metrics else None,
        "training_duration_ms": run.training_duration_ms,
        "is_selected": run.is_selected,
        "status": run.status,
        "created_at": run.created_at.isoformat(),
    }


# ---------------------------------------------------------------------------
# 1. Recommendations
# ---------------------------------------------------------------------------

@router.get("/{project_id}/recommend")
def get_recommendations(
    project_id: str,
    dataset_id: str,
    target_column: str,
    session: Session = Depends(get_session),
):
    """Return algorithm recommendations for this dataset and target column."""
    dataset, df = _load_df(dataset_id, None, session)

    from core.feature_engine import detect_problem_type
    type_result = detect_problem_type(df, target_column)
    problem_type = type_result.get("problem_type") or "regression"

    n_rows, n_features = df.shape
    n_features -= 1  # exclude target

    recommendations = recommend_models(problem_type, n_rows, n_features)
    return {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "target_column": target_column,
        "problem_type": problem_type,
        "problem_type_reason": type_result.get("reason", ""),
        "recommendations": [
            {
                "algorithm": r.algorithm,
                "display_name": r.display_name,
                "description": r.description,
                "reason": r.reason,
                "hyperparameters": r.hyperparameters,
            }
            for r in recommendations
        ],
    }


# ---------------------------------------------------------------------------
# 2. Train
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    dataset_id: str
    target_column: str
    algorithms: list[str]           # e.g. ["random_forest", "logistic_regression"]
    feature_set_id: str | None = None


@router.post("/{project_id}/train", status_code=201)
def train_models(
    project_id: str,
    body: TrainRequest,
    session: Session = Depends(get_session),
):
    """Train one or more algorithms and persist ModelRun records."""
    dataset, df = _load_df(body.dataset_id, body.feature_set_id, session)

    from core.feature_engine import detect_problem_type
    type_result = detect_problem_type(df, body.target_column)
    problem_type = type_result.get("problem_type") or "regression"

    results = []
    for algorithm in body.algorithms:
        try:
            result = train_model(
                df=df,
                target_col=body.target_column,
                problem_type=problem_type,
                algorithm=algorithm,
                hyperparameters={},
                model_dir=MODELS_DIR,
            )
            run = ModelRun(
                project_id=project_id,
                dataset_id=body.dataset_id,
                feature_set_id=body.feature_set_id,
                algorithm=result["algorithm"],
                display_name=result["display_name"],
                target_column=body.target_column,
                problem_type=problem_type,
                metrics=json.dumps(result["metrics"]),
                training_duration_ms=result["training_duration_ms"],
                model_path=result["model_path"],
                status="done",
            )
        except Exception as exc:
            run = ModelRun(
                project_id=project_id,
                dataset_id=body.dataset_id,
                feature_set_id=body.feature_set_id,
                algorithm=algorithm,
                display_name=algorithm.replace("_", " ").title(),
                target_column=body.target_column,
                problem_type=problem_type,
                status="failed",
                metrics=json.dumps({"error": str(exc)}),
            )

        session.add(run)
        session.commit()
        session.refresh(run)
        results.append(_run_to_dict(run))

    return {"runs": results, "problem_type": problem_type}


# ---------------------------------------------------------------------------
# 3. List runs
# ---------------------------------------------------------------------------

@router.get("/{project_id}/runs")
def list_runs(project_id: str, session: Session = Depends(get_session)):
    """Return all ModelRun records for a project, newest first."""
    runs = session.exec(
        select(ModelRun)
        .where(ModelRun.project_id == project_id)
        .order_by(ModelRun.created_at.desc())  # type: ignore[arg-type]
    ).all()
    return {"runs": [_run_to_dict(r) for r in runs]}


# ---------------------------------------------------------------------------
# 4. Compare
# ---------------------------------------------------------------------------

@router.get("/{project_id}/compare")
def compare_project_models(project_id: str, session: Session = Depends(get_session)):
    """Return a ranked comparison with plain-English summary."""
    runs = session.exec(
        select(ModelRun)
        .where(ModelRun.project_id == project_id, ModelRun.status == "done")
        .order_by(ModelRun.created_at.desc())  # type: ignore[arg-type]
    ).all()

    if not runs:
        return {"summary": "No completed models yet.", "rankings": [], "best_model_id": None}

    problem_type = runs[0].problem_type or "regression"
    run_dicts = [_run_to_dict(r) for r in runs]
    return compare_models(run_dicts, problem_type)


# ---------------------------------------------------------------------------
# 5. Select
# ---------------------------------------------------------------------------

@router.post("/{model_run_id}/select")
def select_model(model_run_id: str, session: Session = Depends(get_session)):
    """Mark one model as selected; deselect all others in the same project."""
    run = session.get(ModelRun, model_run_id)
    if not run:
        raise HTTPException(status_code=404, detail="ModelRun not found")

    # Deselect all others
    others = session.exec(
        select(ModelRun).where(
            ModelRun.project_id == run.project_id,
            ModelRun.id != model_run_id,
        )
    ).all()
    for other in others:
        other.is_selected = False
        session.add(other)

    run.is_selected = True
    session.add(run)
    session.commit()

    return {"success": True, "selected_model_id": model_run_id}
