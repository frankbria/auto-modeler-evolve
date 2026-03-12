"""Feature engineering API endpoints.

Workflow:
1. GET  /api/features/{dataset_id}/suggestions   → list of transform suggestions
2. POST /api/features/{dataset_id}/apply         → apply approved transforms, create FeatureSet
3. GET  /api/features/{feature_set_id}/preview   → preview transformed columns
4. POST /api/features/{dataset_id}/target        → set target column + detect problem type
5. GET  /api/features/{dataset_id}/importance    → mutual-information importance scores
"""

import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from core.feature_engine import (
    apply_transformations,
    compute_feature_importance,
    detect_problem_type,
    suggest_features,
)
from db import get_session
from models.dataset import Dataset
from models.feature_set import FeatureSet

router = APIRouter(prefix="/api/features", tags=["features"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_dataset(dataset_id: str, session: Session) -> tuple[Dataset, pd.DataFrame]:
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")
    df = pd.read_csv(file_path)
    return dataset, df


# ---------------------------------------------------------------------------
# 1. Suggestions
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}/suggestions")
def get_suggestions(dataset_id: str, session: Session = Depends(get_session)):
    """Return AI-generated feature transformation suggestions for a dataset."""
    dataset, df = _load_dataset(dataset_id, session)
    column_stats = json.loads(dataset.columns) if dataset.columns else []
    suggestions = suggest_features(df, column_stats)
    return {
        "dataset_id": dataset_id,
        "suggestions": [
            {
                "id": s.id,
                "column": s.column,
                "transform_type": s.transform_type,
                "title": s.title,
                "description": s.description,
                "preview_columns": s.preview_columns,
                "example_values": s.example_values,
            }
            for s in suggestions
        ],
    }


# ---------------------------------------------------------------------------
# 2. Apply transformations
# ---------------------------------------------------------------------------

class ApplyRequest(BaseModel):
    transformations: list[dict]  # list of {column, transform_type, params?}


@router.post("/{dataset_id}/apply", status_code=201)
def apply_feature_transforms(
    dataset_id: str,
    body: ApplyRequest,
    session: Session = Depends(get_session),
):
    """Apply approved transformations and persist the resulting FeatureSet."""
    dataset, df = _load_dataset(dataset_id, session)

    transformed_df, column_mapping = apply_transformations(df, body.transformations)

    # Deactivate any previous active FeatureSet for this dataset
    prev_sets = session.exec(
        select(FeatureSet).where(
            FeatureSet.dataset_id == dataset_id,
            FeatureSet.is_active == True,  # noqa: E712
        )
    ).all()
    for fs in prev_sets:
        fs.is_active = False
        session.add(fs)

    feature_set = FeatureSet(
        dataset_id=dataset_id,
        transformations=json.dumps(body.transformations),
        column_mapping=json.dumps(column_mapping),
        is_active=True,
    )
    session.add(feature_set)
    session.commit()
    session.refresh(feature_set)

    preview = transformed_df.head(5).to_dict(orient="records")
    new_columns = sorted(
        set(transformed_df.columns) - set(df.columns)
    )

    return {
        "feature_set_id": feature_set.id,
        "column_mapping": column_mapping,
        "new_columns": new_columns,
        "total_columns": len(transformed_df.columns),
        "preview": preview,
    }


# ---------------------------------------------------------------------------
# 3. Preview a feature set
# ---------------------------------------------------------------------------

@router.get("/{feature_set_id}/preview")
def preview_feature_set(feature_set_id: str, session: Session = Depends(get_session)):
    """Return a preview of the transformed dataset for a given FeatureSet."""
    feature_set = session.get(FeatureSet, feature_set_id)
    if not feature_set:
        raise HTTPException(status_code=404, detail="FeatureSet not found")

    dataset, df = _load_dataset(feature_set.dataset_id, session)
    transforms = json.loads(feature_set.transformations or "[]")
    transformed_df, column_mapping = apply_transformations(df, transforms)

    preview = transformed_df.head(10).to_dict(orient="records")
    return {
        "feature_set_id": feature_set_id,
        "columns": transformed_df.columns.tolist(),
        "preview": preview,
        "column_mapping": column_mapping,
        "target_column": feature_set.target_column,
        "problem_type": feature_set.problem_type,
    }


# ---------------------------------------------------------------------------
# 4. Set target variable
# ---------------------------------------------------------------------------

class TargetRequest(BaseModel):
    target_column: str
    feature_set_id: str | None = None  # update an existing FeatureSet if provided


@router.post("/{dataset_id}/target")
def set_target(
    dataset_id: str,
    body: TargetRequest,
    session: Session = Depends(get_session),
):
    """Set the target column and detect problem type (classification vs regression)."""
    dataset, df = _load_dataset(dataset_id, session)
    result = detect_problem_type(df, body.target_column)

    # Update FeatureSet if provided, otherwise update the active one
    feature_set = None
    if body.feature_set_id:
        feature_set = session.get(FeatureSet, body.feature_set_id)
    else:
        feature_set = session.exec(
            select(FeatureSet).where(
                FeatureSet.dataset_id == dataset_id,
                FeatureSet.is_active == True,  # noqa: E712
            )
        ).first()

    if feature_set:
        feature_set.target_column = body.target_column
        feature_set.problem_type = result.get("problem_type")
        session.add(feature_set)
        session.commit()

    return {
        "dataset_id": dataset_id,
        "target_column": body.target_column,
        **result,
    }


# ---------------------------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}/importance")
def get_feature_importance(
    dataset_id: str,
    target_column: str,
    session: Session = Depends(get_session),
):
    """Return mutual-information feature importance scores for a target column."""
    dataset, df = _load_dataset(dataset_id, session)
    column_stats = json.loads(dataset.columns) if dataset.columns else []

    type_result = detect_problem_type(df, target_column)
    problem_type = type_result.get("problem_type", "regression")

    importance = compute_feature_importance(df, target_column, problem_type, column_stats)

    return {
        "dataset_id": dataset_id,
        "target_column": target_column,
        "problem_type": problem_type,
        "features": importance,
    }
