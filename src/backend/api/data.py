import json
import shutil
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session, select

from chat.narration import append_bot_message_to_conversation, narrate_upload
from core.analyzer import analyze_dataframe, compute_full_profile
from core.query_engine import run_nl_query
from db import get_session
from models.dataset import Dataset
from models.project import Project

router = APIRouter(prefix="/api/data", tags=["data"])

UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"
SAMPLE_CSV = Path(__file__).parent.parent / "data" / "sample" / "sample_sales.csv"


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@router.post("/upload", status_code=201)
def upload_csv(
    file: UploadFile,
    project_id: str = Form(...),
    session: Session = Depends(get_session),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    project_upload_dir = UPLOAD_DIR / project_id
    project_upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = project_upload_dir / file.filename
    contents = file.file.read()
    file_path.write_bytes(contents)

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400, detail=f"Failed to parse CSV: {exc}"
        ) from exc

    # Full profiling on upload (includes distributions, correlations, patterns)
    profile = compute_full_profile(df)

    dataset = Dataset(
        project_id=project_id,
        filename=file.filename,
        file_path=str(file_path),
        row_count=profile["row_count"],
        column_count=profile["column_count"],
        columns=json.dumps(profile["columns"]),
        profile=json.dumps(profile),
        size_bytes=len(contents),
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)

    preview_rows = df.head(10).to_dict(orient="records")

    # Inject a proactive bot message into the project conversation
    try:
        col_names = [c["name"] for c in profile["columns"]] if profile.get("columns") else list(df.columns)
        narration = narrate_upload(
            filename=dataset.filename,
            row_count=dataset.row_count,
            col_count=dataset.column_count,
            insights=profile.get("insights"),
            column_names=col_names,
        )
        append_bot_message_to_conversation(project_id, narration, session)
    except Exception:  # noqa: BLE001
        pass  # Narration is nice-to-have; never block the upload response

    return {
        "dataset_id": dataset.id,
        "filename": dataset.filename,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "preview": preview_rows,
        "column_stats": profile["columns"],
        "insights": profile.get("insights", []),
    }


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}/preview")
def get_preview(dataset_id: str, session: Session = Depends(get_session)):
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    df = pd.read_csv(file_path)
    preview_rows = df.head(10).to_dict(orient="records")
    column_stats = json.loads(dataset.columns) if dataset.columns else []

    insights = []
    if dataset.profile:
        try:
            insights = json.loads(dataset.profile).get("insights", [])
        except Exception:  # noqa: BLE001
            pass

    return {
        "dataset_id": dataset.id,
        "filename": dataset.filename,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "preview": preview_rows,
        "column_stats": column_stats,
        "insights": insights,
    }


# ---------------------------------------------------------------------------
# Full profile (cached)
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}/profile")
def get_profile(dataset_id: str, session: Session = Depends(get_session)):
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset.profile:
        return json.loads(dataset.profile)

    # Regenerate if missing (e.g. old data uploaded before full profiling)
    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    df = pd.read_csv(file_path)
    profile = compute_full_profile(df)
    dataset.profile = json.dumps(profile)
    session.add(dataset)
    session.commit()

    return profile


# ---------------------------------------------------------------------------
# Natural language query
# ---------------------------------------------------------------------------

class NLQueryRequest(BaseModel):
    question: str


@router.post("/{dataset_id}/query")
def query_dataset(
    dataset_id: str,
    body: NLQueryRequest,
    session: Session = Depends(get_session),
):
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    df = pd.read_csv(file_path)
    column_info = json.loads(dataset.columns) if dataset.columns else []

    result = run_nl_query(body.question, df, column_info)

    return {
        "question": body.question,
        "answer": result.text,
        "chart_spec": result.chart_spec,
        "result_rows": result.result_rows,
    }


# ---------------------------------------------------------------------------
# Sample dataset loader (onboarding)
# ---------------------------------------------------------------------------

class SampleLoadRequest(BaseModel):
    project_id: str


@router.post("/sample", status_code=201)
def load_sample_dataset(body: SampleLoadRequest, session: Session = Depends(get_session)):
    """Copy the bundled sample sales CSV into the given project as its dataset.

    Idempotent: if the project already has a dataset, returns the existing one.
    """
    project = session.get(Project, body.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Check for existing dataset
    existing = session.exec(
        select(Dataset).where(Dataset.project_id == body.project_id)
    ).first()
    if existing:
        df_existing = pd.read_csv(existing.file_path)
        preview_rows = df_existing.head(10).to_dict(orient="records")
        column_stats = json.loads(existing.columns) if existing.columns else []
        return {
            "dataset_id": existing.id,
            "filename": existing.filename,
            "row_count": existing.row_count,
            "column_count": existing.column_count,
            "preview": preview_rows,
            "column_stats": column_stats,
            "insights": [],
            "already_existed": True,
        }

    if not SAMPLE_CSV.exists():
        raise HTTPException(status_code=500, detail="Sample dataset not found on server")

    project_upload_dir = UPLOAD_DIR / body.project_id
    project_upload_dir.mkdir(parents=True, exist_ok=True)
    dest = project_upload_dir / "sample_sales.csv"
    shutil.copy2(SAMPLE_CSV, dest)

    df = pd.read_csv(dest)
    profile = compute_full_profile(df)

    dataset = Dataset(
        project_id=body.project_id,
        filename="sample_sales.csv",
        file_path=str(dest),
        row_count=profile["row_count"],
        column_count=profile["column_count"],
        columns=json.dumps(profile["columns"]),
        profile=json.dumps(profile),
        size_bytes=dest.stat().st_size,
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)

    preview_rows = df.head(10).to_dict(orient="records")

    # Inject a proactive bot message into the project conversation
    try:
        col_names = [c["name"] for c in profile["columns"]] if profile.get("columns") else list(df.columns)
        narration = narrate_upload(
            filename=dataset.filename,
            row_count=dataset.row_count,
            col_count=dataset.column_count,
            insights=profile.get("insights"),
            column_names=col_names,
        )
        append_bot_message_to_conversation(body.project_id, narration, session)
    except Exception:  # noqa: BLE001
        pass  # Narration is nice-to-have; never block the response

    return {
        "dataset_id": dataset.id,
        "filename": dataset.filename,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "preview": preview_rows,
        "column_stats": profile["columns"],
        "insights": profile.get("insights", []),
        "already_existed": False,
    }


@router.get("/sample/info")
def sample_info():
    """Return metadata about the bundled sample dataset (no auth needed)."""
    if not SAMPLE_CSV.exists():
        raise HTTPException(status_code=404, detail="Sample dataset not available")
    df = pd.read_csv(SAMPLE_CSV)
    return {
        "filename": "sample_sales.csv",
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "description": (
            "Monthly sales data with 200 rows across 5 product lines and 4 regions. "
            "Good for predicting revenue."
        ),
    }
