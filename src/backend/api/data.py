import json
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from sqlmodel import Session

from core.analyzer import analyze_dataframe
from db import get_session
from models.dataset import Dataset

router = APIRouter(prefix="/api/data", tags=["data"])

UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"


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

    profile = analyze_dataframe(df)

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

    return {
        "dataset_id": dataset.id,
        "filename": dataset.filename,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "preview": preview_rows,
        "column_stats": profile["columns"],
    }


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

    return {
        "dataset_id": dataset.id,
        "filename": dataset.filename,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "preview": preview_rows,
        "column_stats": column_stats,
    }
