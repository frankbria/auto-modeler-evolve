import io
import json
import math
import re
import shutil
import sqlite3
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session, select

from chat.narration import (
    append_bot_message_to_conversation,
    narrate_data_insights_ai,
    narrate_upload,
)
from core.analyzer import analyze_dataframe, compute_full_profile, detect_time_columns
from core.chart_builder import build_correlation_heatmap, build_timeseries_chart
from core.merger import merge_datasets, suggest_join_keys
from core.query_engine import run_nl_query
from db import get_session
from models.dataset import Dataset
from models.project import Project

router = APIRouter(prefix="/api/data", tags=["data"])

UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"
SAMPLE_CSV = Path(__file__).parent.parent / "data" / "sample" / "sample_sales.csv"

_ACCEPTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
_ACCEPTED_MIME_TYPES = {
    "text/csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
}


def _load_df_from_path(path: Path) -> pd.DataFrame:
    """Load a DataFrame from CSV or Excel file."""
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path, engine="openpyxl")
    return pd.read_csv(path)


def _is_accepted_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in _ACCEPTED_EXTENSIONS


def _sanitize_rows(rows: list[dict]) -> list[dict[str, Any]]:
    """Replace NaN/inf floats with None so JSON serialization never crashes."""
    clean = []
    for row in rows:
        safe: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                safe[k] = None
            else:
                safe[k] = v
        clean.append(safe)
    return clean


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

@router.post("/upload", status_code=201)
def upload_csv(
    file: UploadFile,
    project_id: str = Form(...),
    session: Session = Depends(get_session),
):
    if not file.filename or not _is_accepted_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Only CSV and Excel files (.csv, .xlsx, .xls) are accepted",
        )

    project_upload_dir = UPLOAD_DIR / project_id
    project_upload_dir.mkdir(parents=True, exist_ok=True)

    original_path = project_upload_dir / file.filename
    contents = file.file.read()
    original_path.write_bytes(contents)

    try:
        ext = Path(file.filename).suffix.lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(original_path, engine="openpyxl")
            # Store as CSV so all downstream readers use a consistent format
            csv_filename = Path(file.filename).stem + ".csv"
            file_path = project_upload_dir / csv_filename
            df.to_csv(file_path, index=False)
            original_path.unlink(missing_ok=True)  # drop the xlsx; keep only CSV
            stored_filename = csv_filename
        else:
            df = pd.read_csv(original_path)
            file_path = original_path
            stored_filename = file.filename
    except Exception as exc:
        original_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400, detail=f"Failed to parse file: {exc}"
        ) from exc

    # Full profiling on upload (includes distributions, correlations, patterns)
    profile = compute_full_profile(df)

    dataset = Dataset(
        project_id=project_id,
        filename=stored_filename,
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

    preview_rows = _sanitize_rows(df.head(10).to_dict(orient="records"))

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

        # Follow up with a Claude-generated AI insight (best-effort, async-safe)
        dataset_summary = ", ".join(col_names[:8])
        profile_highlights = json.dumps(
            {k: profile[k] for k in ("patterns", "warnings", "correlations") if k in profile},
            default=str,
        )
        ai_insight = narrate_data_insights_ai(
            dataset_summary=dataset_summary,
            profile_highlights=profile_highlights,
            n_rows=dataset.row_count,
            n_cols=dataset.column_count,
        )
        if ai_insight:
            append_bot_message_to_conversation(project_id, ai_insight, session)
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
    preview_rows = _sanitize_rows(df.head(10).to_dict(orient="records"))
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
        preview_rows = _sanitize_rows(df_existing.head(10).to_dict(orient="records"))
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

    preview_rows = _sanitize_rows(df.head(10).to_dict(orient="records"))

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

        dataset_summary = ", ".join(col_names[:8])
        profile_highlights = json.dumps(
            {k: profile[k] for k in ("patterns", "warnings", "correlations") if k in profile},
            default=str,
        )
        ai_insight = narrate_data_insights_ai(
            dataset_summary=dataset_summary,
            profile_highlights=profile_highlights,
            n_rows=dataset.row_count,
            n_cols=dataset.column_count,
        )
        if ai_insight:
            append_bot_message_to_conversation(body.project_id, ai_insight, session)
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


@router.get("/{dataset_id}/correlations")
def get_correlations(dataset_id: str, session: Session = Depends(get_session)):
    """Return correlation matrix as a heatmap chart spec.

    Uses cached profile when available; falls back to computing on demand.
    Returns 200 with chart_spec=null when there are fewer than 2 numeric columns.
    """
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Try cached profile first
    correlations: dict = {}
    if dataset.profile:
        try:
            profile = json.loads(dataset.profile)
            correlations = profile.get("correlations", {})
        except Exception:  # noqa: BLE001
            pass

    # Recompute if missing
    if not correlations:
        file_path = Path(dataset.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset file not found on disk")
        df = pd.read_csv(file_path)
        profile = compute_full_profile(df)
        correlations = profile.get("correlations", {})

    columns = correlations.get("columns", [])
    matrix = correlations.get("matrix", [])

    if len(columns) < 2:
        return {
            "dataset_id": dataset_id,
            "chart_spec": None,
            "message": "Need at least 2 numeric columns to show a correlation matrix.",
        }

    heatmap = build_correlation_heatmap(matrix, columns)
    return {
        "dataset_id": dataset_id,
        "chart_spec": heatmap,
        "pairs": correlations.get("pairs", []),
    }


@router.get("/{dataset_id}/timeseries")
def get_timeseries(
    dataset_id: str,
    value_column: str | None = None,
    window: int = 7,
    session: Session = Depends(get_session),
):
    """Return a time-series decomposition chart for a dataset.

    Detects date columns automatically. If value_column is not specified,
    uses the first numeric column as the value series.

    Returns:
    - date_columns: list of detected date columns
    - value_columns: list of available numeric columns
    - chart_spec: line chart with original + rolling average + trend (or null if no date column)
    """
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    df = pd.read_csv(file_path)

    # Detect date columns
    date_cols = detect_time_columns(df)

    # Get numeric columns (excluding the date column)
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns.tolist()
        if c not in date_cols
    ]

    if not date_cols:
        return {
            "dataset_id": dataset_id,
            "date_columns": [],
            "value_columns": numeric_cols,
            "chart_spec": None,
            "message": "No date/time column detected. Time-series analysis requires a date column.",
        }

    if not numeric_cols:
        return {
            "dataset_id": dataset_id,
            "date_columns": date_cols,
            "value_columns": [],
            "chart_spec": None,
            "message": "No numeric columns available for time-series analysis.",
        }

    # Use specified value column or fall back to first numeric
    chosen_value_col = value_column if value_column in numeric_cols else numeric_cols[0]
    date_col = date_cols[0]

    # Parse date column and sort
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df_sorted = df.dropna(subset=[date_col]).sort_values(date_col)
    except Exception:  # noqa: BLE001
        df_sorted = df

    dates = df_sorted[date_col].astype(str).tolist()
    values = df_sorted[chosen_value_col].tolist()

    # Limit to 500 points to keep chart rendering fast
    if len(dates) > 500:
        step = len(dates) // 500
        dates = dates[::step]
        values = values[::step]

    chart_spec = build_timeseries_chart(dates, values, chosen_value_col, window=window)

    return {
        "dataset_id": dataset_id,
        "date_columns": date_cols,
        "value_columns": numeric_cols,
        "date_column": date_col,
        "value_column": chosen_value_col,
        "chart_spec": chart_spec,
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


# ---------------------------------------------------------------------------
# Multi-dataset support
# ---------------------------------------------------------------------------

@router.get("/project/{project_id}/datasets")
def list_project_datasets(project_id: str, session: Session = Depends(get_session)):
    """List all datasets uploaded to a project."""
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    datasets = session.exec(
        select(Dataset).where(Dataset.project_id == project_id)
    ).all()

    return [
        {
            "dataset_id": ds.id,
            "filename": ds.filename,
            "row_count": ds.row_count,
            "column_count": ds.column_count,
            "uploaded_at": ds.uploaded_at.isoformat(),
            "size_bytes": ds.size_bytes,
        }
        for ds in datasets
    ]


class JoinKeysRequest(BaseModel):
    dataset_id_1: str
    dataset_id_2: str


@router.post("/join-keys")
def get_join_key_suggestions(body: JoinKeysRequest, session: Session = Depends(get_session)):
    """Suggest candidate join keys for merging two datasets.

    Returns common columns ranked by uniqueness — the best join key candidates first.
    """
    ds1 = session.get(Dataset, body.dataset_id_1)
    ds2 = session.get(Dataset, body.dataset_id_2)
    if not ds1:
        raise HTTPException(status_code=404, detail=f"Dataset {body.dataset_id_1} not found")
    if not ds2:
        raise HTTPException(status_code=404, detail=f"Dataset {body.dataset_id_2} not found")

    path1, path2 = Path(ds1.file_path), Path(ds2.file_path)
    if not path1.exists():
        raise HTTPException(status_code=404, detail="Left dataset file not found on disk")
    if not path2.exists():
        raise HTTPException(status_code=404, detail="Right dataset file not found on disk")

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    suggestions = suggest_join_keys(df1, df2)

    return {
        "dataset_id_1": body.dataset_id_1,
        "dataset_id_2": body.dataset_id_2,
        "join_key_suggestions": suggestions,
        "common_column_count": len(suggestions),
    }


class MergeRequest(BaseModel):
    dataset_id_1: str
    dataset_id_2: str
    join_key: str
    how: str = "inner"  # inner | left | right | outer
    suffix_left: str = "_left"
    suffix_right: str = "_right"
    save_as_filename: str | None = None  # defaults to "{file1}+{file2}_merged.csv"


@router.post("/{project_id}/merge", status_code=201)
def merge_project_datasets(
    project_id: str,
    body: MergeRequest,
    session: Session = Depends(get_session),
):
    """Merge two datasets within a project on a shared join key.

    Creates a new Dataset record for the merged result.
    Returns preview + full column stats for the merged dataset.
    """
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    ds1 = session.get(Dataset, body.dataset_id_1)
    ds2 = session.get(Dataset, body.dataset_id_2)
    if not ds1:
        raise HTTPException(status_code=404, detail=f"Dataset {body.dataset_id_1} not found")
    if not ds2:
        raise HTTPException(status_code=404, detail=f"Dataset {body.dataset_id_2} not found")

    path1, path2 = Path(ds1.file_path), Path(ds2.file_path)
    if not path1.exists():
        raise HTTPException(status_code=404, detail="Left dataset file not found on disk")
    if not path2.exists():
        raise HTTPException(status_code=404, detail="Right dataset file not found on disk")

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    try:
        result = merge_datasets(
            df1,
            df2,
            join_key=body.join_key,
            how=body.how,
            suffix_left=body.suffix_left,
            suffix_right=body.suffix_right,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    merged_df: pd.DataFrame = result["merged_df"]

    # Persist merged CSV
    base1 = Path(ds1.filename).stem
    base2 = Path(ds2.filename).stem
    out_filename = body.save_as_filename or f"{base1}+{base2}_merged.csv"
    project_upload_dir = UPLOAD_DIR / project_id
    project_upload_dir.mkdir(parents=True, exist_ok=True)
    out_path = project_upload_dir / out_filename
    merged_df.to_csv(out_path, index=False)

    profile = compute_full_profile(merged_df)

    dataset = Dataset(
        project_id=project_id,
        filename=out_filename,
        file_path=str(out_path),
        row_count=profile["row_count"],
        column_count=profile["column_count"],
        columns=json.dumps(profile["columns"]),
        profile=json.dumps(profile),
        size_bytes=out_path.stat().st_size,
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)

    return {
        "dataset_id": dataset.id,
        "filename": dataset.filename,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "join_key": body.join_key,
        "how": body.how,
        "conflict_columns": result["conflict_columns"],
        "preview": result["preview_rows"],
        "column_stats": profile["columns"],
    }


# ---------------------------------------------------------------------------
# URL import (Google Sheets + generic CSV URLs)
# ---------------------------------------------------------------------------

_GSHEETS_PATTERN = re.compile(
    r"https://docs\.google\.com/spreadsheets/d/([A-Za-z0-9_\-]+)"
)
_GSHEETS_GID_PATTERN = re.compile(r"gid=([0-9]+)")


def _sheets_to_csv_url(url: str) -> str:
    """Convert a Google Sheets share/edit URL to a direct CSV export URL.

    Preserves the gid (tab ID) if present so multi-sheet workbooks are
    imported from the correct tab.
    """
    match = _GSHEETS_PATTERN.search(url)
    if not match:
        raise ValueError("URL does not look like a Google Sheets link")
    sheet_id = match.group(1)
    gid_match = _GSHEETS_GID_PATTERN.search(url)
    export_url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    )
    if gid_match:
        export_url += f"&gid={gid_match.group(1)}"
    return export_url


def _is_google_sheets_url(url: str) -> bool:
    return bool(_GSHEETS_PATTERN.search(url))


class UrlImportRequest(BaseModel):
    url: str
    project_id: str
    filename: str | None = None  # optional override; defaults to derived from URL


@router.post("/upload-url", status_code=201)
def upload_from_url(body: UrlImportRequest, session: Session = Depends(get_session)):
    """Import a dataset from a URL (Google Sheets public link or direct CSV URL).

    For Google Sheets: the sheet must be shared as "Anyone with the link can view".
    Converts Google Sheets URLs to direct CSV export URLs automatically.
    For other URLs: fetches directly and expects CSV content.
    """
    project = session.get(Project, body.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    url = body.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    # Resolve the actual download URL
    download_url = url
    source_label = "URL"
    if _is_google_sheets_url(url):
        try:
            download_url = _sheets_to_csv_url(url)
            source_label = "Google Sheets"
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Download the CSV content
    try:
        req = urllib.request.Request(
            download_url,
            headers={"User-Agent": "AutoModeler/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw_bytes = resp.read()
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch data from {source_label}: {exc}",
        ) from exc

    # Derive a filename
    if body.filename:
        stored_filename = body.filename if body.filename.endswith(".csv") else body.filename + ".csv"
    elif _is_google_sheets_url(url):
        m = _GSHEETS_PATTERN.search(url)
        stored_filename = f"sheets_{m.group(1)[:12]}.csv" if m else "sheets_import.csv"
    else:
        # Use the last path segment of the URL, fall back to "import.csv"
        path_part = url.rstrip("/").split("/")[-1].split("?")[0]
        stored_filename = path_part if path_part.endswith(".csv") else "import.csv"

    # Parse as CSV
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Downloaded content could not be parsed as CSV: {exc}"
        ) from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="Imported dataset is empty")

    # Save to filesystem
    project_upload_dir = UPLOAD_DIR / body.project_id
    project_upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = project_upload_dir / stored_filename
    df.to_csv(file_path, index=False)

    profile = compute_full_profile(df)

    dataset = Dataset(
        project_id=body.project_id,
        filename=stored_filename,
        file_path=str(file_path),
        row_count=profile["row_count"],
        column_count=profile["column_count"],
        columns=json.dumps(profile["columns"]),
        profile=json.dumps(profile),
        size_bytes=file_path.stat().st_size,
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)

    preview_rows = _sanitize_rows(df.head(10).to_dict(orient="records"))

    # Proactive narration (best-effort)
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
        dataset_summary = ", ".join(col_names[:8])
        profile_highlights = json.dumps(
            {k: profile[k] for k in ("patterns", "warnings", "correlations") if k in profile},
            default=str,
        )
        ai_insight = narrate_data_insights_ai(
            dataset_summary=dataset_summary,
            profile_highlights=profile_highlights,
            n_rows=dataset.row_count,
            n_cols=dataset.column_count,
        )
        if ai_insight:
            append_bot_message_to_conversation(body.project_id, ai_insight, session)
    except Exception:  # noqa: BLE001
        pass

    return {
        "dataset_id": dataset.id,
        "filename": dataset.filename,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "preview": preview_rows,
        "column_stats": profile["columns"],
        "insights": profile.get("insights", []),
        "source": source_label,
    }


# ---------------------------------------------------------------------------
# SQLite database connector
# ---------------------------------------------------------------------------
# Allows analysts to upload a .db file, browse its tables, and extract
# any table (or a custom SELECT query) as a Dataset — same pipeline as CSV.

_DB_UPLOADS_DIR = UPLOAD_DIR.parent / "db_uploads"


@router.post("/upload-db", status_code=201)
async def upload_sqlite_db(
    project_id: str = Form(...),
    file: UploadFile = Form(...),
    session: Session = Depends(get_session),
):
    """Upload a SQLite database file (.db or .sqlite).

    Returns a list of tables in the database so the user can choose which
    table to extract as a Dataset.  The file is stored temporarily under
    data/db_uploads/{project_id}/.
    """
    project = session.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    filename = file.filename or "database.db"
    ext = Path(filename).suffix.lower()
    if ext not in (".db", ".sqlite", ".sqlite3"):
        raise HTTPException(
            status_code=400,
            detail="Only SQLite database files (.db, .sqlite, .sqlite3) are accepted",
        )

    db_dir = _DB_UPLOADS_DIR / project_id
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / filename
    db_path.write_bytes(await file.read())

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
    except sqlite3.DatabaseError as exc:
        db_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400, detail=f"File is not a valid SQLite database: {exc}"
        ) from exc

    if not tables:
        db_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Database contains no tables")

    return {
        "project_id": project_id,
        "db_filename": filename,
        "db_path": str(db_path),
        "tables": tables,
        "table_count": len(tables),
    }


class DbExtractRequest(BaseModel):
    project_id: str
    db_path: str   # path returned by upload-db
    table_name: str
    query: str | None = None  # optional SQL override (must be SELECT)
    save_as_filename: str | None = None


@router.post("/extract-db", status_code=201)
def extract_db_table(body: DbExtractRequest, session: Session = Depends(get_session)):
    """Extract a table (or custom SELECT query) from an uploaded SQLite database.

    Creates a new Dataset record with the extracted data — identical to
    uploading a CSV. The query must be a SELECT statement for safety.
    """
    project = session.get(Project, body.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    db_path = Path(body.db_path)
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database file not found. Upload it first.")

    # Validate query is read-only
    query = body.query or f"SELECT * FROM [{body.table_name}]"
    if not query.strip().upper().startswith("SELECT"):
        raise HTTPException(
            status_code=400,
            detail="Only SELECT queries are allowed for database extraction",
        )

    try:
        conn = sqlite3.connect(str(db_path))
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=400, detail=f"Query failed: {exc}"
        ) from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="Query returned no rows")

    # Persist as CSV so downstream pipeline is unchanged
    out_filename = body.save_as_filename or f"{body.table_name}.csv"
    if not out_filename.endswith(".csv"):
        out_filename += ".csv"
    project_upload_dir = UPLOAD_DIR / body.project_id
    project_upload_dir.mkdir(parents=True, exist_ok=True)
    out_path = project_upload_dir / out_filename
    df.to_csv(out_path, index=False)

    profile = compute_full_profile(df)

    dataset = Dataset(
        project_id=body.project_id,
        filename=out_filename,
        file_path=str(out_path),
        row_count=profile["row_count"],
        column_count=profile["column_count"],
        columns=json.dumps(profile["columns"]),
        profile=json.dumps(profile),
        size_bytes=out_path.stat().st_size,
    )
    session.add(dataset)
    session.commit()
    session.refresh(dataset)

    preview_rows = _sanitize_rows(df.head(10).to_dict(orient="records"))

    # Proactive narration (best-effort)
    try:
        col_names = [c["name"] for c in profile["columns"]] if profile.get("columns") else list(df.columns)
        narration = narrate_upload(
            filename=out_filename,
            row_count=dataset.row_count,
            col_count=dataset.column_count,
            insights=profile.get("insights"),
            column_names=col_names,
        )
        append_bot_message_to_conversation(body.project_id, narration, session)
    except Exception:  # noqa: BLE001
        pass

    return {
        "dataset_id": dataset.id,
        "filename": dataset.filename,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "table_name": body.table_name,
        "query": query,
        "preview": preview_rows,
        "column_stats": profile["columns"],
        "insights": profile.get("insights", []),
        "source": "SQLite",
    }
