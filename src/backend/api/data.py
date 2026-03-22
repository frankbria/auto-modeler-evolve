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
from core.analyzer import (
    analyze_target_correlations,
    compare_segments,
    compute_full_profile,
    compute_group_stats,
    detect_time_columns,
)
from core.anomaly import detect_anomalies
from core.cleaner import (
    cap_outliers,
    drop_column,
    fill_missing,
    filter_rows,
    remove_duplicates,
)
from core.chart_builder import (
    build_boxplot,
    build_correlation_heatmap,
    build_crosstab,
    build_timeseries_chart,
)
from core.forecaster import detect_time_series, forecast_next_periods
from core.readiness import compute_data_readiness
from core.computed import add_computed_column
from core.dictionary import generate_dictionary
from core.merger import merge_datasets, suggest_join_keys
from core.query_engine import run_nl_query
from db import get_session
from models.dataset import Dataset
from models.dataset_filter import DatasetFilter
from models.feature_set import FeatureSet
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
        col_names = (
            [c["name"] for c in profile["columns"]]
            if profile.get("columns")
            else list(df.columns)
        )
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
            {
                k: profile[k]
                for k in ("patterns", "warnings", "correlations")
                if k in profile
            },
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
def load_sample_dataset(
    body: SampleLoadRequest, session: Session = Depends(get_session)
):
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
        raise HTTPException(
            status_code=500, detail="Sample dataset not found on server"
        )

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
        col_names = (
            [c["name"] for c in profile["columns"]]
            if profile.get("columns")
            else list(df.columns)
        )
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
            {
                k: profile[k]
                for k in ("patterns", "warnings", "correlations")
                if k in profile
            },
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
            raise HTTPException(
                status_code=404, detail="Dataset file not found on disk"
            )
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
        c
        for c in df.select_dtypes(include="number").columns.tolist()
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


@router.get("/{dataset_id}/forecast")
def get_forecast(
    dataset_id: str,
    target: str | None = None,
    periods: int = 6,
    session: Session = Depends(get_session),
):
    """Return a time-series forecast for the next *periods* time steps.

    Automatically detects the date column. If *target* is not provided, uses
    the first numeric column. Returns a ForecastResult dict including
    historical points, forecasted points with 95% prediction intervals, trend
    direction, and a plain-English summary.

    Returns 404 when no time-series structure is detected or the dataset file
    is missing. Returns 400 when forecast parameters are invalid.
    """
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    df = pd.read_csv(file_path)

    ts_info = detect_time_series(df)
    if not ts_info:
        raise HTTPException(
            status_code=404,
            detail=(
                "No time-series structure detected. "
                "This dataset needs a date column and at least one numeric column."
            ),
        )

    date_col = ts_info["date_col"]
    value_cols = ts_info["value_cols"]

    # Resolve target column
    if target and target in value_cols:
        value_col = target
    elif target and target not in value_cols:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{target}' not found or not numeric. Available: {value_cols}",
        )
    else:
        value_col = value_cols[0]

    # Clamp periods
    if periods < 1 or periods > 24:
        raise HTTPException(
            status_code=400,
            detail="periods must be between 1 and 24",
        )

    try:
        result = forecast_next_periods(df, date_col, value_col, periods=periods)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "dataset_id": dataset_id,
        "date_columns": [date_col],
        "value_columns": value_cols,
        "forecast": result,
    }


@router.get("/{dataset_id}/readiness-check")
def get_readiness_check(
    dataset_id: str,
    target: str | None = None,
    session: Session = Depends(get_session),
):
    """Return a data-readiness assessment for the dataset.

    Scores the dataset across 5 components (row count, missing values,
    duplicate rows, feature diversity, data type quality) and returns an
    overall 0-100 score, letter grade, status, per-component details, and
    actionable recommendations.

    Optionally pass `target` (target column name) for an additional class-
    balance advisory check.
    """
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    df = pd.read_csv(file_path)

    result = compute_data_readiness(df, target_col=target)
    return {"dataset_id": dataset_id, **result}


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
def get_join_key_suggestions(
    body: JoinKeysRequest, session: Session = Depends(get_session)
):
    """Suggest candidate join keys for merging two datasets.

    Returns common columns ranked by uniqueness — the best join key candidates first.
    """
    ds1 = session.get(Dataset, body.dataset_id_1)
    ds2 = session.get(Dataset, body.dataset_id_2)
    if not ds1:
        raise HTTPException(
            status_code=404, detail=f"Dataset {body.dataset_id_1} not found"
        )
    if not ds2:
        raise HTTPException(
            status_code=404, detail=f"Dataset {body.dataset_id_2} not found"
        )

    path1, path2 = Path(ds1.file_path), Path(ds2.file_path)
    if not path1.exists():
        raise HTTPException(
            status_code=404, detail="Left dataset file not found on disk"
        )
    if not path2.exists():
        raise HTTPException(
            status_code=404, detail="Right dataset file not found on disk"
        )

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
        raise HTTPException(
            status_code=404, detail=f"Dataset {body.dataset_id_1} not found"
        )
    if not ds2:
        raise HTTPException(
            status_code=404, detail=f"Dataset {body.dataset_id_2} not found"
        )

    path1, path2 = Path(ds1.file_path), Path(ds2.file_path)
    if not path1.exists():
        raise HTTPException(
            status_code=404, detail="Left dataset file not found on disk"
        )
    if not path2.exists():
        raise HTTPException(
            status_code=404, detail="Right dataset file not found on disk"
        )

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
    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
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
        raise HTTPException(
            status_code=400, detail="URL must start with http:// or https://"
        )

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
        stored_filename = (
            body.filename if body.filename.endswith(".csv") else body.filename + ".csv"
        )
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
            status_code=400,
            detail=f"Downloaded content could not be parsed as CSV: {exc}",
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
        col_names = (
            [c["name"] for c in profile["columns"]]
            if profile.get("columns")
            else list(df.columns)
        )
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
            {
                k: profile[k]
                for k in ("patterns", "warnings", "correlations")
                if k in profile
            },
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
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
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
    db_path: str  # path returned by upload-db
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
        raise HTTPException(
            status_code=404, detail="Database file not found. Upload it first."
        )

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
        raise HTTPException(status_code=400, detail=f"Query failed: {exc}") from exc

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
        col_names = (
            [c["name"] for c in profile["columns"]]
            if profile.get("columns")
            else list(df.columns)
        )
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


# ---------------------------------------------------------------------------
# Box plot — grouped distribution comparison
# ---------------------------------------------------------------------------


@router.get("/{dataset_id}/boxplot")
def get_boxplot(
    dataset_id: str,
    column: str,
    groupby: str | None = None,
    session: Session = Depends(get_session),
) -> dict:
    """Return a boxplot chart spec for a numeric column, optionally grouped.

    GET /api/data/{dataset_id}/boxplot?column=sales&groupby=region

    Returns a chart spec with chart_type="boxplot" and one box per group.
    When groupby is omitted, returns a single box for the whole column.
    """
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = pd.read_csv(dataset.file_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Could not read dataset: {exc}"
        ) from exc

    if column not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{column}' not found. Available: {list(df.columns)}",
        )

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise HTTPException(
            status_code=400,
            detail=f"Column '{column}' is not numeric. Box plots require a numeric column.",
        )

    if groupby is not None and groupby not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Group column '{groupby}' not found. Available: {list(df.columns)}",
        )

    chart_spec = build_boxplot(df, value_col=column, group_col=groupby)
    return chart_spec


# ---------------------------------------------------------------------------
# Anomaly detection — multi-dimensional outlier identification
# ---------------------------------------------------------------------------


class AnomalyRequest(BaseModel):
    features: list[str]
    contamination: float = 0.05
    n_top: int = 20


@router.post("/{dataset_id}/anomalies")
def run_anomaly_detection(
    dataset_id: str,
    body: AnomalyRequest,
    session: Session = Depends(get_session),
) -> dict:
    """Detect anomalous rows using IsolationForest across selected features.

    Unlike per-column outlier detection, this finds rows that are unusual
    *across multiple features simultaneously* — e.g., low price + high quantity
    in the "Premium" category.

    Args:
        features: Numeric column names to consider (non-numeric are silently dropped).
        contamination: Expected fraction of anomalies (0.01–0.5, default 0.05).
        n_top: Number of top anomalies to return (default 20).

    Returns:
        {
            dataset_id, anomaly_count, total_rows, contamination_used,
            top_anomalies: [{row_index, anomaly_score, is_anomaly, values}],
            summary, features_used
        }
    """
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    if not body.features:
        raise HTTPException(
            status_code=400, detail="At least one feature must be specified"
        )

    df = _load_df_from_path(file_path)

    try:
        result = detect_anomalies(
            df,
            features=body.features,
            contamination=body.contamination,
            n_top=body.n_top,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"dataset_id": dataset_id, **result}


# ---------------------------------------------------------------------------
# Conversational data cleaning
# ---------------------------------------------------------------------------


class CleanRequest(BaseModel):
    """Body for POST /api/data/{dataset_id}/clean.

    operation: one of remove_duplicates | fill_missing | filter_rows |
               cap_outliers | drop_column
    column:    required for fill_missing, filter_rows, cap_outliers, drop_column
    strategy:  for fill_missing — mean | median | mode | zero | value
    fill_value: literal value when strategy="value"
    operator:  for filter_rows — gt | lt | eq | ne | gte | lte | contains | notcontains
    value:     for filter_rows — comparison value
    percentile: for cap_outliers — upper percentile (default 99.0)
    """

    operation: str
    column: str | None = None
    strategy: str | None = None  # fill_missing
    fill_value: float | str | None = None  # fill_missing strategy=value
    operator: str | None = None  # filter_rows
    value: float | str | None = None  # filter_rows
    percentile: float = 99.0  # cap_outliers


@router.post("/{dataset_id}/clean")
def clean_dataset(
    dataset_id: str,
    body: CleanRequest,
    session: Session = Depends(get_session),
) -> dict:
    """Apply a single data cleaning operation to a dataset in-place.

    The cleaned CSV is written back to disk and the Dataset record is updated
    (row_count, column_count, columns JSON, profile JSON).

    Supported operations
    --------------------
    remove_duplicates — no extra params needed
    fill_missing      — column + strategy (mean/median/mode/zero/value) + optional fill_value
    filter_rows       — column + operator (gt/lt/eq/ne/gte/lte/contains/notcontains) + value
    cap_outliers      — column + percentile (default 99.0)
    drop_column       — column

    Returns
    -------
    {
        dataset_id, operation_result: {...}, preview: [first 10 rows],
        updated_stats: {row_count, column_count, columns}
    }
    """
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")

    df = _load_df_from_path(file_path)

    try:
        op = body.operation
        if op == "remove_duplicates":
            cleaned_df, result = remove_duplicates(df)
        elif op == "fill_missing":
            if not body.column:
                raise ValueError("fill_missing requires 'column'.")
            if not body.strategy:
                raise ValueError(
                    "fill_missing requires 'strategy' (mean/median/mode/zero/value)."
                )
            cleaned_df, result = fill_missing(
                df,
                column=body.column,
                strategy=body.strategy,
                fill_value=body.fill_value,
            )
        elif op == "filter_rows":
            if not body.column:
                raise ValueError("filter_rows requires 'column'.")
            if not body.operator:
                raise ValueError("filter_rows requires 'operator'.")
            if body.value is None:
                raise ValueError("filter_rows requires 'value'.")
            cleaned_df, result = filter_rows(
                df, column=body.column, operator=body.operator, value=body.value
            )
        elif op == "cap_outliers":
            if not body.column:
                raise ValueError("cap_outliers requires 'column'.")
            cleaned_df, result = cap_outliers(
                df, column=body.column, percentile=body.percentile
            )
        elif op == "drop_column":
            if not body.column:
                raise ValueError("drop_column requires 'column'.")
            cleaned_df, result = drop_column(df, column=body.column)
        else:
            raise ValueError(
                f"Unknown operation '{op}'. Valid: remove_duplicates, fill_missing, "
                "filter_rows, cap_outliers, drop_column."
            )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Persist cleaned CSV back to the same path
    cleaned_df.to_csv(file_path, index=False)

    # Recompute profile and update Dataset record
    profile = compute_full_profile(cleaned_df)
    dataset.row_count = profile["row_count"]
    dataset.column_count = profile["column_count"]
    dataset.columns = json.dumps(profile["columns"])
    dataset.profile = json.dumps(profile)
    session.add(dataset)
    session.commit()
    session.refresh(dataset)

    preview = _sanitize_rows(cleaned_df.head(10).to_dict(orient="records"))

    return {
        "dataset_id": dataset_id,
        "operation_result": result,
        "preview": preview,
        "updated_stats": {
            "row_count": dataset.row_count,
            "column_count": dataset.column_count,
            "columns": profile["columns"],
        },
    }


# ---------------------------------------------------------------------------
# Dataset refresh — replace data in-place, keep dataset ID + model history
# ---------------------------------------------------------------------------


@router.post("/{dataset_id}/refresh")
def refresh_dataset(
    dataset_id: str,
    file: UploadFile,
    session: Session = Depends(get_session),
):
    """Replace an existing dataset's CSV with a new file, preserving the dataset ID.

    This keeps all foreign-key relationships intact (FeatureSets, ModelRuns,
    Deployments all still reference the same dataset_id).

    Column compatibility check:
    - new_columns: columns in new file that weren't in the old file
    - removed_columns: columns in old file missing from new file
    - feature_columns_missing: columns required by the active FeatureSet that
      are now absent — these must be present to retrain using the existing config
    - compatible: True when no required feature columns are missing

    Returns
    -------
    {
        dataset_id, filename, row_count, column_count,
        new_columns, removed_columns, feature_columns_missing, compatible,
        preview, column_stats
    }
    """
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not file.filename or not _is_accepted_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Only CSV and Excel files (.csv, .xlsx, .xls) are accepted",
        )

    # Load the new file into a DataFrame
    contents = file.file.read()
    try:
        ext = Path(file.filename).suffix.lower()
        if ext in (".xlsx", ".xls"):
            new_df = pd.read_excel(io.BytesIO(contents), engine="openpyxl")
        else:
            new_df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to parse file: {exc}"
        ) from exc

    if new_df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Identify column changes
    old_columns = (
        set(c["name"] for c in json.loads(dataset.columns))
        if dataset.columns
        else set()
    )
    new_columns_set = set(new_df.columns)
    added_columns = sorted(new_columns_set - old_columns)
    removed_columns = sorted(old_columns - new_columns_set)

    # Check against active FeatureSet (if one exists)
    feature_set = session.exec(
        select(FeatureSet)
        .where(FeatureSet.dataset_id == dataset_id, FeatureSet.is_active == True)  # noqa: E712
        .order_by(FeatureSet.created_at.desc())  # type: ignore[arg-type]
    ).first()

    feature_columns_missing: list[str] = []
    if feature_set and feature_set.column_mapping:
        required_cols = set(json.loads(feature_set.column_mapping).keys())
        if feature_set.target_column:
            required_cols.add(feature_set.target_column)
        feature_columns_missing = sorted(required_cols - new_columns_set)

    compatible = len(feature_columns_missing) == 0

    # Write new CSV to the existing path (preserves all downstream references)
    file_path = Path(dataset.file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Store as CSV regardless of input format (same as upload endpoint)
    new_df.to_csv(file_path, index=False)

    # Update filename if a CSV was provided with a new name
    stored_filename = dataset.filename  # keep old name by default
    if ext in (".xlsx", ".xls"):
        # Excel converted to CSV; keep existing CSV filename
        pass
    elif file.filename != dataset.filename:
        stored_filename = file.filename

    # Re-run profiling and update Dataset record
    profile = compute_full_profile(new_df)
    dataset.filename = stored_filename
    dataset.row_count = profile["row_count"]
    dataset.column_count = profile["column_count"]
    dataset.columns = json.dumps(profile["columns"])
    dataset.profile = json.dumps(profile)
    dataset.size_bytes = len(contents)
    session.add(dataset)
    session.commit()
    session.refresh(dataset)

    preview = _sanitize_rows(new_df.head(10).to_dict(orient="records"))

    return {
        "dataset_id": dataset_id,
        "filename": dataset.filename,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "new_columns": added_columns,
        "removed_columns": removed_columns,
        "feature_columns_missing": feature_columns_missing,
        "compatible": compatible,
        "preview": preview,
        "column_stats": profile["columns"],
    }


# ---------------------------------------------------------------------------
# Data Dictionary
# ---------------------------------------------------------------------------


@router.get("/{dataset_id}/dictionary")
def get_dictionary(dataset_id: str, session: Session = Depends(get_session)):
    """Return the stored data dictionary for the dataset.

    The dictionary is derived from the ``columns`` JSON stored on the Dataset
    record. If column descriptions have been generated before (i.e. a
    ``description`` key is present on the first column), those are returned
    as-is. Otherwise a static (rule-based) dictionary is generated on the fly
    without calling Claude.

    Returns::

        {
            "dataset_id": "...",
            "filename": "sales.csv",
            "generated": false,
            "columns": [
                {
                    "name": "revenue",
                    "dtype": "float64",
                    "col_type": "metric",
                    "description": "..."
                },
                ...
            ]
        }
    """
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not dataset.columns:
        return {
            "dataset_id": dataset_id,
            "filename": dataset.filename,
            "generated": False,
            "columns": [],
        }

    try:
        columns_data = json.loads(dataset.columns)
    except Exception:  # noqa: BLE001
        columns_data = []

    # Check whether AI descriptions are already stored
    already_generated = any("description" in c for c in columns_data)

    if not already_generated:
        # Generate static descriptions on the fly
        enriched = generate_dictionary(
            columns_data,
            filename=dataset.filename,
            row_count=dataset.row_count,
        )
    else:
        # Ensure col_type is present even if descriptions were generated by an
        # older version that might not have stored it.
        from core.dictionary import classify_column_type

        enriched = []
        for col in columns_data:
            if "col_type" not in col:
                col_type = classify_column_type(
                    col_name=col["name"],
                    dtype=col["dtype"],
                    unique_count=col.get("unique_count", 0),
                    row_count=dataset.row_count or 1,
                    sample_values=col.get("sample_values", []),
                    null_pct=col.get("null_pct", 0.0),
                )
                enriched.append({**col, "col_type": col_type})
            else:
                enriched.append(col)

    return {
        "dataset_id": dataset_id,
        "filename": dataset.filename,
        "generated": already_generated,
        "columns": enriched,
    }


@router.post("/{dataset_id}/dictionary", status_code=200)
def generate_dataset_dictionary(
    dataset_id: str,
    session: Session = Depends(get_session),
):
    """Generate (or regenerate) AI-powered column descriptions.

    Calls Claude when ANTHROPIC_API_KEY is present; falls back to static
    rule-based descriptions. The generated descriptions are persisted back
    into ``Dataset.columns`` so subsequent GET requests return them instantly.

    Returns the same shape as GET /dictionary with ``generated=true``.
    """
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not dataset.columns:
        raise HTTPException(
            status_code=400, detail="Dataset has no column data to describe"
        )

    try:
        columns_data = json.loads(dataset.columns)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail="Failed to parse column data"
        ) from exc

    # Generate (possibly with Claude)
    enriched = generate_dictionary(
        columns_data,
        filename=dataset.filename,
        row_count=dataset.row_count,
    )

    # Persist enriched columns back to the dataset record
    dataset.columns = json.dumps(enriched)
    session.add(dataset)
    session.commit()

    return {
        "dataset_id": dataset_id,
        "filename": dataset.filename,
        "generated": True,
        "columns": enriched,
    }


# ---------------------------------------------------------------------------
# Cross-tabulation / pivot table
# ---------------------------------------------------------------------------


@router.get("/{dataset_id}/crosstab")
def get_crosstab(
    dataset_id: str,
    rows: str,
    cols: str,
    values: str | None = None,
    agg: str = "sum",
    session: Session = Depends(get_session),
) -> dict:
    """Return a pivot-table (cross-tabulation) for two categorical columns.

    GET /api/data/{dataset_id}/crosstab?rows=region&cols=product&values=revenue&agg=sum

    Parameters
    ----------
    rows:   Column name for the row dimension (e.g. "region")
    cols:   Column name for the column dimension (e.g. "product_category")
    values: Column to aggregate (e.g. "revenue"). If omitted, counts rows.
    agg:    Aggregation function — sum | mean | count | min | max (default: sum)
    """
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = pd.read_csv(dataset.file_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Could not read dataset: {exc}"
        ) from exc

    try:
        result = build_crosstab(
            df, row_col=rows, col_col=cols, value_col=values, agg_func=agg
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return result


# ---------------------------------------------------------------------------
# Computed columns — add derived columns through conversation
# ---------------------------------------------------------------------------


class ComputeRequest(BaseModel):
    """Body for POST /api/data/{dataset_id}/compute."""

    name: str
    expression: str


@router.post("/{dataset_id}/compute")
def compute_column(
    dataset_id: str,
    body: ComputeRequest,
    session: Session = Depends(get_session),
) -> dict:
    """Evaluate *expression* and add the result as a new column named *name*.

    The updated CSV is written back to disk and the Dataset record is refreshed.

    POST /api/data/{dataset_id}/compute
    Body: {"name": "margin", "expression": "revenue / cost"}
    """
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Could not read dataset: {exc}"
        ) from exc

    try:
        updated_df, result = add_computed_column(df, body.name, body.expression)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Persist updated CSV back to the same path
    updated_df.to_csv(file_path, index=False)

    # Recompute profile and update Dataset record
    profile = compute_full_profile(updated_df)
    cols_meta = [
        {
            "name": col,
            "dtype": str(updated_df[col].dtype),
        }
        for col in updated_df.columns
    ]
    dataset.row_count = len(updated_df)
    dataset.column_count = len(updated_df.columns)
    dataset.columns = json.dumps(cols_meta)
    dataset.profile = json.dumps(profile)
    session.add(dataset)
    session.commit()

    preview = _sanitize_rows(updated_df.head(10).to_dict(orient="records"))

    return {
        "dataset_id": dataset_id,
        "compute_result": result,
        "preview": preview,
        "updated_stats": {
            "row_count": len(updated_df),
            "column_count": len(updated_df.columns),
        },
    }


@router.get("/{dataset_id}/compare-segments")
def compare_dataset_segments(
    dataset_id: str,
    col: str,
    val1: str,
    val2: str,
    session: Session = Depends(get_session),
) -> dict:
    """Compare two segments of a dataset on all numeric columns.

    GET /api/data/{dataset_id}/compare-segments?col=region&val1=East&val2=West
    Returns per-column stats (mean/std/median/count) for each segment plus
    effect sizes and a plain-English summary.
    """
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Could not read dataset: {exc}"
        ) from exc

    if col not in df.columns:
        raise HTTPException(
            status_code=400, detail=f"Column '{col}' not found in dataset"
        )

    col_values_lower = df[col].astype(str).str.strip().str.lower().unique().tolist()
    if val1.strip().lower() not in col_values_lower:
        raise HTTPException(
            status_code=400, detail=f"Value '{val1}' not found in column '{col}'"
        )
    if val2.strip().lower() not in col_values_lower:
        raise HTTPException(
            status_code=400, detail=f"Value '{val2}' not found in column '{col}'"
        )

    result = compare_segments(df, col, val1, val2)
    return result


@router.get("/{dataset_id}/target-correlations")
def get_target_correlations(
    dataset_id: str,
    target: str,
    top_n: int = 10,
    session: Session = Depends(get_session),
):
    """Return Pearson correlations between a target column and all other numeric columns.

    Sorted by absolute correlation value (strongest first). Use this to answer
    "what's correlated with revenue?" or "what drives profit?" questions.

    Returns 400 if target column is not found or not numeric.
    """
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Could not read dataset: {exc}"
        ) from exc

    result = analyze_target_correlations(df, target, top_n=max(1, min(top_n, 50)))

    if result.get("error") in ("column_not_found", "not_numeric"):
        raise HTTPException(status_code=400, detail=result["summary"])

    return {"dataset_id": dataset_id, **result}


@router.get("/{dataset_id}/group-stats")
def get_group_stats(
    dataset_id: str,
    group_by: str,
    metrics: str | None = None,
    agg: str = "sum",
    session: Session = Depends(get_session),
):
    """Aggregate numeric columns grouped by a categorical column.

    Query parameters
    ----------------
    group_by : Column to group by (required).
    metrics  : Comma-separated list of numeric columns to aggregate.
               Defaults to all numeric columns.
    agg      : Aggregation function — sum | mean | count | min | max | median.
               Defaults to 'sum'.

    Returns 400 if group_by column is not found or the aggregation fails.
    """
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Could not read dataset: {exc}"
        ) from exc

    value_cols: list[str] | None = None
    if metrics:
        value_cols = [c.strip() for c in metrics.split(",") if c.strip()]

    result = compute_group_stats(df, group_by, value_cols=value_cols, agg=agg)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return {"dataset_id": dataset_id, **result}


# ---------------------------------------------------------------------------
# Column rename
# ---------------------------------------------------------------------------


class RenameColumnRequest(BaseModel):
    old_name: str
    new_name: str


@router.post("/{dataset_id}/rename-column")
def rename_column(
    dataset_id: str,
    body: RenameColumnRequest,
    session: Session = Depends(get_session),
):
    """Rename a column in the dataset CSV and update the stored profile.

    Validates that:
    - The dataset and file exist.
    - old_name is a real column.
    - new_name is non-empty, doesn't already exist, and contains only word
      characters (letters, digits, underscores).

    Returns the old name, new name, and updated row count.
    """
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500, detail=f"Could not read dataset: {exc}"
        ) from exc

    columns = list(df.columns)

    # Validate old name exists
    if body.old_name not in columns:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{body.old_name}' not found. Available: {', '.join(columns)}",
        )

    # Validate new name
    new_name = body.new_name.strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="New column name cannot be empty")
    if not re.match(r"^\w+$", new_name):
        raise HTTPException(
            status_code=400,
            detail="New column name may only contain letters, digits, and underscores",
        )
    if new_name in columns and new_name != body.old_name:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{new_name}' already exists",
        )

    # Apply rename and write back
    df = df.rename(columns={body.old_name: new_name})
    df.to_csv(file_path, index=False)

    # Recompute profile and update Dataset record
    profile = compute_full_profile(df)
    dataset.profile = json.dumps(profile, default=str)
    dataset.columns = json.dumps(profile["columns"])
    dataset.column_count = len(df.columns)
    session.add(dataset)
    session.commit()

    return {
        "dataset_id": dataset_id,
        "old_name": body.old_name,
        "new_name": new_name,
        "row_count": len(df),
        "column_count": len(df.columns),
    }


# ---------------------------------------------------------------------------
# Automated data story
# ---------------------------------------------------------------------------


@router.get("/{dataset_id}/story")
def get_data_story(
    dataset_id: str,
    target: str | None = None,
    session: Session = Depends(get_session),
):
    """Orchestrate a comprehensive data analysis narrative.

    Runs readiness, group-by, target correlations (if target provided), and
    anomaly detection — combines them into a single DataStory response.
    """
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    df = pd.read_csv(file_path)

    from core.storyteller import generate_data_story

    return generate_data_story(
        df,
        dataset_id=dataset_id,
        target_col=target,
        dataset_filename=dataset.filename,
    )


# ---------------------------------------------------------------------------
# Non-destructive data filter
# ---------------------------------------------------------------------------


class FilterRequest(BaseModel):
    conditions: list[dict]  # [{column, operator, value}]


@router.post("/{dataset_id}/set-filter")
def set_dataset_filter(
    dataset_id: str,
    body: FilterRequest,
    session: Session = Depends(get_session),
):
    """Set the active non-destructive filter for a dataset.

    Overwrites any existing filter. Conditions are AND-ed together.
    Returns filter summary with row counts.
    """
    from core.filter_view import (
        apply_active_filter,
        build_filter_summary,
        validate_filter_conditions,
    )

    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = Path(dataset.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file not found")

    df = pd.read_csv(file_path)
    df_columns = list(df.columns)

    errors = validate_filter_conditions(body.conditions, df_columns)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    filtered_df = apply_active_filter(df, body.conditions)
    summary = build_filter_summary(body.conditions)

    # Upsert: replace existing filter for this dataset
    existing = session.exec(
        select(DatasetFilter).where(DatasetFilter.dataset_id == dataset_id)
    ).first()
    if existing:
        session.delete(existing)
        session.commit()

    new_filter = DatasetFilter(
        dataset_id=dataset_id,
        conditions=json.dumps(body.conditions),
        filter_summary=summary,
        original_rows=len(df),
        filtered_rows=len(filtered_df),
    )
    session.add(new_filter)
    session.commit()
    session.refresh(new_filter)

    return {
        "dataset_id": dataset_id,
        "filter_summary": summary,
        "conditions": body.conditions,
        "original_rows": len(df),
        "filtered_rows": len(filtered_df),
        "row_reduction_pct": round((1 - len(filtered_df) / max(len(df), 1)) * 100, 1),
    }


@router.delete("/{dataset_id}/clear-filter")
def clear_dataset_filter(
    dataset_id: str,
    session: Session = Depends(get_session),
):
    """Remove the active filter for a dataset, restoring full dataset view."""
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    existing = session.exec(
        select(DatasetFilter).where(DatasetFilter.dataset_id == dataset_id)
    ).first()
    if existing:
        session.delete(existing)
        session.commit()

    return {"dataset_id": dataset_id, "cleared": True}


@router.get("/{dataset_id}/active-filter")
def get_active_filter(
    dataset_id: str,
    session: Session = Depends(get_session),
):
    """Get the active filter for a dataset, or null if no filter is set."""
    dataset = session.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    active = session.exec(
        select(DatasetFilter).where(DatasetFilter.dataset_id == dataset_id)
    ).first()
    if not active:
        return {"dataset_id": dataset_id, "active": False}

    return {
        "dataset_id": dataset_id,
        "active": True,
        "filter_summary": active.filter_summary,
        "conditions": json.loads(active.conditions),
        "original_rows": active.original_rows,
        "filtered_rows": active.filtered_rows,
        "row_reduction_pct": active.row_reduction_pct,
    }
