"""Tests for Excel (.xlsx/.xls) file upload support."""
import io
import pytest
import pandas as pd
from httpx import AsyncClient, ASGITransport
from sqlmodel import create_engine, SQLModel
import db as db_module


def _make_xlsx_bytes(data: dict) -> bytes:
    """Create an in-memory xlsx file from a dict of {col: [values]}."""
    df = pd.DataFrame(data)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf.read()


SAMPLE_DATA = {
    "product": ["Widget A", "Widget B", "Widget C", "Widget D", "Widget E"],
    "region": ["North", "South", "East", "West", "North"],
    "revenue": [1200.50, 850.00, 2100.75, 450.25, 1650.00],
    "units": [10, 8, 18, 4, 15],
}


@pytest.fixture
async def ac(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.project  # noqa
    import models.dataset  # noqa
    import models.conversation  # noqa
    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    data_module.UPLOAD_DIR = tmp_path / "uploads"

    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.fixture
async def project_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Excel Test Project"})
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# xlsx upload tests
# ---------------------------------------------------------------------------

async def test_upload_xlsx_returns_201(ac, project_id):
    """xlsx file is accepted and returns the same shape as CSV upload."""
    xlsx_bytes = _make_xlsx_bytes(SAMPLE_DATA)
    resp = await ac.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales_data.xlsx", io.BytesIO(xlsx_bytes),
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["row_count"] == 5
    assert data["column_count"] == 4
    assert "dataset_id" in data
    assert len(data["preview"]) == 5


async def test_upload_xlsx_stored_as_csv(ac, project_id, tmp_path):
    """After uploading an xlsx, the stored filename ends in .csv."""
    xlsx_bytes = _make_xlsx_bytes(SAMPLE_DATA)
    resp = await ac.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("report.xlsx", io.BytesIO(xlsx_bytes),
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert resp.status_code == 201
    data = resp.json()
    # stored filename should be the .csv version
    assert data["filename"].endswith(".csv"), f"Expected .csv filename, got {data['filename']}"


async def test_upload_xlsx_columns_preserved(ac, project_id):
    """Column names from the Excel sheet are preserved correctly."""
    xlsx_bytes = _make_xlsx_bytes(SAMPLE_DATA)
    resp = await ac.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.xlsx", io.BytesIO(xlsx_bytes),
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert resp.status_code == 201
    data = resp.json()
    col_names = [c["name"] for c in data["column_stats"]]
    assert "product" in col_names
    assert "revenue" in col_names
    assert "units" in col_names


async def test_upload_xlsx_preview_readable_after_upload(ac, project_id):
    """Preview endpoint works on a dataset created from an xlsx upload."""
    xlsx_bytes = _make_xlsx_bytes(SAMPLE_DATA)
    upload_resp = await ac.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.xlsx", io.BytesIO(xlsx_bytes),
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert upload_resp.status_code == 201
    dataset_id = upload_resp.json()["dataset_id"]

    preview_resp = await ac.get(f"/api/data/{dataset_id}/preview")
    assert preview_resp.status_code == 200
    preview = preview_resp.json()
    assert preview["row_count"] == 5
    assert len(preview["preview"]) == 5


async def test_upload_xlsx_profile_endpoint_works(ac, project_id):
    """Full profile endpoint works for xlsx-sourced datasets."""
    xlsx_bytes = _make_xlsx_bytes(SAMPLE_DATA)
    upload_resp = await ac.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.xlsx", io.BytesIO(xlsx_bytes),
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    dataset_id = upload_resp.json()["dataset_id"]

    profile_resp = await ac.get(f"/api/data/{dataset_id}/profile")
    assert profile_resp.status_code == 200
    profile = profile_resp.json()
    assert profile["row_count"] == 5


async def test_upload_non_excel_non_csv_rejected(ac, project_id):
    """A .pdf file is rejected with 400."""
    resp = await ac.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("report.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    assert resp.status_code == 400
    assert "accepted" in resp.json()["detail"].lower()


async def test_upload_xlsx_accepted_extensions_message(ac, project_id):
    """Error message for rejected files mentions xlsx."""
    resp = await ac.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.json", io.BytesIO(b"{}"), "application/json")},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert ".xlsx" in detail or "Excel" in detail


async def test_upload_xlsx_with_numeric_and_string_columns(ac, project_id):
    """Mixed column types from xlsx are profiled correctly."""
    mixed_data = {
        "name": [f"item_{i}" for i in range(20)],
        "score": [float(i) * 1.5 for i in range(20)],
        "category": ["A" if i % 2 == 0 else "B" for i in range(20)],
        "rank": list(range(1, 21)),
    }
    xlsx_bytes = _make_xlsx_bytes(mixed_data)
    resp = await ac.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("mixed.xlsx", io.BytesIO(xlsx_bytes),
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["row_count"] == 20
    assert data["column_count"] == 4
