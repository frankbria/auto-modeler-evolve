"""Tests for CSV upload and data preview."""

import pytest
import io
from httpx import AsyncClient, ASGITransport
from sqlmodel import create_engine, SQLModel
import db as db_module

SAMPLE_CSV = b"""date,product,region,revenue,units
2024-01-01,Widget A,North,1200.50,10
2024-01-01,Widget B,South,850.00,8
2024-01-02,Widget A,East,2100.75,18
2024-01-02,Widget C,West,450.25,4
2024-01-03,Widget B,North,1650.00,15
"""


@pytest.fixture
async def ac(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    # Import models so their tables are registered with metadata
    import models.project  # noqa
    import models.dataset  # noqa
    import models.conversation  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    # Point upload dir to tmp_path
    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
async def project_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Test Project"})
    return resp.json()["id"]


async def test_upload_valid_csv(ac, project_id):
    resp = await ac.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["row_count"] == 5
    assert data["column_count"] == 5
    assert "dataset_id" in data
    assert len(data["preview"]) == 5
    assert len(data["column_stats"]) == 5


async def test_upload_non_csv_rejected(ac, project_id):
    resp = await ac.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.xlsx", b"fake excel data", "application/vnd.ms-excel")},
    )
    assert resp.status_code == 400


async def test_preview_endpoint(ac, project_id):
    upload_resp = await ac.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    assert upload_resp.status_code == 201, upload_resp.text
    dataset_id = upload_resp.json()["dataset_id"]

    preview_resp = await ac.get(f"/api/data/{dataset_id}/preview")
    assert preview_resp.status_code == 200
    data = preview_resp.json()
    assert "column_stats" in data
    assert "preview" in data
    col_names = [c["name"] for c in data["column_stats"]]
    assert "revenue" in col_names
    assert "product" in col_names


async def test_column_stats_numeric(ac, project_id):
    """Numeric columns should have min, max, mean stats."""
    resp = await ac.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    revenue_col = next(c for c in data["column_stats"] if c["name"] == "revenue")
    assert revenue_col["min"] is not None
    assert revenue_col["max"] is not None
    assert revenue_col["mean"] is not None
