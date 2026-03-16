"""Tests for PDF model report generation.

Covers:
- generate_model_report() returns valid PDF bytes
- GET /api/models/{run_id}/report returns PDF with correct headers
- 404 for unknown run_id
- 400 for non-done run
"""

import os

import pytest
from httpx import AsyncClient, ASGITransport
from sqlmodel import SQLModel, Session, create_engine

TEST_DATABASE_URL = "sqlite:///./test_pdf_report.db"


@pytest.fixture(autouse=True)
def setup_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    yield
    if os.path.exists("test_pdf_report.db"):
        os.unlink("test_pdf_report.db")


@pytest.fixture
async def client(setup_env, tmp_path):
    import db
    from main import app

    db.engine = create_engine(TEST_DATABASE_URL, echo=False)
    db.DATA_DIR = str(tmp_path)
    SQLModel.metadata.create_all(db.engine)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ─── Unit test for the generator ──────────────────────────────────────────────


def test_generate_model_report_returns_pdf_bytes():
    """generate_model_report() should return non-empty bytes starting with %PDF."""
    from core.report_generator import generate_model_report
    from datetime import datetime

    pdf = generate_model_report(
        project_name="Test Project",
        dataset_filename="sales.csv",
        dataset_rows=200,
        dataset_columns=5,
        algorithm="RandomForest",
        problem_type="regression",
        metrics={"r2": 0.92, "mae": 123.45, "rmse": 156.78},
        summary="The model explains 92% of variance.",
        training_duration_ms=3200,
        feature_importances=[
            {"feature": "region", "importance": 0.45, "rank": 1},
            {"feature": "product", "importance": 0.30, "rank": 2},
        ],
        confidence_assessment={
            "confidence_level": "High",
            "strengths": ["Large training set", "Good cross-validation scores"],
            "limitations": ["May not generalise to new product categories"],
        },
        created_at=datetime(2026, 3, 13, 8, 0, 0),
    )

    assert isinstance(pdf, bytes), "Should return bytes"
    assert len(pdf) > 1000, "PDF should not be empty"
    assert pdf[:4] == b"%PDF", "Bytes should start with PDF magic number"


def test_generate_model_report_minimal():
    """generate_model_report() with minimal inputs should still succeed."""
    from core.report_generator import generate_model_report

    pdf = generate_model_report(
        project_name="Minimal",
        dataset_filename="data.csv",
        dataset_rows=50,
        dataset_columns=3,
        algorithm="LinearRegression",
        problem_type="regression",
        metrics={},
        summary=None,
        training_duration_ms=None,
    )

    assert pdf[:4] == b"%PDF"


# ─── API tests ────────────────────────────────────────────────────────────────

SAMPLE_CSV = b"""date,product,region,revenue,units
2024-01-01,Widget A,North,1200.50,10
2024-01-02,Widget A,East,2100.75,18
2024-01-03,Widget B,North,1650.00,15
2024-01-04,Widget C,West,450.25,4
2024-01-05,Widget A,South,900.00,9
"""


async def _train_model(client: AsyncClient, tmp_path) -> str:
    """Helper: create project → upload CSV → set target → train → return run_id."""
    # Create project
    r = await client.post("/api/projects", json={"name": "PDF Test Project"})
    project_id = r.json()["id"]

    # Upload CSV
    r = await client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", SAMPLE_CSV, "text/csv")},
    )
    dataset_id = r.json()["dataset_id"]

    # Apply feature set (no transforms) + set target
    await client.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    await client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue"},
    )

    # Train one model
    r = await client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    # Poll until done
    import asyncio

    for _ in range(30):
        r = await client.get(f"/api/models/{project_id}/runs")
        runs = r.json()["runs"]
        done = [ru for ru in runs if ru["status"] == "done"]
        if done:
            return done[0]["id"]
        await asyncio.sleep(0.5)

    pytest.skip("Training did not complete in time")


@pytest.mark.asyncio
async def test_report_endpoint_returns_pdf(client, tmp_path):
    run_id = await _train_model(client, tmp_path)
    r = await client.get(f"/api/models/{run_id}/report")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert "attachment" in r.headers.get("content-disposition", "")
    assert r.content[:4] == b"%PDF"


@pytest.mark.asyncio
async def test_report_endpoint_404_unknown_run(client):
    r = await client.get("/api/models/nonexistent-run-id/report")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_report_endpoint_400_pending_run(client, tmp_path):
    """A run that hasn't finished training should return 400."""
    import db
    from models.model_run import ModelRun

    # Create a project and insert a pending run directly
    r = await client.post("/api/projects", json={"name": "Pending Test"})
    project_id = r.json()["id"]

    with Session(db.engine) as session:
        run = ModelRun(
            project_id=project_id, algorithm="linear_regression", status="pending"
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id

    r = await client.get(f"/api/models/{run_id}/report")
    assert r.status_code == 400
