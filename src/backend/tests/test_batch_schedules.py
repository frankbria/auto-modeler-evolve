"""Tests for batch prediction scheduling: models, scheduler logic, and REST endpoints."""

import io
import time
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

SAMPLE_CSV = b"""date,product,region,revenue,units
2024-01-01,Widget A,North,1200.50,10
2024-01-01,Widget B,South,850.00,8
2024-01-02,Widget A,East,2100.75,18
2024-01-02,Widget C,West,450.25,4
2024-01-03,Widget B,North,1650.00,15
2024-01-04,Widget A,South,980.00,9
2024-01-04,Widget C,North,1100.25,11
2024-01-05,Widget B,East,1750.00,16
2024-01-05,Widget A,West,2300.50,20
2024-01-06,Widget C,South,620.75,6
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models  # noqa — registers all tables

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_api

    models_api.MODELS_DIR = tmp_path / "models"

    import api.deploy as deploy_module

    deploy_module.DEPLOY_DIR = tmp_path / "deployments"

    import core.scheduler as sched_module

    sched_module.BATCH_OUTPUT_DIR = tmp_path / "batch_outputs"

    from main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def deployed(client):
    """Create a project → upload → features → train → deploy."""
    proj = client.post("/api/projects", json={"name": "Sched Test"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    apply = client.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert apply.status_code in (200, 201)

    client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue"},
    )

    train = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    assert train.status_code == 202
    run_id = train.json()["model_run_ids"][0]

    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next((r for r in runs if r["id"] == run_id), None)
        if run and run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    deploy = client.post(f"/api/deploy/{run_id}")
    assert deploy.status_code == 201
    return deploy.json()["id"]


# ---------------------------------------------------------------------------
# Unit tests: compute_next_run
# ---------------------------------------------------------------------------


def test_next_run_daily():
    from core.scheduler import compute_next_run

    after = datetime(2024, 3, 15, 8, 0, 0)
    nxt = compute_next_run("daily", 9, 0, None, None, after=after)
    assert nxt == datetime(2024, 3, 15, 9, 0, 0)


def test_next_run_daily_past_time():
    from core.scheduler import compute_next_run

    after = datetime(2024, 3, 15, 10, 0, 0)  # already past 09:00
    nxt = compute_next_run("daily", 9, 0, None, None, after=after)
    assert nxt == datetime(2024, 3, 16, 9, 0, 0)


def test_next_run_weekly():
    from core.scheduler import compute_next_run

    # 2024-03-15 is a Friday (weekday=4). Next Monday (0) should be 2024-03-18
    after = datetime(2024, 3, 15, 12, 0, 0)
    nxt = compute_next_run("weekly", 9, 0, 0, None, after=after)
    assert nxt.weekday() == 0
    assert nxt > after


def test_next_run_monthly():
    from core.scheduler import compute_next_run

    after = datetime(2024, 3, 10, 12, 0, 0)
    nxt = compute_next_run("monthly", 9, 0, None, 15, after=after)
    assert nxt.day == 15
    assert nxt.month == 3


def test_next_run_monthly_past():
    from core.scheduler import compute_next_run

    after = datetime(2024, 3, 20, 12, 0, 0)  # past the 15th
    nxt = compute_next_run("monthly", 9, 0, None, 15, after=after)
    assert nxt.day == 15
    assert nxt.month == 4


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


def test_create_schedule(client, deployed):
    resp = client.post(
        f"/api/deploy/{deployed}/schedules",
        json={"frequency": "daily", "run_hour": 9, "run_minute": 0},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["frequency"] == "daily"
    assert data["run_hour"] == 9
    assert data["deployment_id"] == deployed
    assert data["next_run"] is not None
    assert data["is_active"] is True


def test_create_schedule_weekly(client, deployed):
    resp = client.post(
        f"/api/deploy/{deployed}/schedules",
        json={"frequency": "weekly", "run_hour": 8, "run_minute": 30, "day_of_week": 1},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["frequency"] == "weekly"
    assert data["day_of_week"] == 1


def test_create_schedule_monthly(client, deployed):
    resp = client.post(
        f"/api/deploy/{deployed}/schedules",
        json={
            "frequency": "monthly",
            "run_hour": 6,
            "run_minute": 0,
            "day_of_month": 1,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["frequency"] == "monthly"
    assert data["day_of_month"] == 1


def test_create_schedule_invalid_frequency(client, deployed):
    resp = client.post(
        f"/api/deploy/{deployed}/schedules",
        json={"frequency": "hourly", "run_hour": 9, "run_minute": 0},
    )
    assert resp.status_code == 400


def test_create_schedule_invalid_hour(client, deployed):
    resp = client.post(
        f"/api/deploy/{deployed}/schedules",
        json={"frequency": "daily", "run_hour": 25, "run_minute": 0},
    )
    assert resp.status_code == 400


def test_create_schedule_not_found(client):
    resp = client.post(
        "/api/deploy/nonexistent-id/schedules",
        json={"frequency": "daily", "run_hour": 9, "run_minute": 0},
    )
    assert resp.status_code == 404


def test_list_schedules_empty(client, deployed):
    resp = client.get(f"/api/deploy/{deployed}/schedules")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_schedules(client, deployed):
    client.post(
        f"/api/deploy/{deployed}/schedules",
        json={"frequency": "daily", "run_hour": 9, "run_minute": 0},
    )
    client.post(
        f"/api/deploy/{deployed}/schedules",
        json={"frequency": "weekly", "run_hour": 8, "run_minute": 0, "day_of_week": 0},
    )
    resp = client.get(f"/api/deploy/{deployed}/schedules")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_delete_schedule(client, deployed):
    create = client.post(
        f"/api/deploy/{deployed}/schedules",
        json={"frequency": "daily", "run_hour": 9, "run_minute": 0},
    )
    schedule_id = create.json()["id"]

    delete = client.delete(f"/api/deploy/{deployed}/schedules/{schedule_id}")
    assert delete.status_code == 204

    # Schedule still in list but inactive
    lst = client.get(f"/api/deploy/{deployed}/schedules")
    items = [s for s in lst.json() if s["id"] == schedule_id]
    assert len(items) == 1
    assert items[0]["is_active"] is False


def test_delete_schedule_wrong_deployment(client, deployed):
    create = client.post(
        f"/api/deploy/{deployed}/schedules",
        json={"frequency": "daily", "run_hour": 9, "run_minute": 0},
    )
    schedule_id = create.json()["id"]

    resp = client.delete(f"/api/deploy/wrong-id/schedules/{schedule_id}")
    assert resp.status_code == 404


def test_trigger_schedule_run(client, deployed):
    create = client.post(
        f"/api/deploy/{deployed}/schedules",
        json={"frequency": "daily", "run_hour": 9, "run_minute": 0},
    )
    schedule_id = create.json()["id"]

    resp = client.post(f"/api/deploy/{deployed}/schedules/{schedule_id}/run")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"
    assert resp.json()["schedule_id"] == schedule_id


def test_list_schedule_runs_empty(client, deployed):
    create = client.post(
        f"/api/deploy/{deployed}/schedules",
        json={"frequency": "daily", "run_hour": 9, "run_minute": 0},
    )
    schedule_id = create.json()["id"]

    resp = client.get(f"/api/deploy/{deployed}/schedules/{schedule_id}/runs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_download_batch_output_invalid_filename(client):
    """Reject filenames with path traversal attempts."""
    resp = client.get("/api/deploy/batch-outputs/../etc/passwd")
    assert resp.status_code in (400, 404)


def test_download_batch_output_not_found(client):
    resp = client.get("/api/deploy/batch-outputs/nonexistent_file.csv")
    assert resp.status_code == 404
