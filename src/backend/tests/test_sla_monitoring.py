"""Tests for prediction SLA monitoring.

Covers:
- GET /api/deploy/{id}/sla returns 404 when deployment not found
- Returns empty response with sample_count=0 when no timed predictions exist
- p50/p95/p99 computed correctly from sorted latencies
- alert=True when p95 > 500ms
- alert_message is None when p95 <= 500ms
- latency_by_day aggregates correctly
- _percentile helper edge cases (single value, full list)
- response_ms stored on PredictionLog during make_prediction
- make_prediction returns prediction + response_ms stored
- Predictions with NULL response_ms excluded from SLA sample_count
- avg_ms computed correctly
"""

import asyncio
import io

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.deploy import _percentile

# ---------------------------------------------------------------------------
# Unit tests for _percentile helper
# ---------------------------------------------------------------------------


def test_percentile_single_value():
    assert _percentile([100.0], 50) == 100.0
    assert _percentile([100.0], 95) == 100.0
    assert _percentile([100.0], 99) == 100.0


def test_percentile_empty():
    assert _percentile([], 50) == 0.0


def test_percentile_known_values():
    vals = sorted([10.0, 20.0, 30.0, 40.0, 50.0])
    # p50 should be the middle value
    assert _percentile(vals, 50) == 30.0
    # p0 should be the minimum
    assert _percentile(vals, 0) == 10.0
    # p100 should be the maximum
    assert _percentile(vals, 100) == 50.0


def test_percentile_two_values():
    vals = [100.0, 200.0]
    # p50 of [100, 200] should be 150
    assert _percentile(vals, 50) == 150.0


# ---------------------------------------------------------------------------
# Sample CSV
# ---------------------------------------------------------------------------

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
    b"East,175.2,18\nWest,220.4,22\nNorth,190.6,19\nEast,130.8,13\nWest,280.0,28\n"
    b"East,160.0,16\nWest,210.0,21\n"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.batch_schedule  # noqa
    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.dataset_filter  # noqa
    import models.deployment  # noqa
    import models.deployment_version  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa
    import models.webhook_config  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.deploy as deploy_module
    import api.models as models_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"
    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
async def project_id(ac):
    r = await ac.post("/api/projects", json={"name": "SLA Test"})
    return r.json()["id"]


@pytest.fixture()
async def dataset_id(ac, project_id):
    r = await ac.post(
        "/api/data/upload",
        files={"file": ("sales.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        data={"project_id": project_id},
    )
    assert r.status_code == 201, r.text
    return r.json()["dataset_id"]


@pytest.fixture()
async def feature_set_id(ac, dataset_id):
    r = await ac.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert r.status_code == 201, r.text
    fs_id = r.json()["feature_set_id"]
    await ac.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )
    return fs_id


async def _train_and_wait(ac, project_id, feature_set_id):
    r = await ac.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"], "feature_set_id": feature_set_id},
    )
    assert r.status_code == 202, r.text
    run_id = r.json()["model_run_ids"][0]
    for _ in range(40):
        r2 = await ac.get(f"/api/models/{project_id}/runs")
        run = next((x for x in r2.json().get("runs", []) if x["id"] == run_id), None)
        if run and run["status"] == "done":
            return run_id
        await asyncio.sleep(0.2)
    pytest.fail("Training timed out")


@pytest.fixture()
async def deployment_id(ac, project_id, feature_set_id):
    run_id = await _train_and_wait(ac, project_id, feature_set_id)
    r = await ac.post(f"/api/deploy/{run_id}")
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ---------------------------------------------------------------------------
# SLA endpoint: basic cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sla_not_found(ac):
    r = await ac.get("/api/deploy/nonexistent/sla")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_sla_no_predictions(ac, deployment_id):
    r = await ac.get(f"/api/deploy/{deployment_id}/sla")
    assert r.status_code == 200
    data = r.json()
    assert data["sample_count"] == 0
    assert data["p50_ms"] is None
    assert data["p95_ms"] is None
    assert data["p99_ms"] is None
    assert data["avg_ms"] is None
    assert data["alert"] is False
    assert data["alert_message"] is None
    assert data["latency_by_day"] == []


@pytest.mark.asyncio
async def test_sla_after_predictions(ac, deployment_id):
    """After predictions, SLA should have timing data."""
    # Make a prediction to populate response_ms
    r = await ac.get(f"/api/deploy/{deployment_id}")
    schema = r.json()["feature_schema"]
    payload = {
        col["name"]: (1.0 if col["type"] == "numeric" else "East") for col in schema
    }

    pred_r = await ac.post(f"/api/predict/{deployment_id}", json=payload)
    assert pred_r.status_code == 200

    sla_r = await ac.get(f"/api/deploy/{deployment_id}/sla")
    assert sla_r.status_code == 200
    data = sla_r.json()
    assert data["sample_count"] == 1
    assert data["p50_ms"] is not None
    assert data["p95_ms"] is not None
    assert data["p99_ms"] is not None
    assert data["avg_ms"] is not None
    assert data["p50_ms"] >= 0
    assert isinstance(data["latency_by_day"], list)
    assert len(data["latency_by_day"]) == 1
    assert "date" in data["latency_by_day"][0]
    assert "avg_ms" in data["latency_by_day"][0]


@pytest.mark.asyncio
async def test_sla_percentiles_multiple_predictions(ac, deployment_id):
    """With multiple predictions, p50/p95/p99 should be well-ordered."""
    r = await ac.get(f"/api/deploy/{deployment_id}")
    schema = r.json()["feature_schema"]
    payload = {
        col["name"]: (1.0 if col["type"] == "numeric" else "East") for col in schema
    }

    for _ in range(5):
        await ac.post(f"/api/predict/{deployment_id}", json=payload)

    sla_r = await ac.get(f"/api/deploy/{deployment_id}/sla")
    data = sla_r.json()
    assert data["sample_count"] == 5
    # p50 <= p95 <= p99
    assert data["p50_ms"] <= data["p95_ms"]
    assert data["p95_ms"] <= data["p99_ms"]


@pytest.mark.asyncio
async def test_sla_no_alert_for_fast_predictions(ac, deployment_id):
    """Predictions under 500ms p95 should not trigger alert."""
    r = await ac.get(f"/api/deploy/{deployment_id}")
    schema = r.json()["feature_schema"]
    payload = {
        col["name"]: (1.0 if col["type"] == "numeric" else "East") for col in schema
    }

    await ac.post(f"/api/predict/{deployment_id}", json=payload)

    sla_r = await ac.get(f"/api/deploy/{deployment_id}/sla")
    data = sla_r.json()
    # Local sklearn predictions are always well under 500ms
    assert data["alert"] is False
    assert data["alert_message"] is None


@pytest.mark.asyncio
async def test_sla_alert_when_p95_exceeds_500ms(ac, deployment_id):
    """Directly insert high-latency PredictionLog rows and verify alert fires."""
    from sqlmodel import Session
    from models.prediction_log import PredictionLog

    with Session(db_module.engine) as sess:
        for ms in [600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0]:
            sess.add(
                PredictionLog(
                    deployment_id=deployment_id,
                    input_features="{}",
                    prediction='"1.0"',
                    response_ms=ms,
                )
            )
        sess.commit()

    sla_r = await ac.get(f"/api/deploy/{deployment_id}/sla")
    data = sla_r.json()
    assert data["alert"] is True
    assert data["alert_message"] is not None
    assert "500ms" in data["alert_message"]


@pytest.mark.asyncio
async def test_sla_latency_by_day_aggregation(ac, deployment_id):
    """latency_by_day should group by date and average the ms values."""
    from sqlmodel import Session
    from models.prediction_log import PredictionLog
    from datetime import datetime

    with Session(db_module.engine) as sess:
        # Two entries on the same day
        for ms in [100.0, 200.0]:
            sess.add(
                PredictionLog(
                    deployment_id=deployment_id,
                    input_features="{}",
                    prediction='"1.0"',
                    response_ms=ms,
                    created_at=datetime(2025, 1, 15, 12, 0, 0),
                )
            )
        # One entry on a different day
        sess.add(
            PredictionLog(
                deployment_id=deployment_id,
                input_features="{}",
                prediction='"1.0"',
                response_ms=300.0,
                created_at=datetime(2025, 1, 16, 12, 0, 0),
            )
        )
        sess.commit()

    sla_r = await ac.get(f"/api/deploy/{deployment_id}/sla")
    data = sla_r.json()
    by_day = {row["date"]: row["avg_ms"] for row in data["latency_by_day"]}
    assert by_day["2025-01-15"] == 150.0
    assert by_day["2025-01-16"] == 300.0


@pytest.mark.asyncio
async def test_sla_excludes_null_response_ms(ac, deployment_id):
    """Predictions without response_ms (old logs) must not affect sample_count."""
    from sqlmodel import Session
    from models.prediction_log import PredictionLog

    with Session(db_module.engine) as sess:
        # Old-style log with no timing data
        sess.add(
            PredictionLog(
                deployment_id=deployment_id,
                input_features="{}",
                prediction='"1.0"',
                response_ms=None,
            )
        )
        # New-style log with timing
        sess.add(
            PredictionLog(
                deployment_id=deployment_id,
                input_features="{}",
                prediction='"2.0"',
                response_ms=50.0,
            )
        )
        sess.commit()

    sla_r = await ac.get(f"/api/deploy/{deployment_id}/sla")
    data = sla_r.json()
    assert data["sample_count"] == 1  # only the timed one
    assert data["p50_ms"] == 50.0
