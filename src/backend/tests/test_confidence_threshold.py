"""Tests for prediction confidence thresholding.

Covers:
- Deployment model field (confidence_threshold)
- PUT /api/deploy/{id}/confidence-threshold  — set / clear
- GET /api/deploy/{id}/confidence-threshold-status — stats
- make_prediction() below_threshold flag
- _CONFIDENCE_THRESHOLD_PATTERNS — NL intent detection regex
- Chat handler writes threshold to DB
"""

import io
import time

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module

_CLASSIFICATION_CSV = (
    b"age,income,approved\n"
    b"25,40000,yes\n30,55000,no\n35,70000,yes\n40,85000,yes\n45,60000,no\n"
    b"50,90000,yes\n22,35000,no\n28,48000,yes\n33,62000,no\n38,78000,yes\n"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.dataset_filter  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa
    import models.deployment_preset  # noqa
    import models.batch_schedule  # noqa
    import models.webhook_config  # noqa
    import models.ab_test  # noqa
    import models.deployment_version  # noqa

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
async def classification_project(ac):
    resp = await ac.post("/api/projects", json={"name": "Confidence Threshold Test"})
    project_id = resp.json()["id"]

    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("loans.csv", io.BytesIO(_CLASSIFICATION_CSV), "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code == 201
    dataset_id = resp.json()["dataset_id"]

    resp = await ac.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert resp.status_code == 201
    fs_id = resp.json()["feature_set_id"]

    await ac.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "approved", "feature_set_id": fs_id},
    )

    resp = await ac.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["logistic_regression"], "feature_set_id": fs_id},
    )
    assert resp.status_code == 202
    run_id = resp.json()["model_run_ids"][0]
    for _ in range(30):
        r = await ac.get(f"/api/models/{project_id}/runs")
        run = next((x for x in r.json().get("runs", []) if x["id"] == run_id), None)
        if run and run["status"] == "done":
            break
        time.sleep(0.3)
    else:
        pytest.skip("Training did not complete")

    resp = await ac.post(f"/api/deploy/{run_id}", json={})
    assert resp.status_code == 201, resp.text
    deployment_id = resp.json()["id"]

    return {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "deployment_id": deployment_id,
        "run_id": run_id,
    }


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------


def test_confidence_threshold_patterns_set():
    from api.chat import _CONFIDENCE_THRESHOLD_PATTERNS

    phrases = [
        "set confidence threshold to 80%",
        "configure a confidence threshold at 0.7",
        "minimum confidence threshold 75%",
        "only accept predictions above 90% confidence",
        "reject low-confidence predictions",
        "filter out predictions with low confidence",
        "confidence filter at 0.6",
        "confidence cutoff 80%",
    ]
    for phrase in phrases:
        assert _CONFIDENCE_THRESHOLD_PATTERNS.search(phrase), (
            f"Pattern did not match: {phrase!r}"
        )


def test_confidence_threshold_patterns_disable():
    from api.chat import _DISABLE_CONFIDENCE_THRESHOLD_RE

    disable_phrases = [
        "disable the confidence threshold",
        "remove confidence threshold",
        "turn off confidence threshold",
        "clear confidence threshold",
    ]
    for phrase in disable_phrases:
        assert _DISABLE_CONFIDENCE_THRESHOLD_RE.search(phrase), (
            f"Disable pattern did not match: {phrase!r}"
        )


def test_confidence_threshold_patterns_status():
    from api.chat import _CONFIDENCE_THRESHOLD_PATTERNS

    status_phrases = [
        "check my confidence threshold",
        "show confidence threshold",
        "what is my confidence threshold",
        "get confidence threshold",
    ]
    for phrase in status_phrases:
        assert _CONFIDENCE_THRESHOLD_PATTERNS.search(phrase), (
            f"Status pattern did not match: {phrase!r}"
        )


def test_confidence_threshold_value_re_percent():
    from api.chat import _CONFIDENCE_THRESHOLD_VALUE_RE

    m = _CONFIDENCE_THRESHOLD_VALUE_RE.search("set threshold to 80%")
    assert m is not None
    assert m.group(1) == "80"


def test_confidence_threshold_value_re_decimal():
    from api.chat import _CONFIDENCE_THRESHOLD_VALUE_RE

    m = _CONFIDENCE_THRESHOLD_VALUE_RE.search("configure at 0.75")
    assert m is not None
    assert m.group(1) == "0.75"


# ---------------------------------------------------------------------------
# Deployment model field
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_deployment_has_confidence_threshold_field(classification_project, ac):
    dep_id = classification_project["deployment_id"]
    resp = await ac.get(f"/api/deploy/{dep_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "confidence_threshold" in data
    assert data["confidence_threshold"] is None


# ---------------------------------------------------------------------------
# PUT /api/deploy/{id}/confidence-threshold
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_set_confidence_threshold_valid(classification_project, ac):
    dep_id = classification_project["deployment_id"]
    resp = await ac.put(
        f"/api/deploy/{dep_id}/confidence-threshold",
        json={"threshold": 0.8},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["confidence_threshold"] == pytest.approx(0.8)
    assert "80%" in data["message"]


@pytest.mark.anyio
async def test_set_confidence_threshold_persists(classification_project, ac):
    dep_id = classification_project["deployment_id"]
    await ac.put(
        f"/api/deploy/{dep_id}/confidence-threshold",
        json={"threshold": 0.65},
    )
    resp = await ac.get(f"/api/deploy/{dep_id}")
    assert resp.json()["confidence_threshold"] == pytest.approx(0.65)


@pytest.mark.anyio
async def test_clear_confidence_threshold(classification_project, ac):
    dep_id = classification_project["deployment_id"]
    await ac.put(
        f"/api/deploy/{dep_id}/confidence-threshold",
        json={"threshold": 0.8},
    )
    resp = await ac.put(
        f"/api/deploy/{dep_id}/confidence-threshold",
        json={"threshold": None},
    )
    assert resp.status_code == 200
    assert resp.json()["confidence_threshold"] is None
    assert "removed" in resp.json()["message"].lower()


@pytest.mark.anyio
async def test_set_confidence_threshold_out_of_range(classification_project, ac):
    dep_id = classification_project["deployment_id"]
    resp = await ac.put(
        f"/api/deploy/{dep_id}/confidence-threshold",
        json={"threshold": 1.5},
    )
    assert resp.status_code in (400, 422)


@pytest.mark.anyio
async def test_set_confidence_threshold_negative(classification_project, ac):
    dep_id = classification_project["deployment_id"]
    resp = await ac.put(
        f"/api/deploy/{dep_id}/confidence-threshold",
        json={"threshold": -0.1},
    )
    assert resp.status_code in (400, 422)


@pytest.mark.anyio
async def test_set_confidence_threshold_nonexistent(ac):
    resp = await ac.put(
        "/api/deploy/nonexistent-id/confidence-threshold",
        json={"threshold": 0.8},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/deploy/{id}/confidence-threshold-status
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_confidence_threshold_status_no_threshold(classification_project, ac):
    dep_id = classification_project["deployment_id"]
    resp = await ac.get(f"/api/deploy/{dep_id}/confidence-threshold-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["threshold_enabled"] is False
    assert data["confidence_threshold"] is None
    assert "summary" in data


@pytest.mark.anyio
async def test_confidence_threshold_status_with_threshold(classification_project, ac):
    dep_id = classification_project["deployment_id"]
    await ac.put(
        f"/api/deploy/{dep_id}/confidence-threshold",
        json={"threshold": 0.75},
    )
    resp = await ac.get(f"/api/deploy/{dep_id}/confidence-threshold-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["threshold_enabled"] is True
    assert data["confidence_threshold"] == pytest.approx(0.75)
    assert "below_threshold_count_30d" in data
    assert "total_predictions_30d" in data


# ---------------------------------------------------------------------------
# make_prediction() below_threshold flag
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_prediction_without_threshold_has_no_flag(classification_project, ac):
    dep_id = classification_project["deployment_id"]
    resp = await ac.post(
        f"/api/predict/{dep_id}",
        json={"features": {"age": 30, "income": 50000}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "below_threshold" not in data or data["below_threshold"] is False


@pytest.mark.anyio
async def test_prediction_above_threshold_not_flagged(classification_project, ac):
    dep_id = classification_project["deployment_id"]
    # Set a very low threshold so any real prediction is above it
    await ac.put(
        f"/api/deploy/{dep_id}/confidence-threshold",
        json={"threshold": 0.01},
    )
    resp = await ac.post(
        f"/api/predict/{dep_id}",
        json={"features": {"age": 30, "income": 50000}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("below_threshold") is False


@pytest.mark.anyio
async def test_prediction_below_threshold_flagged(classification_project, ac):
    dep_id = classification_project["deployment_id"]
    # Set a threshold of 1.0 to guarantee every prediction is below it
    await ac.put(
        f"/api/deploy/{dep_id}/confidence-threshold",
        json={"threshold": 1.0},
    )
    resp = await ac.post(
        f"/api/predict/{dep_id}",
        json={"features": {"age": 30, "income": 50000}},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("below_threshold") is True
    assert "threshold_message" in data
    assert "threshold" in data["threshold_message"].lower()
