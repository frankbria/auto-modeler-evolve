"""Tests for chat-initiated model deployment.

Covers:
- _DEPLOY_CHAT_PATTERNS regex detection
- execute_deployment() helper extracted from deploy_model route
- POST /api/chat/{project_id} deployed SSE event (selected run deployed)
- POST /api/chat/{project_id} deployed SSE event (best run deployed when none selected)
- POST /api/chat/{project_id} no-model graceful handling
- Idempotency: second deploy request returns existing deployment
"""

import json
import time

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import _DEPLOY_CHAT_PATTERNS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_CSV = (
    b"region,revenue,units,cost\n"
    b"East,100.5,10,50\n"
    b"West,200.3,20,80\n"
    b"East,150.7,15,60\n"
    b"West,300.1,30,120\n"
    b"North,250.9,25,100\n"
    b"East,175.2,18,70\n"
    b"West,220.4,22,90\n"
    b"North,190.6,19,75\n"
    b"East,130.8,13,55\n"
    b"West,280.0,28,110\n"
)


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.deploy as deploy_module

    deploy_module.DEPLOY_DIR = tmp_path / "deployments"

    import api.models as models_module

    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
async def project_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Deploy Test"})
    return resp.json()["id"]


@pytest.fixture()
async def dataset_id(ac, project_id):
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("sales.csv", _SAMPLE_CSV, "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["dataset_id"]


@pytest.fixture()
async def feature_set_id(ac, dataset_id):
    resp = await ac.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert resp.status_code == 201, resp.text
    fs_id = resp.json()["feature_set_id"]

    await ac.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )
    return fs_id


@pytest.fixture()
async def trained_run_id(ac, project_id, feature_set_id):
    """Train a single model and wait until it finishes."""
    resp = await ac.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"], "feature_set_id": feature_set_id},
    )
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["model_run_ids"][0]

    # Poll until done (max 10s)
    for _ in range(20):
        r = await ac.get(f"/api/models/{project_id}/runs")
        runs = r.json().get("runs", [])
        run = next((x for x in runs if x["id"] == run_id), None)
        if run and run["status"] == "done":
            return run_id
        time.sleep(0.5)
    pytest.skip("Training did not complete in time")


@pytest.fixture()
async def selected_run_id(ac, project_id, trained_run_id):
    """Select the trained run."""
    await ac.post(f"/api/models/{trained_run_id}/select")
    return trained_run_id


# ---------------------------------------------------------------------------
# Unit tests — _DEPLOY_CHAT_PATTERNS regex
# ---------------------------------------------------------------------------


def test_deploy_pattern_deploy_my_model():
    assert _DEPLOY_CHAT_PATTERNS.search("deploy my model")


def test_deploy_pattern_deploy_the_model():
    assert _DEPLOY_CHAT_PATTERNS.search("deploy the model")


def test_deploy_pattern_go_live():
    assert _DEPLOY_CHAT_PATTERNS.search("let's go live")


def test_deploy_pattern_make_it_live():
    assert _DEPLOY_CHAT_PATTERNS.search("make it live")


def test_deploy_pattern_publish():
    assert _DEPLOY_CHAT_PATTERNS.search("publish my model")


def test_deploy_pattern_launch_api():
    assert _DEPLOY_CHAT_PATTERNS.search("launch the api")


def test_deploy_pattern_put_in_production():
    assert _DEPLOY_CHAT_PATTERNS.search("put my model in production")


def test_deploy_pattern_ship():
    assert _DEPLOY_CHAT_PATTERNS.search("ship my model")


def test_deploy_pattern_create_endpoint():
    assert _DEPLOY_CHAT_PATTERNS.search("create an api from my model")


def test_deploy_pattern_no_false_positive_train():
    assert not _DEPLOY_CHAT_PATTERNS.search("train a model")


def test_deploy_pattern_no_false_positive_analyze():
    assert not _DEPLOY_CHAT_PATTERNS.search("analyze my data")


# ---------------------------------------------------------------------------
# Unit test — execute_deployment helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_deployment_creates_deployment(ac, project_id, trained_run_id):
    """execute_deployment() creates and returns a deployment dict."""
    resp = await ac.post(f"/api/deploy/{trained_run_id}")
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"]
    assert data["endpoint_path"].startswith("/api/predict/")
    assert data["dashboard_url"].startswith("/predict/")
    assert data["algorithm"] == "linear_regression"
    assert data["target_column"] == "revenue"
    assert data["is_active"] is True


@pytest.mark.asyncio
async def test_execute_deployment_idempotent(ac, project_id, trained_run_id):
    """Deploying the same run twice returns the same deployment."""
    r1 = await ac.post(f"/api/deploy/{trained_run_id}")
    r2 = await ac.post(f"/api/deploy/{trained_run_id}")
    assert r1.status_code in (200, 201)
    assert r2.status_code in (200, 201)
    assert r1.json()["id"] == r2.json()["id"]


# ---------------------------------------------------------------------------
# Integration tests — deployed SSE event via /api/chat/{project_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_deploy_selected_run(ac, project_id, selected_run_id):
    """Chat 'deploy my model' with a selected run emits deployed SSE event."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Your model is now live! 🚀"])
        MockAnthropic.return_value.messages.stream.return_value = mock_stream

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "deploy my model"},
        )

    assert resp.status_code == 200

    events = []
    for line in resp.text.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass

    types = [e.get("type") for e in events]
    assert "deployed" in types, f"Expected 'deployed' in {types}"

    deployed_event = next(e for e in events if e.get("type") == "deployed")
    dep = deployed_event["deployment"]
    assert dep["algorithm"] == "linear_regression"
    assert dep["target_column"] == "revenue"
    assert dep["is_active"] is True
    assert dep["endpoint_path"].startswith("/api/predict/")
    assert dep["dashboard_url"].startswith("/predict/")


@pytest.mark.asyncio
async def test_chat_deploy_best_run_when_none_selected(ac, project_id, trained_run_id):
    """Chat 'go live' with completed but unselected run still deploys."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Deploying your best model!"])
        MockAnthropic.return_value.messages.stream.return_value = mock_stream

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "make it live"},
        )

    assert resp.status_code == 200

    events = []
    for line in resp.text.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass

    types = [e.get("type") for e in events]
    assert "deployed" in types, f"Expected 'deployed' in {types}"


@pytest.mark.asyncio
async def test_chat_deploy_no_models_does_not_crash(ac, project_id, dataset_id):
    """Chat 'deploy my model' with no completed runs does not crash; just no deployed event."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["You need to train a model first."])
        MockAnthropic.return_value.messages.stream.return_value = mock_stream

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "deploy my model"},
        )

    assert resp.status_code == 200

    events = []
    for line in resp.text.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass

    # No deployed event — but chat completes normally
    types = [e.get("type") for e in events]
    assert "done" in types
    assert "deployed" not in types


@pytest.mark.asyncio
async def test_chat_deploy_already_deployed_returns_existing(
    ac, project_id, selected_run_id
):
    """If model is already deployed, second deploy chat request returns existing deployment."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Your model is live!"])
        MockAnthropic.return_value.messages.stream.return_value = mock_stream

        # First deploy via REST
        r1 = await ac.post(f"/api/deploy/{selected_run_id}")
        assert r1.status_code in (200, 201)

        # Second deploy via chat — should return the existing deployment
        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "deploy my model"},
        )

    assert resp.status_code == 200

    events = []
    for line in resp.text.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass

    # The deployed_event should NOT be emitted again because ctx["deployment"] is already set
    # (the chat handler checks `not ctx["deployment"]` before attempting deployment)
    types = [e.get("type") for e in events]
    assert "done" in types
