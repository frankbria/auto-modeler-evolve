"""Tests for chat-initiated model training.

Covers:
- _TRAIN_PATTERNS regex detection
- _detect_train_target() helper
- POST /api/chat/{project_id} training_started SSE event
"""

import json

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import (
    _TRAIN_PATTERNS,
    _detect_train_target,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_CSV = (
    b"region,revenue,units,cost\n"
    b"East,100,10,50\n"
    b"West,200,20,80\n"
    b"East,150,15,60\n"
    b"West,300,30,120\n"
    b"North,250,25,100\n"
    b"East,175,18,70\n"
    b"West,220,22,90\n"
    b"North,190,19,75\n"
    b"East,130,13,55\n"
    b"West,280,28,110\n"
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

    import api.models as models_module

    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
async def project_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Train Test"})
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
    """Create a feature set with target column set."""
    # Apply empty transformations to create feature set
    resp = await ac.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert resp.status_code == 201, resp.text
    fs_id = resp.json()["feature_set_id"]

    # Set target column
    await ac.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )
    return fs_id


# ---------------------------------------------------------------------------
# Unit tests — _TRAIN_PATTERNS regex
# ---------------------------------------------------------------------------


def test_train_patterns_train_a_model():
    assert _TRAIN_PATTERNS.search("train a model")


def test_train_patterns_build_model():
    assert _TRAIN_PATTERNS.search("build a model to predict revenue")


def test_train_patterns_start_training():
    assert _TRAIN_PATTERNS.search("start training now")


def test_train_patterns_i_want_to_train():
    assert _TRAIN_PATTERNS.search("I want to train a model")


def test_train_patterns_lets_build():
    assert _TRAIN_PATTERNS.search("let's build a predictor")


def test_train_patterns_ml_model():
    assert _TRAIN_PATTERNS.search("build a machine learning model")


def test_train_patterns_no_false_positive_analyze():
    # "analyze" should not trigger training
    assert not _TRAIN_PATTERNS.search("analyze my data")


def test_train_patterns_no_false_positive_predict():
    # "predict next 6 months" should NOT trigger training (forecast pattern)
    # because it matches "predict X with model" but not "predict X next N"
    # This is intentionally not matched:
    assert not _TRAIN_PATTERNS.search("predict next quarter revenue")


# ---------------------------------------------------------------------------
# Unit tests — _detect_train_target
# ---------------------------------------------------------------------------


def test_detect_train_target_predict_pattern():
    cols = ["revenue", "region", "units", "cost"]
    result = _detect_train_target("train a model to predict revenue", cols)
    assert result == "revenue"


def test_detect_train_target_target_is_pattern():
    cols = ["sales", "region", "quantity"]
    result = _detect_train_target("train a model, target is sales", cols)
    assert result == "sales"


def test_detect_train_target_column_scan_fallback():
    cols = ["revenue", "region", "units"]
    # No pattern match, but "units" appears in message
    result = _detect_train_target("build a model for units prediction", cols)
    assert result == "units"


def test_detect_train_target_case_insensitive():
    cols = ["Revenue", "region", "units"]
    result = _detect_train_target("predict Revenue please", cols)
    assert result == "Revenue"


def test_detect_train_target_no_match():
    cols = ["revenue", "region", "units"]
    result = _detect_train_target("train a model", cols)
    assert result is None


def test_detect_train_target_returns_original_casing():
    cols = ["TotalRevenue", "region"]
    result = _detect_train_target("predict totalrevenue", cols)
    assert result == "TotalRevenue"


# ---------------------------------------------------------------------------
# Integration tests — SSE event via /api/chat/{project_id}
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_training_started_event(ac, project_id, dataset_id, feature_set_id):
    """When user says 'train a model' with feature set+target ready, get training_started."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Training will begin shortly!"])
        MockAnthropic.return_value.messages.stream.return_value = mock_stream

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "train a model to predict revenue"},
        )

    assert resp.status_code == 200

    # Parse SSE events
    events = []
    for line in resp.text.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass

    types = [e.get("type") for e in events]
    assert "training_started" in types, f"Expected training_started in {types}"

    training_event = next(e for e in events if e.get("type") == "training_started")
    training = training_event["training"]
    assert training["target_column"] == "revenue"
    assert training["status"] == "started"
    assert training["run_count"] >= 1
    assert len(training["algorithms"]) >= 1


@pytest.mark.asyncio
async def test_chat_training_creates_feature_set_when_none_exists(
    ac, project_id, dataset_id
):
    """When no feature set exists, training request with named target should create one."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Starting training!"])
        MockAnthropic.return_value.messages.stream.return_value = mock_stream

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "train a model to predict revenue"},
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
    assert "training_started" in types


@pytest.mark.asyncio
async def test_chat_training_no_target_no_event(ac, project_id, dataset_id):
    """Training request without a nameable target emits no training_started event."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["What would you like to predict?"])
        MockAnthropic.return_value.messages.stream.return_value = mock_stream

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "train a model"},
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
    assert "training_started" not in types


@pytest.mark.asyncio
async def test_chat_training_no_dataset_no_event(ac, project_id):
    """With no dataset, training request emits no training_started event."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Please upload a dataset first."])
        MockAnthropic.return_value.messages.stream.return_value = mock_stream

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "train a model to predict revenue"},
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
    assert "training_started" not in types
