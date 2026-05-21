"""Tests for Proactive Milestone Messages.

These messages fire automatically on the first chat message after a workflow
state transition (upload → train → deploy) WITHOUT the analyst asking.

Backend:  _get_current_milestone_state(), _MILESTONE_ORDER
SSE type: milestone
"""

from __future__ import annotations

import csv
import io
import json
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import _MILESTONE_ORDER, _get_current_milestone_state

# ---------------------------------------------------------------------------
# 1. Pure function unit tests
# ---------------------------------------------------------------------------


def test_milestone_order_has_four_levels():
    assert len(_MILESTONE_ORDER) == 4
    assert _MILESTONE_ORDER[0] is None
    assert _MILESTONE_ORDER[1] == "upload"
    assert _MILESTONE_ORDER[2] == "train"
    assert _MILESTONE_ORDER[3] == "deploy"


def test_get_state_no_data():
    ctx: dict = {"dataset": None, "model_runs": [], "deployment": None}
    assert _get_current_milestone_state(ctx) is None


def test_get_state_upload():
    ctx: dict = {
        "dataset": object(),
        "model_runs": [],
        "deployment": None,
    }
    assert _get_current_milestone_state(ctx) == "upload"


def test_get_state_train():
    class FakeRun:
        status = "done"

    ctx: dict = {
        "dataset": object(),
        "model_runs": [FakeRun()],
        "deployment": None,
    }
    assert _get_current_milestone_state(ctx) == "train"


def test_get_state_train_ignores_failed_runs():
    class FakeRunFailed:
        status = "failed"

    ctx: dict = {
        "dataset": object(),
        "model_runs": [FakeRunFailed()],
        "deployment": None,
    }
    # Only failed runs → still in "upload" milestone stage
    assert _get_current_milestone_state(ctx) == "upload"


def test_get_state_deploy():
    class FakeRun:
        status = "done"

    ctx: dict = {
        "dataset": object(),
        "model_runs": [FakeRun()],
        "deployment": object(),
    }
    assert _get_current_milestone_state(ctx) == "deploy"


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CSV = b"units,price,revenue\n1,5,50\n2,10,100\n3,15,150\n4,20,200\n5,25,250\n6,30,300\n7,35,350\n8,40,400\n9,45,450\n10,50,500\n11,55,550\n12,60,600\n13,65,650\n14,70,700\n15,75,750\n16,80,800\n17,85,850\n18,90,900\n19,95,950\n20,100,1000\n"


@pytest.fixture()
def client(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    db_module.engine = engine

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    (tmp_path / "uploads").mkdir(exist_ok=True)

    import api.models as models_api_module

    models_api_module.MODELS_DIR = tmp_path / "models"
    (tmp_path / "models").mkdir(exist_ok=True)

    from main import app

    with TestClient(app) as c:
        yield c


def _create_project(client: TestClient) -> str:
    resp = client.post("/api/projects", json={"name": "Milestone Test"})
    assert resp.status_code in (200, 201)
    return resp.json()["id"]


def _upload_csv(client: TestClient, project_id: str) -> str:
    resp = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    assert resp.status_code in (200, 201), f"Upload failed ({resp.status_code}): {resp.text[:200]}"
    return resp.json()["dataset_id"]


def _chat_events(client: TestClient, project_id: str, message: str) -> list[dict]:
    """Send a chat message with mocked Anthropic; return all SSE events."""
    with patch("anthropic.Anthropic") as mock_cls:
        mock_c = MagicMock()
        mock_cls.return_value = mock_c
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Great! "])
        mock_c.messages.stream.return_value = mock_stream

        resp = client.post(
            f"/api/chat/{project_id}",
            json={"message": message},
        )
    assert resp.status_code == 200, f"Chat failed: {resp.text}"
    events = []
    for line in resp.content.decode().splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def _find_event(events: list[dict], event_type: str) -> dict | None:
    return next((e for e in events if e.get("type") == event_type), None)


# ---------------------------------------------------------------------------
# 2. Integration tests
# ---------------------------------------------------------------------------


def test_upload_milestone_fires_on_first_chat_after_upload(client):
    """After uploading a dataset, the next chat message triggers the upload milestone."""
    pid = _create_project(client)
    _upload_csv(client, pid)
    events = _chat_events(client, pid, "hello, I just uploaded my data")
    ms_event = _find_event(events, "milestone")
    assert ms_event is not None, "Expected milestone event after first upload"
    ms = ms_event["milestone"]
    assert ms["milestone_type"] == "upload"
    assert ms["progress"] == 20
    assert ms["icon"] == "🎉"
    assert len(ms["actions"]) == 2


def test_upload_milestone_includes_row_count(client):
    """Upload milestone summary should mention the row count."""
    pid = _create_project(client)
    _upload_csv(client, pid)
    events = _chat_events(client, pid, "hi there")
    ms = _find_event(events, "milestone")["milestone"]
    assert "20" in ms["summary"]  # 20 rows uploaded


def test_upload_milestone_only_fires_once(client):
    """Milestone should not repeat on the second message after the same state."""
    pid = _create_project(client)
    _upload_csv(client, pid)
    # First message — fires milestone
    _chat_events(client, pid, "hi")
    # Second message — no milestone (state hasn't changed)
    events = _chat_events(client, pid, "what else can I do?")
    ms_event = _find_event(events, "milestone")
    assert ms_event is None, "Milestone should not fire a second time for same state"


def test_no_milestone_when_no_data(client):
    """No milestone fires when the project has no dataset yet."""
    pid = _create_project(client)
    events = _chat_events(client, pid, "hello")
    ms_event = _find_event(events, "milestone")
    assert ms_event is None


def test_milestone_event_fields(client):
    """Milestone event must contain all required fields."""
    pid = _create_project(client)
    _upload_csv(client, pid)
    events = _chat_events(client, pid, "hi")
    ms = _find_event(events, "milestone")["milestone"]
    for field in (
        "milestone_type",
        "icon",
        "title",
        "subtitle",
        "summary",
        "progress",
        "actions",
    ):
        assert field in ms, f"Missing field: {field}"


def test_milestone_actions_have_label_and_prompt(client):
    """Each action in the milestone card must have 'label' and 'prompt'."""
    pid = _create_project(client)
    _upload_csv(client, pid)
    events = _chat_events(client, pid, "hello")
    ms = _find_event(events, "milestone")["milestone"]
    for action in ms["actions"]:
        assert "label" in action
        assert "prompt" in action
        assert action["label"]
        assert action["prompt"]


def test_milestone_progress_in_valid_range(client):
    """Progress value must be between 0 and 100."""
    pid = _create_project(client)
    _upload_csv(client, pid)
    events = _chat_events(client, pid, "hello")
    ms = _find_event(events, "milestone")["milestone"]
    assert 0 <= ms["progress"] <= 100
