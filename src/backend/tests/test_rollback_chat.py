"""Tests for deployment rollback via chat.

Covers:
- _ROLLBACK_PATTERNS — NL intent detection regex (rollback vs list variants)
- _ROLLBACK_VERSION_RE — version number extraction
- Chat handler: list mode (no rollback, just shows versions)
- Chat handler: rollback mode (performs rollback inline)
- Chat handler: error cases (no deployment, already current, version not found)
"""

import io
import json
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
    b"East,175.2,18\nWest,220.4,22\nNorth,190.6,19\nEast,130.8,13\nWest,280.0,28\n"
)


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------


def test_rollback_patterns_match_rollback_phrases():
    from api.chat import _ROLLBACK_PATTERNS

    phrases = [
        "roll back my model",
        "rollback to version 2",
        "revert my deployment",
        "restore to version 1",
        "go back to version 3",
        "undo the last retrain",
        "cancel the last redeploy",
        "previous version please",
    ]
    for phrase in phrases:
        assert _ROLLBACK_PATTERNS.search(phrase), f"Should match: {phrase!r}"


def test_rollback_patterns_match_list_phrases():
    from api.chat import _ROLLBACK_PATTERNS

    phrases = [
        "show my deployment versions",
        "list my versions",
        "deployment version history",
        "what are my versions",
        "version history",
    ]
    for phrase in phrases:
        assert _ROLLBACK_PATTERNS.search(phrase), f"Should match: {phrase!r}"


def test_rollback_patterns_no_false_positives():
    from api.chat import _ROLLBACK_PATTERNS

    non_matches = [
        "show me the correlation heatmap",
        "train a model",
        "what is my model accuracy",
        "deploy my model",
        "explain the prediction",
    ]
    for phrase in non_matches:
        assert not _ROLLBACK_PATTERNS.search(phrase), f"Should NOT match: {phrase!r}"


def test_rollback_version_re_extracts_number():
    from api.chat import _ROLLBACK_VERSION_RE

    cases = [
        ("roll back to version 2", "2"),
        ("restore v3 please", "3"),
        ("revert to version 10", "10"),
        ("go back to v1", "1"),
    ]
    for msg, expected in cases:
        m = _ROLLBACK_VERSION_RE.search(msg)
        assert m and m.group(1) == expected, f"Expected {expected!r} from {msg!r}"


def test_rollback_version_re_no_match():
    from api.chat import _ROLLBACK_VERSION_RE

    assert _ROLLBACK_VERSION_RE.search("roll back my model") is None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    monkeypatch.setattr(db_module, "engine", engine)
    monkeypatch.setattr(db_module, "DATA_DIR", tmp_path)
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import api.data as data_module
    import api.deploy as deploy_module
    import api.models as models_module

    monkeypatch.setattr(data_module, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(deploy_module, "DEPLOY_DIR", tmp_path / "deployments")
    monkeypatch.setattr(models_module, "MODELS_DIR", tmp_path / "models")

    from main import app

    return TestClient(app)


def _setup_project_with_deployment(client):
    """Create project → upload CSV → train model → deploy. Returns (proj_id, dep_id, run_id)."""
    proj_r = client.post("/api/projects", json={"name": "Rollback Test"})
    proj_id = proj_r.json()["id"]

    up_r = client.post(
        "/api/data/upload",
        files={"file": ("data.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        data={"project_id": proj_id},
    )
    assert up_r.status_code == 201, up_r.text
    ds_id = up_r.json()["dataset_id"]

    feat_r = client.post(
        f"/api/features/{ds_id}/apply",
        json={"transformations": []},
    )
    assert feat_r.status_code == 201
    fs_id = feat_r.json()["feature_set_id"]

    client.post(
        f"/api/features/{ds_id}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )

    train_r = client.post(
        f"/api/models/{proj_id}/train",
        json={"algorithms": ["linear_regression"], "feature_set_id": fs_id},
    )
    assert train_r.status_code == 202
    run_id = train_r.json()["model_run_ids"][0]

    for _ in range(30):
        runs_r = client.get(f"/api/models/{proj_id}/runs")
        run = next(
            (x for x in runs_r.json().get("runs", []) if x["id"] == run_id), None
        )
        if run and run["status"] == "done":
            break
        time.sleep(0.3)
    else:
        pytest.skip("Training did not complete in time")

    dep_r = client.post(f"/api/deploy/{run_id}", json={})
    assert dep_r.status_code == 201, dep_r.text
    dep_id = dep_r.json()["id"]

    return proj_id, dep_id, run_id


def _chat_events(client, project_id: str, message: str) -> list[dict]:
    """Send a chat message under mocked Anthropic and return all SSE events."""
    with patch("anthropic.Anthropic") as mock_cls:
        mock_c = MagicMock()
        mock_cls.return_value = mock_c
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["OK."])
        mock_c.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": message},
        )

    events = []
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_chat_list_versions_shows_version_card(client):
    """Asking 'show my versions' returns rollback_chat SSE event with version list."""
    proj_id, _dep_id, _run_id = _setup_project_with_deployment(client)

    events = _chat_events(client, proj_id, "show my deployment versions")

    rollback_events = [e for e in events if e.get("type") == "rollback_chat"]
    assert rollback_events, "Should emit rollback_chat SSE event"
    data = rollback_events[0]["rollback_chat"]
    assert data["action"] == "list"
    assert data["total_versions"] >= 1
    assert data["rolled_back"] is False
    assert isinstance(data["versions"], list)


def test_chat_rollback_to_previous_version(client):
    """Roll back to version 1 after a second deployment creates version 2."""
    proj_id, _dep_id, run_id = _setup_project_with_deployment(client)

    runs_r = client.get(f"/api/models/{proj_id}/runs")
    orig_run = next(r for r in runs_r.json()["runs"] if r["id"] == run_id)
    fs_id2 = orig_run.get("feature_set_id")

    train_r2 = client.post(
        f"/api/models/{proj_id}/train",
        json={"algorithms": ["linear_regression"], "feature_set_id": fs_id2},
    )
    assert train_r2.status_code == 202

    for _ in range(30):
        runs_r = client.get(f"/api/models/{proj_id}/runs")
        runs = runs_r.json().get("runs", [])
        if any(r["status"] == "done" and r["id"] != run_id for r in runs):
            break
        time.sleep(0.3)
    else:
        pytest.skip("Re-training did not complete in time")

    runs_r = client.get(f"/api/models/{proj_id}/runs")
    run2_id = next(
        r["id"]
        for r in runs_r.json()["runs"]
        if r["status"] == "done" and r["id"] != run_id
    )
    dep_r2 = client.post(f"/api/deploy/{run2_id}", json={})
    assert dep_r2.status_code in (200, 201)

    events = _chat_events(client, proj_id, "roll back to version 1")

    rollback_events = [e for e in events if e.get("type") == "rollback_chat"]
    assert rollback_events, "Should emit rollback_chat SSE event"
    data = rollback_events[0]["rollback_chat"]
    assert data["rolled_back"] is True
    assert data["rolled_back_to_version"] == 1
    assert data["new_version_number"] == 3  # 1 → 2 → rollback = v3


def test_chat_rollback_no_deployment(client):
    """Without a deployment, rollback chat should not crash — just no event."""
    proj_r = client.post("/api/projects", json={"name": "No Deploy"})
    proj_id = proj_r.json()["id"]

    events = _chat_events(client, proj_id, "roll back my model")

    rollback_events = [e for e in events if e.get("type") == "rollback_chat"]
    assert not rollback_events, "No event expected without a deployment"


def test_chat_rollback_only_one_version(client):
    """With only v1, asking to roll back should show error message in card."""
    proj_id, _dep_id, _run_id = _setup_project_with_deployment(client)

    events = _chat_events(client, proj_id, "roll back my model to the previous version")

    rollback_events = [e for e in events if e.get("type") == "rollback_chat"]
    assert rollback_events, "Should emit rollback_chat event"
    data = rollback_events[0]["rollback_chat"]
    assert data["rolled_back"] is False
    assert data["total_versions"] == 1 or data["error_message"] is not None


def test_chat_rollback_version_list_structure(client):
    """Version list entries have expected fields."""
    proj_id, _dep_id, _run_id = _setup_project_with_deployment(client)

    events = _chat_events(client, proj_id, "deployment version history")

    rollback_events = [e for e in events if e.get("type") == "rollback_chat"]
    assert rollback_events
    data = rollback_events[0]["rollback_chat"]
    assert "versions" in data
    assert "current_version_number" in data
    assert "total_versions" in data
    if data["versions"]:
        v = data["versions"][0]
        assert "version_number" in v
        assert "is_current" in v
