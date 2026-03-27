"""Tests for chat-triggered what-if analysis: _WHATIF_CHAT_PATTERNS,
_detect_whatif_request(), the SSE chat handler, and the existing whatif endpoint."""

import io
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

# ---------------------------------------------------------------------------
# Sample CSV
# ---------------------------------------------------------------------------

SAMPLE_CSV = b"""product,region,units,revenue
Widget A,North,10,1200.5
Widget B,South,8,850.0
Widget A,East,18,2100.75
Widget C,West,4,450.25
Widget B,North,15,1650.0
Widget A,South,9,980.0
Widget C,North,11,1100.25
Widget B,East,16,1750.0
Widget A,West,20,2300.5
Widget C,South,6,620.75
Widget A,North,12,1300.0
Widget B,South,9,950.0
Widget A,East,20,2200.0
Widget C,West,5,520.0
Widget B,North,16,1700.0
Widget A,South,10,1050.0
Widget C,North,12,1150.0
Widget B,East,17,1800.0
Widget A,West,21,2350.0
Widget C,South,7,670.0
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    monkeypatch.setattr(db_module, "engine", engine)
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture()
def client():
    from main import app

    return TestClient(app)


@pytest.fixture()
def deployed_project(client):
    """Creates project → uploads CSV → applies features → trains model → deploys."""
    import time

    proj = client.post("/api/projects", json={"name": "WhatIfTest"})
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(f"/api/features/{dataset_id}/target", json={"target_column": "revenue"})

    train_resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    run_id = train_resp.json()["model_run_ids"][0]

    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next(r for r in runs if r["id"] == run_id)
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)
    assert run["status"] == "done"

    deploy_resp = client.post(f"/api/deploy/{run_id}")
    assert deploy_resp.status_code == 201
    deployment_id = deploy_resp.json()["id"]

    return {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "run_id": run_id,
        "deployment_id": deployment_id,
    }


# ---------------------------------------------------------------------------
# Unit: _WHATIF_CHAT_PATTERNS
# ---------------------------------------------------------------------------


def test_whatif_patterns_match():
    from api.chat import _WHATIF_CHAT_PATTERNS

    matching = [
        "what if revenue was 500?",
        "what would happen if units is 20?",
        "if I changed units to 30",
        "if units were 15, what would the prediction be?",
        "suppose revenue is 1000",
        "change units to 5",
        "what would my prediction be if units equals 10?",
        "how would the prediction change if region is North?",
    ]
    for msg in matching:
        assert _WHATIF_CHAT_PATTERNS.search(msg), f"Expected match: {msg!r}"


def test_whatif_patterns_no_match():
    from api.chat import _WHATIF_CHAT_PATTERNS

    non_matching = [
        "show me the top 10 customers",
        "compare 2023 vs 2024",
        "cluster my data",
        "tell me about the revenue column",
    ]
    for msg in non_matching:
        assert not _WHATIF_CHAT_PATTERNS.search(msg), f"Unexpected match: {msg!r}"


# ---------------------------------------------------------------------------
# Unit: _detect_whatif_request
# ---------------------------------------------------------------------------


def test_detect_whatif_simple_numeric():
    from api.chat import _detect_whatif_request

    result = _detect_whatif_request(
        "what if revenue was 500?", ["units", "revenue", "region"]
    )
    assert result is not None
    assert result["feature"] == "revenue"
    assert result["new_value"] == 500.0


def test_detect_whatif_change_to():
    from api.chat import _detect_whatif_request

    result = _detect_whatif_request("change units to 30", ["units", "revenue"])
    assert result is not None
    assert result["feature"] == "units"
    assert result["new_value"] == 30.0


def test_detect_whatif_equals_pattern():
    from api.chat import _detect_whatif_request

    result = _detect_whatif_request(
        "if units equals 15, what happens?", ["units", "revenue"]
    )
    assert result is not None
    assert result["feature"] == "units"
    assert result["new_value"] == 15.0


def test_detect_whatif_string_value():
    from api.chat import _detect_whatif_request

    result = _detect_whatif_request(
        "what if region is North?", ["units", "revenue", "region"]
    )
    assert result is not None
    assert result["feature"] == "region"
    assert result["new_value"] == "North"


def test_detect_whatif_underscore_feature():
    from api.chat import _detect_whatif_request

    result = _detect_whatif_request(
        "what if total revenue was 2000?", ["total_revenue", "units"]
    )
    assert result is not None
    assert result["feature"] == "total_revenue"
    assert result["new_value"] == 2000.0


def test_detect_whatif_double_multiplier():
    from api.chat import _detect_whatif_request

    result = _detect_whatif_request(
        "what if I doubled the units?", ["units", "revenue"]
    )
    assert result is not None
    assert result["feature"] == "units"
    assert isinstance(result["new_value"], str)
    assert "__multiply__2.0" in str(result["new_value"])


def test_detect_whatif_halve_multiplier():
    from api.chat import _detect_whatif_request

    result = _detect_whatif_request("halve the revenue", ["units", "revenue"])
    assert result is not None
    assert result["feature"] == "revenue"
    assert "__multiply__0.5" in str(result["new_value"])


def test_detect_whatif_returns_none_no_match():
    from api.chat import _detect_whatif_request

    result = _detect_whatif_request(
        "show me the top 10 customers", ["units", "revenue"]
    )
    assert result is None


def test_detect_whatif_returns_none_empty_features():
    from api.chat import _detect_whatif_request

    result = _detect_whatif_request("what if revenue was 500?", [])
    assert result is None


# ---------------------------------------------------------------------------
# Integration: whatif endpoint (existing REST, quick sanity check)
# ---------------------------------------------------------------------------


def test_whatif_endpoint_returns_result(client, deployed_project):
    deployment_id = deployed_project["deployment_id"]
    r = client.post(
        f"/api/predict/{deployment_id}/whatif",
        json={"base": {"units": 10.0}, "overrides": {"units": 20.0}},
    )
    assert r.status_code == 200
    data = r.json()
    assert "original_prediction" in data
    assert "modified_prediction" in data
    assert "direction" in data
    assert "summary" in data


def test_whatif_endpoint_404_inactive(client):
    r = client.post(
        "/api/predict/nonexistent-id/whatif",
        json={"base": {"units": 10.0}, "overrides": {"units": 20.0}},
    )
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Integration: chat handler emits whatif_result SSE event
# ---------------------------------------------------------------------------


def test_chat_whatif_sse_event(client, deployed_project):
    """When a deployed model exists and message matches what-if pattern,
    the SSE stream should contain a whatif_result event."""
    project_id = deployed_project["project_id"]

    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["The prediction would change."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": "what if units was 20?"},
        )

    assert response.status_code == 200
    lines = [line for line in response.text.split("\n") if line.startswith("data: ")]
    event_types = []
    for line in lines:
        try:
            parsed = json.loads(line[6:])
            event_types.append(parsed.get("type"))
        except json.JSONDecodeError:
            pass
    assert "whatif_result" in event_types, f"Expected whatif_result in {event_types}"


def test_chat_whatif_sse_event_no_deployment(client, deployed_project):
    """Without an active deployment, no whatif_result event is emitted."""
    # Use a fresh project with no deployment
    r = client.post("/api/projects/", json={"name": "NoDeploy"})
    project_id = r.json()["id"]

    with patch("anthropic.Anthropic") as mock_anthropic_cls:
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["No deployment available."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": "what if units was 20?"},
        )

    lines = [line for line in response.text.split("\n") if line.startswith("data: ")]
    event_types = []
    for line in lines:
        try:
            parsed = json.loads(line[6:])
            event_types.append(parsed.get("type"))
        except json.JSONDecodeError:
            pass
    assert "whatif_result" not in event_types
