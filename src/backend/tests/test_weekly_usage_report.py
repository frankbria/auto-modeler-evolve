"""Tests for Weekly Usage Report feature.

Covers:
- _WEEKLY_USAGE_PATTERNS regex — NL intent detection
- False-positive guards
- Chat handler integration — weekly_usage_report SSE event emitted
"""

import io
import json
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine
from unittest.mock import patch

import db as db_module

SAMPLE_CSV = (
    b"product,region,units,revenue\n"
    b"Widget A,North,10,1200.5\n"
    b"Widget B,South,8,850.0\n"
    b"Widget A,East,18,2100.75\n"
    b"Widget C,West,4,450.25\n"
    b"Widget B,North,15,1650.0\n"
    b"Widget A,South,9,980.0\n"
    b"Widget C,North,11,1100.25\n"
    b"Widget B,East,16,1750.0\n"
    b"Widget A,West,20,2300.5\n"
    b"Widget C,South,7,670.0\n"
    b"Widget A,North,12,1300.0\n"
    b"Widget B,South,9,950.0\n"
    b"Widget A,East,20,2200.0\n"
    b"Widget C,West,5,520.0\n"
    b"Widget B,North,16,1700.0\n"
    b"Widget A,South,10,1050.0\n"
    b"Widget C,North,12,1150.0\n"
    b"Widget B,East,17,1800.0\n"
    b"Widget A,West,21,2350.0\n"
    b"Widget C,South,7,670.0\n"
)


# ---------------------------------------------------------------------------
# Pattern tests (pure regex — no DB needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "weekly usage report",
        "weekly prediction report",
        "give me my weekly summary",
        "how am I doing this week",
        "how did I do this week",
        "this week's predictions",
        "this week's stats",
        "week-over-week prediction comparison",
        "week over week usage trend",
        "prediction trend this week",
        "how many predictions did I get this week",
        "how many predictions did I get last week",
        "usage report",
        "prediction report weekly",
    ],
)
def test_weekly_usage_patterns_match(phrase):
    """NL variants that should trigger the weekly usage report."""
    from api.chat import _WEEKLY_USAGE_PATTERNS

    assert _WEEKLY_USAGE_PATTERNS.search(phrase), (
        f"Pattern should match: {phrase!r}"
    )


@pytest.mark.parametrize(
    "phrase",
    [
        "train a random forest",
        "what is the best model",
        "show me the feature importance",
        "deploy my model",
        "upload a new dataset",
        "what is the accuracy",
        "explore my data",
    ],
)
def test_weekly_usage_patterns_no_false_positives(phrase):
    """Unrelated phrases must NOT trigger the weekly usage report."""
    from api.chat import _WEEKLY_USAGE_PATTERNS

    assert not _WEEKLY_USAGE_PATTERNS.search(phrase), (
        f"Pattern should NOT match: {phrase!r}"
    )


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models  # noqa — registers all tables

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.deploy as deploy_module
    import api.models as models_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"
    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    return TestClient(app)


def _build_deployed_project(client, project_name="Weekly Usage Test"):
    r = client.post("/api/projects", json={"name": project_name})
    project_id = r.json()["id"]

    r = client.post(
        "/api/data/upload",
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
        data={"project_id": project_id},
    )
    assert r.status_code in (200, 201), r.text
    dataset_id = r.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(f"/api/features/{dataset_id}/target", json={"target_column": "revenue"})

    r_train = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    assert r_train.status_code in (200, 202), r_train.text
    run_id = r_train.json()["model_run_ids"][0]

    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next((x for x in runs if x["id"] == run_id), None)
        if run and run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    run = next((x for x in runs if x["id"] == run_id), None)
    if not run or run["status"] != "done":
        pytest.skip("training did not complete")

    r_dep = client.post(f"/api/deploy/{run_id}")
    assert r_dep.status_code in (200, 201), r_dep.text
    deployment_id = r_dep.json()["id"]

    return project_id, deployment_id


def _chat(client, project_id, message):
    """Send a chat message and return parsed SSE events."""
    mock_stream = iter(["Here is your weekly usage report."])
    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream
        with client.stream(
            "POST",
            f"/api/chat/{project_id}",
            json={"message": message, "conversation_id": None},
            headers={"accept": "text/event-stream"},
        ) as resp:
            events = []
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    try:
                        events.append(json.loads(line[6:]))
                    except Exception:
                        pass
    return events


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_weekly_usage_report_event_emitted(client):
    """weekly_usage_report SSE event is emitted for 'weekly report' query."""
    project_id, _ = _build_deployed_project(client)
    events = _chat(client, project_id, "weekly usage report")

    weekly_events = [e for e in events if e.get("type") == "weekly_usage_report"]
    assert weekly_events, "Expected weekly_usage_report SSE event"

    data = weekly_events[0]["weekly_usage_report"]
    assert "this_week_count" in data
    assert "last_week_count" in data
    assert "trend" in data
    assert data["trend"] in ("up", "down", "flat")
    assert "by_day" in data
    assert len(data["by_day"]) == 7
    assert "top_input_patterns" in data
    assert "summary" in data


def test_weekly_usage_report_by_day_has_seven_entries(client):
    """by_day list contains exactly 7 entries with date and count."""
    project_id, _ = _build_deployed_project(client, "Day Entries Test")
    events = _chat(client, project_id, "how did I do this week")

    weekly_events = [e for e in events if e.get("type") == "weekly_usage_report"]
    assert weekly_events

    by_day = weekly_events[0]["weekly_usage_report"]["by_day"]
    assert len(by_day) == 7
    for entry in by_day:
        assert "date" in entry
        assert "count" in entry
        assert isinstance(entry["count"], int)
        assert entry["count"] >= 0


def test_weekly_usage_report_not_emitted_without_deployment(client):
    """weekly_usage_report is NOT emitted when no deployment exists."""
    r = client.post("/api/projects", json={"name": "No Deployment Project"})
    project_id = r.json()["id"]

    events = _chat(client, project_id, "weekly usage report")
    weekly_events = [e for e in events if e.get("type") == "weekly_usage_report"]
    assert not weekly_events, "Should not emit weekly_usage_report without a deployment"
