"""Tests for batch prediction scheduling via chat.

Covers:
- _SCHEDULE_PATTERNS          — NL intent detection regex
- _extract_schedule_params()  — frequency / time / day extraction
- _build_schedule_description() — plain-English descriptions
- Chat handler integration     — create schedule, list schedules
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
    b"feat1,feat2,target\n"
    b"1.0,0.5,10.0\n2.0,1.0,20.0\n3.0,1.5,30.0\n4.0,2.0,40.0\n5.0,2.5,50.0\n"
    b"6.0,3.0,60.0\n7.0,3.5,70.0\n8.0,4.0,80.0\n9.0,4.5,90.0\n10.0,5.0,100.0\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chat_events(client, project_id: str, message: str) -> list[dict]:
    """Send a chat message with mocked Anthropic and return parsed SSE events."""
    with patch("anthropic.Anthropic") as mock_cls:
        mock_c = MagicMock()
        mock_cls.return_value = mock_c
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Scheduled."])
        mock_c.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": message, "project_id": project_id},
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    """TestClient backed by an isolated SQLite DB."""
    from main import app

    test_db = str(tmp_path / "test.db")
    orig_engine = db_module.engine
    db_module.engine = create_engine(
        f"sqlite:///{test_db}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(db_module.engine)
    db_module.create_db_and_tables()

    yield TestClient(app)

    db_module.engine = orig_engine


@pytest.fixture()
def deployed_project(client, tmp_path):
    """Create project, upload CSV, train a model, and deploy it."""
    proj = client.post("/api/projects", json={"name": "ScheduleTest"})
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
    )
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "target", "problem_type": "regression"},
    )

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
        time.sleep(0.3)
    assert run["status"] == "done"

    dep = client.post(f"/api/deploy/{run_id}")
    assert dep.status_code in (200, 201), dep.text

    return {"project_id": project_id, "deployment_id": dep.json()["id"]}


# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


class TestSchedulePatterns:
    def test_schedule_daily_predictions(self):
        from api.chat import _SCHEDULE_PATTERNS
        assert _SCHEDULE_PATTERNS.search("schedule daily predictions at 9am")

    def test_set_up_daily_schedule(self):
        from api.chat import _SCHEDULE_PATTERNS
        assert _SCHEDULE_PATTERNS.search("set up a daily prediction schedule")

    def test_create_weekly_batch_prediction_schedule(self):
        from api.chat import _SCHEDULE_PATTERNS
        assert _SCHEDULE_PATTERNS.search("create a weekly batch prediction schedule")

    def test_run_model_every_day(self):
        from api.chat import _SCHEDULE_PATTERNS
        assert _SCHEDULE_PATTERNS.search("run my model every day at 8am")

    def test_run_every_monday(self):
        from api.chat import _SCHEDULE_PATTERNS
        assert _SCHEDULE_PATTERNS.search("run my batch every Monday at 9am")

    def test_batch_predictions_every_month(self):
        from api.chat import _SCHEDULE_PATTERNS
        assert _SCHEDULE_PATTERNS.search("batch predictions every month")

    def test_configure_monthly_schedule(self):
        from api.chat import _SCHEDULE_PATTERNS
        assert _SCHEDULE_PATTERNS.search("configure a monthly prediction schedule")

    def test_list_schedules(self):
        from api.chat import _SCHEDULE_PATTERNS
        assert _SCHEDULE_PATTERNS.search("show my batch schedules")

    def test_negative_train_model(self):
        from api.chat import _SCHEDULE_PATTERNS
        assert not _SCHEDULE_PATTERNS.search("train a new model")

    def test_negative_make_prediction(self):
        from api.chat import _SCHEDULE_PATTERNS
        assert not _SCHEDULE_PATTERNS.search("make a prediction for units=100")


# ---------------------------------------------------------------------------
# Unit tests: _extract_schedule_params()
# ---------------------------------------------------------------------------


class TestExtractScheduleParams:
    def test_daily_default(self):
        from api.chat import _extract_schedule_params
        p = _extract_schedule_params("schedule daily predictions")
        assert p["frequency"] == "daily"
        assert p["day_of_week"] is None
        assert p["day_of_month"] is None

    def test_daily_time_at_9am(self):
        from api.chat import _extract_schedule_params
        p = _extract_schedule_params("run model every day at 9am")
        assert p["frequency"] == "daily"
        assert p["run_hour"] == 9
        assert p["run_minute"] == 0

    def test_daily_time_2pm(self):
        from api.chat import _extract_schedule_params
        p = _extract_schedule_params("schedule batch predictions at 2pm")
        assert p["run_hour"] == 14

    def test_daily_time_with_minutes(self):
        from api.chat import _extract_schedule_params
        p = _extract_schedule_params("run model at 9:30am every day")
        assert p["run_hour"] == 9
        assert p["run_minute"] == 30

    def test_weekly_monday(self):
        from api.chat import _extract_schedule_params
        p = _extract_schedule_params("run my model every Monday at 9am")
        assert p["frequency"] == "weekly"
        assert p["day_of_week"] == 0

    def test_weekly_friday(self):
        from api.chat import _extract_schedule_params
        p = _extract_schedule_params("schedule batch predictions every Friday")
        assert p["frequency"] == "weekly"
        assert p["day_of_week"] == 4

    def test_monthly_defaults_to_1st(self):
        from api.chat import _extract_schedule_params
        p = _extract_schedule_params("schedule monthly batch predictions")
        assert p["frequency"] == "monthly"
        assert p["day_of_month"] == 1


# ---------------------------------------------------------------------------
# Unit tests: _build_schedule_description()
# ---------------------------------------------------------------------------


class TestBuildScheduleDescription:
    def test_daily(self):
        from api.chat import _build_schedule_description
        desc = _build_schedule_description("daily", 9, 0, None, None)
        assert desc == "Every day at 09:00 UTC"

    def test_weekly_wednesday(self):
        from api.chat import _build_schedule_description
        desc = _build_schedule_description("weekly", 14, 30, 2, None)
        assert "Wednesday" in desc
        assert "14:30 UTC" in desc

    def test_monthly_1st(self):
        from api.chat import _build_schedule_description
        desc = _build_schedule_description("monthly", 8, 0, None, 1)
        assert "1st" in desc
        assert "08:00 UTC" in desc


# ---------------------------------------------------------------------------
# Integration: chat handler creates and lists schedules
# ---------------------------------------------------------------------------


def test_chat_creates_daily_schedule(client, deployed_project):
    """A chat message with schedule intent creates a BatchSchedule record."""
    events = _chat_events(
        client, deployed_project["project_id"],
        "schedule daily batch predictions at 9am"
    )
    types = [e.get("type") for e in events]
    assert "schedule_set" in types, f"Expected schedule_set in {types}"

    se = next(e for e in events if e.get("type") == "schedule_set")["schedule_set"]
    assert se["action"] == "created"
    assert se["frequency"] == "daily"
    assert "description" in se


def test_chat_schedule_has_next_run(client, deployed_project):
    """Created schedule includes next_run timestamp."""
    events = _chat_events(
        client, deployed_project["project_id"],
        "schedule batch predictions every day"
    )
    se_events = [e for e in events if e.get("type") == "schedule_set"]
    assert len(se_events) == 1
    assert se_events[0]["schedule_set"]["next_run"] is not None


def test_chat_no_schedule_without_deployment(client):
    """No schedule_set event is emitted when no deployment exists."""
    proj = client.post("/api/projects", json={"name": "NoDepProj"})
    project_id = proj.json()["id"]
    events = _chat_events(client, project_id, "schedule daily predictions at 9am")
    types = [e.get("type") for e in events]
    assert "schedule_set" not in types


def test_chat_list_schedules(client, deployed_project):
    """'show my schedules' emits a list action event."""
    events = _chat_events(
        client, deployed_project["project_id"],
        "show my batch schedules"
    )
    types = [e.get("type") for e in events]
    assert "schedule_set" in types, f"Expected schedule_set in {types}"
    se = next(e for e in events if e.get("type") == "schedule_set")["schedule_set"]
    assert se["action"] == "list"
    assert "count" in se


def test_chat_schedule_weekly_monday(client, deployed_project):
    """Weekly schedule for Monday sets correct day_of_week."""
    events = _chat_events(
        client, deployed_project["project_id"],
        "run my model every Monday at 8am"
    )
    se_events = [e for e in events if e.get("type") == "schedule_set"]
    assert len(se_events) == 1
    se = se_events[0]["schedule_set"]
    assert se["frequency"] == "weekly"
    assert se["day_of_week"] == 0
    assert se["run_hour"] == 8
