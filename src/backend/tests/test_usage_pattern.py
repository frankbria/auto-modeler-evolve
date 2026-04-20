"""Tests for prediction usage pattern analysis (Track D perpetual)."""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

import db as db_module
from api.chat import _USAGE_PATTERN_PATTERNS
from core.analyzer import compute_usage_pattern


# ---------------------------------------------------------------------------
# Pattern tests
# ---------------------------------------------------------------------------


class TestUsagePatternPatterns:
    POSITIVE = [
        "when is my model busiest?",
        "when is my api most active?",
        "peak usage hours for my model",
        "peak traffic for the endpoint",
        "hourly usage pattern",
        "daily prediction distribution",
        "usage heatmap",
        "what time do people use my model?",
        "when is my api getting called the most?",
        "busiest hours for my endpoint",
        "quietest periods for my model",
        "maintenance window for my api",
    ]
    NEGATIVE = [
        "train a model",
        "quota runway",
        "what is my accuracy",
        "show me the confusion matrix",
        "how many predictions did I make?",
        "is my model drifting?",
        "set rate limit",
        "hello",
        "explain prediction for row 5",
    ]

    @pytest.mark.parametrize("msg", POSITIVE)
    def test_positive(self, msg):
        assert _USAGE_PATTERN_PATTERNS.search(msg), f"Should match: {msg!r}"

    @pytest.mark.parametrize("msg", NEGATIVE)
    def test_negative(self, msg):
        assert not _USAGE_PATTERN_PATTERNS.search(msg), f"Should not match: {msg!r}"


# ---------------------------------------------------------------------------
# compute_usage_pattern unit tests
# ---------------------------------------------------------------------------


def _make_log(hour: int, weekday: int = 0):
    """Create a minimal mock prediction log at a given UTC hour and weekday."""
    delta_days = weekday
    ts = datetime(2024, 1, 1 + delta_days, hour, 0, 0)

    class FakeLog:
        created_at = ts

    return FakeLog()


class TestComputeUsagePattern:
    def test_empty_returns_zero_counts(self):
        result = compute_usage_pattern([])
        assert result["total_predictions"] == 0
        assert sum(result["hour_counts"]) == 0
        assert sum(result["day_counts"]) == 0
        assert result["peak_hour"] is None
        assert "No predictions recorded" in result["summary"]

    def test_single_log_peak_detected(self):
        logs = [_make_log(hour=14, weekday=0)]
        result = compute_usage_pattern(logs)
        assert result["total_predictions"] == 1
        assert result["peak_hour"] == 14
        assert result["peak_hour_count"] == 1

    def test_hour_counts_length_24(self):
        logs = [_make_log(h) for h in range(10)]
        result = compute_usage_pattern(logs)
        assert len(result["hour_counts"]) == 24

    def test_day_counts_length_7(self):
        logs = [_make_log(9, wd) for wd in range(5)]
        result = compute_usage_pattern(logs)
        assert len(result["day_counts"]) == 7

    def test_peak_hour_correct(self):
        # Flood hour 9 with 5 logs, others get 1 each
        logs = [_make_log(9) for _ in range(5)] + [_make_log(h) for h in [6, 12, 18]]
        result = compute_usage_pattern(logs)
        assert result["peak_hour"] == 9
        assert result["peak_hour_count"] == 5

    def test_peak_day_correct(self):
        # 3 on Monday (0), 1 on others
        logs = [_make_log(9, 0) for _ in range(3)] + [_make_log(9, d) for d in [1, 2]]
        result = compute_usage_pattern(logs)
        assert result["peak_day"] == 0
        assert result["peak_day_name"] == "Monday"

    def test_quiet_hours_identified(self):
        # Peak at hour 10 (high count), most others zero
        logs = [_make_log(10) for _ in range(20)]
        result = compute_usage_pattern(logs)
        # Hours other than 10 have 0 predictions → should be quiet
        assert 0 in result["quiet_hours"]
        assert 10 not in result["quiet_hours"]

    def test_total_predictions_count(self):
        logs = [_make_log(h % 24) for h in range(30)]
        result = compute_usage_pattern(logs)
        assert result["total_predictions"] == 30

    def test_summary_mentions_peak(self):
        logs = [_make_log(9) for _ in range(5)]
        result = compute_usage_pattern(logs)
        assert "9am" in result["summary"]

    def test_busiest_period_morning(self):
        # All predictions in morning hours
        logs = [_make_log(h) for h in [7, 8, 9, 10, 11]]
        result = compute_usage_pattern(logs)
        assert result["busiest_period"] is not None
        assert "morning" in result["busiest_period"]

    def test_day_names_returned(self):
        logs = [_make_log(9, 0)]
        result = compute_usage_pattern(logs)
        assert len(result["day_names"]) == 7
        assert result["day_names"][0] == "Mon"

    def test_invalid_created_at_skipped(self):
        """Logs with None or bad created_at are ignored gracefully."""

        class BadLog:
            created_at = None

        result = compute_usage_pattern([BadLog()])
        assert result["total_predictions"] == 0


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.ab_test  # noqa
    import models.batch_schedule  # noqa
    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.dataset_filter  # noqa
    import models.deployment  # noqa
    import models.deployment_preset  # noqa
    import models.deployment_version  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa
    import models.webhook_config  # noqa

    SQLModel.metadata.create_all(db_module.engine)
    from main import app

    return TestClient(app)


def _create_deployment(client, project_id: str) -> str:
    run_id = str(uuid.uuid4())
    dep_id = str(uuid.uuid4())
    with Session(db_module.engine) as sess:
        from models.deployment import Deployment
        from models.model_run import ModelRun

        sess.add(
            ModelRun(
                id=run_id,
                project_id=project_id,
                feature_set_id=str(uuid.uuid4()),
                algorithm="linear_regression",
                status="done",
                metrics=json.dumps({"r2": 0.8}),
                summary="LR",
            )
        )
        sess.add(
            Deployment(
                id=dep_id,
                model_run_id=run_id,
                project_id=project_id,
                endpoint_path=f"/api/predict/{dep_id}",
                dashboard_url=f"/predict/{dep_id}",
                is_active=True,
                algorithm="linear_regression",
                problem_type="regression",
                feature_names=json.dumps(["units"]),
                target_column="revenue",
                metrics=json.dumps({"r2": 0.8}),
            )
        )
        sess.commit()
    return dep_id


class TestUsagePatternEndpoint:
    def test_returns_404_for_unknown_deployment(self, client):
        r = client.get("/api/deploy/does-not-exist/usage-pattern")
        assert r.status_code == 404

    def test_returns_usage_pattern_structure(self, client):
        proj = client.post("/api/projects", json={"name": "UP"}).json()
        dep_id = _create_deployment(client, proj["id"])
        r = client.get(f"/api/deploy/{dep_id}/usage-pattern")
        assert r.status_code == 200
        data = r.json()
        assert "hour_counts" in data
        assert len(data["hour_counts"]) == 24
        assert "day_counts" in data
        assert len(data["day_counts"]) == 7
        assert data["deployment_id"] == dep_id
        assert "total_predictions" in data
        assert "summary" in data

    def test_empty_deployment_no_crash(self, client):
        proj = client.post("/api/projects", json={"name": "Empty"}).json()
        dep_id = _create_deployment(client, proj["id"])
        r = client.get(f"/api/deploy/{dep_id}/usage-pattern")
        assert r.status_code == 200
        assert r.json()["total_predictions"] == 0


# ---------------------------------------------------------------------------
# Chat integration tests
# ---------------------------------------------------------------------------


@contextmanager
def _mock_anthropic():
    with mock.patch("anthropic.Anthropic") as m:
        mock_cli = mock.MagicMock()
        m.return_value = mock_cli
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Here are your usage patterns."])
        mock_cli.messages.stream.return_value = mock_stream
        yield


class TestUsagePatternChatHandler:
    def test_no_event_without_deployment(self, client):
        proj = client.post("/api/projects", json={"name": "NoDeploy"}).json()
        pid = proj["id"]
        with _mock_anthropic():
            with client.stream(
                "POST",
                f"/api/chat/{pid}",
                json={"message": "when is my model busiest?"},
            ) as resp:
                events = [
                    json.loads(line[6:])
                    for line in resp.iter_lines()
                    if line.startswith("data:")
                ]
        types = [e.get("type") for e in events]
        assert "usage_pattern" not in types

    def test_emits_usage_pattern_event(self, client):
        proj = client.post("/api/projects", json={"name": "WithDeploy"}).json()
        pid = proj["id"]
        _create_deployment(client, pid)
        with _mock_anthropic():
            with client.stream(
                "POST",
                f"/api/chat/{pid}",
                json={"message": "when is my model busiest?"},
            ) as resp:
                events = [
                    json.loads(line[6:])
                    for line in resp.iter_lines()
                    if line.startswith("data:")
                ]
        up_events = [e for e in events if e.get("type") == "usage_pattern"]
        assert len(up_events) == 1

    def test_event_required_fields(self, client):
        proj = client.post("/api/projects", json={"name": "Fields"}).json()
        pid = proj["id"]
        _create_deployment(client, pid)
        with _mock_anthropic():
            with client.stream(
                "POST",
                f"/api/chat/{pid}",
                json={"message": "hourly usage pattern"},
            ) as resp:
                events = [
                    json.loads(line[6:])
                    for line in resp.iter_lines()
                    if line.startswith("data:")
                ]
        up_events = [e for e in events if e.get("type") == "usage_pattern"]
        assert up_events, "Expected usage_pattern event"
        payload = up_events[0]["usage_pattern"]
        for field in ("deployment_id", "hour_counts", "day_counts", "total_predictions", "summary"):
            assert field in payload, f"Missing field: {field}"
        assert len(payload["hour_counts"]) == 24
        assert len(payload["day_counts"]) == 7
