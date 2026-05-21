"""Tests for Feedback Accuracy Report via Chat — Track D perpetual.

Covers:
  - compute_feedback_accuracy_report pure function (regression, classification,
    weekly trend, trend direction, empty/no-data paths)
  - _FEEDBACK_ACCURACY_PATTERNS regex (positive + negative)
  - Chat handler integration (emits event, guard, empty-feedback path)
"""

from __future__ import annotations

import io
import json
import time
import unittest.mock as mock
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.analyzer import compute_feedback_accuracy_report

# ─── helpers ──────────────────────────────────────────────────────────────────


def _mk_feedback(
    prediction_log_id: str | None = None,
    actual_value: float | None = None,
    actual_label: str | None = None,
    is_correct: bool | None = None,
    created_at: datetime | None = None,
):
    return SimpleNamespace(
        prediction_log_id=prediction_log_id,
        actual_value=actual_value,
        actual_label=actual_label,
        is_correct=is_correct,
        created_at=created_at or datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
    )


def _mk_log(log_id: str = "log1", prediction_numeric: float | None = None):
    return SimpleNamespace(id=log_id, prediction_numeric=prediction_numeric)


_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
)


# ─── Unit tests: compute_feedback_accuracy_report ─────────────────────────────


class TestComputeFeedbackAccuracyReport:
    def test_empty_feedback_returns_no_data(self):
        result = compute_feedback_accuracy_report([], {}, "regression")
        assert result["status"] == "no_feedback"
        assert result["total_feedback"] == 0
        assert result["has_data"] is False
        assert "No feedback recorded" in result["summary"]

    def test_empty_feedback_classification(self):
        result = compute_feedback_accuracy_report([], {}, "classification")
        assert result["status"] == "no_feedback"
        assert result["has_data"] is False

    def test_regression_feedback_only_no_logs(self):
        fb = _mk_feedback(actual_value=100.0)
        result = compute_feedback_accuracy_report([fb], {}, "regression")
        assert result["status"] == "feedback_only"
        assert result["has_data"] is False
        assert result["total_feedback"] == 1

    def test_regression_computes_mae(self):
        fb1 = _mk_feedback(prediction_log_id="log1", actual_value=100.0)
        fb2 = _mk_feedback(prediction_log_id="log2", actual_value=200.0)
        logs = {"log1": _mk_log("log1", 90.0), "log2": _mk_log("log2", 180.0)}
        result = compute_feedback_accuracy_report([fb1, fb2], logs, "regression")
        assert result["status"] == "computed"
        assert result["has_data"] is True
        assert result["paired_count"] == 2
        # MAE = (|100-90| + |200-180|) / 2 = 15
        assert abs(result["mae"] - 15.0) < 0.01
        assert result["total_feedback"] == 2

    def test_regression_pct_error_and_verdict_excellent(self):
        # actual = 100, predicted = 99 → MAE = 1, avg_actual = 100 → pct_error = 1%
        fb = _mk_feedback(prediction_log_id="log1", actual_value=100.0)
        logs = {"log1": _mk_log("log1", 99.0)}
        result = compute_feedback_accuracy_report([fb], logs, "regression")
        assert result["verdict"] == "excellent"
        assert result["pct_error"] < 5

    def test_regression_verdict_poor_when_high_error(self):
        # actual = 100, predicted = 200 → 100% error
        fb = _mk_feedback(prediction_log_id="log1", actual_value=100.0)
        logs = {"log1": _mk_log("log1", 200.0)}
        result = compute_feedback_accuracy_report([fb], logs, "regression")
        assert result["verdict"] == "poor"

    def test_regression_weekly_trend_computed(self):
        # Two feedback entries two weeks apart
        w1 = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)  # Week 1
        w2 = datetime(2026, 1, 19, 12, 0, tzinfo=timezone.utc)  # Week 3
        fb1 = _mk_feedback(prediction_log_id="log1", actual_value=100.0, created_at=w1)
        fb2 = _mk_feedback(prediction_log_id="log2", actual_value=200.0, created_at=w2)
        logs = {
            "log1": _mk_log("log1", 95.0),
            "log2": _mk_log("log2", 190.0),
        }
        result = compute_feedback_accuracy_report([fb1, fb2], logs, "regression")
        assert result["weekly_trend"] is not None
        assert len(result["weekly_trend"]) >= 1
        assert "mae" in result["weekly_trend"][0]

    def test_classification_feedback_only_no_is_correct(self):
        fb = _mk_feedback(actual_label="yes")
        result = compute_feedback_accuracy_report([fb], {}, "classification")
        assert result["status"] == "feedback_only"
        assert result["has_data"] is False

    def test_classification_computes_accuracy(self):
        fbs = [
            _mk_feedback(is_correct=True),
            _mk_feedback(is_correct=True),
            _mk_feedback(is_correct=False),
        ]
        result = compute_feedback_accuracy_report(fbs, {}, "classification")
        assert result["status"] == "computed"
        assert result["has_data"] is True
        assert result["correct_count"] == 2
        assert result["incorrect_count"] == 1
        assert result["rated_count"] == 3
        assert abs(result["accuracy"] - 2 / 3) < 0.01
        assert abs(result["accuracy_pct"] - 66.7) < 0.2

    def test_classification_verdict_excellent_at_high_accuracy(self):
        fbs = [_mk_feedback(is_correct=True) for _ in range(10)]
        result = compute_feedback_accuracy_report(fbs, {}, "classification")
        assert result["verdict"] == "excellent"

    def test_classification_verdict_poor_below_60pct(self):
        fbs = [_mk_feedback(is_correct=True)] + [
            _mk_feedback(is_correct=False) for _ in range(4)
        ]
        result = compute_feedback_accuracy_report(fbs, {}, "classification")
        assert result["verdict"] == "poor"

    def test_classification_weekly_trend(self):
        w1 = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
        w2 = datetime(2026, 1, 19, 12, 0, tzinfo=timezone.utc)
        fbs = [
            _mk_feedback(is_correct=True, created_at=w1),
            _mk_feedback(is_correct=False, created_at=w2),
        ]
        result = compute_feedback_accuracy_report(fbs, {}, "classification")
        assert result["weekly_trend"] is not None
        assert len(result["weekly_trend"]) >= 1
        assert "accuracy" in result["weekly_trend"][0]

    def test_classification_trend_direction_improving(self):
        # first week: 60%, second week: 90% → improving
        w1 = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)
        w2 = datetime(2026, 1, 19, 12, 0, tzinfo=timezone.utc)
        fbs = [
            _mk_feedback(is_correct=True, created_at=w1),
            _mk_feedback(is_correct=False, created_at=w1),
            _mk_feedback(is_correct=False, created_at=w1),
            _mk_feedback(is_correct=True, created_at=w2),
            _mk_feedback(is_correct=True, created_at=w2),
            _mk_feedback(is_correct=True, created_at=w2),
            _mk_feedback(is_correct=True, created_at=w2),
            _mk_feedback(is_correct=True, created_at=w2),
            _mk_feedback(is_correct=True, created_at=w2),
            _mk_feedback(is_correct=True, created_at=w2),
        ]
        result = compute_feedback_accuracy_report(fbs, {}, "classification")
        assert result["trend_direction"] == "improving"

    def test_summary_contains_key_info(self):
        fbs = [
            _mk_feedback(is_correct=True),
            _mk_feedback(is_correct=True),
            _mk_feedback(is_correct=False),
        ]
        result = compute_feedback_accuracy_report(fbs, {}, "classification")
        assert "2 of 3" in result["summary"] or "66" in result["summary"]


# ─── Regex tests: _FEEDBACK_ACCURACY_PATTERNS ─────────────────────────────────


def _get_pattern():
    from api.chat import _FEEDBACK_ACCURACY_PATTERNS

    return _FEEDBACK_ACCURACY_PATTERNS


@pytest.mark.parametrize(
    "msg",
    [
        "how accurate have my predictions been",
        "how accurate were my predictions",
        "show me real-world accuracy",
        "show me the feedback accuracy report",
        "feedback accuracy summary",
        "how many predictions were correct",
        "how often were my predictions right",
        "did my predictions match reality",
        "real-world model accuracy",
        "production accuracy",
        "live performance report",
        "how well did my model perform in practice",
        "how well has my model done in reality",
        "ground truth accuracy report",
        "actual accuracy report",
        "prediction accuracy feedback stats",
        "were my predictions accurate",
        "how good is my model in production",
    ],
)
def test_feedback_accuracy_pattern_positive(msg):
    pattern = _get_pattern()
    assert pattern.search(msg), f"Pattern should match: {msg!r}"


@pytest.mark.parametrize(
    "msg",
    [
        "train a model",
        "show me a bar chart",
        "what is the correlation",
        "group by region",
        "how is my model confidence trending",
        "show me the latency",
        "how many predictions today",
    ],
)
def test_feedback_accuracy_pattern_negative(msg):
    pattern = _get_pattern()
    assert not pattern.search(msg), f"Pattern should NOT match: {msg!r}"


# ─── Chat handler integration tests ──────────────────────────────────────────


def _make_app_with_db(db_url: str):
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    db_module._engine = engine

    from main import app

    return app


def _mock_stream(text="Feedback accuracy report loaded."):
    ms = mock.MagicMock()
    ms.__enter__ = mock.MagicMock(return_value=ms)
    ms.__exit__ = mock.MagicMock(return_value=False)
    ms.text_stream = iter([text])
    return ms


def _parse_events(resp) -> list[dict]:
    events = []
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            try:
                events.append(json.loads(line[5:]))
            except json.JSONDecodeError:
                pass
    return events


@pytest.fixture(scope="module")
def far_app(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("far_db")
    return _make_app_with_db(f"sqlite:///{tmpdir}/far.db")


@pytest.fixture(scope="module")
async def far_client(far_app):
    async with AsyncClient(
        transport=ASGITransport(app=far_app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture(scope="module")
async def far_project_id(far_client):
    r = await far_client.post("/api/projects", json={"name": "FAR_test"})
    return r.json()["id"]


@pytest.fixture(scope="module")
async def far_dataset_id(far_client, far_project_id):
    r = await far_client.post(
        "/api/data/upload",
        files={"file": ("data.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        data={"project_id": far_project_id},
    )
    assert r.status_code == 201, r.text
    return r.json()["dataset_id"]


@pytest.fixture(scope="module")
async def far_feature_set_id(far_client, far_dataset_id):
    r = await far_client.post(
        f"/api/features/{far_dataset_id}/apply",
        json={"transformations": []},
    )
    assert r.status_code == 201, r.text
    fs_id = r.json()["feature_set_id"]
    await far_client.post(
        f"/api/features/{far_dataset_id}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )
    return fs_id


@pytest.fixture(scope="module")
async def far_trained_run_id(far_client, far_project_id, far_feature_set_id):
    r = await far_client.post(
        f"/api/models/{far_project_id}/train",
        json={
            "algorithms": ["linear_regression"],
            "feature_set_id": far_feature_set_id,
        },
    )
    assert r.status_code == 202, r.text
    run_id = r.json()["model_run_ids"][0]
    for _ in range(30):
        resp = await far_client.get(f"/api/models/{far_project_id}/runs")
        run = next((x for x in resp.json().get("runs", []) if x["id"] == run_id), None)
        if run and run["status"] == "done":
            return run_id
        time.sleep(0.3)
    pytest.skip("Training did not complete")


@pytest.fixture(scope="module")
async def far_deployment_id(far_client, far_trained_run_id):
    r = await far_client.post(f"/api/deploy/{far_trained_run_id}", json={})
    assert r.status_code == 201, r.text
    return r.json()["id"]


@pytest.mark.anyio
async def test_feedback_accuracy_chat_emits_event(
    far_client, far_project_id, far_deployment_id
):
    """Handler emits feedback_accuracy_report SSE event when deployment exists."""
    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = _mock_stream()
        resp = await far_client.post(
            f"/api/chat/{far_project_id}",
            json={"message": "show me my feedback accuracy report", "session_id": "f1"},
        )

    events = _parse_events(resp)
    types = [e.get("type") for e in events]
    assert "feedback_accuracy_report" in types, f"Expected event. Got: {types}"
    ev = next(e for e in events if e.get("type") == "feedback_accuracy_report")
    data = ev["feedback_accuracy_report"]
    assert "status" in data
    assert "summary" in data
    assert "deployment_id" in data
    assert data["deployment_id"] == far_deployment_id


@pytest.mark.anyio
async def test_feedback_accuracy_chat_no_deployment_guard(far_client):
    """Handler does NOT emit event when no deployment exists."""
    r = await far_client.post("/api/projects", json={"name": "FAR_no_deploy"})
    nd_project_id = r.json()["id"]

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = _mock_stream()
        resp = await far_client.post(
            f"/api/chat/{nd_project_id}",
            json={
                "message": "how accurate have my predictions been",
                "session_id": "f2",
            },
        )

    events = _parse_events(resp)
    types = [e.get("type") for e in events]
    assert "feedback_accuracy_report" not in types


@pytest.mark.anyio
async def test_feedback_accuracy_chat_no_feedback_state(
    far_client, far_project_id, far_deployment_id
):
    """With a deployment but no FeedbackRecords, emits status=no_feedback."""
    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = _mock_stream()
        resp = await far_client.post(
            f"/api/chat/{far_project_id}",
            json={
                "message": "how accurate have my real-world predictions been",
                "session_id": "f3",
            },
        )

    events = _parse_events(resp)
    ev = next((e for e in events if e.get("type") == "feedback_accuracy_report"), None)
    assert ev is not None, "Expected feedback_accuracy_report event"
    assert ev["feedback_accuracy_report"]["status"] == "no_feedback"
