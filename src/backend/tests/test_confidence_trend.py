"""Tests for Confidence Trend Analysis — Track D perpetual.

Covers:
  - compute_confidence_trend pure function (daily stats, trend direction, slope)
  - GET /api/deploy/{deployment_id}/confidence-trend REST endpoint
  - _CONFIDENCE_TREND_PATTERNS regex (positive + negative)
  - Chat handler integration (emits event, required fields, no-deployment guard)
"""

from __future__ import annotations

import io
import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.analyzer import compute_confidence_trend


# ─── helpers ──────────────────────────────────────────────────────────────────


def _mk_log(
    confidence: float | None = None,
    created_at: datetime | None = None,
):
    return SimpleNamespace(
        confidence=confidence,
        response_ms=None,
        created_at=created_at or datetime.now(timezone.utc),
    )


NOW = datetime(2026, 4, 21, 12, 0, 0, tzinfo=timezone.utc)

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
)


# ─── Unit tests: compute_confidence_trend ─────────────────────────────────────


class TestComputeConfidenceTrend:
    def test_empty_logs_returns_no_data(self):
        result = compute_confidence_trend([], window_days=30, now_utc=NOW)
        assert result["has_data"] is False
        assert result["daily_stats"] == []
        assert result["overall_avg"] is None
        assert result["sample_count"] == 0
        assert result["trend_direction"] == "stable"

    def test_single_day_no_trend(self):
        logs = [_mk_log(0.9, NOW - timedelta(hours=1))]
        result = compute_confidence_trend(logs, window_days=30, now_utc=NOW)
        assert result["has_data"] is True
        assert result["sample_count"] == 1
        assert result["trend_direction"] == "stable"
        assert len(result["daily_stats"]) == 1
        assert result["daily_stats"][0]["avg_confidence"] == pytest.approx(90.0, abs=0.5)

    def test_logs_outside_window_excluded(self):
        old_log = _mk_log(0.5, NOW - timedelta(days=60))
        new_log = _mk_log(0.9, NOW - timedelta(days=1))
        result = compute_confidence_trend(
            [old_log, new_log], window_days=30, now_utc=NOW
        )
        assert result["sample_count"] == 1
        assert result["overall_avg"] == pytest.approx(90.0, abs=0.5)

    def test_no_confidence_values_returns_no_data(self):
        logs = [_mk_log(None, NOW - timedelta(days=1))]
        result = compute_confidence_trend(logs, window_days=30, now_utc=NOW)
        assert result["has_data"] is False
        assert result["sample_count"] == 0

    def test_improving_trend_detected(self):
        # Confidence increases day by day: 60%, 70%, 80%, 90%
        logs = [
            _mk_log(0.60, NOW - timedelta(days=3, hours=12)),
            _mk_log(0.70, NOW - timedelta(days=2, hours=12)),
            _mk_log(0.80, NOW - timedelta(days=1, hours=12)),
            _mk_log(0.90, NOW - timedelta(hours=12)),
        ]
        result = compute_confidence_trend(logs, window_days=30, now_utc=NOW)
        assert result["trend_direction"] == "improving"
        assert result["trend_rate_per_day"] > 0

    def test_declining_trend_detected(self):
        # Confidence decreases: 90%, 80%, 70%, 60%
        logs = [
            _mk_log(0.90, NOW - timedelta(days=3, hours=12)),
            _mk_log(0.80, NOW - timedelta(days=2, hours=12)),
            _mk_log(0.70, NOW - timedelta(days=1, hours=12)),
            _mk_log(0.60, NOW - timedelta(hours=12)),
        ]
        result = compute_confidence_trend(logs, window_days=30, now_utc=NOW)
        assert result["trend_direction"] == "declining"
        assert result["trend_rate_per_day"] < 0

    def test_stable_trend_detected(self):
        # Confidence stays flat: 80% each day
        logs = [_mk_log(0.80, NOW - timedelta(days=i, hours=12)) for i in range(1, 5)]
        result = compute_confidence_trend(logs, window_days=30, now_utc=NOW)
        assert result["trend_direction"] == "stable"

    def test_daily_stats_sorted_oldest_first(self):
        logs = [
            _mk_log(0.80, NOW - timedelta(days=2, hours=12)),
            _mk_log(0.90, NOW - timedelta(hours=12)),
        ]
        result = compute_confidence_trend(logs, window_days=30, now_utc=NOW)
        dates = [d["date"] for d in result["daily_stats"]]
        assert dates == sorted(dates)

    def test_peak_and_low_day_identified(self):
        logs = [
            _mk_log(0.50, NOW - timedelta(days=2, hours=12)),
            _mk_log(0.95, NOW - timedelta(days=1, hours=12)),
        ]
        result = compute_confidence_trend(logs, window_days=30, now_utc=NOW)
        assert result["peak_value"] == pytest.approx(95.0, abs=0.5)
        assert result["low_value"] == pytest.approx(50.0, abs=0.5)

    def test_summary_contains_direction(self):
        logs = [_mk_log(0.80, NOW - timedelta(hours=6))]
        result = compute_confidence_trend(logs, window_days=30, now_utc=NOW)
        assert "stable" in result["summary"].lower() or "improving" in result["summary"].lower() or "declining" in result["summary"].lower()


# ─── Regex tests: _CONFIDENCE_TREND_PATTERNS ─────────────────────────────────


_POSITIVE_PHRASES = [
    "how is my model confidence trending?",
    "show confidence trend",
    "confidence over time",
    "are my predictions getting less confident?",
    "is my confidence score declining?",
    "are confidence scores dropping?",
    "model confidence history",
    "prediction confidence chart",
    "confidence trend analysis",
    "track my daily confidence level",
    "are my predictions getting more reliable?",
    "how reliable were my predictions last 7 days",
]

_NEGATIVE_PHRASES = [
    "what is the SLA for my deployment",
    "how many predictions today",
    "usage pattern for my model",
    "show me the prediction audit",
    "covariate drift in my model",
    "are my inputs drifting",
    "prediction volume this week",
]


class TestConfidenceTrendPatterns:
    @pytest.fixture(autouse=True)
    def _import_pattern(self):
        from api.chat import _CONFIDENCE_TREND_PATTERNS

        self.pat = _CONFIDENCE_TREND_PATTERNS

    @pytest.mark.parametrize("phrase", _POSITIVE_PHRASES)
    def test_positive(self, phrase):
        assert self.pat.search(phrase), f"Expected match for: {phrase!r}"

    @pytest.mark.parametrize("phrase", _NEGATIVE_PHRASES)
    def test_negative(self, phrase):
        assert not self.pat.search(phrase), f"Expected NO match for: {phrase!r}"


# ─── async fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.dataset_filter  # noqa
    import models.deployment  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.deploy as deploy_module
    import api.models as models_module
    from main import app

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"
    models_module.MODELS_DIR = tmp_path / "models"

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
async def project_id(ac):
    r = await ac.post("/api/projects", json={"name": "ct_test"})
    return r.json()["id"]


@pytest.fixture()
async def dataset_id(ac, project_id):
    r = await ac.post(
        "/api/data/upload",
        files={"file": ("data.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        data={"project_id": project_id},
    )
    assert r.status_code == 201, r.text
    return r.json()["dataset_id"]


@pytest.fixture()
async def feature_set_id(ac, dataset_id):
    r = await ac.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert r.status_code == 201, r.text
    fs_id = r.json()["feature_set_id"]
    await ac.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )
    return fs_id


@pytest.fixture()
async def trained_run_id(ac, project_id, feature_set_id):
    r = await ac.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"], "feature_set_id": feature_set_id},
    )
    assert r.status_code == 202, r.text
    run_id = r.json()["model_run_ids"][0]
    for _ in range(30):
        resp = await ac.get(f"/api/models/{project_id}/runs")
        run = next((x for x in resp.json().get("runs", []) if x["id"] == run_id), None)
        if run and run["status"] == "done":
            return run_id
        time.sleep(0.3)
    pytest.skip("Training did not complete")


@pytest.fixture()
async def deployment_id(ac, trained_run_id):
    r = await ac.post(f"/api/deploy/{trained_run_id}", json={})
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ─── REST endpoint tests ─────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_confidence_trend_no_predictions(ac, deployment_id):
    r = await ac.get(f"/api/deploy/{deployment_id}/confidence-trend")
    assert r.status_code == 200
    body = r.json()
    assert body["has_data"] is False
    assert body["deployment_id"] == deployment_id


@pytest.mark.anyio
async def test_confidence_trend_404_unknown(ac):
    r = await ac.get("/api/deploy/nonexistent-id/confidence-trend")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_confidence_trend_with_window_param(ac, deployment_id):
    r = await ac.get(f"/api/deploy/{deployment_id}/confidence-trend?window=7")
    assert r.status_code == 200
    assert "deployment_id" in r.json()


# ─── Chat handler integration tests ──────────────────────────────────────────


def _mock_stream(text="Confidence trend loaded."):
    import unittest.mock as mock

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


@pytest.mark.anyio
async def test_chat_confidence_trend_emits_event(ac, project_id, deployment_id):
    """Chat with matching phrase + deployment emits confidence_trend SSE event."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = _mock_stream()
        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "how is my model confidence trending?", "session_id": "t1"},
        )
    events = _parse_events(resp)
    types = [e.get("type") for e in events]
    assert "confidence_trend" in types, f"Expected confidence_trend event. Got: {types}"
    ct_event = next(e for e in events if e.get("type") == "confidence_trend")
    data = ct_event["confidence_trend"]
    assert "has_data" in data
    assert "trend_direction" in data
    assert "deployment_id" in data


@pytest.mark.anyio
async def test_chat_confidence_trend_no_deployment(ac):
    """Chat without deployment never fires confidence_trend event."""
    import unittest.mock as mock

    r = await ac.post("/api/projects", json={"name": "ct_no_deploy"})
    nd_project_id = r.json()["id"]

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = _mock_stream()
        resp = await ac.post(
            f"/api/chat/{nd_project_id}",
            json={"message": "show confidence trend", "session_id": "t2"},
        )
    events = _parse_events(resp)
    types = [e.get("type") for e in events]
    assert "confidence_trend" not in types
