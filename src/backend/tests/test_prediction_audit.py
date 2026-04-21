"""Tests for Prediction Audit Report — Track D perpetual.

Covers:
  - compute_prediction_audit pure function (volume, confidence, SLA, quota, status)
  - GET /api/deploy/{deployment_id}/prediction-audit REST endpoint
  - _PRED_AUDIT_PATTERNS regex (positive + negative)
  - Chat handler integration (emits event, required fields, no-deployment guard)
"""

from __future__ import annotations

import io
import json
import re
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.analyzer import compute_prediction_audit


# ─── helpers ──────────────────────────────────────────────────────────────────


def _mk_log(
    created_at: datetime | None = None,
    confidence: float | None = None,
    response_ms: float | None = None,
):
    obj = SimpleNamespace(
        confidence=confidence,
        response_ms=response_ms,
        created_at=created_at or datetime.now(timezone.utc),
    )
    return obj


def _mk_dep(monthly_quota: int | None = None):
    return SimpleNamespace(monthly_quota=monthly_quota)


NOW = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
    b"East,175.2,18\nWest,220.4,22\nNorth,190.6,19\nEast,130.8,13\nWest,280.0,28\n"
)


# ─── async fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.dataset_filter  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.deploy as deploy_module
    import api.models as models_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"
    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
async def project_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Audit Test"})
    return resp.json()["id"]


@pytest.fixture()
async def dataset_id(ac, project_id):
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("sales.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
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
    resp = await ac.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"], "feature_set_id": feature_set_id},
    )
    assert resp.status_code == 202, resp.text
    run_id = resp.json()["model_run_ids"][0]
    for _ in range(30):
        r = await ac.get(f"/api/models/{project_id}/runs")
        run = next((x for x in r.json().get("runs", []) if x["id"] == run_id), None)
        if run and run["status"] == "done":
            return run_id
        time.sleep(0.3)
    pytest.skip("Training did not complete")


@pytest.fixture()
async def deployment_id(ac, trained_run_id):
    resp = await ac.post(f"/api/deploy/{trained_run_id}", json={})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# ─── compute_prediction_audit unit tests ──────────────────────────────────────


class TestComputePredictionAudit:
    def test_empty_logs_returns_zero_volume(self):
        result = compute_prediction_audit([], _mk_dep(), now_utc=NOW)
        assert result["total_predictions"] == 0
        assert result["predictions_today"] == 0
        assert result["predictions_7d"] == 0
        assert result["predictions_30d"] == 0

    def test_empty_logs_overall_healthy(self):
        result = compute_prediction_audit([], _mk_dep(), now_utc=NOW)
        assert result["overall_status"] == "healthy"

    def test_empty_logs_has_summary(self):
        result = compute_prediction_audit([], _mk_dep(), now_utc=NOW)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_volume_counts_correctly(self):
        logs = [
            _mk_log(NOW - timedelta(minutes=30)),  # today
            _mk_log(NOW - timedelta(hours=6)),  # today
            _mk_log(NOW - timedelta(days=2)),  # 7d
            _mk_log(NOW - timedelta(days=15)),  # 30d
            _mk_log(NOW - timedelta(days=40)),  # outside all windows
        ]
        result = compute_prediction_audit(logs, _mk_dep(), now_utc=NOW)
        assert result["total_predictions"] == 5
        assert result["predictions_today"] == 2
        assert result["predictions_7d"] == 3
        assert result["predictions_30d"] == 4

    def test_confidence_distribution_high(self):
        logs = [
            _mk_log(confidence=0.90),
            _mk_log(confidence=0.85),
            _mk_log(confidence=0.80),
        ]
        result = compute_prediction_audit(logs, _mk_dep(), now_utc=NOW)
        assert result["has_confidence_data"] is True
        assert result["confidence_high_pct"] == pytest.approx(100.0, abs=0.5)
        assert result["confidence_medium_pct"] == pytest.approx(0.0, abs=0.5)
        assert result["confidence_low_pct"] == pytest.approx(0.0, abs=0.5)

    def test_confidence_distribution_mixed(self):
        logs = [
            _mk_log(confidence=0.90),  # high
            _mk_log(confidence=0.70),  # medium
            _mk_log(confidence=0.40),  # low
            _mk_log(confidence=0.50),  # low
        ]
        result = compute_prediction_audit(logs, _mk_dep(), now_utc=NOW)
        assert result["confidence_high_pct"] == pytest.approx(25.0, abs=0.5)
        assert result["confidence_medium_pct"] == pytest.approx(25.0, abs=0.5)
        assert result["confidence_low_pct"] == pytest.approx(50.0, abs=0.5)

    def test_no_confidence_data_flag(self):
        logs = [_mk_log(), _mk_log()]
        result = compute_prediction_audit(logs, _mk_dep(), now_utc=NOW)
        assert result["has_confidence_data"] is False

    def test_latency_percentiles(self):
        latencies = [10.0, 20.0, 100.0, 300.0, 600.0]
        logs = [_mk_log(response_ms=ms) for ms in latencies]
        result = compute_prediction_audit(logs, _mk_dep(), now_utc=NOW)
        assert result["has_latency_data"] is True
        assert result["p50_ms"] is not None
        assert result["p95_ms"] is not None

    def test_sla_alert_when_p95_above_500(self):
        logs = [_mk_log(response_ms=float(ms)) for ms in [100, 100, 100, 100, 800]]
        result = compute_prediction_audit(logs, _mk_dep(), now_utc=NOW)
        assert result["sla_alert"] is True

    def test_no_sla_alert_when_fast(self):
        logs = [_mk_log(response_ms=50.0), _mk_log(response_ms=80.0)]
        result = compute_prediction_audit(logs, _mk_dep(), now_utc=NOW)
        assert result["sla_alert"] is False

    def test_no_latency_data_flag(self):
        logs = [_mk_log(), _mk_log()]
        result = compute_prediction_audit(logs, _mk_dep(), now_utc=NOW)
        assert result["has_latency_data"] is False
        assert result["p50_ms"] is None

    def test_quota_enabled_with_usage(self):
        logs = [_mk_log(NOW - timedelta(days=i)) for i in range(10)]
        dep = _mk_dep(monthly_quota=100)
        result = compute_prediction_audit(logs, dep, now_utc=NOW)
        assert result["quota_enabled"] is True
        assert result["monthly_quota"] == 100
        assert result["quota_used"] == 10
        assert result["quota_pct"] == pytest.approx(10.0, abs=0.5)

    def test_quota_disabled(self):
        logs = [_mk_log()]
        result = compute_prediction_audit(
            logs, _mk_dep(monthly_quota=None), now_utc=NOW
        )
        assert result["quota_enabled"] is False
        assert result["quota_pct"] is None

    def test_critical_status_on_sla_alert(self):
        logs = [_mk_log(response_ms=float(ms)) for ms in [100] * 9 + [1000]]
        result = compute_prediction_audit(logs, _mk_dep(), now_utc=NOW)
        assert result["overall_status"] == "critical"
        assert result["overall_label"] == "Critical"

    def test_warning_status_on_high_quota_usage(self):
        logs = [_mk_log(NOW - timedelta(hours=i)) for i in range(75)]
        dep = _mk_dep(monthly_quota=100)
        result = compute_prediction_audit(logs, dep, now_utc=NOW)
        assert result["overall_status"] in ("warning", "critical")

    def test_healthy_status_normal_operation(self):
        logs = [_mk_log(response_ms=50.0, confidence=0.90) for _ in range(5)]
        result = compute_prediction_audit(logs, _mk_dep(), now_utc=NOW)
        assert result["overall_status"] == "healthy"

    def test_result_has_required_fields(self):
        result = compute_prediction_audit([], _mk_dep(), now_utc=NOW)
        required = [
            "total_predictions",
            "predictions_today",
            "predictions_7d",
            "predictions_30d",
            "confidence_high_pct",
            "confidence_medium_pct",
            "confidence_low_pct",
            "has_confidence_data",
            "p50_ms",
            "p95_ms",
            "avg_ms",
            "has_latency_data",
            "sla_alert",
            "quota_used",
            "monthly_quota",
            "quota_pct",
            "quota_enabled",
            "overall_status",
            "overall_label",
            "summary",
        ]
        for field in required:
            assert field in result, f"Missing field: {field}"


# ─── REST endpoint tests ───────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_returns_404_for_unknown_deployment(ac):
    resp = await ac.get("/api/deploy/nonexistent-id/prediction-audit")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_returns_audit_for_active_deployment(ac, deployment_id):
    resp = await ac.get(f"/api/deploy/{deployment_id}/prediction-audit")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deployment_id"] == deployment_id
    assert "total_predictions" in data
    assert "overall_status" in data
    assert data["total_predictions"] == 0


@pytest.mark.anyio
async def test_audit_deployment_id_in_response(ac, deployment_id):
    resp = await ac.get(f"/api/deploy/{deployment_id}/prediction-audit")
    assert resp.status_code == 200
    assert resp.json()["deployment_id"] == deployment_id


# ─── Pattern regex tests ────────────────────────────────────────────────────────


_PAT = re.compile(
    r"(?i)(?:"
    r"(?:deployment|prediction|model)\s+(?:audit|monitoring\s+(?:digest|summary|report)|operational\s+(?:report|summary))\b|"
    r"(?:how|what)\s+(?:is|are|was|were)\s+(?:my\s+)?(?:deployment|model|api|endpoint)\s+doing\b|"
    r"(?:production|deployment|model)\s+(?:status\s+(?:report|summary|check)|health\s+(?:report|digest))\b|"
    r"(?:show|give|get)\s+(?:me\s+)?(?:a\s+)?(?:deployment|model|monitoring)\s+(?:digest|summary|report|snapshot|overview)\b|"
    r"(?:morning|daily|weekly)\s+(?:model|deployment|prediction)\s+(?:report|briefing|check|digest)\b|"
    r"(?:what'?s?|how\s+is)\s+(?:my\s+)?(?:deployment|prediction\s+api|model\s+api)\s+(?:performance|doing|status)\b|"
    r"(?:full|complete|comprehensive)\s+(?:deployment|prediction|monitoring)\s+(?:report|summary|overview)\b|"
    r"audit\s+(?:my\s+)?(?:deployment|predictions?|production\s+model)\b"
    r")",
    re.IGNORECASE,
)


class TestPredAuditPatterns:
    @pytest.mark.parametrize(
        "msg",
        [
            "deployment audit",
            "show me a deployment summary",
            "give me a monitoring digest",
            "model monitoring report",
            "how is my deployment doing",
            "how is my model doing",
            "what is my api doing",
            "model status report",
            "deployment health report",
            "daily model report",
            "morning deployment check",
            "comprehensive deployment report",
            "audit my deployment",
            "audit my predictions",
            "full monitoring overview",
        ],
    )
    def test_positive(self, msg):
        assert _PAT.search(msg), f"Expected match for: {msg!r}"

    @pytest.mark.parametrize(
        "msg",
        [
            "show me recent predictions",
            "export prediction logs",
            "is my model drifting",
            "what is my quota",
            "show SLA metrics",
            "train a model",
            "what are the top features",
        ],
    )
    def test_negative(self, msg):
        assert not _PAT.search(msg), f"Expected NO match for: {msg!r}"


# ─── Chat handler integration tests ────────────────────────────────────────────


def _parse_events(resp) -> list[dict]:
    events = []
    for line in resp.text.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def _mock_stream(text="Audit complete."):
    import unittest.mock as mock

    ms = mock.MagicMock()
    ms.__enter__ = mock.MagicMock(return_value=ms)
    ms.__exit__ = mock.MagicMock(return_value=False)
    ms.text_stream = iter([text])
    return ms


@pytest.mark.anyio
async def test_emits_prediction_audit_event(ac, project_id, deployment_id):
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = _mock_stream()
        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "show me a deployment audit", "session_id": "test"},
        )
    events = _parse_events(resp)
    types = [e.get("type") for e in events]
    assert "prediction_audit" in types, f"Expected prediction_audit event. Got: {types}"


@pytest.mark.anyio
async def test_audit_event_has_required_fields(ac, project_id, deployment_id):
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = _mock_stream()
        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "model monitoring report", "session_id": "test"},
        )
    events = _parse_events(resp)
    audit_events = [e for e in events if e.get("type") == "prediction_audit"]
    assert audit_events, "No prediction_audit event found"
    data = audit_events[0]["prediction_audit"]
    assert "total_predictions" in data
    assert "overall_status" in data
    assert "summary" in data


@pytest.mark.anyio
async def test_no_event_without_deployment(ac, project_id):
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = _mock_stream()
        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "deployment audit", "session_id": "test"},
        )
    events = _parse_events(resp)
    types = [e.get("type") for e in events]
    assert "prediction_audit" not in types
