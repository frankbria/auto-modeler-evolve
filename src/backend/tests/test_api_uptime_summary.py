"""Tests for Prediction API Uptime Summary.

Covers:
- compute_api_uptime_summary pure function (verdicts, daily stats, latency stats)
- REST endpoint: GET /api/deploy/{id}/uptime-summary
- Chat: _UPTIME_PATTERNS regex (positive + false-positive guards)
"""

import uuid
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from api.chat import _UPTIME_PATTERNS
from core.analyzer import compute_api_uptime_summary
from main import app
from models.deployment import Deployment
from models.project import Project

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_log_dicts(
    days_ago_list: list[float], response_ms_list: list[float | None]
) -> list[dict]:
    """Build log dicts with created_at and response_ms."""
    now = datetime.utcnow()
    return [
        {
            "created_at": now - timedelta(days=d),
            "response_ms": ms,
        }
        for d, ms in zip(days_ago_list, response_ms_list)
    ]


# ---------------------------------------------------------------------------
# compute_api_uptime_summary — no data / minimal data
# ---------------------------------------------------------------------------


def test_uptime_idle_when_empty():
    # Zero predictions → idle (deployment exists but no traffic in window)
    result = compute_api_uptime_summary([])
    assert result["verdict"] == "idle"
    assert result["total_predictions"] == 0
    assert result["n_active_days"] == 0
    assert result["overall_avg_latency_ms"] is None
    assert result["overall_p95_latency_ms"] is None


def test_uptime_no_data_too_few():
    logs = _make_log_dicts(
        [0.5, 1.5], [100.0, 200.0]
    )  # only 2 predictions (> 0 but < 3)
    result = compute_api_uptime_summary(logs)
    assert result["verdict"] == "no_data"
    assert result["total_predictions"] == 2


def test_uptime_required_keys():
    result = compute_api_uptime_summary([])
    required = {
        "n_days",
        "total_predictions",
        "n_active_days",
        "n_silent_days",
        "daily_stats",
        "overall_avg_latency_ms",
        "overall_p95_latency_ms",
        "peak_latency_ms",
        "peak_latency_date",
        "degraded_days",
        "verdict",
        "note",
        "summary",
    }
    assert required.issubset(result.keys())


def test_uptime_daily_stats_length():
    result = compute_api_uptime_summary([], n_days=7)
    assert len(result["daily_stats"]) == 7


def test_uptime_daily_stats_custom_window():
    result = compute_api_uptime_summary([], n_days=14)
    assert len(result["daily_stats"]) == 14
    assert result["n_days"] == 14


# ---------------------------------------------------------------------------
# compute_api_uptime_summary — idle verdict
# ---------------------------------------------------------------------------


def test_uptime_idle_verdict():
    # Many predictions but all older than n_days
    logs = _make_log_dicts([10.0, 11.0, 12.0, 13.0, 14.0], [50.0] * 5)
    result = compute_api_uptime_summary(logs, n_days=7)
    assert result["verdict"] == "idle"
    assert result["n_active_days"] == 0
    assert result["total_predictions"] == 0


# ---------------------------------------------------------------------------
# compute_api_uptime_summary — healthy verdict
# ---------------------------------------------------------------------------


def test_uptime_healthy_verdict():
    # 5 predictions spread across 5 of the last 7 days with fast latency
    logs = _make_log_dicts([0.5, 1.5, 2.5, 3.5, 4.5], [100.0, 120.0, 80.0, 90.0, 110.0])
    result = compute_api_uptime_summary(logs, n_days=7)
    assert result["verdict"] == "healthy"
    assert result["degraded_days"] == 0
    assert result["n_active_days"] >= 4


def test_uptime_healthy_latency_stats():
    logs = _make_log_dicts(
        [0.5, 0.5, 1.5, 1.5, 2.5], [100.0, 200.0, 150.0, 250.0, 300.0]
    )
    result = compute_api_uptime_summary(logs, n_days=7)
    assert result["overall_avg_latency_ms"] is not None
    assert result["overall_p95_latency_ms"] is not None
    assert result["overall_avg_latency_ms"] > 0


# ---------------------------------------------------------------------------
# compute_api_uptime_summary — degraded verdict
# ---------------------------------------------------------------------------


def test_uptime_degraded_single_day():
    # One day with very high latency (p95 > 2000ms)
    logs = _make_log_dicts(
        [0.3, 0.4, 0.5, 1.5, 2.5],
        [3000.0, 2500.0, 2800.0, 100.0, 100.0],  # today is slow
    )
    result = compute_api_uptime_summary(logs, n_days=7)
    assert result["verdict"] == "degraded"
    assert result["degraded_days"] >= 1


def test_uptime_degraded_day_status():
    # Today's predictions all have high latency
    logs = _make_log_dicts([0.01, 0.02, 0.03], [3000.0, 3000.0, 3000.0])
    result = compute_api_uptime_summary(logs, n_days=7)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    today_stat = next((s for s in result["daily_stats"] if s["date"] == today), None)
    assert today_stat is not None
    assert today_stat["status"] == "degraded"
    assert today_stat["n_predictions"] == 3


# ---------------------------------------------------------------------------
# compute_api_uptime_summary — mostly_idle verdict
# ---------------------------------------------------------------------------


def test_uptime_mostly_idle_verdict():
    # Predictions all on day 0, giving at most 1-2 active days out of 7 → mostly_idle
    now = datetime.utcnow()
    logs = [
        {"created_at": now - timedelta(hours=1), "response_ms": 100.0},
        {"created_at": now - timedelta(hours=2), "response_ms": 120.0},
        {"created_at": now - timedelta(hours=3), "response_ms": 90.0},
    ]
    result = compute_api_uptime_summary(logs, n_days=7)
    assert result["verdict"] == "mostly_idle"
    assert result["n_active_days"] < 3  # < n_days // 2


# ---------------------------------------------------------------------------
# compute_api_uptime_summary — daily stats structure
# ---------------------------------------------------------------------------


def test_uptime_daily_stat_keys():
    result = compute_api_uptime_summary([])
    for stat in result["daily_stats"]:
        assert set(stat.keys()) == {
            "date",
            "n_predictions",
            "avg_latency_ms",
            "p95_latency_ms",
            "status",
        }


def test_uptime_silent_days_have_no_latency():
    result = compute_api_uptime_summary([], n_days=7)
    for stat in result["daily_stats"]:
        assert stat["status"] == "silent"
        assert stat["avg_latency_ms"] is None
        assert stat["p95_latency_ms"] is None
        assert stat["n_predictions"] == 0


def test_uptime_active_day_has_predictions():
    now = datetime.utcnow()
    # Use exact datetime that is definitely today to avoid date boundary issues
    logs = [{"created_at": now - timedelta(hours=1), "response_ms": 100.0}]
    result = compute_api_uptime_summary(logs, n_days=7)
    today = now.strftime("%Y-%m-%d")
    today_stat = next((s for s in result["daily_stats"] if s["date"] == today), None)
    assert today_stat is not None
    assert today_stat["n_predictions"] == 1
    assert today_stat["status"] == "active"


def test_uptime_n_active_n_silent_sum_to_n_days():
    logs = _make_log_dicts([0.5, 1.5, 2.5, 3.5, 4.5], [100.0] * 5)
    result = compute_api_uptime_summary(logs, n_days=7)
    assert result["n_active_days"] + result["n_silent_days"] == 7


# ---------------------------------------------------------------------------
# compute_api_uptime_summary — peak latency
# ---------------------------------------------------------------------------


def test_uptime_peak_latency_captured():
    logs = _make_log_dicts(
        [0.5, 1.5, 2.5, 3.5, 4.5], [100.0, 200.0, 500.0, 150.0, 300.0]
    )
    result = compute_api_uptime_summary(logs, n_days=7)
    assert result["peak_latency_ms"] == pytest.approx(500.0, abs=1)


def test_uptime_no_peak_when_no_latency_data():
    logs = _make_log_dicts([0.5, 1.5, 2.5], [None, None, None])
    result = compute_api_uptime_summary(logs, n_days=7)
    assert result["peak_latency_ms"] is None
    assert result["peak_latency_date"] is None


# ---------------------------------------------------------------------------
# compute_api_uptime_summary — summary and note content
# ---------------------------------------------------------------------------


def test_uptime_summary_is_string():
    result = compute_api_uptime_summary([])
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0


def test_uptime_note_mentions_logging_limitation():
    result = compute_api_uptime_summary([])
    assert "silent" in result["note"].lower() or "successful" in result["note"].lower()


def test_uptime_healthy_summary_mentions_active_days():
    logs = _make_log_dicts([0.5, 1.5, 2.5, 3.5, 4.5], [100.0] * 5)
    result = compute_api_uptime_summary(logs, n_days=7)
    assert (
        "active" in result["summary"].lower() or "healthy" in result["summary"].lower()
    )


# ---------------------------------------------------------------------------
# REST endpoint — GET /api/deploy/{id}/uptime-summary
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return TestClient(app)


def test_uptime_endpoint_404(client):
    resp = client.get("/api/deploy/nonexistent-id/uptime-summary")
    assert resp.status_code == 404


def test_uptime_endpoint_returns_dict(client):
    # Create project + deployment then hit endpoint
    from db import get_session

    with TestClient(app) as c:
        with next(get_session()) as session:
            proj = Project(owner_id="test-default-owner", id=str(uuid.uuid4()), name="uptime-test")
            session.add(proj)
            dep = Deployment(
                id=str(uuid.uuid4()),
                model_run_id=str(uuid.uuid4()),
                project_id=proj.id,
                endpoint_path=f"/api/predict/{uuid.uuid4()}",
                dashboard_url=f"/predict/{uuid.uuid4()}",
            )
            session.add(dep)
            session.commit()
            dep_id = dep.id

        resp = c.get(f"/api/deploy/{dep_id}/uptime-summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "verdict" in data
        assert "daily_stats" in data
        assert "deployment_id" in data
        assert data["deployment_id"] == dep_id


def test_uptime_endpoint_n_days_param(client):
    from db import get_session

    with TestClient(app) as c:
        with next(get_session()) as session:
            proj = Project(owner_id="test-default-owner", id=str(uuid.uuid4()), name="uptime-test-days")
            session.add(proj)
            dep = Deployment(
                id=str(uuid.uuid4()),
                model_run_id=str(uuid.uuid4()),
                project_id=proj.id,
                endpoint_path=f"/api/predict/{uuid.uuid4()}",
                dashboard_url=f"/predict/{uuid.uuid4()}",
            )
            session.add(dep)
            session.commit()
            dep_id = dep.id

        resp = c.get(f"/api/deploy/{dep_id}/uptime-summary?n_days=14")
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_days"] == 14
        assert len(data["daily_stats"]) == 14


# ---------------------------------------------------------------------------
# _UPTIME_PATTERNS regex — positive matches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "msg",
    [
        "has my prediction endpoint had any downtime this week?",
        "show my deployment uptime",
        "how reliable has my API been?",
        "API availability report for my model",
        "uptime stats for my deployment",
        "was my endpoint down yesterday?",
        "API health history",
        "how available is my deployment?",
    ],
)
def test_uptime_patterns_positive(msg):
    assert _UPTIME_PATTERNS.search(msg), f"Expected match for: {msg!r}"


# ---------------------------------------------------------------------------
# _UPTIME_PATTERNS regex — false-positive guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "msg",
    [
        "show me my prediction history",
        "what is the latency of my model?",
        "how many predictions this week?",
        "throughput assessment for my deployment",
        "show recent predictions table",
    ],
)
def test_uptime_patterns_negative(msg):
    assert not _UPTIME_PATTERNS.search(msg), f"Unexpected match for: {msg!r}"
