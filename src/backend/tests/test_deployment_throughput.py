"""Tests for Deployment Throughput Assessment.

Covers:
- compute_deployment_throughput pure function (latency stats, verdicts, target_n math)
- _format_duration helper
- REST endpoint: GET /api/deploy/{id}/throughput-assessment
- Chat: _THROUGHPUT_PATTERNS regex (positive + false-positive guards)
- Chat: _extract_throughput_n helper
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from api.chat import _THROUGHPUT_PATTERNS, _extract_throughput_n
from core.analyzer import _format_duration, compute_deployment_throughput
from main import app
from models.deployment import Deployment
from models.prediction_log import PredictionLog
from models.project import Project

# ---------------------------------------------------------------------------
# _format_duration
# ---------------------------------------------------------------------------


def test_format_duration_milliseconds():
    assert _format_duration(0.5) == "500 ms"


def test_format_duration_sub_second():
    assert _format_duration(0.001) == "1 ms"


def test_format_duration_seconds():
    result = _format_duration(30.5)
    assert "30" in result and "second" in result


def test_format_duration_one_minute():
    result = _format_duration(60)
    assert "1 minute" in result


def test_format_duration_minutes():
    result = _format_duration(150)  # 2m 30s
    assert "2 minute" in result


def test_format_duration_one_hour():
    result = _format_duration(3600)
    assert "1 hour" in result


def test_format_duration_hours():
    result = _format_duration(7260)  # 2h 1m
    assert "2 hour" in result


# ---------------------------------------------------------------------------
# compute_deployment_throughput — no data
# ---------------------------------------------------------------------------


def test_throughput_no_data_empty():
    result = compute_deployment_throughput([])
    assert result["verdict"] == "no_data"
    assert result["p50_ms"] is None
    assert result["p95_ms"] is None
    assert result["max_rps"] is None
    assert result["serial_seconds"] is None


def test_throughput_no_data_too_few():
    logs = [{"response_ms": 50.0}] * 4  # only 4, need 5
    result = compute_deployment_throughput(logs)
    assert result["verdict"] == "no_data"
    assert result["n_samples"] == 4


def test_throughput_no_data_skips_none():
    logs = [{"response_ms": None}] * 10
    result = compute_deployment_throughput(logs)
    assert result["verdict"] == "no_data"


# ---------------------------------------------------------------------------
# compute_deployment_throughput — latency stats
# ---------------------------------------------------------------------------


def _make_logs(ms_values: list[float]) -> list[dict]:
    return [{"response_ms": ms} for ms in ms_values]


def test_throughput_p50_correct():
    # With [10..100] (10 values), index=int(10*0.50)=5 → sorted[5]=60
    logs = _make_logs([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    result = compute_deployment_throughput(logs)
    assert result["p50_ms"] == pytest.approx(60.0, abs=1)


def test_throughput_p95_correct():
    logs = _make_logs(list(range(10, 110, 10)))  # 10..100, 10 values
    result = compute_deployment_throughput(logs)
    assert result["p95_ms"] == pytest.approx(100.0, abs=5)


def test_throughput_mean_correct():
    logs = _make_logs([100.0] * 10)
    result = compute_deployment_throughput(logs)
    assert result["mean_ms"] == pytest.approx(100.0)


def test_throughput_rps_from_p95():
    # p95 of 100ms → 10 rps
    logs = _make_logs([50] * 9 + [100])  # p95 = 100
    result = compute_deployment_throughput(logs)
    assert result["max_rps"] == pytest.approx(10.0, abs=1)


# ---------------------------------------------------------------------------
# compute_deployment_throughput — verdicts
# ---------------------------------------------------------------------------


def test_throughput_verdict_instant():
    # p95 = 10ms → 1000 predictions = 10 seconds → "instant" (< 5s)
    # Need p95 ≤ 5ms for target_n=1000 to be instant
    logs = _make_logs([2.0] * 5)  # p95 = 2ms → serial = 2s
    result = compute_deployment_throughput(logs, target_n=1000)
    assert result["verdict"] == "instant"


def test_throughput_verdict_fast():
    # p95 = 20ms → 1000 * 20 / 1000 = 20s → "fast" (< 60s)
    logs = _make_logs([20.0] * 5)
    result = compute_deployment_throughput(logs, target_n=1000)
    assert result["verdict"] == "fast"


def test_throughput_verdict_moderate():
    # p95 = 300ms → 1000 * 300 / 1000 = 300s → "moderate" (< 600s)
    logs = _make_logs([300.0] * 5)
    result = compute_deployment_throughput(logs, target_n=1000)
    assert result["verdict"] == "moderate"


def test_throughput_verdict_slow():
    # p95 = 2000ms → 1000 * 2000 / 1000 = 2000s → "slow" (< 3600s)
    logs = _make_logs([2000.0] * 5)
    result = compute_deployment_throughput(logs, target_n=1000)
    assert result["verdict"] == "slow"


def test_throughput_verdict_very_slow():
    # p95 = 5000ms → 1000 * 5000 / 1000 = 5000s → "very_slow" (≥ 3600s)
    logs = _make_logs([5000.0] * 5)
    result = compute_deployment_throughput(logs, target_n=1000)
    assert result["verdict"] == "very_slow"


# ---------------------------------------------------------------------------
# compute_deployment_throughput — target_n scaling
# ---------------------------------------------------------------------------


def test_throughput_target_n_scales_duration():
    logs = _make_logs([100.0] * 5)  # p95 = 100ms → 1 rps
    result_1k = compute_deployment_throughput(logs, target_n=1000)
    result_2k = compute_deployment_throughput(logs, target_n=2000)
    assert result_2k["serial_seconds"] == pytest.approx(
        result_1k["serial_seconds"] * 2, rel=0.01
    )


def test_throughput_default_target_n():
    logs = _make_logs([50.0] * 5)
    result = compute_deployment_throughput(logs)
    assert result["target_n"] == 1000


def test_throughput_required_keys():
    logs = _make_logs([100.0] * 5)
    result = compute_deployment_throughput(logs)
    for key in [
        "verdict",
        "n_samples",
        "target_n",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "mean_ms",
        "max_rps",
        "serial_seconds",
        "serial_duration",
        "summary",
    ]:
        assert key in result, f"missing key: {key}"


def test_throughput_summary_nonempty():
    logs = _make_logs([100.0] * 5)
    result = compute_deployment_throughput(logs)
    assert len(result["summary"]) > 10


# ---------------------------------------------------------------------------
# REST endpoint
# ---------------------------------------------------------------------------


def test_throughput_endpoint_404():
    client = TestClient(app)
    resp = client.get(
        "/api/deploy/00000000-0000-0000-0000-000000000000/throughput-assessment"
    )
    assert resp.status_code == 404


def test_throughput_endpoint_no_data():
    from db import engine

    from sqlmodel import Session as _Session

    project_id = str(uuid.uuid4())
    dep_id = str(uuid.uuid4())

    with _Session(engine) as session:
        project = Project(id=project_id, name="TP Project")
        session.add(project)
        dep = Deployment(
            id=dep_id,
            project_id=project_id,
            model_run_id=str(uuid.uuid4()),
            endpoint_path=f"/api/predict/{dep_id}",
            dashboard_url=f"/predict/{dep_id}",
            is_active=True,
            algorithm="random_forest_regressor",
            target_column="revenue",
        )
        session.add(dep)
        session.commit()

    client = TestClient(app)
    resp = client.get(f"/api/deploy/{dep_id}/throughput-assessment")
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "no_data"
    assert data["deployment_id"] == dep_id


def test_throughput_endpoint_with_data():
    from db import engine

    from sqlmodel import Session as _Session

    project_id = str(uuid.uuid4())
    dep_id = str(uuid.uuid4())

    with _Session(engine) as session:
        project = Project(id=project_id, name="TP Project 2")
        session.add(project)
        dep = Deployment(
            id=dep_id,
            project_id=project_id,
            model_run_id=str(uuid.uuid4()),
            endpoint_path=f"/api/predict/{dep_id}",
            dashboard_url=f"/predict/{dep_id}",
            is_active=True,
            algorithm="random_forest_regressor",
            target_column="revenue",
        )
        session.add(dep)
        for ms in [50.0, 60.0, 70.0, 80.0, 90.0, 100.0]:
            log = PredictionLog(
                deployment_id=dep_id,
                input_features="{}",
                prediction="1.0",
                response_ms=ms,
            )
            session.add(log)
        session.commit()

    client = TestClient(app)
    resp = client.get(f"/api/deploy/{dep_id}/throughput-assessment")
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] != "no_data"
    assert data["p50_ms"] is not None
    assert data["p95_ms"] is not None
    assert data["max_rps"] is not None
    assert data["serial_seconds"] is not None


def test_throughput_endpoint_custom_n():
    from db import engine

    from sqlmodel import Session as _Session

    project_id = str(uuid.uuid4())
    dep_id = str(uuid.uuid4())

    with _Session(engine) as session:
        project = Project(id=project_id, name="TP Project 3")
        session.add(project)
        dep = Deployment(
            id=dep_id,
            project_id=project_id,
            model_run_id=str(uuid.uuid4()),
            endpoint_path=f"/api/predict/{dep_id}",
            dashboard_url=f"/predict/{dep_id}",
            is_active=True,
            algorithm="random_forest_regressor",
            target_column="revenue",
        )
        session.add(dep)
        for ms in [100.0] * 5:
            log = PredictionLog(
                deployment_id=dep_id,
                input_features="{}",
                prediction="1.0",
                response_ms=ms,
            )
            session.add(log)
        session.commit()

    client = TestClient(app)
    resp = client.get(f"/api/deploy/{dep_id}/throughput-assessment?n=500")
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_n"] == 500


# ---------------------------------------------------------------------------
# _THROUGHPUT_PATTERNS regex
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "msg",
    [
        "how long to process 1000 predictions?",
        "how long to batch 500 records?",
        "throughput assessment for this deployment",
        "batch time estimate",
        "how fast can my deployment process requests?",
        "how fast does my model predict?",
        "maximum requests per second",
        "how many predictions per second can this handle?",
        "can it handle 1000 requests per second",
        "time to process my dataset",
    ],
)
def test_throughput_patterns_positive(msg):
    assert _THROUGHPUT_PATTERNS.search(msg), f"expected match: {msg!r}"


@pytest.mark.parametrize(
    "msg",
    [
        "how many predictions do I have left in my quota",
        "cost estimate for 1000 predictions",
    ],
)
def test_throughput_patterns_false_positive_guard(msg):
    assert not _THROUGHPUT_PATTERNS.search(msg), f"unexpected match: {msg!r}"


# ---------------------------------------------------------------------------
# _extract_throughput_n
# ---------------------------------------------------------------------------


def test_extract_throughput_n_plain():
    assert _extract_throughput_n("how long to process 500 records") == 500


def test_extract_throughput_n_k_suffix():
    assert _extract_throughput_n("can it handle 5k requests") == 5000


def test_extract_throughput_n_comma():
    assert _extract_throughput_n("batch 1,000 predictions") == 1000


def test_extract_throughput_n_default():
    assert _extract_throughput_n("what's the throughput?") == 1000
