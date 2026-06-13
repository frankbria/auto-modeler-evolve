"""Tests for Deployment Health Scorecard feature.

Covers:
- compute_deployment_health_scorecard() pure function (20 tests)
- Pattern matching for _DEPLOY_HEALTH_SCORECARD_PATTERNS (8 + 5 = 13 tests)
- GET /api/deploy/{deployment_id}/health-scorecard REST endpoint (5 tests)
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlmodel import Session, SQLModel, create_engine

import db as db_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _make_logs(
    n: int = 10,
    response_ms: float | None = 100.0,
    confidence: float | None = 0.85,
    days_old: int = 1,
) -> list[dict]:
    base = _now() - timedelta(days=days_old)
    return [
        {
            "response_ms": response_ms,
            "confidence": confidence,
            "created_at": base + timedelta(hours=i),
            "prediction_numeric": 42.0,
            "prediction_value": "42.0",
        }
        for i in range(n)
    ]


def _make_feedback(n: int = 15, accuracy: float = 0.9) -> list[dict]:
    correct = int(n * accuracy)
    return [{"is_correct": True}] * correct + [{"is_correct": False}] * (n - correct)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _test_db(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'hs.db'}")
    monkeypatch.setattr(db_module, "engine", engine)
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture()
def client():
    from main import app
    from starlette.testclient import TestClient

    return TestClient(app)


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_healthy_deployment_grade_a_or_b():
    from core.analyzer import compute_deployment_health_scorecard

    logs = _make_logs(20, response_ms=80.0, confidence=0.92, days_old=1)
    fb = _make_feedback(20, accuracy=0.95)
    result = compute_deployment_health_scorecard(
        logs=logs,
        feedback_records=fb,
        created_at=_now() - timedelta(days=5),
        problem_type="classification",
    )

    assert result["overall_grade"] in ("A", "B")
    assert result["overall_score"] >= 70
    assert len(result["signals"]) == 5
    assert result["summary"]


def test_high_latency_signals_red():
    from core.analyzer import compute_deployment_health_scorecard

    logs = _make_logs(10, response_ms=3000.0, confidence=0.85, days_old=1)
    result = compute_deployment_health_scorecard(
        logs=logs,
        feedback_records=[],
        created_at=_now() - timedelta(days=3),
    )

    lat = next(s for s in result["signals"] if s["signal_key"] == "latency")
    assert lat["severity"] == "red"
    assert lat["score"] < 30


def test_moderate_latency_signals_amber():
    from core.analyzer import compute_deployment_health_scorecard

    logs = _make_logs(10, response_ms=700.0, days_old=1)
    result = compute_deployment_health_scorecard(
        logs=logs,
        feedback_records=[],
        created_at=_now() - timedelta(days=5),
    )

    lat = next(s for s in result["signals"] if s["signal_key"] == "latency")
    assert lat["severity"] == "amber"


def test_low_confidence_signals_red():
    from core.analyzer import compute_deployment_health_scorecard

    logs = _make_logs(10, response_ms=100.0, confidence=0.45, days_old=1)
    result = compute_deployment_health_scorecard(
        logs=logs,
        feedback_records=[],
        created_at=_now() - timedelta(days=3),
        problem_type="classification",
    )

    conf = next(s for s in result["signals"] if s["signal_key"] == "confidence")
    assert conf["severity"] == "red"


def test_moderate_confidence_signals_amber():
    from core.analyzer import compute_deployment_health_scorecard

    logs = _make_logs(10, confidence=0.65, days_old=1)
    result = compute_deployment_health_scorecard(
        logs=logs,
        feedback_records=[],
        created_at=_now() - timedelta(days=5),
        problem_type="classification",
    )

    conf = next(s for s in result["signals"] if s["signal_key"] == "confidence")
    assert conf["severity"] == "amber"


def test_no_recent_activity_signals_red():
    from core.analyzer import compute_deployment_health_scorecard

    # Logs are old (> 7 days), so recent_count = 0
    logs = _make_logs(5, response_ms=100.0, confidence=0.85, days_old=14)
    result = compute_deployment_health_scorecard(
        logs=logs,
        feedback_records=[],
        created_at=_now() - timedelta(days=30),
    )

    act = next(s for s in result["signals"] if s["signal_key"] == "activity")
    assert act["severity"] == "red"
    assert act["value"] == 0


def test_new_deployment_activity_gray():
    from core.analyzer import compute_deployment_health_scorecard

    result = compute_deployment_health_scorecard(
        logs=[],
        feedback_records=[],
        created_at=_now() - timedelta(hours=2),
    )

    act = next(s for s in result["signals"] if s["signal_key"] == "activity")
    assert act["severity"] == "gray"


def test_high_activity_signals_green():
    from core.analyzer import compute_deployment_health_scorecard

    logs = _make_logs(60, response_ms=100.0, days_old=1)
    result = compute_deployment_health_scorecard(
        logs=logs,
        feedback_records=[],
        created_at=_now() - timedelta(days=10),
    )

    act = next(s for s in result["signals"] if s["signal_key"] == "activity")
    assert act["severity"] == "green"


def test_low_feedback_accuracy_signals_red():
    from core.analyzer import compute_deployment_health_scorecard

    fb = _make_feedback(15, accuracy=0.40)
    result = compute_deployment_health_scorecard(
        logs=_make_logs(5, days_old=1),
        feedback_records=fb,
        created_at=_now() - timedelta(days=10),
        problem_type="classification",
    )

    acc = next(s for s in result["signals"] if s["signal_key"] == "accuracy")
    assert acc["severity"] == "red"


def test_insufficient_feedback_accuracy_gray():
    from core.analyzer import compute_deployment_health_scorecard

    fb = _make_feedback(5, accuracy=0.9)  # below 10 threshold
    result = compute_deployment_health_scorecard(
        logs=_make_logs(10, days_old=1),
        feedback_records=fb,
        created_at=_now() - timedelta(days=5),
    )

    acc = next(s for s in result["signals"] if s["signal_key"] == "accuracy")
    assert acc["severity"] == "gray"
    assert not acc["is_available"]


def test_old_model_signals_red():
    from core.analyzer import compute_deployment_health_scorecard

    result = compute_deployment_health_scorecard(
        logs=_make_logs(10, days_old=1),
        feedback_records=[],
        created_at=_now() - timedelta(days=100),
    )

    age = next(s for s in result["signals"] if s["signal_key"] == "model_age")
    assert age["severity"] == "red"
    assert age["value"] >= 100


def test_fresh_model_age_green():
    from core.analyzer import compute_deployment_health_scorecard

    result = compute_deployment_health_scorecard(
        logs=_make_logs(5, days_old=1),
        feedback_records=[],
        created_at=_now() - timedelta(days=3),
    )

    age = next(s for s in result["signals"] if s["signal_key"] == "model_age")
    assert age["severity"] == "green"


def test_insufficient_latency_data_gray():
    from core.analyzer import compute_deployment_health_scorecard

    logs = _make_logs(3, response_ms=100.0, days_old=1)  # < 5 threshold
    result = compute_deployment_health_scorecard(
        logs=logs,
        feedback_records=[],
        created_at=_now() - timedelta(days=5),
    )

    lat = next(s for s in result["signals"] if s["signal_key"] == "latency")
    assert lat["severity"] == "gray"
    assert not lat["is_available"]


def test_canary_active_shows_info():
    from core.analyzer import compute_deployment_health_scorecard

    result = compute_deployment_health_scorecard(
        logs=_make_logs(10, days_old=1),
        feedback_records=[],
        created_at=_now() - timedelta(days=5),
        canary_is_active=True,
        canary_traffic_pct=10,
    )

    assert result["canary_info"] is not None
    assert result["canary_info"]["is_active"] is True
    assert result["canary_info"]["traffic_pct"] == 10
    assert "10%" in result["canary_info"]["finding"]


def test_canary_inactive_no_info():
    from core.analyzer import compute_deployment_health_scorecard

    result = compute_deployment_health_scorecard(
        logs=_make_logs(10, days_old=1),
        feedback_records=[],
        created_at=_now() - timedelta(days=5),
        canary_is_active=False,
    )

    assert result["canary_info"] is None


def test_score_in_range():
    from core.analyzer import compute_deployment_health_scorecard

    result = compute_deployment_health_scorecard(
        logs=_make_logs(10, days_old=1),
        feedback_records=[],
        created_at=_now() - timedelta(days=5),
    )

    assert 0 <= result["overall_score"] <= 100


def test_signal_count_totals_five():
    from core.analyzer import compute_deployment_health_scorecard

    result = compute_deployment_health_scorecard(
        logs=_make_logs(10, days_old=1),
        feedback_records=[],
        created_at=_now() - timedelta(days=5),
    )

    gray_count = sum(1 for s in result["signals"] if s["severity"] == "gray")
    non_gray = (
        result["n_signals_healthy"]
        + result["n_signals_warning"]
        + result["n_signals_critical"]
    )
    assert non_gray + gray_count == 5


def test_critical_deployment_gets_low_grade():
    from core.analyzer import compute_deployment_health_scorecard

    logs = _make_logs(10, response_ms=5000.0, confidence=0.35, days_old=30)
    fb = _make_feedback(15, accuracy=0.30)
    result = compute_deployment_health_scorecard(
        logs=logs,
        feedback_records=fb,
        created_at=_now() - timedelta(days=120),
        problem_type="classification",
    )

    assert result["overall_grade"] in ("D", "F")
    assert result["overall_health"] in ("warning", "critical")


def test_grade_a_for_excellent_deployment():
    from core.analyzer import compute_deployment_health_scorecard

    logs = _make_logs(60, response_ms=50.0, confidence=0.95, days_old=1)
    fb = _make_feedback(30, accuracy=0.95)
    result = compute_deployment_health_scorecard(
        logs=logs,
        feedback_records=fb,
        created_at=_now() - timedelta(days=7),
        problem_type="classification",
    )

    assert result["overall_grade"] in ("A", "B")
    assert result["overall_score"] >= 75


def test_empty_logs_handles_gracefully():
    from core.analyzer import compute_deployment_health_scorecard

    result = compute_deployment_health_scorecard(
        logs=[],
        feedback_records=[],
        created_at=_now() - timedelta(days=5),
    )

    assert "overall_grade" in result
    assert len(result["signals"]) == 5


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "show deployment health scorecard",
        "deployment health scorecard",
        "is my deployment healthy",
        "how is my deployment doing",
        "health scorecard",
        "give me a health check",
        "show me the health overview",
        "what is the overall deployment health",
    ],
)
def test_scorecard_patterns_match(message: str):
    from api.chat import _DEPLOY_HEALTH_SCORECARD_PATTERNS

    assert _DEPLOY_HEALTH_SCORECARD_PATTERNS.search(message), f"No match: {message!r}"


@pytest.mark.parametrize(
    "message",
    [
        "what is my model accuracy",
        "show me the canary comparison",
        "monitoring signal digest",
        "should I retrain my model",
        "drift importance ranking",
    ],
)
def test_scorecard_patterns_no_false_positives(message: str):
    from api.chat import _DEPLOY_HEALTH_SCORECARD_PATTERNS

    assert not _DEPLOY_HEALTH_SCORECARD_PATTERNS.search(message), (
        f"Should not match: {message!r}"
    )


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


def _create_deployment(tmp_path, dep_id: str = "hs-dep-1", **kwargs):
    from models.deployment import Deployment

    defaults = dict(
        id=dep_id,
        model_run_id="run-1",
        project_id="proj-1",
        endpoint_path=f"/api/predict/{dep_id}",
        dashboard_url=f"/predict/{dep_id}",
        algorithm="random_forest_regressor",
        target_column="revenue",
        problem_type="regression",
        created_at=_now() - timedelta(days=10),
    )
    defaults.update(kwargs)
    dep = Deployment(**defaults)
    with Session(db_module.engine) as s:
        s.add(dep)
        s.commit()
    return dep_id


def test_endpoint_200_with_required_fields(client, tmp_path):
    dep_id = _create_deployment(tmp_path, dep_id="hs-ep-1")
    resp = client.get(f"/api/deploy/{dep_id}/health-scorecard")
    assert resp.status_code == 200
    data = resp.json()
    assert "overall_grade" in data
    assert data["overall_grade"] in ("A", "B", "C", "D", "F")
    assert "signals" in data
    assert len(data["signals"]) == 5
    assert data["deployment_id"] == dep_id


def test_endpoint_404_for_unknown(client):
    resp = client.get("/api/deploy/nonexistent-999/health-scorecard")
    assert resp.status_code == 404


def test_endpoint_score_in_range(client, tmp_path):
    dep_id = _create_deployment(tmp_path, dep_id="hs-ep-2")
    resp = client.get(f"/api/deploy/{dep_id}/health-scorecard")
    assert resp.status_code == 200
    data = resp.json()
    assert 0 <= data["overall_score"] <= 100


def test_endpoint_includes_algorithm_and_target(client, tmp_path):
    dep_id = _create_deployment(
        tmp_path,
        dep_id="hs-ep-3",
        algorithm="logistic_regression",
        target_column="churn",
        problem_type="classification",
    )
    resp = client.get(f"/api/deploy/{dep_id}/health-scorecard")
    assert resp.status_code == 200
    data = resp.json()
    assert data["algorithm"] == "logistic_regression"
    assert data["target_column"] == "churn"


def test_endpoint_canary_info_included_when_active(client, tmp_path):
    dep_id = _create_deployment(
        tmp_path,
        dep_id="hs-ep-4",
        canary_is_active=True,
        canary_traffic_pct=15,
    )
    resp = client.get(f"/api/deploy/{dep_id}/health-scorecard")
    assert resp.status_code == 200
    data = resp.json()
    assert data["canary_info"] is not None
    assert data["canary_info"]["traffic_pct"] == 15
