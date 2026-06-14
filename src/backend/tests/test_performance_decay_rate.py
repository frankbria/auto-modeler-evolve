"""Tests for Model Performance Decay Rate Analysis.

Covers:
- compute_performance_decay_rate() pure function (no_data variants, stable,
  improving, degrading_slowly, degrading_fast, below_threshold, threshold estimate)
- _PERF_DECAY_PATTERNS regex (8 positive, 3 negative)
- GET /api/deploy/{deployment_id}/performance-decay-rate REST endpoint (200, 404, no_data)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine, Session

import db as db_module

# ---------------------------------------------------------------------------
# Pure-function constants
# ---------------------------------------------------------------------------

_NO_DATA_VERDICT = "no_data"
_STABLE_VERDICT = "stable"
_IMPROVING_VERDICT = "improving"
_DEGRADING_SLOW_VERDICT = "degrading_slowly"
_DEGRADING_FAST_VERDICT = "degrading_fast"
_BELOW_THRESHOLD_VERDICT = "below_threshold"


def test_perf_decay_verdicts_are_constants():
    assert _NO_DATA_VERDICT == "no_data"
    assert _STABLE_VERDICT == "stable"
    assert _IMPROVING_VERDICT == "improving"
    assert _DEGRADING_SLOW_VERDICT == "degrading_slowly"
    assert _DEGRADING_FAST_VERDICT == "degrading_fast"
    assert _BELOW_THRESHOLD_VERDICT == "below_threshold"


# ---------------------------------------------------------------------------
# Pure-function helpers
# ---------------------------------------------------------------------------


def _make_pair(is_correct: bool, week_offset: int = 0) -> dict:
    """Make a feedback pair for a specific week offset from 2025-01-06 (Monday)."""
    from datetime import date, timedelta

    d = date(2025, 1, 6) + timedelta(weeks=week_offset)
    return {"created_at": f"{d.isoformat()}T10:00:00", "is_correct": is_correct}


def _pairs_for_week(is_correct_list: list[bool], week_offset: int) -> list[dict]:
    return [_make_pair(c, week_offset) for c in is_correct_list]


# ---------------------------------------------------------------------------
# no_data: empty list
# ---------------------------------------------------------------------------


def test_perf_decay_empty_list():
    from core.analyzer import compute_performance_decay_rate

    r = compute_performance_decay_rate([])
    assert r["verdict"] == "no_data"
    assert r["weekly_data"] == []
    assert r["slope_pct_per_week"] is None
    assert r["weeks_until_threshold"] is None
    assert r["current_accuracy_pct"] is None


def test_perf_decay_too_few_pairs():
    from core.analyzer import compute_performance_decay_rate

    pairs = [_make_pair(True, 0), _make_pair(False, 0), _make_pair(True, 1)]
    r = compute_performance_decay_rate(pairs)
    assert r["verdict"] == "no_data"


def test_perf_decay_only_one_week():
    from core.analyzer import compute_performance_decay_rate

    # 6 pairs all in the same week → not enough weeks
    pairs = [_make_pair(i % 2 == 0, 0) for i in range(6)]
    r = compute_performance_decay_rate(pairs)
    assert r["verdict"] == "no_data"


# ---------------------------------------------------------------------------
# stable — slope between -0.5 and +0.5
# ---------------------------------------------------------------------------


def test_perf_decay_stable():
    from core.analyzer import compute_performance_decay_rate

    # 5 weeks all at 80% accuracy (10 pairs per week); threshold set low at 60%
    pairs = []
    for w in range(5):
        pairs += _pairs_for_week([True] * 8 + [False] * 2, w)

    r = compute_performance_decay_rate(pairs, threshold_pct=60.0)
    assert r["verdict"] == "stable"
    assert r["slope_pct_per_week"] is not None
    assert abs(r["slope_pct_per_week"]) < 0.5
    assert r["current_accuracy_pct"] is not None


# ---------------------------------------------------------------------------
# improving — slope >= 0.5
# ---------------------------------------------------------------------------


def test_perf_decay_improving():
    from core.analyzer import compute_performance_decay_rate

    # Accuracy improves week by week: 70%, 75%, 80%, 85%, 90%
    # 10 pairs per week
    accuracies = [0.7, 0.75, 0.80, 0.85, 0.90]
    pairs = []
    for w, acc in enumerate(accuracies):
        n_correct = int(acc * 10)
        pairs += _pairs_for_week([True] * n_correct + [False] * (10 - n_correct), w)

    r = compute_performance_decay_rate(pairs, threshold_pct=60.0)
    assert r["verdict"] == "improving"
    assert r["slope_pct_per_week"] is not None and r["slope_pct_per_week"] > 0
    assert r["weeks_until_threshold"] is None  # improving, so no ETA below threshold


# ---------------------------------------------------------------------------
# degrading_slowly — slope between -2 and -0.5
# ---------------------------------------------------------------------------


def test_perf_decay_degrading_slowly():
    from core.analyzer import compute_performance_decay_rate

    # Accuracy drops: 95%, 94%, 93%, 92%, 91% (−1% per week); threshold at 80%
    # so current is clearly above threshold
    accuracies = [0.95, 0.94, 0.93, 0.92, 0.91]
    pairs = []
    for w, acc in enumerate(accuracies):
        n_correct = int(acc * 100)
        pairs += _pairs_for_week([True] * n_correct + [False] * (100 - n_correct), w)

    r = compute_performance_decay_rate(pairs, threshold_pct=80.0)
    assert r["verdict"] == "degrading_slowly"
    assert r["slope_pct_per_week"] is not None and r["slope_pct_per_week"] < 0
    # Weeks until threshold should be estimated
    assert r["weeks_until_threshold"] is not None and r["weeks_until_threshold"] > 0


# ---------------------------------------------------------------------------
# degrading_fast — slope <= -2
# ---------------------------------------------------------------------------


def test_perf_decay_degrading_fast():
    from core.analyzer import compute_performance_decay_rate

    # Accuracy drops fast: 95%, 90%, 85%, 80%, 75% (−5% per week)
    accuracies = [0.95, 0.90, 0.85, 0.80, 0.75]
    pairs = []
    for w, acc in enumerate(accuracies):
        n_correct = int(acc * 20)
        pairs += _pairs_for_week([True] * n_correct + [False] * (20 - n_correct), w)

    r = compute_performance_decay_rate(pairs, threshold_pct=80.0)
    # degrading_fast OR below_threshold if current <= 80
    assert r["verdict"] in ("degrading_fast", "below_threshold")
    assert r["slope_pct_per_week"] is not None and r["slope_pct_per_week"] < -2


# ---------------------------------------------------------------------------
# below_threshold
# ---------------------------------------------------------------------------


def test_perf_decay_below_threshold():
    from core.analyzer import compute_performance_decay_rate

    # 5 weeks all at ~70% — below 80% threshold
    pairs = []
    for w in range(5):
        pairs += _pairs_for_week([True] * 7 + [False] * 3, w)

    r = compute_performance_decay_rate(pairs, threshold_pct=80.0)
    assert r["verdict"] == "below_threshold"
    assert r["weeks_until_threshold"] == 0.0
    assert "below" in r["summary"].lower() or "threshold" in r["summary"].lower()


# ---------------------------------------------------------------------------
# threshold estimation sanity
# ---------------------------------------------------------------------------


def test_perf_decay_threshold_estimate_reasonable():
    from core.analyzer import compute_performance_decay_rate

    # Current accuracy ~90%, slope ~−1% per week, threshold 80%
    # Expect weeks_until ~ 10
    accuracies = [0.90, 0.89, 0.88, 0.87, 0.86]
    pairs = []
    for w, acc in enumerate(accuracies):
        n_correct = int(acc * 100)
        pairs += _pairs_for_week([True] * n_correct + [False] * (100 - n_correct), w)

    r = compute_performance_decay_rate(pairs, threshold_pct=80.0)
    assert r["weeks_until_threshold"] is not None
    # Should be roughly (86 - 80) / 1 = ~6 weeks from the current end point
    assert 2 <= r["weeks_until_threshold"] <= 30  # generous range


# ---------------------------------------------------------------------------
# Threshold parameter is reflected in result
# ---------------------------------------------------------------------------


def test_perf_decay_threshold_pct_reflected():
    from core.analyzer import compute_performance_decay_rate

    pairs = []
    for w in range(4):
        pairs += _pairs_for_week([True] * 8 + [False] * 2, w)

    r = compute_performance_decay_rate(pairs, threshold_pct=70.0)
    assert r["threshold_pct"] == 70.0


# ---------------------------------------------------------------------------
# weekly_data structure
# ---------------------------------------------------------------------------


def test_perf_decay_weekly_data_structure():
    from core.analyzer import compute_performance_decay_rate

    pairs = []
    for w in range(3):
        pairs += _pairs_for_week([True] * 8 + [False] * 2, w)

    r = compute_performance_decay_rate(pairs, threshold_pct=80.0)
    assert len(r["weekly_data"]) == 3
    for entry in r["weekly_data"]:
        assert "week" in entry
        assert "accuracy_pct" in entry
        assert "n_samples" in entry
        assert 0 <= entry["accuracy_pct"] <= 100


def test_perf_decay_weekly_data_sorted():
    """Weeks must be in ascending chronological order."""
    from core.analyzer import compute_performance_decay_rate

    pairs = []
    for w in range(4):
        pairs += _pairs_for_week([True] * 7 + [False] * 3, w)

    r = compute_performance_decay_rate(pairs, threshold_pct=80.0)
    weeks = [e["week"] for e in r["weekly_data"]]
    assert weeks == sorted(weeks)


# ---------------------------------------------------------------------------
# Regex: _PERF_DECAY_PATTERNS
# ---------------------------------------------------------------------------


def test_perf_decay_pattern_how_fast_degrading():
    from api.chat import _PERF_DECAY_PATTERNS

    assert _PERF_DECAY_PATTERNS.search("how fast is my model degrading?")


def test_perf_decay_pattern_model_decay():
    from api.chat import _PERF_DECAY_PATTERNS

    assert _PERF_DECAY_PATTERNS.search("model decay rate")


def test_perf_decay_pattern_performance_decay():
    from api.chat import _PERF_DECAY_PATTERNS

    assert _PERF_DECAY_PATTERNS.search("show me the performance decay over time")


def test_perf_decay_pattern_accuracy_decline():
    from api.chat import _PERF_DECAY_PATTERNS

    assert _PERF_DECAY_PATTERNS.search("accuracy decline rate")


def test_perf_decay_pattern_how_fast_accuracy_dropping():
    from api.chat import _PERF_DECAY_PATTERNS

    assert _PERF_DECAY_PATTERNS.search("how fast is accuracy dropping?")


def test_perf_decay_pattern_when_will_accuracy_fall():
    from api.chat import _PERF_DECAY_PATTERNS

    assert _PERF_DECAY_PATTERNS.search("when will my model accuracy fall below 80%?")


def test_perf_decay_pattern_weeks_until_degrades():
    from api.chat import _PERF_DECAY_PATTERNS

    assert _PERF_DECAY_PATTERNS.search("weeks until my model degrades?")


def test_perf_decay_pattern_is_model_getting_worse():
    from api.chat import _PERF_DECAY_PATTERNS

    assert _PERF_DECAY_PATTERNS.search("is my model getting worse over time?")


def test_perf_decay_pattern_no_match_train():
    from api.chat import _PERF_DECAY_PATTERNS

    assert not _PERF_DECAY_PATTERNS.search("train a random forest")


def test_perf_decay_pattern_no_match_batch():
    from api.chat import _PERF_DECAY_PATTERNS

    assert not _PERF_DECAY_PATTERNS.search("schedule a batch job")


def test_perf_decay_pattern_no_match_upload():
    from api.chat import _PERF_DECAY_PATTERNS

    assert not _PERF_DECAY_PATTERNS.search("upload my dataset")


# ---------------------------------------------------------------------------
# REST endpoint
# ---------------------------------------------------------------------------


@pytest.fixture()
def _test_engine_and_client(tmp_path):
    """In-memory SQLite + TestClient with an active deployment."""
    from main import app
    from models.project import Project
    from models.dataset import Dataset
    from models.model_run import ModelRun
    from models.deployment import Deployment
    from models.feedback_record import FeedbackRecord
    from datetime import datetime, date, timedelta
    import uuid

    sqlite_url = f"sqlite:///{tmp_path}/test.db"
    engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)

    _orig = db_module.engine
    db_module.engine = engine

    try:
        with Session(engine) as s:
            proj = Project(
                id=str(uuid.uuid4()), name="Test", created_at=datetime.utcnow()
            )
            ds = Dataset(
                id=str(uuid.uuid4()),
                project_id=proj.id,
                filename="t.csv",
                file_path="/tmp/t.csv",
                created_at=datetime.utcnow(),
            )
            run = ModelRun(
                id=str(uuid.uuid4()),
                project_id=proj.id,
                algorithm="random_forest",
                status="done",
                created_at=datetime.utcnow(),
            )
            _dep_id = str(uuid.uuid4())
            dep = Deployment(
                id=_dep_id,
                project_id=proj.id,
                model_run_id=run.id,
                is_active=True,
                endpoint_path=f"/api/predict/{_dep_id}",
                dashboard_url=f"/dashboard/{_dep_id}",
                created_at=datetime.utcnow(),
            )
            s.add_all([proj, ds, run, dep])

            # Add 5 weeks of feedback (10 pairs per week, 85% accuracy)
            for w in range(5):
                day = date(2025, 1, 6) + timedelta(weeks=w)
                for i in range(10):
                    fb = FeedbackRecord(
                        id=str(uuid.uuid4()),
                        deployment_id=dep.id,
                        is_correct=(i < 8),  # 8/10 = 80%
                        created_at=datetime(day.year, day.month, day.day, 10, 0, 0),
                    )
                    s.add(fb)
            s.commit()

        yield engine, TestClient(app), _dep_id

    finally:
        db_module.engine = _orig


def test_perf_decay_rest_200(_test_engine_and_client):
    _, client, dep_id = _test_engine_and_client
    resp = client.get(f"/api/deploy/{dep_id}/performance-decay-rate")
    assert resp.status_code == 200
    data = resp.json()
    assert data["deployment_id"] == dep_id
    assert "verdict" in data
    assert "weekly_data" in data
    assert "summary" in data


def test_perf_decay_rest_404(_test_engine_and_client):
    _, client, _dep_id = _test_engine_and_client
    resp = client.get("/api/deploy/nonexistent-id/performance-decay-rate")
    assert resp.status_code == 404


def test_perf_decay_rest_no_feedback(_test_engine_and_client, tmp_path):
    """Deployment with no feedback returns no_data."""
    from main import app
    from models.project import Project
    from models.model_run import ModelRun
    from models.deployment import Deployment
    from datetime import datetime
    import uuid

    _, _, _ = _test_engine_and_client  # just to trigger the fixture setup

    engine = create_engine(
        f"sqlite:///{tmp_path}/test2.db", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)
    orig = db_module.engine
    db_module.engine = engine
    try:
        with Session(engine) as s:
            proj = Project(
                id=str(uuid.uuid4()), name="P2", created_at=datetime.utcnow()
            )
            run = ModelRun(
                id=str(uuid.uuid4()),
                project_id=proj.id,
                algorithm="random_forest",
                status="done",
                created_at=datetime.utcnow(),
            )
            _dep_id = str(uuid.uuid4())
            dep = Deployment(
                id=_dep_id,
                project_id=proj.id,
                model_run_id=run.id,
                is_active=True,
                endpoint_path=f"/api/predict/{_dep_id}",
                dashboard_url=f"/dashboard/{_dep_id}",
                created_at=datetime.utcnow(),
            )
            s.add_all([proj, run, dep])
            s.commit()
        with TestClient(app) as c:
            resp = c.get(f"/api/deploy/{_dep_id}/performance-decay-rate")
        assert resp.status_code == 200
        assert resp.json()["verdict"] == "no_data"
    finally:
        db_module.engine = orig
