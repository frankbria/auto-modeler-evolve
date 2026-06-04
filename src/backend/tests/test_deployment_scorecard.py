"""Tests for Deployment Comparison Scorecard.

Covers:
- Pure-function scoring helpers (_usage_score, _sla_score, _freshness_score)
- compute_deployment_scorecard ranking and output shape
- REST endpoint: GET /api/projects/{project_id}/deployment-scorecard
- Chat: _DEPLOY_SCORECARD_PATTERNS regex (positive + false-positive guards)
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from api.chat import _DEPLOY_SCORECARD_PATTERNS
from core.analyzer import (
    _freshness_score,
    _sla_score,
    _usage_score,
    compute_deployment_scorecard,
)
from main import app
from models.deployment import Deployment


# ---------------------------------------------------------------------------
# _usage_score
# ---------------------------------------------------------------------------


def test_usage_score_zero():
    assert _usage_score(0) == 0


def test_usage_score_one():
    assert _usage_score(1) == 0  # log10(1) = 0 → 0 * 25 = 0


def test_usage_score_ten():
    assert _usage_score(10) == 25  # log10(10) = 1 → 25


def test_usage_score_hundred():
    assert _usage_score(100) == 50  # log10(100) = 2 → 50


def test_usage_score_thousand():
    assert _usage_score(1000) == 75  # log10(1000) = 3 → 75


def test_usage_score_ten_thousand_capped():
    assert _usage_score(10000) == 100  # log10(10000) = 4 → 100, capped at 100


def test_usage_score_very_large_capped():
    assert _usage_score(1_000_000) == 100


# ---------------------------------------------------------------------------
# _sla_score
# ---------------------------------------------------------------------------


def test_sla_score_none_returns_none():
    assert _sla_score(None) is None


def test_sla_score_100ms():
    assert _sla_score(100.0) == 100


def test_sla_score_below_100ms():
    assert _sla_score(50.0) == 100


def test_sla_score_above_2000ms():
    assert _sla_score(2000.0) == 0


def test_sla_score_very_slow():
    assert _sla_score(5000.0) == 0


def test_sla_score_mid_range():
    # Linear decay: 100ms → 2000ms over 100 points
    # At 1100ms: 100 - (1100-100)*100/1900 ≈ 47
    score = _sla_score(1100.0)
    assert 40 <= score <= 55


def test_sla_score_non_negative():
    assert _sla_score(3000.0) >= 0


# ---------------------------------------------------------------------------
# _freshness_score
# ---------------------------------------------------------------------------


def test_freshness_under_7_days():
    assert _freshness_score(3) == 100


def test_freshness_7_to_30():
    assert _freshness_score(15) == 80


def test_freshness_30_to_60():
    assert _freshness_score(45) == 60


def test_freshness_60_to_90():
    assert _freshness_score(75) == 40


def test_freshness_90_to_180():
    assert _freshness_score(120) == 20


def test_freshness_over_180():
    assert _freshness_score(200) == 0


# ---------------------------------------------------------------------------
# compute_deployment_scorecard
# ---------------------------------------------------------------------------


def _entry(
    dep_id="dep-1",
    algorithm="random_forest_regressor",
    algorithm_plain="Random Forest",
    target="revenue",
    environment="production",
    request_count=100,
    feedback_accuracy=None,
    p95_ms=None,
    age_days=10,
):
    return {
        "deployment_id": dep_id,
        "algorithm": algorithm,
        "algorithm_plain": algorithm_plain,
        "target_column": target,
        "environment": environment,
        "request_count": request_count,
        "feedback_accuracy": feedback_accuracy,
        "p95_ms": p95_ms,
        "age_days": age_days,
        "deployed_at": None,
        "dashboard_url": None,
    }


def test_empty_returns_no_winner():
    result = compute_deployment_scorecard([])
    assert result["total"] == 0
    assert result["winner_id"] is None
    assert isinstance(result["summary"], str)
    assert result["entries"] == []


def test_single_entry_required_keys():
    result = compute_deployment_scorecard([_entry()])
    assert result["total"] == 1
    assert result["winner_id"] == "dep-1"
    entry = result["entries"][0]
    for key in ["rank", "composite_score", "usage_score", "freshness_score"]:
        assert key in entry


def test_single_entry_rank_one():
    result = compute_deployment_scorecard([_entry()])
    assert result["entries"][0]["rank"] == 1


def test_composite_score_in_range():
    result = compute_deployment_scorecard([_entry(request_count=50, age_days=20)])
    score = result["entries"][0]["composite_score"]
    assert 0 <= score <= 100


def test_ranking_higher_usage_wins():
    entries = [
        _entry("low", request_count=1, age_days=5),
        _entry("high", request_count=10000, age_days=5),
    ]
    result = compute_deployment_scorecard(entries)
    assert result["winner_id"] == "high"
    assert result["entries"][0]["rank"] == 1
    assert result["entries"][1]["rank"] == 2


def test_feedback_accuracy_boosts_score():
    low = _entry("low", request_count=100, feedback_accuracy=0.50)
    high = _entry("high", request_count=100, feedback_accuracy=0.95)
    result = compute_deployment_scorecard([low, high])
    assert result["winner_id"] == "high"


def test_sla_score_included_when_provided():
    entry = _entry(p95_ms=150.0)
    result = compute_deployment_scorecard([entry])
    assert result["entries"][0]["sla_score"] is not None
    assert result["entries"][0]["sla_score"] > 0


def test_sla_score_none_when_not_provided():
    entry = _entry(p95_ms=None)
    result = compute_deployment_scorecard([entry])
    assert result["entries"][0]["sla_score"] is None


def test_accuracy_score_none_when_no_feedback():
    entry = _entry(feedback_accuracy=None)
    result = compute_deployment_scorecard([entry])
    assert result["entries"][0]["accuracy_score"] is None


def test_accuracy_score_computed_from_feedback():
    entry = _entry(feedback_accuracy=0.75)
    result = compute_deployment_scorecard([entry])
    assert result["entries"][0]["accuracy_score"] == 75


def test_summary_is_nonempty_string():
    result = compute_deployment_scorecard([_entry()])
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0


def test_multi_entry_summary_mentions_winner():
    e1 = _entry("d1", algorithm_plain="Random Forest", request_count=100)
    e2 = _entry("d2", algorithm_plain="Linear Regression", request_count=1)
    result = compute_deployment_scorecard([e1, e2])
    assert "Random Forest" in result["summary"]


def test_three_entries_correct_rank_order():
    entries = [
        _entry("d3", request_count=1, age_days=200),
        _entry("d1", request_count=1000, age_days=5),
        _entry("d2", request_count=100, age_days=30),
    ]
    result = compute_deployment_scorecard(entries)
    assert result["entries"][0]["deployment_id"] == "d1"
    assert result["entries"][-1]["deployment_id"] == "d3"


# ---------------------------------------------------------------------------
# REST endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    return TestClient(app)


def test_scorecard_empty_project(client):
    project_id = str(uuid.uuid4())
    resp = client.get(f"/api/projects/{project_id}/deployment-scorecard")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["winner_id"] is None


def test_scorecard_returns_active_deployments(client):
    from db import engine

    project_id = str(uuid.uuid4())
    dep_id = str(uuid.uuid4())
    with Session(engine) as s:
        dep = Deployment(
            id=dep_id,
            project_id=project_id,
            model_run_id=str(uuid.uuid4()),
            algorithm="linear_regression",
            target_column="revenue",
            endpoint_path=f"/api/predict/{dep_id}",
            dashboard_url=f"/predict/{dep_id}",
            is_active=True,
            request_count=42,
        )
        s.add(dep)
        s.commit()

    resp = client.get(f"/api/projects/{project_id}/deployment-scorecard")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["winner_id"] == dep_id
    entry = data["entries"][0]
    assert entry["request_count"] == 42
    assert entry["composite_score"] >= 0


def test_scorecard_inactive_not_included(client):
    from db import engine

    project_id = str(uuid.uuid4())
    dep_id = str(uuid.uuid4())
    with Session(engine) as s:
        dep = Deployment(
            id=dep_id,
            project_id=project_id,
            model_run_id=str(uuid.uuid4()),
            algorithm="linear_regression",
            target_column="revenue",
            endpoint_path=f"/api/predict/{dep_id}",
            dashboard_url=f"/predict/{dep_id}",
            is_active=False,
            request_count=100,
        )
        s.add(dep)
        s.commit()

    resp = client.get(f"/api/projects/{project_id}/deployment-scorecard")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0


# ---------------------------------------------------------------------------
# Chat regex patterns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "msg",
    [
        "rank my deployments by performance",
        "rank all my deployments",
        "deployment scorecard",
        "deployment comparison scorecard",
        "which deployment performs best",
        "which deployment is best",
        "compare my deployments by performance",
        "deployment leaderboard",
        "best-performing deployment",
        "how do my deployments compare",
        "production performance ranking",
    ],
)
def test_scorecard_pattern_positive(msg):
    assert _DEPLOY_SCORECARD_PATTERNS.search(msg), f"Should match: {msg!r}"


@pytest.mark.parametrize(
    "msg",
    [
        "train a model",
        "how is my model doing",
        "show me the correlation matrix",
        "deployment dashboard",
        "list my deployments",
        "rollback my model",
    ],
)
def test_scorecard_pattern_false_positive_guard(msg):
    assert not _DEPLOY_SCORECARD_PATTERNS.search(msg), f"Should NOT match: {msg!r}"
