"""Tests for the Model Quality Score feature.

Covers:
- compute_model_quality_score() pure function
- /api/models/{run_id}/quality-score REST endpoint
- _QUALITY_PATTERNS regex
"""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")

# ---------------------------------------------------------------------------
# Module-level anyio backend (required for @pytest.mark.anyio tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_regression_excellent():
    from core.advisor import compute_model_quality_score

    r = compute_model_quality_score({"r2": 0.90}, "regression")
    assert r["quality_label"] == "Excellent"
    assert r["color"] == "emerald"
    assert r["quality_score"] > 90
    assert r["is_stable"] is True
    assert "R²" in r["primary_metric_name"]


def test_regression_good():
    from core.advisor import compute_model_quality_score

    r = compute_model_quality_score({"r2": 0.75}, "regression")
    assert r["quality_label"] == "Good"
    assert r["color"] == "blue"


def test_regression_acceptable():
    from core.advisor import compute_model_quality_score

    r = compute_model_quality_score({"r2": 0.60}, "regression")
    assert r["quality_label"] == "Acceptable"
    assert r["color"] == "amber"


def test_regression_needs_work():
    from core.advisor import compute_model_quality_score

    r = compute_model_quality_score({"r2": 0.40}, "regression")
    assert r["quality_label"] == "Needs Work"
    assert r["color"] == "rose"


def test_classification_excellent():
    from core.advisor import compute_model_quality_score

    r = compute_model_quality_score({"accuracy": 0.95}, "classification")
    assert r["quality_label"] == "Excellent"


def test_classification_good():
    from core.advisor import compute_model_quality_score

    r = compute_model_quality_score({"accuracy": 0.82}, "classification")
    assert r["quality_label"] == "Good"


def test_classification_acceptable():
    from core.advisor import compute_model_quality_score

    r = compute_model_quality_score({"accuracy": 0.72}, "classification")
    assert r["quality_label"] == "Acceptable"


def test_classification_needs_work():
    from core.advisor import compute_model_quality_score

    r = compute_model_quality_score({"accuracy": 0.55}, "classification")
    assert r["quality_label"] == "Needs Work"


def test_cv_instability_downgrades_label():
    """High cv_std downgrades quality label one step."""
    from core.advisor import compute_model_quality_score

    # R²=0.90 would normally be Excellent; with cv_std=0.15 → Good
    r = compute_model_quality_score({"r2": 0.90, "cv_std": 0.15, "cv_mean": 0.85}, "regression")
    assert r["quality_label"] == "Good"
    assert r["is_stable"] is False


def test_cv_stability_no_downgrade_when_low():
    """Low cv_std (≤ 0.10) does not downgrade."""
    from core.advisor import compute_model_quality_score

    r = compute_model_quality_score({"r2": 0.90, "cv_std": 0.05, "cv_mean": 0.88}, "regression")
    assert r["quality_label"] == "Excellent"
    assert r["is_stable"] is True


def test_reasoning_bullets_present():
    from core.advisor import compute_model_quality_score

    r = compute_model_quality_score({"r2": 0.73, "cv_mean": 0.71, "cv_std": 0.04}, "regression")
    assert len(r["reasoning"]) >= 2
    assert any("R²" in b or "73" in b for b in r["reasoning"])
    assert any("cross-validation" in b.lower() or "cv" in b.lower() for b in r["reasoning"])


def test_recommendation_populated():
    from core.advisor import compute_model_quality_score

    r = compute_model_quality_score({"r2": 0.60}, "regression")
    assert isinstance(r["recommendation"], str)
    assert len(r["recommendation"]) > 10


def test_small_dataset_adds_note():
    from core.advisor import compute_model_quality_score

    r = compute_model_quality_score({"r2": 0.80}, "regression", n_rows=50)
    assert any("50" in b or "row" in b.lower() for b in r["reasoning"])


def test_nonlinear_recommendation_for_linear_algo():
    from core.advisor import compute_model_quality_score

    r = compute_model_quality_score(
        {"accuracy": 0.72}, "classification", algorithm="logistic_regression"
    )
    # Acceptable + linear algo → suggest nonlinear algorithm
    assert "random" in r["recommendation"].lower() or "xgboost" in r["recommendation"].lower()


def test_nonlinear_recommendation_skipped_for_ensemble():
    from core.advisor import compute_model_quality_score

    r = compute_model_quality_score(
        {"accuracy": 0.72}, "classification", algorithm="voting_classifier"
    )
    # Already nonlinear — should suggest features/data instead
    assert "random forest" not in r["recommendation"].lower()


def test_quality_score_within_range():
    from core.advisor import compute_model_quality_score

    for metric in [0.10, 0.40, 0.60, 0.73, 0.85, 0.95]:
        r = compute_model_quality_score({"r2": metric}, "regression")
        assert 0 <= r["quality_score"] <= 100, f"score out of range for r2={metric}"


def test_cv_fields_returned():
    from core.advisor import compute_model_quality_score

    r = compute_model_quality_score({"r2": 0.80, "cv_mean": 0.78, "cv_std": 0.03}, "regression")
    assert r["cv_mean"] == pytest.approx(0.78)
    assert r["cv_std"] == pytest.approx(0.03)


def test_cv_fields_none_when_absent():
    from core.advisor import compute_model_quality_score

    r = compute_model_quality_score({"r2": 0.80}, "regression")
    assert r["cv_mean"] is None
    assert r["cv_std"] is None


def test_uses_f1_fallback_when_no_accuracy():
    from core.advisor import compute_model_quality_score

    r = compute_model_quality_score({"f1": 0.82}, "classification")
    assert r["primary_metric"] == pytest.approx(0.82)


# ---------------------------------------------------------------------------
# _QUALITY_PATTERNS regex tests
# ---------------------------------------------------------------------------


def _load_pattern():
    from api.chat import _QUALITY_PATTERNS

    return _QUALITY_PATTERNS


def test_pattern_how_good():
    assert _load_pattern().search("how good is my model?")


def test_pattern_is_model_reliable():
    assert _load_pattern().search("is my model reliable enough?")


def test_pattern_rate_model():
    assert _load_pattern().search("can you rate my model?")


def test_pattern_model_quality():
    assert _load_pattern().search("tell me about my model quality")


def test_pattern_is_accuracy_good():
    assert _load_pattern().search("is my accuracy good enough?")


def test_pattern_should_use():
    assert _load_pattern().search("should I use this model?")


def test_pattern_production_ready():
    assert _load_pattern().search("is my model production ready?")


def test_pattern_evaluate_model():
    assert _load_pattern().search("evaluate my model please")


def test_pattern_how_well():
    assert _load_pattern().search("how well does my model perform?")


def test_pattern_assess():
    assert _load_pattern().search("assess my model performance")


def test_no_false_positive_improvement():
    """'improve my model' should NOT match _QUALITY_PATTERNS."""
    assert not _load_pattern().search("how do I improve my model?")


def test_no_false_positive_compare():
    """'compare my models' should NOT match _QUALITY_PATTERNS."""
    assert not _load_pattern().search("compare my models and tell me which is better")


# ---------------------------------------------------------------------------
# REST endpoint tests (async pattern — matches codebase convention)
# ---------------------------------------------------------------------------


async def _insert_quality_run(
    run_id: str,
    project_id: str,
    *,
    algorithm: str = "linear_regression",
    metrics: dict | None = None,
    status: str = "done",
) -> None:
    """Insert a Project + ModelRun directly into the test DB."""
    import db
    from models.model_run import ModelRun
    from models.project import Project

    if metrics is None:
        metrics = {"r2": 0.80, "cv_mean": 0.78, "cv_std": 0.03}

    with next(db.get_session()) as session:
        proj = Project(id=project_id, name=f"Quality Test ({project_id})")
        session.merge(proj)
        run = ModelRun(
            id=run_id,
            project_id=project_id,
            algorithm=algorithm,
            status=status,
            metrics=json.dumps(metrics),
        )
        session.merge(run)
        session.commit()


@pytest.mark.anyio
async def test_quality_score_endpoint_returns_200(client):
    await _insert_quality_run("qs-run-1", "qs-proj-1")
    resp = await client.get("/api/models/qs-run-1/quality-score")
    assert resp.status_code == 200
    data = resp.json()
    assert "quality_label" in data
    assert data["quality_label"] in ("Excellent", "Good", "Acceptable", "Needs Work")
    assert "quality_score" in data
    assert "reasoning" in data
    assert "recommendation" in data
    assert "run_id" in data
    assert data["run_id"] == "qs-run-1"


@pytest.mark.anyio
async def test_quality_score_endpoint_404_on_bad_run(client):
    resp = await client.get("/api/models/nonexistent-run-id/quality-score")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_quality_score_endpoint_contains_primary_metric(client):
    await _insert_quality_run("qs-run-2", "qs-proj-2")
    resp = await client.get("/api/models/qs-run-2/quality-score")
    assert resp.status_code == 200
    data = resp.json()
    assert "primary_metric" in data
    assert isinstance(data["primary_metric"], float)
    assert "primary_metric_name" in data


@pytest.mark.anyio
async def test_quality_score_endpoint_reasoning_is_list(client):
    await _insert_quality_run("qs-run-3", "qs-proj-3")
    resp = await client.get("/api/models/qs-run-3/quality-score")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["reasoning"], list)
    assert len(data["reasoning"]) >= 1
