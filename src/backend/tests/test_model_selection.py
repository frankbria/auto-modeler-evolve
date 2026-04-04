"""Tests for core/advisor.py — Model Selection Advisor.

Pure-function tests: compute_model_selection, scoring helpers, criteria detection.
API integration tests: GET /api/models/{project_id}/model-selection.
"""

import json
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel

from core.advisor import (
    compute_model_selection,
    _score_run,
    _accuracy_score,
    _explainability_score,
    _stability_score,
    _speed_score,
)

# ---------------------------------------------------------------------------
# Sample run data
# ---------------------------------------------------------------------------

REGRESSION_RUNS = [
    {
        "run_id": "r1",
        "algorithm": "linear_regression",
        "metrics": {"r2": 0.72, "cv_mean": 0.70, "cv_std": 0.02},
        "problem_type": "regression",
        "is_selected": True,
        "is_deployed": False,
    },
    {
        "run_id": "r2",
        "algorithm": "random_forest_regressor",
        "metrics": {"r2": 0.88, "cv_mean": 0.86, "cv_std": 0.05},
        "problem_type": "regression",
        "is_selected": False,
        "is_deployed": True,
    },
    {
        "run_id": "r3",
        "algorithm": "xgboost_regressor",
        "metrics": {"r2": 0.91, "cv_mean": 0.89, "cv_std": 0.08},
        "problem_type": "regression",
        "is_selected": False,
        "is_deployed": False,
    },
]

CLASSIFICATION_RUNS = [
    {
        "run_id": "c1",
        "algorithm": "logistic_regression",
        "metrics": {"accuracy": 0.82, "f1": 0.80, "cv_mean": 0.81, "cv_std": 0.01},
        "problem_type": "classification",
        "is_selected": False,
        "is_deployed": False,
    },
    {
        "run_id": "c2",
        "algorithm": "random_forest_classifier",
        "metrics": {"accuracy": 0.91, "f1": 0.89, "cv_mean": 0.89, "cv_std": 0.04},
        "problem_type": "classification",
        "is_selected": True,
        "is_deployed": False,
    },
]


# ---------------------------------------------------------------------------
# _accuracy_score
# ---------------------------------------------------------------------------


def test_accuracy_score_clamps_above_one():
    assert _accuracy_score(1.5) == 1.0


def test_accuracy_score_clamps_below_zero():
    assert _accuracy_score(-0.3) == 0.0


def test_accuracy_score_passthrough():
    assert abs(_accuracy_score(0.84) - 0.84) < 1e-6


# ---------------------------------------------------------------------------
# _explainability_score
# ---------------------------------------------------------------------------


def test_linear_regression_most_explainable():
    score_linear = _explainability_score("linear_regression")
    score_rf = _explainability_score("random_forest_regressor")
    score_xgb = _explainability_score("xgboost_regressor")
    assert score_linear > score_rf > score_xgb


def test_unknown_algorithm_gets_lowest_explainability():
    score_known = _explainability_score("stacking_regressor")
    score_unknown = _explainability_score("totally_unknown_algo")
    assert score_unknown <= score_known


def test_explainability_score_range():
    score = _explainability_score("logistic_regression")
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# _stability_score
# ---------------------------------------------------------------------------


def test_stability_score_no_cv_data():
    score = _stability_score({})
    assert score == 0.5


def test_stability_score_very_stable():
    score = _stability_score({"cv_mean": 0.85, "cv_std": 0.001})
    assert score > 0.9


def test_stability_score_unstable():
    score = _stability_score({"cv_mean": 0.6, "cv_std": 0.3})
    assert score < 0.1


def test_stability_score_zero_mean_fallback():
    assert _stability_score({"cv_mean": 0.0, "cv_std": 0.1}) == 0.5


# ---------------------------------------------------------------------------
# _speed_score
# ---------------------------------------------------------------------------


def test_linear_regression_fastest():
    score_linear = _speed_score("linear_regression")
    score_mlp = _speed_score("mlp_regressor")
    assert score_linear > score_mlp


def test_speed_score_range():
    assert 0.0 <= _speed_score("stacking_classifier") <= 1.0


# ---------------------------------------------------------------------------
# _score_run
# ---------------------------------------------------------------------------


def test_score_run_returns_required_fields():
    scored = _score_run(REGRESSION_RUNS[0], "accuracy")
    for key in [
        "run_id",
        "algorithm",
        "algorithm_plain",
        "score",
        "primary_metric",
        "primary_metric_name",
        "component_scores",
        "why",
        "is_selected",
        "is_deployed",
    ]:
        assert key in scored, f"Missing key: {key}"


def test_score_run_accuracy_criteria_uses_primary_metric():
    scored = _score_run(REGRESSION_RUNS[2], "accuracy")  # xgboost r2=0.91
    assert abs(scored["score"] - 0.91) < 1e-4


def test_score_run_explainability_linear_beats_xgb():
    scored_linear = _score_run(REGRESSION_RUNS[0], "explainability")
    scored_xgb = _score_run(REGRESSION_RUNS[2], "explainability")
    assert scored_linear["score"] > scored_xgb["score"]


def test_score_run_balanced_between_zero_and_one():
    for run in REGRESSION_RUNS:
        scored = _score_run(run, "balanced")
        assert 0.0 <= scored["score"] <= 1.0


# ---------------------------------------------------------------------------
# compute_model_selection
# ---------------------------------------------------------------------------


def test_empty_runs_returns_no_winner():
    result = compute_model_selection([], criteria="accuracy")
    assert result["winner"] is None
    assert result["ranked_runs"] == []
    assert result["n_runs"] == 0


def test_accuracy_criteria_picks_highest_r2():
    result = compute_model_selection(REGRESSION_RUNS, criteria="accuracy")
    assert result["winner"]["algorithm"] == "xgboost_regressor"


def test_explainability_criteria_picks_linear():
    result = compute_model_selection(REGRESSION_RUNS, criteria="explainability")
    assert result["winner"]["algorithm"] == "linear_regression"


def test_balanced_criteria_returns_winner():
    result = compute_model_selection(REGRESSION_RUNS, criteria="balanced")
    assert result["winner"] is not None
    assert result["winner"]["rank"] == 1


def test_ranked_runs_sorted_by_score_descending():
    result = compute_model_selection(REGRESSION_RUNS, criteria="accuracy")
    scores = [r["score"] for r in result["ranked_runs"]]
    assert scores == sorted(scores, reverse=True)


def test_ranks_are_sequential_from_one():
    result = compute_model_selection(REGRESSION_RUNS, criteria="balanced")
    ranks = [r["rank"] for r in result["ranked_runs"]]
    assert ranks == list(range(1, len(ranks) + 1))


def test_invalid_criteria_falls_back_to_balanced():
    result = compute_model_selection(REGRESSION_RUNS, criteria="nonsense_criteria")
    assert result["criteria"] == "balanced"


def test_single_run_still_works():
    result = compute_model_selection([REGRESSION_RUNS[0]], criteria="accuracy")
    assert result["n_runs"] == 1
    assert result["winner"]["rank"] == 1


def test_classification_uses_accuracy_metric():
    result = compute_model_selection(CLASSIFICATION_RUNS, criteria="accuracy")
    winner = result["winner"]
    assert winner["primary_metric_name"] == "accuracy"
    assert winner["algorithm"] == "random_forest_classifier"


def test_speed_criteria_picks_fastest():
    result = compute_model_selection(REGRESSION_RUNS, criteria="speed")
    assert result["winner"]["algorithm"] == "linear_regression"


def test_stability_criteria_picks_most_consistent():
    result = compute_model_selection(CLASSIFICATION_RUNS, criteria="stability")
    assert result["winner"]["algorithm"] == "logistic_regression"


def test_result_has_criteria_description():
    result = compute_model_selection(REGRESSION_RUNS, criteria="explainability")
    assert "explainable" in result["criteria_description"].lower()


def test_summary_is_non_empty_string():
    result = compute_model_selection(REGRESSION_RUNS, criteria="accuracy")
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 0


def test_is_selected_flag_preserved():
    result = compute_model_selection(REGRESSION_RUNS, criteria="accuracy")
    run_r1 = next(r for r in result["ranked_runs"] if r["run_id"] == "r1")
    assert run_r1["is_selected"] is True


def test_is_deployed_flag_preserved():
    result = compute_model_selection(REGRESSION_RUNS, criteria="balanced")
    run_r2 = next(r for r in result["ranked_runs"] if r["run_id"] == "r2")
    assert run_r2["is_deployed"] is True


def test_component_scores_all_present():
    result = compute_model_selection(REGRESSION_RUNS, criteria="balanced")
    for run in result["ranked_runs"]:
        for key in ["accuracy", "explainability", "stability", "speed"]:
            assert key in run["component_scores"]
            assert 0.0 <= run["component_scores"][key] <= 1.0


# ---------------------------------------------------------------------------
# _detect_selection_criteria (chat helper in api/chat.py)
# ---------------------------------------------------------------------------


def test_detect_criteria_explainability():
    from api.chat import _detect_selection_criteria as _dsc

    assert _dsc("pick the most explainable model") == "explainability"
    assert _dsc("I want a model I can explain to my boss") == "explainability"
    assert _dsc("transparent model for audit") == "explainability"


def test_detect_criteria_accuracy():
    from api.chat import _detect_selection_criteria as _dsc

    assert _dsc("I want the most accurate model") == "accuracy"
    assert _dsc("highest accuracy please") == "accuracy"


def test_detect_criteria_speed():
    from api.chat import _detect_selection_criteria as _dsc

    assert _dsc("fastest model for real-time API") == "speed"
    assert _dsc("low latency predictions") == "speed"


def test_detect_criteria_stability():
    from api.chat import _detect_selection_criteria as _dsc

    assert _dsc("most stable and reliable model") == "stability"
    assert _dsc("consistent model") == "stability"


def test_detect_criteria_balanced_default():
    from api.chat import _detect_selection_criteria as _dsc

    assert _dsc("which model should I use?") == "balanced"
    assert _dsc("pick the best model for me") == "balanced"
    assert _dsc("recommend a model") == "balanced"


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


async def _make_project_with_runs(tmp_path):
    """Create a project with two completed model runs and return project_id."""
    import db
    from models.project import Project
    from models.model_run import ModelRun

    SQLModel.metadata.create_all(db.engine)
    with next(db.get_session()) as session:
        proj = Project(id="sel-proj-1", name="Selection Test Project")
        session.merge(proj)
        run1 = ModelRun(
            id="sel-run-1",
            project_id="sel-proj-1",
            algorithm="linear_regression",
            status="done",
            metrics=json.dumps({"r2": 0.75, "cv_mean": 0.73, "cv_std": 0.02}),
            is_selected=True,
        )
        run2 = ModelRun(
            id="sel-run-2",
            project_id="sel-proj-1",
            algorithm="random_forest_regressor",
            status="done",
            metrics=json.dumps({"r2": 0.90, "cv_mean": 0.88, "cv_std": 0.05}),
            is_selected=False,
        )
        session.merge(run1)
        session.merge(run2)
        session.commit()
    return "sel-proj-1"


@pytest.mark.anyio
async def test_model_selection_endpoint_accuracy(tmp_path, set_test_env):
    from main import app
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / "sel_test.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    project_id = await _make_project_with_runs(tmp_path)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get(
            f"/api/models/{project_id}/model-selection",
            params={"criteria": "accuracy"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["criteria"] == "accuracy"
    assert data["winner"] is not None
    assert data["winner"]["algorithm"] == "random_forest_regressor"
    assert len(data["ranked_runs"]) == 2


@pytest.mark.anyio
async def test_model_selection_endpoint_explainability(tmp_path, set_test_env):
    from main import app
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / "sel_expl.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    project_id = await _make_project_with_runs(tmp_path)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get(
            f"/api/models/{project_id}/model-selection",
            params={"criteria": "explainability"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["winner"]["algorithm"] == "linear_regression"


@pytest.mark.anyio
async def test_model_selection_endpoint_invalid_criteria(tmp_path, set_test_env):
    from main import app
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / "sel_bad.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    project_id = await _make_project_with_runs(tmp_path)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get(
            f"/api/models/{project_id}/model-selection",
            params={"criteria": "bogus"},
        )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_model_selection_endpoint_unknown_project(tmp_path, set_test_env):
    from main import app
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / "sel_404.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/models/nonexistent-id/model-selection")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_model_selection_endpoint_no_completed_runs(tmp_path, set_test_env):
    from main import app
    import db
    from models.project import Project
    from sqlmodel import create_engine

    test_db = str(tmp_path / "sel_norun.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)

    with next(db.get_session()) as session:
        proj = Project(id="empty-proj", name="Empty Project")
        session.merge(proj)
        session.commit()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/models/empty-proj/model-selection")
    assert resp.status_code == 404
