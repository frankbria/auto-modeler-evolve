"""Tests for compute_low_accuracy_guidance() and the proactive chat handler."""

from __future__ import annotations

import json
import unittest.mock as mock

import pytest
from core.advisor import (
    LOW_ACCURACY_GUIDANCE_THRESHOLD,
    compute_low_accuracy_guidance,
)

# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


class TestLowAccuracyGuidanceThreshold:
    def test_threshold_is_float(self):
        assert isinstance(LOW_ACCURACY_GUIDANCE_THRESHOLD, float)

    def test_threshold_below_ensemble_auto(self):
        """Guidance threshold must be below ensemble auto-suggest threshold (0.75)."""
        from api.chat import _ENSEMBLE_AUTO_THRESHOLD

        assert LOW_ACCURACY_GUIDANCE_THRESHOLD < _ENSEMBLE_AUTO_THRESHOLD

    def test_threshold_value(self):
        assert LOW_ACCURACY_GUIDANCE_THRESHOLD == 0.70


class TestComputeLowAccuracyGuidance:
    def _call(self, **kwargs):
        defaults = dict(
            problem_type="regression",
            best_score=0.55,
            n_rows=300,
            has_ensemble=True,
            has_date_columns=False,
            n_features=5,
        )
        defaults.update(kwargs)
        return compute_low_accuracy_guidance(**defaults)

    # --- return shape ---
    def test_returns_dict(self):
        result = self._call()
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        result = self._call()
        keys = {
            "problem_type",
            "best_score",
            "metric_name",
            "score_label",
            "is_very_low",
            "tips",
            "summary",
            "n_tips",
        }
        assert keys.issubset(result.keys())

    def test_tips_is_list(self):
        result = self._call()
        assert isinstance(result["tips"], list)

    def test_each_tip_has_required_keys(self):
        result = self._call()
        tip_keys = {"icon", "title", "description", "action"}
        for tip in result["tips"]:
            assert tip_keys.issubset(tip.keys())

    def test_n_tips_matches_len_tips(self):
        result = self._call()
        assert result["n_tips"] == len(result["tips"])

    # --- metric_name ---
    def test_regression_metric_name(self):
        result = self._call(problem_type="regression")
        assert result["metric_name"] == "R²"

    def test_classification_metric_name(self):
        result = self._call(problem_type="classification")
        assert result["metric_name"] == "accuracy"

    # --- score_label ---
    def test_score_label_contains_pct(self):
        result = self._call(best_score=0.55)
        assert "55.0%" == result["score_label"]

    def test_score_label_rounds_correctly(self):
        result = self._call(best_score=0.6789)
        assert "67.9%" == result["score_label"]

    # --- is_very_low ---
    def test_very_low_when_score_below_050(self):
        result = self._call(best_score=0.45)
        assert result["is_very_low"] is True

    def test_not_very_low_when_score_above_050(self):
        result = self._call(best_score=0.55)
        assert result["is_very_low"] is False

    def test_not_very_low_at_exactly_050(self):
        result = self._call(best_score=0.50)
        assert result["is_very_low"] is False

    # --- small-dataset tip ---
    def test_small_dataset_tip_for_few_rows(self):
        result = self._call(n_rows=50)
        # First tip should mention adding data
        assert any(
            "50" in t["description"] or "data" in t["title"].lower()
            for t in result["tips"]
        )

    def test_large_dataset_generic_data_tip(self):
        result = self._call(n_rows=1000)
        # Should still include a data tip but without row count callout
        titles = [t["title"] for t in result["tips"]]
        assert any("data" in title.lower() for title in titles)

    # --- date column tip ---
    def test_date_tip_included_when_has_dates(self):
        result = self._call(has_date_columns=True)
        tips_text = " ".join(t["title"] + t["description"] for t in result["tips"])
        assert "chronological" in tips_text.lower() or "date" in tips_text.lower()

    def test_no_date_tip_when_no_dates(self):
        result = self._call(has_date_columns=False)
        tips_text = " ".join(t["description"] for t in result["tips"])
        assert "chronological" not in tips_text.lower()

    # --- target / leakage tips for very low scores ---
    def test_very_low_score_includes_target_review_tip(self):
        result = self._call(best_score=0.30)
        tips_text = " ".join(t["title"] for t in result["tips"])
        assert "target" in tips_text.lower() or "quality" in tips_text.lower()

    # --- summary ---
    def test_summary_mentions_ensemble_when_has_ensemble(self):
        result = self._call(has_ensemble=True)
        assert "ensemble" in result["summary"].lower()

    def test_summary_without_ensemble_mention(self):
        result = self._call(has_ensemble=False)
        # Summary should still be informative
        assert len(result["summary"]) > 20

    # --- tip cap ---
    def test_at_most_5_tips(self):
        result = self._call(has_date_columns=True, best_score=0.30, n_rows=50)
        assert len(result["tips"]) <= 5

    # --- problem_type passthrough ---
    def test_problem_type_in_result(self):
        result = self._call(problem_type="classification")
        assert result["problem_type"] == "classification"

    def test_best_score_in_result(self):
        result = self._call(best_score=0.63)
        assert result["best_score"] == pytest.approx(0.63)


# ---------------------------------------------------------------------------
# Integration tests — verify handler logic conditions (mock-based)
# Mirrors the pattern used in test_ensemble_auto_suggest.py
# ---------------------------------------------------------------------------


def _make_run(
    algorithm: str,
    score: float,
    metric: str = "r2",
    ensemble_type: str | None = None,
    status: str = "done",
) -> mock.MagicMock:
    r = mock.MagicMock()
    r.algorithm = algorithm
    r.status = status
    metrics: dict = {metric: score}
    if ensemble_type:
        metrics["ensemble_type"] = ensemble_type
    r.metrics = json.dumps(metrics)
    return r


def _make_feature_set(problem_type: str = "regression") -> mock.MagicMock:
    fs = mock.MagicMock()
    fs.problem_type = problem_type
    fs.target_column = "revenue"
    fs.approved_features = json.dumps(["a", "b"])
    return fs


def _make_project(
    *,
    last_lag_run_count: int | None = None,
    last_ensemble_suggest_run_count: int | None = None,
) -> mock.MagicMock:
    p = mock.MagicMock()
    p.last_low_accuracy_guidance_run_count = last_lag_run_count
    p.last_ensemble_suggest_run_count = last_ensemble_suggest_run_count
    p.last_milestone_state = "train"
    p.last_insight_dataset_id = None
    p.last_type_check_dataset_id = None
    p.auto_retrain = False
    return p


def _apply_guidance_logic(ctx: dict, project: mock.MagicMock) -> dict | None:
    """Replicate the proactive low-accuracy guidance logic from chat.py."""
    from core.advisor import (
        LOW_ACCURACY_GUIDANCE_THRESHOLD,
        compute_low_accuracy_guidance,
    )

    fset = ctx["feature_set"]
    prob = fset.problem_type if fset else "regression"
    metric_name = "r2" if prob == "regression" else "accuracy"

    done_runs = [r for r in ctx["model_runs"] if r.status == "done"]
    if not done_runs:
        return None

    has_ens = False
    best: float | None = None
    for r in done_runs:
        m = json.loads(r.metrics or "{}")
        if m.get("ensemble_type"):
            has_ens = True
        v = m.get(metric_name)
        if v is not None:
            v = float(v)
            if best is None or v > best:
                best = v

    if best is None:
        return None

    run_count = len(done_runs)
    last_count = project.last_low_accuracy_guidance_run_count
    ens_was_suggested = project.last_ensemble_suggest_run_count is not None

    if not (has_ens or ens_was_suggested):
        return None

    if best >= LOW_ACCURACY_GUIDANCE_THRESHOLD:
        return None

    if run_count == last_count:
        return None  # already shown for this batch

    ds = ctx.get("dataset")
    n_rows = ds.row_count if ds else 0
    return compute_low_accuracy_guidance(
        problem_type=prob,
        best_score=best,
        n_rows=n_rows,
        has_ensemble=has_ens,
        has_date_columns=False,
        n_features=2,
    )


class TestLowAccuracyGuidanceChatHandler:
    """Verify guidance fires / does-not-fire under various conditions."""

    def _build_ctx(self, runs: list, problem_type: str = "regression") -> dict:
        ds = mock.MagicMock()
        ds.row_count = 300
        ds.id = "ds-1"
        ds.column_info = json.dumps([{"name": "a", "dtype": "int64"}])
        return {
            "feature_set": _make_feature_set(problem_type),
            "model_runs": runs,
            "dataset": ds,
            "deployment": None,
            "conversation": [],
        }

    def test_fires_when_ensemble_exists_and_score_below_threshold(self):
        """Guidance fires when ensemble run exists and best score < 0.70."""
        runs = [
            _make_run("linear_regression", 0.55),
            _make_run("voting_regressor", 0.60, ensemble_type="voting"),
        ]
        ctx = self._build_ctx(runs)
        project = _make_project(last_lag_run_count=None)
        result = _apply_guidance_logic(ctx, project)
        assert result is not None
        assert result["best_score"] == pytest.approx(0.60)

    def test_fires_when_ens_was_suggested_but_not_trained(self):
        """Guidance fires when ensemble was auto-suggested (project field set) even without ensemble run."""
        runs = [_make_run("linear_regression", 0.55)]
        ctx = self._build_ctx(runs)
        project = _make_project(
            last_lag_run_count=None,
            last_ensemble_suggest_run_count=1,
        )
        result = _apply_guidance_logic(ctx, project)
        assert result is not None

    def test_no_fire_when_score_at_threshold(self):
        """Guidance does NOT fire when best score is exactly 0.70."""
        runs = [
            _make_run("linear_regression", 0.70),
            _make_run("voting_regressor", 0.70, ensemble_type="voting"),
        ]
        ctx = self._build_ctx(runs)
        project = _make_project(last_lag_run_count=None)
        result = _apply_guidance_logic(ctx, project)
        assert result is None

    def test_no_fire_when_score_above_threshold(self):
        """Guidance does NOT fire when best score > 0.70."""
        runs = [
            _make_run("random_forest_regressor", 0.82),
            _make_run("voting_regressor", 0.85, ensemble_type="voting"),
        ]
        ctx = self._build_ctx(runs)
        project = _make_project(last_lag_run_count=None)
        result = _apply_guidance_logic(ctx, project)
        assert result is None

    def test_no_fire_without_ensemble_and_no_prior_suggest(self):
        """Guidance does NOT fire if no ensemble exists and was never auto-suggested."""
        runs = [_make_run("linear_regression", 0.55)]
        ctx = self._build_ctx(runs)
        project = _make_project(
            last_lag_run_count=None,
            last_ensemble_suggest_run_count=None,
        )
        result = _apply_guidance_logic(ctx, project)
        assert result is None

    def test_no_fire_when_run_count_unchanged(self):
        """Guidance does NOT fire again for same run batch (cooldown)."""
        runs = [
            _make_run("linear_regression", 0.55),
            _make_run("voting_regressor", 0.60, ensemble_type="voting"),
        ]
        ctx = self._build_ctx(runs)
        project = _make_project(
            last_lag_run_count=2,  # same as current run count
        )
        result = _apply_guidance_logic(ctx, project)
        assert result is None

    def test_fires_again_when_new_run_added(self):
        """Guidance fires again when run count changes (new model trained)."""
        runs = [
            _make_run("linear_regression", 0.55),
            _make_run("voting_regressor", 0.58, ensemble_type="voting"),
            _make_run("gradient_boosting_regressor", 0.62),
        ]
        ctx = self._build_ctx(runs)
        project = _make_project(
            last_lag_run_count=2,  # was shown after 2 runs
        )
        result = _apply_guidance_logic(ctx, project)
        assert result is not None

    def test_no_runs_no_fire(self):
        """Guidance does NOT fire when there are no done runs."""
        ctx = self._build_ctx([])
        project = _make_project()
        result = _apply_guidance_logic(ctx, project)
        assert result is None

    def test_classification_fires_correctly(self):
        """Guidance fires for classification problem with accuracy < 0.70."""
        runs = [
            _make_run("logistic_regression", 0.62, metric="accuracy"),
            _make_run(
                "voting_classifier", 0.65, metric="accuracy", ensemble_type="voting"
            ),
        ]
        ctx = self._build_ctx(runs, problem_type="classification")
        project = _make_project(last_lag_run_count=None)
        result = _apply_guidance_logic(ctx, project)
        assert result is not None
        assert result["metric_name"] == "accuracy"
        assert result["best_score"] == pytest.approx(0.65)

    def test_result_has_required_keys(self):
        """Returned guidance dict has all expected keys."""
        runs = [
            _make_run("linear_regression", 0.55),
            _make_run("voting_regressor", 0.60, ensemble_type="voting"),
        ]
        ctx = self._build_ctx(runs)
        project = _make_project(last_lag_run_count=None)
        result = _apply_guidance_logic(ctx, project)
        assert result is not None
        assert "tips" in result
        assert "summary" in result
        assert "n_tips" in result
        assert "is_very_low" in result
