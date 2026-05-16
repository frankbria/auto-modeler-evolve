"""Tests for compute_cross_model_feature_importance in core/advisor.py."""

import pytest

from core.advisor import compute_cross_model_feature_importance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(
    algo_plain: str,
    importances: list[tuple[str, float, int]],
) -> dict:
    """Build a runs_with_importances entry."""
    return {
        "run_id": f"run_{algo_plain}",
        "algorithm": algo_plain.lower().replace(" ", "_"),
        "algorithm_plain": algo_plain,
        "importances": [
            {"feature": f, "importance": imp, "rank": rank}
            for f, imp, rank in importances
        ],
    }


FEATURES_A = [("age", 0.40, 1), ("income", 0.35, 2), ("score", 0.25, 3)]
FEATURES_B = [("income", 0.45, 1), ("age", 0.30, 2), ("score", 0.25, 3)]
FEATURES_C = [("age", 0.50, 1), ("score", 0.30, 2), ("income", 0.20, 3)]


# ---------------------------------------------------------------------------
# Empty / degenerate inputs
# ---------------------------------------------------------------------------


def test_empty_runs():
    result = compute_cross_model_feature_importance([])
    assert result["n_models"] == 0
    assert result["features"] == []
    assert result["consensus_features"] == []
    assert result["top_feature"] is None
    assert "No trained models" in result["summary"]


def test_run_with_no_importances():
    run = {
        "run_id": "r1",
        "algorithm": "rf",
        "algorithm_plain": "Random Forest",
        "importances": [],
    }
    result = compute_cross_model_feature_importance([run])
    assert result["n_models"] == 1
    assert result["features"] == []
    assert "returned no feature importances" in result["summary"]


def test_run_missing_importances_key():
    run = {"run_id": "r1", "algorithm": "rf", "algorithm_plain": "Random Forest"}
    result = compute_cross_model_feature_importance([run])
    assert result["features"] == []


# ---------------------------------------------------------------------------
# Single model
# ---------------------------------------------------------------------------


def test_single_model_basic():
    result = compute_cross_model_feature_importance([_run("Random Forest", FEATURES_A)])
    assert result["n_models"] == 1
    assert len(result["features"]) == 3
    assert result["top_feature"] == "age"
    assert "train more" in result["summary"].lower()


def test_single_model_feature_order():
    result = compute_cross_model_feature_importance([_run("Random Forest", FEATURES_A)])
    feats = [f["feature"] for f in result["features"]]
    assert feats == ["age", "income", "score"]


def test_single_model_mean_importance_equals_importance():
    result = compute_cross_model_feature_importance([_run("Random Forest", FEATURES_A)])
    age_entry = next(f for f in result["features"] if f["feature"] == "age")
    assert abs(age_entry["mean_importance"] - 0.40) < 1e-9


# ---------------------------------------------------------------------------
# Two models
# ---------------------------------------------------------------------------


def test_two_models_mean_importance():
    runs = [_run("RF", FEATURES_A), _run("LR", FEATURES_B)]
    result = compute_cross_model_feature_importance(runs)
    income = next(f for f in result["features"] if f["feature"] == "income")
    # income: 0.35 + 0.45 = 0.80 / 2 = 0.40
    assert abs(income["mean_importance"] - 0.40) < 1e-9


def test_two_models_agreement_count():
    runs = [_run("RF", FEATURES_A), _run("LR", FEATURES_B)]
    result = compute_cross_model_feature_importance(runs)
    # All 3 features rank ≤ 5 in both models
    for entry in result["features"]:
        assert entry["agreement_count"] == 2


def test_two_models_consensus_features():
    runs = [_run("RF", FEATURES_A), _run("LR", FEATURES_B)]
    result = compute_cross_model_feature_importance(runs)
    # All features appear in top-5 of both models
    assert set(result["consensus_features"]) == {"age", "income", "score"}


def test_two_models_n_models_with_data():
    runs = [_run("RF", FEATURES_A), _run("LR", FEATURES_B)]
    result = compute_cross_model_feature_importance(runs)
    for entry in result["features"]:
        assert entry["n_models_with_data"] == 2


# ---------------------------------------------------------------------------
# Three models: consensus, consistency, summary
# ---------------------------------------------------------------------------


def test_three_models_top_feature():
    runs = [_run("RF", FEATURES_A), _run("LR", FEATURES_B), _run("GB", FEATURES_C)]
    result = compute_cross_model_feature_importance(runs)
    # age mean: (0.40+0.30+0.50)/3 = 0.40; income: (0.35+0.45+0.20)/3 = 0.333; score: 0.266
    assert result["top_feature"] == "age"


def test_three_models_consensus_summary_mentions_consensus():
    runs = [_run("RF", FEATURES_A), _run("LR", FEATURES_B), _run("GB", FEATURES_C)]
    result = compute_cross_model_feature_importance(runs)
    assert len(result["consensus_features"]) > 0
    assert "3 models" in result["summary"] or "all" in result["summary"].lower()


def test_consistency_high_when_low_variance():
    # All models give age the same importance — coefficient of variation is 0
    uniform = [("age", 0.40, 1), ("income", 0.35, 2), ("score", 0.25, 3)]
    runs = [_run("RF", uniform), _run("LR", uniform), _run("GB", uniform)]
    result = compute_cross_model_feature_importance(runs)
    age = next(f for f in result["features"] if f["feature"] == "age")
    assert age["consistency"] == "high"


def test_consistency_variable_when_high_variance():
    # income is top in one model, last in another
    runs = [
        _run("RF", [("age", 0.40, 1), ("income", 0.01, 3)]),
        _run("LR", [("age", 0.05, 2), ("income", 0.90, 1)]),
    ]
    result = compute_cross_model_feature_importance(runs)
    income = next(f for f in result["features"] if f["feature"] == "income")
    assert income["consistency"] in {"medium", "variable"}


# ---------------------------------------------------------------------------
# Feature cap
# ---------------------------------------------------------------------------


def test_features_capped_at_15():
    many_features = [(f"feat_{i}", 1.0 / (i + 1), i + 1) for i in range(20)]
    result = compute_cross_model_feature_importance([_run("RF", many_features)])
    assert len(result["features"]) <= 15


def test_features_sorted_descending():
    runs = [_run("RF", FEATURES_A), _run("LR", FEATURES_B)]
    result = compute_cross_model_feature_importance(runs)
    means = [f["mean_importance"] for f in result["features"]]
    assert means == sorted(means, reverse=True)


# ---------------------------------------------------------------------------
# Return shape
# ---------------------------------------------------------------------------


def test_return_keys():
    result = compute_cross_model_feature_importance([_run("RF", FEATURES_A)])
    for key in ("n_models", "features", "consensus_features", "top_feature", "summary"):
        assert key in result


def test_feature_entry_keys():
    result = compute_cross_model_feature_importance([_run("RF", FEATURES_A)])
    entry = result["features"][0]
    for key in (
        "feature",
        "mean_importance",
        "n_models_with_data",
        "agreement_count",
        "consistency",
        "per_model",
    ):
        assert key in entry


def test_per_model_entry_keys():
    result = compute_cross_model_feature_importance([_run("RF", FEATURES_A)])
    pm = result["features"][0]["per_model"][0]
    for key in ("algorithm_plain", "importance", "rank"):
        assert key in pm
