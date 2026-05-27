"""Tests for compute_feature_engineering_impact pure function and related endpoint."""

import pytest

from api.chat import _FE_IMPACT_PATTERNS
from core.trainer import compute_feature_engineering_impact


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_importance(feature: str, importance: float, rank: int) -> dict:
    return {"feature": feature, "importance": importance, "rank": rank}


IMPORTANCES_MIXED = [
    _make_importance("age", 0.40, 1),
    _make_importance("income_log", 0.30, 2),
    _make_importance("score", 0.20, 3),
    _make_importance("income_bin", 0.10, 4),
]

TRANSFORMS_MIXED = [
    {"column": "income", "transform_type": "log_transform"},
    {"column": "income", "transform_type": "bin_quartile"},
]

COL_MAP_MIXED = {
    "income": ["income_log", "income_bin"],
}


# ---------------------------------------------------------------------------
# Pure function: basic grouping
# ---------------------------------------------------------------------------


def test_groups_original_and_engineered():
    result = compute_feature_engineering_impact(
        ["age", "income_log", "score", "income_bin"],
        IMPORTANCES_MIXED,
        TRANSFORMS_MIXED,
        COL_MAP_MIXED,
    )
    orig_group = next(g for g in result["groups"] if g["name"] == "Original")
    eng_group = next(g for g in result["groups"] if g["name"] == "Engineered")

    assert {f["name"] for f in orig_group["features"]} == {"age", "score"}
    assert {f["name"] for f in eng_group["features"]} == {"income_log", "income_bin"}


def test_total_importances_sum_correctly():
    result = compute_feature_engineering_impact(
        ["age", "income_log", "score", "income_bin"],
        IMPORTANCES_MIXED,
        TRANSFORMS_MIXED,
        COL_MAP_MIXED,
    )
    orig_group = next(g for g in result["groups"] if g["name"] == "Original")
    eng_group = next(g for g in result["groups"] if g["name"] == "Engineered")

    assert abs(orig_group["total_importance"] - 0.60) < 1e-5
    assert abs(eng_group["total_importance"] - 0.40) < 1e-5


def test_engineered_columns_list():
    result = compute_feature_engineering_impact(
        ["age", "income_log", "score", "income_bin"],
        IMPORTANCES_MIXED,
        TRANSFORMS_MIXED,
        COL_MAP_MIXED,
    )
    assert set(result["engineered_columns"]) == {"income_log", "income_bin"}
    assert set(result["original_columns"]) == {"age", "score"}


def test_source_column_populated():
    result = compute_feature_engineering_impact(
        ["age", "income_log", "score", "income_bin"],
        IMPORTANCES_MIXED,
        TRANSFORMS_MIXED,
        COL_MAP_MIXED,
    )
    eng_group = next(g for g in result["groups"] if g["name"] == "Engineered")
    for feat in eng_group["features"]:
        assert feat["source"] == "income"


def test_original_source_is_none():
    result = compute_feature_engineering_impact(
        ["age", "income_log", "score", "income_bin"],
        IMPORTANCES_MIXED,
        TRANSFORMS_MIXED,
        COL_MAP_MIXED,
    )
    orig_group = next(g for g in result["groups"] if g["name"] == "Original")
    for feat in orig_group["features"]:
        assert feat["source"] is None


# ---------------------------------------------------------------------------
# Pure function: no engineering case
# ---------------------------------------------------------------------------


def test_no_engineering_all_original():
    result = compute_feature_engineering_impact(
        ["age", "score"],
        [_make_importance("age", 0.6, 1), _make_importance("score", 0.4, 2)],
        [],
        {},
    )
    assert result["has_engineering"] is False
    eng_group = next(g for g in result["groups"] if g["name"] == "Engineered")
    assert eng_group["features"] == []
    assert eng_group["total_importance"] == 0.0


def test_no_engineering_verdict():
    result = compute_feature_engineering_impact(
        ["age", "score"],
        [_make_importance("age", 0.6, 1), _make_importance("score", 0.4, 2)],
        [],
        {},
    )
    assert "No feature engineering" in result["verdict"]


# ---------------------------------------------------------------------------
# Pure function: verdict wording
# ---------------------------------------------------------------------------


def test_verdict_worthwhile_when_engineered_dominates():
    result = compute_feature_engineering_impact(
        ["age", "income_log"],
        [_make_importance("age", 0.20, 2), _make_importance("income_log", 0.80, 1)],
        [{"column": "income", "transform_type": "log_transform"}],
        {"income": ["income_log"]},
    )
    assert "worthwhile" in result["verdict"]


def test_verdict_minimal_when_engineered_low():
    result = compute_feature_engineering_impact(
        ["age", "income_log"],
        [_make_importance("age", 0.97, 1), _make_importance("income_log", 0.03, 2)],
        [{"column": "income", "transform_type": "log_transform"}],
        {"income": ["income_log"]},
    )
    assert "minimal" in result["verdict"]


def test_verdict_added_value_middle():
    result = compute_feature_engineering_impact(
        ["age", "income_log"],
        [_make_importance("age", 0.70, 1), _make_importance("income_log", 0.30, 2)],
        [{"column": "income", "transform_type": "log_transform"}],
        {"income": ["income_log"]},
    )
    assert "added value" in result["verdict"]


# ---------------------------------------------------------------------------
# Pure function: missing importances fallback
# ---------------------------------------------------------------------------


def test_feature_missing_from_importances_gets_zero():
    result = compute_feature_engineering_impact(
        ["age", "income_log", "orphan_col"],
        [_make_importance("age", 0.6, 1), _make_importance("income_log", 0.4, 2)],
        [{"column": "income", "transform_type": "log_transform"}],
        {"income": ["income_log"]},
    )
    orig_group = next(g for g in result["groups"] if g["name"] == "Original")
    orphan = next(f for f in orig_group["features"] if f["name"] == "orphan_col")
    assert orphan["importance"] == 0.0


# ---------------------------------------------------------------------------
# Pure function: features sorted by importance descending
# ---------------------------------------------------------------------------


def test_features_sorted_descending():
    result = compute_feature_engineering_impact(
        ["age", "income_log", "score"],
        [
            _make_importance("age", 0.10, 3),
            _make_importance("income_log", 0.60, 1),
            _make_importance("score", 0.30, 2),
        ],
        [{"column": "income", "transform_type": "log_transform"}],
        {"income": ["income_log"]},
    )
    orig_group = next(g for g in result["groups"] if g["name"] == "Original")
    imps = [f["importance"] for f in orig_group["features"]]
    assert imps == sorted(imps, reverse=True)


# ---------------------------------------------------------------------------
# Pure function: return keys
# ---------------------------------------------------------------------------


def test_result_has_required_keys():
    result = compute_feature_engineering_impact(
        ["age", "income_log"],
        [_make_importance("age", 0.6, 1), _make_importance("income_log", 0.4, 2)],
        [{"column": "income", "transform_type": "log_transform"}],
        {"income": ["income_log"]},
    )
    for key in (
        "groups",
        "engineered_columns",
        "original_columns",
        "verdict",
        "has_engineering",
    ):
        assert key in result, f"Missing key: {key}"


def test_groups_has_two_entries():
    result = compute_feature_engineering_impact(
        ["age", "income_log"],
        [_make_importance("age", 0.6, 1), _make_importance("income_log", 0.4, 2)],
        [{"column": "income", "transform_type": "log_transform"}],
        {"income": ["income_log"]},
    )
    assert len(result["groups"]) == 2
    names = {g["name"] for g in result["groups"]}
    assert names == {"Original", "Engineered"}


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "which features helped the model",
        "what features improved accuracy",
        "feature engineering impact",
        "did the feature engineering help",
        "did the transformations improve my model",
        "show me what the engineering added",
        "impact of feature engineering",
        "were the transformations worthwhile",
        "how much did feature engineering help",
        "how much did engineering improve things",
    ],
)
def test_fe_impact_pattern_matches(phrase):
    assert _FE_IMPACT_PATTERNS.search(phrase), f"Pattern did not match: {phrase!r}"


@pytest.mark.parametrize(
    "phrase",
    [
        "train a model",
        "show me predictions",
        "what columns are there",
        "run a regression",
    ],
)
def test_fe_impact_pattern_no_false_positives(phrase):
    assert not _FE_IMPACT_PATTERNS.search(phrase), (
        f"Pattern falsely matched: {phrase!r}"
    )
