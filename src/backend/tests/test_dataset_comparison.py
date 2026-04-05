"""Tests for Dataset Distribution Comparison feature.

Covers:
- compute_dataset_comparison() pure function — numeric drifts, categorical changes,
  drift score, schema changes, summary
- GET /api/data/compare endpoint
- _DATASET_COMPARE_PATTERNS chat intent detection
"""

import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# compute_dataset_comparison — pure function tests
# ---------------------------------------------------------------------------


def test_identical_datasets_zero_drift():
    from core.analyzer import compute_dataset_comparison

    df = _make_df(revenue=[100, 200, 300], region=["East", "West", "North"])
    result = compute_dataset_comparison(df, df.copy())

    assert result["drift_score"] == 0
    assert result["numeric_drifts"] == []
    assert result["categorical_drifts"] == []
    assert result["new_columns"] == []
    assert result["dropped_columns"] == []
    assert (
        "similar" in result["summary"].lower() or "match" in result["summary"].lower()
    )


def test_large_numeric_mean_shift_detected():
    from core.analyzer import compute_dataset_comparison

    old_df = _make_df(revenue=[100.0, 110.0, 120.0])
    new_df = _make_df(revenue=[200.0, 220.0, 240.0])  # ~100% increase
    result = compute_dataset_comparison(old_df, new_df)

    assert len(result["numeric_drifts"]) == 1
    drift = result["numeric_drifts"][0]
    assert drift["col"] == "revenue"
    assert drift["pct_change"] > 80
    assert drift["severity"] == "high"


def test_small_numeric_shift_not_reported():
    from core.analyzer import compute_dataset_comparison

    old_df = _make_df(revenue=[100.0, 110.0, 120.0])
    new_df = _make_df(revenue=[103.0, 113.0, 123.0])  # ~3% change
    result = compute_dataset_comparison(old_df, new_df)

    assert result["numeric_drifts"] == []


def test_new_categorical_category_detected():
    from core.analyzer import compute_dataset_comparison

    old_df = _make_df(region=["East", "West", "East", "West"])
    new_df = _make_df(region=["East", "West", "North", "South"])  # 2 new
    result = compute_dataset_comparison(old_df, new_df)

    assert len(result["categorical_drifts"]) == 1
    drift = result["categorical_drifts"][0]
    assert drift["col"] == "region"
    assert "North" in drift["new_categories"] or "South" in drift["new_categories"]


def test_dropped_categorical_category_detected():
    from core.analyzer import compute_dataset_comparison

    old_df = _make_df(region=["East", "West", "North"])
    new_df = _make_df(region=["East", "West", "East"])  # North dropped
    result = compute_dataset_comparison(old_df, new_df)

    cat = next((d for d in result["categorical_drifts"] if d["col"] == "region"), None)
    assert cat is not None
    assert "North" in cat["dropped_categories"]


def test_new_column_detected():
    from core.analyzer import compute_dataset_comparison

    old_df = _make_df(revenue=[100, 200])
    new_df = _make_df(revenue=[100, 200], quantity=[10, 20])
    result = compute_dataset_comparison(old_df, new_df)

    assert "quantity" in result["new_columns"]
    assert result["dropped_columns"] == []


def test_dropped_column_detected():
    from core.analyzer import compute_dataset_comparison

    old_df = _make_df(revenue=[100, 200], quantity=[10, 20])
    new_df = _make_df(revenue=[100, 200])
    result = compute_dataset_comparison(old_df, new_df)

    assert "quantity" in result["dropped_columns"]
    assert result["new_columns"] == []


def test_row_count_change_reported():
    from core.analyzer import compute_dataset_comparison

    old_df = _make_df(revenue=[100.0, 200.0, 300.0, 400.0])
    new_df = _make_df(revenue=[100.0, 200.0])  # 50% fewer rows
    result = compute_dataset_comparison(old_df, new_df)

    assert result["row_count_old"] == 4
    assert result["row_count_new"] == 2
    assert result["row_count_change_pct"] == -50.0


def test_drift_score_zero_for_identical():
    from core.analyzer import compute_dataset_comparison

    df = _make_df(x=[1.0, 2.0, 3.0], y=["a", "b", "a"])
    result = compute_dataset_comparison(df, df.copy())
    assert result["drift_score"] == 0


def test_drift_score_high_for_large_changes():
    from core.analyzer import compute_dataset_comparison

    old_df = _make_df(revenue=[100.0, 200.0, 300.0], region=["East", "West", "North"])
    # Dramatic shift: 5x revenue and completely new region categories
    new_df = _make_df(
        revenue=[500.0, 1000.0, 1500.0],
        region=["Alpha", "Beta", "Gamma"],
    )
    result = compute_dataset_comparison(old_df, new_df)
    assert result["drift_score"] >= 30  # substantial drift


def test_summary_mentions_significant_for_high_drift():
    from core.analyzer import compute_dataset_comparison

    old_df = _make_df(revenue=[100.0, 200.0, 300.0])
    new_df = _make_df(revenue=[1000.0, 2000.0, 3000.0])
    result = compute_dataset_comparison(old_df, new_df)

    assert (
        "significant" in result["summary"].lower()
        or "moderate" in result["summary"].lower()
    )


def test_col_count_fields_present():
    from core.analyzer import compute_dataset_comparison

    old_df = _make_df(a=[1, 2], b=[3, 4])
    new_df = _make_df(a=[1, 2], b=[3, 4], c=[5, 6])
    result = compute_dataset_comparison(old_df, new_df)

    assert result["col_count_old"] == 2
    assert result["col_count_new"] == 3


def test_empty_old_df_zero_change_pct():
    from core.analyzer import compute_dataset_comparison

    old_df = pd.DataFrame({"revenue": pd.Series([], dtype=float)})
    new_df = _make_df(revenue=[100.0, 200.0])
    result = compute_dataset_comparison(old_df, new_df)

    assert result["row_count_change_pct"] == 0.0  # division by zero guard


# ---------------------------------------------------------------------------
# API integration — GET /api/data/compare
# ---------------------------------------------------------------------------

_OLD_CSV = b"revenue,region\n100,East\n200,West\n300,North\n"
_NEW_CSV = b"revenue,region\n500,East\n1000,West\n1500,North\n"
_SHIFT_CSV = b"revenue\n500\n550\n600\n650\n700\n"
_BASELINE_CSV = b"revenue\n100\n110\n120\n130\n140\n"


@pytest.mark.anyio
async def test_compare_datasets_returns_report(client):
    r_old = await client.post(
        "/api/data/upload",
        files={"file": ("old.csv", _OLD_CSV, "text/csv")},
        data={"project_id": "cmp-proj"},
    )
    assert r_old.status_code == 201
    baseline_id = r_old.json()["dataset_id"]

    r_new = await client.post(
        "/api/data/upload",
        files={"file": ("new.csv", _NEW_CSV, "text/csv")},
        data={"project_id": "cmp-proj"},
    )
    assert r_new.status_code == 201
    new_id = r_new.json()["dataset_id"]

    resp = await client.get(
        f"/api/data/compare?baseline_id={baseline_id}&new_id={new_id}"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["baseline_id"] == baseline_id
    assert data["new_id"] == new_id
    assert "drift_score" in data
    assert "summary" in data
    assert "numeric_drifts" in data
    assert "categorical_drifts" in data


@pytest.mark.anyio
async def test_compare_404_unknown_baseline(client):
    resp = await client.get(
        "/api/data/compare?baseline_id=unknown-abc&new_id=unknown-xyz"
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_compare_detects_revenue_shift(client):
    r1 = await client.post(
        "/api/data/upload",
        files={"file": ("baseline.csv", _BASELINE_CSV, "text/csv")},
        data={"project_id": "shift-proj"},
    )
    assert r1.status_code == 201

    r2 = await client.post(
        "/api/data/upload",
        files={"file": ("shifted.csv", _SHIFT_CSV, "text/csv")},
        data={"project_id": "shift-proj"},
    )
    assert r2.status_code == 201

    resp = await client.get(
        f"/api/data/compare?baseline_id={r1.json()['dataset_id']}&new_id={r2.json()['dataset_id']}"
    )
    assert resp.status_code == 200
    data = resp.json()
    numeric = data["numeric_drifts"]
    revenue_drift = next((d for d in numeric if d["col"] == "revenue"), None)
    assert revenue_drift is not None
    assert revenue_drift["severity"] in ("medium", "high")


# ---------------------------------------------------------------------------
# Chat intent detection — _DATASET_COMPARE_PATTERNS
# ---------------------------------------------------------------------------


def test_pattern_matches_what_changed():
    from api.chat import _DATASET_COMPARE_PATTERNS

    assert _DATASET_COMPARE_PATTERNS.search("What changed in my new data?")


def test_pattern_matches_how_does_compare():
    from api.chat import _DATASET_COMPARE_PATTERNS

    assert _DATASET_COMPARE_PATTERNS.search(
        "How does my new dataset compare to the original?"
    )


def test_pattern_matches_distribution_shift():
    from api.chat import _DATASET_COMPARE_PATTERNS

    assert _DATASET_COMPARE_PATTERNS.search("Are there any distribution shifts?")


def test_pattern_matches_has_data_changed():
    from api.chat import _DATASET_COMPARE_PATTERNS

    assert _DATASET_COMPARE_PATTERNS.search("Has my data changed?")


def test_pattern_matches_new_vs_old():
    from api.chat import _DATASET_COMPARE_PATTERNS

    assert _DATASET_COMPARE_PATTERNS.search("Show me new vs old data")


def test_pattern_matches_is_new_data_different():
    from api.chat import _DATASET_COMPARE_PATTERNS

    assert _DATASET_COMPARE_PATTERNS.search(
        "Is my new data different from what you trained on?"
    )


def test_pattern_does_not_match_unrelated():
    from api.chat import _DATASET_COMPARE_PATTERNS

    assert not _DATASET_COMPARE_PATTERNS.search("Train a model on my data")
    assert not _DATASET_COMPARE_PATTERNS.search("What are the top features?")
