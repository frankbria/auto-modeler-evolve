"""Tests for the data readiness feature.

Covers:
  - core/readiness.py unit tests (all 5 component scorers + overall scoring)
  - GET /api/data/{id}/readiness-check endpoint
  - Chat intent detection (_DATA_READINESS_PATTERNS)
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from core.readiness import (
    _score_class_balance,
    _score_data_type_quality,
    _score_duplicates,
    _score_feature_diversity,
    _score_missing_values,
    _score_row_count,
    _score_to_grade,
    _score_to_status,
    compute_data_readiness,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def make_good_df(n: int = 300) -> pd.DataFrame:
    """A clean, ready-to-train dataset."""
    import numpy as np

    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "revenue": rng.normal(1000, 200, n),
            "quantity": rng.integers(1, 100, n),
            "region": rng.choice(["East", "West", "North", "South"], n),
            "product": rng.choice(["A", "B", "C"], n),
        }
    )


def make_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


# ---------------------------------------------------------------------------
# Row count scorer
# ---------------------------------------------------------------------------


def test_row_count_critical():
    df = pd.DataFrame({"a": range(30)})
    comp = _score_row_count(df)
    assert comp["score"] == 0
    assert comp["status"] == "critical"
    assert comp["max_score"] == 25
    assert "recommendation" in comp


def test_row_count_warning():
    df = pd.DataFrame({"a": range(100)})
    comp = _score_row_count(df)
    assert comp["score"] == 12
    assert comp["status"] == "warning"


def test_row_count_good_medium():
    df = pd.DataFrame({"a": range(500)})
    comp = _score_row_count(df)
    assert comp["score"] == 20
    assert comp["status"] == "good"


def test_row_count_good_large():
    df = pd.DataFrame({"a": range(2000)})
    comp = _score_row_count(df)
    assert comp["score"] == 25
    assert comp["status"] == "good"


# ---------------------------------------------------------------------------
# Missing values scorer
# ---------------------------------------------------------------------------


def test_missing_values_none():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    comp = _score_missing_values(df)
    assert comp["score"] == 25
    assert comp["status"] == "good"


def test_missing_values_critical():
    df = pd.DataFrame({"a": [1, None] * 10, "b": range(20)})
    comp = _score_missing_values(df)
    assert comp["status"] == "critical"
    assert comp["score"] <= 5
    assert "recommendation" in comp


def test_missing_values_warning_moderate():
    # 30% missing in col a
    a = [1.0] * 7 + [None] * 3
    df = pd.DataFrame({"a": a, "b": range(10)})
    comp = _score_missing_values(df)
    assert comp["status"] == "warning"
    assert comp["score"] == 15


def test_missing_values_minor():
    df = pd.DataFrame({"a": [1, 2, None, 4, 5, 6, 7, 8, 9, 10]})
    comp = _score_missing_values(df)
    assert comp["status"] == "warning"
    assert comp["score"] == 20


def test_missing_values_empty_df():
    comp = _score_missing_values(pd.DataFrame())
    assert comp["status"] == "critical"
    assert comp["score"] == 0


# ---------------------------------------------------------------------------
# Duplicate rows scorer
# ---------------------------------------------------------------------------


def test_duplicates_none():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    comp = _score_duplicates(df)
    assert comp["score"] == 20
    assert comp["status"] == "good"


def test_duplicates_critical():
    # 30 identical rows + 100 unique = 130 total
    # df.duplicated() marks 29 of 30 identical rows → 29/130 = 22.3% > 20% → critical
    row = {"a": 1, "b": 2}
    df = pd.DataFrame([row] * 30 + [{"a": i, "b": i} for i in range(100)])
    comp = _score_duplicates(df)
    assert comp["status"] == "critical"
    assert comp["score"] == 5
    assert "recommendation" in comp


def test_duplicates_warning():
    dupes = [{"a": 1, "b": 2}] * 8
    unique = [{"a": i, "b": i} for i in range(100)]
    df = pd.DataFrame(dupes + unique)
    comp = _score_duplicates(df)
    assert comp["status"] == "warning"


def test_duplicates_minor():
    dupes = [{"a": 1, "b": 2}] * 2
    unique = [{"a": i, "b": i} for i in range(100)]
    df = pd.DataFrame(dupes + unique)
    comp = _score_duplicates(df)
    assert comp["status"] == "warning"
    assert comp["score"] == 17


# ---------------------------------------------------------------------------
# Feature diversity scorer
# ---------------------------------------------------------------------------


def test_feature_diversity_good_mix():
    df = pd.DataFrame({"num1": [1, 2, 3], "num2": [4, 5, 6], "cat": ["a", "b", "c"]})
    comp = _score_feature_diversity(df)
    assert comp["score"] == 15
    assert comp["status"] == "good"


def test_feature_diversity_single_column():
    df = pd.DataFrame({"a": [1, 2, 3]})
    comp = _score_feature_diversity(df)
    assert comp["score"] == 0
    assert comp["status"] == "critical"
    assert "recommendation" in comp


def test_feature_diversity_no_numeric():
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})
    comp = _score_feature_diversity(df)
    assert comp["status"] == "critical"
    assert "recommendation" in comp


def test_feature_diversity_numeric_only_few():
    df = pd.DataFrame({"a": [1, 2, 3]})
    comp = _score_feature_diversity(df)
    # only 1 col → critical path
    assert comp["status"] == "critical"


def test_feature_diversity_numeric_only_many():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    comp = _score_feature_diversity(df)
    assert comp["status"] == "good"
    assert comp["score"] == 12


# ---------------------------------------------------------------------------
# Data type quality scorer
# ---------------------------------------------------------------------------


def test_data_type_quality_clean():
    # Use low-cardinality categories so they don't get flagged as ID-like
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": ["x", "y", "x", "y", "x"]})
    comp = _score_data_type_quality(df)
    assert comp["score"] == 15
    assert comp["status"] == "good"


def test_data_type_quality_all_null_column():
    df = pd.DataFrame({"a": [None, None, None], "b": [1, 2, 3]})
    comp = _score_data_type_quality(df)
    assert comp["status"] == "critical"
    assert comp["score"] == 5
    assert "recommendation" in comp


def test_data_type_quality_high_cardinality():
    # 20 unique strings in 20 rows — 100% unique → high cardinality
    df = pd.DataFrame(
        {
            "id_col": [f"uid-{i}" for i in range(20)],
            "another_id": [f"x-{i}" for i in range(20)],
            "value": range(20),
        }
    )
    comp = _score_data_type_quality(df)
    assert comp["status"] == "warning"


def test_data_type_quality_empty():
    comp = _score_data_type_quality(pd.DataFrame())
    assert comp["status"] == "critical"


# ---------------------------------------------------------------------------
# Class balance scorer
# ---------------------------------------------------------------------------


def test_class_balance_good():
    df = pd.DataFrame({"target": ["A"] * 50 + ["B"] * 50})
    comp = _score_class_balance(df, "target")
    assert comp is not None
    assert comp["status"] == "good"
    assert comp["advisory"] is True


def test_class_balance_critical():
    df = pd.DataFrame({"target": ["A"] * 95 + ["B"] * 5})
    comp = _score_class_balance(df, "target")
    assert comp is not None
    assert comp["status"] == "critical"


def test_class_balance_warning():
    df = pd.DataFrame({"target": ["A"] * 85 + ["B"] * 15})
    comp = _score_class_balance(df, "target")
    assert comp is not None
    assert comp["status"] == "warning"


def test_class_balance_skip_numeric_target():
    # High unique values → skip (not a classification target)
    df = pd.DataFrame({"target": list(range(100))})
    comp = _score_class_balance(df, "target")
    assert comp is None


def test_class_balance_missing_col():
    df = pd.DataFrame({"a": [1, 2, 3]})
    comp = _score_class_balance(df, "nonexistent")
    assert comp is None


# ---------------------------------------------------------------------------
# Grade / status helpers
# ---------------------------------------------------------------------------


def test_score_to_grade():
    assert _score_to_grade(95) == "A"
    assert _score_to_grade(80) == "B"
    assert _score_to_grade(65) == "C"
    assert _score_to_grade(50) == "D"
    assert _score_to_grade(30) == "F"


def test_score_to_status():
    assert _score_to_status(90) == "ready"
    assert _score_to_status(60) == "needs_attention"
    assert _score_to_status(30) == "not_ready"


# ---------------------------------------------------------------------------
# compute_data_readiness integration
# ---------------------------------------------------------------------------


def test_compute_readiness_good_dataset():
    df = make_good_df()
    result = compute_data_readiness(df)
    assert result["score"] >= 75
    assert result["grade"] in ("A", "B")
    assert result["status"] == "ready"
    assert len(result["components"]) == 5  # 5 core components
    assert "summary" in result


def test_compute_readiness_tiny_dataset():
    # Only 3 rows → row_count=0; 1 col → feature diversity score is 0 too
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = compute_data_readiness(df)
    # row_count=0, feature_diversity=0, rest may be high → score still below 75
    assert result["score"] < 75
    # Status should not be "ready"
    assert result["status"] != "ready"


def test_compute_readiness_with_target_adds_class_balance():
    df = make_good_df()
    result = compute_data_readiness(df, target_col="region")
    # Should have 6 components (5 core + 1 advisory)
    advisory = [c for c in result["components"] if c.get("advisory")]
    assert len(advisory) == 1
    assert "Class Balance" in advisory[0]["name"]


def test_compute_readiness_class_balance_not_in_score():
    df = make_good_df()
    base = compute_data_readiness(df)
    with_target = compute_data_readiness(df, target_col="region")
    # Score should be the same (advisory doesn't affect weighted total)
    assert base["score"] == with_target["score"]


def test_compute_readiness_recommendations_capped():
    """Recommendations are capped at 5."""
    df = pd.DataFrame({"a": [None] * 10 + [1] * 10})
    result = compute_data_readiness(df)
    assert len(result["recommendations"]) <= 5


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_readiness_endpoint(client, tmp_path):
    """GET /api/data/{id}/readiness-check returns score + components."""
    df = make_good_df()
    response = await client.post(
        "/api/data/upload",
        files={"file": ("test.csv", make_csv_bytes(df), "text/csv")},
        data={"project_id": "proj-readiness"},
    )
    assert response.status_code in (200, 201)
    dataset_id = response.json()["dataset_id"]

    resp = await client.get(f"/api/data/{dataset_id}/readiness-check")
    assert resp.status_code == 200
    data = resp.json()
    assert data["dataset_id"] == dataset_id
    assert "score" in data
    assert "grade" in data
    assert "status" in data
    assert "components" in data
    assert len(data["components"]) >= 5
    assert "recommendations" in data


@pytest.mark.asyncio
async def test_readiness_endpoint_with_target(client, tmp_path):
    """GET /readiness-check?target=region includes class balance advisory."""
    df = make_good_df()
    response = await client.post(
        "/api/data/upload",
        files={"file": ("test.csv", make_csv_bytes(df), "text/csv")},
        data={"project_id": "proj-readiness-target"},
    )
    dataset_id = response.json()["dataset_id"]

    resp = await client.get(f"/api/data/{dataset_id}/readiness-check?target=region")
    assert resp.status_code == 200
    data = resp.json()
    advisory = [c for c in data["components"] if c.get("advisory")]
    assert len(advisory) == 1


@pytest.mark.asyncio
async def test_readiness_endpoint_404(client):
    """Non-existent dataset returns 404."""
    resp = await client.get("/api/data/nonexistent-id/readiness-check")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Chat intent detection
# ---------------------------------------------------------------------------


def test_data_readiness_patterns():
    """_DATA_READINESS_PATTERNS matches relevant queries."""
    from api.chat import _DATA_READINESS_PATTERNS

    matches = [
        "is my data ready?",
        "Is the data ready to train?",
        "can I start training now?",
        "check my data",
        "is this data good enough?",
        "data readiness check",
        "ready to model",
        "is my data clean enough?",
        "assess my data",
        "data suitable for modeling",
    ]
    for msg in matches:
        assert _DATA_READINESS_PATTERNS.search(msg), f"Should match: {msg!r}"


def test_data_readiness_patterns_no_false_positives():
    """_DATA_READINESS_PATTERNS does not match unrelated queries."""
    from api.chat import _DATA_READINESS_PATTERNS

    non_matches = [
        "what is the average revenue?",
        "show me a chart",
        "train a model on revenue",
        "which column should I use as target?",
    ]
    for msg in non_matches:
        assert not _DATA_READINESS_PATTERNS.search(msg), f"Should NOT match: {msg!r}"
