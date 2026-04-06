"""Tests for prediction opportunity discovery feature.

Covers:
- _PREDICT_OPP_PATTERNS detection in chat.py
- compute_prediction_opportunities() pure function
- GET /api/data/{dataset_id}/prediction-opportunities endpoint
"""

import pytest

# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def test_pattern_what_can_i_predict():
    from api.chat import _PREDICT_OPP_PATTERNS

    assert _PREDICT_OPP_PATTERNS.search("what can I predict with this data?")


def test_pattern_what_should_i_model():
    from api.chat import _PREDICT_OPP_PATTERNS

    assert _PREDICT_OPP_PATTERNS.search("what should I model?")


def test_pattern_suggest_target():
    from api.chat import _PREDICT_OPP_PATTERNS

    assert _PREDICT_OPP_PATTERNS.search("suggest a prediction target")
    assert _PREDICT_OPP_PATTERNS.search("suggest a target for my model")


def test_pattern_help_choose():
    from api.chat import _PREDICT_OPP_PATTERNS

    assert _PREDICT_OPP_PATTERNS.search("help me choose a prediction target")


def test_pattern_what_columns():
    from api.chat import _PREDICT_OPP_PATTERNS

    assert _PREDICT_OPP_PATTERNS.search("what columns can I predict?")
    assert _PREDICT_OPP_PATTERNS.search("what variable should I model?")


def test_pattern_prediction_opportunities():
    from api.chat import _PREDICT_OPP_PATTERNS

    assert _PREDICT_OPP_PATTERNS.search("what prediction opportunities are there?")
    assert _PREDICT_OPP_PATTERNS.search("show me prediction opportunities")


def test_pattern_no_match_unrelated():
    from api.chat import _PREDICT_OPP_PATTERNS

    assert not _PREDICT_OPP_PATTERNS.search("show me a bar chart")
    assert not _PREDICT_OPP_PATTERNS.search("deploy my model")
    assert not _PREDICT_OPP_PATTERNS.search("what is the model accuracy?")


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


def _make_col(
    name, dtype, null_pct=0.0, unique_count=None, mean=None, std=None, sample_count=100
):
    """Helper to build a col_stats dict for testing."""
    stat: dict = {
        "name": name,
        "dtype": dtype,
        "null_pct": null_pct,
        "unique_count": unique_count if unique_count is not None else sample_count,
        "non_null_count": int(sample_count * (1 - null_pct / 100)),
    }
    if mean is not None:
        stat["mean"] = mean
    if std is not None:
        stat["std"] = std
    return stat


def test_basic_regression_opportunity():
    from core.analyzer import compute_prediction_opportunities

    cols = [
        _make_col("revenue", "float64", mean=1000.0, std=200.0, unique_count=90),
        _make_col("units", "int64", mean=50.0, std=10.0, unique_count=20),
        _make_col("region", "object", unique_count=4),
        _make_col("product", "object", unique_count=5),
    ]
    result = compute_prediction_opportunities(cols, row_count=100)
    assert len(result) > 0
    target_cols = [o["target_col"] for o in result]
    # revenue should be top due to high business value
    assert "revenue" in target_cols


def test_regression_problem_type():
    from core.analyzer import compute_prediction_opportunities

    cols = [
        _make_col("sales", "float64", mean=500.0, std=100.0, unique_count=80),
        _make_col("region", "object", unique_count=3),
    ]
    result = compute_prediction_opportunities(cols, row_count=100)
    sales_opp = next((o for o in result if o["target_col"] == "sales"), None)
    assert sales_opp is not None
    assert sales_opp["problem_type"] == "regression"


def test_classification_opportunity():
    from core.analyzer import compute_prediction_opportunities

    cols = [
        _make_col("churn", "object", unique_count=2),
        _make_col("revenue", "float64", mean=1000.0, std=200.0, unique_count=90),
        _make_col("tenure", "int64", mean=24.0, std=12.0, unique_count=30),
    ]
    result = compute_prediction_opportunities(cols, row_count=100)
    churn_opp = next((o for o in result if o["target_col"] == "churn"), None)
    assert churn_opp is not None
    assert churn_opp["problem_type"] == "classification"


def test_high_business_value_ranked_higher():
    from core.analyzer import compute_prediction_opportunities

    cols = [
        _make_col("revenue", "float64", mean=1000.0, std=200.0, unique_count=90),
        _make_col("generic_metric", "float64", mean=100.0, std=20.0, unique_count=80),
        _make_col("region", "object", unique_count=4),
    ]
    result = compute_prediction_opportunities(cols, row_count=100)
    assert len(result) >= 1
    # revenue should rank before generic_metric due to high business value
    target_cols = [o["target_col"] for o in result]
    assert target_cols.index("revenue") < target_cols.index("generic_metric")


def test_id_column_excluded():
    from core.analyzer import compute_prediction_opportunities

    cols = [
        _make_col("customer_id", "int64", unique_count=100, mean=50.0, std=30.0),
        _make_col("revenue", "float64", mean=1000.0, std=200.0, unique_count=80),
        _make_col("region", "object", unique_count=4),
    ]
    result = compute_prediction_opportunities(cols, row_count=100)
    target_cols = [o["target_col"] for o in result]
    assert "customer_id" not in target_cols


def test_high_missing_data_excluded():
    from core.analyzer import compute_prediction_opportunities

    cols = [
        _make_col(
            "sparse_col",
            "float64",
            null_pct=50.0,
            unique_count=40,
            mean=100.0,
            std=20.0,
        ),
        _make_col(
            "revenue", "float64", null_pct=2.0, unique_count=80, mean=1000.0, std=200.0
        ),
        _make_col("region", "object", unique_count=4),
    ]
    result = compute_prediction_opportunities(cols, row_count=100)
    target_cols = [o["target_col"] for o in result]
    assert "sparse_col" not in target_cols


def test_too_many_categories_excluded():
    from core.analyzer import compute_prediction_opportunities

    cols = [
        _make_col("free_text", "object", unique_count=95),
        _make_col("revenue", "float64", mean=1000.0, std=200.0, unique_count=80),
        _make_col("region", "object", unique_count=4),
    ]
    result = compute_prediction_opportunities(cols, row_count=100)
    target_cols = [o["target_col"] for o in result]
    assert "free_text" not in target_cols


def test_constant_column_excluded():
    from core.analyzer import compute_prediction_opportunities

    cols = [
        _make_col("always_1", "float64", mean=1.0, std=0.0, unique_count=1),
        _make_col("revenue", "float64", mean=1000.0, std=200.0, unique_count=80),
        _make_col("region", "object", unique_count=4),
    ]
    result = compute_prediction_opportunities(cols, row_count=100)
    target_cols = [o["target_col"] for o in result]
    assert "always_1" not in target_cols


def test_max_five_results():
    from core.analyzer import compute_prediction_opportunities

    # Create many eligible columns
    cols = [
        _make_col(
            f"metric_{i}",
            "float64",
            mean=float(100 + i),
            std=float(10 + i),
            unique_count=80,
        )
        for i in range(10)
    ] + [_make_col("region", "object", unique_count=4)]
    result = compute_prediction_opportunities(cols, row_count=100)
    assert len(result) <= 5


def test_empty_dataset_returns_empty():
    from core.analyzer import compute_prediction_opportunities

    result = compute_prediction_opportunities([], row_count=0)
    assert result == []


def test_too_few_rows_returns_empty():
    from core.analyzer import compute_prediction_opportunities

    cols = [_make_col("revenue", "float64", mean=100.0, std=20.0, unique_count=5)]
    result = compute_prediction_opportunities(cols, row_count=5)
    assert result == []


def test_result_has_required_fields():
    from core.analyzer import compute_prediction_opportunities

    cols = [
        _make_col("revenue", "float64", mean=1000.0, std=200.0, unique_count=80),
        _make_col("region", "object", unique_count=4),
    ]
    result = compute_prediction_opportunities(cols, row_count=100)
    assert len(result) > 0
    opp = result[0]
    assert "target_col" in opp
    assert "problem_type" in opp
    assert "feasibility_score" in opp
    assert "reason" in opp
    assert "business_value" in opp
    assert "example_question" in opp
    assert "predictor_count" in opp


def test_feasibility_score_range():
    from core.analyzer import compute_prediction_opportunities

    cols = [
        _make_col("revenue", "float64", mean=1000.0, std=200.0, unique_count=80),
        _make_col("region", "object", unique_count=4),
        _make_col("product", "object", unique_count=5),
        _make_col("units", "int64", mean=50.0, std=10.0, unique_count=20),
    ]
    result = compute_prediction_opportunities(cols, row_count=100)
    for opp in result:
        assert 0 <= opp["feasibility_score"] <= 100


def test_business_value_high_for_revenue():
    from core.analyzer import compute_prediction_opportunities

    cols = [
        _make_col(
            "revenue", "float64", null_pct=0.0, mean=1000.0, std=200.0, unique_count=80
        ),
        _make_col("region", "object", unique_count=4),
    ]
    result = compute_prediction_opportunities(cols, row_count=100)
    rev_opp = next(o for o in result if o["target_col"] == "revenue")
    assert rev_opp["business_value"] == "high"


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_endpoint_returns_opportunities(tmp_path, client):
    """Upload a CSV and verify the endpoint returns ranked opportunities."""
    # Upload a file
    csv_content = b"""product,region,revenue,units,churn
Widget A,North,1200.50,10,0
Widget B,South,850.00,8,1
Widget A,East,2100.75,18,0
Widget C,West,450.25,4,1
Widget B,North,1650.00,15,0
Widget A,South,1100.00,9,0
Widget B,East,2000.00,17,0
Widget C,North,600.00,5,1
Widget A,West,950.00,8,0
Widget B,South,1300.00,12,0
"""
    resp = await client.post(
        "/api/data/upload",
        files={"file": ("data.csv", csv_content, "text/csv")},
        data={"project_id": "proj-1"},
    )
    assert resp.status_code == 201
    dataset_id = resp.json()["dataset_id"]

    resp2 = await client.get(f"/api/data/{dataset_id}/prediction-opportunities")
    assert resp2.status_code == 200
    data = resp2.json()
    assert "opportunities" in data
    assert "total" in data
    assert data["total"] >= 1


@pytest.mark.anyio
async def test_endpoint_revenue_in_results(tmp_path, client):
    """revenue column should appear as a suggestion."""
    csv_content = b"""product,region,revenue,units
Widget A,North,1200.50,10
Widget B,South,850.00,8
Widget A,East,2100.75,18
Widget C,West,450.25,4
Widget B,North,1650.00,15
Widget A,South,1100.00,9
Widget B,East,2000.00,17
Widget C,North,600.00,5
Widget A,West,950.00,8
Widget B,South,1300.00,12
"""
    resp = await client.post(
        "/api/data/upload",
        files={"file": ("sales.csv", csv_content, "text/csv")},
        data={"project_id": "proj-2"},
    )
    dataset_id = resp.json()["dataset_id"]

    resp2 = await client.get(f"/api/data/{dataset_id}/prediction-opportunities")
    assert resp2.status_code == 200
    target_cols = [o["target_col"] for o in resp2.json()["opportunities"]]
    assert "revenue" in target_cols


@pytest.mark.anyio
async def test_endpoint_404_unknown_dataset(client):
    resp = await client.get("/api/data/nonexistent-id/prediction-opportunities")
    assert resp.status_code == 404
