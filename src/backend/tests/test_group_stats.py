"""Tests for group-by analysis: compute_group_stats + /api/data/{id}/group-stats endpoint."""

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.analyzer import compute_group_stats


# ---------------------------------------------------------------------------
# Unit tests — compute_group_stats
# ---------------------------------------------------------------------------


@pytest.fixture
def sales_df():
    return pd.DataFrame(
        {
            "region": ["East", "West", "East", "West", "East", "North"],
            "product": ["A", "A", "B", "B", "C", "A"],
            "revenue": [100.0, 200.0, 150.0, 300.0, 80.0, 250.0],
            "units": [10, 20, 15, 30, 8, 25],
            "cost": [50.0, 80.0, 60.0, 120.0, 40.0, 100.0],
        }
    )


def test_group_stats_sum(sales_df):
    result = compute_group_stats(sales_df, "region", value_cols=["revenue"], agg="sum")
    assert "error" not in result
    assert result["group_col"] == "region"
    assert result["value_col"] == "revenue"
    assert result["agg"] == "sum"
    # East: 100+150+80=330, West: 200+300=500, North: 250
    rows_by_group = {r["group"]: r["revenue"] for r in result["rows"]}
    assert abs(rows_by_group["East"] - 330.0) < 0.01
    assert abs(rows_by_group["West"] - 500.0) < 0.01
    assert abs(rows_by_group["North"] - 250.0) < 0.01
    # Sorted descending (West first)
    assert result["rows"][0]["group"] == "West"


def test_group_stats_mean(sales_df):
    result = compute_group_stats(sales_df, "region", value_cols=["revenue"], agg="mean")
    assert result["agg"] == "mean"
    rows_by_group = {r["group"]: r["revenue"] for r in result["rows"]}
    # East mean: (100+150+80)/3 = 110
    assert abs(rows_by_group["East"] - 110.0) < 0.01


def test_group_stats_count(sales_df):
    result = compute_group_stats(sales_df, "region", agg="count")
    assert result["agg"] == "count"
    rows_by_group = {r["group"]: r["count"] for r in result["rows"]}
    assert rows_by_group["East"] == 3
    assert rows_by_group["West"] == 2
    assert rows_by_group["North"] == 1


def test_group_stats_min(sales_df):
    result = compute_group_stats(sales_df, "region", value_cols=["revenue"], agg="min")
    assert result["agg"] == "min"
    rows_by_group = {r["group"]: r["revenue"] for r in result["rows"]}
    assert abs(rows_by_group["East"] - 80.0) < 0.01


def test_group_stats_max(sales_df):
    result = compute_group_stats(sales_df, "region", value_cols=["revenue"], agg="max")
    assert result["agg"] == "max"
    rows_by_group = {r["group"]: r["revenue"] for r in result["rows"]}
    assert abs(rows_by_group["West"] - 300.0) < 0.01


def test_group_stats_median(sales_df):
    result = compute_group_stats(sales_df, "region", value_cols=["revenue"], agg="median")
    assert result["agg"] == "median"
    # East median: sorted [80, 100, 150] → 100
    rows_by_group = {r["group"]: r["revenue"] for r in result["rows"]}
    assert abs(rows_by_group["East"] - 100.0) < 0.01


def test_group_stats_multiple_value_cols(sales_df):
    result = compute_group_stats(
        sales_df, "region", value_cols=["revenue", "units"], agg="sum"
    )
    assert "error" not in result
    # Both columns should be present in each row
    row_east = next(r for r in result["rows"] if r["group"] == "East")
    assert "revenue" in row_east
    assert "units" in row_east


def test_group_stats_default_value_cols(sales_df):
    """When value_cols=None, all numeric columns except group_col are used."""
    result = compute_group_stats(sales_df, "region")
    assert "error" not in result
    assert result["value_cols"]  # should have at least one column


def test_group_stats_invalid_group_col(sales_df):
    result = compute_group_stats(sales_df, "nonexistent")
    assert "error" in result


def test_group_stats_invalid_value_col(sales_df):
    result = compute_group_stats(sales_df, "region", value_cols=["does_not_exist"])
    # Should fall back gracefully or return error
    # If value_cols is empty after filtering, we fall back to count
    assert "agg" in result or "error" in result


def test_group_stats_no_rows(sales_df):
    """Empty DataFrame."""
    empty_df = sales_df.iloc[0:0]
    result = compute_group_stats(empty_df, "region")
    assert "error" in result


def test_group_stats_summary_present(sales_df):
    result = compute_group_stats(sales_df, "region", value_cols=["revenue"])
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 10
    # Summary should mention the top group
    assert result["rows"][0]["group"] in result["summary"]


def test_group_stats_total_present(sales_df):
    result = compute_group_stats(sales_df, "region", value_cols=["revenue"], agg="sum")
    assert result["total"] is not None
    # Total should equal sum across all regions
    assert abs(result["total"] - sales_df["revenue"].sum()) < 0.01


def test_group_stats_capped_at_max_groups():
    """More than 30 groups should be capped."""
    df = pd.DataFrame(
        {
            "cat": [f"cat_{i}" for i in range(50)],
            "val": list(range(50)),
        }
    )
    result = compute_group_stats(df, "cat", value_cols=["val"], agg="sum")
    assert "error" not in result
    assert len(result["rows"]) <= 30


def test_group_stats_by_product(sales_df):
    result = compute_group_stats(sales_df, "product", value_cols=["revenue"], agg="sum")
    assert "error" not in result
    assert len(result["rows"]) == 3  # A, B, C


# ---------------------------------------------------------------------------
# API endpoint tests — GET /api/data/{id}/group-stats
# ---------------------------------------------------------------------------

_SAMPLE_CSV = (
    b"region,product,revenue,units\n"
    b"East,A,100,10\n"
    b"West,A,200,20\n"
    b"East,B,150,15\n"
    b"West,B,300,30\n"
)


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.feature_set  # noqa
    import models.model_run  # noqa
    import models.project  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
async def dataset_id(ac):
    proj_resp = await ac.post("/api/projects", json={"name": "Group Test"})
    project_id = proj_resp.json()["id"]
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("sales.csv", _SAMPLE_CSV, "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["dataset_id"]


async def test_group_stats_endpoint_sum(ac, dataset_id):
    resp = await ac.get(
        f"/api/data/{dataset_id}/group-stats",
        params={"group_by": "region", "agg": "sum"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["group_col"] == "region"
    assert data["agg"] == "sum"
    assert len(data["rows"]) == 2  # East, West


async def test_group_stats_endpoint_with_metrics(ac, dataset_id):
    resp = await ac.get(
        f"/api/data/{dataset_id}/group-stats",
        params={"group_by": "region", "metrics": "revenue", "agg": "mean"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["agg"] == "mean"
    rows = {r["group"]: r for r in data["rows"]}
    assert "East" in rows
    assert "West" in rows


async def test_group_stats_endpoint_count(ac, dataset_id):
    resp = await ac.get(
        f"/api/data/{dataset_id}/group-stats",
        params={"group_by": "region", "agg": "count"},
    )
    assert resp.status_code == 200
    rows = {r["group"]: r["count"] for r in resp.json()["rows"]}
    assert rows["East"] == 2
    assert rows["West"] == 2


async def test_group_stats_endpoint_invalid_group(ac, dataset_id):
    resp = await ac.get(
        f"/api/data/{dataset_id}/group-stats",
        params={"group_by": "nonexistent"},
    )
    assert resp.status_code == 400


async def test_group_stats_endpoint_dataset_not_found(ac):
    resp = await ac.get(
        "/api/data/00000000-0000-0000-0000-000000000000/group-stats",
        params={"group_by": "region"},
    )
    assert resp.status_code == 404


async def test_group_stats_endpoint_summary_in_response(ac, dataset_id):
    resp = await ac.get(
        f"/api/data/{dataset_id}/group-stats",
        params={"group_by": "product", "agg": "sum"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert isinstance(data["summary"], str)


async def test_group_stats_endpoint_dataset_id_in_response(ac, dataset_id):
    resp = await ac.get(
        f"/api/data/{dataset_id}/group-stats",
        params={"group_by": "region"},
    )
    assert resp.status_code == 200
    assert resp.json()["dataset_id"] == dataset_id


# ---------------------------------------------------------------------------
# Chat intent detection tests
# ---------------------------------------------------------------------------


@pytest.fixture
def chat_df():
    return pd.DataFrame(
        {
            "region": ["East", "West", "East", "North"],
            "product": ["A", "B", "A", "C"],
            "revenue": [100.0, 200.0, 150.0, 300.0],
            "units": [10, 20, 15, 30],
        }
    )


def test_detect_group_request_by_region(chat_df):
    from api.chat import _detect_group_request

    result = _detect_group_request("show me revenue by region", chat_df)
    assert result is not None
    assert result["group_col"] == "region"
    assert "revenue" in (result.get("value_cols") or [])


def test_detect_group_request_breakdown(chat_df):
    from api.chat import _detect_group_request

    result = _detect_group_request("breakdown by product", chat_df)
    assert result is not None
    assert result["group_col"] == "product"


def test_detect_group_request_agg_mean(chat_df):
    from api.chat import _detect_group_request

    result = _detect_group_request("average revenue by region", chat_df)
    assert result is not None
    assert result["agg"] == "mean"


def test_detect_group_request_agg_count(chat_df):
    from api.chat import _detect_group_request

    result = _detect_group_request("count by region", chat_df)
    assert result is not None
    assert result["agg"] == "count"


def test_detect_group_request_no_columns(chat_df):
    from api.chat import _detect_group_request

    # Message mentions nothing that matches a column
    result = _detect_group_request("hello world", chat_df)
    assert result is None


def test_group_patterns_match():
    from api.chat import _GROUP_PATTERNS

    patterns_that_should_match = [
        "show me revenue by region",
        "breakdown by product",
        "group by department",
        "total revenue per category",
        "average sales per region",
        "sum units by product",
        "count by status",
        "break down sales by region",
    ]
    for msg in patterns_that_should_match:
        assert _GROUP_PATTERNS.search(msg), f"Should match: {msg!r}"
