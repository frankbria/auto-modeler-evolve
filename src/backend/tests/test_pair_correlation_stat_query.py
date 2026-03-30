"""Tests for pair correlation and stat query features."""

import io
import pytest
import pandas as pd
import numpy as np
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel

from core.analyzer import compute_pair_correlation, compute_stat_query

# ---------------------------------------------------------------------------
# Test DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def corr_df():
    """DataFrame with known correlation between columns."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    return pd.DataFrame(
        {
            "revenue": x * 100 + 500,
            "cost": x * 80 + 200,  # strong positive correlation with revenue
            "units": -x * 50 + 100,  # strong negative correlation with revenue
            "random": rng.standard_normal(100),  # negligible correlation
        }
    )


@pytest.fixture
def small_df():
    """DataFrame with only 2 rows — insufficient for correlation."""
    return pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})


@pytest.fixture
def null_df():
    """DataFrame with null values in some rows."""
    return pd.DataFrame(
        {
            "x": [1.0, 2.0, None, 4.0, 5.0],
            "y": [2.0, 4.0, 6.0, None, 10.0],
        }
    )


@pytest.fixture
def sales_df():
    """Simple sales DataFrame for stat query tests."""
    return pd.DataFrame(
        {
            "region": ["East", "West", "East", "North", "West"],
            "revenue": [5000.0, 1200.0, 8500.0, 300.0, 4200.0],
            "units": [50, 12, 85, 3, 42],
        }
    )


# ---------------------------------------------------------------------------
# compute_pair_correlation tests
# ---------------------------------------------------------------------------


class TestComputePairCorrelation:
    def test_strong_positive_correlation(self, corr_df):
        result = compute_pair_correlation(corr_df, "revenue", "cost")
        assert result["r"] is not None
        assert result["r"] > 0.9
        assert result["strength"] in ("very strong", "strong")
        assert result["direction"] == "positive"
        assert result["n"] == 100

    def test_strong_negative_correlation(self, corr_df):
        result = compute_pair_correlation(corr_df, "revenue", "units")
        assert result["r"] is not None
        assert result["r"] < -0.7
        assert result["direction"] == "negative"

    def test_negligible_correlation(self, corr_df):
        result = compute_pair_correlation(corr_df, "revenue", "random")
        assert result["r"] is not None
        assert result["strength"] in ("negligible", "weak", "moderate")

    def test_returns_p_value(self, corr_df):
        result = compute_pair_correlation(corr_df, "revenue", "cost")
        assert result["p_value"] is not None
        assert 0.0 <= result["p_value"] <= 1.0

    def test_significant_correlation_flagged(self, corr_df):
        result = compute_pair_correlation(corr_df, "revenue", "cost")
        assert "significant" in result["significant"].lower()

    def test_insufficient_data(self, small_df):
        result = compute_pair_correlation(small_df, "a", "b")
        assert result["r"] is None
        assert result["n"] == 2
        assert (
            "insufficient" in result["strength"].lower()
            or "insufficient" in result["summary"].lower()
        )

    def test_missing_col1_raises(self, corr_df):
        with pytest.raises(ValueError, match="not found"):
            compute_pair_correlation(corr_df, "nonexistent", "revenue")

    def test_missing_col2_raises(self, corr_df):
        with pytest.raises(ValueError, match="not found"):
            compute_pair_correlation(corr_df, "revenue", "nonexistent")

    def test_handles_null_values(self, null_df):
        result = compute_pair_correlation(null_df, "x", "y")
        # Should use only rows where both are non-null (rows 0, 1 → 2 valid? no, 3 valid)
        assert result["n"] >= 2

    def test_result_has_required_fields(self, corr_df):
        result = compute_pair_correlation(corr_df, "revenue", "cost")
        assert "col1" in result
        assert "col2" in result
        assert "r" in result
        assert "p_value" in result
        assert "n" in result
        assert "strength" in result
        assert "direction" in result
        assert "significant" in result
        assert "summary" in result

    def test_col1_col2_recorded(self, corr_df):
        result = compute_pair_correlation(corr_df, "revenue", "cost")
        assert result["col1"] == "revenue"
        assert result["col2"] == "cost"

    def test_strength_labels_valid(self, corr_df):
        result = compute_pair_correlation(corr_df, "revenue", "cost")
        assert result["strength"] in (
            "very strong",
            "strong",
            "moderate",
            "weak",
            "negligible",
        )

    def test_direction_valid(self, corr_df):
        result = compute_pair_correlation(corr_df, "revenue", "cost")
        assert result["direction"] in ("positive", "negative")

    def test_summary_contains_r_value(self, corr_df):
        result = compute_pair_correlation(corr_df, "revenue", "cost")
        assert "r =" in result["summary"]

    def test_r_rounded_to_4_decimals(self, corr_df):
        result = compute_pair_correlation(corr_df, "revenue", "cost")
        if result["r"] is not None:
            r_str = str(result["r"])
            decimal_part = r_str.split(".")[-1] if "." in r_str else ""
            assert len(decimal_part) <= 4


# ---------------------------------------------------------------------------
# compute_stat_query tests
# ---------------------------------------------------------------------------


class TestComputeStatQuery:
    def test_mean_basic(self, sales_df):
        result = compute_stat_query(sales_df, agg="mean", col="revenue")
        expected = sales_df["revenue"].mean()
        assert abs(result["value"] - expected) < 0.01
        assert result["agg"] == "mean"
        assert result["col"] == "revenue"

    def test_sum_basic(self, sales_df):
        result = compute_stat_query(sales_df, agg="sum", col="revenue")
        assert result["value"] == sales_df["revenue"].sum()
        assert result["label"] == "total"

    def test_median_basic(self, sales_df):
        result = compute_stat_query(sales_df, agg="median", col="revenue")
        assert result["value"] == sales_df["revenue"].median()

    def test_max_basic(self, sales_df):
        result = compute_stat_query(sales_df, agg="max", col="revenue")
        assert result["value"] == sales_df["revenue"].max()
        assert result["label"] == "maximum"

    def test_min_basic(self, sales_df):
        result = compute_stat_query(sales_df, agg="min", col="revenue")
        assert result["value"] == sales_df["revenue"].min()
        assert result["label"] == "minimum"

    def test_std_basic(self, sales_df):
        result = compute_stat_query(sales_df, agg="std", col="revenue")
        assert abs(result["value"] - sales_df["revenue"].std()) < 0.01

    def test_count_with_col(self, sales_df):
        result = compute_stat_query(sales_df, agg="count", col="revenue")
        assert result["value"] == 5
        assert result["agg"] == "count"

    def test_count_without_col(self, sales_df):
        result = compute_stat_query(sales_df, agg="count")
        assert result["value"] == len(sales_df)
        assert result["col"] is None

    def test_formatted_value_large_number(self):
        df = pd.DataFrame({"sales": [1_500_000.0, 2_000_000.0]})
        result = compute_stat_query(df, agg="sum", col="sales")
        assert "M" in result["formatted_value"] or "k" in result["formatted_value"]

    def test_formatted_value_thousands(self, sales_df):
        result = compute_stat_query(sales_df, agg="sum", col="revenue")
        # 19200 → "19.20k"
        assert (
            "k" in result["formatted_value"]
            or result["formatted_value"].replace(",", "").replace(".", "").isdigit()
        )

    def test_n_rows_recorded(self, sales_df):
        result = compute_stat_query(sales_df, agg="mean", col="revenue")
        assert result["n_rows"] == len(sales_df)

    def test_n_valid_recorded(self, sales_df):
        result = compute_stat_query(sales_df, agg="mean", col="revenue")
        assert result["n_valid"] == 5

    def test_summary_contains_result(self, sales_df):
        result = compute_stat_query(sales_df, agg="mean", col="revenue")
        assert "average" in result["summary"].lower()
        assert "revenue" in result["summary"]

    def test_invalid_agg_raises(self, sales_df):
        with pytest.raises(ValueError, match="Unknown aggregation"):
            compute_stat_query(sales_df, agg="variance", col="revenue")

    def test_missing_col_raises(self, sales_df):
        with pytest.raises(ValueError, match="not found"):
            compute_stat_query(sales_df, agg="mean", col="nonexistent")

    def test_no_col_non_count_raises(self, sales_df):
        with pytest.raises(ValueError, match="Column name is required"):
            compute_stat_query(sales_df, agg="mean")

    def test_result_has_required_fields(self, sales_df):
        result = compute_stat_query(sales_df, agg="mean", col="revenue")
        assert "agg" in result
        assert "col" in result
        assert "value" in result
        assert "n_rows" in result
        assert "formatted_value" in result
        assert "summary" in result


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    from main import app as fastapi_app

    return fastapi_app


@pytest.fixture
def sample_csv():
    content = "region,revenue,cost,units\n"
    content += "East,5000,3000,50\n"
    content += "West,1200,800,12\n"
    content += "East,8500,5000,85\n"
    content += "North,300,200,3\n"
    content += "West,4200,2500,42\n"
    return content.encode()


@pytest.fixture(autouse=True)
def reset_db():
    from db import engine as _engine

    SQLModel.metadata.drop_all(bind=_engine)
    SQLModel.metadata.create_all(bind=_engine)
    yield


async def _create_dataset(client, sample_csv: bytes, suffix: str = "") -> str:
    """Helper: create a project, upload CSV, return dataset_id."""
    proj_resp = await client.post(
        "/api/projects", json={"name": f"Test Project {suffix}"}
    )
    assert proj_resp.status_code == 201
    project_id = proj_resp.json()["id"]

    upload_resp = await client.post(
        "/api/data/upload",
        files={"file": ("data.csv", io.BytesIO(sample_csv), "text/csv")},
        data={"project_id": project_id},
    )
    assert upload_resp.status_code == 201
    return upload_resp.json()["dataset_id"]


@pytest.mark.asyncio
async def test_pair_correlation_endpoint(app, sample_csv):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        dataset_id = await _create_dataset(client, sample_csv, "1")

        resp2 = await client.get(
            f"/api/data/{dataset_id}/pair-correlation?col1=revenue&col2=cost"
        )
        assert resp2.status_code == 200
        data = resp2.json()
        assert "r" in data
        assert data["r"] is not None
        assert data["col1"] == "revenue"
        assert data["col2"] == "cost"


@pytest.mark.asyncio
async def test_pair_correlation_missing_col(app, sample_csv):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        dataset_id = await _create_dataset(client, sample_csv, "2")

        resp2 = await client.get(
            f"/api/data/{dataset_id}/pair-correlation?col1=revenue&col2=nonexistent"
        )
        assert resp2.status_code == 400


@pytest.mark.asyncio
async def test_pair_correlation_non_numeric_col(app, sample_csv):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        dataset_id = await _create_dataset(client, sample_csv, "3")

        resp2 = await client.get(
            f"/api/data/{dataset_id}/pair-correlation?col1=region&col2=revenue"
        )
        assert resp2.status_code == 400


@pytest.mark.asyncio
async def test_stat_query_mean(app, sample_csv):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        dataset_id = await _create_dataset(client, sample_csv, "4")

        resp2 = await client.get(
            f"/api/data/{dataset_id}/stat-query?agg=mean&col=revenue"
        )
        assert resp2.status_code == 200
        data = resp2.json()
        assert "value" in data
        assert "formatted_value" in data
        assert data["agg"] == "mean"


@pytest.mark.asyncio
async def test_stat_query_count_no_col(app, sample_csv):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        dataset_id = await _create_dataset(client, sample_csv, "5")

        resp2 = await client.get(f"/api/data/{dataset_id}/stat-query?agg=count")
        assert resp2.status_code == 200
        assert resp2.json()["value"] == 5


@pytest.mark.asyncio
async def test_stat_query_invalid_agg(app, sample_csv):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        dataset_id = await _create_dataset(client, sample_csv, "6")

        resp2 = await client.get(
            f"/api/data/{dataset_id}/stat-query?agg=variance&col=revenue"
        )
        assert resp2.status_code == 400


@pytest.mark.asyncio
async def test_stat_query_missing_col(app, sample_csv):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        dataset_id = await _create_dataset(client, sample_csv, "7")

        resp2 = await client.get(
            f"/api/data/{dataset_id}/stat-query?agg=mean&col=nonexistent"
        )
        assert resp2.status_code == 400


# ---------------------------------------------------------------------------
# Chat intent detection tests
# ---------------------------------------------------------------------------


class TestPairCorrPatterns:
    def setup_method(self):
        from api.chat import _PAIR_CORR_PATTERNS, _detect_pair_corr_cols

        self.pattern = _PAIR_CORR_PATTERNS
        self.detect = _detect_pair_corr_cols

    def test_correlation_between_x_and_y(self):
        assert self.pattern.search("correlation between revenue and cost")

    def test_how_correlated_are(self):
        assert self.pattern.search("how correlated are revenue and cost?")

    def test_pearson_r(self):
        assert self.pattern.search("what's the Pearson r for these columns?")

    def test_pearson_correlation(self):
        assert self.pattern.search("show me the Pearson correlation")

    def test_does_correlate_with(self):
        assert self.pattern.search("does revenue correlate with units?")

    def test_correlation_of_with(self):
        assert self.pattern.search("correlation of revenue with cost")

    def test_does_not_match_summarize(self):
        assert not self.pattern.search("summarize my data")

    def test_does_not_match_group_by(self):
        assert not self.pattern.search("show revenue by region")

    def test_detect_two_mentioned_cols(self):
        df = pd.DataFrame({"revenue": [1.0], "cost": [2.0], "region": ["East"]})
        result = self.detect("correlation between revenue and cost", df)
        assert result is not None
        assert set(result) == {"revenue", "cost"}

    def test_detect_fallback_first_two(self):
        df = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
        result = self.detect("pearson correlation", df)
        assert result is not None
        assert len(result) == 2

    def test_detect_returns_none_no_numeric(self):
        df = pd.DataFrame({"region": ["East"], "product": ["A"]})
        result = self.detect("correlation between region and product", df)
        assert result is None


class TestStatQueryPatterns:
    def setup_method(self):
        from api.chat import _STAT_QUERY_PATTERNS, _detect_stat_query

        self.pattern = _STAT_QUERY_PATTERNS
        self.detect = _detect_stat_query

    def test_whats_the_average(self):
        assert self.pattern.search("what's the average of revenue?")

    def test_whats_the_total(self):
        assert self.pattern.search("what's the total revenue?")

    def test_sum_of(self):
        assert self.pattern.search("sum of units")

    def test_max_value(self):
        assert self.pattern.search("maximum value of cost")

    def test_count_rows(self):
        assert self.pattern.search("count the rows")

    def test_how_many_rows(self):
        assert self.pattern.search("how many total rows?")

    def test_detect_mean_col(self):
        df = pd.DataFrame({"revenue": [1.0, 2.0], "region": ["East", "West"]})
        result = self.detect("what's the average of revenue?", df)
        assert result is not None
        assert result["agg"] == "mean"
        assert result["col"] == "revenue"

    def test_detect_sum_col(self):
        df = pd.DataFrame({"revenue": [1.0, 2.0]})
        result = self.detect("sum of revenue", df)
        assert result is not None
        assert result["agg"] == "sum"

    def test_detect_count_no_col(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = self.detect("how many total rows?", df)
        assert result is not None
        assert result["agg"] == "count"

    def test_detect_max(self):
        df = pd.DataFrame({"cost": [1.0, 2.0]})
        result = self.detect("maximum value of cost", df)
        assert result is not None
        assert result["agg"] == "max"

    def test_detect_min(self):
        df = pd.DataFrame({"cost": [1.0, 2.0]})
        result = self.detect("minimum cost", df)
        assert result is not None
        assert result["agg"] == "min"
