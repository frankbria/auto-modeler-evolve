"""Tests for the enhanced data profiling and analysis features."""

import json
import pytest
import pandas as pd
import numpy as np

from core.analyzer import (
    analyze_dataframe,
    compute_full_profile,
    _detect_outliers,
    _numeric_distribution,
    _categorical_distribution,
    _looks_like_date,
)
from core.chart_builder import (
    build_bar_chart,
    build_line_chart,
    build_histogram,
    build_pie_chart,
    chart_from_query_result,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sales_df():
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "product": ["Widget A", "Widget B", "Widget A", "Widget C", "Widget B"],
        "region": ["North", "South", "East", "West", "North"],
        "revenue": [1200.50, 850.00, 2100.75, 450.25, 1650.00],
        "units": [10, 8, 18, 4, 15],
    })


@pytest.fixture
def df_with_nulls():
    return pd.DataFrame({
        "a": [1.0, None, 3.0, None, 5.0],
        "b": ["x", "y", None, "w", "v"],
    })


@pytest.fixture
def df_with_outliers():
    # IQR: Q1=2, Q3=4, IQR=2, upper fence=7; value 100 is an outlier
    return pd.DataFrame({"values": [1, 2, 3, 4, 5, 100]})


# ---------------------------------------------------------------------------
# analyze_dataframe
# ---------------------------------------------------------------------------

class TestAnalyzeDataframe:
    def test_row_and_column_counts(self, sales_df):
        result = analyze_dataframe(sales_df)
        assert result["row_count"] == 5
        assert result["column_count"] == 5

    def test_numeric_column_has_stats(self, sales_df):
        result = analyze_dataframe(sales_df)
        revenue = next(c for c in result["columns"] if c["name"] == "revenue")
        assert revenue["min"] == pytest.approx(450.25)
        assert revenue["max"] == pytest.approx(2100.75)
        assert revenue["mean"] is not None
        assert revenue["std"] is not None

    def test_null_pct_calculated(self, df_with_nulls):
        result = analyze_dataframe(df_with_nulls)
        col_a = next(c for c in result["columns"] if c["name"] == "a")
        assert col_a["null_count"] == 2
        assert col_a["null_pct"] == pytest.approx(40.0)

    def test_sample_values_are_json_serializable(self, sales_df):
        result = analyze_dataframe(sales_df)
        for col in result["columns"]:
            for v in col["sample_values"]:
                assert not isinstance(v, np.generic), f"numpy type in sample_values: {type(v)}"


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

class TestOutlierDetection:
    def test_detects_obvious_outlier(self, df_with_outliers):
        result = _detect_outliers(df_with_outliers["values"])
        assert result["count"] == 1
        assert result["upper_fence"] is not None

    def test_no_outliers_in_normal_data(self, sales_df):
        result = _detect_outliers(sales_df["units"])
        assert isinstance(result["count"], int)
        assert result["lower_fence"] is not None

    def test_empty_series_returns_zero(self):
        result = _detect_outliers(pd.Series([], dtype=float))
        assert result["count"] == 0
        assert result["lower_fence"] is None


# ---------------------------------------------------------------------------
# Distribution computation
# ---------------------------------------------------------------------------

class TestDistributions:
    def test_numeric_distribution_has_bins_and_counts(self, sales_df):
        dist = _numeric_distribution(sales_df["revenue"])
        assert len(dist["bins"]) > 0
        assert len(dist["counts"]) == len(dist["bins"])
        assert all(isinstance(c, int) for c in dist["counts"])

    def test_categorical_distribution_returns_labels(self, sales_df):
        dist = _categorical_distribution(sales_df["region"])
        assert "North" in dist["labels"]
        assert len(dist["labels"]) == len(dist["counts"])


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------

class TestPatternDetection:
    def test_detects_high_missing(self):
        df = pd.DataFrame({"a": [1, None, None, None, None]})
        profile = compute_full_profile(df)
        insight_types = [i["type"] for i in profile["insights"]]
        assert "missing_values" in insight_types

    def test_detects_date_column(self, sales_df):
        profile = compute_full_profile(sales_df)
        date_insights = [i for i in profile["insights"] if i["type"] == "date_column"]
        assert len(date_insights) >= 1
        assert date_insights[0]["title"].startswith("'date'")

    def test_detects_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
        profile = compute_full_profile(df)
        dup_insights = [i for i in profile["insights"] if i["type"] == "duplicates"]
        assert len(dup_insights) == 1

    def test_no_false_positives_on_clean_data(self, sales_df):
        profile = compute_full_profile(sales_df)
        # Should not detect missing values (there are none)
        missing_warnings = [
            i for i in profile["insights"]
            if i["type"] == "missing_values" and i["severity"] == "warning"
        ]
        assert len(missing_warnings) == 0


# ---------------------------------------------------------------------------
# Full profile
# ---------------------------------------------------------------------------

class TestComputeFullProfile:
    def test_contains_all_keys(self, sales_df):
        profile = compute_full_profile(sales_df)
        assert "row_count" in profile
        assert "column_count" in profile
        assert "columns" in profile
        assert "correlations" in profile
        assert "insights" in profile

    def test_correlations_computed_for_numeric_columns(self, sales_df):
        profile = compute_full_profile(sales_df)
        assert len(profile["correlations"]["pairs"]) > 0
        pair = profile["correlations"]["pairs"][0]
        assert "col_a" in pair and "col_b" in pair and "correlation" in pair

    def test_columns_have_distribution(self, sales_df):
        profile = compute_full_profile(sales_df)
        for col in profile["columns"]:
            assert "distribution" in col, f"No distribution for column {col['name']}"

    def test_json_serializable(self, sales_df):
        profile = compute_full_profile(sales_df)
        # Must not raise
        json.dumps(profile)

    def test_looks_like_date(self):
        assert _looks_like_date("2024-01-01") is True
        assert _looks_like_date("01/15/2023") is True
        assert _looks_like_date("Widget A") is False
        assert _looks_like_date("1200.50") is False


# ---------------------------------------------------------------------------
# Chart builder
# ---------------------------------------------------------------------------

class TestChartBuilder:
    def test_bar_chart_structure(self):
        chart = build_bar_chart({"North": 1200, "South": 850, "East": 2100}, "Revenue by Region")
        assert chart["chart_type"] == "bar"
        assert chart["x_key"] == "label"
        assert "value" in chart["y_keys"]
        assert len(chart["data"]) == 3
        # Should be sorted descending
        assert chart["data"][0]["value"] == 2100

    def test_bar_chart_from_series(self, sales_df):
        series = sales_df.groupby("region")["revenue"].sum()
        chart = build_bar_chart(series, "Revenue by Region")
        assert chart["chart_type"] == "bar"
        assert len(chart["data"]) == len(series)

    def test_histogram_structure(self):
        chart = build_histogram(
            bins=[0.0, 10.0, 20.0, 30.0],
            counts=[5, 15, 8, 2],
            title="Revenue Distribution",
        )
        assert chart["chart_type"] == "histogram"
        assert chart["x_key"] == "bin"
        assert len(chart["data"]) == 4

    def test_line_chart_structure(self):
        chart = build_line_chart(
            x_values=["Jan", "Feb", "Mar"],
            y_series={"Revenue": [1000, 1200, 1100]},
            title="Monthly Revenue",
        )
        assert chart["chart_type"] == "line"
        assert len(chart["data"]) == 3
        assert chart["data"][0]["x"] == "Jan"
        assert chart["data"][0]["Revenue"] == 1000

    def test_pie_chart_rolls_up_tail(self):
        data = {f"item_{i}": i * 10 for i in range(15)}
        chart = build_pie_chart(data, "Items", limit=5)
        labels = [d["name"] for d in chart["data"]]
        assert "Other" in labels
        assert len(chart["data"]) <= 6  # 5 top + Other

    def test_chart_from_query_result_two_col(self, sales_df):
        result_df = sales_df.groupby("region")["revenue"].sum().reset_index()
        result_df.columns = ["region", "revenue"]
        chart = chart_from_query_result(result_df, "revenue by region")
        assert chart is not None
        assert chart["chart_type"] == "bar"

    def test_chart_from_query_result_empty(self):
        empty_df = pd.DataFrame({"region": [], "revenue": []})
        chart = chart_from_query_result(empty_df, "revenue by region")
        assert chart is None


# ---------------------------------------------------------------------------
# API endpoint tests (upload with full profile)
# ---------------------------------------------------------------------------

class TestUploadWithProfile:
    def test_upload_returns_insights(self, client, sample_csv_content):
        import asyncio

        async def run():
            # Create a project first
            proj_resp = await client.post(
                "/api/projects", json={"name": "Test Project"}
            )
            assert proj_resp.status_code == 201
            project_id = proj_resp.json()["id"]

            resp = await client.post(
                "/api/data/upload",
                data={"project_id": project_id},
                files={"file": ("sales.csv", sample_csv_content, "text/csv")},
            )
            assert resp.status_code == 201
            body = resp.json()
            assert "insights" in body
            assert isinstance(body["insights"], list)
            assert "column_stats" in body
            # Check column stats have distributions
            for col_stat in body["column_stats"]:
                assert "distribution" in col_stat

        asyncio.get_event_loop().run_until_complete(run())

    def test_profile_endpoint(self, client, sample_csv_content):
        import asyncio

        async def run():
            proj_resp = await client.post(
                "/api/projects", json={"name": "Profile Test"}
            )
            project_id = proj_resp.json()["id"]

            upload_resp = await client.post(
                "/api/data/upload",
                data={"project_id": project_id},
                files={"file": ("sales.csv", sample_csv_content, "text/csv")},
            )
            dataset_id = upload_resp.json()["dataset_id"]

            profile_resp = await client.get(f"/api/data/{dataset_id}/profile")
            assert profile_resp.status_code == 200
            profile = profile_resp.json()
            assert "correlations" in profile
            assert "insights" in profile

        asyncio.get_event_loop().run_until_complete(run())


# ---------------------------------------------------------------------------
# GET /api/data/{dataset_id}/correlations
# ---------------------------------------------------------------------------

class TestCorrelationsEndpoint:
    async def test_returns_heatmap_spec(self, client, sample_csv_content):
        # Create project + upload data
        proj = await client.post("/api/projects", json={"name": "corr-test"})
        assert proj.status_code == 201
        pid = proj.json()["id"]

        up = await client.post(
            "/api/data/upload",
            data={"project_id": pid},
            files={"file": ("data.csv", sample_csv_content, "text/csv")},
        )
        assert up.status_code == 201
        dataset_id = up.json()["dataset_id"]

        resp = await client.get(f"/api/data/{dataset_id}/correlations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["chart_spec"] is not None
        assert data["chart_spec"]["chart_type"] == "heatmap"
        assert "pairs" in data

    async def test_nonexistent_dataset_404(self, client):
        resp = await client.get("/api/data/no-such-dataset/correlations")
        assert resp.status_code == 404

    async def test_no_numeric_columns(self, client):
        # Upload CSV with only categorical columns
        proj = await client.post("/api/projects", json={"name": "no-numeric"})
        pid = proj.json()["id"]
        csv = b"color,size\nred,small\nblue,large\ngreen,medium\n"
        up = await client.post(
            "/api/data/upload",
            data={"project_id": pid},
            files={"file": ("cat.csv", csv, "text/csv")},
        )
        assert up.status_code == 201
        dataset_id = up.json()["dataset_id"]

        resp = await client.get(f"/api/data/{dataset_id}/correlations")
        assert resp.status_code == 200
        # Should return null chart_spec with a helpful message
        data = resp.json()
        assert data["chart_spec"] is None
        assert "message" in data
