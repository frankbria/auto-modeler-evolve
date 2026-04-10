"""Tests for non-destructive data filter feature.

Covers:
  - core/filter_view.py: parse, apply, summarise, validate
  - POST /api/data/{id}/set-filter
  - DELETE /api/data/{id}/clear-filter
  - GET /api/data/{id}/active-filter
"""

import pandas as pd
import pytest
from httpx import AsyncClient

# ---------------------------------------------------------------------------
# Unit tests — core/filter_view.py
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "region": ["North", "South", "East", "North", "West"],
            "category": ["A", "B", "A", "C", "B"],
            "revenue": [1200.0, 850.0, 2100.0, 1650.0, 450.0],
            "units": [10, 8, 18, 15, 4],
            "status": ["active", "inactive", "active", "active", "inactive"],
        }
    )


class TestApplyActiveFilter:
    def test_eq_string(self, sample_df):
        from core.filter_view import apply_active_filter

        result = apply_active_filter(
            sample_df, [{"column": "region", "operator": "eq", "value": "North"}]
        )
        assert len(result) == 2
        assert all(result["region"] == "North")

    def test_eq_case_insensitive(self, sample_df):
        from core.filter_view import apply_active_filter

        result = apply_active_filter(
            sample_df, [{"column": "region", "operator": "eq", "value": "north"}]
        )
        assert len(result) == 2

    def test_ne_string(self, sample_df):
        from core.filter_view import apply_active_filter

        result = apply_active_filter(
            sample_df, [{"column": "status", "operator": "ne", "value": "inactive"}]
        )
        assert len(result) == 3
        assert all(result["status"] == "active")

    def test_gt_numeric(self, sample_df):
        from core.filter_view import apply_active_filter

        result = apply_active_filter(
            sample_df, [{"column": "revenue", "operator": "gt", "value": 1000.0}]
        )
        assert len(result) == 3
        assert all(result["revenue"] > 1000.0)

    def test_lt_numeric(self, sample_df):
        from core.filter_view import apply_active_filter

        result = apply_active_filter(
            sample_df, [{"column": "units", "operator": "lt", "value": 10.0}]
        )
        assert len(result) == 2

    def test_gte_numeric(self, sample_df):
        from core.filter_view import apply_active_filter

        result = apply_active_filter(
            sample_df, [{"column": "revenue", "operator": "gte", "value": 1200.0}]
        )
        assert len(result) == 3

    def test_lte_numeric(self, sample_df):
        from core.filter_view import apply_active_filter

        result = apply_active_filter(
            sample_df, [{"column": "units", "operator": "lte", "value": 10.0}]
        )
        assert len(result) == 3

    def test_contains_string(self, sample_df):
        from core.filter_view import apply_active_filter

        result = apply_active_filter(
            sample_df, [{"column": "status", "operator": "contains", "value": "act"}]
        )
        assert len(result) == 5  # "active" and "inactive" both contain "act"

    def test_not_contains_string(self, sample_df):
        from core.filter_view import apply_active_filter

        result = apply_active_filter(
            sample_df,
            [{"column": "status", "operator": "not_contains", "value": "in"}],
        )
        assert len(result) == 3  # Only "active" rows (not "inactive")

    def test_multiple_conditions_and_logic(self, sample_df):
        from core.filter_view import apply_active_filter

        result = apply_active_filter(
            sample_df,
            [
                {"column": "region", "operator": "eq", "value": "North"},
                {"column": "revenue", "operator": "gt", "value": 1000.0},
            ],
        )
        assert len(result) == 2

    def test_unknown_column_silently_skipped(self, sample_df):
        from core.filter_view import apply_active_filter

        result = apply_active_filter(
            sample_df,
            [{"column": "nonexistent_column", "operator": "eq", "value": "X"}],
        )
        assert len(result) == len(sample_df)

    def test_empty_conditions_returns_full_df(self, sample_df):
        from core.filter_view import apply_active_filter

        result = apply_active_filter(sample_df, [])
        assert len(result) == len(sample_df)


class TestBuildFilterSummary:
    def test_eq_condition(self):
        from core.filter_view import build_filter_summary

        summary = build_filter_summary(
            [{"column": "region", "operator": "eq", "value": "North"}]
        )
        assert "region" in summary
        assert "North" in summary

    def test_multiple_conditions(self):
        from core.filter_view import build_filter_summary

        summary = build_filter_summary(
            [
                {"column": "region", "operator": "eq", "value": "North"},
                {"column": "revenue", "operator": "gt", "value": 1000},
            ]
        )
        assert "AND" in summary
        assert "region" in summary
        assert "revenue" in summary

    def test_empty_conditions(self):
        from core.filter_view import build_filter_summary

        summary = build_filter_summary([])
        assert "no filter" in summary.lower()


class TestValidateFilterConditions:
    def test_valid_conditions(self):
        from core.filter_view import validate_filter_conditions

        errors = validate_filter_conditions(
            [{"column": "region", "operator": "eq", "value": "North"}],
            ["region", "revenue"],
        )
        assert errors == []

    def test_invalid_column(self):
        from core.filter_view import validate_filter_conditions

        errors = validate_filter_conditions(
            [{"column": "nonexistent", "operator": "eq", "value": "X"}],
            ["region", "revenue"],
        )
        assert len(errors) == 1
        assert "nonexistent" in errors[0]

    def test_invalid_operator(self):
        from core.filter_view import validate_filter_conditions

        errors = validate_filter_conditions(
            [{"column": "region", "operator": "invalid_op", "value": "X"}],
            ["region", "revenue"],
        )
        assert len(errors) == 1

    def test_multiple_errors(self):
        from core.filter_view import validate_filter_conditions

        errors = validate_filter_conditions(
            [
                {"column": "bad_col", "operator": "bad_op", "value": "X"},
            ],
            ["region", "revenue"],
        )
        assert len(errors) == 2


class TestParseFilterRequest:
    def test_explicit_numeric_comparison(self):
        from core.filter_view import parse_filter_request

        result = parse_filter_request(
            "filter to revenue > 1000", ["region", "revenue", "units"]
        )
        assert result is not None
        assert any(c["column"] == "revenue" and c["operator"] == "gt" for c in result)

    def test_equality_pattern(self):
        from core.filter_view import parse_filter_request

        result = parse_filter_request(
            "region is North and revenue > 500", ["region", "revenue"]
        )
        assert result is not None
        assert any(c["column"] == "revenue" for c in result)

    def test_no_column_match_returns_none(self):
        from core.filter_view import parse_filter_request

        result = parse_filter_request("hello world", ["revenue", "region"])
        assert result is None

    def test_gte_operator(self):
        from core.filter_view import parse_filter_request

        result = parse_filter_request("units >= 10", ["units", "revenue"])
        assert result is not None
        assert any(c["operator"] == "gte" for c in result)


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
async def dataset_with_csv(client: AsyncClient, tmp_path):
    """Upload a sample CSV and return the dataset_id."""
    csv_content = b"region,revenue,units\nNorth,1200,10\nSouth,850,8\nNorth,2100,18\n"
    resp = await client.post(
        "/api/data/upload",
        files={"file": ("sales.csv", csv_content, "text/csv")},
        data={"project_id": "proj-filter-test"},
    )
    assert resp.status_code == 201
    return resp.json()["dataset_id"]


class TestFilterEndpoints:
    async def test_set_filter_valid(self, client: AsyncClient, dataset_with_csv):
        dataset_id = dataset_with_csv
        resp = await client.post(
            f"/api/data/{dataset_id}/set-filter",
            json={
                "conditions": [{"column": "region", "operator": "eq", "value": "North"}]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filtered_rows"] == 2
        assert data["original_rows"] == 3
        assert "region" in data["filter_summary"]

    async def test_set_filter_invalid_column(
        self, client: AsyncClient, dataset_with_csv
    ):
        resp = await client.post(
            f"/api/data/{dataset_with_csv}/set-filter",
            json={
                "conditions": [
                    {"column": "nonexistent", "operator": "eq", "value": "X"}
                ]
            },
        )
        assert resp.status_code == 400

    async def test_get_active_filter_before_set(
        self, client: AsyncClient, dataset_with_csv
    ):
        resp = await client.get(f"/api/data/{dataset_with_csv}/active-filter")
        assert resp.status_code == 200
        assert resp.json()["active"] is False

    async def test_get_active_filter_after_set(
        self, client: AsyncClient, dataset_with_csv
    ):
        await client.post(
            f"/api/data/{dataset_with_csv}/set-filter",
            json={
                "conditions": [{"column": "region", "operator": "eq", "value": "North"}]
            },
        )
        resp = await client.get(f"/api/data/{dataset_with_csv}/active-filter")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is True
        assert data["filtered_rows"] == 2
        assert len(data["conditions"]) == 1

    async def test_clear_filter(self, client: AsyncClient, dataset_with_csv):
        await client.post(
            f"/api/data/{dataset_with_csv}/set-filter",
            json={
                "conditions": [{"column": "region", "operator": "eq", "value": "North"}]
            },
        )
        resp = await client.delete(f"/api/data/{dataset_with_csv}/clear-filter")
        assert resp.status_code == 200
        assert resp.json()["cleared"] is True

        # Verify cleared
        resp2 = await client.get(f"/api/data/{dataset_with_csv}/active-filter")
        assert resp2.json()["active"] is False

    async def test_set_filter_overwrites_existing(
        self, client: AsyncClient, dataset_with_csv
    ):
        await client.post(
            f"/api/data/{dataset_with_csv}/set-filter",
            json={
                "conditions": [{"column": "region", "operator": "eq", "value": "North"}]
            },
        )
        # Overwrite with new filter
        resp = await client.post(
            f"/api/data/{dataset_with_csv}/set-filter",
            json={
                "conditions": [{"column": "revenue", "operator": "gt", "value": 1000}]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "revenue" in data["filter_summary"]
        assert data["filtered_rows"] == 2

    async def test_clear_filter_no_op_if_none_set(
        self, client: AsyncClient, dataset_with_csv
    ):
        resp = await client.delete(f"/api/data/{dataset_with_csv}/clear-filter")
        assert resp.status_code == 200
        assert resp.json()["cleared"] is True

    async def test_set_filter_404_for_missing_dataset(self, client: AsyncClient):
        resp = await client.post(
            "/api/data/nonexistent-dataset/set-filter",
            json={
                "conditions": [{"column": "region", "operator": "eq", "value": "North"}]
            },
        )
        assert resp.status_code == 404

    async def test_get_active_filter_404_for_missing_dataset(self, client: AsyncClient):
        resp = await client.get("/api/data/nonexistent-dataset/active-filter")
        assert resp.status_code == 404

    async def test_row_reduction_pct(self, client: AsyncClient, dataset_with_csv):
        resp = await client.post(
            f"/api/data/{dataset_with_csv}/set-filter",
            json={
                "conditions": [{"column": "region", "operator": "eq", "value": "South"}]
            },
        )
        data = resp.json()
        assert data["filtered_rows"] == 1
        assert data["original_rows"] == 3
        assert data["row_reduction_pct"] == pytest.approx(66.7, abs=0.2)

    async def test_set_filter_multiple_conditions(
        self, client: AsyncClient, dataset_with_csv
    ):
        resp = await client.post(
            f"/api/data/{dataset_with_csv}/set-filter",
            json={
                "conditions": [
                    {"column": "region", "operator": "eq", "value": "North"},
                    {"column": "revenue", "operator": "gt", "value": 1500},
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filtered_rows"] == 1  # Only North + revenue>1500
        assert "AND" in data["filter_summary"]


# ---------------------------------------------------------------------------
# Unit tests — parse_date_filter_request + date_range apply
# ---------------------------------------------------------------------------


@pytest.fixture
def date_df():
    """DataFrame with a date column spanning 2022–2024."""
    import pandas as pd

    dates = pd.date_range("2022-01-01", periods=36, freq="MS")  # 36 months
    return pd.DataFrame(
        {
            "sale_date": dates.strftime("%Y-%m-%d"),
            "revenue": range(36),
        }
    )


class TestParseDateFilterRequest:
    def test_quarter_numeric(self, date_df):
        from core.filter_view import parse_date_filter_request

        result = parse_date_filter_request("show Q1 2023 data", date_df)
        assert result is not None
        assert len(result) == 1
        cond = result[0]
        assert cond["operator"] == "date_range"
        assert cond["value"]["start"] == "2023-01-01"
        assert cond["value"]["end"] == "2023-03-31"

    def test_quarter_q4(self, date_df):
        from core.filter_view import parse_date_filter_request

        result = parse_date_filter_request("filter to Q4 2022", date_df)
        assert result is not None
        assert result[0]["value"]["start"] == "2022-10-01"
        assert result[0]["value"]["end"] == "2022-12-31"

    def test_quarter_word(self, date_df):
        from core.filter_view import parse_date_filter_request

        result = parse_date_filter_request("show third quarter 2023", date_df)
        assert result is not None
        assert result[0]["value"]["start"] == "2023-07-01"
        assert result[0]["value"]["end"] == "2023-09-30"

    def test_year_only(self, date_df):
        from core.filter_view import parse_date_filter_request

        result = parse_date_filter_request("show 2023 data", date_df)
        assert result is not None
        assert result[0]["value"]["start"] == "2023-01-01"
        assert result[0]["value"]["end"] == "2023-12-31"

    def test_month_range(self, date_df):
        from core.filter_view import parse_date_filter_request

        result = parse_date_filter_request(
            "filter to January through March 2023", date_df
        )
        assert result is not None
        assert result[0]["value"]["start"] == "2023-01-01"
        assert result[0]["value"]["end"] == "2023-03-31"

    def test_single_month_with_year(self, date_df):
        from core.filter_view import parse_date_filter_request

        result = parse_date_filter_request("narrow to June 2023", date_df)
        assert result is not None
        assert result[0]["value"]["start"] == "2023-06-01"
        assert result[0]["value"]["end"] == "2023-06-30"

    def test_last_n_months(self, date_df):
        from datetime import date

        from core.filter_view import parse_date_filter_request

        result = parse_date_filter_request("show last 3 months", date_df)
        assert result is not None
        cond = result[0]
        assert cond["operator"] == "date_range"
        # start should be ~90 days ago
        start = date.fromisoformat(cond["value"]["start"])
        today = date.today()
        delta = (today - start).days
        assert 85 <= delta <= 95  # ~90 days (3 × 30)

    def test_last_n_years(self, date_df):
        from datetime import date

        from core.filter_view import parse_date_filter_request

        result = parse_date_filter_request("filter to last 2 years", date_df)
        assert result is not None
        start = date.fromisoformat(result[0]["value"]["start"])
        today = date.today()
        delta = (today - start).days
        assert 720 <= delta <= 740  # ~730 days

    def test_this_year(self, date_df):
        from datetime import date

        from core.filter_view import parse_date_filter_request

        result = parse_date_filter_request("show this year", date_df)
        assert result is not None
        assert result[0]["value"]["start"] == f"{date.today().year}-01-01"
        assert result[0]["value"]["end"] == f"{date.today().year}-12-31"

    def test_last_year(self, date_df):
        from datetime import date

        from core.filter_view import parse_date_filter_request

        result = parse_date_filter_request("filter to last year", date_df)
        assert result is not None
        last_yr = date.today().year - 1
        assert result[0]["value"]["start"] == f"{last_yr}-01-01"
        assert result[0]["value"]["end"] == f"{last_yr}-12-31"

    def test_no_date_column_returns_none(self):
        import pandas as pd

        from core.filter_view import parse_date_filter_request

        df = pd.DataFrame({"region": ["North", "South"], "revenue": [100, 200]})
        result = parse_date_filter_request("show Q1 2023", df)
        assert result is None

    def test_unrelated_message_returns_none(self, date_df):
        from core.filter_view import parse_date_filter_request

        result = parse_date_filter_request("what is the average revenue", date_df)
        assert result is None

    def test_date_range_column_detection_by_value(self):
        """Column named 'period' (no common date keyword) should still be detected by values."""
        import pandas as pd

        from core.filter_view import parse_date_filter_request

        df = pd.DataFrame(
            {
                "period": ["2023-01-01", "2023-04-01", "2023-07-01"],
                "revenue": [100, 200, 300],
            }
        )
        result = parse_date_filter_request("show Q2 2023", df)
        # column 'period' contains date strings → detected; Q2 → filter returned
        assert result is not None
        assert result[0]["value"]["start"] == "2023-04-01"


class TestApplyDateRangeFilter:
    def test_date_range_filters_rows(self, date_df):
        from core.filter_view import apply_active_filter

        conditions = [
            {
                "column": "sale_date",
                "operator": "date_range",
                "value": {"start": "2023-01-01", "end": "2023-03-31"},
            }
        ]
        result = apply_active_filter(date_df, conditions)
        assert len(result) == 3
        assert all(result["sale_date"].str.startswith("2023-0"))

    def test_date_range_full_year(self, date_df):
        from core.filter_view import apply_active_filter

        conditions = [
            {
                "column": "sale_date",
                "operator": "date_range",
                "value": {"start": "2023-01-01", "end": "2023-12-31"},
            }
        ]
        result = apply_active_filter(date_df, conditions)
        assert len(result) == 12

    def test_date_range_empty_result(self, date_df):
        from core.filter_view import apply_active_filter

        conditions = [
            {
                "column": "sale_date",
                "operator": "date_range",
                "value": {"start": "2025-01-01", "end": "2025-12-31"},
            }
        ]
        result = apply_active_filter(date_df, conditions)
        assert len(result) == 0


class TestBuildDateRangeSummary:
    def test_date_range_summary(self):
        from core.filter_view import build_filter_summary

        summary = build_filter_summary(
            [
                {
                    "column": "sale_date",
                    "operator": "date_range",
                    "value": {"start": "2023-01-01", "end": "2023-03-31"},
                }
            ]
        )
        assert "sale_date" in summary
        assert "2023-01-01" in summary
        assert "2023-03-31" in summary
        assert "between" in summary
