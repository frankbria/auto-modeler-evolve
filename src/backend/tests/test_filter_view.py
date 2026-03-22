"""Tests for non-destructive data filter feature.

Covers:
  - core/filter_view.py: parse, apply, summarise, validate
  - POST /api/data/{id}/set-filter
  - DELETE /api/data/{id}/clear-filter
  - GET /api/data/{id}/active-filter
"""

import json
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
