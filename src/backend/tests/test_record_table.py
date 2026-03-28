"""Tests for record table viewer: sample_records() + endpoint + chat patterns."""

import os

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel

from core.analyzer import sample_records


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sales_df():
    """Simple sales dataset."""
    return pd.DataFrame(
        {
            "customer": ["Alice", "Bob", "Carol", "Dave", "Eve"],
            "revenue": [5000.0, 1200.0, 8500.0, 300.0, 4200.0],
            "region": ["East", "West", "East", "North", "West"],
            "units": [50, 12, 85, 3, 42],
        }
    )


@pytest.fixture
def large_df():
    """60-row dataset for paging tests."""
    return pd.DataFrame(
        {"id": range(60), "value": [float(i) * 1.5 for i in range(60)]}
    )


# ---------------------------------------------------------------------------
# Unit tests: sample_records()
# ---------------------------------------------------------------------------


class TestSampleRecords:
    def test_basic_sample_returns_n_rows(self, sales_df):
        result = sample_records(sales_df, n=3)
        assert result["shown_rows"] == 3
        assert result["total_rows"] == 5
        assert result["filtered"] is False
        assert len(result["rows"]) == 3

    def test_returns_all_if_n_exceeds_rows(self, sales_df):
        result = sample_records(sales_df, n=20)
        assert result["shown_rows"] == 5
        assert len(result["rows"]) == 5

    def test_caps_n_at_50(self, large_df):
        result = sample_records(large_df, n=100)
        assert result["shown_rows"] == 50

    def test_min_n_is_1(self, sales_df):
        result = sample_records(sales_df, n=0)
        assert result["shown_rows"] == 1

    def test_columns_capped_at_8(self):
        df = pd.DataFrame({f"col_{i}": range(5) for i in range(12)})
        result = sample_records(df, n=5)
        assert len(result["columns"]) == 8

    def test_with_condition_filters_rows(self, sales_df):
        conditions = [{"column": "region", "operator": "eq", "value": "East"}]
        result = sample_records(sales_df, n=10, conditions=conditions)
        assert result["filtered"] is True
        assert result["filtered_rows"] == 2  # Alice and Carol are East
        assert len(result["rows"]) == 2

    def test_with_numeric_condition(self, sales_df):
        conditions = [{"column": "revenue", "operator": "gt", "value": 3000.0}]
        result = sample_records(sales_df, n=10, conditions=conditions)
        assert result["filtered"] is True
        assert result["filtered_rows"] == 3  # Alice 5000, Carol 8500, Eve 4200

    def test_no_match_returns_empty_rows(self, sales_df):
        conditions = [{"column": "region", "operator": "eq", "value": "Antarctica"}]
        result = sample_records(sales_df, n=10, conditions=conditions)
        assert result["filtered_rows"] == 0
        assert len(result["rows"]) == 0
        assert "No rows match" in result["summary"]

    def test_condition_summary_built(self, sales_df):
        conditions = [{"column": "region", "operator": "eq", "value": "East"}]
        result = sample_records(sales_df, n=10, conditions=conditions)
        assert "region" in result["condition_summary"]
        assert "East" in result["condition_summary"]

    def test_offset_paging(self, large_df):
        page1 = sample_records(large_df, n=10, offset=0)
        page2 = sample_records(large_df, n=10, offset=10)
        assert page1["rows"][0]["id"] == 0
        assert page2["rows"][0]["id"] == 10

    def test_summary_unfiltered(self, sales_df):
        result = sample_records(sales_df, n=3)
        assert "Showing 3" in result["summary"]

    def test_summary_filtered(self, sales_df):
        conditions = [{"column": "region", "operator": "eq", "value": "East"}]
        result = sample_records(sales_df, n=10, conditions=conditions)
        assert "matching" in result["summary"].lower()

    def test_nan_values_serialized_as_none(self):
        df = pd.DataFrame({"a": [1.0, float("nan"), 3.0], "b": ["x", "y", "z"]})
        result = sample_records(df, n=3)
        none_count = sum(1 for r in result["rows"] if r["a"] is None)
        assert none_count == 1

    def test_row_dicts_contain_all_display_columns(self, sales_df):
        result = sample_records(sales_df, n=2)
        expected_cols = set(result["columns"])
        for row in result["rows"]:
            assert set(row.keys()) == expected_cols


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
    from main import app as fastapi_app

    SQLModel.metadata.create_all(
        fastapi_app.state.engine if hasattr(fastapi_app.state, "engine") else __import__("db").engine
    )
    return fastapi_app


@pytest.fixture
def csv_file(tmp_path):
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Carol", "Dave", "Eve"],
            "score": [90, 45, 78, 92, 60],
            "grade": ["A", "F", "C", "A", "D"],
        }
    )
    p = tmp_path / "test.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.mark.asyncio
async def test_records_endpoint_returns_rows(app, csv_file):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Upload dataset
        with open(csv_file, "rb") as f:
            upload = await client.post(
                "/api/data/upload",
                files={"file": ("test.csv", f, "text/csv")},
                data={"project_id": "proj-rec-1"},
            )
        assert upload.status_code in (200, 201)
        dataset_id = upload.json()["dataset_id"]

        resp = await client.get(f"/api/data/{dataset_id}/records?n=3")
        assert resp.status_code == 200
        body = resp.json()
        assert body["shown_rows"] == 3
        assert len(body["rows"]) == 3
        assert "name" in body["columns"]


@pytest.mark.asyncio
async def test_records_endpoint_with_where(app, csv_file):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with open(csv_file, "rb") as f:
            upload = await client.post(
                "/api/data/upload",
                files={"file": ("test.csv", f, "text/csv")},
                data={"project_id": "proj-rec-2"},
            )
        dataset_id = upload.json()["dataset_id"]

        resp = await client.get(
            f"/api/data/{dataset_id}/records?n=10&where=score+>+80"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["filtered"] is True
        assert body["filtered_rows"] == 2  # Alice (90) and Dave (92)


@pytest.mark.asyncio
async def test_records_endpoint_unknown_dataset(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/data/nonexistent-id/records")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Chat pattern tests
# ---------------------------------------------------------------------------


def test_records_pattern_show_data():
    from api.chat import _RECORDS_PATTERNS

    assert _RECORDS_PATTERNS.search("show me the data")
    assert _RECORDS_PATTERNS.search("show me my data")
    assert _RECORDS_PATTERNS.search("show me the rows")
    assert _RECORDS_PATTERNS.search("display the records")
    assert _RECORDS_PATTERNS.search("preview the data")
    assert _RECORDS_PATTERNS.search("let me see the data")
    assert _RECORDS_PATTERNS.search("what does the data look like")
    assert _RECORDS_PATTERNS.search("show first 20 rows")
    assert _RECORDS_PATTERNS.search("give me a sample of the data")
    assert _RECORDS_PATTERNS.search("sample the records")
    assert _RECORDS_PATTERNS.search("show rows where region = East")
    assert _RECORDS_PATTERNS.search("find records where revenue > 1000")
    assert _RECORDS_PATTERNS.search("peek at the data")


def test_records_pattern_no_false_positives():
    from api.chat import _RECORDS_PATTERNS

    # Top-N queries are NOT record viewer
    assert not _RECORDS_PATTERNS.search("show me top 10 customers by revenue")
    assert not _RECORDS_PATTERNS.search("show me the best regions")
    # Error analysis
    assert not _RECORDS_PATTERNS.search("show me the prediction errors")
    assert not _RECORDS_PATTERNS.search("show me the model errors")


def test_detect_records_request_extracts_n():
    import pandas as pd
    from api.chat import _detect_records_request

    df = pd.DataFrame({"a": range(5), "b": range(5)})
    result = _detect_records_request("show first 15 rows", df)
    assert result["n"] == 15


def test_detect_records_request_default_n():
    import pandas as pd
    from api.chat import _detect_records_request

    df = pd.DataFrame({"a": range(5)})
    result = _detect_records_request("show me the data", df)
    assert result["n"] == 20


def test_detect_records_request_with_where_clause():
    import pandas as pd
    from api.chat import _detect_records_request

    df = pd.DataFrame({"region": ["East", "West"], "revenue": [100.0, 200.0]})
    result = _detect_records_request("show rows where region = East", df)
    assert result["conditions"] is not None
    assert len(result["conditions"]) > 0
    assert result["conditions"][0]["column"] == "region"
