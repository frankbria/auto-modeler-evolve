"""Tests for group trend analysis: compute_group_trends() and the REST endpoint."""

import pytest
import pandas as pd
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel

from core.analyzer import compute_group_trends

# ---------------------------------------------------------------------------
# Test DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def trend_df():
    """DataFrame with clear trends: East growing, West declining, North flat."""
    rows = []
    for month in range(1, 13):
        rows.append(
            {
                "date": f"2024-{month:02d}-01",
                "region": "East",
                "revenue": 1000 + month * 100,  # growing
            }
        )
        rows.append(
            {
                "date": f"2024-{month:02d}-01",
                "region": "West",
                "revenue": 5000 - month * 200,  # declining
            }
        )
        rows.append(
            {
                "date": f"2024-{month:02d}-01",
                "region": "North",
                "revenue": 3000 + (month % 2) * 10,  # near-flat
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def single_record_df():
    """Each group has only 1 record — can't compute a trend."""
    return pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "region": ["East", "West"],
            "revenue": [1000.0, 2000.0],
        }
    )


@pytest.fixture
def no_dates_df():
    """DataFrame with no parseable date column."""
    return pd.DataFrame(
        {
            "label": ["foo", "bar"],
            "region": ["East", "West"],
            "revenue": [1000.0, 2000.0],
        }
    )


# ---------------------------------------------------------------------------
# Unit tests: compute_group_trends()
# ---------------------------------------------------------------------------


class TestComputeGroupTrends:
    def test_basic_trends(self, trend_df):
        result = compute_group_trends(
            trend_df, date_col="date", group_col="region", value_col="revenue"
        )
        assert "error" not in result
        assert result["date_col"] == "date"
        assert result["group_col"] == "region"
        assert result["value_col"] == "revenue"
        groups = {r["group"]: r for r in result["groups"]}
        assert "East" in groups
        assert "West" in groups
        assert groups["East"]["direction"] == "up"
        assert groups["West"]["direction"] == "down"

    def test_sorted_by_slope_descending(self, trend_df):
        result = compute_group_trends(
            trend_df, date_col="date", group_col="region", value_col="revenue"
        )
        slopes = [r["slope"] for r in result["groups"]]
        assert slopes == sorted(slopes, reverse=True)

    def test_rank_field(self, trend_df):
        result = compute_group_trends(
            trend_df, date_col="date", group_col="region", value_col="revenue"
        )
        ranks = [r["rank"] for r in result["groups"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_rising_falling_counts(self, trend_df):
        result = compute_group_trends(
            trend_df, date_col="date", group_col="region", value_col="revenue"
        )
        assert result["rising"] >= 1
        assert result["falling"] >= 1

    def test_summary_contains_top_group(self, trend_df):
        result = compute_group_trends(
            trend_df, date_col="date", group_col="region", value_col="revenue"
        )
        assert "East" in result["summary"]

    def test_pct_change_direction_consistency(self, trend_df):
        result = compute_group_trends(
            trend_df, date_col="date", group_col="region", value_col="revenue"
        )
        groups = {r["group"]: r for r in result["groups"]}
        east = groups["East"]
        assert east["pct_change"] > 0
        assert east["direction"] == "up"
        west = groups["West"]
        assert west["pct_change"] < 0
        assert west["direction"] == "down"

    def test_missing_date_col_returns_error(self, trend_df):
        result = compute_group_trends(
            trend_df, date_col="nonexistent", group_col="region", value_col="revenue"
        )
        assert "error" in result

    def test_missing_group_col_returns_error(self, trend_df):
        result = compute_group_trends(
            trend_df, date_col="date", group_col="nonexistent", value_col="revenue"
        )
        assert "error" in result

    def test_missing_value_col_returns_error(self, trend_df):
        result = compute_group_trends(
            trend_df, date_col="date", group_col="region", value_col="nonexistent"
        )
        assert "error" in result

    def test_high_cardinality_group_col_returns_error(self):
        """Groups with >50 unique values should return an error."""
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=60),
                "region": [f"group_{i}" for i in range(60)],
                "revenue": range(60),
            }
        )
        result = compute_group_trends(
            df, date_col="date", group_col="region", value_col="revenue"
        )
        assert "error" in result
        assert "50" in result["error"]

    def test_single_record_per_group_skipped(self, single_record_df):
        """Groups with only 1 record should be skipped; result may have no groups."""
        result = compute_group_trends(
            single_record_df, date_col="date", group_col="region", value_col="revenue"
        )
        # Either error (no groups) or empty groups list
        assert "error" in result or len(result.get("groups", [])) == 0

    def test_result_fields_present(self, trend_df):
        result = compute_group_trends(
            trend_df, date_col="date", group_col="region", value_col="revenue"
        )
        assert "groups" in result
        assert "rising" in result
        assert "falling" in result
        assert "flat" in result
        assert "summary" in result
        for row in result["groups"]:
            for key in (
                "group",
                "slope",
                "pct_change",
                "direction",
                "first_value",
                "last_value",
                "n_periods",
                "rank",
            ):
                assert key in row, f"Missing key: {key}"

    def test_with_null_dates_and_values(self):
        """Rows with null dates or values should be dropped gracefully."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", None, "2024-03-01", "2024-04-01", "2024-05-01"],
                "region": ["East", "East", "East", "West", "West"],
                "revenue": [1000.0, 1500.0, None, 2000.0, 2500.0],
            }
        )
        result = compute_group_trends(
            df, date_col="date", group_col="region", value_col="revenue"
        )
        # Should not raise; East may have < 2 valid rows but West should work
        assert "error" in result or "groups" in result

    def test_direction_flat(self):
        """Groups with identical values across time should be flat."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-02-01", "2024-03-01"] * 2,
                "region": ["East"] * 3 + ["West"] * 3,
                "revenue": [1000.0, 1000.0, 1000.0, 2000.0, 2000.0, 2000.0],
            }
        )
        result = compute_group_trends(
            df, date_col="date", group_col="region", value_col="revenue"
        )
        assert "error" not in result
        for row in result["groups"]:
            assert row["direction"] == "flat"
        assert result["flat"] == 2


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def trend_csv(tmp_path, trend_df):
    """Write trend_df to a CSV and return the path."""
    path = tmp_path / "trends.csv"
    trend_df.to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_group_trends_endpoint_success(trend_csv):
    from main import app
    from db import get_session, engine
    from models.dataset import Dataset

    SQLModel.metadata.create_all(engine)
    with next(get_session()) as session:
        ds = Dataset(
            id="gt-ds-1",
            project_id="proj-1",
            filename="trends.csv",
            file_path=str(trend_csv),
            row_count=36,
            column_count=3,
            columns="{}",
        )
        session.merge(ds)
        session.commit()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get(
            "/api/data/gt-ds-1/group-trends",
            params={"date_col": "date", "group_col": "region", "value_col": "revenue"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "groups" in data
    assert len(data["groups"]) == 3
    assert "summary" in data


@pytest.mark.anyio
async def test_group_trends_endpoint_404(tmp_path):
    from main import app
    from db import engine

    SQLModel.metadata.create_all(engine)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get(
            "/api/data/nonexistent-ds/group-trends",
            params={"date_col": "date", "group_col": "region", "value_col": "revenue"},
        )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_group_trends_endpoint_400_bad_col(trend_csv):
    from main import app
    from db import get_session, engine
    from models.dataset import Dataset

    SQLModel.metadata.create_all(engine)
    with next(get_session()) as session:
        ds = Dataset(
            id="gt-ds-2",
            project_id="proj-1",
            filename="trends.csv",
            file_path=str(trend_csv),
            row_count=36,
            column_count=3,
            columns="{}",
        )
        session.merge(ds)
        session.commit()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get(
            "/api/data/gt-ds-2/group-trends",
            params={
                "date_col": "date",
                "group_col": "nonexistent",
                "value_col": "revenue",
            },
        )
    assert resp.status_code == 400
    assert (
        "error" in resp.json().get("detail", "").lower()
        or "not found" in resp.json().get("detail", "").lower()
    )
