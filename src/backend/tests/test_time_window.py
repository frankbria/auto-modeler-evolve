"""Tests for time-period comparison: compare_time_windows() + endpoint + chat patterns."""

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

from core.analyzer import compare_time_windows, _build_timewindow_summary


# ---------------------------------------------------------------------------
# Test DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def sales_df():
    """Multi-year sales dataset with a date column."""
    return pd.DataFrame(
        {
            "date": [
                "2023-01-15",
                "2023-04-10",
                "2023-07-20",
                "2023-10-05",
                "2024-02-01",
                "2024-05-15",
                "2024-08-22",
                "2024-11-30",
            ],
            "revenue": [1000.0, 1200.0, 900.0, 1100.0, 1400.0, 1600.0, 1300.0, 1800.0],
            "units": [10, 12, 9, 11, 14, 16, 13, 18],
            "cost": [500.0, 600.0, 450.0, 550.0, 700.0, 800.0, 650.0, 900.0],
        }
    )


@pytest.fixture
def single_year_df():
    """Dataset with a single year for H1/H2 split testing."""
    return pd.DataFrame(
        {
            "date": [
                "2024-01-10",
                "2024-02-15",
                "2024-03-20",
                "2024-07-05",
                "2024-08-12",
                "2024-09-25",
            ],
            "revenue": [500.0, 600.0, 550.0, 750.0, 800.0, 720.0],
            "units": [5, 6, 5, 8, 8, 7],
        }
    )


@pytest.fixture
def no_date_df():
    return pd.DataFrame({"revenue": [1, 2, 3], "units": [10, 20, 30]})


@pytest.fixture
def no_numeric_df():
    return pd.DataFrame(
        {"date": ["2024-01-01", "2024-02-01"], "region": ["North", "South"]}
    )


# ---------------------------------------------------------------------------
# Unit tests — compare_time_windows
# ---------------------------------------------------------------------------


def test_basic_year_comparison(sales_df):
    result = compare_time_windows(
        sales_df,
        "date",
        "2023",
        "2023-01-01",
        "2023-12-31",
        "2024",
        "2024-01-01",
        "2024-12-31",
    )
    assert "error" not in result
    assert result["period1"]["name"] == "2023"
    assert result["period2"]["name"] == "2024"
    assert result["period1"]["row_count"] == 4
    assert result["period2"]["row_count"] == 4


def test_pct_change_direction(sales_df):
    result = compare_time_windows(
        sales_df,
        "date",
        "2023",
        "2023-01-01",
        "2023-12-31",
        "2024",
        "2024-01-01",
        "2024-12-31",
    )
    # Revenue is higher in 2024 → direction should be "up"
    rev_col = next(c for c in result["columns"] if c["column"] == "revenue")
    assert rev_col["direction"] == "up"
    assert rev_col["pct_change"] > 0


def test_notable_changes_flagged(sales_df):
    result = compare_time_windows(
        sales_df,
        "date",
        "2023",
        "2023-01-01",
        "2023-12-31",
        "2024",
        "2024-01-01",
        "2024-12-31",
    )
    # Revenue goes from avg ~1050 to avg ~1525 → ~45% change — should be notable
    rev_col = next(c for c in result["columns"] if c["column"] == "revenue")
    assert rev_col["notable"] is True
    assert "revenue" in result["notable_changes"]


def test_columns_list_structure(sales_df):
    result = compare_time_windows(
        sales_df,
        "date",
        "2023",
        "2023-01-01",
        "2023-12-31",
        "2024",
        "2024-01-01",
        "2024-12-31",
    )
    assert len(result["columns"]) == 3  # revenue, units, cost
    for col in result["columns"]:
        assert "column" in col
        assert "p1_mean" in col
        assert "p2_mean" in col
        assert "pct_change" in col
        assert col["direction"] in ("up", "down", "flat")
        assert isinstance(col["notable"], bool)


def test_summary_is_string(sales_df):
    result = compare_time_windows(
        sales_df,
        "date",
        "2023",
        "2023-01-01",
        "2023-12-31",
        "2024",
        "2024-01-01",
        "2024-12-31",
    )
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 20
    assert "2023" in result["summary"]
    assert "2024" in result["summary"]


def test_unknown_date_col_returns_error(sales_df):
    result = compare_time_windows(
        sales_df,
        "nonexistent",
        "2023",
        "2023-01-01",
        "2023-12-31",
        "2024",
        "2024-01-01",
        "2024-12-31",
    )
    assert "error" in result


def test_no_rows_in_period_returns_error(sales_df):
    result = compare_time_windows(
        sales_df,
        "date",
        "P1",
        "2020-01-01",
        "2020-12-31",
        "P2",
        "2024-01-01",
        "2024-12-31",
    )
    assert "error" in result
    assert "P1" in result["error"]


def test_no_numeric_columns_returns_error(no_numeric_df):
    result = compare_time_windows(
        no_numeric_df,
        "date",
        "A",
        "2024-01-01",
        "2024-01-31",
        "B",
        "2024-02-01",
        "2024-02-28",
    )
    assert "error" in result


def test_flat_direction_near_zero_change():
    df = pd.DataFrame(
        {
            "date": ["2023-06-01", "2023-07-01", "2024-06-01", "2024-07-01"],
            "revenue": [1000.0, 1000.0, 1005.0, 995.0],  # <1% change
        }
    )
    result = compare_time_windows(
        df,
        "date",
        "H1-2023",
        "2023-06-01",
        "2023-07-01",
        "H1-2024",
        "2024-06-01",
        "2024-07-01",
    )
    assert "error" not in result
    rev_col = next(c for c in result["columns"] if c["column"] == "revenue")
    assert rev_col["direction"] == "flat"


def test_invalid_date_boundaries_returns_error(sales_df):
    result = compare_time_windows(
        sales_df,
        "date",
        "P1",
        "not-a-date",
        "also-not",
        "P2",
        "2024-01-01",
        "2024-12-31",
    )
    assert "error" in result


def test_build_timewindow_summary_no_notable():
    cols = [
        {
            "column": "revenue",
            "p1_mean": 1000.0,
            "p2_mean": 1050.0,
            "pct_change": 5.0,
            "direction": "up",
            "notable": False,
        },
    ]
    summary = _build_timewindow_summary("2023", "2024", 10, 12, cols, [])
    assert "2023" in summary
    assert "no metrics changed by more than 20%" in summary


def test_build_timewindow_summary_with_notable():
    cols = [
        {
            "column": "revenue",
            "p1_mean": 1000.0,
            "p2_mean": 1500.0,
            "pct_change": 50.0,
            "direction": "up",
            "notable": True,
        },
        {
            "column": "cost",
            "p1_mean": 500.0,
            "p2_mean": 400.0,
            "pct_change": -20.0,
            "direction": "down",
            "notable": True,
        },
    ]
    summary = _build_timewindow_summary(
        "2023", "2024", 10, 12, cols, ["revenue", "cost"]
    )
    assert "revenue" in summary or "cost" in summary
    assert "increased" in summary or "decreased" in summary


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
async def client_with_dataset(tmp_path, set_test_env):
    import db
    from main import app

    test_db = str(tmp_path / "test.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db.DATA_DIR = tmp_path
    SQLModel.metadata.create_all(db.engine)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        # Upload a CSV with date + numeric columns
        csv_data = (
            "date,revenue,units\n"
            "2023-01-01,1000.0,10\n"
            "2023-07-01,1100.0,11\n"
            "2024-01-01,1400.0,14\n"
            "2024-07-01,1500.0,15\n"
        )
        # Create project first
        proj_resp = await ac.post("/api/projects", json={"name": "Test Project"})
        assert proj_resp.status_code == 201
        project_id = proj_resp.json()["id"]

        # Upload
        upload_resp = await ac.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("sales.csv", csv_data.encode(), "text/csv")},
        )
        assert upload_resp.status_code == 201
        dataset_id = upload_resp.json()["dataset_id"]

        yield ac, dataset_id


@pytest.mark.anyio
async def test_endpoint_returns_comparison(client_with_dataset):
    ac, dataset_id = client_with_dataset
    resp = await ac.get(
        f"/api/data/{dataset_id}/compare-time-windows",
        params={
            "date_col": "date",
            "p1_name": "2023",
            "p1_start": "2023-01-01",
            "p1_end": "2023-12-31",
            "p2_name": "2024",
            "p2_start": "2024-01-01",
            "p2_end": "2024-12-31",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["period1"]["name"] == "2023"
    assert data["period2"]["name"] == "2024"
    assert len(data["columns"]) >= 1
    assert "summary" in data


@pytest.mark.anyio
async def test_endpoint_404_on_missing_dataset(client_with_dataset):
    ac, _ = client_with_dataset
    resp = await ac.get(
        "/api/data/nonexistent-id/compare-time-windows",
        params={
            "date_col": "date",
            "p1_name": "2023",
            "p1_start": "2023-01-01",
            "p1_end": "2023-12-31",
            "p2_name": "2024",
            "p2_start": "2024-01-01",
            "p2_end": "2024-12-31",
        },
    )
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_endpoint_400_on_unknown_date_col(client_with_dataset):
    ac, dataset_id = client_with_dataset
    resp = await ac.get(
        f"/api/data/{dataset_id}/compare-time-windows",
        params={
            "date_col": "nonexistent",
            "p1_name": "2023",
            "p1_start": "2023-01-01",
            "p1_end": "2023-12-31",
            "p2_name": "2024",
            "p2_start": "2024-01-01",
            "p2_end": "2024-12-31",
        },
    )
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_endpoint_400_on_empty_period(client_with_dataset):
    ac, dataset_id = client_with_dataset
    resp = await ac.get(
        f"/api/data/{dataset_id}/compare-time-windows",
        params={
            "date_col": "date",
            "p1_name": "2020",
            "p1_start": "2020-01-01",
            "p1_end": "2020-12-31",
            "p2_name": "2024",
            "p2_start": "2024-01-01",
            "p2_end": "2024-12-31",
        },
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Chat pattern tests
# ---------------------------------------------------------------------------


def test_timewindow_pattern_year_vs_year():
    from api.chat import _TIMEWINDOW_PATTERNS

    assert _TIMEWINDOW_PATTERNS.search("compare 2023 vs 2024")
    assert _TIMEWINDOW_PATTERNS.search("2023 vs 2024 revenue")
    assert _TIMEWINDOW_PATTERNS.search("how did 2023 compare to 2024?")


def test_timewindow_pattern_quarter_vs_quarter():
    from api.chat import _TIMEWINDOW_PATTERNS

    assert _TIMEWINDOW_PATTERNS.search("Q1 vs Q2")
    assert _TIMEWINDOW_PATTERNS.search("Q3 vs Q4 performance")
    assert _TIMEWINDOW_PATTERNS.search("compare Q1 2023 vs Q2 2023")


def test_timewindow_pattern_yoy():
    from api.chat import _TIMEWINDOW_PATTERNS

    assert _TIMEWINDOW_PATTERNS.search("year over year growth")
    assert _TIMEWINDOW_PATTERNS.search("yoy comparison")
    assert _TIMEWINDOW_PATTERNS.search("this year vs last year")


def test_timewindow_pattern_mom():
    from api.chat import _TIMEWINDOW_PATTERNS

    assert _TIMEWINDOW_PATTERNS.search("month over month change")
    assert _TIMEWINDOW_PATTERNS.search("mom trend")
    assert _TIMEWINDOW_PATTERNS.search("this month vs last month")


def test_timewindow_pattern_half_year():
    from api.chat import _TIMEWINDOW_PATTERNS

    assert _TIMEWINDOW_PATTERNS.search("first half vs second half")
    assert _TIMEWINDOW_PATTERNS.search("H1 vs H2 performance")


def test_timewindow_pattern_no_match():
    from api.chat import _TIMEWINDOW_PATTERNS

    assert not _TIMEWINDOW_PATTERNS.search("what is the average revenue?")
    assert not _TIMEWINDOW_PATTERNS.search("show me anomalies")
    assert not _TIMEWINDOW_PATTERNS.search("cluster my customers")


# ---------------------------------------------------------------------------
# _detect_timewindow_request tests
# ---------------------------------------------------------------------------


def test_detect_year_vs_year(sales_df):
    from api.chat import _detect_timewindow_request

    result = _detect_timewindow_request("compare 2023 vs 2024", sales_df)
    assert result is not None
    assert result["period1_name"] == "2023"
    assert result["period2_name"] == "2024"
    assert result["period1_start"] == "2023-01-01"
    assert result["period1_end"] == "2023-12-31"
    assert result["period2_start"] == "2024-01-01"
    assert result["period2_end"] == "2024-12-31"


def test_detect_quarter_vs_quarter(sales_df):
    from api.chat import _detect_timewindow_request

    result = _detect_timewindow_request("Q1 vs Q2 performance", sales_df)
    assert result is not None
    assert "Q1" in result["period1_name"]
    assert "Q2" in result["period2_name"]
    # Q1 ends March 31, Q2 starts April 1
    assert result["period1_end"] == f"{result['period1_name'][-4:]}-03-31"
    assert result["period2_start"] == f"{result['period2_name'][-4:]}-04-01"


def test_detect_yoy_uses_data_years(sales_df):
    from api.chat import _detect_timewindow_request

    result = _detect_timewindow_request("year over year comparison", sales_df)
    assert result is not None
    # Most recent year in sales_df is 2024 → latest year
    assert result["period2_name"] == "2024"
    assert result["period1_name"] == "2023"


def test_detect_fallback_bisects_range(sales_df):
    from api.chat import _detect_timewindow_request

    # No specific pattern → bisect
    result = _detect_timewindow_request("compare the two periods", sales_df)
    assert result is not None
    assert result["date_col"] == "date"
    assert result["period1_start"] < result["period2_start"]


def test_detect_returns_none_on_no_date_col(no_date_df):
    from api.chat import _detect_timewindow_request

    result = _detect_timewindow_request("compare 2023 vs 2024", no_date_df)
    assert result is None
