"""Tests for the time-series forecasting feature.

Covers:
  - core/forecaster.py unit tests (detect_time_series, forecast_next_periods)
  - GET /api/data/{id}/forecast API endpoint
  - Chat intent detection patterns (_FORECAST_PATTERNS, _detect_forecast_request)
"""

from __future__ import annotations

import io
from datetime import date, timedelta

import pandas as pd
import pytest

from core.forecaster import (
    _detect_frequency,
    _fmt_value,
    detect_time_series,
    forecast_next_periods,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_monthly_df(n: int = 24, start: str = "2022-01-01") -> pd.DataFrame:
    """Create a simple monthly time-series DataFrame."""
    base = date.fromisoformat(start)
    rows = []
    for i in range(n):
        d = date(
            base.year + (base.month + i - 1) // 12, (base.month + i - 1) % 12 + 1, 1
        )
        rows.append({"month": str(d), "revenue": 1000 + i * 50 + (i % 3) * 20})
    return pd.DataFrame(rows)


def make_daily_df(n: int = 30) -> pd.DataFrame:
    base = date(2024, 1, 1)
    return pd.DataFrame(
        [{"day": str(base + timedelta(days=i)), "sales": 200 + i * 5} for i in range(n)]
    )


def make_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


# ---------------------------------------------------------------------------
# detect_time_series
# ---------------------------------------------------------------------------


class TestDetectTimeSeries:
    def test_returns_none_for_no_date_column(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert detect_time_series(df) is None

    def test_returns_none_for_fewer_than_4_rows(self):
        df = pd.DataFrame(
            {"date": ["2024-01-01", "2024-01-02", "2024-01-03"], "val": [1, 2, 3]}
        )
        assert detect_time_series(df) is None

    def test_returns_none_when_no_numeric_columns(self):
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
                "label": ["a", "b", "c", "d"],
            }
        )
        assert detect_time_series(df) is None

    def test_detects_monthly_data(self):
        df = make_monthly_df(12)
        result = detect_time_series(df)
        assert result is not None
        assert result["date_col"] == "month"
        assert "revenue" in result["value_cols"]

    def test_detects_daily_data(self):
        df = make_daily_df(10)
        result = detect_time_series(df)
        assert result is not None
        assert "day" in result["date_col"] or "date" in result["date_col"]

    def test_returns_multiple_value_cols(self):
        df = make_monthly_df(12)
        df["units"] = range(12)
        result = detect_time_series(df)
        assert result is not None
        assert "revenue" in result["value_cols"]
        assert "units" in result["value_cols"]


# ---------------------------------------------------------------------------
# _detect_frequency
# ---------------------------------------------------------------------------


class TestDetectFrequency:
    def test_daily(self):
        dates = [pd.Timestamp(f"2024-01-{i:02d}") for i in range(1, 10)]
        label, delta = _detect_frequency(dates)
        assert label == "day"
        assert delta.days == 1

    def test_weekly(self):
        base = pd.Timestamp("2024-01-01")
        dates = [base + pd.Timedelta(weeks=i) for i in range(8)]
        label, delta = _detect_frequency(dates)
        assert label == "week"

    def test_monthly(self):
        dates = [pd.Timestamp(f"2024-{m:02d}-01") for m in range(1, 13)]
        label, delta = _detect_frequency(dates)
        assert label == "month"

    def test_single_date_returns_default(self):
        label, delta = _detect_frequency([pd.Timestamp("2024-01-01")])
        assert label == "month"


# ---------------------------------------------------------------------------
# forecast_next_periods
# ---------------------------------------------------------------------------


class TestForecastNextPeriods:
    def test_returns_required_keys(self):
        df = make_monthly_df(24)
        result = forecast_next_periods(df, "month", "revenue", periods=6)
        for key in (
            "chart_type",
            "date_col",
            "value_col",
            "historical",
            "forecast",
            "period_label",
            "trend",
            "growth_pct",
            "summary",
            "ci_level",
        ):
            assert key in result, f"Missing key: {key}"

    def test_chart_type_is_forecast(self):
        df = make_monthly_df(12)
        result = forecast_next_periods(df, "month", "revenue", periods=3)
        assert result["chart_type"] == "forecast"

    def test_historical_length_matches_data(self):
        df = make_monthly_df(12)
        result = forecast_next_periods(df, "month", "revenue", periods=6)
        assert len(result["historical"]) == 12

    def test_forecast_length_matches_periods(self):
        df = make_monthly_df(12)
        result = forecast_next_periods(df, "month", "revenue", periods=6)
        assert len(result["forecast"]) == 6

    def test_forecast_points_have_ci(self):
        df = make_monthly_df(12)
        result = forecast_next_periods(df, "month", "revenue", periods=3)
        for fp in result["forecast"]:
            assert "lower" in fp and "upper" in fp
            assert (
                fp["lower"] <= fp["value"] <= fp["upper"] or fp["lower"] <= fp["upper"]
            )

    def test_periods_clamped_to_max_24(self):
        df = make_monthly_df(12)
        result = forecast_next_periods(df, "month", "revenue", periods=100)
        assert len(result["forecast"]) == 24

    def test_periods_clamped_to_min_1(self):
        df = make_monthly_df(12)
        result = forecast_next_periods(df, "month", "revenue", periods=0)
        assert len(result["forecast"]) == 1

    def test_trend_up_for_increasing_series(self):
        rows = [{"date": f"2024-{m:02d}-01", "val": m * 100} for m in range(1, 13)]
        df = pd.DataFrame(rows)
        result = forecast_next_periods(df, "date", "val", periods=3)
        assert result["trend"] == "up"

    def test_trend_down_for_decreasing_series(self):
        rows = [
            {"date": f"2024-{m:02d}-01", "val": (13 - m) * 100} for m in range(1, 13)
        ]
        df = pd.DataFrame(rows)
        result = forecast_next_periods(df, "date", "val", periods=3)
        assert result["trend"] == "down"

    def test_summary_is_nonempty_string(self):
        df = make_monthly_df(12)
        result = forecast_next_periods(df, "month", "revenue", periods=3)
        assert isinstance(result["summary"], str) and len(result["summary"]) > 20

    def test_raises_on_too_few_rows(self):
        df = pd.DataFrame({"date": ["2024-01-01", "2024-02-01"], "val": [1, 2]})
        with pytest.raises(ValueError):
            forecast_next_periods(df, "date", "val", periods=3)

    def test_daily_data_forecast(self):
        df = make_daily_df(20)
        result = forecast_next_periods(df, "day", "sales", periods=7)
        assert len(result["forecast"]) == 7
        assert result["period_label"] == "day"


# ---------------------------------------------------------------------------
# _fmt_value
# ---------------------------------------------------------------------------


class TestFmtValue:
    def test_millions(self):
        assert "M" in _fmt_value(1_500_000)

    def test_thousands(self):
        assert "K" in _fmt_value(2_500)

    def test_small(self):
        assert "K" not in _fmt_value(999)
        assert "M" not in _fmt_value(999)


# ---------------------------------------------------------------------------
# API endpoint: GET /api/data/{id}/forecast
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forecast_endpoint_returns_forecast(client, tmp_path):
    """Upload monthly data, then call the forecast endpoint."""
    df = make_monthly_df(24)
    csv_bytes = make_csv_bytes(df)

    # Create project
    proj_resp = await client.post("/api/projects", json={"name": "ForecastTest"})
    project_id = proj_resp.json()["id"]

    # Upload dataset
    upload_resp = await client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("monthly.csv", csv_bytes, "text/csv")},
    )
    assert upload_resp.status_code in (200, 201)
    dataset_id = upload_resp.json()["dataset_id"]

    # Call forecast endpoint
    resp = await client.get(f"/api/data/{dataset_id}/forecast?periods=6")
    assert resp.status_code == 200
    body = resp.json()
    assert "forecast" in body
    fc = body["forecast"]
    assert fc["chart_type"] == "forecast"
    assert len(fc["historical"]) == 24
    assert len(fc["forecast"]) == 6


@pytest.mark.asyncio
async def test_forecast_endpoint_with_target_column(client, tmp_path):
    df = make_monthly_df(12)
    df["units"] = range(12, 24)
    csv_bytes = make_csv_bytes(df)

    proj_resp = await client.post("/api/projects", json={"name": "ForecastTarget"})
    project_id = proj_resp.json()["id"]

    upload_resp = await client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("monthly.csv", csv_bytes, "text/csv")},
    )
    dataset_id = upload_resp.json()["dataset_id"]

    resp = await client.get(f"/api/data/{dataset_id}/forecast?target=units&periods=3")
    assert resp.status_code == 200
    fc = resp.json()["forecast"]
    assert fc["value_col"] == "units"


@pytest.mark.asyncio
async def test_forecast_endpoint_invalid_target(client, tmp_path):
    df = make_monthly_df(12)
    csv_bytes = make_csv_bytes(df)

    proj_resp = await client.post("/api/projects", json={"name": "BadTarget"})
    project_id = proj_resp.json()["id"]

    upload_resp = await client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("monthly.csv", csv_bytes, "text/csv")},
    )
    dataset_id = upload_resp.json()["dataset_id"]

    resp = await client.get(f"/api/data/{dataset_id}/forecast?target=nonexistent_col")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_forecast_endpoint_no_time_series(client, tmp_path):
    """Dataset without a date column should return 404."""
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
    csv_bytes = make_csv_bytes(df)

    proj_resp = await client.post("/api/projects", json={"name": "NoTS"})
    project_id = proj_resp.json()["id"]

    upload_resp = await client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("numeric.csv", csv_bytes, "text/csv")},
    )
    dataset_id = upload_resp.json()["dataset_id"]

    resp = await client.get(f"/api/data/{dataset_id}/forecast")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_forecast_endpoint_periods_out_of_range(client, tmp_path):
    df = make_monthly_df(12)
    csv_bytes = make_csv_bytes(df)

    proj_resp = await client.post("/api/projects", json={"name": "PeriodRange"})
    project_id = proj_resp.json()["id"]
    upload_resp = await client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("m.csv", csv_bytes, "text/csv")},
    )
    dataset_id = upload_resp.json()["dataset_id"]

    resp = await client.get(f"/api/data/{dataset_id}/forecast?periods=99")
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_forecast_endpoint_dataset_not_found(client):
    resp = await client.get("/api/data/nonexistent-id/forecast")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Chat intent detection
# ---------------------------------------------------------------------------


class TestForecastChatPatterns:
    """Verify that _FORECAST_PATTERNS matches expected phrases."""

    @pytest.fixture(autouse=True)
    def import_patterns(self):
        from api.chat import _FORECAST_PATTERNS, _detect_forecast_request

        self.pattern = _FORECAST_PATTERNS
        self.detector = _detect_forecast_request

    def test_matches_forecast(self):
        assert self.pattern.search("Can you forecast next quarter?")

    def test_matches_predict_next(self):
        assert self.pattern.search("predict the next 6 months of revenue")

    def test_matches_project_future(self):
        assert self.pattern.search("project future sales")

    def test_matches_next_n_months(self):
        assert self.pattern.search("what about next 3 months?")

    def test_matches_extrapolate(self):
        assert self.pattern.search("extrapolate from the trend")

    def test_no_match_on_normal_query(self):
        assert not self.pattern.search("what is the highest revenue region?")

    def test_detect_extracts_periods_and_unit(self):
        result = self.detector("predict next 6 months")
        assert result["periods"] == 6
        assert result["period_unit"] == "month"

    def test_detect_defaults_to_6_periods(self):
        result = self.detector("forecast the trend")
        assert result["periods"] == 6

    def test_detect_clamps_large_periods(self):
        result = self.detector("predict next 100 weeks")
        assert result["periods"] == 24

    def test_detect_quarter_unit(self):
        result = self.detector("project next 4 quarters")
        assert result["period_unit"] == "quarter"
        assert result["periods"] == 4
