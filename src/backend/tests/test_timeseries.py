"""test_timeseries.py

Tests for time-series decomposition:
  - detect_time_columns helper (analyzer.py)
  - build_timeseries_chart (chart_builder.py)
  - GET /api/data/{dataset_id}/timeseries endpoint (api/data.py)
"""

from __future__ import annotations


import pandas as pd
import pytest
from httpx import AsyncClient, ASGITransport
from sqlmodel import SQLModel, create_engine

import db as db_module


# ===========================================================================
# detect_time_columns (unit tests)
# ===========================================================================


class TestDetectTimeColumns:
    def test_detects_iso_date_column(self):
        from core.analyzer import detect_time_columns

        df = pd.DataFrame(
            {
                "date": [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-06",
                    "2024-01-07",
                    "2024-01-08",
                    "2024-01-09",
                    "2024-01-10",
                ],
                "value": range(10),
            }
        )
        result = detect_time_columns(df)
        assert "date" in result

    def test_does_not_detect_pure_numeric_column(self):
        from core.analyzer import detect_time_columns

        df = pd.DataFrame({"revenue": [100, 200, 300, 400, 500]})
        result = detect_time_columns(df)
        assert "revenue" not in result

    def test_does_not_detect_random_string_column(self):
        from core.analyzer import detect_time_columns

        df = pd.DataFrame(
            {"product": ["Widget A", "Widget B", "Widget C", "Widget D", "Widget E"]}
        )
        result = detect_time_columns(df)
        assert "product" not in result

    def test_returns_empty_for_no_date_columns(self):
        from core.analyzer import detect_time_columns

        df = pd.DataFrame({"revenue": [100, 200], "units": [10, 20]})
        result = detect_time_columns(df)
        assert result == []

    def test_detects_slash_format_date(self):
        from core.analyzer import detect_time_columns

        df = pd.DataFrame(
            {
                "date": [
                    "01/01/2024",
                    "01/02/2024",
                    "01/03/2024",
                    "01/04/2024",
                    "01/05/2024",
                    "01/06/2024",
                    "01/07/2024",
                    "01/08/2024",
                    "01/09/2024",
                    "01/10/2024",
                ],
                "value": range(10),
            }
        )
        result = detect_time_columns(df)
        assert "date" in result

    def test_detects_pandas_datetime_dtype(self):
        from core.analyzer import detect_time_columns

        df = pd.DataFrame(
            {
                "ts": pd.date_range("2024-01-01", periods=5),
                "value": range(5),
            }
        )
        result = detect_time_columns(df)
        assert "ts" in result

    def test_all_null_column_not_detected(self):
        from core.analyzer import detect_time_columns

        df = pd.DataFrame({"date": [None, None, None], "value": [1, 2, 3]})
        result = detect_time_columns(df)
        assert "date" not in result


# ===========================================================================
# build_timeseries_chart (unit tests)
# ===========================================================================


class TestBuildTimeseriesChart:
    def test_returns_line_chart(self):
        from core.chart_builder import build_timeseries_chart

        dates = [
            "2024-01",
            "2024-02",
            "2024-03",
            "2024-04",
            "2024-05",
            "2024-06",
            "2024-07",
            "2024-08",
            "2024-09",
            "2024-10",
        ]
        values = [100.0, 120.0, 110.0, 130.0, 140.0, 125.0, 135.0, 150.0, 145.0, 160.0]
        result = build_timeseries_chart(dates, values, "revenue")
        assert result["chart_type"] == "line"
        assert result["x_key"] == "x"
        assert "revenue" in result["y_keys"]
        assert len(result["data"]) == 10

    def test_includes_rolling_average_series(self):
        from core.chart_builder import build_timeseries_chart

        dates = [f"2024-{i:02d}" for i in range(1, 13)]
        values = [100.0 + i * 5 for i in range(12)]
        result = build_timeseries_chart(dates, values, "sales", window=3)
        avg_key = next((k for k in result["y_keys"] if "avg" in k.lower()), None)
        assert avg_key is not None

    def test_includes_trend_series(self):
        from core.chart_builder import build_timeseries_chart

        dates = [f"2024-{i:02d}" for i in range(1, 13)]
        values = [100.0 + i * 10 for i in range(12)]
        result = build_timeseries_chart(dates, values, "revenue", window=3)
        assert "Trend" in result["y_keys"]

    def test_empty_values_returns_empty_chart(self):
        from core.chart_builder import build_timeseries_chart

        result = build_timeseries_chart([], [], "revenue")
        assert result["chart_type"] == "line"
        assert result["data"] == []

    def test_short_series_adjusts_window(self):
        """Window auto-adjusts for very short series."""
        from core.chart_builder import build_timeseries_chart

        # 4 points is less than 6 → effective_window=1
        dates = ["2024-01", "2024-02", "2024-03", "2024-04"]
        values = [100.0, 110.0, 105.0, 115.0]
        result = build_timeseries_chart(dates, values, "revenue", window=7)
        assert result["chart_type"] == "line"
        assert len(result["data"]) == 4

    def test_with_none_values(self):
        """None (missing) values in the series should not crash."""
        from core.chart_builder import build_timeseries_chart

        dates = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"]
        values = [100.0, None, 110.0, None, 120.0]
        result = build_timeseries_chart(dates, values, "revenue")
        assert result["chart_type"] == "line"
        assert len(result["data"]) == 5

    def test_trend_line_increases_for_monotonic_data(self):
        """Trend should be increasing for monotonically increasing data."""
        from core.chart_builder import build_timeseries_chart

        dates = [f"2024-{i:02d}" for i in range(1, 13)]
        values = [float(i * 10) for i in range(12)]
        result = build_timeseries_chart(dates, values, "revenue", window=3)
        trend_values = [
            pt["Trend"] for pt in result["data"] if pt.get("Trend") is not None
        ]
        assert len(trend_values) >= 2
        # Trend should be non-decreasing
        assert trend_values[-1] > trend_values[0]


# ===========================================================================
# GET /api/data/{dataset_id}/timeseries endpoint
# ===========================================================================


@pytest.fixture
async def client(tmp_path):
    test_db = str(tmp_path / "timeseries_test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.project  # noqa
    import models.dataset  # noqa
    import models.conversation  # noqa
    import models.feature_set  # noqa
    import models.model_run  # noqa
    import models.deployment  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    data_module.UPLOAD_DIR = upload_dir

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
async def project_id(client):
    resp = await client.post("/api/projects", json={"name": "TS Test"})
    return resp.json()["id"]


TIMESERIES_CSV = b"""date,product,revenue,units
2024-01-01,Widget A,1200,10
2024-01-02,Widget B,850,8
2024-01-03,Widget A,2100,18
2024-01-04,Widget C,450,4
2024-01-05,Widget B,1650,15
2024-01-06,Widget A,1800,14
2024-01-07,Widget C,500,5
2024-01-08,Widget B,900,9
2024-01-09,Widget A,2200,20
2024-01-10,Widget C,600,6
"""

NO_DATE_CSV = b"""product,revenue,units
Widget A,1200,10
Widget B,850,8
Widget C,450,4
"""


class TestTimeseriesEndpoint:
    async def _upload(self, client, project_id, csv_bytes, filename="data.csv"):
        resp = await client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": (filename, csv_bytes, "text/csv")},
        )
        assert resp.status_code == 201, resp.text
        return resp.json()["dataset_id"]

    async def test_returns_404_for_nonexistent_dataset(self, client):
        resp = await client.get("/api/data/nonexistent-id-999/timeseries")
        assert resp.status_code == 404

    async def test_detects_date_column_and_returns_chart(self, client, project_id):
        dataset_id = await self._upload(client, project_id, TIMESERIES_CSV)
        resp = await client.get(f"/api/data/{dataset_id}/timeseries")
        assert resp.status_code == 200
        body = resp.json()
        assert "date" in body["date_columns"]
        assert body["chart_spec"] is not None
        assert body["chart_spec"]["chart_type"] == "line"

    async def test_returns_null_chart_when_no_date_column(self, client, project_id):
        dataset_id = await self._upload(client, project_id, NO_DATE_CSV, "no_date.csv")
        resp = await client.get(f"/api/data/{dataset_id}/timeseries")
        assert resp.status_code == 200
        body = resp.json()
        assert body["chart_spec"] is None
        assert "message" in body

    async def test_returns_value_column_options(self, client, project_id):
        dataset_id = await self._upload(client, project_id, TIMESERIES_CSV)
        resp = await client.get(f"/api/data/{dataset_id}/timeseries")
        body = resp.json()
        assert "revenue" in body["value_columns"] or "units" in body["value_columns"]

    async def test_accepts_explicit_value_column(self, client, project_id):
        dataset_id = await self._upload(client, project_id, TIMESERIES_CSV)
        resp = await client.get(f"/api/data/{dataset_id}/timeseries?value_column=units")
        assert resp.status_code == 200
        body = resp.json()
        assert body["value_column"] == "units"
        assert body["chart_spec"] is not None

    async def test_chart_has_three_series(self, client, project_id):
        dataset_id = await self._upload(client, project_id, TIMESERIES_CSV)
        resp = await client.get(f"/api/data/{dataset_id}/timeseries")
        body = resp.json()
        chart = body["chart_spec"]
        assert len(chart["y_keys"]) == 3  # original + rolling avg + trend
        assert "Trend" in chart["y_keys"]
        assert "revenue" in chart["y_keys"]

    async def test_custom_window_parameter(self, client, project_id):
        dataset_id = await self._upload(client, project_id, TIMESERIES_CSV)
        resp = await client.get(f"/api/data/{dataset_id}/timeseries?window=3")
        assert resp.status_code == 200
        body = resp.json()
        assert body["chart_spec"] is not None
        # rolling avg label should reflect the window (auto-adjusted to data length)
        avg_key = next(
            (k for k in body["chart_spec"]["y_keys"] if "avg" in k.lower()), None
        )
        assert avg_key is not None
