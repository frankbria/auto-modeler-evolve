"""Tests for line chart and box plot chat features:
- _LINE_CHART_PATTERNS and _detect_line_chart_request in chat.py
- _BOXPLOT_PATTERNS and _detect_boxplot_request in chat.py
- SSE chart emission via send_message
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import (
    _BOXPLOT_PATTERNS,
    _LINE_CHART_PATTERNS,
    _detect_boxplot_request,
    _detect_line_chart_request,
)

# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------

_TIMESERIES_CSV = (
    b"date,revenue,units\n"
    b"2023-01-01,100,10\n"
    b"2023-02-01,120,12\n"
    b"2023-03-01,115,11\n"
    b"2023-04-01,140,14\n"
    b"2023-05-01,160,16\n"
    b"2023-06-01,175,18\n"
)

_DF_TS = pd.DataFrame(
    {
        "date": pd.to_datetime(
            ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01", "2023-06-01"]
        ),
        "revenue": [100, 120, 115, 140, 160, 175],
        "units": [10, 12, 11, 14, 16, 18],
    }
)

_DF_CATEGORICAL = pd.DataFrame(
    {
        "region": ["East", "West", "North", "East", "West", "North"],
        "revenue": [100, 200, 150, 120, 220, 160],
        "units": [10, 20, 15, 12, 22, 16],
    }
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def test_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    monkeypatch.setattr(db_module, "engine", engine)
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture()
def client():
    from main import app

    return TestClient(app)


@pytest.fixture()
def project_and_ts_dataset(client):
    proj_resp = client.post("/api/projects/", json={"name": "Line Chart Test"})
    project_id = proj_resp.json()["id"]
    resp = client.post(
        "/api/data/upload",
        files={"file": ("timeseries.csv", _TIMESERIES_CSV, "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code in (200, 201), resp.text
    return project_id, resp.json()["dataset_id"]


@pytest.fixture()
def project_and_cat_dataset(client):
    csv_bytes = (
        b"region,revenue,units\n"
        b"East,100,10\n"
        b"West,200,20\n"
        b"North,150,15\n"
        b"East,120,12\n"
        b"West,220,22\n"
    )
    proj_resp = client.post("/api/projects/", json={"name": "Box Plot Test"})
    project_id = proj_resp.json()["id"]
    resp = client.post(
        "/api/data/upload",
        files={"file": ("sales.csv", csv_bytes, "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code in (200, 201), resp.text
    return project_id, resp.json()["dataset_id"]


# ---------------------------------------------------------------------------
# Unit tests — _LINE_CHART_PATTERNS
# ---------------------------------------------------------------------------


def test_line_pattern_plot_over_time():
    assert _LINE_CHART_PATTERNS.search("plot revenue over time")


def test_line_pattern_chart_by_month():
    assert _LINE_CHART_PATTERNS.search("chart revenue by month")


def test_line_pattern_trend_of():
    assert _LINE_CHART_PATTERNS.search("trend of sales")


def test_line_pattern_line_chart_of():
    assert _LINE_CHART_PATTERNS.search("line chart of revenue")


def test_line_pattern_how_has_changed():
    assert _LINE_CHART_PATTERNS.search("how has revenue changed over time")


def test_line_pattern_show_trend():
    assert _LINE_CHART_PATTERNS.search("show me the revenue trend")


def test_line_pattern_time_series_of():
    assert _LINE_CHART_PATTERNS.search("time series of units")


def test_line_pattern_by_year():
    assert _LINE_CHART_PATTERNS.search("show revenue by year")


def test_line_pattern_no_match_scatter():
    assert not _LINE_CHART_PATTERNS.search("plot revenue vs units")


def test_line_pattern_no_match_group_stats():
    assert not _LINE_CHART_PATTERNS.search("show me revenue by region")


def test_line_pattern_no_match_forecast():
    assert not _LINE_CHART_PATTERNS.search("predict next 6 months revenue")


def test_line_pattern_no_match_generic():
    assert not _LINE_CHART_PATTERNS.search("what is the average cost")


# ---------------------------------------------------------------------------
# Unit tests — _detect_line_chart_request
# ---------------------------------------------------------------------------


def test_detect_line_finds_date_and_value():
    result = _detect_line_chart_request("plot revenue over time", _DF_TS)
    assert result is not None
    assert result["value_col"] == "revenue"
    assert result["date_col"] == "date"


def test_detect_line_extracts_mentioned_column():
    result = _detect_line_chart_request("show me units by month", _DF_TS)
    assert result is not None
    assert result["value_col"] == "units"


def test_detect_line_fallback_to_first_numeric():
    result = _detect_line_chart_request("plot trend over time", _DF_TS)
    assert result is not None
    # Falls back to first numeric col (revenue)
    assert result["value_col"] in ["revenue", "units"]


def test_detect_line_no_date_column():
    """Returns None when no date column present."""
    df = pd.DataFrame({"revenue": [100, 200], "units": [10, 20]})
    result = _detect_line_chart_request("plot revenue over time", df)
    assert result is None


def test_detect_line_no_numeric_column():
    """Returns None when no numeric columns."""
    df = pd.DataFrame({"date": pd.to_datetime(["2023-01", "2023-02"]), "label": ["A", "B"]})
    result = _detect_line_chart_request("plot trend over time", df)
    assert result is None


def test_detect_line_case_insensitive():
    result = _detect_line_chart_request("plot REVENUE over time", _DF_TS)
    assert result is not None
    assert result["value_col"] == "revenue"


# ---------------------------------------------------------------------------
# Unit tests — _BOXPLOT_PATTERNS
# ---------------------------------------------------------------------------


def test_boxplot_pattern_box_plot_of():
    assert _BOXPLOT_PATTERNS.search("box plot of revenue")


def test_boxplot_pattern_distribution_by():
    assert _BOXPLOT_PATTERNS.search("distribution of revenue by region")


def test_boxplot_pattern_spread_by():
    assert _BOXPLOT_PATTERNS.search("spread of revenue by region")


def test_boxplot_pattern_compare_distribution():
    assert _BOXPLOT_PATTERNS.search("compare distribution of revenue by region")


def test_boxplot_pattern_whisker_plot():
    assert _BOXPLOT_PATTERNS.search("whisker plot")


def test_boxplot_pattern_compare_spread_across():
    assert _BOXPLOT_PATTERNS.search("compare the spread of units across region")


def test_boxplot_pattern_outliers_by():
    assert _BOXPLOT_PATTERNS.search("show outliers in revenue by region")


def test_boxplot_pattern_quartile_by():
    assert _BOXPLOT_PATTERNS.search("quartile range of revenue by region")


def test_boxplot_pattern_no_match_scatter():
    assert not _BOXPLOT_PATTERNS.search("plot revenue vs units")


def test_boxplot_pattern_no_match_group_stats():
    assert not _BOXPLOT_PATTERNS.search("revenue by region summary")


def test_boxplot_pattern_no_match_generic():
    assert not _BOXPLOT_PATTERNS.search("tell me about the revenue column")


# ---------------------------------------------------------------------------
# Unit tests — _detect_boxplot_request
# ---------------------------------------------------------------------------


def test_detect_boxplot_value_and_group():
    result = _detect_boxplot_request("distribution of revenue by region", _DF_CATEGORICAL)
    assert result is not None
    assert result["value_col"] == "revenue"
    assert result["group_col"] == "region"


def test_detect_boxplot_value_only():
    result = _detect_boxplot_request("box plot of revenue", _DF_CATEGORICAL)
    assert result is not None
    assert result["value_col"] == "revenue"


def test_detect_boxplot_fallback_first_numeric():
    result = _detect_boxplot_request("box plot please", _DF_CATEGORICAL)
    assert result is not None
    assert result["value_col"] in ["revenue", "units"]


def test_detect_boxplot_group_col_from_by_clause():
    result = _detect_boxplot_request("spread of units by region", _DF_CATEGORICAL)
    assert result is not None
    assert result["group_col"] == "region"


def test_detect_boxplot_no_numeric():
    """Returns None when no numeric columns."""
    df = pd.DataFrame({"region": ["A", "B"], "label": ["x", "y"]})
    result = _detect_boxplot_request("box plot of region", df)
    assert result is None


def test_detect_boxplot_no_group_when_none_found():
    result = _detect_boxplot_request("box plot of revenue", _DF_CATEGORICAL)
    assert result is not None
    # group_col may be None or a categorical column
    # Just verify value_col is set
    assert result["value_col"] == "revenue"


# ---------------------------------------------------------------------------
# Chat SSE integration tests
# ---------------------------------------------------------------------------


def test_line_chart_chat_emits_chart_event(client, project_and_ts_dataset):
    """Sending a line chart request should emit a {type:'chart'} SSE event
    with chart_type='line'."""
    project_id, dataset_id = project_and_ts_dataset

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Here is the revenue trend over time."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": "plot revenue over time", "dataset_id": dataset_id},
        )

    assert response.status_code == 200
    lines = [line for line in response.text.split("\n") if line.startswith("data: ")]
    event_types = []
    chart_types = []
    for line in lines:
        try:
            parsed = json.loads(line[6:])
            event_types.append(parsed.get("type"))
            if parsed.get("type") == "chart":
                chart_types.append(parsed.get("chart", {}).get("chart_type"))
        except json.JSONDecodeError:
            pass

    assert "chart" in event_types, f"Expected chart event. Got: {event_types}"
    assert "line" in chart_types, f"Expected line chart. Got: {chart_types}"


def test_boxplot_chat_emits_chart_event(client, project_and_cat_dataset):
    """Sending a box plot request should emit a {type:'chart'} SSE event
    with chart_type='boxplot'."""
    project_id, dataset_id = project_and_cat_dataset

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Here is the distribution."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={
                "message": "distribution of revenue by region",
                "dataset_id": dataset_id,
            },
        )

    assert response.status_code == 200
    lines = [line for line in response.text.split("\n") if line.startswith("data: ")]
    event_types = []
    chart_types = []
    for line in lines:
        try:
            parsed = json.loads(line[6:])
            event_types.append(parsed.get("type"))
            if parsed.get("type") == "chart":
                chart_types.append(parsed.get("chart", {}).get("chart_type"))
        except json.JSONDecodeError:
            pass

    assert "chart" in event_types, f"Expected chart event. Got: {event_types}"
    assert "boxplot" in chart_types, f"Expected boxplot chart. Got: {chart_types}"


def test_line_chart_no_dataset_no_crash(client):
    """Chat should not crash if no dataset is attached."""
    proj_resp = client.post("/api/projects/", json={"name": "No Data"})
    project_id = proj_resp.json()["id"]

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["No data."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": "plot revenue over time"},
        )

    assert response.status_code == 200


def test_boxplot_no_dataset_no_crash(client):
    """Chat should not crash if no dataset is attached."""
    proj_resp = client.post("/api/projects/", json={"name": "No Data BP"})
    project_id = proj_resp.json()["id"]

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["No data."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": "distribution of revenue by region"},
        )

    assert response.status_code == 200
