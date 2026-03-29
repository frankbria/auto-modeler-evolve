"""Tests for bar chart chat feature:
- _BAR_CHART_PATTERNS and _detect_bar_chart_request in chat.py
- SSE bar chart emission via send_message
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import (
    _BAR_CHART_PATTERNS,
    _detect_bar_chart_request,
)

# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------

_SAMPLE_CSV = (
    b"region,revenue,units,cost\n"
    b"East,100,10,50\n"
    b"West,200,20,80\n"
    b"East,150,15,60\n"
    b"West,300,30,120\n"
    b"North,250,25,100\n"
)

_DF = pd.DataFrame(
    {
        "region": ["East", "West", "East", "West", "North"],
        "revenue": [100, 200, 150, 300, 250],
        "units": [10, 20, 15, 30, 25],
        "cost": [50, 80, 60, 120, 100],
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
def project_and_dataset(client):
    proj_resp = client.post("/api/projects/", json={"name": "Bar Test"})
    project_id = proj_resp.json()["id"]
    resp = client.post(
        "/api/data/upload",
        files={"file": ("sales.csv", _SAMPLE_CSV, "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code in (200, 201), resp.text
    return project_id, resp.json()["dataset_id"]


# ---------------------------------------------------------------------------
# Unit tests — _BAR_CHART_PATTERNS
# ---------------------------------------------------------------------------


def test_bar_pattern_bar_chart():
    assert _BAR_CHART_PATTERNS.search("bar chart of revenue by region")


def test_bar_pattern_bar_chart_bare():
    assert _BAR_CHART_PATTERNS.search("bar chart please")


def test_bar_pattern_column_chart():
    assert _BAR_CHART_PATTERNS.search("column chart of sales by product")


def test_bar_pattern_vertical_bar():
    assert _BAR_CHART_PATTERNS.search("vertical bar of units by category")


def test_bar_pattern_show_me_bar():
    assert _BAR_CHART_PATTERNS.search("show me a bar chart")


def test_bar_pattern_create_bar():
    assert _BAR_CHART_PATTERNS.search("create a bar chart of revenue")


def test_bar_pattern_bar_chart_of():
    assert _BAR_CHART_PATTERNS.search("bar chart of cost by region")


def test_bar_pattern_column_chart_showing():
    assert _BAR_CHART_PATTERNS.search("column chart showing revenue")


# Non-matches — other intents should NOT trigger bar chart
def test_bar_pattern_no_match_scatter():
    assert not _BAR_CHART_PATTERNS.search("plot revenue vs units")


def test_bar_pattern_no_match_pie():
    assert not _BAR_CHART_PATTERNS.search("pie chart of revenue by region")


def test_bar_pattern_no_match_generic_group():
    assert not _BAR_CHART_PATTERNS.search("show me revenue by region")


def test_bar_pattern_no_match_top_n():
    assert not _BAR_CHART_PATTERNS.search("show top 10 customers by revenue")


# ---------------------------------------------------------------------------
# Unit tests — _detect_bar_chart_request
# ---------------------------------------------------------------------------


def test_detect_bar_explicit_cols():
    result = _detect_bar_chart_request("bar chart of revenue by region", _DF)
    assert result is not None
    assert result["value_col"] == "revenue"
    assert result["group_col"] == "region"


def test_detect_bar_value_col_mentioned():
    result = _detect_bar_chart_request("show me a bar chart of units by region", _DF)
    assert result is not None
    assert result["value_col"] == "units"
    assert result["group_col"] == "region"


def test_detect_bar_fallback_first_cols():
    """When no specific columns mentioned, uses first numeric + first categorical."""
    result = _detect_bar_chart_request("show me a bar chart", _DF)
    assert result is not None
    assert result["value_col"] in {"revenue", "units", "cost"}
    assert result["group_col"] == "region"


def test_detect_bar_agg_mean():
    result = _detect_bar_chart_request("bar chart of average revenue by region", _DF)
    assert result is not None
    assert result["agg"] == "mean"


def test_detect_bar_agg_count():
    result = _detect_bar_chart_request("bar chart count by region", _DF)
    assert result is not None
    assert result["agg"] == "count"


def test_detect_bar_default_agg_sum():
    result = _detect_bar_chart_request("bar chart of revenue by region", _DF)
    assert result is not None
    assert result["agg"] == "sum"


def test_detect_bar_no_numeric_returns_none():
    """Returns None when dataset has no numeric columns."""
    df_all_cat = pd.DataFrame({"region": ["East", "West"], "product": ["A", "B"]})
    result = _detect_bar_chart_request("bar chart of region by product", df_all_cat)
    assert result is None


def test_detect_bar_case_insensitive():
    result = _detect_bar_chart_request("BAR CHART OF REVENUE BY REGION", _DF)
    assert result is not None
    assert result["value_col"] == "revenue"
    assert result["group_col"] == "region"


def test_detect_bar_cost_col():
    result = _detect_bar_chart_request("bar chart of cost by region", _DF)
    assert result is not None
    assert result["value_col"] == "cost"


# ---------------------------------------------------------------------------
# Chat SSE integration tests
# ---------------------------------------------------------------------------


def test_bar_chat_emits_chart_event(client, project_and_dataset):
    """Sending a bar chart request via chat should emit a {type:'chart'} SSE event
    with a bar chart spec."""
    project_id, dataset_id = project_and_dataset

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Here is the bar chart."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={
                "message": "show me a bar chart of revenue by region",
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
    assert "bar" in chart_types, f"Expected bar chart. Got chart_types: {chart_types}"


def test_bar_chart_data_has_groups(client, project_and_dataset):
    """The emitted bar chart should have data entries with label + value keys."""
    project_id, dataset_id = project_and_dataset

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Bar chart done."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={
                "message": "bar chart of revenue by region",
                "dataset_id": dataset_id,
            },
        )

    lines = [line for line in response.text.split("\n") if line.startswith("data: ")]
    bar_event = None
    for line in lines:
        try:
            parsed = json.loads(line[6:])
            if (
                parsed.get("type") == "chart"
                and parsed.get("chart", {}).get("chart_type") == "bar"
            ):
                bar_event = parsed
                break
        except json.JSONDecodeError:
            pass

    assert bar_event is not None, "No bar chart event found"
    chart = bar_event["chart"]
    assert chart["chart_type"] == "bar"
    assert chart["title"] == "Sum of revenue by region"
    assert len(chart["data"]) > 0
    # Each data point should have label (x) and value (y)
    first_point = chart["data"][0]
    assert "label" in first_point
    assert "value" in first_point
