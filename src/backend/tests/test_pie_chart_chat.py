"""Tests for pie chart chat feature:
- _PIE_CHART_PATTERNS and _detect_pie_chart_request in chat.py
- SSE pie chart emission via send_message
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import (
    _PIE_CHART_PATTERNS,
    _detect_pie_chart_request,
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
    proj_resp = client.post("/api/projects/", json={"name": "Pie Test"})
    project_id = proj_resp.json()["id"]
    resp = client.post(
        "/api/data/upload",
        files={"file": ("sales.csv", _SAMPLE_CSV, "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code in (200, 201), resp.text
    return project_id, resp.json()["dataset_id"]


# ---------------------------------------------------------------------------
# Unit tests — _PIE_CHART_PATTERNS
# ---------------------------------------------------------------------------


def test_pie_pattern_pie_chart():
    assert _PIE_CHART_PATTERNS.search("show me a pie chart of revenue by region")


def test_pie_pattern_pie_chart_bare():
    assert _PIE_CHART_PATTERNS.search("pie chart please")


def test_pie_pattern_donut_chart():
    assert _PIE_CHART_PATTERNS.search("donut chart of sales by product")


def test_pie_pattern_doughnut_chart():
    assert _PIE_CHART_PATTERNS.search("doughnut chart of costs")


def test_pie_pattern_show_me_a_pie():
    assert _PIE_CHART_PATTERNS.search("show me a pie of revenue")


def test_pie_pattern_composition_of():
    assert _PIE_CHART_PATTERNS.search("composition of revenue by region")


def test_pie_pattern_share_by():
    assert _PIE_CHART_PATTERNS.search("share of sales by product")


def test_pie_pattern_proportion_chart():
    assert _PIE_CHART_PATTERNS.search("proportion chart of units by region")


def test_pie_pattern_breakdown_chart():
    assert _PIE_CHART_PATTERNS.search("breakdown chart of revenue")


# Non-matches — group stats and other intents should NOT trigger pie chart
def test_pie_pattern_no_match_group_stats():
    assert not _PIE_CHART_PATTERNS.search("show me revenue by region")


def test_pie_pattern_no_match_scatter():
    assert not _PIE_CHART_PATTERNS.search("plot revenue vs units")


def test_pie_pattern_no_match_top_n():
    assert not _PIE_CHART_PATTERNS.search("show top 10 customers by revenue")


def test_pie_pattern_no_match_generic():
    assert not _PIE_CHART_PATTERNS.search("what is the average revenue")


# ---------------------------------------------------------------------------
# Unit tests — _detect_pie_chart_request
# ---------------------------------------------------------------------------


def test_detect_pie_explicit_cols():
    result = _detect_pie_chart_request("pie chart of revenue by region", _DF)
    assert result is not None
    assert result["value_col"] == "revenue"
    assert result["slice_col"] == "region"


def test_detect_pie_numeric_col_mentioned():
    result = _detect_pie_chart_request("show me units by region as a pie", _DF)
    assert result is not None
    assert result["value_col"] == "units"
    assert result["slice_col"] == "region"


def test_detect_pie_fallback_first_cols():
    """When no specific columns mentioned, uses first numeric + first categorical."""
    result = _detect_pie_chart_request("show me a pie chart", _DF)
    assert result is not None
    assert result["value_col"] in {"revenue", "units", "cost"}
    assert result["slice_col"] == "region"


def test_detect_pie_no_categorical_returns_none():
    """Returns None when dataset has no categorical columns with 2-30 unique values."""
    df_all_numeric = pd.DataFrame(
        {"revenue": [100, 200], "cost": [50, 80], "units": [10, 20]}
    )
    result = _detect_pie_chart_request("pie chart", df_all_numeric)
    assert result is None


def test_detect_pie_no_numeric_returns_none():
    """Returns None when dataset has no numeric columns."""
    df_all_cat = pd.DataFrame({"region": ["East", "West"], "product": ["A", "B"]})
    result = _detect_pie_chart_request("pie chart of region by product", df_all_cat)
    assert result is None


def test_detect_pie_case_insensitive():
    result = _detect_pie_chart_request("PIE CHART OF REVENUE BY REGION", _DF)
    assert result is not None
    assert result["value_col"] == "revenue"
    assert result["slice_col"] == "region"


def test_detect_pie_cost_col():
    result = _detect_pie_chart_request("pie chart of cost by region", _DF)
    assert result is not None
    assert result["value_col"] == "cost"
    assert result["slice_col"] == "region"


# ---------------------------------------------------------------------------
# Chat SSE integration tests
# ---------------------------------------------------------------------------


def test_pie_chat_emits_chart_event(client, project_and_dataset):
    """Sending a pie chart request via chat should emit a {type:'chart'} SSE event
    with a pie chart spec."""
    project_id, dataset_id = project_and_dataset

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Here is the pie chart."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={
                "message": "show me a pie chart of revenue by region",
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
    assert "pie" in chart_types, f"Expected pie chart. Got chart_types: {chart_types}"


def test_pie_chart_data_has_slices(client, project_and_dataset):
    """The emitted pie chart should have data entries with name + value keys."""
    project_id, dataset_id = project_and_dataset

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Pie chart done."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={
                "message": "pie chart of revenue by region",
                "dataset_id": dataset_id,
            },
        )

    lines = [line for line in response.text.split("\n") if line.startswith("data: ")]
    pie_event = None
    for line in lines:
        try:
            parsed = json.loads(line[6:])
            if (
                parsed.get("type") == "chart"
                and parsed.get("chart", {}).get("chart_type") == "pie"
            ):
                pie_event = parsed
                break
        except json.JSONDecodeError:
            pass

    assert pie_event is not None, "No pie chart event found"
    chart = pie_event["chart"]
    assert len(chart["data"]) >= 2, "Expected at least 2 slice entries"
    first_entry = chart["data"][0]
    assert "name" in first_entry
    assert "value" in first_entry


def test_pie_chart_no_dataset_no_crash(client):
    """Chat should not crash if no dataset is attached."""
    proj_resp = client.post("/api/projects/", json={"name": "No Data"})
    project_id = proj_resp.json()["id"]

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["No data available."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": "pie chart of revenue by region"},
        )

    assert response.status_code == 200  # Must not crash
