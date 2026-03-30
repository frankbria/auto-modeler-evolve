"""Tests for histogram via chat feature:
- _HISTOGRAM_PATTERNS and _detect_histogram_col in chat.py
- SSE histogram chart emission via send_message
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import (
    _HISTOGRAM_PATTERNS,
    _detect_histogram_col,
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
    proj_resp = client.post("/api/projects/", json={"name": "Histogram Test"})
    project_id = proj_resp.json()["id"]
    resp = client.post(
        "/api/data/upload",
        files={"file": ("sales.csv", _SAMPLE_CSV, "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code in (200, 201), resp.text
    return project_id, resp.json()["dataset_id"]


# ---------------------------------------------------------------------------
# Unit tests — _HISTOGRAM_PATTERNS
# ---------------------------------------------------------------------------


def test_histogram_pattern_basic():
    assert _HISTOGRAM_PATTERNS.search("histogram of revenue")


def test_histogram_pattern_show_histogram():
    assert _HISTOGRAM_PATTERNS.search("show me a histogram")


def test_histogram_pattern_create_histogram():
    assert _HISTOGRAM_PATTERNS.search("create a histogram")


def test_histogram_pattern_frequency_histogram():
    assert _HISTOGRAM_PATTERNS.search("frequency histogram of units")


def test_histogram_pattern_binned_distribution():
    assert _HISTOGRAM_PATTERNS.search("binned distribution of cost")


def test_histogram_pattern_frequency_chart():
    assert _HISTOGRAM_PATTERNS.search("frequency chart of revenue")


def test_histogram_pattern_distribution_chart():
    assert _HISTOGRAM_PATTERNS.search("distribution chart of units")


def test_histogram_pattern_distribution_histogram():
    assert _HISTOGRAM_PATTERNS.search("distribution histogram of sales")


# Non-matches
def test_histogram_no_match_bar_chart():
    assert not _HISTOGRAM_PATTERNS.search("bar chart of revenue by region")


def test_histogram_no_match_scatter():
    assert not _HISTOGRAM_PATTERNS.search("plot revenue vs units")


def test_histogram_no_match_line():
    assert not _HISTOGRAM_PATTERNS.search("line chart of revenue over time")


def test_histogram_no_match_box_plot():
    assert not _HISTOGRAM_PATTERNS.search("box plot of revenue by region")


def test_histogram_no_match_generic_distribution():
    # "distribution of revenue by region" should NOT match (goes to column profile or box plot)
    assert not _HISTOGRAM_PATTERNS.search("distribution of revenue by region")


# ---------------------------------------------------------------------------
# Unit tests — _detect_histogram_col
# ---------------------------------------------------------------------------


def test_detect_histogram_explicit_col():
    assert _detect_histogram_col("histogram of revenue", _DF) == "revenue"


def test_detect_histogram_units_col():
    assert _detect_histogram_col("frequency histogram of units", _DF) == "units"


def test_detect_histogram_cost_col():
    assert _detect_histogram_col("show me a distribution histogram of cost", _DF) == "cost"


def test_detect_histogram_fallback_to_first_numeric():
    """When no numeric col mentioned, returns first numeric column."""
    result = _detect_histogram_col("show me a histogram", _DF)
    assert result in {"revenue", "units", "cost"}


def test_detect_histogram_case_insensitive():
    assert _detect_histogram_col("HISTOGRAM OF REVENUE", _DF) == "revenue"


def test_detect_histogram_no_numeric_returns_none():
    df_all_cat = pd.DataFrame({"region": ["East", "West"], "product": ["A", "B"]})
    assert _detect_histogram_col("histogram of region", df_all_cat) is None


def test_detect_histogram_space_variant():
    """Column name with underscores matched by space variant in message."""
    df = pd.DataFrame({"total_revenue": [100, 200], "cost": [50, 80]})
    result = _detect_histogram_col("histogram of total revenue", df)
    assert result == "total_revenue"


# ---------------------------------------------------------------------------
# Chat SSE integration tests
# ---------------------------------------------------------------------------


def test_histogram_chat_emits_chart_event(client, project_and_dataset):
    """Sending a histogram request should emit a {type:'chart'} SSE event
    with chart_type='histogram'."""
    project_id, dataset_id = project_and_dataset

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Here is the histogram."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": "show me a histogram of revenue", "dataset_id": dataset_id},
        )

    assert response.status_code == 200
    lines = [line for line in response.text.split("\n") if line.startswith("data: ")]
    chart_types = []
    for line in lines:
        try:
            parsed = json.loads(line[6:])
            if parsed.get("type") == "chart":
                chart_types.append(parsed.get("chart", {}).get("chart_type"))
        except json.JSONDecodeError:
            pass

    assert "histogram" in chart_types, f"Expected histogram event. Got: {chart_types}"


def test_histogram_chart_has_bins_and_counts(client, project_and_dataset):
    """The histogram chart should have bin/count data entries."""
    project_id, dataset_id = project_and_dataset

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Histogram done."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": "frequency histogram of revenue", "dataset_id": dataset_id},
        )

    lines = [line for line in response.text.split("\n") if line.startswith("data: ")]
    hist_event = None
    for line in lines:
        try:
            parsed = json.loads(line[6:])
            if parsed.get("type") == "chart" and parsed.get("chart", {}).get("chart_type") == "histogram":
                hist_event = parsed
                break
        except json.JSONDecodeError:
            pass

    assert hist_event is not None, "No histogram chart event found"
    chart = hist_event["chart"]
    assert chart["chart_type"] == "histogram"
    assert "Distribution of" in chart["title"]
    assert len(chart["data"]) > 0
    first_point = chart["data"][0]
    assert "bin" in first_point
    assert "count" in first_point


def test_histogram_no_dataset_no_event(client):
    """Without a dataset, no histogram event should be emitted."""
    proj_resp = client.post("/api/projects/", json={"name": "Empty Hist Test"})
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
            json={"message": "show me a histogram"},
        )

    assert response.status_code == 200
    lines = [line for line in response.text.split("\n") if line.startswith("data: ")]
    chart_types = [
        json.loads(line[6:]).get("chart", {}).get("chart_type")
        for line in lines
        if line.startswith("data: ")
        and "chart" in json.loads(line[6:]).get("type", "")
    ]
    assert "histogram" not in chart_types
