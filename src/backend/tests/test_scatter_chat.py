"""Tests for scatter plot chat feature:
- _SCATTER_PATTERNS and _detect_scatter_request in chat.py
- SSE scatter chart emission via send_message
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import (
    _SCATTER_PATTERNS,
    _detect_scatter_request,
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
    proj_resp = client.post("/api/projects/", json={"name": "Scatter Test"})
    project_id = proj_resp.json()["id"]
    resp = client.post(
        "/api/data/upload",
        files={"file": ("sales.csv", _SAMPLE_CSV, "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code in (200, 201), resp.text
    return project_id, resp.json()["dataset_id"]


# ---------------------------------------------------------------------------
# Unit tests — _SCATTER_PATTERNS
# ---------------------------------------------------------------------------


def test_scatter_pattern_plot_vs():
    assert _SCATTER_PATTERNS.search("plot revenue vs units")


def test_scatter_pattern_chart_versus():
    assert _SCATTER_PATTERNS.search("chart revenue versus cost")


def test_scatter_pattern_scatter_against():
    assert _SCATTER_PATTERNS.search("scatter revenue against units")


def test_scatter_pattern_scatter_plot():
    assert _SCATTER_PATTERNS.search("scatter plot please")


def test_scatter_pattern_show_relationship():
    assert _SCATTER_PATTERNS.search("show me the relationship between revenue and cost")


def test_scatter_pattern_visualize_relationship():
    assert _SCATTER_PATTERNS.search("visualize the relationship between units and cost")


def test_scatter_pattern_relationship_noun():
    assert _SCATTER_PATTERNS.search("relationship between revenue and units")


def test_scatter_pattern_how_does_relate():
    assert _SCATTER_PATTERNS.search("how does revenue relate to cost")


def test_scatter_pattern_no_match_group_stats():
    assert not _SCATTER_PATTERNS.search("show me revenue by region")


def test_scatter_pattern_no_match_top_n():
    assert not _SCATTER_PATTERNS.search("show me top 10 revenue")


def test_scatter_pattern_no_match_generic():
    assert not _SCATTER_PATTERNS.search("what is the average cost")


def test_scatter_pattern_no_match_filter():
    assert not _SCATTER_PATTERNS.search("filter to rows where region is East")


# ---------------------------------------------------------------------------
# Unit tests — _detect_scatter_request
# ---------------------------------------------------------------------------


def test_detect_scatter_vs_separator():
    result = _detect_scatter_request("plot revenue vs units", _DF)
    assert result is not None
    assert result["x_col"] == "revenue"
    assert result["y_col"] == "units"


def test_detect_scatter_versus_separator():
    result = _detect_scatter_request("revenue versus cost", _DF)
    assert result is not None
    assert result["x_col"] == "revenue"
    assert result["y_col"] == "cost"


def test_detect_scatter_against_separator():
    result = _detect_scatter_request("scatter revenue against cost", _DF)
    assert result is not None
    assert result["x_col"] == "revenue"
    assert result["y_col"] == "cost"


def test_detect_scatter_between_and():
    result = _detect_scatter_request(
        "show me the relationship between revenue and units", _DF
    )
    assert result is not None
    assert set([result["x_col"], result["y_col"]]) == {"revenue", "units"}


def test_detect_scatter_fallback_two_mentioned():
    """If no separator found, uses first two numeric columns mentioned."""
    result = _detect_scatter_request("tell me about revenue and cost", _DF)
    assert result is not None
    # Both revenue and cost are numeric
    assert result["x_col"] in {"revenue", "cost"}
    assert result["y_col"] in {"revenue", "cost"}
    assert result["x_col"] != result["y_col"]


def test_detect_scatter_case_insensitive():
    result = _detect_scatter_request("plot REVENUE vs UNITS", _DF)
    assert result is not None
    assert result["x_col"] == "revenue"
    assert result["y_col"] == "units"


def test_detect_scatter_no_numeric_columns():
    """Returns None when < 2 numeric columns."""
    df_cat = pd.DataFrame({"region": ["East", "West"], "label": ["A", "B"]})
    result = _detect_scatter_request("plot region vs label", df_cat)
    assert result is None


def test_detect_scatter_one_numeric_column():
    """Returns None when only 1 numeric column."""
    df = pd.DataFrame({"region": ["East", "West"], "revenue": [100, 200]})
    result = _detect_scatter_request("plot revenue vs region", df)
    assert result is None


def test_detect_scatter_no_columns_in_message():
    """Returns None if no column names found in message."""
    result = _detect_scatter_request("show me some chart please", _DF)
    assert result is None


def test_detect_scatter_same_column_not_returned():
    """Does not return same column for both x and y."""
    result = _detect_scatter_request("plot revenue vs revenue", _DF)
    # Should be None (can't scatter x=revenue, y=revenue) or pick different cols
    if result is not None:
        assert result["x_col"] != result["y_col"]


# ---------------------------------------------------------------------------
# Chat SSE integration test
# ---------------------------------------------------------------------------


def test_scatter_chat_emits_chart_event(client, project_and_dataset):
    """Sending a scatter request via chat should emit a {type:'chart'} SSE event
    with a scatter chart spec."""
    project_id, dataset_id = project_and_dataset

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Here is the scatter plot."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": "plot revenue vs units", "dataset_id": dataset_id},
        )

    assert response.status_code == 200
    lines = [l for l in response.text.split("\n") if l.startswith("data: ")]
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
    assert "scatter" in chart_types, (
        f"Expected scatter chart. Got chart_types: {chart_types}"
    )


def test_scatter_chat_no_dataset_no_crash(client):
    """Chat should not crash if no dataset is attached (just returns text)."""
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
            json={"message": "plot revenue vs units"},
        )

    assert response.status_code == 200  # Must not crash
