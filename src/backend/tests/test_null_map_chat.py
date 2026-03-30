"""Tests for missing values overview (null map) via chat feature:
- _NULL_MAP_PATTERNS in chat.py
- SSE null_map event emission via send_message
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import _NULL_MAP_PATTERNS

# ---------------------------------------------------------------------------
# Shared sample data — include some missing values
# ---------------------------------------------------------------------------

_SAMPLE_CSV_WITH_NULLS = (
    b"region,revenue,units,notes\n"
    b"East,100,10,\n"
    b"West,,20,important\n"
    b"East,150,15,\n"
    b"West,300,,note\n"
    b"North,250,25,\n"
)

_SAMPLE_CSV_COMPLETE = b"region,revenue,units\nEast,100,10\nWest,200,20\nNorth,250,25\n"


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
def project_with_nulls(client):
    proj_resp = client.post("/api/projects/", json={"name": "Null Map Test"})
    project_id = proj_resp.json()["id"]
    resp = client.post(
        "/api/data/upload",
        files={"file": ("nulls.csv", _SAMPLE_CSV_WITH_NULLS, "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code in (200, 201), resp.text
    return project_id, resp.json()["dataset_id"]


@pytest.fixture()
def project_complete(client):
    proj_resp = client.post("/api/projects/", json={"name": "Complete Test"})
    project_id = proj_resp.json()["id"]
    resp = client.post(
        "/api/data/upload",
        files={"file": ("complete.csv", _SAMPLE_CSV_COMPLETE, "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code in (200, 201), resp.text
    return project_id, resp.json()["dataset_id"]


# ---------------------------------------------------------------------------
# Unit tests — _NULL_MAP_PATTERNS
# ---------------------------------------------------------------------------


def test_null_pattern_show_missing_values():
    assert _NULL_MAP_PATTERNS.search("show me the missing values")


def test_null_pattern_which_columns_have_missing():
    assert _NULL_MAP_PATTERNS.search("which columns have missing data?")


def test_null_pattern_null_values_overview():
    assert _NULL_MAP_PATTERNS.search("null values overview")


def test_null_pattern_missing_data_summary():
    assert _NULL_MAP_PATTERNS.search("missing data summary")


def test_null_pattern_data_completeness_overview():
    assert _NULL_MAP_PATTERNS.search("data completeness overview")


def test_null_pattern_data_completeness_by_column():
    assert _NULL_MAP_PATTERNS.search("data completeness by column")


def test_null_pattern_how_many_missing():
    assert _NULL_MAP_PATTERNS.search("how many missing values do I have?")


def test_null_pattern_missing_count_per_column():
    assert _NULL_MAP_PATTERNS.search("missing count per column")


def test_null_pattern_where_is_missing_data():
    assert _NULL_MAP_PATTERNS.search("where is my missing data?")


def test_null_pattern_display_missing_fields():
    assert _NULL_MAP_PATTERNS.search("display missing fields")


def test_null_pattern_null_rate_by_column():
    assert _NULL_MAP_PATTERNS.search("null rate by column")


def test_null_pattern_which_columns_contain_null():
    assert _NULL_MAP_PATTERNS.search("which columns contain null values?")


# Non-matches
def test_null_no_match_readiness():
    assert not _NULL_MAP_PATTERNS.search("is my data ready for training?")


def test_null_no_match_bar_chart():
    assert not _NULL_MAP_PATTERNS.search("bar chart of revenue by region")


def test_null_no_match_top_n():
    assert not _NULL_MAP_PATTERNS.search("show me top 10 customers")


def test_null_no_match_filter():
    assert not _NULL_MAP_PATTERNS.search("filter to region East")


# ---------------------------------------------------------------------------
# Chat SSE integration tests
# ---------------------------------------------------------------------------


def _extract_null_map_event(response_text: str) -> dict | None:
    for line in response_text.split("\n"):
        if not line.startswith("data: "):
            continue
        try:
            parsed = json.loads(line[6:])
            if parsed.get("type") == "null_map" and parsed.get("null_map"):
                return parsed["null_map"]
        except json.JSONDecodeError:
            pass
    return None


def test_null_map_chat_emits_event(client, project_with_nulls):
    """Chat with null map query should emit {type:'null_map'} SSE event."""
    project_id, _ = project_with_nulls

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Here is the missing values overview."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": "show me the missing values"},
        )

    assert response.status_code == 200
    null_map = _extract_null_map_event(response.text)
    assert null_map is not None, "No null_map event found in SSE stream"


def test_null_map_columns_sorted_most_missing_first(client, project_with_nulls):
    """Columns with the most missing values should appear first."""
    project_id, _ = project_with_nulls

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Missing values overview."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": "which columns have missing data?"},
        )

    null_map = _extract_null_map_event(response.text)
    assert null_map is not None

    cols = null_map["columns"]
    # Verify sorted descending by null_pct
    for i in range(len(cols) - 1):
        assert cols[i]["null_pct"] >= cols[i + 1]["null_pct"]


def test_null_map_has_required_fields(client, project_with_nulls):
    """Null map event should have all required structural fields."""
    project_id, _ = project_with_nulls

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Done."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": "null values overview"},
        )

    null_map = _extract_null_map_event(response.text)
    assert null_map is not None

    assert "total_rows" in null_map
    assert "total_columns" in null_map
    assert "columns_with_nulls" in null_map
    assert "fully_complete_columns" in null_map
    assert "overall_completeness" in null_map
    assert "summary" in null_map
    assert "columns" in null_map
    assert len(null_map["columns"]) > 0

    first_col = null_map["columns"][0]
    assert "column" in first_col
    assert "null_count" in first_col
    assert "null_pct" in first_col
    assert "complete_pct" in first_col


def test_null_map_columns_with_nulls_count(client, project_with_nulls):
    """columns_with_nulls should count only columns that have at least one null."""
    project_id, _ = project_with_nulls

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Done."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": "missing data summary"},
        )

    null_map = _extract_null_map_event(response.text)
    assert null_map is not None
    # We uploaded CSV with nulls in revenue, units, notes columns
    assert null_map["columns_with_nulls"] > 0
    assert (
        null_map["columns_with_nulls"] + null_map["fully_complete_columns"]
        == null_map["total_columns"]
    )


def test_null_map_complete_dataset_shows_zero_missing(client, project_complete):
    """A complete dataset should report 0 columns with nulls."""
    project_id, _ = project_complete

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["All complete."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": "show me the missing values"},
        )

    null_map = _extract_null_map_event(response.text)
    assert null_map is not None
    assert null_map["columns_with_nulls"] == 0
    assert null_map["overall_completeness"] == 100.0


def test_null_map_complete_pct_plus_null_pct_equals_100(client, project_with_nulls):
    """For each column, complete_pct + null_pct should equal 100."""
    project_id, _ = project_with_nulls

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Done."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": "data completeness overview"},
        )

    null_map = _extract_null_map_event(response.text)
    assert null_map is not None
    for col in null_map["columns"]:
        assert abs(col["complete_pct"] + col["null_pct"] - 100) < 0.2


def test_null_map_no_dataset_no_event(client):
    """Without a dataset, no null_map event should be emitted."""
    proj_resp = client.post("/api/projects/", json={"name": "Empty Null Test"})
    project_id = proj_resp.json()["id"]

    with patch("anthropic.Anthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client

        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["No dataset."])
        mock_client.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": "show me the missing values"},
        )

    assert response.status_code == 200
    null_map = _extract_null_map_event(response.text)
    assert null_map is None
