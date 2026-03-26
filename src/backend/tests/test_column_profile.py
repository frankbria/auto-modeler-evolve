"""Tests for column profile deep-dive feature.

Covers:
- compute_column_profile() unit tests (core/analyzer.py)
  - numeric column stats, distribution, issues (skewed, constant, potential_id)
  - categorical column stats, distribution, issues (high_cardinality, constant, dominant)
  - date column stats
  - high null rate issue
  - missing column error
- GET /api/data/{dataset_id}/column-profile endpoint
  - 200 with valid numeric column
  - 200 with valid categorical column
  - 400 on unknown column
  - 404 on unknown dataset
- _COLUMN_PROFILE_PATTERNS regex detection
- _detect_profile_col() helper
- POST /api/chat/{project_id} emits column_profile SSE event
"""

import io
import json

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import _COLUMN_PROFILE_PATTERNS, _detect_profile_col
from core.analyzer import compute_column_profile

# ---------------------------------------------------------------------------
# Sample CSV with mixed column types
# ---------------------------------------------------------------------------

_SAMPLE_CSV = (
    b"order_date,region,revenue,category\n"
    b"2024-01-01,East,100.5,A\n"
    b"2024-02-01,West,200.3,B\n"
    b"2024-03-01,East,150.7,A\n"
    b"2024-04-01,West,300.1,C\n"
    b"2024-05-01,North,250.9,A\n"
    b"2024-06-01,East,175.2,B\n"
    b"2024-07-01,West,220.4,C\n"
    b"2024-08-01,North,190.6,A\n"
    b"2024-09-01,East,130.8,B\n"
    b"2024-10-01,West,280.0,C\n"
)


# ---------------------------------------------------------------------------
# Unit tests: compute_column_profile()
# ---------------------------------------------------------------------------


class TestComputeColumnProfileNumeric:
    def _df(self):
        return pd.DataFrame(
            {"revenue": [100.5, 200.3, 150.7, 300.1, 250.9, 175.2, 220.4, 190.6, 130.8, 280.0]}
        )

    def test_col_type_is_numeric(self):
        result = compute_column_profile(self._df(), "revenue")
        assert result["col_type"] == "numeric"

    def test_stats_keys_present(self):
        result = compute_column_profile(self._df(), "revenue")
        stats = result["stats"]
        for key in ("min", "max", "mean", "median", "std", "p25", "p75"):
            assert key in stats, f"Missing stat key: {key}"

    def test_stats_values_correct(self):
        result = compute_column_profile(self._df(), "revenue")
        stats = result["stats"]
        assert stats["total_rows"] == 10
        assert stats["null_count"] == 0
        assert stats["null_pct"] == 0.0
        assert stats["unique_count"] == 10
        assert abs(stats["mean"] - 199.95) < 1.0

    def test_distribution_is_histogram(self):
        result = compute_column_profile(self._df(), "revenue")
        dist = result["distribution"]
        assert dist["type"] == "histogram"
        assert isinstance(dist["bins"], list)
        assert isinstance(dist["counts"], list)
        assert len(dist["bins"]) == len(dist["counts"])

    def test_summary_contains_numeric_info(self):
        result = compute_column_profile(self._df(), "revenue")
        summary = result["summary"].lower()
        assert "numeric" in summary or "100" in result["summary"] or "range" in summary

    def test_no_issues_for_clean_column(self):
        result = compute_column_profile(self._df(), "revenue")
        assert result["issues"] == []

    def test_high_null_issue_detected(self):
        df = pd.DataFrame({"revenue": [1.0, None, None, None, None, None, 2.0, None, None, None]})
        result = compute_column_profile(df, "revenue")
        issue_types = [i["type"] for i in result["issues"]]
        assert "high_null_rate" in issue_types

    def test_skewed_issue_detected(self):
        vals = [1, 1, 1, 1, 1, 1, 1, 2, 3, 1000]
        df = pd.DataFrame({"val": vals})
        result = compute_column_profile(df, "val")
        issue_types = [i["type"] for i in result["issues"]]
        assert "skewed" in issue_types

    def test_constant_value_issue(self):
        df = pd.DataFrame({"revenue": [5.0] * 10})
        result = compute_column_profile(df, "revenue")
        issue_types = [i["type"] for i in result["issues"]]
        assert "constant_value" in issue_types

    def test_potential_id_issue(self):
        df = pd.DataFrame({"id": list(range(20))})
        result = compute_column_profile(df, "id")
        issue_types = [i["type"] for i in result["issues"]]
        assert "potential_id" in issue_types


class TestComputeColumnProfileCategorical:
    def _df(self):
        return pd.DataFrame(
            {"region": ["East", "West", "East", "West", "North", "East", "West", "North", "East", "West"]}
        )

    def test_col_type_is_categorical(self):
        result = compute_column_profile(self._df(), "region")
        assert result["col_type"] == "categorical"

    def test_stats_keys(self):
        result = compute_column_profile(self._df(), "region")
        stats = result["stats"]
        for key in ("most_common", "most_common_pct", "top_categories", "unique_count"):
            assert key in stats, f"Missing: {key}"

    def test_most_common_value(self):
        result = compute_column_profile(self._df(), "region")
        assert result["stats"]["most_common"] == "East"

    def test_distribution_is_bar(self):
        result = compute_column_profile(self._df(), "region")
        dist = result["distribution"]
        assert dist["type"] == "bar"
        assert isinstance(dist["labels"], list)
        assert isinstance(dist["counts"], list)

    def test_high_cardinality_issue(self):
        df = pd.DataFrame({"col": [f"val_{i}" for i in range(60)]})
        result = compute_column_profile(df, "col")
        issue_types = [i["type"] for i in result["issues"]]
        assert "high_cardinality" in issue_types

    def test_constant_value_categorical(self):
        df = pd.DataFrame({"region": ["East"] * 10})
        result = compute_column_profile(df, "region")
        issue_types = [i["type"] for i in result["issues"]]
        assert "constant_value" in issue_types

    def test_dominant_value_issue(self):
        df = pd.DataFrame({"region": ["East"] * 19 + ["West"]})
        result = compute_column_profile(df, "region")
        issue_types = [i["type"] for i in result["issues"]]
        assert "dominant_value" in issue_types


class TestComputeColumnProfileDate:
    def _df(self):
        return pd.DataFrame(
            {
                "order_date": [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-04-01",
                    "2024-05-01",
                ]
            }
        )

    def test_date_column_detected(self):
        result = compute_column_profile(self._df(), "order_date")
        assert result["col_type"] == "date"

    def test_date_stats_present(self):
        result = compute_column_profile(self._df(), "order_date")
        stats = result["stats"]
        assert "min_date" in stats
        assert "max_date" in stats


class TestComputeColumnProfileErrors:
    def test_missing_column_returns_error(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = compute_column_profile(df, "nonexistent")
        assert "error" in result

    def test_all_null_numeric_does_not_crash(self):
        df = pd.DataFrame({"val": [None, None, None]})
        result = compute_column_profile(df, "val")
        assert result is not None
        assert "col_type" in result


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------


class TestColumnProfilePatterns:
    def test_tell_me_about(self):
        assert _COLUMN_PROFILE_PATTERNS.search("tell me about the revenue column")

    def test_profile_column(self):
        assert _COLUMN_PROFILE_PATTERNS.search("profile the region column")

    def test_describe_column(self):
        assert _COLUMN_PROFILE_PATTERNS.search("describe the age column")

    def test_show_stats_for(self):
        assert _COLUMN_PROFILE_PATTERNS.search("show me stats for price")

    def test_distribution_of(self):
        assert _COLUMN_PROFILE_PATTERNS.search("distribution of revenue")

    def test_histogram_for(self):
        assert _COLUMN_PROFILE_PATTERNS.search("histogram for sales")

    def test_explore_column(self):
        assert _COLUMN_PROFILE_PATTERNS.search("explore the category column")

    def test_what_values_in(self):
        assert _COLUMN_PROFILE_PATTERNS.search("what are the values in region")


class TestDetectProfileCol:
    def _df(self):
        return pd.DataFrame(columns=["revenue", "region", "order_date", "category"])

    def test_exact_match(self):
        assert _detect_profile_col("tell me about revenue", self._df()) == "revenue"

    def test_case_insensitive(self):
        assert _detect_profile_col("show me REVENUE distribution", self._df()) == "revenue"

    def test_categorical_match(self):
        assert _detect_profile_col("profile the category column", self._df()) == "category"

    def test_returns_none_no_match(self):
        result = _detect_profile_col("what time is it", self._df())
        assert result is None


# ---------------------------------------------------------------------------
# Async fixture for API tests (matches pattern used across test suite)
# ---------------------------------------------------------------------------


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.dataset_filter  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.deploy as deploy_module

    deploy_module.DEPLOY_DIR = tmp_path / "deployments"

    import api.models as models_module

    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
async def project_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Column Profile Test"})
    return resp.json()["id"]


@pytest.fixture()
async def dataset_id(ac, project_id):
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("data.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["dataset_id"]


# ---------------------------------------------------------------------------
# Endpoint tests: GET /api/data/{id}/column-profile
# ---------------------------------------------------------------------------


class TestColumnProfileEndpoint:
    async def test_numeric_column(self, ac, dataset_id):
        resp = await ac.get(f"/api/data/{dataset_id}/column-profile?col=revenue")
        assert resp.status_code == 200
        data = resp.json()
        assert data["col_type"] == "numeric"
        assert data["col_name"] == "revenue"
        assert "stats" in data
        assert "distribution" in data
        assert "issues" in data
        assert "summary" in data

    async def test_categorical_column(self, ac, dataset_id):
        resp = await ac.get(f"/api/data/{dataset_id}/column-profile?col=region")
        assert resp.status_code == 200
        data = resp.json()
        assert data["col_type"] == "categorical"
        assert data["stats"]["unique_count"] == 3  # East, West, North

    async def test_unknown_column_returns_400(self, ac, dataset_id):
        resp = await ac.get(f"/api/data/{dataset_id}/column-profile?col=nonexistent")
        assert resp.status_code == 400

    async def test_unknown_dataset_returns_404(self, ac):
        resp = await ac.get(
            "/api/data/00000000-0000-0000-0000-000000000000/column-profile?col=revenue"
        )
        assert resp.status_code == 404

    async def test_distribution_bins_present(self, ac, dataset_id):
        resp = await ac.get(f"/api/data/{dataset_id}/column-profile?col=revenue")
        dist = resp.json()["distribution"]
        assert dist["type"] == "histogram"
        assert len(dist["bins"]) > 0
        assert len(dist["counts"]) > 0


# ---------------------------------------------------------------------------
# Chat SSE integration test
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_chat_emits_column_profile_event(ac, project_id, dataset_id):
    """POST /api/chat/{project_id} should emit a column_profile SSE event."""
    import unittest.mock as mock

    mock_stream = mock.MagicMock()
    mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = mock.MagicMock(return_value=False)
    mock_stream.text_stream = iter(["Here is the profile for the revenue column."])

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = mock_stream

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "tell me about the revenue column"},
            timeout=15,
        )
    assert resp.status_code == 200

    events = [
        json.loads(line[6:])
        for line in resp.text.split("\n")
        if line.startswith("data: ")
    ]
    event_types = [e.get("type") for e in events]
    assert "column_profile" in event_types, f"column_profile not in: {event_types}"

    profile_event = next(e for e in events if e.get("type") == "column_profile")
    assert "column_profile" in profile_event
    assert profile_event["column_profile"]["col_name"] == "revenue"
    assert profile_event["column_profile"]["col_type"] == "numeric"
