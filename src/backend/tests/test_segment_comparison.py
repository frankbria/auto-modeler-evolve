"""Tests for segment comparison feature.

Covers:
- compare_segments() unit tests (core/analyzer.py)
- GET /api/data/{dataset_id}/compare-segments endpoint
- _detect_compare_request() helper in api/chat.py
"""

import io

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.analyzer import compare_segments

# ---------------------------------------------------------------------------
# Sample data: two clear segments (East vs West) with known differences
# ---------------------------------------------------------------------------

SAMPLE_CSV = (
    b"region,product,revenue,units,cost\n"
    b"East,Widget A,2000.0,20,800.0\n"
    b"East,Widget B,1800.0,18,700.0\n"
    b"East,Widget A,2200.0,22,850.0\n"
    b"East,Widget C,1900.0,19,750.0\n"
    b"West,Widget A,500.0,5,200.0\n"
    b"West,Widget B,600.0,6,220.0\n"
    b"West,Widget C,550.0,5,210.0\n"
    b"West,Widget A,480.0,5,195.0\n"
)


def make_df():
    return pd.read_csv(io.BytesIO(SAMPLE_CSV))


# ---------------------------------------------------------------------------
# Unit tests: compare_segments()
# ---------------------------------------------------------------------------


class TestCompareSegments:
    def test_basic_output_structure(self):
        df = make_df()
        result = compare_segments(df, "region", "East", "West")
        assert result["group_col"] == "region"
        assert result["val1"] == "East"
        assert result["val2"] == "West"
        assert result["count1"] == 4
        assert result["count2"] == 4
        assert isinstance(result["columns"], list)
        assert isinstance(result["notable_diffs"], list)
        assert isinstance(result["summary"], str)
        assert "East" in result["summary"]
        assert "West" in result["summary"]

    def test_numeric_columns_covered(self):
        df = make_df()
        result = compare_segments(df, "region", "East", "West")
        col_names = {c["name"] for c in result["columns"]}
        assert "revenue" in col_names
        assert "units" in col_names
        assert "cost" in col_names
        # non-numeric (product, region) should NOT appear
        assert "product" not in col_names
        assert "region" not in col_names

    def test_mean_values_correct(self):
        df = make_df()
        result = compare_segments(df, "region", "East", "West")
        rev_col = next(c for c in result["columns"] if c["name"] == "revenue")
        # East revenue: 2000, 1800, 2200, 1900 → mean 1975
        assert abs(rev_col["mean1"] - 1975.0) < 0.1
        # West revenue: 500, 600, 550, 480 → mean 532.5
        assert abs(rev_col["mean2"] - 532.5) < 0.1

    def test_effect_size_direction(self):
        df = make_df()
        result = compare_segments(df, "region", "East", "West")
        rev_col = next(c for c in result["columns"] if c["name"] == "revenue")
        # East revenue >> West revenue → direction higher_in_val1
        assert rev_col["direction"] == "higher_in_val1"

    def test_notable_diffs_sorted_by_magnitude(self):
        df = make_df()
        result = compare_segments(df, "region", "East", "West")
        effects = [abs(n["effect_size"]) for n in result["notable_diffs"]]
        assert effects == sorted(effects, reverse=True)

    def test_large_effect_flagged_notable(self):
        df = make_df()
        result = compare_segments(df, "region", "East", "West")
        # Revenue has a very large effect (East >> West) → must be notable
        notable_names = {n["name"] for n in result["notable_diffs"]}
        assert "revenue" in notable_names

    def test_case_insensitive_matching(self):
        df = make_df()
        result = compare_segments(df, "region", "east", "WEST")
        assert result["count1"] == 4
        assert result["count2"] == 4

    def test_empty_segment_returns_none_stats(self):
        df = make_df()
        # val2 "North" doesn't exist — counts should be 0, stats None
        result = compare_segments(df, "region", "East", "North")
        assert result["count2"] == 0
        for col in result["columns"]:
            assert col["mean2"] is None

    def test_summary_mentions_notable(self):
        df = make_df()
        result = compare_segments(df, "region", "East", "West")
        assert len(result["summary"]) > 0
        # Summary should reference notable differences
        if result["notable_diffs"]:
            assert "higher" in result["summary"].lower()

    def test_excludes_group_col_itself(self):
        df = make_df()
        result = compare_segments(df, "region", "East", "West")
        col_names = {c["name"] for c in result["columns"]}
        assert "region" not in col_names


# ---------------------------------------------------------------------------
# API tests: GET /api/data/{dataset_id}/compare-segments
# ---------------------------------------------------------------------------


@pytest.fixture
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.feature_set  # noqa
    import models.model_run  # noqa
    import models.project  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
async def dataset_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Segment Test"})
    project_id = resp.json()["id"]
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("data.csv", SAMPLE_CSV, "text/csv")},
        data={"project_id": project_id},
    )
    return resp.json()["dataset_id"]


class TestCompareSegmentsEndpoint:
    async def test_basic_comparison(self, dataset_id, ac):
        resp = await ac.get(
            f"/api/data/{dataset_id}/compare-segments",
            params={"col": "region", "val1": "East", "val2": "West"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["group_col"] == "region"
        assert data["count1"] == 4
        assert data["count2"] == 4
        assert isinstance(data["columns"], list)
        assert len(data["columns"]) > 0

    async def test_unknown_column_returns_400(self, dataset_id, ac):
        resp = await ac.get(
            f"/api/data/{dataset_id}/compare-segments",
            params={"col": "nonexistent", "val1": "East", "val2": "West"},
        )
        assert resp.status_code == 400

    async def test_unknown_val1_returns_400(self, dataset_id, ac):
        resp = await ac.get(
            f"/api/data/{dataset_id}/compare-segments",
            params={"col": "region", "val1": "Unknown", "val2": "West"},
        )
        assert resp.status_code == 400

    async def test_unknown_val2_returns_400(self, dataset_id, ac):
        resp = await ac.get(
            f"/api/data/{dataset_id}/compare-segments",
            params={"col": "region", "val1": "East", "val2": "Unknown"},
        )
        assert resp.status_code == 400

    async def test_404_for_missing_dataset(self, ac):
        resp = await ac.get(
            "/api/data/nonexistent-id/compare-segments",
            params={"col": "region", "val1": "East", "val2": "West"},
        )
        assert resp.status_code == 404

    async def test_response_contains_summary(self, dataset_id, ac):
        resp = await ac.get(
            f"/api/data/{dataset_id}/compare-segments",
            params={"col": "region", "val1": "East", "val2": "West"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert len(data["summary"]) > 0

    async def test_notable_diffs_populated(self, dataset_id, ac):
        resp = await ac.get(
            f"/api/data/{dataset_id}/compare-segments",
            params={"col": "region", "val1": "East", "val2": "West"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # East >> West on revenue — should have notable differences
        assert isinstance(data["notable_diffs"], list)
        assert len(data["notable_diffs"]) > 0


# ---------------------------------------------------------------------------
# Chat helper tests: _detect_compare_request()
# ---------------------------------------------------------------------------


class TestDetectCompareRequest:
    def _df(self):
        return make_df()

    def test_compare_vs_pattern(self):
        from api.chat import _detect_compare_request

        df = self._df()
        result = _detect_compare_request("compare East vs West", df)
        assert result is not None
        assert result["group_col"] == "region"
        assert result["val1"].lower() == "east"
        assert result["val2"].lower() == "west"

    def test_versus_pattern(self):
        from api.chat import _detect_compare_request

        df = self._df()
        result = _detect_compare_request("East versus West", df)
        assert result is not None
        assert result["group_col"] == "region"

    def test_difference_between_pattern(self):
        from api.chat import _detect_compare_request

        df = self._df()
        result = _detect_compare_request(
            "what is the difference between East and West?", df
        )
        assert result is not None
        assert result["group_col"] == "region"

    def test_no_match_when_values_not_in_data(self):
        from api.chat import _detect_compare_request

        df = self._df()
        result = _detect_compare_request("compare Alpha vs Beta", df)
        assert result is None

    def test_no_match_on_unrelated_message(self):
        from api.chat import _detect_compare_request

        df = self._df()
        result = _detect_compare_request("show me the correlation heatmap", df)
        assert result is None
