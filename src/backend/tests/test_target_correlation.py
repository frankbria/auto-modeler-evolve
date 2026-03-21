"""Tests for target correlation analysis feature.

Covers:
- analyze_target_correlations() unit tests (core/analyzer.py)
- GET /api/data/{dataset_id}/target-correlations endpoint
- _detect_correlation_target_request() helper in api/chat.py
- _CORRELATION_TARGET_PATTERNS regex
"""

import io

import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.analyzer import analyze_target_correlations

# ---------------------------------------------------------------------------
# Sample data: known correlations
# revenue is strongly correlated with units (positive) and inversely with discount
# ---------------------------------------------------------------------------

SAMPLE_CSV = (
    b"region,product,revenue,units,cost,discount\n"
    b"East,Widget A,2000.0,20,800.0,0.05\n"
    b"East,Widget B,1800.0,18,700.0,0.08\n"
    b"East,Widget A,2200.0,22,850.0,0.03\n"
    b"East,Widget C,1900.0,19,750.0,0.06\n"
    b"West,Widget A,500.0,5,200.0,0.25\n"
    b"West,Widget B,600.0,6,220.0,0.22\n"
    b"West,Widget C,550.0,5,210.0,0.23\n"
    b"West,Widget A,480.0,5,195.0,0.28\n"
    b"North,Widget B,1200.0,12,480.0,0.12\n"
    b"North,Widget C,1100.0,11,440.0,0.14\n"
)


def make_df():
    return pd.read_csv(io.BytesIO(SAMPLE_CSV))


# ---------------------------------------------------------------------------
# Unit tests: analyze_target_correlations()
# ---------------------------------------------------------------------------


class TestAnalyzeTargetCorrelations:
    def test_basic_output_structure(self):
        df = make_df()
        result = analyze_target_correlations(df, "revenue")
        assert result["target_col"] == "revenue"
        assert "correlations" in result
        assert "summary" in result
        assert "error" not in result

    def test_correlations_sorted_by_absolute_value(self):
        df = make_df()
        result = analyze_target_correlations(df, "revenue")
        corrs = result["correlations"]
        abs_vals = [abs(e["correlation"]) for e in corrs]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_units_strongly_correlated_with_revenue(self):
        df = make_df()
        result = analyze_target_correlations(df, "revenue")
        corr_map = {e["column"]: e for e in result["correlations"]}
        assert "units" in corr_map
        assert abs(corr_map["units"]["correlation"]) > 0.9

    def test_discount_negatively_correlated_with_revenue(self):
        df = make_df()
        result = analyze_target_correlations(df, "revenue")
        corr_map = {e["column"]: e for e in result["correlations"]}
        assert "discount" in corr_map
        assert corr_map["discount"]["direction"] == "negative"

    def test_direction_field(self):
        df = make_df()
        result = analyze_target_correlations(df, "revenue")
        for entry in result["correlations"]:
            assert entry["direction"] in ("positive", "negative")
            expected_dir = "positive" if entry["correlation"] >= 0 else "negative"
            assert entry["direction"] == expected_dir

    def test_strength_labels(self):
        df = make_df()
        result = analyze_target_correlations(df, "revenue")
        valid_strengths = {"very strong", "strong", "moderate", "weak", "negligible"}
        for entry in result["correlations"]:
            assert entry["strength"] in valid_strengths

    def test_target_excluded_from_correlations(self):
        df = make_df()
        result = analyze_target_correlations(df, "revenue")
        columns = [e["column"] for e in result["correlations"]]
        assert "revenue" not in columns

    def test_top_n_limits_results(self):
        df = make_df()
        result = analyze_target_correlations(df, "revenue", top_n=2)
        assert len(result["correlations"]) <= 2

    def test_column_not_found_returns_error(self):
        df = make_df()
        result = analyze_target_correlations(df, "nonexistent_col")
        assert result["error"] == "column_not_found"
        assert result["correlations"] == []

    def test_non_numeric_target_returns_error(self):
        df = make_df()
        result = analyze_target_correlations(df, "region")
        assert result["error"] == "not_numeric"
        assert result["correlations"] == []

    def test_no_other_numeric_columns(self):
        df = pd.DataFrame({"revenue": [1.0, 2.0, 3.0], "name": ["a", "b", "c"]})
        result = analyze_target_correlations(df, "revenue")
        assert result["error"] == "no_numeric_columns"

    def test_summary_is_non_empty(self):
        df = make_df()
        result = analyze_target_correlations(df, "revenue")
        assert len(result["summary"]) > 20

    def test_summary_mentions_target_col(self):
        df = make_df()
        result = analyze_target_correlations(df, "revenue")
        assert "revenue" in result["summary"].lower()

    def test_correlation_values_in_range(self):
        df = make_df()
        result = analyze_target_correlations(df, "revenue")
        for entry in result["correlations"]:
            assert -1.0 <= entry["correlation"] <= 1.0

    def test_nan_resistant(self):
        # Some NaN values in the data should not crash the function
        df = make_df()
        df.loc[0, "units"] = float("nan")
        result = analyze_target_correlations(df, "revenue")
        assert "error" not in result
        assert result["correlations"]

    def test_default_top_n_is_ten(self):
        # Create a wide DataFrame with many numeric columns
        import numpy as np

        rng = np.random.default_rng(42)
        data = {"target": rng.random(50)}
        for i in range(15):
            data[f"col_{i}"] = rng.random(50)
        df = pd.DataFrame(data)
        result = analyze_target_correlations(df, "target")
        assert len(result["correlations"]) <= 10


# ---------------------------------------------------------------------------
# Endpoint tests: GET /api/data/{dataset_id}/target-correlations
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


@pytest.fixture()
async def dataset_id(ac):
    proj_resp = await ac.post("/api/projects", json={"name": "Corr Test"})
    project_id = proj_resp.json()["id"]
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("test.csv", SAMPLE_CSV, "text/csv")},
        data={"project_id": project_id},
    )
    return resp.json()["dataset_id"]


class TestTargetCorrelationsEndpoint:
    async def test_basic_response_structure(self, ac, dataset_id):
        resp = await ac.get(
            f"/api/data/{dataset_id}/target-correlations", params={"target": "revenue"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["target_col"] == "revenue"
        assert "correlations" in data
        assert "summary" in data
        assert data["dataset_id"] == dataset_id

    async def test_404_for_missing_dataset(self, ac):
        resp = await ac.get(
            "/api/data/no-such-id/target-correlations", params={"target": "revenue"}
        )
        assert resp.status_code == 404

    async def test_400_for_missing_target_column(self, ac, dataset_id):
        resp = await ac.get(
            f"/api/data/{dataset_id}/target-correlations",
            params={"target": "nonexistent"},
        )
        assert resp.status_code == 400

    async def test_400_for_non_numeric_target(self, ac, dataset_id):
        resp = await ac.get(
            f"/api/data/{dataset_id}/target-correlations", params={"target": "region"}
        )
        assert resp.status_code == 400

    async def test_top_n_param_respected(self, ac, dataset_id):
        resp = await ac.get(
            f"/api/data/{dataset_id}/target-correlations",
            params={"target": "revenue", "top_n": 2},
        )
        assert resp.status_code == 200
        assert len(resp.json()["correlations"]) <= 2

    async def test_correlations_sorted_strongest_first(self, ac, dataset_id):
        resp = await ac.get(
            f"/api/data/{dataset_id}/target-correlations", params={"target": "revenue"}
        )
        corrs = resp.json()["correlations"]
        abs_vals = [abs(e["correlation"]) for e in corrs]
        assert abs_vals == sorted(abs_vals, reverse=True)


# ---------------------------------------------------------------------------
# Chat helper tests: _detect_correlation_target_request and patterns
# ---------------------------------------------------------------------------


class TestDetectCorrelationTargetRequest:
    def setup_method(self):
        from api.chat import _detect_correlation_target_request

        self._detect = _detect_correlation_target_request

    def test_detects_column_in_message(self):
        cols = ["revenue", "units", "cost"]
        result = self._detect("what's correlated with revenue?", cols)
        assert result == "revenue"

    def test_case_insensitive(self):
        cols = ["Revenue", "units"]
        result = self._detect("What drives REVENUE?", cols)
        assert result == "Revenue"

    def test_returns_none_when_no_column_mentioned(self):
        cols = ["revenue", "units"]
        result = self._detect("what's going on?", cols)
        assert result is None

    def test_first_matching_column_returned(self):
        cols = ["revenue", "units"]
        result = self._detect("what drives revenue and units?", cols)
        # Either is valid — just not None
        assert result in cols


class TestCorrelationTargetPatterns:
    def setup_method(self):
        from api.chat import _CORRELATION_TARGET_PATTERNS

        self._pattern = _CORRELATION_TARGET_PATTERNS

    def test_matches_correlated_with(self):
        assert self._pattern.search("what's correlated with revenue?")

    def test_matches_what_drives(self):
        assert self._pattern.search("what drives sales?")

    def test_matches_what_predicts(self):
        assert self._pattern.search("what predicts profit?")

    def test_matches_show_correlations_for(self):
        assert self._pattern.search("show correlations for margin")

    def test_matches_which_factors(self):
        assert self._pattern.search("which factors affect revenue?")

    def test_matches_what_influences(self):
        assert self._pattern.search("what influences profit?")

    def test_does_not_match_unrelated_message(self):
        assert not self._pattern.search("train my model")

    def test_case_insensitive(self):
        assert self._pattern.search("WHAT DRIVES REVENUE?")
