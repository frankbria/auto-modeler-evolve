"""Tests for conversational data cleaning: core logic + API endpoint + chat patterns.

Coverage:
  - core/cleaner.py: remove_duplicates, fill_missing, filter_rows, cap_outliers, drop_column
  - POST /api/data/{dataset_id}/clean
  - chat._CLEAN_PATTERNS regex + _detect_clean_op() helper
  - chat SSE emits cleaning_suggestion event on match
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from core.cleaner import (
    cap_outliers,
    drop_column,
    fill_missing,
    filter_rows,
    remove_duplicates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "revenue": [100.0, 200.0, 300.0, 200.0, None, 5000.0],
            "region": ["North", "South", "East", "South", "West", "North"],
            "quantity": [10, 20, 30, 20, 15, None],
        }
    )


def _upload_csv(client: TestClient, content: bytes, project_id: str) -> str:
    """Upload CSV and return dataset_id."""
    r = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.csv", io.BytesIO(content), "text/csv")},
    )
    assert r.status_code == 201, r.text
    return r.json()["dataset_id"]


def _make_project(client: TestClient) -> str:
    r = client.post("/api/projects", json={"name": "Clean Test"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


CSV_WITH_DUPS = b"""revenue,region,quantity
100,North,10
200,South,20
200,South,20
300,East,30
"""

CSV_WITH_MISSING = b"""revenue,region,quantity
100,North,10
200,South,
300,East,30
,West,15
"""

CSV_NUMERIC = b"""revenue,quantity
100,10
200,20
300,30
5000,500
50,5
"""


# ---------------------------------------------------------------------------
# Unit: remove_duplicates
# ---------------------------------------------------------------------------

class TestRemoveDuplicates:
    def test_removes_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        cleaned, result = remove_duplicates(df)
        assert result["modified_count"] == 1
        assert result["after_rows"] == 2
        assert result["before_rows"] == 3

    def test_no_duplicates_returns_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        cleaned, result = remove_duplicates(df)
        assert result["modified_count"] == 0
        assert "unchanged" in result["summary"]

    def test_result_has_correct_keys(self):
        df = pd.DataFrame({"a": [1, 1]})
        _, result = remove_duplicates(df)
        assert all(k in result for k in ("operation", "before_rows", "after_rows", "modified_count", "summary"))


# ---------------------------------------------------------------------------
# Unit: fill_missing
# ---------------------------------------------------------------------------

class TestFillMissing:
    def test_fill_mean(self):
        df = pd.DataFrame({"revenue": [10.0, 20.0, None, 30.0]})
        cleaned, result = fill_missing(df, "revenue", "mean")
        assert cleaned["revenue"].isna().sum() == 0
        assert result["modified_count"] == 1
        assert "mean" in result["summary"]

    def test_fill_median(self):
        df = pd.DataFrame({"revenue": [10.0, None, 30.0]})
        cleaned, result = fill_missing(df, "revenue", "median")
        assert cleaned["revenue"].isna().sum() == 0

    def test_fill_mode(self):
        df = pd.DataFrame({"region": ["North", None, "North", "South"]})
        cleaned, result = fill_missing(df, "region", "mode")
        assert cleaned["region"].isna().sum() == 0
        assert "North" in result["summary"]

    def test_fill_zero(self):
        df = pd.DataFrame({"quantity": [10, None, 20]})
        cleaned, result = fill_missing(df, "quantity", "zero")
        assert cleaned["quantity"].iloc[1] == 0

    def test_fill_literal_value(self):
        df = pd.DataFrame({"score": [1.0, None, 3.0]})
        cleaned, result = fill_missing(df, "score", "value", fill_value=99.0)
        assert cleaned["score"].iloc[1] == 99.0
        assert "99" in result["summary"]

    def test_no_missing_returns_unchanged(self):
        df = pd.DataFrame({"revenue": [1.0, 2.0, 3.0]})
        _, result = fill_missing(df, "revenue", "mean")
        assert result["modified_count"] == 0

    def test_unknown_column_raises(self):
        df = pd.DataFrame({"revenue": [1.0]})
        with pytest.raises(ValueError, match="not found"):
            fill_missing(df, "nonexistent", "mean")

    def test_mean_on_string_column_raises(self):
        df = pd.DataFrame({"region": ["North", None]})
        with pytest.raises(ValueError, match="numeric"):
            fill_missing(df, "region", "mean")

    def test_unknown_strategy_raises(self):
        df = pd.DataFrame({"revenue": [1.0, None]})
        with pytest.raises(ValueError, match="Unknown strategy"):
            fill_missing(df, "revenue", "bogus")


# ---------------------------------------------------------------------------
# Unit: filter_rows
# ---------------------------------------------------------------------------

class TestFilterRows:
    def test_filter_gt(self):
        df = pd.DataFrame({"quantity": [10, 0, -5, 20]})
        cleaned, result = filter_rows(df, "quantity", "lt", 0)
        # Removes rows where quantity < 0 → removes -5
        assert len(cleaned) == 3
        assert result["modified_count"] == 1

    def test_filter_eq(self):
        df = pd.DataFrame({"region": ["North", "South", "North"]})
        cleaned, result = filter_rows(df, "region", "eq", "North")
        assert len(cleaned) == 1  # only "South" remains

    def test_filter_contains(self):
        df = pd.DataFrame({"region": ["North America", "South", "Northern Europe"]})
        cleaned, result = filter_rows(df, "region", "contains", "north")
        assert len(cleaned) == 1  # only "South" remains

    def test_no_match_returns_unchanged(self):
        df = pd.DataFrame({"revenue": [10, 20, 30]})
        cleaned, result = filter_rows(df, "revenue", "gt", 1000)
        assert result["modified_count"] == 0
        assert "unchanged" in result["summary"]

    def test_unknown_column_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="not found"):
            filter_rows(df, "nonexistent", "gt", 0)

    def test_unknown_operator_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="Unknown operator"):
            filter_rows(df, "a", "between", 0)


# ---------------------------------------------------------------------------
# Unit: cap_outliers
# ---------------------------------------------------------------------------

class TestCapOutliers:
    def test_caps_extreme_values(self):
        df = pd.DataFrame({"revenue": [100.0, 200.0, 300.0, 50000.0]})
        cleaned, result = cap_outliers(df, "revenue", percentile=95.0)
        # 50000 should be clipped
        assert cleaned["revenue"].max() < 50000
        assert result["modified_count"] >= 1

    def test_no_extremes_returns_unchanged(self):
        # Use identical values so all percentile bounds collapse to the same number → no clipping.
        df = pd.DataFrame({"revenue": [200.0] * 50})
        cleaned, result = cap_outliers(df, "revenue", percentile=99.0)
        assert result["modified_count"] == 0

    def test_non_numeric_column_raises(self):
        df = pd.DataFrame({"region": ["North", "South"]})
        with pytest.raises(ValueError, match="numeric"):
            cap_outliers(df, "region")

    def test_unknown_column_raises(self):
        df = pd.DataFrame({"revenue": [1.0]})
        with pytest.raises(ValueError, match="not found"):
            cap_outliers(df, "nonexistent")


# ---------------------------------------------------------------------------
# Unit: drop_column
# ---------------------------------------------------------------------------

class TestDropColumn:
    def test_drops_column(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        cleaned, result = drop_column(df, "b")
        assert "b" not in cleaned.columns
        assert result["after_columns"] == 2
        assert "2 columns" in result["summary"]

    def test_unknown_column_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="not found"):
            drop_column(df, "nonexistent")


# ---------------------------------------------------------------------------
# API: POST /api/data/{id}/clean
# ---------------------------------------------------------------------------

class TestCleanEndpoint:
    def test_remove_duplicates_endpoint(self):
        from main import app
        client = TestClient(app)
        from sqlmodel import create_engine, SQLModel
        import db

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db.engine = create_engine(f"sqlite:///{tmp}/test.db", echo=False)
            SQLModel.metadata.create_all(db.engine)
            db.DATA_DIR = Path(tmp)

            r = client.post("/api/projects", json={"name": "Clean Test"})
            project_id = r.json()["id"]

            ds_id = _upload_csv(client, CSV_WITH_DUPS, project_id)

            r = client.post(f"/api/data/{ds_id}/clean", json={"operation": "remove_duplicates"})
            assert r.status_code == 200
            body = r.json()
            assert body["operation_result"]["modified_count"] == 1
            assert body["updated_stats"]["row_count"] == 3

    def test_fill_missing_endpoint(self):
        from main import app
        client = TestClient(app)
        from sqlmodel import create_engine, SQLModel
        import db

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db.engine = create_engine(f"sqlite:///{tmp}/test.db", echo=False)
            SQLModel.metadata.create_all(db.engine)
            db.DATA_DIR = Path(tmp)

            r = client.post("/api/projects", json={"name": "Clean Test"})
            project_id = r.json()["id"]
            ds_id = _upload_csv(client, CSV_WITH_MISSING, project_id)

            r = client.post(
                f"/api/data/{ds_id}/clean",
                json={"operation": "fill_missing", "column": "quantity", "strategy": "median"},
            )
            assert r.status_code == 200
            assert r.json()["operation_result"]["modified_count"] >= 1

    def test_cap_outliers_endpoint(self):
        from main import app
        client = TestClient(app)
        from sqlmodel import create_engine, SQLModel
        import db

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db.engine = create_engine(f"sqlite:///{tmp}/test.db", echo=False)
            SQLModel.metadata.create_all(db.engine)
            db.DATA_DIR = Path(tmp)

            r = client.post("/api/projects", json={"name": "Clean Test"})
            project_id = r.json()["id"]
            ds_id = _upload_csv(client, CSV_NUMERIC, project_id)

            r = client.post(
                f"/api/data/{ds_id}/clean",
                json={"operation": "cap_outliers", "column": "revenue", "percentile": 80.0},
            )
            assert r.status_code == 200

    def test_unknown_operation_returns_400(self):
        from main import app
        client = TestClient(app)
        from sqlmodel import create_engine, SQLModel
        import db

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db.engine = create_engine(f"sqlite:///{tmp}/test.db", echo=False)
            SQLModel.metadata.create_all(db.engine)
            db.DATA_DIR = Path(tmp)

            r = client.post("/api/projects", json={"name": "Clean Test"})
            project_id = r.json()["id"]
            ds_id = _upload_csv(client, CSV_WITH_DUPS, project_id)

            r = client.post(f"/api/data/{ds_id}/clean", json={"operation": "bogus_op"})
            assert r.status_code == 400

    def test_missing_column_param_returns_400(self):
        from main import app
        client = TestClient(app)
        from sqlmodel import create_engine, SQLModel
        import db

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db.engine = create_engine(f"sqlite:///{tmp}/test.db", echo=False)
            SQLModel.metadata.create_all(db.engine)
            db.DATA_DIR = Path(tmp)

            r = client.post("/api/projects", json={"name": "Clean Test"})
            project_id = r.json()["id"]
            ds_id = _upload_csv(client, CSV_WITH_MISSING, project_id)

            r = client.post(f"/api/data/{ds_id}/clean", json={"operation": "fill_missing"})
            assert r.status_code == 400

    def test_dataset_not_found_returns_404(self):
        from main import app
        client = TestClient(app)
        from sqlmodel import create_engine, SQLModel
        import db

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            db.engine = create_engine(f"sqlite:///{tmp}/test.db", echo=False)
            SQLModel.metadata.create_all(db.engine)
            db.DATA_DIR = Path(tmp)

            r = client.post("/api/data/nonexistent-id/clean", json={"operation": "remove_duplicates"})
            assert r.status_code == 404


# ---------------------------------------------------------------------------
# Chat: _CLEAN_PATTERNS regex + _detect_clean_op helper
# ---------------------------------------------------------------------------

class TestCleanChatPatterns:
    def test_fill_missing_matches(self):
        from api.chat import _CLEAN_PATTERNS
        assert _CLEAN_PATTERNS.search("fill missing values with median")
        assert _CLEAN_PATTERNS.search("fix missing data")
        assert _CLEAN_PATTERNS.search("Fill the null values")

    def test_remove_duplicates_matches(self):
        from api.chat import _CLEAN_PATTERNS
        assert _CLEAN_PATTERNS.search("remove duplicate rows")
        assert _CLEAN_PATTERNS.search("deduplicate the dataset")
        assert _CLEAN_PATTERNS.search("drop duplicates")

    def test_filter_rows_matches(self):
        from api.chat import _CLEAN_PATTERNS
        assert _CLEAN_PATTERNS.search("remove rows where quantity < 0")
        assert _CLEAN_PATTERNS.search("drop rows where revenue is 0")
        assert _CLEAN_PATTERNS.search("filter out negative values")

    def test_cap_outliers_matches(self):
        from api.chat import _CLEAN_PATTERNS
        assert _CLEAN_PATTERNS.search("cap outliers in revenue")
        assert _CLEAN_PATTERNS.search("handle outliers in the data")
        assert _CLEAN_PATTERNS.search("remove outliers")

    def test_irrelevant_message_does_not_match(self):
        from api.chat import _CLEAN_PATTERNS
        assert not _CLEAN_PATTERNS.search("what is the average revenue?")
        assert not _CLEAN_PATTERNS.search("train a model to predict sales")

    def test_detect_clean_op_fill_median(self):
        from api.chat import _detect_clean_op
        op = _detect_clean_op("fill missing revenue with median", ["revenue", "region", "quantity"])
        assert op is not None
        assert op["operation"] == "fill_missing"
        assert op["column"] == "revenue"
        assert op["strategy"] == "median"

    def test_detect_clean_op_remove_duplicates(self):
        from api.chat import _detect_clean_op
        op = _detect_clean_op("remove duplicate rows", ["a", "b"])
        assert op is not None
        assert op["operation"] == "remove_duplicates"

    def test_detect_clean_op_filter_rows(self):
        from api.chat import _detect_clean_op
        op = _detect_clean_op("drop rows where quantity < 0", ["revenue", "quantity", "region"])
        assert op is not None
        assert op["operation"] == "filter_rows"
        assert op["column"] == "quantity"
        assert op["operator"] == "lt"
        assert op["value"] == 0.0

    def test_detect_clean_op_unrecognised_returns_none(self):
        from api.chat import _detect_clean_op
        op = _detect_clean_op("what is the average revenue?", ["revenue"])
        assert op is None
