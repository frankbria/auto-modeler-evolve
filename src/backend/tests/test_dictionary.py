"""Tests for the AI data dictionary: core logic + API endpoints."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Core: classify_column_type
# ---------------------------------------------------------------------------


class TestClassifyColumnType:
    def _classify(self, name, dtype, unique, row_count, samples=None, null_pct=0.0):
        from core.dictionary import classify_column_type
        return classify_column_type(name, dtype, unique, row_count, samples or [], null_pct)

    def test_date_hint_in_name(self):
        assert self._classify("order_date", "object", 100, 100) == "date"

    def test_date_dtype(self):
        assert self._classify("ts", "datetime64[ns]", 100, 100) == "date"

    def test_created_at_hint(self):
        assert self._classify("created_at", "object", 100, 100) == "date"

    def test_flag_boolean_dtype(self):
        assert self._classify("is_active", "bool", 2, 100) == "flag"

    def test_flag_two_unique_values(self):
        assert self._classify("churned", "int64", 2, 100) == "flag"

    def test_id_name_hint(self):
        assert self._classify("customer_id", "int64", 100, 100) == "id"

    def test_id_high_cardinality_object(self):
        # 95% unique — should be id if no metric hint
        assert self._classify("uuid_key", "object", 95, 100) == "id"

    def test_metric_name_hint(self):
        assert self._classify("revenue", "float64", 50, 100) == "metric"

    def test_numeric_high_cardinality_is_metric(self):
        # float with 80% unique — metric even without name hint
        assert self._classify("x_value", "float64", 80, 100) == "metric"

    def test_dimension_low_cardinality_string(self):
        # 5 unique / 100 rows = 5% ratio → dimension
        assert self._classify("region", "object", 5, 100) == "dimension"

    def test_dimension_name_hint(self):
        assert self._classify("product_category", "object", 20, 100) == "dimension"

    def test_text_long_strings(self):
        long_sample = ["This is a very long description that exceeds sixty characters easily for testing"]
        assert self._classify("notes", "object", 90, 100, long_sample) == "text"

    def test_zero_rows_returns_unknown(self):
        assert self._classify("col", "int64", 0, 0) == "unknown"


# ---------------------------------------------------------------------------
# Core: generate_dictionary (static fallback — no Claude)
# ---------------------------------------------------------------------------


class TestGenerateDictionary:
    def _make_columns(self):
        return [
            {
                "name": "revenue",
                "dtype": "float64",
                "unique_count": 80,
                "null_pct": 0.0,
                "null_count": 0,
                "sample_values": [100.0, 200.0, 150.0],
                "min": 50.0,
                "max": 500.0,
                "mean": 200.0,
                "std": 50.0,
            },
            {
                "name": "region",
                "dtype": "object",
                "unique_count": 4,
                "null_pct": 2.5,
                "null_count": 2,
                "sample_values": ["North", "South", "East"],
            },
            {
                "name": "order_date",
                "dtype": "object",
                "unique_count": 90,
                "null_pct": 0.0,
                "null_count": 0,
                "sample_values": ["2024-01-01", "2024-01-02"],
            },
        ]

    def _gen(self, extra_kwargs=None):
        """Run generate_dictionary with Claude patched out."""
        from core.dictionary import generate_dictionary

        with patch("core.dictionary._call_claude_for_dictionary", return_value=None):
            return generate_dictionary(self._make_columns(), row_count=100)

    def test_returns_list_with_same_length(self):
        assert len(self._gen()) == 3

    def test_each_column_has_col_type(self):
        for col in self._gen():
            assert "col_type" in col

    def test_each_column_has_description(self):
        for col in self._gen():
            assert "description" in col
            assert len(col["description"]) > 10

    def test_revenue_classified_as_metric(self):
        revenue = next(c for c in self._gen() if c["name"] == "revenue")
        assert revenue["col_type"] == "metric"

    def test_region_classified_as_dimension(self):
        region = next(c for c in self._gen() if c["name"] == "region")
        assert region["col_type"] == "dimension"

    def test_order_date_classified_as_date(self):
        date_col = next(c for c in self._gen() if c["name"] == "order_date")
        assert date_col["col_type"] == "date"

    def test_missing_pct_mentioned_in_static_description(self):
        """Static fallback description mentions missing %."""
        from core.dictionary import generate_dictionary

        # Force static descriptions by patching Claude to return None
        with patch("core.dictionary._call_claude_for_dictionary", return_value=None):
            result = generate_dictionary(self._make_columns(), row_count=100)

        region = next(c for c in result if c["name"] == "region")
        assert "2.5" in region["description"] or "missing" in region["description"].lower()

    def test_claude_descriptions_override_fallback(self):
        """When Claude returns JSON, use those descriptions."""
        from core.dictionary import generate_dictionary

        mock_claude = {"revenue": "Total sales amount.", "region": "Geographic area.", "order_date": "When ordered."}

        with patch("core.dictionary._call_claude_for_dictionary", return_value=mock_claude):
            result = generate_dictionary(self._make_columns(), row_count=100)

        revenue = next(c for c in result if c["name"] == "revenue")
        assert revenue["description"] == "Total sales amount."

    def test_claude_failure_falls_back_to_static(self):
        """When Claude returns None, static descriptions are used."""
        from core.dictionary import generate_dictionary

        with patch("core.dictionary._call_claude_for_dictionary", return_value=None):
            result = generate_dictionary(self._make_columns(), row_count=100)

        for col in result:
            assert len(col["description"]) > 5


# ---------------------------------------------------------------------------
# API: GET + POST /api/data/{id}/dictionary
# ---------------------------------------------------------------------------


@pytest.fixture()
def dict_client(tmp_path):
    """Sync TestClient with a pre-uploaded dataset."""
    import api.data as data_module
    from main import app

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    CSV_CONTENT = (
        b"revenue,region,order_date,is_active\n"
        b"1000,North,2024-01-01,1\n"
        b"2000,South,2024-01-02,0\n"
        b"1500,East,2024-01-03,1\n"
    )

    with patch.object(data_module, "UPLOAD_DIR", upload_dir):
        with TestClient(app) as c:
            proj = c.post("/api/projects", json={"name": "DictTest"}).json()
            pid = proj["id"]
            r = c.post(
                "/api/data/upload",
                data={"project_id": pid},
                files={"file": ("sales.csv", CSV_CONTENT, "text/csv")},
            )
            assert r.status_code == 201, r.text
            did = r.json()["dataset_id"]
            yield c, pid, did


class TestDictionaryGetEndpoint:
    def test_get_returns_200(self, dict_client):
        c, _, did = dict_client
        resp = c.get(f"/api/data/{did}/dictionary")
        assert resp.status_code == 200

    def test_get_returns_expected_keys(self, dict_client):
        c, _, did = dict_client
        data = c.get(f"/api/data/{did}/dictionary").json()
        assert "dataset_id" in data
        assert "filename" in data
        assert "columns" in data
        assert "generated" in data

    def test_get_returns_all_columns(self, dict_client):
        c, _, did = dict_client
        data = c.get(f"/api/data/{did}/dictionary").json()
        col_names = {col["name"] for col in data["columns"]}
        assert "revenue" in col_names
        assert "region" in col_names

    def test_get_columns_have_col_type(self, dict_client):
        c, _, did = dict_client
        data = c.get(f"/api/data/{did}/dictionary").json()
        for col in data["columns"]:
            assert "col_type" in col

    def test_get_404_for_unknown_dataset(self, dict_client):
        c, _, _ = dict_client
        resp = c.get("/api/data/nonexistent-id/dictionary")
        assert resp.status_code == 404


class TestDictionaryPostEndpoint:
    def test_post_returns_200(self, dict_client):
        c, _, did = dict_client
        resp = c.post(f"/api/data/{did}/dictionary")
        assert resp.status_code == 200

    def test_post_returns_generated_true(self, dict_client):
        c, _, did = dict_client
        data = c.post(f"/api/data/{did}/dictionary").json()
        assert data["generated"] is True

    def test_post_persists_descriptions(self, dict_client):
        """After POST, GET should return descriptions stored in DB."""
        c, _, did = dict_client
        c.post(f"/api/data/{did}/dictionary")
        get_data = c.get(f"/api/data/{did}/dictionary").json()
        assert get_data["generated"] is True
        for col in get_data["columns"]:
            assert "description" in col

    def test_post_404_for_unknown_dataset(self, dict_client):
        c, _, _ = dict_client
        resp = c.post("/api/data/nonexistent-id/dictionary")
        assert resp.status_code == 404

    def test_post_column_descriptions_non_empty(self, dict_client):
        c, _, did = dict_client
        data = c.post(f"/api/data/{did}/dictionary").json()
        for col in data["columns"]:
            desc = col.get("description", "")
            assert isinstance(desc, str) and len(desc) > 5
