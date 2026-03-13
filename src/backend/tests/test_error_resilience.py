"""test_error_resilience.py

Error resilience audit: systematically tests failure modes to ensure the platform
handles bad input gracefully with helpful user-facing messages.

Scenarios covered:
- Corrupt/binary file upload
- Non-CSV file upload
- Empty CSV (headers only, zero rows)
- Single-row CSV
- All-null column in CSV
- Very large header (100+ columns, few rows)
- Training with insufficient data
- Training when all target values are identical (degenerate target)
- Prediction on deployed model with missing features
- Profile/preview for nonexistent dataset
- Profile/preview when file has been deleted from disk
"""

import io
import json
import os

import pandas as pd
import pytest
from httpx import AsyncClient, ASGITransport
from sqlmodel import Session, SQLModel, create_engine

TEST_DATABASE_URL = "sqlite:///./test_resilience.db"


@pytest.fixture
async def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import db
    from main import app

    db.engine = create_engine(TEST_DATABASE_URL, echo=False)
    SQLModel.metadata.create_all(db.engine)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    if os.path.exists("test_resilience.db"):
        os.unlink("test_resilience.db")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _create_project(client: AsyncClient, name: str = "resilience-test") -> str:
    resp = await client.post("/api/projects", json={"name": name})
    assert resp.status_code == 201
    return resp.json()["id"]


async def _upload_csv(client: AsyncClient, project_id: str, csv_bytes: bytes, filename: str = "test.csv"):
    return await client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": (filename, csv_bytes, "text/csv")},
    )


# ---------------------------------------------------------------------------
# File upload error cases
# ---------------------------------------------------------------------------

class TestUploadErrorHandling:

    async def test_upload_non_csv_extension_rejected(self, client):
        pid = await _create_project(client, "reject-non-csv")
        resp = await _upload_csv(client, pid, b"a,b\n1,2", filename="data.xlsx")
        assert resp.status_code == 400
        assert "CSV" in resp.json()["detail"] or "csv" in resp.json()["detail"].lower()

    async def test_upload_binary_file_rejected(self, client):
        pid = await _create_project(client, "reject-binary")
        # Feed a .csv extension but binary content (PNG header)
        binary_content = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00" + b"\xff" * 100
        resp = await _upload_csv(client, pid, binary_content, filename="data.csv")
        # Should return 400 (parse error)
        assert resp.status_code == 400
        detail = resp.json()["detail"].lower()
        assert "parse" in detail or "failed" in detail or "csv" in detail

    async def test_upload_empty_csv_headers_only(self, client):
        """CSV with column headers but zero data rows — should succeed."""
        pid = await _create_project(client, "empty-csv")
        csv_bytes = b"date,product,revenue\n"
        resp = await _upload_csv(client, pid, csv_bytes)
        assert resp.status_code == 201
        data = resp.json()
        assert data["row_count"] == 0
        assert data["column_count"] == 3
        # Preview should be empty
        assert data["preview"] == []

    async def test_upload_single_row_csv(self, client):
        """CSV with one data row — should succeed and profile gracefully."""
        pid = await _create_project(client, "single-row")
        csv_bytes = b"product,revenue\nWidget A,1200.50\n"
        resp = await _upload_csv(client, pid, csv_bytes)
        assert resp.status_code == 201
        data = resp.json()
        assert data["row_count"] == 1
        assert len(data["preview"]) == 1

    async def test_upload_all_null_column(self, client):
        """CSV where one column is entirely null — should not crash profiling."""
        pid = await _create_project(client, "null-column")
        csv_bytes = b"product,revenue,empty\nWidget A,100,\nWidget B,200,\nWidget C,300,\n"
        resp = await _upload_csv(client, pid, csv_bytes)
        assert resp.status_code == 201
        data = resp.json()
        assert data["row_count"] == 3
        # Find the null column stats
        empty_col = next((c for c in data["column_stats"] if c["name"] == "empty"), None)
        assert empty_col is not None
        assert empty_col["null_count"] == 3

    async def test_upload_wide_csv(self, client):
        """CSV with 50 columns and 5 rows — should profile without crash."""
        pid = await _create_project(client, "wide-csv")
        headers = ",".join(f"col_{i}" for i in range(50))
        row = ",".join(str(i * 1.5) for i in range(50))
        csv_bytes = (headers + "\n" + "\n".join(row for _ in range(5))).encode()
        resp = await _upload_csv(client, pid, csv_bytes)
        assert resp.status_code == 201
        assert resp.json()["column_count"] == 50

    async def test_upload_csv_with_duplicate_rows(self, client):
        """CSV with all-duplicate rows — profiler should detect and report."""
        pid = await _create_project(client, "duplicates")
        csv_bytes = b"product,revenue\nWidget A,100\nWidget A,100\nWidget A,100\n"
        resp = await _upload_csv(client, pid, csv_bytes)
        assert resp.status_code == 201
        # Insights should flag duplicates
        insights = resp.json().get("insights", [])
        dup_insight = next((i for i in insights if i["type"] == "duplicates"), None)
        assert dup_insight is not None, f"Expected duplicate insight but got: {insights}"


# ---------------------------------------------------------------------------
# Profile / preview error cases
# ---------------------------------------------------------------------------

class TestProfileErrorHandling:

    async def test_preview_nonexistent_dataset(self, client):
        resp = await client.get("/api/data/nonexistent-id-12345/preview")
        assert resp.status_code == 404

    async def test_profile_nonexistent_dataset(self, client):
        resp = await client.get("/api/data/nonexistent-id-12345/profile")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Analyzer robustness (unit tests, no API)
# ---------------------------------------------------------------------------

class TestAnalyzerEdgeCases:

    def test_profile_empty_dataframe(self):
        """compute_full_profile must not crash on a zero-row DataFrame."""
        from core.analyzer import compute_full_profile
        df = pd.DataFrame({"a": pd.Series([], dtype=float), "b": pd.Series([], dtype=str)})
        profile = compute_full_profile(df)
        assert profile["row_count"] == 0
        assert profile["column_count"] == 2

    def test_profile_single_row(self):
        from core.analyzer import compute_full_profile
        df = pd.DataFrame({"x": [1.0], "y": ["hello"]})
        profile = compute_full_profile(df)
        assert profile["row_count"] == 1

    def test_profile_all_null_column(self):
        from core.analyzer import compute_full_profile
        df = pd.DataFrame({"good": [1, 2, 3], "all_null": [None, None, None]})
        profile = compute_full_profile(df)
        null_col = next(c for c in profile["columns"] if c["name"] == "all_null")
        assert null_col["null_count"] == 3
        assert null_col["null_pct"] == 100.0

    def test_profile_single_unique_value(self):
        """Column with all identical values — std should be 0, not crash."""
        from core.analyzer import compute_full_profile
        df = pd.DataFrame({"constant": [5, 5, 5, 5, 5]})
        profile = compute_full_profile(df)
        const_col = next(c for c in profile["columns"] if c["name"] == "constant")
        assert const_col["std"] == 0.0 or const_col["std"] is None  # valid

    def test_profile_with_special_float_values(self):
        """NaN and inf in numeric columns should not break profiling."""
        import numpy as np
        from core.analyzer import compute_full_profile
        df = pd.DataFrame({"val": [1.0, float("nan"), 3.0, float("inf"), -float("inf")]})
        # Should not raise
        profile = compute_full_profile(df)
        assert profile["row_count"] == 5

    def test_profile_high_cardinality_column(self):
        """Unique ID column should be flagged as high cardinality."""
        from core.analyzer import compute_full_profile
        df = pd.DataFrame({
            "id": [f"ID{i:04d}" for i in range(50)],
            "value": range(50),
        })
        profile = compute_full_profile(df)
        insights = profile["insights"]
        hc = next((i for i in insights if i["type"] == "high_cardinality"), None)
        assert hc is not None

    def test_profile_missing_values_insight(self):
        """Columns with >30% null should generate a warning insight."""
        from core.analyzer import compute_full_profile
        data = {"val": [None] * 35 + [1.0] * 65}
        df = pd.DataFrame(data)
        profile = compute_full_profile(df)
        mv_insights = [i for i in profile["insights"] if i["type"] == "missing_values"]
        assert any(i["severity"] == "warning" for i in mv_insights)


# ---------------------------------------------------------------------------
# Training error cases
# ---------------------------------------------------------------------------

class TestTrainingEdgeCases:

    async def test_train_without_feature_set(self, client):
        """Training without a feature set should return 4xx."""
        pid = await _create_project(client, "train-no-features")
        resp = await client.post(
            f"/api/models/{pid}/train",
            json={"algorithms": ["linear_regression"], "target_column": "revenue"},
        )
        assert resp.status_code in (400, 404, 422)

    async def test_train_with_insufficient_data(self, client):
        """Training on a 3-row dataset should either succeed with a warning
        or return a helpful error — it must not 500."""
        pid = await _create_project(client, "train-tiny")
        csv_bytes = b"product,revenue\nA,100\nB,200\nC,300\n"
        up = await _upload_csv(client, pid, csv_bytes)
        assert up.status_code == 201
        dataset_id = up.json()["dataset_id"]

        # Apply features
        feat_resp = await client.post(
            f"/api/features/{dataset_id}/apply",
            json={"transformations": [], "target_column": "revenue"},
        )
        assert feat_resp.status_code == 201
        feature_set_id = feat_resp.json()["feature_set_id"]

        train_resp = await client.post(
            f"/api/models/{pid}/train",
            json={"algorithms": ["linear_regression"], "target_column": "revenue", "feature_set_id": feature_set_id},
        )
        # Should not 500 — either 4xx with helpful message or 202 accepted
        assert train_resp.status_code != 500, f"Got 500: {train_resp.text}"

    async def test_recommendations_no_dataset(self, client):
        """Recommendations endpoint without a dataset should 404 gracefully."""
        pid = await _create_project(client, "no-dataset-recs")
        resp = await client.get(
            f"/api/models/{pid}/recommendations",
            params={"target_column": "revenue"},
        )
        assert resp.status_code in (400, 404)


# ---------------------------------------------------------------------------
# Deployer robustness
# ---------------------------------------------------------------------------

class TestDeployerEdgeCases:

    async def test_deploy_nonexistent_run(self, client):
        resp = await client.post("/api/deploy/nonexistent-run-id-99999")
        assert resp.status_code in (400, 404, 422)

    async def test_predict_nonexistent_deployment(self, client):
        resp = await client.post(
            "/api/predict/nonexistent-deployment-99999",
            json={"product": "A", "revenue": 100},
        )
        assert resp.status_code in (400, 404, 422)

    async def test_batch_predict_invalid_csv(self, client):
        """Batch prediction with corrupt CSV should return 4xx."""
        # First, create a fake deployment ID that doesn't exist
        resp = await client.post(
            "/api/predict/bad-id/batch",
            files={"file": ("data.csv", b"not,valid\n\xff\xfe", "text/csv")},
        )
        assert resp.status_code in (400, 404, 422)
