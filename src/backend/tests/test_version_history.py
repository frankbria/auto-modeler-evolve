"""Tests for the data version history feature.

Covers:
- Pure function: compute_version_history edge cases
- API: GET endpoint (via chat pattern detection)
- Chat: _VERSION_HISTORY_PATTERNS detects intent correctly
"""

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module


# ---------------------------------------------------------------------------
# Test DB fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.project  # noqa
    import models.dataset  # noqa
    import models.feature_set  # noqa
    import models.conversation  # noqa
    import models.model_run  # noqa
    import models.deployment  # noqa
    import models.prediction_log  # noqa
    import models.feedback_record  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_api_module

    models_api_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as c:
        yield c


def _make_ds(idx, filename="data.csv", row_count=100, col_count=5, size=1024):
    return {
        "id": f"ds-{idx}",
        "filename": filename,
        "row_count": row_count,
        "column_count": col_count,
        "uploaded_at": f"2024-0{idx}-01T00:00:00",
        "size_bytes": size,
    }


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestComputeVersionHistory:
    def test_empty_datasets_returns_zero_versions(self):
        from core.analyzer import compute_version_history

        result = compute_version_history([], [])
        assert result["version_count"] == 0
        assert result["versions"] == []
        assert result["overall_stability"] == "stable"

    def test_single_dataset_no_drift(self):
        from core.analyzer import compute_version_history

        ds = [_make_ds(1, filename="sales.csv", row_count=200)]
        df = pd.DataFrame({"revenue": [100, 200, 300]})
        result = compute_version_history(ds, [df])
        assert result["version_count"] == 1
        assert len(result["versions"]) == 1
        assert result["versions"][0]["drift_from_previous"] is None
        assert result["versions"][0]["version"] == 1
        assert result["versions"][0]["filename"] == "sales.csv"

    def test_two_identical_datasets_stable(self):
        from core.analyzer import compute_version_history

        ds = [_make_ds(1), _make_ds(2)]
        df1 = pd.DataFrame(
            {"revenue": [100.0, 200.0, 300.0], "region": ["East", "West", "East"]}
        )
        df2 = df1.copy()
        result = compute_version_history(ds, [df1, df2])
        assert result["version_count"] == 2
        # Second version has drift_from_previous
        v2 = result["versions"][1]
        assert v2["drift_from_previous"] is not None
        assert v2["drift_from_previous"]["drift_score"] == 0
        assert result["overall_stability"] == "stable"

    def test_high_drift_detected(self):
        from core.analyzer import compute_version_history

        ds = [_make_ds(1), _make_ds(2)]
        df1 = pd.DataFrame({"revenue": [100.0, 200.0, 300.0]})
        # 5x revenue shift → high drift
        df2 = pd.DataFrame({"revenue": [500.0, 1000.0, 1500.0]})
        result = compute_version_history(ds, [df1, df2])
        v2 = result["versions"][1]
        assert v2["drift_from_previous"]["drift_score"] > 0
        assert result["overall_stability"] in ("stable", "moderate", "high")

    def test_required_fields_present(self):
        from core.analyzer import compute_version_history

        ds = [_make_ds(1, size=2048)]
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = compute_version_history(ds, [df])
        v = result["versions"][0]
        assert "version" in v
        assert "dataset_id" in v
        assert "filename" in v
        assert "row_count" in v
        assert "column_count" in v
        assert "uploaded_at" in v
        assert "size_bytes" in v
        assert "drift_from_previous" in v

    def test_version_numbers_sequential(self):
        from core.analyzer import compute_version_history

        ds = [_make_ds(i) for i in range(1, 4)]
        dfs = [pd.DataFrame({"x": [i * 10, i * 20]}) for i in range(1, 4)]
        result = compute_version_history(ds, dfs)
        nums = [v["version"] for v in result["versions"]]
        assert nums == [1, 2, 3]

    def test_summary_present_in_output(self):
        from core.analyzer import compute_version_history

        ds = [_make_ds(1)]
        df = pd.DataFrame({"x": [1]})
        result = compute_version_history(ds, [df])
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_drift_fields_in_second_version(self):
        from core.analyzer import compute_version_history

        ds = [_make_ds(1), _make_ds(2)]
        df1 = pd.DataFrame({"revenue": [100.0, 200.0]})
        df2 = pd.DataFrame({"revenue": [110.0, 205.0]})
        result = compute_version_history(ds, [df1, df2])
        drift = result["versions"][1]["drift_from_previous"]
        assert "drift_score" in drift
        assert "summary" in drift
        assert "changed_columns" in drift
        assert "new_columns" in drift
        assert "dropped_columns" in drift
        assert "row_count_change_pct" in drift

    def test_new_dropped_columns_detected(self):
        from core.analyzer import compute_version_history

        ds = [_make_ds(1), _make_ds(2)]
        df1 = pd.DataFrame({"revenue": [100.0], "region": ["East"]})
        df2 = pd.DataFrame({"revenue": [100.0], "category": ["A"]})
        result = compute_version_history(ds, [df1, df2])
        drift = result["versions"][1]["drift_from_previous"]
        assert "region" in drift["dropped_columns"]
        assert "category" in drift["new_columns"]

    def test_overall_stability_stable_when_no_drift(self):
        from core.analyzer import compute_version_history

        ds = [_make_ds(i) for i in range(1, 3)]
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = compute_version_history(ds, [df, df.copy()])
        assert result["overall_stability"] == "stable"


# ---------------------------------------------------------------------------
# Chat pattern tests
# ---------------------------------------------------------------------------


class TestVersionHistoryPatterns:
    def _matches(self, text: str) -> bool:
        from api.chat import _VERSION_HISTORY_PATTERNS

        return bool(_VERSION_HISTORY_PATTERNS.search(text))

    def test_show_upload_history(self):
        assert self._matches("show my upload history")

    def test_data_version_timeline(self):
        assert self._matches("data version history")

    def test_how_many_uploads(self):
        assert self._matches("how many uploads do I have")

    def test_list_datasets(self):
        assert self._matches("what datasets do I have")

    def test_how_has_data_evolved(self):
        assert self._matches("how has my data evolved over time")

    def test_previous_uploads(self):
        assert self._matches("show my previous uploads")

    def test_upload_timeline(self):
        assert self._matches("upload timeline")

    def test_negative_unrelated_query(self):
        assert not self._matches("train a model")

    def test_negative_distribution_shift(self):
        # distribution shift is handled by dataset comparison, not version history
        assert not self._matches("distribution shift")


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


class TestVersionHistoryAPI:
    def _create_project(self, client) -> str:
        resp = client.post(
            "/api/projects/", json={"name": "History Test", "description": ""}
        )
        assert resp.status_code == 201
        return resp.json()["id"]

    def _upload_csv(
        self, client, project_id: str, rows: list[dict], filename: str
    ) -> dict:
        import csv
        import io

        if not rows:
            return {}
        fieldnames = list(rows[0].keys())
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        buf.seek(0)
        resp = client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": (filename, buf.getvalue().encode(), "text/csv")},
        )
        assert resp.status_code == 201
        return resp.json()

    def test_version_history_endpoint_single_upload(self, client):
        pid = self._create_project(client)
        self._upload_csv(client, pid, [{"revenue": 100}], "v1.csv")
        resp = client.get(f"/api/data/{pid}/version-history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version_count"] == 1
        assert len(data["versions"]) == 1

    def test_version_history_endpoint_two_uploads(self, client):
        pid = self._create_project(client)
        self._upload_csv(client, pid, [{"revenue": 100}], "v1.csv")
        self._upload_csv(client, pid, [{"revenue": 500}], "v2.csv")
        resp = client.get(f"/api/data/{pid}/version-history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["version_count"] == 2
        v2 = data["versions"][1]
        assert v2["drift_from_previous"] is not None

    def test_version_history_unknown_project_404(self, client):
        resp = client.get("/api/data/nonexistent-project/version-history")
        assert resp.status_code == 404
