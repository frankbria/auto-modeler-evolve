"""Tests for POST /api/projects/{id}/narrative — AI project narrative.

Covers:
- 404 for unknown project
- Basic response shape (project_id, narrative string, generated_at)
- Narrative includes project name
- Works at each workflow stage (no data, data only, trained model, deployed)
- Static fallback works without ANTHROPIC_API_KEY
- Context dict includes expected keys
"""

from __future__ import annotations

import io
import json
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

# ──────────────────────────────────────────────────────────────────────────────
# Helpers & Fixtures
# ──────────────────────────────────────────────────────────────────────────────

SAMPLE_CSV = b"""product,region,revenue,units
Widget A,North,1200.50,10
Widget B,South,850.00,8
Widget A,East,2100.75,18
Widget C,West,450.25,4
Widget B,North,1650.00,15
Widget A,South,980.00,9
Widget C,North,1100.25,11
Widget B,East,1750.00,16
Widget A,West,2300.50,20
Widget C,South,620.75,6
Widget A,North,1400.00,12
Widget B,West,900.00,9
Widget C,East,750.50,7
Widget A,South,1800.25,16
Widget B,North,2000.00,18
"""


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

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
    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_api_module
    models_api_module.MODELS_DIR = tmp_path / "models"

    import api.deploy as deploy_module
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"

    from main import app
    with TestClient(app) as c:
        yield c


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestProjectNarrativeBasic:

    def test_404_for_unknown_project(self, client):
        r = client.post("/api/projects/nonexistent-id/narrative")
        assert r.status_code == 404

    def test_narrative_for_empty_project(self, client):
        """Project with no data still generates a narrative."""
        proj = client.post("/api/projects", json={"name": "Empty Project"}).json()
        r = client.post(f"/api/projects/{proj['id']}/narrative")
        assert r.status_code == 200
        body = r.json()
        assert body["project_id"] == proj["id"]
        assert body["project_name"] == "Empty Project"
        assert isinstance(body["narrative"], str)
        assert len(body["narrative"]) > 20

    def test_narrative_includes_project_name(self, client):
        proj = client.post("/api/projects", json={"name": "Revenue Analysis 2024"}).json()
        r = client.post(f"/api/projects/{proj['id']}/narrative")
        body = r.json()
        assert "Revenue Analysis 2024" in body["narrative"]

    def test_narrative_has_generated_at(self, client):
        proj = client.post("/api/projects", json={"name": "Test"}).json()
        r = client.post(f"/api/projects/{proj['id']}/narrative")
        body = r.json()
        assert "generated_at" in body
        assert body["generated_at"] is not None

    def test_narrative_has_context(self, client):
        proj = client.post("/api/projects", json={"name": "Test"}).json()
        r = client.post(f"/api/projects/{proj['id']}/narrative")
        body = r.json()
        assert "context" in body
        assert isinstance(body["context"], dict)

    def test_narrative_context_has_project_info(self, client):
        proj = client.post("/api/projects", json={"name": "Context Test"}).json()
        r = client.post(f"/api/projects/{proj['id']}/narrative")
        ctx = r.json()["context"]
        assert ctx["project_name"] == "Context Test"
        assert "created_at" in ctx


class TestProjectNarrativeWithData:

    def _upload(self, client, project_id):
        r = client.post(
            "/api/data/upload",
            files={"file": ("data.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
            data={"project_id": project_id},
        )
        return r.json()["dataset_id"]

    def test_narrative_with_dataset(self, client):
        proj = client.post("/api/projects", json={"name": "Sales Analysis"}).json()
        self._upload(client, proj["id"])
        r = client.post(f"/api/projects/{proj['id']}/narrative")
        assert r.status_code == 200
        body = r.json()
        assert "dataset" in body["context"]
        assert body["context"]["dataset"]["filename"] == "data.csv"

    def test_narrative_dataset_context_has_rows(self, client):
        proj = client.post("/api/projects", json={"name": "Dataset Test"}).json()
        self._upload(client, proj["id"])
        ctx = client.post(f"/api/projects/{proj['id']}/narrative").json()["context"]
        assert ctx["dataset"]["rows"] > 0
        assert ctx["dataset"]["columns"] > 0

    def test_narrative_mentions_rows_in_text(self, client):
        proj = client.post("/api/projects", json={"name": "Row Test"}).json()
        self._upload(client, proj["id"])
        narrative = client.post(f"/api/projects/{proj['id']}/narrative").json()["narrative"]
        # Static narrative includes row count
        assert "15" in narrative or "rows" in narrative.lower() or "data.csv" in narrative

    def test_narrative_with_feature_set_and_target(self, client):
        proj = client.post("/api/projects", json={"name": "With Features"}).json()
        did = self._upload(client, proj["id"])
        fs = client.post(f"/api/features/{did}/apply", json={"transformations": []}).json()
        client.post(f"/api/features/{did}/target", json={
            "target_column": "revenue",
            "feature_set_id": fs["feature_set_id"],
        })
        r = client.post(f"/api/projects/{proj['id']}/narrative")
        assert r.status_code == 200
        ctx = r.json()["context"]
        assert "features" in ctx
        assert ctx["features"]["target_column"] == "revenue"

    def test_narrative_features_context_has_problem_type(self, client):
        proj = client.post("/api/projects", json={"name": "Problem Type Test"}).json()
        did = self._upload(client, proj["id"])
        fs = client.post(f"/api/features/{did}/apply", json={"transformations": []}).json()
        client.post(f"/api/features/{did}/target", json={
            "target_column": "revenue",
            "feature_set_id": fs["feature_set_id"],
        })
        ctx = client.post(f"/api/projects/{proj['id']}/narrative").json()["context"]
        assert ctx["features"]["problem_type"] in ("regression", "classification", None)


class TestProjectNarrativeWithModel:

    def _full_setup(self, client):
        proj = client.post("/api/projects", json={"name": "Full Pipeline"}).json()
        pid = proj["id"]

        r = client.post(
            "/api/data/upload",
            files={"file": ("d.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
            data={"project_id": pid},
        )
        did = r.json()["dataset_id"]
        fs = client.post(f"/api/features/{did}/apply", json={"transformations": []}).json()
        client.post(f"/api/features/{did}/target", json={
            "target_column": "revenue",
            "feature_set_id": fs["feature_set_id"],
        })
        client.post(f"/api/models/{pid}/train", json={"algorithms": ["linear_regression"]})
        for _ in range(60):
            runs = client.get(f"/api/models/{pid}/runs").json()["runs"]
            if all(r["status"] in ("done", "failed") for r in runs):
                break
            time.sleep(0.1)
        return pid, did

    def test_narrative_with_trained_model(self, client):
        pid, _ = self._full_setup(client)
        r = client.post(f"/api/projects/{pid}/narrative")
        assert r.status_code == 200
        ctx = r.json()["context"]
        assert "model" in ctx
        assert ctx["model"]["algorithm"] is not None
        assert ctx["model"]["n_models_compared"] >= 1

    def test_narrative_model_context_has_metrics(self, client):
        pid, _ = self._full_setup(client)
        ctx = client.post(f"/api/projects/{pid}/narrative").json()["context"]
        assert "metrics" in ctx["model"]
        assert isinstance(ctx["model"]["metrics"], dict)

    def test_narrative_deployment_not_live_without_deploy(self, client):
        pid, _ = self._full_setup(client)
        ctx = client.post(f"/api/projects/{pid}/narrative").json()["context"]
        assert ctx["deployment"]["is_live"] is False

    def test_narrative_static_fallback_mentions_model(self, client):
        pid, _ = self._full_setup(client)
        narrative = client.post(f"/api/projects/{pid}/narrative").json()["narrative"]
        # Static narrative should mention the algorithm
        assert "linear" in narrative.lower() or "regression" in narrative.lower()


class TestStaticNarrativeFunction:
    """Unit tests for _static_narrative() directly."""

    def test_static_narrative_empty_context(self):
        from api.projects import _static_narrative
        ctx = {"project_name": "Test", "created_at": "January 01, 2024"}
        result = _static_narrative(ctx)
        assert "Test" in result
        assert len(result) > 10

    def test_static_narrative_with_dataset(self):
        from api.projects import _static_narrative
        ctx = {
            "project_name": "Sales",
            "created_at": "March 15, 2026",
            "dataset": {"filename": "sales.csv", "rows": 1500, "columns": 8, "missing_pct": 2.5, "has_outliers": False},
        }
        result = _static_narrative(ctx)
        assert "sales.csv" in result
        assert "1,500" in result or "1500" in result

    def test_static_narrative_with_regression_model(self):
        from api.projects import _static_narrative
        ctx = {
            "project_name": "Revenue Forecast",
            "created_at": "March 15, 2026",
            "dataset": {"filename": "d.csv", "rows": 200, "columns": 5, "missing_pct": 0, "has_outliers": False},
            "model": {"algorithm": "random_forest_regressor", "metrics": {"r2": 0.87, "mae": 120.5}, "summary": "Good fit.", "is_selected": True, "n_models_compared": 2},
        }
        result = _static_narrative(ctx)
        assert "0.87" in result or "87" in result

    def test_static_narrative_with_classification_model(self):
        from api.projects import _static_narrative
        ctx = {
            "project_name": "Churn Prediction",
            "created_at": "March 15, 2026",
            "model": {"algorithm": "random_forest_classifier", "metrics": {"accuracy": 0.92, "f1": 0.91}, "summary": "Great.", "is_selected": True, "n_models_compared": 1},
        }
        result = _static_narrative(ctx)
        assert "92" in result or "92.0" in result

    def test_static_narrative_with_deployment(self):
        from api.projects import _static_narrative
        ctx = {
            "project_name": "Live Model",
            "created_at": "March 01, 2026",
            "deployment": {"is_live": True, "endpoint": "/api/predict/abc", "dashboard_url": "/predict/abc", "prediction_count": 42, "created_at": "March 10, 2026"},
        }
        result = _static_narrative(ctx)
        assert "live" in result.lower() or "42" in result

    def test_static_narrative_not_deployed(self):
        from api.projects import _static_narrative
        ctx = {
            "project_name": "Pending",
            "created_at": "March 15, 2026",
            "deployment": {"is_live": False},
        }
        result = _static_narrative(ctx)
        assert "not yet" in result.lower() or "deployed" in result.lower() or "deploy" in result.lower()
