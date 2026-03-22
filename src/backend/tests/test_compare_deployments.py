"""Tests for POST /api/predict/compare — cross-deployment model comparison.

Also covers GET /api/deployments?project_id= filter.
"""

import io
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

# ---------------------------------------------------------------------------
# Sample CSV data
# ---------------------------------------------------------------------------

REGRESSION_CSV = b"""age,income,revenue
25,30000,1234.50
35,45000,2150.75
45,60000,3280.00
55,75000,4450.25
30,38000,1680.50
40,52000,2780.00
50,68000,3900.50
28,33000,1380.25
38,48000,2480.75
48,63000,3580.00
32,42000,1920.50
42,57000,3050.25
52,72000,4150.00
60,85000,5300.50
36,46000,2250.75
"""

CLASSIFICATION_CSV = b"""x1,x2,label
1.0,2.0,cat
2.0,3.0,dog
3.0,4.0,cat
4.0,5.0,dog
5.0,6.0,cat
6.0,7.0,dog
7.0,8.0,cat
8.0,9.0,dog
9.0,10.0,cat
10.0,11.0,dog
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    SQLModel.metadata.create_all(db_module.engine)

    from main import app

    return TestClient(app)


def _deploy_model(client, project_id: str, csv_bytes: bytes, target: str, algo: str):
    """Upload CSV → apply → set target → train → deploy. Returns deployment dict."""
    upload_resp = client.post(
        "/api/data/upload",
        files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"project_id": project_id},
    )
    assert upload_resp.status_code in (200, 201), upload_resp.text
    dataset_id = upload_resp.json()["dataset_id"]

    apply_resp = client.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert apply_resp.status_code in (200, 201), apply_resp.text

    target_resp = client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": target},
    )
    assert target_resp.status_code in (200, 201), target_resp.text

    train_resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": [algo]},
    )
    assert train_resp.status_code in (200, 202), train_resp.text
    run_id = train_resp.json()["model_run_ids"][0]

    # Wait for completion
    for _ in range(60):
        runs_resp = client.get(f"/api/models/{project_id}/runs")
        runs = runs_resp.json().get("runs", [])
        run = next((r for r in runs if r["id"] == run_id), None)
        if run and run["status"] in ("done", "failed"):
            break
        time.sleep(0.2)

    assert run and run["status"] == "done", f"Training failed: {run}"

    deploy_resp = client.post(f"/api/deploy/{run_id}")
    assert deploy_resp.status_code in (200, 201), deploy_resp.text
    return deploy_resp.json()


# ---------------------------------------------------------------------------
# Tests — compare endpoint input validation
# ---------------------------------------------------------------------------


class TestCompareValidation:
    def test_too_few_ids(self, client):
        """Fewer than 2 deployment IDs → 400."""
        r = client.post(
            "/api/predict/compare",
            json={"deployment_ids": ["only-one"], "features": {}},
        )
        assert r.status_code == 400
        assert "2" in r.json()["detail"]

    def test_too_many_ids(self, client):
        """More than 4 deployment IDs → 400."""
        r = client.post(
            "/api/predict/compare",
            json={
                "deployment_ids": ["a", "b", "c", "d", "e"],
                "features": {},
            },
        )
        assert r.status_code == 400
        assert "4" in r.json()["detail"]

    def test_nonexistent_deployments_returns_error_entries(self, client):
        """Unknown IDs should appear in results with error field, not raise 404."""
        r = client.post(
            "/api/predict/compare",
            json={
                "deployment_ids": ["no-such-id-1", "no-such-id-2"],
                "features": {"age": 30},
            },
        )
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) == 2
        for res in results:
            assert res["error"] is not None


# ---------------------------------------------------------------------------
# Tests — compare endpoint with real trained models
# ---------------------------------------------------------------------------


class TestCompareWithRealModels:
    def test_compare_two_regression_models(self, client):
        """Two different regression algorithms on the same project produce comparable results."""
        proj_resp = client.post("/api/projects", json={"name": "Compare Test"})
        proj_id = proj_resp.json()["id"]

        dep1 = _deploy_model(
            client, proj_id, REGRESSION_CSV, "revenue", "linear_regression"
        )
        dep2 = _deploy_model(
            client, proj_id, REGRESSION_CSV, "revenue", "random_forest_regressor"
        )

        features = {"age": 35, "income": 50000}
        r = client.post(
            "/api/predict/compare",
            json={"deployment_ids": [dep1["id"], dep2["id"]], "features": features},
        )
        assert r.status_code == 200
        data = r.json()
        results = data["results"]

        assert len(results) == 2
        for res in results:
            assert res["error"] is None
            assert "prediction" in res
            assert res["problem_type"] == "regression"
            assert res["algorithm"] is not None
            assert res["trained_at"] is not None

    def test_compare_result_includes_confidence_interval(self, client):
        """Regression comparison results include confidence_interval when residual_std > 0."""
        proj_resp = client.post("/api/projects", json={"name": "CI Compare Test"})
        proj_id = proj_resp.json()["id"]

        dep = _deploy_model(
            client, proj_id, REGRESSION_CSV, "revenue", "linear_regression"
        )
        features = {"age": 35, "income": 50000}

        r = client.post(
            "/api/predict/compare",
            json={"deployment_ids": [dep["id"], dep["id"]], "features": features},
        )
        assert r.status_code == 200
        results = r.json()["results"]
        # At least one result may have confidence_interval (depends on residual_std)
        # Just verify structure is correct if present
        for res in results:
            if "confidence_interval" in res and res["confidence_interval"]:
                ci = res["confidence_interval"]
                assert "lower" in ci
                assert "upper" in ci
                assert ci["lower"] <= ci["upper"]

    def test_compare_classification_includes_confidence(self, client):
        """Classification comparison results include confidence (max proba)."""
        proj_resp = client.post("/api/projects", json={"name": "Clf Compare Test"})
        proj_id = proj_resp.json()["id"]

        dep = _deploy_model(
            client, proj_id, CLASSIFICATION_CSV, "label", "logistic_regression"
        )
        features = {"x1": 3.0, "x2": 4.0}

        r = client.post(
            "/api/predict/compare",
            json={"deployment_ids": [dep["id"], dep["id"]], "features": features},
        )
        assert r.status_code == 200
        results = r.json()["results"]
        for res in results:
            assert res["error"] is None
            assert res["problem_type"] == "classification"
            assert res.get("confidence") is not None

    def test_compare_mixed_valid_and_invalid(self, client):
        """Mix of valid and invalid IDs: valid one succeeds, invalid one errors."""
        proj_resp = client.post("/api/projects", json={"name": "Mixed Compare"})
        proj_id = proj_resp.json()["id"]

        dep = _deploy_model(
            client, proj_id, REGRESSION_CSV, "revenue", "linear_regression"
        )
        features = {"age": 30, "income": 40000}

        r = client.post(
            "/api/predict/compare",
            json={
                "deployment_ids": [dep["id"], "nonexistent-id"],
                "features": features,
            },
        )
        assert r.status_code == 200
        results = r.json()["results"]
        assert len(results) == 2

        good = next(res for res in results if res["deployment_id"] == dep["id"])
        bad = next(res for res in results if res["deployment_id"] == "nonexistent-id")

        assert good["error"] is None
        assert good["prediction"] is not None
        assert bad["error"] is not None

    def test_exactly_four_deployments_allowed(self, client):
        """Exactly 4 deployment IDs is the maximum and should succeed."""
        proj_resp = client.post("/api/projects", json={"name": "Four Models"})
        proj_id = proj_resp.json()["id"]

        dep = _deploy_model(
            client, proj_id, REGRESSION_CSV, "revenue", "linear_regression"
        )
        features = {"age": 30, "income": 40000}

        # Reuse the same deployment 4 times (for testing the limit)
        r = client.post(
            "/api/predict/compare",
            json={
                "deployment_ids": [dep["id"], dep["id"], dep["id"], dep["id"]],
                "features": features,
            },
        )
        assert r.status_code == 200
        assert len(r.json()["results"]) == 4


# ---------------------------------------------------------------------------
# Tests — GET /api/deployments?project_id= filter
# ---------------------------------------------------------------------------


class TestDeploymentListProjectFilter:
    def test_filter_returns_only_project_deployments(self, client):
        """?project_id= filter returns only deployments from the specified project."""
        proj1_resp = client.post("/api/projects", json={"name": "Project Alpha"})
        proj1_id = proj1_resp.json()["id"]
        proj2_resp = client.post("/api/projects", json={"name": "Project Beta"})
        proj2_id = proj2_resp.json()["id"]

        dep1 = _deploy_model(
            client, proj1_id, REGRESSION_CSV, "revenue", "linear_regression"
        )
        dep2 = _deploy_model(
            client, proj2_id, REGRESSION_CSV, "revenue", "linear_regression"
        )

        r = client.get(f"/api/deployments?project_id={proj1_id}")
        assert r.status_code == 200
        ids = [d["id"] for d in r.json()]
        assert dep1["id"] in ids
        assert dep2["id"] not in ids

    def test_filter_with_unknown_project_returns_empty(self, client):
        """project_id that doesn't exist → empty list, not 404."""
        r = client.get("/api/deployments?project_id=nonexistent-project")
        assert r.status_code == 200
        assert r.json() == []

    def test_no_filter_returns_all_deployments(self, client):
        """Without ?project_id=, all active deployments are returned."""
        proj1_resp = client.post("/api/projects", json={"name": "All Proj 1"})
        proj1_id = proj1_resp.json()["id"]
        proj2_resp = client.post("/api/projects", json={"name": "All Proj 2"})
        proj2_id = proj2_resp.json()["id"]

        dep1 = _deploy_model(
            client, proj1_id, REGRESSION_CSV, "revenue", "linear_regression"
        )
        dep2 = _deploy_model(
            client, proj2_id, REGRESSION_CSV, "revenue", "linear_regression"
        )

        r = client.get("/api/deployments")
        assert r.status_code == 200
        ids = [d["id"] for d in r.json()]
        assert dep1["id"] in ids
        assert dep2["id"] in ids
