"""Tests for GET /api/deploy/{id}/integration — developer code-snippet endpoint."""

import io
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

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
"""

CLASSIFICATION_CSV = b"""feature1,feature2,label
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

    import api.deploy as deploy_module
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"

    from main import app
    with TestClient(app) as c:
        yield c


def _setup_and_deploy(client, csv_bytes=SAMPLE_CSV, target="revenue", algo="linear_regression"):
    """Helper: upload → feature apply → set target → train → deploy → return deployment_id."""
    proj = client.post("/api/projects", json={"name": "Snippet Test"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(f"/api/features/{dataset_id}/target", json={"target_column": target})

    train = client.post(f"/api/models/{project_id}/train", json={"algorithms": [algo]})
    assert train.status_code == 202
    run_id = train.json()["model_run_ids"][0]

    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next(r for r in runs if r["id"] == run_id)
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    assert run["status"] == "done"

    deploy = client.post(f"/api/deploy/{run_id}")
    assert deploy.status_code == 201
    return deploy.json()["id"]


class TestIntegrationSnippets:
    def test_integration_returns_200(self, client):
        dep_id = _setup_and_deploy(client)
        resp = client.get(f"/api/deploy/{dep_id}/integration")
        assert resp.status_code == 200

    def test_integration_has_required_fields(self, client):
        dep_id = _setup_and_deploy(client)
        data = client.get(f"/api/deploy/{dep_id}/integration").json()
        required = ["deployment_id", "endpoint_url", "example_input", "curl", "python", "javascript", "openapi_url", "batch_url", "batch_note"]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_endpoint_url_contains_deployment_id(self, client):
        dep_id = _setup_and_deploy(client)
        data = client.get(f"/api/deploy/{dep_id}/integration").json()
        assert dep_id in data["endpoint_url"]

    def test_example_input_has_feature_keys(self, client):
        dep_id = _setup_and_deploy(client)
        data = client.get(f"/api/deploy/{dep_id}/integration").json()
        # Should have keys matching the feature columns
        assert isinstance(data["example_input"], dict)
        assert len(data["example_input"]) > 0

    def test_curl_snippet_contains_endpoint(self, client):
        dep_id = _setup_and_deploy(client)
        data = client.get(f"/api/deploy/{dep_id}/integration").json()
        assert dep_id in data["curl"]
        assert "POST" in data["curl"]
        assert "Content-Type" in data["curl"]

    def test_python_snippet_contains_endpoint(self, client):
        dep_id = _setup_and_deploy(client)
        data = client.get(f"/api/deploy/{dep_id}/integration").json()
        assert dep_id in data["python"]
        assert "requests" in data["python"]
        assert "prediction" in data["python"]

    def test_javascript_snippet_contains_endpoint(self, client):
        dep_id = _setup_and_deploy(client)
        data = client.get(f"/api/deploy/{dep_id}/integration").json()
        assert dep_id in data["javascript"]
        assert "fetch" in data["javascript"]
        assert "prediction" in data["javascript"]

    def test_regression_python_has_confidence_interval_code(self, client):
        dep_id = _setup_and_deploy(client)
        data = client.get(f"/api/deploy/{dep_id}/integration").json()
        assert data["problem_type"] == "regression"
        assert "confidence_interval" in data["python"]

    def test_regression_js_has_confidence_interval_code(self, client):
        dep_id = _setup_and_deploy(client)
        data = client.get(f"/api/deploy/{dep_id}/integration").json()
        assert "confidence_interval" in data["javascript"]

    def test_classification_python_has_confidence_code(self, client):
        dep_id = _setup_and_deploy(
            client,
            csv_bytes=CLASSIFICATION_CSV,
            target="label",
            algo="random_forest_classifier",
        )
        data = client.get(f"/api/deploy/{dep_id}/integration").json()
        assert data["problem_type"] == "classification"
        assert "confidence" in data["python"]
        assert "confidence_interval" not in data["python"]

    def test_custom_base_url(self, client):
        dep_id = _setup_and_deploy(client)
        resp = client.get(
            f"/api/deploy/{dep_id}/integration",
            params={"base_url": "https://api.mycompany.com"},
        )
        data = resp.json()
        assert "https://api.mycompany.com" in data["endpoint_url"]
        assert "https://api.mycompany.com" in data["curl"]
        assert "https://api.mycompany.com" in data["python"]
        assert "https://api.mycompany.com" in data["javascript"]

    def test_openapi_url_points_to_docs(self, client):
        dep_id = _setup_and_deploy(client)
        data = client.get(f"/api/deploy/{dep_id}/integration").json()
        assert data["openapi_url"].endswith("/docs")

    def test_batch_url_contains_deployment_id(self, client):
        dep_id = _setup_and_deploy(client)
        data = client.get(f"/api/deploy/{dep_id}/integration").json()
        assert dep_id in data["batch_url"]
        assert "batch" in data["batch_url"]

    def test_batch_note_mentions_csv(self, client):
        dep_id = _setup_and_deploy(client)
        data = client.get(f"/api/deploy/{dep_id}/integration").json()
        assert "csv" in data["batch_note"].lower() or "CSV" in data["batch_note"]

    def test_algorithm_and_target_in_response(self, client):
        dep_id = _setup_and_deploy(client)
        data = client.get(f"/api/deploy/{dep_id}/integration").json()
        assert data["algorithm"] == "linear_regression"
        assert data["target_column"] == "revenue"

    def test_404_for_nonexistent_deployment(self, client):
        resp = client.get("/api/deploy/nonexistent-id/integration")
        assert resp.status_code == 404

    def test_example_input_values_are_sensible(self, client):
        """Example input values should be 1.0 for numeric or 'value' for categorical."""
        dep_id = _setup_and_deploy(client)
        data = client.get(f"/api/deploy/{dep_id}/integration").json()
        example = data["example_input"]
        for val in example.values():
            assert val in (1.0, "value"), f"Unexpected example value: {val}"

    def test_deployment_id_in_response_matches_request(self, client):
        dep_id = _setup_and_deploy(client)
        data = client.get(f"/api/deploy/{dep_id}/integration").json()
        assert data["deployment_id"] == dep_id
