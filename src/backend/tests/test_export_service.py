"""Tests for GET /api/deploy/{id}/export — self-contained prediction service ZIP."""

import io
import time
import zipfile

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


def _setup_and_deploy(
    client, csv_bytes=SAMPLE_CSV, target="revenue", algo="linear_regression"
):
    """Helper: upload → apply features → set target → train → deploy → return dep_id."""
    proj = client.post("/api/projects", json={"name": "Export Test"})
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


class TestExportService:
    def test_export_returns_200(self, client):
        dep_id = _setup_and_deploy(client)
        resp = client.get(f"/api/deploy/{dep_id}/export")
        assert resp.status_code == 200

    def test_export_content_type_is_zip(self, client):
        dep_id = _setup_and_deploy(client)
        resp = client.get(f"/api/deploy/{dep_id}/export")
        assert "application/zip" in resp.headers["content-type"]

    def test_export_content_disposition_is_attachment(self, client):
        dep_id = _setup_and_deploy(client)
        resp = client.get(f"/api/deploy/{dep_id}/export")
        assert "attachment" in resp.headers["content-disposition"]
        assert ".zip" in resp.headers["content-disposition"]

    def test_export_zip_contains_required_files(self, client):
        dep_id = _setup_and_deploy(client)
        resp = client.get(f"/api/deploy/{dep_id}/export")
        assert resp.status_code == 200
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
        assert "server.py" in names
        assert "model_pipeline.joblib" in names
        assert "model.joblib" in names
        assert "requirements.txt" in names
        assert "README.md" in names

    def test_server_py_is_valid_python(self, client):
        dep_id = _setup_and_deploy(client)
        resp = client.get(f"/api/deploy/{dep_id}/export")
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            server_src = zf.read("server.py").decode("utf-8")
        # Should be syntactically valid Python
        compile(server_src, "server.py", "exec")

    def test_server_py_contains_fastapi_app(self, client):
        dep_id = _setup_and_deploy(client)
        resp = client.get(f"/api/deploy/{dep_id}/export")
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            server_src = zf.read("server.py").decode("utf-8")
        assert "FastAPI" in server_src
        assert "predict" in server_src
        assert "joblib" in server_src

    def test_server_py_has_predict_endpoint(self, client):
        dep_id = _setup_and_deploy(client)
        resp = client.get(f"/api/deploy/{dep_id}/export")
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            server_src = zf.read("server.py").decode("utf-8")
        assert '@app.post("/predict"' in server_src

    def test_server_py_has_health_endpoint(self, client):
        dep_id = _setup_and_deploy(client)
        resp = client.get(f"/api/deploy/{dep_id}/export")
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            server_src = zf.read("server.py").decode("utf-8")
        assert '@app.get("/health"' in server_src

    def test_requirements_txt_has_core_deps(self, client):
        dep_id = _setup_and_deploy(client)
        resp = client.get(f"/api/deploy/{dep_id}/export")
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            reqs = zf.read("requirements.txt").decode("utf-8")
        for pkg in ["fastapi", "uvicorn", "scikit-learn", "numpy", "pandas", "joblib"]:
            assert pkg in reqs, f"Missing dependency: {pkg}"

    def test_readme_contains_target_column(self, client):
        dep_id = _setup_and_deploy(client)
        resp = client.get(f"/api/deploy/{dep_id}/export")
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            readme = zf.read("README.md").decode("utf-8")
        assert "revenue" in readme

    def test_readme_contains_algorithm(self, client):
        dep_id = _setup_and_deploy(client)
        resp = client.get(f"/api/deploy/{dep_id}/export")
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            readme = zf.read("README.md").decode("utf-8")
        assert "linear_regression" in readme

    def test_readme_contains_uvicorn_instructions(self, client):
        dep_id = _setup_and_deploy(client)
        resp = client.get(f"/api/deploy/{dep_id}/export")
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            readme = zf.read("README.md").decode("utf-8")
        assert "uvicorn" in readme.lower()

    def test_pipeline_file_is_valid_joblib(self, client):
        dep_id = _setup_and_deploy(client)
        resp = client.get(f"/api/deploy/{dep_id}/export")
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            pipeline_bytes = zf.read("model_pipeline.joblib")
        import joblib

        pipeline = joblib.load(io.BytesIO(pipeline_bytes))
        assert hasattr(pipeline, "feature_names")
        assert hasattr(pipeline, "transform")

    def test_model_file_is_valid_joblib(self, client):
        dep_id = _setup_and_deploy(client)
        resp = client.get(f"/api/deploy/{dep_id}/export")
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            model_bytes = zf.read("model.joblib")
        import joblib

        model = joblib.load(io.BytesIO(model_bytes))
        assert hasattr(model, "predict")

    def test_classification_export_works(self, client):
        dep_id = _setup_and_deploy(
            client,
            csv_bytes=CLASSIFICATION_CSV,
            target="label",
            algo="random_forest_classifier",
        )
        resp = client.get(f"/api/deploy/{dep_id}/export")
        assert resp.status_code == 200
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
        assert "server.py" in names
        assert "model.joblib" in names

    def test_export_pipeline_and_model_usable_for_prediction(self, client):
        """Exported pipeline + model can make a real prediction."""
        dep_id = _setup_and_deploy(client)
        resp = client.get(f"/api/deploy/{dep_id}/export")
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            pipeline_bytes = zf.read("model_pipeline.joblib")
            model_bytes = zf.read("model.joblib")

        import joblib

        pipeline = joblib.load(io.BytesIO(pipeline_bytes))
        model = joblib.load(io.BytesIO(model_bytes))

        # Build a sample prediction input using training medians
        sample = {
            f: pipeline.medians.get(f, 0.0)
            for f in pipeline.feature_names
            if pipeline.column_types.get(f) == "numeric"
        }
        for f in pipeline.feature_names:
            if pipeline.column_types.get(f) == "categorical":
                le = pipeline.label_encoders.get(f)
                sample[f] = le.classes_[0] if le and len(le.classes_) else "unknown"

        X = pipeline.transform(sample)
        pred = model.predict(X)
        assert len(pred) == 1

    def test_404_for_nonexistent_deployment(self, client):
        resp = client.get("/api/deploy/nonexistent-id/export")
        assert resp.status_code == 404

    def test_filename_contains_target_and_algorithm(self, client):
        dep_id = _setup_and_deploy(client)
        resp = client.get(f"/api/deploy/{dep_id}/export")
        disposition = resp.headers["content-disposition"]
        assert "revenue" in disposition
        assert "linear_regression" in disposition
