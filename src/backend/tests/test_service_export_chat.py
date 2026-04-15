"""Tests for Service Export Chat Integration.

Covers:
- _SERVICE_EXPORT_PATTERNS regex matching (positive and negative)
- Chat handler emitting service_export SSE event when deployment exists
- No service_export event emitted when deployment is absent
"""

from __future__ import annotations

import io
import json
import time
import unittest.mock as mock

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
    b"East,175.2,18\nWest,220.4,22\nNorth,190.6,19\nEast,130.8,13\nWest,280.0,28\n"
    b"East,160.0,16\nWest,210.0,21\n"
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.ab_test  # noqa
    import models.batch_schedule  # noqa
    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.dataset_filter  # noqa
    import models.deployment  # noqa
    import models.deployment_preset  # noqa
    import models.deployment_version  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa
    import models.webhook_config  # noqa
    import models.webhook_event  # noqa
    import models.analysis_template  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.deploy as deploy_module
    import api.models as models_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"
    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as c:
        yield c


def _setup_and_deploy(client):
    """Helper: create project → upload → features → target → train → deploy."""
    proj = client.post("/api/projects", json={"name": "SvcExport Test"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "problem_type": "regression"},
    )

    train = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    assert train.status_code == 202
    run_id = train.json()["model_run_ids"][0]

    for _ in range(60):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next((r for r in runs if r["id"] == run_id), None)
        if run and run["status"] in ("done", "failed"):
            break
        time.sleep(0.2)

    assert run and run["status"] == "done"

    deploy = client.post(f"/api/deploy/{run_id}")
    assert deploy.status_code == 201
    return project_id, deploy.json()["id"]


# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


class TestServiceExportPatterns:
    def setup_method(self):
        from api.chat import _SERVICE_EXPORT_PATTERNS

        self.pattern = _SERVICE_EXPORT_PATTERNS

    def test_package_my_model(self):
        assert self.pattern.search("package my model")

    def test_export_model_as_service(self):
        assert self.pattern.search("export my model as a service")

    def test_download_the_prediction_service(self):
        assert self.pattern.search("download the prediction service")

    def test_standalone_prediction_service(self):
        assert self.pattern.search("standalone prediction service")

    def test_self_contained_api(self):
        assert self.pattern.search("self-contained API")

    def test_deploy_elsewhere(self):
        assert self.pattern.search("deploy this elsewhere")

    def test_package_for_deployment(self):
        assert self.pattern.search("package my model for deployment")

    def test_export_prediction_service(self):
        assert self.pattern.search("export my prediction service")

    # Negative tests — should NOT match
    def test_negative_train_model(self):
        assert not self.pattern.search("train a new model on my data")

    def test_negative_make_prediction(self):
        assert not self.pattern.search("make a prediction for units=100")


# ---------------------------------------------------------------------------
# Integration tests (chat SSE handler)
# ---------------------------------------------------------------------------


class TestServiceExportChatHandler:
    def test_service_export_event_emitted_with_deployment(self, client):
        """service_export SSE event is emitted when deployment exists."""
        project_id, _dep_id = _setup_and_deploy(client)

        with mock.patch("anthropic.Anthropic") as mock_ant:
            mock_client = mock.MagicMock()
            mock_ant.return_value = mock_client
            mock_stream = mock.MagicMock()
            mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = mock.MagicMock(return_value=False)
            mock_stream.text_stream = iter(["Model packaged."])
            mock_client.messages.stream.return_value = mock_stream

            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "package my model for deployment"},
            )

        assert resp.status_code == 200
        events = [
            json.loads(line[6:])
            for line in resp.text.splitlines()
            if line.startswith("data: ") and line[6:].strip()
        ]
        types = [e.get("type") for e in events]
        assert "service_export" in types

    def test_service_export_event_has_required_fields(self, client):
        """service_export event contains deployment_id, download_url, included_files."""
        project_id, _dep_id = _setup_and_deploy(client)

        with mock.patch("anthropic.Anthropic") as mock_ant:
            mock_client = mock.MagicMock()
            mock_ant.return_value = mock_client
            mock_stream = mock.MagicMock()
            mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = mock.MagicMock(return_value=False)
            mock_stream.text_stream = iter(["Done."])
            mock_client.messages.stream.return_value = mock_stream

            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "export my model as a standalone service"},
            )

        events = [
            json.loads(line[6:])
            for line in resp.text.splitlines()
            if line.startswith("data: ") and line[6:].strip()
        ]
        svc_events = [e for e in events if e.get("type") == "service_export"]
        assert len(svc_events) == 1
        data = svc_events[0]["service_export"]
        for key in ("deployment_id", "download_url", "included_files", "target_column"):
            assert key in data, f"Missing key: {key}"
        assert data["download_url"].startswith("/api/deploy/")
        assert data["download_url"].endswith("/export")
        assert "server.py" in data["included_files"]
        assert "model.joblib" in data["included_files"]

    def test_no_service_export_without_deployment(self, client):
        """service_export event is NOT emitted when no deployment exists."""
        proj = client.post("/api/projects", json={"name": "No Deploy Project"})
        project_id = proj.json()["id"]

        # Upload only — no model, no deployment
        client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("sales.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        )

        with mock.patch("anthropic.Anthropic") as mock_ant:
            mock_client = mock.MagicMock()
            mock_ant.return_value = mock_client
            mock_stream = mock.MagicMock()
            mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = mock.MagicMock(return_value=False)
            mock_stream.text_stream = iter(["No deployment yet."])
            mock_client.messages.stream.return_value = mock_stream

            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "package my model"},
            )

        events = [
            json.loads(line[6:])
            for line in resp.text.splitlines()
            if line.startswith("data: ") and line[6:].strip()
        ]
        types = [e.get("type") for e in events]
        assert "service_export" not in types
