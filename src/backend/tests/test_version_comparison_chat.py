"""Tests for Deployment Version Comparison Chat Integration.

Covers:
- _VERSION_COMPARE_PATTERNS regex (positive + negative)
- Chat handler emitting version_comparison SSE event when 2+ versions exist
- No-comparison event emitted when only 1 version exists
- No version_comparison event when no deployment exists
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
    proj = client.post("/api/projects", json={"name": "VersionCompare Test"})
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


def _make_second_version(client, project_id):
    """Train a second model with a different algorithm and deploy it (creates v2)."""
    train2 = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["random_forest_regressor"]},
    )
    assert train2.status_code == 202
    run_id2 = train2.json()["model_run_ids"][0]

    for _ in range(60):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next((r for r in runs if r["id"] == run_id2), None)
        if run and run["status"] in ("done", "failed"):
            break
        time.sleep(0.2)

    assert run and run["status"] == "done"

    # Deploy the new run — execute_deployment archives v1 and creates v2
    redeploy = client.post(f"/api/deploy/{run_id2}")
    assert redeploy.status_code in (200, 201)
    return redeploy.json()["id"]


def _chat(client, project_id, message):
    with mock.patch("anthropic.Anthropic") as mock_ant:
        mock_client = mock.MagicMock()
        mock_ant.return_value = mock_client
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Comparison shown."])
        mock_client.messages.stream.return_value = mock_stream

        resp = client.post(f"/api/chat/{project_id}", json={"message": message})
    return resp


def _parse_events(resp):
    return [
        json.loads(line[6:])
        for line in resp.text.splitlines()
        if line.startswith("data: ") and line[6:].strip()
    ]


# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


class TestVersionComparePatterns:
    def setup_method(self):
        from api.chat import _VERSION_COMPARE_PATTERNS

        self.pattern = _VERSION_COMPARE_PATTERNS

    def test_compare_deployment_versions(self):
        assert self.pattern.search("compare my deployment versions")

    def test_how_did_retrain_improve(self):
        assert self.pattern.search("how did my retrain improve")

    def test_did_retrain_help(self):
        assert self.pattern.search("did my retrain help")

    def test_how_much_improved_after_retrain(self):
        assert self.pattern.search("how much improved after the retrain")

    def test_current_version_vs_previous(self):
        assert self.pattern.search("current version vs previous")

    def test_previous_model_vs_current(self):
        assert self.pattern.search("previous model vs current")

    def test_show_version_history_metrics(self):
        assert self.pattern.search("show my version history metrics")

    def test_is_new_version_better(self):
        assert self.pattern.search("is the new version better")

    # Negative tests — should NOT match
    def test_negative_train_model(self):
        assert not self.pattern.search("train a new model on my data")

    def test_negative_make_prediction(self):
        assert not self.pattern.search("make a prediction for units=100")


# ---------------------------------------------------------------------------
# Integration tests (chat SSE handler)
# ---------------------------------------------------------------------------


class TestVersionCompareChatHandler:
    def test_no_deployment_no_event(self, client):
        """No version_comparison event when no deployment exists."""
        proj = client.post("/api/projects", json={"name": "NoDepProj"})
        project_id = proj.json()["id"]

        resp = _chat(client, project_id, "compare my deployment versions")
        assert resp.status_code == 200
        events = _parse_events(resp)
        types = [e.get("type") for e in events]
        assert "version_comparison" not in types

    def test_single_version_emits_no_comparison(self, client):
        """When only 1 version exists, event has has_comparison=False."""
        project_id, _dep_id = _setup_and_deploy(client)

        resp = _chat(client, project_id, "compare my deployment versions")
        assert resp.status_code == 200
        events = _parse_events(resp)
        vc_events = [e for e in events if e.get("type") == "version_comparison"]
        assert len(vc_events) == 1
        assert vc_events[0]["version_comparison"]["has_comparison"] is False

    def test_two_versions_emits_comparison(self, client):
        """When 2+ versions exist, event has has_comparison=True with metric_diffs."""
        project_id, dep_id = _setup_and_deploy(client)
        _make_second_version(client, project_id)

        resp = _chat(client, project_id, "did my retrain improve things?")
        assert resp.status_code == 200
        events = _parse_events(resp)
        vc_events = [e for e in events if e.get("type") == "version_comparison"]
        assert len(vc_events) == 1
        data = vc_events[0]["version_comparison"]
        assert data["has_comparison"] is True
        assert "current_version" in data
        assert "previous_version" in data
        assert "summary" in data
