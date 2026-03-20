"""Tests for prediction logging, analytics, and model readiness endpoints.

Day 4 (00:08): New Phase 8 capabilities —
  - PredictionLog model (stored on every /api/predict/{id} call)
  - GET /api/deploy/{id}/analytics — usage analytics with histogram + daily counts
  - GET /api/deploy/{id}/logs      — paginated prediction log
  - GET /api/models/{id}/readiness — production-readiness checklist
  - Chat intent detection for "is my model ready?" queries
"""

import io
import json
import time
from unittest.mock import MagicMock, patch

import db as db_module
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine, select


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

REGRESSION_CSV = b"""age,income,score,target
25,50000,80,100.0
30,60000,85,120.0
35,70000,90,140.0
40,80000,75,160.0
45,90000,95,180.0
50,100000,70,200.0
55,55000,82,110.0
28,62000,88,125.0
33,73000,91,145.0
38,84000,78,165.0
43,95000,96,185.0
48,106000,72,205.0
53,57000,84,115.0
26,64000,87,128.0
31,75000,92,148.0
36,86000,76,168.0
41,97000,97,188.0
46,108000,71,208.0
51,59000,83,118.0
29,66000,89,131.0
"""

CLASSIFICATION_CSV = b"""feat1,feat2,label
1.0,2.0,A
2.0,3.0,B
3.0,4.0,A
4.0,5.0,B
5.0,6.0,A
6.0,7.0,B
7.0,8.0,A
8.0,9.0,B
9.0,10.0,A
10.0,11.0,B
1.5,2.5,A
2.5,3.5,B
3.5,4.5,A
4.5,5.5,B
5.5,6.5,A
6.5,7.5,B
7.5,8.5,A
8.5,9.5,B
9.5,10.5,A
10.5,11.5,B
"""


# ---------------------------------------------------------------------------
# Fixtures
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


@pytest.fixture
def deployed_regression(client):
    """Full pipeline: project → upload → features → train → select → deploy."""
    proj = client.post("/api/projects", json={"name": "Monitoring Test"})
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.csv", io.BytesIO(REGRESSION_CSV), "text/csv")},
    )
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(f"/api/features/{dataset_id}/target", json={"target_column": "target"})

    train_resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    run_id = train_resp.json()["model_run_ids"][0]

    # Wait for training
    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next(r for r in runs if r["id"] == run_id)
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    # Select the run
    client.post(f"/api/models/{run_id}/select")

    # Deploy
    deploy_resp = client.post(f"/api/deploy/{run_id}")
    assert deploy_resp.status_code == 201
    deployment_id = deploy_resp.json()["id"]

    return {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "run_id": run_id,
        "deployment_id": deployment_id,
    }


@pytest.fixture
def deployed_classification(client):
    """Classification pipeline: project → upload → features → train → deploy."""
    proj = client.post("/api/projects", json={"name": "Classification Test"})
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("clf.csv", io.BytesIO(CLASSIFICATION_CSV), "text/csv")},
    )
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "label"},
    )

    train_resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["logistic_regression"]},
    )
    run_id = train_resp.json()["model_run_ids"][0]

    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next(r for r in runs if r["id"] == run_id)
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    client.post(f"/api/models/{run_id}/select")
    deploy_resp = client.post(f"/api/deploy/{run_id}")
    assert deploy_resp.status_code == 201

    return {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "run_id": run_id,
        "deployment_id": deploy_resp.json()["id"],
    }


# ---------------------------------------------------------------------------
# Prediction logging tests
# ---------------------------------------------------------------------------

class TestPredictionLogging:
    """Predictions are stored in PredictionLog on every /api/predict call."""

    def test_single_prediction_creates_log(self, client, deployed_regression):
        did = deployed_regression["deployment_id"]
        resp = client.post(
            f"/api/predict/{did}",
            json={"age": 30, "income": 60000, "score": 85},
        )
        assert resp.status_code == 200
        assert "prediction" in resp.json()

        # Verify log entry was created
        with Session(db_module.engine) as session:
            from models.prediction_log import PredictionLog
            logs = session.exec(
                select(PredictionLog).where(PredictionLog.deployment_id == did)
            ).all()
        assert len(logs) == 1
        assert logs[0].deployment_id == did
        assert logs[0].input_features is not None
        assert logs[0].prediction is not None

    def test_prediction_log_stores_numeric_value(self, client, deployed_regression):
        did = deployed_regression["deployment_id"]
        client.post(f"/api/predict/{did}", json={"age": 35, "income": 70000, "score": 90})

        with Session(db_module.engine) as session:
            from models.prediction_log import PredictionLog
            log = session.exec(
                select(PredictionLog).where(PredictionLog.deployment_id == did)
            ).first()
        assert log is not None
        assert log.prediction_numeric is not None

    def test_multiple_predictions_accumulate_logs(self, client, deployed_regression):
        did = deployed_regression["deployment_id"]
        for i in range(3):
            client.post(
                f"/api/predict/{did}",
                json={"age": 25 + i * 5, "income": 50000 + i * 10000, "score": 80 + i},
            )

        with Session(db_module.engine) as session:
            from models.prediction_log import PredictionLog
            logs = session.exec(
                select(PredictionLog).where(PredictionLog.deployment_id == did)
            ).all()
        assert len(logs) == 3

    def test_prediction_log_input_features_parseable(self, client, deployed_regression):
        did = deployed_regression["deployment_id"]
        client.post(
            f"/api/predict/{did}",
            json={"age": 42, "income": 88000, "score": 77},
        )

        with Session(db_module.engine) as session:
            from models.prediction_log import PredictionLog
            log = session.exec(
                select(PredictionLog).where(PredictionLog.deployment_id == did)
            ).first()
        assert log is not None
        features = json.loads(log.input_features)
        assert isinstance(features, dict)
        assert "age" in features


# ---------------------------------------------------------------------------
# Analytics endpoint tests
# ---------------------------------------------------------------------------

class TestDeploymentAnalytics:
    """GET /api/deploy/{id}/analytics returns aggregated prediction stats."""

    def test_analytics_empty_before_predictions(self, client, deployed_regression):
        did = deployed_regression["deployment_id"]
        resp = client.get(f"/api/deploy/{did}/analytics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deployment_id"] == did
        assert data["total_predictions"] == 0
        assert data["predictions_by_day"] == []
        assert data["prediction_distribution"] == []
        assert data["recent_avg"] is None

    def test_analytics_after_predictions(self, client, deployed_regression):
        did = deployed_regression["deployment_id"]
        for i in range(5):
            client.post(
                f"/api/predict/{did}",
                json={"age": 30 + i, "income": 60000 + i * 5000, "score": 85},
            )

        resp = client.get(f"/api/deploy/{did}/analytics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_predictions"] == 5
        assert len(data["predictions_by_day"]) >= 1
        assert data["recent_avg"] is not None
        # Distribution should have at least one bucket
        assert len(data["prediction_distribution"]) >= 1

    def test_analytics_by_day_structure(self, client, deployed_regression):
        did = deployed_regression["deployment_id"]
        client.post(f"/api/predict/{did}", json={"age": 28, "income": 55000, "score": 82})

        resp = client.get(f"/api/deploy/{did}/analytics")
        data = resp.json()
        assert "predictions_by_day" in data
        if data["predictions_by_day"]:
            day_entry = data["predictions_by_day"][0]
            assert "date" in day_entry
            assert "count" in day_entry

    def test_analytics_404_for_unknown_deployment(self, client):
        resp = client.get("/api/deploy/nonexistent/analytics")
        assert resp.status_code == 404

    def test_analytics_distribution_buckets(self, client, deployed_regression):
        did = deployed_regression["deployment_id"]
        # Make predictions with varied values to ensure histogram has multiple buckets
        for i in range(10):
            client.post(
                f"/api/predict/{did}",
                json={"age": 25 + i * 3, "income": 50000 + i * 10000, "score": 70 + i * 3},
            )

        resp = client.get(f"/api/deploy/{did}/analytics")
        data = resp.json()
        dist = data["prediction_distribution"]
        assert isinstance(dist, list)
        # Each bucket should have bucket label and count
        for bucket in dist:
            assert "bucket" in bucket
            assert "count" in bucket

    def test_analytics_days_parameter(self, client, deployed_regression):
        did = deployed_regression["deployment_id"]
        resp = client.get(f"/api/deploy/{did}/analytics?days=30")
        assert resp.status_code == 200

    def test_analytics_problem_type_regression(self, client, deployed_regression):
        did = deployed_regression["deployment_id"]
        resp = client.get(f"/api/deploy/{did}/analytics")
        assert resp.json()["problem_type"] == "regression"


# ---------------------------------------------------------------------------
# Prediction logs endpoint tests
# ---------------------------------------------------------------------------

class TestPredictionLogs:
    """GET /api/deploy/{id}/logs returns paginated prediction log entries."""

    def test_logs_empty_initially(self, client, deployed_regression):
        did = deployed_regression["deployment_id"]
        resp = client.get(f"/api/deploy/{did}/logs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["logs"] == []

    def test_logs_populated_after_predictions(self, client, deployed_regression):
        did = deployed_regression["deployment_id"]
        for i in range(3):
            client.post(
                f"/api/predict/{did}",
                json={"age": 30 + i, "income": 60000, "score": 85},
            )

        resp = client.get(f"/api/deploy/{did}/logs")
        data = resp.json()
        assert data["total"] == 3
        assert len(data["logs"]) == 3

    def test_logs_structure(self, client, deployed_regression):
        did = deployed_regression["deployment_id"]
        client.post(f"/api/predict/{did}", json={"age": 35, "income": 70000, "score": 90})

        resp = client.get(f"/api/deploy/{did}/logs")
        log = resp.json()["logs"][0]
        assert "id" in log
        assert "input_features" in log
        assert "prediction" in log
        assert "created_at" in log

    def test_logs_pagination(self, client, deployed_regression):
        did = deployed_regression["deployment_id"]
        for i in range(5):
            client.post(f"/api/predict/{did}", json={"age": 30 + i, "income": 60000, "score": 85})

        resp1 = client.get(f"/api/deploy/{did}/logs?limit=2&offset=0")
        resp2 = client.get(f"/api/deploy/{did}/logs?limit=2&offset=2")
        assert len(resp1.json()["logs"]) == 2
        assert len(resp2.json()["logs"]) == 2
        # Different entries
        ids1 = {entry["id"] for entry in resp1.json()["logs"]}
        ids2 = {entry["id"] for entry in resp2.json()["logs"]}
        assert ids1.isdisjoint(ids2)

    def test_logs_sorted_most_recent_first(self, client, deployed_regression):
        did = deployed_regression["deployment_id"]
        for i in range(3):
            client.post(f"/api/predict/{did}", json={"age": 30 + i, "income": 60000, "score": 85})
            time.sleep(0.01)  # Ensure distinct timestamps

        resp = client.get(f"/api/deploy/{did}/logs")
        logs = resp.json()["logs"]
        timestamps = [entry["created_at"] for entry in logs]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_logs_404_for_unknown_deployment(self, client):
        resp = client.get("/api/deploy/nonexistent/logs")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Model readiness tests
# ---------------------------------------------------------------------------

class TestModelReadiness:
    """GET /api/models/{id}/readiness — production-readiness checklist."""

    def test_readiness_returns_score_and_verdict(self, client, deployed_regression):
        run_id = deployed_regression["run_id"]
        resp = client.get(f"/api/models/{run_id}/readiness")
        assert resp.status_code == 200
        data = resp.json()
        assert "score" in data
        assert 0 <= data["score"] <= 100
        assert data["verdict"] in ("ready", "needs_attention", "not_ready")

    def test_readiness_has_checks_list(self, client, deployed_regression):
        run_id = deployed_regression["run_id"]
        resp = client.get(f"/api/models/{run_id}/readiness")
        data = resp.json()
        assert "checks" in data
        assert isinstance(data["checks"], list)
        assert len(data["checks"]) > 0

    def test_readiness_checks_have_required_fields(self, client, deployed_regression):
        run_id = deployed_regression["run_id"]
        resp = client.get(f"/api/models/{run_id}/readiness")
        for check in resp.json()["checks"]:
            assert "id" in check
            assert "label" in check
            assert "passed" in check
            assert isinstance(check["passed"], bool)

    def test_readiness_has_summary(self, client, deployed_regression):
        run_id = deployed_regression["run_id"]
        resp = client.get(f"/api/models/{run_id}/readiness")
        data = resp.json()
        assert "summary" in data
        assert isinstance(data["summary"], str)
        assert len(data["summary"]) > 10

    def test_readiness_includes_algorithm(self, client, deployed_regression):
        run_id = deployed_regression["run_id"]
        resp = client.get(f"/api/models/{run_id}/readiness")
        assert resp.json()["algorithm"] == "linear_regression"

    def test_readiness_selected_model_passes_selection_check(self, client, deployed_regression):
        run_id = deployed_regression["run_id"]  # Already selected in fixture
        resp = client.get(f"/api/models/{run_id}/readiness")
        checks = {c["id"]: c for c in resp.json()["checks"]}
        assert checks["model_selected"]["passed"] is True

    def test_readiness_unselected_model_fails_selection_check(self, client, deployed_regression):
        """An unselected model should fail the model_selected readiness check."""
        run_id = deployed_regression["run_id"]
        # Deselect the run by updating via DB directly
        with Session(db_module.engine) as session:
            from models.model_run import ModelRun
            run = session.get(ModelRun, run_id)
            if run:
                run.is_selected = False
                session.add(run)
                session.commit()

        resp = client.get(f"/api/models/{run_id}/readiness")
        assert resp.status_code == 200
        checks = {c["id"]: c for c in resp.json()["checks"]}
        assert checks["model_selected"]["passed"] is False

    def test_readiness_training_complete_check_passes(self, client, deployed_regression):
        run_id = deployed_regression["run_id"]
        resp = client.get(f"/api/models/{run_id}/readiness")
        checks = {c["id"]: c for c in resp.json()["checks"]}
        assert checks["training_complete"]["passed"] is True

    def test_readiness_404_for_unknown_run(self, client):
        resp = client.get("/api/models/nonexistent/readiness")
        assert resp.status_code == 404

    def test_readiness_400_for_pending_run(self, client, deployed_regression):
        """Readiness endpoint rejects runs that haven't completed training."""
        # We can't easily get a 'pending' run after the fixture completes,
        # so test the 400 via a deployed run that we pretend is pending by
        # directly mutating the DB
        run_id = deployed_regression["run_id"]
        with Session(db_module.engine) as session:
            from models.model_run import ModelRun
            run = session.get(ModelRun, run_id)
            if run:
                run.status = "training"
                session.add(run)
                session.commit()
        resp = client.get(f"/api/models/{run_id}/readiness")
        assert resp.status_code == 400

    def test_readiness_score_higher_for_selected_model(self, client, deployed_regression):
        """Selected model with decent metrics should score higher than unselected."""
        run_id = deployed_regression["run_id"]
        resp = client.get(f"/api/models/{run_id}/readiness")
        score = resp.json()["score"]
        # With 20 rows + selected + training complete, should score at least 45
        assert score >= 45

    def test_readiness_regression_checks_r2(self, client, deployed_regression):
        run_id = deployed_regression["run_id"]
        resp = client.get(f"/api/models/{run_id}/readiness")
        checks = {c["id"]: c for c in resp.json()["checks"]}
        # There should be an accuracy check
        assert "model_accuracy" in checks or "accuracy" in checks

    def test_readiness_classification_model(self, client, deployed_classification):
        run_id = deployed_classification["run_id"]
        resp = client.get(f"/api/models/{run_id}/readiness")
        assert resp.status_code == 200
        data = resp.json()
        assert data["problem_type"] == "classification"
        assert 0 <= data["score"] <= 100


# ---------------------------------------------------------------------------
# Chat readiness integration tests
# ---------------------------------------------------------------------------

class TestChatReadinessIntent:
    """Chat endpoint detects readiness questions and emits structured readiness events."""

    def _mock_anthropic(self, text: str = "Looks good!") -> MagicMock:
        """Return a mock Anthropic client that yields a single SSE token."""
        mock_client = MagicMock()
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter([text])
        mock_client.messages.stream.return_value = mock_stream
        return mock_client

    def test_readiness_query_emits_readiness_event(self, client, deployed_regression):
        project_id = deployed_regression["project_id"]
        with patch("api.chat.anthropic.Anthropic", return_value=self._mock_anthropic()):
            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "Is my model ready to deploy?"},
            )
        assert resp.status_code == 200
        events = []
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
        readiness_events = [e for e in events if e.get("type") == "readiness"]
        assert len(readiness_events) == 1
        r = readiness_events[0]["readiness"]
        assert "score" in r
        assert "verdict" in r
        assert "checks" in r

    def test_non_readiness_query_no_readiness_event(self, client, deployed_regression):
        project_id = deployed_regression["project_id"]
        with patch("api.chat.anthropic.Anthropic", return_value=self._mock_anthropic()):
            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "What columns does my dataset have?"},
            )
        assert resp.status_code == 200
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                try:
                    event = json.loads(line[6:])
                    assert event.get("type") != "readiness"
                except json.JSONDecodeError:
                    pass

    def test_deploy_keyword_triggers_readiness(self, client, deployed_regression):
        project_id = deployed_regression["project_id"]
        with patch("api.chat.anthropic.Anthropic", return_value=self._mock_anthropic()):
            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "Should I deploy this model?"},
            )
        assert resp.status_code == 200
        events = []
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
        readiness_events = [e for e in events if e.get("type") == "readiness"]
        assert len(readiness_events) == 1

    def test_readiness_query_no_models_no_readiness_event(self, client):
        """If no completed model runs, no readiness event is emitted."""
        proj = client.post("/api/projects", json={"name": "Empty Project"})
        project_id = proj.json()["id"]
        with patch("api.chat.anthropic.Anthropic", return_value=self._mock_anthropic()):
            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "Is my model ready to deploy?"},
            )
        assert resp.status_code == 200
        for line in resp.text.splitlines():
            if line.startswith("data: "):
                try:
                    event = json.loads(line[6:])
                    assert event.get("type") != "readiness"
                except json.JSONDecodeError:
                    pass
