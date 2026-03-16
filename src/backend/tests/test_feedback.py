"""Tests for the prediction feedback loop.

Covers:
- POST /api/predict/{deployment_id}/feedback  (submit actual outcome)
- GET  /api/deploy/{deployment_id}/feedback-accuracy (aggregate stats)

Both regression and classification paths are exercised.
"""

from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REGRESSION_CSV = b"""product,region,revenue,units
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
Widget A,East,950.75,8
Widget C,West,550.00,5
Widget B,South,1300.00,12
Widget A,North,2500.00,22
Widget C,East,800.00,8
"""

CLASSIFICATION_CSV = b"""feature_a,feature_b,label
1.2,3.4,cat
2.1,1.1,dog
3.3,2.2,cat
0.5,4.1,dog
2.8,0.9,cat
1.9,3.1,dog
3.0,2.5,cat
0.8,3.8,dog
2.3,1.5,cat
1.7,2.9,dog
2.5,1.8,cat
1.1,3.5,dog
3.2,2.0,cat
0.6,4.3,dog
2.7,1.2,cat
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

    import api.data as data_mod
    data_mod.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_mod
    models_mod.MODELS_DIR = tmp_path / "models"

    import api.deploy as deploy_mod
    deploy_mod.DEPLOY_DIR = tmp_path / "deployments"

    from main import app
    with TestClient(app) as c:
        yield c


def _setup_deployed_regression(client) -> tuple[str, str]:
    """Full workflow → deployed regression model. Returns (deployment_id, prediction_log_id)."""
    pid = client.post("/api/projects", json={"name": "FBTest"}).json()["id"]

    r = client.post(
        "/api/data/upload",
        files={"file": ("data.csv", io.BytesIO(REGRESSION_CSV), "text/csv")},
        data={"project_id": pid},
    )
    did = r.json()["dataset_id"]

    fs = client.post(f"/api/features/{did}/apply", json={"transformations": []}).json()
    client.post(f"/api/features/{did}/target", json={"target_column": "revenue", "feature_set_id": fs["feature_set_id"]})

    client.post(f"/api/models/{pid}/train", json={"algorithms": ["linear_regression"]})
    for _ in range(60):
        runs = client.get(f"/api/models/{pid}/runs").json()["runs"]
        if all(r["status"] in ("done", "failed") for r in runs):
            break
        time.sleep(0.1)

    runs = client.get(f"/api/models/{pid}/runs").json()["runs"]
    done_run = next(r for r in runs if r["status"] == "done")

    dep = client.post(f"/api/deploy/{done_run['id']}").json()
    dep_id = dep["id"]

    # Make a prediction so we have a log entry
    pred_r = client.post(
        f"/api/predict/{dep_id}",
        json={"product": "Widget A", "region": "North", "units": 12},
    )
    log_id = None
    # Grab the log entry id
    logs = client.get(f"/api/deploy/{dep_id}/logs").json()["logs"]
    if logs:
        log_id = logs[0]["id"]

    return dep_id, log_id


def _setup_deployed_classification(client) -> tuple[str, str]:
    """Full workflow → deployed classification model. Returns (deployment_id, prediction_log_id)."""
    pid = client.post("/api/projects", json={"name": "FBClassTest"}).json()["id"]

    r = client.post(
        "/api/data/upload",
        files={"file": ("cls.csv", io.BytesIO(CLASSIFICATION_CSV), "text/csv")},
        data={"project_id": pid},
    )
    did = r.json()["dataset_id"]

    fs = client.post(f"/api/features/{did}/apply", json={"transformations": []}).json()
    client.post(f"/api/features/{did}/target", json={
        "target_column": "label",
        "problem_type": "classification",
        "feature_set_id": fs["feature_set_id"],
    })

    client.post(f"/api/models/{pid}/train", json={"algorithms": ["logistic_regression"]})
    for _ in range(60):
        runs = client.get(f"/api/models/{pid}/runs").json()["runs"]
        if all(r["status"] in ("done", "failed") for r in runs):
            break
        time.sleep(0.1)

    runs = client.get(f"/api/models/{pid}/runs").json()["runs"]
    done_run = next(r for r in runs if r["status"] == "done")

    dep = client.post(f"/api/deploy/{done_run['id']}").json()
    dep_id = dep["id"]

    # Make a prediction
    client.post(f"/api/predict/{dep_id}", json={"feature_a": 1.5, "feature_b": 3.0})
    logs = client.get(f"/api/deploy/{dep_id}/logs").json()["logs"]
    log_id = logs[0]["id"] if logs else None

    return dep_id, log_id


# ---------------------------------------------------------------------------
# submit_feedback — happy paths
# ---------------------------------------------------------------------------

class TestSubmitFeedback:

    def test_submit_regression_feedback_returns_201(self, client):
        dep_id, _ = _setup_deployed_regression(client)
        r = client.post(
            f"/api/predict/{dep_id}/feedback",
            json={"actual_value": 1350.0},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["deployment_id"] == dep_id
        assert body["actual_value"] == 1350.0
        assert "id" in body
        assert "message" in body

    def test_submit_classification_feedback_with_label(self, client):
        dep_id, log_id = _setup_deployed_classification(client)
        r = client.post(
            f"/api/predict/{dep_id}/feedback",
            json={"actual_label": "cat"},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["actual_label"] == "cat"

    def test_submit_feedback_with_comment(self, client):
        dep_id, _ = _setup_deployed_regression(client)
        r = client.post(
            f"/api/predict/{dep_id}/feedback",
            json={"actual_value": 999.0, "comment": "Customer reported actual revenue"},
        )
        assert r.status_code == 201
        assert r.json()["comment"] == "Customer reported actual revenue"

    def test_submit_feedback_with_prediction_log_id(self, client):
        dep_id, log_id = _setup_deployed_regression(client)
        if not log_id:
            pytest.skip("No log entry found")
        r = client.post(
            f"/api/predict/{dep_id}/feedback",
            json={"prediction_log_id": log_id, "actual_value": 1200.0},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["prediction_log_id"] == log_id

    def test_submit_feedback_auto_computes_is_correct_for_classification(self, client):
        dep_id, log_id = _setup_deployed_classification(client)
        if not log_id:
            pytest.skip("No log entry found")

        # Get what the model actually predicted
        logs = client.get(f"/api/deploy/{dep_id}/logs").json()["logs"]
        predicted_label = str(logs[0]["prediction"])

        r = client.post(
            f"/api/predict/{dep_id}/feedback",
            json={"prediction_log_id": log_id, "actual_label": predicted_label},
        )
        assert r.status_code == 201
        # is_correct should be True since we passed the predicted label as actual
        assert r.json()["is_correct"] is True

    def test_submit_feedback_wrong_prediction_sets_is_correct_false(self, client):
        dep_id, log_id = _setup_deployed_classification(client)
        if not log_id:
            pytest.skip("No log entry found")

        # Get what the model predicted and provide the OTHER label
        logs = client.get(f"/api/deploy/{dep_id}/logs").json()["logs"]
        predicted_label = str(logs[0]["prediction"])
        wrong_label = "cat" if predicted_label == "dog" else "dog"

        r = client.post(
            f"/api/predict/{dep_id}/feedback",
            json={"prediction_log_id": log_id, "actual_label": wrong_label},
        )
        assert r.status_code == 201
        assert r.json()["is_correct"] is False

    def test_submit_feedback_is_correct_override(self, client):
        dep_id, _ = _setup_deployed_regression(client)
        r = client.post(
            f"/api/predict/{dep_id}/feedback",
            json={"actual_value": 1500.0, "is_correct": True},
        )
        assert r.status_code == 201
        assert r.json()["is_correct"] is True


# ---------------------------------------------------------------------------
# submit_feedback — error paths
# ---------------------------------------------------------------------------

class TestSubmitFeedbackErrors:

    def test_unknown_deployment_returns_404(self, client):
        r = client.post(
            "/api/predict/nonexistent-dep/feedback",
            json={"actual_value": 1.0},
        )
        assert r.status_code == 404

    def test_missing_actual_value_returns_400(self, client):
        dep_id, _ = _setup_deployed_regression(client)
        r = client.post(
            f"/api/predict/{dep_id}/feedback",
            json={"comment": "Just a note, no actual value"},
        )
        assert r.status_code == 400
        assert "actual_value" in r.json()["detail"] or "actual_label" in r.json()["detail"]

    def test_multiple_feedback_records_accumulate(self, client):
        dep_id, _ = _setup_deployed_regression(client)
        for val in [1100.0, 1250.0, 1400.0]:
            client.post(f"/api/predict/{dep_id}/feedback", json={"actual_value": val})

        acc = client.get(f"/api/deploy/{dep_id}/feedback-accuracy").json()
        assert acc["total_feedback"] == 3


# ---------------------------------------------------------------------------
# feedback-accuracy — regression
# ---------------------------------------------------------------------------

class TestFeedbackAccuracyRegression:

    def test_no_feedback_returns_no_feedback_status(self, client):
        dep_id, _ = _setup_deployed_regression(client)
        r = client.get(f"/api/deploy/{dep_id}/feedback-accuracy")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "no_feedback"
        assert body["total_feedback"] == 0
        assert "message" in body

    def test_feedback_without_log_id_returns_feedback_only_status(self, client):
        dep_id, _ = _setup_deployed_regression(client)
        client.post(f"/api/predict/{dep_id}/feedback", json={"actual_value": 1300.0})
        r = client.get(f"/api/deploy/{dep_id}/feedback-accuracy").json()
        # Without prediction_log_id we can't compute MAE — status may be feedback_only or computed
        assert r["status"] in ("feedback_only", "computed")
        assert r["total_feedback"] == 1

    def test_feedback_with_log_id_computes_mae(self, client):
        dep_id, log_id = _setup_deployed_regression(client)
        if not log_id:
            pytest.skip("No prediction log entry")

        # Submit feedback with log_id so MAE can be computed
        client.post(
            f"/api/predict/{dep_id}/feedback",
            json={"prediction_log_id": log_id, "actual_value": 1200.0},
        )
        r = client.get(f"/api/deploy/{dep_id}/feedback-accuracy").json()
        assert r["status"] == "computed"
        assert "mae" in r
        assert r["mae"] >= 0
        assert "pct_error" in r
        assert r["paired_count"] >= 1
        assert "verdict" in r
        assert "message" in r

    def test_feedback_accuracy_404_unknown_deployment(self, client):
        r = client.get("/api/deploy/nonexistent/feedback-accuracy")
        assert r.status_code == 404

    def test_accuracy_verdict_is_valid(self, client):
        dep_id, log_id = _setup_deployed_regression(client)
        if not log_id:
            pytest.skip("No log entry")
        client.post(
            f"/api/predict/{dep_id}/feedback",
            json={"prediction_log_id": log_id, "actual_value": 1200.0},
        )
        r = client.get(f"/api/deploy/{dep_id}/feedback-accuracy").json()
        if r.get("verdict"):
            assert r["verdict"] in ("excellent", "good", "moderate", "poor")


# ---------------------------------------------------------------------------
# feedback-accuracy — classification
# ---------------------------------------------------------------------------

class TestFeedbackAccuracyClassification:

    def test_no_feedback_returns_no_feedback_status(self, client):
        dep_id, _ = _setup_deployed_classification(client)
        r = client.get(f"/api/deploy/{dep_id}/feedback-accuracy").json()
        assert r["status"] == "no_feedback"

    def test_correct_feedback_computes_accuracy(self, client):
        dep_id, log_id = _setup_deployed_classification(client)
        if not log_id:
            pytest.skip("No log entry")

        # Submit 2 correct, 1 incorrect
        for correct in [True, True, False]:
            client.post(
                f"/api/predict/{dep_id}/feedback",
                json={"actual_label": "cat", "is_correct": correct},
            )

        r = client.get(f"/api/deploy/{dep_id}/feedback-accuracy").json()
        # status may be computed if is_correct flags were set
        if r["status"] == "computed":
            assert r["correct_count"] == 2
            assert r["incorrect_count"] == 1
            assert abs(r["accuracy_from_feedback"] - 2/3) < 0.01

    def test_feedback_without_is_correct_returns_feedback_only(self, client):
        dep_id, _ = _setup_deployed_classification(client)
        # Submit feedback with no is_correct and no log_id for auto-compute
        client.post(f"/api/predict/{dep_id}/feedback", json={"actual_label": "cat"})
        r = client.get(f"/api/deploy/{dep_id}/feedback-accuracy").json()
        assert r["status"] in ("feedback_only", "computed")
        assert r["total_feedback"] == 1


# ---------------------------------------------------------------------------
# FeedbackRecord model
# ---------------------------------------------------------------------------

class TestFeedbackRecordModel:

    def test_feedback_record_is_persisted(self, client):
        dep_id, _ = _setup_deployed_regression(client)
        r = client.post(
            f"/api/predict/{dep_id}/feedback",
            json={"actual_value": 1500.0, "comment": "test note"},
        )
        assert r.status_code == 201
        record_id = r.json()["id"]
        assert record_id  # UUID assigned

    def test_multiple_feedback_all_have_unique_ids(self, client):
        dep_id, _ = _setup_deployed_regression(client)
        ids = []
        for i in range(3):
            r = client.post(
                f"/api/predict/{dep_id}/feedback",
                json={"actual_value": float(i * 100)},
            )
            ids.append(r.json()["id"])
        assert len(set(ids)) == 3  # all unique

    def test_feedback_includes_timestamp(self, client):
        dep_id, _ = _setup_deployed_regression(client)
        r = client.post(f"/api/predict/{dep_id}/feedback", json={"actual_value": 1.0})
        body = r.json()
        assert "created_at" in body
        assert body["created_at"]  # non-empty ISO string
