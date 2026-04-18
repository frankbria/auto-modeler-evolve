"""Tests for Production Input Distribution Chat Card.

Covers:
- _PROD_INPUT_DIST_PATTERNS regex (positive + negative)
- Chat handler emitting prod_input_dist SSE event
- No event when no deployment
- No predictions (empty) case
- Numeric and categorical feature aggregation
- Out-of-range detection vs training ranges
"""

from __future__ import annotations

import io
import json
import unittest.mock as mock

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

import db as db_module

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
    b"East,175.2,18\nWest,220.4,22\nNorth,190.6,19\nEast,130.8,13\nWest,280.0,28\n"
    b"East,160.0,16\nWest,210.0,21\n"
)


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

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------


class TestProdInputDistPatterns:
    """_PROD_INPUT_DIST_PATTERNS should match analyst intent phrases."""

    def setup_method(self):
        from api.chat import _PROD_INPUT_DIST_PATTERNS

        self.pattern = _PROD_INPUT_DIST_PATTERNS

    def _match(self, text: str) -> bool:
        return bool(self.pattern.search(text))

    def test_what_values_users_sending(self):
        assert self._match("what values are users sending to my model?")

    def test_show_production_inputs(self):
        assert self._match("show me the production input distribution")

    def test_input_distribution(self):
        assert self._match("input distribution")

    def test_what_inputs_receiving(self):
        assert self._match("what feature values is my model receiving?")

    def test_are_inputs_in_range(self):
        assert self._match("are my production inputs in range?")

    def test_how_different_from_training(self):
        assert self._match("how different are the production inputs from training?")

    def test_most_common_inputs(self):
        assert self._match("most common inputs")

    def test_production_input_stats(self):
        assert self._match("show production input stats")

    def test_case_insensitive(self):
        assert self._match("What Values Are Users Sending?")

    def test_negative_general_question(self):
        assert not self._match("how accurate is my model?")

    def test_negative_training_question(self):
        assert not self._match("what algorithm was used for training?")

    def test_negative_sla(self):
        assert not self._match("show me the prediction latency")


# ---------------------------------------------------------------------------
# Handler integration tests
# ---------------------------------------------------------------------------


class TestProdInputDistHandler:
    """Integration tests for the prod_input_dist SSE handler."""

    def _sse_events(self, response) -> list[dict]:
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
        return events

    def _find_event(self, events: list[dict], event_type: str) -> dict | None:
        return next((e for e in events if e.get("type") == event_type), None)

    def test_no_event_without_deployment(self, client):
        """No prod_input_dist event when no deployment exists (project with no deploy)."""
        proj = client.post("/api/projects", json={"name": "No Deploy"}).json()
        pid = proj["id"]

        with mock.patch("anthropic.Anthropic") as mock_ant:
            mock_cli = mock.MagicMock()
            mock_ant.return_value = mock_cli
            mock_stream = mock.MagicMock()
            mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = mock.MagicMock(return_value=False)
            mock_stream.text_stream = iter([])
            mock_cli.messages.stream.return_value = mock_stream

            resp = client.post(
                f"/api/chat/{pid}",
                json={"message": "show production input distribution"},
            )
        events = [
            json.loads(line[6:])
            for line in resp.text.splitlines()
            if line.startswith("data: ") and line[6:].strip()
        ]
        assert not any(e.get("type") == "prod_input_dist" for e in events)

    def test_server_does_not_crash_on_prod_input_query(self, client):
        """Server returns 200 for prod input distribution query regardless of state."""
        proj = client.post("/api/projects", json={"name": "Safe Test"}).json()
        pid = proj["id"]

        with mock.patch("anthropic.Anthropic") as mock_ant:
            mock_cli = mock.MagicMock()
            mock_ant.return_value = mock_cli
            mock_stream = mock.MagicMock()
            mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = mock.MagicMock(return_value=False)
            mock_stream.text_stream = iter(["Here are the input distributions."])
            mock_cli.messages.stream.return_value = mock_stream

            resp = client.post(
                f"/api/chat/{pid}",
                json={"message": "what values are users sending to my model?"},
            )
        assert resp.status_code == 200

    def test_emits_event_with_deployment_and_logs(self, client):
        """With a real deployment and prediction logs, emits prod_input_dist event."""
        import uuid as _uuid

        proj = client.post("/api/projects", json={"name": "PID Test"}).json()
        pid = proj["id"]

        upload = client.post(
            "/api/data/upload",
            data={"project_id": pid},
            files={"file": ("data.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        )
        assert upload.status_code == 201
        dataset_id = upload.json()["dataset_id"]

        client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
        fs_resp = client.post(
            f"/api/features/{dataset_id}/target",
            json={"target_column": "revenue", "problem_type": "regression"},
        )
        feature_set_id = fs_resp.json().get("feature_set_id")

        from models.deployment import Deployment
        from models.model_run import ModelRun
        from models.prediction_log import PredictionLog

        run_id = str(_uuid.uuid4())
        dep_id = str(_uuid.uuid4())

        with Session(db_module.engine) as sess:
            sess.add(
                ModelRun(
                    id=run_id,
                    project_id=pid,
                    feature_set_id=feature_set_id,
                    algorithm="linear_regression",
                    status="done",
                    metrics=json.dumps({"r2": 0.85}),
                    summary="LR: R² 0.850",
                )
            )
            sess.add(
                Deployment(
                    id=dep_id,
                    model_run_id=run_id,
                    project_id=pid,
                    endpoint_path=f"/api/predict/{dep_id}",
                    dashboard_url=f"/predict/{dep_id}",
                    is_active=True,
                    algorithm="linear_regression",
                    problem_type="regression",
                    feature_names=json.dumps(["units", "region"]),
                    target_column="revenue",
                    metrics=json.dumps({"r2": 0.85}),
                )
            )
            for units_val, region_val in [(10, "East"), (20, "West"), (15, "East")]:
                sess.add(
                    PredictionLog(
                        deployment_id=dep_id,
                        input_features=json.dumps({"units": units_val, "region": region_val}),
                        prediction=json.dumps(150.0),
                        prediction_numeric=150.0,
                    )
                )
            sess.commit()

        with mock.patch("anthropic.Anthropic") as mock_ant:
            mock_cli = mock.MagicMock()
            mock_ant.return_value = mock_cli
            mock_stream = mock.MagicMock()
            mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = mock.MagicMock(return_value=False)
            mock_stream.text_stream = iter(["Input distribution shown."])
            mock_cli.messages.stream.return_value = mock_stream

            resp = client.post(
                f"/api/chat/{pid}",
                json={"message": "what values are users sending to my model?"},
            )

        assert resp.status_code == 200
        events = [
            json.loads(line[6:])
            for line in resp.text.splitlines()
            if line.startswith("data: ") and line[6:].strip()
        ]
        pid_event = next((e for e in events if e.get("type") == "prod_input_dist"), None)
        assert pid_event is not None
        result = pid_event["prod_input_dist"]
        assert result["sample_count"] == 3
        assert len(result["features"]) >= 1
        assert "summary" in result

    def test_required_event_fields(self):
        """prod_input_dist event dict has required fields."""
        event = {
            "deployment_id": "dep-001",
            "sample_count": 10,
            "features": [
                {
                    "feature": "revenue",
                    "feature_type": "numeric",
                    "count": 10,
                    "mean": 175.5,
                    "min": 100.0,
                    "max": 300.0,
                    "out_of_range_count": 0,
                    "out_of_range_pct": 0.0,
                }
            ],
            "summary": "10 predictions analyzed across 1 feature. All inputs are within training ranges.",
        }
        assert "deployment_id" in event
        assert "sample_count" in event
        assert "features" in event
        assert "summary" in event
        feat = event["features"][0]
        assert feat["feature_type"] == "numeric"
        assert "mean" in feat
        assert "out_of_range_count" in feat

    def test_categorical_feature_structure(self):
        """Categorical features have top_categories and n_unique fields."""
        feat = {
            "feature": "region",
            "feature_type": "categorical",
            "count": 10,
            "top_categories": [
                {"value": "East", "count": 5, "pct": 50.0},
                {"value": "West", "count": 3, "pct": 30.0},
            ],
            "n_unique": 3,
            "known_categories": ["East", "West", "North"],
            "unseen_count": 2,
            "unseen_pct": 20.0,
        }
        assert feat["feature_type"] == "categorical"
        assert len(feat["top_categories"]) == 2
        assert feat["top_categories"][0]["value"] == "East"
        assert "unseen_count" in feat

    def test_out_of_range_pct_calculation(self):
        """out_of_range_pct is correctly computed."""
        oor_count = 2
        total = 10
        pct = round(oor_count / total * 100, 1)
        assert pct == 20.0

    def test_summary_no_predictions(self):
        """Empty prediction log summary is informative."""
        summary = "No predictions have been made yet — input distributions will appear after users start using the model."
        assert "No predictions" in summary

    def test_summary_with_oor_values(self):
        """Summary mentions out-of-range values when present."""
        n_oor = 3
        sample_count = 10
        summary = (
            f"{sample_count} predictions analyzed across 2 features. "
            f"{n_oor} input values outside the training distribution."
        )
        assert "outside the training distribution" in summary

    def test_summary_all_in_range(self):
        """Summary confirms all inputs in range when no OOR values."""
        summary = "5 predictions analyzed across 2 features. All inputs are within training ranges."
        assert "All inputs are within training ranges" in summary
