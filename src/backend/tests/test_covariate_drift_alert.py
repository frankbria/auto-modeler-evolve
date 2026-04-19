"""Tests for Covariate Drift Alert.

Covers:
- compute_covariate_drift_alert() pure function (unit tests)
- _COVARIATE_DRIFT_PATTERNS regex (positive + negative)
- Chat handler emitting covariate_drift_alert SSE event
- No event without a deployment
- GET /api/deploy/{id}/covariate-drift REST endpoint
"""

from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

import db as db_module
from core.analyzer import compute_covariate_drift_alert

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
    b"East,175.2,18\nWest,220.4,22\nNorth,190.6,19\nEast,130.8,13\nWest,280.0,28\n"
)


# ---------------------------------------------------------------------------
# Unit tests for compute_covariate_drift_alert()
# ---------------------------------------------------------------------------


class TestComputeCovariateDriftAlert:
    def _ranges(self):
        return {
            "revenue": {"min": 100.0, "max": 300.0},
            "units": {"min": 10, "max": 30},
            "region": {"known_categories": ["East", "West", "North"]},
        }

    def test_empty_inputs_returns_no_alerts(self):
        result = compute_covariate_drift_alert([], self._ranges())
        assert result["has_alerts"] is False
        assert result["severity"] == "low"
        assert result["sample_count"] == 0
        assert result["alerts"] == []

    def test_all_in_range_returns_low_severity(self):
        inputs = [
            {"revenue": 150.0, "units": 15, "region": "East"},
            {"revenue": 200.0, "units": 20, "region": "West"},
        ]
        result = compute_covariate_drift_alert(inputs, self._ranges())
        assert result["has_alerts"] is False
        assert result["severity"] == "low"
        assert result["alert_count"] == 0

    def test_numeric_oor_medium_severity(self):
        # 20% OOR (≥15% threshold) → medium
        inputs = [{"revenue": float(v)} for v in [50, 150, 150, 150, 150]]
        ranges = {"revenue": {"min": 100.0, "max": 300.0}}
        result = compute_covariate_drift_alert(inputs, ranges)
        assert result["has_alerts"] is True
        assert result["severity"] == "medium"
        assert result["alerts"][0]["feature"] == "revenue"
        assert result["alerts"][0]["oor_pct"] == 20.0

    def test_numeric_oor_high_severity(self):
        # 40% OOR (≥30% threshold) → high
        inputs = [{"revenue": float(v)} for v in [50, 60, 150, 150, 150]]
        ranges = {"revenue": {"min": 100.0, "max": 300.0}}
        result = compute_covariate_drift_alert(inputs, ranges)
        assert result["severity"] == "high"
        assert result["alerts"][0]["severity"] == "high"

    def test_categorical_unseen_medium_severity(self):
        # 20% unseen (≥15% threshold) → medium
        inputs = [{"region": v} for v in ["East", "West", "East", "West", "UNKNOWN"]]
        ranges = {"region": {"known_categories": ["East", "West", "North"]}}
        result = compute_covariate_drift_alert(inputs, ranges)
        assert result["has_alerts"] is True
        assert result["severity"] == "medium"
        assert result["alerts"][0]["feature"] == "region"
        assert result["alerts"][0]["feature_type"] == "categorical"
        assert result["alerts"][0]["unseen_pct"] == 20.0

    def test_categorical_unseen_high_severity(self):
        # 40% unseen → high
        inputs = [{"region": v} for v in ["East", "NEW1", "NEW2", "East", "East"]]
        ranges = {"region": {"known_categories": ["East", "West"]}}
        result = compute_covariate_drift_alert(inputs, ranges)
        assert result["severity"] == "high"

    def test_no_feature_ranges_returns_no_alerts(self):
        inputs = [{"revenue": 500.0}, {"revenue": 600.0}]
        result = compute_covariate_drift_alert(inputs, {})
        assert result["has_alerts"] is False

    def test_missing_min_max_skips_numeric(self):
        inputs = [{"revenue": 500.0}]
        ranges = {"revenue": {"p5": 100.0, "p95": 300.0}}  # no min/max
        result = compute_covariate_drift_alert(inputs, ranges)
        assert result["has_alerts"] is False

    def test_empty_known_categories_skips_categorical(self):
        inputs = [{"region": "UNKNOWN"}]
        ranges = {"region": {"known_categories": []}}
        result = compute_covariate_drift_alert(inputs, ranges)
        assert result["has_alerts"] is False

    def test_max_features_cap(self):
        # 11 features — only first 10 analyzed
        inputs = [{f"feat_{i}": 999.0 for i in range(11)}]
        ranges = {f"feat_{i}": {"min": 0.0, "max": 1.0} for i in range(11)}
        result = compute_covariate_drift_alert(inputs, ranges, max_features=10)
        assert result["feature_count"] == 10
        assert result["alert_count"] <= 10

    def test_summary_present_for_all_severities(self):
        for inputs, exp_has in [
            ([], False),
            ([{"revenue": 150.0}], False),
        ]:
            result = compute_covariate_drift_alert(
                inputs, {"revenue": {"min": 100.0, "max": 300.0}}
            )
            assert isinstance(result["summary"], str)
            assert len(result["summary"]) > 0

    def test_result_contains_required_keys(self):
        result = compute_covariate_drift_alert([], {})
        required = {
            "has_alerts",
            "severity",
            "severity_label",
            "sample_count",
            "feature_count",
            "alert_count",
            "alerts",
            "summary",
        }
        assert required.issubset(result.keys())


# ---------------------------------------------------------------------------
# Pattern regex tests
# ---------------------------------------------------------------------------


class TestCovariateDriftPatterns:
    def _pattern(self):
        from api.chat import _COVARIATE_DRIFT_PATTERNS

        return _COVARIATE_DRIFT_PATTERNS

    @pytest.mark.parametrize(
        "msg",
        [
            "covariate drift",
            "covariate shift",
            "input drift",
            "input feature drift",
            "production data drift",
            "production input drift",
            "are my inputs drifting",
            "are my production inputs drifting?",
            "check input drift",
            "detect feature drift",
            "show drift alert",
            "feature distribution drift",
            "any input drift alerts?",
            "drift alert",
            "drift monitor",
            "drift warning",
            "drift detection",
        ],
    )
    def test_positive_matches(self, msg):
        assert self._pattern().search(msg), f"Expected match for: {msg!r}"

    @pytest.mark.parametrize(
        "msg",
        [
            "show production input distribution",
            "what values are users sending",
            "how is my model doing",
            "show feature importance",
            "train my model",
            "deploy the model",
        ],
    )
    def test_negative_no_match(self, msg):
        assert not self._pattern().search(msg), f"Expected no match for: {msg!r}"


# ---------------------------------------------------------------------------
# Integration tests using TestClient
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

    SQLModel.metadata.create_all(db_module.engine)

    from main import app

    return TestClient(app)


def _setup_project_with_deployment(client, tmp_path):
    """Create project, then insert ModelRun + Deployment directly into DB."""
    from models.deployment import Deployment
    from models.model_run import ModelRun

    proj = client.post("/api/projects", json={"name": "DriftTest"}).json()
    project_id = proj["id"]

    run_id = str(uuid.uuid4())
    dep_id = str(uuid.uuid4())

    with Session(db_module.engine) as sess:
        sess.add(
            ModelRun(
                id=run_id,
                project_id=project_id,
                feature_set_id=str(uuid.uuid4()),
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
                project_id=project_id,
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
        sess.commit()

    return project_id, run_id, dep_id


class TestCovariateDriftRestEndpoint:
    def test_404_for_missing_deployment(self, client):
        resp = client.get("/api/deploy/nonexistent-id/covariate-drift")
        assert resp.status_code == 404

    def test_no_predictions_returns_low_severity(self, client, tmp_path):
        _, _, deployment_id = _setup_project_with_deployment(client, tmp_path)
        resp = client.get(f"/api/deploy/{deployment_id}/covariate-drift")
        assert resp.status_code == 200
        data = resp.json()
        assert data["has_alerts"] is False
        assert data["severity"] == "low"
        assert data["deployment_id"] == deployment_id

    def test_response_contains_required_fields(self, client, tmp_path):
        _, _, deployment_id = _setup_project_with_deployment(client, tmp_path)
        resp = client.get(f"/api/deploy/{deployment_id}/covariate-drift")
        assert resp.status_code == 200
        data = resp.json()
        for key in [
            "has_alerts",
            "severity",
            "severity_label",
            "sample_count",
            "feature_count",
            "alert_count",
            "alerts",
            "summary",
            "deployment_id",
        ]:
            assert key in data, f"Missing key: {key}"


def _mock_anthropic():
    from unittest import mock

    mock_ant = mock.patch("anthropic.Anthropic")
    patcher = mock_ant.start()
    mock_cli = mock.MagicMock()
    patcher.return_value = mock_cli
    mock_stream = mock.MagicMock()
    mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = mock.MagicMock(return_value=False)
    mock_stream.text_stream = iter(["Drift checked."])
    mock_cli.messages.stream.return_value = mock_stream
    return mock_ant


class TestCovariateDriftChatHandler:
    def test_no_event_without_deployment(self, client):
        from unittest import mock

        proj = client.post("/api/projects", json={"name": "NoDeploy"}).json()
        project_id = proj["id"]

        with mock.patch("anthropic.Anthropic") as m:
            mock_cli = mock.MagicMock()
            m.return_value = mock_cli
            mock_stream = mock.MagicMock()
            mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = mock.MagicMock(return_value=False)
            mock_stream.text_stream = iter(["No deployment."])
            mock_cli.messages.stream.return_value = mock_stream

            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "check input drift"},
            )

        assert resp.status_code == 200
        assert "covariate_drift_alert" not in resp.text

    def test_emits_event_with_deployment(self, client, tmp_path):
        from unittest import mock

        project_id, _, _ = _setup_project_with_deployment(client, tmp_path)

        with mock.patch("anthropic.Anthropic") as m:
            mock_cli = mock.MagicMock()
            m.return_value = mock_cli
            mock_stream = mock.MagicMock()
            mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = mock.MagicMock(return_value=False)
            mock_stream.text_stream = iter(["Drift checked."])
            mock_cli.messages.stream.return_value = mock_stream

            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "check input drift"},
            )

        assert resp.status_code == 200
        events = [
            json.loads(line[6:])
            for line in resp.text.splitlines()
            if line.startswith("data: ") and line[6:].strip()
        ]
        types = [e.get("type") for e in events]
        assert "covariate_drift_alert" in types

    def test_event_structure(self, client, tmp_path):
        from unittest import mock

        project_id, _, _ = _setup_project_with_deployment(client, tmp_path)

        with mock.patch("anthropic.Anthropic") as m:
            mock_cli = mock.MagicMock()
            m.return_value = mock_cli
            mock_stream = mock.MagicMock()
            mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = mock.MagicMock(return_value=False)
            mock_stream.text_stream = iter(["Drift checked."])
            mock_cli.messages.stream.return_value = mock_stream

            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "covariate drift"},
            )

        assert resp.status_code == 200
        events = [
            json.loads(line[6:])
            for line in resp.text.splitlines()
            if line.startswith("data: ")
            and line[6:].strip()
            and '"covariate_drift_alert"' in line
        ]
        assert len(events) >= 1
        ev = events[0]["covariate_drift_alert"]
        assert "severity" in ev
        assert "has_alerts" in ev
        assert "summary" in ev
        assert "alerts" in ev
