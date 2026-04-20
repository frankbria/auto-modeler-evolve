"""Tests for Prediction Log CSV Export (Track D perpetual)."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

import db as db_module
from api.chat import _PRED_LOG_EXPORT_PATTERNS


@contextmanager
def _mock_anthropic():
    with mock.patch("anthropic.Anthropic") as m:
        mock_cli = mock.MagicMock()
        m.return_value = mock_cli
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Here is your export download link."])
        mock_cli.messages.stream.return_value = mock_stream
        yield


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------


class TestPredLogExportPatterns:
    POSITIVE = [
        "export prediction history",
        "download prediction logs",
        "download prediction history",
        "save predictions as csv",
        "export my model's predictions",
        "download all predictions",
        "export prediction log",
        "prediction log export",
        "save prediction log to csv",
        "give me the prediction logs as a csv",
        "export model output log",
        "download my api call history",
        "download the prediction data",
        "export prediction records",
        "save prediction results as csv",
        "get my prediction history",
    ]
    NEGATIVE = [
        "how many predictions have I made?",
        "show prediction analytics",
        "train a model",
        "what is my quota runway?",
        "export my data",  # data export, not predictions
        "is my model drifting?",
        "latency report",
        "peak traffic hours",
        "hello",
        "show confusion matrix",
    ]

    @pytest.mark.parametrize("msg", POSITIVE)
    def test_positive(self, msg):
        assert _PRED_LOG_EXPORT_PATTERNS.search(msg), f"Should match: {msg!r}"

    @pytest.mark.parametrize("msg", NEGATIVE)
    def test_negative(self, msg):
        assert not _PRED_LOG_EXPORT_PATTERNS.search(msg), f"Should not match: {msg!r}"


# ---------------------------------------------------------------------------
# Fixtures
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

    SQLModel.metadata.create_all(db_module.engine)

    from main import app

    return TestClient(app)


@contextmanager
def _session():
    with Session(db_module.engine) as s:
        yield s


def _create_project_and_deployment(session, tmp_path):
    from models.deployment import Deployment
    from models.model_run import ModelRun
    from models.project import Project

    proj = Project(name="Test", status="deployed")
    session.add(proj)
    session.commit()
    session.refresh(proj)

    run = ModelRun(
        project_id=proj.id,
        algorithm="linear_regression",
        status="done",
        metrics=json.dumps({"r2": 0.85}),
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    dep = Deployment(
        model_run_id=run.id,
        project_id=proj.id,
        endpoint_path=f"/api/predict/{run.id}",
        dashboard_url=f"/predict/{run.id}",
        is_active=True,
        problem_type="regression",
        target_column="revenue",
    )
    session.add(dep)
    session.commit()
    session.refresh(dep)

    return proj, run, dep


def _add_prediction_logs(session, deployment_id: str, n: int = 5):
    from models.prediction_log import PredictionLog

    logs = []
    for i in range(n):
        log = PredictionLog(
            deployment_id=deployment_id,
            input_features=json.dumps({"units": 10 + i, "price": 50.0}),
            prediction=json.dumps(500.0 + i * 10),
            prediction_numeric=500.0 + i * 10,
            confidence=0.85 + i * 0.01,
            response_ms=12.5 + i,
            created_at=datetime(2024, 1, 1 + i, 9, 0, 0),
        )
        session.add(log)
        logs.append(log)
    session.commit()
    return logs


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


class TestPredictionLogExportEndpoint:
    def test_unknown_deployment_returns_404(self, client):
        resp = client.get("/api/deploy/nonexistent-id/prediction-logs/export")
        assert resp.status_code == 404

    def test_empty_logs_returns_csv_header_only(self, client):
        with _session() as s:
            _, _, dep = _create_project_and_deployment(s, None)
            dep_id = dep.id

        resp = client.get(f"/api/deploy/{dep_id}/prediction-logs/export")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        lines = resp.text.strip().splitlines()
        # Only the header row — no data rows
        assert len(lines) == 1
        assert "id" in lines[0]
        assert "created_at" in lines[0]
        assert "prediction" in lines[0]

    def test_returns_csv_with_correct_row_count(self, client):
        with _session() as s:
            _, _, dep = _create_project_and_deployment(s, None)
            dep_id = dep.id
            _add_prediction_logs(s, dep_id, n=3)

        resp = client.get(f"/api/deploy/{dep_id}/prediction-logs/export")
        assert resp.status_code == 200
        lines = resp.text.strip().splitlines()
        # 1 header + 3 data rows
        assert len(lines) == 4

    def test_csv_contains_feature_columns(self, client):
        with _session() as s:
            _, _, dep = _create_project_and_deployment(s, None)
            dep_id = dep.id
            _add_prediction_logs(s, dep_id, n=2)

        resp = client.get(f"/api/deploy/{dep_id}/prediction-logs/export")
        assert resp.status_code == 200
        header = resp.text.strip().splitlines()[0]
        assert "units" in header
        assert "price" in header

    def test_csv_content_disposition_header(self, client):
        with _session() as s:
            _, _, dep = _create_project_and_deployment(s, None)
            dep_id = dep.id

        resp = client.get(f"/api/deploy/{dep_id}/prediction-logs/export")
        assert resp.status_code == 200
        cd = resp.headers.get("content-disposition", "")
        assert "attachment" in cd
        assert ".csv" in cd

    def test_csv_data_values_correct(self, client):
        with _session() as s:
            _, _, dep = _create_project_and_deployment(s, None)
            dep_id = dep.id
            _add_prediction_logs(s, dep_id, n=1)

        resp = client.get(f"/api/deploy/{dep_id}/prediction-logs/export")
        assert resp.status_code == 200
        lines = resp.text.strip().splitlines()
        # Second line is first data row
        assert "500.0" in lines[1]
        assert "10" in lines[1]  # units value


# ---------------------------------------------------------------------------
# Chat handler integration tests
# ---------------------------------------------------------------------------


class TestPredLogExportChatHandler:
    def test_no_event_without_deployment(self, client):
        """When there is no active deployment, the export event should not fire."""
        proj = client.post("/api/projects", json={"name": "NoDeploy"}).json()
        pid = proj["id"]

        with _mock_anthropic():
            events = []
            with client.stream(
                "POST",
                f"/api/chat/{pid}",
                json={"message": "export prediction history"},
            ) as r:
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        events.append(json.loads(line[6:]))

        types = [e.get("type") for e in events]
        assert "prediction_log_export" not in types

    def test_emits_event_with_deployment(self, client, tmp_path):
        """When a deployment exists, the export event should fire."""
        with _session() as s:
            proj, run, dep = _create_project_and_deployment(s, tmp_path)
            _add_prediction_logs(s, dep.id, n=4)
            proj_id = proj.id

        with _mock_anthropic():
            events = []
            with client.stream(
                "POST",
                f"/api/chat/{proj_id}",
                json={"message": "export prediction history"},
            ) as r:
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        events.append(json.loads(line[6:]))

        export_events = [e for e in events if e.get("type") == "prediction_log_export"]
        assert len(export_events) == 1
        payload = export_events[0]["prediction_log_export"]
        assert payload["total_predictions"] == 4
        assert "/api/deploy/" in payload["download_url"]
        assert payload["first_prediction_at"] is not None
        assert payload["last_prediction_at"] is not None

    def test_event_has_required_fields(self, client, tmp_path):
        with _session() as s:
            proj, run, dep = _create_project_and_deployment(s, tmp_path)
            _add_prediction_logs(s, dep.id, n=2)
            proj_id = proj.id

        with _mock_anthropic():
            events = []
            with client.stream(
                "POST",
                f"/api/chat/{proj_id}",
                json={"message": "download prediction logs"},
            ) as r:
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        events.append(json.loads(line[6:]))

        export_events = [e for e in events if e.get("type") == "prediction_log_export"]
        assert export_events
        payload = export_events[0]["prediction_log_export"]
        for field in (
            "deployment_id",
            "total_predictions",
            "download_url",
            "first_prediction_at",
            "last_prediction_at",
        ):
            assert field in payload, f"Missing field: {field}"
