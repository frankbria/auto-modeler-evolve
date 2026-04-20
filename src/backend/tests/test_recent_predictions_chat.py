"""Tests for Recent Predictions Table via Chat (Track D perpetual)."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

import db as db_module
from api.chat import _RECENT_PRED_LOG_PATTERNS, _extract_recent_pred_n


@contextmanager
def _mock_anthropic():
    with mock.patch("anthropic.Anthropic") as m:
        mock_cli = mock.MagicMock()
        m.return_value = mock_cli
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Here are your recent predictions."])
        mock_cli.messages.stream.return_value = mock_stream
        yield


# ---------------------------------------------------------------------------
# Regex pattern tests
# ---------------------------------------------------------------------------


class TestRecentPredLogPatterns:
    POSITIVE = [
        "show me recent predictions",
        "show recent predictions",
        "show me the last 10 predictions",
        "what were the last 5 predictions",
        "list recent predictions",
        "recent api calls",
        "latest prediction requests",
        "show today's predictions",
        "show me yesterday's predictions",
        "browse predictions",
        "inspect recent predictions",
        "view my recent predictions",
        "prediction log table",
        "prediction feed",
        "show me what my model predicted recently",
        "last 20 prediction results",
    ]
    NEGATIVE = [
        "export prediction history",
        "download prediction logs",
        "how many predictions have I made?",
        "show prediction analytics",
        "train a model",
        "what is my quota runway?",
        "when is my model busiest?",
        "is my model drifting?",
        "latency report",
        "hello",
        "set rate limit to 100 rpm",
    ]

    @pytest.mark.parametrize("msg", POSITIVE)
    def test_positive(self, msg):
        assert _RECENT_PRED_LOG_PATTERNS.search(msg), f"Should match: {msg!r}"

    @pytest.mark.parametrize("msg", NEGATIVE)
    def test_negative(self, msg):
        assert not _RECENT_PRED_LOG_PATTERNS.search(msg), f"Should not match: {msg!r}"


# ---------------------------------------------------------------------------
# Helper extractor tests
# ---------------------------------------------------------------------------


class TestExtractRecentPredN:
    def test_default_is_10(self):
        assert _extract_recent_pred_n("show me recent predictions") == 10

    def test_parses_last_n(self):
        assert _extract_recent_pred_n("show me the last 5 predictions") == 5

    def test_parses_recent_n(self):
        assert _extract_recent_pred_n("show recent 20 predictions") == 20

    def test_parses_n_before_keyword(self):
        assert _extract_recent_pred_n("15 recent predictions please") == 15

    def test_clamps_to_1(self):
        assert _extract_recent_pred_n("last 0 predictions") == 1

    def test_clamps_to_50(self):
        assert _extract_recent_pred_n("last 999 predictions") == 50

    def test_whole_number_correctly_parsed(self):
        assert _extract_recent_pred_n("what were the last 3 predictions") == 3


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


def _create_project_and_deployment(session):
    from models.deployment import Deployment
    from models.model_run import ModelRun
    from models.project import Project

    proj = Project(name="RPTest", status="deployed")
    session.add(proj)
    session.commit()
    session.refresh(proj)

    run = ModelRun(
        project_id=proj.id,
        algorithm="random_forest_regressor",
        status="done",
        metrics=json.dumps({"r2": 0.90}),
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

    for i in range(n):
        log = PredictionLog(
            deployment_id=deployment_id,
            input_features=json.dumps({"units": 10 + i, "region": "East"}),
            prediction=json.dumps(500.0 + i * 10),
            prediction_numeric=500.0 + i * 10,
            confidence=0.80 + i * 0.02,
            response_ms=10.0 + i,
            created_at=datetime(2024, 3, 1 + i, 12, 0, 0),
        )
        session.add(log)
    session.commit()


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


class TestRecentPredictionsEndpoint:
    def test_unknown_deployment_returns_404(self, client):
        resp = client.get("/api/deploy/nonexistent/recent-predictions")
        assert resp.status_code == 404

    def test_empty_log_returns_zero_total(self, client):
        with _session() as s:
            _, _, dep = _create_project_and_deployment(s)
            dep_id = dep.id

        resp = client.get(f"/api/deploy/{dep_id}/recent-predictions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_all_time"] == 0
        assert body["n_shown"] == 0
        assert body["predictions"] == []

    def test_returns_predictions_in_desc_order(self, client):
        with _session() as s:
            _, _, dep = _create_project_and_deployment(s)
            dep_id = dep.id
            _add_prediction_logs(s, dep_id, n=3)

        resp = client.get(f"/api/deploy/{dep_id}/recent-predictions?n=10")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_shown"] == 3
        assert body["total_all_time"] == 3
        # most recent first (day 3 > day 2 > day 1)
        created_at_list = [p["created_at"] for p in body["predictions"]]
        assert created_at_list == sorted(created_at_list, reverse=True)

    def test_n_param_limits_results(self, client):
        with _session() as s:
            _, _, dep = _create_project_and_deployment(s)
            dep_id = dep.id
            _add_prediction_logs(s, dep_id, n=10)

        resp = client.get(f"/api/deploy/{dep_id}/recent-predictions?n=3")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_shown"] == 3
        assert body["total_all_time"] == 10

    def test_response_fields_present(self, client):
        with _session() as s:
            _, _, dep = _create_project_and_deployment(s)
            dep_id = dep.id
            _add_prediction_logs(s, dep_id, n=2)

        resp = client.get(f"/api/deploy/{dep_id}/recent-predictions")
        body = resp.json()
        for field in ("deployment_id", "n_shown", "total_all_time", "predictions", "export_url", "summary"):
            assert field in body, f"Missing field: {field}"

    def test_prediction_row_fields(self, client):
        with _session() as s:
            _, _, dep = _create_project_and_deployment(s)
            dep_id = dep.id
            _add_prediction_logs(s, dep_id, n=1)

        resp = client.get(f"/api/deploy/{dep_id}/recent-predictions")
        row = resp.json()["predictions"][0]
        for field in ("id", "created_at", "prediction", "confidence", "response_ms", "input_summary"):
            assert field in row, f"Missing row field: {field}"

    def test_input_summary_max_3_items(self, client):
        with _session() as s:
            _, _, dep = _create_project_and_deployment(s)
            dep_id = dep.id
            _add_prediction_logs(s, dep_id, n=1)

        resp = client.get(f"/api/deploy/{dep_id}/recent-predictions")
        row = resp.json()["predictions"][0]
        assert len(row["input_summary"]) <= 3


# ---------------------------------------------------------------------------
# Chat handler integration tests
# ---------------------------------------------------------------------------


class TestRecentPredictionsChatHandler:
    def test_no_event_without_deployment(self, client):
        proj = client.post("/api/projects", json={"name": "NoDeploy2"}).json()
        pid = proj["id"]

        with _mock_anthropic():
            events = []
            with client.stream(
                "POST",
                f"/api/chat/{pid}",
                json={"message": "show me recent predictions"},
            ) as r:
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        events.append(json.loads(line[6:]))

        types = [e.get("type") for e in events]
        assert "recent_predictions" not in types

    def test_emits_event_with_deployment(self, client):
        with _session() as s:
            proj, _, dep = _create_project_and_deployment(s)
            _add_prediction_logs(s, dep.id, n=3)
            proj_id = proj.id

        with _mock_anthropic():
            events = []
            with client.stream(
                "POST",
                f"/api/chat/{proj_id}",
                json={"message": "show me recent predictions"},
            ) as r:
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        events.append(json.loads(line[6:]))

        rp_events = [e for e in events if e.get("type") == "recent_predictions"]
        assert len(rp_events) == 1
        payload = rp_events[0]["recent_predictions"]
        assert payload["total_all_time"] == 3
        assert payload["n_shown"] <= 3

    def test_event_required_fields(self, client):
        with _session() as s:
            proj, _, dep = _create_project_and_deployment(s)
            _add_prediction_logs(s, dep.id, n=2)
            proj_id = proj.id

        with _mock_anthropic():
            events = []
            with client.stream(
                "POST",
                f"/api/chat/{proj_id}",
                json={"message": "list recent predictions"},
            ) as r:
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        events.append(json.loads(line[6:]))

        rp_events = [e for e in events if e.get("type") == "recent_predictions"]
        assert rp_events
        payload = rp_events[0]["recent_predictions"]
        for field in ("deployment_id", "n_shown", "total_all_time", "predictions", "export_url", "summary"):
            assert field in payload, f"Missing: {field}"

    def test_empty_log_emits_zero_total(self, client):
        with _session() as s:
            proj, _, dep = _create_project_and_deployment(s)
            proj_id = proj.id

        with _mock_anthropic():
            events = []
            with client.stream(
                "POST",
                f"/api/chat/{proj_id}",
                json={"message": "show recent predictions"},
            ) as r:
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        events.append(json.loads(line[6:]))

        rp_events = [e for e in events if e.get("type") == "recent_predictions"]
        assert rp_events
        assert rp_events[0]["recent_predictions"]["total_all_time"] == 0

    def test_export_event_takes_priority_over_recent(self, client):
        """When message triggers export pattern, recent_predictions should NOT fire."""
        with _session() as s:
            proj, _, dep = _create_project_and_deployment(s)
            _add_prediction_logs(s, dep.id, n=2)
            proj_id = proj.id

        with _mock_anthropic():
            events = []
            with client.stream(
                "POST",
                f"/api/chat/{proj_id}",
                json={"message": "download prediction logs as csv"},
            ) as r:
                for line in r.iter_lines():
                    if line.startswith("data:"):
                        events.append(json.loads(line[6:]))

        types = [e.get("type") for e in events]
        assert "prediction_log_export" in types
        assert "recent_predictions" not in types
