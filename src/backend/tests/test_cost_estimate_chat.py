"""Tests for deployment cost / capacity estimate chat card."""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

import db as db_module
from api.chat import _COST_ESTIMATE_PATTERNS, _extract_cost_n

# ---------------------------------------------------------------------------
# Pattern tests
# ---------------------------------------------------------------------------


class TestCostEstimatePatterns:
    POSITIVE = [
        "how much would 1000 predictions cost?",
        "how much would predictions cost?",
        "estimate prediction cost",
        "estimate the capacity",
        "cost for 500 predictions",
        "how many users can my model handle?",
        "how many requests can the api handle?",
        "prediction capacity planning",
        "optimal rate limit for this use case",
        "quota capacity planning",
    ]
    NEGATIVE = [
        "train a model",
        "quota runway",
        "will my quota last",
        "show me the confusion matrix",
        "set rate limit to 100",
        "hello",
        "what is my accuracy",
        "how many predictions did I make",
    ]

    @pytest.mark.parametrize("msg", POSITIVE)
    def test_positive(self, msg):
        assert _COST_ESTIMATE_PATTERNS.search(msg), f"Should match: {msg!r}"

    @pytest.mark.parametrize("msg", NEGATIVE)
    def test_negative(self, msg):
        assert not _COST_ESTIMATE_PATTERNS.search(msg), f"Should not match: {msg!r}"


# ---------------------------------------------------------------------------
# _extract_cost_n tests
# ---------------------------------------------------------------------------


class TestExtractCostN:
    def test_plain_number(self):
        assert _extract_cost_n("how much would 1000 predictions cost?") == 1000

    def test_comma_formatted(self):
        assert _extract_cost_n("cost for 10,000 predictions") == 10000

    def test_k_suffix(self):
        assert _extract_cost_n("how many 5k predictions can I make?") == 5000

    def test_m_suffix(self):
        assert _extract_cost_n("1m users per month") == 1_000_000

    def test_requests_word(self):
        assert _extract_cost_n("handle 200 requests per day") == 200

    def test_users_word(self):
        assert _extract_cost_n("500 users daily") == 500

    def test_no_number_defaults_to_1000(self):
        assert _extract_cost_n("how much would predictions cost?") == 1000

    def test_minimum_one(self):
        assert _extract_cost_n("0 predictions") >= 1


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

    SQLModel.metadata.create_all(db_module.engine)
    from main import app

    return TestClient(app)


@contextmanager
def _mock_anthropic():
    with mock.patch("anthropic.Anthropic") as m:
        mock_cli = mock.MagicMock()
        m.return_value = mock_cli
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Capacity analysis complete."])
        mock_cli.messages.stream.return_value = mock_stream
        yield


def _setup_project_with_deployment(client, *, monthly_quota=None, rate_limit_rpm=None):
    from models.deployment import Deployment
    from models.model_run import ModelRun

    proj = client.post("/api/projects", json={"name": "CostTest"}).json()
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
        dep = Deployment(
            id=dep_id,
            model_run_id=run_id,
            project_id=project_id,
            endpoint_path=f"/api/predict/{dep_id}",
            dashboard_url=f"/predict/{dep_id}",
            is_active=True,
            algorithm="linear_regression",
            problem_type="regression",
            feature_names=json.dumps(["units"]),
            target_column="revenue",
            metrics=json.dumps({"r2": 0.85}),
        )
        if monthly_quota is not None:
            dep.monthly_quota = monthly_quota
        if rate_limit_rpm is not None:
            dep.rate_limit_rpm = rate_limit_rpm
        sess.add(dep)
        sess.commit()

    return project_id, run_id, dep_id


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestCostEstimateChatHandler:
    def test_no_event_without_deployment(self, client):
        """Cost estimate event requires a deployment context."""
        proj = client.post("/api/projects", json={"name": "NoDeploy"}).json()
        pid = proj["id"]
        with _mock_anthropic():
            with client.stream(
                "POST",
                f"/api/chat/{pid}",
                json={
                    "message": "how much would 1000 predictions cost?",
                    "conversation_history": [],
                },
            ) as resp:
                events = [
                    line
                    for line in resp.iter_lines()
                    if line.startswith("data:") and "cost_estimate" in line
                ]
        assert len(events) == 0

    def test_emits_cost_estimate_event(self, client):
        """Emits cost_estimate SSE event when deployment is active."""
        project_id, _, _ = _setup_project_with_deployment(client)

        with _mock_anthropic():
            with client.stream(
                "POST",
                f"/api/chat/{project_id}",
                json={
                    "message": "how much would 1000 predictions cost?",
                    "conversation_history": [],
                },
            ) as resp:
                events = [
                    line for line in resp.iter_lines() if '"cost_estimate"' in line
                ]

        assert len(events) >= 1
        payload = json.loads(events[0][len("data:") :])
        assert payload["type"] == "cost_estimate"
        assert "cost_estimate" in payload

    def test_event_required_fields(self, client):
        """Cost estimate event contains all required fields."""
        project_id, _, _ = _setup_project_with_deployment(client)

        with _mock_anthropic():
            with client.stream(
                "POST",
                f"/api/chat/{project_id}",
                json={
                    "message": "estimate prediction cost",
                    "conversation_history": [],
                },
            ) as resp:
                events = [
                    line for line in resp.iter_lines() if '"cost_estimate"' in line
                ]

        assert len(events) >= 1
        result = json.loads(events[0][len("data:") :])["cost_estimate"]
        required = {
            "deployment_id",
            "n_predictions",
            "monthly_quota",
            "used_this_month",
            "quota_pct",
            "within_quota",
            "current_rpm",
            "daily_capacity",
            "avg_per_day",
            "days_needed",
            "recommended_rpm",
        }
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_no_quota_state(self, client):
        """Without monthly quota, quota_pct and within_quota are None."""
        project_id, _, _ = _setup_project_with_deployment(client)

        with _mock_anthropic():
            with client.stream(
                "POST",
                f"/api/chat/{project_id}",
                json={
                    "message": "how much would 500 predictions cost?",
                    "conversation_history": [],
                },
            ) as resp:
                events = [
                    line for line in resp.iter_lines() if '"cost_estimate"' in line
                ]

        result = json.loads(events[0][len("data:") :])["cost_estimate"]
        assert result["monthly_quota"] is None
        assert result["quota_pct"] is None
        assert result["within_quota"] is None

    def test_with_quota_set(self, client):
        """With monthly quota set, quota_pct and within_quota are computed."""
        project_id, _, _ = _setup_project_with_deployment(client, monthly_quota=2000)

        with _mock_anthropic():
            with client.stream(
                "POST",
                f"/api/chat/{project_id}",
                json={
                    "message": "how much would 1000 predictions cost?",
                    "conversation_history": [],
                },
            ) as resp:
                events = [
                    line for line in resp.iter_lines() if '"cost_estimate"' in line
                ]

        result = json.loads(events[0][len("data:") :])["cost_estimate"]
        assert result["monthly_quota"] == 2000
        assert result["quota_pct"] == 50.0  # 1000/2000
        assert result["within_quota"] is True

    def test_n_predictions_extracted_from_message(self, client):
        """n_predictions in event matches number extracted from message."""
        project_id, _, _ = _setup_project_with_deployment(client)

        with _mock_anthropic():
            with client.stream(
                "POST",
                f"/api/chat/{project_id}",
                json={
                    "message": "how much would 500 predictions cost?",
                    "conversation_history": [],
                },
            ) as resp:
                events = [
                    line for line in resp.iter_lines() if '"cost_estimate"' in line
                ]

        result = json.loads(events[0][len("data:") :])["cost_estimate"]
        assert result["n_predictions"] == 500

    def test_recommended_rpm_positive(self, client):
        """recommended_rpm is always a positive integer."""
        project_id, _, _ = _setup_project_with_deployment(client)

        with _mock_anthropic():
            with client.stream(
                "POST",
                f"/api/chat/{project_id}",
                json={
                    "message": "prediction capacity planning",
                    "conversation_history": [],
                },
            ) as resp:
                events = [
                    line for line in resp.iter_lines() if '"cost_estimate"' in line
                ]

        result = json.loads(events[0][len("data:") :])["cost_estimate"]
        assert result["recommended_rpm"] >= 1

    def test_rate_limit_sets_daily_capacity(self, client):
        """daily_capacity is computed when rate_limit_rpm is set."""
        project_id, _, _ = _setup_project_with_deployment(client, rate_limit_rpm=60)

        with _mock_anthropic():
            with client.stream(
                "POST",
                f"/api/chat/{project_id}",
                json={
                    "message": "how many users can my model handle?",
                    "conversation_history": [],
                },
            ) as resp:
                events = [
                    line for line in resp.iter_lines() if '"cost_estimate"' in line
                ]

        result = json.loads(events[0][len("data:") :])["cost_estimate"]
        assert result["current_rpm"] == 60
        # 60 RPM × 60 min × 24 h = 86400
        assert result["daily_capacity"] == 86400
