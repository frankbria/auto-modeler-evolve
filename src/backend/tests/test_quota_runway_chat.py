"""Tests for quota runway / capacity planning chat card."""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

import db as db_module
from api.chat import _QUOTA_RUNWAY_PATTERNS

# ---------------------------------------------------------------------------
# Pattern tests
# ---------------------------------------------------------------------------


class TestQuotaRunwayPatterns:
    POSITIVE = [
        "will my quota last the month?",
        "at this rate when will I run out?",
        "quota runway",
        "quota forecast",
        "quota projection",
        "how long until my quota runs out",
        "quota exhaustion",
        "prediction budget analysis",
        "when will my monthly quota run out?",
    ]
    NEGATIVE = [
        "train a model",
        "show me the confusion matrix",
        "how many predictions left",  # covered by rate_limit patterns
        "set rate limit to 100",
        "hello",
        "what is my accuracy",
    ]

    @pytest.mark.parametrize("msg", POSITIVE)
    def test_positive(self, msg):
        assert _QUOTA_RUNWAY_PATTERNS.search(msg), f"Should match: {msg!r}"

    @pytest.mark.parametrize("msg", NEGATIVE)
    def test_negative(self, msg):
        assert not _QUOTA_RUNWAY_PATTERNS.search(msg), f"Should not match: {msg!r}"


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
        mock_stream.text_stream = iter(["Quota analysis complete."])
        mock_cli.messages.stream.return_value = mock_stream
        yield


def _setup_project_with_deployment(client, *, monthly_quota=None, rate_limit_rpm=None):
    """Create project + inject ModelRun + Deployment directly."""
    from models.deployment import Deployment
    from models.model_run import ModelRun

    proj = client.post("/api/projects", json={"name": "QuotaTest"}).json()
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
        dep_kwargs = dict(
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
        dep = Deployment(**dep_kwargs)
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


class TestQuotaRunwayChatHandler:
    def test_no_event_without_deployment(self, client):
        """Quota runway event requires a deployment context."""
        proj = client.post("/api/projects", json={"name": "NoDeploy"}).json()
        pid = proj["id"]
        with _mock_anthropic():
            with client.stream(
                "POST",
                f"/api/chat/{pid}",
                json={"message": "quota runway", "conversation_history": []},
            ) as resp:
                events = [
                    line
                    for line in resp.iter_lines()
                    if line.startswith("data:") and "quota_runway" in line
                ]
        assert len(events) == 0

    def test_emits_quota_runway_event(self, client):
        """Emits quota_runway SSE event when deployment is active."""
        project_id, _, _ = _setup_project_with_deployment(client)

        with _mock_anthropic():
            with client.stream(
                "POST",
                f"/api/chat/{project_id}",
                json={"message": "quota runway", "conversation_history": []},
            ) as resp:
                events = [line for line in resp.iter_lines() if line.startswith("data:")]

        quota_events = [e for e in events if '"quota_runway"' in e]
        assert len(quota_events) >= 1
        payload = json.loads(quota_events[0][len("data:"):])
        assert payload["type"] == "quota_runway"
        assert "quota_runway" in payload

    def test_event_required_fields(self, client):
        """Quota runway event contains all required fields."""
        project_id, _, _ = _setup_project_with_deployment(client)

        with _mock_anthropic():
            with client.stream(
                "POST",
                f"/api/chat/{project_id}",
                json={"message": "quota forecast", "conversation_history": []},
            ) as resp:
                events = [line for line in resp.iter_lines() if '"quota_runway"' in line]

        assert len(events) >= 1
        result = json.loads(events[0][len("data:"):])["quota_runway"]
        required = {
            "deployment_id", "has_quota", "monthly_quota", "used_this_month",
            "remaining", "avg_per_day", "days_left_at_rate", "est_month_total",
            "days_remaining_in_month", "rate_limit_rpm", "will_exhaust",
        }
        for field in required:
            assert field in result, f"Missing field: {field}"

    def test_no_quota_state(self, client):
        """When no monthly quota is set, has_quota is False and will_exhaust is False."""
        project_id, _, _ = _setup_project_with_deployment(client)

        with _mock_anthropic():
            with client.stream(
                "POST",
                f"/api/chat/{project_id}",
                json={"message": "quota runway", "conversation_history": []},
            ) as resp:
                events = [line for line in resp.iter_lines() if '"quota_runway"' in line]

        result = json.loads(events[0][len("data:"):])["quota_runway"]
        assert result["has_quota"] is False
        assert result["monthly_quota"] is None
        assert result["remaining"] is None
        assert result["will_exhaust"] is False

    def test_with_quota_set(self, client):
        """When monthly quota is set, has_quota is True and remaining is computed."""
        project_id, _, _ = _setup_project_with_deployment(
            client, monthly_quota=500
        )

        with _mock_anthropic():
            with client.stream(
                "POST",
                f"/api/chat/{project_id}",
                json={"message": "quota projection", "conversation_history": []},
            ) as resp:
                events = [line for line in resp.iter_lines() if '"quota_runway"' in line]

        result = json.loads(events[0][len("data:"):])["quota_runway"]
        assert result["has_quota"] is True
        assert result["monthly_quota"] == 500
        assert result["remaining"] == 500  # no predictions yet
        assert isinstance(result["est_month_total"], int)

    def test_rate_limit_included(self, client):
        """Rate limit RPM is included in the event when set."""
        project_id, _, _ = _setup_project_with_deployment(
            client, rate_limit_rpm=100
        )

        with _mock_anthropic():
            with client.stream(
                "POST",
                f"/api/chat/{project_id}",
                json={"message": "quota runway", "conversation_history": []},
            ) as resp:
                events = [line for line in resp.iter_lines() if '"quota_runway"' in line]

        result = json.loads(events[0][len("data:"):])["quota_runway"]
        assert result["rate_limit_rpm"] == 100
