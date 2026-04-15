"""Tests for Executive Briefing feature.

Covers:
- _BRIEFING_PATTERNS regex matching (positive and negative)
- generate_executive_briefing() pure function outputs
- GET /api/projects/{id}/executive-briefing endpoint
- Chat handler emitting executive_briefing SSE event
"""

from __future__ import annotations

import json
import unittest.mock as mock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
    b"East,175.2,18\nWest,220.4,22\nNorth,190.6,19\nEast,130.8,13\nWest,280.0,28\n"
    b"East,160.0,16\nWest,210.0,21\n"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def ac(tmp_path):
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

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
async def project_id(ac):
    r = await ac.post("/api/projects", json={"name": "Briefing Test Project"})
    return r.json()["id"]


# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


class TestBriefingPatterns:
    def setup_method(self):
        from api.chat import _BRIEFING_PATTERNS

        self.pattern = _BRIEFING_PATTERNS

    def test_write_briefing_for_vp(self):
        assert self.pattern.search("write a briefing for my VP")

    def test_create_executive_summary(self):
        assert self.pattern.search("create an executive summary")

    def test_prepare_summary_for_boss(self):
        assert self.pattern.search("prepare a summary for my boss")

    def test_explain_to_executive(self):
        assert self.pattern.search("explain this to my executive team")

    def test_talking_points_for_meeting(self):
        assert self.pattern.search("talking points for my VP meeting")

    def test_non_technical_summary(self):
        assert self.pattern.search("non-technical summary of the analysis")

    def test_present_to_stakeholder(self):
        assert self.pattern.search("how do I present results to my stakeholder")

    def test_write_up_the_findings(self):
        assert self.pattern.search("write up the findings")

    # Negative tests — should NOT match
    def test_negative_train_model(self):
        assert not self.pattern.search("train a new model")

    def test_negative_show_data(self):
        assert not self.pattern.search("show me my data")


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestGenerateExecutiveBriefing:
    def test_returns_required_keys(self):
        from core.storyteller import generate_executive_briefing

        result = generate_executive_briefing(
            project_name="Sales Forecast",
            dataset_filename="q1_sales.csv",
            row_count=200,
            col_count=8,
            target_column="revenue",
            problem_type="regression",
            algorithm="random_forest_regressor",
            primary_metric_name="r2",
            primary_metric_value=0.87,
            deployment_url="http://localhost:3000/predict/abc123",
            request_count=42,
            conversation_snippet=None,
        )
        for key in (
            "project_name",
            "target_column",
            "sections",
            "summary",
            "action_items",
            "algorithm",
        ):
            assert key in result, f"Missing key: {key}"

    def test_sections_have_heading_and_body(self):
        from core.storyteller import generate_executive_briefing

        result = generate_executive_briefing(
            project_name="Churn Model",
            dataset_filename="customers.csv",
            row_count=500,
            col_count=12,
            target_column="churned",
            problem_type="classification",
            algorithm="logistic_regression",
            primary_metric_name="accuracy",
            primary_metric_value=0.82,
            deployment_url=None,
            request_count=None,
            conversation_snippet=None,
        )
        assert len(result["sections"]) > 0
        for section in result["sections"]:
            assert "heading" in section
            assert "body" in section
            assert len(section["body"]) > 0

    def test_summary_contains_project_name(self):
        from core.storyteller import generate_executive_briefing

        result = generate_executive_briefing(
            project_name="Revenue Predictor",
            dataset_filename=None,
            row_count=None,
            col_count=None,
            target_column="revenue",
            problem_type="regression",
            algorithm="linear_regression",
            primary_metric_name="r2",
            primary_metric_value=0.75,
            deployment_url=None,
            request_count=None,
            conversation_snippet=None,
        )
        assert "Revenue Predictor" in result["summary"]

    def test_excellent_r2_metric_label(self):
        from core.storyteller import generate_executive_briefing

        result = generate_executive_briefing(
            project_name="P",
            dataset_filename=None,
            row_count=None,
            col_count=None,
            target_column=None,
            problem_type="regression",
            algorithm=None,
            primary_metric_name="r2",
            primary_metric_value=0.90,
            deployment_url=None,
            request_count=None,
            conversation_snippet=None,
        )
        # Metric label should mention R²
        assert result["metric_label"] is not None
        assert "R²" in result["metric_label"] or "r2" in result["metric_label"].lower()

    def test_action_items_present(self):
        from core.storyteller import generate_executive_briefing

        result = generate_executive_briefing(
            project_name="P",
            dataset_filename=None,
            row_count=None,
            col_count=None,
            target_column="sales",
            problem_type="regression",
            algorithm="ridge",
            primary_metric_name="r2",
            primary_metric_value=0.80,
            deployment_url="http://example.com/predict/1",
            request_count=5,
            conversation_snippet=None,
        )
        assert len(result["action_items"]) > 0

    def test_minimal_call_no_crash(self):
        """Should not crash when called with minimal info (no model yet)."""
        from core.storyteller import generate_executive_briefing

        result = generate_executive_briefing(
            project_name="Empty Project",
            dataset_filename=None,
            row_count=None,
            col_count=None,
            target_column=None,
            problem_type=None,
            algorithm=None,
            primary_metric_name=None,
            primary_metric_value=None,
            deployment_url=None,
            request_count=None,
            conversation_snippet=None,
        )
        assert result["project_name"] == "Empty Project"
        assert isinstance(result["sections"], list)


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestExecutiveBriefingEndpoint:
    async def test_returns_404_for_unknown_project(self, ac):
        r = await ac.get("/api/projects/nonexistent-id/executive-briefing")
        assert r.status_code == 404

    async def test_returns_200_for_empty_project(self, ac, project_id):
        r = await ac.get(f"/api/projects/{project_id}/executive-briefing")
        assert r.status_code == 200
        data = r.json()
        assert data["project_name"] == "Briefing Test Project"
        assert "sections" in data
        assert "summary" in data
        assert "action_items" in data

    async def test_project_id_present_in_response(self, ac, project_id):
        r = await ac.get(f"/api/projects/{project_id}/executive-briefing")
        assert r.status_code == 200
        data = r.json()
        assert data["project_id"] == project_id


# ---------------------------------------------------------------------------
# Chat integration test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBriefingChatHandler:
    async def test_briefing_event_emitted(self, ac, project_id):
        """Chat handler emits executive_briefing SSE event for briefing phrases."""
        with mock.patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = mock.MagicMock()
            mock_anthropic.return_value = mock_client
            mock_stream = mock.MagicMock()
            mock_client.messages.stream.return_value.__enter__ = mock.Mock(
                return_value=mock_stream
            )
            mock_client.messages.stream.return_value.__exit__ = mock.Mock(
                return_value=False
            )
            mock_stream.text_stream = iter(["Here is your briefing."])

            r = await ac.post(
                f"/api/chat/{project_id}",
                json={"message": "write a briefing for my VP"},
            )
        assert r.status_code == 200
        events = [
            json.loads(line[6:])
            for line in r.text.splitlines()
            if line.startswith("data:") and line[6:].strip()
        ]
        event_types = [e.get("type") for e in events]
        assert "executive_briefing" in event_types

    async def test_briefing_event_has_required_fields(self, ac, project_id):
        """executive_briefing event payload contains all required fields."""
        with mock.patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = mock.MagicMock()
            mock_anthropic.return_value = mock_client
            mock_stream = mock.MagicMock()
            mock_client.messages.stream.return_value.__enter__ = mock.Mock(
                return_value=mock_stream
            )
            mock_client.messages.stream.return_value.__exit__ = mock.Mock(
                return_value=False
            )
            mock_stream.text_stream = iter(["Briefing ready."])

            r = await ac.post(
                f"/api/chat/{project_id}",
                json={"message": "create an executive summary"},
            )

        events = [
            json.loads(line[6:])
            for line in r.text.splitlines()
            if line.startswith("data:") and line[6:].strip()
        ]
        briefing_event = next(
            (e for e in events if e.get("type") == "executive_briefing"), None
        )
        assert briefing_event is not None
        payload = briefing_event["executive_briefing"]
        assert "sections" in payload
        assert "summary" in payload
        assert "action_items" in payload

    async def test_no_briefing_event_on_unrelated_message(self, ac, project_id):
        """Unrelated messages do not emit executive_briefing event."""
        with mock.patch("anthropic.Anthropic") as mock_anthropic:
            mock_client = mock.MagicMock()
            mock_anthropic.return_value = mock_client
            mock_stream = mock.MagicMock()
            mock_client.messages.stream.return_value.__enter__ = mock.Mock(
                return_value=mock_stream
            )
            mock_client.messages.stream.return_value.__exit__ = mock.Mock(
                return_value=False
            )
            mock_stream.text_stream = iter(["Hello!"])

            r = await ac.post(
                f"/api/chat/{project_id}",
                json={"message": "show me my data"},
            )

        events = [
            json.loads(line[6:])
            for line in r.text.splitlines()
            if line.startswith("data:") and line[6:].strip()
        ]
        event_types = [e.get("type") for e in events]
        assert "executive_briefing" not in event_types
