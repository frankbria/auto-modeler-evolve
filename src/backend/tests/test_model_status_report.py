"""Tests for Model Status Report Export feature.

Tests:
- HTML generation pure function (generate_model_status_report_html)
- REST endpoint (GET /api/deploy/{id}/status-report)
- Chat pattern matching (_STATUS_REPORT_PATTERNS)
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# HTML generation tests
# ---------------------------------------------------------------------------


class TestGenerateModelStatusReportHtml:
    def _gen(self, **kwargs):
        from core.report_generator import generate_model_status_report_html

        defaults = {
            "project_name": "Sales Forecast",
            "algorithm_plain": "Random Forest",
            "problem_type": "regression",
            "target_column": "revenue",
            "is_active": True,
            "deployed_at": "January 15, 2025",
        }
        defaults.update(kwargs)
        return generate_model_status_report_html(**defaults)

    def test_returns_html_string(self):
        html = self._gen()
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_contains_project_name(self):
        html = self._gen(project_name="My Revenue Model")
        assert "My Revenue Model" in html

    def test_contains_algorithm(self):
        html = self._gen(algorithm_plain="XGBoost")
        assert "XGBoost" in html

    def test_contains_target_column(self):
        html = self._gen(target_column="churn")
        assert "churn" in html

    def test_healthy_score_shows_healthy(self):
        html = self._gen(health_score=85, health_status="healthy")
        assert "Healthy" in html

    def test_warning_score_shows_warning(self):
        html = self._gen(health_score=60, health_status="warning")
        assert "Warning" in html

    def test_critical_score_shows_critical(self):
        html = self._gen(health_score=30, health_status="critical")
        assert "Critical" in html

    def test_volume_numbers_appear(self):
        html = self._gen(
            predictions_today=42,
            predictions_7d=300,
            predictions_30d=1200,
            predictions_total=9999,
        )
        assert "42" in html
        assert "300" in html
        assert "1,200" in html or "1200" in html

    def test_sla_alert_badge_appears(self):
        html = self._gen(sla_alert=True, p95_ms=800.0)
        assert "SLA Alert" in html

    def test_sla_ok_badge_appears(self):
        html = self._gen(sla_alert=False, p95_ms=120.0)
        assert "Within SLA" in html

    def test_alerts_section_appears(self):
        html = self._gen(
            alerts=[
                {
                    "severity": "critical",
                    "message": "Model is 95 days old",
                    "recommendation": "Retrain soon.",
                }
            ]
        )
        assert "Model is 95 days old" in html
        assert "Retrain soon." in html

    def test_no_alerts_shows_passing(self):
        html = self._gen(alerts=[])
        assert "No Active Alerts" in html

    def test_feedback_accuracy_shown(self):
        html = self._gen(
            feedback_accuracy=0.87,
            feedback_accuracy_label="good",
        )
        assert "87.0%" in html or "87%" in html
        assert "good" in html

    def test_recommendations_shown(self):
        html = self._gen(recommendations=["Consider retraining.", "Check SLA metrics."])
        assert "Consider retraining." in html
        assert "Check SLA metrics." in html

    def test_html_escape_prevents_xss(self):
        html = self._gen(project_name="<script>alert(1)</script>")
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_production_environment_badge(self):
        html = self._gen(environment="production")
        assert "Production" in html

    def test_staging_environment_badge(self):
        html = self._gen(environment="staging")
        assert "Staging" in html

    def test_footer_has_automodeler_branding(self):
        html = self._gen()
        assert "AutoModeler" in html

    def test_contains_generated_at_timestamp(self):
        html = self._gen()
        assert "Generated" in html and "UTC" in html


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


class TestModelStatusReportEndpoint:
    @pytest.mark.asyncio
    async def test_404_unknown_deployment(self, client):
        resp = await client.get("/api/deploy/nonexistent-id/status-report")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_html_for_live_deployment(self, client, tmp_path):
        """Create a minimal deployment+run and verify the endpoint returns HTML."""
        import uuid
        import db
        from sqlmodel import Session
        from models.deployment import Deployment
        from models.model_run import ModelRun
        from models.project import Project

        dep_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        proj_id = str(uuid.uuid4())
        with Session(db.engine) as s:
            proj = Project(id=proj_id, name="Test Project")
            s.add(proj)
            run = ModelRun(
                id=run_id,
                project_id=proj_id,
                algorithm="linear_regression",
                status="done",
                metrics="{}",
            )
            s.add(run)
            dep = Deployment(
                id=dep_id,
                model_run_id=run_id,
                project_id=proj_id,
                endpoint_path=f"/api/predict/{dep_id}",
                dashboard_url=f"/predict/{dep_id}",
                is_active=True,
                request_count=5,
                problem_type="regression",
                target_column="revenue",
            )
            s.add(dep)
            s.commit()

        resp = await client.get(f"/api/deploy/{dep_id}/status-report")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "<!DOCTYPE html>" in resp.text
        assert "Model Status Report" in resp.text
        assert 'filename="status_report_' in resp.headers.get("content-disposition", "")


# ---------------------------------------------------------------------------
# Chat pattern tests
# ---------------------------------------------------------------------------


class TestStatusReportPatterns:
    @pytest.fixture(autouse=True)
    def pattern(self):
        from api.chat import _STATUS_REPORT_PATTERNS

        self._pat = _STATUS_REPORT_PATTERNS

    def _match(self, text: str) -> bool:
        return bool(self._pat.search(text))

    # Positive matches
    def test_generate_status_report(self):
        assert self._match("generate a status report")

    def test_production_health_report(self):
        assert self._match("production health report")

    def test_monitoring_briefing(self):
        assert self._match("monitoring briefing for my VP")

    def test_download_deployment_health_report(self):
        assert self._match("download the deployment health report")

    def test_model_health_briefing(self):
        assert self._match("model health briefing for my manager")

    def test_shareable_monitoring_report(self):
        assert self._match("shareable monitoring report")

    def test_weekly_status_report(self):
        assert self._match("generate a weekly model status report")

    def test_monitoring_report_for_stakeholders(self):
        assert self._match("status report for stakeholders")

    # False-positive guards
    def test_model_health_check_no_match(self):
        # "model health" alone without report/briefing/etc. should NOT match
        assert not self._match("do a model health check please")

    def test_generic_report_no_match(self):
        assert not self._match("generate a model report")

    def test_pdf_report_no_match(self):
        assert not self._match("download the pdf report")

    def test_conversation_export_no_match(self):
        assert not self._match("export this conversation")
