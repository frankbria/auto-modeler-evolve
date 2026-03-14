"""Tests for the template projects API.

Covers:
- GET /api/templates — list all templates
- GET /api/templates/{id} — get template detail
- POST /api/templates/{id}/apply — create project from template
"""

from __future__ import annotations

import pytest


class TestListTemplates:
    async def test_returns_template_list(self, client):
        resp = await client.get("/api/templates")
        assert resp.status_code == 200
        body = resp.json()
        assert "templates" in body
        templates = body["templates"]
        assert len(templates) >= 3

    async def test_template_has_required_fields(self, client):
        resp = await client.get("/api/templates")
        templates = resp.json()["templates"]
        for tpl in templates:
            assert "id" in tpl
            assert "name" in tpl
            assert "description" in tpl
            assert "use_case" in tpl
            assert "target_column" in tpl
            assert "problem_type" in tpl
            assert "tags" in tpl
            assert "difficulty" in tpl

    async def test_includes_sales_forecast(self, client):
        resp = await client.get("/api/templates")
        ids = [t["id"] for t in resp.json()["templates"]]
        assert "sales_forecast" in ids

    async def test_includes_customer_churn(self, client):
        resp = await client.get("/api/templates")
        ids = [t["id"] for t in resp.json()["templates"]]
        assert "customer_churn" in ids

    async def test_includes_demand_forecast(self, client):
        resp = await client.get("/api/templates")
        ids = [t["id"] for t in resp.json()["templates"]]
        assert "demand_forecast" in ids


class TestGetTemplate:
    async def test_sales_forecast_template_detail(self, client):
        resp = await client.get("/api/templates/sales_forecast")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "sales_forecast"
        assert body["target_column"] == "revenue"
        assert body["problem_type"] == "regression"
        assert "conversation_starter" in body
        assert len(body["conversation_starter"]) > 50

    async def test_customer_churn_template_detail(self, client):
        resp = await client.get("/api/templates/customer_churn")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "customer_churn"
        assert body["target_column"] == "churn"
        assert body["problem_type"] == "classification"
        assert "suggested_algorithms" in body
        assert len(body["suggested_algorithms"]) >= 2

    async def test_nonexistent_template_404(self, client):
        resp = await client.get("/api/templates/doesnt_exist_xyz")
        assert resp.status_code == 404


class TestApplyTemplate:
    async def test_apply_sales_forecast_creates_project(self, client):
        resp = await client.post("/api/templates/sales_forecast/apply")
        assert resp.status_code == 201
        body = resp.json()
        assert "project_id" in body
        assert "dataset_id" in body
        assert body["template_id"] == "sales_forecast"
        assert body["target_column"] == "revenue"
        assert body["problem_type"] == "regression"
        assert body["row_count"] > 0

    async def test_apply_creates_accessible_project(self, client):
        """The created project should be retrievable via GET /api/projects/{id}."""
        apply_resp = await client.post("/api/templates/sales_forecast/apply")
        assert apply_resp.status_code == 201
        project_id = apply_resp.json()["project_id"]

        project_resp = await client.get(f"/api/projects/{project_id}")
        assert project_resp.status_code == 200
        project = project_resp.json()
        assert project["id"] == project_id
        assert project["name"] == "Sales Revenue Forecast"

    async def test_apply_creates_dataset(self, client):
        """Dataset should exist and have a preview."""
        apply_resp = await client.post("/api/templates/customer_churn/apply")
        assert apply_resp.status_code == 201
        body = apply_resp.json()
        dataset_id = body["dataset_id"]

        preview_resp = await client.get(f"/api/data/{dataset_id}/preview")
        assert preview_resp.status_code == 200
        preview = preview_resp.json()
        assert preview["row_count"] == 300
        assert "churn" in [c["name"] for c in preview["column_stats"]]

    async def test_apply_demand_forecast(self, client):
        resp = await client.post("/api/templates/demand_forecast/apply")
        assert resp.status_code == 201
        body = resp.json()
        assert body["target_column"] == "units_sold"
        assert body["problem_type"] == "regression"
        assert "units_sold" in body["columns"]

    async def test_apply_includes_conversation_starter(self, client):
        resp = await client.post("/api/templates/customer_churn/apply")
        assert resp.status_code == 201
        body = resp.json()
        assert "conversation_starter" in body
        starter = body["conversation_starter"]
        assert len(starter) > 100  # Should be a meaningful prompt
        assert "churn" in starter.lower()

    async def test_apply_suggests_algorithms(self, client):
        resp = await client.post("/api/templates/customer_churn/apply")
        assert resp.status_code == 201
        body = resp.json()
        assert "suggested_algorithms" in body
        algos = body["suggested_algorithms"]
        assert len(algos) >= 2

    async def test_apply_nonexistent_template_404(self, client):
        resp = await client.post("/api/templates/not_a_real_template/apply")
        assert resp.status_code == 404

    async def test_multiple_applies_create_separate_projects(self, client):
        """Each call creates a fresh independent project."""
        r1 = await client.post("/api/templates/sales_forecast/apply")
        r2 = await client.post("/api/templates/sales_forecast/apply")
        assert r1.status_code == 201
        assert r2.status_code == 201
        assert r1.json()["project_id"] != r2.json()["project_id"]
        assert r1.json()["dataset_id"] != r2.json()["dataset_id"]

    async def test_apply_churn_dataset_has_correct_columns(self, client):
        resp = await client.post("/api/templates/customer_churn/apply")
        body = resp.json()
        expected_cols = {"tenure_months", "monthly_charge", "support_calls", "churn"}
        assert expected_cols.issubset(set(body["columns"]))

    async def test_sales_dataset_row_count(self, client):
        resp = await client.post("/api/templates/sales_forecast/apply")
        assert resp.json()["row_count"] == 200

    async def test_churn_dataset_row_count(self, client):
        resp = await client.post("/api/templates/customer_churn/apply")
        assert resp.json()["row_count"] == 300

    async def test_demand_dataset_row_count(self, client):
        resp = await client.post("/api/templates/demand_forecast/apply")
        assert resp.json()["row_count"] == 250
