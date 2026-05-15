"""Tests for generate_model_card_html(), export-model-card endpoint, and chat regex."""

from __future__ import annotations


import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

# ---------------------------------------------------------------------------
# Shared fixture data
# ---------------------------------------------------------------------------

CLASSIFICATION_CSV = b"""age,income,churned
25,45000,yes
45,80000,no
30,35000,yes
55,120000,no
28,42000,yes
60,95000,no
35,55000,no
40,70000,yes
22,30000,yes
50,100000,no
33,48000,no
48,88000,no
27,38000,yes
52,110000,no
31,41000,yes
"""

_MINIMAL_CARD_KWARGS: dict = dict(
    project_name="Demo Project",
    algorithm_plain="Random Forest",
    problem_type="classification",
    target_column="churned",
    metric_name="f1",
    metric_display="0.84",
    metric_plain_english="84% F1 score — correctly identifies positive cases 84% of the time",
    row_count=1000,
    feature_count=5,
    top_features=[
        {"feature": "income", "importance": 0.45},
        {"feature": "age", "importance": 0.30},
    ],
    limitations=["Trained on historical data", "May not generalise to new markets"],
    trained_at="2026-05-15T12:00:00",
)


# ---------------------------------------------------------------------------
# Unit tests: generate_model_card_html()
# ---------------------------------------------------------------------------


class TestGenerateModelCardHtml:
    def setup_method(self):
        from core.report_generator import generate_model_card_html

        self.fn = generate_model_card_html

    def test_returns_html_string(self):
        html = self.fn(**_MINIMAL_CARD_KWARGS)
        assert isinstance(html, str)
        assert html.startswith("<!DOCTYPE html>") or "<html" in html

    def test_contains_project_name(self):
        html = self.fn(**_MINIMAL_CARD_KWARGS)
        assert "Demo Project" in html

    def test_contains_algorithm(self):
        html = self.fn(**_MINIMAL_CARD_KWARGS)
        assert "Random Forest" in html

    def test_contains_metric(self):
        html = self.fn(**_MINIMAL_CARD_KWARGS)
        assert "0.84" in html

    def test_contains_top_features(self):
        html = self.fn(**_MINIMAL_CARD_KWARGS)
        assert "income" in html
        assert "age" in html

    def test_limitations_included(self):
        html = self.fn(**_MINIMAL_CARD_KWARGS)
        assert "Trained on historical data" in html

    def test_calibration_section_when_provided(self):
        html = self.fn(
            **_MINIMAL_CARD_KWARGS,
            calibration_note="Well calibrated",
            brier_score=0.08,
        )
        assert "Calibration" in html
        assert "0.08" in html

    def test_no_calibration_section_when_absent(self):
        html = self.fn(**_MINIMAL_CARD_KWARGS)
        assert "Brier" not in html

    def test_deployment_section_when_deployed(self):
        html = self.fn(
            **_MINIMAL_CARD_KWARGS,
            is_deployed=True,
            deployment_endpoint="/api/predict/demo",
        )
        assert "/api/predict/demo" in html

    def test_html_escaping_in_project_name(self):
        html = self.fn(
            **{**_MINIMAL_CARD_KWARGS, "project_name": "<script>alert(1)</script>"}
        )
        assert "<script>" not in html


# ---------------------------------------------------------------------------
# Integration tests: GET /api/models/{run_id}/export-model-card
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    db_module.engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", echo=False)
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

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def trained_run(client):
    """Create a project with a trained model run and return run_id."""
    import io
    import time

    proj = client.post("/api/projects", json={"name": "CardTest"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        files={"file": ("data.csv", io.BytesIO(CLASSIFICATION_CSV), "text/csv")},
        data={"project_id": project_id},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "churned", "problem_type": "classification"},
    )

    train = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["logistic_regression"]},
    )
    assert train.status_code == 202
    run_id = train.json()["model_run_ids"][0]

    for _ in range(60):
        runs = client.get(f"/api/models/{project_id}/runs").json().get("runs", [])
        run = next((r for r in runs if r["id"] == run_id), None)
        if run and run["status"] in ("done", "failed"):
            break
        time.sleep(0.2)

    assert run and run["status"] == "done", "training did not complete"
    return run_id


class TestExportModelCardEndpoint:
    def test_returns_html_content_type(self, client, trained_run):
        r = client.get(f"/api/models/{trained_run}/export-model-card")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")

    def test_returns_attachment_header(self, client, trained_run):
        r = client.get(f"/api/models/{trained_run}/export-model-card")
        assert "attachment" in r.headers.get("content-disposition", "")

    def test_404_for_nonexistent_run(self, client):
        r = client.get("/api/models/nonexistent-run-id/export-model-card")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Unit tests: chat intent regex _MODEL_CARD_EXPORT_PATTERNS
# ---------------------------------------------------------------------------


class TestModelCardExportPatterns:
    def setup_method(self):
        from api.chat import _MODEL_CARD_EXPORT_PATTERNS

        self.pat = _MODEL_CARD_EXPORT_PATTERNS

    def _matches(self, text: str) -> bool:
        return bool(self.pat.search(text))

    def test_export_model_card(self):
        assert self._matches("export model card")

    def test_export_my_model_card(self):
        assert self._matches("export my model card")

    def test_download_model_card(self):
        assert self._matches("download model card")

    def test_generate_a_model_card(self):
        assert self._matches("generate a model card")

    def test_create_model_card(self):
        assert self._matches("create model card")

    def test_model_card_export(self):
        assert self._matches("model card export")

    def test_model_card_for_compliance(self):
        assert self._matches("model card for compliance")

    def test_model_card_document(self):
        assert self._matches("model card document")

    def test_share_model_documentation(self):
        assert self._matches("share model documentation")

    def test_export_model_documentation(self):
        assert self._matches("export model documentation")

    def test_model_documentation_download(self):
        assert self._matches("model documentation download")

    def test_no_false_positive_on_generic(self):
        assert not self._matches("what is the model accuracy?")
