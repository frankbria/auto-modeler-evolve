"""Tests for SDK Generation feature.

Covers:
- Pattern detection: _SDK_PATTERNS
- Helper functions: _sdk_class_name, _generate_python_sdk, _generate_javascript_sdk
- API endpoint: GET /api/deploy/{id}/sdk?language=python|javascript
- API validation: invalid language, inactive deployment
"""

import json
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine

import db as db_module
from api.chat import _SDK_PATTERNS
from api.deploy import _generate_javascript_sdk, _generate_python_sdk, _sdk_class_name


# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


def test_sdk_pattern_generate_python():
    assert _SDK_PATTERNS.search("generate a python sdk for my model")


def test_sdk_pattern_generate_sdk():
    assert _SDK_PATTERNS.search("generate sdk for my deployment")


def test_sdk_pattern_download_python():
    assert _SDK_PATTERNS.search("download the python sdk")


def test_sdk_pattern_download_js():
    assert _SDK_PATTERNS.search("download the javascript sdk")


def test_sdk_pattern_create_client():
    assert _SDK_PATTERNS.search("create a python client library for my model")


def test_sdk_pattern_developer_sdk():
    assert _SDK_PATTERNS.search("developer sdk")


def test_sdk_pattern_how_to_integrate():
    assert _SDK_PATTERNS.search("how do developers use my prediction api")


def test_sdk_pattern_no_match():
    assert not _SDK_PATTERNS.search("show me the top 10 predictions")


# ---------------------------------------------------------------------------
# _sdk_class_name helper
# ---------------------------------------------------------------------------


def test_sdk_class_name_basic():
    assert _sdk_class_name("revenue", "linear_regression") == "RevenuePredictor"


def test_sdk_class_name_underscores():
    assert _sdk_class_name("churn_risk", "random_forest") == "ChurnRiskPredictor"


def test_sdk_class_name_hyphen():
    # Hyphens should be treated like underscores
    assert _sdk_class_name("sales-forecast", "xgboost") == "SalesForecastPredictor"


# ---------------------------------------------------------------------------
# _generate_python_sdk helper
# ---------------------------------------------------------------------------

_FEATURE_SCHEMA_NUM = [
    {"name": "units", "type": "numeric"},
    {"name": "price", "type": "numeric"},
]
_FEATURE_SCHEMA_MIXED = [
    {"name": "region", "type": "categorical"},
    {"name": "revenue", "type": "numeric"},
]


def test_python_sdk_contains_class():
    sdk = _generate_python_sdk(
        deployment_id="dep-abc",
        target_column="revenue",
        algorithm="linear_regression",
        problem_type="regression",
        feature_schema=_FEATURE_SCHEMA_NUM,
        base_url="http://localhost:8000",
        export_date="2026-04-10",
    )
    assert "class RevenuePredictor" in sdk


def test_python_sdk_contains_predict():
    sdk = _generate_python_sdk(
        deployment_id="dep-abc",
        target_column="revenue",
        algorithm="linear_regression",
        problem_type="regression",
        feature_schema=_FEATURE_SCHEMA_NUM,
        base_url="http://localhost:8000",
        export_date="2026-04-10",
    )
    assert "def predict(" in sdk
    assert "def predict_batch(" in sdk


def test_python_sdk_contains_feature_params():
    sdk = _generate_python_sdk(
        deployment_id="dep-abc",
        target_column="revenue",
        algorithm="linear_regression",
        problem_type="regression",
        feature_schema=_FEATURE_SCHEMA_NUM,
        base_url="http://localhost:8000",
        export_date="2026-04-10",
    )
    assert "units" in sdk
    assert "price" in sdk


def test_python_sdk_endpoint_url():
    sdk = _generate_python_sdk(
        deployment_id="my-dep-id",
        target_column="target",
        algorithm="rf",
        problem_type="regression",
        feature_schema=[],
        base_url="https://api.example.com",
        export_date="2026-04-10",
    )
    assert "my-dep-id" in sdk


def test_python_sdk_regression_returns_doc():
    sdk = _generate_python_sdk(
        deployment_id="x",
        target_column="sales",
        algorithm="gbm",
        problem_type="regression",
        feature_schema=[],
        base_url="http://localhost:8000",
        export_date="2026-04-10",
    )
    assert "confidence_interval" in sdk


def test_python_sdk_classification_returns_doc():
    sdk = _generate_python_sdk(
        deployment_id="x",
        target_column="churn",
        algorithm="logistic",
        problem_type="classification",
        feature_schema=[],
        base_url="http://localhost:8000",
        export_date="2026-04-10",
    )
    assert "probabilities" in sdk


def test_python_sdk_mixed_feature_types():
    sdk = _generate_python_sdk(
        deployment_id="x",
        target_column="revenue",
        algorithm="rf",
        problem_type="regression",
        feature_schema=_FEATURE_SCHEMA_MIXED,
        base_url="http://localhost:8000",
        export_date="2026-04-10",
    )
    assert "region" in sdk
    assert "str" in sdk  # categorical → str type annotation
    assert "revenue" in sdk


# ---------------------------------------------------------------------------
# _generate_javascript_sdk helper
# ---------------------------------------------------------------------------


def test_js_sdk_contains_class():
    sdk = _generate_javascript_sdk(
        deployment_id="dep-abc",
        target_column="revenue",
        algorithm="linear_regression",
        problem_type="regression",
        feature_schema=_FEATURE_SCHEMA_NUM,
        base_url="http://localhost:8000",
        export_date="2026-04-10",
    )
    assert "class RevenuePredictor" in sdk


def test_js_sdk_contains_predict():
    sdk = _generate_javascript_sdk(
        deployment_id="dep-abc",
        target_column="revenue",
        algorithm="linear_regression",
        problem_type="regression",
        feature_schema=_FEATURE_SCHEMA_NUM,
        base_url="http://localhost:8000",
        export_date="2026-04-10",
    )
    assert "async predict(" in sdk
    assert "async predictBatch(" in sdk


def test_js_sdk_export_keyword():
    sdk = _generate_javascript_sdk(
        deployment_id="x",
        target_column="sales",
        algorithm="rf",
        problem_type="regression",
        feature_schema=[],
        base_url="http://localhost:8000",
        export_date="2026-04-10",
    )
    assert sdk.startswith("/**") or "export class" in sdk


def test_js_sdk_endpoint_url():
    sdk = _generate_javascript_sdk(
        deployment_id="my-dep-id",
        target_column="target",
        algorithm="rf",
        problem_type="regression",
        feature_schema=[],
        base_url="https://api.example.com",
        export_date="2026-04-10",
    )
    assert "my-dep-id" in sdk


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_with_deployment(tmp_path):
    """TestClient with a seeded active deployment (no ML pipeline on disk)."""
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)
    db_module.engine = engine

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    import api.models as models_api_module

    models_api_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as client:
        # Seed a Deployment row directly (no ML pipeline file on disk).
        from models.deployment import Deployment

        dep = Deployment(
            id=str(uuid.uuid4()),
            model_run_id=str(uuid.uuid4()),
            project_id=str(uuid.uuid4()),
            endpoint_path="/api/predict/test",
            dashboard_url="/predict/test",
            is_active=True,
            algorithm="linear_regression",
            problem_type="regression",
            feature_names=json.dumps(["units", "price"]),
            target_column="revenue",
            metrics='{"r2": 0.9}',
        )
        with Session(engine) as sess:
            sess.add(dep)
            sess.commit()
            sess.refresh(dep)
            deployment_id = dep.id

        yield client, deployment_id

    SQLModel.metadata.drop_all(engine)


def test_sdk_endpoint_python(client_with_deployment):
    client, deployment_id = client_with_deployment
    r = client.get(f"/api/deploy/{deployment_id}/sdk?language=python")
    assert r.status_code == 200
    assert (
        "text/x-python" in r.headers["content-type"]
        or "text" in r.headers["content-type"]
    )
    assert "class" in r.text
    assert "predict" in r.text


def test_sdk_endpoint_javascript(client_with_deployment):
    client, deployment_id = client_with_deployment
    r = client.get(f"/api/deploy/{deployment_id}/sdk?language=javascript")
    assert r.status_code == 200
    assert "predict" in r.text
    assert "class" in r.text


def test_sdk_endpoint_invalid_language(client_with_deployment):
    client, deployment_id = client_with_deployment
    r = client.get(f"/api/deploy/{deployment_id}/sdk?language=ruby")
    assert r.status_code == 400


def test_sdk_endpoint_not_found(client_with_deployment):
    client, _ = client_with_deployment
    r = client.get("/api/deploy/nonexistent-id/sdk?language=python")
    assert r.status_code == 404


def test_sdk_endpoint_default_language(client_with_deployment):
    """Default language is python when not specified."""
    client, deployment_id = client_with_deployment
    r = client.get(f"/api/deploy/{deployment_id}/sdk")
    assert r.status_code == 200
    assert "predict" in r.text
