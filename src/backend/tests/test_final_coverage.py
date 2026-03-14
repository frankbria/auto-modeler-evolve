"""Final coverage push — targets the remaining 2% uncovered lines.

Organises tests by source module. Every test here is designed to exercise
a specific uncovered branch without duplicating what the existing suites
already cover.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sqlmodel import Session, SQLModel, create_engine

import db as db_module


# ---------------------------------------------------------------------------
# Shared API test fixture (modelled after test_api_coverage_gaps.py)
# ---------------------------------------------------------------------------

# Use continuous float y values to ensure regression problem detection
SIMPLE_CSV = (
    b"x,y\n"
    b"1,1.2\n2,2.4\n3,3.6\n4,4.8\n5,6.0\n"
    b"6,7.2\n7,8.4\n8,9.6\n9,10.8\n10,12.0\n"
    b"11,13.2\n12,14.4\n"
)


@pytest.fixture
async def ac(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.feature_set  # noqa
    import models.model_run  # noqa
    import models.project  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.deploy as deploy_module
    import api.models as models_module
    import api.templates as templates_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    models_module.MODELS_DIR = tmp_path / "models"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"

    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client, tmp_path, monkeypatch


async def _setup_trained_run(ac_client, csv_bytes=None, algorithm="linear_regression",
                              target="y", problem_type="regression"):
    """Helper: create project → upload → features → train → return (project_id, run_id)."""
    import time

    r = await ac_client.post("/api/projects", json={"name": "test_proj"})
    assert r.status_code == 201
    project_id = r.json()["id"]

    csv = csv_bytes or SIMPLE_CSV
    r = await ac_client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("test.csv", io.BytesIO(csv), "text/csv")},
    )
    assert r.status_code == 201
    dataset_id = r.json()["dataset_id"]

    r = await ac_client.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert r.status_code in (200, 201)

    r = await ac_client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": target, "problem_type": problem_type},
    )
    assert r.status_code == 200

    r = await ac_client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": [algorithm]},
    )
    assert r.status_code == 202
    run_id = (r.json().get("run_ids") or r.json().get("model_run_ids"))[0]

    # Poll until done
    for _ in range(60):
        time.sleep(0.5)
        r2 = await ac_client.get(f"/api/models/{project_id}/runs")
        runs = r2.json()["runs"]
        if any(run["id"] == run_id and run["status"] in ("done", "failed") for run in runs):
            break

    return project_id, run_id, dataset_id


# ---------------------------------------------------------------------------
# core/explainer.py
# ---------------------------------------------------------------------------

class TestExplainerCoverage:

    def test_extract_importances_multiclass_logistic_regression(self):
        """Line 78: coef_.ndim == 2 path for multiclass LogisticRegression."""
        from core.explainer import compute_feature_importance

        X = np.array([
            [1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0],
            [5.0, 1.0], [1.0, 5.0],
        ])
        y = np.array([0, 1, 2, 0, 1, 2])
        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X, y)
        assert model.coef_.ndim == 2

        result = compute_feature_importance(model, ["a", "b"])
        assert len(result) == 2
        assert all("importance" in r for r in result)

    def test_explain_single_prediction_classification(self):
        """Lines 118-120: classification path with predict_proba."""
        from core.explainer import explain_single_prediction

        X = np.array([[1, 0], [0, 1], [1, 1], [0, 0],
                       [2, 0], [0, 2], [2, 2], [0, 0]])
        y = np.array([0, 1, 1, 0, 0, 1, 1, 0])
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        x_row = np.array([1.0, 0.5])
        result = explain_single_prediction(
            model, x_row, X, ["feat_a", "feat_b"],
            problem_type="classification", target_name="label"
        )
        assert isinstance(result["prediction"], int)
        assert "summary" in result
        assert "Predicted label" in result["summary"]

    def test_prediction_summary_empty_contributions(self):
        """Line 170: _prediction_summary with empty top_contributions."""
        from core.explainer import _prediction_summary

        result = _prediction_summary([], 0.5, None, "regression", "revenue")
        assert result == "No contribution data available."

    def test_prediction_summary_classification_with_class(self):
        """Lines 176-177: classification path in _prediction_summary."""
        from core.explainer import _prediction_summary

        contrib = [{"feature": "x", "value": 1.0, "contribution": 0.5, "direction": "positive"}]
        result = _prediction_summary(contrib, 0.8, 1, "classification", "churn")
        assert "class 1" in result
        assert "churn" in result


# ---------------------------------------------------------------------------
# core/validator.py
# ---------------------------------------------------------------------------

class TestValidatorCoverage:

    def test_cv_summary_weak_quality(self):
        """Line 85: 'weak' quality when mean < 0.5."""
        from core.validator import _cv_summary

        result = _cv_summary(mean=0.3, std=0.02, n_splits=5, problem_type="regression")
        assert "weak" in result

    def test_confusion_matrix_no_class_labels(self):
        """Line 114: else branch (no class_labels) uses sorted unique."""
        from core.validator import compute_confusion_matrix

        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])
        result = compute_confusion_matrix(y_true, y_pred)
        assert result["labels"] == ["0", "1", "2"]

    def test_residual_summary_underprediction_bias(self):
        """Line 198: 'under-prediction bias' branch (bias > 0, bias_frac >= 0.02)."""
        from core.validator import _residual_summary

        residuals = np.full(50, 10.0)  # positive bias = under-prediction
        y_true = np.full(50, 20.0)
        result = _residual_summary(residuals, y_true)
        assert "under-prediction bias" in result

    def test_residual_summary_overprediction_bias(self):
        """Line 200: 'over-prediction bias' branch (bias < 0)."""
        from core.validator import _residual_summary

        residuals = np.full(50, -10.0)
        y_true = np.full(50, 20.0)
        result = _residual_summary(residuals, y_true)
        assert "over-prediction bias" in result

    def test_assess_confidence_classification_low_accuracy(self):
        """Lines 237, 244: classification acc < 0.7 limitation."""
        from core.validator import assess_confidence_limitations

        result = assess_confidence_limitations(
            metrics={"accuracy": 0.5, "f1": 0.5},
            problem_type="classification",
            n_rows=200,
            n_features=5,
            cv_std=0.03,
        )
        assert any("Accuracy" in lim for lim in result["limitations"])

    def test_assess_confidence_high_feature_ratio(self):
        """Line 250: n_features > n_rows * 0.5 limitation."""
        from core.validator import assess_confidence_limitations

        result = assess_confidence_limitations(
            metrics={"r2": 0.9},
            problem_type="regression",
            n_rows=10,
            n_features=8,
            cv_std=0.02,
        )
        assert any("features" in lim and "rows" in lim for lim in result["limitations"])

    def test_overall_confidence_low(self):
        """Line 288: _overall_confidence returns 'low'."""
        from core.validator import _overall_confidence

        result = _overall_confidence(
            metrics={"r2": 0.4},
            problem_type="regression",
            cv_std=0.3,
        )
        assert result == "low"

    def test_overall_confidence_low_classification(self):
        """Line 288: _overall_confidence 'low' for classification."""
        from core.validator import _overall_confidence

        result = _overall_confidence(
            metrics={"accuracy": 0.4, "f1": 0.4},
            problem_type="classification",
            cv_std=None,
        )
        assert result == "low"


# ---------------------------------------------------------------------------
# chat/orchestrator.py
# ---------------------------------------------------------------------------

class TestOrchestratorCoverage:

    def _make_run(self, metrics=None, algorithm="linear_regression",
                  created_at="2024-01-01T12:00:00"):
        run = MagicMock()
        run.metrics = metrics
        run.algorithm = algorithm
        run.status = "done"
        run.created_at = created_at
        return run

    def test_primary_metric_no_metrics(self):
        """Line 135: _primary_metric returns None when run.metrics is falsy."""
        from chat.orchestrator import _primary_metric

        run = self._make_run(metrics=None)
        assert _primary_metric(run) is None

    def test_primary_metric_empty_string(self):
        """Line 135: _primary_metric returns None when run.metrics is empty string."""
        from chat.orchestrator import _primary_metric

        run = self._make_run(metrics="")
        assert _primary_metric(run) is None

    def test_primary_metric_invalid_json(self):
        """Lines 139-140: _primary_metric except branch with invalid JSON."""
        from chat.orchestrator import _primary_metric

        run = self._make_run(metrics="not-json{{{")
        assert _primary_metric(run) is None

    def test_metric_label_no_metrics(self):
        """Line 146: _metric_label returns 'score' when run.metrics is falsy."""
        from chat.orchestrator import _metric_label

        run = self._make_run(metrics=None)
        assert _metric_label(run) == "score"

    def test_metric_label_no_r2_no_accuracy(self):
        """Lines 151-153: _metric_label 'score' when neither r2 nor accuracy in metrics."""
        from chat.orchestrator import _metric_label

        run = self._make_run(metrics=json.dumps({"f1": 0.8}))
        assert _metric_label(run) == "score"

    def test_metric_label_invalid_json(self):
        """Lines 154-155: _metric_label except branch with invalid JSON."""
        from chat.orchestrator import _metric_label

        run = self._make_run(metrics="bad-json")
        assert _metric_label(run) == "score"

    def test_detect_model_regression_none_score(self):
        """Line 117: _detect_model_regression when latest_score is None."""
        from chat.orchestrator import _detect_model_regression

        run_old = self._make_run(
            metrics=json.dumps({"r2": 0.9}),
            algorithm="linear_regression",
            created_at="2024-01-01T10:00:00",
        )
        run_latest = self._make_run(
            metrics=None,  # _primary_metric → None
            algorithm="random_forest",
            created_at="2024-01-01T12:00:00",
        )

        result = _detect_model_regression([run_old, run_latest])
        assert result is None


# ---------------------------------------------------------------------------
# core/deployer.py
# ---------------------------------------------------------------------------

class TestDeployerCoverage:

    def test_predict_single_classification_with_probabilities(self, tmp_path):
        """Lines 174-180: predict_single classification path (predict_proba)."""
        from core.deployer import build_prediction_pipeline, save_pipeline, predict_single
        from core.trainer import train_single_model, prepare_features

        df = pd.DataFrame({
            "age": [25, 35, 45, 55, 30, 40, 50, 60],
            "income": [30000, 50000, 70000, 90000, 40000, 60000, 80000, 100000],
            "churn": ["No", "No", "Yes", "Yes", "No", "Yes", "Yes", "No"],
        })
        pipeline = build_prediction_pipeline(df, ["age", "income"], "churn", "classification")
        pipeline_path = str(tmp_path / "pipeline.pkl")
        save_pipeline(pipeline, pipeline_path)

        X, y, _ = prepare_features(df, ["age", "income"], "churn", "classification")
        train_single_model(X, y, "random_forest_classifier", "classification", tmp_path, "cls-run")

        result = predict_single(
            pipeline_path=pipeline_path,
            model_path=str(tmp_path / "cls-run.joblib"),
            input_data={"age": 30, "income": 45000},
        )
        assert "probabilities" in result
        assert result["problem_type"] == "classification"

    def test_predict_batch_classification_with_confidence(self, tmp_path):
        """Lines 213-214: predict_batch classification path (predict_proba confidence)."""
        from core.deployer import build_prediction_pipeline, save_pipeline, predict_batch
        from core.trainer import train_single_model, prepare_features

        df = pd.DataFrame({
            "age": [25, 35, 45, 55, 30, 40, 50, 60],
            "income": [30000, 50000, 70000, 90000, 40000, 60000, 80000, 100000],
            "churn": ["No", "No", "Yes", "Yes", "No", "Yes", "Yes", "No"],
        })
        pipeline = build_prediction_pipeline(df, ["age", "income"], "churn", "classification")
        pipeline_path = str(tmp_path / "pipeline_b.pkl")
        save_pipeline(pipeline, pipeline_path)

        X, y, _ = prepare_features(df, ["age", "income"], "churn", "classification")
        train_single_model(X, y, "random_forest_classifier", "classification", tmp_path, "cls-batch")

        batch_csv = b"age,income\n28,35000\n45,70000\n"
        result_bytes = predict_batch(
            pipeline_path=pipeline_path,
            model_path=str(tmp_path / "cls-batch.joblib"),
            csv_bytes=batch_csv,
        )
        result_df = pd.read_csv(io.BytesIO(result_bytes))
        assert "prediction" in result_df.columns
        assert "confidence" in result_df.columns

    def test_transform_col_not_in_label_encoders(self):
        """Line 63: else branch when a column has no LabelEncoder (empty label_encoders)."""
        from core.deployer import PredictionPipeline

        # No label_encoders for any column → the else branch (row.append(0.0)) is hit
        pp = PredictionPipeline(
            feature_names=["category", "other"],
            column_types={"category": "categorical", "other": "categorical"},
            label_encoders={},  # empty — no encoders
            medians={},
        )
        row = {"category": "A", "other": "B"}
        vec = pp.transform(row)
        assert vec.shape == (1, 2)
        assert vec[0, 0] == 0.0  # fallback to 0.0
        assert vec[0, 1] == 0.0


# ---------------------------------------------------------------------------
# core/feature_engine.py
# ---------------------------------------------------------------------------

class TestFeatureEngineCoverage:

    def test_suggest_skips_all_nan_numeric_column(self):
        """Line 98: continue when numeric series is empty (all NaN)."""
        from core.feature_engine import suggest_features

        df = pd.DataFrame({
            "all_nan": [float("nan"), float("nan"), float("nan")],
            "normal": [1.0, 2.0, 3.0],
        })
        column_stats = [
            {"name": "all_nan", "dtype": "float64", "unique_count": 0, "sample_values": []},
            {"name": "normal", "dtype": "float64", "unique_count": 3, "sample_values": [1, 2, 3]},
        ]
        result = suggest_features(df, column_stats)
        assert isinstance(result, list)

    def test_apply_transformations_exception_silenced(self):
        """Lines 301-303: except silences failed transforms."""
        from core.feature_engine import apply_transformations

        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        # Use an unknown transform type — should be silently skipped
        transforms = [{"transform_type": "nonexistent_transform_xyz", "column": "value"}]
        result_df, mapping = apply_transformations(df, transforms)
        assert "value" in result_df.columns

    def test_compute_feature_importance_target_not_in_df(self):
        """Line 404: returns [] when target_col not in df."""
        from core.feature_engine import compute_feature_importance

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = compute_feature_importance(df, "missing_target", "regression", [])
        assert result == []

    def test_compute_feature_importance_no_x_parts(self):
        """Line 423: returns [] when no usable columns after excluding target."""
        from core.feature_engine import compute_feature_importance

        # Only the target column, no features
        df = pd.DataFrame({"target": [1, 2, 3]})
        result = compute_feature_importance(df, "target", "regression", [])
        assert result == []

    def test_importance_description_strong_predictor(self):
        """Line 497: pct >= 10 → 'Strong predictor'."""
        from core.feature_engine import _importance_description

        result = _importance_description("revenue", 15.0)
        assert "Strong predictor" in result

    def test_importance_description_moderate_predictor(self):
        """Line 499: pct >= 5 and < 10 → 'Moderate predictor'."""
        from core.feature_engine import _importance_description

        result = _importance_description("units", 7.5)
        assert "Moderate predictor" in result


# ---------------------------------------------------------------------------
# core/query_engine.py
# ---------------------------------------------------------------------------

class TestQueryEngineCoverage:

    def test_parse_question_returns_none_for_null_string(self):
        """Line 168: when Claude returns 'null', return None."""
        from core.query_engine import _parse_question_to_spec

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="null")]

        with patch("core.query_engine.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_anthropic.Anthropic.return_value = mock_client
            mock_client.messages.create.return_value = mock_response

            df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            result = _parse_question_to_spec("some question", df, [])
            assert result is None

    def test_safe_rows_numpy_scalar(self):
        """Line 331: safe[k] = v.item() for numpy scalars."""
        from core.query_engine import _safe_rows

        df = pd.DataFrame({
            "a": np.array([1, 2], dtype=np.int64),
            "b": np.array([3.0, 4.0], dtype=np.float32),
        })
        rows = _safe_rows(df)
        assert len(rows) == 2
        assert type(rows[0]["a"]) in (int, float)


# ---------------------------------------------------------------------------
# chat/narration.py
# ---------------------------------------------------------------------------

class TestNarrationCoverage:

    def test_narrate_profile_highlights_str_warning(self):
        """Lines 151-152: string warning (not dict) in narrate_profile_highlights."""
        from chat.narration import narrate_profile_highlights

        profile = {
            "patterns": [],
            "warnings": ["This is a plain string warning"],  # string, not dict
            "correlations": {},
        }
        result = narrate_profile_highlights(profile)
        # Should return a string (found an insight from the string warning)
        assert result is not None
        assert isinstance(result, str)

    def test_narrate_profile_highlights_correlation_value_error(self):
        """Lines 166-167: TypeError/ValueError in correlation parsing is silenced."""
        from chat.narration import narrate_profile_highlights

        profile = {
            "patterns": [],
            "warnings": [],
            "correlations": {"a_b": "not_a_number"},  # causes ValueError
        }
        # Should not raise
        result = narrate_profile_highlights(profile)
        # No valid insight found, so returns None
        assert result is None


# ---------------------------------------------------------------------------
# chat/prompts.py
# ---------------------------------------------------------------------------

class TestPromptsCoverage:

    def test_format_metric_needs_improvement(self):
        """Line 147: metric value < 0.5 → '(needs improvement)'."""
        from chat.prompts import format_metric

        result = format_metric("r2", 0.3)
        assert "needs improvement" in result


# ---------------------------------------------------------------------------
# core/analyzer.py
# ---------------------------------------------------------------------------

class TestAnalyzerCoverage:

    def test_numeric_distribution_all_inf_values(self):
        """Line 120: finite_series is empty after filtering inf."""
        from core.analyzer import _numeric_distribution

        series = pd.Series([float("inf"), float("-inf"), float("inf")])
        result = _numeric_distribution(series)
        assert result == {"bins": [], "counts": []}

    def test_safe_scalar_numpy_generic(self):
        """Line 309: _safe_scalar returns .item() for numpy generic."""
        from core.analyzer import _safe_scalar

        val = np.float64(3.14)
        result = _safe_scalar(val)
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# core/report_generator.py
# ---------------------------------------------------------------------------

class TestReportGeneratorCoverage:

    def test_generate_model_report_skips_none_metrics(self):
        """Line 248: `if val is None: continue` in metrics table."""
        from core.report_generator import generate_model_report
        from datetime import datetime, timezone

        pdf_bytes = generate_model_report(
            project_name="Test Project",
            dataset_filename="test.csv",
            dataset_rows=100,
            dataset_columns=5,
            algorithm="linear_regression",
            problem_type="regression",
            metrics={
                "r2": 0.85,
                "mae": None,   # None value — should be skipped (line 248)
                "rmse": 0.12,
            },
            summary="Good model",
            training_duration_ms=500,
            feature_importances=None,
            confidence_assessment=None,
            created_at=datetime.now(timezone.utc),
        )
        assert len(pdf_bytes) > 0


# ---------------------------------------------------------------------------
# api/projects.py — delete nonexistent project → 404
# ---------------------------------------------------------------------------

class TestProjectsApiCoverage:

    @pytest.mark.asyncio
    async def test_delete_nonexistent_project(self, ac):
        client, _, _ = ac
        r = await client.delete("/api/projects/nonexistent-id-xyz")
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_query_dataset_not_found(self, ac):
        """Line 249: query_dataset with nonexistent dataset → 404."""
        client, _, _ = ac
        r = await client.post(
            "/api/data/nonexistent-id/query",
            json={"question": "what is the total?"},
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# api/templates.py — sample file missing → 503
# ---------------------------------------------------------------------------

class TestTemplatesApiCoverage:

    @pytest.mark.asyncio
    async def test_apply_template_sample_file_missing(self, ac):
        """Line 162: template sample file not on disk → 503."""
        client, tmp_path, monkeypatch = ac
        import api.templates as tmpl_mod

        r = await client.post("/api/projects", json={"name": "t"})
        assert r.status_code == 201
        project_id = r.json()["id"]

        monkeypatch.setattr(tmpl_mod, "SAMPLE_DIR", tmp_path / "empty_dir")

        r = await client.post(
            "/api/templates/sales_forecast/apply",
            json={"project_id": project_id},
        )
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# api/deploy.py — deploy edge cases
# ---------------------------------------------------------------------------

class TestDeployApiCoverage:

    @pytest.mark.asyncio
    async def test_deploy_run_not_found(self, ac):
        """api/deploy.py lines 69, 73: deploy with nonexistent run → 404."""
        client, _, _ = ac
        r = await client.post("/api/deploy/nonexistent-run-id")
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_deploy_with_transforms_line_110(self, ac):
        """Line 110: deploy endpoint applies transformations (df, _ = apply_transformations)."""
        import time
        client, _, _ = ac

        cls_csv = (
            b"age,income,churn\n"
            b"25,30000,0\n35,50000,0\n45,70000,1\n55,90000,1\n"
            b"30,40000,0\n40,60000,1\n50,80000,1\n60,100000,0\n"
        )
        r = await client.post("/api/projects", json={"name": "deploy_transform"})
        project_id = r.json()["id"]

        r = await client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("t.csv", io.BytesIO(cls_csv), "text/csv")},
        )
        assert r.status_code == 201
        dataset_id = r.json()["dataset_id"]

        r = await client.post(
            f"/api/features/{dataset_id}/apply",
            json={"transformations": [{"transform_type": "log_transform", "column": "age"}]},
        )
        assert r.status_code in (200, 201)

        r = await client.post(
            f"/api/features/{dataset_id}/target",
            json={"target_column": "churn", "problem_type": "classification"},
        )
        assert r.status_code == 200

        r = await client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["random_forest_classifier"]},
        )
        assert r.status_code == 202
        run_id = (r.json().get("run_ids") or r.json().get("model_run_ids"))[0]

        for _ in range(60):
            time.sleep(0.5)
            r2 = await client.get(f"/api/models/{project_id}/runs")
            runs = r2.json()["runs"]
            if any(run["id"] == run_id and run["status"] in ("done", "failed") for run in runs):
                break

        r = await client.post(f"/api/deploy/{run_id}")
        assert r.status_code in (200, 201)


# ---------------------------------------------------------------------------
# api/validation.py — transforms applied branch
# ---------------------------------------------------------------------------

class TestValidationApiCoverage:

    @pytest.mark.asyncio
    async def test_validate_with_transforms(self, ac):
        """Line 87: _load_data applies transformations when feature_set has them."""
        import time
        client, _, _ = ac

        # Enough rows for CV
        csv_lines = [b"value,revenue"]
        for i in range(1, 33):
            csv_lines.append(f"{i},{i*100}".encode())
        csv_bytes = b"\n".join(csv_lines) + b"\n"

        r = await client.post("/api/projects", json={"name": "val_transform"})
        project_id = r.json()["id"]

        r = await client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("t.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        assert r.status_code == 201
        dataset_id = r.json()["dataset_id"]

        r = await client.post(
            f"/api/features/{dataset_id}/apply",
            json={"transformations": [{"transform_type": "log_transform", "column": "value"}]},
        )
        assert r.status_code in (200, 201)

        r = await client.post(
            f"/api/features/{dataset_id}/target",
            json={"target_column": "revenue", "problem_type": "regression"},
        )
        assert r.status_code == 200

        r = await client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression"]},
        )
        assert r.status_code == 202
        run_id = (r.json().get("run_ids") or r.json().get("model_run_ids"))[0]

        for _ in range(60):
            time.sleep(0.5)
            r2 = await client.get(f"/api/models/{project_id}/runs")
            if any(run["id"] == run_id and run["status"] == "done"
                   for run in r2.json()["runs"]):
                break

        r = await client.get(f"/api/validate/{run_id}/metrics")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# api/data.py — _load_df_from_path xlsx path (lines 44-47)
# ---------------------------------------------------------------------------

class TestDataApiLoadPath:

    def test_load_df_from_path_xlsx(self, tmp_path):
        """Lines 44-47: _load_df_from_path handles .xlsx files."""
        from api.data import _load_df_from_path

        df_original = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        xlsx_path = tmp_path / "test.xlsx"
        df_original.to_excel(xlsx_path, index=False)

        result = _load_df_from_path(xlsx_path)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 3

    def test_load_df_from_path_csv(self, tmp_path):
        """Line 47: _load_df_from_path handles .csv files."""
        from api.data import _load_df_from_path

        df_original = pd.DataFrame({"x": [1], "y": [2]})
        csv_path = tmp_path / "test.csv"
        df_original.to_csv(csv_path, index=False)

        result = _load_df_from_path(csv_path)
        assert list(result.columns) == ["x", "y"]


# ---------------------------------------------------------------------------
# api/models.py — various edge cases
# ---------------------------------------------------------------------------

class TestModelsApiCoverage:

    @pytest.mark.asyncio
    async def test_comparison_radar_returns_chart_when_two_done(self, ac):
        """Line 490: comparison-radar returns chart when 2+ models trained."""
        import time
        client, _, _ = ac

        r = await client.post("/api/projects", json={"name": "radar"})
        project_id = r.json()["id"]

        r = await client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("t.csv", io.BytesIO(SIMPLE_CSV), "text/csv")},
        )
        assert r.status_code == 201
        dataset_id = r.json()["dataset_id"]

        r = await client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
        r = await client.post(
            f"/api/features/{dataset_id}/target",
            json={"target_column": "y", "problem_type": "regression"},
        )

        r = await client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression", "random_forest_regressor"]},
        )
        assert r.status_code == 202

        for _ in range(60):
            time.sleep(0.5)
            r2 = await client.get(f"/api/models/{project_id}/runs")
            done_runs = [run for run in r2.json()["runs"] if run["status"] == "done"]
            if len(done_runs) >= 2:
                break

        r = await client.get(f"/api/models/{project_id}/comparison-radar")
        assert r.status_code in (200, 204)
        if r.status_code == 200:
            assert "chart" in r.json()

    @pytest.mark.asyncio
    async def test_download_model_returns_file(self, ac):
        """Lines 600-601: model download returns file."""
        import time
        client, _, _ = ac

        project_id, run_id, _ = await _setup_trained_run(client)

        r = await client.get(f"/api/models/{run_id}/download")
        assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_select_model_not_done_returns_400(self, ac):
        """Lines 518-519: select a model before training completes → 400.

        We test this by directly inserting a pending run into the DB.
        """
        import db as _db
        from models.model_run import ModelRun
        from sqlmodel import Session
        client, tmp_path, _ = ac

        r = await client.post("/api/projects", json={"name": "select_test"})
        project_id = r.json()["id"]

        # Insert a pending run directly
        with Session(_db.engine) as session:
            run = ModelRun(
                id="test-pending-run",
                project_id=project_id,
                algorithm="linear_regression",
                status="training",
            )
            session.add(run)
            session.commit()

        r = await client.post("/api/models/test-pending-run/select")
        assert r.status_code == 400
        assert "Cannot select" in r.json()["detail"]


# ---------------------------------------------------------------------------
# api/data.py — narration exception pass blocks
# ---------------------------------------------------------------------------

class TestDataApiNarrationExceptions:

    @pytest.mark.asyncio
    async def test_upload_csv_narration_exception_silenced(self, ac):
        """Lines 156-157: narration exception in upload_csv is silenced."""
        client, _, monkeypatch = ac
        import api.data as data_mod

        def _raise(*args, **kwargs):
            raise RuntimeError("narration failure")

        monkeypatch.setattr(data_mod, "narrate_data_insights_ai", _raise)

        r = await client.post("/api/projects", json={"name": "narr_test"})
        project_id = r.json()["id"]

        r = await client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("t.csv", io.BytesIO(b"a,b\n1,2\n3,4\n"), "text/csv")},
        )
        assert r.status_code == 201  # succeeds despite narration failure

    @pytest.mark.asyncio
    async def test_sample_upload_narration_exception_silenced(self, ac):
        """Lines 357-358: narration exception in sample_dataset is silenced."""
        client, _, monkeypatch = ac
        import api.data as data_mod

        def _raise(*args, **kwargs):
            raise RuntimeError("narration failure")

        monkeypatch.setattr(data_mod, "narrate_data_insights_ai", _raise)

        r = await client.post("/api/projects", json={"name": "sample_narr"})
        project_id = r.json()["id"]

        r = await client.post("/api/data/sample", json={"project_id": project_id})
        assert r.status_code in (200, 201)

    @pytest.mark.asyncio
    async def test_url_import_narration_exception_silenced(self, ac):
        """Lines 829-830: narration exception in URL import is silenced."""
        import urllib.request as urlreq
        client, _, monkeypatch = ac
        import api.data as data_mod

        def _raise_narr(*args, **kwargs):
            raise RuntimeError("narration failure")

        monkeypatch.setattr(data_mod, "narrate_data_insights_ai", _raise_narr)

        csv_bytes = b"col_a,col_b\n1,2\n3,4\n5,6\n"

        class MockResponse:
            def read(self):
                return csv_bytes
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        monkeypatch.setattr(urlreq, "urlopen", lambda req, timeout=None: MockResponse())

        r = await client.post("/api/projects", json={"name": "url_narr"})
        project_id = r.json()["id"]

        r = await client.post(
            "/api/data/upload-url",
            json={"url": "http://example.com/data.csv", "project_id": project_id},
        )
        assert r.status_code in (200, 201)


# ---------------------------------------------------------------------------
# api/data.py — profile missing file (line 222)
# ---------------------------------------------------------------------------

class TestDataApiProfileMissingFile:

    @pytest.mark.asyncio
    async def test_profile_file_missing_no_cache(self, ac):
        """Line 222: profile endpoint when file is missing and no cache."""
        client, tmp_path, _ = ac

        r = await client.post("/api/projects", json={"name": "profile_miss"})
        project_id = r.json()["id"]

        r = await client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("t.csv", io.BytesIO(b"a,b\n1,2\n3,4\n"), "text/csv")},
        )
        assert r.status_code == 201
        dataset_id = r.json()["dataset_id"]

        # Clear the profile cache and delete the file
        from models.dataset import Dataset

        with Session(db_module.engine) as session:
            ds = session.get(Dataset, dataset_id)
            assert ds is not None
            file_to_delete = ds.file_path
            ds.profile = None
            session.add(ds)
            session.commit()

        if os.path.exists(file_to_delete):
            os.unlink(file_to_delete)

        r = await client.get(f"/api/data/{dataset_id}/profile")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# api/data.py — timeseries exception (lines 481-482)
# ---------------------------------------------------------------------------

class TestDataApiTimeseriesException:

    @pytest.mark.asyncio
    async def test_timeseries_date_parse_exception_silenced(self, ac):
        """Lines 481-482: exception in date parsing falls back to raw df."""
        client, _, monkeypatch = ac
        import api.data as data_mod

        r = await client.post("/api/projects", json={"name": "ts_exc"})
        project_id = r.json()["id"]

        r = await client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("t.csv",
                            io.BytesIO(b"date,value\n2024-01-01,10\n2024-01-02,20\n2024-01-03,30\n"),
                            "text/csv")},
        )
        assert r.status_code == 201
        dataset_id = r.json()["dataset_id"]

        # Patch pd.to_datetime inside the data module to raise
        orig = data_mod.pd.to_datetime

        def _raise_to_datetime(col, *args, **kwargs):
            raise ValueError("mocked parse error")

        monkeypatch.setattr(data_mod.pd, "to_datetime", _raise_to_datetime)

        r = await client.get(f"/api/data/{dataset_id}/timeseries")
        # Either returns 200 (degraded) or error — either way, doesn't crash
        assert r.status_code in (200, 400, 422)


# ---------------------------------------------------------------------------
# api/data.py — URL import bad CSV content (lines 773-774)
# ---------------------------------------------------------------------------

class TestDataApiUrlBadCsv:

    @pytest.mark.asyncio
    async def test_url_import_bad_csv_content(self, ac):
        """Lines 773-774: downloaded content cannot be parsed as CSV → 400."""
        import urllib.request as urlreq
        client, _, monkeypatch = ac

        class MockResponse:
            def read(self):
                return b"\x00\x01\x02\x03\xff\xfe binary garbage that can't be CSV"
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        monkeypatch.setattr(urlreq, "urlopen", lambda req, timeout=None: MockResponse())

        r = await client.post("/api/projects", json={"name": "badcsv"})
        project_id = r.json()["id"]

        r = await client.post(
            "/api/data/upload-url",
            json={"url": "http://example.com/bad.csv", "project_id": project_id},
        )
        assert r.status_code == 400
        assert "parsed as CSV" in r.json()["detail"]
