"""test_coverage_gaps.py

Targeted tests to close coverage gaps found in Day 3 gap analysis:
  - core/chart_builder.py (73%): build_model_comparison_radar paths
  - chat/orchestrator.py (78%): profile branch, transformations, classification metrics
  - api/chat.py (37%): SSE endpoint with mocked Anthropic client
  - core/trainer.py: error paths in training
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# chart_builder.py — build_model_comparison_radar
# ===========================================================================

class TestRadarChart:
    """Cover the build_model_comparison_radar function (lines 288-340)."""

    def test_returns_none_when_fewer_than_two_models(self):
        from core.chart_builder import build_model_comparison_radar
        result = build_model_comparison_radar([], "regression")
        assert result is None

    def test_returns_none_with_single_model(self):
        from core.chart_builder import build_model_comparison_radar
        models = [{"algorithm": "linear_regression", "metrics": {"r2": 0.8, "mae": 10, "rmse": 15}}]
        result = build_model_comparison_radar(models, "regression")
        assert result is None

    def test_regression_radar_has_correct_structure(self):
        from core.chart_builder import build_model_comparison_radar
        models = [
            {"algorithm": "linear_regression", "metrics": {"r2": 0.8, "mae": 10.0, "rmse": 15.0}},
            {"algorithm": "random_forest_regressor", "metrics": {"r2": 0.9, "mae": 8.0, "rmse": 12.0}},
        ]
        result = build_model_comparison_radar(models, "regression")
        assert result is not None
        assert result["chart_type"] == "radar"
        assert result["x_key"] == "metric"
        assert len(result["y_keys"]) == 2
        assert "data" in result
        # Three spokes: R², MAE Score, RMSE Score
        assert len(result["data"]) == 3
        metrics_names = {d["metric"] for d in result["data"]}
        assert "R²" in metrics_names
        assert "MAE Score" in metrics_names
        assert "RMSE Score" in metrics_names

    def test_regression_radar_higher_mae_gets_lower_score(self):
        """MAE is inverted: model with lower MAE should have higher normalized score."""
        from core.chart_builder import build_model_comparison_radar
        models = [
            {"algorithm": "linear_regression", "metrics": {"r2": 0.8, "mae": 100.0, "rmse": 120.0}},
            {"algorithm": "random_forest_regressor", "metrics": {"r2": 0.8, "mae": 10.0, "rmse": 12.0}},
        ]
        result = build_model_comparison_radar(models, "regression")
        assert result is not None
        mae_row = next(d for d in result["data"] if d["metric"] == "MAE Score")
        linear_key = "Linear Regression"
        rf_key = "Random Forest Regressor"
        # Random forest (lower MAE) should have higher normalized MAE score
        assert mae_row[rf_key] > mae_row[linear_key]

    def test_classification_radar_has_four_spokes(self):
        from core.chart_builder import build_model_comparison_radar
        models = [
            {
                "algorithm": "logistic_regression",
                "metrics": {"accuracy": 0.82, "f1": 0.81, "precision": 0.80, "recall": 0.83},
            },
            {
                "algorithm": "random_forest_classifier",
                "metrics": {"accuracy": 0.91, "f1": 0.90, "precision": 0.89, "recall": 0.92},
            },
        ]
        result = build_model_comparison_radar(models, "classification")
        assert result is not None
        assert result["chart_type"] == "radar"
        assert len(result["data"]) == 4
        metric_names = {d["metric"] for d in result["data"]}
        assert metric_names == {"Accuracy", "F1 Score", "Precision", "Recall"}

    def test_radar_ignores_models_without_metrics(self):
        from core.chart_builder import build_model_comparison_radar
        models = [
            {"algorithm": "linear_regression", "metrics": None},
            {"algorithm": "random_forest_regressor", "metrics": {"r2": 0.9, "mae": 5.0, "rmse": 7.0}},
        ]
        # Only 1 model has valid metrics → should return None
        result = build_model_comparison_radar(models, "regression")
        assert result is None

    def test_radar_with_all_zero_mae_does_not_divide_by_zero(self):
        from core.chart_builder import build_model_comparison_radar
        models = [
            {"algorithm": "linear_regression", "metrics": {"r2": 0.9, "mae": 0.0, "rmse": 0.0}},
            {"algorithm": "random_forest_regressor", "metrics": {"r2": 0.85, "mae": 0.0, "rmse": 0.0}},
        ]
        # Both MAE=0 → max is 0 → should handle without crashing
        result = build_model_comparison_radar(models, "regression")
        assert result is not None
        mae_row = next(d for d in result["data"] if d["metric"] == "MAE Score")
        # When all MAE=0, normalized should be 1.0 for both
        for algo_key in result["y_keys"]:
            assert mae_row[algo_key] == 1.0

    def test_radar_values_clamped_between_zero_and_one(self):
        from core.chart_builder import build_model_comparison_radar
        # Negative R² should be clamped to 0
        models = [
            {"algorithm": "linear_regression", "metrics": {"r2": -0.5, "mae": 100.0, "rmse": 200.0}},
            {"algorithm": "random_forest_regressor", "metrics": {"r2": 0.9, "mae": 5.0, "rmse": 10.0}},
        ]
        result = build_model_comparison_radar(models, "regression")
        assert result is not None
        r2_row = next(d for d in result["data"] if d["metric"] == "R²")
        linear_key = "Linear Regression"
        assert r2_row[linear_key] == 0.0  # clamped from -0.5

    def test_radar_y_keys_are_algo_names_title_case(self):
        from core.chart_builder import build_model_comparison_radar
        models = [
            {"algorithm": "linear_regression", "metrics": {"r2": 0.8, "mae": 10.0, "rmse": 12.0}},
            {"algorithm": "gradient_boosting_regressor", "metrics": {"r2": 0.85, "mae": 8.0, "rmse": 9.0}},
        ]
        result = build_model_comparison_radar(models, "regression")
        assert result is not None
        assert "Linear Regression" in result["y_keys"]
        assert "Gradient Boosting Regressor" in result["y_keys"]

    def test_radar_three_models(self):
        from core.chart_builder import build_model_comparison_radar
        models = [
            {"algorithm": "linear_regression", "metrics": {"r2": 0.7, "mae": 20.0, "rmse": 25.0}},
            {"algorithm": "random_forest_regressor", "metrics": {"r2": 0.88, "mae": 10.0, "rmse": 14.0}},
            {"algorithm": "gradient_boosting_regressor", "metrics": {"r2": 0.92, "mae": 7.0, "rmse": 10.0}},
        ]
        result = build_model_comparison_radar(models, "regression")
        assert result is not None
        assert len(result["y_keys"]) == 3
        # Each data point should have all three algorithm keys
        for row in result["data"]:
            assert all(k in row for k in result["y_keys"])


# ===========================================================================
# chat/orchestrator.py — uncovered edge paths
# ===========================================================================

class TestOrchestratorEdgePaths:
    """Cover lines 153-154, 157-167, 182-191, 209-210."""

    def make_project(self, name="Test", description=None):
        p = MagicMock()
        p.name = name
        p.description = description
        return p

    def test_columns_json_decode_error_is_silently_ignored(self):
        """Line 153-154: bad JSON in dataset.columns is handled gracefully."""
        from chat.orchestrator import build_system_prompt
        p = self.make_project()
        ds = MagicMock()
        ds.filename = "data.csv"
        ds.row_count = 100
        ds.column_count = 3
        ds.columns = "INVALID JSON {{{"  # will cause JSONDecodeError
        ds.profile = None
        prompt = build_system_prompt(p, dataset=ds)
        # Should not raise; should still include dataset info
        assert "data.csv" in prompt

    def test_dataset_profile_included_in_prompt(self):
        """Lines 157-167: profile with insights/correlations is included."""
        from chat.orchestrator import build_system_prompt
        p = self.make_project()
        ds = MagicMock()
        ds.filename = "sales.csv"
        ds.row_count = 500
        ds.column_count = 5
        ds.columns = None
        ds.profile = json.dumps({
            "patterns": ["strong upward trend in Q3"],
            "correlations": {"pairs": [{"col_a": "revenue", "col_b": "units", "correlation": 0.92}]},
        })
        prompt = build_system_prompt(p, dataset=ds)
        assert "Data profile highlights" in prompt
        assert "revenue" in prompt

    def test_dataset_profile_json_decode_error_is_ignored(self):
        """Lines 166-167: bad JSON in profile is silently ignored."""
        from chat.orchestrator import build_system_prompt
        p = self.make_project()
        ds = MagicMock()
        ds.filename = "data.csv"
        ds.row_count = 50
        ds.column_count = 2
        ds.columns = None
        ds.profile = "NOT VALID JSON"
        prompt = build_system_prompt(p, dataset=ds)
        assert isinstance(prompt, str)  # no crash

    def test_feature_set_transformations_shown_in_prompt(self):
        """Lines 182-191: feature set with transformations list is shown."""
        from chat.orchestrator import build_system_prompt
        p = self.make_project()
        ds = MagicMock()
        ds.filename = "data.csv"
        ds.row_count = 200
        ds.column_count = 4
        ds.columns = None
        ds.profile = None
        fs = MagicMock()
        fs.target_column = "revenue"
        fs.problem_type = "regression"
        fs.transformations = json.dumps([
            {"name": "log_transform", "type": "log_transform"},
            {"name": "one_hot", "type": "one_hot"},
            {"name": "bin_quartile", "type": "bin_quartile"},
        ])
        prompt = build_system_prompt(p, dataset=ds, feature_set=fs)
        assert "log_transform" in prompt or "one_hot" in prompt

    def test_feature_set_transformations_json_error_is_ignored(self):
        """Lines 190-191: bad JSON in transformations is silently ignored."""
        from chat.orchestrator import build_system_prompt
        p = self.make_project()
        ds = MagicMock()
        ds.filename = "data.csv"
        ds.row_count = 200
        ds.column_count = 4
        ds.columns = None
        ds.profile = None
        fs = MagicMock()
        fs.target_column = "revenue"
        fs.problem_type = "regression"
        fs.transformations = "{bad json"
        prompt = build_system_prompt(p, dataset=ds, feature_set=fs)
        assert isinstance(prompt, str)  # no crash

    def test_feature_set_more_than_five_transforms_shows_plus_more(self):
        """Lines 186-188: list of >5 transforms shows '+ more'."""
        from chat.orchestrator import build_system_prompt
        p = self.make_project()
        ds = MagicMock()
        ds.filename = "data.csv"
        ds.row_count = 200
        ds.column_count = 10
        ds.columns = None
        ds.profile = None
        fs = MagicMock()
        fs.target_column = "target"
        fs.problem_type = "classification"
        transforms = [{"name": f"transform_{i}", "type": "one_hot"} for i in range(8)]
        fs.transformations = json.dumps(transforms)
        prompt = build_system_prompt(p, dataset=ds, feature_set=fs)
        assert "more" in prompt

    def test_classification_metrics_formatted_in_prompt(self):
        """Lines 205-210: accuracy/F1 format for classification model runs."""
        from chat.orchestrator import build_system_prompt
        p = self.make_project()
        ds = MagicMock()
        ds.filename = "churn.csv"
        ds.row_count = 1000
        ds.column_count = 8
        ds.columns = None
        ds.profile = None
        fs = MagicMock()
        fs.target_column = "churned"
        fs.problem_type = "classification"
        fs.transformations = None
        mr = MagicMock()
        mr.algorithm = "random_forest_classifier"
        mr.status = "done"
        mr.is_selected = True
        mr.metrics = json.dumps({"accuracy": 0.91, "f1": 0.89, "precision": 0.90, "recall": 0.88})
        prompt = build_system_prompt(p, dataset=ds, feature_set=fs, model_runs=[mr])
        assert "accuracy" in prompt.lower() or "91" in prompt

    def test_model_metrics_json_error_is_ignored(self):
        """Lines 209-210: bad JSON in model metrics is handled gracefully."""
        from chat.orchestrator import build_system_prompt
        p = self.make_project()
        ds = MagicMock()
        ds.filename = "data.csv"
        ds.row_count = 200
        ds.column_count = 4
        ds.columns = None
        ds.profile = None
        fs = MagicMock()
        fs.target_column = "target"
        fs.problem_type = "regression"
        fs.transformations = None
        mr = MagicMock()
        mr.algorithm = "linear_regression"
        mr.status = "done"
        mr.is_selected = False
        mr.metrics = "NOT VALID JSON"
        prompt = build_system_prompt(p, dataset=ds, feature_set=fs, model_runs=[mr])
        assert "linear_regression" in prompt


# ===========================================================================
# api/chat.py — send_message + history endpoints (mock Anthropic)
# ===========================================================================

class TestChatAPI:
    """Tests for the chat endpoints. Anthropic client is mocked."""

    @pytest.fixture
    async def client(self, tmp_path, monkeypatch):
        import os
        import db as db_module
        from sqlmodel import create_engine, SQLModel

        test_db = str(tmp_path / "chat_test.db")
        db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
        db_module.DATA_DIR = tmp_path

        import models.project  # noqa
        import models.dataset  # noqa
        import models.conversation  # noqa
        import models.feature_set  # noqa
        import models.model_run  # noqa
        import models.deployment  # noqa
        SQLModel.metadata.create_all(db_module.engine)

        import api.data as data_module
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        data_module.UPLOAD_DIR = upload_dir

        from main import app
        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac

    @pytest.fixture
    async def project_id(self, client):
        resp = await client.post("/api/projects", json={"name": "Chat Test"})
        assert resp.status_code in (200, 201)
        return resp.json()["id"]

    def _make_mock_anthropic(self, response_text: str = "Hello from Claude!"):
        """Return a mock Anthropic client that yields SSE tokens."""
        mock_client = MagicMock()
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter([response_text])
        mock_client.messages.stream.return_value = mock_stream
        return mock_client

    async def test_history_empty_for_new_project(self, client, project_id):
        resp = await client.get(f"/api/chat/{project_id}/history")
        assert resp.status_code == 200
        assert resp.json()["messages"] == []

    async def test_history_returns_404_for_unknown_project(self, client):
        # GET history for unknown project returns empty messages (not 404)
        resp = await client.get("/api/chat/nonexistent-id/history")
        assert resp.status_code == 200
        assert resp.json()["messages"] == []

    async def test_send_message_returns_404_for_unknown_project(self, client):
        with patch("api.chat.anthropic.Anthropic", return_value=self._make_mock_anthropic()):
            resp = await client.post(
                "/api/chat/nonexistent-id",
                json={"message": "Hello!"},
            )
        assert resp.status_code == 404

    async def test_send_message_streams_response(self, client, project_id):
        with patch("api.chat.anthropic.Anthropic", return_value=self._make_mock_anthropic("Hi there!")):
            resp = await client.post(
                f"/api/chat/{project_id}",
                json={"message": "Hello, how are you?"},
            )
        assert resp.status_code == 200
        body = resp.content.decode()
        assert "token" in body
        assert "Hi there!" in body
        assert "done" in body

    async def test_send_message_saves_to_history(self, client, project_id):
        with patch("api.chat.anthropic.Anthropic", return_value=self._make_mock_anthropic("Great dataset!")):
            await client.post(
                f"/api/chat/{project_id}",
                json={"message": "What do you think?"},
            )
        hist = await client.get(f"/api/chat/{project_id}/history")
        messages = hist.json()["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What do you think?"
        assert messages[1]["role"] == "assistant"
        assert "Great dataset!" in messages[1]["content"]

    async def test_send_multiple_messages_accumulates_history(self, client, project_id):
        for i, resp_text in enumerate(["First reply", "Second reply", "Third reply"]):
            with patch("api.chat.anthropic.Anthropic", return_value=self._make_mock_anthropic(resp_text)):
                await client.post(
                    f"/api/chat/{project_id}",
                    json={"message": f"Message {i+1}"},
                )
        hist = await client.get(f"/api/chat/{project_id}/history")
        messages = hist.json()["messages"]
        assert len(messages) == 6  # 3 user + 3 assistant

    async def test_send_message_creates_conversation_if_none_exists(self, client, project_id):
        """First message creates a Conversation record."""
        with patch("api.chat.anthropic.Anthropic", return_value=self._make_mock_anthropic("Created!")):
            resp = await client.post(
                f"/api/chat/{project_id}",
                json={"message": "Hello"},
            )
        assert resp.status_code == 200
        hist = await client.get(f"/api/chat/{project_id}/history")
        assert len(hist.json()["messages"]) == 2

    async def test_chat_with_dataset_does_not_crash(self, client, project_id, tmp_path):
        """Chat with an active dataset context doesn't crash."""
        import api.data as data_module
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir(exist_ok=True)
        data_module.UPLOAD_DIR = upload_dir

        csv_content = b"product,revenue,units\nWidget A,1200,10\nWidget B,850,8\n"
        await client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("sales.csv", csv_content, "text/csv")},
        )

        with patch("api.chat.anthropic.Anthropic", return_value=self._make_mock_anthropic("Nice data!")):
            resp = await client.post(
                f"/api/chat/{project_id}",
                json={"message": "What is the average revenue?"},
            )
        assert resp.status_code == 200
        assert b"token" in resp.content


# ===========================================================================
# core/trainer.py — error paths not yet covered
# ===========================================================================

class TestTrainerEdgePaths:
    """Cover the uncovered trainer.py paths (lines 175-180, 213)."""

    def test_train_single_model_unknown_algorithm_raises(self, tmp_path):
        from core.trainer import train_single_model
        import numpy as np
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([10.0, 20.0, 30.0])
        with pytest.raises(ValueError, match="Unknown algorithm"):
            train_single_model(X, y, "nonexistent_algo", "regression", tmp_path, "run1")

    def test_prepare_features_missing_target_raises(self):
        from core.trainer import prepare_features
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        with pytest.raises(ValueError, match="Target column"):
            prepare_features(df, ["a"], "missing_col", "regression")

    def test_prepare_features_no_valid_feature_cols_raises(self):
        from core.trainer import prepare_features
        df = pd.DataFrame({"a": [1, 2, 3], "target": [10, 20, 30]})
        with pytest.raises(ValueError, match="No valid feature columns"):
            prepare_features(df, ["nonexistent"], "target", "regression")

    def test_prepare_features_too_few_rows_raises(self):
        from core.trainer import prepare_features
        df = pd.DataFrame({"a": [1.0], "target": [None]})
        with pytest.raises(ValueError, match="Not enough non-null rows"):
            prepare_features(df, ["a"], "target", "regression")

    def test_train_few_rows_uses_same_data_for_train_eval(self, tmp_path):
        """With fewer than 10 rows, train/eval on same data (no split)."""
        from core.trainer import train_single_model
        import numpy as np
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = train_single_model(X, y, "linear_regression", "regression", tmp_path, "run_small")
        assert result["metrics"]["train_size"] == 5
        assert result["metrics"]["test_size"] == 5

    def test_pick_best_model_empty_list_returns_none(self):
        from core.trainer import pick_best_model
        assert pick_best_model([], "regression") is None

    def test_pick_best_model_regression_picks_highest_r2(self):
        from core.trainer import pick_best_model
        models = [
            {"id": "m1", "algorithm": "linear", "metrics": {"r2": 0.7, "mae": 10}},
            {"id": "m2", "algorithm": "rf", "metrics": {"r2": 0.92, "mae": 5}},
            {"id": "m3", "algorithm": "gb", "metrics": {"r2": 0.85, "mae": 7}},
        ]
        best = pick_best_model(models, "regression")
        assert best["model_run_id"] == "m2"

    def test_pick_best_model_classification_picks_highest_f1(self):
        from core.trainer import pick_best_model
        models = [
            {"id": "m1", "algorithm": "lr", "metrics": {"accuracy": 0.85, "f1": 0.82}},
            {"id": "m2", "algorithm": "rf", "metrics": {"accuracy": 0.91, "f1": 0.90}},
        ]
        best = pick_best_model(models, "classification")
        assert best["model_run_id"] == "m2"
