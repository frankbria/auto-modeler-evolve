"""Tests for local explanation (feature contribution waterfall) chat card."""

import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Unit tests for _extract_row_index helper
# ---------------------------------------------------------------------------


class TestExtractRowIndex:
    @pytest.fixture
    def extractor(self):
        from api.chat import _extract_row_index

        return _extract_row_index

    def test_explicit_row_number(self, extractor):
        assert extractor("explain prediction for row 5") == 5

    def test_record_keyword(self, extractor):
        assert extractor("explain record 10") == 10

    def test_index_keyword(self, extractor):
        assert extractor("explain index 3") == 3

    def test_hash_notation(self, extractor):
        assert extractor("show feature contributions for row #7") == 7

    def test_hash_alone(self, extractor):
        assert extractor("explain #12") == 12

    def test_no_number_defaults_zero(self, extractor):
        assert extractor("explain this specific prediction") == 0

    def test_generic_message_defaults_zero(self, extractor):
        assert extractor("show me SHAP values") == 0

    def test_first_number_taken(self, extractor):
        # row 2 appears first in message
        assert extractor("explain prediction for row 2 using index 5") == 2


# ---------------------------------------------------------------------------
# Unit tests for explain_single_prediction
# ---------------------------------------------------------------------------


class TestExplainSinglePrediction:
    def _make_regression_model(self):
        from sklearn.linear_model import LinearRegression

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        model = LinearRegression()
        model.fit(X, y)
        return model, X, y

    def _make_classification_model(self):
        from sklearn.linear_model import LogisticRegression

        X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=float)
        y = np.array([0, 1, 1, 0])
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        return model, X, y

    def test_regression_returns_required_fields(self):
        from core.explainer import explain_single_prediction

        model, X, y = self._make_regression_model()
        result = explain_single_prediction(
            model, X[0], X, ["f1", "f2"], "regression", "target"
        )
        for field in ("prediction", "prediction_value", "contributions", "summary"):
            assert field in result

    def test_regression_contributions_have_correct_structure(self):
        from core.explainer import explain_single_prediction

        model, X, y = self._make_regression_model()
        result = explain_single_prediction(
            model, X[0], X, ["f1", "f2"], "regression", "target"
        )
        for c in result["contributions"]:
            assert "feature" in c
            assert "value" in c
            assert "mean_value" in c
            assert "contribution" in c
            assert c["direction"] in ("positive", "negative")

    def test_regression_sorted_by_abs_contribution(self):
        from core.explainer import explain_single_prediction

        model, X, y = self._make_regression_model()
        result = explain_single_prediction(
            model, X[0], X, ["f1", "f2"], "regression", "target"
        )
        contribs = [abs(c["contribution"]) for c in result["contributions"]]
        assert contribs == sorted(contribs, reverse=True)

    def test_regression_feature_count_matches(self):
        from core.explainer import explain_single_prediction

        model, X, y = self._make_regression_model()
        result = explain_single_prediction(
            model, X[0], X, ["f1", "f2"], "regression", "target"
        )
        assert len(result["contributions"]) == 2

    def test_classification_returns_prediction_class(self):
        from core.explainer import explain_single_prediction

        model, X, y = self._make_classification_model()
        result = explain_single_prediction(
            model, X[0], X, ["f1", "f2"], "classification", "label"
        )
        assert result["prediction"] in (0, 1)

    def test_classification_summary_mentions_target(self):
        from core.explainer import explain_single_prediction

        model, X, y = self._make_classification_model()
        result = explain_single_prediction(
            model, X[0], X, ["f1", "f2"], "classification", "my_label"
        )
        assert "my_label" in result["summary"]

    def test_regression_summary_non_empty(self):
        from core.explainer import explain_single_prediction

        model, X, y = self._make_regression_model()
        result = explain_single_prediction(
            model, X[0], X, ["f1", "f2"], "regression", "revenue"
        )
        assert len(result["summary"]) > 5


# ---------------------------------------------------------------------------
# Pattern matching tests
# ---------------------------------------------------------------------------


class TestExplainRowPatterns:
    @pytest.fixture
    def pattern(self):
        from api.chat import _EXPLAIN_ROW_PATTERNS

        return _EXPLAIN_ROW_PATTERNS

    @pytest.mark.parametrize(
        "msg",
        [
            "explain prediction for row 5",
            "explain this specific prediction",
            "show me SHAP values",
            "show me the feature contributions",
            "give me the local explanation",
            "what drove this prediction",
            "what caused the prediction",
            "why did the model predict that",
            "individual prediction explanation",
            "local model explanation",
            "waterfall chart",
            "feature contributions for row 3",
            "explain record 10",
            "explain index 0",
        ],
    )
    def test_positive_matches(self, pattern, msg):
        assert pattern.search(msg), f"Should match: {msg!r}"

    @pytest.mark.parametrize(
        "msg",
        [
            "train a model",
            "show me the data",
            "how consistent is my model",
            "partial dependence for price",
            "confusion matrix",
            "show me prediction errors",
        ],
    )
    def test_negative_no_match(self, pattern, msg):
        assert not pattern.search(msg), f"Should not match: {msg!r}"


# ---------------------------------------------------------------------------
# Integration: chat handler emits local_explanation event
# ---------------------------------------------------------------------------


class TestLocalExplanationChatIntegration:
    """End-to-end: chat endpoint emits {type:'local_explanation'} SSE event."""

    @pytest.fixture(autouse=True)
    def _patch_db(self, tmp_path, monkeypatch):
        from sqlmodel import SQLModel, create_engine
        import db as db_module
        import models.project  # noqa: F401
        import models.dataset  # noqa: F401
        import models.feature_set  # noqa: F401
        import models.model_run  # noqa: F401
        import models.conversation  # noqa: F401
        import models.deployment  # noqa: F401

        engine = create_engine(f"sqlite:///{tmp_path}/test.db")
        monkeypatch.setattr(db_module, "engine", engine)
        SQLModel.metadata.create_all(engine)
        self._engine = engine

    @pytest.fixture()
    def client(self):
        from main import app

        return TestClient(app)

    @pytest.fixture()
    def project_with_model(self, _patch_db, tmp_path):
        import csv
        import uuid

        import joblib
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sqlmodel import Session
        from core.trainer import prepare_features
        from models.project import Project
        from models.dataset import Dataset
        from models.feature_set import FeatureSet
        from models.model_run import ModelRun

        csv_path = tmp_path / "data.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["units", "price", "revenue"])
            for i in range(30):
                writer.writerow([i + 1, (i + 1) * 2, (i + 1) * 5 + 10])

        df = pd.read_csv(csv_path)
        X, y, _ = prepare_features(df, ["units", "price"], "revenue", "regression")
        model = LinearRegression()
        model.fit(X, y)
        model_path = str(tmp_path / f"{uuid.uuid4()}_model.joblib")
        joblib.dump(model, model_path)

        with Session(self._engine) as session:
            project = Project(name="ExplainTest", status="exploring")
            session.add(project)
            session.flush()

            dataset = Dataset(
                project_id=project.id,
                filename="data.csv",
                file_path=str(csv_path),
                row_count=30,
                column_count=3,
                columns=json.dumps(
                    [{"name": "units"}, {"name": "price"}, {"name": "revenue"}]
                ),
            )
            session.add(dataset)
            session.flush()

            feature_set = FeatureSet(
                dataset_id=dataset.id,
                target_column="revenue",
                problem_type="regression",
                transformations="[]",
                is_active=True,
            )
            session.add(feature_set)
            session.flush()

            run = ModelRun(
                project_id=project.id,
                feature_set_id=feature_set.id,
                algorithm="linear_regression",
                status="done",
                model_path=model_path,
                metrics=json.dumps({"r2": 0.99, "mae": 0.5}),
                is_selected=True,
            )
            session.add(run)
            session.commit()
            return project.id

    def _chat_events(self, client, project_id, message):
        with patch("anthropic.Anthropic") as mock_cls:
            mock_c = MagicMock()
            mock_cls.return_value = mock_c
            mock_stream = MagicMock()
            mock_stream.__enter__ = MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = MagicMock(return_value=False)
            mock_stream.text_stream = iter(["OK"])
            mock_c.messages.stream.return_value = mock_stream
            response = client.post(f"/api/chat/{project_id}", json={"message": message})

        events = []
        for line in response.text.split("\n"):
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
        return events

    def test_local_explanation_event_emitted(self, client, project_with_model):
        events = self._chat_events(
            client, project_with_model, "explain this specific prediction"
        )
        assert any(e.get("type") == "local_explanation" for e in events)

    def test_local_explanation_event_has_required_fields(
        self, client, project_with_model
    ):
        events = self._chat_events(
            client, project_with_model, "show me feature contributions for row 0"
        )
        le_events = [e for e in events if e.get("type") == "local_explanation"]
        assert len(le_events) >= 1
        payload = le_events[0]["local_explanation"]
        for field in (
            "row_index",
            "algorithm",
            "target_col",
            "problem_type",
            "contributions",
            "summary",
            "actual_value",
            "predicted_value",
        ):
            assert field in payload, f"Missing field: {field}"

    def test_local_explanation_contributions_structure(
        self, client, project_with_model
    ):
        events = self._chat_events(
            client, project_with_model, "show me the feature contributions"
        )
        le_events = [e for e in events if e.get("type") == "local_explanation"]
        assert le_events
        for c in le_events[0]["local_explanation"]["contributions"]:
            assert "feature" in c
            assert "contribution" in c
            assert c["direction"] in ("positive", "negative")

    def test_local_explanation_row_index_defaults_zero(
        self, client, project_with_model
    ):
        events = self._chat_events(
            client, project_with_model, "what drove this prediction"
        )
        le_events = [e for e in events if e.get("type") == "local_explanation"]
        assert le_events
        assert le_events[0]["local_explanation"]["row_index"] == 0

    def test_local_explanation_specific_row_index(self, client, project_with_model):
        events = self._chat_events(
            client, project_with_model, "explain prediction for row 5"
        )
        le_events = [e for e in events if e.get("type") == "local_explanation"]
        assert le_events
        assert le_events[0]["local_explanation"]["row_index"] == 5

    def test_no_local_explanation_without_model(self, client, tmp_path):
        """No event when project has no trained model."""
        from sqlmodel import Session
        from models.project import Project
        from models.dataset import Dataset

        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("units,revenue\n1,5\n")

        with Session(self._engine) as session:
            project = Project(name="NoModel", status="exploring")
            session.add(project)
            session.flush()
            dataset = Dataset(
                project_id=project.id,
                filename="empty.csv",
                file_path=str(csv_path),
                row_count=1,
                column_count=2,
                columns=json.dumps([]),
            )
            session.add(dataset)
            session.commit()
            pid = project.id

        events = self._chat_events(client, pid, "explain this specific prediction")
        assert not any(e.get("type") == "local_explanation" for e in events)
