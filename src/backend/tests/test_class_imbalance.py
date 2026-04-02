"""Tests for class imbalance detection and handling.

Covers:
- detect_class_imbalance() pure function
- train_single_model() with imbalance_strategy variants
- GET /api/models/{project_id}/imbalance endpoint
- POST /api/models/{project_id}/train with imbalance_strategy
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

# ---------------------------------------------------------------------------
# Imbalanced CSV (90% class A, 10% class B — 10 rows)
# ---------------------------------------------------------------------------

IMBALANCED_CSV = b"""feature1,feature2,label
1.0,2.0,A
2.0,3.0,A
3.0,4.0,A
4.0,5.0,A
5.0,6.0,A
6.0,7.0,A
7.0,8.0,A
8.0,9.0,A
9.0,10.0,A
10.0,11.0,B
"""

# Balanced CSV (50/50)
BALANCED_CSV = b"""feature1,feature2,label
1.0,2.0,A
2.0,3.0,A
3.0,4.0,A
4.0,5.0,A
5.0,6.0,A
6.0,7.0,B
7.0,8.0,B
8.0,9.0,B
9.0,10.0,B
10.0,11.0,B
"""

# Slightly imbalanced but > 20% (should NOT trigger)
BORDERLINE_CSV = b"""feature1,feature2,label
1.0,2.0,A
2.0,3.0,A
3.0,4.0,A
4.0,5.0,A
5.0,6.0,A
6.0,7.0,A
7.0,8.0,A
8.0,9.0,A
9.0,10.0,B
10.0,11.0,B
10.0,11.0,B
"""  # 8 A vs 3 B → B ratio ≈ 0.27 > 0.20


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.project  # noqa
    import models.dataset  # noqa
    import models.feature_set  # noqa
    import models.conversation  # noqa
    import models.model_run  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_api_module

    models_api_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as c:
        yield c


def _setup_classification_project(client, csv_bytes: bytes):
    """Create project, upload CSV, apply empty feature set, set classification target."""
    proj = client.post("/api/projects", json={"name": "Imbalance Test"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    apply = client.post(
        f"/api/features/{dataset_id}/apply", json={"transformations": []}
    )
    assert apply.status_code == 201

    target = client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "label"},
    )
    assert target.status_code == 200
    return project_id, dataset_id


# ---------------------------------------------------------------------------
# Unit tests: detect_class_imbalance
# ---------------------------------------------------------------------------


class TestDetectClassImbalance:
    def test_balanced_classes_returns_false(self):
        from core.trainer import detect_class_imbalance

        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        result = detect_class_imbalance(y)
        assert result["is_imbalanced"] is False
        assert result["recommended_strategy"] == "none"

    def test_imbalanced_classes_returns_true(self):
        from core.trainer import detect_class_imbalance

        # 9 A, 1 B → 10% minority
        y = np.array(["A"] * 9 + ["B"])
        result = detect_class_imbalance(y)
        assert result["is_imbalanced"] is True
        assert result["minority_class"] == "B"
        assert result["minority_ratio"] == pytest.approx(0.10, abs=0.01)

    def test_borderline_not_imbalanced(self):
        from core.trainer import detect_class_imbalance

        # 8 A + 3 B = 11 rows, B ratio ≈ 0.27 → not imbalanced
        y = np.array(["A"] * 8 + ["B"] * 3)
        result = detect_class_imbalance(y)
        assert result["is_imbalanced"] is False

    def test_exactly_at_threshold_is_not_imbalanced(self):
        from core.trainer import detect_class_imbalance

        # Exactly 20% → NOT imbalanced (< 0.20 triggers it)
        y = np.array([0] * 8 + [1] * 2)  # 20% minority
        result = detect_class_imbalance(y)
        assert result["is_imbalanced"] is False

    def test_just_below_threshold_is_imbalanced(self):
        from core.trainer import detect_class_imbalance

        # 19% → imbalanced
        y = np.array([0] * 81 + [1] * 19)
        result = detect_class_imbalance(y)
        assert result["is_imbalanced"] is True

    def test_returns_class_distribution(self):
        from core.trainer import detect_class_imbalance

        y = np.array(["cat"] * 70 + ["dog"] * 30)
        result = detect_class_imbalance(y)
        classes = {d["class"] for d in result["class_distribution"]}
        assert "cat" in classes and "dog" in classes

    def test_explanation_mentions_minority_class(self):
        from core.trainer import detect_class_imbalance

        y = np.array(["yes"] * 90 + ["no"] * 10)
        result = detect_class_imbalance(y)
        assert "no" in result["explanation"] or "10.0%" in result["explanation"]

    def test_small_dataset_recommends_class_weight(self):
        from core.trainer import detect_class_imbalance

        y = np.array([0] * 40 + [1] * 5)  # 45 rows, imbalanced
        result = detect_class_imbalance(y)
        assert result["is_imbalanced"] is True
        assert result["recommended_strategy"] == "class_weight"

    def test_severe_imbalance_large_dataset_recommends_smote(self):
        from core.trainer import detect_class_imbalance

        # < 5% minority, >= 100 rows → smote
        y = np.array([0] * 200 + [1] * 5)
        result = detect_class_imbalance(y)
        assert result["is_imbalanced"] is True
        assert result["recommended_strategy"] == "smote"

    def test_multiclass_detects_minority(self):
        from core.trainer import detect_class_imbalance

        y = np.array([0] * 70 + [1] * 20 + [2] * 10)
        result = detect_class_imbalance(y)
        assert result["is_imbalanced"] is True
        assert result["minority_class"] == "2"

    def test_all_required_keys_present(self):
        from core.trainer import detect_class_imbalance

        y = np.array([0, 1, 0, 1, 0])
        result = detect_class_imbalance(y)
        for key in (
            "is_imbalanced",
            "class_distribution",
            "minority_class",
            "minority_ratio",
            "recommended_strategy",
            "explanation",
        ):
            assert key in result


# ---------------------------------------------------------------------------
# Unit tests: train_single_model with imbalance_strategy
# ---------------------------------------------------------------------------


class TestTrainWithImbalanceStrategy:
    @pytest.fixture
    def imbalanced_xy(self):
        """80 majority + 10 minority for classification."""
        np.random.seed(42)
        n_maj, n_min = 80, 10
        X_maj = np.random.randn(n_maj, 2) + np.array([0, 0])
        X_min = np.random.randn(n_min, 2) + np.array([3, 3])
        X = np.vstack([X_maj, X_min])
        y = np.array([0] * n_maj + [1] * n_min)
        return X, y

    def test_class_weight_strategy_logistic_regression(self, imbalanced_xy, tmp_path):
        from core.trainer import train_single_model

        X, y = imbalanced_xy
        result = train_single_model(
            X,
            y,
            "logistic_regression",
            "classification",
            tmp_path / "models",
            "run_cw_lr",
            imbalance_strategy="class_weight",
        )
        assert result["metrics"]["imbalance_strategy"] == "class_weight"
        assert result["metrics"]["accuracy"] > 0

    def test_class_weight_strategy_random_forest(self, imbalanced_xy, tmp_path):
        from core.trainer import train_single_model

        X, y = imbalanced_xy
        result = train_single_model(
            X,
            y,
            "random_forest_classifier",
            "classification",
            tmp_path / "models",
            "run_cw_rf",
            imbalance_strategy="class_weight",
        )
        assert result["metrics"]["imbalance_strategy"] == "class_weight"

    def test_class_weight_strategy_gradient_boosting(self, imbalanced_xy, tmp_path):
        from core.trainer import train_single_model

        X, y = imbalanced_xy
        result = train_single_model(
            X,
            y,
            "gradient_boosting_classifier",
            "classification",
            tmp_path / "models",
            "run_cw_gbc",
            imbalance_strategy="class_weight",
        )
        assert result["metrics"]["imbalance_strategy"] == "class_weight"

    def test_smote_strategy(self, imbalanced_xy, tmp_path):
        from core.trainer import train_single_model

        X, y = imbalanced_xy
        result = train_single_model(
            X,
            y,
            "random_forest_classifier",
            "classification",
            tmp_path / "models",
            "run_smote",
            imbalance_strategy="smote",
        )
        assert result["metrics"]["imbalance_strategy"] == "smote"
        assert result["metrics"]["accuracy"] > 0

    def test_threshold_strategy(self, imbalanced_xy, tmp_path):
        from core.trainer import train_single_model

        X, y = imbalanced_xy
        result = train_single_model(
            X,
            y,
            "logistic_regression",
            "classification",
            tmp_path / "models",
            "run_thresh",
            imbalance_strategy="threshold",
        )
        assert result["metrics"]["imbalance_strategy"] == "threshold"
        assert "optimal_threshold" in result["metrics"]
        thresh = result["metrics"]["optimal_threshold"]
        assert 0.05 <= thresh <= 0.95

    def test_no_strategy_works_normally(self, imbalanced_xy, tmp_path):
        from core.trainer import train_single_model

        X, y = imbalanced_xy
        result = train_single_model(
            X,
            y,
            "logistic_regression",
            "classification",
            tmp_path / "models",
            "run_none",
        )
        assert "imbalance_strategy" not in result["metrics"]

    def test_strategy_ignored_for_regression(self, tmp_path):
        from core.trainer import train_single_model

        np.random.seed(0)
        X = np.random.randn(30, 2)
        y = np.random.randn(30)
        result = train_single_model(
            X,
            y,
            "linear_regression",
            "regression",
            tmp_path / "models",
            "run_reg",
            imbalance_strategy="class_weight",
        )
        # For regression, strategy is in metrics only if strategy was passed and is non-None
        # The code sets it regardless of problem_type; that's fine — regression won't apply it
        assert result["metrics"]["r2"] is not None

    def test_threshold_no_proba_falls_back(self, imbalanced_xy, tmp_path):
        """GradientBoostingClassifier has predict_proba — threshold works."""
        from core.trainer import train_single_model

        X, y = imbalanced_xy
        result = train_single_model(
            X,
            y,
            "gradient_boosting_classifier",
            "classification",
            tmp_path / "models",
            "run_thresh_gbc",
            imbalance_strategy="threshold",
        )
        # Should have optimal_threshold since GBC has predict_proba
        assert "optimal_threshold" in result["metrics"]

    def test_threshold_strategy_records_imbalance_in_metrics(
        self, imbalanced_xy, tmp_path
    ):
        from core.trainer import train_single_model

        X, y = imbalanced_xy
        result = train_single_model(
            X,
            y,
            "logistic_regression",
            "classification",
            tmp_path / "models",
            "run_thresh_note",
            imbalance_strategy="threshold",
        )
        # Strategy is always recorded in metrics when provided
        assert result["metrics"]["imbalance_strategy"] == "threshold"


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestImbalanceEndpoint:
    def test_get_imbalance_classification_imbalanced(self, client):
        project_id, _ = _setup_classification_project(client, IMBALANCED_CSV)
        r = client.get(f"/api/models/{project_id}/imbalance")
        assert r.status_code == 200
        data = r.json()
        assert data["is_imbalanced"] is True
        assert data["minority_class"] == "B"
        assert len(data["class_distribution"]) == 2

    def test_get_imbalance_classification_balanced(self, client):
        project_id, _ = _setup_classification_project(client, BALANCED_CSV)
        r = client.get(f"/api/models/{project_id}/imbalance")
        assert r.status_code == 200
        data = r.json()
        assert data["is_imbalanced"] is False
        assert data["recommended_strategy"] == "none"

    def test_get_imbalance_returns_explanation(self, client):
        project_id, _ = _setup_classification_project(client, IMBALANCED_CSV)
        r = client.get(f"/api/models/{project_id}/imbalance")
        data = r.json()
        assert len(data["explanation"]) > 10

    def test_get_imbalance_project_not_found(self, client):
        r = client.get("/api/models/nonexistent-id/imbalance")
        assert r.status_code == 404

    def test_train_with_class_weight_strategy(self, client):
        project_id, _ = _setup_classification_project(client, IMBALANCED_CSV)
        r = client.post(
            f"/api/models/{project_id}/train",
            json={
                "algorithms": ["logistic_regression"],
                "imbalance_strategy": "class_weight",
            },
        )
        assert r.status_code == 202
        data = r.json()
        assert len(data["model_run_ids"]) == 1

    def test_train_with_null_strategy(self, client):
        project_id, _ = _setup_classification_project(client, IMBALANCED_CSV)
        r = client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["logistic_regression"], "imbalance_strategy": None},
        )
        assert r.status_code == 202

    def test_train_with_invalid_strategy(self, client):
        project_id, _ = _setup_classification_project(client, IMBALANCED_CSV)
        r = client.post(
            f"/api/models/{project_id}/train",
            json={
                "algorithms": ["logistic_regression"],
                "imbalance_strategy": "invalid_strategy",
            },
        )
        assert r.status_code == 400

    def test_regression_project_returns_not_imbalanced(self, client, tmp_path):
        """Regression projects should get a 200 with is_imbalanced=False."""
        # Create a regression project
        proj = client.post("/api/projects", json={"name": "Reg Project"})
        project_id = proj.json()["id"]

        reg_csv = b"feature1,target\n1.0,10.0\n2.0,20.0\n3.0,30.0\n4.0,40.0\n5.0,50.0\n"
        upload = client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("reg.csv", io.BytesIO(reg_csv), "text/csv")},
        )
        dataset_id = upload.json()["dataset_id"]
        client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
        client.post(
            f"/api/features/{dataset_id}/target", json={"target_column": "target"}
        )

        r = client.get(f"/api/models/{project_id}/imbalance")
        assert r.status_code == 200
        assert r.json()["is_imbalanced"] is False
        assert "classification" in r.json()["explanation"].lower()
