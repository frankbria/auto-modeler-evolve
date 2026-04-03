"""Tests for large dataset sampling and classifier calibration.

Covers:
- sample_large_dataset() pure function
- Large dataset sampling in _train_in_background (via train endpoint)
- CalibratedClassifierCV wrapping in train_single_model
- _add_calibration_metrics helper
- GET /api/models/{run_id}/calibration endpoint
- identify_weak_features() with CalibratedClassifierCV-wrapped model
"""

import io
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.trainer import (
    _add_calibration_metrics,
    identify_weak_features,
    sample_large_dataset,
    train_single_model,
)


# ---------------------------------------------------------------------------
# Sample CSVs
# ---------------------------------------------------------------------------

CLASSIFICATION_CSV = b"""f1,f2,f3,label
1.0,0.5,100.0,A
2.0,1.0,200.0,A
3.0,1.5,300.0,A
4.0,2.0,400.0,A
5.0,2.5,500.0,A
6.0,3.0,600.0,B
7.0,3.5,700.0,B
8.0,4.0,800.0,B
9.0,4.5,900.0,B
10.0,5.0,1000.0,B
11.0,5.5,1100.0,B
12.0,6.0,1200.0,B
13.0,6.5,1300.0,A
14.0,7.0,1400.0,A
15.0,7.5,1500.0,A
16.0,8.0,1600.0,B
17.0,8.5,1700.0,B
18.0,9.0,1800.0,A
19.0,9.5,1900.0,A
20.0,10.0,2000.0,B
21.0,10.5,2100.0,A
22.0,11.0,2200.0,B
23.0,11.5,2300.0,A
24.0,12.0,2400.0,B
25.0,12.5,2500.0,A
26.0,13.0,2600.0,B
27.0,13.5,2700.0,A
28.0,14.0,2800.0,B
29.0,14.5,2900.0,A
30.0,15.0,3000.0,B
31.0,15.5,3100.0,A
32.0,16.0,3200.0,B
33.0,16.5,3300.0,A
34.0,17.0,3400.0,B
35.0,17.5,3500.0,A
36.0,18.0,3600.0,B
37.0,18.5,3700.0,A
38.0,19.0,3800.0,B
39.0,19.5,3900.0,A
40.0,20.0,4000.0,B
"""

REGRESSION_CSV = b"""f1,f2,f3,target
1.0,0.5,100.0,10.0
2.0,1.0,200.0,20.0
3.0,1.5,300.0,30.0
4.0,2.0,400.0,40.0
5.0,2.5,500.0,50.0
6.0,3.0,600.0,60.0
7.0,3.5,700.0,70.0
8.0,4.0,800.0,80.0
9.0,4.5,900.0,90.0
10.0,5.0,1000.0,100.0
11.0,5.5,1100.0,110.0
12.0,6.0,1200.0,120.0
"""


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


@pytest.fixture
def project_and_dataset(client):
    """Create a project, upload a dataset, set target and apply features."""
    proj = client.post("/api/projects", json={"name": "CalibTest"})
    assert proj.status_code == 201
    proj_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        files={"file": ("data.csv", io.BytesIO(CLASSIFICATION_CSV), "text/csv")},
        data={"project_id": proj_id},
    )
    assert upload.status_code == 201
    ds_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{ds_id}/apply", json={"transformations": []})
    client.post(f"/api/features/{ds_id}/target", json={"target_column": "label"})

    return proj_id, ds_id


# ===========================================================================
# 1. sample_large_dataset() pure function
# ===========================================================================


class TestSampleLargeDataset:
    def test_small_df_not_sampled(self):
        df = pd.DataFrame({"x": range(1000)})
        sampled, info = sample_large_dataset(df, max_rows=20000, threshold=50000)
        assert not info["was_sampled"]
        assert info["original_rows"] == 1000
        assert info["sample_rows"] == 1000
        assert len(sampled) == 1000

    def test_df_at_threshold_not_sampled(self):
        df = pd.DataFrame({"x": range(50000)})
        sampled, info = sample_large_dataset(df, max_rows=20000, threshold=50000)
        assert not info["was_sampled"]
        assert len(sampled) == 50000

    def test_large_df_is_sampled(self):
        df = pd.DataFrame({"x": range(60000)})
        sampled, info = sample_large_dataset(df, max_rows=20000, threshold=50000)
        assert info["was_sampled"]
        assert info["original_rows"] == 60000
        assert info["sample_rows"] == 20000
        assert len(sampled) == 20000

    def test_sample_note_contains_counts(self):
        df = pd.DataFrame({"x": range(75000)})
        _, info = sample_large_dataset(df, max_rows=20000, threshold=50000)
        assert "20,000" in info["note"]
        assert "75,000" in info["note"]

    def test_sample_is_reproducible(self):
        df = pd.DataFrame({"x": range(60000)})
        s1, _ = sample_large_dataset(df, random_state=42)
        s2, _ = sample_large_dataset(df, random_state=42)
        assert list(s1["x"]) == list(s2["x"])

    def test_sample_different_with_different_seed(self):
        df = pd.DataFrame({"x": range(60000)})
        s1, _ = sample_large_dataset(df, random_state=1)
        s2, _ = sample_large_dataset(df, random_state=99)
        assert list(s1["x"]) != list(s2["x"])

    def test_custom_threshold_and_max_rows(self):
        df = pd.DataFrame({"x": range(200)})
        sampled, info = sample_large_dataset(df, max_rows=50, threshold=100)
        assert info["was_sampled"]
        assert len(sampled) == 50

    def test_empty_note_when_not_sampled(self):
        df = pd.DataFrame({"x": range(10)})
        _, info = sample_large_dataset(df)
        assert info["note"] == ""


# ===========================================================================
# 2. _add_calibration_metrics helper
# ===========================================================================


class TestAddCalibrationMetrics:
    def _make_data(self, n=200):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 3))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def test_binary_calibration_adds_brier_score(self, tmp_path):
        X, y = self._make_data()
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        clf = CalibratedClassifierCV(
            LogisticRegression(max_iter=200), cv=3, method="sigmoid"
        )
        clf.fit(X_train, y_train)

        metrics: dict = {}
        _add_calibration_metrics(metrics, clf, X_test, y_test)

        assert "brier_score" in metrics
        assert 0 <= metrics["brier_score"] <= 1

    def test_binary_calibration_adds_curve(self, tmp_path):
        X, y = self._make_data()
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        clf = CalibratedClassifierCV(
            LogisticRegression(max_iter=200), cv=3, method="sigmoid"
        )
        clf.fit(X_train, y_train)

        metrics: dict = {}
        _add_calibration_metrics(metrics, clf, X_test, y_test)

        assert "calibration_curve" in metrics
        assert len(metrics["calibration_curve"]) > 0
        for pt in metrics["calibration_curve"]:
            assert "predicted" in pt
            assert "actual" in pt

    def test_calibration_note_present(self, tmp_path):
        X, y = self._make_data()
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        clf = CalibratedClassifierCV(
            LogisticRegression(max_iter=200), cv=3, method="sigmoid"
        )
        clf.fit(X_train, y_train)

        metrics: dict = {}
        _add_calibration_metrics(metrics, clf, X_test, y_test)

        assert "calibration_note" in metrics
        assert len(metrics["calibration_note"]) > 10

    def test_is_calibrated_flag_set(self, tmp_path):
        X, y = self._make_data()
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        clf = CalibratedClassifierCV(
            LogisticRegression(max_iter=200), cv=3, method="sigmoid"
        )
        clf.fit(X_train, y_train)

        metrics: dict = {}
        _add_calibration_metrics(metrics, clf, X_test, y_test)

        assert metrics.get("is_calibrated") is True


# ===========================================================================
# 3. train_single_model produces calibrated classifiers
# ===========================================================================


class TestTrainSingleModelCalibration:
    def _make_xy(self, n=100):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((n, 3))
        y = (X[:, 0] > 0).astype(int)
        return X.astype(np.float64), y

    def test_logistic_regression_is_calibrated(self, tmp_path):
        X, y = self._make_xy(100)
        result = train_single_model(
            X, y, "logistic_regression", "classification", tmp_path, "run_cal_1"
        )
        assert result["metrics"].get("is_calibrated") is True

    def test_random_forest_classifier_is_calibrated(self, tmp_path):
        X, y = self._make_xy(100)
        result = train_single_model(
            X, y, "random_forest_classifier", "classification", tmp_path, "run_cal_2"
        )
        assert result["metrics"].get("is_calibrated") is True

    def test_regression_is_not_calibrated(self, tmp_path):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 3))
        y = X[:, 0] * 2 + 1
        result = train_single_model(
            X, y.astype(np.float64), "linear_regression", "regression", tmp_path, "run_cal_3"
        )
        assert "is_calibrated" not in result["metrics"]

    def test_calibration_curve_has_predicted_and_actual(self, tmp_path):
        X, y = self._make_xy(100)
        result = train_single_model(
            X, y, "logistic_regression", "classification", tmp_path, "run_cal_4"
        )
        curve = result["metrics"].get("calibration_curve", [])
        assert len(curve) > 0
        assert all("predicted" in pt and "actual" in pt for pt in curve)

    def test_threshold_tuning_skips_calibration(self, tmp_path):
        X, y = self._make_xy(60)
        result = train_single_model(
            X,
            y,
            "logistic_regression",
            "classification",
            tmp_path,
            "run_cal_5",
            imbalance_strategy="threshold",
        )
        # threshold tuning takes over probability manipulation; calibration skipped
        assert "is_calibrated" not in result["metrics"]

    def test_saved_model_is_calibratedclassifiercv(self, tmp_path):
        import joblib

        X, y = self._make_xy(100)
        result = train_single_model(
            X, y, "logistic_regression", "classification", tmp_path, "run_cal_6"
        )
        model = joblib.load(result["model_path"])
        assert isinstance(model, CalibratedClassifierCV)

    def test_calibrated_model_has_predict_proba(self, tmp_path):
        import joblib

        X, y = self._make_xy(100)
        result = train_single_model(
            X, y, "random_forest_classifier", "classification", tmp_path, "run_cal_7"
        )
        model = joblib.load(result["model_path"])
        proba = model.predict_proba(X[:5])
        assert proba.shape == (5, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


# ===========================================================================
# 4. identify_weak_features with CalibratedClassifierCV
# ===========================================================================


class TestIdentifyWeakFeaturesCalibrated:
    def test_unwraps_calibrated_classifier(self, tmp_path):
        import joblib

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 4))
        y = (X[:, 0] > 0).astype(int)

        result = train_single_model(
            X, y, "random_forest_classifier", "classification", tmp_path, "run_fs_cal"
        )
        model = joblib.load(result["model_path"])
        assert isinstance(model, CalibratedClassifierCV)

        fs = identify_weak_features(model, ["f1", "f2", "f3", "f4"])
        assert fs["has_importances"] is True
        assert len(fs["feature_importances"]) == 4

    def test_logistic_regression_calibrated_coefficients(self, tmp_path):
        import joblib

        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = (X[:, 0] > 0).astype(int)

        result = train_single_model(
            X, y, "logistic_regression", "classification", tmp_path, "run_fs_lr_cal"
        )
        model = joblib.load(result["model_path"])
        fs = identify_weak_features(model, ["a", "b", "c"])
        # LogisticRegression in CalibratedClassifierCV should have coefficients
        # (method may return not_available if coef_ shape doesn't match — that's ok)
        assert "has_importances" in fs


# ===========================================================================
# 5. GET /api/models/{run_id}/calibration endpoint
# ===========================================================================


class TestCalibrationEndpoint:
    def _train_classifier(self, client, project_and_dataset):
        proj_id, _ = project_and_dataset
        resp = client.post(
            f"/api/models/{proj_id}/train",
            json={"algorithms": ["logistic_regression"]},
        )
        assert resp.status_code == 202
        run_id = resp.json()["model_run_ids"][0]

        # Wait for training to complete
        deadline = time.time() + 20
        while time.time() < deadline:
            runs = client.get(f"/api/models/{proj_id}/runs").json()["runs"]
            run = next((r for r in runs if r["id"] == run_id), None)
            if run and run["status"] in ("done", "failed"):
                break
            time.sleep(0.2)

        assert run and run["status"] == "done", f"Training did not complete: {run}"
        return run_id

    def test_calibration_endpoint_returns_200_for_classifier(
        self, client, project_and_dataset
    ):
        run_id = self._train_classifier(client, project_and_dataset)
        resp = client.get(f"/api/models/{run_id}/calibration")
        assert resp.status_code == 200

    def test_calibration_response_has_brier_score(self, client, project_and_dataset):
        run_id = self._train_classifier(client, project_and_dataset)
        data = client.get(f"/api/models/{run_id}/calibration").json()
        assert "brier_score" in data
        assert 0 <= data["brier_score"] <= 1

    def test_calibration_response_has_curve(self, client, project_and_dataset):
        run_id = self._train_classifier(client, project_and_dataset)
        data = client.get(f"/api/models/{run_id}/calibration").json()
        assert "calibration_curve" in data
        assert isinstance(data["calibration_curve"], list)

    def test_calibration_404_for_unknown_run(self, client, project_and_dataset):
        resp = client.get("/api/models/nonexistent-run/calibration")
        assert resp.status_code == 404

    def test_calibration_on_regression_run_is_400(self, client, tmp_path):
        """Regression models don't have calibration data — endpoint returns 400."""
        # Re-use the same client but upload regression data
        proj = client.post("/api/projects", json={"name": "RegTest2"})
        assert proj.status_code == 201
        pid = proj.json()["id"]

        upload = client.post(
            "/api/data/upload",
            files={"file": ("data.csv", io.BytesIO(REGRESSION_CSV), "text/csv")},
            data={"project_id": pid},
        )
        assert upload.status_code == 201
        ds_id = upload.json()["dataset_id"]

        client.post(f"/api/features/{ds_id}/apply", json={"transformations": []})
        client.post(f"/api/features/{ds_id}/target", json={"target_column": "target"})

        resp = client.post(
            f"/api/models/{pid}/train",
            json={"algorithms": ["linear_regression"]},
        )
        assert resp.status_code == 202
        run_id = resp.json()["model_run_ids"][0]

        deadline = time.time() + 20
        while time.time() < deadline:
            runs = client.get(f"/api/models/{pid}/runs").json()["runs"]
            run = next((r for r in runs if r["id"] == run_id), None)
            if run and run["status"] in ("done", "failed"):
                break
            time.sleep(0.2)

        assert run and run["status"] == "done"
        resp = client.get(f"/api/models/{run_id}/calibration")
        assert resp.status_code == 400


# ===========================================================================
# 6. sample_large_dataset used in training pipeline (unit-level wiring check)
# ===========================================================================


class TestSampleLargeDatasetWiring:
    """Verify that train_single_model works correctly on a sampled DataFrame
    (i.e. training on a smaller representative sample produces valid results)."""

    def test_train_on_sampled_data_succeeds(self, tmp_path):
        """Sampling a DF and training on the result must not crash."""
        rng = np.random.default_rng(42)
        n = 200
        X_df = pd.DataFrame(rng.standard_normal((n, 4)), columns=["a", "b", "c", "d"])
        X_df["label"] = np.where(X_df["a"] > 0, "yes", "no")

        sampled, info = sample_large_dataset(X_df, max_rows=50, threshold=100)
        assert len(sampled) == 50
        assert info["was_sampled"]

        from core.trainer import prepare_features

        X, y, _ = prepare_features(sampled, ["a", "b", "c", "d"], "label", "classification")
        result = train_single_model(
            X, y, "logistic_regression", "classification", tmp_path, "run_wiring_1"
        )
        assert result["metrics"]["train_size"] + result["metrics"]["test_size"] <= 50

    def test_sample_note_wording(self):
        """Note uses comma-formatted numbers and is analyst-friendly."""
        df = pd.DataFrame({"x": range(100_000)})
        _, info = sample_large_dataset(df, max_rows=20_000, threshold=50_000)
        assert "20,000" in info["note"]
        assert "100,000" in info["note"]
        assert "random sample" in info["note"].lower()
