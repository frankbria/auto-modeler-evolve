"""Tests for compute_per_class_threshold_analysis() pure function, REST endpoint, and chat regex."""

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.validator import compute_per_class_threshold_analysis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_multiclass_data(n=200, n_classes=3, seed=42):
    """Return (y_true, y_proba) for a multiclass problem."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, n_classes, size=n)
    # Proba: each class gets a higher score when it's the true class
    y_proba = rng.dirichlet(alpha=[0.5] * n_classes, size=n)
    for i, cls in enumerate(y_true):
        y_proba[i, cls] += 0.5
        y_proba[i] /= y_proba[i].sum()
    return y_true.astype(int), y_proba.astype(float)


def _make_binary_data(n=100, positive_ratio=0.3, seed=42):
    """Return (y_true, y_proba) for a binary problem."""
    rng = np.random.default_rng(seed)
    y_true = (rng.random(n) < positive_ratio).astype(int)
    p1 = np.clip(y_true * 0.6 + rng.random(n) * 0.4, 0.01, 0.99)
    y_proba = np.column_stack([1 - p1, p1])
    return y_true, y_proba


# ---------------------------------------------------------------------------
# Pure function: required-keys contract
# ---------------------------------------------------------------------------


class TestPerClassThresholdRequiredKeys:
    def test_top_level_keys(self):
        y, p = _make_multiclass_data()
        result = compute_per_class_threshold_analysis(y, p)
        for key in ["classes", "n_classes", "n_samples", "n_actionable", "summary"]:
            assert key in result, f"Missing key: {key}"

    def test_class_entry_keys(self):
        y, p = _make_multiclass_data()
        result = compute_per_class_threshold_analysis(y, p)
        entry = result["classes"][0]
        for key in [
            "class_name",
            "class_index",
            "n_positive",
            "prevalence",
            "sweep",
            "optimal_threshold",
            "optimal_f1",
            "optimal_precision",
            "optimal_recall",
            "default_f1",
            "f1_gain",
            "direction",
            "recommendation",
            "is_actionable",
        ]:
            assert key in entry, f"Missing class entry key: {key}"

    def test_n_classes_matches_classes_list(self):
        y, p = _make_multiclass_data(n_classes=3)
        result = compute_per_class_threshold_analysis(y, p)
        assert result["n_classes"] == 3
        assert len(result["classes"]) == 3

    def test_n_samples_correct(self):
        y, p = _make_multiclass_data(n=150)
        result = compute_per_class_threshold_analysis(y, p)
        assert result["n_samples"] == 150


# ---------------------------------------------------------------------------
# Pure function: sweep correctness
# ---------------------------------------------------------------------------


class TestPerClassThresholdSweep:
    def test_sweep_has_19_points_per_class(self):
        y, p = _make_multiclass_data()
        result = compute_per_class_threshold_analysis(y, p)
        for cls in result["classes"]:
            assert (
                len(cls["sweep"]) == 19
            ), f"Class {cls['class_name']} sweep has wrong length"

    def test_sweep_thresholds_range(self):
        y, p = _make_multiclass_data()
        result = compute_per_class_threshold_analysis(y, p)
        for cls in result["classes"]:
            thresholds = [pt["threshold"] for pt in cls["sweep"]]
            assert abs(thresholds[0] - 0.05) < 0.001
            assert abs(thresholds[-1] - 0.95) < 0.001

    def test_optimal_threshold_in_range(self):
        y, p = _make_multiclass_data()
        result = compute_per_class_threshold_analysis(y, p)
        for cls in result["classes"]:
            assert 0.05 <= cls["optimal_threshold"] <= 0.95

    def test_direction_values(self):
        y, p = _make_multiclass_data()
        result = compute_per_class_threshold_analysis(y, p)
        valid = {"raise", "lower", "keep"}
        for cls in result["classes"]:
            assert (
                cls["direction"] in valid
            ), f"Unexpected direction: {cls['direction']}"

    def test_direction_logic(self):
        y, p = _make_multiclass_data()
        result = compute_per_class_threshold_analysis(y, p)
        for cls in result["classes"]:
            if cls["optimal_threshold"] > 0.50:
                assert cls["direction"] == "raise"
            elif cls["optimal_threshold"] < 0.50:
                assert cls["direction"] == "lower"
            else:
                assert cls["direction"] == "keep"


# ---------------------------------------------------------------------------
# Pure function: binary case
# ---------------------------------------------------------------------------


class TestPerClassThresholdBinary:
    def test_binary_two_classes(self):
        y, p = _make_binary_data()
        result = compute_per_class_threshold_analysis(y, p)
        assert result["n_classes"] == 2
        assert len(result["classes"]) == 2

    def test_binary_class_indices(self):
        y, p = _make_binary_data()
        result = compute_per_class_threshold_analysis(y, p)
        indices = sorted(c["class_index"] for c in result["classes"])
        assert indices == [0, 1]


# ---------------------------------------------------------------------------
# Pure function: class names
# ---------------------------------------------------------------------------


class TestPerClassThresholdClassNames:
    def test_class_names_used(self):
        y, p = _make_multiclass_data(n_classes=3)
        names = ["low", "medium", "high"]
        result = compute_per_class_threshold_analysis(y, p, class_names=names)
        returned_names = [c["class_name"] for c in result["classes"]]
        assert returned_names == names

    def test_fallback_to_int_string(self):
        y, p = _make_multiclass_data(n_classes=3)
        result = compute_per_class_threshold_analysis(y, p, class_names=None)
        for cls in result["classes"]:
            assert cls["class_name"].isdigit() or cls["class_name"] in ["0", "1", "2"]


# ---------------------------------------------------------------------------
# Pure function: edge cases & validation
# ---------------------------------------------------------------------------


class TestPerClassThresholdEdgeCases:
    def test_raises_on_too_few_samples(self):
        y = np.array([0, 1, 0, 1, 0])
        p = np.column_stack(
            [
                1 - np.array([0.3, 0.7, 0.4, 0.8, 0.2]),
                np.array([0.3, 0.7, 0.4, 0.8, 0.2]),
            ]
        )
        with pytest.raises(ValueError, match="10 samples"):
            compute_per_class_threshold_analysis(y, p)

    def test_raises_on_proba_shape_mismatch(self):
        y, _ = _make_multiclass_data(n=50, n_classes=3)
        bad_proba = np.random.dirichlet([1, 1], size=50)  # 2 cols but 3 classes
        with pytest.raises(ValueError, match="y_proba must be"):
            compute_per_class_threshold_analysis(y, bad_proba)

    def test_f1_gain_non_negative(self):
        y, p = _make_multiclass_data()
        result = compute_per_class_threshold_analysis(y, p)
        for cls in result["classes"]:
            assert (
                cls["f1_gain"] >= 0
            ), f"Negative f1_gain for class {cls['class_name']}"

    def test_prevalence_sums_approximately_one(self):
        y, p = _make_multiclass_data()
        result = compute_per_class_threshold_analysis(y, p)
        total = sum(c["prevalence"] for c in result["classes"])
        assert (
            abs(total - 1.0) < 0.05
        )  # approximately 1 (one-vs-rest, so should be close)

    def test_n_actionable_consistent(self):
        y, p = _make_multiclass_data()
        result = compute_per_class_threshold_analysis(y, p)
        manual_count = sum(1 for c in result["classes"] if c["is_actionable"])
        assert result["n_actionable"] == manual_count


# ---------------------------------------------------------------------------
# Pure function: summary text
# ---------------------------------------------------------------------------


class TestPerClassThresholdSummary:
    def test_summary_non_empty(self):
        y, p = _make_multiclass_data()
        result = compute_per_class_threshold_analysis(y, p)
        assert len(result["summary"]) > 10

    def test_no_actionable_summary_mentions_default(self):
        # Force all optimal thresholds to 0.50 by using uniform probabilities
        n = 100
        y = np.tile([0, 1, 2], n // 3 + 1)[:n]
        p = np.ones((n, 3)) / 3  # uniform → all thresholds near default
        result = compute_per_class_threshold_analysis(y, p)
        if result["n_actionable"] == 0:
            assert (
                "default" in result["summary"].lower()
                or "optimal" in result["summary"].lower()
            )


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def db_engine(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")

    import models.ab_test  # noqa: F401
    import models.batch_schedule  # noqa: F401
    import models.conversation  # noqa: F401
    import models.dataset  # noqa: F401
    import models.deployment  # noqa: F401
    import models.deployment_changelog  # noqa: F401
    import models.deployment_preset  # noqa: F401
    import models.deployment_version  # noqa: F401
    import models.feature_set  # noqa: F401
    import models.feedback_record  # noqa: F401
    import models.model_run  # noqa: F401
    import models.prediction_log  # noqa: F401
    import models.project  # noqa: F401
    import models.webhook_config  # noqa: F401
    import models.webhook_event  # noqa: F401
    import models.weekly_digest_config  # noqa: F401

    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def client(db_engine):
    from main import app

    db_module.engine = db_engine
    import importlib

    import api.validation as val_mod

    importlib.reload(val_mod)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_endpoint_404_unknown_run(client):
    async with client as c:
        resp = await c.get("/api/models/nonexistent-run/per-class-threshold")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_endpoint_400_regression_model(client, tmp_path):
    """Regression model should return 400."""
    from sqlmodel import Session

    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun
    from models.project import Project

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b,target\n1,2,3\n4,5,6\n7,8,9\n")

    with Session(db_module.engine) as sess:
        proj = Project(name="p1")
        sess.add(proj)
        sess.flush()
        ds = Dataset(
            project_id=proj.id,
            filename="data.csv",
            file_path=str(csv_path),
            row_count=3,
            column_count=3,
            columns="{}",
        )
        sess.add(ds)
        sess.flush()
        fs = FeatureSet(
            project_id=proj.id,
            dataset_id=ds.id,
            target_column="target",
            problem_type="regression",
            transformations="[]",
            feature_columns='["a","b"]',
        )
        sess.add(fs)
        sess.flush()
        # Create a placeholder model file so _load_run_context passes the path check
        model_path = tmp_path / "model.joblib"
        model_path.write_bytes(b"placeholder")
        run = ModelRun(
            project_id=proj.id,
            feature_set_id=fs.id,
            algorithm="linear_regression",
            status="done",
            metrics="{}",
            model_path=str(model_path),
            is_selected=True,
        )
        sess.add(run)
        sess.commit()
        run_id = run.id

    async with client as c:
        resp = await c.get(f"/api/models/{run_id}/per-class-threshold")
    assert resp.status_code == 400
    assert "classification" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Chat pattern tests
# ---------------------------------------------------------------------------


class TestPerClassThresholdPatterns:
    @pytest.fixture(autouse=True)
    def load_pattern(self):
        from api.chat import _PER_CLASS_THRESHOLD_PATTERNS

        self.pat = _PER_CLASS_THRESHOLD_PATTERNS

    def _m(self, text: str) -> bool:
        return bool(self.pat.search(text))

    def test_per_class_threshold_tuning(self):
        assert self._m("per-class threshold tuning")

    def test_optimize_threshold_for_each_class(self):
        assert self._m("optimize the threshold for each class")

    def test_class_specific_threshold(self):
        assert self._m("class-specific confidence threshold")

    def test_independent_threshold_per_class(self):
        assert self._m("independent threshold per class")

    def test_multiclass_threshold_analysis(self):
        assert self._m("multiclass threshold analysis")

    def test_different_threshold_for_each_label(self):
        assert self._m("different threshold for each label")

    def test_optimize_independently(self):
        assert self._m("optimize threshold independently for each class")

    def test_what_threshold_per_class(self):
        assert self._m("what threshold should I use per class")

    # False-positive guards
    def test_no_match_generic_threshold(self):
        assert not self._m("what probability threshold should I use")

    def test_no_match_confidence_distribution(self):
        assert not self._m("show me the confidence distribution")

    def test_no_match_calibration(self):
        assert not self._m("how well-calibrated is my model")
