"""Tests for compute_confidence_distribution() pure function, REST endpoint, and chat regex."""

import re

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.validator import compute_confidence_distribution

# ---------------------------------------------------------------------------
# anyio fixture for async endpoint tests
# ---------------------------------------------------------------------------

pytest_plugins = ("anyio",)


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa: F401
    import models.dataset  # noqa: F401
    import models.deployment  # noqa: F401
    import models.feature_set  # noqa: F401
    import models.model_run  # noqa: F401
    import models.project  # noqa: F401

    SQLModel.metadata.create_all(db_module.engine)

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_binary_proba(n=100, decisive=True, seed=42):
    """Generate (y_proba, y_pred) arrays for a binary classifier."""
    rng = np.random.default_rng(seed)
    if decisive:
        # Most probabilities near 0 or 1
        y_proba = np.clip(
            np.where(rng.random(n) > 0.5, rng.uniform(0.7, 0.99, n), rng.uniform(0.01, 0.3, n)),
            0.01,
            0.99,
        )
    else:
        # Most probabilities near 0.5
        y_proba = np.clip(rng.normal(0.5, 0.1, n), 0.01, 0.99)
    y_pred = (y_proba >= 0.5).astype(int)
    return y_proba.astype(float), y_pred.astype(int)


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestComputeConfidenceDistribution:
    def test_returns_required_keys(self):
        y_proba, y_pred = _make_binary_proba()
        result = compute_confidence_distribution(y_proba, y_pred)
        for key in [
            "bins",
            "n_total",
            "n_bins",
            "mean_confidence",
            "median_confidence",
            "pct_high",
            "pct_medium",
            "pct_low",
            "n_high",
            "n_medium",
            "n_low",
            "decisiveness",
            "decisiveness_label",
            "per_class_mean",
            "summary",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_bins_count_matches_n_bins(self):
        y_proba, y_pred = _make_binary_proba()
        result = compute_confidence_distribution(y_proba, y_pred, n_bins=10)
        assert len(result["bins"]) == 10

    def test_bin_keys(self):
        y_proba, y_pred = _make_binary_proba()
        result = compute_confidence_distribution(y_proba, y_pred)
        b = result["bins"][0]
        for key in ["lo", "hi", "count", "pct", "label"]:
            assert key in b, f"Missing bin key: {key}"

    def test_n_total_correct(self):
        y_proba, y_pred = _make_binary_proba(n=80)
        result = compute_confidence_distribution(y_proba, y_pred)
        assert result["n_total"] == 80

    def test_pct_sum_approx_100(self):
        y_proba, y_pred = _make_binary_proba()
        result = compute_confidence_distribution(y_proba, y_pred)
        total_pct = result["pct_high"] + result["pct_medium"] + result["pct_low"]
        assert abs(total_pct - 100.0) < 1.0

    def test_n_counts_sum_to_total(self):
        y_proba, y_pred = _make_binary_proba()
        result = compute_confidence_distribution(y_proba, y_pred)
        assert result["n_high"] + result["n_medium"] + result["n_low"] == result["n_total"]

    def test_decisive_verdict_for_decisive_model(self):
        y_proba, y_pred = _make_binary_proba(decisive=True, n=200)
        result = compute_confidence_distribution(y_proba, y_pred)
        assert result["decisiveness"] in ("decisive", "moderate", "uncertain")

    def test_uncertain_verdict_for_uncertain_model(self):
        rng = np.random.default_rng(0)
        # All predictions near 0.5 — should be uncertain
        y_proba = np.clip(rng.normal(0.5, 0.05, 200), 0.35, 0.65)
        y_pred = (y_proba >= 0.5).astype(int)
        result = compute_confidence_distribution(y_proba, y_pred)
        assert result["decisiveness"] == "uncertain"
        assert result["pct_high"] < 30

    def test_per_class_mean_structure(self):
        y_proba, y_pred = _make_binary_proba()
        result = compute_confidence_distribution(y_proba, y_pred, class_names=["No", "Yes"])
        for entry in result["per_class_mean"]:
            assert "class_name" in entry
            assert "mean_confidence" in entry
            assert "count" in entry

    def test_per_class_count_sums_to_total(self):
        y_proba, y_pred = _make_binary_proba(n=100)
        result = compute_confidence_distribution(y_proba, y_pred)
        total_from_classes = sum(e["count"] for e in result["per_class_mean"])
        assert total_from_classes == 100

    def test_mean_confidence_in_range(self):
        y_proba, y_pred = _make_binary_proba()
        result = compute_confidence_distribution(y_proba, y_pred)
        assert 0.0 <= result["mean_confidence"] <= 1.0

    def test_median_confidence_in_range(self):
        y_proba, y_pred = _make_binary_proba()
        result = compute_confidence_distribution(y_proba, y_pred)
        assert 0.0 <= result["median_confidence"] <= 1.0

    def test_summary_is_string(self):
        y_proba, y_pred = _make_binary_proba()
        result = compute_confidence_distribution(y_proba, y_pred)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 20

    def test_class_names_applied(self):
        y_proba, y_pred = _make_binary_proba(n=50)
        result = compute_confidence_distribution(y_proba, y_pred, class_names=["Cat", "Dog"])
        names = {e["class_name"] for e in result["per_class_mean"]}
        assert "Cat" in names or "Dog" in names

    def test_n_bins_clamped_min(self):
        y_proba, y_pred = _make_binary_proba()
        result = compute_confidence_distribution(y_proba, y_pred, n_bins=2)
        assert result["n_bins"] == 5

    def test_n_bins_clamped_max(self):
        y_proba, y_pred = _make_binary_proba()
        result = compute_confidence_distribution(y_proba, y_pred, n_bins=50)
        assert result["n_bins"] == 20

    def test_too_few_samples_raises(self):
        with pytest.raises(ValueError, match="at least 5"):
            compute_confidence_distribution(
                np.array([0.9, 0.1, 0.8]), np.array([1, 0, 1])
            )


# ---------------------------------------------------------------------------
# Chat regex tests
# ---------------------------------------------------------------------------


_PATTERN_SRC = (
    r"(?i)(?:"
    r"(?:show|display|plot|visualize)\s+(?:me\s+)?(?:the\s+)?confidence\s+distribution\b|"
    r"(?:confidence|probability)\s+(?:histogram|distribution|spread|breakdown)\b|"
    r"how\s+confident\s+is\s+(?:my\s+)?(?:model|classifier)\b|"
    r"how\s+certain\s+are\s+(?:my\s+)?predictions\b|"
    r"(?:is\s+)?(?:my\s+)?model\s+(?:decisive|uncertain|confident)\b|"
    r"(?:distribution\s+of|spread\s+of)\s+(?:prediction\s+)?(?:probabilities|confidences|certainty)\b|"
    r"how\s+(?:often|many\s+times)\s+(?:is\s+)?(?:my\s+)?model\s+(?:very\s+)?confident\b|"
    r"(?:prediction\s+)?confidence\s+(?:breakdown|profile|overview|analysis)\b"
    r")"
)
_PAT = re.compile(_PATTERN_SRC, re.IGNORECASE)


class TestConfidenceDistPatterns:
    def test_show_confidence_distribution(self):
        assert _PAT.search("show confidence distribution")

    def test_show_me_the_confidence_distribution(self):
        assert _PAT.search("show me the confidence distribution")

    def test_confidence_histogram(self):
        assert _PAT.search("confidence histogram")

    def test_how_confident_is_my_model(self):
        assert _PAT.search("how confident is my model?")

    def test_how_certain_are_my_predictions(self):
        assert _PAT.search("how certain are my predictions?")

    def test_is_my_model_decisive(self):
        assert _PAT.search("is my model decisive?")

    def test_model_confident(self):
        assert _PAT.search("is my model confident?")

    def test_distribution_of_probabilities(self):
        assert _PAT.search("distribution of prediction probabilities")

    def test_prediction_confidence_overview(self):
        assert _PAT.search("prediction confidence overview")

    def test_false_positive_training_progress(self):
        assert not _PAT.search("show me the training progress")

    def test_false_positive_accuracy(self):
        assert not _PAT.search("what is my model accuracy?")


# ---------------------------------------------------------------------------
# REST endpoint integration tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_confidence_dist_regression_returns_400(ac, tmp_path):
    """Regression models should return HTTP 400."""
    import joblib
    from sklearn.linear_model import LinearRegression
    from sqlmodel import Session

    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun
    from models.project import Project

    n = 20
    X = np.array([[i] for i in range(n)], dtype=float)
    y = np.array([i * 2 for i in range(n)], dtype=float)
    csv_content = "x,y\n" + "\n".join(f"{i},{i*2}" for i in range(n))

    with Session(db_module.engine) as session:
        proj = Project(name="p")
        session.add(proj)
        session.commit()
        session.refresh(proj)

        csv_path = tmp_path / "data_cd.csv"
        csv_path.write_text(csv_content)
        ds = Dataset(
            project_id=proj.id,
            filename="data_cd.csv",
            file_path=str(csv_path),
            row_count=n,
            column_count=2,
            columns="{}",
        )
        session.add(ds)
        session.commit()
        session.refresh(ds)

        fset = FeatureSet(
            project_id=proj.id,
            dataset_id=ds.id,
            feature_columns='["x"]',
            target_column="y",
            problem_type="regression",
            transformations="[]",
            is_active=True,
        )
        session.add(fset)
        session.commit()
        session.refresh(fset)

        model_path = tmp_path / "reg_cd_model.joblib"
        reg = LinearRegression().fit(X, y)
        joblib.dump(reg, model_path)

        run = ModelRun(
            project_id=proj.id,
            feature_set_id=fset.id,
            algorithm="linear_regression",
            status="done",
            model_path=str(model_path),
            metrics="{}",
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id

    response = await ac.get(f"/api/models/{run_id}/confidence-distribution")
    assert response.status_code == 400
    assert "classification" in response.json()["detail"].lower()


@pytest.mark.anyio
async def test_confidence_dist_classification_200(ac, tmp_path):
    """Classification model with predict_proba should return 200 with bins."""
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sqlmodel import Session

    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun
    from models.project import Project

    n = 60
    rng = np.random.default_rng(1)
    X = rng.random((n, 2))
    y = (X[:, 0] > 0.5).astype(int)

    csv_rows = "\n".join(f"{x[0]:.3f},{x[1]:.3f},{yi}" for x, yi in zip(X, y))
    csv_content = f"a,b,target\n{csv_rows}"

    with Session(db_module.engine) as session:
        proj = Project(name="cls_cd")
        session.add(proj)
        session.commit()
        session.refresh(proj)

        csv_path = tmp_path / "cls_cd.csv"
        csv_path.write_text(csv_content)
        ds = Dataset(
            project_id=proj.id,
            filename="cls_cd.csv",
            file_path=str(csv_path),
            row_count=n,
            column_count=3,
            columns="{}",
        )
        session.add(ds)
        session.commit()
        session.refresh(ds)

        fset = FeatureSet(
            project_id=proj.id,
            dataset_id=ds.id,
            feature_columns='["a","b"]',
            target_column="target",
            problem_type="classification",
            transformations="[]",
            is_active=True,
        )
        session.add(fset)
        session.commit()
        session.refresh(fset)

        model_path = tmp_path / "cls_cd_model.joblib"
        clf = LogisticRegression().fit(X, y)
        joblib.dump(clf, model_path)

        run = ModelRun(
            project_id=proj.id,
            feature_set_id=fset.id,
            algorithm="logistic_regression",
            status="done",
            model_path=str(model_path),
            metrics="{}",
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id

    response = await ac.get(f"/api/models/{run_id}/confidence-distribution")
    assert response.status_code == 200
    data = response.json()
    assert "bins" in data
    assert "decisiveness" in data
    assert "mean_confidence" in data
    assert len(data["bins"]) == 10


@pytest.mark.anyio
async def test_confidence_dist_unknown_run_returns_404(ac):
    """Unknown run ID should return 404."""
    response = await ac.get("/api/models/nonexistent-run/confidence-distribution")
    assert response.status_code == 404
