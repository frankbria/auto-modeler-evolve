"""Tests for compute_sample_size_adequacy() pure function, REST endpoint, and chat regex."""

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import _SAMPLE_SIZE_PATTERNS
from core.analyzer import compute_sample_size_adequacy


# ---------------------------------------------------------------------------
# Pure function: structure
# ---------------------------------------------------------------------------


class TestComputeSampleSizeAdequacyPure:
    def test_returns_required_keys_regression(self):
        result = compute_sample_size_adequacy(
            n_rows=200, n_features=5, problem_type="regression"
        )
        for key in [
            "n_rows",
            "n_features",
            "n_classes",
            "problem_type",
            "recommended_n",
            "shortfall",
            "coverage_pct",
            "ratio",
            "verdict",
            "verdict_label",
            "cv_std",
            "cv_stable",
            "summary",
            "recommendations",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_regression_recommended_n(self):
        # Regression: max(50, 10 * n_features)
        result = compute_sample_size_adequacy(n_rows=100, n_features=8, problem_type="regression")
        assert result["recommended_n"] == max(50, 10 * 8)

    def test_regression_recommended_n_minimum_50(self):
        result = compute_sample_size_adequacy(n_rows=100, n_features=3, problem_type="regression")
        assert result["recommended_n"] == 50

    def test_classification_recommended_n(self):
        # Classification: max(50 * n_classes, 10 * n_features * n_classes)
        result = compute_sample_size_adequacy(
            n_rows=300, n_features=5, problem_type="classification", n_classes=3
        )
        assert result["recommended_n"] == max(50 * 3, 10 * 5 * 3)

    def test_n_classes_none_for_regression(self):
        result = compute_sample_size_adequacy(n_rows=200, n_features=5, problem_type="regression")
        assert result["n_classes"] is None

    def test_n_classes_present_for_classification(self):
        result = compute_sample_size_adequacy(
            n_rows=200, n_features=5, problem_type="classification", n_classes=2
        )
        assert result["n_classes"] == 2

    def test_adequate_verdict(self):
        # n_rows >= recommended_n → adequate
        result = compute_sample_size_adequacy(
            n_rows=500, n_features=5, problem_type="regression"
        )
        assert result["verdict"] == "adequate"

    def test_borderline_verdict(self):
        # 60% <= n_rows < recommended_n → borderline
        result = compute_sample_size_adequacy(
            n_rows=40, n_features=5, problem_type="regression"
        )
        # recommended = max(50, 50) = 50; 40/50 = 0.80 → borderline
        assert result["verdict"] == "borderline"

    def test_insufficient_verdict(self):
        # n_rows < 60% of recommended_n → insufficient
        result = compute_sample_size_adequacy(
            n_rows=10, n_features=5, problem_type="regression"
        )
        # recommended = 50; 10/50 = 0.20 → insufficient
        assert result["verdict"] == "insufficient"

    def test_shortfall_zero_when_adequate(self):
        result = compute_sample_size_adequacy(
            n_rows=500, n_features=5, problem_type="regression"
        )
        assert result["shortfall"] == 0

    def test_shortfall_positive_when_insufficient(self):
        result = compute_sample_size_adequacy(
            n_rows=10, n_features=5, problem_type="regression"
        )
        assert result["shortfall"] > 0
        assert result["shortfall"] == result["recommended_n"] - result["n_rows"]

    def test_ratio_computed_correctly(self):
        result = compute_sample_size_adequacy(
            n_rows=100, n_features=10, problem_type="regression"
        )
        assert abs(result["ratio"] - 10 / 100) < 0.001

    def test_coverage_pct_capped_at_100(self):
        result = compute_sample_size_adequacy(
            n_rows=10000, n_features=5, problem_type="regression"
        )
        assert result["coverage_pct"] <= 100.0

    def test_cv_std_none_gives_cv_stable_none(self):
        result = compute_sample_size_adequacy(
            n_rows=200, n_features=5, problem_type="regression", cv_std=None
        )
        assert result["cv_stable"] is None
        assert result["cv_std"] is None

    def test_cv_std_low_gives_cv_stable_true(self):
        result = compute_sample_size_adequacy(
            n_rows=200, n_features=5, problem_type="regression", cv_std=0.05
        )
        assert result["cv_stable"] is True

    def test_cv_std_high_gives_cv_stable_false(self):
        result = compute_sample_size_adequacy(
            n_rows=200, n_features=5, problem_type="regression", cv_std=0.15
        )
        assert result["cv_stable"] is False

    def test_summary_is_string(self):
        result = compute_sample_size_adequacy(
            n_rows=200, n_features=5, problem_type="regression"
        )
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 10

    def test_recommendations_is_list(self):
        result = compute_sample_size_adequacy(
            n_rows=200, n_features=5, problem_type="regression"
        )
        assert isinstance(result["recommendations"], list)

    def test_insufficient_has_recommendations(self):
        result = compute_sample_size_adequacy(
            n_rows=10, n_features=5, problem_type="regression"
        )
        assert len(result["recommendations"]) > 0

    def test_too_few_rows_raises_value_error(self):
        with pytest.raises(ValueError, match="5 rows"):
            compute_sample_size_adequacy(
                n_rows=2, n_features=5, problem_type="regression"
            )

    def test_zero_features_raises_value_error(self):
        with pytest.raises(ValueError, match="1 feature"):
            compute_sample_size_adequacy(
                n_rows=100, n_features=0, problem_type="regression"
            )


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------


class TestSampleSizePatterns:
    def test_enough_data(self):
        assert _SAMPLE_SIZE_PATTERNS.search("do I have enough data to train?")

    def test_dataset_big_enough(self):
        assert _SAMPLE_SIZE_PATTERNS.search("is my dataset big enough?")

    def test_how_many_more_rows(self):
        assert _SAMPLE_SIZE_PATTERNS.search("how many more rows do I need?")

    def test_dataset_too_small(self):
        assert _SAMPLE_SIZE_PATTERNS.search("is my dataset too small")

    def test_sample_size_check(self):
        assert _SAMPLE_SIZE_PATTERNS.search("sample size check")

    def test_is_data_sufficient(self):
        assert _SAMPLE_SIZE_PATTERNS.search("is my training data sufficient?")

    def test_enough_data_to_train(self):
        assert _SAMPLE_SIZE_PATTERNS.search("do I have enough data to train a model?")

    def test_minimum_required_samples(self):
        assert _SAMPLE_SIZE_PATTERNS.search("minimum required samples for training")

    def test_false_positive_train_model(self):
        assert not _SAMPLE_SIZE_PATTERNS.search("can you train a model for me?")

    def test_false_positive_explain_data(self):
        assert not _SAMPLE_SIZE_PATTERNS.search("explain my data to me")

    def test_data_size_adequacy(self):
        assert _SAMPLE_SIZE_PATTERNS.search("dataset size adequacy")


# ---------------------------------------------------------------------------
# Endpoint integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
async def ac(tmp_path):
    """Async HTTP test client with isolated DB."""
    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.feature_set  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa
    from main import app

    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path
    SQLModel.metadata.create_all(db_module.engine)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.mark.anyio
async def test_sample_size_regression_returns_200(ac, tmp_path):
    """Regression model returns 200 with adequacy verdict."""
    import joblib
    from sklearn.linear_model import LinearRegression
    from sqlmodel import Session

    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun
    from models.project import Project

    n = 60
    rng = np.random.default_rng(42)
    X = rng.random((n, 2))
    y = X[:, 0] * 2 + X[:, 1] + rng.random(n) * 0.1

    csv_rows = "\n".join(f"{x[0]:.3f},{x[1]:.3f},{yi:.3f}" for x, yi in zip(X, y))
    csv_content = f"a,b,target\n{csv_rows}"

    with Session(db_module.engine) as session:
        proj = Project(name="ssa_reg")
        session.add(proj)
        session.commit()
        session.refresh(proj)

        csv_path = tmp_path / "ssa_reg.csv"
        csv_path.write_text(csv_content)
        ds = Dataset(
            project_id=proj.id,
            filename="ssa_reg.csv",
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
            problem_type="regression",
            transformations="[]",
            is_active=True,
        )
        session.add(fset)
        session.commit()
        session.refresh(fset)

        model_path = tmp_path / "ssa_model.joblib"
        model = LinearRegression().fit(X, y)
        joblib.dump(model, model_path)

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

    response = await ac.get(f"/api/models/{run_id}/sample-size-adequacy")
    assert response.status_code == 200
    body = response.json()
    assert "verdict" in body
    assert body["verdict"] in ("adequate", "borderline", "insufficient")
    assert body["n_rows"] == n
    assert body["n_features"] == 2
    assert body["target_col"] == "target"
    assert body["algorithm"] == "linear_regression"
    assert "recommended_n" in body
    assert "shortfall" in body
    assert "summary" in body


@pytest.mark.anyio
async def test_sample_size_unknown_run_returns_404(ac):
    response = await ac.get("/api/models/nonexistent-run-id/sample-size-adequacy")
    assert response.status_code == 404


@pytest.mark.anyio
async def test_sample_size_not_done_run_returns_400(ac, tmp_path):
    """A run that is not 'done' should return 400."""
    from sqlmodel import Session

    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun
    from models.project import Project

    with Session(db_module.engine) as session:
        proj = Project(name="ssa_pending")
        session.add(proj)
        session.commit()
        session.refresh(proj)

        csv_path = tmp_path / "dummy.csv"
        csv_path.write_text("a,b,target\n1,2,3\n")
        ds = Dataset(
            project_id=proj.id,
            filename="dummy.csv",
            file_path=str(csv_path),
            row_count=2,
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
            problem_type="regression",
            transformations="[]",
            is_active=True,
        )
        session.add(fset)
        session.commit()

        run = ModelRun(
            project_id=proj.id,
            feature_set_id=fset.id,
            algorithm="linear_regression",
            status="training",
            metrics="{}",
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id

    response = await ac.get(f"/api/models/{run_id}/sample-size-adequacy")
    assert response.status_code == 400
    assert "not done" in response.json()["detail"].lower()
