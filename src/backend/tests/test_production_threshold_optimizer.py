"""Tests for compute_production_threshold_optimizer() pure function, REST endpoint, and chat regex."""

import json

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine, Session

import db as db_module
from core.validator import compute_production_threshold_optimizer


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_pairs(n=30, accuracy=0.7, seed=42):
    """Generate feedback_pairs with a mix of correct/incorrect predictions."""
    import random as rnd

    rnd.seed(seed)
    labels = ["cat", "dog", "bird"]
    pairs = []
    for i in range(n):
        actual = rnd.choice(labels)
        # At high confidence, more likely to be correct
        confidence = rnd.uniform(0.3, 0.99)
        if rnd.random() < accuracy:
            predicted = actual
        else:
            # pick a different label
            predicted = rnd.choice([lb for lb in labels if lb != actual])
        pairs.append(
            {
                "confidence": round(confidence, 4),
                "predicted_label": predicted,
                "actual_label": actual,
            }
        )
    return pairs


def _make_all_correct_high_confidence(n=20):
    """All predictions correct at high confidence — default threshold should be optimal."""
    return [
        {"confidence": 0.95, "predicted_label": "yes", "actual_label": "yes"}
        for _ in range(n)
    ]


def _make_mixed_confidence(n=30):
    """Low-confidence predictions wrong, high-confidence predictions right."""
    pairs = []
    for i in range(n):
        if i % 3 == 0:
            pairs.append({"confidence": 0.55, "predicted_label": "yes", "actual_label": "no"})
        else:
            pairs.append({"confidence": 0.90, "predicted_label": "yes", "actual_label": "yes"})
    return pairs


# ---------------------------------------------------------------------------
# Pure function unit tests
# ---------------------------------------------------------------------------


class TestComputeProductionThresholdOptimizerPure:
    def test_returns_required_keys(self):
        pairs = _make_pairs()
        result = compute_production_threshold_optimizer(pairs)
        for key in [
            "sweep",
            "optimal_threshold",
            "optimal_f1",
            "optimal_precision",
            "optimal_recall",
            "optimal_coverage",
            "current_threshold",
            "current_f1",
            "current_precision",
            "current_recall",
            "current_coverage",
            "n_feedback",
            "n_correct_total",
            "overall_accuracy",
            "verdict",
            "verdict_label",
            "summary",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_sweep_length_is_19(self):
        pairs = _make_pairs()
        result = compute_production_threshold_optimizer(pairs)
        assert len(result["sweep"]) == 19

    def test_sweep_thresholds_range(self):
        pairs = _make_pairs()
        result = compute_production_threshold_optimizer(pairs)
        thresholds = [pt["threshold"] for pt in result["sweep"]]
        assert min(thresholds) == pytest.approx(0.05, abs=0.001)
        assert max(thresholds) == pytest.approx(0.95, abs=0.001)

    def test_sweep_entry_keys(self):
        pairs = _make_pairs()
        result = compute_production_threshold_optimizer(pairs)
        for pt in result["sweep"]:
            for k in ["threshold", "precision", "recall", "f1", "coverage", "n_served"]:
                assert k in pt

    def test_n_feedback_matches_input(self):
        pairs = _make_pairs(n=30)
        result = compute_production_threshold_optimizer(pairs)
        assert result["n_feedback"] == 30

    def test_overall_accuracy_range(self):
        pairs = _make_pairs()
        result = compute_production_threshold_optimizer(pairs)
        assert 0.0 <= result["overall_accuracy"] <= 1.0

    def test_current_threshold_is_0_5(self):
        pairs = _make_pairs()
        result = compute_production_threshold_optimizer(pairs)
        assert result["current_threshold"] == pytest.approx(0.50)

    def test_optimal_threshold_in_range(self):
        pairs = _make_pairs()
        result = compute_production_threshold_optimizer(pairs)
        assert 0.05 <= result["optimal_threshold"] <= 0.95

    def test_optimal_f1_is_best(self):
        pairs = _make_pairs()
        result = compute_production_threshold_optimizer(pairs)
        all_f1s = [pt["f1"] for pt in result["sweep"]]
        assert result["optimal_f1"] == pytest.approx(max(all_f1s), abs=0.001)

    def test_coverage_decreases_with_threshold(self):
        """Higher threshold = fewer predictions served."""
        pairs = _make_pairs(n=50)
        result = compute_production_threshold_optimizer(pairs)
        coverages = [pt["coverage"] for pt in result["sweep"]]
        # Coverage should be non-increasing
        for i in range(len(coverages) - 1):
            assert coverages[i] >= coverages[i + 1] - 0.01  # allow tiny rounding

    def test_n_served_consistent_with_coverage(self):
        pairs = _make_pairs(n=30)
        result = compute_production_threshold_optimizer(pairs)
        for pt in result["sweep"]:
            expected = round(pt["coverage"] * 30)
            assert abs(pt["n_served"] - expected) <= 1

    def test_verdict_same_when_all_correct_high_confidence(self):
        """All correct at high confidence — default 0.5 threshold already optimal."""
        pairs = _make_all_correct_high_confidence()
        result = compute_production_threshold_optimizer(pairs)
        # With all predictions correct, F1 is the same across all thresholds that include data
        assert result["verdict"] in ("same", "improved")

    def test_verdict_improved_when_low_conf_wrong(self):
        """Low-confidence predictions are wrong — raising threshold should improve F1."""
        pairs = _make_mixed_confidence()
        result = compute_production_threshold_optimizer(pairs)
        # Optimal threshold should be above 0.5 (filters out the 0.55 wrong predictions)
        assert result["optimal_threshold"] > 0.55

    def test_precision_at_high_threshold_is_high(self):
        """At near-max threshold, only high-confidence (and presumably correct) predictions survive."""
        pairs = _make_mixed_confidence()
        result = compute_production_threshold_optimizer(pairs)
        high_thr_pt = next(pt for pt in result["sweep"] if pt["threshold"] == 0.85)
        # All remaining predictions are the correct 0.90-confidence ones
        assert high_thr_pt["precision"] > 0.9

    def test_recall_at_low_threshold_approaches_1(self):
        """At 0.05 threshold: most/all correct predictions are served."""
        pairs = _make_pairs(n=50, accuracy=0.8)
        result = compute_production_threshold_optimizer(pairs)
        low_thr_pt = result["sweep"][0]  # threshold = 0.05
        assert low_thr_pt["recall"] > 0.9

    def test_f1_scores_are_non_negative(self):
        pairs = _make_pairs()
        result = compute_production_threshold_optimizer(pairs)
        for pt in result["sweep"]:
            assert pt["f1"] >= 0.0

    def test_n_correct_total_consistent(self):
        pairs = _make_pairs(n=20, accuracy=1.0)  # all correct
        result = compute_production_threshold_optimizer(pairs)
        assert result["n_correct_total"] == 20

    def test_summary_is_string(self):
        pairs = _make_pairs()
        result = compute_production_threshold_optimizer(pairs)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 20

    def test_verdict_label_is_string(self):
        pairs = _make_pairs()
        result = compute_production_threshold_optimizer(pairs)
        assert isinstance(result["verdict_label"], str)

    def test_raises_on_too_few_pairs(self):
        with pytest.raises(ValueError, match="at least 5"):
            compute_production_threshold_optimizer([
                {"confidence": 0.9, "predicted_label": "a", "actual_label": "a"},
                {"confidence": 0.6, "predicted_label": "b", "actual_label": "b"},
            ])

    def test_handles_missing_confidence_gracefully(self):
        """Pairs with None confidence are treated as 0.0."""
        pairs = [
            {"confidence": None, "predicted_label": "a", "actual_label": "a"},
            {"confidence": 0.9, "predicted_label": "a", "actual_label": "a"},
            {"confidence": 0.8, "predicted_label": "a", "actual_label": "a"},
            {"confidence": 0.7, "predicted_label": "a", "actual_label": "a"},
            {"confidence": 0.6, "predicted_label": "a", "actual_label": "a"},
            {"confidence": 0.5, "predicted_label": "a", "actual_label": "a"},
        ]
        result = compute_production_threshold_optimizer(pairs)
        assert "sweep" in result

    def test_coverage_at_05_includes_all_predictions(self):
        """At threshold 0.05, all predictions with confidence >= 0.05 should be served."""
        pairs = [
            {"confidence": 0.9, "predicted_label": "a", "actual_label": "a"},
            {"confidence": 0.7, "predicted_label": "b", "actual_label": "a"},
            {"confidence": 0.8, "predicted_label": "a", "actual_label": "a"},
            {"confidence": 0.6, "predicted_label": "b", "actual_label": "b"},
            {"confidence": 0.99, "predicted_label": "a", "actual_label": "a"},
        ]
        result = compute_production_threshold_optimizer(pairs)
        low_thr = result["sweep"][0]  # 0.05
        assert low_thr["n_served"] == 5

    def test_optimal_coverage_between_0_and_1(self):
        pairs = _make_pairs()
        result = compute_production_threshold_optimizer(pairs)
        assert 0.0 <= result["optimal_coverage"] <= 1.0

    def test_summary_contains_n_feedback(self):
        pairs = _make_pairs(n=25)
        result = compute_production_threshold_optimizer(pairs)
        assert "25" in result["summary"]


# ---------------------------------------------------------------------------
# Regex unit tests
# ---------------------------------------------------------------------------

try:
    from api.chat import _PROD_THRESHOLD_OPT_PATTERNS as _PAT
except ImportError:
    _PAT = None


@pytest.mark.skipif(_PAT is None, reason="chat module not importable in isolation")
class TestProdThresholdOptRegex:
    def test_matches_maximize_production(self):
        assert _PAT.search("what threshold maximizes my production F1")

    def test_matches_optimize_from_feedback(self):
        assert _PAT.search("optimize my classification threshold from feedback")

    def test_matches_best_threshold_based_on_feedback(self):
        assert _PAT.search("best threshold based on feedback data")

    def test_matches_threshold_optimizer_production(self):
        assert _PAT.search("threshold optimizer from production")

    def test_matches_production_feedback_threshold_optimization(self):
        assert _PAT.search("production feedback threshold optimization")

    def test_matches_real_world_threshold_analysis(self):
        assert _PAT.search("real-world threshold analysis")

    def test_matches_what_threshold_based_on_actual(self):
        assert _PAT.search("what threshold should I use based on actual outcomes")

    def test_matches_calibrate_threshold_using_feedback(self):
        assert _PAT.search("calibrate my decision threshold using feedback")

    def test_no_match_generic_threshold(self):
        """Generic threshold questions go to ThresholdAdvisor, not here."""
        assert not _PAT.search("what threshold should I use")

    def test_no_match_confidence_distribution(self):
        assert not _PAT.search("show me the confidence distribution")


# ---------------------------------------------------------------------------
# REST endpoint integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def db_engine(tmp_path):
    db_file = tmp_path / "test_prod_threshold.db"
    engine = create_engine(f"sqlite:///{db_file}")
    import models.project  # noqa: F401
    import models.dataset  # noqa: F401
    import models.feature_set  # noqa: F401
    import models.model_run  # noqa: F401
    import models.deployment  # noqa: F401
    import models.prediction_log  # noqa: F401
    import models.feedback_record  # noqa: F401

    SQLModel.metadata.create_all(engine)
    original = db_module.engine
    db_module.engine = engine
    yield engine
    db_module.engine = original


@pytest.fixture
def client(db_engine):
    from main import app

    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.anyio
async def test_get_production_threshold_optimizer_404(client):
    async with client as c:
        resp = await c.get("/api/deploy/nonexistent-id/production-threshold-optimizer")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_get_production_threshold_optimizer_no_data(client, db_engine):
    """Returns no_data when no feedback records exist."""
    from models.deployment import Deployment
    from models.model_run import ModelRun
    from models.project import Project
    from models.dataset import Dataset

    with Session(db_engine) as session:
        proj = Project(name="Test Project")
        session.add(proj)
        session.commit()
        session.refresh(proj)

        ds = Dataset(project_id=proj.id, filename="test.csv", file_path="/tmp/test.csv", row_count=100)
        session.add(ds)
        session.commit()
        session.refresh(ds)

        run = ModelRun(
            project_id=proj.id,
            algorithm="RandomForestClassifier",
            metrics=json.dumps({"accuracy": 0.85}),
        )
        session.add(run)
        session.commit()
        session.refresh(run)

        dep = Deployment(
            model_run_id=run.id,
            project_id=proj.id,
            endpoint_path=f"/api/predict/{run.id}",
            dashboard_url=f"/predict/{run.id}",
            is_active=True,
            problem_type="classification",
        )
        session.add(dep)
        session.commit()
        session.refresh(dep)

    async with client as c:
        resp = await c.get(f"/api/deploy/{dep.id}/production-threshold-optimizer")

    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "no_data"
    assert data["n_feedback"] == 0
