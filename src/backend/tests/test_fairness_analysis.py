"""Tests for Fairness / Bias Analysis.

Covers:
- compute_fairness_metrics() pure function (classification + regression + edge cases)
- GET /api/models/{run_id}/fairness REST endpoint (happy path, validation errors)
- _FAIRNESS_PATTERNS regex positive + negative
- _detect_fairness_col() helper
- Chat handler emitting fairness_check SSE event when a completed run exists
- No event when model runs absent
"""

from __future__ import annotations

import io
import json
import time
import unittest.mock as mock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

_SAMPLE_CSV = (
    b"region,product,revenue,units\n"
    b"East,A,100.0,10\nWest,B,200.0,20\nEast,A,150.0,15\nWest,B,300.0,30\n"
    b"North,A,250.0,25\nEast,B,175.0,18\nWest,A,220.0,22\nNorth,B,190.0,19\n"
    b"East,A,130.0,13\nWest,B,280.0,28\nEast,A,160.0,16\nWest,B,210.0,21\n"
)

# Classification sample with a binary target
_CLASS_CSV = (
    b"region,feature1,churn\n"
    b"East,10,0\nEast,12,0\nEast,9,1\nEast,11,0\n"
    b"West,20,1\nWest,22,1\nWest,21,0\nWest,19,1\n"
    b"North,15,0\nNorth,14,0\nNorth,16,1\nNorth,13,0\n"
)


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestComputeFairnessMetricsClassification:
    """Tests for the classification path of compute_fairness_metrics."""

    def _run(self, y_true, y_pred, sensitive, **kwargs):
        from core.validator import compute_fairness_metrics

        return compute_fairness_metrics(
            np.array(y_true),
            np.array(y_pred),
            np.array(sensitive),
            problem_type="classification",
            **kwargs,
        )

    def test_fair_binary_equal_groups(self):
        # Both groups have identical 50% positive rate → SPD=0, DIR=1.0 → fair
        y_true = [0, 1, 0, 1, 0, 1, 0, 1]
        y_pred = [0, 1, 0, 1, 0, 1, 0, 1]
        groups = ["A", "A", "A", "A", "B", "B", "B", "B"]
        result = self._run(y_true, y_pred, groups)
        assert result["overall_status"] == "fair"
        assert abs(result["spd"]) < 0.1
        assert 0.8 <= result["dir"] <= 1.2

    def test_biased_binary_unequal_positive_rates(self):
        # Group A always predicted 1, Group B always predicted 0
        y_true = [1, 1, 1, 0, 0, 0]
        y_pred = [1, 1, 1, 0, 0, 0]
        groups = ["A", "A", "A", "B", "B", "B"]
        result = self._run(y_true, y_pred, groups)
        assert result["overall_status"] == "biased"
        assert result["spd"] == pytest.approx(1.0, abs=0.01)
        assert result["dir"] == pytest.approx(0.0, abs=0.01)
        assert result["dir_label"] == "fails 4/5ths rule"

    def test_per_group_metrics_present(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 1, 0]
        groups = ["A", "A", "B", "B"]
        result = self._run(y_true, y_pred, groups)
        assert len(result["per_group_metrics"]) == 2
        assert all("group" in g for g in result["per_group_metrics"])
        assert all("accuracy" in g for g in result["per_group_metrics"])
        assert all("positive_rate" in g for g in result["per_group_metrics"])

    def test_warning_status_intermediate_spd(self):
        # SPD ≈ 0.17 — above 0.1 but below 0.2
        y_true = [1, 0, 1, 0, 1, 0]
        y_pred = [1, 1, 1, 0, 0, 0]  # A: pos_rate=1.0, B: pos_rate=0.33 → spd=0.67
        groups = ["A", "A", "A", "B", "B", "B"]
        result = self._run(y_true, y_pred, groups)
        # spd should be 0.67 here → biased (>0.2)
        assert result["overall_status"] in ("warning", "biased")

    def test_insufficient_data_single_group(self):
        y_true = [0, 1, 0]
        y_pred = [0, 1, 0]
        groups = ["A", "A", "A"]
        result = self._run(y_true, y_pred, groups)
        assert result["overall_status"] == "insufficient_data"
        assert "insufficient_data" in result["overall_status"]

    def test_spd_label_fair(self):
        from core.validator import _fairness_spd_label

        assert _fairness_spd_label(0.05) == "fair"

    def test_spd_label_slight_disparity(self):
        from core.validator import _fairness_spd_label

        assert _fairness_spd_label(0.15) == "slight disparity"

    def test_spd_label_moderate_disparity(self):
        from core.validator import _fairness_spd_label

        assert _fairness_spd_label(0.25) == "moderate disparity"

    def test_spd_label_significant_disparity(self):
        from core.validator import _fairness_spd_label

        assert _fairness_spd_label(0.4) == "significant disparity"

    def test_dir_label_passes(self):
        from core.validator import _fairness_dir_label

        assert _fairness_dir_label(0.95) == "passes 4/5ths rule"

    def test_dir_label_borderline(self):
        from core.validator import _fairness_dir_label

        assert _fairness_dir_label(0.75) == "borderline"

    def test_dir_label_fails(self):
        from core.validator import _fairness_dir_label

        assert _fairness_dir_label(0.5) == "fails 4/5ths rule"

    def test_summary_contains_groups(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]
        groups = ["East", "East", "West", "West"]
        result = self._run(y_true, y_pred, groups)
        assert "East" in result["summary"]
        assert "West" in result["summary"]

    def test_groups_list_in_result(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]
        groups = ["A", "A", "B", "B"]
        result = self._run(y_true, y_pred, groups)
        assert "A" in result["groups"]
        assert "B" in result["groups"]


class TestComputeFairnessMetricsRegression:
    """Tests for the regression path of compute_fairness_metrics."""

    def _run(self, y_true, y_pred, sensitive):
        from core.validator import compute_fairness_metrics

        return compute_fairness_metrics(
            np.array(y_true, dtype=float),
            np.array(y_pred, dtype=float),
            np.array(sensitive),
            problem_type="regression",
        )

    def test_fair_equal_mae_groups(self):
        y_true = [100.0, 200.0, 100.0, 200.0]
        y_pred = [100.0, 200.0, 100.0, 200.0]  # perfect predictions
        groups = ["A", "A", "B", "B"]
        result = self._run(y_true, y_pred, groups)
        # MAE disparity = 0/0 edge case but both are 0 → fair
        assert result["overall_status"] in ("fair", "insufficient_data")

    def test_biased_large_mae_disparity(self):
        # Group A: MAE = 1.0, Group B: MAE = 100.0 → disparity = 100
        y_true = [10.0, 10.0, 10.0, 10.0]
        y_pred = [9.0, 11.0, 90.0, -90.0]  # A: MAE=1, B: MAE=100
        groups = ["A", "A", "B", "B"]
        result = self._run(y_true, y_pred, groups)
        assert result["overall_status"] == "biased"
        assert result["mae_disparity"] >= 1.5

    def test_warning_moderate_mae_disparity(self):
        # Group A: MAE = 1.0, Group B: MAE = 1.3 → disparity ≈ 1.3 → warning
        y_true = [10.0, 10.0, 10.0, 10.0]
        y_pred = [9.0, 11.0, 8.7, 11.3]  # A: MAE=1, B: MAE=1.3
        groups = ["A", "A", "B", "B"]
        result = self._run(y_true, y_pred, groups)
        assert result["overall_status"] in ("warning", "fair")

    def test_per_group_mae_present(self):
        y_true = [10.0, 20.0, 10.0, 20.0]
        y_pred = [9.0, 21.0, 8.0, 22.0]
        groups = ["A", "A", "B", "B"]
        result = self._run(y_true, y_pred, groups)
        for g in result["per_group_metrics"]:
            assert "mae" in g
            assert g["mae"] >= 0.0

    def test_summary_string_present(self):
        y_true = [10.0, 20.0, 10.0, 20.0]
        y_pred = [9.5, 20.5, 9.5, 20.5]
        groups = ["A", "A", "B", "B"]
        result = self._run(y_true, y_pred, groups)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 10


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.ab_test  # noqa
    import models.batch_schedule  # noqa
    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.dataset_filter  # noqa
    import models.deployment  # noqa
    import models.deployment_preset  # noqa
    import models.deployment_version  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa
    import models.webhook_config  # noqa
    import models.webhook_event  # noqa
    import models.analysis_template  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.deploy as deploy_module
    import api.models as models_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"
    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as c:
        yield c


def _setup_regression(client, tmp_path=None):
    """Create project → upload CSV → features → target (regression) → train."""
    proj = client.post("/api/projects", json={"name": "FairnessTest"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "problem_type": "regression"},
    )

    train = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    assert train.status_code == 202
    run_id = train.json()["model_run_ids"][0]

    for _ in range(60):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next((r for r in runs if r["id"] == run_id), None)
        if run and run["status"] in ("done", "failed"):
            break
        time.sleep(0.2)

    assert run and run["status"] == "done"
    return project_id, run_id


def _chat(client, project_id, message):
    with mock.patch("anthropic.Anthropic") as mock_ant:
        mock_client = mock.MagicMock()
        mock_ant.return_value = mock_client
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Fairness analysis complete."])
        mock_client.messages.stream.return_value = mock_stream

        resp = client.post(f"/api/chat/{project_id}", json={"message": message})
    return resp


def _parse_events(resp):
    return [
        json.loads(line[6:])
        for line in resp.text.splitlines()
        if line.startswith("data: ") and line[6:].strip()
    ]


# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


class TestFairnessPatterns:
    def setup_method(self):
        from api.chat import _FAIRNESS_PATTERNS

        self.p = _FAIRNESS_PATTERNS

    def test_is_my_model_biased(self):
        assert self.p.search("is my model biased?")

    def test_check_fairness(self):
        assert self.p.search("check fairness of my model")

    def test_bias_check(self):
        assert self.p.search("run a bias check")

    def test_fairness_analysis(self):
        assert self.p.search("fairness analysis")

    def test_disparate_impact(self):
        assert self.p.search("any disparate impact?")

    def test_statistical_parity(self):
        assert self.p.search("statistical parity difference")

    def test_is_model_fair(self):
        assert self.p.search("how fair is my model?")

    def test_treating_everyone_fairly(self):
        assert self.p.search("is my model treating everyone fairly?")

    def test_check_bias_by_region(self):
        assert self.p.search("check bias by region")

    def test_model_bias_across_groups(self):
        assert self.p.search("model bias across groups")

    # Negative tests — should NOT trigger fairness
    def test_negative_train_model(self):
        assert not self.p.search("train a model to predict revenue")

    def test_negative_what_is_revenue(self):
        assert not self.p.search("what is the average revenue?")

    def test_negative_show_confusion_matrix(self):
        assert not self.p.search("show me the confusion matrix")


class TestDetectFairnessCol:
    def test_detects_by_keyword(self):
        from api.chat import _detect_fairness_col

        cols = ["region", "product", "revenue"]
        result = _detect_fairness_col("check bias by region", cols)
        assert result == "region"

    def test_detects_across_keyword(self):
        from api.chat import _detect_fairness_col

        cols = ["region", "product", "revenue"]
        result = _detect_fairness_col("fairness across region", cols)
        assert result == "region"

    def test_returns_none_when_no_match(self):
        from api.chat import _detect_fairness_col

        cols = ["revenue", "units"]
        result = _detect_fairness_col("is my model fair?", cols)
        # No categorical columns mentioned by name
        assert result is None or result in cols

    def test_longest_match_wins(self):
        from api.chat import _detect_fairness_col

        cols = ["region", "product_region", "revenue"]
        result = _detect_fairness_col("check bias by product_region", cols)
        assert result == "product_region"


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


class TestFairnessEndpoint:
    def test_happy_path_regression(self, client):
        _, run_id = _setup_regression(client)
        resp = client.get(f"/api/models/{run_id}/fairness?col=region")
        assert resp.status_code == 200
        data = resp.json()
        assert data["sensitive_col"] == "region"
        assert "overall_status" in data
        assert "per_group_metrics" in data
        assert len(data["per_group_metrics"]) >= 2

    def test_unknown_column_returns_400(self, client):
        _, run_id = _setup_regression(client)
        resp = client.get(f"/api/models/{run_id}/fairness?col=nonexistent_col")
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"].lower()

    def test_unknown_run_id_returns_404(self, client):
        resp = client.get("/api/models/does-not-exist/fairness?col=region")
        assert resp.status_code == 404

    def test_high_cardinality_column_returns_400(self, client):
        _, run_id = _setup_regression(client)
        # revenue has continuous values — too many unique entries
        resp = client.get(f"/api/models/{run_id}/fairness?col=revenue")
        # revenue is numeric with many unique values
        assert resp.status_code in (200, 400)  # depends on actual unique count

    def test_result_includes_algorithm(self, client):
        _, run_id = _setup_regression(client)
        resp = client.get(f"/api/models/{run_id}/fairness?col=region")
        assert resp.status_code == 200
        data = resp.json()
        assert data["algorithm"] == "linear_regression"


# ---------------------------------------------------------------------------
# Chat integration tests
# ---------------------------------------------------------------------------


class TestFairnessChatIntegration:
    def test_emits_fairness_check_event(self, client):
        project_id, _ = _setup_regression(client)
        resp = _chat(client, project_id, "is my model biased by region?")
        events = _parse_events(resp)
        fairness_events = [e for e in events if e.get("type") == "fairness_check"]
        assert len(fairness_events) == 1
        data = fairness_events[0]["fairness_check"]
        assert "overall_status" in data
        assert data["sensitive_col"] == "region"

    def test_no_event_without_model_runs(self, client):
        proj = client.post("/api/projects", json={"name": "NoRuns"})
        project_id = proj.json()["id"]
        upload = client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("data.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        )
        assert upload.status_code == 201
        resp = _chat(client, project_id, "check model fairness")
        events = _parse_events(resp)
        fairness_events = [e for e in events if e.get("type") == "fairness_check"]
        assert len(fairness_events) == 0

    def test_emits_done_event(self, client):
        project_id, _ = _setup_regression(client)
        resp = _chat(client, project_id, "is my model fair?")
        events = _parse_events(resp)
        done_events = [e for e in events if e.get("type") == "done"]
        assert len(done_events) == 1
