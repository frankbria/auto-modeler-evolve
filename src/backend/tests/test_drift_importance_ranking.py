"""Tests for Drift × Importance Ranking.

Covers:
- compute_drift_importance_ranking() pure function (unit tests)
- _DRIFT_IMPORTANCE_PATTERNS regex (positive + negative)
- GET /api/deploy/{id}/drift-importance-ranking REST endpoint
- Chat handler emitting drift_importance_ranking SSE event
- No event without a deployment
"""

from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, SQLModel, create_engine

import db as db_module
from core.analyzer import compute_drift_importance_ranking

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
    b"East,175.2,18\nWest,220.4,22\nNorth,190.6,19\nEast,130.8,13\nWest,280.0,28\n"
)


def _importances(names: list[str], values: list[float]) -> list[dict]:
    total = sum(values)
    normed = [v / total for v in values]
    sorted_pairs = sorted(zip(normed, names), reverse=True)
    ranks = {n: i + 1 for i, (_, n) in enumerate(sorted_pairs)}
    return [
        {
            "name": n,
            "importance": round(v / total, 6),
            "rank": ranks[n],
            "is_weak": (v / total) < (sorted_pairs[-1][0] * 1.2),
        }
        for n, v in zip(names, values)
    ]


def _ranges():
    return {
        "revenue": {"min": 100.0, "max": 300.0},
        "units": {"min": 10, "max": 30},
        "region": {"known_categories": ["East", "West", "North"]},
    }


# ---------------------------------------------------------------------------
# Unit tests for compute_drift_importance_ranking()
# ---------------------------------------------------------------------------


class TestComputeDriftImportanceRanking:
    def test_empty_feature_importances_returns_no_importances(self):
        result = compute_drift_importance_ranking([], {}, [])
        assert result["has_importances"] is False
        assert result["verdict"] == "no_importances"
        assert result["ranked_features"] == []

    def test_none_importances_returns_no_importances(self):
        fi = [{"name": "a", "importance": None, "rank": 1, "is_weak": False}]
        result = compute_drift_importance_ranking([], {}, fi)
        assert result["has_importances"] is False

    def test_clear_verdict_when_no_drift(self):
        inputs = [
            {"revenue": 150.0, "units": 15, "region": "East"},
            {"revenue": 200.0, "units": 20, "region": "West"},
        ]
        fi = _importances(["revenue", "units", "region"], [0.6, 0.3, 0.1])
        result = compute_drift_importance_ranking(inputs, _ranges(), fi)
        assert result["has_importances"] is True
        assert result["verdict"] == "clear"
        assert result["critical_count"] == 0
        assert result["high_count"] == 0
        assert result["no_drift_count"] == len(result["ranked_features"])

    def test_critical_verdict_high_drift_important_feature(self):
        # revenue is most important AND >20% OOR
        inputs = [{"revenue": float(v)} for v in [500, 600, 150, 150, 150]]
        ranges = {"revenue": {"min": 100.0, "max": 300.0}}
        fi = _importances(["revenue"], [1.0])
        result = compute_drift_importance_ranking(inputs, ranges, fi)
        assert result["verdict"] == "action_required"
        assert result["critical_count"] >= 1

    def test_high_verdict_moderate_drift(self):
        # 20% OOR but feature is below median importance → "high" not "critical"
        inputs = [
            {"revenue": 500.0, "units": 20},
            {"revenue": 150.0, "units": 15},
            {"revenue": 150.0, "units": 15},
            {"revenue": 150.0, "units": 15},
            {"revenue": 150.0, "units": 15},
        ]
        ranges = {
            "revenue": {"min": 100.0, "max": 300.0},
            "units": {"min": 10, "max": 30},
        }
        fi = _importances(["revenue", "units"], [0.1, 0.9])
        result = compute_drift_importance_ranking(inputs, ranges, fi)
        assert result["verdict"] in ("attention", "monitoring", "action_required")

    def test_ranking_sorted_by_risk_score_descending(self):
        # revenue: 40% OOR, importance 0.6  → risk_score = 0.4 * 0.6
        # units: 0% OOR, importance 0.3     → risk_score = 0
        inputs = [{"revenue": float(v), "units": 15} for v in [500, 600, 150, 150, 150]]
        ranges = {
            "revenue": {"min": 100.0, "max": 300.0},
            "units": {"min": 10, "max": 30},
        }
        fi = _importances(["revenue", "units"], [0.6, 0.3])
        result = compute_drift_importance_ranking(inputs, ranges, fi)
        ranked = result["ranked_features"]
        assert len(ranked) == 2
        assert ranked[0]["name"] == "revenue"
        assert ranked[0]["risk_score"] >= ranked[1]["risk_score"]

    def test_categorical_drift_detected(self):
        inputs = [{"region": v} for v in ["East", "West", "UNKNOWN", "ALIEN", "East"]]
        ranges = {"region": {"known_categories": ["East", "West", "North"]}}
        fi = _importances(["region"], [1.0])
        result = compute_drift_importance_ranking(inputs, ranges, fi)
        ranked = result["ranked_features"]
        assert len(ranked) == 1
        assert ranked[0]["feature_type"] == "categorical"
        assert ranked[0]["drift_pct"] == 40.0

    def test_numeric_drift_pct_is_correct(self):
        # 2 out of 5 values OOR: 500>300, 600>300 = 40%
        inputs = [{"x": float(v)} for v in [500, 600, 150, 200.0, 250.0]]
        ranges = {"x": {"min": 100.0, "max": 300.0}}
        fi = _importances(["x"], [1.0])
        result = compute_drift_importance_ranking(inputs, ranges, fi)
        ranked = result["ranked_features"]
        assert ranked[0]["drift_pct"] == 40.0
        assert ranked[0]["feature_type"] == "numeric"

    def test_max_features_cap(self):
        names = [f"feat_{i}" for i in range(20)]
        inputs = [{n: 1.0 for n in names}]
        ranges = {n: {"min": 0.0, "max": 2.0} for n in names}
        fi = _importances(names, [1.0] * 20)
        result = compute_drift_importance_ranking(inputs, ranges, fi, max_features=10)
        assert result["n_features"] == 10

    def test_empty_inputs_returns_no_drift(self):
        fi = _importances(["revenue", "units"], [0.7, 0.3])
        result = compute_drift_importance_ranking([], _ranges(), fi)
        assert result["has_importances"] is True
        # No inputs → no drift
        for r in result["ranked_features"]:
            assert r["priority"] == "no_drift"

    def test_result_contains_required_keys(self):
        fi = _importances(["revenue"], [1.0])
        result = compute_drift_importance_ranking([], {}, fi)
        required = {
            "has_importances",
            "ranked_features",
            "n_features",
            "n_samples",
            "critical_count",
            "high_count",
            "medium_count",
            "low_count",
            "no_drift_count",
            "verdict",
            "summary",
        }
        assert required.issubset(result.keys())

    def test_feature_row_contains_required_keys(self):
        inputs = [{"revenue": 150.0}]
        ranges = {"revenue": {"min": 100.0, "max": 300.0}}
        fi = _importances(["revenue"], [1.0])
        result = compute_drift_importance_ranking(inputs, ranges, fi)
        row = result["ranked_features"][0]
        for key in [
            "name",
            "importance_pct",
            "rank",
            "drift_pct",
            "risk_score",
            "priority",
            "feature_type",
            "drift_details",
        ]:
            assert key in row, f"Missing key: {key}"

    def test_summary_is_non_empty_string(self):
        fi = _importances(["revenue"], [1.0])
        result = compute_drift_importance_ranking([], {}, fi)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_missing_feature_ranges_gives_unknown_type(self):
        inputs = [{"revenue": 150.0}]
        fi = _importances(["revenue"], [1.0])
        result = compute_drift_importance_ranking(inputs, {}, fi)
        # Feature is present in inputs but no ranges → feature_type unknown or numeric
        row = result["ranked_features"][0]
        assert row["drift_pct"] == 0.0

    def test_priority_no_drift_for_in_range(self):
        inputs = [{"revenue": 150.0}]
        ranges = {"revenue": {"min": 100.0, "max": 300.0}}
        fi = _importances(["revenue"], [1.0])
        result = compute_drift_importance_ranking(inputs, ranges, fi)
        assert result["ranked_features"][0]["priority"] == "no_drift"

    def test_priority_medium_for_small_drift(self):
        # 4% OOR → medium (≥2%)
        inputs = [{"revenue": float(v)} for v in [500] + [150.0] * 24]
        ranges = {"revenue": {"min": 100.0, "max": 300.0}}
        fi = _importances(["revenue"], [1.0])
        result = compute_drift_importance_ranking(inputs, ranges, fi)
        assert result["ranked_features"][0]["priority"] in ("medium", "low")


# ---------------------------------------------------------------------------
# Regex tests
# ---------------------------------------------------------------------------


class TestDriftImportancePatterns:
    def _pattern(self):
        from api.chat import _DRIFT_IMPORTANCE_PATTERNS

        return _DRIFT_IMPORTANCE_PATTERNS

    @pytest.mark.parametrize(
        "msg",
        [
            "which features drifted the most",
            "what input features drifted the most",
            "drift ranking",
            "drift ranked",
            "drift by importance",
            "rank my drifted features",
            "prioritize drifted features",
            "important drifted features",
            "most important drifting features",
            "which drift matters most",
            "what drift affects my model",
            "feature drift risk",
            "feature drift ranking",
            "feature drift priority",
            "drift vs importance",
            "drift versus importance",
            "highest risk drift features",
            "top risk drifted features",
        ],
    )
    def test_positive_matches(self, msg):
        assert self._pattern().search(msg), f"Expected match for: {msg!r}"

    @pytest.mark.parametrize(
        "msg",
        [
            "check input drift",
            "covariate drift",
            "show drift alert",
            "drift warning",
            "are my inputs drifting",
            "show feature importance",
            "train my model",
            "deploy the model",
        ],
    )
    def test_negative_no_match(self, msg):
        assert not self._pattern().search(msg), f"Expected no match for: {msg!r}"


# ---------------------------------------------------------------------------
# REST endpoint tests
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

    SQLModel.metadata.create_all(db_module.engine)

    from main import app

    return TestClient(app)


def _setup_project_with_deployment(client, tmp_path):
    from models.deployment import Deployment
    from models.model_run import ModelRun

    proj = client.post("/api/projects", json={"name": "DriftImportTest"}).json()
    project_id = proj["id"]

    run_id = str(uuid.uuid4())
    dep_id = str(uuid.uuid4())

    with Session(db_module.engine) as sess:
        sess.add(
            ModelRun(
                id=run_id,
                project_id=project_id,
                feature_set_id=str(uuid.uuid4()),
                algorithm="random_forest",
                status="done",
                metrics=json.dumps({"accuracy": 0.87}),
                summary="RF: acc 0.87",
            )
        )
        sess.add(
            Deployment(
                id=dep_id,
                model_run_id=run_id,
                project_id=project_id,
                endpoint_path=f"/api/predict/{dep_id}",
                dashboard_url=f"/predict/{dep_id}",
                is_active=True,
                algorithm="random_forest",
                problem_type="classification",
                feature_names=json.dumps(["revenue", "units", "region"]),
                target_column="churn",
                metrics=json.dumps({"accuracy": 0.87}),
            )
        )
        sess.commit()

    return project_id, run_id, dep_id


class TestDriftImportanceRankingEndpoint:
    def test_404_for_missing_deployment(self, client):
        resp = client.get("/api/deploy/nonexistent-id/drift-importance-ranking")
        assert resp.status_code == 404

    def test_no_predictions_returns_result(self, client, tmp_path):
        _, _, dep_id = _setup_project_with_deployment(client, tmp_path)
        resp = client.get(f"/api/deploy/{dep_id}/drift-importance-ranking")
        assert resp.status_code == 200
        data = resp.json()
        assert "verdict" in data
        assert "summary" in data
        assert "deployment_id" in data
        assert data["deployment_id"] == dep_id

    def test_response_contains_required_fields(self, client, tmp_path):
        _, _, dep_id = _setup_project_with_deployment(client, tmp_path)
        resp = client.get(f"/api/deploy/{dep_id}/drift-importance-ranking")
        assert resp.status_code == 200
        data = resp.json()
        for key in [
            "has_importances",
            "ranked_features",
            "n_features",
            "n_samples",
            "critical_count",
            "high_count",
            "medium_count",
            "low_count",
            "no_drift_count",
            "verdict",
            "summary",
            "deployment_id",
        ]:
            assert key in data, f"Missing key: {key}"

    def test_no_model_path_returns_no_importances(self, client, tmp_path):
        _, _, dep_id = _setup_project_with_deployment(client, tmp_path)
        resp = client.get(f"/api/deploy/{dep_id}/drift-importance-ranking")
        assert resp.status_code == 200
        data = resp.json()
        # No real model file → no importances
        assert data["has_importances"] is False


# ---------------------------------------------------------------------------
# Chat handler tests
# ---------------------------------------------------------------------------


class TestDriftImportanceChatHandler:
    def _mock_anthropic(self, text="Drift ranked."):
        from unittest import mock

        m = mock.patch("anthropic.Anthropic")
        patcher = m.start()
        mock_cli = mock.MagicMock()
        patcher.return_value = mock_cli
        mock_stream = mock.MagicMock()
        mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = mock.MagicMock(return_value=False)
        mock_stream.text_stream = iter([text])
        mock_cli.messages.stream.return_value = mock_stream
        return m

    def test_no_event_without_deployment(self, client):
        from unittest import mock

        proj = client.post("/api/projects", json={"name": "NoDeploy"}).json()
        project_id = proj["id"]

        with mock.patch("anthropic.Anthropic") as m:
            mock_cli = mock.MagicMock()
            m.return_value = mock_cli
            mock_stream = mock.MagicMock()
            mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = mock.MagicMock(return_value=False)
            mock_stream.text_stream = iter(["No deployment."])
            mock_cli.messages.stream.return_value = mock_stream

            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "which features drifted the most"},
            )

        assert resp.status_code == 200
        assert "drift_importance_ranking" not in resp.text

    def test_emits_event_with_deployment(self, client, tmp_path):
        from unittest import mock

        project_id, _, _ = _setup_project_with_deployment(client, tmp_path)

        with mock.patch("anthropic.Anthropic") as m:
            mock_cli = mock.MagicMock()
            m.return_value = mock_cli
            mock_stream = mock.MagicMock()
            mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = mock.MagicMock(return_value=False)
            mock_stream.text_stream = iter(["Drift ranked."])
            mock_cli.messages.stream.return_value = mock_stream

            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "which features drifted the most"},
            )

        assert resp.status_code == 200
        events = [
            json.loads(line[6:])
            for line in resp.text.splitlines()
            if line.startswith("data: ") and line[6:].strip()
        ]
        types = [e.get("type") for e in events]
        assert "drift_importance_ranking" in types

    def test_event_structure(self, client, tmp_path):
        from unittest import mock

        project_id, _, _ = _setup_project_with_deployment(client, tmp_path)

        with mock.patch("anthropic.Anthropic") as m:
            mock_cli = mock.MagicMock()
            m.return_value = mock_cli
            mock_stream = mock.MagicMock()
            mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = mock.MagicMock(return_value=False)
            mock_stream.text_stream = iter(["Drift ranked."])
            mock_cli.messages.stream.return_value = mock_stream

            resp = client.post(
                f"/api/chat/{project_id}",
                json={"message": "drift ranking by importance"},
            )

        assert resp.status_code == 200
        events = [
            json.loads(line[6:])
            for line in resp.text.splitlines()
            if line.startswith("data: ")
            and line[6:].strip()
            and '"drift_importance_ranking"' in line
        ]
        assert len(events) >= 1
        ev = events[0]["drift_importance_ranking"]
        for key in ["verdict", "summary", "has_importances", "ranked_features"]:
            assert key in ev, f"Missing key: {key}"
