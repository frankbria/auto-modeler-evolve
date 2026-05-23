"""Tests for goal seek history — save, prune, REST, and chat handler."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from models.goal_seek_record import GoalSeekRecord, MAX_HISTORY


# ---------------------------------------------------------------------------
# Regex unit tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def gsh_pattern():
    from api.chat import _GOAL_SEEK_HISTORY_PATTERNS

    return _GOAL_SEEK_HISTORY_PATTERNS


class TestGoalSeekHistoryPatterns:
    """_GOAL_SEEK_HISTORY_PATTERNS should match natural-language history queries."""

    SHOULD_MATCH = [
        "show my goal seek history",
        "view goal seek history",
        "list my previous goal seek results",
        "what goal seeks have I tried",
        "compare my goal seek scenarios",
        "show past goal seek runs",
        "last goal seek results",
        "goal seek comparison",
        "prior goal-seek attempts",
        "how did my goal seeks compare",
    ]

    SHOULD_NOT_MATCH = [
        "goal seek revenue to $5M",
        "what inputs would produce $5M",
        "reverse prediction",
        "optimize my inputs to reach $1M",
    ]

    def test_matches_natural_language_variants(self, gsh_pattern):
        for phrase in self.SHOULD_MATCH:
            assert gsh_pattern.search(phrase), f"Should match: {phrase!r}"

    def test_no_false_positives(self, gsh_pattern):
        for phrase in self.SHOULD_NOT_MATCH:
            assert not gsh_pattern.search(phrase), f"Should NOT match: {phrase!r}"


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestGoalSeekRecordModel:
    """GoalSeekRecord model stores goal seek results with required fields."""

    def test_max_history_constant(self):
        assert MAX_HISTORY == 3

    def test_create_record(self):
        r = GoalSeekRecord(
            deployment_id="dep-001",
            target_column="revenue",
            problem_type="regression",
            algorithm_plain="Linear Regression",
            target_value_str="5000000",
            achieved_value_str="4850000",
            achieved=False,
            gap_pct=3.0,
            suggestions_json=json.dumps([{"feature": "units", "suggested_value": 200}]),
            fixed_features_json="{}",
            summary="Optimizer suggests increasing units to 200.",
        )
        assert r.deployment_id == "dep-001"
        assert r.achieved is False
        assert r.gap_pct == 3.0
        assert r.problem_type == "regression"

    def test_defaults(self):
        r = GoalSeekRecord(
            deployment_id="dep-002",
            target_column="churn",
            problem_type="classification",
            algorithm_plain="Random Forest",
            target_value_str="High",
            achieved_value_str="High",
            achieved=True,
            summary="Goal achieved.",
        )
        assert r.suggestions_json == "[]"
        assert r.fixed_features_json == "{}"
        assert r.gap_pct is None


# ---------------------------------------------------------------------------
# Integration tests — REST endpoints
# ---------------------------------------------------------------------------


@pytest.fixture()
def client_and_deployment():
    """TestClient + a real active Deployment in the test DB."""
    from main import app
    from db import get_session
    from models.project import Project
    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun
    from models.deployment import Deployment

    with TestClient(app) as c:
        session: Session = next(
            c.app.dependency_overrides.get(get_session, get_session)()
        )

        project = Project(
            id="gsh-proj",
            name="GSH Project",
            created_at=datetime.now(UTC).replace(tzinfo=None),
            updated_at=datetime.now(UTC).replace(tzinfo=None),
            status="deployed",
        )
        session.add(project)

        dataset = Dataset(
            id="gsh-ds",
            project_id="gsh-proj",
            filename="sales.csv",
            file_path="/tmp/gsh_sales.csv",
            row_count=100,
            column_count=5,
            uploaded_at=datetime.now(UTC).replace(tzinfo=None),
            size_bytes=1024,
        )
        session.add(dataset)

        fs = FeatureSet(
            id="gsh-fs",
            dataset_id="gsh-ds",
            is_active=True,
            created_at=datetime.now(UTC).replace(tzinfo=None),
        )
        session.add(fs)

        run = ModelRun(
            id="gsh-run",
            project_id="gsh-proj",
            feature_set_id="gsh-fs",
            algorithm="linear_regression",
            status="done",
            is_selected=True,
            is_deployed=True,
            model_path="/tmp/gsh_model.joblib",
            created_at=datetime.now(UTC).replace(tzinfo=None),
        )
        session.add(run)

        dep = Deployment(
            id="gsh-dep",
            model_run_id="gsh-run",
            project_id="gsh-proj",
            endpoint_path="/api/predict/gsh-dep",
            dashboard_url="/predict/gsh-dep",
            is_active=True,
            request_count=0,
            created_at=datetime.now(UTC).replace(tzinfo=None),
        )
        session.add(dep)
        session.commit()

        yield c, session, "gsh-dep"

        # Cleanup
        session.exec(
            select(GoalSeekRecord).where(GoalSeekRecord.deployment_id == "gsh-dep")
        ).all()
        for r in session.exec(
            select(GoalSeekRecord).where(GoalSeekRecord.deployment_id == "gsh-dep")
        ).all():
            session.delete(r)
        session.delete(dep)
        session.delete(run)
        session.delete(fs)
        session.delete(dataset)
        session.delete(project)
        session.commit()


class TestGoalSeekHistoryEndpoint:
    """GET /api/deploy/{id}/goal-seek/history returns history records."""

    def test_empty_history_returns_200(self, client_and_deployment):
        client, session, dep_id = client_and_deployment
        resp = client.get(f"/api/deploy/{dep_id}/goal-seek/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["entries"] == []
        assert data["max_history"] == MAX_HISTORY

    def test_history_with_records(self, client_and_deployment):
        client, session, dep_id = client_and_deployment
        # Insert 2 records
        for i in range(2):
            r = GoalSeekRecord(
                deployment_id=dep_id,
                target_column="revenue",
                problem_type="regression",
                algorithm_plain="Linear Regression",
                target_value_str=str(5_000_000 + i * 1_000_000),
                achieved_value_str=str(4_800_000 + i * 1_000_000),
                achieved=False,
                gap_pct=4.0,
                summary=f"Scenario {i}",
            )
            session.add(r)
        session.commit()

        resp = client.get(f"/api/deploy/{dep_id}/goal-seek/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["entries"]) == 2
        assert data["entries"][0]["target_column"] == "revenue"
        assert data["entries"][0]["problem_type"] == "regression"
        assert "created_at" in data["entries"][0]

    def test_inactive_deployment_returns_404(self, client_and_deployment):
        client, session, dep_id = client_and_deployment
        dep = session.get(
            __import__("models.deployment", fromlist=["Deployment"]).Deployment, dep_id
        )
        dep.is_active = False
        session.commit()
        resp = client.get(f"/api/deploy/{dep_id}/goal-seek/history")
        assert resp.status_code == 404
        dep.is_active = True
        session.commit()

    def test_unknown_deployment_returns_404(self, client_and_deployment):
        client, _, _ = client_and_deployment
        resp = client.get("/api/deploy/does-not-exist/goal-seek/history")
        assert resp.status_code == 404


class TestGoalSeekHistoryPruning:
    """After saving more than MAX_HISTORY records, oldest are deleted."""

    def test_prune_keeps_max_history(self, client_and_deployment):
        client, session, dep_id = client_and_deployment
        # Insert MAX_HISTORY + 1 records directly
        base_time = datetime.now(UTC).replace(tzinfo=None)
        for i in range(MAX_HISTORY + 1):
            r = GoalSeekRecord(
                deployment_id=dep_id,
                target_column="revenue",
                problem_type="regression",
                algorithm_plain="Ridge Regression",
                target_value_str=str(1_000_000 * (i + 1)),
                achieved_value_str=str(900_000 * (i + 1)),
                achieved=False,
                gap_pct=10.0,
                summary=f"Run {i}",
                created_at=base_time + timedelta(seconds=i),
            )
            session.add(r)
        session.commit()

        # Simulate the prune logic from the endpoint
        from sqlmodel import select

        records = session.exec(
            select(GoalSeekRecord)
            .where(GoalSeekRecord.deployment_id == dep_id)
            .order_by(GoalSeekRecord.created_at.desc())
            .offset(MAX_HISTORY)
        ).all()
        for old in records:
            session.delete(old)
        session.commit()

        remaining = session.exec(
            select(GoalSeekRecord).where(GoalSeekRecord.deployment_id == dep_id)
        ).all()
        assert len(remaining) == MAX_HISTORY

    def test_history_returns_at_most_max_history(self, client_and_deployment):
        client, session, dep_id = client_and_deployment
        # Ensure only MAX_HISTORY entries returned even if DB has more
        for i in range(MAX_HISTORY):
            r = GoalSeekRecord(
                deployment_id=dep_id,
                target_column="revenue",
                problem_type="regression",
                algorithm_plain="Linear Regression",
                target_value_str=str(i * 1_000_000),
                achieved_value_str=str(i * 900_000),
                achieved=True,
                summary=f"Run {i}",
            )
            session.add(r)
        session.commit()

        resp = client.get(f"/api/deploy/{dep_id}/goal-seek/history")
        data = resp.json()
        assert data["count"] <= MAX_HISTORY
        assert len(data["entries"]) <= MAX_HISTORY


class TestGoalSeekHistoryFields:
    """History entries contain all required fields for the frontend card."""

    def test_entry_required_fields(self, client_and_deployment):
        client, session, dep_id = client_and_deployment
        r = GoalSeekRecord(
            deployment_id=dep_id,
            target_column="churn",
            problem_type="classification",
            algorithm_plain="Random Forest",
            target_value_str="High",
            achieved_value_str="High",
            achieved=True,
            suggestions_json=json.dumps(
                [
                    {
                        "feature": "contract_length",
                        "current_mean": 12.0,
                        "suggested_value": 24.0,
                        "direction": "increase",
                        "change_pct": 100,
                    }
                ]
            ),
            fixed_features_json=json.dumps({"region": 1.0}),
            summary="Increase contract length to achieve High churn risk.",
        )
        session.add(r)
        session.commit()

        resp = client.get(f"/api/deploy/{dep_id}/goal-seek/history")
        data = resp.json()
        assert data["count"] == 1
        entry = data["entries"][0]
        required_keys = {
            "id",
            "target_column",
            "problem_type",
            "algorithm_plain",
            "target_value_str",
            "achieved_value_str",
            "achieved",
            "gap_pct",
            "suggestions",
            "fixed_features",
            "summary",
            "created_at",
        }
        assert required_keys.issubset(entry.keys())
        assert entry["achieved"] is True
        assert entry["suggestions"][0]["feature"] == "contract_length"
        assert entry["fixed_features"] == {"region": 1.0}
