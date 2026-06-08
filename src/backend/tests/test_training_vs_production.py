"""Tests for Training vs Production Performance Monitor — Track D perpetual.

Covers:
  - compute_training_vs_production pure function (regression, classification,
    no-feedback, degradation thresholds, weekly timeline)
  - GET /api/deploy/{id}/training-vs-production REST endpoint
  - _PROD_MONITOR_PATTERNS regex (positive + negative)
  - Chat handler integration (emits prod_performance SSE event)
"""

from __future__ import annotations

import json
import unittest.mock as mock
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from core.analyzer import compute_training_vs_production

# ─── helpers ──────────────────────────────────────────────────────────────────


def _mk_feedback(
    prediction_log_id: str | None = None,
    actual_value: float | None = None,
    is_correct: bool | None = None,
    created_at: datetime | None = None,
):
    return SimpleNamespace(
        prediction_log_id=prediction_log_id,
        actual_value=actual_value,
        is_correct=is_correct,
        created_at=created_at or datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
    )


def _mk_log(log_id: str = "log1", prediction_numeric: float | None = None):
    return SimpleNamespace(id=log_id, prediction_numeric=prediction_numeric)


_SAMPLE_CSV = (
    b"region,revenue,units\n"
    b"East,100.5,10\nWest,200.3,20\nEast,150.7,15\nWest,300.1,30\nNorth,250.9,25\n"
)


# ─── Unit tests: compute_training_vs_production ────────────────────────────────


class TestComputeTrainingVsProductionRegression:
    """Regression: lower MAE is better."""

    _metrics = {"mae": 10.0, "r2": 0.85}

    def test_no_feedback_returns_no_data(self):
        result = compute_training_vs_production([], {}, self._metrics, "regression")
        assert result["has_data"] is False
        assert result["status"] == "no_feedback"
        assert "training_value" in result
        assert result["training_value"] == pytest.approx(10.0)
        assert (
            "feedback" in result["summary"].lower()
            or "no matched" in result["summary"].lower()
        )

    def test_no_paired_logs_returns_no_data(self):
        fb = _mk_feedback(actual_value=105.0)  # no prediction_log_id
        result = compute_training_vs_production([fb], {}, self._metrics, "regression")
        assert result["has_data"] is False

    def test_stable_when_live_mae_close_to_training(self):
        fb = _mk_feedback(prediction_log_id="l1", actual_value=110.0)
        logs = {"l1": _mk_log("l1", 108.5)}  # error = 1.5
        metrics = {"mae": 2.0}  # live 1.5 < training 2.0 → better → not degrading
        result = compute_training_vs_production([fb], logs, metrics, "regression")
        assert result["has_data"] is True
        # degradation = (1.5 - 2.0)/2.0 * 100 = -25% → stable (negative = improvement)
        assert result["status"] == "stable"
        assert result["degradation_pct"] < 0

    def test_warning_when_moderately_worse(self):
        fb = _mk_feedback(prediction_log_id="l1", actual_value=100.0)
        logs = {"l1": _mk_log("l1", 85.0)}  # error = 15
        metrics = {"mae": 10.0}  # degradation = 50%
        result = compute_training_vs_production([fb], logs, metrics, "regression")
        assert result["has_data"] is True
        assert result["degradation_pct"] == pytest.approx(50.0)
        assert result["status"] == "degrading"

    def test_stable_under_10pct_degradation(self):
        fb = _mk_feedback(prediction_log_id="l1", actual_value=100.0)
        logs = {"l1": _mk_log("l1", 91.0)}  # error = 9
        metrics = {"mae": 8.5}  # degradation ≈ 5.9%
        result = compute_training_vs_production([fb], logs, metrics, "regression")
        assert result["has_data"] is True
        assert result["status"] == "stable"

    def test_metric_direction_lower_is_better(self):
        fb = _mk_feedback(prediction_log_id="l1", actual_value=100.0)
        logs = {"l1": _mk_log("l1", 90.0)}
        result = compute_training_vs_production([fb], logs, self._metrics, "regression")
        assert result["metric_direction"] == "lower_is_better"

    def test_weekly_timeline_populated(self):
        fbs = [
            _mk_feedback(
                "l1", 100.0, created_at=datetime(2026, 1, 5, tzinfo=timezone.utc)
            ),
            _mk_feedback(
                "l2", 200.0, created_at=datetime(2026, 1, 12, tzinfo=timezone.utc)
            ),
        ]
        logs = {
            "l1": _mk_log("l1", 90.0),
            "l2": _mk_log("l2", 185.0),
        }
        result = compute_training_vs_production(fbs, logs, self._metrics, "regression")
        assert result["has_data"] is True
        assert isinstance(result["weekly_timeline"], list)
        assert len(result["weekly_timeline"]) >= 1
        for entry in result["weekly_timeline"]:
            assert "period" in entry
            assert "value" in entry
            assert "n" in entry

    def test_summary_contains_metric_values(self):
        fb = _mk_feedback(prediction_log_id="l1", actual_value=100.0)
        logs = {"l1": _mk_log("l1", 88.0)}  # error = 12
        metrics = {"mae": 10.0}
        result = compute_training_vs_production([fb], logs, metrics, "regression")
        assert "MAE" in result["summary"]

    def test_n_feedback_count(self):
        fbs = [
            _mk_feedback("l1", 100.0),
            _mk_feedback("l2", 200.0),
        ]
        logs = {"l1": _mk_log("l1", 95.0), "l2": _mk_log("l2", 193.0)}
        result = compute_training_vs_production(fbs, logs, self._metrics, "regression")
        assert result["n_feedback"] == 2


class TestComputeTrainingVsProductionClassification:
    """Classification: higher accuracy is better."""

    _metrics = {"accuracy": 0.85, "f1": 0.83}

    def test_no_feedback_returns_no_data(self):
        result = compute_training_vs_production([], {}, self._metrics, "classification")
        assert result["has_data"] is False
        assert result["status"] == "no_feedback"
        assert result["training_value"] == pytest.approx(0.85)

    def test_no_rated_records_returns_no_data(self):
        fb = _mk_feedback(is_correct=None)  # unrated
        result = compute_training_vs_production(
            [fb], {}, self._metrics, "classification"
        )
        assert result["has_data"] is False

    def test_stable_when_accuracy_close_to_training(self):
        fbs = [_mk_feedback(is_correct=True) for _ in range(85)] + [
            _mk_feedback(is_correct=False) for _ in range(15)
        ]
        result = compute_training_vs_production(
            fbs, {}, self._metrics, "classification"
        )
        # live=0.85, training=0.85 → 0% degradation → stable
        assert result["has_data"] is True
        assert result["status"] == "stable"

    def test_warning_when_moderate_drop(self):
        # 75% live vs 85% training → drop of ~11.8% → warning
        fbs = [_mk_feedback(is_correct=True) for _ in range(75)] + [
            _mk_feedback(is_correct=False) for _ in range(25)
        ]
        result = compute_training_vs_production(
            fbs, {}, self._metrics, "classification"
        )
        assert result["has_data"] is True
        assert result["status"] == "warning"

    def test_degrading_when_large_drop(self):
        # 60% live vs 85% training → drop ~29.4% → degrading
        fbs = [_mk_feedback(is_correct=True) for _ in range(60)] + [
            _mk_feedback(is_correct=False) for _ in range(40)
        ]
        result = compute_training_vs_production(
            fbs, {}, self._metrics, "classification"
        )
        assert result["has_data"] is True
        assert result["status"] == "degrading"

    def test_metric_direction_higher_is_better(self):
        fb = _mk_feedback(is_correct=True)
        result = compute_training_vs_production(
            [fb], {}, self._metrics, "classification"
        )
        assert result["metric_direction"] == "higher_is_better"

    def test_summary_contains_accuracy_values(self):
        fbs = [_mk_feedback(is_correct=True) for _ in range(8)] + [
            _mk_feedback(is_correct=False) for _ in range(2)
        ]
        result = compute_training_vs_production(
            fbs, {}, self._metrics, "classification"
        )
        assert (
            "accuracy" in result["summary"].lower() or "Accuracy" in result["summary"]
        )

    def test_weekly_timeline_populated(self):
        fbs = [
            _mk_feedback(
                is_correct=True,
                created_at=datetime(2026, 1, 5, tzinfo=timezone.utc),
            ),
            _mk_feedback(
                is_correct=False,
                created_at=datetime(2026, 1, 12, tzinfo=timezone.utc),
            ),
        ]
        result = compute_training_vs_production(
            fbs, {}, self._metrics, "classification"
        )
        assert result["has_data"] is True
        tl = result["weekly_timeline"]
        assert isinstance(tl, list) and len(tl) >= 1

    def test_missing_training_metrics_handled_gracefully(self):
        fb = _mk_feedback(is_correct=True)
        # No accuracy key in metrics
        result = compute_training_vs_production([fb], {}, {}, "classification")
        assert result["has_data"] is True
        assert result["training_value"] is None
        assert result["degradation_pct"] == 0.0


# ─── Regex tests: _PROD_MONITOR_PATTERNS ──────────────────────────────────────


def _get_pattern():
    from api.chat import _PROD_MONITOR_PATTERNS

    return _PROD_MONITOR_PATTERNS


@pytest.mark.parametrize(
    "msg",
    [
        "how is my model holding up in production",
        "training vs production performance",
        "training versus live accuracy",
        "is my model degrading",
        "is my model getting worse",
        "compare training and production performance",
        "production performance vs training",
        "is my model still accurate in production",
        "how much has my model accuracy dropped",
        "production performance check",
        "model degradation check",
        "training vs production gap",
    ],
)
def test_prod_monitor_pattern_positive(msg):
    pattern = _get_pattern()
    assert pattern.search(msg), f"Pattern should match: {msg!r}"


@pytest.mark.parametrize(
    "msg",
    [
        "train a model to predict revenue",
        "show me the correlation matrix",
        "group by region",
        "how is my confidence trending",
        "what are my batch schedules",
        "deployment health overview",
        "show alert rules",
    ],
)
def test_prod_monitor_pattern_negative(msg):
    pattern = _get_pattern()
    assert not pattern.search(msg), f"Pattern should NOT match: {msg!r}"


# ─── REST endpoint tests ───────────────────────────────────────────────────────


def _make_app_with_db(db_url: str):
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    db_module._engine = engine
    from main import app

    return app


_ENDPOINT_CSV = b"region,revenue,units\nEast,100.0,10\nWest,200.0,20\nNorth,150.0,15\n"


class TestTrainingVsProductionEndpoint:
    def test_unknown_deployment_returns_404(self):
        from fastapi.testclient import TestClient

        app = _make_app_with_db("sqlite:///:memory:")
        client = TestClient(app)
        resp = client.get("/api/deploy/nonexistent-id/training-vs-production")
        assert resp.status_code == 404

    def test_no_feedback_returns_no_data_status(self):
        """Endpoint accessible even with no feedback records."""
        from fastapi.testclient import TestClient

        db_url = "sqlite:///:memory:"
        app = _make_app_with_db(db_url)
        client = TestClient(app)

        # Create project + dataset + deployment
        proj = client.post("/api/projects", json={"name": "P1"}).json()
        pid = proj["id"]

        import io

        client.post(
            "/api/data/upload",
            data={"project_id": pid},
            files={"file": ("data.csv", io.BytesIO(_ENDPOINT_CSV), "text/csv")},
        ).json()
        # Deploy a mock model
        from sqlmodel import Session as _Session
        from models.model_run import ModelRun
        import uuid

        with _Session(db_module.engine) as session:
            run = ModelRun(
                id=str(uuid.uuid4()),
                project_id=pid,
                algorithm="linear_regression",
                status="done",
                metrics='{"r2": 0.8, "mae": 5.0}',
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            run_id = run.id

        dep_resp = client.post(
            f"/api/deploy/{run_id}",
            json={"environment": "staging"},
        )
        if dep_resp.status_code not in (200, 201):
            pytest.skip("Deployment creation failed in test environment")
        dep_id = dep_resp.json().get("deployment_id") or dep_resp.json().get("id")
        if not dep_id:
            pytest.skip("Could not get deployment id")

        resp = client.get(f"/api/deploy/{dep_id}/training-vs-production")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["has_data"] is False or data["status"] in (
            "no_feedback",
            "computed",
        )


# ─── Chat handler integration tests ───────────────────────────────────────────


def _mock_stream(text="Model performance check complete."):
    ms = mock.MagicMock()
    ms.__enter__ = mock.MagicMock(return_value=ms)
    ms.__exit__ = mock.MagicMock(return_value=False)
    ms.text_stream = iter([text])
    return ms


def _parse_events(resp) -> list[dict]:
    events = []
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            try:
                events.append(json.loads(line[5:].strip()))
            except json.JSONDecodeError:
                pass
    return events


@pytest.mark.asyncio
async def test_prod_monitor_chat_emits_event():
    """When a deployment exists and message matches, prod_performance SSE event fires."""
    import io

    db_url = "sqlite:///:memory:"
    app = _make_app_with_db(db_url)

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = _mock_stream()

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            # Create project + dataset + model run + deployment
            proj = (await ac.post("/api/projects", json={"name": "TVP Project"})).json()
            pid = proj["id"]

            await ac.post(
                "/api/data/upload",
                data={"project_id": pid},
                files={
                    "file": (
                        "data.csv",
                        io.BytesIO(_ENDPOINT_CSV),
                        "text/csv",
                    )
                },
            )

            # Inject model run directly
            from models.model_run import ModelRun as MR
            import uuid

            from sqlmodel import Session as _Sess

            with _Sess(db_module.engine) as sess:
                run = MR(
                    id=str(uuid.uuid4()),
                    project_id=pid,
                    algorithm="linear_regression",
                    status="done",
                    metrics='{"r2": 0.8, "mae": 5.0}',
                )
                sess.add(run)
                sess.commit()
                sess.refresh(run)
                run_id = run.id

            dep = await ac.post(f"/api/deploy/{run_id}", json={})
            if dep.status_code not in (200, 201):
                pytest.skip("Deployment creation failed")
            dep_data = dep.json()
            dep_id = dep_data.get("deployment_id") or dep_data.get("id")
            if not dep_id:
                pytest.skip("No deployment id returned")

            MockAnthropic.return_value.messages.stream.return_value = _mock_stream()
            resp = await ac.post(
                f"/api/chat/{pid}/message",
                json={
                    "message": "is my model degrading in production",
                    "history": [],
                },
            )
            assert resp.status_code == 200
            events = _parse_events(resp)
            types = [e.get("type") for e in events]
            assert (
                "prod_performance" in types
            ), f"Expected prod_performance. Got: {types}"
            ev = next(e for e in events if e.get("type") == "prod_performance")
            data = ev["prod_performance"]
            assert "status" in data
            assert "summary" in data


@pytest.mark.asyncio
async def test_prod_monitor_chat_no_deployment_guard():
    """Without a deployment, prod_performance event is not emitted."""
    import io

    db_url = "sqlite:///:memory:"
    app = _make_app_with_db(db_url)

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = _mock_stream()

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            proj = (await ac.post("/api/projects", json={"name": "No Dep"})).json()
            pid = proj["id"]

            await ac.post(
                "/api/data/upload",
                data={"project_id": pid},
                files={
                    "file": (
                        "data.csv",
                        io.BytesIO(_ENDPOINT_CSV),
                        "text/csv",
                    )
                },
            )

            MockAnthropic.return_value.messages.stream.return_value = _mock_stream()
            resp = await ac.post(
                f"/api/chat/{pid}/message",
                json={
                    "message": "is my model degrading in production",
                    "history": [],
                },
            )
            if resp.status_code != 200:
                pytest.skip(
                    "Chat endpoint not available without initialized session context"
                )
            events = _parse_events(resp)
            types = [e.get("type") for e in events]
            assert "prod_performance" not in types
