"""Tests for ensemble model recommendation via chat.

Covers:
- _ENSEMBLE_PATTERNS regex matching (8 positive, 2 negative)
- Chat handler emitting ensemble_recommendation SSE event with model runs
- No model runs → no event emitted
- Required fields present in event payload
"""

from __future__ import annotations

import io
import json
import unittest.mock as mock

import pytest
from sqlmodel import Session, SQLModel, create_engine

import db as db_module

# Small regression CSV
_REGRESSION_CSV = (
    b"region,units,price,revenue\n"
    + b"East,10,5.0,50.0\n" * 40
    + b"West,20,8.0,160.0\n" * 40
    + b"North,15,6.0,90.0\n" * 20
)


# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


class TestEnsemblePatterns:
    """Verify _ENSEMBLE_PATTERNS matches all intended phrases."""

    @pytest.fixture(autouse=True)
    def _import_pattern(self):
        from api.chat import _ENSEMBLE_PATTERNS

        self.pattern = _ENSEMBLE_PATTERNS

    def test_best_ensemble_for_problem(self):
        assert self.pattern.search("what's the best ensemble for this problem?")

    def test_should_try_ensemble(self):
        assert self.pattern.search("should I try an ensemble model?")

    def test_should_use_ensemble(self):
        assert self.pattern.search("should I use an ensemble?")

    def test_voting_classifier(self):
        assert self.pattern.search("can I use a voting classifier here?")

    def test_stacking_regressor(self):
        assert self.pattern.search("try a stacking regressor for this")

    def test_combine_models(self):
        assert self.pattern.search("combine my models together")

    def test_can_ensemble_improve(self):
        assert self.pattern.search("can an ensemble improve my accuracy?")

    def test_ensemble_recommendation(self):
        assert self.pattern.search("ensemble recommendation for this dataset")

    def test_negative_train_model(self):
        assert not self.pattern.search("train a random forest model")

    def test_negative_class_imbalance(self):
        assert not self.pattern.search("is my data imbalanced?")


# ---------------------------------------------------------------------------
# Chat handler integration tests (synchronous TestClient)
# ---------------------------------------------------------------------------


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def _chat_events(client, project_id: str, message: str) -> list[dict]:
    """Send chat message with mocked Anthropic and return parsed SSE events."""
    with mock.patch("anthropic.Anthropic") as mock_cls:
        mc = mock.MagicMock()
        mock_cls.return_value = mc
        ms = mock.MagicMock()
        ms.__enter__ = mock.MagicMock(return_value=ms)
        ms.__exit__ = mock.MagicMock(return_value=False)
        ms.text_stream = iter(["Done."])
        mc.messages.stream.return_value = ms
        resp = client.post(
            f"/api/chat/{project_id}",
            json={"message": message, "project_id": project_id},
        )
    return _parse_sse(resp.text)


@pytest.fixture()
def sync_client(tmp_path):
    """Synchronous TestClient with isolated SQLite DB."""
    from fastapi.testclient import TestClient
    from main import app

    test_db = str(tmp_path / "ensemble_chat_test.db")
    orig_engine = db_module.engine
    db_module.engine = create_engine(
        f"sqlite:///{test_db}", connect_args={"check_same_thread": False}
    )
    import models  # noqa — registers all tables

    SQLModel.metadata.create_all(db_module.engine)
    db_module.create_db_and_tables()

    import api.data as dm
    import api.deploy as dep
    import api.models as mm

    dm.UPLOAD_DIR = tmp_path / "uploads_ens"
    dep.DEPLOY_DIR = tmp_path / "deployments_ens"
    mm.MODELS_DIR = tmp_path / "models_ens"

    yield TestClient(app)
    db_module.engine = orig_engine


def _setup_with_model_run(client):
    """Create project, upload CSV, set target, inject a done ModelRun. Returns project_id.

    Injects the ModelRun directly to avoid the background-training thread / DB-engine
    mismatch that occurs when the real training path captures the module-level engine
    reference before the test fixture can swap it.
    """
    from models.model_run import ModelRun

    proj = client.post("/api/projects", json={"name": "EnsembleTest"})
    pid = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": pid},
        files={"file": ("reg.csv", io.BytesIO(_REGRESSION_CSV), "text/csv")},
    )
    assert upload.status_code == 201, upload.text
    did = upload.json()["dataset_id"]

    fs_r = client.post(f"/api/features/{did}/apply", json={"transformations": []})
    assert fs_r.status_code == 201, fs_r.text
    fs_id = fs_r.json()["feature_set_id"]

    client.post(
        f"/api/features/{did}/target",
        json={"target_column": "revenue", "feature_set_id": fs_id},
    )

    # Inject a completed ModelRun directly — no background thread needed
    with Session(db_module.engine) as session:
        run = ModelRun(
            project_id=pid,
            feature_set_id=fs_id,
            algorithm="linear_regression",
            status="done",
            metrics=json.dumps({"r2": 0.92, "mae": 5.0}),
            summary="Linear Regression: R² 0.920",
        )
        session.add(run)
        session.commit()

    return pid


class TestEnsembleChatHandler:
    """Integration tests for the ensemble recommendation chat handler."""

    def test_no_model_runs_no_event(self, sync_client):
        """Without any model runs, no ensemble_recommendation event is emitted."""
        proj = sync_client.post("/api/projects", json={"name": "NoRuns"})
        pid = proj.json()["id"]
        events = _chat_events(sync_client, pid, "should I use an ensemble?")
        assert not any(e.get("type") == "ensemble_recommendation" for e in events)

    def test_with_model_run_emits_event(self, sync_client):
        """With a completed model run, ensemble_recommendation event is emitted."""
        pid = _setup_with_model_run(sync_client)
        events = _chat_events(sync_client, pid, "what's the best ensemble for this problem?")
        ens_events = [e for e in events if e.get("type") == "ensemble_recommendation"]
        assert len(ens_events) == 1

    def test_event_has_required_fields(self, sync_client):
        """Event payload includes all required fields."""
        pid = _setup_with_model_run(sync_client)
        events = _chat_events(sync_client, pid, "should I try an ensemble?")
        payload = next(
            e["ensemble_recommendation"]
            for e in events
            if e.get("type") == "ensemble_recommendation"
        )
        for field in (
            "problem_type",
            "metric_name",
            "dataset_size",
            "recommended_algorithm",
            "recommended_name",
            "options",
            "summary",
        ):
            assert field in payload, f"Missing field: {field}"

    def test_options_has_two_entries(self, sync_client):
        """Options list contains voting and stacking entries."""
        pid = _setup_with_model_run(sync_client)
        events = _chat_events(sync_client, pid, "ensemble recommendation please")
        payload = next(
            e["ensemble_recommendation"]
            for e in events
            if e.get("type") == "ensemble_recommendation"
        )
        assert len(payload["options"]) == 2
        types = {o["ensemble_type"] for o in payload["options"]}
        assert types == {"voting", "stacking"}

    def test_regression_uses_regression_algos(self, sync_client):
        """Regression problem type uses voting/stacking regressor algo names."""
        pid = _setup_with_model_run(sync_client)
        events = _chat_events(sync_client, pid, "voting regressor options?")
        payload = next(
            e["ensemble_recommendation"]
            for e in events
            if e.get("type") == "ensemble_recommendation"
        )
        assert payload["problem_type"] == "regression"
        algos = {o["algorithm"] for o in payload["options"]}
        assert "voting_regressor" in algos
        assert "stacking_regressor" in algos

    def test_summary_is_nonempty_string(self, sync_client):
        """Summary is a non-empty plain-English string."""
        pid = _setup_with_model_run(sync_client)
        events = _chat_events(sync_client, pid, "can an ensemble improve my results?")
        payload = next(
            e["ensemble_recommendation"]
            for e in events
            if e.get("type") == "ensemble_recommendation"
        )
        assert isinstance(payload["summary"], str)
        assert len(payload["summary"]) > 20
