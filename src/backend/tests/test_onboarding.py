"""Tests for the guided onboarding wizard.

Covers:
- Pure function: compute_onboarding_state edge cases and step progression
- API: GET /api/projects/{id}/onboarding returns correct state
- Chat: _ONBOARDING_PATTERNS detects intent correctly
"""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

from core.onboarding import compute_onboarding_state

# ---------------------------------------------------------------------------
# Test DB fixture
# ---------------------------------------------------------------------------


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
    import models.deployment  # noqa
    import models.prediction_log  # noqa
    import models.feedback_record  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_api_module

    models_api_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestComputeOnboardingState:
    def test_fresh_project_step0(self):
        """New project → step 0 (upload)."""
        state = compute_onboarding_state(
            has_dataset=False,
            message_count=0,
            has_target=False,
            has_model_run=False,
            has_cross_val=False,
            has_deployment=False,
        )
        assert state["step_index"] == 0
        assert state["is_complete"] is False
        assert state["completion_pct"] == 0
        assert state["current_step"]["name"] == "upload"

    def test_after_upload_step1(self):
        """Dataset uploaded → step 1 (explore)."""
        state = compute_onboarding_state(
            has_dataset=True,
            message_count=1,
            has_target=False,
            has_model_run=False,
            has_cross_val=False,
            has_deployment=False,
        )
        assert state["step_index"] == 1
        assert state["current_step"]["name"] == "explore"

    def test_after_explore_step2(self):
        """Dataset + 2+ messages → step 2 (target)."""
        state = compute_onboarding_state(
            has_dataset=True,
            message_count=3,
            has_target=False,
            has_model_run=False,
            has_cross_val=False,
            has_deployment=False,
        )
        assert state["step_index"] == 2
        assert state["current_step"]["name"] == "target"

    def test_after_target_step3(self):
        """Target set → step 3 (train)."""
        state = compute_onboarding_state(
            has_dataset=True,
            message_count=5,
            has_target=True,
            has_model_run=False,
            has_cross_val=False,
            has_deployment=False,
        )
        assert state["step_index"] == 3
        assert state["current_step"]["name"] == "train"

    def test_after_training_step4(self):
        """Model run done → step 4 (validate)."""
        state = compute_onboarding_state(
            has_dataset=True,
            message_count=5,
            has_target=True,
            has_model_run=True,
            has_cross_val=False,
            has_deployment=False,
        )
        assert state["step_index"] == 4
        assert state["current_step"]["name"] == "validate"

    def test_after_validation_step5(self):
        """Model run + cross-val → step 5 (deploy)."""
        state = compute_onboarding_state(
            has_dataset=True,
            message_count=5,
            has_target=True,
            has_model_run=True,
            has_cross_val=True,
            has_deployment=False,
        )
        assert state["step_index"] == 5
        assert state["current_step"]["name"] == "deploy"

    def test_complete_when_deployed(self):
        """All steps done → is_complete True."""
        state = compute_onboarding_state(
            has_dataset=True,
            message_count=8,
            has_target=True,
            has_model_run=True,
            has_cross_val=True,
            has_deployment=True,
        )
        assert state["is_complete"] is True
        assert state["completion_pct"] == 100
        assert state["current_step"] is None
        assert (
            "complete" in state["summary"].lower() or "set" in state["summary"].lower()
        )

    def test_total_steps_is_six(self):
        state = compute_onboarding_state(False, 0, False, False, False, False)
        assert state["total_steps"] == 6

    def test_steps_list_length(self):
        state = compute_onboarding_state(False, 0, False, False, False, False)
        assert len(state["steps"]) == 6

    def test_done_steps_have_is_done_true(self):
        """Steps before current should all be is_done=True."""
        state = compute_onboarding_state(
            has_dataset=True,
            message_count=5,
            has_target=True,
            has_model_run=False,
            has_cross_val=False,
            has_deployment=False,
        )
        # Steps 0, 1, 2 are done; step 3 is current
        for i in range(3):
            assert state["steps"][i]["is_done"] is True
        assert state["steps"][3]["is_current"] is True

    def test_completion_pct_monotone(self):
        """Completion percentage increases as each step is completed."""
        pcts = []
        for n_done in range(7):
            flags = [n_done > i for i in range(6)]
            state = compute_onboarding_state(
                has_dataset=flags[0],
                message_count=5 if flags[1] else 0,
                has_target=flags[2],
                has_model_run=flags[3],
                has_cross_val=flags[4],
                has_deployment=flags[5],
            )
            pcts.append(state["completion_pct"])
        assert pcts == sorted(pcts)

    def test_summary_mentions_step_name(self):
        """Summary for a mid-journey state mentions the next step."""
        state = compute_onboarding_state(
            has_dataset=True,
            message_count=5,
            has_target=False,
            has_model_run=False,
            has_cross_val=False,
            has_deployment=False,
        )
        assert (
            "prediction target" in state["summary"].lower()
            or "target" in state["summary"].lower()
        )


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


def test_onboarding_endpoint_returns_state(client):
    """GET /api/projects/{id}/onboarding returns onboarding state."""
    proj = client.post("/api/projects", json={"name": "Onboarding Test"})
    assert proj.status_code == 201
    project_id = proj.json()["id"]

    resp = client.get(f"/api/projects/{project_id}/onboarding")
    assert resp.status_code == 200
    data = resp.json()
    assert "step_index" in data
    assert "steps" in data
    assert "is_complete" in data
    assert data["project_id"] == project_id


def test_onboarding_endpoint_fresh_project(client):
    """Fresh project (no dataset) → step 0."""
    proj = client.post("/api/projects", json={"name": "Fresh Project"})
    project_id = proj.json()["id"]

    resp = client.get(f"/api/projects/{project_id}/onboarding")
    data = resp.json()
    assert data["step_index"] == 0
    assert data["is_complete"] is False


def test_onboarding_endpoint_404_on_unknown(client):
    """Unknown project → 404."""
    resp = client.get("/api/projects/00000000-0000-0000-0000-000000000000/onboarding")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Chat pattern tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "guide me through this",
        "help me get started",
        "walk me through the steps",
        "show me the guide",
        "what should I do first",
        "I'm new here, first steps please",
        "onboarding",
        "how do I use this",
    ],
)
def test_onboarding_pattern_matches(phrase):
    from api.chat import _ONBOARDING_PATTERNS

    assert _ONBOARDING_PATTERNS.search(phrase), f"Expected match for: {phrase!r}"


@pytest.mark.parametrize(
    "phrase",
    [
        "what is the average revenue",
        "train a model",
        "show me the scatter plot",
    ],
)
def test_onboarding_pattern_no_match(phrase):
    from api.chat import _ONBOARDING_PATTERNS

    assert not _ONBOARDING_PATTERNS.search(phrase), f"Unexpected match for: {phrase!r}"
