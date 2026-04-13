"""Tests for A/B test chat integration.

Covers:
- _AB_TEST_PATTERNS      — NL intent detection regex
- _AB_PROMOTE_RE         — promote action detection
- _AB_END_RE             — end action detection
- Chat handler           — status, no-test, promote, end via SSE
"""

import io
import json
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

_SAMPLE_CSV = (
    b"feat1,feat2,target\n"
    b"1.0,0.5,10.0\n2.0,1.0,20.0\n3.0,1.5,30.0\n4.0,2.0,40.0\n5.0,2.5,50.0\n"
    b"6.0,3.0,60.0\n7.0,3.5,70.0\n8.0,4.0,80.0\n9.0,4.5,90.0\n10.0,5.0,100.0\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chat_events(client, project_id: str, message: str) -> list[dict]:
    """Send a chat message with mocked Anthropic and return parsed SSE events."""
    with patch("anthropic.Anthropic") as mock_cls:
        mock_c = MagicMock()
        mock_cls.return_value = mock_c
        mock_stream = MagicMock()
        mock_stream.__enter__ = MagicMock(return_value=mock_stream)
        mock_stream.__exit__ = MagicMock(return_value=False)
        mock_stream.text_stream = iter(["Done."])
        mock_c.messages.stream.return_value = mock_stream

        response = client.post(
            f"/api/chat/{project_id}",
            json={"message": message, "project_id": project_id},
        )

    events = []
    for line in response.text.split("\n"):
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    """TestClient backed by an isolated SQLite DB."""
    from main import app

    test_db = str(tmp_path / "test.db")
    orig_engine = db_module.engine
    db_module.engine = create_engine(
        f"sqlite:///{test_db}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(db_module.engine)
    db_module.create_db_and_tables()

    yield TestClient(app)

    db_module.engine = orig_engine


@pytest.fixture()
def deployed_project(client, tmp_path):
    """Create project, upload CSV, train, and deploy."""
    proj = client.post("/api/projects", json={"name": "ABTestChatTest"})
    project_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
    )
    dataset_id = upload.json()["dataset_id"]

    client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    client.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "target", "problem_type": "regression"},
    )

    train_resp = client.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    run_id = train_resp.json()["model_run_ids"][0]
    for _ in range(30):
        runs = client.get(f"/api/models/{project_id}/runs").json()["runs"]
        run = next(r for r in runs if r["id"] == run_id)
        if run["status"] in ("done", "failed"):
            break
        time.sleep(0.3)
    assert run["status"] == "done"

    dep = client.post(f"/api/deploy/{run_id}")
    assert dep.status_code in (200, 201), dep.text

    return {
        "project_id": project_id,
        "deployment_id": dep.json()["id"],
        "run_id": run_id,
    }


# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


class TestABTestPatterns:
    def test_how_is_ab_test_going(self):
        from api.chat import _AB_TEST_PATTERNS

        assert _AB_TEST_PATTERNS.search("how is my A/B test going?")

    def test_check_ab_test(self):
        from api.chat import _AB_TEST_PATTERNS

        assert _AB_TEST_PATTERNS.search("check my A/B test")

    def test_show_ab_test_status(self):
        from api.chat import _AB_TEST_PATTERNS

        assert _AB_TEST_PATTERNS.search("show A/B test status")

    def test_is_challenger_doing_better(self):
        from api.chat import _AB_TEST_PATTERNS

        assert _AB_TEST_PATTERNS.search("is the challenger doing better?")

    def test_ab_test_results(self):
        from api.chat import _AB_TEST_PATTERNS

        assert _AB_TEST_PATTERNS.search("A/B test results")

    def test_promote_challenger(self):
        from api.chat import _AB_TEST_PATTERNS

        assert _AB_TEST_PATTERNS.search("promote the challenger")

    def test_end_ab_test(self):
        from api.chat import _AB_TEST_PATTERNS

        assert _AB_TEST_PATTERNS.search("end the A/B test")

    def test_stop_split_test(self):
        from api.chat import _AB_TEST_PATTERNS

        assert _AB_TEST_PATTERNS.search("stop the split test")

    def test_negative_train_model(self):
        from api.chat import _AB_TEST_PATTERNS

        assert not _AB_TEST_PATTERNS.search("train a new model")

    def test_negative_make_prediction(self):
        from api.chat import _AB_TEST_PATTERNS

        assert not _AB_TEST_PATTERNS.search("make a prediction for units=100")

    def test_promote_re_matches(self):
        from api.chat import _AB_PROMOTE_RE

        assert _AB_PROMOTE_RE.search("promote the challenger")

    def test_end_re_matches(self):
        from api.chat import _AB_END_RE

        assert _AB_END_RE.search("end the A/B test")

    def test_end_re_stop(self):
        from api.chat import _AB_END_RE

        assert _AB_END_RE.search("stop the split test")


# ---------------------------------------------------------------------------
# Integration tests — chat handler
# ---------------------------------------------------------------------------


class TestABTestChatHandler:
    def test_no_active_test_emits_none_event(self, deployed_project, client):
        """Asking about A/B test with no active test returns action=none."""
        events = _chat_events(
            client,
            deployed_project["project_id"],
            "how is my A/B test going?",
        )
        ab_events = [e for e in events if e.get("type") == "ab_test_result"]
        assert ab_events, "Expected ab_test_result SSE event"
        event = ab_events[0]["ab_test_result"]
        assert event["action"] == "none"
        assert "summary" in event

    def test_status_emits_event_with_test_data(self, deployed_project, client, tmp_path):
        """With an active A/B test, status query returns action=status with metrics.

        Creates a second independent deployment (different project) as the challenger
        — required because the deployment system keeps a single deployment per project.
        """
        dep_id = deployed_project["deployment_id"]
        project_id = deployed_project["project_id"]

        # Create a second independent project + deployment to use as challenger
        proj2 = client.post("/api/projects", json={"name": "ChallengerProject"})
        proj2_id = proj2.json()["id"]
        up2 = client.post(
            "/api/data/upload",
            data={"project_id": proj2_id},
            files={"file": ("d2.csv", io.BytesIO(_SAMPLE_CSV), "text/csv")},
        )
        dataset_id2 = up2.json()["dataset_id"]
        client.post(f"/api/features/{dataset_id2}/apply", json={"transformations": []})
        client.post(
            f"/api/features/{dataset_id2}/target",
            json={"target_column": "target", "problem_type": "regression"},
        )
        t2 = client.post(
            f"/api/models/{proj2_id}/train",
            json={"algorithms": ["linear_regression"]},
        )
        run_id2 = t2.json()["model_run_ids"][0]
        for _ in range(30):
            runs = client.get(f"/api/models/{proj2_id}/runs").json()["runs"]
            run = next((r for r in runs if r["id"] == run_id2), None)
            if run and run["status"] in ("done", "failed"):
                break
            time.sleep(0.3)
        dep2 = client.post(f"/api/deploy/{run_id2}")
        assert dep2.status_code in (200, 201), dep2.text
        challenger_id = dep2.json()["id"]

        # Create A/B test linking the two deployments
        ab_resp = client.post(
            f"/api/deploy/{dep_id}/ab-test",
            json={"challenger_id": challenger_id, "champion_split_pct": 80},
        )
        assert ab_resp.status_code == 201, ab_resp.text

        events = _chat_events(
            client,
            project_id,
            "show me the A/B test status",
        )
        ab_events = [e for e in events if e.get("type") == "ab_test_result"]
        assert ab_events, "Expected ab_test_result SSE event"
        event = ab_events[0]["ab_test_result"]
        assert event["action"] == "status"
        assert "champion_split_pct" in event
        assert "challenger_split_pct" in event
        assert "champion_metrics" in event
        assert "challenger_metrics" in event
        assert "significance" in event
        assert event["champion_split_pct"] == 80
        assert event["challenger_split_pct"] == 20

    def test_no_deployment_no_ab_event(self, client):
        """Without a deployment, A/B test chat does not emit an event."""
        proj = client.post("/api/projects", json={"name": "NoDep"})
        project_id = proj.json()["id"]
        events = _chat_events(client, project_id, "how is my A/B test going?")
        ab_events = [e for e in events if e.get("type") == "ab_test_result"]
        assert not ab_events, "No ab_test_result when no deployment"
