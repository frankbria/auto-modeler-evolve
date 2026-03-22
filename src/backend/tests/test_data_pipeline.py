"""Tests for the incremental transformation pipeline (steps) endpoints.

Covers:
- GET  /api/features/{feature_set_id}/steps  (list steps)
- POST /api/features/{feature_set_id}/steps  (add step)
- DELETE /api/features/{feature_set_id}/steps/{idx}  (remove/undo step)

All tests use the shared async client fixture from conftest.
"""

from __future__ import annotations

import pathlib

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_project(client) -> str:
    r = await client.post("/api/projects", json={"name": "Pipeline Test"})
    assert r.status_code == 201
    return r.json()["id"]


async def _upload_dataset(client, project_id: str) -> str:
    sample = pathlib.Path(__file__).parent.parent / "data/sample/sample_sales.csv"
    with open(sample, "rb") as f:
        r = await client.post(
            "/api/data/upload",
            files={"file": ("sales.csv", f, "text/csv")},
            data={"project_id": project_id},
        )
    assert r.status_code == 201
    return r.json()["dataset_id"]


async def _create_empty_feature_set(client, dataset_id: str) -> str:
    """Create an empty FeatureSet (no transformations)."""
    r = await client.post(
        f"/api/features/{dataset_id}/apply",
        json={"transformations": []},
    )
    assert r.status_code == 201
    return r.json()["feature_set_id"]


# ---------------------------------------------------------------------------
# Tests: list steps
# ---------------------------------------------------------------------------


class TestListSteps:
    async def test_empty_pipeline(self, client):
        pid = await _create_project(client)
        did = await _upload_dataset(client, pid)
        fsid = await _create_empty_feature_set(client, did)

        r = await client.get(f"/api/features/{fsid}/steps")
        assert r.status_code == 200
        body = r.json()
        assert body["feature_set_id"] == fsid
        assert body["step_count"] == 0
        assert body["steps"] == []

    async def test_lists_existing_steps(self, client):
        pid = await _create_project(client)
        did = await _upload_dataset(client, pid)
        # Apply 2 transformations at once
        r = await client.post(
            f"/api/features/{did}/apply",
            json={
                "transformations": [
                    {"column": "revenue", "transform_type": "log_transform"},
                    {"column": "units", "transform_type": "log_transform"},
                ]
            },
        )
        assert r.status_code == 201
        fsid = r.json()["feature_set_id"]

        r2 = await client.get(f"/api/features/{fsid}/steps")
        assert r2.status_code == 200
        body = r2.json()
        assert body["step_count"] == 2
        assert body["steps"][0]["index"] == 0
        assert body["steps"][0]["column"] == "revenue"
        assert body["steps"][1]["index"] == 1
        assert body["steps"][1]["column"] == "units"

    async def test_list_steps_feature_set_not_found(self, client):
        r = await client.get("/api/features/nonexistent-fsid/steps")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Tests: add step
# ---------------------------------------------------------------------------


class TestAddStep:
    async def test_add_single_step_to_empty_pipeline(self, client):
        pid = await _create_project(client)
        did = await _upload_dataset(client, pid)
        fsid = await _create_empty_feature_set(client, did)

        r = await client.post(
            f"/api/features/{fsid}/steps",
            json={"column": "revenue", "transform_type": "log_transform"},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["step_index"] == 0
        assert body["step_count"] == 1
        assert "revenue_log" in body["new_columns"]
        assert body["total_columns"] >= 1

    async def test_add_multiple_steps_incrementally(self, client):
        pid = await _create_project(client)
        did = await _upload_dataset(client, pid)
        fsid = await _create_empty_feature_set(client, did)

        # Step 1: log transform revenue
        r1 = await client.post(
            f"/api/features/{fsid}/steps",
            json={"column": "revenue", "transform_type": "log_transform"},
        )
        assert r1.status_code == 201
        assert r1.json()["step_count"] == 1

        # Step 2: log transform units
        r2 = await client.post(
            f"/api/features/{fsid}/steps",
            json={"column": "units", "transform_type": "log_transform"},
        )
        assert r2.status_code == 201
        assert r2.json()["step_count"] == 2
        assert "units_log" in r2.json()["new_columns"]

        # Verify both steps are listed
        r3 = await client.get(f"/api/features/{fsid}/steps")
        assert r3.json()["step_count"] == 2

    async def test_add_step_preview_included(self, client):
        pid = await _create_project(client)
        did = await _upload_dataset(client, pid)
        fsid = await _create_empty_feature_set(client, did)

        r = await client.post(
            f"/api/features/{fsid}/steps",
            json={"column": "revenue", "transform_type": "log_transform"},
        )
        assert r.status_code == 201
        body = r.json()
        assert "preview" in body
        assert isinstance(body["preview"], list)
        assert len(body["preview"]) <= 5

    async def test_add_step_feature_set_not_found(self, client):
        r = await client.post(
            "/api/features/nonexistent-fsid/steps",
            json={"column": "revenue", "transform_type": "log_transform"},
        )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Tests: remove step (undo)
# ---------------------------------------------------------------------------


class TestRemoveStep:
    async def test_undo_last_step(self, client):
        pid = await _create_project(client)
        did = await _upload_dataset(client, pid)
        fsid = await _create_empty_feature_set(client, did)

        await client.post(
            f"/api/features/{fsid}/steps",
            json={"column": "revenue", "transform_type": "log_transform"},
        )
        await client.post(
            f"/api/features/{fsid}/steps",
            json={"column": "units", "transform_type": "log_transform"},
        )

        # Undo the last step (index 1)
        r = await client.delete(f"/api/features/{fsid}/steps/1")
        assert r.status_code == 200
        body = r.json()
        assert body["step_count"] == 1
        assert body["removed_step"]["column"] == "units"
        assert "units_log" not in body["new_columns"]
        assert "revenue_log" in body["new_columns"]

    async def test_undo_first_step(self, client):
        pid = await _create_project(client)
        did = await _upload_dataset(client, pid)
        fsid = await _create_empty_feature_set(client, did)

        await client.post(
            f"/api/features/{fsid}/steps",
            json={"column": "revenue", "transform_type": "log_transform"},
        )
        await client.post(
            f"/api/features/{fsid}/steps",
            json={"column": "units", "transform_type": "log_transform"},
        )

        # Remove first step (index 0)
        r = await client.delete(f"/api/features/{fsid}/steps/0")
        assert r.status_code == 200
        body = r.json()
        assert body["step_count"] == 1
        assert body["removed_step"]["column"] == "revenue"
        assert body["steps"][0]["column"] == "units"

    async def test_undo_all_steps(self, client):
        pid = await _create_project(client)
        did = await _upload_dataset(client, pid)
        fsid = await _create_empty_feature_set(client, did)

        await client.post(
            f"/api/features/{fsid}/steps",
            json={"column": "revenue", "transform_type": "log_transform"},
        )

        r = await client.delete(f"/api/features/{fsid}/steps/0")
        assert r.status_code == 200
        body = r.json()
        assert body["step_count"] == 0
        assert body["new_columns"] == []

    async def test_undo_invalid_index(self, client):
        pid = await _create_project(client)
        did = await _upload_dataset(client, pid)
        fsid = await _create_empty_feature_set(client, did)

        await client.post(
            f"/api/features/{fsid}/steps",
            json={"column": "revenue", "transform_type": "log_transform"},
        )

        # Index 5 is out of range (pipeline has 1 step)
        r = await client.delete(f"/api/features/{fsid}/steps/5")
        assert r.status_code == 422

    async def test_undo_on_empty_pipeline(self, client):
        pid = await _create_project(client)
        did = await _upload_dataset(client, pid)
        fsid = await _create_empty_feature_set(client, did)

        r = await client.delete(f"/api/features/{fsid}/steps/0")
        assert r.status_code == 422  # empty pipeline, no step 0

    async def test_remove_step_feature_set_not_found(self, client):
        r = await client.delete("/api/features/nonexistent-fsid/steps/0")
        assert r.status_code == 404

    async def test_pipeline_steps_persisted_across_requests(self, client):
        """Verify that steps survive between requests (not just in-memory)."""
        pid = await _create_project(client)
        did = await _upload_dataset(client, pid)
        fsid = await _create_empty_feature_set(client, did)

        await client.post(
            f"/api/features/{fsid}/steps",
            json={"column": "revenue", "transform_type": "log_transform"},
        )
        await client.post(
            f"/api/features/{fsid}/steps",
            json={"column": "units", "transform_type": "log_transform"},
        )

        # Undo last step
        await client.delete(f"/api/features/{fsid}/steps/1")

        # Re-fetch steps — should only have 1
        r = await client.get(f"/api/features/{fsid}/steps")
        body = r.json()
        assert body["step_count"] == 1
        assert body["steps"][0]["column"] == "revenue"
