"""Tenant isolation / IDOR tests.

Proves the new auth layer enforces the two core guarantees the audit demanded:

1. Unauthenticated callers are rejected (401) on management endpoints.
2. An authenticated tenant cannot read/mutate another tenant's resources
   (cross-tenant access returns 404, never leaking existence).

``client`` authenticates as user A (DEFAULT_USER_ID); ``second_client`` is a
distinct user B sharing the same database; ``anon_client`` sends no token.
"""

import pytest
from sqlmodel import Session

from tests.conftest import DEFAULT_USER_ID

A_OWNER = DEFAULT_USER_ID


def _seed_owned_stack(owner_id: str) -> dict:
    """Insert a project + dataset + model run + deployment owned by ``owner_id``.

    Returns the created ids. Uses the shared test engine that ``client`` set up.
    """
    import db
    from models.dataset import Dataset
    from models.deployment import Deployment
    from models.model_run import ModelRun
    from models.project import Project

    ids = {
        "project": "iso-proj-A",
        "dataset": "iso-ds-A",
        "run": "iso-run-A",
        "deployment": "iso-dep-A",
    }
    with Session(db.engine) as s:
        s.add(Project(id=ids["project"], owner_id=owner_id, name="A's project"))
        s.add(
            Dataset(
                id=ids["dataset"],
                project_id=ids["project"],
                filename="a.csv",
                file_path="/tmp/iso_a.csv",
                row_count=5,
            )
        )
        s.add(
            ModelRun(
                id=ids["run"],
                project_id=ids["project"],
                algorithm="random_forest",
                status="done",
            )
        )
        s.add(
            Deployment(
                id=ids["deployment"],
                project_id=ids["project"],
                model_run_id=ids["run"],
                endpoint_path=f"/api/predict/{ids['deployment']}",
                dashboard_url=f"/predict/{ids['deployment']}",
            )
        )
        s.commit()
    return ids


# ---------------------------------------------------------------------------
# 1. Unauthenticated access is rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path,body",
    [
        ("get", "/api/projects", None),
        ("post", "/api/projects", {"name": "x"}),
        ("get", "/api/projects/any-id", None),
        ("patch", "/api/projects/any-id", {"name": "y"}),
        ("delete", "/api/projects/any-id", None),
        ("get", "/api/data/any-id/preview", None),
        ("get", "/api/data/project/any-id/datasets", None),
        ("get", "/api/models/any-id/runs", None),
        ("get", "/api/deployments", None),
        ("get", "/api/deploy/any-id", None),
        ("post", "/api/chat/any-id", {"message": "hi"}),
    ],
)
@pytest.mark.asyncio
async def test_unauthenticated_management_requests_rejected(
    anon_client, method, path, body
):
    resp = await getattr(anon_client, method)(
        path, **({"json": body} if body is not None else {})
    )
    assert resp.status_code == 401, f"{method} {path} -> {resp.status_code}"


@pytest.mark.asyncio
async def test_public_predict_endpoint_not_blocked_by_auth(anon_client):
    """The public serving surface must NOT 401 on a missing user token."""
    resp = await anon_client.post("/api/predict/nonexistent", json={"features": {}})
    # 404 (no such deployment) — crucially NOT 401.
    assert resp.status_code != 401


@pytest.mark.asyncio
async def test_public_prediction_form_endpoints_not_blocked_by_auth(anon_client):
    """The anonymous /predict/[id] page bootstraps via the public prediction
    surface; those read endpoints must load for a real deployment without a
    user token (issue #25)."""
    ids = _seed_owned_stack(A_OWNER)
    dep = ids["deployment"]

    info = await anon_client.get(f"/api/predict/{dep}/info")
    assert info.status_code == 200, info.text

    # The remaining form endpoints must be reachable anonymously (never 401).
    for path in (
        f"/api/predict/{dep}/presets",
        f"/api/predict/{dep}/dashboard-config",
        f"/api/predict/{dep}/dashboard-metadata",
    ):
        resp = await anon_client.get(path)
        assert resp.status_code != 401, f"{path} -> {resp.status_code}"


@pytest.mark.asyncio
async def test_owner_scoped_deploy_reads_still_require_auth(anon_client):
    """Opening the public /api/predict/* mirrors must NOT open the owner-scoped
    /api/deploy/* management endpoints they shadow."""
    ids = _seed_owned_stack(A_OWNER)
    dep = ids["deployment"]
    for path in (
        f"/api/deploy/{dep}",
        f"/api/deploy/{dep}/presets",
        f"/api/deploy/{dep}/dashboard-config",
        f"/api/deploy/{dep}/dashboard-metadata",
    ):
        resp = await anon_client.get(path)
        assert resp.status_code == 401, f"{path} -> {resp.status_code}"


# ---------------------------------------------------------------------------
# 2. Cross-tenant access is forbidden (404)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_tenant_project_forbidden(client, second_client):
    created = await client.post("/api/projects", json={"name": "A secret"})
    assert created.status_code == 201
    pid = created.json()["id"]

    # Owner can read it.
    assert (await client.get(f"/api/projects/{pid}")).status_code == 200

    # Other tenant cannot read / update / delete it.
    assert (await second_client.get(f"/api/projects/{pid}")).status_code == 404
    assert (
        await second_client.patch(f"/api/projects/{pid}", json={"name": "z"})
    ).status_code == 404
    assert (await second_client.delete(f"/api/projects/{pid}")).status_code == 404


@pytest.mark.asyncio
async def test_list_projects_scoped_to_owner(client, second_client):
    await client.post("/api/projects", json={"name": "A-1"})
    await client.post("/api/projects", json={"name": "A-2"})
    await second_client.post("/api/projects", json={"name": "B-1"})

    a_ids = {p["id"] for p in (await client.get("/api/projects")).json()}
    b_list = (await second_client.get("/api/projects")).json()
    b_ids = {p["id"] for p in b_list}

    assert len(a_ids) == 2
    assert len(b_ids) == 1
    assert a_ids.isdisjoint(b_ids)


@pytest.mark.asyncio
async def test_cross_tenant_dataset_forbidden(client, second_client):
    proj = (await client.post("/api/projects", json={"name": "A data"})).json()
    sample = await client.post("/api/data/sample", json={"project_id": proj["id"]})
    assert sample.status_code == 201
    dataset_id = sample.json()["dataset_id"]

    assert (await client.get(f"/api/data/{dataset_id}/preview")).status_code == 200
    assert (
        await second_client.get(f"/api/data/{dataset_id}/preview")
    ).status_code == 404


@pytest.mark.asyncio
async def test_cross_tenant_merge_via_body_forbidden(client, second_client):
    """A caller must not merge ANOTHER tenant's datasets (body-keyed) into their
    own project. Source dataset ids come from the body, not the path."""
    # User A owns a dataset.
    proj_a = (await client.post("/api/projects", json={"name": "A merge"})).json()
    ds_a = (
        await client.post("/api/data/sample", json={"project_id": proj_a["id"]})
    ).json()["dataset_id"]

    # User B owns a destination project, and tries to pull in A's dataset.
    proj_b = (await second_client.post("/api/projects", json={"name": "B"})).json()
    resp = await second_client.post(
        f"/api/data/{proj_b['id']}/merge",
        json={
            "dataset_id_1": ds_a,
            "dataset_id_2": ds_a,
            "join_key": "product",
            "how": "inner",
        },
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cross_tenant_model_and_deployment_forbidden(client, second_client):
    ids = _seed_owned_stack(A_OWNER)

    # Owner reaches its own resources.
    assert (await client.get(f"/api/deploy/{ids['deployment']}")).status_code == 200
    assert (await client.get(f"/api/models/{ids['project']}/runs")).status_code == 200

    # Other tenant is blocked everywhere.
    assert (
        await second_client.get(f"/api/deploy/{ids['deployment']}")
    ).status_code == 404
    assert (
        await second_client.get(f"/api/models/{ids['project']}/runs")
    ).status_code == 404
    assert (
        await second_client.delete(f"/api/deploy/{ids['deployment']}")
    ).status_code == 404

    # List endpoints never surface another tenant's deployment.
    dep_ids = {d["id"] for d in (await second_client.get("/api/deployments")).json()}
    assert ids["deployment"] not in dep_ids
