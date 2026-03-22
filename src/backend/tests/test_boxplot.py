"""Integration tests for GET /api/data/{dataset_id}/boxplot endpoint.

Tests cover:
- Happy path: single column, grouped column
- Error paths: 404 (dataset not found), 400 (column not found), 400 (non-numeric column),
               400 (groupby not found)
- Chart spec shape validation
"""

import io
import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module

SAMPLE_CSV = (
    b"product,region,revenue,units\n"
    b"Widget A,North,1200.0,10\n"
    b"Widget B,South,850.0,8\n"
    b"Widget A,East,2100.0,18\n"
    b"Widget C,West,450.0,4\n"
    b"Widget B,North,1650.0,15\n"
    b"Widget A,South,980.0,9\n"
    b"Widget C,East,3200.0,28\n"
    b"Widget B,West,760.0,7\n"
)


@pytest.fixture
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.feature_set  # noqa
    import models.model_run  # noqa
    import models.project  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
async def project_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Boxplot Test"})
    assert resp.status_code == 201
    return resp.json()["id"]


@pytest.fixture
async def dataset_id(ac, project_id):
    resp = await ac.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    assert resp.status_code == 201
    return resp.json()["dataset_id"]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestBoxplotEndpointHappy:
    async def test_single_column_returns_boxplot_spec(self, ac, dataset_id):
        resp = await ac.get(f"/api/data/{dataset_id}/boxplot?column=revenue")
        assert resp.status_code == 200
        body = resp.json()
        assert body["chart_type"] == "boxplot"
        assert len(body["data"]) == 1
        assert body["data"][0]["group"] == "revenue"

    async def test_grouped_returns_one_box_per_region(self, ac, dataset_id):
        resp = await ac.get(
            f"/api/data/{dataset_id}/boxplot?column=revenue&groupby=region"
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["chart_type"] == "boxplot"
        groups = {r["group"] for r in body["data"]}
        assert groups == {"North", "South", "East", "West"}

    async def test_grouped_sorted_by_median_desc(self, ac, dataset_id):
        resp = await ac.get(
            f"/api/data/{dataset_id}/boxplot?column=revenue&groupby=region"
        )
        body = resp.json()
        medians = [r["median"] for r in body["data"]]
        assert medians == sorted(medians, reverse=True)

    async def test_each_box_has_five_number_summary(self, ac, dataset_id):
        resp = await ac.get(f"/api/data/{dataset_id}/boxplot?column=revenue")
        body = resp.json()
        box = body["data"][0]
        for key in ("min", "q1", "median", "q3", "max"):
            assert key in box, f"Missing key: {key}"

    async def test_y_label_is_value_column(self, ac, dataset_id):
        resp = await ac.get(f"/api/data/{dataset_id}/boxplot?column=revenue")
        body = resp.json()
        assert body["y_label"] == "revenue"

    async def test_x_label_is_group_column_when_grouped(self, ac, dataset_id):
        resp = await ac.get(
            f"/api/data/{dataset_id}/boxplot?column=revenue&groupby=region"
        )
        body = resp.json()
        assert body["x_label"] == "region"

    async def test_units_column_also_works(self, ac, dataset_id):
        resp = await ac.get(f"/api/data/{dataset_id}/boxplot?column=units")
        assert resp.status_code == 200
        assert resp.json()["chart_type"] == "boxplot"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestBoxplotEndpointErrors:
    async def test_dataset_not_found_returns_404(self, ac):
        resp = await ac.get("/api/data/nonexistent-id/boxplot?column=revenue")
        assert resp.status_code == 404

    async def test_column_not_found_returns_400(self, ac, dataset_id):
        resp = await ac.get(f"/api/data/{dataset_id}/boxplot?column=no_such_column")
        assert resp.status_code == 400
        assert "no_such_column" in resp.json()["detail"]

    async def test_non_numeric_column_returns_400(self, ac, dataset_id):
        resp = await ac.get(f"/api/data/{dataset_id}/boxplot?column=region")
        assert resp.status_code == 400
        assert "not numeric" in resp.json()["detail"].lower()

    async def test_groupby_not_found_returns_400(self, ac, dataset_id):
        resp = await ac.get(
            f"/api/data/{dataset_id}/boxplot?column=revenue&groupby=missing_col"
        )
        assert resp.status_code == 400
        assert "missing_col" in resp.json()["detail"]
