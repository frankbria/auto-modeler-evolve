"""Regression test for issue #23: data endpoints must not leak raw exception
detail to clients on a 500. The dataset-read failure path returns a generic
message; the underlying error is logged server-side, not echoed in the body.
"""

import io

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, Session, create_engine, select

import db as db_module

SAMPLE_CSV = b"product,revenue\nWidget A,1200.0\nWidget B,850.0\n"


@pytest.fixture
async def ac(tmp_path):
    db_module.engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", echo=False)
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
async def dataset_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Leak Test"})
    project_id = resp.json()["id"]
    resp = await ac.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    assert resp.status_code == 201
    return resp.json()["dataset_id"]


async def test_unreadable_dataset_returns_generic_500_without_exception_detail(
    ac, dataset_id
):
    from models.dataset import Dataset

    # Corrupt the backing file so pandas raises during read.
    with Session(db_module.engine) as session:
        ds = session.exec(select(Dataset).where(Dataset.id == dataset_id)).one()
        file_path = ds.file_path
    with open(file_path, "w") as fh:
        fh.write("")  # empty file -> pandas EmptyDataError

    resp = await ac.get(f"/api/data/{dataset_id}/boxplot?column=revenue")

    assert resp.status_code == 500
    detail = resp.json()["detail"]
    # Generic message only — no interpolated exception text or file path.
    assert detail == "Could not read dataset"
    assert "EmptyData" not in detail
    assert file_path not in detail
