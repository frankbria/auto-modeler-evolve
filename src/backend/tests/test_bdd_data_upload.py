"""BDD step definitions for data upload scenarios.

Uses FastAPI's synchronous TestClient since pytest-bdd does not natively
support async step functions.
"""
import io
import pytest
from pytest_bdd import scenarios, given, when, then
from fastapi.testclient import TestClient
from sqlmodel import create_engine, SQLModel
import db as db_module

scenarios("features/data_upload.feature")

SAMPLE_CSV = b"""date,product,region,revenue,units
2024-01-01,Widget A,North,1200.50,10
2024-01-01,Widget B,South,850.00,8
2024-01-02,Widget A,East,2100.75,18
2024-01-02,Widget C,West,450.25,4
2024-01-03,Widget B,North,1650.00,15
"""


@pytest.fixture
def client(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.project  # noqa
    import models.dataset  # noqa
    import models.conversation  # noqa
    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    data_module.UPLOAD_DIR = tmp_path / "uploads"

    from main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def context():
    return {}


@given("a project exists")
def a_project_exists(client, context):
    resp = client.post("/api/projects", json={"name": "BDD Test Project"})
    assert resp.status_code == 201
    context["project_id"] = resp.json()["id"]


@given("a project with an uploaded CSV dataset")
def a_project_with_dataset(client, context):
    resp = client.post("/api/projects", json={"name": "BDD Dataset Project"})
    context["project_id"] = resp.json()["id"]
    upload = client.post(
        "/api/data/upload",
        data={"project_id": context["project_id"]},
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    assert upload.status_code == 201, upload.text
    context["dataset_id"] = upload.json()["dataset_id"]


@when("I upload a CSV file with sales data")
def upload_csv(client, context):
    resp = client.post(
        "/api/data/upload",
        data={"project_id": context["project_id"]},
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    context["upload_response"] = resp
    context["upload_data"] = resp.json()


@when("I upload a non-CSV file")
def upload_non_csv(client, context):
    resp = client.post(
        "/api/data/upload",
        data={"project_id": context["project_id"]},
        files={"file": ("data.pdf", b"fake pdf", "application/pdf")},
    )
    context["upload_response"] = resp


@when("I request the dataset preview")
def request_preview(client, context):
    resp = client.get(f"/api/data/{context['dataset_id']}/preview")
    context["preview_response"] = resp
    context["preview_data"] = resp.json()


@then("the dataset is created with correct row and column counts")
def check_row_column_counts(context):
    assert context["upload_response"].status_code == 201
    data = context["upload_data"]
    assert data["row_count"] == 5
    assert data["column_count"] == 5


@then("the preview contains the first rows of data")
def check_preview_rows(context):
    data = context["upload_data"]
    assert len(data["preview"]) == 5
    assert "product" in data["preview"][0]


@then("each column has statistics")
def check_column_stats(context):
    data = context["upload_data"]
    assert len(data["column_stats"]) == 5
    for col in data["column_stats"]:
        assert "name" in col
        assert "dtype" in col
        assert "null_count" in col


@then("the upload is rejected with an error")
def check_upload_rejected(context):
    assert context["upload_response"].status_code == 400


@then("I receive the column statistics and sample rows")
def check_preview_data(context):
    assert context["preview_response"].status_code == 200
    data = context["preview_data"]
    assert "column_stats" in data
    assert "preview" in data
    assert len(data["column_stats"]) > 0
