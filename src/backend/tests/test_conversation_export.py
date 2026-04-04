"""Tests for conversation export as HTML report.

Covers:
- _CONV_EXPORT_PATTERNS detection
- _build_export_html() pure function
- GET /api/chat/{project_id}/export endpoint (200, 404, with model, empty conversation)
"""

import json

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel


# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def test_conv_export_pattern_export_conversation():
    from api.chat import _CONV_EXPORT_PATTERNS

    assert _CONV_EXPORT_PATTERNS.search("export this conversation")
    assert _CONV_EXPORT_PATTERNS.search("download the conversation")
    assert _CONV_EXPORT_PATTERNS.search("save this analysis")


def test_conv_export_pattern_share_report():
    from api.chat import _CONV_EXPORT_PATTERNS

    assert _CONV_EXPORT_PATTERNS.search("share this report")
    assert _CONV_EXPORT_PATTERNS.search("generate a report")
    assert _CONV_EXPORT_PATTERNS.search("export my analysis")


def test_conv_export_pattern_summary():
    from api.chat import _CONV_EXPORT_PATTERNS

    assert _CONV_EXPORT_PATTERNS.search("download a summary of this chat")
    assert _CONV_EXPORT_PATTERNS.search("export my findings")
    assert _CONV_EXPORT_PATTERNS.search("share my findings with the team")


def test_conv_export_pattern_no_match():
    from api.chat import _CONV_EXPORT_PATTERNS

    assert not _CONV_EXPORT_PATTERNS.search("show me a chart")
    assert not _CONV_EXPORT_PATTERNS.search("train a model")
    assert not _CONV_EXPORT_PATTERNS.search("what is r squared?")


# ---------------------------------------------------------------------------
# _build_export_html pure function tests
# ---------------------------------------------------------------------------


def _make_project(name="Test Project"):
    from models.project import Project

    return Project(id="test-proj", name=name)


def _make_dataset():
    from models.dataset import Dataset

    return Dataset(
        id="test-ds",
        project_id="test-proj",
        filename="sales_data.csv",
        file_path="/tmp/sales.csv",
        row_count=500,
        column_count=8,
    )


def _make_run(algorithm="random_forest_regressor", r2=0.87):
    from models.model_run import ModelRun

    return ModelRun(
        id="run-1",
        project_id="test-proj",
        algorithm=algorithm,
        metrics=json.dumps({"r2": r2, "mae": 12.5}),
        summary="This model explains 87% of variation.",
        status="done",
    )


def test_build_export_html_contains_project_name():
    from api.chat import _build_export_html

    project = _make_project("My Sales Analysis")
    html = _build_export_html(project, None, None, [])
    assert "My Sales Analysis" in html


def test_build_export_html_contains_dataset_info():
    from api.chat import _build_export_html

    project = _make_project()
    dataset = _make_dataset()
    html = _build_export_html(project, dataset, None, [])
    assert "sales_data.csv" in html
    assert "500" in html


def test_build_export_html_contains_model_results():
    from api.chat import _build_export_html

    project = _make_project()
    run = _make_run()
    html = _build_export_html(project, None, run, [])
    assert "Model Results" in html
    assert "0.870" in html
    assert "87% of variation" in html  # from summary


def test_build_export_html_renders_messages():
    from api.chat import _build_export_html

    project = _make_project()
    messages = [
        {"role": "user", "content": "What are the top selling products?"},
        {"role": "assistant", "content": "Here are the top selling products..."},
    ]
    html = _build_export_html(project, None, None, messages)
    assert "What are the top selling products?" in html
    assert "Here are the top selling products" in html
    assert "You" in html
    assert "AutoModeler" in html


def test_build_export_html_escapes_content():
    from api.chat import _build_export_html

    project = _make_project()
    messages = [
        {"role": "user", "content": "<script>alert('xss')</script>"},
    ]
    html = _build_export_html(project, None, None, messages)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_build_export_html_empty_conversation():
    from api.chat import _build_export_html

    project = _make_project()
    html = _build_export_html(project, None, None, [])
    assert "No messages" in html
    assert "DOCTYPE html" in html


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


async def _setup_db(tmp_path, db_name: str):
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / f"{db_name}.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)
    return test_db


@pytest.mark.anyio
async def test_export_returns_html(tmp_path, set_test_env):
    from main import app
    import db
    from models.conversation import Conversation
    from models.project import Project
    from sqlmodel import create_engine

    test_db = str(tmp_path / "export_html.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)

    project_id = "exp-html-1"
    with next(db.get_session()) as session:
        session.merge(Project(id=project_id, name="Revenue Analysis"))
        msgs = json.dumps([
            {"role": "user", "content": "Show me the trend"},
            {"role": "assistant", "content": "The trend is upward."},
        ])
        session.merge(Conversation(id="conv-1", project_id=project_id, messages=msgs))
        session.commit()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get(f"/api/chat/{project_id}/export")

    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Revenue Analysis" in resp.text
    assert "The trend is upward" in resp.text


@pytest.mark.anyio
async def test_export_returns_attachment_header(tmp_path, set_test_env):
    from main import app
    import db
    from models.project import Project
    from sqlmodel import create_engine

    test_db = str(tmp_path / "export_hdr.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)

    project_id = "exp-hdr-1"
    with next(db.get_session()) as session:
        session.merge(Project(id=project_id, name="My Project"))
        session.commit()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get(f"/api/chat/{project_id}/export")

    assert resp.status_code == 200
    assert "content-disposition" in resp.headers
    assert "attachment" in resp.headers["content-disposition"]
    assert ".html" in resp.headers["content-disposition"]


@pytest.mark.anyio
async def test_export_unknown_project_returns_404(tmp_path, set_test_env):
    from main import app
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / "export_404.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/chat/no-such-project/export")

    assert resp.status_code == 404


@pytest.mark.anyio
async def test_export_includes_model_info(tmp_path, set_test_env):
    from main import app
    import db
    from models.model_run import ModelRun
    from models.project import Project
    from sqlmodel import create_engine

    test_db = str(tmp_path / "export_model.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)

    project_id = "exp-model-1"
    with next(db.get_session()) as session:
        session.merge(Project(id=project_id, name="Churn Project"))
        run = ModelRun(
            id="run-export-1",
            project_id=project_id,
            algorithm="logistic_regression",
            metrics=json.dumps({"accuracy": 0.91}),
            summary="91% accuracy on test data.",
            status="done",
            is_selected=True,
        )
        session.add(run)
        session.commit()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get(f"/api/chat/{project_id}/export")

    assert resp.status_code == 200
    assert "Model Results" in resp.text
    assert "0.910" in resp.text
    assert "91% accuracy" in resp.text
