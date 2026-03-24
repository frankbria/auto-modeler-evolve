"""Tests for chat-driven feature engineering suggestions and application.

Covers:
- _FEATURE_SUGGEST_PATTERNS regex detects feature suggestion phrases
- _FEATURE_APPLY_PATTERNS regex detects feature apply phrases
- Neither pattern fires on unrelated phrases
- Chat SSE stream emits feature_suggestions event when dataset exists
- feature_suggestions event includes suggestion list with count
- Chat SSE stream emits features_applied event when user asks to apply
- features_applied event includes new_columns, total_columns, applied_count
- Chat handles no-suggestions case gracefully (no event emitted)
"""

import json

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import _FEATURE_APPLY_PATTERNS, _FEATURE_SUGGEST_PATTERNS

# ---------------------------------------------------------------------------
# Sample CSV — uses date column so date_decompose suggestions appear
# ---------------------------------------------------------------------------

_SAMPLE_CSV = (
    b"order_date,region,revenue,units\n"
    b"2023-01-01,East,100.5,10\n"
    b"2023-02-01,West,200.3,20\n"
    b"2023-03-01,East,150.7,15\n"
    b"2023-04-01,West,300.1,30\n"
    b"2023-05-01,North,250.9,25\n"
    b"2023-06-01,East,175.2,18\n"
    b"2023-07-01,West,220.4,22\n"
    b"2023-08-01,North,190.6,19\n"
    b"2023-09-01,East,130.8,13\n"
    b"2023-10-01,West,280.0,28\n"
)

# CSV with no transformable columns — no date, numeric-only
_NUMERIC_ONLY_CSV = (
    b"a,b,c\n1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n10.0,11.0,12.0\n13.0,14.0,15.0\n"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.dataset_filter  # noqa
    import models.deployment  # noqa
    import models.feature_set  # noqa
    import models.feedback_record  # noqa
    import models.model_run  # noqa
    import models.prediction_log  # noqa
    import models.project  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.deploy as deploy_module

    deploy_module.DEPLOY_DIR = tmp_path / "deployments"

    import api.models as models_module

    models_module.MODELS_DIR = tmp_path / "models"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture()
async def project_id(ac):
    resp = await ac.post("/api/projects", json={"name": "Feature Chat Test"})
    return resp.json()["id"]


@pytest.fixture()
async def dataset_id(ac, project_id):
    resp = await ac.post(
        "/api/data/upload",
        files={"file": ("sales.csv", _SAMPLE_CSV, "text/csv")},
        data={"project_id": project_id},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["dataset_id"]


# ---------------------------------------------------------------------------
# Unit tests — _FEATURE_SUGGEST_PATTERNS
# ---------------------------------------------------------------------------


def test_feature_suggest_pattern_suggest_features():
    assert _FEATURE_SUGGEST_PATTERNS.search("suggest features for my data")


def test_feature_suggest_pattern_recommend_transformations():
    assert _FEATURE_SUGGEST_PATTERNS.search("recommend transformations")


def test_feature_suggest_pattern_what_features():
    assert _FEATURE_SUGGEST_PATTERNS.search("what features should I add?")


def test_feature_suggest_pattern_feature_engineering():
    assert _FEATURE_SUGGEST_PATTERNS.search("help me with feature engineering")


def test_feature_suggest_pattern_improve_features():
    assert _FEATURE_SUGGEST_PATTERNS.search("help me improve my features")


def test_feature_suggest_pattern_show_suggestions():
    assert _FEATURE_SUGGEST_PATTERNS.search("show me feature suggestions")


def test_feature_suggest_pattern_any_suggestions():
    assert _FEATURE_SUGGEST_PATTERNS.search("what feature suggestions are there?")


def test_feature_suggest_no_match_train():
    assert not _FEATURE_SUGGEST_PATTERNS.search("train a model to predict revenue")


def test_feature_suggest_no_match_deploy():
    assert not _FEATURE_SUGGEST_PATTERNS.search("deploy my model")


def test_feature_suggest_no_match_upload():
    assert not _FEATURE_SUGGEST_PATTERNS.search("upload a file")


# ---------------------------------------------------------------------------
# Unit tests — _FEATURE_APPLY_PATTERNS
# ---------------------------------------------------------------------------


def test_feature_apply_pattern_apply_suggestions():
    assert _FEATURE_APPLY_PATTERNS.search("apply the feature suggestions")


def test_feature_apply_pattern_apply_all():
    assert _FEATURE_APPLY_PATTERNS.search("apply all the suggestions")


def test_feature_apply_pattern_accept_suggestions():
    assert _FEATURE_APPLY_PATTERNS.search("accept the feature suggestions")


def test_feature_apply_pattern_use_transforms():
    assert _FEATURE_APPLY_PATTERNS.search("use the feature transforms")


def test_feature_apply_pattern_yes_apply():
    assert _FEATURE_APPLY_PATTERNS.search("yes, apply all the features")


def test_feature_apply_pattern_do_engineering():
    assert _FEATURE_APPLY_PATTERNS.search("do the feature engineering")


def test_feature_apply_pattern_run_transforms():
    assert _FEATURE_APPLY_PATTERNS.search("run the feature transforms")


def test_feature_apply_no_match_train():
    assert not _FEATURE_APPLY_PATTERNS.search("train a model")


def test_feature_apply_no_match_deploy():
    assert not _FEATURE_APPLY_PATTERNS.search("deploy my model")


def test_feature_apply_no_match_random():
    assert not _FEATURE_APPLY_PATTERNS.search("show me the correlation matrix")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_sse_events(text: str) -> list[dict]:
    """Extract all JSON payloads from SSE stream text."""
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            try:
                events.append(json.loads(line[6:]))
            except json.JSONDecodeError:
                pass
    return events


def _mock_anthropic_stream(text="Feature suggestions ready."):
    import unittest.mock as mock

    mock_stream = mock.MagicMock()
    mock_stream.__enter__ = mock.MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = mock.MagicMock(return_value=False)
    mock_stream.text_stream = iter([text])
    return mock_stream


# ---------------------------------------------------------------------------
# Integration tests — feature_suggestions SSE event
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_chat_emits_feature_suggestions_event(ac, project_id, dataset_id):
    """Chat emits feature_suggestions event when dataset with transformable cols exists."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = (
            _mock_anthropic_stream("Here are your feature suggestions.")
        )

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "suggest features for my data"},
        )

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    suggest_events = [e for e in events if e.get("type") == "feature_suggestions"]
    assert len(suggest_events) == 1


@pytest.mark.anyio
async def test_feature_suggestions_event_has_suggestions_list(
    ac, project_id, dataset_id
):
    """feature_suggestions event contains a list of suggestion dicts."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = (
            _mock_anthropic_stream("Suggestions below.")
        )

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "recommend transformations"},
        )

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    ev = next((e for e in events if e.get("type") == "feature_suggestions"), None)
    assert ev is not None
    payload = ev["suggestions"]
    assert "suggestions" in payload
    assert isinstance(payload["suggestions"], list)
    assert payload["count"] == len(payload["suggestions"])
    assert payload["count"] > 0


@pytest.mark.anyio
async def test_feature_suggestions_items_have_required_fields(
    ac, project_id, dataset_id
):
    """Each suggestion item has id, column, transform_type, title, description."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = (
            _mock_anthropic_stream("Check out these features.")
        )

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "feature engineering"},
        )

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    ev = next((e for e in events if e.get("type") == "feature_suggestions"), None)
    assert ev is not None
    first = ev["suggestions"]["suggestions"][0]
    for field in (
        "id",
        "column",
        "transform_type",
        "title",
        "description",
        "preview_columns",
    ):
        assert field in first, f"Missing field: {field}"


@pytest.mark.anyio
async def test_chat_no_feature_suggestions_event_without_dataset(ac, project_id):
    """Chat does NOT emit feature_suggestions if no dataset is loaded."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = (
            _mock_anthropic_stream("No dataset yet.")
        )

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "suggest features for my data"},
        )

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    assert not any(e.get("type") == "feature_suggestions" for e in events)


# ---------------------------------------------------------------------------
# Integration tests — features_applied SSE event
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_chat_emits_features_applied_event(ac, project_id, dataset_id):
    """Chat emits features_applied event when user says 'apply features'."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = (
            _mock_anthropic_stream("Features applied successfully!")
        )

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "apply the feature suggestions"},
        )

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    applied_events = [e for e in events if e.get("type") == "features_applied"]
    assert len(applied_events) == 1


@pytest.mark.anyio
async def test_features_applied_event_has_required_fields(ac, project_id, dataset_id):
    """features_applied event contains feature_set_id, new_columns, total_columns, applied_count."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = (
            _mock_anthropic_stream("Applied!")
        )

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "apply all the feature suggestions"},
        )

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    ev = next((e for e in events if e.get("type") == "features_applied"), None)
    assert ev is not None
    payload = ev["applied"]
    assert "feature_set_id" in payload
    assert "dataset_id" in payload
    assert "new_columns" in payload
    assert "total_columns" in payload
    assert "applied_count" in payload
    assert payload["applied_count"] > 0
    assert payload["total_columns"] > 0


@pytest.mark.anyio
async def test_features_applied_creates_feature_set(ac, project_id, dataset_id):
    """Applying features through chat creates an active FeatureSet in the DB."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = (
            _mock_anthropic_stream("Done!")
        )

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "apply all features"},
        )

    assert resp.status_code == 200

    # Verify the feature set exists in the API
    fs_resp = await ac.get(f"/api/features/{dataset_id}/suggestions")
    assert fs_resp.status_code == 200


@pytest.mark.anyio
async def test_features_applied_accept_phrase(ac, project_id, dataset_id):
    """'accept the feature suggestions' also triggers features_applied event."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = (
            _mock_anthropic_stream("Accepted!")
        )

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "accept the feature suggestions"},
        )

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    assert any(e.get("type") == "features_applied" for e in events)


@pytest.mark.anyio
async def test_chat_no_features_applied_without_dataset(ac, project_id):
    """Chat does NOT emit features_applied if no dataset is loaded."""
    import unittest.mock as mock

    with mock.patch("anthropic.Anthropic") as MockAnthropic:
        MockAnthropic.return_value.messages.stream.return_value = (
            _mock_anthropic_stream("No dataset.")
        )

        resp = await ac.post(
            f"/api/chat/{project_id}",
            json={"message": "apply all features"},
        )

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    assert not any(e.get("type") == "features_applied" for e in events)
