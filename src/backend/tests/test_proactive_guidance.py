"""Tests for proactive data-aware upload suggestions and next-step guidance chips.

Covers:
- generate_upload_suggestions: data-aware chip generation from profile
- get_next_step_chips: per-state action chips
- Upload API response includes suggestions
- Sample load API response includes suggestions
- Training stream all_done event includes next_step_chips
"""

import io
import json
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from chat.orchestrator import generate_upload_suggestions, get_next_step_chips

# ---------------------------------------------------------------------------
# Sample CSVs
# ---------------------------------------------------------------------------

SALES_CSV = b"""date,product,region,revenue,units
2024-01-01,Widget A,East,1200.50,10
2024-02-01,Widget B,West,850.00,8
2024-03-01,Widget A,East,2100.75,18
2024-04-01,Widget C,West,450.25,4
2024-05-01,Widget B,North,1650.00,15
2024-06-01,Widget A,South,1900.00,20
2024-07-01,Widget B,East,700.00,7
2024-08-01,Widget C,North,550.00,5
2024-09-01,Widget A,West,2200.00,22
2024-10-01,Widget B,South,1100.00,11
"""

REGRESSION_CSV = b"""f1,f2,f3,target
1.0,0.5,100.0,10.0
2.0,1.0,200.0,20.0
3.0,1.5,300.0,30.0
4.0,2.0,400.0,40.0
5.0,2.5,500.0,50.0
6.0,3.0,600.0,60.0
7.0,3.5,700.0,70.0
8.0,4.0,800.0,80.0
9.0,4.5,900.0,90.0
10.0,5.0,1000.0,100.0
11.0,5.5,1100.0,110.0
12.0,6.0,1200.0,120.0
"""


# ---------------------------------------------------------------------------
# Client fixture
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

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    import api.models as models_api_module

    models_api_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# generate_upload_suggestions — pure function tests
# ---------------------------------------------------------------------------


def _make_profile(columns=None, correlations=None, insights=None):
    return {
        "columns": columns or [],
        "correlations": correlations or {},
        "insights": insights or [],
    }


def test_upload_suggestions_date_and_numeric():
    """Date + numeric columns → trend question appears."""
    profile = _make_profile(
        columns=[
            {"name": "date", "dtype": "datetime64[ns]", "null_pct": 0},
            {"name": "revenue", "dtype": "float64", "null_pct": 0},
        ]
    )
    result = generate_upload_suggestions(profile, ["date", "revenue"])
    assert any("revenue" in s and "trend" in s.lower() for s in result)


def test_upload_suggestions_strong_correlation():
    """Strong correlation pair → relationship question appears."""
    profile = _make_profile(
        columns=[
            {"name": "cost", "dtype": "float64", "null_pct": 0},
            {"name": "price", "dtype": "float64", "null_pct": 0},
        ],
        correlations={
            "pairs": [{"col_a": "cost", "col_b": "price", "correlation": 0.87}]
        },
    )
    result = generate_upload_suggestions(profile, ["cost", "price"])
    assert any("cost" in s and "price" in s for s in result)


def test_upload_suggestions_cat_and_numeric():
    """Categorical + numeric → group-by question appears."""
    profile = _make_profile(
        columns=[
            {"name": "region", "dtype": "object", "null_pct": 0},
            {"name": "sales", "dtype": "float64", "null_pct": 0},
        ]
    )
    result = generate_upload_suggestions(profile, ["region", "sales"])
    assert any("sales" in s and "region" in s for s in result)


def test_upload_suggestions_missing_values():
    """Missing values > 5% → cleaning question appears."""
    profile = _make_profile(
        columns=[
            {"name": "revenue", "dtype": "float64", "null_pct": 15},
            {"name": "cost", "dtype": "float64", "null_pct": 0},
        ]
    )
    result = generate_upload_suggestions(profile, ["revenue", "cost"])
    assert any("missing" in s.lower() for s in result)


def test_upload_suggestions_no_columns_fallback():
    """Empty profile returns at least one suggestion (walk-through fallback)."""
    profile = _make_profile(columns=[])
    result = generate_upload_suggestions(profile, [])
    assert len(result) >= 1
    assert any(
        "walk" in s.lower() or "summary" in s.lower() or "interesting" in s.lower()
        for s in result
    )


def test_upload_suggestions_max_five():
    """Never returns more than 5 suggestions even for rich profiles."""
    profile = _make_profile(
        columns=[
            {"name": "date", "dtype": "datetime64[ns]", "null_pct": 10},
            {"name": "revenue", "dtype": "float64", "null_pct": 10},
            {"name": "cost", "dtype": "float64", "null_pct": 0},
            {"name": "region", "dtype": "object", "null_pct": 0},
            {"name": "product", "dtype": "object", "null_pct": 0},
        ],
        correlations={
            "pairs": [{"col_a": "revenue", "col_b": "cost", "correlation": 0.92}]
        },
        insights=[{"title": "Pattern", "detail": "Something"}],
    )
    result = generate_upload_suggestions(
        profile, ["date", "revenue", "cost", "region", "product"]
    )
    assert len(result) <= 5


def test_upload_suggestions_weak_correlation_skipped():
    """Correlation below 0.5 should not generate a relationship question."""
    profile = _make_profile(
        columns=[
            {"name": "a", "dtype": "float64", "null_pct": 0},
            {"name": "b", "dtype": "float64", "null_pct": 0},
        ],
        correlations={"pairs": [{"col_a": "a", "col_b": "b", "correlation": 0.2}]},
    )
    result = generate_upload_suggestions(profile, ["a", "b"])
    assert not any("relate" in s.lower() for s in result)


def test_upload_suggestions_name_hint_date_detection():
    """Columns named 'month', 'year', 'period' are treated as date columns."""
    profile = _make_profile(
        columns=[
            {"name": "month", "dtype": "int64", "null_pct": 0},
            {"name": "sales", "dtype": "float64", "null_pct": 0},
        ]
    )
    result = generate_upload_suggestions(profile, ["month", "sales"])
    assert any("sales" in s and "trend" in s.lower() for s in result)


def test_upload_suggestions_returns_list_of_strings():
    """Return value is always a list of non-empty strings."""
    profile = _make_profile(columns=[{"name": "x", "dtype": "float64", "null_pct": 0}])
    result = generate_upload_suggestions(profile, ["x"])
    assert isinstance(result, list)
    assert all(isinstance(s, str) and len(s) > 0 for s in result)


# ---------------------------------------------------------------------------
# get_next_step_chips — pure function tests
# ---------------------------------------------------------------------------


def test_next_step_chips_explore():
    chips = get_next_step_chips("explore")
    assert len(chips) == 3
    assert any(
        "model" in c.lower() or "predict" in c.lower() or "feature" in c.lower()
        for c in chips
    )


def test_next_step_chips_shape():
    chips = get_next_step_chips("shape")
    assert len(chips) == 3
    assert any("train" in c.lower() or "model" in c.lower() for c in chips)


def test_next_step_chips_validate():
    chips = get_next_step_chips("validate")
    assert len(chips) == 3
    assert any("deploy" in c.lower() or "share" in c.lower() for c in chips)


def test_next_step_chips_deploy():
    chips = get_next_step_chips("deploy")
    assert len(chips) == 3
    assert any(
        "share" in c.lower()
        or "team" in c.lower()
        or "api" in c.lower()
        or "dashboard" in c.lower()
        or "link" in c.lower()
        for c in chips
    )


def test_next_step_chips_unknown_state_returns_empty():
    """Unknown state returns an empty list rather than crashing."""
    chips = get_next_step_chips("nonexistent_state")
    assert chips == []


def test_next_step_chips_all_strings():
    """All chip values are non-empty strings."""
    for state in ("explore", "shape", "validate", "deploy"):
        chips = get_next_step_chips(state)
        assert all(isinstance(c, str) and len(c) > 0 for c in chips)


# ---------------------------------------------------------------------------
# Upload API includes suggestions
# ---------------------------------------------------------------------------


def test_upload_returns_suggestions(client):
    """Upload endpoint response includes a suggestions list."""
    proj = client.post("/api/projects", json={"name": "TestProj"})
    assert proj.status_code == 201
    proj_id = proj.json()["id"]

    resp = client.post(
        "/api/data/upload",
        files={"file": ("sales.csv", io.BytesIO(SALES_CSV), "text/csv")},
        data={"project_id": proj_id},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert "suggestions" in body
    assert isinstance(body["suggestions"], list)
    assert len(body["suggestions"]) >= 1
    assert all(isinstance(s, str) for s in body["suggestions"])


def test_upload_suggestions_reference_actual_columns(client):
    """Suggestions for a date+numeric dataset should mention those columns."""
    proj = client.post("/api/projects", json={"name": "TestProj2"})
    proj_id = proj.json()["id"]

    resp = client.post(
        "/api/data/upload",
        files={"file": ("sales.csv", io.BytesIO(SALES_CSV), "text/csv")},
        data={"project_id": proj_id},
    )
    body = resp.json()
    # SALES_CSV has 'date' and 'revenue' — expect trend suggestion
    suggestions_text = " ".join(body["suggestions"]).lower()
    assert "revenue" in suggestions_text or "trend" in suggestions_text


def test_sample_returns_suggestions(client):
    """Load-sample endpoint response includes a suggestions list."""
    proj = client.post("/api/projects", json={"name": "SampleTest"})
    proj_id = proj.json()["id"]

    resp = client.post("/api/data/sample", json={"project_id": proj_id})
    assert resp.status_code == 201
    body = resp.json()
    assert "suggestions" in body
    assert isinstance(body["suggestions"], list)
    assert len(body["suggestions"]) >= 1


# ---------------------------------------------------------------------------
# Training stream all_done includes next_step_chips
# ---------------------------------------------------------------------------


def test_training_stream_all_done_has_chips(client):
    """all_done event in training SSE stream includes next_step_chips."""
    # Set up project + dataset + feature set + target
    proj = client.post("/api/projects", json={"name": "TrainTest"})
    proj_id = proj.json()["id"]

    upload = client.post(
        "/api/data/upload",
        files={"file": ("data.csv", io.BytesIO(REGRESSION_CSV), "text/csv")},
        data={"project_id": proj_id},
    )
    ds_id = upload.json()["dataset_id"]

    # Create feature set with target
    client.post(f"/api/features/{ds_id}/apply", json={"transformations": []})
    client.post(f"/api/features/{ds_id}/target", json={"target_column": "target"})

    # Start training
    train_resp = client.post(
        f"/api/models/{proj_id}/train",
        json={"algorithms": ["linear_regression"], "excluded_features": None},
    )
    assert train_resp.status_code in (200, 202)

    # Wait briefly for background thread to complete, then consume SSE stream
    time.sleep(3)

    # GET the stream — with TestClient this reads the full body synchronously
    resp = client.get(f"/api/models/{proj_id}/training-stream")
    assert resp.status_code == 200

    # Parse all events from the SSE response body
    all_done_event = None
    for line in resp.text.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            data = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        if data.get("type") == "all_done":
            all_done_event = data
            break

    assert all_done_event is not None, "No all_done event received in SSE stream"
    assert "next_step_chips" in all_done_event
    assert isinstance(all_done_event["next_step_chips"], list)
    assert len(all_done_event["next_step_chips"]) == 3
    # Chips should guide toward validation
    chips_text = " ".join(all_done_event["next_step_chips"]).lower()
    assert "deploy" in chips_text or "model" in chips_text or "share" in chips_text
