"""Tests for Cross-Project Model Comparison.

Covers:
- Pattern detection: _CROSS_PROJECT_PATTERNS
- Pure function: compute_cross_project_comparison
- Normalisation helper: _normalize_metric
- API endpoint: GET /api/projects/cross-comparison
- Chat SSE integration: cross_project_comparison event emitted on match
"""

import json
import uuid
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import _CROSS_PROJECT_PATTERNS
from core.advisor import _normalize_metric, compute_cross_project_comparison

# ---------------------------------------------------------------------------
# Pattern detection tests
# ---------------------------------------------------------------------------


def test_cross_project_pattern_compare_my_models():
    assert _CROSS_PROJECT_PATTERNS.search("compare my revenue model vs my churn model")


def test_cross_project_pattern_cross_project_comparison():
    assert _CROSS_PROJECT_PATTERNS.search("cross-project comparison")


def test_cross_project_pattern_compare_models_across_all_projects():
    assert _CROSS_PROJECT_PATTERNS.search("compare models across my projects")


def test_cross_project_pattern_head_to_head():
    assert _CROSS_PROJECT_PATTERNS.search("head-to-head project comparison")


def test_cross_project_pattern_which_model_is_best_overall():
    assert _CROSS_PROJECT_PATTERNS.search("which of my models performs best overall")


def test_cross_project_pattern_rank_all_my_models():
    assert _CROSS_PROJECT_PATTERNS.search("rank all my models side by side")


def test_cross_project_pattern_how_do_models_compare():
    assert _CROSS_PROJECT_PATTERNS.search(
        "how do all my models compare to each other across projects"
    )


def test_cross_project_pattern_no_match_train():
    assert not _CROSS_PROJECT_PATTERNS.search("train a random forest")


def test_cross_project_pattern_no_match_show_chart():
    assert not _CROSS_PROJECT_PATTERNS.search("show me a bar chart of revenue")


# ---------------------------------------------------------------------------
# _normalize_metric unit tests
# ---------------------------------------------------------------------------


def test_normalize_metric_r2():
    assert _normalize_metric("r2", 0.85) == 85.0


def test_normalize_metric_accuracy():
    assert _normalize_metric("accuracy", 0.92) == 92.0


def test_normalize_metric_f1():
    assert _normalize_metric("f1", 0.75) == 75.0


def test_normalize_metric_mae_invert():
    # MAE = 1.0 → 1/(1+1)*100 = 50
    assert _normalize_metric("mae", 1.0) == 50.0


def test_normalize_metric_mae_small():
    # MAE = 0 → 1/(1+0)*100 = 100
    assert _normalize_metric("mae", 0.0) == 100.0


def test_normalize_metric_none_returns_zero():
    assert _normalize_metric("r2", None) == 0.0


def test_normalize_metric_clamp_above_100():
    assert _normalize_metric("r2", 1.5) == 100.0


def test_normalize_metric_clamp_below_zero():
    assert _normalize_metric("r2", -0.5) == 0.0


# ---------------------------------------------------------------------------
# compute_cross_project_comparison pure function tests
# ---------------------------------------------------------------------------


def _make_project(
    name,
    model_count=1,
    best_metric_name="r2",
    best_metric_value=None,
    best_problem_type="regression",
    best_target_column="revenue",
    best_algorithm="random_forest",
    has_deployment=False,
    prediction_count=0,
):
    return {
        "project_id": str(uuid.uuid4()),
        "name": name,
        "model_count": model_count,
        "best_algorithm": best_algorithm if model_count > 0 else None,
        "best_metric_name": best_metric_name if model_count > 0 else None,
        "best_metric_value": best_metric_value,
        "best_problem_type": best_problem_type if model_count > 0 else None,
        "best_target_column": best_target_column if model_count > 0 else None,
        "has_deployment": has_deployment,
        "prediction_count": prediction_count,
    }


def test_cross_compare_empty_list():
    result = compute_cross_project_comparison([])
    assert result["n_projects"] == 0
    assert result["n_with_models"] == 0
    assert result["winner"] is None
    assert result["projects_compared"] == []
    assert result["insights"] == []
    assert "No trained models" in result["summary"]


def test_cross_compare_no_models():
    summaries = [_make_project("A", model_count=0, best_metric_value=None)]
    result = compute_cross_project_comparison(summaries)
    assert result["n_with_models"] == 0
    assert result["winner"] is None


def test_cross_compare_single_project():
    summaries = [_make_project("Revenue", best_metric_value=0.88)]
    result = compute_cross_project_comparison(summaries)
    assert result["n_with_models"] == 1
    assert result["winner"]["name"] == "Revenue"
    assert result["winner"]["performance_score"] == pytest.approx(88.0)
    assert result["winner"]["rank"] == 1


def test_cross_compare_ranking():
    summaries = [
        _make_project("Low", best_metric_value=0.60),
        _make_project("High", best_metric_value=0.90),
        _make_project("Mid", best_metric_value=0.75),
    ]
    result = compute_cross_project_comparison(summaries)
    rows = result["projects_compared"]
    assert rows[0]["name"] == "High"
    assert rows[0]["rank"] == 1
    assert rows[1]["name"] == "Mid"
    assert rows[2]["name"] == "Low"


def test_cross_compare_winner_is_top_ranked():
    summaries = [
        _make_project("A", best_metric_value=0.72),
        _make_project("B", best_metric_value=0.95),
    ]
    result = compute_cross_project_comparison(summaries)
    assert result["winner"]["name"] == "B"


def test_cross_compare_n_projects_total():
    summaries = [
        _make_project("A", best_metric_value=0.80),
        _make_project("B", model_count=0, best_metric_value=None),
    ]
    result = compute_cross_project_comparison(summaries)
    assert result["n_projects"] == 2
    assert result["n_with_models"] == 1


def test_cross_compare_insights_clear_leader():
    summaries = [
        _make_project("Dominant", best_metric_value=0.95),
        _make_project("Weak", best_metric_value=0.50),
    ]
    result = compute_cross_project_comparison(summaries)
    assert any("leads by" in i or "clear winner" in i for i in result["insights"])


def test_cross_compare_insights_close_match():
    summaries = [
        _make_project("A", best_metric_value=0.91),
        _make_project("B", best_metric_value=0.90),
    ]
    result = compute_cross_project_comparison(summaries)
    assert any("close" in i or "very close" in i for i in result["insights"])


def test_cross_compare_deployment_insight():
    summaries = [
        _make_project("Deployed", best_metric_value=0.88, has_deployment=True),
        _make_project("NotDeployed", best_metric_value=0.80, has_deployment=False),
    ]
    result = compute_cross_project_comparison(summaries)
    assert any(
        "deployment" in i.lower() or "NotDeployed" in i for i in result["insights"]
    )


def test_cross_compare_summary_text():
    summaries = [_make_project("Revenue", best_metric_value=0.88)]
    result = compute_cross_project_comparison(summaries)
    assert "Revenue" in result["summary"]
    assert "1 project" in result["summary"]


def test_cross_compare_row_has_algorithm_plain():
    summaries = [_make_project("A", best_metric_value=0.80)]
    result = compute_cross_project_comparison(summaries)
    row = result["projects_compared"][0]
    assert row["algorithm_plain"]  # not empty
    assert "_" not in row["algorithm_plain"]  # humanised


def test_cross_compare_mixed_problem_types():
    summaries = [
        _make_project("Reg", best_metric_value=0.82, best_problem_type="regression"),
        _make_project(
            "Cls",
            best_metric_value=0.89,
            best_metric_name="accuracy",
            best_problem_type="classification",
        ),
    ]
    result = compute_cross_project_comparison(summaries)
    assert result["n_with_models"] == 2
    # regression and classification insights generated
    assert any(
        "regression" in i.lower() or "classification" in i.lower()
        for i in result["insights"]
    )


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    SQLModel.metadata.create_all(engine)
    db_module.engine = engine

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    import api.models as models_api_module

    models_api_module.MODELS_DIR = tmp_path / "models"

    from main import app

    with TestClient(app) as c:
        yield c


def test_cross_comparison_endpoint_returns_200(client):
    response = client.get("/api/projects/cross-comparison")
    assert response.status_code == 200


def test_cross_comparison_endpoint_empty_returns_zero(client):
    response = client.get("/api/projects/cross-comparison")
    data = response.json()
    assert data["n_projects"] == 0
    assert data["n_with_models"] == 0
    assert data["winner"] is None
    assert data["projects_compared"] == []


def test_cross_comparison_endpoint_with_project(client):
    resp = client.post("/api/projects", json={"name": "CrossTest"})
    assert resp.status_code == 201

    response = client.get("/api/projects/cross-comparison")
    assert response.status_code == 200
    data = response.json()
    assert data["n_projects"] == 1
    # No model trained, so n_with_models = 0
    assert data["n_with_models"] == 0


# ---------------------------------------------------------------------------
# Chat SSE integration test
# ---------------------------------------------------------------------------


def _chat(client, project_id, message):
    """Send a chat message and return parsed SSE events."""
    mock_stream = iter(["Here is the cross-project comparison."])
    with patch("anthropic.Anthropic") as mock_cls:
        mock_inst = mock_cls.return_value
        mock_stream_cm = mock_inst.messages.stream.return_value.__enter__.return_value
        mock_stream_cm.text_stream = mock_stream
        with client.stream(
            "POST",
            f"/api/chat/{project_id}",
            json={"message": message, "conversation_id": None},
            headers={"accept": "text/event-stream"},
        ) as resp:
            events = []
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    try:
                        events.append(json.loads(line[6:]))
                    except Exception:
                        pass
    return events


def test_chat_emits_cross_project_comparison_event(client):
    """SSE chat response includes cross_project_comparison event on pattern match."""
    # Create a project (no model trained, but pattern still fires and returns empty comparison)
    resp = client.post("/api/projects", json={"name": "Alpha"})
    project_id = resp.json()["id"]

    events = _chat(client, project_id, "cross-project comparison")
    xp_events = [e for e in events if e.get("type") == "cross_project_comparison"]
    assert xp_events, "Expected cross_project_comparison SSE event"

    data = xp_events[0]["cross_project_comparison"]
    assert "n_projects" in data
    assert "n_with_models" in data
    assert "projects_compared" in data
    assert "summary" in data
