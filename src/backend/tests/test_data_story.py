"""Tests for the automated data story feature.

Covers:
  - core/storyteller.py: generate_data_story() with various datasets
  - GET /api/data/{id}/story endpoint (404, 200 variants)
  - Chat intent detection (_STORY_PATTERNS)
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from core.storyteller import _build_summary, _recommend_next_step, generate_data_story


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def make_sales_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "revenue": rng.normal(1000, 200, n),
            "quantity": rng.integers(1, 100, n),
            "region": rng.choice(["East", "West", "North", "South"], n),
            "product": rng.choice(["A", "B", "C"], n),
        }
    )


def make_minimal_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})


def make_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


# ---------------------------------------------------------------------------
# core/storyteller.py unit tests
# ---------------------------------------------------------------------------


class TestGenerateDataStory:
    def test_returns_required_keys(self):
        df = make_sales_df()
        result = generate_data_story(df, dataset_id="ds-1", dataset_filename="sales.csv")
        assert "dataset_id" in result
        assert "row_count" in result
        assert "col_count" in result
        assert "readiness_score" in result
        assert "readiness_grade" in result
        assert "sections" in result
        assert "summary" in result
        assert "recommended_next_step" in result

    def test_row_and_col_counts(self):
        df = make_sales_df(150)
        result = generate_data_story(df, dataset_id="ds-2")
        assert result["row_count"] == 150
        assert result["col_count"] == 4

    def test_sections_include_readiness(self):
        df = make_sales_df()
        result = generate_data_story(df, dataset_id="ds-3")
        section_types = [s["type"] for s in result["sections"]]
        assert "readiness" in section_types

    def test_sections_include_group_by_for_categorical_data(self):
        df = make_sales_df()
        result = generate_data_story(df, dataset_id="ds-4")
        section_types = [s["type"] for s in result["sections"]]
        assert "group_by" in section_types

    def test_no_group_by_when_no_categorical_cols(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0] * 50, "b": [4.0, 5.0, 6.0] * 50})
        result = generate_data_story(df, dataset_id="ds-5")
        section_types = [s["type"] for s in result["sections"]]
        assert "group_by" not in section_types

    def test_correlations_section_when_target_provided(self):
        df = make_sales_df()
        result = generate_data_story(df, dataset_id="ds-6", target_col="revenue")
        section_types = [s["type"] for s in result["sections"]]
        assert "correlations" in section_types

    def test_no_correlations_when_no_target(self):
        df = make_sales_df()
        result = generate_data_story(df, dataset_id="ds-7")
        section_types = [s["type"] for s in result["sections"]]
        assert "correlations" not in section_types

    def test_anomaly_section_present_for_large_df(self):
        df = make_sales_df(200)
        result = generate_data_story(df, dataset_id="ds-8")
        # Anomaly section only appears if anomalies are found (contamination=0.05)
        # With 200 rows, IsolationForest should flag ~10 rows
        section_types = [s["type"] for s in result["sections"]]
        # IsolationForest always flags contamination fraction — so anomalies section should appear
        assert "anomalies" in section_types

    def test_readiness_score_in_range(self):
        df = make_sales_df()
        result = generate_data_story(df, dataset_id="ds-9")
        assert 0 <= result["readiness_score"] <= 100

    def test_readiness_grade_valid(self):
        df = make_sales_df()
        result = generate_data_story(df, dataset_id="ds-10")
        assert result["readiness_grade"] in ("A", "B", "C", "D", "F")

    def test_summary_is_nonempty_string(self):
        df = make_sales_df()
        result = generate_data_story(df, dataset_id="ds-11")
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 20

    def test_summary_contains_row_count(self):
        df = make_sales_df(150)
        result = generate_data_story(df, dataset_id="ds-12")
        assert "150" in result["summary"]

    def test_recommended_next_step_nonempty(self):
        df = make_sales_df()
        result = generate_data_story(df, dataset_id="ds-13")
        assert isinstance(result["recommended_next_step"], str)
        assert len(result["recommended_next_step"]) > 5

    def test_filename_stored(self):
        df = make_sales_df()
        result = generate_data_story(df, dataset_id="ds-14", dataset_filename="q4.csv")
        assert result["filename"] == "q4.csv"

    def test_minimal_df_does_not_crash(self):
        df = make_minimal_df()
        result = generate_data_story(df, dataset_id="ds-15")
        assert "row_count" in result
        assert result["row_count"] == 3

    def test_with_missing_values(self):
        df = make_sales_df()
        df.loc[:10, "revenue"] = None
        result = generate_data_story(df, dataset_id="ds-16")
        assert result["readiness_score"] < 100  # should note missing values


class TestBuildSummary:
    def test_includes_row_count(self):
        text = _build_summary(500, 10, 80, "B", 0, None, [])
        assert "500" in text

    def test_includes_grade(self):
        text = _build_summary(100, 5, 90, "A", 0, None, [])
        assert "A" in text

    def test_mentions_anomalies_when_present(self):
        text = _build_summary(200, 4, 75, "B", 8, None, [])
        assert "8" in text
        assert "anomal" in text.lower()

    def test_no_anomaly_mention_when_zero(self):
        text = _build_summary(200, 4, 75, "B", 0, None, [])
        assert "anomal" not in text.lower()


class TestRecommendNextStep:
    def test_not_ready_returns_fix_data(self):
        step = _recommend_next_step("not_ready", None, [])
        assert "quality" in step.lower() or "fix" in step.lower() or "data" in step.lower()

    def test_no_target_returns_set_target_prompt(self):
        step = _recommend_next_step("ready", None, [])
        assert "target" in step.lower() or "predict" in step.lower()

    def test_ready_with_target_suggests_training(self):
        step = _recommend_next_step("ready", "revenue", [])
        assert "revenue" in step
        assert "train" in step.lower() or "model" in step.lower()


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture()
async def ac(tmp_path):
    from sqlmodel import SQLModel, create_engine

    import db as db_module

    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa: F401
    import models.dataset  # noqa: F401
    import models.deployment  # noqa: F401
    import models.feature_set  # noqa: F401
    import models.feedback_record  # noqa: F401
    import models.model_run  # noqa: F401
    import models.prediction_log  # noqa: F401
    import models.project  # noqa: F401

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"

    from httpx import ASGITransport, AsyncClient

    from main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.fixture()
async def story_dataset(ac):
    df = make_sales_df(200)
    csv_bytes = make_csv_bytes(df)

    r = await ac.post("/api/projects", json={"name": "StoryTest", "description": ""})
    project_id = r.json()["id"]

    r = await ac.post(
        "/api/data/upload",
        files={"file": ("sales.csv", csv_bytes, "text/csv")},
        data={"project_id": project_id},
    )
    dataset_id = r.json()["dataset_id"]

    return ac, project_id, dataset_id


@pytest.mark.anyio
async def test_story_404_unknown_dataset(ac):
    r = await ac.get("/api/data/nonexistent-id/story")
    assert r.status_code == 404


@pytest.mark.anyio
async def test_story_200_without_target(story_dataset):
    client, _, dataset_id = story_dataset
    r = await client.get(f"/api/data/{dataset_id}/story")
    assert r.status_code == 200
    data = r.json()
    assert "row_count" in data
    assert "sections" in data
    assert "summary" in data
    assert "recommended_next_step" in data


@pytest.mark.anyio
async def test_story_200_with_target(story_dataset):
    client, _, dataset_id = story_dataset
    r = await client.get(f"/api/data/{dataset_id}/story?target=revenue")
    assert r.status_code == 200
    data = r.json()
    section_types = [s["type"] for s in data["sections"]]
    assert "correlations" in section_types


@pytest.mark.anyio
async def test_story_200_has_readiness_section(story_dataset):
    client, _, dataset_id = story_dataset
    r = await client.get(f"/api/data/{dataset_id}/story")
    assert r.status_code == 200
    data = r.json()
    section_types = [s["type"] for s in data["sections"]]
    assert "readiness" in section_types


@pytest.mark.anyio
async def test_story_200_has_group_by_section(story_dataset):
    client, _, dataset_id = story_dataset
    r = await client.get(f"/api/data/{dataset_id}/story")
    assert r.status_code == 200
    data = r.json()
    section_types = [s["type"] for s in data["sections"]]
    assert "group_by" in section_types


# ---------------------------------------------------------------------------
# Chat intent detection
# ---------------------------------------------------------------------------


def test_story_pattern_imports():
    """Verify the pattern is importable from chat."""
    from api.chat import _STORY_PATTERNS

    assert _STORY_PATTERNS is not None


@pytest.mark.parametrize(
    "msg",
    [
        "analyze my data",
        "walk me through this dataset",
        "give me a full analysis",
        "what's interesting in this data?",
        "summarize my data",
        "what are the key insights?",
        "tell me the story",
        "give me the complete picture",
        "what should I know about this?",
        "data overview",
        "full data analysis",
        "what do you see in my data",
    ],
)
def test_story_patterns_match(msg):
    from api.chat import _STORY_PATTERNS

    assert _STORY_PATTERNS.search(msg), f"Pattern did not match: {msg!r}"


@pytest.mark.parametrize(
    "msg",
    [
        "train a model to predict revenue",
        "show me revenue by region",
        "fill missing values with median",
        "compare East vs West",
    ],
)
def test_story_patterns_no_false_positives(msg):
    from api.chat import _STORY_PATTERNS

    assert not _STORY_PATTERNS.search(msg), f"Pattern incorrectly matched: {msg!r}"
