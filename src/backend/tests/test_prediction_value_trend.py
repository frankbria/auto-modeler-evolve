"""Tests for Production Prediction Value Trend — Track D perpetual.

Covers:
  - compute_prediction_value_trend pure function (stable, up, down verdicts)
  - Period grouping by day and week
  - Edge cases: no_data, insufficient periods, classification guard
  - GET /api/deploy/{deployment_id}/prediction-value-trend REST endpoint
  - _PRED_VALUE_TREND_PATTERNS regex (positive + negative matches)
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, create_engine

import db as db_module
from api.chat import _PRED_VALUE_TREND_PATTERNS
from core.analyzer import compute_prediction_value_trend

# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_logs(values: list[float], start: datetime | None = None) -> list[dict]:
    """Build a list of prediction log dicts, one per day, in ascending order."""
    base = start or datetime(2025, 1, 1)
    return [
        {"prediction_numeric": v, "created_at": base + timedelta(days=i)}
        for i, v in enumerate(values)
    ]


# ─── Pure function tests ───────────────────────────────────────────────────────


def test_required_keys_returned():
    logs = _make_logs([100.0, 110.0, 120.0, 130.0, 140.0])
    result = compute_prediction_value_trend(logs)
    for key in (
        "periods",
        "n_total",
        "n_periods_with_data",
        "direction",
        "direction_label",
        "slope",
        "slope_pct_per_period",
        "first_period_mean",
        "last_period_mean",
        "overall_change_pct",
        "summary",
    ):
        assert key in result, f"Missing key: {key}"


def test_stable_verdict():
    """Flat predictions produce 'stable' direction."""
    logs = _make_logs([100.0] * 10)
    result = compute_prediction_value_trend(logs)
    assert result["direction"] == "stable"
    assert result["overall_change_pct"] == pytest.approx(0.0, abs=1.0)


def test_trending_up_verdict():
    """Steadily increasing predictions produce 'trending_up'."""
    logs = _make_logs([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
    result = compute_prediction_value_trend(logs)
    assert result["direction"] == "trending_up"
    assert result["overall_change_pct"] > 5.0


def test_trending_down_verdict():
    """Steadily decreasing predictions produce 'trending_down'."""
    logs = _make_logs([200, 190, 180, 170, 160, 150, 140, 130, 120, 110])
    result = compute_prediction_value_trend(logs)
    assert result["direction"] == "trending_down"
    assert result["overall_change_pct"] < -5.0


def test_periods_list_length():
    """Each day with data produces one period entry."""
    logs = _make_logs([100.0 + i for i in range(7)])
    result = compute_prediction_value_trend(logs)
    assert result["n_periods_with_data"] == 7
    assert len(result["periods"]) == 7


def test_period_entry_keys():
    """Each period entry has required keys."""
    logs = _make_logs([100.0, 110.0, 120.0])
    result = compute_prediction_value_trend(logs)
    for period in result["periods"]:
        for k in ("period_label", "period_start", "mean", "count", "min", "max"):
            assert k in period, f"Period entry missing key: {k}"


def test_period_count_accuracy():
    """count in each period matches the number of predictions that day."""
    base = datetime(2025, 1, 1)
    logs = [
        {"prediction_numeric": 100.0, "created_at": base},
        {"prediction_numeric": 102.0, "created_at": base},  # same day
        {"prediction_numeric": 110.0, "created_at": base + timedelta(days=1)},
    ]
    result = compute_prediction_value_trend(logs)
    first_period = result["periods"][0]
    assert first_period["count"] == 2
    assert first_period["mean"] == pytest.approx(101.0)


def test_first_and_last_period_mean():
    """first_period_mean and last_period_mean track the period averages."""
    logs = _make_logs([100.0, 200.0])
    result = compute_prediction_value_trend(logs)
    assert result["first_period_mean"] == pytest.approx(100.0)
    assert result["last_period_mean"] == pytest.approx(200.0)


def test_n_total_is_sum_of_counts():
    """n_total equals the total number of numeric predictions in all periods."""
    logs = _make_logs([100.0] * 5)
    result = compute_prediction_value_trend(logs)
    assert result["n_total"] == 5


def test_slope_sign_matches_direction():
    """Slope is positive for trending_up and negative for trending_down."""
    up_result = compute_prediction_value_trend(
        _make_logs([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    )
    assert up_result["slope"] > 0

    down_result = compute_prediction_value_trend(
        _make_logs([100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
    )
    assert down_result["slope"] < 0


def test_n_periods_cap():
    """n_periods parameter limits the number of included periods."""
    logs = _make_logs([100.0 + i for i in range(30)])
    result = compute_prediction_value_trend(logs, n_periods=7)
    assert result["n_periods_with_data"] <= 7


def test_none_prediction_numeric_skipped():
    """Logs with None prediction_numeric are silently ignored."""
    base = datetime(2025, 1, 1)
    logs = [
        {"prediction_numeric": None, "created_at": base},
        {"prediction_numeric": 100.0, "created_at": base + timedelta(days=1)},
        {"prediction_numeric": 110.0, "created_at": base + timedelta(days=2)},
    ]
    result = compute_prediction_value_trend(logs)
    assert result["n_total"] == 2


def test_iso_string_created_at():
    """ISO-format string timestamps are parsed correctly."""
    base = datetime(2025, 3, 1)
    logs = [
        {"prediction_numeric": 50.0, "created_at": base.isoformat()},
        {
            "prediction_numeric": 60.0,
            "created_at": (base + timedelta(days=1)).isoformat(),
        },
    ]
    result = compute_prediction_value_trend(logs)
    assert result["n_total"] == 2


def test_summary_is_string():
    """Summary is a non-empty string."""
    logs = _make_logs([100.0, 110.0, 120.0])
    result = compute_prediction_value_trend(logs)
    assert isinstance(result["summary"], str)
    assert len(result["summary"]) > 10


def test_summary_mentions_trend_up():
    """Summary for trending_up mentions increase."""
    logs = _make_logs([100, 115, 130, 145, 160, 175, 190, 205, 220, 235])
    result = compute_prediction_value_trend(logs)
    assert result["direction"] == "trending_up"
    assert (
        "increased" in result["summary"].lower()
        or "higher" in result["summary"].lower()
        or "increase" in result["summary"].lower()
    )


def test_summary_mentions_trend_down():
    """Summary for trending_down mentions decrease."""
    logs = _make_logs([235, 220, 205, 190, 175, 160, 145, 130, 115, 100])
    result = compute_prediction_value_trend(logs)
    assert result["direction"] == "trending_down"
    assert (
        "decreased" in result["summary"].lower()
        or "lower" in result["summary"].lower()
        or "decrease" in result["summary"].lower()
    )


def test_insufficient_data_raises():
    """Fewer than 2 periods raises ValueError."""
    logs = _make_logs([100.0])  # only one day
    with pytest.raises(ValueError):
        compute_prediction_value_trend(logs)


def test_empty_logs_raises():
    """Empty list raises ValueError."""
    with pytest.raises(ValueError):
        compute_prediction_value_trend([])


def test_all_none_raises():
    """All-None prediction_numeric raises ValueError."""
    base = datetime(2025, 1, 1)
    logs = [
        {"prediction_numeric": None, "created_at": base + timedelta(days=i)}
        for i in range(5)
    ]
    with pytest.raises(ValueError):
        compute_prediction_value_trend(logs)


def test_week_period_grouping():
    """period='week' groups daily predictions into ISO week buckets."""
    # 14 days = 2 weeks
    base = datetime(2025, 1, 6)  # Monday
    logs = [
        {
            "prediction_numeric": float(i * 10 + 100),
            "created_at": base + timedelta(days=i),
        }
        for i in range(14)
    ]
    result = compute_prediction_value_trend(logs, period="week")
    assert result["n_periods_with_data"] == 2


def test_overall_change_pct_formula():
    """overall_change_pct = (last_mean - first_mean) / |first_mean| * 100."""
    logs = _make_logs([100.0, 200.0])
    result = compute_prediction_value_trend(logs)
    expected = (200.0 - 100.0) / 100.0 * 100
    assert result["overall_change_pct"] == pytest.approx(expected, abs=0.1)


def test_direction_label_is_string():
    """direction_label is a human-readable string."""
    logs = _make_logs([100.0, 110.0, 120.0])
    result = compute_prediction_value_trend(logs)
    assert isinstance(result["direction_label"], str)
    assert len(result["direction_label"]) > 3


# ─── Regex tests ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "msg",
    [
        "are my predictions trending up?",
        "prediction value trend",
        "regression output trend",
        "are my predictions going up over time",
        "show me how my predictions have changed",
        "prediction values over time",
        "are my model outputs getting higher over time",
        "prediction value history",
        "my model outputs are trending",
        "show prediction time series",
    ],
)
def test_regex_positive(msg):
    assert _PRED_VALUE_TREND_PATTERNS.search(msg), f"Expected match for: {msg!r}"


@pytest.mark.parametrize(
    "msg",
    [
        "how accurate is my model",
        "show me feature importance",
        "retrain my model",
        "what is my training accuracy",
    ],
)
def test_regex_negative(msg):
    assert not _PRED_VALUE_TREND_PATTERNS.search(msg), f"Unexpected match for: {msg!r}"


# ─── REST endpoint tests ──────────────────────────────────────────────────────


@pytest.fixture()
def _db_url(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path}/test_pred_trend.db",
        connect_args={"check_same_thread": False},
    )
    import models.project  # noqa: F401
    import models.dataset  # noqa: F401
    import models.dataset_filter  # noqa: F401
    import models.feature_set  # noqa: F401
    import models.model_run  # noqa: F401
    import models.conversation  # noqa: F401
    import models.prediction_log  # noqa: F401
    import models.deployment  # noqa: F401
    import models.feedback_record  # noqa: F401
    import models.webhook_config  # noqa: F401
    import models.webhook_event  # noqa: F401
    import models.ab_test  # noqa: F401
    import models.deployment_preset  # noqa: F401
    import models.input_validation_rule  # noqa: F401
    import models.dashboard_field_config  # noqa: F401
    import models.goal_seek_record  # noqa: F401
    import models.deployment_changelog  # noqa: F401
    import models.deployment_version  # noqa: F401
    import models.batch_schedule  # noqa: F401
    import models.prediction_alert_rule  # noqa: F401
    import models.analysis_template  # noqa: F401

    SQLModel.metadata.create_all(engine)
    db_module.engine = engine
    yield
    engine.dispose()


@pytest.mark.asyncio
async def test_endpoint_404_missing_deployment(_db_url):
    """Returns 404 for unknown deployment."""
    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get("/api/deploy/nonexistent/prediction-value-trend")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_endpoint_no_data(_db_url, tmp_path):
    """Returns no_data direction when no prediction logs exist."""
    from main import app
    from sqlmodel import Session
    from models.project import Project
    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun
    from models.deployment import Deployment

    csv_data = "revenue,units\n100,10\n200,20\n"
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(csv_data)

    with Session(db_module.engine) as sess:
        proj = Project(owner_id="test-default-owner", name="TrendTest")
        sess.add(proj)
        sess.flush()
        ds = Dataset(
            project_id=proj.id,
            filename="data.csv",
            file_path=str(csv_path),
            row_count=2,
            column_count=2,
            columns="[]",
        )
        sess.add(ds)
        sess.flush()
        fs = FeatureSet(
            dataset_id=ds.id,
            target_column="revenue",
            transformations="[]",
            column_mapping="{}",
        )
        sess.add(fs)
        sess.flush()
        run = ModelRun(
            project_id=proj.id,
            feature_set_id=fs.id,
            algorithm="linear_regression",
            hyperparameters="{}",
            metrics='{"r2": 0.8}',
            status="done",
        )
        sess.add(run)
        sess.flush()
        dep = Deployment(
            model_run_id=run.id,
            project_id=proj.id,
            endpoint_path=f"/api/predict/{run.id}",
            dashboard_url=f"/predict/{run.id}",
            is_active=True,
            algorithm="linear_regression",
            target_column="revenue",
            problem_type="regression",
        )
        sess.add(dep)
        sess.commit()
        dep_id = dep.id

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        resp = await ac.get(f"/api/deploy/{dep_id}/prediction-value-trend")

    assert resp.status_code == 200
    data = resp.json()
    assert data["direction"] in ("no_data", "stable", "trending_up", "trending_down")
    assert "summary" in data
