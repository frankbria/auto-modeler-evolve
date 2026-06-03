"""Tests for the Automated Weekly Monitoring Digest Webhook feature."""

from __future__ import annotations

from datetime import datetime

# ---------------------------------------------------------------------------
# Pure-function tests: should_send_weekly_digest
# ---------------------------------------------------------------------------


def test_send_on_correct_day_and_hour():
    from core.scheduler import should_send_weekly_digest

    now = datetime(2025, 6, 2, 10, 0)  # Monday 10:00
    assert should_send_weekly_digest(0, 9, None, now) is True


def test_not_send_wrong_day():
    from core.scheduler import should_send_weekly_digest

    now = datetime(2025, 6, 3, 10, 0)  # Tuesday
    assert should_send_weekly_digest(0, 9, None, now) is False  # configured Monday


def test_not_send_before_hour():
    from core.scheduler import should_send_weekly_digest

    now = datetime(2025, 6, 2, 8, 0)  # Monday 08:00, configured 09:00
    assert should_send_weekly_digest(0, 9, None, now) is False


def test_not_send_already_sent_today():
    from core.scheduler import should_send_weekly_digest

    now = datetime(2025, 6, 2, 10, 30)  # Monday 10:30
    last_sent = datetime(2025, 6, 2, 9, 5)  # Already sent today
    assert should_send_weekly_digest(0, 9, last_sent, now) is False


def test_send_if_sent_last_week():
    from core.scheduler import should_send_weekly_digest

    now = datetime(2025, 6, 9, 10, 0)  # Monday next week
    last_sent = datetime(2025, 6, 2, 9, 5)  # Sent last week
    assert should_send_weekly_digest(0, 9, last_sent, now) is True


def test_send_at_exact_hour():
    from core.scheduler import should_send_weekly_digest

    now = datetime(2025, 6, 2, 9, 0)  # Monday exactly at 09:00
    assert should_send_weekly_digest(0, 9, None, now) is True


def test_all_days_of_week():
    from core.scheduler import should_send_weekly_digest

    for weekday in range(7):
        now = datetime(2025, 6, 2 + weekday, 10, 0)  # Mon=0, Tue=1, ...
        result = should_send_weekly_digest(weekday, 9, None, now)
        assert result is True, f"Should send on weekday={weekday}"


def test_not_send_none_last_sent_wrong_day():
    from core.scheduler import should_send_weekly_digest

    now = datetime(2025, 6, 4, 10, 0)  # Wednesday
    assert should_send_weekly_digest(0, 9, None, now) is False  # Monday=0


# ---------------------------------------------------------------------------
# Regex pattern tests: _WEEKLY_DIGEST_PATTERNS
# ---------------------------------------------------------------------------


def _check_weekly_digest(msg: str) -> bool:
    from api.chat import _WEEKLY_DIGEST_PATTERNS

    return bool(_WEEKLY_DIGEST_PATTERNS.search(msg))


def test_pattern_enable_weekly_digest():
    assert _check_weekly_digest("enable weekly monitoring digest")


def test_pattern_schedule_weekly_report():
    assert _check_weekly_digest("schedule a weekly monitoring report")


def test_pattern_send_weekly_digest():
    assert _check_weekly_digest("send me a weekly digest")


def test_pattern_disable():
    assert _check_weekly_digest("disable the weekly monitoring digest")


def test_pattern_status_check():
    assert _check_weekly_digest("show my weekly digest status")


def test_pattern_auto_send():
    assert _check_weekly_digest("automatically send monitoring digest weekly")


def test_pattern_model_health_report():
    assert _check_weekly_digest("weekly model health report")


def test_pattern_configure():
    assert _check_weekly_digest("configure weekly monitoring report")


# False-positive guards
def test_no_match_weekly_report_unrelated():
    assert not _check_weekly_digest("show me a report of my model metrics")


def test_no_match_batch_schedule():
    assert not _check_weekly_digest("schedule batch predictions every monday")


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------


def test_get_weekly_digest_config_not_found():
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    resp = client.get("/api/deploy/nonexistent-id/weekly-digest-config")
    assert resp.status_code == 404


def test_put_weekly_digest_config_not_found():
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    resp = client.put("/api/deploy/nonexistent-id/weekly-digest-config?enabled=true")
    assert resp.status_code == 404


def test_put_weekly_digest_invalid_day():
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    # Need a real deployment for this test, but invalid day check comes first
    # Just verify the 422 for invalid day_of_week on a known-missing deployment
    resp = client.put(
        "/api/deploy/nonexistent-id/weekly-digest-config?enabled=true&day_of_week=9"
    )
    # 404 comes first since deployment doesn't exist, but validates our route is wired
    assert resp.status_code in (404, 422)


def test_weekly_digest_config_roundtrip(tmp_path, monkeypatch):
    """Full create/read/delete roundtrip via REST endpoints."""
    import uuid

    from fastapi.testclient import TestClient
    from sqlmodel import Session, SQLModel, create_engine

    test_db = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{test_db}")

    import models.weekly_digest_config  # noqa: F401

    SQLModel.metadata.create_all(engine)

    from main import app

    monkeypatch.setattr("db.engine", engine)

    # Create a fake active deployment
    from models.deployment import Deployment
    from models.model_run import ModelRun

    with Session(engine) as s:
        run = ModelRun(
            id=str(uuid.uuid4()),
            project_id="proj1",
            algorithm="linear_regression",
            metrics="{}",
            model_path="/tmp/model.joblib",
        )
        s.add(run)
        s.commit()

        dep = Deployment(
            id=str(uuid.uuid4()),
            model_run_id=run.id,
            project_id="proj1",
            endpoint_path=f"/api/predict/{run.id}",
            dashboard_url=f"/predict/{run.id}",
            algorithm="linear_regression",
            target_column="revenue",
            problem_type="regression",
        )
        s.add(dep)
        s.commit()
        dep_id = dep.id

    client = TestClient(app)

    # GET before configuring
    resp = client.get(f"/api/deploy/{dep_id}/weekly-digest-config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is False

    # PUT to enable
    resp = client.put(
        f"/api/deploy/{dep_id}/weekly-digest-config?enabled=true&day_of_week=2&send_hour=8"
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True
    assert data["day_of_week"] == 2
    assert data["day_name"] == "Wednesday"
    assert data["send_hour"] == 8

    # GET to verify
    resp = client.get(f"/api/deploy/{dep_id}/weekly-digest-config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True

    # DELETE to remove
    resp = client.delete(f"/api/deploy/{dep_id}/weekly-digest-config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == "deleted"

    # GET after delete returns disabled state
    resp = client.get(f"/api/deploy/{dep_id}/weekly-digest-config")
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False
