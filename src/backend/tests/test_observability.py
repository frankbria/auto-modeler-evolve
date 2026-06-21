"""Tests for issue #12 — logging config, global exception handler, scheduler liveness."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

import core.obs as obs

# ---------------------------------------------------------------------------
# log_and_swallow
# ---------------------------------------------------------------------------


def test_log_and_swallow_logs_and_swallows(caplog):
    log = logging.getLogger("test.swallow")
    with caplog.at_level(logging.ERROR, logger="test.swallow"):
        with obs.log_and_swallow("doing a thing", log=log):
            raise ValueError("boom")  # must NOT propagate

    assert "doing a thing" in caplog.text
    # logger.exception records the traceback
    assert any(r.exc_info for r in caplog.records)
    assert "boom" in caplog.text


def test_log_and_swallow_passes_through_on_success():
    ran = []
    with obs.log_and_swallow("ok"):
        ran.append(1)
    assert ran == [1]


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


def test_configure_logging_honors_level_and_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    monkeypatch.setattr(obs, "_configured", False)
    try:
        obs.configure_logging()
        assert root.level == logging.DEBUG
        rotating = [
            h for h in root.handlers if h.__class__.__name__ == "RotatingFileHandler"
        ]
        assert rotating, "expected a RotatingFileHandler on the root logger"
        assert (tmp_path / "logs" / "app.log").parent.exists()

        # Idempotent: second call (flag now set) does not add handlers.
        count = len(root.handlers)
        obs.configure_logging()
        assert len(root.handlers) == count
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


# ---------------------------------------------------------------------------
# global exception handler
# ---------------------------------------------------------------------------


def test_internal_error_handler_returns_generic_500_with_request_id(caplog):
    app = FastAPI()
    app.add_exception_handler(Exception, obs.internal_error_handler)

    @app.get("/boom")
    def boom():
        raise RuntimeError("kaboom")

    client = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level(logging.ERROR):
        resp = client.get("/boom")

    assert resp.status_code == 500
    body = resp.json()
    assert body["request_id"]
    assert "kaboom" not in body["detail"]  # no leak of internals to the client
    # request id and traceback land in the log
    assert body["request_id"] in caplog.text
    assert "kaboom" in caplog.text


# ---------------------------------------------------------------------------
# /health scheduler liveness
# ---------------------------------------------------------------------------


async def test_health_reports_scheduler_liveness(anon_client, monkeypatch):
    import core.scheduler as scheduler

    monkeypatch.setattr(scheduler, "_last_tick", datetime.now(UTC))
    resp = await anon_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["scheduler_last_tick"] is not None
    assert data["scheduler_alive"] is True


async def test_health_reports_dead_scheduler(anon_client, monkeypatch):
    import core.scheduler as scheduler

    # No tick ever, and a stale tick, both read as not-alive.
    monkeypatch.setattr(scheduler, "_last_tick", None)
    resp = await anon_client.get("/health")
    assert resp.json()["scheduler_alive"] is False

    monkeypatch.setattr(scheduler, "_last_tick", datetime.now(UTC) - timedelta(hours=1))
    resp = await anon_client.get("/health")
    assert resp.json()["scheduler_alive"] is False
