"""Observability helpers: logging config, background-swallow logging, error handler.

Centralizes the three things issue #12 asked for:
- ``configure_logging()`` — a ``dictConfig`` (console + rotating file) wired into
  the app lifespan, so the ~19 background logger calls actually go somewhere
  instead of the lastResort handler.
- ``log_and_swallow(label)`` — a context manager for background-daemon paths that
  must never crash the request but must NOT fail silently either.
- ``internal_error_handler`` — a global exception handler that logs the traceback
  with a request id and returns a generic 500 (no stack trace to the client).
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import contextmanager
from logging.config import dictConfig

from fastapi import Request
from fastapi.responses import JSONResponse

from core import storage

logger = logging.getLogger(__name__)

_configured = False


def configure_logging() -> None:
    """Install a console + rotating-file logging config (idempotent).

    Level comes from the ``LOG_LEVEL`` env var (default ``INFO``). The file
    handler writes under the DATA_DIR-aware ``logs/`` dir so rotation shares the
    same root as every other artifact.
    """
    global _configured
    if _configured:
        return

    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_dir = storage.logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "formatter": "standard",
                    "filename": str(log_dir / "app.log"),
                    "maxBytes": 10 * 1024 * 1024,
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
            },
            "root": {
                "level": level,
                "handlers": ["console", "file"],
            },
        }
    )
    _configured = True


@contextmanager
def log_and_swallow(label: str, *, log: logging.Logger | None = None):
    """Run a background best-effort block; log any exception, never re-raise.

    Replaces ``except Exception: pass`` in background-daemon/correctness paths so
    a monitoring/alert feature that breaks leaves a traceback in the log instead
    of failing silently forever.
    """
    try:
        yield
    except Exception:  # noqa: BLE001 — that is the point: swallow after logging
        (log or logger).exception("Swallowed background error: %s", label)


async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log an unhandled exception with a request id; return a generic 500.

    The north star is "fail gracefully — no stack traces": the client gets an
    opaque message plus a correlation id they can quote, while the full traceback
    lands in the server log under that same id.
    """
    request_id = uuid.uuid4().hex
    logger.exception(
        "Unhandled error [%s] on %s %s",
        request_id,
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error. Please try again.",
            "request_id": request_id,
        },
    )
