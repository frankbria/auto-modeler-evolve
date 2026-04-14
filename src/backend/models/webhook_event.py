"""Webhook event log model — records every webhook dispatch attempt."""

from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class WebhookEvent(SQLModel, table=True):
    """One row per webhook dispatch attempt.

    Written by ``core.webhook._dispatch_in_thread`` after each fire.
    ``status_code`` is 0 when the HTTP request itself failed (network error,
    timeout, etc.).
    """

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    webhook_id: str = Field(index=True)
    deployment_id: str = Field(index=True)
    event_type: str  # "batch_complete" | "drift_detected" | etc.
    fired_at: datetime = Field(default_factory=_utcnow)
    status_code: Optional[int] = None  # HTTP response code; None = not yet recorded
