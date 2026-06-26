"""Webhook notification configuration model."""

from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class WebhookConfig(SQLModel, table=True):
    """Stores a registered webhook endpoint for a deployment.

    AutoModeler dispatches a signed POST request to ``url`` whenever one of the
    registered ``event_types`` fires:
      - "batch_complete"  — a scheduled batch job finished (success or failure)
      - "drift_detected"  — prediction distribution drift score >= 50
      - "health_degraded" — model health score drops below 60
    """

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    deployment_id: str = Field(index=True)
    url: str
    # Random 32-byte hex secret for HMAC-SHA256 signing, encrypted at rest
    # (core.secret_box) and shown in cleartext only once at registration.
    secret: str = Field(default_factory=lambda: secrets_token())
    # JSON-serialised list of event type strings
    event_types: str = Field(
        default='["batch_complete","drift_detected","health_degraded"]'
    )
    is_active: bool = True
    created_at: datetime = Field(default_factory=_utcnow)
    last_fired_at: Optional[datetime] = None
    last_status_code: Optional[int] = None


def secrets_token() -> str:
    """Encrypted-at-rest default secret. Used only when a WebhookConfig is built
    without an explicit secret; the create endpoint sets its own so it can return
    the cleartext once."""
    import secrets

    from core.secret_box import encrypt

    return encrypt(secrets.token_hex(32))
