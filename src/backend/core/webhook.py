"""Webhook dispatch logic for AutoModeler deployment notifications.

Dispatches signed POST requests to registered webhook URLs when key deployment
events occur:
  - "batch_complete"  — a scheduled batch prediction job finished
  - "drift_detected"  — prediction distribution drift score >= 50
  - "health_degraded" — model health score drops below 60

Payloads are signed with HMAC-SHA256 using the webhook's secret, delivered in
the ``X-AutoModeler-Signature`` header (hex digest of the JSON body).

All dispatches happen in a background daemon thread — they are best-effort and
never block or crash the calling request.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# Event type constants
EVENT_BATCH_COMPLETE = "batch_complete"
EVENT_DRIFT_DETECTED = "drift_detected"
EVENT_HEALTH_DEGRADED = "health_degraded"
EVENT_QUOTA_ALERT = "quota_alert"

ALL_EVENTS = {
    EVENT_BATCH_COMPLETE,
    EVENT_DRIFT_DETECTED,
    EVENT_HEALTH_DEGRADED,
    EVENT_QUOTA_ALERT,
}


def _sign_payload(secret: str, payload_bytes: bytes) -> str:
    """Return the HMAC-SHA256 hex digest of payload_bytes using secret."""
    return hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()


def _do_dispatch(
    webhook_id: str,
    url: str,
    secret: str,
    payload: dict[str, Any],
) -> int:
    """Send the signed payload to url. Returns HTTP status code (or 0 on error)."""
    try:
        import urllib.request

        body = json.dumps(payload).encode()
        sig = _sign_payload(secret, body)
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-AutoModeler-Signature": sig,
                "X-AutoModeler-Event": payload.get("event_type", ""),
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            return resp.status
    except Exception as exc:
        logger.warning("Webhook %s dispatch to %s failed: %s", webhook_id, url, exc)
        return 0


def _dispatch_in_thread(
    webhook_id: str,
    url: str,
    secret: str,
    payload: dict[str, Any],
) -> None:
    """Update last_fired_at / last_status_code after dispatch, and log the event."""
    status = _do_dispatch(webhook_id, url, secret, payload)

    # Update the DB record and write an event log entry (best-effort)
    try:
        from db import engine
        from models.webhook_config import WebhookConfig
        from models.webhook_event import WebhookEvent
        from sqlmodel import Session

        deployment_id = payload.get("deployment_id", "")
        event_type = payload.get("event_type", "")

        with Session(engine) as session:
            wh = session.get(WebhookConfig, webhook_id)
            if wh:
                wh.last_fired_at = datetime.now(UTC).replace(tzinfo=None)
                wh.last_status_code = status
                session.add(wh)

            # Write one row to the event log regardless of outcome
            evt = WebhookEvent(
                webhook_id=webhook_id,
                deployment_id=deployment_id,
                event_type=event_type,
                status_code=status,
            )
            session.add(evt)
            session.commit()
    except Exception:
        pass


def dispatch_webhooks(
    deployment_id: str,
    event_type: str,
    payload: dict[str, Any],
) -> None:
    """Fire all active webhooks registered for this deployment + event_type.

    Non-blocking — starts one daemon thread per matching webhook.
    ``payload`` should contain at minimum ``deployment_id`` and ``event_type``;
    callers may add any extra context they like.
    """
    if event_type not in ALL_EVENTS:
        return

    try:
        from db import engine
        from models.webhook_config import WebhookConfig
        from sqlmodel import Session, select

        with Session(engine) as session:
            hooks = session.exec(
                select(WebhookConfig).where(
                    WebhookConfig.deployment_id == deployment_id,
                    WebhookConfig.is_active == True,  # noqa: E712
                )
            ).all()
            hooks_data = [(h.id, h.url, h.secret, h.event_types) for h in hooks]

        full_payload = {
            "deployment_id": deployment_id,
            "event_type": event_type,
            "fired_at": datetime.now(UTC).isoformat(),
            **payload,
        }

        for hook_id, url, secret, event_types_json in hooks_data:
            event_types_list: list[str] = json.loads(event_types_json or "[]")
            if event_type in event_types_list:
                t = threading.Thread(
                    target=_dispatch_in_thread,
                    args=(hook_id, url, secret, full_payload),
                    daemon=True,
                )
                t.start()

    except Exception as exc:
        logger.warning("dispatch_webhooks failed for %s: %s", deployment_id, exc)
