"""Reversible encryption-at-rest for secrets we must later use in cleartext.

Unlike a password or API key (which we only ever *verify*, so a one-way hash is
right), a webhook signing secret is needed in cleartext at dispatch time to
compute the outbound HMAC-SHA256 signature. So it must be encrypted, not hashed.

Ciphertext is tagged with an ``enc:`` prefix so :func:`decrypt` can transparently
pass through any legacy plaintext secret written before this was introduced.

The key is derived from ``WEBHOOK_SECRET_KEY`` (falling back to ``AUTH_SECRET``)
via SHA-256 → urlsafe-base64, the 32-byte form Fernet requires. Rotating either
env var makes existing ciphertext undecryptable, so dispatch falls back to
re-issuing — acceptable for best-effort webhooks.
"""

from __future__ import annotations

import base64
import hashlib
import os

from cryptography.fernet import Fernet, InvalidToken

_PREFIX = "enc:"

# Same insecure dev fallback discipline as auth.security: works locally, logs
# nothing here because the webhook path is best-effort, but real deployments set
# AUTH_SECRET (or WEBHOOK_SECRET_KEY) anyway.
_DEV_KEY_MATERIAL = "automodeler-insecure-dev-secret"


def _fernet() -> Fernet:
    material = (
        os.environ.get("WEBHOOK_SECRET_KEY")
        or os.environ.get("AUTH_SECRET")
        or _DEV_KEY_MATERIAL
    )
    key = base64.urlsafe_b64encode(hashlib.sha256(material.encode()).digest())
    return Fernet(key)


def encrypt(plaintext: str) -> str:
    """Encrypt a secret for storage. Returns an ``enc:``-prefixed token."""
    return _PREFIX + _fernet().encrypt(plaintext.encode()).decode()


def decrypt(stored: str) -> str:
    """Recover a secret. Pass plaintext (un-prefixed legacy) values through.

    Returns ``stored`` unchanged if it isn't our ciphertext or can't be
    decrypted (wrong key), so dispatch degrades rather than crashing.
    """
    if not stored.startswith(_PREFIX):
        return stored
    try:
        return _fernet().decrypt(stored[len(_PREFIX) :].encode()).decode()
    except InvalidToken:
        return stored
