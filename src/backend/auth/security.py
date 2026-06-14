"""Password hashing and JWT access-token helpers.

The signing secret is read from the ``AUTH_SECRET`` environment variable at
call time (so tests can set it). When unset, a clearly-insecure development
default is used and a warning is emitted at startup — production deployments
MUST set ``AUTH_SECRET``.
"""

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Optional

import bcrypt
import jwt

logger = logging.getLogger(__name__)

ALGORITHM = "HS256"
# Marked nosec/noqa: this is an obvious dev placeholder, never a real secret.
_DEV_SECRET = "dev-insecure-secret-change-me-please-set-auth-secret"  # noqa: S105
# Default token lifetime: 7 days.
ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.environ.get("AUTH_TOKEN_EXPIRE_MINUTES", str(60 * 24 * 7))
)
# bcrypt only considers the first 72 bytes of a password.
MAX_PASSWORD_BYTES = 72


def get_secret() -> str:
    return os.environ.get("AUTH_SECRET") or _DEV_SECRET


def using_insecure_default_secret() -> bool:
    return not os.environ.get("AUTH_SECRET")


def hash_password(password: str) -> str:
    pw = password.encode("utf-8")[:MAX_PASSWORD_BYTES]
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        pw = password.encode("utf-8")[:MAX_PASSWORD_BYTES]
        return bcrypt.checkpw(pw, hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(user_id: str, expires_minutes: Optional[int] = None) -> str:
    now = datetime.now(UTC)
    minutes = (
        expires_minutes if expires_minutes is not None else ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
    }
    return jwt.encode(payload, get_secret(), algorithm=ALGORITHM)


def decode_access_token(token: str) -> Optional[str]:
    """Return the subject (user id) for a valid token, else None."""
    try:
        payload = jwt.decode(token, get_secret(), algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
    sub = payload.get("sub")
    return sub if isinstance(sub, str) else None
