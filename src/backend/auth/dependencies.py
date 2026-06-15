"""FastAPI authentication dependencies."""

from typing import Optional

from fastapi import Depends, Header, HTTPException, Request
from sqlmodel import Session

from auth.security import decode_access_token
from db import get_session
from models.user import User

# Public, unauthenticated path prefixes (prediction-serving surface, which is
# gated by per-deployment API keys, not user auth).
PUBLIC_PATH_PREFIXES = ("/api/predict", "/predict")

_UNAUTHENTICATED = HTTPException(
    status_code=401,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


def _extract_token(
    authorization: Optional[str], request: Optional[Request]
) -> Optional[str]:
    """Pull the JWT from the Authorization header, falling back to the
    ``access_token`` query parameter.

    The query fallback exists for browser callers that cannot set headers —
    ``EventSource`` (SSE streams) and direct download links (``<a href>`` /
    ``window.open``). Such URLs are same-origin and short-lived; the token is
    only read here, never logged by the app.
    """
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    if request is not None:
        query_token = request.query_params.get("access_token")
        if query_token:
            return query_token.strip()
    return None


def _user_from_token(token: Optional[str], session: Session) -> Optional[User]:
    if not token:
        return None
    user_id = decode_access_token(token)
    if not user_id:
        return None
    user = session.get(User, user_id)
    if not user or not user.is_active:
        return None
    return user


def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    session: Session = Depends(get_session),
) -> User:
    """Resolve the authenticated user or raise 401.

    Attach to any management route with ``Depends(get_current_user)``.
    """
    user = _user_from_token(_extract_token(authorization, request), session)
    if user is None:
        raise _UNAUTHENTICATED
    return user


def get_optional_user(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    session: Session = Depends(get_session),
) -> Optional[User]:
    """Like ``get_current_user`` but returns None instead of raising.

    For endpoints that are public but can personalize when a token is present.
    """
    return _user_from_token(_extract_token(authorization, request), session)


def require_owner(
    request: Request,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> User:
    """Router-level dependency: authenticate AND authorize the path resource.

    Attach via ``APIRouter(dependencies=[Depends(require_owner)])``. Rejects
    unauthenticated requests (401) and any request whose path-parameter
    resource is not owned by the caller (404). Covers every route in the
    router with a single line.
    """
    from auth.scoping import authorize_path_params

    authorize_path_params(request.path_params, current_user, session)
    return current_user


def require_owner_allow_public(
    request: Request,
    user: Optional[User] = Depends(get_optional_user),
    session: Session = Depends(get_session),
) -> Optional[User]:
    """Like ``require_owner`` but lets public prediction-serving paths through.

    Used by the deploy router, which mixes management endpoints (owner-scoped)
    with the public ``/api/predict/*`` serving surface (gated by per-deployment
    API keys, not user auth).
    """
    from auth.scoping import authorize_path_params

    # The prediction-serving surface (/api/predict/*, /predict/*) is public —
    # gated by per-deployment API keys inside the handlers, not user auth.
    # (The cross-deployment /api/predict/compare tool re-locks itself with its
    # own get_current_user dependency + per-id ownership checks.)
    if any(request.url.path.startswith(p) for p in PUBLIC_PATH_PREFIXES):
        return None
    if user is None:
        raise _UNAUTHENTICATED
    authorize_path_params(request.path_params, user, session)
    return user
