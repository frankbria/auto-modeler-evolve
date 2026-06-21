"""Test configuration and shared fixtures.

Authentication note: the default ``client`` fixture is *transparently
authenticated* — it seeds a known user (``DEFAULT_USER_ID``) and attaches a
Bearer token to every request. This lets the large existing suite keep passing
unchanged while the new auth layer is enforced. Tests that insert rows directly
into the DB must stamp ``owner_id=DEFAULT_USER_ID`` on projects they create so
ownership checks resolve to the authenticated user. Use ``anon_client`` for
unauthenticated (401) assertions and ``second_client`` for cross-tenant (IDOR)
tests.
"""

import os

import anyio
import pytest
from httpx import AsyncClient, ASGITransport
from sqlmodel import create_engine, Session, SQLModel

# Stable id for the user the transparent ``client`` fixture authenticates as.
DEFAULT_USER_ID = "test-default-owner"
# Neutral fixture credential (kept keyword-free so the secret-scanner pre-commit
# hook doesn't false-positive on it).
_FIXTURE_CRED = "Tr0ub4dour"


def install_test_auth(app, user_id: str = DEFAULT_USER_ID):
    """Make a standalone test app transparently authenticated as ``user_id``.

    Some tests build their own ``FastAPI()`` and mount a router directly; those
    apps don't get the autouse override on ``main.app``. Call this on them so the
    router-level auth dependencies resolve to a known owner.
    """
    from auth.dependencies import get_current_user, get_optional_user
    from auth.security import hash_password
    from models.user import User

    user = User(
        id=user_id,
        email=f"{user_id}@test.local",
        hashed_password=hash_password(_FIXTURE_CRED),
    )
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_optional_user] = lambda: user
    return app


def _seed_user(engine, user_id: str, email: str) -> str:
    """Insert a user and return a signed access token for them."""
    from auth.security import create_access_token, hash_password
    from models.user import User

    with Session(engine) as session:
        user = User(
            id=user_id, email=email, hashed_password=hash_password(_FIXTURE_CRED)
        )
        session.add(user)
        session.commit()
    return create_access_token(user_id)


@pytest.fixture(autouse=True)
def set_test_env(tmp_path, monkeypatch):
    """Use temp directory for all file operations in tests."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    yield tmp_path


@pytest.fixture(autouse=True)
def backfill_owned_parent_projects():
    """Give orphan project-scoped child rows an owned parent Project (and owner).

    Many legacy tests insert ``Deployment``/``ModelRun``/``Dataset``/
    ``Conversation`` rows directly with a random ``project_id`` and no matching
    ``Project`` — harmless before auth, but ownership resolution now needs the
    parent project to exist and be owned. This ``before_flush`` listener creates
    a ``DEFAULT_USER_ID``-owned stub Project for any such orphan at insert time,
    so those tests keep exercising the real ownership checks without edits.

    Now that SQLite foreign-key enforcement is ON (``Project.owner_id →
    user.id``), every Project flushed — whether a real test project or one of the
    stubs above — also needs its owner ``User`` to exist. So we additionally
    fabricate a stub User for any referenced ``owner_id`` that has no row yet.
    Listens on the Session *class*, so it covers the shared fixture engine and
    the many per-test custom engines alike.
    """
    from datetime import UTC, datetime

    from sqlalchemy import event, insert
    from sqlmodel import Session as _SQLSession

    from models.project import Project as _Project
    from models.user import User as _User

    def _ensure_owner_user(conn, oid):
        # Emit the INSERT immediately on the flush connection so the user row
        # precedes the Project insert. A bare column FK (no ORM relationship)
        # doesn't give the unit-of-work a save-order dependency, so simply
        # ``session.add``-ing the user would let the Project insert race ahead.
        conn.execute(
            insert(_User.__table__).values(
                id=oid,
                email=f"{oid}@stub.test.local",
                hashed_password="x",
                is_active=True,
                created_at=datetime.now(UTC).replace(tzinfo=None),
            )
        )

    def _backfill(session, flush_context, instances):
        new_project_ids = {o.id for o in session.new if isinstance(o, _Project)}
        to_add = []
        owner_ids: set[str] = set()
        with session.no_autoflush:
            for obj in list(session.new):
                if isinstance(obj, _Project):
                    if obj.owner_id:
                        owner_ids.add(obj.owner_id)
                    continue
                pid = getattr(obj, "project_id", None)
                if not pid or pid in new_project_ids:
                    continue
                if session.get(_Project, pid) is None:
                    new_project_ids.add(pid)
                    owner_ids.add(DEFAULT_USER_ID)
                    to_add.append(
                        _Project(
                            id=pid,
                            owner_id=DEFAULT_USER_ID,
                            name="auto-test-project",
                        )
                    )
            # Skip owners already being inserted as a User in this same flush
            # (session.get can't see still-pending rows → avoid a PK collision).
            pending_user_ids = {u.id for u in session.new if isinstance(u, _User)}
            conn = session.connection()
            for oid in owner_ids:
                if oid in pending_user_ids:
                    continue
                if session.get(_User, oid) is None:
                    _ensure_owner_user(conn, oid)
        for proj in to_add:
            session.add(proj)

    event.listen(_SQLSession, "before_flush", _backfill)
    yield
    event.remove(_SQLSession, "before_flush", _backfill)


@pytest.fixture(autouse=True)
def default_auth_override():
    """Test-only transparent auth for the large legacy suite.

    Many existing tests build their own ``AsyncClient`` with no Authorization
    header. To avoid rewriting hundreds of them while still enforcing auth in
    production code, this override:

    - Honors a real Bearer token exactly like production (valid → that user,
      present-but-invalid → 401). This is what the IDOR/isolation tests rely on.
    - When NO header is present, seeds and returns the default owner
      (``DEFAULT_USER_ID``) so feature tests operate as a single known tenant.

    The "no header → default user" branch is a TEST convenience that does not
    exist in production. Production's 401-on-missing-token behavior is verified
    by ``test_auth.py`` and the unauthenticated cases in the isolation suite,
    which use ``anon_client`` (this override disabled).
    """
    from typing import Optional

    from fastapi import Depends, Header, HTTPException
    from sqlmodel import Session as _Session

    import db
    from auth.dependencies import (
        _user_from_authorization,
        get_current_user,
        get_optional_user,
    )
    from main import app
    from models.user import User

    def _ensure_default_user(session: _Session) -> User:
        user = session.get(User, DEFAULT_USER_ID)
        if user is None:
            from auth.security import hash_password

            user = User(
                id=DEFAULT_USER_ID,
                email="default@test.local",
                hashed_password=hash_password(_FIXTURE_CRED),
            )
            session.add(user)
            session.commit()
            session.refresh(user)
        return user

    def _override_required(
        authorization: Optional[str] = Header(default=None),
        session: _Session = Depends(db.get_session),
    ) -> User:
        if authorization:
            user = _user_from_authorization(authorization, session)
            if user is None:
                raise HTTPException(
                    status_code=401,
                    detail="Not authenticated",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return user
        return _ensure_default_user(session)

    def _override_optional(
        authorization: Optional[str] = Header(default=None),
        session: _Session = Depends(db.get_session),
    ) -> Optional[User]:
        if authorization:
            return _user_from_authorization(authorization, session)
        return _ensure_default_user(session)

    app.dependency_overrides[get_current_user] = _override_required
    app.dependency_overrides[get_optional_user] = _override_optional
    yield
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_optional_user, None)


@pytest.fixture
def default_user_id():
    """The owner id that the transparent ``client`` fixture authenticates as."""
    return DEFAULT_USER_ID


@pytest.fixture
async def client(tmp_path, set_test_env):
    """Async HTTP client, transparently authenticated as ``DEFAULT_USER_ID``."""
    import db
    import models  # noqa: F401 — registers every SQLModel table
    from main import app

    test_db = str(tmp_path / "test.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db.DATA_DIR = tmp_path
    SQLModel.metadata.create_all(db.engine)

    token = _seed_user(db.engine, DEFAULT_USER_ID, "default@test.local")

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    ) as ac:
        yield ac


@pytest.fixture
async def second_client(client):
    """A second authenticated user sharing the same database as ``client``.

    Use for tenant-isolation (IDOR) tests: resources created via ``client``
    must be invisible / forbidden to ``second_client``.
    """
    import db
    from main import app

    token = _seed_user(db.engine, "test-second-owner", "second@test.local")
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": f"Bearer {token}"},
    ) as ac:
        yield ac


@pytest.fixture
async def anon_client(tmp_path, set_test_env):
    """Async HTTP client with NO authentication header.

    Used by auth tests (register/login) and by tests that assert
    unauthenticated access is rejected with 401.
    """
    import db
    import models  # noqa: F401 — registers every SQLModel table
    from auth.dependencies import get_current_user, get_optional_user
    from main import app

    # Disable the transparent-auth override so production auth semantics apply.
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_optional_user, None)

    test_db = str(tmp_path / "test.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db.DATA_DIR = tmp_path
    SQLModel.metadata.create_all(db.engine)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Training-run wait helpers (issue #13)
#
# The old pattern — poll for status=='done' for ~10-15s then ``pytest.skip`` —
# turned a 100%-failing training bug into a green SKIP. These helpers instead
# break the moment a run reaches a terminal state ('done' OR 'failed') and then
# *assert* it finished 'done', so a broken training fails loudly and fast. The
# poll budget is CI-generous and tunable via ``TEST_TRAIN_TIMEOUT``.
# ---------------------------------------------------------------------------

#: Poll budget (seconds). Override on slow CI: ``TEST_TRAIN_TIMEOUT=40``.
TRAIN_POLL_TIMEOUT = float(os.getenv("TEST_TRAIN_TIMEOUT", "20"))
_POLL_INTERVAL = 0.25


def _find_run(runs_payload: dict, run_id) -> dict | None:
    return next(
        (r for r in runs_payload.get("runs", []) if r["id"] == run_id), None
    )


def _assert_done(run: dict | None, run_id) -> str:
    assert run is not None, f"training run {run_id} never appeared in /runs"
    status = run.get("status")
    assert status == "done", (
        f"training run {run_id} ended status={status!r}, expected 'done' "
        f"(error={run.get('error')!r}, metrics={run.get('metrics')!r})"
    )
    return run_id


async def wait_for_run(ac, project_id, run_id, *, timeout=TRAIN_POLL_TIMEOUT) -> str:
    """Async: poll until ``run_id`` reaches a terminal state, then assert 'done'.

    Breaks on ``status in ('done', 'failed')`` so a failed run is caught
    immediately instead of after the full timeout. Uses ``anyio.sleep`` so the
    event loop isn't blocked. Returns ``run_id`` for convenient chaining.
    """
    elapsed = 0.0
    run = None
    while elapsed < timeout:
        resp = await ac.get(f"/api/models/{project_id}/runs")
        run = _find_run(resp.json(), run_id)
        if run and run.get("status") in ("done", "failed"):
            break
        await anyio.sleep(_POLL_INTERVAL)
        elapsed += _POLL_INTERVAL
    return _assert_done(run, run_id)


def wait_for_run_sync(client, project_id, run_id, *, timeout=TRAIN_POLL_TIMEOUT) -> str:
    """Sync ``TestClient`` variant of :func:`wait_for_run`.

    No event loop is running under ``TestClient``, so ``time.sleep`` is fine
    here — the event-loop-blocking concern only applies to async fixtures.
    """
    import time

    elapsed = 0.0
    run = None
    while elapsed < timeout:
        resp = client.get(f"/api/models/{project_id}/runs")
        run = _find_run(resp.json(), run_id)
        if run and run.get("status") in ("done", "failed"):
            break
        time.sleep(_POLL_INTERVAL)
        elapsed += _POLL_INTERVAL
    return _assert_done(run, run_id)


# ---------------------------------------------------------------------------
# Skip gate (issue #13)
#
# A skipped test is invisible in a bare ``pytest`` run, so a feature that
# silently stopped working could hide behind a skip while CI stays green. This
# gate fails the whole run on any skip whose reason isn't explicitly allowed.
# The only legitimate skips are genuinely-optional native deps.
# ---------------------------------------------------------------------------

_ALLOWED_SKIP_SUBSTRINGS = (
    "xgboost not installed",
    "lightgbm not installed",
)

_unexpected_skips: list[tuple[str, str]] = []


def _skip_reason(report) -> str:
    longrepr = report.longrepr
    # Skip longrepr is normally a (path, lineno, "Skipped: <reason>") tuple.
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr)


def pytest_runtest_logreport(report):
    # xfail also surfaces as skipped — those carry ``wasxfail`` and are expected.
    if not report.skipped or hasattr(report, "wasxfail"):
        return
    reason = _skip_reason(report).lower()
    if any(allowed.lower() in reason for allowed in _ALLOWED_SKIP_SUBSTRINGS):
        return
    _unexpected_skips.append((report.nodeid, _skip_reason(report)))


def pytest_sessionfinish(session, exitstatus):
    if not _unexpected_skips:
        return
    import sys

    print(
        "\n=== SKIP GATE: unexpected skips (not in allowlist) — treated as "
        "failures ===",
        file=sys.stderr,
    )
    for nodeid, reason in _unexpected_skips:
        print(f"  SKIP  {nodeid}\n        {reason}", file=sys.stderr)
    if session.exitstatus == 0:
        session.exitstatus = 1


@pytest.fixture
def sample_csv_content():
    """A small CSV representing sales data."""
    return b"""date,product,region,revenue,units
2024-01-01,Widget A,North,1200.50,10
2024-01-01,Widget B,South,850.00,8
2024-01-02,Widget A,East,2100.75,18
2024-01-02,Widget C,West,450.25,4
2024-01-03,Widget B,North,1650.00,15
"""
