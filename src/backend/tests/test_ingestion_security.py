"""Security regression tests for issue #3 (P1.3).

Covers the four data-ingestion vectors the audit flagged:
  1. SSRF via URL import (POST /api/data/upload-url)
  2. SSRF via webhook register + dispatch
  3. Path traversal on file write (upload / merge / extract-db output)
  4. Arbitrary local file read via extract-db db_path

Plus unit tests for the two guards (core.ssrf_guard, core.path_safety) that
back every call site.
"""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path
from unittest import mock

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel, Session, create_engine

import db as db_module

_SAMPLE_CSV = b"id,region,revenue\n" b"1,North,100\n2,South,200\n3,East,300\n"


# ===========================================================================
# Unit tests — core.ssrf_guard
# ===========================================================================


class TestSsrfGuard:
    def test_blocks_loopback_ip_literal(self):
        from core.ssrf_guard import UnsafeURLError, assert_safe_url

        with pytest.raises(UnsafeURLError):
            assert_safe_url("http://127.0.0.1/x")

    def test_blocks_metadata_endpoint(self):
        from core.ssrf_guard import UnsafeURLError, assert_safe_url

        with pytest.raises(UnsafeURLError):
            assert_safe_url("http://169.254.169.254/latest/meta-data/")

    @pytest.mark.parametrize(
        "url",
        [
            "http://10.0.0.5/x",
            "http://172.16.3.4/x",
            "http://192.168.1.1/x",
            "http://0.0.0.0/x",
            "http://[::1]/x",
            "http://[fc00::1]/x",
        ],
    )
    def test_blocks_private_ranges(self, url):
        from core.ssrf_guard import UnsafeURLError, assert_safe_url

        with pytest.raises(UnsafeURLError):
            assert_safe_url(url)

    def test_blocks_ipv4_mapped_ipv6_loopback(self):
        from core.ssrf_guard import UnsafeURLError, assert_safe_url

        with pytest.raises(UnsafeURLError):
            assert_safe_url("http://[::ffff:127.0.0.1]/x")

    def test_rejects_non_http_scheme(self):
        from core.ssrf_guard import UnsafeURLError, assert_safe_url

        with pytest.raises(UnsafeURLError):
            assert_safe_url("ftp://example.com/x")
        with pytest.raises(UnsafeURLError):
            assert_safe_url("file:///etc/passwd")

    def test_allows_public_host(self):
        from core.ssrf_guard import assert_safe_url

        # Resolves to a public address — must not raise.
        assert_safe_url("https://example.com/data.csv")

    def test_localhost_hostname_blocked(self):
        from core.ssrf_guard import UnsafeURLError, assert_safe_url

        with pytest.raises(UnsafeURLError):
            assert_safe_url("http://localhost:8000/x")

    def test_unresolved_host_respects_flag(self):
        from core.ssrf_guard import UnsafeURLError, assert_safe_url

        host = "http://nonexistent.invalid-tld-xyz.example/x"
        # allow_unresolved=True (registration) — tolerate a host that won't resolve.
        assert_safe_url(host, allow_unresolved=True)
        # allow_unresolved=False (about to connect) — reject it.
        with pytest.raises(UnsafeURLError):
            assert_safe_url(host, allow_unresolved=False)

    def test_private_literal_blocked_even_when_unresolved_allowed(self):
        from core.ssrf_guard import UnsafeURLError, assert_safe_url

        with pytest.raises(UnsafeURLError):
            assert_safe_url("http://127.0.0.1/x", allow_unresolved=True)


# ===========================================================================
# Unit tests — core.path_safety
# ===========================================================================


class TestPathSafety:
    def test_sanitize_strips_directory(self):
        from core.path_safety import sanitize_filename

        assert sanitize_filename("../../../etc/passwd") == "passwd"
        assert sanitize_filename("/abs/evil.csv") == "evil.csv"
        assert sanitize_filename("data.csv") == "data.csv"

    @pytest.mark.parametrize("name", ["", "   ", "..", ".", "/", "../"])
    def test_sanitize_rejects_unsafe(self, name):
        from core.path_safety import UnsafePathError, sanitize_filename

        with pytest.raises(UnsafePathError):
            sanitize_filename(name)

    def test_resolve_within_contains(self, tmp_path):
        from core.path_safety import resolve_within

        out = resolve_within(tmp_path, "../../evil.csv")
        assert out.parent == tmp_path.resolve()
        assert out.name == "evil.csv"

    def test_assert_within_blocks_escape(self, tmp_path):
        from core.path_safety import UnsafePathError, assert_within

        base = tmp_path / "uploads"
        base.mkdir()
        with pytest.raises(UnsafePathError):
            assert_within(base, Path("/etc/passwd"))

    def test_assert_within_allows_contained(self, tmp_path):
        from core.path_safety import assert_within

        base = tmp_path / "uploads"
        base.mkdir()
        ok = base / "child.db"
        assert assert_within(base, ok) == ok.resolve()


# ===========================================================================
# Integration fixtures
# ===========================================================================


@pytest.fixture
async def ac(tmp_path):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models  # noqa: F401 — register every table

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    data_module._DB_UPLOADS_DIR = tmp_path / "db_uploads"

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
async def project_id(ac):
    r = await ac.post("/api/projects", json={"name": "Security Test"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _make_sqlite_bytes(tmp_path: Path) -> bytes:
    p = tmp_path / "_in.db"
    conn = sqlite3.connect(str(p))
    conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "x"), (2, "y")])
    conn.commit()
    conn.close()
    return p.read_bytes()


# ===========================================================================
# SSRF — URL import
# ===========================================================================


class TestUrlImportSsrf:
    @pytest.mark.parametrize(
        "url",
        [
            "http://127.0.0.1/data.csv",
            "http://169.254.169.254/latest/meta-data/",
            "http://10.0.0.1/data.csv",
            "http://192.168.0.5/data.csv",
            "http://localhost:6379/data.csv",
        ],
    )
    async def test_private_url_rejected_before_fetch(self, ac, project_id, url):
        """SSRF targets are rejected with 400 and the fetch is never reached."""
        with mock.patch("api.data.safe_urlopen") as m:
            r = await ac.post(
                "/api/data/upload-url",
                json={"project_id": project_id, "url": url},
            )
        assert r.status_code == 400, r.text
        m.assert_not_called()

    async def test_public_url_still_works(self, ac, project_id):
        """A public host passes the guard and imports normally (fetch mocked)."""
        cm = mock.MagicMock()
        resp = mock.MagicMock()
        resp.read.return_value = _SAMPLE_CSV
        cm.__enter__ = mock.MagicMock(return_value=resp)
        cm.__exit__ = mock.MagicMock(return_value=False)
        with mock.patch("api.data.safe_urlopen", return_value=cm):
            r = await ac.post(
                "/api/data/upload-url",
                json={"project_id": project_id, "url": "https://example.com/data.csv"},
            )
        assert r.status_code == 201, r.text

    def test_redirect_to_private_host_blocked(self):
        """A 3xx Location pointing at an internal host is rejected per-hop."""
        import io as _io
        import urllib.request

        from core.ssrf_guard import UnsafeURLError, _ValidatingRedirectHandler

        handler = _ValidatingRedirectHandler(allow_unresolved=False)
        req = urllib.request.Request("http://example.com/")
        with pytest.raises(UnsafeURLError):
            handler.redirect_request(
                req,
                _io.BytesIO(b""),
                302,
                "Found",
                {},
                "http://169.254.169.254/latest/meta-data/",
            )


# ===========================================================================
# Path traversal — file writes
# ===========================================================================


class TestPathTraversal:
    async def test_upload_filename_traversal_contained(self, ac, project_id, tmp_path):
        r = await ac.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={
                "file": (
                    "../../../../tmp/pwned.csv",
                    io.BytesIO(_SAMPLE_CSV),
                    "text/csv",
                )
            },
        )
        assert r.status_code == 201, r.text
        # Nothing escaped the uploads dir.
        assert not (tmp_path / "pwned.csv").exists()
        assert not (Path("/tmp") / "pwned.csv").exists()
        assert (tmp_path / "uploads" / project_id / "pwned.csv").exists()

    async def test_merge_save_as_traversal_contained(self, ac, project_id, tmp_path):
        # Upload two datasets sharing the "id" column.
        for name in ("a.csv", "b.csv"):
            r = await ac.post(
                "/api/data/upload",
                data={"project_id": project_id},
                files={"file": (name, io.BytesIO(_SAMPLE_CSV), "text/csv")},
            )
            assert r.status_code == 201, r.text

        ds = (await ac.get(f"/api/data/project/{project_id}/datasets")).json()
        ids = [d["dataset_id"] for d in ds]

        r = await ac.post(
            f"/api/data/{project_id}/merge",
            json={
                "dataset_id_1": ids[0],
                "dataset_id_2": ids[1],
                "join_key": "id",
                "save_as_filename": "../../../../tmp/merged_pwned.csv",
            },
        )
        assert r.status_code == 201, r.text
        assert not (Path("/tmp") / "merged_pwned.csv").exists()
        assert (tmp_path / "uploads" / project_id / "merged_pwned.csv").exists()

    async def test_extract_db_save_as_traversal_contained(
        self, ac, project_id, tmp_path
    ):
        up = await ac.post(
            "/api/data/upload-db",
            data={"project_id": project_id},
            files={
                "file": (
                    "d.db",
                    io.BytesIO(_make_sqlite_bytes(tmp_path)),
                    "application/octet-stream",
                )
            },
        )
        assert up.status_code == 201, up.text
        db_path = up.json()["db_path"]

        r = await ac.post(
            "/api/data/extract-db",
            json={
                "project_id": project_id,
                "db_path": db_path,
                "table_name": "t",
                "save_as_filename": "../../../../tmp/extract_pwned",
            },
        )
        assert r.status_code == 201, r.text
        assert not (Path("/tmp") / "extract_pwned.csv").exists()
        assert (tmp_path / "uploads" / project_id / "extract_pwned.csv").exists()


# ===========================================================================
# Arbitrary file read — extract-db db_path confinement
# ===========================================================================


class TestExtractDbConfinement:
    async def test_absolute_db_path_outside_dir_rejected(self, ac, project_id):
        """A db_path pointing outside the project's upload dir is 404 (no leak)."""
        r = await ac.post(
            "/api/data/extract-db",
            json={
                "project_id": project_id,
                "db_path": "/etc/hosts",
                "table_name": "t",
            },
        )
        assert r.status_code == 404, r.text

    async def test_traversal_db_path_rejected(self, ac, project_id, tmp_path):
        """A real SQLite file outside the upload dir cannot be read via ../."""
        secret = tmp_path / "secret.db"
        conn = sqlite3.connect(str(secret))
        conn.execute("CREATE TABLE s (v TEXT)")
        conn.execute("INSERT INTO s VALUES ('tenant-data')")
        conn.commit()
        conn.close()

        # Craft a traversal from the (expected) db_uploads/{project} dir.
        traversal = f"db_uploads/{project_id}/../../secret.db"
        r = await ac.post(
            "/api/data/extract-db",
            json={
                "project_id": project_id,
                "db_path": str(tmp_path / traversal),
                "table_name": "s",
            },
        )
        assert r.status_code == 404, r.text

    async def test_legitimate_db_path_still_works(self, ac, project_id, tmp_path):
        up = await ac.post(
            "/api/data/upload-db",
            data={"project_id": project_id},
            files={
                "file": (
                    "d.db",
                    io.BytesIO(_make_sqlite_bytes(tmp_path)),
                    "application/octet-stream",
                )
            },
        )
        db_path = up.json()["db_path"]
        r = await ac.post(
            "/api/data/extract-db",
            json={"project_id": project_id, "db_path": db_path, "table_name": "t"},
        )
        assert r.status_code == 201, r.text
        assert r.json()["row_count"] == 2

    async def test_hardened_connection_blocks_writes_and_attach(
        self, ac, project_id, tmp_path
    ):
        """The hardened connection blocks both writes (mode=ro) and ATTACH.

        ATTACH is the real arbitrary-file-read vector — ``mode=ro`` alone does
        NOT stop it, so the authorizer must.
        """
        from api.data import _open_readonly_sqlite

        up = await ac.post(
            "/api/data/upload-db",
            data={"project_id": project_id},
            files={
                "file": (
                    "d.db",
                    io.BytesIO(_make_sqlite_bytes(tmp_path)),
                    "application/octet-stream",
                )
            },
        )
        db_path = Path(up.json()["db_path"])

        # A real SQLite file outside the upload dir that ATTACH would try to read.
        victim = tmp_path / "victim.db"
        vconn = sqlite3.connect(str(victim))
        vconn.execute("CREATE TABLE secret (v TEXT)")
        vconn.execute("INSERT INTO secret VALUES ('tenant-data')")
        vconn.commit()
        vconn.close()

        conn = _open_readonly_sqlite(db_path)
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO t VALUES (9, 'z')")
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(f"ATTACH DATABASE '{victim}' AS v")
        conn.close()


# ===========================================================================
# SSRF — webhook register + dispatch
# ===========================================================================


def _seed_deployment(project_id: str) -> str:
    """Insert a minimal active Deployment owned (transitively) by the project."""
    from models.deployment import Deployment

    with Session(db_module.engine) as session:
        dep = Deployment(
            model_run_id="run-x",
            project_id=project_id,
            endpoint_path="/api/predict/x",
            dashboard_url="/predict/x",
            is_active=True,
        )
        session.add(dep)
        session.commit()
        session.refresh(dep)
        return dep.id


class TestWebhookSsrf:
    async def test_register_private_url_rejected(self, ac, project_id):
        dep_id = _seed_deployment(project_id)
        r = await ac.post(
            f"/api/deploy/{dep_id}/webhooks",
            json={
                "url": "http://169.254.169.254/hook",
                "event_types": ["batch_complete"],
            },
        )
        assert r.status_code == 400, r.text

    async def test_register_public_url_allowed(self, ac, project_id):
        dep_id = _seed_deployment(project_id)
        r = await ac.post(
            f"/api/deploy/{dep_id}/webhooks",
            json={"url": "https://example.com/hook", "event_types": ["batch_complete"]},
        )
        assert r.status_code == 201, r.text

    async def test_register_unresolvable_host_allowed(self, ac, project_id):
        """A not-yet-resolving staging host can still be registered."""
        dep_id = _seed_deployment(project_id)
        r = await ac.post(
            f"/api/deploy/{dep_id}/webhooks",
            json={
                "url": "https://staging.invalid-tld-xyz.example/hook",
                "event_types": ["batch_complete"],
            },
        )
        assert r.status_code == 201, r.text

    def test_dispatch_blocks_private_without_network(self):
        """_do_dispatch refuses a private target and never calls urlopen."""
        from core.webhook import _do_dispatch

        with mock.patch("urllib.request.urlopen") as m:
            status = _do_dispatch("wid", "http://127.0.0.1:9999/hook", "secret", {})
        assert status == 0
        m.assert_not_called()
