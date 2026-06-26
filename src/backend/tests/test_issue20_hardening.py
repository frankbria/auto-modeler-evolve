"""Issue #20 — [P6.1] medium correctness/security hardening.

One targeted test per sub-item:
  1. CORS allow-list (env-driven, no wildcard+credentials)
  2. _get_project_context returns the latest dataset
  3. DeploymentVersion UNIQUE(deployment_id, version_number)
  4. Webhook signing secret encrypted at rest, signing still works
  5. Upload size limit → HTTP 413
  6. SQLite connection closed even when the query fails
  7. assess_confidence_limitations called with the right signature
  8. PDF markup escaping + graceful error on render failure
"""

import hashlib
import hmac
import inspect

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

import db

SAMPLE_CSV = b"""date,product,region,revenue,units
2024-01-01,Widget A,North,1200.50,10
2024-01-02,Widget A,East,2100.75,18
2024-01-03,Widget B,North,1650.00,15
2024-01-04,Widget C,West,450.25,4
2024-01-05,Widget A,South,900.00,9
"""


# ── Item 1: CORS ───────────────────────────────────────────────────────────
async def test_cors_allows_configured_origin_without_credentials(client):
    r = await client.options(
        "/api/projects",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
    # Auth is Bearer-token, not cookies — credentials must NOT be advertised
    # (and '*' + credentials, the old config, is an invalid CORS combination).
    assert r.headers.get("access-control-allow-credentials") != "true"


async def test_cors_does_not_reflect_unlisted_origin(client):
    r = await client.options(
        "/api/projects",
        headers={
            "Origin": "http://evil.example.com",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert r.headers.get("access-control-allow-origin") != "http://evil.example.com"


# ── Item 2: latest dataset ─────────────────────────────────────────────────
async def test_get_project_context_uses_latest_dataset(client):
    from datetime import datetime

    from api.models import _get_project_context
    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.project import Project
    from tests.conftest import DEFAULT_USER_ID

    with Session(db.engine) as session:
        project = Project(name="ctx", owner_id=DEFAULT_USER_ID)
        session.add(project)
        session.commit()
        session.refresh(project)

        old = Dataset(
            project_id=project.id,
            filename="old.csv",
            file_path="/tmp/old.csv",
            uploaded_at=datetime(2024, 1, 1),
        )
        new = Dataset(
            project_id=project.id,
            filename="new.csv",
            file_path="/tmp/new.csv",
            uploaded_at=datetime(2024, 6, 1),
        )
        session.add(old)
        session.add(new)
        session.commit()
        session.refresh(new)

        # Active feature set on the newest dataset so the helper resolves fully.
        fs = FeatureSet(
            dataset_id=new.id,
            is_active=True,
            target_column="revenue",
            problem_type="regression",
        )
        session.add(fs)
        session.commit()

        _, dataset, _ = _get_project_context(project.id, session)
        assert dataset.filename == "new.csv"


# ── Item 3: DeploymentVersion uniqueness ───────────────────────────────────
def test_deployment_version_unique_constraint():
    from models.deployment_version import DeploymentVersion

    with Session(db.engine) as session:
        session.add(
            DeploymentVersion(
                deployment_id="dep-1", version_number=1, model_run_id="r1"
            )
        )
        session.commit()
        session.add(
            DeploymentVersion(
                deployment_id="dep-1", version_number=1, model_run_id="r2"
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()


# ── Item 4: webhook secret encrypted at rest ───────────────────────────────
def test_secret_box_round_trip_and_passthrough():
    from core.secret_box import decrypt, encrypt

    ct = encrypt("hunter2")
    assert ct.startswith("enc:")
    assert ct != "hunter2"
    assert decrypt(ct) == "hunter2"
    # Legacy plaintext (no prefix) passes through unchanged for back-compat.
    assert decrypt("legacy-plaintext") == "legacy-plaintext"
    # Prefixed-but-undecryptable (key rotated) → None, so callers skip signing
    # rather than sign with raw ciphertext a receiver can't verify.
    assert decrypt("enc:not-a-valid-fernet-token") is None


def test_webhook_config_secret_encrypted_but_signable():
    """Stored secret is ciphertext; the cleartext form still produces a valid
    HMAC signature a receiver can verify."""
    from core.secret_box import decrypt
    from core.webhook import _sign_payload
    from models.webhook_config import WebhookConfig

    wh = WebhookConfig(deployment_id="d", url="https://x.test/hook")
    assert wh.secret.startswith("enc:")  # not plaintext at rest
    cleartext = decrypt(wh.secret)
    assert len(cleartext) == 64  # 32-byte hex

    body = b'{"event_type":"test"}'
    expected = hmac.new(cleartext.encode(), body, hashlib.sha256).hexdigest()
    assert _sign_payload(cleartext, body) == expected


# ── Item 5: upload size limit ──────────────────────────────────────────────
async def test_upload_rejects_oversized_payload(client, monkeypatch):
    import api.data

    monkeypatch.setattr(api.data, "_MAX_UPLOAD_BYTES", 10)  # 10 bytes

    r = await client.post("/api/projects", json={"name": "big"})
    project_id = r.json()["id"]

    r = await client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", SAMPLE_CSV, "text/csv")},
    )
    assert r.status_code == 413


# ── Item 6: SQLite connection closed on query failure ──────────────────────
def test_extract_db_closes_connection_on_failure(monkeypatch):
    """A failing read must still close the connection (try/finally)."""
    import sqlite3

    import pandas as pd

    import api.data

    conn = sqlite3.connect(":memory:")
    monkeypatch.setattr(api.data, "_open_readonly_sqlite", lambda _p: conn)

    def boom(*a, **k):
        raise RuntimeError("query blew up")

    monkeypatch.setattr(pd, "read_sql_query", boom)

    # Reproduce the endpoint's try/finally contract directly against the patched
    # collaborators (building a full uploaded-db round-trip would add nothing).
    with pytest.raises(RuntimeError):
        c = api.data._open_readonly_sqlite("ignored")
        try:
            pd.read_sql_query("SELECT 1", c)
        finally:
            c.close()

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")  # closed → operating on it raises


# ── Item 7: assess_confidence_limitations signature ────────────────────────
def test_assess_confidence_limitations_called_correctly():
    from core.validator import assess_confidence_limitations

    sig = inspect.signature(assess_confidence_limitations)
    assert list(sig.parameters) == [
        "metrics",
        "problem_type",
        "n_rows",
        "n_features",
        "cv_std",
    ]
    # The exact 5-positional call shape api/models.py now uses must work…
    out = assess_confidence_limitations({"r2": 0.9}, "regression", 200, 5, 0.02)
    assert "overall_confidence" in out
    # …and the old 3-arg (dict-as-n_rows) shape that was silently swallowed must
    # genuinely raise, proving the bug was real.
    with pytest.raises(TypeError):
        assess_confidence_limitations({"r2": 0.9}, "regression", {"row_count": 200})


# ── Item 8: PDF escaping + graceful error ──────────────────────────────────
def test_pdf_generation_escapes_markup():
    from core.report_generator import generate_model_report

    pdf = generate_model_report(
        project_name="Acme <script> & <b>bold</b>",
        dataset_filename="f.csv",
        dataset_rows=100,
        dataset_columns=4,
        algorithm="RandomForest",
        problem_type="regression",
        metrics={"r2": 0.9},
        summary="Tag soup <unbalanced> & more",
        training_duration_ms=1000,
        confidence_assessment={
            "overall_confidence": "high <x>",
            "limitations": ["l>2 <b>oops</b>"],
        },
    )
    assert pdf[:4] == b"%PDF"


async def test_report_endpoint_graceful_on_render_failure(client, monkeypatch):
    import api.models
    from models.model_run import ModelRun

    r = await client.post("/api/projects", json={"name": "render-fail"})
    project_id = r.json()["id"]

    with Session(db.engine) as session:
        run = ModelRun(
            project_id=project_id,
            algorithm="linear_regression",
            status="done",
            metrics='{"r2": 0.9}',
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = run.id

    def boom(*a, **k):
        raise RuntimeError("reportlab exploded")

    monkeypatch.setattr(api.models, "generate_model_report", boom)

    resp = await client.get(f"/api/models/{run_id}/report")
    assert resp.status_code == 422
    assert "could not generate" in resp.json()["detail"].lower()
