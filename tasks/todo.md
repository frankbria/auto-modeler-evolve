# Issue #20 — [P6.1] Medium correctness/security hardening

8 sub-items, grouped in 3 phases. Backend only. TDD: targeted test per item.
Line numbers verified live.

All 8 items implemented + tested. Also fixed a latent report bug: report read
`confidence_level`/`strengths` but the validator emits `overall_confidence`/
`limitations` — aligned the report to the real shape (the section never rendered
before because #7's call always raised).

## Phase 1 — HTTP layer (main.py, api/data.py)
- [x] **1. CORS** (`main.py:77-83`) — env-driven `CORS_ORIGINS` (comma-sep) allow-list;
      default `http://localhost:3000`. Per owner comment (auth is Bearer, not cookie):
      `allow_credentials=False`. Test asserts configured origins, no wildcard+creds.
- [x] **5. Upload size limit** — `MAX_UPLOAD_SIZE_MB` env (default 100). Helper checks
      `Content-Length` → 413 before read; also length-check bytes after read (header is
      spoofable/absent). Apply to `/upload` (160), `/upload-db` (1184), `/{id}/refresh`
      (1624), `/upload-url` (after fetch). Test: oversized → 413.
- [x] **6. SQLite leak** — try/finally close in `/upload-db` (1187-1193) and
      `/extract-db` (1261-1263). Test: failing query still closes conn.

## Phase 2 — Persistence (models/, api/models.py)
- [x] **2. Dataset ordering** (`api/models.py:91-93`) — add
      `.order_by(Dataset.uploaded_at.desc())`. Test: latest dataset returned.
- [x] **3. DeploymentVersion UNIQUE** (`models/deployment_version.py`) — add
      `UniqueConstraint("deployment_id","version_number")`. Counter increment
      (`deploy.py:339`) is read-modify-write; SQLite write-lock serializes, UNIQUE
      enforces integrity. Test: duplicate (dep_id, ver) raises IntegrityError.
- [x] **4. Webhook secret at rest** — DEVIATION: secret signs OUTBOUND HMAC
      (`core/webhook.py:71`), so it must be REVERSIBLE → encrypt (Fernet, cryptography
      41 already installed), key from `WEBHOOK_SECRET_KEY`/derived from `AUTH_SECRET`.
      NOT hash (would break signing). API already returns secret once at creation;
      `list_webhooks` excludes it (done). Store ciphertext w/ marker prefix for
      plaintext back-compat; decrypt at dispatch. Test: stored value != plaintext,
      signing still verifies.
- [x] **7. assess_confidence_limitations call** (`api/models.py:1627`) — real bug:
      fn takes `(metrics, problem_type, n_rows, n_features, cv_std)` (5 args), call
      passes 3 with a dict 3rd → TypeError swallowed → section dropped. Fix:
      `(metrics, problem_type, dataset_rows, dataset_columns, metrics.get("cv_std"))`.
      Narrow except. Regression test: section present.

## Phase 3 — PDF (core/report_generator.py, api/models.py)
- [x] **8. ReportLab injection** — `xml_escape = html_escape` alias; escape
      `project_name` (118,136), `summary` (163), `level/strengths/limitations`
      (191,197,203), feature names (180). Wrap `generate_model_report()` in
      `download_report` (1633) try/except → 422 + log. Test: `<>` in project name
      generates OK; failure → graceful error.
