# Issue #4 — Cascade project delete + artifact GC

*2026-06-17T01:35:45Z*

Issue #4 (P2.1, blocker, data-integrity): `delete_project` removed only the Project row, orphaning ~23 child tables and leaking every on-disk artifact. This demo proves each acceptance criterion with before/after outcome evidence, fully isolated under a temp DATA_DIR (nothing touches the real data tree).

The driver seeds a project with one row in EVERY child table (24 rows across 4 levels) plus all four artifact types, then deletes via `delete_project_cascade` and prints row counts + artifact existence before/after. Criterion 1: all child rows + artifact dirs removed in one commit. Criterion 2: PRAGMA foreign_keys=ON on a fresh connection. Criterion 3: the janitor reaps orphan artifacts, keeps live-project artifacts, and age-gates upload retention.

```bash
uv run python demo_issue4.py 2>/dev/null
```

```output
========================================================================
CRITERION 1 — cascade delete: all child rows + artifacts in one commit
========================================================================

Row counts BEFORE delete:
  Project                  1
  Dataset                  1
  ModelRun                 1
  Deployment               1
  Conversation             1
  AnalysisTemplate         1
  FeatureSet               1
  DatasetFilter            1
  PredictionLog            1
  FeedbackRecord           1
  DeploymentChangelog      1
  DeploymentPreset         1
  DashboardFieldConfig     1
  InputValidationRule      1
  PredictionAlertRule      1
  SavedScenario            1
  WebhookConfig            1
  WebhookEvent             1
  WeeklyDigestConfig       1
  GoalSeekRecord           1
  DeploymentVersion        1
  BatchSchedule            1
  BatchJobRun              1
  ABTest                   1
  TOTAL ROWS               24

Artifact dirs/files BEFORE delete:
  [EXISTS] /tmp/demo_issue4_yq50tywd/uploads/demo-project-1
  [EXISTS] /tmp/demo_issue4_yq50tywd/db_uploads/demo-project-1
  [EXISTS] /tmp/demo_issue4_yq50tywd/models/demo-project-1
  [EXISTS] deployments/*_pipeline.joblib (1 file)

>>> delete_project_cascade() summary: {'removed_dirs': 3, 'removed_files': 1}

Row counts AFTER delete:
  Project                  0
  Dataset                  0
  ModelRun                 0
  Deployment               0
  Conversation             0
  AnalysisTemplate         0
  FeatureSet               0
  DatasetFilter            0
  PredictionLog            0
  FeedbackRecord           0
  DeploymentChangelog      0
  DeploymentPreset         0
  DashboardFieldConfig     0
  InputValidationRule      0
  PredictionAlertRule      0
  SavedScenario            0
  WebhookConfig            0
  WebhookEvent             0
  WeeklyDigestConfig       0
  GoalSeekRecord           0
  DeploymentVersion        0
  BatchSchedule            0
  BatchJobRun              0
  ABTest                   0
  TOTAL ROWS               0

Artifact dirs/files AFTER delete:
  [gone  ] /tmp/demo_issue4_yq50tywd/uploads/demo-project-1
  [gone  ] /tmp/demo_issue4_yq50tywd/db_uploads/demo-project-1
  [gone  ] /tmp/demo_issue4_yq50tywd/models/demo-project-1
  [gone  ] deployments/*_pipeline.joblib (0 file)

RESULT: 24 rows -> 0, all artifacts removed. PASS

========================================================================
CRITERION 2 — PRAGMA foreign_keys=ON on a fresh connection
========================================================================

  PRAGMA foreign_keys -> 1   (ON)
RESULT: foreign-key enforcement is ON. PASS

========================================================================
CRITERION 3 — janitor GCs orphan artifacts, keeps live, enforces retention
========================================================================

  live   upload dir exists: True  (/tmp/demo_issue4_yq50tywd/uploads/live-project)
  orphan upload dir exists: True  (/tmp/demo_issue4_yq50tywd/uploads/ghost-project)
  orphan models dir exists: True  (/tmp/demo_issue4_yq50tywd/models/ghost-project)

>>> collect_orphans() summary: {'removed_dirs': 2, 'removed_files': 0}

  live   upload dir exists: True  (kept)
  orphan upload dir exists: False  (reaped)
  orphan models dir exists: False  (reaped)

  enforce_upload_retention(30d): removed 0, dir exists=True
  enforce_upload_retention(7d):  removed 1, dir exists=False
RESULT: orphans reaped, live kept, retention age-gated. PASS

ALL CRITERIA DEMONSTRATED WITH OUTCOME EVIDENCE.
```

Criterion 4 — the test suite asserts zero orphan rows across ALL 23 child tables + removed artifacts after delete, cross-tenant DELETE is 404 and non-destructive, and the janitor/storage behaviors hold. Run the issue-#4 tests through the real API (authenticated `client` fixture):

```bash
uv run pytest tests/test_project_cascade_delete.py tests/test_janitor.py tests/test_storage_layout.py -p no:cacheprovider -v 2>/dev/null | grep -E "PASSED|FAILED|passed|failed"
```

```output
tests/test_project_cascade_delete.py::test_delete_removes_all_child_rows_and_artifacts PASSED [  9%]
tests/test_project_cascade_delete.py::test_delete_is_scoped_to_one_project PASSED [ 18%]
tests/test_project_cascade_delete.py::test_cross_tenant_delete_forbidden_and_nondestructive PASSED [ 27%]
tests/test_project_cascade_delete.py::test_sqlite_foreign_keys_pragma_enabled PASSED [ 36%]
tests/test_janitor.py::test_collect_orphans_removes_orphan_dirs_keeps_live PASSED [ 45%]
tests/test_janitor.py::test_collect_orphans_removes_orphan_pipelines_and_batch_outputs PASSED [ 54%]
tests/test_janitor.py::test_collect_orphans_respects_upload_grace PASSED [ 63%]
tests/test_janitor.py::test_enforce_upload_retention_age_gated PASSED    [ 72%]
tests/test_janitor.py::test_run_janitor_never_raises_and_reports PASSED  [ 81%]
tests/test_storage_layout.py::test_storage_layout_matches_api_constants PASSED [ 90%]
tests/test_storage_layout.py::test_pipeline_path_matches_deploy_naming PASSED [100%]
======================= 11 passed, 4 warnings in 10.84s ========================
```

**All four acceptance criteria demonstrated with outcome evidence:** (1) 24 child rows → 0 and all artifact dirs/files removed in one commit; (2) `PRAGMA foreign_keys` returns 1; (3) janitor reaps orphans, keeps live, age-gates retention; (4) 11/11 tests pass through the real authenticated API, including the cross-tenant-404 non-destructive guard.
