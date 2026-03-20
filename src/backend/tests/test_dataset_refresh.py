"""Tests for dataset refresh: POST /api/data/{dataset_id}/refresh.

Coverage:
  - Successful refresh — column counts updated, profile recomputed
  - Added columns detected in new_columns list
  - Removed columns detected in removed_columns list
  - Feature set compatibility check (feature_columns_missing, compatible flag)
  - 404 on unknown dataset
  - 400 on invalid file type
  - 400 on empty file
  - chat._REFRESH_PATTERNS regex matches expected phrases
  - Chat SSE emits {type: refresh_prompt} event when dataset present
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CSV_ORIGINAL = b"""date,product,region,revenue,units
2024-01-01,Widget A,North,1200.50,10
2024-01-01,Widget B,South,850.00,8
2024-01-02,Widget A,East,2100.75,18
"""

CSV_COMPATIBLE = b"""date,product,region,revenue,units
2024-02-01,Widget A,North,1400.00,12
2024-02-01,Widget B,South,900.00,9
2024-02-02,Widget C,East,2200.00,20
2024-02-03,Widget D,West,500.00,5
"""

CSV_WITH_NEW_COL = b"""date,product,region,revenue,units,discount
2024-02-01,Widget A,North,1400.00,12,5.0
2024-02-01,Widget B,South,900.00,9,2.5
"""

CSV_MISSING_COL = b"""date,product,region,revenue
2024-02-01,Widget A,North,1400.00
2024-02-01,Widget B,South,900.00
"""

CSV_EMPTY = b"date,product\n"


def _setup_client_and_db(tmp: str):
    """Create a fresh in-memory DB and return a configured TestClient."""
    from main import app
    from sqlmodel import create_engine, SQLModel
    import db
    import models.project, models.dataset, models.feature_set  # noqa
    import models.conversation, models.model_run, models.deployment  # noqa
    import models.prediction_log  # noqa

    db.engine = create_engine(f"sqlite:///{tmp}/test.db", echo=False)
    SQLModel.metadata.create_all(db.engine)
    db.DATA_DIR = Path(tmp)
    return TestClient(app)


def _make_project(client: TestClient) -> str:
    r = client.post("/api/projects", json={"name": "Refresh Test"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def _upload(client: TestClient, content: bytes, project_id: str) -> str:
    r = client.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("data.csv", io.BytesIO(content), "text/csv")},
    )
    assert r.status_code == 201, r.text
    return r.json()["dataset_id"]


def _refresh(client: TestClient, dataset_id: str, content: bytes, filename: str = "data.csv"):
    return client.post(
        f"/api/data/{dataset_id}/refresh",
        files={"file": (filename, io.BytesIO(content), "text/csv")},
    )


# ---------------------------------------------------------------------------
# API: POST /api/data/{dataset_id}/refresh
# ---------------------------------------------------------------------------


class TestRefreshEndpoint:
    def test_compatible_refresh_updates_row_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = _setup_client_and_db(tmp)
            pid = _make_project(client)
            ds_id = _upload(client, CSV_ORIGINAL, pid)

            r = _refresh(client, ds_id, CSV_COMPATIBLE)
            assert r.status_code == 200, r.text
            body = r.json()

            assert body["dataset_id"] == ds_id
            assert body["row_count"] == 4   # CSV_COMPATIBLE has 4 data rows
            assert body["compatible"] is True
            assert body["new_columns"] == []
            assert body["removed_columns"] == []
            assert body["feature_columns_missing"] == []

    def test_refresh_detects_added_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = _setup_client_and_db(tmp)
            pid = _make_project(client)
            ds_id = _upload(client, CSV_ORIGINAL, pid)

            r = _refresh(client, ds_id, CSV_WITH_NEW_COL)
            assert r.status_code == 200, r.text
            body = r.json()

            assert "discount" in body["new_columns"]
            assert body["compatible"] is True    # no feature set → not blocking

    def test_refresh_detects_removed_columns_no_feature_set(self):
        """Without a FeatureSet, removed columns are reported but compatible=True."""
        with tempfile.TemporaryDirectory() as tmp:
            client = _setup_client_and_db(tmp)
            pid = _make_project(client)
            ds_id = _upload(client, CSV_ORIGINAL, pid)

            r = _refresh(client, ds_id, CSV_MISSING_COL)
            assert r.status_code == 200, r.text
            body = r.json()

            assert "units" in body["removed_columns"]
            # No feature set → compatible=True (just a warning)
            assert body["compatible"] is True
            assert body["feature_columns_missing"] == []

    def test_refresh_incompatible_when_feature_set_requires_missing_col(self):
        """When active FeatureSet requires 'units' and new CSV drops it → incompatible."""
        with tempfile.TemporaryDirectory() as tmp:
            client = _setup_client_and_db(tmp)
            pid = _make_project(client)
            ds_id = _upload(client, CSV_ORIGINAL, pid)

            # Create a feature set referencing "units" column
            from sqlmodel import Session
            import db as _db
            from models.feature_set import FeatureSet

            with Session(_db.engine) as sess:
                fs = FeatureSet(
                    dataset_id=ds_id,
                    column_mapping=json.dumps({"units": ["units"], "revenue": ["revenue"]}),
                    target_column="revenue",
                    is_active=True,
                )
                sess.add(fs)
                sess.commit()

            r = _refresh(client, ds_id, CSV_MISSING_COL)
            assert r.status_code == 200, r.text
            body = r.json()

            assert body["compatible"] is False
            assert "units" in body["feature_columns_missing"]

    def test_refresh_preview_matches_new_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = _setup_client_and_db(tmp)
            pid = _make_project(client)
            ds_id = _upload(client, CSV_ORIGINAL, pid)

            r = _refresh(client, ds_id, CSV_COMPATIBLE)
            body = r.json()

            assert len(body["preview"]) > 0
            # First row of CSV_COMPATIBLE is 2024-02-01
            assert body["preview"][0]["date"] == "2024-02-01"

    def test_refresh_updates_profile_in_db(self):
        """After refresh, GET /profile returns updated stats."""
        with tempfile.TemporaryDirectory() as tmp:
            client = _setup_client_and_db(tmp)
            pid = _make_project(client)
            ds_id = _upload(client, CSV_ORIGINAL, pid)

            _refresh(client, ds_id, CSV_COMPATIBLE)

            r = client.get(f"/api/data/{ds_id}/profile")
            assert r.status_code == 200
            profile = r.json()
            assert profile["row_count"] == 4

    def test_refresh_dataset_not_found_returns_404(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = _setup_client_and_db(tmp)
            r = client.post(
                "/api/data/nonexistent-id/refresh",
                files={"file": ("data.csv", io.BytesIO(CSV_COMPATIBLE), "text/csv")},
            )
            assert r.status_code == 404

    def test_refresh_invalid_file_type_returns_400(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = _setup_client_and_db(tmp)
            pid = _make_project(client)
            ds_id = _upload(client, CSV_ORIGINAL, pid)

            r = client.post(
                f"/api/data/{ds_id}/refresh",
                files={"file": ("data.txt", io.BytesIO(b"not a csv"), "text/plain")},
            )
            assert r.status_code == 400

    def test_refresh_empty_file_returns_400(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = _setup_client_and_db(tmp)
            pid = _make_project(client)
            ds_id = _upload(client, CSV_ORIGINAL, pid)

            r = _refresh(client, ds_id, CSV_EMPTY)
            assert r.status_code == 400

    def test_refresh_returns_column_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = _setup_client_and_db(tmp)
            pid = _make_project(client)
            ds_id = _upload(client, CSV_ORIGINAL, pid)

            r = _refresh(client, ds_id, CSV_COMPATIBLE)
            body = r.json()

            assert isinstance(body["column_stats"], list)
            assert len(body["column_stats"]) > 0
            col_names = [c["name"] for c in body["column_stats"]]
            assert "revenue" in col_names


# ---------------------------------------------------------------------------
# Chat: _REFRESH_PATTERNS regex
# ---------------------------------------------------------------------------


class TestRefreshChatPatterns:
    @pytest.fixture
    def pattern(self):
        from api.chat import _REFRESH_PATTERNS
        return _REFRESH_PATTERNS

    def test_matches_new_data(self, pattern):
        assert pattern.search("I have new data to upload")

    def test_matches_updated_data(self, pattern):
        assert pattern.search("I have updated data")

    def test_matches_refresh_dataset(self, pattern):
        assert pattern.search("refresh my dataset please")

    def test_matches_replace_data(self, pattern):
        assert pattern.search("I want to replace my data")

    def test_matches_latest_data(self, pattern):
        assert pattern.search("here is the latest data")

    def test_matches_new_csv(self, pattern):
        assert pattern.search("I have a new CSV file")

    def test_matches_new_spreadsheet(self, pattern):
        assert pattern.search("I have a new spreadsheet")

    def test_does_not_match_unrelated(self, pattern):
        assert not pattern.search("what is the accuracy of my model?")

    def test_does_not_match_train(self, pattern):
        assert not pattern.search("train my model")


# ---------------------------------------------------------------------------
# Chat: SSE emits refresh_prompt event
# ---------------------------------------------------------------------------


def _mock_anthropic():
    """Return a context manager that replaces anthropic.Anthropic with a stub."""
    class _FakeStream:
        @property
        def text_stream(self):
            yield "I'll help you update your data."

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class _FakeMessages:
        def stream(self, **kwargs):
            return _FakeStream()

    class _FakeClient:
        messages = _FakeMessages()

    return patch("api.chat.anthropic.Anthropic", return_value=_FakeClient())


class TestRefreshChatSSE:
    def test_refresh_intent_emits_sse_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = _setup_client_and_db(tmp)
            pid = _make_project(client)
            _upload(client, CSV_ORIGINAL, pid)

            with _mock_anthropic():
                r = client.post(
                    f"/api/chat/{pid}",
                    json={"message": "I have new data to upload"},
                )

            assert r.status_code == 200
            events = [
                json.loads(line[6:])
                for line in r.text.splitlines()
                if line.startswith("data: ")
            ]
            types = [e["type"] for e in events]
            assert "refresh_prompt" in types

    def test_refresh_event_contains_dataset_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = _setup_client_and_db(tmp)
            pid = _make_project(client)
            _upload(client, CSV_ORIGINAL, pid)

            with _mock_anthropic():
                r = client.post(
                    f"/api/chat/{pid}",
                    json={"message": "I want to replace my data with updated data"},
                )

            events = {
                json.loads(line[6:])["type"]: json.loads(line[6:])
                for line in r.text.splitlines()
                if line.startswith("data: ")
            }
            assert "refresh_prompt" in events
            refresh = events["refresh_prompt"]["refresh"]
            assert "dataset_id" in refresh
            assert "required_columns" in refresh
            assert "current_filename" in refresh

    def test_no_refresh_event_without_dataset(self):
        """If no dataset uploaded, no refresh_prompt event should be emitted."""
        with tempfile.TemporaryDirectory() as tmp:
            client = _setup_client_and_db(tmp)
            pid = _make_project(client)
            # No dataset uploaded

            with _mock_anthropic():
                r = client.post(
                    f"/api/chat/{pid}",
                    json={"message": "I have new data to upload"},
                )

            events = [
                json.loads(line[6:])
                for line in r.text.splitlines()
                if line.startswith("data: ")
            ]
            types = [e["type"] for e in events]
            assert "refresh_prompt" not in types
