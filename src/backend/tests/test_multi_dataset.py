"""Tests for multi-dataset support.

Covers:
- core/merger.py: suggest_join_keys, merge_datasets
- GET  /api/data/project/{project_id}/datasets
- POST /api/data/join-keys
- POST /api/data/{project_id}/merge
"""

from __future__ import annotations

import io
import pathlib

import pandas as pd
import pytest

from core.merger import merge_datasets, suggest_join_keys

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _df(data: dict) -> pd.DataFrame:
    return pd.DataFrame(data)


SALES_CSV = pathlib.Path(__file__).parent.parent / "data/sample/sample_sales.csv"


def _make_csv_bytes(data: dict) -> bytes:
    return _df(data).to_csv(index=False).encode()


async def _create_project(client) -> str:
    r = await client.post("/api/projects", json={"name": "Multi Dataset Test"})
    assert r.status_code == 201
    return r.json()["id"]


async def _upload_csv(client, project_id: str, filename: str, csv_bytes: bytes) -> str:
    r = await client.post(
        "/api/data/upload",
        files={"file": (filename, io.BytesIO(csv_bytes), "text/csv")},
        data={"project_id": project_id},
    )
    assert r.status_code == 201
    return r.json()["dataset_id"]


# ---------------------------------------------------------------------------
# Unit tests: core/merger.py — suggest_join_keys
# ---------------------------------------------------------------------------


class TestSuggestJoinKeys:
    def test_common_columns_returned(self):
        df1 = _df({"id": [1, 2, 3], "value": [10, 20, 30]})
        df2 = _df({"id": [1, 2, 4], "score": [0.1, 0.2, 0.4]})
        suggestions = suggest_join_keys(df1, df2)
        assert len(suggestions) == 1
        assert suggestions[0]["name"] == "id"

    def test_no_common_columns(self):
        df1 = _df({"a": [1, 2], "b": [3, 4]})
        df2 = _df({"c": [1, 2], "d": [3, 4]})
        assert suggest_join_keys(df1, df2) == []

    def test_multiple_common_columns_sorted_by_uniqueness(self):
        # 'id' has perfect uniqueness; 'category' has low uniqueness
        df1 = _df({"id": [1, 2, 3, 4], "category": ["A", "A", "B", "B"]})
        df2 = _df({"id": [1, 2, 3, 4], "category": ["A", "B", "A", "B"]})
        suggestions = suggest_join_keys(df1, df2)
        assert suggestions[0]["name"] == "id"  # higher uniqueness first

    def test_recommended_flag_for_high_uniqueness(self):
        df1 = _df({"id": [1, 2, 3, 4, 5]})
        df2 = _df({"id": [1, 2, 3, 4, 5]})
        suggestions = suggest_join_keys(df1, df2)
        assert suggestions[0]["recommended"] is True

    def test_not_recommended_for_low_uniqueness(self):
        # All same value → uniqueness = 0.2
        df1 = _df({"cat": ["A"] * 5})
        df2 = _df({"cat": ["A"] * 5})
        suggestions = suggest_join_keys(df1, df2)
        assert suggestions[0]["recommended"] is False

    def test_returns_dtype_info(self):
        df1 = _df({"id": [1, 2, 3]})
        df2 = _df({"id": [1, 2, 3]})
        s = suggest_join_keys(df1, df2)[0]
        assert "dtype_left" in s
        assert "dtype_right" in s
        assert "unique_left" in s
        assert "unique_right" in s


# ---------------------------------------------------------------------------
# Unit tests: core/merger.py — merge_datasets
# ---------------------------------------------------------------------------


class TestMergeDatasets:
    def setup_method(self):
        self.left = _df(
            {"id": [1, 2, 3], "name": ["A", "B", "C"], "revenue": [100, 200, 300]}
        )
        self.right = _df(
            {"id": [1, 2, 4], "score": [0.9, 0.8, 0.7], "revenue": [110, 210, 410]}
        )

    def test_inner_join_row_count(self):
        result = merge_datasets(self.left, self.right, join_key="id", how="inner")
        assert result["row_count"] == 2  # ids 1 and 2 match

    def test_left_join_row_count(self):
        result = merge_datasets(self.left, self.right, join_key="id", how="left")
        assert result["row_count"] == 3  # all left rows kept

    def test_right_join_row_count(self):
        result = merge_datasets(self.left, self.right, join_key="id", how="right")
        assert result["row_count"] == 3  # all right rows kept

    def test_outer_join_row_count(self):
        result = merge_datasets(self.left, self.right, join_key="id", how="outer")
        assert result["row_count"] == 4  # union of ids 1,2,3,4

    def test_conflict_columns_identified(self):
        result = merge_datasets(self.left, self.right, join_key="id", how="inner")
        assert "revenue" in result["conflict_columns"]

    def test_suffix_applied_to_conflict_columns(self):
        result = merge_datasets(self.left, self.right, join_key="id", how="inner")
        assert "revenue_left" in result["columns"]
        assert "revenue_right" in result["columns"]
        assert "revenue" not in result["columns"]

    def test_custom_suffix(self):
        result = merge_datasets(
            self.left,
            self.right,
            join_key="id",
            how="inner",
            suffix_left="_A",
            suffix_right="_B",
        )
        assert "revenue_A" in result["columns"]
        assert "revenue_B" in result["columns"]

    def test_preview_rows_returned(self):
        result = merge_datasets(self.left, self.right, join_key="id", how="inner")
        assert isinstance(result["preview_rows"], list)
        assert len(result["preview_rows"]) <= 10

    def test_invalid_join_key_raises(self):
        with pytest.raises(ValueError, match="not found in left dataset"):
            merge_datasets(self.left, self.right, join_key="nonexistent")

    def test_invalid_join_key_right_raises(self):
        df_no_id = self.right.drop(columns=["id"])
        with pytest.raises(ValueError, match="not found in right dataset"):
            merge_datasets(self.left, df_no_id, join_key="id")

    def test_invalid_how_raises(self):
        with pytest.raises(ValueError, match="how must be one of"):
            merge_datasets(self.left, self.right, join_key="id", how="cross")


# ---------------------------------------------------------------------------
# API integration tests
# ---------------------------------------------------------------------------


class TestListProjectDatasets:
    async def test_empty_project_returns_empty_list(self, client):
        pid = await _create_project(client)
        r = await client.get(f"/api/data/project/{pid}/datasets")
        assert r.status_code == 200
        assert r.json() == []

    async def test_lists_uploaded_datasets(self, client):
        pid = await _create_project(client)
        csv1 = _make_csv_bytes({"id": [1, 2], "value": [10, 20]})
        csv2 = _make_csv_bytes({"id": [3, 4], "val2": [30, 40]})
        await _upload_csv(client, pid, "first.csv", csv1)
        await _upload_csv(client, pid, "second.csv", csv2)

        r = await client.get(f"/api/data/project/{pid}/datasets")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 2
        filenames = {ds["filename"] for ds in body}
        assert filenames == {"first.csv", "second.csv"}

    async def test_404_for_missing_project(self, client):
        r = await client.get("/api/data/project/nonexistent/datasets")
        assert r.status_code == 404

    async def test_response_shape(self, client):
        pid = await _create_project(client)
        csv1 = _make_csv_bytes({"id": [1, 2, 3], "score": [0.1, 0.2, 0.3]})
        await _upload_csv(client, pid, "data.csv", csv1)

        r = await client.get(f"/api/data/project/{pid}/datasets")
        body = r.json()
        ds = body[0]
        assert "dataset_id" in ds
        assert "filename" in ds
        assert "row_count" in ds
        assert "column_count" in ds
        assert "uploaded_at" in ds
        assert "size_bytes" in ds


class TestJoinKeySuggestions:
    async def test_returns_common_columns(self, client):
        pid = await _create_project(client)
        csv1 = _make_csv_bytes({"id": [1, 2, 3], "revenue": [100, 200, 300]})
        csv2 = _make_csv_bytes({"id": [1, 2, 4], "score": [0.9, 0.8, 0.7]})
        did1 = await _upload_csv(client, pid, "a.csv", csv1)
        did2 = await _upload_csv(client, pid, "b.csv", csv2)

        r = await client.post(
            "/api/data/join-keys",
            json={"dataset_id_1": did1, "dataset_id_2": did2},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["common_column_count"] == 1
        assert body["join_key_suggestions"][0]["name"] == "id"

    async def test_404_for_missing_dataset(self, client):
        r = await client.post(
            "/api/data/join-keys",
            json={"dataset_id_1": "nonexistent", "dataset_id_2": "also-bad"},
        )
        assert r.status_code == 404


class TestMergeEndpoint:
    async def _setup(self, client):
        pid = await _create_project(client)
        csv1 = _make_csv_bytes(
            {"id": [1, 2, 3], "name": ["A", "B", "C"], "revenue": [100, 200, 300]}
        )
        csv2 = _make_csv_bytes({"id": [1, 2, 4], "region": ["North", "South", "East"]})
        did1 = await _upload_csv(client, pid, "left.csv", csv1)
        did2 = await _upload_csv(client, pid, "right.csv", csv2)
        return pid, did1, did2

    async def test_merge_creates_new_dataset(self, client):
        pid, did1, did2 = await self._setup(client)
        r = await client.post(
            f"/api/data/{pid}/merge",
            json={
                "dataset_id_1": did1,
                "dataset_id_2": did2,
                "join_key": "id",
                "how": "inner",
            },
        )
        assert r.status_code == 201
        body = r.json()
        assert "dataset_id" in body
        assert body["row_count"] == 2  # ids 1 and 2 match

    async def test_merge_result_listed_in_project_datasets(self, client):
        pid, did1, did2 = await self._setup(client)
        await client.post(
            f"/api/data/{pid}/merge",
            json={
                "dataset_id_1": did1,
                "dataset_id_2": did2,
                "join_key": "id",
                "how": "inner",
            },
        )
        r = await client.get(f"/api/data/project/{pid}/datasets")
        assert len(r.json()) == 3  # original 2 + merged 1

    async def test_merge_returns_preview(self, client):
        pid, did1, did2 = await self._setup(client)
        r = await client.post(
            f"/api/data/{pid}/merge",
            json={
                "dataset_id_1": did1,
                "dataset_id_2": did2,
                "join_key": "id",
                "how": "inner",
            },
        )
        body = r.json()
        assert "preview" in body
        assert len(body["preview"]) == 2

    async def test_merge_400_for_bad_join_key(self, client):
        pid, did1, did2 = await self._setup(client)
        r = await client.post(
            f"/api/data/{pid}/merge",
            json={
                "dataset_id_1": did1,
                "dataset_id_2": did2,
                "join_key": "nonexistent",
                "how": "inner",
            },
        )
        assert r.status_code == 400

    async def test_merge_400_for_invalid_how(self, client):
        pid, did1, did2 = await self._setup(client)
        r = await client.post(
            f"/api/data/{pid}/merge",
            json={
                "dataset_id_1": did1,
                "dataset_id_2": did2,
                "join_key": "id",
                "how": "cross",
            },
        )
        assert r.status_code == 400

    async def test_merge_404_for_missing_project(self, client):
        r = await client.post(
            "/api/data/nonexistent/merge",
            json={
                "dataset_id_1": "a",
                "dataset_id_2": "b",
                "join_key": "id",
                "how": "inner",
            },
        )
        assert r.status_code == 404

    async def test_merge_custom_filename(self, client):
        pid, did1, did2 = await self._setup(client)
        r = await client.post(
            f"/api/data/{pid}/merge",
            json={
                "dataset_id_1": did1,
                "dataset_id_2": did2,
                "join_key": "id",
                "how": "inner",
                "save_as_filename": "custom_name.csv",
            },
        )
        assert r.status_code == 201
        assert r.json()["filename"] == "custom_name.csv"

    async def test_merge_left_join_keeps_all_left_rows(self, client):
        pid, did1, did2 = await self._setup(client)
        r = await client.post(
            f"/api/data/{pid}/merge",
            json={
                "dataset_id_1": did1,
                "dataset_id_2": did2,
                "join_key": "id",
                "how": "left",
            },
        )
        assert r.status_code == 201
        assert r.json()["row_count"] == 3  # all 3 left rows
