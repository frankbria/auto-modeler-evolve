"""Integration tests: full backend pipeline — upload → profile → features → train → deploy → predict.

These tests differ from unit tests by exercising real cross-boundary flows:
each step consumes the real output of the previous step, so they catch contract
mismatches that unit tests cannot (e.g. model file path written by trainer must
be readable by deployer).

No browser, no LLM calls (Anthropic is monkeypatched), real SQLite + real sklearn.
"""

from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import db as db_module

# ─────────────────────────────────────────────────────────────────────────────
# Shared sample data — 20 rows so sklearn can do a train/test split
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_CSV = b"""product,region,price,units,revenue
Widget A,North,120.0,10,1200.0
Widget B,South,85.0,8,680.0
Widget A,East,120.0,18,2160.0
Widget C,West,45.0,4,180.0
Widget B,North,85.0,15,1275.0
Widget A,South,120.0,9,1080.0
Widget C,North,45.0,11,495.0
Widget B,East,85.0,16,1360.0
Widget A,West,120.0,20,2400.0
Widget C,South,45.0,6,270.0
Widget A,North,125.0,12,1500.0
Widget B,East,90.0,7,630.0
Widget C,West,50.0,9,450.0
Widget A,South,125.0,14,1750.0
Widget B,North,90.0,11,990.0
Widget C,East,50.0,5,250.0
Widget A,West,130.0,17,2210.0
Widget B,South,88.0,13,1144.0
Widget C,North,48.0,3,144.0
Widget A,East,130.0,22,2860.0
"""


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient wired to a fresh temp SQLite DB and file dirs."""
    test_db = str(tmp_path / "integration.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    # Import all models so their tables are registered with metadata
    import models.project  # noqa
    import models.dataset  # noqa
    import models.feature_set  # noqa
    import models.conversation  # noqa
    import models.model_run  # noqa
    import models.deployment  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.models as models_api_module
    import api.deploy as deploy_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    models_api_module.MODELS_DIR = tmp_path / "models"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"

    # Stub the Anthropic client so no real LLM calls are made during chat
    monkeypatch.setattr(
        "api.chat.anthropic.Anthropic", lambda *a, **kw: _FakeAnthropic()
    )

    from main import app

    with TestClient(app) as c:
        yield c


class _FakeStream:
    """Mimics the Anthropic streaming context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def text_stream(self):
        yield "Here is an analysis of your data."

    def get_final_message(self):
        class _Msg:
            content = [type("B", (), {"text": "Analysis complete."})()]

        return _Msg()


class _FakeAnthropic:
    class messages:
        @staticmethod
        def stream(*args, **kwargs):
            return _FakeStream()


def _wait_for_training(
    client: TestClient, project_id: str, timeout: int = 30
) -> list[dict]:
    """Poll GET /api/models/{project_id}/runs until all runs finish."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/models/{project_id}/runs")
        assert resp.status_code == 200
        runs = resp.json()["runs"]
        if runs and all(r["status"] in ("done", "failed") for r in runs):
            return runs
        time.sleep(0.3)
    raise TimeoutError(f"Training did not complete within {timeout}s")


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline integration test
# ─────────────────────────────────────────────────────────────────────────────


class TestFullPipeline:
    """Exercises the complete backend pipeline as one connected flow."""

    def test_upload_creates_project_and_dataset(self, client):
        """Phase 1: project creation + upload returns a valid dataset."""
        project = client.post("/api/projects", json={"name": "Integration Test"})
        assert project.status_code == 201
        project_id = project.json()["id"]

        upload = client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
        )
        assert upload.status_code == 201, upload.text
        body = upload.json()
        assert body["row_count"] == 20
        assert body["column_count"] == 5
        assert len(body["preview"]) == 10  # API returns first 10 rows as preview
        assert len(body["column_stats"]) == 5

    def test_profile_is_generated_on_upload(self, client):
        """Phase 2: profile endpoint returns distributions and insights."""
        project_id = client.post("/api/projects", json={"name": "Profile Test"}).json()[
            "id"
        ]
        upload = client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
        )
        dataset_id = upload.json()["dataset_id"]

        profile = client.get(f"/api/data/{dataset_id}/profile")
        assert profile.status_code == 200, profile.text
        body = profile.json()
        assert "columns" in body
        assert len(body["columns"]) == 5
        # At least some insights detected
        assert "insights" in body

    def test_feature_suggestions_and_apply(self, client):
        """Phase 3: feature suggestions returned and can be applied."""
        project_id = client.post("/api/projects", json={"name": "Feature Test"}).json()[
            "id"
        ]
        upload = client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
        )
        dataset_id = upload.json()["dataset_id"]

        suggestions = client.get(f"/api/features/{dataset_id}/suggestions")
        assert suggestions.status_code == 200, suggestions.text
        sugg_body = suggestions.json()
        assert "suggestions" in sugg_body

        # Apply transforms (empty list just creates the FeatureSet record)
        apply = client.post(
            f"/api/features/{dataset_id}/apply",
            json={"transformations": []},
        )
        assert apply.status_code == 201, apply.text
        assert "feature_set_id" in apply.json()

    def test_train_and_compare(self, client):
        """Phase 4: training completes and compare returns a recommendation."""
        project_id = client.post("/api/projects", json={"name": "Train Test"}).json()[
            "id"
        ]
        dataset_id = client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
        ).json()["dataset_id"]

        client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
        client.post(
            f"/api/features/{dataset_id}/target", json={"target_column": "revenue"}
        )

        train = client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression"]},
        )
        assert train.status_code == 202, train.text

        runs = _wait_for_training(client, project_id)
        assert len(runs) == 1
        assert runs[0]["status"] == "done"
        assert runs[0]["metrics"] is not None

        compare = client.get(f"/api/models/{project_id}/compare")
        assert compare.status_code == 200
        cmp = compare.json()
        assert len(cmp["models"]) == 1
        assert cmp["recommendation"] is not None

    def test_deploy_and_single_predict(self, client):
        """Phase 6: deployed model accepts JSON input and returns a prediction."""
        project_id = client.post("/api/projects", json={"name": "Deploy Test"}).json()[
            "id"
        ]
        dataset_id = client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
        ).json()["dataset_id"]

        client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
        client.post(
            f"/api/features/{dataset_id}/target", json={"target_column": "revenue"}
        )

        run_id = client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression"]},
        ).json()["model_run_ids"][0]

        runs = _wait_for_training(client, project_id)
        assert any(r["id"] == run_id and r["status"] == "done" for r in runs)

        # Deploy the model
        deploy = client.post(f"/api/deploy/{run_id}")
        assert deploy.status_code == 201, deploy.text
        deployment_id = deploy.json()["id"]

        # Single prediction
        predict = client.post(
            f"/api/predict/{deployment_id}",
            json={
                "product": "Widget A",
                "region": "North",
                "price": 120.0,
                "units": 10,
            },
        )
        assert predict.status_code == 200, predict.text
        pred = predict.json()
        assert "prediction" in pred
        assert isinstance(pred["prediction"], (int, float))

    def test_batch_predict(self, client):
        """Phase 6: batch prediction returns a CSV with a prediction column."""
        project_id = client.post("/api/projects", json={"name": "Batch Test"}).json()[
            "id"
        ]
        dataset_id = client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
        ).json()["dataset_id"]

        client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
        client.post(
            f"/api/features/{dataset_id}/target", json={"target_column": "revenue"}
        )

        run_id = client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression"]},
        ).json()["model_run_ids"][0]
        _wait_for_training(client, project_id)

        client.post(f"/api/deploy/{run_id}")
        dep_id = client.get("/api/deployments").json()[0]["id"]

        batch_csv = b"""product,region,price,units
Widget A,North,120.0,10
Widget B,South,85.0,8
Widget C,East,45.0,4
"""
        resp = client.post(
            f"/api/predict/{dep_id}/batch",
            files={"file": ("batch.csv", io.BytesIO(batch_csv), "text/csv")},
        )
        assert resp.status_code == 200, resp.text
        assert resp.headers["content-type"].startswith("text/csv")
        lines = resp.text.strip().split("\n")
        # Header row + 3 data rows
        assert len(lines) == 4
        # Prediction column present in header
        assert "prediction" in lines[0].lower()

    def test_deploy_then_undeploy(self, client):
        """Deploying and then undeploying a model updates is_active."""
        project_id = client.post(
            "/api/projects", json={"name": "Undeploy Test"}
        ).json()["id"]
        dataset_id = client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
        ).json()["dataset_id"]

        client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
        client.post(
            f"/api/features/{dataset_id}/target", json={"target_column": "revenue"}
        )

        run_id = client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression"]},
        ).json()["model_run_ids"][0]
        _wait_for_training(client, project_id)

        dep_id = client.post(f"/api/deploy/{run_id}").json()["id"]
        assert client.get("/api/deployments").json()[0]["is_active"] is True

        undeploy = client.delete(f"/api/deploy/{dep_id}")
        assert undeploy.status_code in (200, 204)
        remaining = client.get("/api/deployments").json()
        assert all(not d["is_active"] for d in remaining)

    def test_multiple_models_comparison(self, client):
        """Training two models produces a comparison with a recommended winner."""
        project_id = client.post("/api/projects", json={"name": "Multi Test"}).json()[
            "id"
        ]
        dataset_id = client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
        ).json()["dataset_id"]

        client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
        client.post(
            f"/api/features/{dataset_id}/target", json={"target_column": "revenue"}
        )

        client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression", "random_forest_regressor"]},
        )
        runs = _wait_for_training(client, project_id, timeout=60)
        assert len(runs) == 2
        assert all(r["status"] == "done" for r in runs)

        cmp = client.get(f"/api/models/{project_id}/compare").json()
        assert len(cmp["models"]) == 2
        assert cmp["recommendation"] is not None
        # Recommendation points to one of the run IDs
        run_ids = {r["id"] for r in runs}
        assert cmp["recommendation"]["model_run_id"] in run_ids

    def test_upload_narration_appears_in_chat(self, client):
        """Upload triggers an auto-narration message in the conversation."""
        project_id = client.post("/api/projects", json={"name": "Narrate Test"}).json()[
            "id"
        ]
        client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
        )

        history = client.get(f"/api/chat/{project_id}/history")
        assert history.status_code == 200
        messages = history.json()["messages"]
        # The narration module injects an assistant message after upload
        assert any(m["role"] == "assistant" for m in messages)
        # The message should mention the dataset dimensions
        assistant_texts = " ".join(
            m["content"] for m in messages if m["role"] == "assistant"
        )
        assert "20" in assistant_texts or "column" in assistant_texts.lower()

    def test_validation_metrics_after_training(self, client):
        """Phase 5: validation metrics endpoint returns CV results for a trained model."""
        project_id = client.post(
            "/api/projects", json={"name": "Validate Test"}
        ).json()["id"]
        dataset_id = client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
        ).json()["dataset_id"]

        client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
        client.post(
            f"/api/features/{dataset_id}/target", json={"target_column": "revenue"}
        )

        run_id = client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["linear_regression"]},
        ).json()["model_run_ids"][0]
        _wait_for_training(client, project_id)

        val = client.get(f"/api/validate/{run_id}/metrics")
        assert val.status_code == 200, val.text
        body = val.json()
        assert "cross_validation" in body
        assert "error_analysis" in body
        assert "confidence" in body

    def test_feature_importance_after_training(self, client):
        """Phase 5: explainer returns global feature importance for a trained model."""
        project_id = client.post("/api/projects", json={"name": "Explain Test"}).json()[
            "id"
        ]
        dataset_id = client.post(
            "/api/data/upload",
            data={"project_id": project_id},
            files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
        ).json()["dataset_id"]

        client.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
        client.post(
            f"/api/features/{dataset_id}/target", json={"target_column": "revenue"}
        )

        run_id = client.post(
            f"/api/models/{project_id}/train",
            json={"algorithms": ["random_forest_regressor"]},
        ).json()["model_run_ids"][0]
        _wait_for_training(client, project_id, timeout=60)

        explain = client.get(f"/api/validate/{run_id}/explain")
        assert explain.status_code == 200, explain.text
        body = explain.json()
        assert "feature_importance" in body
        assert len(body["feature_importance"]) > 0
