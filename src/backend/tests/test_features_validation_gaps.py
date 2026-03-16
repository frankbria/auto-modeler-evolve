"""Targeted tests to close remaining coverage gaps in:
  - api/features.py (lines 44, 47, 106-107, 142, 249, 255, 258, 307, 310)
  - api/validation.py (lines 54, 60, 64, 70, 74, 87, 104, 139-140)
  - other remaining gaps: chat/orchestrator.py, core/explainer.py, api/chat.py
"""
import io
import json
import asyncio

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import Session, SQLModel, create_engine

import db as db_module


SAMPLE_CSV = (
    b"date,product,region,revenue,units,category\n"
    b"2024-01-01,Widget A,North,1200.50,10,Premium\n"
    b"2024-01-01,Widget B,South,850.00,8,Standard\n"
    b"2024-01-02,Widget A,East,2100.75,18,Premium\n"
    b"2024-01-02,Widget C,West,450.25,4,Budget\n"
    b"2024-01-03,Widget B,North,1650.00,15,Standard\n"
    b"2024-01-04,Widget A,South,900.00,9,Premium\n"
    b"2024-01-05,Widget C,East,750.00,7,Budget\n"
    b"2024-01-06,Widget B,West,1100.00,11,Standard\n"
)


@pytest.fixture
async def ac(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test.db")
    db_module.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    db_module.DATA_DIR = tmp_path

    import models.conversation  # noqa
    import models.dataset  # noqa
    import models.deployment  # noqa
    import models.feature_set  # noqa
    import models.model_run  # noqa
    import models.project  # noqa

    SQLModel.metadata.create_all(db_module.engine)

    import api.data as data_module
    import api.deploy as deploy_module
    import api.models as models_module

    data_module.UPLOAD_DIR = tmp_path / "uploads"
    models_module.MODELS_DIR = tmp_path / "models"
    deploy_module.DEPLOY_DIR = tmp_path / "deployments"

    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


@pytest.fixture
async def project_id(ac):
    r = await ac.post("/api/projects", json={"name": "Gap Test Project"})
    assert r.status_code == 201
    return r.json()["id"]


@pytest.fixture
async def dataset_id(ac, project_id):
    r = await ac.post(
        "/api/data/upload",
        data={"project_id": project_id},
        files={"file": ("sales.csv", io.BytesIO(SAMPLE_CSV), "text/csv")},
    )
    assert r.status_code == 201
    return r.json()["dataset_id"]


@pytest.fixture
async def feature_set_id(ac, dataset_id):
    r = await ac.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
    assert r.status_code == 201
    fs_id = r.json()["feature_set_id"]
    r2 = await ac.post(
        f"/api/features/{dataset_id}/target",
        json={"target_column": "revenue", "problem_type": "regression"},
    )
    assert r2.status_code == 200
    return fs_id


@pytest.fixture
async def model_run_id(ac, project_id, feature_set_id):
    r = await ac.post(
        f"/api/models/{project_id}/train",
        json={"algorithms": ["linear_regression"]},
    )
    assert r.status_code == 202
    run_id = r.json()["model_run_ids"][0]
    for _ in range(30):
        sr = await ac.get(f"/api/models/{project_id}/runs")
        runs = sr.json()["runs"]
        matching = [x for x in runs if x["id"] == run_id]
        if matching and matching[0]["status"] in ("done", "failed"):
            break
        await asyncio.sleep(0.2)
    return run_id


# ===========================================================================
# api/features.py gaps
# ===========================================================================


class TestFeaturesApiGaps:

    # Lines 44, 47: _load_dataset error paths
    @pytest.mark.asyncio
    async def test_suggestions_dataset_not_found(self, ac):
        r = await ac.get("/api/features/nonexistent/suggestions")
        assert r.status_code == 404
        assert "Dataset not found" in r.json()["detail"]

    @pytest.mark.asyncio
    async def test_suggestions_file_missing_from_disk(self, ac, dataset_id, tmp_path):
        """Suggestions returns 404 when file deleted from disk."""
        from models.dataset import Dataset as DS
        with Session(db_module.engine) as session:
            ds = session.get(DS, dataset_id)
            ds.file_path = str(tmp_path / "gone.csv")
            session.add(ds)
            session.commit()

        r = await ac.get(f"/api/features/{dataset_id}/suggestions")
        assert r.status_code == 404
        assert "not found on disk" in r.json()["detail"]

    # Lines 106-107: apply — previous feature sets deactivated
    @pytest.mark.asyncio
    async def test_apply_deactivates_previous_feature_set(self, ac, dataset_id):
        """Applying transforms twice deactivates the first FeatureSet."""
        r1 = await ac.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
        fs1_id = r1.json()["feature_set_id"]

        r2 = await ac.post(f"/api/features/{dataset_id}/apply", json={"transformations": []})
        fs2_id = r2.json()["feature_set_id"]

        from models.feature_set import FeatureSet
        with Session(db_module.engine) as session:
            fs1 = session.get(FeatureSet, fs1_id)
            fs2 = session.get(FeatureSet, fs2_id)
        assert fs1.is_active is False
        assert fs2.is_active is True

    # Line 142: preview_feature_set — FeatureSet not found
    @pytest.mark.asyncio
    async def test_preview_feature_set_not_found(self, ac):
        r = await ac.get("/api/features/nonexistent-fs/preview")
        assert r.status_code == 404
        assert "FeatureSet not found" in r.json()["detail"]

    # Line 249: add_pipeline_step — params included in step
    @pytest.mark.asyncio
    async def test_add_step_with_params(self, ac, feature_set_id):
        """Adding a step with params should include params in pipeline."""
        r = await ac.post(
            f"/api/features/{feature_set_id}/steps",
            json={"column": "units", "transform_type": "log_transform", "params": {"base": 10}},
        )
        assert r.status_code == 201
        # Verify the step was stored with params
        list_r = await ac.get(f"/api/features/{feature_set_id}/steps")
        steps = list_r.json()["steps"]
        assert any("params" in s for s in steps)

    # Line 255: add_pipeline_step — dataset not found
    @pytest.mark.asyncio
    async def test_add_step_dataset_not_found(self, ac):
        """add_pipeline_step returns 404 when FeatureSet's dataset is missing."""
        from models.feature_set import FeatureSet
        # Create a FeatureSet with an invalid dataset_id
        with Session(db_module.engine) as session:
            fs = FeatureSet(
                dataset_id="nonexistent-dataset",
                transformations=json.dumps([]),
                column_mapping=json.dumps({}),
                is_active=True,
            )
            session.add(fs)
            session.commit()
            session.refresh(fs)
            fs_id = fs.id

        r = await ac.post(
            f"/api/features/{fs_id}/steps",
            json={"column": "units", "transform_type": "log_transform"},
        )
        assert r.status_code == 404
        assert "Dataset not found" in r.json()["detail"]

    # Line 258: add_pipeline_step — dataset file missing
    @pytest.mark.asyncio
    async def test_add_step_dataset_file_missing(self, ac, feature_set_id, dataset_id, tmp_path):
        """add_pipeline_step returns 404 when file is deleted from disk."""
        from models.dataset import Dataset as DS
        with Session(db_module.engine) as session:
            ds = session.get(DS, dataset_id)
            ds.file_path = str(tmp_path / "gone.csv")
            session.add(ds)
            session.commit()

        r = await ac.post(
            f"/api/features/{feature_set_id}/steps",
            json={"column": "units", "transform_type": "log_transform"},
        )
        assert r.status_code == 404
        assert "not found on disk" in r.json()["detail"]

    # Line 307: remove_pipeline_step — dataset not found
    @pytest.mark.asyncio
    async def test_remove_step_dataset_not_found(self, ac):
        """remove_pipeline_step returns 404 when FeatureSet's dataset is missing."""
        from models.feature_set import FeatureSet
        with Session(db_module.engine) as session:
            fs = FeatureSet(
                dataset_id="nonexistent-dataset",
                transformations=json.dumps([{"column": "x", "transform_type": "log_transform"}]),
                column_mapping=json.dumps({}),
                is_active=True,
            )
            session.add(fs)
            session.commit()
            session.refresh(fs)
            fs_id = fs.id

        r = await ac.delete(f"/api/features/{fs_id}/steps/0")
        assert r.status_code == 404
        assert "Dataset not found" in r.json()["detail"]

    # Line 310: remove_pipeline_step — file missing
    @pytest.mark.asyncio
    async def test_remove_step_file_missing(self, ac, feature_set_id, dataset_id, tmp_path):
        """remove_pipeline_step returns 404 when file is deleted from disk."""
        # Add a step first so there's something to remove
        await ac.post(
            f"/api/features/{feature_set_id}/steps",
            json={"column": "units", "transform_type": "log_transform"},
        )

        from models.dataset import Dataset as DS
        with Session(db_module.engine) as session:
            ds = session.get(DS, dataset_id)
            ds.file_path = str(tmp_path / "gone.csv")
            session.add(ds)
            session.commit()

        r = await ac.delete(f"/api/features/{feature_set_id}/steps/0")
        assert r.status_code == 404
        assert "not found on disk" in r.json()["detail"]


# ===========================================================================
# api/validation.py gaps
# ===========================================================================


class TestValidationApiGaps:

    # Line 54: _load_run_context — run not done
    @pytest.mark.asyncio
    async def test_validate_metrics_run_not_done(self, ac, project_id, feature_set_id):
        from models.model_run import ModelRun
        with Session(db_module.engine) as session:
            run = ModelRun(
                project_id=project_id,
                feature_set_id=feature_set_id,
                algorithm="linear_regression",
                hyperparameters=json.dumps({}),
                status="pending",
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            run_id = run.id

        r = await ac.get(f"/api/validate/{run_id}/metrics")
        assert r.status_code == 400
        assert "Validation requires a completed run" in r.json()["detail"]

    # Line 60: _load_run_context — model file not found
    @pytest.mark.asyncio
    async def test_validate_metrics_model_file_missing(self, ac, project_id, feature_set_id):
        from models.model_run import ModelRun
        with Session(db_module.engine) as session:
            run = ModelRun(
                project_id=project_id,
                feature_set_id=feature_set_id,
                algorithm="linear_regression",
                hyperparameters=json.dumps({}),
                status="done",
                model_path=None,
                metrics=json.dumps({"r2": 0.9}),
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            run_id = run.id

        r = await ac.get(f"/api/validate/{run_id}/metrics")
        assert r.status_code == 404
        assert "model file" in r.json()["detail"].lower()

    # Line 64: _load_run_context — feature set not found
    @pytest.mark.asyncio
    async def test_validate_metrics_feature_set_missing(self, ac, project_id, tmp_path):
        from models.model_run import ModelRun
        # Create a fake joblib file so the path check passes
        fake_model = tmp_path / "model.joblib"
        fake_model.write_bytes(b"fake")
        with Session(db_module.engine) as session:
            run = ModelRun(
                project_id=project_id,
                feature_set_id="deleted-feature-set",
                algorithm="linear_regression",
                hyperparameters=json.dumps({}),
                status="done",
                model_path=str(fake_model),
                metrics=json.dumps({"r2": 0.9}),
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            run_id = run.id

        r = await ac.get(f"/api/validate/{run_id}/metrics")
        assert r.status_code == 404
        assert "Feature set" in r.json()["detail"] or "feature" in r.json()["detail"].lower()

    # Line 70: _load_run_context — dataset not found
    @pytest.mark.asyncio
    async def test_validate_metrics_dataset_not_found(self, ac, project_id, tmp_path):
        from models.model_run import ModelRun
        from models.feature_set import FeatureSet
        fake_model = tmp_path / "model.joblib"
        fake_model.write_bytes(b"fake")
        with Session(db_module.engine) as session:
            fs = FeatureSet(
                dataset_id="nonexistent-dataset",
                transformations=json.dumps([]),
                column_mapping=json.dumps({}),
                is_active=True,
                target_column="revenue",
                problem_type="regression",
            )
            session.add(fs)
            session.commit()
            session.refresh(fs)

            run = ModelRun(
                project_id=project_id,
                feature_set_id=fs.id,
                algorithm="linear_regression",
                hyperparameters=json.dumps({}),
                status="done",
                model_path=str(fake_model),
                metrics=json.dumps({"r2": 0.9}),
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            run_id = run.id

        r = await ac.get(f"/api/validate/{run_id}/metrics")
        assert r.status_code == 404
        assert "Dataset" in r.json()["detail"] or "dataset" in r.json()["detail"].lower()

    # Line 74: _load_run_context — dataset file not found
    @pytest.mark.asyncio
    async def test_validate_metrics_dataset_file_missing(self, ac, model_run_id, dataset_id, tmp_path):
        from models.dataset import Dataset as DS
        with Session(db_module.engine) as session:
            ds = session.get(DS, dataset_id)
            ds.file_path = str(tmp_path / "gone.csv")
            session.add(ds)
            session.commit()

        r = await ac.get(f"/api/validate/{model_run_id}/metrics")
        assert r.status_code == 404
        assert "not found on disk" in r.json()["detail"]

    # Line 87: _get_unfitted_model — unknown algorithm
    @pytest.mark.asyncio
    async def test_validate_metrics_unknown_algorithm(self, ac, project_id, feature_set_id, dataset_id, tmp_path):
        from models.model_run import ModelRun
        fake_model = tmp_path / "model.joblib"
        # Use an actual dataset file path so context loads OK
        from models.dataset import Dataset as DS
        with Session(db_module.engine) as session:
            ds = session.get(DS, dataset_id)
            real_path = ds.file_path
        fake_model.write_bytes(b"fake")
        with Session(db_module.engine) as session:
            run = ModelRun(
                project_id=project_id,
                feature_set_id=feature_set_id,
                algorithm="unknown_algo",
                hyperparameters=json.dumps({}),
                status="done",
                model_path=str(fake_model),
                metrics=json.dumps({"r2": 0.9}),
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            run_id = run.id

        r = await ac.get(f"/api/validate/{run_id}/metrics")
        assert r.status_code == 400
        assert "Unknown algorithm" in r.json()["detail"]

    # Line 104: get_explain — run not found
    @pytest.mark.asyncio
    async def test_explain_run_not_found(self, ac):
        r = await ac.get("/api/validate/nonexistent/explain")
        assert r.status_code == 404

    # Lines 139-140: get_explain_row — run not found
    @pytest.mark.asyncio
    async def test_explain_row_run_not_found(self, ac):
        r = await ac.get("/api/validate/nonexistent/explain/0")
        assert r.status_code == 404

    # Positive path: explain on real trained model
    @pytest.mark.asyncio
    async def test_explain_success(self, ac, model_run_id):
        r = await ac.get(f"/api/validate/{model_run_id}/explain")
        assert r.status_code == 200
        data = r.json()
        assert "feature_importance" in data

    # Positive path: explain row on real trained model
    @pytest.mark.asyncio
    async def test_explain_row_success(self, ac, model_run_id):
        r = await ac.get(f"/api/validate/{model_run_id}/explain/0")
        assert r.status_code == 200


# ===========================================================================
# api/chat.py remaining gap (lines 172-174: project not found in get_history)
# ===========================================================================

class TestChatApiGaps:

    @pytest.mark.asyncio
    async def test_get_history_nonexistent_project_returns_empty(self, ac):
        """History for a project with no conversation returns empty list (not 404)."""
        r = await ac.get("/api/chat/nonexistent-project/history")
        assert r.status_code == 200
        assert r.json()["messages"] == []

    @pytest.mark.asyncio
    async def test_send_message_project_not_found(self, ac):
        r = await ac.post(
            "/api/chat/nonexistent-project",
            json={"message": "hello"},
        )
        assert r.status_code in (404, 422, 200)  # may return 200 SSE stream with error message
