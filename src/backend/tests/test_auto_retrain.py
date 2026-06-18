"""Tests for auto-retrain feature.

Covers:
- _AUTO_RETRAIN_PATTERNS detection in chat.py
- GET/PUT /api/projects/{project_id}/auto-retrain endpoints
- trigger_auto_retrain() core function
- Upload endpoint response includes auto_retrain field
"""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlmodel import SQLModel

# ---------------------------------------------------------------------------
# Pattern detection
# ---------------------------------------------------------------------------


def test_auto_retrain_pattern_enable():
    from api.chat import _AUTO_RETRAIN_PATTERNS

    assert _AUTO_RETRAIN_PATTERNS.search("enable auto-retrain for this project")
    assert _AUTO_RETRAIN_PATTERNS.search("turn on auto retrain")
    assert _AUTO_RETRAIN_PATTERNS.search("auto retrain when I upload new data")


def test_auto_retrain_pattern_disable():
    from api.chat import _AUTO_RETRAIN_PATTERNS

    assert _AUTO_RETRAIN_PATTERNS.search("disable auto-retrain")
    assert _AUTO_RETRAIN_PATTERNS.search("turn off auto retrain")


def test_auto_retrain_pattern_status():
    from api.chat import _AUTO_RETRAIN_PATTERNS

    assert _AUTO_RETRAIN_PATTERNS.search("what is the auto-retrain status?")
    assert _AUTO_RETRAIN_PATTERNS.search("is auto retrain enabled?")
    assert _AUTO_RETRAIN_PATTERNS.search("retrain automatically when upload")


def test_auto_retrain_pattern_fresh():
    from api.chat import _AUTO_RETRAIN_PATTERNS

    assert _AUTO_RETRAIN_PATTERNS.search("keep model fresh with new data")
    assert _AUTO_RETRAIN_PATTERNS.search("keep model current automatically")


def test_auto_retrain_pattern_no_match():
    from api.chat import _AUTO_RETRAIN_PATTERNS

    assert not _AUTO_RETRAIN_PATTERNS.search("show me a histogram")
    assert not _AUTO_RETRAIN_PATTERNS.search("what is the model accuracy?")
    assert not _AUTO_RETRAIN_PATTERNS.search("deploy the model")


# ---------------------------------------------------------------------------
# API fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def anyio_backend():
    return "asyncio"


async def _make_project(tmp_path, project_id: str):
    import db
    from models.project import Project
    from sqlmodel import create_engine

    test_db = str(tmp_path / f"{project_id}.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)
    with next(db.get_session()) as session:
        proj = Project(
            owner_id="test-default-owner", id=project_id, name="Test Project"
        )
        session.merge(proj)
        session.commit()
    return project_id


# ---------------------------------------------------------------------------
# GET /api/projects/{project_id}/auto-retrain
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_auto_retrain_default_disabled(tmp_path, set_test_env):
    from main import app
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / "ar_get.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    project_id = await _make_project(tmp_path, "ar-get-1")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get(f"/api/projects/{project_id}/auto-retrain")

    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is False
    assert data["project_id"] == project_id


@pytest.mark.anyio
async def test_get_auto_retrain_not_found(tmp_path, set_test_env):
    from main import app
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / "ar_404.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.get("/api/projects/no-such-id/auto-retrain")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# PUT /api/projects/{project_id}/auto-retrain
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_put_auto_retrain_enable(tmp_path, set_test_env):
    from main import app
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / "ar_put.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    project_id = await _make_project(tmp_path, "ar-put-1")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.put(
            f"/api/projects/{project_id}/auto-retrain",
            json={"enabled": True},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is True
    assert "enabled" in data["message"].lower()


@pytest.mark.anyio
async def test_put_auto_retrain_disable(tmp_path, set_test_env):
    from main import app
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / "ar_put2.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    project_id = await _make_project(tmp_path, "ar-put-2")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Enable first
        await client.put(
            f"/api/projects/{project_id}/auto-retrain",
            json={"enabled": True},
        )
        # Then disable
        resp = await client.put(
            f"/api/projects/{project_id}/auto-retrain",
            json={"enabled": False},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["enabled"] is False


@pytest.mark.anyio
async def test_put_auto_retrain_persists(tmp_path, set_test_env):
    """Enable auto-retrain and verify GET returns updated state."""
    from main import app
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / "ar_persist.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    project_id = await _make_project(tmp_path, "ar-persist-1")

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.put(
            f"/api/projects/{project_id}/auto-retrain",
            json={"enabled": True},
        )
        resp = await client.get(f"/api/projects/{project_id}/auto-retrain")

    assert resp.status_code == 200
    assert resp.json()["enabled"] is True


@pytest.mark.anyio
async def test_put_auto_retrain_not_found(tmp_path, set_test_env):
    from main import app
    import db
    from sqlmodel import create_engine

    test_db = str(tmp_path / "ar_put404.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.put(
            "/api/projects/no-such-id/auto-retrain",
            json={"enabled": True},
        )

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# trigger_auto_retrain() — skips gracefully when no selected model
# ---------------------------------------------------------------------------


def test_trigger_auto_retrain_no_selected_model(tmp_path):
    """trigger_auto_retrain returns None when no selected model exists."""
    import db
    from models.project import Project
    from sqlmodel import create_engine, SQLModel

    test_db = str(tmp_path / "ar_core.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)

    with next(db.get_session()) as session:
        proj = Project(owner_id="test-default-owner", id="ar-core-1", name="Core Test")
        session.add(proj)
        session.commit()

    from core.retrain import trigger_auto_retrain

    result = trigger_auto_retrain("ar-core-1", "some-dataset-id")
    assert result is None


def test_trigger_auto_retrain_nonexistent_project(tmp_path):
    """trigger_auto_retrain returns None (does not raise) for missing project."""
    import db
    from sqlmodel import create_engine, SQLModel

    test_db = str(tmp_path / "ar_core2.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)

    from core.retrain import trigger_auto_retrain

    result = trigger_auto_retrain("no-such-project", "no-such-dataset")
    assert result is None


# ---------------------------------------------------------------------------
# Upload endpoint includes auto_retrain field in response
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_upload_response_includes_auto_retrain_field(tmp_path, set_test_env):
    """Upload response always has 'auto_retrain' key (None when not triggered)."""
    import io
    from main import app
    import db
    from models.project import Project
    from sqlmodel import create_engine

    test_db = str(tmp_path / "ar_upload.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)

    with next(db.get_session()) as session:
        proj = Project(
            owner_id="test-default-owner", id="ar-upload-1", name="Upload Test"
        )
        session.add(proj)
        session.commit()

    csv_data = b"col_a,col_b\n1,2\n3,4\n5,6\n"

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post(
            "/api/data/upload",
            data={"project_id": "ar-upload-1"},
            files={"file": ("test.csv", io.BytesIO(csv_data), "text/csv")},
        )

    assert resp.status_code == 201
    data = resp.json()
    assert "auto_retrain" in data


# ---------------------------------------------------------------------------
# Issue #6: auto-retrain must not rebind the live feature_set before training
# succeeds. A skipped/failed retrain must leave prior state untouched.
# ---------------------------------------------------------------------------


def _bootstrap_retrain_project(tmp_path, project_id: str):
    """Create a project with a selected/done ModelRun + a FeatureSet on OLD data.

    Returns (engine, old_dataset_id, old_feature_set_id, old_model_run_id).
    """
    import db
    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun
    from models.project import Project
    from sqlmodel import Session, SQLModel, create_engine

    test_db = str(tmp_path / f"{project_id}.db")
    db.engine = create_engine(f"sqlite:///{test_db}", echo=False)
    SQLModel.metadata.create_all(db.engine)

    # A real CSV backing the OLD dataset (so the feature set is well-formed)
    old_csv = tmp_path / "old.csv"
    old_csv.write_text("f1,f2,target\n1,2,10\n3,4,20\n5,6,30\n")

    with Session(db.engine) as session:
        proj = Project(
            owner_id="test-default-owner", id=project_id, name="Retrain Test"
        )
        session.add(proj)

        old_ds = Dataset(
            project_id=project_id,
            filename="old.csv",
            file_path=str(old_csv),
            row_count=3,
            column_count=3,
        )
        session.add(old_ds)
        session.flush()

        old_fs = FeatureSet(
            dataset_id=old_ds.id,
            target_column="target",
            problem_type="regression",
            transformations="[]",
            column_mapping=None,
            is_active=True,
        )
        session.add(old_fs)
        session.flush()

        old_run = ModelRun(
            project_id=project_id,
            feature_set_id=old_fs.id,
            algorithm="linear_regression",
            hyperparameters="{}",
            metrics="{}",
            status="done",
            is_selected=True,
        )
        session.add(old_run)
        session.commit()
        session.refresh(old_ds)
        session.refresh(old_fs)
        session.refresh(old_run)
        return db.engine, old_ds.id, old_fs.id, old_run.id


def test_retrain_skipped_leaves_feature_set_unchanged(tmp_path):
    """Reproduce #6: missing new-dataset file must NOT rebind the live feature_set.

    The selected/done run's FeatureSet points at OLD. We point retrain at a NEW
    dataset whose backing file is absent, so training is skipped. The live
    feature_set.dataset_id must stay OLD, and no new FeatureSet/ModelRun rows
    may be created.
    """
    from sqlmodel import Session, select

    from models.feature_set import FeatureSet
    from models.model_run import ModelRun

    engine, old_ds_id, old_fs_id, _old_run_id = _bootstrap_retrain_project(
        tmp_path, "r-skipped-1"
    )

    # NEW dataset row exists in DB, but its file is missing on disk
    from models.dataset import Dataset

    with Session(engine) as session:
        new_ds = Dataset(
            project_id="r-skipped-1",
            filename="new.csv",
            file_path=str(tmp_path / "does-not-exist.csv"),
            row_count=3,
            column_count=3,
        )
        session.add(new_ds)
        session.commit()
        session.refresh(new_ds)
        new_ds_id = new_ds.id

    from core.retrain import trigger_auto_retrain

    result = trigger_auto_retrain("r-skipped-1", new_ds_id)

    assert result is None  # skipped because the file is absent

    with Session(engine) as session:
        # Live feature_set untouched
        fs = session.get(FeatureSet, old_fs_id)
        assert fs.dataset_id == old_ds_id  # NOT rebound to the new dataset
        # No new feature sets / model runs were created
        all_fs = list(session.exec(select(FeatureSet)).all())
        all_runs = list(session.exec(select(ModelRun)).all())
        assert len(all_fs) == 1
        assert len(all_runs) == 1  # only the original selected run


def test_retrain_empty_feature_cols_leaves_state_untouched(tmp_path):
    """Skipped retrain (zero feature columns) must leave prior state untouched."""
    from sqlmodel import Session, select

    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun

    engine, old_ds_id, old_fs_id, _old_run_id = _bootstrap_retrain_project(
        tmp_path, "r-empty-1"
    )

    # NEW dataset file has ONLY the target column → zero feature columns
    new_csv = tmp_path / "new_target_only.csv"
    new_csv.write_text("target\n10\n20\n30\n")
    with Session(engine) as session:
        new_ds = Dataset(
            project_id="r-empty-1",
            filename="new_target_only.csv",
            file_path=str(new_csv),
            row_count=3,
            column_count=1,
        )
        session.add(new_ds)
        session.commit()
        session.refresh(new_ds)
        new_ds_id = new_ds.id

    from core.retrain import trigger_auto_retrain

    result = trigger_auto_retrain("r-empty-1", new_ds_id)

    assert result is None  # skipped — no feature columns

    with Session(engine) as session:
        fs = session.get(FeatureSet, old_fs_id)
        assert fs.dataset_id == old_ds_id  # untouched
        all_fs = list(session.exec(select(FeatureSet)).all())
        all_runs = list(session.exec(select(ModelRun)).all())
        assert len(all_fs) == 1
        assert len(all_runs) == 1


def test_retrain_creates_new_feature_set_not_in_place(tmp_path, monkeypatch):
    """Success path: a NEW run-scoped FeatureSet is created; the original is never
    mutated. Verified by stubbing the background trainer so we isolate the
    trigger's DB logic (training correctness is covered elsewhere).
    """
    from sqlmodel import Session, select

    from models.dataset import Dataset
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun

    engine, old_ds_id, old_fs_id, _old_run_id = _bootstrap_retrain_project(
        tmp_path, "r-newfs-1"
    )

    # NEW dataset file with real features → success path
    new_csv = tmp_path / "new.csv"
    new_csv.write_text("f1,f2,target\n7,8,40\n9,10,50\n11,12,60\n")
    with Session(engine) as session:
        new_ds = Dataset(
            project_id="r-newfs-1",
            filename="new.csv",
            file_path=str(new_csv),
            row_count=3,
            column_count=3,
        )
        session.add(new_ds)
        session.commit()
        session.refresh(new_ds)
        new_ds_id = new_ds.id

    # Stub the background trainer to record the launch, and make the worker
    # thread run synchronously so the launch is observable before the trigger
    # returns (deterministic — no scheduler race on the `launched` assertion).
    import api.models as _am
    import core.retrain as _cr

    launched: list[tuple] = []
    monkeypatch.setattr(_am, "_train_in_background", lambda *a, **k: launched.append(a))

    class _SyncThread:
        def __init__(self, target=(), args=(), **_kw):
            self._target = target
            self._args = args

        def start(self):
            self._target(*self._args)

    monkeypatch.setattr(_cr.threading, "Thread", _SyncThread)

    from core.retrain import trigger_auto_retrain

    result = trigger_auto_retrain("r-newfs-1", new_ds_id)

    assert result is not None
    assert result["triggered"] is True
    new_run_id = result["run_id"]

    with Session(engine) as session:
        all_fs = list(session.exec(select(FeatureSet)).all())

        # Original feature set is untouched
        old_fs = session.get(FeatureSet, old_fs_id)
        assert old_fs.dataset_id == old_ds_id
        assert old_fs.target_column == "target"

        # A NEW feature set was created pointing at the NEW dataset
        new_fs = next(f for f in all_fs if f.id != old_fs_id)
        assert new_fs.dataset_id == new_ds_id
        assert new_fs.target_column == "target"
        assert new_fs.problem_type == "regression"
        assert new_fs.is_active is True

        # The new run references the NEW feature set, not the original
        new_run = session.get(ModelRun, new_run_id)
        assert new_run.feature_set_id == new_fs.id
        assert new_run.feature_set_id != old_fs_id

    assert launched, "background trainer was launched"
    assert launched[0][0] == new_run_id  # launched with the new run id


def test_retrain_queue_counter_set_under_lock(tmp_path, monkeypatch):
    """Queue/counter are set up (under _lock) when a retrain is triggered."""
    import api.models as _am

    monkeypatch.setattr(_am, "_train_in_background", lambda *a, **k: None)

    from models.dataset import Dataset
    from sqlmodel import Session

    engine, _old_ds_id, _old_fs_id, _old_run_id = _bootstrap_retrain_project(
        tmp_path, "r-lock-1"
    )

    new_csv = tmp_path / "new_lock.csv"
    new_csv.write_text("f1,f2,target\n7,8,40\n9,10,50\n11,12,60\n")
    with Session(engine) as session:
        new_ds = Dataset(
            project_id="r-lock-1",
            filename="new_lock.csv",
            file_path=str(new_csv),
            row_count=3,
            column_count=3,
        )
        session.add(new_ds)
        session.commit()
        session.refresh(new_ds)
        new_ds_id = new_ds.id

    from core.retrain import trigger_auto_retrain

    trigger_auto_retrain("r-lock-1", new_ds_id)

    # The project now has an SSE queue and a running-thread counter entry
    assert "r-lock-1" in _am._training_queues
    assert _am._training_counters.get("r-lock-1") == 1
