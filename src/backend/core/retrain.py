"""Auto-retrain trigger utility.

Called from data.py after a successful upload when project.auto_retrain is True.
Imports api.models internals lazily to avoid circular imports at module load.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "models" / "trained"


def trigger_auto_retrain(project_id: str, new_dataset_id: str) -> dict | None:
    """Trigger background retraining using the selected model's algorithm.

    Returns a summary dict if a retrain was started, or None if skipped
    (no selected model, no feature set, etc.).  Never raises — caller is
    upload endpoint where we don't want to break the response.
    """
    try:
        return _do_trigger(project_id, new_dataset_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("Auto-retrain trigger failed for project %s: %s", project_id, exc)
        return None


def _do_trigger(project_id: str, new_dataset_id: str) -> dict | None:
    import db
    from models.feature_set import FeatureSet
    from models.model_run import ModelRun
    from sqlmodel import Session, select

    with Session(db.engine) as session:
        # Find the selected model run
        selected_run = session.exec(
            select(ModelRun).where(
                ModelRun.project_id == project_id,
                ModelRun.is_selected == True,  # noqa: E712
                ModelRun.status == "done",
            )
        ).first()
        if not selected_run:
            logger.info(
                "Auto-retrain: no selected model for project %s — skipping", project_id
            )
            return None

        # Get the feature set
        feature_set = session.get(FeatureSet, selected_run.feature_set_id)
        if not feature_set or not feature_set.target_column:
            logger.info(
                "Auto-retrain: no usable feature set for project %s — skipping",
                project_id,
            )
            return None

        algorithm = selected_run.algorithm
        target_col = feature_set.target_column
        problem_type = feature_set.problem_type or "regression"
        transformations = feature_set.transformations

        # Update the feature set to point to the new dataset
        feature_set.dataset_id = new_dataset_id
        session.add(feature_set)

        # Create a new model run
        new_run = ModelRun(
            project_id=project_id,
            feature_set_id=feature_set.id,
            algorithm=algorithm,
            hyperparameters=json.dumps({}),
            status="pending",
        )
        session.add(new_run)
        session.commit()
        session.refresh(new_run)
        new_run_id = new_run.id

        # Get the new dataset's file path
        from models.dataset import Dataset

        dataset = session.get(Dataset, new_dataset_id)
        if not dataset or not dataset.file_path:
            logger.info(
                "Auto-retrain: dataset file not found for %s — skipping", new_dataset_id
            )
            return None
        file_path = Path(dataset.file_path)
        if not file_path.exists():
            return None

    # Load dataframe and launch background training
    import pandas as pd

    from core.feature_engine import apply_transformations

    df = pd.read_csv(file_path)
    transforms = json.loads(transformations or "[]")
    if transforms:
        df, _ = apply_transformations(df, transforms)

    feature_cols = [c for c in df.columns if c != target_col]
    if not feature_cols:
        return None

    model_dir = MODELS_DIR / project_id

    # Import training helpers lazily to avoid circular imports
    from api.models import _training_counters, _training_queues
    import queue

    _training_queues[project_id] = queue.Queue()
    _training_counters[project_id] = 1

    from api.models import _train_in_background

    t = threading.Thread(
        target=_train_in_background,
        args=(
            new_run_id,
            project_id,
            df,
            feature_cols,
            target_col,
            algorithm,
            problem_type,
            model_dir,
        ),
        daemon=True,
        name=f"AutoRetrain-{project_id}",
    )
    t.start()

    logger.info(
        "Auto-retrain started for project %s: run_id=%s algorithm=%s",
        project_id,
        new_run_id,
        algorithm,
    )
    return {
        "run_id": new_run_id,
        "algorithm": algorithm,
        "triggered": True,
    }
