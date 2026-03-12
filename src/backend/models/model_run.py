from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


class ModelRun(SQLModel, table=True):
    """A single model training run: one algorithm trained on one feature set."""

    id: Optional[str] = Field(
        default_factory=lambda: str(uuid4()), primary_key=True
    )
    project_id: str = Field(foreign_key="project.id")
    dataset_id: str = Field(foreign_key="dataset.id")
    feature_set_id: Optional[str] = Field(default=None, foreign_key="featureset.id")
    algorithm: str                      # e.g. "random_forest"
    display_name: str                   # e.g. "Random Forest"
    target_column: Optional[str] = None
    problem_type: Optional[str] = None  # "classification" | "regression"
    hyperparameters: Optional[str] = None  # JSON
    metrics: Optional[str] = None          # JSON
    training_duration_ms: Optional[int] = None
    model_path: Optional[str] = None    # filesystem path to serialized pipeline
    is_selected: bool = False
    is_deployed: bool = False
    status: str = "done"                # "done" | "failed"
    created_at: datetime = Field(default_factory=datetime.utcnow)
