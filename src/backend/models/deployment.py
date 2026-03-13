from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


class Deployment(SQLModel, table=True):
    """A deployed model: active prediction endpoint backed by a trained ModelRun."""

    id: Optional[str] = Field(
        default_factory=lambda: str(uuid4()), primary_key=True
    )
    model_run_id: str = Field(foreign_key="modelrun.id")
    project_id: str = Field(foreign_key="project.id")
    is_active: bool = True
    request_count: int = 0
    feature_schema: Optional[str] = None  # JSON: [{name, dtype, sample_values}, ...]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_predicted_at: Optional[datetime] = None
