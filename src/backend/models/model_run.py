from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class ModelRun(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    project_id: str = Field(index=True)
    feature_set_id: Optional[str] = None
    algorithm: str
    hyperparameters: Optional[str] = None  # JSON
    metrics: Optional[str] = None  # JSON {r2/accuracy, mae/f1, ...}
    summary: Optional[str] = None  # Plain-English metric summary
    training_duration_ms: Optional[int] = None
    model_path: Optional[str] = None
    is_selected: bool = False
    is_deployed: bool = False
    status: str = Field(default="pending")  # pending | training | done | failed
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)
