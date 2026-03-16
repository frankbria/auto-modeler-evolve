from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class FeatureSet(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    dataset_id: str = Field(index=True)
    transformations: Optional[str] = None      # JSON list of transformation dicts
    column_mapping: Optional[str] = None        # JSON dict: source → [new_cols]
    target_column: Optional[str] = None
    problem_type: Optional[str] = None          # "classification" | "regression"
    is_active: bool = True
    created_at: datetime = Field(default_factory=_utcnow)
