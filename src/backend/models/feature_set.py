from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


class FeatureSet(SQLModel, table=True):
    """Records a set of transformations applied to a dataset.

    Each row represents one snapshot of engineered features — the ordered
    list of approved transformations and the resulting column mapping.
    """

    id: Optional[str] = Field(
        default_factory=lambda: str(uuid4()), primary_key=True
    )
    dataset_id: str = Field(foreign_key="dataset.id")
    transformations: Optional[str] = None  # JSON: ordered list of applied transforms
    column_mapping: Optional[str] = None   # JSON: original → [new_cols]
    target_column: Optional[str] = None
    problem_type: Optional[str] = None     # "classification" | "regression"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True
