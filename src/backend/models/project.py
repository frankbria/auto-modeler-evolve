from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Project(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    owner_id: str = Field(foreign_key="user.id", index=True)
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    status: str = Field(default="exploring")
    settings: Optional[str] = None  # JSON string for preferences
    auto_retrain: bool = Field(default=False)
    last_milestone_state: Optional[str] = Field(
        default=None
    )  # None | "upload" | "train" | "deploy"
    last_insight_dataset_id: Optional[str] = Field(
        default=None
    )  # dataset.id of the most-recently auto-analyzed dataset
    last_type_check_dataset_id: Optional[str] = Field(
        default=None
    )  # dataset.id of the most-recently type-checked dataset
    last_ensemble_suggest_run_count: Optional[int] = Field(
        default=None
    )  # number of done non-ensemble runs when auto-suggest last fired
    last_low_accuracy_guidance_run_count: Optional[int] = Field(
        default=None
    )  # total done runs when low-accuracy guidance last fired
