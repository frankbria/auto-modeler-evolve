from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class DashboardFieldConfig(SQLModel, table=True):
    """Per-field visibility/lock configuration for the public prediction dashboard."""

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    deployment_id: str = Field(index=True)
    feature_name: str
    is_visible: bool = True
    is_locked: bool = False
    locked_value: Optional[str] = None  # JSON-encoded value when locked
    display_label: Optional[str] = None  # human-friendly override label
    display_order: Optional[int] = None  # ascending; None = natural schema order
    created_at: datetime = Field(default_factory=_utcnow)
