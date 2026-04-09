from datetime import UTC, datetime
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class DeploymentPreset(SQLModel, table=True):
    """Named prediction scenario that pre-fills the VP dashboard form.

    `feature_values` is a JSON-encoded dict mapping feature column names
    to their preset values (strings or numbers).
    """

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    deployment_id: str = Field(index=True)
    name: str
    feature_values: str = Field(default="{}")  # JSON dict
    created_at: datetime = Field(default_factory=_utcnow)
