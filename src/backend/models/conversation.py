from datetime import UTC, datetime
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Conversation(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    project_id: str = Field(index=True)
    messages: str = Field(default="[]")  # JSON list of {role, content, timestamp}
    state: str = Field(default="upload")  # upload | explore | shape | model | validate | deploy
    updated_at: datetime = Field(default_factory=_utcnow)
