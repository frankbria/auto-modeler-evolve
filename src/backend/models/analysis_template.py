from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class AnalysisTemplate(SQLModel, table=True):
    """Named set of chat queries the analyst can replay on any dataset.

    `queries` is a JSON-encoded list[str] of user messages captured from
    the conversation history at the time the template was saved.
    """

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    project_id: str = Field(index=True)
    name: str
    queries: str = Field(default="[]")  # JSON list of query strings
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)
