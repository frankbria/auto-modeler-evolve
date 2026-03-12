from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


class Conversation(SQLModel, table=True):
    id: Optional[str] = Field(
        default_factory=lambda: str(uuid4()), primary_key=True
    )
    project_id: str = Field(foreign_key="project.id")
    messages: str = "[]"  # JSON string list of {role, content, timestamp}
    state: str = "upload"  # upload, explore, shape, model, validate, deploy
    updated_at: datetime = Field(default_factory=datetime.utcnow)
