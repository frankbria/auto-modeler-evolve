from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


class Project(SQLModel, table=True):
    id: Optional[str] = Field(
        default_factory=lambda: str(uuid4()), primary_key=True
    )
    name: str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "exploring"  # exploring, modeling, deployed
    settings: Optional[str] = None  # JSON string
