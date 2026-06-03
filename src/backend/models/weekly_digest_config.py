"""WeeklyDigestConfig — per-deployment weekly monitoring digest schedule."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class WeeklyDigestConfig(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    deployment_id: str = Field(index=True)
    enabled: bool = Field(default=True)
    day_of_week: int = Field(default=0)  # 0=Monday … 6=Sunday
    send_hour: int = Field(default=9)  # UTC hour 0-23
    last_sent_at: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
