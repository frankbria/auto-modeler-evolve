from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class ABTest(SQLModel, table=True):
    """Champion-challenger A/B test splitting live prediction traffic.

    The champion is the currently deployed model.  The challenger is another
    active deployment (typically a freshly retrained model).  Incoming
    prediction requests are routed to the challenger with probability
    (1 - champion_split_pct / 100) so analysts can compare performance on
    real traffic before promoting the challenger.
    """

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    champion_id: str = Field(index=True)  # Deployment.id — receives champion traffic
    challenger_id: str = Field(index=True)  # Deployment.id — receives challenger traffic
    champion_split_pct: int = Field(default=80)  # 1–99; % routed to champion
    is_active: bool = Field(default=True)
    auto_promote: bool = Field(default=False)  # future: auto-promote on significance
    created_at: datetime = Field(default_factory=_utcnow)
    ended_at: Optional[datetime] = None
    winner: Optional[str] = None  # "champion" | "challenger" | None
