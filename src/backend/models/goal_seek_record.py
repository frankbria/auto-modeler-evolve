"""GoalSeekRecord — stores the last N goal-seek results per deployment.

Each time an analyst runs goal seek ("what inputs would produce $5M revenue?"),
one record is saved. Only the most recent `MAX_HISTORY` per deployment are kept
so analysts can compare alternative scenarios without unbounded storage growth.
"""

from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel

MAX_HISTORY = 3  # keep only the last 3 runs per deployment


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class GoalSeekRecord(SQLModel, table=True):
    """One saved goal-seek result for a deployment."""

    __tablename__ = "goalseekrecord"  # type: ignore[assignment]

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    deployment_id: str = Field(index=True)

    # Target and result
    target_column: str
    problem_type: str  # "regression" | "classification"
    algorithm_plain: str
    target_value_str: str  # display string, e.g. "5000000" or "High"
    achieved_value_str: str  # display string
    achieved: bool
    gap_pct: Optional[float] = None  # regression only

    # Top suggestions (JSON-encoded list of dicts)
    suggestions_json: str = Field(default="[]")

    # Fixed features pinned by the analyst (JSON-encoded dict)
    fixed_features_json: str = Field(default="{}")

    # Plain-English summary from the optimizer
    summary: str

    created_at: datetime = Field(default_factory=_utcnow)
