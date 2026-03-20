"""FeedbackRecord — tracks actual outcomes recorded by users after predictions.

This closes the feedback loop: a user makes a prediction, later observes the
real outcome, and records it here. The system can then compute real-world
accuracy and suggest retraining when error rates climb.
"""

from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class FeedbackRecord(SQLModel, table=True):
    """One user-provided ground-truth label for a past prediction."""

    __tablename__ = "feedbackrecord"  # type: ignore[assignment]

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    deployment_id: str = Field(index=True)
    prediction_log_id: Optional[str] = Field(default=None, index=True)

    # Ground-truth value(s) — at least one should be populated
    actual_value: Optional[float] = None  # For regression: the true numeric outcome
    actual_label: Optional[str] = None  # For classification: the true class label

    # Convenience flag — can be set directly by the user or computed at record time
    is_correct: Optional[bool] = None  # For classification: True if prediction matched

    # Free-text note from the user ("Customer actually churned in Q3")
    comment: Optional[str] = None

    created_at: datetime = Field(default_factory=_utcnow)
