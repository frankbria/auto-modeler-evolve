"""Prediction alert rule — business-rule-based alerts on prediction values."""

from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class PredictionAlertRule(SQLModel, table=True):
    """A user-defined rule that fires when a live prediction meets a condition.

    condition_type values:
      "prediction_value" — numeric prediction compared with condition_op / condition_value
      "confidence"       — model confidence % compared with condition_op / condition_value
      "predicted_class"  — classification output equals condition_class (string match)

    condition_op values (numeric types): "lt", "gt", "lte", "gte", "eq"
    """

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    deployment_id: str = Field(index=True)
    name: str
    condition_type: str  # "prediction_value" | "confidence" | "predicted_class"
    condition_op: str = Field(default="lt")  # "lt" | "gt" | "lte" | "gte" | "eq"
    condition_value: Optional[float] = None
    condition_class: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=_utcnow)
    last_triggered_at: Optional[datetime] = None
    trigger_count: int = 0
