"""Input validation rule — business-rule-based constraints on prediction inputs."""

from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class InputValidationRule(SQLModel, table=True):
    """A user-defined constraint applied to incoming prediction inputs before the model runs.

    rule_type values:
      "range"   — feature value must be between min_val and max_val (inclusive)
      "one_of"  — feature value must be one of the strings in allowed_values (JSON list)
      "not_null" — feature value must be present and non-null
    """

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    deployment_id: str = Field(index=True)
    feature_name: str
    rule_type: str  # "range" | "one_of" | "not_null"
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    allowed_values: Optional[str] = None  # JSON-encoded list of strings
    created_at: datetime = Field(default_factory=_utcnow)
