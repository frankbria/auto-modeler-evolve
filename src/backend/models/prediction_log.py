from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class PredictionLog(SQLModel, table=True):
    """Records a single prediction request for analytics and monitoring."""

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    deployment_id: str = Field(index=True)
    input_features: str  # JSON: dict of feature_name → value
    prediction: str  # JSON: the raw prediction result (value or class label)
    prediction_numeric: Optional[float] = None  # parsed numeric value for aggregation
    confidence: Optional[float] = None  # probability / confidence score if available
    created_at: datetime = Field(default_factory=_utcnow)
