from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import Index
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class PredictionLog(SQLModel, table=True):
    """Records a single prediction request for analytics and monitoring."""

    # Composite index serves both deployment_id lookups (left-prefix) and the
    # near-universal "deployment_id + created_at time window/sort" analytics
    # queries (issue #19). It replaces the old standalone deployment_id index.
    __table_args__ = (
        Index("ix_predictionlog_dep_created", "deployment_id", "created_at"),
    )

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    deployment_id: str
    input_features: str  # JSON: dict of feature_name → value
    prediction: str  # JSON: the raw prediction result (value or class label)
    prediction_numeric: Optional[float] = None  # parsed numeric value for aggregation
    confidence: Optional[float] = None  # probability / confidence score if available
    response_ms: Optional[float] = None  # prediction latency in milliseconds
    ab_variant: Optional[str] = (
        None  # "champion" | "challenger" when serving an A/B test
    )
    served_by_canary: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_utcnow)
