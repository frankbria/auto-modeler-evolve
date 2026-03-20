from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Deployment(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    model_run_id: str = Field(index=True)
    project_id: str = Field(index=True)
    endpoint_path: str  # /api/predict/{id}
    dashboard_url: str  # /predict/{id}
    pipeline_path: Optional[str] = None  # serialized PredictionPipeline
    is_active: bool = True
    request_count: int = 0
    # Metadata cached at deploy-time for fast serving
    algorithm: Optional[str] = None
    problem_type: Optional[str] = None
    feature_names: Optional[str] = None  # JSON list
    target_column: Optional[str] = None
    metrics: Optional[str] = None  # JSON metrics dict
    created_at: datetime = Field(default_factory=_utcnow)
    last_predicted_at: Optional[datetime] = None
