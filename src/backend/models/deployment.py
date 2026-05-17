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
    # API key authentication (optional protection for prediction endpoints)
    api_key_enabled: bool = Field(default=False)
    api_key_hash: Optional[str] = None  # sha256(salt + ":" + key)
    api_key_salt: Optional[str] = None  # random hex salt
    # Rate limiting (optional — per-minute requests and monthly quota)
    rate_limit_rpm: Optional[int] = None  # max requests per minute; None = unlimited
    monthly_quota: Optional[int] = (
        None  # max predictions per rolling 30 days; None = unlimited
    )
    # Quota alert: fire webhooks when usage crosses this % of monthly_quota (None = disabled)
    quota_alert_threshold_pct: Optional[int] = None
    # Accuracy degradation alert
    accuracy_alert_threshold: Optional[float] = (
        None  # 0-1 (classification) or 0-100 (regression); None = disabled
    )
    accuracy_alert_fired: bool = Field(
        default=False
    )  # True after first alert fires; reset when threshold changes
    # Versioning — monotonically incremented each time the deployment is updated
    current_version_number: int = Field(default=1)
    # Environment: "staging" (default) or "production"
    environment: str = Field(default="staging")
    # Confidence threshold: reject/flag predictions below this value (0-1, classification only)
    confidence_threshold: Optional[float] = None
