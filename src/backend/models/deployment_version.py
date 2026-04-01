from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class DeploymentVersion(SQLModel, table=True):
    """Snapshot of a deployment at a specific point in time.

    Each time a deployment is updated (new model deployed or rollback executed),
    the previous state is archived here. The version_number is monotonically
    increasing per deployment_id. is_current=True marks the version the
    deployment is actively serving.
    """

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    deployment_id: str = Field(index=True)
    version_number: int  # 1-based, monotonically increasing per deployment
    model_run_id: str
    algorithm: Optional[str] = None
    problem_type: Optional[str] = None
    target_column: Optional[str] = None
    metrics: Optional[str] = None  # JSON metrics dict
    pipeline_path: Optional[str] = None
    deployed_at: datetime = Field(default_factory=_utcnow)
    is_current: bool = True  # True = this version is what the deployment serves now
