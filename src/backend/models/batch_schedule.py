"""Batch prediction schedule and job run models."""

from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class BatchSchedule(SQLModel, table=True):
    """Recurring batch prediction schedule for a deployment."""

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    deployment_id: str = Field(index=True)
    # Frequency: "daily" | "weekly" | "monthly"
    frequency: str = Field(default="daily")
    # Time of day to run (UTC)
    run_hour: int = Field(default=9)
    run_minute: int = Field(default=0)
    # For weekly: 0=Monday … 6=Sunday
    day_of_week: Optional[int] = None
    # For monthly: 1-28
    day_of_month: Optional[int] = None
    is_active: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    last_output_path: Optional[str] = None
    last_row_count: Optional[int] = None
    last_error: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)


class BatchJobRun(SQLModel, table=True):
    """Record of a single batch schedule execution."""

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    schedule_id: str = Field(index=True)
    deployment_id: str = Field(index=True)
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = None
    # "running" | "success" | "failed"
    status: str = Field(default="running")
    output_path: Optional[str] = None
    row_count: Optional[int] = None
    error: Optional[str] = None
