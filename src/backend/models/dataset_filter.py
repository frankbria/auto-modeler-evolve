from datetime import UTC, datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class DatasetFilter(SQLModel, table=True):
    """Stores the active non-destructive filter for a dataset.

    One record per dataset (unique on dataset_id). The filter narrows the
    analytical lens without modifying the underlying CSV. All analysis
    operations in the chat handler apply this filter before running.
    """

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    dataset_id: str = Field(index=True, unique=True)
    conditions: str  # JSON list of {column, operator, value} dicts
    filter_summary: (
        str  # Plain-English description e.g. "region = North AND revenue > 1000"
    )
    original_rows: int
    filtered_rows: int
    created_at: datetime = Field(default_factory=_utcnow)

    @property
    def row_reduction_pct(self) -> float:
        if self.original_rows == 0:
            return 0.0
        return round((1 - self.filtered_rows / self.original_rows) * 100, 1)
