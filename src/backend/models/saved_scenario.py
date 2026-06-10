from datetime import UTC, datetime
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class SavedScenario(SQLModel, table=True):
    """Named prediction scenario saved by an analyst for comparison.

    `inputs` is a JSON-encoded dict of feature overrides used to produce
    the prediction.  `prediction_numeric` stores the numeric value for
    regression comparisons; for classification, `confidence` holds the
    winning-class probability.
    """

    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    deployment_id: str = Field(index=True)
    name: str
    inputs: str = Field(default="{}")  # JSON dict of feature overrides
    prediction_value: str = ""  # decoded prediction label/value string
    prediction_numeric: float | None = Field(default=None)
    confidence: float | None = Field(default=None)
    created_at: datetime = Field(default_factory=_utcnow)
