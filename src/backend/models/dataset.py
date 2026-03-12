from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


class Dataset(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    project_id: str = Field(index=True)
    filename: str
    file_path: str
    row_count: int = 0
    column_count: int = 0
    columns: Optional[str] = None   # JSON list of column stat dicts
    profile: Optional[str] = None   # JSON full profiling results
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)
    size_bytes: int = 0
