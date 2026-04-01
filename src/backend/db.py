from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

DATABASE_URL = f"sqlite:///{DATA_DIR}/automodeler.db"
engine = create_engine(DATABASE_URL, echo=False)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)
    # Inline migrations: add columns that may be missing in existing DBs
    _apply_migrations()


def _apply_migrations():
    """Add any columns missing from pre-existing tables."""
    migrations = [
        ("deployment", "api_key_enabled", "INTEGER NOT NULL DEFAULT 0"),
        ("deployment", "api_key_hash", "TEXT"),
        ("deployment", "api_key_salt", "TEXT"),
        ("deployment", "current_version_number", "INTEGER NOT NULL DEFAULT 1"),
        ("predictionlog", "response_ms", "REAL"),
    ]
    with engine.connect() as conn:
        for table, col, definition in migrations:
            try:
                conn.execute(
                    __import__("sqlalchemy").text(
                        f"ALTER TABLE {table} ADD COLUMN {col} {definition}"
                    )
                )
                conn.commit()
            except Exception:
                pass  # Column already exists


def get_session():
    with Session(engine) as session:
        yield session
