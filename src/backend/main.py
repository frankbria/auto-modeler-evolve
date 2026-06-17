import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import models  # noqa: F401 — ensures all SQLModel tables are registered before create_all

from core.path_safety import UnsafePathError
from core.ssrf_guard import UnsafeURLError

from api.chat import router as chat_router
from api.data import router as data_router
from api.deploy import router as deploy_router
from api.features import router as features_router
from api.models import router as models_router
from api.projects import router as projects_router
from api.templates import router as templates_router
from api.validation import router as validation_router
from api.analysis_templates import router as analysis_templates_router
from api.auth import router as auth_router
from auth.security import using_insecure_default_secret
from core.scheduler import start_scheduler
from db import create_db_and_tables

DATA_DIR = Path(__file__).parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    create_db_and_tables()

    # Reclaim disk from any artifacts orphaned by crashes / out-of-band deletes.
    # Runs in a daemon thread so a large first-boot backlog (the audit found
    # thousands of leaked dirs) can't block the app from serving. run_janitor
    # never raises, and opens its own short-lived session.
    def _sweep() -> None:
        try:
            from sqlmodel import Session

            from core.janitor import run_janitor
            from db import engine as _engine

            with Session(_engine) as _session:
                run_janitor(_session)
        except Exception:  # noqa: BLE001 - background best-effort
            logging.getLogger(__name__).exception("Startup janitor sweep failed")

    threading.Thread(target=_sweep, name="artifact-janitor", daemon=True).start()
    if using_insecure_default_secret():
        logging.getLogger(__name__).warning(
            "AUTH_SECRET is not set — using an insecure development secret. "
            "Set AUTH_SECRET before any non-local deployment."
        )
    start_scheduler()
    yield


app = FastAPI(
    title="AutoModeler",
    description="AI-powered conversational data modeling platform",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(projects_router)
app.include_router(data_router)
app.include_router(chat_router)
app.include_router(features_router)
app.include_router(models_router)
app.include_router(validation_router)
app.include_router(deploy_router)
app.include_router(templates_router)
app.include_router(analysis_templates_router)


@app.exception_handler(UnsafeURLError)
async def _unsafe_url_handler(request: Request, exc: UnsafeURLError) -> JSONResponse:
    """A rejected outbound URL (SSRF guard) is a client error, not a 500."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(UnsafePathError)
async def _unsafe_path_handler(request: Request, exc: UnsafePathError) -> JSONResponse:
    """A rejected filename / path (traversal guard) is a client error, not a 500."""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}
