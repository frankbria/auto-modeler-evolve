from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import models  # noqa: F401 — ensures all SQLModel tables are registered before create_all

from api.chat import router as chat_router
from api.data import router as data_router
from api.deploy import deploy_router, deployments_router, predict_router
from api.features import router as features_router
from api.models import router as models_router
from api.projects import router as projects_router
from api.validation import router as validation_router
from db import create_db_and_tables

DATA_DIR = Path(__file__).parent / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
MODELS_DIR = DATA_DIR / "models"


@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    create_db_and_tables()
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

app.include_router(projects_router)
app.include_router(data_router)
app.include_router(chat_router)
app.include_router(features_router)
app.include_router(models_router)
app.include_router(validation_router)
app.include_router(deploy_router)
app.include_router(predict_router)
app.include_router(deployments_router)


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}
