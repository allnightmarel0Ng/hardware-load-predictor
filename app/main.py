"""
Application entrypoint.
Creates all DB tables, mounts routers, starts the background scheduler.
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import Base, engine
from app.modules.request_handler import config_router, train_router, forecast_router, accuracy_router, jobs_router, groups_router
from app.modules.accuracy_monitor import start_scheduler, stop_scheduler
from app.modules.job_runner import start_executor, stop_executor

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# Create tables (use Alembic for migrations in production)
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Hardware Load Predictor",
    description=(
        "Predicts CPU, RAM and Network load from business metrics "
        "using machine learning."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(groups_router)
app.include_router(config_router)
app.include_router(train_router)
app.include_router(forecast_router)
app.include_router(accuracy_router)
app.include_router(jobs_router)


@app.on_event("startup")
def on_startup() -> None:
    start_executor()
    start_scheduler()


@app.on_event("shutdown")
def on_shutdown() -> None:
    stop_scheduler()
    stop_executor()


@app.get("/health", tags=["System"])
def health():
    return {"status": "ok"}
