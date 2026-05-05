from __future__ import annotations

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from app.core.database import SessionLocal
from app.models.db_models import ForecastingConfig, JobStatus, TrainingJob
from app.modules.data_collector import fetch_historical_data
from app.modules.correlation_analyzer import analyze
from app.modules.model_trainer import train_model

logger = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None
MAX_WORKERS = 4


def start_executor() -> None:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="trainer")
        logger.info("Training job executor started (max_workers=%d)", MAX_WORKERS)


def stop_executor() -> None:
    global _executor
    if _executor:
        _executor.shutdown(wait=False, cancel_futures=False)
        logger.info("Training job executor stopped.")
        _executor = None


def _run_training_job(job_id: int) -> None:
    db = SessionLocal()
    try:
        job = db.get(TrainingJob, job_id)
        if not job:
            logger.error("Job %d not found in DB", job_id)
            return

        job.status     = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        db.commit()
        logger.info("Job %d started (config_id=%d lookback=%dd)",
                    job_id, job.config_id, job.lookback_days)

        config = db.get(ForecastingConfig, job.config_id)
        if not config:
            raise RuntimeError(f"Config {job.config_id} not found.")

        bundle = fetch_historical_data(
            host=config.host,
            port=config.port,
            business_formula=config.business_metric_formula,
            lookback_days=job.lookback_days,
            instance_label=getattr(config, "instance_label", None),
        )
        report = analyze(bundle)
        if report.is_business_constant:
            raise ValueError(
                "Business metric series has near-zero variance — appears constant. "
                "Check the PromQL formula or increase the lookback window."
            )
        model = train_model(db, config, bundle, report)

        job.status      = JobStatus.DONE
        job.model_id    = model.id
        job.finished_at = datetime.utcnow()
        db.commit()
        logger.info(
            "Job %d done — model_id=%d  algo=%s  R²_cpu=%.3f  duration=%.1fs",
            job_id, model.id, model.algorithm,
            model.metrics.get("r2_cpu", 0) if model.metrics else 0,
            job.duration_seconds or 0,
        )

    except Exception:
        err = traceback.format_exc()
        logger.exception("Job %d failed", job_id)
        try:
            job = db.get(TrainingJob, job_id)
            if job:
                job.status        = JobStatus.FAILED
                job.finished_at   = datetime.utcnow()
                job.error_message = err[-2000:]   # cap to avoid giant DB rows
                db.commit()
        except Exception:
            logger.exception("Failed to write FAILED status for job %d", job_id)
    finally:
        db.close()


def submit_training_job(config_id: int, lookback_days: int) -> TrainingJob:
    db = SessionLocal()
    try:
        job = TrainingJob(
            config_id=config_id,
            lookback_days=lookback_days,
            status=JobStatus.QUEUED,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id
        logger.info("Training job %d queued for config_id=%d", job_id, config_id)
    finally:
        db.close()

    if _executor is None:
        raise RuntimeError(
            "Job executor is not running. "
            "Ensure start_executor() was called at app startup."
        )
    _executor.submit(_run_training_job, job_id)
    return job


def get_job(job_id: int) -> TrainingJob | None:
    db = SessionLocal()
    try:
        return db.get(TrainingJob, job_id)
    finally:
        db.close()


def list_jobs(config_id: int, limit: int = 20) -> list[TrainingJob]:
    db = SessionLocal()
    try:
        return (
            db.query(TrainingJob)
            .filter_by(config_id=config_id)
            .order_by(TrainingJob.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        db.close()
