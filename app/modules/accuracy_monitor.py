"""
Module 6 — Accuracy Monitor
Periodically evaluates model quality against recent actuals and triggers
retraining if metrics fall below the configured threshold.

STATUS: Actual metric computation is MOCKED (no real actuals yet).
        The scheduler wiring, DB queries, and retrain trigger are REAL.
"""
from __future__ import annotations

import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.db_models import ForecastingConfig, TrainedModel, ModelStatus

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None


# ---------------------------------------------------------------------------
# Metric computation (mocked)
# ---------------------------------------------------------------------------

def _compute_accuracy_metrics(
    db: Session, model: TrainedModel
) -> dict:
    """
    Compare stored forecasts against actual system metrics.

    TODO: In production, query Prometheus for actual values at the
          timestamps of each ForecastResult, then compute:
              mae  = mean(|actual - predicted|)
              r2   = 1 - SS_res / SS_tot
          and return them.

    MOCK: returns stored training metrics (simulates stable accuracy).
    """
    logger.debug("Computing accuracy for model_id=%d (MOCKED)", model.id)
    return model.metrics or {
        "mae_cpu": 4.2,
        "r2_cpu": 0.67,
        "mae_ram": 0.7,
        "r2_ram": 0.55,
        "mae_net": 8.3,
        "r2_net": 0.48,
        "mape_overall": 7.5,
    }


def _needs_retraining(metrics: dict, threshold: float) -> bool:
    r2_values = [v for k, v in metrics.items() if k.startswith("r2_")]
    if not r2_values:
        return False
    avg_r2 = sum(r2_values) / len(r2_values)
    needs = avg_r2 < threshold
    logger.info("Average R²=%.3f  threshold=%.2f  needs_retraining=%s",
                avg_r2, threshold, needs)
    return needs


# ---------------------------------------------------------------------------
# Evaluation job (called by scheduler)
# ---------------------------------------------------------------------------

def _evaluate_all_models() -> None:
    """Evaluate every READY model and retrain if accuracy dropped."""
    db: Session = SessionLocal()
    try:
        models = db.query(TrainedModel).filter_by(status=ModelStatus.READY).all()
        logger.info("Accuracy check: evaluating %d model(s)", len(models))

        for model in models:
            try:
                metrics = _compute_accuracy_metrics(db, model)
                if _needs_retraining(metrics, settings.accuracy_threshold):
                    logger.warning(
                        "Model %d (config_id=%d) accuracy below threshold — "
                        "triggering retraining.",
                        model.id, model.config_id,
                    )
                    _trigger_retrain(db, model.config_id)
                else:
                    logger.info("Model %d accuracy OK.", model.id)
            except Exception:
                logger.exception("Failed to evaluate model_id=%d", model.id)
    finally:
        db.close()


def _trigger_retrain(db: Session, config_id: int) -> None:
    """
    Initiate background retraining for a config.
    Imports are deferred to avoid circular dependencies.
    """
    from app.modules.data_collector import fetch_historical_data
    from app.modules.correlation_analyzer import analyze
    from app.modules.model_trainer import train_model

    config = db.get(ForecastingConfig, config_id)
    if not config:
        logger.error("Retrain requested for unknown config_id=%d", config_id)
        return

    logger.info("Retraining model for config_id=%d (%s)", config_id, config.name)
    bundle = fetch_historical_data(config.host, config.port, config.business_metric_formula)
    report = analyze(bundle)
    if not report.any_significant:
        logger.warning("No significant correlations found; skipping retrain.")
        return
    train_model(db, config, bundle, report)


# ---------------------------------------------------------------------------
# Scheduler lifecycle
# ---------------------------------------------------------------------------

def start_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        return

    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        _evaluate_all_models,
        trigger="interval",
        hours=settings.retrain_interval_hours,
        id="accuracy_monitor",
        replace_existing=True,
        next_run_time=datetime.utcnow(),   # run once at startup too
    )
    _scheduler.start()
    logger.info(
        "Accuracy monitor scheduler started (interval=%dh)",
        settings.retrain_interval_hours,
    )


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Accuracy monitor scheduler stopped.")
