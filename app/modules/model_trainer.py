"""
Module 4 — Model Trainer
Trains regression models on historical data and persists them to disk +
the database.

STATUS: ML MOCKED — the "model" is a simple linear coefficient computed
        from mock correlation data.  The persistence, versioning, and
        database bookkeeping are REAL and production-ready.

        To plug in real ML:
        1. Replace `_fit_mock_model()` with scikit-learn training code.
        2. Serialize with joblib.dump() to artifact_path.
        3. Store actual hyperparameters / metrics in the DB.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.db_models import ForecastingConfig, ModelStatus, TrainedModel
from app.modules.correlation_analyzer import CorrelationReport
from app.modules.data_collector import MetricsBundle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _next_version(db: Session, config_id: int) -> int:
    latest = (
        db.query(TrainedModel)
        .filter_by(config_id=config_id)
        .order_by(TrainedModel.version.desc())
        .first()
    )
    return (latest.version + 1) if latest else 1


def _artifact_path(config_id: int, version: int) -> str:
    directory = Path(settings.model_storage_path) / str(config_id)
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / f"model_v{version}.json")


def _fit_mock_model(
    bundle: MetricsBundle,
    report: CorrelationReport,
) -> dict:
    """
    MOCK training — derives linear scaling coefficients from the last
    data point ratio.  Replace with joblib.dump(sklearn_model, path).

    Returns a dict of model parameters to be stored in the DB.
    """
    # Use the median business value as a normalisation baseline
    biz_values = [p["value"] for p in bundle.business]
    baseline = sorted(biz_values)[len(biz_values) // 2] or 1.0

    cpu_values = [p["value"] for p in bundle.cpu]
    ram_values = [p["value"] for p in bundle.ram]
    net_values = [p["value"] for p in bundle.network]

    cpu_median  = sorted(cpu_values)[len(cpu_values) // 2]
    ram_median  = sorted(ram_values)[len(ram_values) // 2]
    net_median  = sorted(net_values)[len(net_values) // 2]

    return {
        "type": "mock_linear",
        "baseline_business": baseline,
        "cpu_coeff":  cpu_median / baseline,
        "ram_coeff":  ram_median / baseline,
        "net_coeff":  net_median / baseline,
        "cpu_lag_min": report.cpu.lag_minutes,
        "ram_lag_min": report.ram.lag_minutes,
        "net_lag_min": report.network.lag_minutes,
    }


def _mock_metrics(report: CorrelationReport) -> dict:
    """Return plausible evaluation metrics (MOCKED)."""
    return {
        "mae_cpu":  4.2,
        "rmse_cpu": 6.1,
        "r2_cpu":   report.cpu.pearson_r ** 2,
        "mae_ram":  0.7,
        "rmse_ram": 1.1,
        "r2_ram":   report.ram.pearson_r ** 2,
        "mae_net":  8.3,
        "rmse_net": 12.4,
        "r2_net":   report.network.pearson_r ** 2,
        "mape_overall": 7.5,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_model(
    db: Session,
    config: ForecastingConfig,
    bundle: MetricsBundle,
    report: CorrelationReport,
) -> TrainedModel:
    """
    Create a TrainedModel record, fit the model, persist the artifact,
    and update the record with metrics + status.
    """
    version = _next_version(db, config.id)
    artifact = _artifact_path(config.id, version)

    record = TrainedModel(
        config_id=config.id,
        version=version,
        algorithm="mock_linear",   # change to e.g. "gradient_boosting" later
        status=ModelStatus.TRAINING,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    try:
        logger.info(
            "Training model v%d for config_id=%d (MOCKED)", version, config.id
        )
        params = _fit_mock_model(bundle, report)
        metrics = _mock_metrics(report)

        # Persist mock model as JSON (replace with joblib.dump for real ML)
        with open(artifact, "w") as fh:
            json.dump(params, fh)

        record.parameters = params
        record.metrics = metrics
        record.artifact_path = artifact
        record.lag_minutes = report.best_lag()
        record.status = ModelStatus.READY
        record.trained_at = datetime.utcnow()

    except Exception as exc:
        logger.exception("Training failed for config_id=%d", config.id)
        record.status = ModelStatus.FAILED
        db.commit()
        raise

    db.commit()
    db.refresh(record)
    logger.info("Model v%d ready: metrics=%s", version, metrics)
    return record


def get_latest_ready_model(db: Session, config_id: int) -> TrainedModel | None:
    """Return the most recent READY model for a given config, or None."""
    return (
        db.query(TrainedModel)
        .filter_by(config_id=config_id, status=ModelStatus.READY)
        .order_by(TrainedModel.version.desc())
        .first()
    )
