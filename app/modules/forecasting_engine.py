"""
Module 5 — Forecasting Engine
Loads a trained model and produces CPU / RAM / Network predictions for a
given business metric value.

STATUS: Uses mock_linear models stored as JSON.
        When real sklearn models are persisted with joblib, replace
        `_load_model_params()` and `_predict()` accordingly.
"""
from __future__ import annotations

import json
import logging

from sqlalchemy.orm import Session

from app.models.db_models import ForecastingConfig, ForecastResult, TrainedModel
from app.modules.model_trainer import get_latest_ready_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_model_params(model: TrainedModel) -> dict:
    """
    Load model parameters from disk.

    Real ML version:
        import joblib
        return joblib.load(model.artifact_path)
    """
    if not model.artifact_path:
        raise ValueError(f"Model {model.id} has no artifact_path set.")
    with open(model.artifact_path) as fh:
        return json.load(fh)


def _predict(params: dict, business_value: float) -> tuple[float, float, float]:
    """
    Run inference.

    Real ML version (e.g. multi-output regressor):
        X = [[business_value]]
        preds = model.predict(X)[0]
        return float(preds[0]), float(preds[1]), float(preds[2])

    Mock version: simple linear scaling.
    """
    baseline = params["baseline_business"] or 1.0
    ratio = business_value / baseline

    cpu = round(min(100.0, max(0.0, params["cpu_coeff"] * business_value)), 2)
    ram = round(max(0.0, params["ram_coeff"] * business_value), 2)
    net = round(max(0.0, params["net_coeff"] * business_value), 2)
    return cpu, ram, net


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forecast(
    db: Session,
    config: ForecastingConfig,
    business_metric_value: float,
) -> ForecastResult:
    """
    Produce and persist a forecast for the given business metric value.

    Raises:
        ValueError: if no ready model exists for this config.
    """
    model = get_latest_ready_model(db, config.id)
    if model is None:
        raise ValueError(
            f"No ready model found for config '{config.name}' (id={config.id}). "
            "Please train a model first."
        )

    logger.info(
        "Running forecast for config_id=%d model_id=%d biz_value=%.2f",
        config.id, model.id, business_metric_value,
    )

    params = _load_model_params(model)
    cpu, ram, net = _predict(params, business_metric_value)

    result = ForecastResult(
        config_id=config.id,
        model_id=model.id,
        business_metric_value=business_metric_value,
        predicted_cpu_percent=cpu,
        predicted_ram_gb=ram,
        predicted_network_mbps=net,
    )
    db.add(result)
    db.commit()
    db.refresh(result)

    logger.info(
        "Forecast result: cpu=%.1f%% ram=%.2fGB net=%.1fMbps",
        cpu, ram, net,
    )
    return result
