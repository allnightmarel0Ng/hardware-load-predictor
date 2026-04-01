"""
Module 5 — Forecasting Engine
Produces CPU / RAM / Network predictions from a trained model artifact.

Two capabilities added in Tier 1:

  1. Prediction intervals (80% coverage, 10th–90th percentile)
     Each forecast response includes:
       predicted_*   — point estimate (median GBR)
       lower_*       — 10th-percentile bound
       upper_*       — 90th-percentile bound

  2. Forecast horizon (multi-step)
     The engineer supplies a list of (business_value, minutes_ahead) pairs
     and receives a prediction per step.

Artifact schema (written by model_trainer._fit_and_evaluate):
    {
      "model":       MultiOutputRegressor  (point estimator)
      "model_lower": MultiOutputRegressor  (10th percentile) | None
      "model_upper": MultiOutputRegressor  (90th percentile) | None
      "scaler":      StandardScaler
    }
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import NamedTuple

import joblib
import numpy as np
from sqlalchemy.orm import Session

from app.models.db_models import (
    ForecastHorizonResult,
    ForecastingConfig,
    ForecastResult,
    TrainedModel,
)
from app.modules.model_trainer import get_latest_ready_model

logger = logging.getLogger(__name__)


class TargetPrediction(NamedTuple):
    """Point estimate + 80% prediction interval for one target."""
    point: float
    lower: float | None
    upper: float | None


@dataclass
class InferencePrediction:
    """Full prediction for all three targets at one timestep."""
    cpu:     TargetPrediction
    ram:     TargetPrediction
    network: TargetPrediction


def _load_artifact(model: TrainedModel) -> dict:
    if not model.artifact_path:
        raise ValueError(f"Model {model.id} has no artifact_path set.")
    artifact = joblib.load(model.artifact_path)
    artifact.setdefault("model_lower", None)
    artifact.setdefault("model_upper", None)
    return artifact


def _build_inference_feature(
    business_value: float,
    at_time: datetime | None = None,
) -> np.ndarray:
    t    = at_time or datetime.utcnow()
    hour = t.hour + t.minute / 60.0
    dow  = t.weekday()
    return np.array([[
        business_value,
        business_value,
        0.0,
        business_value,
        0.0,
        np.sin(2 * np.pi * hour / 24.0),
        np.cos(2 * np.pi * hour / 24.0),
        np.sin(2 * np.pi * dow  / 7.0),
        np.cos(2 * np.pi * dow  / 7.0),
    ]], dtype=float)


def _run_inference(
    artifact: dict,
    business_value: float,
    at_time: datetime | None = None,
) -> InferencePrediction:
    """
    Run point + quantile inference for all three targets.
    CPU clamped [0,100]; RAM/network to [0,inf).
    Intervals enforced: lower <= point <= upper always holds.
    """
    scaler      = artifact["scaler"]
    model_point = artifact["model"]
    model_lower = artifact.get("model_lower")
    model_upper = artifact.get("model_upper")

    X        = _build_inference_feature(business_value, at_time)
    X_scaled = scaler.transform(X)

    pt      = model_point.predict(X_scaled)[0]
    cpu_pt  = round(float(np.clip(pt[0], 0.0, 100.0)), 2)
    ram_pt  = round(float(max(0.0, pt[1])), 2)
    net_pt  = round(float(max(0.0, pt[2])), 2)

    if model_lower is not None and model_upper is not None:
        lo = model_lower.predict(X_scaled)[0]
        hi = model_upper.predict(X_scaled)[0]
        cpu_lo = round(float(np.clip(lo[0], 0.0, cpu_pt)), 2)
        cpu_hi = round(float(np.clip(hi[0], cpu_pt, 100.0)), 2)
        ram_lo = round(float(max(0.0, min(lo[1], ram_pt))), 2)
        ram_hi = round(float(max(ram_pt, hi[1])), 2)
        net_lo = round(float(max(0.0, min(lo[2], net_pt))), 2)
        net_hi = round(float(max(net_pt, hi[2])), 2)
    else:
        cpu_lo = cpu_hi = ram_lo = ram_hi = net_lo = net_hi = None

    return InferencePrediction(
        cpu=TargetPrediction(point=cpu_pt, lower=cpu_lo, upper=cpu_hi),
        ram=TargetPrediction(point=ram_pt, lower=ram_lo, upper=ram_hi),
        network=TargetPrediction(point=net_pt, lower=net_lo, upper=net_hi),
    )


def forecast(
    db: Session,
    config: ForecastingConfig,
    business_metric_value: float,
) -> ForecastResult:
    """
    Produce and persist a single-step forecast.
    Includes prediction intervals when the model supports them.
    """
    model = get_latest_ready_model(db, config.id)
    if model is None:
        raise ValueError(
            f"No ready model for config '{config.name}' (id={config.id}). "
            "Train first via POST /configs/{id}/train/."
        )

    logger.info(
        "Forecast: config_id=%d model_id=%d (v%d) biz=%.2f",
        config.id, model.id, model.version, business_metric_value,
    )

    artifact = _load_artifact(model)
    pred     = _run_inference(artifact, business_metric_value)

    result = ForecastResult(
        config_id=config.id,
        model_id=model.id,
        business_metric_value=business_metric_value,
        predicted_cpu_percent=pred.cpu.point,
        predicted_ram_gb=pred.ram.point,
        predicted_network_mbps=pred.network.point,
        lower_cpu_percent=pred.cpu.lower,
        lower_ram_gb=pred.ram.lower,
        lower_network_mbps=pred.network.lower,
        upper_cpu_percent=pred.cpu.upper,
        upper_ram_gb=pred.ram.upper,
        upper_network_mbps=pred.network.upper,
    )
    db.add(result)
    db.commit()
    db.refresh(result)

    has_iv = pred.cpu.lower is not None
    logger.info(
        "Result: cpu=%.1f%%%s  ram=%.2f GB  net=%.1f Mbps",
        pred.cpu.point,
        f"(±{round((pred.cpu.upper - pred.cpu.lower) / 2, 1)})" if has_iv else "",
        pred.ram.point, pred.network.point,
    )
    return result


def forecast_horizon(
    db: Session,
    config: ForecastingConfig,
    steps: list[dict],
) -> list[ForecastHorizonResult]:
    """
    Produce and persist a multi-step forecast for a schedule of business values.

    Args:
        steps: list of {"business_metric_value": float, "minutes_ahead": int}

    Returns:
        List of ForecastHorizonResult rows ordered by step index.

    Raises:
        ValueError: if no ready model or steps is empty.
    """
    if not steps:
        raise ValueError("steps list must not be empty.")

    model = get_latest_ready_model(db, config.id)
    if model is None:
        raise ValueError(
            f"No ready model for config '{config.name}' (id={config.id})."
        )

    artifact = _load_artifact(model)
    now      = datetime.utcnow()
    results: list[ForecastHorizonResult] = []

    logger.info(
        "Horizon forecast: config_id=%d model_id=%d %d steps",
        config.id, model.id, len(steps),
    )

    for i, spec in enumerate(steps):
        biz_val     = float(spec["business_metric_value"])
        minutes_fwd = int(spec["minutes_ahead"])
        at_time     = now + timedelta(minutes=minutes_fwd)

        pred = _run_inference(artifact, biz_val, at_time=at_time)

        row = ForecastHorizonResult(
            config_id=config.id,
            model_id=model.id,
            step=i,
            minutes_ahead=minutes_fwd,
            business_metric_value=biz_val,
            predicted_cpu_percent=pred.cpu.point,
            predicted_ram_gb=pred.ram.point,
            predicted_network_mbps=pred.network.point,
            lower_cpu_percent=pred.cpu.lower,
            lower_ram_gb=pred.ram.lower,
            lower_network_mbps=pred.network.lower,
            upper_cpu_percent=pred.cpu.upper,
            upper_ram_gb=pred.ram.upper,
            upper_network_mbps=pred.network.upper,
        )
        results.append(row)

    db.add_all(results)
    db.commit()
    for r in results:
        db.refresh(r)

    logger.info(
        "Horizon done: %d steps  cpu range [%.1f%%, %.1f%%]",
        len(results),
        min(r.predicted_cpu_percent for r in results),
        max(r.predicted_cpu_percent for r in results),
    )
    return results
