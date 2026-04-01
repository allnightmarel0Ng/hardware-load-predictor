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
    """Full prediction for all five targets at one timestep."""
    cpu:         TargetPrediction
    ram_gb:      TargetPrediction
    ram_percent: TargetPrediction
    network:     TargetPrediction
    disk:        TargetPrediction


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
    Run point + quantile inference for all five targets.
    Percentages clamped [0,100]; GB/Mbps to [0,∞).
    Intervals enforced: lower <= point <= upper always holds.
    """
    scaler      = artifact["scaler"]
    model_point = artifact["model"]
    model_lower = artifact.get("model_lower")
    model_upper = artifact.get("model_upper")

    X        = _build_inference_feature(business_value, at_time)
    X_scaled = scaler.transform(X)

    pt = model_point.predict(X_scaled)[0]   # shape: (5,)
    # col: 0=cpu_pct, 1=ram_gb, 2=ram_pct, 3=net_mbps, 4=disk_pct
    cpu_pt  = round(float(np.clip(pt[0], 0.0, 100.0)), 2)
    rgb_pt  = round(float(max(0.0, pt[1])), 2)
    rpt_pt  = round(float(np.clip(pt[2], 0.0, 100.0)), 2)
    net_pt  = round(float(max(0.0, pt[3])), 2)
    dsk_pt  = round(float(np.clip(pt[4], 0.0, 100.0)), 2)

    if model_lower is not None and model_upper is not None:
        lo = model_lower.predict(X_scaled)[0]
        hi = model_upper.predict(X_scaled)[0]

        def _bounds_pct(lo_v, pt_v, hi_v):
            return (
                round(float(np.clip(lo_v, 0.0, pt_v)), 2),
                round(float(np.clip(hi_v, pt_v, 100.0)), 2),
            )
        def _bounds_abs(lo_v, pt_v, hi_v):
            return (
                round(float(max(0.0, min(lo_v, pt_v))), 2),
                round(float(max(pt_v, hi_v)), 2),
            )

        cpu_lo, cpu_hi = _bounds_pct(lo[0], cpu_pt, hi[0])
        rgb_lo, rgb_hi = _bounds_abs(lo[1], rgb_pt, hi[1])
        rpt_lo, rpt_hi = _bounds_pct(lo[2], rpt_pt, hi[2])
        net_lo, net_hi = _bounds_abs(lo[3], net_pt, hi[3])
        dsk_lo, dsk_hi = _bounds_pct(lo[4], dsk_pt, hi[4])
    else:
        cpu_lo = cpu_hi = rgb_lo = rgb_hi = None
        rpt_lo = rpt_hi = net_lo = net_hi = dsk_lo = dsk_hi = None

    return InferencePrediction(
        cpu=         TargetPrediction(point=cpu_pt, lower=cpu_lo, upper=cpu_hi),
        ram_gb=      TargetPrediction(point=rgb_pt, lower=rgb_lo, upper=rgb_hi),
        ram_percent= TargetPrediction(point=rpt_pt, lower=rpt_lo, upper=rpt_hi),
        network=     TargetPrediction(point=net_pt, lower=net_lo, upper=net_hi),
        disk=        TargetPrediction(point=dsk_pt, lower=dsk_lo, upper=dsk_hi),
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
        predicted_ram_gb=pred.ram_gb.point,
        predicted_ram_percent=pred.ram_percent.point,
        predicted_network_mbps=pred.network.point,
        predicted_disk_io_percent=pred.disk.point,
        lower_cpu_percent=pred.cpu.lower,
        lower_ram_gb=pred.ram_gb.lower,
        lower_ram_percent=pred.ram_percent.lower,
        lower_network_mbps=pred.network.lower,
        lower_disk_io_percent=pred.disk.lower,
        upper_cpu_percent=pred.cpu.upper,
        upper_ram_gb=pred.ram_gb.upper,
        upper_ram_percent=pred.ram_percent.upper,
        upper_network_mbps=pred.network.upper,
        upper_disk_io_percent=pred.disk.upper,
    )
    db.add(result)
    db.commit()
    db.refresh(result)

    has_iv = pred.cpu.lower is not None
    logger.info(
        "Result: cpu=%.1f%%%s  ram=%.2fGB(%.1f%%)  net=%.1fMbps  disk=%.1f%%",
        pred.cpu.point,
        f"(±{round((pred.cpu.upper - pred.cpu.lower) / 2, 1)})" if has_iv else "",
        pred.ram_gb.point, pred.ram_percent.point,
        pred.network.point, pred.disk.point,
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
            predicted_ram_gb=pred.ram_gb.point,
            predicted_ram_percent=pred.ram_percent.point,
            predicted_network_mbps=pred.network.point,
            predicted_disk_io_percent=pred.disk.point,
            lower_cpu_percent=pred.cpu.lower,
            lower_ram_gb=pred.ram_gb.lower,
            lower_ram_percent=pred.ram_percent.lower,
            lower_network_mbps=pred.network.lower,
            lower_disk_io_percent=pred.disk.lower,
            upper_cpu_percent=pred.cpu.upper,
            upper_ram_gb=pred.ram_gb.upper,
            upper_ram_percent=pred.ram_percent.upper,
            upper_network_mbps=pred.network.upper,
            upper_disk_io_percent=pred.disk.upper,
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
