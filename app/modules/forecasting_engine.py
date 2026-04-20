"""
Module 5 — Forecasting Engine  (v3: per-target adaptive inference)

Artifact schema (v3):
    {
      "per_target": {
          "cpu":     {"model": ..., "scaler": ..., "lag": int,
                      "model_type": str, "mean_val": float, ...},
          "ram_gb":  {...},
          "ram_pct": {...},
          "net":     {...},
          "disk":    {...},
      },
      "version": "v3_per_target",
    }

At inference time:
  - Fetches last AR_WINDOW points of system metrics and business metric
    from Prometheus (same host:port as in ForecastingConfig).
  - Builds the same advanced feature vector used during training,
    using each target's individual lag.
  - Runs each target's model independently.
  - Falls back gracefully to mean_val if model is None (mean_baseline).

Backward compat: if artifact has old schema (key "model" at top level),
  falls back to legacy single-model inference.
"""
from __future__ import annotations

import logging
import math
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
# Imported lazily inside functions to avoid circular import with model_trainer
# (request_handler imports both modules at module level)

logger = logging.getLogger(__name__)

# Mirror constants from model_trainer — kept in sync manually
LOOKBACK    = 30
AR_WINDOW   = LOOKBACK + 5
TARGET_KEYS = ["cpu", "ram_gb", "ram_pct", "net", "disk"]


def _get_latest_ready_model(db, config_id):
    """Lazy import wrapper to avoid circular dependency."""
    from app.modules.model_trainer import get_latest_ready_model
    return get_latest_ready_model(db, config_id)


def _get_target_extra_features(target_key: str, hist: "np.ndarray") -> list:
    """Lazy import wrapper to avoid circular dependency."""
    from app.modules.model_trainer import _target_extra_features
    return _target_extra_features(target_key, hist)


class TargetPrediction(NamedTuple):
    point: float
    lower: float | None
    upper: float | None


@dataclass
class InferencePrediction:
    cpu:         TargetPrediction
    ram_gb:      TargetPrediction
    ram_percent: TargetPrediction
    network:     TargetPrediction
    disk:        TargetPrediction


# ── Artifact loading ──────────────────────────────────────────────────────────

def _load_artifact(model: TrainedModel) -> dict:
    if not model.artifact_path:
        raise ValueError(f"Model {model.id} has no artifact_path.")
    return joblib.load(model.artifact_path)


def _is_v3(artifact: dict) -> bool:
    return artifact.get("version") == "v3_per_target"


# ── Prometheus fetch helpers ──────────────────────────────────────────────────

def _fetch_series(config: ForecastingConfig, query: str, n_points: int) -> np.ndarray:
    from app.modules.data_collector import _query_prometheus
    end   = datetime.utcnow()
    start = end - timedelta(minutes=n_points + 5)
    try:
        series = _query_prometheus(config.host, config.port, query,
                                   start, end, step_seconds=60)
        if not series:
            return np.zeros(n_points)
        vals = np.array([p["value"] for p in series[-n_points:]], dtype=float)
        if len(vals) < n_points:
            vals = np.pad(vals, (n_points - len(vals), 0), mode="edge")
        return vals[-n_points:]
    except Exception as exc:
        logger.warning("Prometheus fetch failed (%s): %s", query[:40], exc)
        return np.zeros(n_points)


def _fetch_system_context(config: ForecastingConfig) -> dict[str, np.ndarray]:
    from app.core.config import settings
    instance = getattr(config, "instance_label", None)
    ifilter  = f'instance="{instance}"' if instance else 'instance=~".+"'
    def q(tmpl): return tmpl.format(instance=ifilter)
    return {
        "cpu":     _fetch_series(config, q(settings.prometheus_cpu_query),     AR_WINDOW),
        "ram_pct": _fetch_series(config, q(settings.prometheus_ram_pct_query), AR_WINDOW),
        "net":     _fetch_series(config, q(settings.prometheus_net_query),     AR_WINDOW),
        "disk":    _fetch_series(config, q(settings.prometheus_disk_query),    AR_WINDOW),
    }


def _fetch_biz_history(config: ForecastingConfig, current_value: float) -> np.ndarray:
    from app.modules.data_collector import _query_prometheus
    end   = datetime.utcnow()
    start = end - timedelta(minutes=AR_WINDOW + 5)
    try:
        series = _query_prometheus(
            config.host, config.port, config.business_metric_formula,
            start, end, step_seconds=60,
        )
        if not series:
            raise ValueError("empty")
        vals = np.array([p["value"] for p in series[-AR_WINDOW:]], dtype=float)
        if len(vals) < AR_WINDOW:
            vals = np.pad(vals, (AR_WINDOW - len(vals), 0), mode="edge")
        return np.append(vals[1:], current_value)
    except Exception as exc:
        logger.warning("Biz history fetch failed: %s", exc)
        return np.full(AR_WINDOW, current_value)


# ── Single-target inference feature vector ────────────────────────────────────

def _build_inference_row(
    biz_hist: np.ndarray,
    sys_ctx: dict[str, np.ndarray],
    target_key: str,
    tau: int,
    at_time: datetime | None = None,
    max_biz_train: float | None = None,
) -> np.ndarray:
    """Build exactly one feature row (same logic as _build_features_for_target).

    max_biz_train: stored in the model artifact at training time.
        Used to compute biz_above_max — the extrapolation signal.
    """
    t   = at_time or datetime.utcnow()
    i   = len(biz_hist) - 1
    n   = len(biz_hist)

    hour  = t.hour + t.minute / 60.0
    dow   = t.weekday()
    sin_h = math.sin(2 * math.pi * hour / 24)
    cos_h = math.cos(2 * math.pi * hour / 24)
    sin_d = math.sin(2 * math.pi * dow  / 7)
    cos_d = math.cos(2 * math.pi * dow  / 7)
    trend = 1.0

    bl   = biz_hist[i - tau] if i >= tau else biz_hist[0]
    rm30 = biz_hist[max(0, i - 30):i].mean() if i > 0 else bl
    bn   = bl / (rm30 + 1e-9)
    d1   = biz_hist[i] - biz_hist[i-1] if i >= 1 else 0.0
    d2   = biz_hist[i-1] - biz_hist[i-2] if i >= 2 else 0.0
    mu5b = biz_hist[max(0, i-5):i].mean() if i > 0 else bl
    bz   = (biz_hist[i] - mu5b) / (biz_hist[max(0, i-5):i].std() + 1e-9) if i > 0 else 0.0

    biz_above_max = max(0.0, bl - max_biz_train) if max_biz_train is not None else 0.0
    biz_relative  = bl / (max_biz_train + 1e-9)  if max_biz_train is not None else 1.0

    feats: list[float] = [
        sin_h, cos_h, sin_d, cos_d, trend,
        bl, bn, d1, d2, bz, bl * sin_h, bl * cos_h,
        biz_above_max, biz_relative,
    ]

    for key in ("cpu", "ram_pct", "net", "disk"):
        arr = sys_ctx.get(key, np.zeros(AR_WINDOW))
        l1  = arr[-1]
        l2  = arr[-2] if len(arr) >= 2 else arr[-1]
        l3  = arr[-3] if len(arr) >= 3 else arr[-1]
        m5  = arr[-5:].mean()  if len(arr) >= 5  else arr.mean()
        m15 = arr[-15:].mean() if len(arr) >= 15 else arr.mean()
        m30 = arr[-30:].mean() if len(arr) >= 30 else arr.mean()
        s5  = arr[-5:].std()   + 1e-9 if len(arr) >= 5  else 1.0
        s15 = arr[-15:].std()  + 1e-9 if len(arr) >= 15 else 1.0
        feats.extend([l1, l2, l3, m5, m15, m30, s5, s15])

    # Use sys_ctx for the target's own history (ram_gb not in ctx → use ram_pct proxy)
    ctx_key = "ram_pct" if target_key == "ram_gb" else target_key
    hist = sys_ctx.get(ctx_key, np.zeros(AR_WINDOW))
    feats.extend(_get_target_extra_features(target_key, hist))

    return np.array([feats], dtype=float)


# ── Core inference ────────────────────────────────────────────────────────────

def _infer_one_target(
    info: dict,
    biz_hist: np.ndarray,
    sys_ctx: dict[str, np.ndarray],
    target_key: str,
    at_time: datetime | None = None,
    hypothetical: bool = False,
) -> float:
    """Predict one target. Returns point estimate (float).

    hypothetical=True: replace live AR context with training-time means.
    Use this when the user asks 'what if RPS=X?' rather than
    'what will happen in the next N minutes?'.
    """
    if info["model_type"] == "mean_baseline" or info["model"] is None:
        return float(info["mean_val"])

    tau           = info["lag"]
    scaler        = info["scaler"]
    model         = info["model"]
    max_biz_train = info.get("max_biz_train")

    if hypothetical and "mean_sys_ctx" in info:
        # Replace live AR features with training-time mean repeated AR_WINDOW times
        mean_ctx = info["mean_sys_ctx"]
        ctx = {k: np.full(AR_WINDOW, v) for k, v in mean_ctx.items()}
    else:
        ctx = sys_ctx

    X_row = _build_inference_row(biz_hist, ctx, target_key, tau, at_time,
                                  max_biz_train=max_biz_train)

    X_sc = scaler.transform(X_row) if scaler is not None else X_row

    return float(model.predict(X_sc)[0])


def _run_inference_v3(
    artifact: dict,
    biz_hist: np.ndarray,
    sys_ctx: dict[str, np.ndarray],
    at_time: datetime | None = None,
    hypothetical: bool = False,
) -> InferencePrediction:
    """Full v3 inference: five independent models."""
    pt = artifact["per_target"]

    cpu_pt = round(float(np.clip(_infer_one_target(pt["cpu"],     biz_hist, sys_ctx, "cpu",     at_time, hypothetical), 0, 100)), 2)
    rgb_pt = round(max(0.0, _infer_one_target(pt["ram_gb"],  biz_hist, sys_ctx, "ram_gb",  at_time, hypothetical)), 2)
    rpt_pt = round(float(np.clip(_infer_one_target(pt["ram_pct"], biz_hist, sys_ctx, "ram_pct", at_time, hypothetical), 0, 100)), 2)
    net_pt = round(max(0.0, _infer_one_target(pt["net"],     biz_hist, sys_ctx, "net",     at_time, hypothetical)), 2)
    dsk_pt = round(float(np.clip(_infer_one_target(pt["disk"],    biz_hist, sys_ctx, "disk",    at_time, hypothetical), 0, 100)), 2)

    # No quantile intervals in v3 (can be added later per target)
    return InferencePrediction(
        cpu=         TargetPrediction(point=cpu_pt, lower=None, upper=None),
        ram_gb=      TargetPrediction(point=rgb_pt, lower=None, upper=None),
        ram_percent= TargetPrediction(point=rpt_pt, lower=None, upper=None),
        network=     TargetPrediction(point=net_pt, lower=None, upper=None),
        disk=        TargetPrediction(point=dsk_pt, lower=None, upper=None),
    )


def _run_inference_legacy(
    artifact: dict,
    business_value: float,
    at_time: datetime | None = None,
) -> InferencePrediction:
    """Backward-compatible inference for old single-model artifacts."""
    t    = at_time or datetime.utcnow()
    hour = t.hour + t.minute / 60.0
    dow  = t.weekday()
    X = np.array([[
        business_value, business_value, 0.0, business_value, 0.0,
        np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
        np.sin(2*np.pi*dow/7),  np.cos(2*np.pi*dow/7),
    ]])
    scaler = artifact["scaler"]
    model  = artifact["model"]
    pt     = model.predict(scaler.transform(X))[0]
    return InferencePrediction(
        cpu=         TargetPrediction(round(float(np.clip(pt[0],0,100)),2), None, None),
        ram_gb=      TargetPrediction(round(float(max(0,pt[1])),2),        None, None),
        ram_percent= TargetPrediction(round(float(np.clip(pt[2],0,100)),2),None, None),
        network=     TargetPrediction(round(float(max(0,pt[3])),2),        None, None),
        disk=        TargetPrediction(round(float(np.clip(pt[4],0,100)),2),None, None),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def forecast(
    db: Session,
    config: ForecastingConfig,
    business_metric_value: float,
    hypothetical: bool = True,
) -> ForecastResult:
    """Produce and persist a single-step forecast.

    hypothetical=True (default): AR context is replaced with training-time
    means so the prediction reflects 'what would happen at biz=X' rather
    than 'what will happen in the next few minutes given the current state'.
    Set hypothetical=False for short-horizon operational forecasts.
    """
    model_rec = _get_latest_ready_model(db, config.id)
    if model_rec is None:
        raise ValueError(
            f"No ready model for config '{config.name}' (id={config.id}). "
            "Train first via POST /configs/{id}/train/."
        )

    logger.info(
        "Forecast: config_id=%d model_id=%d (v%d) biz=%.2f hypothetical=%s",
        config.id, model_rec.id, model_rec.version, business_metric_value, hypothetical,
    )

    artifact = _load_artifact(model_rec)

    if _is_v3(artifact):
        biz_hist = _fetch_biz_history(config, business_metric_value)
        sys_ctx  = _fetch_system_context(config)
        pred     = _run_inference_v3(artifact, biz_hist, sys_ctx,
                                     hypothetical=hypothetical)
    else:
        pred = _run_inference_legacy(artifact, business_metric_value)

    result = ForecastResult(
        config_id=config.id,
        model_id=model_rec.id,
        business_metric_value=business_metric_value,
        predicted_cpu_percent=pred.cpu.point,
        predicted_ram_gb=pred.ram_gb.point,
        predicted_ram_percent=pred.ram_percent.point,
        predicted_network_mbps=pred.network.point,
        predicted_disk_io_percent=pred.disk.point,
        lower_cpu_percent=None, lower_ram_gb=None,
        lower_ram_percent=None, lower_network_mbps=None,
        lower_disk_io_percent=None,
        upper_cpu_percent=None, upper_ram_gb=None,
        upper_ram_percent=None, upper_network_mbps=None,
        upper_disk_io_percent=None,
    )
    db.add(result)
    db.commit()
    db.refresh(result)

    logger.info(
        "Result: cpu=%.1f%%  ram=%.2fGB(%.1f%%)  net=%.1fMbps  disk=%.1f%%",
        pred.cpu.point, pred.ram_gb.point, pred.ram_percent.point,
        pred.network.point, pred.disk.point,
    )
    return result


def forecast_horizon(
    db: Session,
    config: ForecastingConfig,
    steps: list[dict],
) -> list[ForecastHorizonResult]:
    if not steps:
        raise ValueError("steps list must not be empty.")

    model_rec = _get_latest_ready_model(db, config.id)
    if model_rec is None:
        raise ValueError(f"No ready model for config '{config.name}'.")

    artifact = _load_artifact(model_rec)
    sys_ctx  = _fetch_system_context(config) if _is_v3(artifact) else {}
    now      = datetime.utcnow()
    results: list[ForecastHorizonResult] = []

    for i, spec in enumerate(steps):
        biz_val     = float(spec["business_metric_value"])
        minutes_fwd = int(spec["minutes_ahead"])
        at_time     = now + timedelta(minutes=minutes_fwd)

        if _is_v3(artifact):
            biz_hist = _fetch_biz_history(config, biz_val)
            pred     = _run_inference_v3(artifact, biz_hist, sys_ctx, at_time)
        else:
            pred = _run_inference_legacy(artifact, biz_val, at_time)

        results.append(ForecastHorizonResult(
            config_id=config.id, model_id=model_rec.id,
            step=i, minutes_ahead=minutes_fwd,
            business_metric_value=biz_val,
            predicted_cpu_percent=pred.cpu.point,
            predicted_ram_gb=pred.ram_gb.point,
            predicted_ram_percent=pred.ram_percent.point,
            predicted_network_mbps=pred.network.point,
            predicted_disk_io_percent=pred.disk.point,
            lower_cpu_percent=None, lower_ram_gb=None,
            lower_ram_percent=None, lower_network_mbps=None,
            lower_disk_io_percent=None,
            upper_cpu_percent=None, upper_ram_gb=None,
            upper_ram_percent=None, upper_network_mbps=None,
            upper_disk_io_percent=None,
        ))

    db.add_all(results)
    db.commit()
    for r in results: db.refresh(r)
    return results
