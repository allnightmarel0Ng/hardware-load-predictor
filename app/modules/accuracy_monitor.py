"""
Module 6 — Accuracy Monitor
Evaluates post-deployment model accuracy by comparing stored forecasts
against actual system metric values fetched from Prometheus.

How it works:
  1. The scheduler wakes up every RETRAIN_INTERVAL_HOURS.
  2. For each READY model it finds all ForecastResult rows that:
       a. were created more than LAG_FETCH_BUFFER_MINUTES ago
          (so the actual values have had time to materialise in Prometheus)
       b. have not yet had their actuals fetched (actuals_fetched_at IS NULL)
  3. It queries Prometheus for the actual cpu/ram/net values at each
     forecast timestamp using an instant-query at (created_at + lag_minutes).
  4. It writes the actuals back into ForecastResult and computes MAE,
     RMSE, MAPE, R² across all evaluated pairs for that model, storing
     the result in ModelEvaluation.
  5. If avg R² < accuracy_threshold it triggers retraining.

The Prometheus fetch is isolated in _fetch_actuals_from_prometheus() so it
can be monkeypatched in tests without network access.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

import httpx
import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.db_models import (
    ForecastingConfig,
    ForecastResult,
    ModelEvaluation,
    ModelStatus,
    TrainedModel,
)

logger = logging.getLogger(__name__)

_scheduler: BackgroundScheduler | None = None

# After a forecast is issued, wait at least this many minutes before
# trying to fetch actuals from Prometheus (so data has propagated).
LAG_FETCH_BUFFER_MINUTES: int = 10

# Minimum number of forecast-actual pairs required before we compute
# evaluation metrics. Below this we skip to avoid noisy estimates.
MIN_EVAL_SAMPLES: int = 5


# ── Prometheus queries ────────────────────────────────────────────────────────

class ActualValues(NamedTuple):
    cpu_percent:      float
    ram_gb:           float
    ram_percent:      float
    network_mbps:     float
    disk_io_percent:  float


def _fetch_actuals_from_prometheus(
    host: str,
    port: int,
    at_time: datetime,
) -> ActualValues | None:
    """
    Fetch actual system metric values from Prometheus at a specific timestamp
    using the instant-query API (GET /api/v1/query?query=<expr>&time=<ts>).

    Fetches five targets: CPU %, RAM GB, RAM %, Network Mbps, Disk IO %.
    Returns None if Prometheus is unreachable or any metric returns no data.
    """
    ts       = at_time.timestamp()
    base_url = f"http://{host}:{port}/api/v1/query"

    queries = {
        "cpu":      settings.prometheus_cpu_query,
        "ram_gb":   settings.prometheus_ram_gb_query,
        "ram_pct":  settings.prometheus_ram_pct_query,
        "net":      settings.prometheus_net_query,
        "disk":     settings.prometheus_disk_query,
    }

    results: dict[str, float] = {}
    try:
        with httpx.Client(timeout=10.0) as client:
            for key, expr in queries.items():
                resp = client.get(base_url, params={"query": expr, "time": ts})
                resp.raise_for_status()
                data = resp.json()
                result_list = data.get("data", {}).get("result", [])
                if not result_list:
                    logger.debug("Prometheus: no data for %s at %s", key, at_time)
                    return None
                results[key] = float(result_list[0]["value"][1])
    except httpx.HTTPError as exc:
        logger.warning("Prometheus unreachable for actuals fetch: %s", exc)
        return None
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning("Unexpected Prometheus response format: %s", exc)
        return None

    return ActualValues(
        cpu_percent=     max(0.0, min(100.0, results["cpu"])),
        ram_gb=          max(0.0, results["ram_gb"]),
        ram_percent=     max(0.0, min(100.0, results["ram_pct"])),
        network_mbps=    max(0.0, results["net"]),
        disk_io_percent= max(0.0, min(100.0, results["disk"])),
    )


# ── Actuals back-fill ─────────────────────────────────────────────────────────

def _backfill_actuals(db: Session, model: TrainedModel) -> list[ForecastResult]:
    """
    For all ForecastResult rows tied to this model that:
      - were issued more than LAG_FETCH_BUFFER_MINUTES ago
      - have not yet had actuals fetched
    fetch the actual metric values from Prometheus and write them back.

    Returns the updated rows that now have all three actuals populated.
    """
    cutoff = datetime.utcnow() - timedelta(minutes=LAG_FETCH_BUFFER_MINUTES)

    pending = (
        db.query(ForecastResult)
        .filter(
            ForecastResult.model_id == model.id,
            ForecastResult.actuals_fetched_at.is_(None),
            ForecastResult.created_at <= cutoff,
        )
        .order_by(ForecastResult.created_at)
        .all()
    )

    if not pending:
        logger.debug("No pending actuals for model_id=%d", model.id)
        return []

    logger.info(
        "Fetching actuals for %d forecast(s) from model_id=%d",
        len(pending), model.id,
    )

    config = db.get(ForecastingConfig, model.config_id)
    lag = model.lag_minutes or 0
    filled: list[ForecastResult] = []

    for fr in pending:
        # The actual values materialise at created_at + lag (the forecast horizon)
        actual_ts = fr.created_at + timedelta(minutes=lag)
        actuals = _fetch_actuals_from_prometheus(config.host, config.port, actual_ts)

        if actuals is None:
            # Prometheus unavailable or no data — skip this row, try again later
            continue

        fr.actual_cpu_percent     = actuals.cpu_percent
        fr.actual_ram_gb          = actuals.ram_gb
        fr.actual_ram_percent     = actuals.ram_percent
        fr.actual_network_mbps    = actuals.network_mbps
        fr.actual_disk_io_percent = actuals.disk_io_percent
        fr.actuals_fetched_at     = datetime.utcnow()
        filled.append(fr)

    if filled:
        db.commit()
        logger.info("Wrote actuals for %d forecast(s)", len(filled))

    return filled


# ── Metric computation ────────────────────────────────────────────────────────

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _compute_post_deployment_metrics(rows: list[ForecastResult]) -> dict:
    """
    Compute MAE, RMSE, MAPE, R² from a list of ForecastResult rows
    that have all five predicted and actual values populated.
    """
    cpu_pred = np.array([r.predicted_cpu_percent     for r in rows])
    rgb_pred = np.array([r.predicted_ram_gb           for r in rows])
    rpt_pred = np.array([r.predicted_ram_percent      for r in rows])
    net_pred = np.array([r.predicted_network_mbps     for r in rows])
    dsk_pred = np.array([r.predicted_disk_io_percent  for r in rows])

    cpu_true = np.array([r.actual_cpu_percent     for r in rows])
    rgb_true = np.array([r.actual_ram_gb           for r in rows])
    rpt_true = np.array([r.actual_ram_percent      for r in rows])
    net_true = np.array([r.actual_network_mbps     for r in rows])
    dsk_true = np.array([r.actual_disk_io_percent  for r in rows])

    def _mae(t, p):  return float(np.mean(np.abs(t - p)))
    def _rmse(t, p): return float(np.sqrt(np.mean((t - p) ** 2)))
    def _r2(t, p):
        ss_res = float(np.sum((t - p) ** 2))
        ss_tot = float(np.sum((t - np.mean(t)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    mape_all = _mape(
        np.concatenate([cpu_true, rgb_true, rpt_true, net_true, dsk_true]),
        np.concatenate([cpu_pred, rgb_pred, rpt_pred, net_pred, dsk_pred]),
    )

    return {
        "mae_cpu":     round(_mae(cpu_true,  cpu_pred),  4),
        "mae_ram_gb":  round(_mae(rgb_true,  rgb_pred),  4),
        "mae_ram_pct": round(_mae(rpt_true,  rpt_pred),  4),
        "mae_net":     round(_mae(net_true,  net_pred),  4),
        "mae_disk":    round(_mae(dsk_true,  dsk_pred),  4),
        "rmse_cpu":    round(_rmse(cpu_true, cpu_pred),  4),
        "rmse_ram_gb": round(_rmse(rgb_true, rgb_pred),  4),
        "rmse_ram_pct":round(_rmse(rpt_true, rpt_pred),  4),
        "rmse_net":    round(_rmse(net_true, net_pred),  4),
        "rmse_disk":   round(_rmse(dsk_true, dsk_pred),  4),
        "r2_cpu":      round(_r2(cpu_true,   cpu_pred),  4),
        "r2_ram_gb":   round(_r2(rgb_true,   rgb_pred),  4),
        "r2_ram_pct":  round(_r2(rpt_true,   rpt_pred),  4),
        "r2_net":      round(_r2(net_true,   net_pred),  4),
        "r2_disk":     round(_r2(dsk_true,   dsk_pred),  4),
        "mape_overall": round(mape_all, 4),
    }


def _needs_retraining(metrics: dict, threshold: float) -> bool:
    r2_keys = ("r2_cpu", "r2_ram_gb", "r2_ram_pct", "r2_net", "r2_disk")
    r2_values = [metrics[k] for k in r2_keys if k in metrics]
    if not r2_values:
        return False
    avg_r2 = sum(r2_values) / len(r2_values)
    needs = avg_r2 < threshold
    logger.info(
        "Post-deployment avg R²=%.3f  threshold=%.2f  needs_retraining=%s",
        avg_r2, threshold, needs,
    )
    return needs


# ── Per-model evaluation ──────────────────────────────────────────────────────

def _evaluate_model(db: Session, model: TrainedModel) -> ModelEvaluation | None:
    """
    Back-fill actuals, compute post-deployment metrics + PSI drift check,
    persist a ModelEvaluation row, and return it (or None if insufficient data).
    """
    from app.modules.drift_detector import check_drift_from_snapshot

    # Step 1: try to fill in any pending actuals
    _backfill_actuals(db, model)

    # Step 2: collect all rows that now have actuals
    evaluated_rows = (
        db.query(ForecastResult)
        .filter(
            ForecastResult.model_id == model.id,
            ForecastResult.actuals_fetched_at.isnot(None),
        )
        .all()
    )

    if len(evaluated_rows) < MIN_EVAL_SAMPLES:
        logger.info(
            "Model %d has only %d evaluated forecast(s) — need %d, skipping.",
            model.id, len(evaluated_rows), MIN_EVAL_SAMPLES,
        )
        return None

    # Step 3: compute output accuracy metrics (MAE, RMSE, MAPE, R²)
    metrics = _compute_post_deployment_metrics(evaluated_rows)

    # Step 4: check input distribution drift (PSI)
    drift_triggered = False
    psi_value       = None
    psi_level       = "stable"

    ref_dist = (model.parameters or {}).get("input_distribution")
    if ref_dist:
        current_biz = np.array(
            [r.business_metric_value for r in evaluated_rows], dtype=float
        )
        drift = check_drift_from_snapshot(ref_dist, current_biz)
        psi_value = drift.psi
        psi_level = drift.level
        logger.info(
            "Drift check model %d: PSI=%.4f  level=%s  drifted=%s",
            model.id, drift.psi, drift.level, drift.is_drifted,
        )
        if drift.is_drifted:
            logger.warning(
                "Significant input drift detected for model %d "
                "(PSI=%.4f ≥ %.2f) — triggering retraining.",
                model.id, drift.psi, 0.20,
            )
            drift_triggered = True
    else:
        logger.debug(
            "Model %d has no stored input_distribution — skipping PSI check.", model.id
        )

    # Step 5: decide on retraining from R² or drift
    r2_triggered = _needs_retraining(metrics, settings.accuracy_threshold)
    retrain      = r2_triggered or drift_triggered

    # Step 6: persist evaluation record
    evaluation = ModelEvaluation(
        model_id=model.id,
        config_id=model.config_id,
        n_samples=len(evaluated_rows),
        triggered_retrain=retrain,
        psi_value=psi_value,
        psi_level=psi_level,
        **metrics,
    )
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)

    logger.info(
        "Evaluation model %d: n=%d  "
        "R²(cpu=%.3f ram_gb=%.3f ram_pct=%.3f net=%.3f disk=%.3f)  "
        "MAPE=%.1f%%  PSI=%.4f(%s)  retrain=%s",
        model.id, len(evaluated_rows),
        metrics["r2_cpu"],    metrics["r2_ram_gb"],  metrics["r2_ram_pct"],
        metrics["r2_net"],    metrics["r2_disk"],
        metrics["mape_overall"], psi_value or 0.0, psi_level, retrain,
    )

    if retrain:
        _trigger_retrain(db, model.config_id)

    return evaluation


# ── Retrain trigger ───────────────────────────────────────────────────────────

def _trigger_retrain(db: Session, config_id: int) -> None:
    """Initiate background retraining. Imports are deferred to avoid circular deps."""
    from app.modules.data_collector import fetch_historical_data
    from app.modules.correlation_analyzer import analyze
    from app.modules.model_trainer import train_model

    config = db.get(ForecastingConfig, config_id)
    if not config:
        logger.error("Retrain requested for unknown config_id=%d", config_id)
        return

    logger.warning("Triggering retraining for config_id=%d (%s)", config_id, config.name)
    try:
        bundle = fetch_historical_data(
            config.host, config.port, config.business_metric_formula
        )
        report = analyze(bundle)
        if report.is_business_constant:
            logger.warning(
                "Business metric appears constant for config_id=%d — skipping retrain.",
                config_id,
            )
            return
        train_model(db, config, bundle, report)
        logger.info("Retraining complete for config_id=%d", config_id)
    except Exception:
        logger.exception("Retraining failed for config_id=%d", config_id)


# ── Scheduler job ─────────────────────────────────────────────────────────────

def _evaluate_all_models() -> None:
    """
    Main job run by the scheduler. Evaluates every READY model.
    Creates a fresh DB session per run so the job is independent
    of any HTTP request lifecycle.
    """
    db: Session = SessionLocal()
    try:
        models = db.query(TrainedModel).filter_by(status=ModelStatus.READY).all()
        logger.info("Accuracy check: evaluating %d model(s)", len(models))
        for model in models:
            try:
                _evaluate_model(db, model)
            except Exception:
                logger.exception("Evaluation failed for model_id=%d", model.id)
    finally:
        db.close()


# ── Public API ────────────────────────────────────────────────────────────────

def get_accuracy_status(db: Session, model_id: int) -> dict:
    """
    Return the latest ModelEvaluation for a given model plus health status.
    Called by the request handler to serve GET /models/{id}/accuracy.
    """
    model = db.get(TrainedModel, model_id)
    if not model:
        return {}

    evaluations = (
        db.query(ModelEvaluation)
        .filter_by(model_id=model_id)
        .order_by(ModelEvaluation.evaluated_at.desc())
        .all()
    )

    latest = evaluations[0] if evaluations else None
    n_samples_total = sum(e.n_samples for e in evaluations)

    is_healthy = True
    health_reason = None

    if latest is None:
        is_healthy = False
        health_reason = "No evaluations have been run yet."
    else:
        r2_vals = [v for v in [latest.r2_cpu, latest.r2_ram, latest.r2_net] if v is not None]
        if r2_vals:
            avg_r2 = sum(r2_vals) / len(r2_vals)
            if avg_r2 < settings.accuracy_threshold:
                is_healthy = False
                health_reason = (
                    f"Average R² ({avg_r2:.3f}) is below threshold "
                    f"({settings.accuracy_threshold})."
                )

    config = db.get(ForecastingConfig, model.config_id)

    return {
        "model_id": model_id,
        "config_id": model.config_id,
        "config_name": config.name if config else "unknown",
        "n_evaluations": len(evaluations),
        "n_samples_total": n_samples_total,
        "latest_evaluation": latest,
        "is_healthy": is_healthy,
        "health_reason": health_reason,
    }


def force_evaluate(db: Session, model_id: int) -> ModelEvaluation | None:
    """
    Trigger an immediate evaluation for a specific model.
    Called by POST /models/{id}/accuracy/evaluate.
    """
    model = db.get(TrainedModel, model_id)
    if not model or model.status != ModelStatus.READY:
        return None
    return _evaluate_model(db, model)


# ── Scheduler lifecycle ───────────────────────────────────────────────────────

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
        next_run_time=datetime.utcnow(),
    )
    _scheduler.start()
    logger.info(
        "Accuracy monitor started (interval=%dh  min_samples=%d  threshold=%.2f)",
        settings.retrain_interval_hours, MIN_EVAL_SAMPLES, settings.accuracy_threshold,
    )


def stop_scheduler() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Accuracy monitor stopped.")

