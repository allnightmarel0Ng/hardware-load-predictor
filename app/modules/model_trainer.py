"""
Module 4 — Model Trainer
Trains regression models on historical data and persists them to disk +
the database.

Implementation:
  - Feature engineering: lag-shifted business metric + rolling statistics
    (mean, std over 5-min, 15-min windows) + hour-of-day + day-of-week.
  - Algorithm: MultiOutputRegressor wrapping GradientBoostingRegressor,
    one estimator per target (cpu%, ram_gb, net_mbps).
    Research confirmed Gradient Boosting as the top performer on structured
    tabular time-series vs SVR, RF, MLP, LSTM on moderate-size datasets.
  - Serialisation: joblib dump to disk; model type stored in DB.
  - Evaluation: MAE, RMSE, MAPE, R² computed on a 20% held-out test split
    (temporal order preserved — no shuffling).
  - Fallback: if fewer than MIN_TRAIN_SAMPLES points are available after
    feature engineering, Ridge Regression is used instead.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.db_models import ForecastingConfig, ModelStatus, TrainedModel
from app.modules.correlation_analyzer import CorrelationReport
from app.modules.data_collector import MetricsBundle

logger = logging.getLogger(__name__)

MIN_TRAIN_SAMPLES = 50   # minimum rows for gradient boosting; below → Ridge
TEST_SPLIT_RATIO  = 0.20 # last 20% of data used for evaluation


# ── Artifact path helpers ────────────────────────────────────────────────────

def _artifact_path(config_id: int, version: int) -> str:
    directory = Path(settings.model_storage_path) / str(config_id)
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / f"model_v{version}.joblib")


def _next_version(db: Session, config_id: int) -> int:
    latest = (
        db.query(TrainedModel)
        .filter_by(config_id=config_id)
        .order_by(TrainedModel.version.desc())
        .first()
    )
    return (latest.version + 1) if latest else 1


# ── Feature engineering ──────────────────────────────────────────────────────

def _build_features(
    bundle: MetricsBundle,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct feature matrix X and target matrix y from the MetricsBundle.

    Features (per timestep t):
      - biz[t - lag]              lagged business metric value
      - biz_roll5_mean            rolling mean over prev 5 points
      - biz_roll5_std             rolling std over prev 5 points
      - biz_roll15_mean           rolling mean over prev 15 points
      - biz_roll15_std            rolling std over prev 15 points
      - hour_sin, hour_cos        hour-of-day encoded as sine/cosine pair
      - dow_sin, dow_cos          day-of-week encoded as sine/cosine pair

    Targets:
      - cpu_percent at time t
      - ram_gb      at time t
      - net_mbps    at time t

    The lag shift means we need at least `lag` leading points before
    we can form a valid row — those rows are dropped.
    """
    biz = np.array([p["value"]     for p in bundle.business], dtype=float)
    cpu = np.array([p["value"]     for p in bundle.cpu],      dtype=float)
    ram = np.array([p["value"]     for p in bundle.ram],      dtype=float)
    net = np.array([p["value"]     for p in bundle.network],  dtype=float)
    ts  = [p["timestamp"] for p in bundle.business]

    n = len(biz)
    start = max(lag, 15)  # need 15 points for the longest rolling window

    rows_X, rows_y = [], []

    for t in range(start, n):
        biz_lagged = biz[t - lag] if lag > 0 else biz[t]

        window5  = biz[max(0, t - 5):t]
        window15 = biz[max(0, t - 15):t]

        roll5_mean  = float(np.mean(window5))  if len(window5)  > 0 else biz[t]
        roll5_std   = float(np.std(window5))   if len(window5)  > 1 else 0.0
        roll15_mean = float(np.mean(window15)) if len(window15) > 0 else biz[t]
        roll15_std  = float(np.std(window15))  if len(window15) > 1 else 0.0

        # Cyclical encoding of time features
        hour = ts[t].hour + ts[t].minute / 60.0
        dow  = ts[t].weekday()
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        dow_sin  = np.sin(2 * np.pi * dow  / 7.0)
        dow_cos  = np.cos(2 * np.pi * dow  / 7.0)

        rows_X.append([
            biz_lagged,
            roll5_mean, roll5_std,
            roll15_mean, roll15_std,
            hour_sin, hour_cos,
            dow_sin, dow_cos,
        ])
        rows_y.append([cpu[t], ram[t], net[t]])

    X = np.array(rows_X, dtype=float)
    y = np.array(rows_y, dtype=float)
    return X, y


# ── Evaluation helpers ───────────────────────────────────────────────────────

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (ignores zero targets)."""
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, MAPE, R² for each of the three targets
    (columns: 0=cpu, 1=ram, 2=net) plus an overall MAPE.
    """
    names = ["cpu", "ram", "net"]
    metrics: dict = {}
    for i, name in enumerate(names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        metrics[f"mae_{name}"]  = round(float(mean_absolute_error(yt, yp)), 4)
        metrics[f"rmse_{name}"] = round(float(np.sqrt(mean_squared_error(yt, yp))), 4)
        metrics[f"mape_{name}"] = round(_mape(yt, yp), 4)
        metrics[f"r2_{name}"]   = round(float(r2_score(yt, yp)), 4)

    # Overall MAPE (flat across all outputs)
    metrics["mape_overall"] = round(
        float(np.mean([metrics[f"mape_{n}"] for n in names])), 4
    )
    return metrics


# ── Model selection & training ───────────────────────────────────────────────

def _build_model(n_samples: int):
    """
    Return the appropriate sklearn model for the dataset size.
    Below MIN_TRAIN_SAMPLES use Ridge (more stable on tiny data);
    otherwise use GradientBoostingRegressor wrapped for multi-output.
    """
    if n_samples < MIN_TRAIN_SAMPLES:
        logger.info("Small dataset (%d samples) — using Ridge Regression", n_samples)
        return MultiOutputRegressor(Ridge(alpha=1.0)), "ridge"

    logger.info("Using Gradient Boosting Regressor (%d training samples)", n_samples)
    gbr = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )
    return MultiOutputRegressor(gbr), "gradient_boosting"


def _build_quantile_model(alpha: float, n_samples: int):
    """
    Build a quantile GBR for a single alpha level.
    Used to produce prediction interval bounds.
    Falls back to Ridge (which has no quantile mode) when data is small —
    in that case the caller should skip interval computation.
    """
    if n_samples < MIN_TRAIN_SAMPLES:
        return None  # Ridge has no quantile mode; intervals not available
    return MultiOutputRegressor(
        GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=100,   # fewer trees than point model — intervals are cheaper
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
    )


def _fit_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    artifact: str,
) -> tuple[dict, dict, str]:
    """
    Split data, fit point + quantile models, evaluate on test split,
    persist artifact.
    Returns (parameters_dict, metrics_dict, algorithm_name).

    Artifact schema:
        {
          "model":       MultiOutputRegressor  (point estimator)
          "model_lower": MultiOutputRegressor  (10th percentile, or None)
          "model_upper": MultiOutputRegressor  (90th percentile, or None)
          "scaler":      StandardScaler
        }
    """
    n = len(X)
    split = max(1, int(n * (1 - TEST_SPLIT_RATIO)))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model, algo_name = _build_model(len(X_train))

    # Scale features (important for Ridge; harmless for GBR)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Point model ───────────────────────────────────────────────────────────
    model.fit(X_train_s, y_train)

    # ── Quantile models (80% prediction interval: 10th–90th percentile) ──────
    model_lower = _build_quantile_model(0.10, len(X_train))
    model_upper = _build_quantile_model(0.90, len(X_train))

    if model_lower is not None:
        model_lower.fit(X_train_s, y_train)
        model_upper.fit(X_train_s, y_train)
        logger.info("Quantile models fitted (80%% prediction interval)")
    else:
        logger.info("Quantile models skipped (too few samples for Ridge)")

    # ── Evaluate point model on test split ────────────────────────────────────
    if len(X_test_s) > 0:
        y_pred = model.predict(X_test_s)
        metrics = _compute_metrics(y_test, y_pred)
    else:
        y_pred = model.predict(X_train_s)
        metrics = _compute_metrics(y_train, y_pred)
        logger.warning("Test split was empty; metrics computed on training data.")

    params = {
        "algorithm": algo_name,
        "n_estimators": 200 if algo_name == "gradient_boosting" else None,
        "max_depth": 4 if algo_name == "gradient_boosting" else None,
        "learning_rate": 0.05 if algo_name == "gradient_boosting" else None,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "n_features": int(X.shape[1]),
        "has_intervals": model_lower is not None,
        "interval_coverage": 0.80 if model_lower is not None else None,
        "feature_names": [
            "biz_lagged", "roll5_mean", "roll5_std",
            "roll15_mean", "roll15_std",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        ],
    }

    # Persist point model, quantile models, and scaler together
    joblib.dump({
        "model":       model,
        "model_lower": model_lower,
        "model_upper": model_upper,
        "scaler":      scaler,
    }, artifact)
    logger.info("Model artifact saved to %s", artifact)

    return params, metrics, algo_name


# ── Public API ────────────────────────────────────────────────────────────────

def train_model(
    db: Session,
    config: ForecastingConfig,
    bundle: MetricsBundle,
    report: CorrelationReport,
) -> TrainedModel:
    """
    Build features from the bundle using the lag discovered by the Correlation
    Analyser, train a GradientBoosting (or Ridge) regressor, evaluate it on a
    held-out test split, persist the artifact, and update the DB record.
    """
    version = _next_version(db, config.id)
    artifact = _artifact_path(config.id, version)

    record = TrainedModel(
        config_id=config.id,
        version=version,
        algorithm="pending",
        status=ModelStatus.TRAINING,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    try:
        lag = report.best_lag()
        logger.info(
            "Training model v%d for config_id=%d  lag=%d min  n_points=%d",
            version, config.id, lag, report.n_points,
        )

        X, y = _build_features(bundle, lag)
        if len(X) < 2:
            raise ValueError(
                f"Not enough data to train: only {len(X)} feature rows after "
                "applying lag and rolling windows."
            )

        params, metrics, algo_name = _fit_and_evaluate(X, y, artifact)

        # Compute and store reference distribution for future drift detection
        biz_values = np.array([p["value"] for p in bundle.business], dtype=float)
        from app.modules.drift_detector import compute_reference_distribution
        params["input_distribution"] = compute_reference_distribution(biz_values)

        record.algorithm     = algo_name
        record.parameters    = params
        record.metrics       = metrics
        record.artifact_path = artifact
        record.lag_minutes   = lag
        record.status        = ModelStatus.READY
        record.trained_at    = datetime.utcnow()

        logger.info(
            "Model v%d ready [%s]  "
            "MAE cpu=%.2f ram=%.2f net=%.2f  "
            "R²  cpu=%.3f ram=%.3f net=%.3f",
            version, algo_name,
            metrics["mae_cpu"], metrics["mae_ram"], metrics["mae_net"],
            metrics["r2_cpu"],  metrics["r2_ram"],  metrics["r2_net"],
        )

    except Exception:
        logger.exception("Training failed for config_id=%d", config.id)
        record.status = ModelStatus.FAILED
        db.commit()
        raise

    db.commit()
    db.refresh(record)
    return record


def get_latest_ready_model(db: Session, config_id: int) -> TrainedModel | None:
    """Return the most recent READY model for a given config, or None."""
    return (
        db.query(TrainedModel)
        .filter_by(config_id=config_id, status=ModelStatus.READY)
        .order_by(TrainedModel.version.desc())
        .first()
    )
