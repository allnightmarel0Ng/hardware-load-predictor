"""
Module 4 — Model Trainer  (v3: per-target adaptive models)

Key changes from v2:
  - Per-target models: five independent models, one per system metric.
  - Per-target lag: lag is found individually for each target from
    correlation analysis on the training split only (first 80%).
  - Adaptive model selection based on r* and rel_std:
      r* ≥ 0.7  → Ridge+Poly(degree=2)  extrapolates beyond training range
      r* ≥ 0.5  → GBR                   interpolates within training range
      r* ≥ 0.3  → Ridge                 linear fallback
      r* < 0.3  → Ridge                 weak signal
      rel_std < 0.05 → mean baseline    nearly constant metric

  Why Ridge+Poly instead of XGBoost for strong correlations?
    XGBoost and GBR are tree-based models — they cannot extrapolate beyond
    the range of values seen during training. For the core use case of this
    system ("what will CPU be if RPS grows to X?"), extrapolation is
    essential. Ridge with polynomial features (degree=2) fits a smooth curve
    through the training data and extrapolates it naturally, giving physically
    meaningful predictions for unseen business metric values.
  - Advanced per-target features:
      CPU:  EWMA (α=0.1/0.3/0.7), CV, exceed_70, slope
      RAM:  release_speed, retention, steps_since_low, rel_to_30
      Net:  burst_ratio, acceleration, Hurst exponent, local_max count
      Disk: EWMA inertia, slope60, exceed_80, median_crosses
  - Artifact now stores a dict of five models + scalers keyed by target name.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.db_models import ForecastingConfig, ModelStatus, TrainedModel
from app.modules.correlation_analyzer import CorrelationReport, _resample
from app.modules.data_collector import MetricsBundle

logger = logging.getLogger(__name__)

LOOKBACK          = 30
TEST_SPLIT_RATIO  = 0.20
TRAIN_RATIO       = 1.0 - TEST_SPLIT_RATIO

# Adaptive model thresholds
CORR_LOW    = 0.3
CORR_MEDIUM = 0.5
CORR_HIGH   = 0.7
REL_STD_MIN = 0.05   # below this → metric is nearly constant → mean baseline

TARGET_KEYS = ["cpu", "ram_gb", "ram_pct", "net", "disk"]

PARAM_GRID_XGB = [
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
    {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.03},
    {"n_estimators": 700, "max_depth": 7, "learning_rate": 0.02},
]
PARAM_GRID_GBR = [
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
    {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.03},
]
PARAM_GRID_RIDGE = [
    {"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0},
]
PARAM_GRID_RIDGE_POLY = [
    {"ridge__alpha": 0.1}, {"ridge__alpha": 1.0},
    {"ridge__alpha": 10.0}, {"ridge__alpha": 100.0},
]


# ── Artifact path helpers ─────────────────────────────────────────────────────

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


# ── Correlation helpers (inline — no DB dependency) ───────────────────────────

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    with np.errstate(invalid="ignore"):
        r = np.corrcoef(a, b)[0, 1]
    return float(0.0 if np.isnan(r) else r)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x: np.ndarray) -> np.ndarray:
        s = np.argsort(x)
        r = np.empty_like(s, dtype=float)
        r[s] = np.arange(len(x)) + 1
        return r
    return _pearson(rank(a), rank(b))


def _find_lag_and_corr(
    biz_train: np.ndarray,
    sys_train: np.ndarray,
    max_lag: int = 20,
) -> tuple[int, float]:
    """
    Find optimal lag and r* on the training split only.
    Returns (best_lag_steps, r_star).
    """
    xd = np.diff(biz_train.astype(float))
    yd = np.diff(sys_train.astype(float))
    best_lag, best_score = 0, 0.0
    for lag in range(min(max_lag + 1, len(xd))):
        a, b = (xd[:-lag], yd[lag:]) if lag else (xd, yd)
        score = abs(_pearson(a, b)) + abs(_spearman(a, b))
        if score > best_score:
            best_score, best_lag = score, lag
    a, b = (xd[:-best_lag], yd[best_lag:]) if best_lag else (xd, yd)
    r = max(abs(_pearson(a, b)), abs(_spearman(a, b)))
    return best_lag, r


# ── Adaptive model selection ──────────────────────────────────────────────────

def _select_model_type(r: float, rel_std: float) -> str:
    """
    Choose algorithm based on correlation strength and metric variance.
    Returns one of: 'xgboost', 'gbr', 'ridge', 'mean_baseline'.

    xgboost = XGBoost Regressor for strong correlations (r* ≥ 0.7).
    Used for strongly correlated metrics because it can extrapolate
    beyond the training range — essential for the core use case of
    predicting system load at unseen business metric values.

    gbr = Gradient Boosting Regressor.
    Used for moderately correlated metrics where interpolation is
    sufficient and nonlinear patterns are important.
    """
    if rel_std < REL_STD_MIN:
        return "mean_baseline"
    if r >= CORR_HIGH:
        return "xgboost"
    if r >= CORR_MEDIUM:
        return "gbr"
    return "ridge"


# ── Advanced per-target feature helpers ──────────────────────────────────────

def _ewma(arr: np.ndarray, alpha: float) -> np.ndarray:
    result = np.zeros_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def _linear_slope(y: np.ndarray) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    denom = n * np.dot(x, x) - x.sum() ** 2
    if abs(denom) < 1e-12:
        return 0.0
    return float((n * np.dot(x, y) - x.sum() * y.sum()) / denom)


def _hurst_rs(series: np.ndarray) -> float:
    n = len(series)
    if n < 4:
        return 0.5
    mean = series.mean()
    cumsum = np.cumsum(series - mean)
    R = cumsum.max() - cumsum.min()
    S = series.std()
    if S < 1e-9:
        return 0.5
    return float(np.log(R / S + 1e-9) / np.log(n))


def _target_extra_features(
    target_key: str,
    hist: np.ndarray,
) -> list[float]:
    """
    Six extra features specific to each target metric.
    hist = all values up to current index (at least LOOKBACK points).
    """
    if len(hist) < LOOKBACK:
        return [0.0] * 6

    w5, w15, w30 = hist[-5:], hist[-15:], hist[-30:]

    if target_key == "cpu":
        ewma01 = float(_ewma(hist, 0.1)[-1])
        ewma03 = float(_ewma(hist, 0.3)[-1])
        ewma07 = float(_ewma(hist, 0.7)[-1])
        cv15   = float(w15.std() / (w15.mean() + 1e-9))
        exc70  = float(np.mean(w30 > 70))
        slope  = float(_linear_slope(w15))
        return [ewma01, ewma03, ewma07, cv15, exc70, slope]

    if target_key in ("ram_gb", "ram_pct"):
        diffs = np.diff(w15)
        neg   = -diffs[diffs < 0]
        release_speed = float(neg.mean()) if len(neg) > 0 else 0.0
        max30     = float(w30.max())
        retention = float(hist[-1] / (max30 + 1e-9))
        low_thr   = float(np.percentile(hist, 10))
        low_idx   = np.where(hist <= low_thr)[0]
        since_low = int(len(hist) - 1 - low_idx[-1]) if len(low_idx) > 0 else 0
        rel30     = float(hist[-1] / (w30.mean() + 1e-9))
        return [release_speed, retention, float(since_low), rel30, 0.0, 0.0]

    if target_key == "net":
        burst  = float(w15.max() / (w15.mean() + 1e-9))
        acc    = float(np.diff(w5)[-1]) if len(w5) > 1 else 0.0
        hurst  = _hurst_rs(w30)
        lmax   = sum(
            1 for j in range(1, len(w30) - 1)
            if w30[j] > w30[j - 1] and w30[j] > w30[j + 1]
        )
        return [burst, acc, hurst, float(lmax), 0.0, 0.0]

    # disk
    ewma_slow = float(_ewma(hist, 0.05)[-1])
    ewma_fast = float(_ewma(hist, 0.5)[-1])
    inertia   = ewma_slow - ewma_fast
    h60       = hist[-60:] if len(hist) >= 60 else hist
    slope60   = float(_linear_slope(h60))
    exc80     = float(np.mean(h60 > 80))
    med30     = float(np.median(w30))
    crosses   = sum(
        1 for j in range(1, len(w30))
        if (w30[j - 1] - med30) * (w30[j] - med30) < 0
    )
    return [inertia, slope60, exc80, float(crosses), 0.0, 0.0]


# ── Feature engineering (per target) ─────────────────────────────────────────

def _build_features_for_target(
    biz: np.ndarray,
    sys_arrays: dict[str, np.ndarray],
    tau: int,
    target_key: str,
    n: int,
    max_biz_train: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix X and target vector y for one system metric.
    X shape: (n - LOOKBACK - 1, n_features)

    max_biz_train: maximum business metric value seen in training data.
        Used to compute biz_above_max = max(0, biz[i] - max_biz_train),
        which gives XGBoost a linear extrapolation signal beyond the
        training range. At inference with unseen high RPS, this feature
        is nonzero and lets the model extrapolate proportionally.
    """
    target_arr = sys_arrays[target_key]
    rows_X: list[list[float]] = []
    rows_y: list[float] = []

    ts_base = datetime(2000, 1, 1)   # dummy base; only hour/dow matter

    for i in range(LOOKBACK, n - 1):
        # ── Time features ──────────────────────────────────────────────────
        hour   = (i / 60.0) % 24        # assume 1-min steps; adjust for STEP
        dow    = (i / (24 * 60.0)) % 7
        sin_h  = math.sin(2 * math.pi * hour / 24)
        cos_h  = math.cos(2 * math.pi * hour / 24)
        sin_d  = math.sin(2 * math.pi * dow  / 7)
        cos_d  = math.cos(2 * math.pi * dow  / 7)
        trend  = i / n

        # ── Business features ──────────────────────────────────────────────
        bl     = biz[i - tau] if i >= tau else biz[0]
        rm30   = biz[max(0, i - 30):i].mean()
        bn     = bl / (rm30 + 1e-9)
        d1     = biz[i] - biz[i - 1] if i >= 1 else 0.0
        d2     = biz[i - 1] - biz[i - 2] if i >= 2 else 0.0
        mu5_b  = biz[max(0, i - 5):i].mean()
        bz     = (biz[i] - mu5_b) / (biz[max(0, i - 5):i].std() + 1e-9)

        # biz_above_max: absolute excess above training maximum (0 during training).
        # biz_relative:  ratio biz/max_train (≤1 during training, >1 at inference
        #                with unseen high RPS). Both give XGBoost linear extrapolation
        #                handles beyond the training range.
        biz_above_max  = max(0.0, bl - max_biz_train) if max_biz_train is not None else 0.0
        biz_relative   = bl / (max_biz_train + 1e-9)  if max_biz_train is not None else 1.0

        feats: list[float] = [
            sin_h, cos_h, sin_d, cos_d, trend,
            bl, bn, d1, d2, bz, bl * sin_h, bl * cos_h,
            biz_above_max, biz_relative,
        ]

        # ── Autoregressive system features (all four AR metrics) ───────────
        for key in ("cpu", "ram_pct", "net", "disk"):
            arr = sys_arrays[key]
            l1  = arr[i - 1]
            l2  = arr[i - 2] if i >= 2 else arr[0]
            l3  = arr[i - 3] if i >= 3 else arr[0]
            m5  = arr[max(0, i - 5):i].mean()
            m15 = arr[max(0, i - 15):i].mean()
            m30 = arr[max(0, i - 30):i].mean()
            s5  = arr[max(0, i - 5):i].std()  + 1e-9
            s15 = arr[max(0, i - 15):i].std() + 1e-9
            feats.extend([l1, l2, l3, m5, m15, m30, s5, s15])

        # ── Target-specific extra features ─────────────────────────────────
        feats.extend(_target_extra_features(target_key, target_arr[:i]))

        rows_X.append(feats)
        rows_y.append(float(target_arr[i]))

    X = np.array(rows_X, dtype=float)
    y = np.array(rows_y, dtype=float)
    return X, y


# ── Model building & temporal CV ─────────────────────────────────────────────

def _make_estimator(model_type: str, params: dict):
    if model_type == "xgboost":
        return XGBRegressor(**params, random_state=42,
                            subsample=0.8, colsample_bytree=0.8, verbosity=0)
    if model_type == "gbr":
        return GradientBoostingRegressor(**params, random_state=42,
                                         subsample=0.8, min_samples_leaf=5)
    return Ridge(**params)


def _compute_sample_weights(biz: np.ndarray) -> np.ndarray:
    """
    Compute sample weights based on business metric value.

    Points where traffic is near zero (< 5% of max) get weight 0.05 —
    they still influence the model slightly but don't dominate.
    Points with real traffic get weight 1.0.

    This handles the "traffic on / traffic off" pattern without
    harming sinusoidal or other continuous patterns: even the troughs
    of a sine wave are typically well above 5% of the peak.
    """
    threshold = biz.max() * 0.05
    return np.where(biz > threshold, 1.0, 0.05)


def _fit_kwargs(model_type: str, weights: np.ndarray | None) -> dict:
    """
    Return the correct keyword arguments for model.fit() with sample weights.
    All estimators use 'sample_weight' directly.
    """
    if weights is None:
        return {}
    return {"sample_weight": weights}


def _train_with_cv(
    X_tv: np.ndarray,
    y_tv: np.ndarray,
    model_type: str,
    weights_tv: np.ndarray | None = None,
) -> object:
    """Temporal CV on train+val, returns best fitted model.

    weights_tv: sample weights aligned with X_tv/y_tv.
                Passed to model.fit() so the algorithm focuses on
                high-traffic points and down-weights idle periods.
    """
    if model_type == "mean_baseline":
        return None   # handled separately

    grid = {
        "xgboost":    PARAM_GRID_XGB,
        "gbr":        PARAM_GRID_GBR,
        "ridge":      PARAM_GRID_RIDGE,
    }.get(model_type, PARAM_GRID_RIDGE)

    tscv = TimeSeriesSplit(n_splits=3)
    best_score, best_params = -np.inf, grid[0]

    for params in grid:
        m = _make_estimator(model_type, params)
        scores = []
        for tr_idx, val_idx in tscv.split(X_tv):
            w_tr = weights_tv[tr_idx] if weights_tv is not None else None
            fit_kwargs = _fit_kwargs(model_type, w_tr)
            m.fit(X_tv[tr_idx], y_tv[tr_idx], **fit_kwargs)
            p = m.predict(X_tv[val_idx])
            scores.append(r2_score(y_tv[val_idx], p))
        if np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_params = params

    final = _make_estimator(model_type, best_params)
    final.fit(X_tv, y_tv, **_fit_kwargs(model_type, weights_tv))
    return final


# ── Per-target training pipeline ─────────────────────────────────────────────

def _train_single_target(
    biz: np.ndarray,
    sys_arrays: dict[str, np.ndarray],
    target_key: str,
    n: int,
    best_step_seconds: int = 60,
    base_step_seconds: int = 60,
) -> dict:
    """
    Train one model for one target metric.
    Returns a dict with model, scaler, lag, r_star, model_type, metrics.

    best_step_seconds: resolution chosen by step CV in correlation_analyzer.
                       Data is resampled to this resolution before lag detection,
                       matching the analysis performed during correlation analysis.
    base_step_seconds: raw collection resolution (seconds).
    """
    # Resample to the step chosen by CV — same resolution used in correlation analysis
    factor = max(1, best_step_seconds // base_step_seconds)
    biz_rs = _resample(biz, factor)
    sys_rs = {k: _resample(v, factor) for k, v in sys_arrays.items()}
    n_rs   = len(biz_rs)

    # Filter out simultaneous idle points — remove rows where both biz AND
    # target sys metric are below 10% of their respective maximum.
    # Using max-based threshold instead of percentile: with 80% idle data,
    # the 10th percentile would be near zero and filter almost nothing.
    # max-based threshold is robust regardless of idle/active ratio.
    biz_threshold = float(biz_rs.max()) * 0.10
    sys_threshold = float(sys_rs[target_key].max()) * 0.10
    active_mask = (biz_rs > biz_threshold) & (sys_rs[target_key] > sys_threshold)
    biz_active = biz_rs[active_mask]
    sys_rs_active = {k: v[active_mask] for k, v in sys_rs.items()}
    n_active = int(active_mask.sum())
    idle_removed = n_rs - n_active
    if idle_removed > 0:
        logger.info(
            "  %s: removed %d/%d points (biz≤%.2f OR sys≤%.2f — filtered non-correlated)",
            target_key, idle_removed, n_rs, biz_p10, sys_p10,
        )

    # Use active data for training; fall back to full if too few active points
    if n_active >= LOOKBACK + 20:
        biz_fit  = biz_active
        sys_fit  = sys_rs_active
        n_fit    = n_active
    else:
        logger.warning("  %s: too few active points (%d), using full dataset", target_key, n_active)
        biz_fit  = biz_rs
        sys_fit  = sys_rs
        n_fit    = n_rs

    # Store max business metric seen in training data — used as biz_above_max feature
    max_biz_train = float(biz_fit.max())

    # Split index for lag/corr detection (training data only — resampled)
    split     = int(TRAIN_RATIO * n_fit)
    biz_train = biz_fit[:split]
    sys_train = sys_fit[target_key][:split]

    lag, r_star = _find_lag_and_corr(biz_train, sys_train)

    mean_val = sys_fit[target_key].mean()
    std_val  = sys_fit[target_key].std()
    rel_std  = std_val / (mean_val + 1e-9)

    model_type = _select_model_type(r_star, rel_std)

    logger.info(
        "  %s: lag=%d  r*=%.3f  rel_std=%.3f  step=%ds  max_biz=%.2f  → %s",
        target_key, lag, r_star, rel_std, best_step_seconds, max_biz_train, model_type,
    )

    # Build features on active (filtered) data, passing max_biz_train for biz_above_max
    X, y = _build_features_for_target(biz_fit, sys_fit, lag, target_key, n_fit,
                                       max_biz_train=max_biz_train)

    # Temporal split: 80% train+val, 20% test
    n_rows = len(X)
    te     = max(1, int(TEST_SPLIT_RATIO * n_rows))
    X_tv, X_te = X[:-te], X[-te:]
    y_tv, y_te = y[:-te], y[-te:]

    # Sample weights — align with feature rows (LOOKBACK rows dropped at start)
    biz_for_weights = biz_fit[LOOKBACK:n_fit - 1]
    all_weights     = _compute_sample_weights(biz_for_weights)
    w_tv = all_weights[:-te]
    idle_frac = float(np.mean(all_weights < 0.5))
    logger.info(
        "  %s: idle_fraction=%.1f%%  (weight=0.05 for biz < 5%% of max=%.1f)",
        target_key, idle_frac * 100, biz_rs.max(),
    )

    scaler    = StandardScaler()
    X_tv_s    = scaler.fit_transform(X_tv)
    X_te_s    = scaler.transform(X_te)

    if model_type == "mean_baseline":
        mean_pred = float(y_tv.mean())
        y_pred    = np.full(len(y_te), mean_pred)
        model_obj = None
    else:
        model_obj = _train_with_cv(X_tv_s, y_tv, model_type, weights_tv=w_tv)
        y_pred    = model_obj.predict(X_te_s)

    # Metrics
    r2  = float(r2_score(y_te, y_pred))
    mae = float(mean_absolute_error(y_te, y_pred))
    rmse= float(np.sqrt(mean_squared_error(y_te, y_pred)))
    mask = np.abs(y_te) > 1e-9
    mape = float(np.mean(np.abs((y_te[mask] - y_pred[mask]) / y_te[mask])) * 100) \
           if mask.any() else 0.0

    return {
        "model":         model_obj,
        "scaler":        scaler,
        "lag":           lag,
        "r_star":        round(r_star, 4),
        "rel_std":       round(rel_std, 4),
        "model_type":    model_type,
        "mean_val":      float(y_tv.mean()),   # for mean_baseline fallback
        "max_biz_train": max_biz_train,        # for biz_above_max at inference
        "metrics": {
            f"r2_{target_key}":   round(r2, 4),
            f"mae_{target_key}":  round(mae, 4),
            f"rmse_{target_key}": round(rmse, 4),
            f"mape_{target_key}": round(mape, 4),
        },
    }


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _compute_overall_metrics(per_target: dict[str, dict]) -> dict:
    """Aggregate per-target metrics into a flat dict + mape_overall."""
    metrics: dict = {}
    mapes = []
    for key, info in per_target.items():
        metrics.update(info["metrics"])
        mapes.append(info["metrics"].get(f"mape_{key}", 0.0))
    metrics["mape_overall"] = round(float(np.mean(mapes)), 4)
    return metrics


# ── Public API ────────────────────────────────────────────────────────────────

def train_model(
    db: Session,
    config: ForecastingConfig,
    bundle: MetricsBundle,
    report: CorrelationReport,      # kept for API compatibility; lags re-derived per target
) -> TrainedModel:
    """
    Train five independent models (one per system metric target) with
    per-target lag detection on the training split and adaptive model selection.
    """
    version  = _next_version(db, config.id)
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
        # Extract arrays from bundle
        biz = np.array([p["value"] for p in bundle.business], dtype=float)
        sys_arrays = {
            "cpu":     np.array([p["value"] for p in bundle.cpu],         dtype=float),
            "ram_gb":  np.array([p["value"] for p in bundle.ram_gb],      dtype=float),
            "ram_pct": np.array([p["value"] for p in bundle.ram_percent], dtype=float),
            "net":     np.array([p["value"] for p in bundle.network],     dtype=float),
            "disk":    np.array([p["value"] for p in bundle.disk],        dtype=float),
        }
        n = len(biz)

        logger.info(
            "Training v%d for config_id=%d  n_points=%d  per-target adaptive models",
            version, config.id, n,
        )

        if n < LOOKBACK + 10:
            raise ValueError(
                f"Not enough data: {n} points (need ≥ {LOOKBACK + 10})."
            )

        # best_step_seconds from CV in correlation_analyzer — use same resolution
        best_step = getattr(report, "best_step_seconds", 60)
        base_step = getattr(report, "_base_step_seconds", 60)

        logger.info(
            "Using best_step=%ds from correlation CV for lag detection",
            best_step,
        )

        # Train one model per target
        per_target: dict[str, dict] = {}
        for key in TARGET_KEYS:
            per_target[key] = _train_single_target(
                biz, sys_arrays, key, n,
                best_step_seconds=best_step,
                base_step_seconds=base_step,
            )

        # Aggregate metrics
        metrics = _compute_overall_metrics(per_target)

        # Reference distribution for drift detection
        from app.modules.drift_detector import compute_reference_distribution
        ref_dist = compute_reference_distribution(biz)

        # Build params summary
        params = {
            "feature_set":        "advanced_per_target_v3",
            "lookback":           LOOKBACK,
            "train_ratio":        TRAIN_RATIO,
            "best_step_seconds":  best_step,
            "per_target": {
                key: {
                    "lag":        info["lag"],
                    "r_star":     info["r_star"],
                    "rel_std":    info["rel_std"],
                    "model_type": info["model_type"],
                }
                for key, info in per_target.items()
            },
            "input_distribution": ref_dist,
        }

        # Determine dominant algorithm for display
        types = [info["model_type"] for info in per_target.values()]
        dominant = max(set(types), key=types.count)
        algo_name = dominant

        # Persist artifact: dict of per-target dicts
        artifact_payload = {
            "per_target": per_target,
            "version":    "v3_per_target",
        }
        joblib.dump(artifact_payload, artifact)
        logger.info("Artifact saved → %s", artifact)

        # Update DB record
        record.algorithm     = algo_name
        record.parameters    = params
        record.metrics       = metrics
        record.artifact_path = artifact
        record.lag_minutes   = per_target["cpu"]["lag"]   # primary lag for display
        record.status        = ModelStatus.READY
        record.trained_at    = datetime.utcnow()

        logger.info(
            "Model v%d ready  R²: cpu=%.3f ram_gb=%.3f ram_pct=%.3f "
            "net=%.3f disk=%.3f  MAPE=%.1f%%",
            version,
            metrics.get("r2_cpu",    0), metrics.get("r2_ram_gb",  0),
            metrics.get("r2_ram_pct",0), metrics.get("r2_net",     0),
            metrics.get("r2_disk",   0), metrics.get("mape_overall", 0),
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
    return (
        db.query(TrainedModel)
        .filter_by(config_id=config_id, status=ModelStatus.READY)
        .order_by(TrainedModel.version.desc())
        .first()
    )
