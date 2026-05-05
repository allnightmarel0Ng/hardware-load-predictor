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
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.db_models import ForecastingConfig, ModelStatus, TrainedModel
from app.modules.correlation_analyzer import CorrelationReport, _resample
from app.modules.data_collector import MetricsBundle

logger = logging.getLogger(__name__)

LOOKBACK          = 30
N_BIZ_FEATURES    = 14
BIZ_ONLY_IDXS     = [0,1,2,3,4,5,6,7,8,9,12,13]
TEST_SPLIT_RATIO  = 0.20
TRAIN_RATIO       = 1.0 - TEST_SPLIT_RATIO

CORR_LOW    = 0.3
CORR_MEDIUM = 0.5
CORR_HIGH   = 0.7
REL_STD_MIN = 0.05
REL_STD_MAX = 2.0

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


def _select_model_type(r: float, rel_std: float) -> str:
    if rel_std < REL_STD_MIN:
        return "mean_baseline"
    if rel_std > REL_STD_MAX:
        return "mean_baseline"
    if r < CORR_LOW:
        return "mean_baseline"
    if r >= CORR_HIGH:
        return "xgboost"
    if r >= CORR_MEDIUM:
        return "gbr"
    return "ridge"


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


def _build_features_for_target(
    biz: np.ndarray,
    sys_arrays: dict[str, np.ndarray],
    tau: int,
    target_key: str,
    n: int,
    max_biz_train: float | None = None,
    step_seconds: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    target_arr = sys_arrays[target_key]
    rows_X: list[list[float]] = []
    rows_y: list[float] = []

    ts_base = datetime(2000, 1, 1)

    for i in range(LOOKBACK, n - 1):
        elapsed_min = i * step_seconds / 60.0
        hour   = elapsed_min % (24 * 60) / 60.0
        dow    = (elapsed_min / (24 * 60.0)) % 7
        sin_h  = math.sin(2 * math.pi * hour / 24)
        cos_h  = math.cos(2 * math.pi * hour / 24)
        sin_d  = math.sin(2 * math.pi * dow  / 7)
        cos_d  = math.cos(2 * math.pi * dow  / 7)
        trend  = i / n

        bl     = biz[i - tau] if i >= tau else biz[0]
        rm30   = biz[max(0, i - 30):i].mean()
        bn     = bl / (rm30 + 1e-9)
        d1     = biz[i] - biz[i - 1] if i >= 1 else 0.0
        d2     = biz[i - 1] - biz[i - 2] if i >= 2 else 0.0
        mu5_b  = biz[max(0, i - 5):i].mean()
        bz     = (biz[i] - mu5_b) / (biz[max(0, i - 5):i].std() + 1e-9)

        biz_above_max  = max(0.0, bl - max_biz_train) if max_biz_train is not None else 0.0
        biz_relative   = bl / (max_biz_train + 1e-9)  if max_biz_train is not None else 1.0

        feats: list[float] = [
            sin_h, cos_h, sin_d, cos_d, trend,
            bl, bn, d1, d2, bz, bl * sin_h, bl * cos_h,
            biz_above_max, biz_relative,
        ]

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

        feats.extend(_target_extra_features(target_key, target_arr[:i]))

        rows_X.append(feats)
        rows_y.append(float(target_arr[i]))

    X = np.array(rows_X, dtype=float)
    y = np.array(rows_y, dtype=float)
    return X, y


def _make_estimator(model_type: str, params: dict):
    if model_type == "xgboost":
        return XGBRegressor(**params, random_state=42,
                            subsample=0.8, colsample_bytree=0.8, verbosity=0)
    if model_type == "gbr":
        return GradientBoostingRegressor(**params, random_state=42,
                                         subsample=0.8, min_samples_leaf=5)
    return Ridge(**params)


def _fit_extrapolation_ridge(
    X_tv_s: np.ndarray,
    y_tv: np.ndarray,
    biz_values: np.ndarray,
    max_biz_train: float,
) -> dict:
    n = len(biz_values)
    fallback = {
        "biz_only_ridge":  None,
        "biz_only_scaler": None,
        "extrap_ridge":    None,
        "boundary_pred":   float(y_tv.mean()),
    }
    if n < 20 or max_biz_train <= 0:
        return fallback

    X_biz_s = X_tv_s[:, BIZ_ONLY_IDXS]
    biz_only_ridge = Ridge(alpha=1.0)
    biz_only_ridge.fit(X_biz_s, y_tv)

    extrap_ridge = Ridge(alpha=1.0)
    extrap_ridge.fit(X_tv_s, y_tv)

    threshold = np.percentile(biz_values, 70)
    mask = biz_values >= threshold
    boundary_pred = float(y_tv[mask].mean()) if mask.sum() >= 5 else float(y_tv.mean())

    logger.info(
        "  extrap Ridge: biz_only(raw) + full fitted  boundary_pred=%.2f  n_top=%d/%d",
        boundary_pred, int(mask.sum()), n,
    )
    return {
        "biz_only_ridge":  biz_only_ridge,
        "extrap_ridge":    extrap_ridge,
        "boundary_pred":   boundary_pred,
    }


def _compute_sample_weights(biz: np.ndarray) -> np.ndarray:
    threshold = biz.max() * 0.05
    return np.where(biz > threshold, 1.0, 0.05)


def _fit_kwargs(model_type: str, weights: np.ndarray | None) -> dict:
    if weights is None:
        return {}
    return {"sample_weight": weights}


def _train_with_cv(
    X_tv: np.ndarray,
    y_tv: np.ndarray,
    model_type: str,
    weights_tv: np.ndarray | None = None,
) -> object:
    if model_type == "mean_baseline":
        return None

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


def _train_single_target(
    biz: np.ndarray,
    sys_arrays: dict[str, np.ndarray],
    target_key: str,
    n: int,
    best_step_seconds: int = 60,
    base_step_seconds: int = 60,
) -> dict:
    factor = max(1, best_step_seconds // base_step_seconds)
    biz_rs = _resample(biz, factor)
    sys_rs = {k: _resample(v, factor) for k, v in sys_arrays.items()}
    n_rs   = len(biz_rs)

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
            target_key, idle_removed, n_rs, biz_threshold, sys_threshold,
        )

    if n_active >= LOOKBACK + 20:
        biz_fit  = biz_active
        sys_fit  = sys_rs_active
        n_fit    = n_active
    else:
        logger.warning("  %s: too few active points (%d), using full dataset", target_key, n_active)
        biz_fit  = biz_rs
        sys_fit  = sys_rs
        n_fit    = n_rs

    max_biz_train = float(biz_fit.max())

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

    X, y = _build_features_for_target(biz_fit, sys_fit, lag, target_key, n_fit,
                                       max_biz_train=max_biz_train,
                                       step_seconds=best_step_seconds)

    n_rows = len(X)
    te     = max(1, int(TEST_SPLIT_RATIO * n_rows))
    X_tv, X_te = X[:-te], X[-te:]
    y_tv, y_te = y[:-te], y[-te:]

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

    biz_for_extrap = biz_fit[LOOKBACK:n_fit - 1][:-te]
    if model_type == "mean_baseline" or model_obj is None:
        extrap = {
            "extrap_ridge":  None,
            "extrap_scaler": scaler,
            "boundary_pred": float(y_tv.mean()),
        }
    else:
        extrap = _fit_extrapolation_ridge(
            X_tv_s, y_tv, biz_for_extrap, max_biz_train
        )

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
        "mean_sys_ctx": {
            k: float(arr.mean()) for k, arr in sys_fit.items()
        },
        "biz_only_ridge":       extrap.get("biz_only_ridge"),
        "biz_only_scaler":      extrap.get("biz_only_scaler"),
        "extrap_ridge":         extrap.get("extrap_ridge"),
        "extrap_boundary_pred": extrap["boundary_pred"],
        "metrics": {
            f"r2_{target_key}":   round(r2, 4),
            f"mae_{target_key}":  round(mae, 4),
            f"rmse_{target_key}": round(rmse, 4),
            f"mape_{target_key}": round(mape, 4),
        },
    }


def _compute_overall_metrics(per_target: dict[str, dict]) -> dict:
    metrics: dict = {}
    mapes = []
    for key, info in per_target.items():
        metrics.update(info["metrics"])
        mapes.append(info["metrics"].get(f"mape_{key}", 0.0))
    metrics["mape_overall"] = round(float(np.mean(mapes)), 4)
    return metrics


def train_model(
    db: Session,
    config: ForecastingConfig,
    bundle: MetricsBundle,
    report: CorrelationReport,
) -> TrainedModel:
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

        best_step = getattr(report, "best_step_seconds", 60)
        base_step = getattr(report, "_base_step_seconds", 60)

        logger.info(
            "Using best_step=%ds from correlation CV for lag detection",
            best_step,
        )

        per_target: dict[str, dict] = {}
        for key in TARGET_KEYS:
            per_target[key] = _train_single_target(
                biz, sys_arrays, key, n,
                best_step_seconds=best_step,
                base_step_seconds=base_step,
            )

        metrics = _compute_overall_metrics(per_target)

        from app.modules.drift_detector import compute_reference_distribution
        ref_dist = compute_reference_distribution(biz)

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

        types = [info["model_type"] for info in per_target.values()]
        dominant = max(set(types), key=types.count)
        algo_name = dominant

        artifact_payload = {
            "per_target": per_target,
            "version":    "v3_per_target",
        }
        joblib.dump(artifact_payload, artifact)
        logger.info("Artifact saved → %s", artifact)

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
