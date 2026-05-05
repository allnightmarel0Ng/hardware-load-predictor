from __future__ import annotations

import argparse
import math
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")


STEP_SECONDS = 300
DAYS         = 8
N            = DAYS * 24 * 3600 // STEP_SECONDS
TRUE_LAG     = 1
LOOKBACK     = 30

rng = np.random.default_rng(42)
t   = np.arange(N)

def generate_system_metrics(biz: np.ndarray, lag: int = TRUE_LAG) -> dict:
    n = len(biz)
    driven = np.empty(n)
    driven[:n - lag] = biz[lag:]
    driven[n - lag:] = biz[-1]
    norm = (driven - driven.min()) / (driven.max() - driven.min() + 1e-9)

    cpu  = np.clip(20 + 55 * norm + 6 * np.sin(2*np.pi*t/(24*3600/STEP_SECONDS))
                   + rng.normal(0, 3, n), 1, 99)
    kernel   = np.ones(30) / 30
    ram_pct  = np.clip(np.convolve(cpu, kernel, "same") * 0.6 + 35
                       + rng.normal(0, 2, n), 10, 95)
    ram_gb   = ram_pct / 100 * 32
    net      = np.clip(5 + 80 * norm + rng.normal(0, 4, n), 0, 200)
    disk     = np.clip(8 + 25 * norm * (rng.random(n) > 0.5)
                       + rng.normal(0, 5, n), 0, 80)

    return {"cpu": cpu, "ram_pct": ram_pct, "ram_gb": ram_gb,
            "net": net, "disk": disk}

def pattern_sinusoidal() -> np.ndarray:
    daily  = 50 + 40 * np.sin(2 * np.pi * (t / (24*3600/STEP_SECONDS) - 0.25))
    weekly = 1 + 0.15 * np.sin(2 * np.pi * t / (7 * 24 * 3600 / STEP_SECONDS))
    return np.clip(daily * weekly + rng.normal(0, 3, N), 1, 200)

def pattern_step() -> np.ndarray:
    levels    = [15, 45, 80, 110, 45, 15, 80, 45, 110, 15]
    durations = [rng.integers(20, 60) for _ in levels]
    biz = np.zeros(N)
    idx = 0
    for lvl, dur in zip(levels, durations):
        end = min(idx + dur, N)
        biz[idx:end] = lvl + rng.normal(0, 2, end - idx)
        idx = end
        if idx >= N:
            break
    if idx < N:
        biz[idx:] = 45 + rng.normal(0, 2, N - idx)
    return np.clip(biz, 1, 200)

def pattern_spiky() -> np.ndarray:
    biz = 10 + rng.exponential(2, N)
    for _ in range(40):
        c = rng.integers(0, N)
        w = rng.integers(3, 12)
        biz += rng.uniform(40, 100) * np.exp(-0.5 * ((t - c) / w) ** 2)
    return np.clip(biz, 1, 250)

def pattern_drifting() -> np.ndarray:
    trend = np.linspace(20, 100, N)
    amp   = np.linspace(10, 40, N)
    return np.clip(trend + amp * np.sin(2*np.pi*(t/(24*3600/STEP_SECONDS)-0.25))
                   + rng.normal(0, 3, N), 1, 200)

def pattern_multimodal() -> np.ndarray:
    dow      = (t // (24 * 3600 // STEP_SECONDS)) % 7
    weekend  = (dow >= 5).astype(float)
    weekday  = (60 + 35 * np.sin(2*np.pi*(t/(24*3600/STEP_SECONDS)-0.25))
                + rng.normal(0, 4, N))
    wkend    = 15 + 8 * np.sin(2*np.pi*t/(24*3600/STEP_SECONDS)) + rng.normal(0, 2, N)
    return np.clip((1 - weekend) * weekday + weekend * wkend, 1, 200)

PATTERNS = {
    "sinusoidal": ("Sinusoidal (SaaS daily wave)",       pattern_sinusoidal),
    "step":       ("Step function (batch / ETL)",        pattern_step),
    "spiky":      ("Spiky bursts (event-driven)",        pattern_spiky),
    "drifting":   ("Drifting trend (growing audience)",  pattern_drifting),
    "multimodal": ("Multimodal (weekday/weekend split)", pattern_multimodal),
}


def train_prophet(biz_train: np.ndarray, sys_train: np.ndarray,
                  biz_test: np.ndarray) -> np.ndarray:
    base_ts = datetime(2024, 1, 1)
    n_tr = len(biz_train)
    n_te = len(biz_test)

    df_train = pd.DataFrame({
        "ds": [base_ts + timedelta(seconds=i * STEP_SECONDS) for i in range(n_tr)],
        "y":  sys_train.astype(float),
        "biz": biz_train.astype(float),
    })

    df_future = pd.DataFrame({
        "ds":  [base_ts + timedelta(seconds=i * STEP_SECONDS)
                for i in range(n_tr, n_tr + n_te)],
        "biz": biz_test.astype(float),
    })

    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
    )
    m.add_regressor("biz")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df_train)

    forecast = m.predict(df_future)
    return forecast["yhat"].values


def build_features_xgb(biz: np.ndarray, sys_arr: np.ndarray,
                        tau: int, max_biz_train: float) -> tuple:
    rows_X, rows_y = [], []
    for i in range(LOOKBACK, len(biz) - 1):
        bl   = biz[i - tau] if i >= tau else biz[0]
        rm30 = biz[max(0, i-30):i].mean()
        bn   = bl / (rm30 + 1e-9)
        d1   = biz[i] - biz[i-1] if i >= 1 else 0.0
        d2   = biz[i-1] - biz[i-2] if i >= 2 else 0.0
        mu5  = biz[max(0, i-5):i].mean()
        bz   = (biz[i] - mu5) / (biz[max(0, i-5):i].std() + 1e-9)
        h    = (i * STEP_SECONDS / 3600) % 24
        d    = (i * STEP_SECONDS / 86400) % 7

        # Extrapolation features
        biz_above = max(0.0, bl - max_biz_train)
        biz_rel   = bl / (max_biz_train + 1e-9)

        rows_X.append([
            bl, bn, rm30, d1, d2, bz,
            math.sin(2*math.pi*h/24), math.cos(2*math.pi*h/24),
            math.sin(2*math.pi*d/7),  math.cos(2*math.pi*d/7),
            i / len(biz),  # trend
            biz_above, biz_rel,
        ])
        rows_y.append(float(sys_arr[i]))
    return np.array(rows_X), np.array(rows_y)


def train_xgb(biz: np.ndarray, sys_arr: np.ndarray,
              tau: int, split: int) -> tuple:
    max_biz_train = float(biz[:split].max())
    X, y = build_features_xgb(biz, sys_arr, tau, max_biz_train)

    feature_split = split - LOOKBACK - 1
    feature_split = max(1, min(feature_split, len(X) - 1))

    X_tr, X_te = X[:feature_split], X[feature_split:]
    y_tr, y_te = y[:feature_split], y[feature_split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    if HAS_XGB:
        model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, verbosity=0)
    else:
        model = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                          learning_rate=0.05, subsample=0.8,
                                          random_state=42)
    model.fit(X_tr_s, y_tr)
    pred = model.predict(X_te_s)
    return pred, y_te


def train_hybrid(biz: np.ndarray, sys_arr: np.ndarray,
                 tau: int, split: int) -> tuple:
    base_ts   = datetime(2024, 1, 1)
    n_tr      = split
    n_te      = len(biz) - split
    max_biz_train = float(biz[:split].max())

    biz_lag = np.roll(biz, tau)
    biz_lag[:tau] = biz[0]

    # ── Step 1: fit Prophet on training data ─────────────────────────────────
    df_train = pd.DataFrame({
        "ds":  [base_ts + timedelta(seconds=i * STEP_SECONDS) for i in range(n_tr)],
        "y":   sys_arr[:n_tr].astype(float),
        "biz": biz_lag[:n_tr].astype(float),
    })
    df_all = pd.DataFrame({
        "ds":  [base_ts + timedelta(seconds=i * STEP_SECONDS) for i in range(len(biz))],
        "biz": biz_lag.astype(float),
    })

    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
    )
    m.add_regressor("biz")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(df_train)

    forecast_all = m.predict(df_all)
    prophet_all  = forecast_all["yhat"].values

    residuals_train = sys_arr[:n_tr] - prophet_all[:n_tr]

    X, _ = build_features_xgb(biz, sys_arr, tau, max_biz_train)
    res_full = sys_arr - prophet_all
    rows_y_res = [float(res_full[i]) for i in range(LOOKBACK, len(biz) - 1)]
    y_res = np.array(rows_y_res)

    feature_split = split - LOOKBACK - 1
    feature_split = max(1, min(feature_split, len(X) - 1))

    X_tr, X_te   = X[:feature_split], X[feature_split:]
    y_res_tr      = y_res[:feature_split]
    y_te_actual   = sys_arr[split:]

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    if HAS_XGB:
        xgb_model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, verbosity=0)
    else:
        xgb_model = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                               learning_rate=0.05, subsample=0.8,
                                               random_state=42)
    xgb_model.fit(X_tr_s, y_res_tr)
    xgb_residual_pred = xgb_model.predict(X_te_s)

    prophet_te = prophet_all[split: split + len(xgb_residual_pred)]
    final_pred = prophet_te + xgb_residual_pred

    min_len = min(len(final_pred), len(y_te_actual))
    return final_pred[:min_len], y_te_actual[:min_len]


def train_ridge(biz: np.ndarray, sys_arr: np.ndarray,
                tau: int, split: int) -> tuple:
    max_biz_train = float(biz[:split].max())
    X, y = build_features_xgb(biz, sys_arr, tau, max_biz_train)

    feature_split = split - LOOKBACK - 1
    feature_split = max(1, min(feature_split, len(X) - 1))

    X_tr, X_te = X[:feature_split], X[feature_split:]
    y_tr, y_te = y[:feature_split], y[feature_split:]

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    model = Ridge(alpha=1.0)
    model.fit(X_tr_s, y_tr)
    pred = model.predict(X_te_s)
    return pred, y_te

def train_combined(biz: np.ndarray, sys_arr: np.ndarray,
                   tau: int, split: int) -> tuple:
    max_biz_train = float(biz[:split].max())
    X, y = build_features_xgb(biz, sys_arr, tau, max_biz_train)

    feature_split = split - LOOKBACK - 1
    feature_split = max(1, min(feature_split, len(X) - 1))

    X_tr, X_te = X[:feature_split], X[feature_split:]
    y_tr, y_te = y[:feature_split], y[feature_split:]

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    if HAS_XGB:
        xgb_model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, verbosity=0)
    else:
        xgb_model = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                               learning_rate=0.05, subsample=0.8,
                                               random_state=42)
    ridge_model = Ridge(alpha=1.0)

    xgb_model.fit(X_tr_s, y_tr)
    ridge_model.fit(X_tr_s, y_tr)

    xgb_pred   = xgb_model.predict(X_te_s)
    ridge_pred = ridge_model.predict(X_te_s)

    biz_lag = biz.copy()
    if tau > 0:
        biz_lag = np.roll(biz, tau)
        biz_lag[:tau] = biz[0]

    test_indices = range(LOOKBACK + feature_split, LOOKBACK + feature_split + len(X_te))
    alphas = np.zeros(len(X_te))
    for i, idx in enumerate(test_indices):
        if idx < len(biz_lag):
            bl = biz_lag[idx]
            biz_above = max(0.0, bl - max_biz_train)
            alphas[i] = min(1.0, biz_above / (max_biz_train + 1e-9))

    combined_pred = (1.0 - alphas) * xgb_pred + alphas * ridge_pred

    return combined_pred, y_te


def train_lgbm_linear(biz: np.ndarray, sys_arr: np.ndarray,
                      tau: int, split: int) -> tuple:
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("pip install lightgbm>=3.2.0")

    max_biz_train = float(biz[:split].max())
    X, y = build_features_xgb(biz, sys_arr, tau, max_biz_train)

    feature_split = split - LOOKBACK - 1
    feature_split = max(1, min(feature_split, len(X) - 1))

    X_tr, X_te = X[:feature_split], X[feature_split:]
    y_tr, y_te = y[:feature_split], y[feature_split:]

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    train_data = lgb.Dataset(X_tr_s, label=y_tr)

    params = {
        "objective":         "regression",
        "metric":            "rmse",
        "linear_tree":       True,
        "num_leaves":        8,
        "min_data_in_leaf":  20,
        "learning_rate":     0.05,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "reg_alpha":         0.1,
        "reg_lambda":        1.0,
        "linear_lambda":     1.0,
        "verbosity":         -1,
        "seed":              42,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        callbacks=[lgb.log_evaluation(period=-1)],
    )
    pred = model.predict(X_te_s)
    return pred, y_te


def find_lag(biz: np.ndarray, sys: np.ndarray, max_lag: int = 20) -> int:
    xd, yd = np.diff(biz.astype(float)), np.diff(sys.astype(float))
    best_lag, best = 0, 0.0
    for lag in range(min(max_lag + 1, len(xd))):
        a, b = (xd[:-lag], yd[lag:]) if lag else (xd, yd)
        if len(a) < 3:
            continue
        p = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else 0.0
        if abs(p) > best:
            best, best_lag = abs(p), lag
    return best_lag


def run_pattern(name: str, label: str, biz_fn, target: str = "cpu") -> dict:
    biz     = biz_fn()
    metrics = generate_system_metrics(biz)
    sys_arr = metrics[target]
    n       = len(biz)
    split   = int(n * 0.8)

    tau = find_lag(biz[:split], sys_arr[:split])

    print(f"\n  {'─'*68}")
    print(f"  Pattern : {label}")
    print(f"  Target  : {target}  |  lag={tau} steps ({tau*STEP_SECONDS//60} min)")
    print(f"  Biz range train: [{biz[:split].min():.1f}, {biz[:split].max():.1f}]  "
          f"test: [{biz[split:].min():.1f}, {biz[split:].max():.1f}]")
    extrapolation_needed = biz[split:].max() > biz[:split].max()
    if extrapolation_needed:
        print(f"  ⚠  EXTRAPOLATION: test biz max ({biz[split:].max():.1f}) > "
              f"train biz max ({biz[:split].max():.1f})")

    results = {}

    t0 = time.time()
    try:
        biz_lag_tr = np.roll(biz[:split], tau);  biz_lag_tr[:tau] = biz[0]
        biz_lag_te = np.roll(biz[split:], tau);  biz_lag_te[:tau] = biz[split]
        pred_prophet = train_prophet(biz_lag_tr, sys_arr[:split], biz_lag_te)
        y_te         = sys_arr[split:]
        # Align lengths
        min_len = min(len(pred_prophet), len(y_te))
        r2_p  = r2_score(y_te[:min_len], pred_prophet[:min_len])
        mae_p = mean_absolute_error(y_te[:min_len], pred_prophet[:min_len])
        results["prophet"] = {"r2": r2_p, "mae": mae_p, "time": time.time() - t0}
        verdict = "✓" if r2_p >= 0.85 else "✗"
        print(f"  Prophet  → R²={r2_p:+.4f}  MAE={mae_p:.2f}  "
              f"time={results['prophet']['time']:.1f}s  {verdict}")
    except Exception as e:
        print(f"  Prophet  → ERROR: {e}")
        results["prophet"] = {"r2": None, "mae": None, "time": None}

    t0 = time.time()
    try:
        pred_xgb, y_te_xgb = train_xgb(biz, sys_arr, tau, split)
        r2_x  = r2_score(y_te_xgb, pred_xgb)
        mae_x = mean_absolute_error(y_te_xgb, pred_xgb)
        results["xgboost"] = {"r2": r2_x, "mae": mae_x, "time": time.time() - t0}
        verdict = "✓" if r2_x >= 0.85 else "✗"
        print(f"  XGBoost  → R²={r2_x:+.4f}  MAE={mae_x:.2f}  "
              f"time={results['xgboost']['time']:.1f}s  {verdict}")
    except Exception as e:
        print(f"  XGBoost  → ERROR: {e}")
        results["xgboost"] = {"r2": None, "mae": None, "time": None}

    t0 = time.time()
    try:
        pred_r, y_te_r = train_ridge(biz, sys_arr, tau, split)
        r2_r  = r2_score(y_te_r, pred_r)
        mae_r = mean_absolute_error(y_te_r, pred_r)
        results["ridge"] = {"r2": r2_r, "mae": mae_r, "time": time.time() - t0}
        verdict = "✓" if r2_r >= 0.85 else "✗"
        print(f"  Ridge    → R²={r2_r:+.4f}  MAE={mae_r:.2f}  "
              f"time={results['ridge']['time']:.1f}s  {verdict}")
    except Exception as e:
        print(f"  Ridge    → ERROR: {e}")
        results["ridge"] = {"r2": None, "mae": None, "time": None}

    t0 = time.time()
    try:
        pred_h, y_te_h = train_hybrid(biz, sys_arr, tau, split)
        r2_h  = r2_score(y_te_h, pred_h)
        mae_h = mean_absolute_error(y_te_h, pred_h)
        results["hybrid"] = {"r2": r2_h, "mae": mae_h, "time": time.time() - t0}
        verdict = "✓" if r2_h >= 0.85 else "✗"
        print(f"  Hybrid   → R²={r2_h:+.4f}  MAE={mae_h:.2f}  "
              f"time={results['hybrid']['time']:.1f}s  {verdict}")
    except Exception as e:
        print(f"  Hybrid   → ERROR: {e}")
        results["hybrid"] = {"r2": None, "mae": None, "time": None}

    t0 = time.time()
    try:
        pred_c, y_te_c = train_combined(biz, sys_arr, tau, split)
        r2_c  = r2_score(y_te_c, pred_c)
        mae_c = mean_absolute_error(y_te_c, pred_c)
        results["combined"] = {"r2": r2_c, "mae": mae_c, "time": time.time() - t0}
        verdict = "✓" if r2_c >= 0.85 else "✗"
        print(f"  Combined → R²={r2_c:+.4f}  MAE={mae_c:.2f}  "
              f"time={results['combined']['time']:.1f}s  {verdict}")
    except Exception as e:
        print(f"  Combined → ERROR: {e}")
        results["combined"] = {"r2": None, "mae": None, "time": None}

    t0 = time.time()
    try:
        pred_l, y_te_l = train_lgbm_linear(biz, sys_arr, tau, split)
        r2_l  = r2_score(y_te_l, pred_l)
        mae_l = mean_absolute_error(y_te_l, pred_l)
        results["lgbm"] = {"r2": r2_l, "mae": mae_l, "time": time.time() - t0}
        verdict = "✓" if r2_l >= 0.85 else "✗"
        print(f"  LGBM-lin → R²={r2_l:+.4f}  MAE={mae_l:.2f}  "
              f"time={results['lgbm']['time']:.1f}s  {verdict}")
    except Exception as e:
        print(f"  LGBM-lin → ERROR: {e}")
        results["lgbm"] = {"r2": None, "mae": None, "time": None}

    all_keys = ("prophet", "xgboost", "ridge", "hybrid", "combined", "lgbm")
    scores = {k: results[k]["r2"] for k in all_keys
              if results.get(k, {}).get("r2") is not None}
    if scores:
        best_k  = max(scores, key=scores.get)
        best_v  = scores[best_k]
        vals    = sorted(scores.values(), reverse=True)
        second  = vals[1] if len(vals) > 1 else best_v
        winner  = best_k.capitalize() if best_v > second + 0.01 else "tie"
        delta_cx = (results.get("combined", {}).get("r2") or 0) - (results.get("xgboost", {}).get("r2") or 0)
        print(f"  Winner   → {winner}  (Combined−XGBoost ΔR²={delta_cx:+.4f})")
        results["winner"]   = winner
        results["delta_cx"] = delta_cx

    return results


def test_extrapolation():
    print("\n" + "="*70)
    print("  EXTRAPOLATION TEST")
    print("  Train: RPS 0..50  |  Test: RPS 50..150 (never seen in training)")
    print("="*70)

    n_tr = int(N * 0.7)
    n_te = N - n_tr

    biz_full = np.clip(
        np.linspace(5, 150, N) + 10 * np.sin(2*np.pi*t/(24*3600/STEP_SECONDS))
        + rng.normal(0, 3, N), 1, 200
    )

    metrics  = generate_system_metrics(biz_full)
    sys_full = metrics["cpu"]
    tau      = find_lag(biz_full[:n_tr], sys_full[:n_tr])

    print(f"\n  Train biz range: [{biz_full[:n_tr].min():.1f}, {biz_full[:n_tr].max():.1f}]")
    print(f"  Test  biz range: [{biz_full[n_tr:].min():.1f}, {biz_full[n_tr:].max():.1f}]")
    print(f"  Lag detected: {tau} steps ({tau*STEP_SECONDS//60} min)\n")

    try:
        biz_lag = np.roll(biz_full, tau); biz_lag[:tau] = biz_full[0]
        pred_p = train_prophet(biz_lag[:n_tr], sys_full[:n_tr], biz_lag[n_tr:])
        y_te   = sys_full[n_tr:]
        min_l  = min(len(pred_p), len(y_te))
        r2_p   = r2_score(y_te[:min_l], pred_p[:min_l])
        mae_p  = mean_absolute_error(y_te[:min_l], pred_p[:min_l])
        print(f"  Prophet  → R²={r2_p:+.4f}  MAE={mae_p:.2f}  {'✓ EXTRAPOLATES' if r2_p>0.7 else '✗ FAILS'}")
    except Exception as e:
        print(f"  Prophet  → ERROR: {e}")
        r2_p = None

    try:
        pred_x, y_te_x = train_xgb(biz_full, sys_full, tau, n_tr)
        r2_x  = r2_score(y_te_x, pred_x)
        mae_x = mean_absolute_error(y_te_x, pred_x)
        print(f"  XGBoost  → R²={r2_x:+.4f}  MAE={mae_x:.2f}  {'✓ EXTRAPOLATES' if r2_x>0.7 else '✗ FAILS'}")
    except Exception as e:
        print(f"  XGBoost  → ERROR: {e}")
        r2_x = None

    try:
        pred_r, y_te_r = train_ridge(biz_full, sys_full, tau, n_tr)
        r2_r  = r2_score(y_te_r, pred_r)
        mae_r = mean_absolute_error(y_te_r, pred_r)
        print(f"  Ridge    → R²={r2_r:+.4f}  MAE={mae_r:.2f}  {'✓ EXTRAPOLATES' if r2_r>0.7 else '✗ FAILS'}")
    except Exception as e:
        print(f"  Ridge    → ERROR: {e}")
        r2_r = None

    try:
        pred_h, y_te_h = train_hybrid(biz_full, sys_full, tau, n_tr)
        r2_h  = r2_score(y_te_h, pred_h)
        mae_h = mean_absolute_error(y_te_h, pred_h)
        print(f"  Hybrid   → R²={r2_h:+.4f}  MAE={mae_h:.2f}  {'✓ EXTRAPOLATES' if r2_h>0.7 else '✗ FAILS'}")
    except Exception as e:
        print(f"  Hybrid   → ERROR: {e}")
        r2_h = None

    try:
        pred_c, y_te_c = train_combined(biz_full, sys_full, tau, n_tr)
        r2_c  = r2_score(y_te_c, pred_c)
        mae_c = mean_absolute_error(y_te_c, pred_c)
        print(f"  Combined → R²={r2_c:+.4f}  MAE={mae_c:.2f}  {'✓ EXTRAPOLATES' if r2_c>0.7 else '✗ FAILS'}")
    except Exception as e:
        print(f"  Combined → ERROR: {e}")
        r2_c = None

    try:
        pred_l, y_te_l = train_lgbm_linear(biz_full, sys_full, tau, n_tr)
        r2_l  = r2_score(y_te_l, pred_l)
        mae_l = mean_absolute_error(y_te_l, pred_l)
        print(f"  LGBM-lin → R²={r2_l:+.4f}  MAE={mae_l:.2f}  {'✓ EXTRAPOLATES' if r2_l>0.7 else '✗ FAILS'}")
    except Exception as e:
        print(f"  LGBM-lin → ERROR: {e}")
        r2_l = None

    scores = {k: v for k, v in [("Prophet",r2_p),("XGBoost",r2_x),("Ridge",r2_r),
                                  ("Hybrid",r2_h),("Combined",r2_c),("LGBM-lin",r2_l)]
              if v is not None}
    if scores:
        best = max(scores, key=scores.get)
        print(f"\n  Extrapolation winner: {best}")
        for name, r2_val in sorted(scores.items(), key=lambda x: -x[1]):
            delta = r2_val - (r2_x or 0)
            marker = " ← baseline" if name == "XGBoost" else f"  ΔvsXGB={delta:+.4f}"
            print(f"  {name:<10} R²={r2_val:+.4f}{marker}")


def main():
    parser = argparse.ArgumentParser(description="Prophet vs XGBoost for load prediction")
    parser.add_argument("--pattern",  choices=list(PATTERNS) + ["all"], default="all")
    parser.add_argument("--target",   choices=["cpu","ram_pct","ram_gb","net","disk"],
                        default="cpu")
    parser.add_argument("--test-extrapolation", action="store_true",
                        help="Run explicit extrapolation benchmark")
    args = parser.parse_args()

    algo = "XGBoost" if HAS_XGB else "GradientBoosting"
    print("=" * 70)
    print("  Prophet vs XGBoost — Load Prediction Evaluation")
    print(f"  Tree algorithm: {algo}  |  Target: {args.target}")
    print(f"  Data: {DAYS} days × {STEP_SECONDS}s = {N:,} points per pattern")
    print("=" * 70)

    if args.test_extrapolation:
        test_extrapolation()
        return

    to_run = list(PATTERNS.items()) if args.pattern == "all" \
             else [(args.pattern, PATTERNS[args.pattern])]

    all_results = []
    for pname, (label, fn) in to_run:
        r = run_pattern(pname, label, fn, args.target)
        all_results.append((label, r))

    print(f"\n{'='*108}")
    print("  SUMMARY")
    print(f"{'='*108}")
    print(f"  {'Pattern':<30} {'Prophet':>8} {'XGBoost':>8} {'Ridge':>8} {'Hybrid':>8} {'Combined':>8} {'LGBM-lin':>8} {'C−X':>7}  Winner")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*7}  {'─'*7}")
    wins = {k: 0 for k in ("prophet","xgboost","ridge","hybrid","combined","lgbm","tie")}
    for label, r in all_results:
        rp  = r.get("prophet",  {}).get("r2")
        rx  = r.get("xgboost",  {}).get("r2")
        rr  = r.get("ridge",    {}).get("r2")
        rh  = r.get("hybrid",   {}).get("r2")
        rc  = r.get("combined", {}).get("r2")
        rl  = r.get("lgbm",     {}).get("r2")
        w   = r.get("winner", "?").lower()
        dcx = r.get("delta_cx", 0.0)
        fmt = lambda v: f"{v:+.4f}" if v is not None else "  ERROR"
        print(f"  {label:<30} {fmt(rp):>8} {fmt(rx):>8} {fmt(rr):>8} {fmt(rh):>8} {fmt(rc):>8} {fmt(rl):>8} {dcx:>+7.4f}  {w}")
        wins[w] = wins.get(w, 0) + 1

    print(f"\n  ", "  ".join(f"{k.capitalize()}:{v}" for k, v in wins.items() if v > 0))
    print(f"  Threshold: R² ≥ 0.85  |  C−X = Combined minus XGBoost")
    print(f"  Key: does LGBM-lin beat Ridge on extrapolation while staying competitive on normal patterns?")
    print(f"\n  Run --test-extrapolation for pure out-of-range benchmark.")


if __name__ == "__main__":
    main()
