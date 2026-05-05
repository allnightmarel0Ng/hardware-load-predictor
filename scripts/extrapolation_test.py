from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Tuple, Callable, List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingRegressor

SCRIPTS_DIR = Path(__file__).parent
ALIBABA_FILE = SCRIPTS_DIR / "machine_usage_days_1_to_8_grouped_300_seconds.csv"
GOOGLE_FILE = SCRIPTS_DIR / "instance_usage_grouped_300_seconds_month.csv"
ALIBABA_URL = "https://zenodo.org/records/14564935/files/machine_usage_days_1_to_8_grouped_300_seconds.csv?download=1"
GOOGLE_URL = "https://zenodo.org/records/14564935/files/instance_usage_grouped_300_seconds_month.csv?download=1"

STEP_SECONDS = 300
TRUE_LAG = 1
SIGNIFICANCE = 0.6
LOOKBACK = 30
TRAIN_RATIO = 0.8
MAX_LAG = 20
REL_STD_THRESHOLD = 0.05

CORR_LOW = 0.3
CORR_MEDIUM = 0.5
CORR_HIGH = 0.7

RPS_MAX_QUANTILE = 0.8

TARGET_KEYS = ["cpu", "ram_gb", "ram_pct", "net", "disk"]
RNG = np.random.default_rng(42)

def load_alibaba() -> Dict[str, np.ndarray]:
    if not ALIBABA_FILE.exists():
        raise FileNotFoundError(f"Alibaba not found.\ncurl -L '{ALIBABA_URL}' -o {ALIBABA_FILE}")
    df = pd.read_csv(ALIBABA_FILE, header=None, skiprows=1,
                     names=["cpu", "mem_pct", "net_in", "net_out", "disk"])
    df.replace([-1, 101], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(inplace=True)
    cpu = df["cpu"].to_numpy(float)
    print(f"  Alibaba 2018: {len(df):,} rows  cpu_mean={cpu.mean():.1f}%")
    return {
        "cpu": cpu,
        "ram_pct": df["mem_pct"].to_numpy(float),
        "ram_gb": df["mem_pct"].to_numpy(float) / 100 * 128,
        "net": df["net_in"].to_numpy(float) / 100 * 500,
        "disk": df["disk"].to_numpy(float),
        "n": len(df),
        "name": "Alibaba 2018",
    }

def load_google() -> Dict[str, np.ndarray]:
    if not GOOGLE_FILE.exists():
        raise FileNotFoundError(f"Google not found.\ncurl -L '{GOOGLE_URL}' -o {GOOGLE_FILE}")
    df = pd.read_csv(GOOGLE_FILE)
    cpu = (df["avg_cpu"] * 100).clip(0, 99)
    mem_pct = (df["avg_mem"] * 100).clip(0, 99)
    cpu_arr = cpu.to_numpy(float)
    net = pd.Series(np.clip(8 + 0.7 * cpu_arr + RNG.normal(0, 5, len(cpu_arr)), 0, 400))
    disk = pd.Series(np.clip(5 + 0.25 * cpu_arr + RNG.normal(0, 4, len(cpu_arr)), 0, 80))
    df.dropna(subset=["avg_cpu", "avg_mem"], inplace=True)
    print(f"  Google 2019:  {len(cpu):,} rows  cpu_mean={cpu.mean():.1f}%")
    return {
        "cpu": cpu_arr,
        "ram_pct": mem_pct.to_numpy(float),
        "ram_gb": mem_pct.to_numpy(float) / 100 * 256,
        "net": net.to_numpy(float),
        "disk": disk.to_numpy(float),
        "n": len(cpu),
        "name": "Google 2019",
    }

def _cpu_to_proxy(cpu: np.ndarray, lag: int, coeff: float = 0.5, noise_std: float = 2.0) -> np.ndarray:
    n = len(cpu)
    shifted = np.empty(n)
    shifted[:n - lag] = cpu[lag:]
    shifted[n - lag:] = cpu[-1]
    proxy = shifted / coeff + RNG.normal(0, noise_std, n)
    return np.maximum(0.5, proxy)

def pattern_sinusoidal(cpu: np.ndarray, lag: int) -> np.ndarray:
    n = len(cpu)
    t = np.arange(n)
    base = _cpu_to_proxy(cpu, lag)
    modulation = 1.0 + 0.2 * np.sin(2 * np.pi * t / (24 * 3600 / STEP_SECONDS))
    return np.clip(base * modulation, 0.5, 300)

def pattern_step(cpu: np.ndarray, lag: int) -> np.ndarray:
    base = _cpu_to_proxy(cpu, lag)
    p25, p50, p75 = np.percentile(base, [25, 50, 75])
    stepped = np.where(base < p25, p25 * 0.5,
              np.where(base < p50, p25,
              np.where(base < p75, p50, p75 * 1.3)))
    return np.clip(stepped + RNG.normal(0, 1.5, len(cpu)), 0.5, 300)

def pattern_spiky(cpu: np.ndarray, lag: int) -> np.ndarray:
    n = len(cpu)
    t = np.arange(n)
    base = _cpu_to_proxy(cpu, lag, coeff=0.8)
    bg = np.full(n, np.percentile(base, 20))
    n_spikes = max(30, n // 80)
    for _ in range(n_spikes):
        center = RNG.integers(0, n)
        width = RNG.integers(3, 12)
        height = RNG.uniform(0.8, 2.5) * np.percentile(base, 90)
        bg += height * np.exp(-0.5 * ((t - center) / width) ** 2)
    return np.clip(bg + RNG.normal(0, 1, n), 0.5, 300)

def pattern_drifting(cpu: np.ndarray, lag: int) -> np.ndarray:
    n = len(cpu)
    base = _cpu_to_proxy(cpu, lag)
    trend = np.linspace(0.6, 1.4, n)
    return np.clip(base * trend + RNG.normal(0, 1.5, n), 0.5, 300)

def pattern_multimodal(cpu: np.ndarray, lag: int) -> np.ndarray:
    n = len(cpu)
    t = np.arange(n)
    base = _cpu_to_proxy(cpu, lag)
    steps_per_day = 24 * 3600 // STEP_SECONDS
    day_of_week = (t // steps_per_day) % 7
    weekend_mask = (day_of_week >= 5).astype(float)
    scale = 1.0 - 0.7 * weekend_mask
    return np.clip(base * scale + RNG.normal(0, 1, n), 0.5, 300)

def pattern_long_spiky(cpu: np.ndarray, lag: int) -> np.ndarray:
    n = len(cpu)
    biz = np.zeros(n)
    n_spikes = RNG.integers(3, 8)
    for _ in range(n_spikes):
        duration = RNG.integers(12, 67)
        amplitude = RNG.uniform(100, 300)
        start = RNG.integers(LOOKBACK, n - duration - LOOKBACK)
        biz[start:start+duration] = amplitude + RNG.normal(0, 5, duration)
    zero_mask = biz == 0
    biz[zero_mask] = RNG.normal(0, 1, np.sum(zero_mask))
    return np.clip(biz, 0.5, 300)

PATTERNS: Dict[str, Tuple[str, Callable]] = {
    "sinusoidal": ("Sinusoidal (SaaS daily wave)", pattern_sinusoidal),
    "step":       ("Step function (batch / ETL)",   pattern_step),
    "spiky":      ("Spiky bursts (event-driven)",   pattern_spiky),
    "drifting":   ("Drifting trend (growing audience)", pattern_drifting),
    "multimodal": ("Multimodal (weekday/weekend split)", pattern_multimodal),
    "long_spiky": ("Long spikes (long idle + burst hours)", pattern_long_spiky),
}

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def rank(x):
        s = np.argsort(x)
        r = np.empty_like(s, float)
        r[s] = np.arange(len(x)) + 1
        return r
    return _pearson(rank(a), rank(b))

def find_lag_and_corr(biz: np.ndarray, sys: np.ndarray, max_lag: int = MAX_LAG) -> Tuple[int, float]:
    xd = np.diff(biz.astype(float))
    yd = np.diff(sys.astype(float))
    best_lag, best_score = 0, 0.0
    max_possible = min(max_lag + 1, len(xd))
    for lag in range(max_possible):
        if lag == 0:
            a, b = xd, yd
        else:
            a = xd[:-lag]
            b = yd[lag:]
        score = abs(_pearson(a, b)) + abs(_spearman(a, b))
        if score > best_score:
            best_score, best_lag = score, lag
    if best_lag == 0:
        a, b = xd, yd
    else:
        a = xd[:-best_lag]
        b = yd[best_lag:]
    p = _pearson(a, b)
    s = _spearman(a, b)
    r = max(abs(p), abs(s))
    return best_lag, r

def build_features_full(biz: np.ndarray, ds: Dict[str, np.ndarray], tau: int) -> np.ndarray:
    n = ds["n"]
    rows = []
    for i in range(LOOKBACK, n - 1):
        hour = (i * STEP_SECONDS / 3600) % 24
        dow = (i * STEP_SECONDS / 86400) % 7
        sin_h = math.sin(2 * math.pi * hour / 24)
        cos_h = math.cos(2 * math.pi * hour / 24)
        sin_d = math.sin(2 * math.pi * dow / 7)
        cos_d = math.cos(2 * math.pi * dow / 7)
        trend = i / n

        bl = biz[i - tau] if i >= tau else biz[0]
        rm30 = biz[max(0, i-30):i].mean()
        bn = bl / (rm30 + 1e-9)
        d1 = biz[i] - biz[i-1] if i >= 1 else 0.0
        d2 = biz[i-1] - biz[i-2] if i >= 2 else 0.0
        mu5 = biz[max(0, i-5):i].mean()
        bz = (biz[i] - mu5) / (biz[max(0, i-5):i].std() + 1e-9)
        biz_feats = [bl, bn, d1, d2, bz, bl * sin_h, bl * cos_h]

        sys_feats = []
        for key in ("cpu", "ram_pct", "net", "disk"):
            arr = ds[key]
            l1 = arr[i-1]
            l2 = arr[i-2] if i >= 2 else arr[0]
            l3 = arr[i-3] if i >= 3 else arr[0]
            m5 = arr[max(0, i-5):i].mean()
            m15 = arr[max(0, i-15):i].mean()
            m30 = arr[max(0, i-30):i].mean()
            s5 = arr[max(0, i-5):i].std() + 1e-9
            s15 = arr[max(0, i-15):i].std() + 1e-9
            sys_feats.extend([l1, l2, l3, m5, m15, m30, s5, s15])

        row = biz_feats + [sin_h, cos_h, sin_d, cos_d, trend] + sys_feats
        rows.append(row)

    X = StandardScaler().fit_transform(np.array(rows))
    return X

def train_ridge(X, y):
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    return ridge

def train_xgboost(X, y):
    if HAS_XGB:
        xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, random_state=42, verbosity=0)
    else:
        xgb = GradientBoostingRegressor(n_estimators=500, max_depth=6, learning_rate=0.03, random_state=42)
    xgb.fit(X, y)
    return xgb

def train_hybrid(X, y):
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    y_ridge = ridge.predict(X)
    residuals = y - y_ridge
    if HAS_XGB:
        xgb = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0)
    else:
        xgb = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    xgb.fit(X, residuals)
    return ridge, xgb

def predict_hybrid(ridge, xgb, X):
    return ridge.predict(X) + xgb.predict(X)

def evaluate_reg(pred, true):
    r2 = r2_score(true, pred)
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / (true + 1e-9))) * 100
    return r2, mae, mape

def run_extrapolation_test(ds: Dict[str, np.ndarray], biz: np.ndarray, lag_true: int):
    n = ds["n"]
    lags = {}
    for key in TARGET_KEYS:
        lag, _ = find_lag_and_corr(biz, ds[key])
        lags[key] = lag

    X_dict = {}
    y_dict = {}
    for key in TARGET_KEYS:
        tau = lags[key]
        X = build_features_full(biz, ds, tau)
        y = ds[key][LOOKBACK:n-1]
        X_dict[key] = X
        y_dict[key] = y

    biz_current = np.array([biz[i] for i in range(LOOKBACK, n-1)])
    n_samples = len(biz_current)
    split_idx = int(TRAIN_RATIO * n_samples)
    biz_train_temporal = biz_current[:split_idx]
    max_rps_train = np.percentile(biz_train_temporal, 100 * RPS_MAX_QUANTILE)
    test_mask = biz_current >= max_rps_train
    train_mask = ~test_mask
    rps_above = np.maximum(0, biz_current - max_rps_train).reshape(-1, 1)

    print(f"\n  Train samples: {np.sum(train_mask)}, Test samples: {np.sum(test_mask)}")
    print(f"  Max RPS in train (threshold): {max_rps_train:.2f}, Max RPS in test: {np.max(biz_current[test_mask]):.2f}")

    all_res = {}
    for key in TARGET_KEYS:
        X = X_dict[key]
        y = y_dict[key]
        X_aug = np.hstack([X, rps_above])
        X_train = X_aug[train_mask]
        y_train = y[train_mask]
        X_test = X_aug[test_mask]
        y_test = y[test_mask]
        if len(X_test) == 0:
            continue

        ridge = train_ridge(X_train, y_train)
        pred_ridge = ridge.predict(X_test)
        r2_r, mae_r, mape_r = evaluate_reg(pred_ridge, y_test)

        xgb = train_xgboost(X_train, y_train)
        pred_xgb = xgb.predict(X_test)
        r2_x, mae_x, mape_x = evaluate_reg(pred_xgb, y_test)

        ridge_h, xgb_h = train_hybrid(X_train, y_train)
        pred_hybrid = predict_hybrid(ridge_h, xgb_h, X_test)
        r2_h, mae_h, mape_h = evaluate_reg(pred_hybrid, y_test)

        all_res[key] = {
            'ridge': (r2_r, mae_r, mape_r),
            'xgb': (r2_x, mae_x, mape_x),
            'hybrid': (r2_h, mae_h, mape_h),
        }
    return all_res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["alibaba","google","both"], default="alibaba")
    parser.add_argument("--pattern", choices=list(PATTERNS)+["all"], default="all")
    parser.add_argument("--lag", type=int, default=TRUE_LAG)
    args = parser.parse_args()

    datasets = []
    for name, loader in [("alibaba", load_alibaba), ("google", load_google)]:
        if args.dataset not in ("both", name):
            continue
        try:
            datasets.append(loader())
        except FileNotFoundError as e:
            print(e)
    if not datasets:
        sys.exit(1)

    to_run = list(PATTERNS.items()) if args.pattern == "all" else [(args.pattern, PATTERNS[args.pattern])]

    print("=" * 80)
    print("  EXTRAPOLATION TEST (including LONG_SPIKY pattern)")
    print("  Train on RPS < {:.0f}% of max RPS in training, test on higher RPS".format(RPS_MAX_QUANTILE*100))
    print("  Added feature: rps_above_max = max(0, RPS - max_train_RPS)")
    print("=" * 80)

    for ds in datasets:
        print(f"\n{'━'*80}\n  Dataset: {ds['name']}\n{'━'*80}")
        for pname, (label, fn) in to_run:
            biz = fn(ds["cpu"], args.lag)
            print(f"\n  Pattern: {label}")
            res = run_extrapolation_test(ds, biz, args.lag)
            if not res:
                print("  No extrapolation data (all RPS below threshold).")
                continue
            print("\n  Target |   Ridge R²/MAE/MAPE   |   XGBoost R²/MAE/MAPE  |   Hybrid R²/MAE/MAPE  ")
            for key in TARGET_KEYS:
                if key not in res:
                    continue
                r2_r, mae_r, mape_r = res[key]['ridge']
                r2_x, mae_x, mape_x = res[key]['xgb']
                r2_h, mae_h, mape_h = res[key]['hybrid']
                print(f"  {key:6} | {r2_r:6.3f} {mae_r:6.2f} {mape_r:5.1f}% | {r2_x:6.3f} {mae_x:6.2f} {mape_x:5.1f}% | {r2_h:6.3f} {mae_h:6.2f} {mape_h:5.1f}%")

if __name__ == "__main__":
    main()