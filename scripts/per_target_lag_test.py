from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_XGB = False

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

CORR_STRONG = 0.7
CORR_MEDIUM = 0.4

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

rng = np.random.default_rng(42)

def _cpu_to_proxy(cpu, lag, coeff=0.5, noise_std=2.0):
    n=len(cpu)
    s=np.empty(n)

    s[:n-lag]=cpu[lag:]; s[n-lag:]=cpu[-1]

    return np.maximum(0.5, s/coeff+rng.normal(0,noise_std,n))

def pattern_sinusoidal(metrics: Dict[str, np.ndarray], lag: int) -> np.ndarray:
    n = len(metrics['cpu'])
    t = np.arange(n)
    base = _cpu_to_proxy(metrics['cpu'], lag)
    modulation = 1.0 + 0.2 * np.sin(2 * np.pi * t / (24 * 3600 / STEP_SECONDS))
    return np.clip(base * modulation, 0.5, 300)

def pattern_step(metrics: Dict[str, np.ndarray], lag: int) -> np.ndarray:
    base = _cpu_to_proxy(metrics['cpu'], lag)
    p25, p50, p75 = np.percentile(base, [25, 50, 75])
    stepped = np.where(base < p25, p25 * 0.5,
              np.where(base < p50, p25,
              np.where(base < p75, p50, p75 * 1.3)))
    return np.clip(stepped + RNG.normal(0, 1.5, len(base)), 0.5, 300)

def pattern_spiky(metrics: Dict[str, np.ndarray], lag: int) -> np.ndarray:
    n = len(metrics['cpu'])
    t = np.arange(n)
    base = _cpu_to_proxy(metrics['cpu'], lag)
    bg = np.full(n, np.percentile(base, 20))
    n_spikes = max(30, n // 80)
    for _ in range(n_spikes):
        center = RNG.integers(0, n)
        width = RNG.integers(3, 12)
        height = RNG.uniform(0.8, 2.5) * np.percentile(base, 90)
        bg += height * np.exp(-0.5 * ((t - center) / width) ** 2)
    return np.clip(bg + RNG.normal(0, 1, n), 0.5, 300)

def pattern_drifting(metrics: Dict[str, np.ndarray], lag: int) -> np.ndarray:
    n = len(metrics['cpu'])
    base = _cpu_to_proxy(metrics['cpu'], lag)
    trend = np.linspace(0.6, 1.4, n)
    return np.clip(base * trend + RNG.normal(0, 1.5, n), 0.5, 300)

def pattern_multimodal(metrics: Dict[str, np.ndarray], lag: int) -> np.ndarray:
    n = len(metrics['cpu'])
    t = np.arange(n)
    base = _cpu_to_proxy(metrics['cpu'], lag)
    steps_per_day = 24 * 3600 // STEP_SECONDS
    day_of_week = (t // steps_per_day) % 7
    weekend_mask = (day_of_week >= 5).astype(float)
    scale = 1.0 - 0.7 * weekend_mask
    return np.clip(base * scale + RNG.normal(0, 1, n), 0.5, 300)

PATTERNS: Dict[str, Tuple[str, Callable]] = {
    "sinusoidal": ("Sinusoidal (SaaS daily wave)", pattern_sinusoidal),
    "step":       ("Step function (batch / ETL)",   pattern_step),
    "spiky":      ("Spiky bursts (event-driven)",   pattern_spiky),
    "drifting":   ("Drifting trend (growing audience)", pattern_drifting),
    "multimodal": ("Multimodal (weekday/weekend split)", pattern_multimodal),
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

def find_lag(biz: np.ndarray, sys: np.ndarray, max_lag: int = 20) -> Tuple[int, float, float, float, bool]:
    """Находит оптимальный лаг (0..max_lag) по сумме абсолютных корреляций Пирсона и Спирмена на разностях."""
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
    return best_lag, p, s, r, r >= SIGNIFICANCE

def get_correlation_info(biz: np.ndarray, sys: np.ndarray, train_ratio: float = 0.8, max_lag: int = 20) -> Dict:
    split = int(train_ratio * len(biz))
    biz_train = biz[:split]
    sys_train = sys[:split]
    lag, _, _, r, sig = find_lag(biz_train, sys_train, max_lag)
    if not sig or r < CORR_MEDIUM:
        strength = 'none'
    elif r < CORR_STRONG:
        strength = 'medium'
    else:
        strength = 'strong'
    return {'lag': lag, 'r': r, 'significant': sig, 'strength': strength}

def build_features_adaptive(biz: np.ndarray, ds: Dict[str, np.ndarray],
                            tau: int, biz_strength: str) -> np.ndarray:
    n = ds["n"]
    rows = []
    for i in range(LOOKBACK, n - 1):
        # ---- временные признаки (всегда) ----
        hour = (i * STEP_SECONDS / 3600) % 24
        dow = (i * STEP_SECONDS / 86400) % 7
        sin_h = math.sin(2 * math.pi * hour / 24)
        cos_h = math.cos(2 * math.pi * hour / 24)
        sin_d = math.sin(2 * math.pi * dow / 7)
        cos_d = math.cos(2 * math.pi * dow / 7)
        trend = i / n

        biz_feats = []
        if biz_strength != 'none':
            bl = biz[i - tau] if i >= tau else biz[0]
            biz_feats.append(bl)
            if biz_strength == 'strong':
                rm30 = biz[max(0, i-30):i].mean()
                bn = bl / (rm30 + 1e-9)
                biz_feats.append(bn)
                d1 = biz[i] - biz[i-1] if i >= 1 else 0.0
                d2 = biz[i-1] - biz[i-2] if i >= 2 else 0.0
                biz_feats.extend([d1, d2])
                mu5 = biz[max(0, i-5):i].mean()
                bz = (biz[i] - mu5) / (biz[max(0, i-5):i].std() + 1e-9)
                biz_feats.append(bz)
                biz_feats.extend([bl * sin_h, bl * cos_h])
            elif biz_strength == 'medium':
                d1 = biz[i] - biz[i-1] if i >= 1 else 0.0
                biz_feats.append(d1)

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

PARAM_GRID = [
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
    {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.03},
    {"n_estimators": 700, "max_depth": 7, "learning_rate": 0.02},
]

def _make_base(params: Dict):
    if HAS_XGB:
        return XGBRegressor(**params, random_state=42, subsample=0.8,
                            colsample_bytree=0.8, verbosity=0)
    return GradientBoostingRegressor(**params, random_state=42,
                                     subsample=0.8, min_samples_leaf=5)

def train_cv_single(X: np.ndarray, y: np.ndarray) -> Tuple[object, np.ndarray, np.ndarray]:
    n = len(X)
    test_size = int(0.2 * n)
    X_tv, X_test = X[:n - test_size], X[n - test_size:]
    y_tv, y_test = y[:n - test_size], y[n - test_size:]

    tscv = TimeSeriesSplit(n_splits=3)
    best_score, best_params = -np.inf, PARAM_GRID[0]
    for params in PARAM_GRID:
        scores = []
        for train_idx, val_idx in tscv.split(X_tv):
            base = _make_base(params)
            base.fit(X_tv[train_idx], y_tv[train_idx])
            pred = base.predict(X_tv[val_idx])
            scores.append(r2_score(y_tv[val_idx], pred))
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    final_model = _make_base(best_params)
    final_model.fit(X_tv, y_tv)
    return final_model, X_test, y_test

def run(ds: Dict[str, np.ndarray], label: str, biz_fn: Callable, lag_true: int) -> Dict[str, float]:
    biz = biz_fn(ds, lag_true)
    n = ds["n"]
    algo = "XGBoost" if HAS_XGB else "GBM"

    print(f"\n  {'─'*70}")
    print(f"  Pattern : {label}  |  Dataset : {ds['name']}  |  Algo: {algo}+CV")
    print(f"  Biz: mean={biz.mean():.1f}  std={biz.std():.1f}")

    # ---- 1. Определение лагов и силы корреляции (на train 80%) ----
    print("  Correlation & strength (only on first 80% of data):")
    lags = {}
    strengths = {}
    for tname, key in [("CPU","cpu"), ("RAM_GB","ram_gb"), ("RAM%","ram_pct"),
                       ("Net","net"), ("Disk","disk")]:
        info = get_correlation_info(biz, ds[key], train_ratio=TRAIN_RATIO)
        lags[key] = info['lag']
        strengths[key] = info['strength']
        ok = "✓" if info['lag'] == lag_true else f"✗(true={lag_true})"
        print(f"    {tname:<7} lag={info['lag']:2d}  r*={info['r']:.3f}  "
              f"sig={'Yes' if info['significant'] else 'No':3s}  "
              f"strength={info['strength']:6s}  {ok}")

    global_lag = int(np.median(list(lags.values())))
    print(f"  Global lag (median): {global_lag}step = {global_lag*STEP_SECONDS//60}min")

    y_targets = {}
    for key in TARGET_KEYS:
        y_targets[key] = ds[key][LOOKBACK:n-1]

    print(f"\n  {'Target':<10} {'τ_A':>4} {'R²_A':>8} {'τ_B':>4} {'R²_B':>8} "
          f"{'ΔR²':>8}  Winner")
    print(f"  {'-'*10} {'-'*4} {'-'*8} {'-'*4} {'-'*8} {'-'*8}  {'-'*6}")

    deltas = {}
    for key in TARGET_KEYS:
        y_t = y_targets[key]

        strength_a = strengths[key]
        X_a = build_features_adaptive(biz, ds, global_lag, strength_a)
        m_a, Xte_a, yte_a = train_cv_single(X_a, y_t)
        r2_a = r2_score(yte_a, m_a.predict(Xte_a))

        tau_b = lags.get(key, 0)
        strength_b = strengths[key]
        X_b = build_features_adaptive(biz, ds, tau_b, strength_b)
        m_b, Xte_b, yte_b = train_cv_single(X_b, y_t)
        r2_b = r2_score(yte_b, m_b.predict(Xte_b))

        delta = r2_b - r2_a
        deltas[key] = delta
        winner = "B ✓" if delta > 0.005 else ("A ✓" if delta < -0.005 else "≈ tie")
        tau_a_min = global_lag * STEP_SECONDS // 60
        tau_b_min = tau_b * STEP_SECONDS // 60
        print(f"  {key:<10} {tau_a_min:>3}m {r2_a:>+8.4f} {tau_b_min:>3}m "
              f"{r2_b:>+8.4f} {delta:>+8.4f}  {winner}")

    avg = np.mean(list(deltas.values()))
    print(f"  {'─'*70}")
    print(f"  Avg ΔR² (B−A): {avg:>+.4f}  "
          f"→ {'B wins ✓' if avg>0.002 else ('A wins' if avg<-0.002 else 'no clear winner')}")
    return deltas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["alibaba","google","both"], default="alibaba")
    parser.add_argument("--pattern", choices=list(PATTERNS)+["all"], default="sinusoidal")
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

    print("=" * 72)
    print("  ADAPTIVE per-target lag vs global lag — A/B comparison")
    print(f"  Algorithm: {'XGBoost' if HAS_XGB else 'GradientBoosting'} + temporal CV")
    print(f"  Embedded lag: {args.lag}step = {args.lag*STEP_SECONDS//60}min")
    print(f"  Adaptive thresholds: strong ≥ {CORR_STRONG}, medium ≥ {CORR_MEDIUM}, else none")
    print("=" * 72)

    all_deltas = []
    for ds in datasets:
        print(f"\n{'━'*72}\n  Dataset: {ds['name']}\n{'━'*72}")
        for pname, (label, fn) in to_run:
            d = run(ds, label, fn, args.lag)
            all_deltas.append((ds["name"], label, d))

    if len(all_deltas) > 1:
        print(f"\n{'='*72}\n  SUMMARY — ΔR² per target (B − A, positive = B better)\n{'='*72}")
        print(f"  {'Dataset':<14} {'Pattern':<35}", end="")
        for k in TARGET_KEYS:
            print(f" {k[:6]:>7}", end="")
        print(f"  {'avg':>7}")
        print(f"  {'-'*14} {'-'*35}" + " -------"*5 + "  -------")
        for ds_name, label, d in all_deltas:
            avg = np.mean(list(d.values()))
            print(f"  {ds_name:<14} {label:<35}", end="")
            for k in TARGET_KEYS:
                print(f" {d.get(k, 0):>+7.4f}", end="")
            print(f"  {avg:>+7.4f}")

if __name__ == "__main__":
    main()