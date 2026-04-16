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
    """Всегда используем полные бизнес-признаки (сдвиг, нормализация, разности, z-score, взаимодействия)."""
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

def select_model_and_metric(r: float, rel_std: float) -> Tuple[str, str, float, List]:
    """
    Возвращает: (model_type, eval_metric, success_threshold, param_grid)
    model_type: 'mean_baseline', 'ridge', 'gbm', 'xgboost'
    eval_metric: 'rel_mae', 'mae', 'r2'
    success_threshold: порог для PASS/FAIL (None если не оцениваем)
    param_grid: список словарей для гиперпараметров
    """
    if rel_std < REL_STD_THRESHOLD:
        return 'mean_baseline', 'rel_mae', 0.05, [{}]

    if r < CORR_LOW:
        # Ridge, метрика MAE (порог не задаём)
        return 'ridge', 'mae', None, [{'alpha': 0.1}, {'alpha': 1.0}, {'alpha': 10.0}]
    elif r < CORR_MEDIUM:
        # Ridge, метрика R² (порог 0.70)
        return 'ridge', 'r2', 0.70, [{'alpha': 0.1}, {'alpha': 1.0}, {'alpha': 10.0}]
    elif r < CORR_HIGH:
        # GBM (depth 4), метрика R² (порог 0.85)
        if HAS_XGB:
            # можно и XGBoost использовать, но для средних r лучше GBM
            return 'gbm', 'r2', 0.85, [{'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05}]
        else:
            return 'gbm', 'r2', 0.85, [{'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05}]
    else:
        # XGBoost (depth 6), метрика R² (порог 0.85)
        if HAS_XGB:
            return 'xgboost', 'r2', 0.85, [{'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.03}]
        else:
            # fallback на GBM с глубиной 6
            return 'gbm', 'r2', 0.85, [{'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.03}]

def train_model(X, y, model_type, param_grid):
    """Обучает модель с временным CV, возвращает предсказания на тесте (последние 20%)."""
    n = len(X)
    test_size = int(0.2 * n)
    X_tv, X_test = X[:n - test_size], X[n - test_size:]
    y_tv, y_test = y[:n - test_size], y[n - test_size:]

    if model_type == 'mean_baseline':
        pred_test = np.full_like(y_test, y_tv.mean())
        return pred_test, y_test

    # Подбор гиперпараметров через TimeSeriesSplit (3 folds)
    tscv = TimeSeriesSplit(n_splits=3)
    best_score = -np.inf
    best_params = param_grid[0] if param_grid else {}

    for params in param_grid:
        scores = []
        for train_idx, val_idx in tscv.split(X_tv):
            if model_type == 'ridge':
                model = Ridge(**params, random_state=42)
            elif model_type == 'gbm':
                if HAS_XGB:
                    # Используем XGBRegressor, но с параметрами GBM (n_estimators, max_depth, learning_rate)
                    model = XGBRegressor(**params, random_state=42, verbosity=0)
                else:
                    model = GradientBoostingRegressor(**params, random_state=42)
            elif model_type == 'xgboost':
                if HAS_XGB:
                    model = XGBRegressor(**params, random_state=42, verbosity=0)
                else:
                    # fallback
                    model = GradientBoostingRegressor(**params, random_state=42)
            else:
                raise ValueError(f"Unknown model_type {model_type}")

            model.fit(X_tv[train_idx], y_tv[train_idx])
            pred = model.predict(X_tv[val_idx])
            # Используем R² как метрику для выбора гиперпараметров
            sc = r2_score(y_tv[val_idx], pred)
            scores.append(sc)
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    # Финальная модель на всей train+val
    if model_type == 'ridge':
        final_model = Ridge(**best_params, random_state=42)
    elif model_type == 'gbm':
        if HAS_XGB:
            final_model = XGBRegressor(**best_params, random_state=42, verbosity=0)
        else:
            final_model = GradientBoostingRegressor(**best_params, random_state=42)
    elif model_type == 'xgboost':
        if HAS_XGB:
            final_model = XGBRegressor(**best_params, random_state=42, verbosity=0)
        else:
            final_model = GradientBoostingRegressor(**best_params, random_state=42)
    else:
        raise ValueError(f"Unknown model_type {model_type}")

    final_model.fit(X_tv, y_tv)
    pred_test = final_model.predict(X_test)
    return pred_test, y_test

def evaluate(pred, true, eval_metric, threshold):
    """Возвращает (значение_метрики, passed)."""
    if eval_metric == 'r2':
        val = r2_score(true, pred)
        passed = (threshold is not None) and (val >= threshold)
    elif eval_metric == 'mae':
        val = mean_absolute_error(true, pred)
        passed = (threshold is not None) and (val <= threshold)
    elif eval_metric == 'rel_mae':
        val = np.mean(np.abs(pred - true) / (np.abs(true) + 1e-9))
        passed = (threshold is not None) and (val <= threshold)
    else:
        val = 0.0
        passed = False
    return val, passed

def run_non_adaptive(ds: Dict[str, np.ndarray], label: str, biz_fn: Callable, lag_true: int):
    cpu = ds["cpu"]
    biz = biz_fn(cpu, lag_true)
    n = ds["n"]

    print(f"\n  {'─'*70}")
    print(f"  Pattern : {label}  |  Dataset : {ds['name']}")
    print(f"  Biz: mean={biz.mean():.1f}  std={biz.std():.1f}")

    # Для каждого таргета определяем лаг, r*, rel_std
    results = []
    print("\n  Target   |  lag  |   r*   | rel_std | model      | metric  |  value  | PASS?")
    print("  ---------+-------+--------+---------+------------+---------+---------+------")
    for key in TARGET_KEYS:
        # Определяем лаг и r* на обучающей выборке
        split = int(TRAIN_RATIO * n)
        biz_train = biz[:split]
        sys_train = ds[key][:split]
        lag, r = find_lag_and_corr(biz_train, sys_train)
        # Относительная дисперсия на всей выборке
        mean_val = ds[key].mean()
        std_val = ds[key].std()
        rel_std = std_val / (mean_val + 1e-9)

        # Выбираем модель и метрику
        model_type, eval_metric, threshold, param_grid = select_model_and_metric(r, rel_std)

        # Строим признаки (всегда полные)
        X = build_features_full(biz, ds, lag)
        y = ds[key][LOOKBACK:n-1]

        # Обучаем
        pred, y_test = train_model(X, y, model_type, param_grid)
        value, passed = evaluate(pred, y_test, eval_metric, threshold)

        # Для красивого вывода
        model_str = model_type[:10] if len(model_type) <= 10 else model_type[:7]+'...'
        metric_str = eval_metric[:7]
        pass_str = 'YES' if passed else ('NO' if threshold is not None else '---')
        print(f"  {key:<8} | {lag:3d}   | {r:6.3f} | {rel_std:7.4f} | {model_str:10s} | {metric_str:7s} | {value:7.4f} | {pass_str:>4}")

        results.append((key, r, rel_std, model_type, eval_metric, value, passed))

    return results

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

    print("=" * 80)
    print("  NON-ADAPTIVE (always full features) with MODEL SELECTION")
    print("  Business metric: _cpu_to_proxy (only CPU)")
    print(f"  Embedded lag: {args.lag}step = {args.lag*STEP_SECONDS//60}min")
    print("  Model selection based on r* and rel_std (see table)")
    print("=" * 80)

    for ds in datasets:
        print(f"\n{'━'*80}\n  Dataset: {ds['name']}\n{'━'*80}")
        for pname, (label, fn) in to_run:
            run_non_adaptive(ds, label, fn, args.lag)

if __name__ == "__main__":
    main()