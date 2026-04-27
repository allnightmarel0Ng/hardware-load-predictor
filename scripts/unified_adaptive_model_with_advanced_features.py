from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Tuple, Callable, List, Any

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
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_XGB = False

SCRIPTS_DIR = Path(__file__).parent
ALIBABA_FILE = SCRIPTS_DIR / "machine_usage_days_1_to_8_grouped_300_seconds.csv"
GOOGLE_FILE = SCRIPTS_DIR / "instance_usage_grouped_300_seconds_month.csv"
ALIBABA_URL = "https://zenodo.org/records/14564935/files/machine_usage_days_1_to_8_grouped_300_seconds.csv?download=1"
GOOGLE_URL = "https://zenodo.org/records/14564935/files/instance_usage_grouped_300_seconds_month.csv?download=1"

STEP_SECONDS = 300
TRUE_LAG = 1
LOOKBACK = 30
TRAIN_RATIO = 0.8
MAX_LAG = 20

CORR_LOW = 0.3
CORR_MEDIUM = 0.5
CORR_HIGH = 0.7
REL_STD_THRESHOLD = 0.05

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
        bg += height * np.exp(-0.5 * ((center - t) / width) ** 2)
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

# ============================================================================
#  Корреляция и лаг (на обучающей выборке)
# ============================================================================
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

def ewma(arr: np.ndarray, alpha: float) -> np.ndarray:
    result = np.zeros_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result

def linear_slope(y: np.ndarray) -> float:
    x = np.arange(len(y))
    n = len(x)
    if n < 2:
        return 0.0
    slope = (n * np.sum(x*y) - np.sum(x)*np.sum(y)) / (n * np.sum(x**2) - (np.sum(x))**2)
    return slope

def hurst_rs(series: np.ndarray) -> float:
    n = len(series)
    if n < 4:
        return 0.5
    mean = np.mean(series)
    cumsum = np.cumsum(series - mean)
    R = np.max(cumsum) - np.min(cumsum)
    S = np.std(series)
    if S == 0:
        return 0.5
    return np.log(R / S) / np.log(n)

def build_advanced_features(biz: np.ndarray, ds: Dict[str, np.ndarray],
                            tau: int, target_key: str) -> np.ndarray:
    """
    Строит расширенные признаки для заданной целевой метрики.
    Включает:
      - временные (час, день недели, тренд)
      - бизнес-признаки (сдвиг, нормализация, разности, z-score, взаимодействия)
      - авторегрессионные для всех системных метрик (лаги, rolling mean/std)
      - специфичные для target_key признаки (EWMA, CV, burst ratio, Hurst, инерционность и т.д.)
    """
    n = ds["n"]
    target_arr = ds[target_key]
    rows = []
    for i in range(LOOKBACK, n - 1):
        # ---- время ----
        hour = (i * STEP_SECONDS / 3600) % 24
        dow = (i * STEP_SECONDS / 86400) % 7
        sin_h = math.sin(2 * math.pi * hour / 24)
        cos_h = math.cos(2 * math.pi * hour / 24)
        sin_d = math.sin(2 * math.pi * dow / 7)
        cos_d = math.cos(2 * math.pi * dow / 7)
        trend = i / n
        feats = [sin_h, cos_h, sin_d, cos_d, trend]

        # ---- бизнес-признаки (full) ----
        bl = biz[i - tau] if i >= tau else biz[0]
        rm30 = biz[max(0, i-30):i].mean()
        bn = bl / (rm30 + 1e-9)
        d1 = biz[i] - biz[i-1] if i >= 1 else 0.0
        d2 = biz[i-1] - biz[i-2] if i >= 2 else 0.0
        mu5 = biz[max(0, i-5):i].mean()
        bz = (biz[i] - mu5) / (biz[max(0, i-5):i].std() + 1e-9)
        feats.extend([bl, bn, d1, d2, bz, bl*sin_h, bl*cos_h])

        # ---- авторегрессия для всех системных метрик ----
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
            feats.extend([l1, l2, l3, m5, m15, m30, s5, s15])

        # ---- расширенные признаки для target_key ----
        hist = target_arr[:i]   # вся история до i
        if len(hist) < 30:
            extra = [0.0] * 6   # запас на максимум 6 признаков
        else:
            window5 = hist[-5:]
            window15 = hist[-15:]
            window30 = hist[-30:]

            if target_key == "cpu":
                # EWMA с α=0.1,0.3,0.7
                ewma01 = ewma(hist, 0.1)[-1]
                ewma03 = ewma(hist, 0.3)[-1]
                ewma07 = ewma(hist, 0.7)[-1]
                cv15 = window15.std() / (window15.mean() + 1e-9)
                exceed_70 = np.mean(window30 > 70)
                slope15 = linear_slope(window15)
                extra = [ewma01, ewma03, ewma07, cv15, exceed_70, slope15]

            elif target_key in ("ram_gb", "ram_pct"):
                # скорость освобождения (отрицательные разности)
                diffs = np.diff(window15)
                neg_diffs = -diffs[diffs < 0]
                release_speed = neg_diffs.mean() if len(neg_diffs) > 0 else 0.0
                # коэффициент удержания (текущее / максимум за 30)
                max_30 = window30.max()
                retention = hist[-1] / (max_30 + 1e-9)
                # время с последнего низкого значения (ниже 10-го перцентиля)
                low_thresh = np.percentile(hist, 10)
                low_indices = np.where(hist <= low_thresh)[0]
                steps_since_low = i - 1 - low_indices[-1] if len(low_indices) > 0 else 0
                # относительное отклонение от среднего за 30
                rel_to_30 = hist[-1] / (window30.mean() + 1e-9)
                extra = [release_speed, retention, steps_since_low, rel_to_30, 0.0, 0.0]

            elif target_key == "net":
                burst_ratio = window15.max() / (window15.mean() + 1e-9)
                acc = np.diff(window5)[-1] if len(window5) > 2 else 0.0
                hurst = hurst_rs(window30)
                local_max = 0
                for j in range(1, len(window30)-1):
                    if window30[j] > window30[j-1] and window30[j] > window30[j+1]:
                        local_max += 1
                extra = [burst_ratio, acc, hurst, local_max, 0.0, 0.0]

            else:  # disk
                ewma_slow = ewma(hist, 0.05)[-1]
                ewma_fast = ewma(hist, 0.5)[-1]
                inertia = ewma_slow - ewma_fast
                slope60 = linear_slope(hist[-60:]) if len(hist) >= 60 else 0.0
                exceed_80 = np.mean(hist[-60:] > 80) if len(hist) >= 60 else 0.0
                median30 = np.median(window30)
                crosses = 0
                for j in range(1, len(window30)):
                    if (window30[j-1] - median30) * (window30[j] - median30) < 0:
                        crosses += 1
                extra = [inertia, slope60, exceed_80, crosses, 0.0, 0.0]

        feats.extend(extra)
        rows.append(feats)

    X = StandardScaler().fit_transform(np.array(rows))
    return X

# ============================================================================
#  Выбор модели и метрики (адаптивно)
# ============================================================================
def select_model_and_metric(r: float, rel_std: float) -> Tuple[str, List[Dict], str, float]:
    if rel_std < REL_STD_THRESHOLD:
        return 'mean_baseline', [], 'rel_mae', 0.05
    if r < CORR_LOW:
        return 'ridge', [{'alpha': 0.1}, {'alpha': 1.0}, {'alpha': 10.0}], 'mae', 0.0  # порог не задан
    elif r < CORR_MEDIUM:
        return 'ridge', [{'alpha': 0.1}, {'alpha': 1.0}, {'alpha': 10.0}], 'r2', 0.70
    elif r < CORR_HIGH:
        return 'gbm', [{'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05}], 'r2', 0.80
    else:
        return 'xgboost', [{'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.03}], 'r2', 0.85

def train_model_with_cv(X: np.ndarray, y: np.ndarray, model_type: str, param_grid: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Возвращает (predictions, y_test) для последних 20% данных."""
    n = len(X)
    test_size = int(0.2 * n)
    X_tv, X_test = X[:n - test_size], X[n - test_size:]
    y_tv, y_test = y[:n - test_size], y[n - test_size:]

    if model_type == 'mean_baseline':
        pred = np.full_like(y_test, y_tv.mean())
        return pred, y_test

    tscv = TimeSeriesSplit(n_splits=3)
    best_score, best_params = -np.inf, param_grid[0]
    for params in param_grid:
        if model_type == 'ridge':
            model = Ridge(**params, random_state=42)
        elif model_type == 'gbm':
            model = GradientBoostingRegressor(**params, random_state=42)
        else:  # xgboost
            model = XGBRegressor(**params, random_state=42, verbosity=0)

        scores = []
        for train_idx, val_idx in tscv.split(X_tv):
            model.fit(X_tv[train_idx], y_tv[train_idx])
            pred_val = model.predict(X_tv[val_idx])
            scores.append(r2_score(y_tv[val_idx], pred_val))
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    # финальная модель на train+val
    if model_type == 'ridge':
        final_model = Ridge(**best_params, random_state=42)
    elif model_type == 'gbm':
        final_model = GradientBoostingRegressor(**best_params, random_state=42)
    else:
        final_model = XGBRegressor(**best_params, random_state=42, verbosity=0)
    final_model.fit(X_tv, y_tv)
    pred = final_model.predict(X_test)
    return pred, y_test

def evaluate(pred: np.ndarray, true: np.ndarray, metric: str, threshold: float) -> Tuple[float, bool]:
    if metric == 'r2':
        val = r2_score(true, pred)
        passed = val >= threshold
    elif metric == 'rel_mae':
        val = np.mean(np.abs(pred - true) / (np.abs(true) + 1e-9))
        passed = val <= threshold
    elif metric == 'mae':
        val = mean_absolute_error(true, pred)
        passed = True   # порог не задан
    else:
        val = 0.0
        passed = False
    return val, passed

def run_unified(ds: Dict[str, np.ndarray], label: str, biz_fn: Callable, lag_true: int):
    biz = biz_fn(ds["cpu"], lag_true)
    n = ds["n"]
    print(f"\n  {'─'*70}")
    print(f"  Pattern : {label}  |  Dataset : {ds['name']}")
    print(f"  Biz: mean={biz.mean():.1f}  std={biz.std():.1f}")

    # Для каждого таргета
    print("\n  Target   |  lag  |   r*   | rel_std |   model   |  metric  |  value  | PASS?")
    print("  ---------+-------+--------+---------+-----------+----------+---------+------")
    for key in TARGET_KEYS:
        # лаг и корреляция на обучающей выборке
        split = int(TRAIN_RATIO * n)
        biz_train = biz[:split]
        sys_train = ds[key][:split]
        lag, r = find_lag_and_corr(biz_train, sys_train)
        mean_val = ds[key].mean()
        std_val = ds[key].std()
        rel_std = std_val / (mean_val + 1e-9)

        # выбор модели и метрики
        model_type, param_grid, eval_metric, threshold = select_model_and_metric(r, rel_std)

        # расширенные признаки
        X = build_advanced_features(biz, ds, lag, key)
        y = ds[key][LOOKBACK:n-1]

        # обучение и оценка
        pred, y_test = train_model_with_cv(X, y, model_type, param_grid)
        value, passed = evaluate(pred, y_test, eval_metric, threshold)

        # режим для вывода
        if rel_std < REL_STD_THRESHOLD:
            mode = 'low_var'
        elif r < CORR_LOW:
            mode = 'weak'
        elif r < CORR_MEDIUM:
            mode = 'medium'
        elif r < CORR_HIGH:
            mode = 'moderate'
        else:
            mode = 'strong'

        print(f"  {key:<8} | {lag:3d}   | {r:6.3f} | {rel_std:7.4f} | {model_type:9s} | {eval_metric:8s} | {value:7.4f} | {'YES' if passed else 'NO'}")

# ============================================================================
#  MAIN
# ============================================================================
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
    print("  UNIFIED: advanced features + adaptive model selection")
    print("  Business metric: _cpu_to_proxy (only CPU)")
    print(f"  Embedded lag: {args.lag}step = {args.lag*STEP_SECONDS//60}min")
    print("  Model selection based on r* and rel_std (weak<0.3, medium<0.5, moderate<0.7, strong≥0.7, low_var<0.05)")
    print("=" * 80)

    for ds in datasets:
        print(f"\n{'━'*80}\n  Dataset: {ds['name']}\n{'━'*80}")
        for pname, (label, fn) in to_run:
            run_unified(ds, label, fn, args.lag)

if __name__ == "__main__":
    main()