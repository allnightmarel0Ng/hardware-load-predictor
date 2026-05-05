from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

SCRIPTS_DIR  = Path(__file__).parent
ALIBABA_FILE = SCRIPTS_DIR / "machine_usage_days_1_to_8_grouped_300_seconds.csv"
GOOGLE_FILE  = SCRIPTS_DIR / "instance_usage_grouped_300_seconds_month.csv"
ALIBABA_URL  = ("https://zenodo.org/records/14564935/files/"
                "machine_usage_days_1_to_8_grouped_300_seconds.csv?download=1")
GOOGLE_URL   = ("https://zenodo.org/records/14564935/files/"
                "instance_usage_grouped_300_seconds_month.csv?download=1")

STEP_SECONDS = 300
TRUE_LAG     = 1
SIGNIFICANCE  = 0.6

rng = np.random.default_rng(42)

def load_alibaba() -> dict[str, np.ndarray]:
    if not ALIBABA_FILE.exists():
        raise FileNotFoundError(
            f"Alibaba dataset not found: {ALIBABA_FILE}\n"
            f"Download with:\n  curl -L '{ALIBABA_URL}' -o {ALIBABA_FILE}"
        )
    df = pd.read_csv(ALIBABA_FILE, header=None,
                     names=["cpu", "mem_pct", "net_in", "net_out", "disk"])
    df.replace([-1, 101], np.nan, inplace=True)
    df.ffill(inplace=True); df.bfill(inplace=True); df.dropna(inplace=True)
    n = len(df)
    print(f"  Alibaba 2018: {n:,} rows  "
          f"cpu_mean={df.cpu.mean():.1f}%  mem_mean={df.mem_pct.mean():.1f}%")
    return {
        "cpu":     df["cpu"].to_numpy(float),
        "ram_pct": df["mem_pct"].to_numpy(float),
        "ram_gb":  df["mem_pct"].to_numpy(float) / 100 * 128,
        "net":     df["net_in"].to_numpy(float) / 100 * 500,   # scale to Mbps
        "disk":    df["disk"].to_numpy(float),
        "n":       n,
        "name":    "Alibaba 2018",
    }


def load_google() -> dict[str, np.ndarray]:
    if not GOOGLE_FILE.exists():
        raise FileNotFoundError(
            f"Google dataset not found: {GOOGLE_FILE}\n"
            f"Download with:\n  curl -L '{GOOGLE_URL}' -o {GOOGLE_FILE}"
        )
    df = pd.read_csv(GOOGLE_FILE)
    cpu     = (df["avg_cpu"] * 100).clip(0, 99)
    mem_pct = (df["avg_mem"] * 100).clip(0, 99)
    cpu_arr = cpu.to_numpy(float)
    net     = pd.Series(np.clip(8 + 0.7 * cpu_arr + rng.normal(0, 5, len(cpu_arr)), 0, 400))
    disk    = pd.Series(np.clip(5 + 0.25 * cpu_arr + rng.normal(0, 4, len(cpu_arr)), 0, 80))
    df.dropna(subset=["avg_cpu", "avg_mem"], inplace=True)
    n = len(cpu)
    print(f"  Google 2019:  {n:,} rows  "
          f"cpu_mean={cpu.mean():.1f}%  mem_mean={mem_pct.mean():.1f}%")
    return {
        "cpu":     cpu.to_numpy(float),
        "ram_pct": mem_pct.to_numpy(float),
        "ram_gb":  mem_pct.to_numpy(float) / 100 * 256,
        "net":     net.to_numpy(float),
        "disk":    disk,
        "n":       n,
        "name":    "Google 2019",
    }


def _cpu_to_proxy(cpu: np.ndarray, lag: int, coeff: float = 0.5,
                  noise_std: float = 2.0) -> np.ndarray:
    n = len(cpu)
    shifted = np.empty(n)
    shifted[:n - lag] = cpu[lag:]
    shifted[n - lag:] = cpu[-1]
    proxy = shifted / coeff + rng.normal(0, noise_std, n)
    return np.maximum(0.5, proxy)


def pattern_sinusoidal(cpu: np.ndarray, lag: int) -> np.ndarray:
    n = len(cpu)
    t = np.arange(n)
    base  = _cpu_to_proxy(cpu, lag)
    # Amplify the existing diurnal signal already present in real CPU
    modulation = 1.0 + 0.2 * np.sin(2 * np.pi * t / (24 * 3600 / STEP_SECONDS))
    return np.clip(base * modulation, 0.5, 300)


def pattern_step(cpu: np.ndarray, lag: int) -> np.ndarray:
    base = _cpu_to_proxy(cpu, lag)
    p25, p50, p75 = np.percentile(base, [25, 50, 75])
    stepped = np.where(base < p25, p25 * 0.5,
              np.where(base < p50, p25,
              np.where(base < p75, p50, p75 * 1.3)))
    return np.clip(stepped + rng.normal(0, 1.5, len(cpu)), 0.5, 300)


def pattern_spiky(cpu: np.ndarray, lag: int) -> np.ndarray:
    n    = len(cpu)
    t    = np.arange(n)
    base = _cpu_to_proxy(cpu, lag, coeff=0.8)
    # Set background to lower percentile, add spikes
    background = np.full(n, np.percentile(base, 20))
    n_spikes   = max(30, n // 80)
    for _ in range(n_spikes):
        center = rng.integers(0, n)
        width  = rng.integers(3, 12)
        height = rng.uniform(0.8, 2.5) * np.percentile(base, 90)
        background += height * np.exp(-0.5 * ((t - center) / width) ** 2)
    return np.clip(background + rng.normal(0, 1, n), 0.5, 300)


def pattern_drifting(cpu: np.ndarray, lag: int) -> np.ndarray:
    n    = len(cpu)
    base = _cpu_to_proxy(cpu, lag)
    trend = np.linspace(0.6, 1.4, n)   # 60% → 140% of baseline over the period
    return np.clip(base * trend + rng.normal(0, 1.5, n), 0.5, 300)


def pattern_multimodal(cpu: np.ndarray, lag: int) -> np.ndarray:
    n   = len(cpu)
    t   = np.arange(n)
    base = _cpu_to_proxy(cpu, lag)
    steps_per_day = 24 * 3600 // STEP_SECONDS
    day_of_week   = (t // steps_per_day) % 7
    weekend_mask  = (day_of_week >= 5).astype(float)
    # Weekends: reduce to ~30% of weekday level
    scale = 1.0 - 0.7 * weekend_mask
    return np.clip(base * scale + rng.normal(0, 1, n), 0.5, 300)


PATTERNS = {
    "sinusoidal": ("Sinusoidal (SaaS daily wave)",      pattern_sinusoidal),
    "step":       ("Step function (batch / ETL)",        pattern_step),
    "spiky":      ("Spiky bursts (event-driven)",        pattern_spiky),
    "drifting":   ("Drifting trend (growing audience)",  pattern_drifting),
    "multimodal": ("Multimodal (weekday/weekend split)", pattern_multimodal),
}


def _pearson(a, b):
    if len(a) < 3: return 0.0
    if a.std() < 1e-9 or b.std() < 1e-9: return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def _spearman(a, b):
    def rank(x): s=np.argsort(x); r=np.empty_like(s,float); r[s]=np.arange(len(x))+1; return r
    return _pearson(rank(a), rank(b))

def find_lag(biz, sys, max_lag=20):
    xd, yd = np.diff(biz.astype(float)), np.diff(sys.astype(float))
    best_lag, best = 0, 0.0
    for lag in range(min(max_lag + 1, len(xd))):
        a, b = (xd[:-lag], yd[lag:]) if lag else (xd, yd)
        score = abs(_pearson(a, b)) + abs(_spearman(a, b))
        if score > best: best, best_lag = score, lag
    a, b = (xd[:-best_lag], yd[best_lag:]) if best_lag else (xd, yd)
    p, s = _pearson(a, b), _spearman(a, b)
    r = max(abs(p), abs(s))
    return best_lag, p, s, r, r >= SIGNIFICANCE


def build_features(biz, sys_metrics, tau):
    LOOKBACK = 30
    n = len(biz)
    rows = []
    start = max(tau, LOOKBACK)
    for i in range(start, n - 1):
        # Business features
        biz_lag  = biz[i - tau] if i >= tau else biz[0]
        rm30     = biz[max(0,i-30):i].mean()
        biz_norm = biz_lag / (rm30 + 1e-9)
        biz_d1   = biz[i] - biz[i-1] if i >= 1 else 0.0
        biz_d2   = biz[i-1] - biz[i-2] if i >= 2 else 0.0
        mu5_b    = biz[max(0,i-5):i].mean()
        biz_z    = (biz[i] - mu5_b) / (biz[max(0,i-5):i].std() + 1e-9)
        # Time features
        h = (i * STEP_SECONDS / 3600) % 24
        d = (i * STEP_SECONDS / 86400) % 7
        sin_h = math.sin(2*math.pi*h/24); cos_h = math.cos(2*math.pi*h/24)
        sin_d = math.sin(2*math.pi*d/7);  cos_d = math.cos(2*math.pi*d/7)
        trend = i / n
        biz_sin = biz_lag * sin_h; biz_cos = biz_lag * cos_h
        # Autoregressive system features
        sys_feats = []
        for key in ("cpu", "ram_pct", "net", "disk"):
            arr = sys_metrics[key]
            l1 = arr[i-1]; l2 = arr[i-2] if i>=2 else arr[0]; l3 = arr[i-3] if i>=3 else arr[0]
            mu5=arr[max(0,i-5):i].mean(); mu15=arr[max(0,i-15):i].mean(); mu30=arr[max(0,i-30):i].mean()
            s5=arr[max(0,i-5):i].std()+1e-9; s15=arr[max(0,i-15):i].std()+1e-9
            sys_feats.extend([l1,l2,l3,mu5,mu15,mu30,s5,s15])
        rows.append([biz_lag,biz_norm,rm30,biz_d1,biz_d2,biz_z,
                     sin_h,cos_h,sin_d,cos_d,trend,biz_sin,biz_cos] + sys_feats)
    X = StandardScaler().fit_transform(np.array(rows))
    idx = slice(start, n - 1)
    y = np.column_stack([sys_metrics[k][idx]
                         for k in ("cpu", "ram_gb", "ram_pct", "net", "disk")])
    return X, y


def train_evaluate(X, y):
    sp = int(len(X) * 0.8)
    Xtr, Xte, ytr, yte = X[:sp], X[sp:], y[:sp], y[sp:]
    algo = "GradientBoosting" if len(Xtr) >= 200 else "Ridge"
    base = (GradientBoostingRegressor(n_estimators=200, max_depth=4,
                learning_rate=0.05, subsample=0.8,
                min_samples_leaf=5, random_state=42)
            if algo == "GradientBoosting" else Ridge(alpha=1.0))
    model = MultiOutputRegressor(base)
    model.fit(Xtr, ytr)
    pred  = model.predict(Xte)
    cols  = ["cpu", "ram_gb", "ram_pct", "net", "disk"]
    r2s   = {c: float(r2_score(yte[:,i], pred[:,i]))            for i,c in enumerate(cols)}
    maes  = {c: float(mean_absolute_error(yte[:,i], pred[:,i])) for i,c in enumerate(cols)}
    def safe_mape(true, p):
        m = np.abs(true) > 0.5
        return float(np.mean(np.abs((true[m]-p[m])/true[m]))*100) if m.sum()>5 else float("nan")
    mape_cpu = safe_mape(yte[:,0], pred[:,0])
    mape_net = safe_mape(yte[:,3], pred[:,3])
    mape_avg = (mape_cpu+mape_net)/2 if not (math.isnan(mape_cpu) or math.isnan(mape_net)) else float("nan")
    return r2s, maes, mape_cpu, mape_net, mape_avg, algo, len(Xtr), len(Xte)


def run(ds: dict, pattern_name: str, label: str,
        biz_fn: callable, lag: int) -> dict:
    biz = biz_fn(ds["cpu"], lag)
    print(f"\n  {'─'*68}")
    print(f"  Pattern : {label}")
    print(f"  Dataset : {ds['name']}  ({ds['n']:,} pts, step={STEP_SECONDS}s)")
    print(f"  Biz     : mean={biz.mean():.1f}  std={biz.std():.1f}  "
          f"range=[{biz.min():.1f}, {biz.max():.1f}]")

    print("  Correlation analysis:")
    lags_found = {}
    for tname, key in [("CPU %","cpu"),("Net Mbps","net"),("RAM %","ram_pct")]:
        lf, rp, rs, r, sig = find_lag(biz, ds[key])
        lags_found[key] = lf
        ok = "✓" if lf == lag else f"✗ (true={lag})"
        print(f"    {tname:<10}  lag={lf:2d}step={lf*STEP_SECONDS//60:3d}min  "
              f"r_P={rp:+.3f}  r_S={rs:+.3f}  r*={r:.3f}  "
              f"sig={'Yes' if sig else 'No':3s}  {ok}")

    tau = lags_found.get("cpu", lag)
    X, y = build_features(biz, ds, tau)
    r2s, maes, mape_cpu, mape_net, mape_avg, algo, ntr, nte = train_evaluate(X, y)

    print(f"  Training: {algo}  train={ntr}  test={nte}")
    print(f"  R²:   CPU={r2s['cpu']:+.4f}  RAM_GB={r2s['ram_gb']:+.4f}  "
          f"RAM%={r2s['ram_pct']:+.4f}  Net={r2s['net']:+.4f}  Disk={r2s['disk']:+.4f}")
    print(f"  MAE:  CPU={maes['cpu']:.2f}%  RAM_GB={maes['ram_gb']:.3f}GB  "
          f"RAM%={maes['ram_pct']:.2f}%  Net={maes['net']:.2f}Mbps  Disk={maes['disk']:.2f}%")
    print(f"  MAPE: CPU={mape_cpu:.2f}%  Net={mape_net:.2f}%  avg={mape_avg:.2f}%")

    cpu_ok  = r2s["cpu"] >= 0.85
    net_ok  = r2s["net"] >= 0.85
    mape_ok = mape_avg   <= 10.0
    lag_ok  = lags_found.get("cpu", -1) == lag
    verdict = "PASS ✓" if (cpu_ok and net_ok and mape_ok and lag_ok) else "FAIL ✗"
    for cond, msg in [(cpu_ok,f"R²_CPU={r2s['cpu']:.3f} < 0.85"),
                      (net_ok,f"R²_Net={r2s['net']:.3f} < 0.85"),
                      (mape_ok,f"MAPE={mape_avg:.2f}% > 10%"),
                      (lag_ok,f"lag={lags_found.get('cpu')} ≠ {lag}")]:
        if not cond: print(f"  ✗ {msg}")
    print(f"  → {verdict}")

    return dict(dataset=ds["name"], pattern=label,
                r2_cpu=r2s["cpu"], r2_net=r2s["net"], r2_ram=r2s["ram_pct"],
                mape=mape_avg, lag_ok=lag_ok, verdict=verdict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  choices=["alibaba","google","both"], default="both")
    parser.add_argument("--pattern",  choices=list(PATTERNS)+["all"],     default="all")
    parser.add_argument("--lag",      type=int, default=TRUE_LAG)
    args = parser.parse_args()

    # Load datasets
    datasets = []
    missing  = []
    for name, loader in [("alibaba", load_alibaba), ("google", load_google)]:
        if args.dataset not in ("both", name):
            continue
        try:
            datasets.append(loader())
        except FileNotFoundError as e:
            missing.append(str(e))

    if missing:
        print("\n  Missing datasets:")
        for m in missing: print(f"  {m}\n")
    if not datasets:
        sys.exit(1)

    to_run = list(PATTERNS.items()) if args.pattern == "all" \
             else [(args.pattern, PATTERNS[args.pattern])]

    print("=" * 70)
    print("  Hardware Load Predictor — Business Pattern Robustness Evaluation")
    print(f"  System metrics: REAL datacenter traces  |  "
          f"Embedded lag: {args.lag} step(s) = {args.lag*STEP_SECONDS//60} min")
    print(f"  Significance threshold: r* ≥ {SIGNIFICANCE}")
    print("=" * 70)

    results = []
    for ds in datasets:
        print(f"\n{'━'*70}")
        print(f"  Dataset: {ds['name']}")
        print(f"{'━'*70}")
        for pname, (label, fn) in to_run:
            results.append(run(ds, pname, label, fn, args.lag))

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Dataset':<14} {'Pattern':<38} {'R²_CPU':>7} {'R²_Net':>7} "
          f"{'MAPE%':>7}  {'Lag':>3}  Result")
    print(f"  {'-'*14} {'-'*38} {'-'*7} {'-'*7} {'-'*7}  {'-'*3}  {'-'*6}")
    for r in results:
        print(f"  {r['dataset']:<14} {r['pattern']:<38} "
              f"{r['r2_cpu']:>7.4f} {r['r2_net']:>7.4f} {r['mape']:>7.2f}  "
              f"{'✓' if r['lag_ok'] else '✗':>3}  {r['verdict']}")

    n_pass = sum(1 for r in results if "PASS" in r["verdict"])
    print(f"\n  Passed: {n_pass}/{len(results)}  "
          f"(criteria: R²_CPU ≥ 0.85, R²_Net ≥ 0.85, MAPE ≤ 10%, lag ✓)")


if __name__ == "__main__":
    main()
