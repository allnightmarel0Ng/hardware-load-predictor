"""
evaluate_on_real_data.py
─────────────────────────────────────────────────────────────────────────────
Evaluates the hardware-load-predictor ML pipeline against real production
datacenter traces instead of synthetic stub data.

Datasets used (both freely downloadable, CC-BY 4.0):

  [A] Alibaba 2018 machine usage — 8 days, 300-second granularity
      cpu_util_percent, mem_util_percent, net_in, net_out
      Download (202 KB):
      https://zenodo.org/records/14564935/files/machine_usage_days_1_to_8_grouped_300_seconds.csv?download=1

  [B] Google 2019 instance usage — 1 month, 300-second granularity
      avg_cpu, avg_mem, avg_assigned_mem, avg_cycles_per_instruction
      Download (607 KB):
      https://zenodo.org/records/14564935/files/instance_usage_grouped_300_seconds_month.csv?download=1

Why these datasets?
  Both are pre-processed aggregates of real production clusters (Alibaba Borg
  co-location cluster; Google's internal Borg scheduler). They contain real
  noise, daily seasonality, load spikes, and multi-modal CPU distributions
  that synthetic sine-wave data cannot replicate.

Why no "business metric" in these files?
  No public dataset pairs application-level business metrics (orders/min,
  HTTP RPS) with server resource utilisation simultaneously — companies do
  not release both together for confidentiality reasons. The standard
  academic approach (used in workload prediction literature) is to construct
  a synthetic business proxy from the real system metrics, then evaluate
  whether the model recovers the known lag and predicts the real values.

  Business metric construction used here:
      business(t) = cpu_util(t - LAG_MINUTES) / COEFF + noise
  This inverts the ground-truth relationship, giving a realistic input that
  is causally prior to the CPU signal by a known lag. Evaluation is then
  against REAL cpu/mem/net values — not another synthetic series.

Usage:
  1. Download the two CSV files into this scripts/ directory (or pass --data-dir)
  2. pip install numpy scikit-learn pandas joblib
  3. python scripts/evaluate_on_real_data.py

  Or to run only one dataset:
      python scripts/evaluate_on_real_data.py --dataset alibaba
      python scripts/evaluate_on_real_data.py --dataset google
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import NamedTuple

import numpy as np
import pandas as pd

# ── allow running from repo root ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.modules.correlation_analyzer import analyze, CorrelationReport
from app.modules.model_trainer import _build_features, _fit_and_evaluate, _compute_metrics
from app.modules.data_collector import MetricsBundle

# ── constants ─────────────────────────────────────────────────────────────────

ALIBABA_FILE = "machine_usage_days_1_to_8_grouped_300_seconds.csv"
GOOGLE_FILE  = "instance_usage_grouped_300_seconds_month.csv"

ALIBABA_URL = (
    "https://zenodo.org/records/14564935/files/"
    "machine_usage_days_1_to_8_grouped_300_seconds.csv?download=1"
)
GOOGLE_URL = (
    "https://zenodo.org/records/14564935/files/"
    "instance_usage_grouped_300_seconds_month.csv?download=1"
)

STEP_SECONDS = 300   # both datasets are 5-minute intervals
LAG_MINUTES  = 5     # synthetic business proxy lag (in 300-sec steps → 1 step)
BIZ_COEFF    = 0.45  # cpu ≈ biz * COEFF  (realistic: ~45 cpu% per 100 biz units)
BIZ_NOISE_SD = 3.0   # std of Gaussian noise added to business proxy


# ── dataset loaders ───────────────────────────────────────────────────────────

def _load_alibaba(path: Path) -> pd.DataFrame:
    """
    Load Alibaba 2018 machine_usage_days_1_to_8_grouped_300_seconds.csv

    Schema (no header in file):
        cpu_util_percent, mem_util_percent, net_in, net_out, disk_io_percent
    All values [0, 100]; invalid values are -1 or 101 → replaced with NaN.
    """
    print(f"  Loading Alibaba 2018 from {path.name} ...")
    df = pd.read_csv(
        path,
        header=None,
        names=["cpu_util_percent", "mem_util_percent",
               "net_in", "net_out", "disk_io_percent"],
    )
    # Replace sentinel values with NaN, then forward-fill
    df.replace([-1, 101], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.dropna(inplace=True)
    print(f"    {len(df):,} rows after cleaning  "
          f"(cpu mean={df.cpu_util_percent.mean():.1f}%  "
          f"mem mean={df.mem_util_percent.mean():.1f}%)")
    return df


def _load_google(path: Path) -> pd.DataFrame:
    """
    Load Google 2019 instance_usage_grouped_300_seconds_month.csv

    Schema (has header):
        avg_cpu [0,1], avg_mem [0,1], avg_assigned_mem [0,1],
        avg_cycles_per_instruction
    Scale avg_cpu and avg_mem to [0, 100] to match Alibaba units.
    """
    print(f"  Loading Google 2019 from {path.name} ...")
    df = pd.read_csv(path)
    # Normalise to [0, 100] scale
    df["cpu_util_percent"] = df["avg_cpu"] * 100.0
    df["mem_util_percent"] = df["avg_mem"] * 100.0
    # Synthesise a network column from cycles_per_instruction (proxy)
    cpi = df["avg_cycles_per_instruction"].clip(0, 10)
    df["net_in"] = (cpi / cpi.max() * 80.0).fillna(0)
    df.dropna(subset=["cpu_util_percent", "mem_util_percent"], inplace=True)
    print(f"    {len(df):,} rows after cleaning  "
          f"(cpu mean={df.cpu_util_percent.mean():.1f}%  "
          f"mem mean={df.mem_util_percent.mean():.1f}%)")
    return df


# ── business proxy construction ───────────────────────────────────────────────

def _build_business_proxy(cpu: np.ndarray, lag_steps: int, rng: np.random.Generator) -> np.ndarray:
    """
    Construct a synthetic business metric that causally precedes the CPU signal.

    business(t) = cpu(t + lag_steps) / BIZ_COEFF + Gaussian noise

    This means: given business(t), the CPU will follow BIZ_COEFF * business(t)
    after lag_steps × 300 seconds. We evaluate whether the ML pipeline can
    discover this relationship and predict the REAL cpu values.
    """
    n = len(cpu)
    # Shift CPU forward by lag to get the "cause"
    shifted = np.zeros(n)
    shifted[:n - lag_steps] = cpu[lag_steps:]
    shifted[n - lag_steps:] = cpu[-1]   # pad end with last value
    business = shifted / BIZ_COEFF + rng.normal(0, BIZ_NOISE_SD, n)
    return np.maximum(0.0, business)


# ── MetricsBundle builder ─────────────────────────────────────────────────────

def _df_to_bundle(df: pd.DataFrame, dataset_name: str, rng: np.random.Generator) -> MetricsBundle:
    """Convert a cleaned dataframe to a MetricsBundle for the pipeline."""
    n = len(df)
    base_ts = datetime(2019, 5, 1)   # arbitrary start date

    cpu_arr = df["cpu_util_percent"].to_numpy(dtype=float)
    mem_arr = df["mem_util_percent"].to_numpy(dtype=float)
    net_arr = df["net_in"].to_numpy(dtype=float)

    # Convert % to GB-like units for RAM (assume 128 GB server, scale accordingly)
    ram_gb = mem_arr / 100.0 * 128.0

    # Scale network to Mbps-like units (0–1000 Mbps range)
    net_mbps = net_arr / 100.0 * 500.0

    # Build business proxy from CPU with a known lag
    lag_steps = LAG_MINUTES   # LAG_MINUTES steps of 300s each = LAG * 5 minutes
    biz_arr = _build_business_proxy(cpu_arr, lag_steps, rng)

    def _series(arr: np.ndarray) -> list[dict]:
        return [
            {"timestamp": base_ts + timedelta(seconds=i * STEP_SECONDS), "value": float(v)}
            for i, v in enumerate(arr)
        ]

    return MetricsBundle(
        business=_series(biz_arr),
        cpu=_series(cpu_arr),
        ram=_series(ram_gb),
        network=_series(net_mbps),
    )


# ── single-dataset evaluation ─────────────────────────────────────────────────

class EvalResult(NamedTuple):
    dataset:     str
    n_rows:      int
    lag_found:   int
    lag_true:    int
    lag_correct: bool
    r2_cpu:      float
    r2_ram:      float
    r2_net:      float
    mae_cpu:     float
    mae_ram:     float
    mae_net:     float
    mape_overall: float
    algorithm:   str


def _evaluate_dataset(name: str, df: pd.DataFrame, rng: np.random.Generator) -> EvalResult:
    print(f"\n{'─'*60}")
    print(f"  Dataset: {name}  ({len(df):,} rows, step={STEP_SECONDS}s)")

    bundle = _df_to_bundle(df, name, rng)

    # ── Step 1: Correlation analysis ──────────────────────────────────────────
    print("  Running correlation analysis ...")
    report = analyze(bundle, max_lag=20)
    print(f"    CPU  lag={report.cpu.lag_minutes:2d} steps  "
          f"pearson={report.cpu.pearson_r:+.3f}  "
          f"spearman={report.cpu.spearman_r:+.3f}  "
          f"significant={report.cpu.is_significant}")
    print(f"    RAM  lag={report.ram.lag_minutes:2d} steps  "
          f"pearson={report.ram.pearson_r:+.3f}  "
          f"spearman={report.ram.spearman_r:+.3f}  "
          f"significant={report.ram.is_significant}")
    print(f"    Net  lag={report.network.lag_minutes:2d} steps  "
          f"pearson={report.network.pearson_r:+.3f}  "
          f"spearman={report.network.spearman_r:+.3f}  "
          f"significant={report.network.is_significant}")

    lag_found = report.best_lag()
    lag_correct = (lag_found == LAG_MINUTES)
    print(f"    Best lag found: {lag_found}  (true: {LAG_MINUTES})  "
          f"{'✓ CORRECT' if lag_correct else f'✗ off by {abs(lag_found - LAG_MINUTES)}'}")

    # ── Step 2: Feature engineering + training ────────────────────────────────
    print("  Training model ...")
    X, y = _build_features(bundle, lag=lag_found)
    print(f"    Feature matrix: X={X.shape}  y={y.shape}")

    artifact = f"/tmp/eval_{name.lower().replace(' ', '_')}_model.joblib"
    os.makedirs("/tmp", exist_ok=True)
    params, metrics, algo = _fit_and_evaluate(X, y, artifact)

    print(f"    Algorithm: {algo}")
    print(f"    R²   cpu={metrics['r2_cpu']:.4f}  "
          f"ram={metrics['r2_ram']:.4f}  "
          f"net={metrics['r2_net']:.4f}")
    print(f"    MAE  cpu={metrics['mae_cpu']:.3f}%  "
          f"ram={metrics['mae_ram']:.3f} GB  "
          f"net={metrics['mae_net']:.3f} Mbps")
    print(f"    MAPE overall={metrics['mape_overall']:.2f}%")

    return EvalResult(
        dataset=name,
        n_rows=len(df),
        lag_found=lag_found,
        lag_true=LAG_MINUTES,
        lag_correct=lag_correct,
        r2_cpu=metrics["r2_cpu"],
        r2_ram=metrics["r2_ram"],
        r2_net=metrics["r2_net"],
        mae_cpu=metrics["mae_cpu"],
        mae_ram=metrics["mae_ram"],
        mae_net=metrics["mae_net"],
        mape_overall=metrics["mape_overall"],
        algorithm=algo,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate ML pipeline on real datacenter traces")
    parser.add_argument("--data-dir", default="scripts",
                        help="Directory containing the downloaded CSV files")
    parser.add_argument("--dataset", choices=["alibaba", "google", "both"], default="both")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    rng = np.random.default_rng(42)
    results: list[EvalResult] = []

    print("=" * 60)
    print("  Hardware Load Predictor — Real Data Evaluation")
    print("=" * 60)

    # ── Alibaba 2018 ──────────────────────────────────────────────────────────
    if args.dataset in ("alibaba", "both"):
        alibaba_path = data_dir / ALIBABA_FILE
        if not alibaba_path.exists():
            print(f"\n  [ALIBABA] File not found: {alibaba_path}")
            print(f"  Download with:")
            print(f"    curl -L '{ALIBABA_URL}' -o {alibaba_path}")
        else:
            df_ali = _load_alibaba(alibaba_path)
            results.append(_evaluate_dataset("Alibaba 2018", df_ali, rng))

    # ── Google 2019 ───────────────────────────────────────────────────────────
    if args.dataset in ("google", "both"):
        google_path = data_dir / GOOGLE_FILE
        if not google_path.exists():
            print(f"\n  [GOOGLE] File not found: {google_path}")
            print(f"  Download with:")
            print(f"    curl -L '{GOOGLE_URL}' -o {google_path}")
        else:
            df_goo = _load_google(google_path)
            results.append(_evaluate_dataset("Google 2019", df_goo, rng))

    # ── Summary table ─────────────────────────────────────────────────────────
    if not results:
        print("\n  No datasets found. Please download the files and re-run.")
        print(f"\n  Alibaba 2018 (202 KB):\n    curl -L '{ALIBABA_URL}' -o scripts/{ALIBABA_FILE}")
        print(f"\n  Google 2019 (607 KB):\n    curl -L '{GOOGLE_URL}' -o scripts/{GOOGLE_FILE}")
        return

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Dataset':<20}  {'Rows':>7}  {'Lag':>8}  {'Algorithm':<20}  "
          f"{'R²_cpu':>8}  {'R²_ram':>8}  {'MAPE%':>7}")
    print(f"  {'-'*20}  {'-'*7}  {'-'*8}  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*7}")
    for r in results:
        lag_str = f"{r.lag_found} {'✓' if r.lag_correct else '✗'}"
        print(f"  {r.dataset:<20}  {r.n_rows:>7,}  {lag_str:>8}  {r.algorithm:<20}  "
              f"{r.r2_cpu:>8.4f}  {r.r2_ram:>8.4f}  {r.mape_overall:>7.2f}")

    print(f"\n  Significance threshold for paper: R² ≥ 0.85, MAPE ≤ 10%")
    for r in results:
        cpu_ok   = r.r2_cpu >= 0.85
        mape_ok  = r.mape_overall <= 10.0
        lag_ok   = r.lag_correct
        overall  = "PASS" if (cpu_ok and mape_ok) else "FAIL"
        print(f"  {r.dataset}: R²_cpu {'✓' if cpu_ok else '✗'}  "
              f"MAPE {'✓' if mape_ok else '✗'}  "
              f"lag_detection {'✓' if lag_ok else '✗'}  → {overall}")


if __name__ == "__main__":
    main()
