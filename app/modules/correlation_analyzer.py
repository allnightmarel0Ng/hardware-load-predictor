"""
Module 3 — Correlation Analyzer
Identifies the time lag at which a business metric best predicts each
system metric, and quantifies the strength of that relationship.

Implementation — two-method parallel CCF:

  1. Pearson cross-correlation function (CCF) swept over lags 0..MAX_LAG.
     Fast (O(n log n) via FFT convolution), gives signed linear correlation.
     Series are first-differenced before CCF to remove trend/non-stationarity,
     preventing spurious correlations from shared drifts.

  2. Spearman rank CCF swept in parallel.
     Captures monotonic non-linear relationships (e.g. logarithmic RAM growth)
     and is robust to outliers/load spikes.

Reported strength = max(|pearson_r|, |spearman_r|) at the optimal lag.
Significance threshold: |r| >= 0.6  (standard convention).

Five targets are analysed independently:
  cpu, ram_gb, ram_percent, network, disk

Behaviour when correlation is not found (best_r < 0.6):
  Training is NOT blocked.  The target is flagged as insignificant and its
  lag is set to 0.  The model will be trained anyway — for a truly
  uncorrelated target the model learns the mean (R² ≈ 0), which is still a
  useful baseline (e.g. RAM is often nearly constant regardless of request
  rate, and "predict mean RAM" is a valid capacity estimate).

  Training is blocked only if the business metric series itself has no
  variance (std ≈ 0), which indicates a broken PromQL formula or a constant
  feed — in that case no lag-detection is meaningful.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from app.modules.data_collector import MetricsBundle

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_LAG_MINUTES:      int   = 60    # search window: 0..60 minute lags
SIGNIFICANCE_THRESHOLD: float = 0.6  # |r| >= 0.6 → significant
MIN_POINTS:           int   = 30    # minimum aligned points for reliable CCF
CONSTANT_STD_THRESHOLD: float = 1e-6  # business series is "constant" below this

TargetName = Literal["cpu", "ram_gb", "ram_percent", "network", "disk"]
ALL_TARGETS: tuple[TargetName, ...] = ("cpu", "ram_gb", "ram_percent", "network", "disk")


# ── Public data types ─────────────────────────────────────────────────────────

@dataclass
class CorrelationResult:
    """
    Holds the discovered lag and correlation coefficients for one
    (business metric → system metric) pair.
    """
    target_metric: str    # "cpu" | "ram_gb" | "ram_percent" | "network" | "disk"
    lag_minutes:   int    # optimal lag in minutes (0 = no detectable lag)
    pearson_r:     float  # Pearson CCF at optimal lag  [-1, 1]
    spearman_r:    float  # Spearman CCF at optimal lag [-1, 1]
    best_r:        float  # max(|pearson_r|, |spearman_r|) — reported strength
    is_significant: bool  # True if best_r >= SIGNIFICANCE_THRESHOLD


@dataclass
class CorrelationReport:
    """
    CCF results for all five system metric targets.

    Key design decision:
      any_significant is INFORMATIONAL only — training proceeds regardless.
      Use per_target_lag() to get the appropriate lag per target;
      insignificant targets get lag=0 (train on current business value).

      Training is blocked upstream only if the business metric is constant
      (zero variance) — checked by is_business_constant.
    """
    cpu:         CorrelationResult
    ram_gb:      CorrelationResult
    ram_percent: CorrelationResult
    network:     CorrelationResult
    disk:        CorrelationResult
    n_points:    int
    # True if the business metric series itself has effectively zero variance.
    # This is the only condition that makes training meaningless.
    is_business_constant: bool = False
    # Computed in __post_init__
    any_significant: bool = field(init=False)

    def __post_init__(self) -> None:
        self.any_significant = any(
            r.is_significant
            for r in [self.cpu, self.ram_gb, self.ram_percent, self.network, self.disk]
        )

    def all_results(self) -> list[CorrelationResult]:
        """All five results in a consistent order."""
        return [self.cpu, self.ram_gb, self.ram_percent, self.network, self.disk]

    def per_target_lag(self) -> dict[str, int]:
        """
        Return the best lag per target.

        For significant targets: the discovered lag.
        For insignificant targets: 0 (train on contemporaneous business value).

        This is the correct input to _build_features when using a single global
        lag.  For multi-lag training (future work), each target column would
        use its own lag independently.
        """
        return {r.target_metric: (r.lag_minutes if r.is_significant else 0)
                for r in self.all_results()}

    def best_lag(self) -> int:
        """
        Return the median lag across all significant targets.
        Falls back to 0 if none are significant (training still proceeds).
        """
        significant = [r for r in self.all_results() if r.is_significant]
        if not significant:
            return 0
        lags = sorted(r.lag_minutes for r in significant)
        return lags[len(lags) // 2]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _extract_values(series: list[dict]) -> np.ndarray:
    return np.array([p["value"] for p in series], dtype=float)


def _first_difference(x: np.ndarray) -> np.ndarray:
    """
    Remove non-stationarity by computing first differences x[t] - x[t-1].
    Result is 1 element shorter than input.
    """
    return np.diff(x)


def _pearson_at_lag(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """
    Pearson correlation between x and y shifted by `lag` steps.
    x leads y: correlates x[0..n-lag] with y[lag..n].
    Returns 0.0 for constant series or too-short windows.
    """
    a, b = (x, y) if lag == 0 else (x[:-lag], y[lag:])
    if len(a) < 2:
        return 0.0
    with np.errstate(invalid="ignore"):
        r = np.corrcoef(a, b)[0, 1]
    return float(0.0 if np.isnan(r) else r)


def _spearman_at_lag(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """
    Spearman rank correlation between x and y shifted by `lag` steps.
    Achieved by rank-transforming the aligned slices then computing Pearson.
    """
    a, b = (x, y) if lag == 0 else (x[:-lag], y[lag:])
    if len(a) < 2:
        return 0.0
    with np.errstate(invalid="ignore"):
        r = np.corrcoef(_rank(a), _rank(b))[0, 1]
    return float(0.0 if np.isnan(r) else r)


def _rank(x: np.ndarray) -> np.ndarray:
    """Convert values to 1-based ranks, averaging ties."""
    temp = np.argsort(x)
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(1, len(x) + 1, dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[temp[i]] == x[temp[j]]:
            j += 1
        if j > i + 1:
            avg = (ranks[temp[i]] + ranks[temp[j - 1]]) / 2.0
            ranks[temp[i:j]] = avg
        i = j
    return ranks


def _best_lag_and_coeffs(
    biz: np.ndarray,
    sys: np.ndarray,
    max_lag: int,
) -> tuple[int, float, float]:
    """
    Sweep lags 0..max_lag and return (best_lag, pearson_r, spearman_r)
    at the lag that maximises |pearson_r| + |spearman_r|.
    """
    best_lag      = 0
    best_pearson  = 0.0
    best_spearman = 0.0
    best_combined = -1.0
    actual_max    = min(max_lag, len(biz) - 2)

    for lag in range(0, actual_max + 1):
        p = _pearson_at_lag(biz, sys, lag)
        s = _spearman_at_lag(biz, sys, lag)
        combined = abs(p) + abs(s)
        if combined > best_combined:
            best_combined = combined
            best_lag      = lag
            best_pearson  = p
            best_spearman = s

    return best_lag, best_pearson, best_spearman


def _analyze_pair(
    biz_diff: np.ndarray,
    sys_diff: np.ndarray,
    target_metric: str,
    max_lag: int,
) -> CorrelationResult:
    """Run the full CCF analysis for one business → system metric pair."""
    lag, pearson_r, spearman_r = _best_lag_and_coeffs(biz_diff, sys_diff, max_lag)
    best_r         = max(abs(pearson_r), abs(spearman_r))
    is_significant = best_r >= SIGNIFICANCE_THRESHOLD

    logger.debug(
        "  %-12s lag=%2d min  pearson=%+.3f  spearman=%+.3f  best=%.3f  sig=%s",
        target_metric, lag, pearson_r, spearman_r, best_r, is_significant,
    )
    return CorrelationResult(
        target_metric=target_metric,
        lag_minutes=lag,
        pearson_r=round(pearson_r, 4),
        spearman_r=round(spearman_r, 4),
        best_r=round(best_r, 4),
        is_significant=is_significant,
    )


# ── Public API ────────────────────────────────────────────────────────────────

def analyze(
    bundle: MetricsBundle,
    max_lag: int = MAX_LAG_MINUTES,
) -> CorrelationReport:
    """
    Analyse lag-shifted Pearson + Spearman correlations between the
    business metric and each of the five system metrics.

    Steps:
      1. Check business metric series has non-zero variance (constant = broken).
      2. First-difference all series (removes trend / non-stationarity).
      3. For each target, sweep lags 0..max_lag and record the lag that
         maximises the combined Pearson + Spearman strength.
      4. Flag significant if best_r >= SIGNIFICANCE_THRESHOLD (0.6).

    Important: insignificant results do NOT block training.
    The caller should use report.per_target_lag() to pick lag=0 for
    uncorrelated targets and the discovered lag for correlated ones.

    is_business_constant=True is the only signal that should block training,
    as it means the business metric feed is broken.

    Args:
        bundle:  MetricsBundle from the data collector.
        max_lag: Maximum lag to search in minutes (default 60).

    Returns:
        CorrelationReport with results for all five targets.
    """
    n = len(bundle.business)
    logger.info(
        "Correlation analysis: %d points  max_lag=%d min  targets=%s",
        n, max_lag, list(ALL_TARGETS),
    )

    if n < MIN_POINTS:
        logger.warning(
            "Only %d data points (minimum %d) — results may be unreliable.",
            n, MIN_POINTS,
        )

    biz = _extract_values(bundle.business)

    # Detect constant business metric — training would be meaningless
    biz_std = float(np.std(biz))
    is_constant = biz_std < CONSTANT_STD_THRESHOLD
    if is_constant:
        logger.error(
            "Business metric has near-zero variance (std=%.2e). "
            "Check the PromQL formula — the series appears constant.",
            biz_std,
        )

    # First-difference all series
    biz_d   = _first_difference(biz)
    cpu_d   = _first_difference(_extract_values(bundle.cpu))
    rgb_d   = _first_difference(_extract_values(bundle.ram_gb))
    rpt_d   = _first_difference(_extract_values(bundle.ram_percent))
    net_d   = _first_difference(_extract_values(bundle.network))
    dsk_d   = _first_difference(_extract_values(bundle.disk))

    # Analyse each pair
    cpu_res = _analyze_pair(biz_d, cpu_d, "cpu",         max_lag)
    rgb_res = _analyze_pair(biz_d, rgb_d, "ram_gb",      max_lag)
    rpt_res = _analyze_pair(biz_d, rpt_d, "ram_percent", max_lag)
    net_res = _analyze_pair(biz_d, net_d, "network",     max_lag)
    dsk_res = _analyze_pair(biz_d, dsk_d, "disk",        max_lag)

    report = CorrelationReport(
        cpu=cpu_res, ram_gb=rgb_res, ram_percent=rpt_res,
        network=net_res, disk=dsk_res,
        n_points=n,
        is_business_constant=is_constant,
    )

    sig_count = sum(1 for r in report.all_results() if r.is_significant)
    logger.info(
        "Analysis complete: %d/%d targets significant  best_lag=%d min  "
        "constant_biz=%s",
        sig_count, len(ALL_TARGETS), report.best_lag(), is_constant,
    )
    for r in report.all_results():
        logger.info(
            "  %-12s  lag=%2d min  best_r=%.3f  sig=%s",
            r.target_metric, r.lag_minutes, r.best_r, r.is_significant,
        )

    return report
