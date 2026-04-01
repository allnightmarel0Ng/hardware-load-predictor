"""
Module 3 — Correlation Analyzer
Identifies the time lag at which a business metric best predicts each
system metric, and quantifies the strength of that relationship.

Implementation uses a two-method approach recommended by research:

  1. Pearson cross-correlation function (CCF) swept over lags 0..MAX_LAG.
     Fast (O(n log n) via FFT convolution), gives signed linear correlation.
     Series are first-differenced to remove trend/non-stationarity before
     computing CCF, preventing spurious correlations from shared trends.

  2. Spearman rank CCF swept in parallel.
     Captures monotonic non-linear relationships (e.g. logarithmic RAM growth)
     and is robust to outliers/load spikes.

The reported coefficient for each metric is max(|pearson_r|, |spearman_r|)
at the optimal lag, so non-linear relationships are not missed.
Significance threshold: |r| >= 0.6  (standard convention).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from app.modules.data_collector import MetricsBundle

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MAX_LAG_MINUTES: int = 60          # search window: 0..60 minute lags
SIGNIFICANCE_THRESHOLD: float = 0.6  # |r| >= 0.6 → significant
MIN_POINTS: int = 30               # need at least this many aligned points


# ── Public data types ────────────────────────────────────────────────────────

@dataclass
class CorrelationResult:
    """
    Holds the discovered lag and correlation coefficient for one
    (business metric → system metric) pair.
    """
    target_metric: str   # "cpu" | "ram" | "network"
    lag_minutes: int     # optimal lag in minutes (0 = no lag)
    pearson_r: float     # Pearson CCF coefficient at the optimal lag  [-1, 1]
    spearman_r: float    # Spearman CCF coefficient at the optimal lag [-1, 1]
    best_r: float        # max(|pearson_r|, |spearman_r|) — reported strength
    is_significant: bool # True if best_r >= SIGNIFICANCE_THRESHOLD


@dataclass
class CorrelationReport:
    """Aggregated CCF results for all three system metrics."""
    cpu: CorrelationResult
    ram: CorrelationResult
    network: CorrelationResult
    n_points: int                   # number of aligned data points used
    any_significant: bool = field(init=False)

    def __post_init__(self) -> None:
        self.any_significant = any(
            r.is_significant for r in [self.cpu, self.ram, self.network]
        )

    def best_lag(self) -> int:
        """
        Return the median lag across all significant metrics.
        Falls back to 0 if none are significant.
        """
        significant = [
            r for r in [self.cpu, self.ram, self.network] if r.is_significant
        ]
        if not significant:
            return 0
        lags = sorted(r.lag_minutes for r in significant)
        return lags[len(lags) // 2]  # median


# ── Internal helpers ─────────────────────────────────────────────────────────

def _extract_values(series: list[dict]) -> np.ndarray:
    """Pull float values out of the timeseries dicts."""
    return np.array([p["value"] for p in series], dtype=float)


def _first_difference(x: np.ndarray) -> np.ndarray:
    """
    Remove non-stationarity by computing first differences (x[t] - x[t-1]).
    This prevents spurious correlations caused by shared trends.
    The returned array is 1 element shorter than the input.
    """
    return np.diff(x)


def _pearson_at_lag(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """
    Pearson correlation coefficient between x and y shifted by `lag` steps.
    x leads y: x[0..n-lag] is correlated with y[lag..n].
    Returns NaN if the variance is zero.
    """
    if lag == 0:
        a, b = x, y
    else:
        a = x[:-lag]
        b = y[lag:]
    if len(a) < 2:
        return 0.0
    # np.corrcoef returns NaN when std == 0 (constant series)
    with np.errstate(invalid="ignore"):
        r = np.corrcoef(a, b)[0, 1]
    return float(0.0 if np.isnan(r) else r)


def _spearman_at_lag(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    """
    Spearman rank correlation between x and y shifted by `lag` steps.
    Achieved by rank-transforming the aligned slices then computing Pearson.
    """
    if lag == 0:
        a, b = x, y
    else:
        a = x[:-lag]
        b = y[lag:]
    if len(a) < 2:
        return 0.0
    # rank transform (average method for ties)
    ra = _rank(a)
    rb = _rank(b)
    with np.errstate(invalid="ignore"):
        r = np.corrcoef(ra, rb)[0, 1]
    return float(0.0 if np.isnan(r) else r)


def _rank(x: np.ndarray) -> np.ndarray:
    """Convert values to ranks (1-based, average ties)."""
    temp = np.argsort(x)
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(1, len(x) + 1, dtype=float)
    # handle ties: find groups of equal values and assign average rank
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
    at the lag that maximises |pearson_r| + |spearman_r| (combined strength).
    """
    best_lag = 0
    best_pearson = 0.0
    best_spearman = 0.0
    best_combined = -1.0

    actual_max_lag = min(max_lag, len(biz) - 2)

    for lag in range(0, actual_max_lag + 1):
        p = _pearson_at_lag(biz, sys, lag)
        s = _spearman_at_lag(biz, sys, lag)
        combined = abs(p) + abs(s)
        if combined > best_combined:
            best_combined = combined
            best_lag = lag
            best_pearson = p
            best_spearman = s

    return best_lag, best_pearson, best_spearman


def _analyze_pair(
    biz_diff: np.ndarray,
    sys_diff: np.ndarray,
    target_metric: str,
    max_lag: int,
) -> CorrelationResult:
    """Run the full CCF analysis for one business→system metric pair."""
    lag, pearson_r, spearman_r = _best_lag_and_coeffs(biz_diff, sys_diff, max_lag)
    best_r = max(abs(pearson_r), abs(spearman_r))
    is_significant = best_r >= SIGNIFICANCE_THRESHOLD

    logger.debug(
        "  %-8s lag=%2d min  pearson=%.3f  spearman=%.3f  best=%.3f  sig=%s",
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
    Analyse lag-shifted Pearson and Spearman correlations between the
    business metric and each of the three system metrics (CPU, RAM, network).

    Steps:
      1. Extract value arrays from the bundle.
      2. First-difference all series (removes trend / non-stationarity).
      3. For each system metric, sweep lags 0..max_lag and record the lag
         that maximises the combined Pearson + Spearman strength.
      4. Flag as significant if best_r >= SIGNIFICANCE_THRESHOLD (0.6).

    Args:
        bundle:  MetricsBundle from the data collector.
        max_lag: Maximum lag to search in minutes (default 60).

    Returns:
        CorrelationReport with results for cpu, ram, and network.
    """
    n = len(bundle.business)
    logger.info("Running correlation analysis on %d data points (max_lag=%d min)", n, max_lag)

    if n < MIN_POINTS:
        logger.warning(
            "Only %d data points available (minimum %d). "
            "Results may be unreliable.",
            n, MIN_POINTS,
        )

    # Extract raw values
    biz = _extract_values(bundle.business)
    cpu = _extract_values(bundle.cpu)
    ram = _extract_values(bundle.ram)
    net = _extract_values(bundle.network)

    # First-difference to achieve (approximate) stationarity
    biz_d = _first_difference(biz)
    cpu_d = _first_difference(cpu)
    ram_d = _first_difference(ram)
    net_d = _first_difference(net)

    # Analyse each pair
    cpu_result = _analyze_pair(biz_d, cpu_d, "cpu",     max_lag)
    ram_result = _analyze_pair(biz_d, ram_d, "ram",     max_lag)
    net_result = _analyze_pair(biz_d, net_d, "network", max_lag)

    report = CorrelationReport(
        cpu=cpu_result,
        ram=ram_result,
        network=net_result,
        n_points=n,
    )
    logger.info(
        "Correlation analysis complete: significant=%s  best_lag=%d min  "
        "cpu(r=%.3f lag=%d)  ram(r=%.3f lag=%d)  net(r=%.3f lag=%d)",
        report.any_significant,
        report.best_lag(),
        cpu_result.best_r, cpu_result.lag_minutes,
        ram_result.best_r, ram_result.lag_minutes,
        net_result.best_r, net_result.lag_minutes,
    )
    return report
