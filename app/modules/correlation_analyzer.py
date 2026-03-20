"""
Module 3 — Correlation Analyzer
Identifies the time lag at which a business metric best predicts each
system metric, and quantifies the strength of that relationship.

STATUS: MOCKED — returns hard-coded plausible values.
        Replace the body of `analyze()` with real cross-correlation logic
        (e.g. numpy.correlate or statsmodels ccf) when ready.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.modules.data_collector import MetricsBundle

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """
    Holds the discovered lag and correlation coefficient for one
    (business metric → system metric) pair.
    """
    target_metric: str          # "cpu" | "ram" | "network"
    lag_minutes: int            # optimal lag in minutes
    pearson_r: float            # correlation coefficient at that lag  [-1, 1]
    is_significant: bool        # True if |r| >= threshold


@dataclass
class CorrelationReport:
    """Aggregated results for all three system metrics."""
    cpu: CorrelationResult
    ram: CorrelationResult
    network: CorrelationResult
    any_significant: bool = field(init=False)

    def __post_init__(self) -> None:
        self.any_significant = any(
            r.is_significant for r in [self.cpu, self.ram, self.network]
        )

    def best_lag(self) -> int:
        """Return the most commonly found lag across significant metrics."""
        significant = [
            r for r in [self.cpu, self.ram, self.network] if r.is_significant
        ]
        if not significant:
            return 0
        return round(sum(r.lag_minutes for r in significant) / len(significant))


_SIGNIFICANCE_THRESHOLD = 0.6  # |r| >= 0.6 → significant


def analyze(bundle: MetricsBundle) -> CorrelationReport:
    """
    Analyse correlations between the business metric and each system metric.

    TODO: Replace mock with real implementation, e.g.:
        import numpy as np
        def _cross_correlate(x, y, max_lag):
            best_lag, best_r = 0, 0
            for lag in range(0, max_lag + 1):
                r = np.corrcoef(x[lag:], y[:len(x)-lag])[0, 1]
                if abs(r) > abs(best_r):
                    best_r, best_lag = r, lag
            return best_lag, best_r
    """
    logger.info(
        "Running correlation analysis (MOCKED) on %d data points",
        len(bundle.business),
    )

    # ── MOCK VALUES ──────────────────────────────────────────────────────────
    # These are realistic for an e-commerce workload:
    # business metric (orders/min) has a ~5 min lag to CPU, ~3 min to RAM,
    # and ~8 min to network traffic.
    cpu_result = CorrelationResult(
        target_metric="cpu",
        lag_minutes=5,
        pearson_r=0.82,
        is_significant=True,
    )
    ram_result = CorrelationResult(
        target_metric="ram",
        lag_minutes=3,
        pearson_r=0.74,
        is_significant=True,
    )
    network_result = CorrelationResult(
        target_metric="network",
        lag_minutes=8,
        pearson_r=0.69,
        is_significant=True,
    )
    # ── END MOCK ─────────────────────────────────────────────────────────────

    report = CorrelationReport(cpu=cpu_result, ram=ram_result, network=network_result)
    logger.info(
        "Correlation analysis complete. Significant=%s, best_lag=%d min",
        report.any_significant,
        report.best_lag(),
    )
    return report
