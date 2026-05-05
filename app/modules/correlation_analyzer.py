from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from app.modules.data_collector import MetricsBundle

logger = logging.getLogger(__name__)


MAX_LAG_MINUTES:        int   = 60
SIGNIFICANCE_THRESHOLD: float = 0.6
MIN_POINTS:             int   = 10    # after resampling
CONSTANT_STD_THRESHOLD: float = 1e-6

STEP_CANDIDATES: tuple[int, ...] = (60, 120, 300, 600)

TargetName = Literal["cpu", "ram_gb", "ram_percent", "network", "disk"]
ALL_TARGETS: tuple[TargetName, ...] = ("cpu", "ram_gb", "ram_percent", "network", "disk")


@dataclass
class CorrelationResult:
    target_metric:  str
    lag_minutes:    int
    pearson_r:      float
    spearman_r:     float
    best_r:         float
    is_significant: bool


@dataclass
class CorrelationReport:
    cpu:         CorrelationResult
    ram_gb:      CorrelationResult
    ram_percent: CorrelationResult
    network:     CorrelationResult
    disk:        CorrelationResult
    n_points:    int
    is_business_constant: bool = False
    best_step_seconds: int = 60
    _base_step_seconds: int = 60
    any_significant: bool = field(init=False)

    def __post_init__(self) -> None:
        self.any_significant = any(
            r.is_significant
            for r in [self.cpu, self.ram_gb, self.ram_percent, self.network, self.disk]
        )

    def all_results(self) -> list[CorrelationResult]:
        return [self.cpu, self.ram_gb, self.ram_percent, self.network, self.disk]

    def per_target_lag(self) -> dict[str, int]:
        return {r.target_metric: (r.lag_minutes if r.is_significant else 0)
                for r in self.all_results()}

    def best_lag(self) -> int:
        significant = [r for r in self.all_results() if r.is_significant]
        if not significant:
            return 0
        lags = sorted(r.lag_minutes for r in significant)
        return lags[len(lags) // 2]


def _resample(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return arr
    n_blocks = len(arr) // factor
    if n_blocks == 0:
        return arr
    return arr[: n_blocks * factor].reshape(n_blocks, factor).mean(axis=1)


def _resample_bundle(
    raw: dict[str, np.ndarray],
    base_step: int,
    target_step: int,
) -> dict[str, np.ndarray]:
    factor = max(1, target_step // base_step)
    return {k: _resample(v, factor) for k, v in raw.items()}


def _score_step(
    biz: np.ndarray,
    cpu: np.ndarray,
    max_lag_steps: int,
) -> float:
    if len(biz) < MIN_POINTS or biz.std() < CONSTANT_STD_THRESHOLD:
        return 0.0
    biz_d = np.diff(biz)
    cpu_d = np.diff(cpu)
    _, p, s = _best_lag_and_coeffs(biz_d, cpu_d, min(max_lag_steps, len(biz_d) - 1))
    return max(abs(p), abs(s))


def _select_best_step(
    raw: dict[str, np.ndarray],
    base_step_seconds: int,
    max_lag_minutes: int,
) -> int:
    best_step  = base_step_seconds
    best_score = -1.0

    logger.info("Step-seconds CV over candidates %s", STEP_CANDIDATES)

    for step in STEP_CANDIDATES:
        if step < base_step_seconds:
            continue

        resampled  = _resample_bundle(raw, base_step_seconds, step)
        biz_r      = resampled["biz"]
        cpu_r      = resampled["cpu"]
        max_lag_st = max_lag_minutes * 60 // step

        if len(biz_r) < MIN_POINTS:
            logger.debug("  step=%ds → only %d points, skipping", step, len(biz_r))
            continue

        score = _score_step(biz_r, cpu_r, max_lag_st)
        logger.info(
            "  step=%4ds  n=%4d  r*(biz→cpu)=%.3f",
            step, len(biz_r), score,
        )

        if score > best_score:
            best_score = score
            best_step  = step

    logger.info(
        "Best step: %ds (r*=%.3f)", best_step, best_score
    )
    return best_step


def _extract_values(series: list[dict]) -> np.ndarray:
    return np.array([p["value"] for p in series], dtype=float)


def _first_difference(x: np.ndarray) -> np.ndarray:
    return np.diff(x)


def _pearson_at_lag(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    a, b = (x, y) if lag == 0 else (x[:-lag], y[lag:])
    if len(a) < 2:
        return 0.0
    with np.errstate(invalid="ignore"):
        r = np.corrcoef(a, b)[0, 1]
    return float(0.0 if np.isnan(r) else r)


def _spearman_at_lag(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    a, b = (x, y) if lag == 0 else (x[:-lag], y[lag:])
    if len(a) < 2:
        return 0.0
    with np.errstate(invalid="ignore"):
        r = np.corrcoef(_rank(a), _rank(b))[0, 1]
    return float(0.0 if np.isnan(r) else r)


def _rank(x: np.ndarray) -> np.ndarray:
    temp  = np.argsort(x)
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
    lag, pearson_r, spearman_r = _best_lag_and_coeffs(biz_diff, sys_diff, max_lag)
    best_r         = max(abs(pearson_r), abs(spearman_r))
    is_significant = best_r >= SIGNIFICANCE_THRESHOLD

    logger.debug(
        "  %-12s lag=%2d  pearson=%+.3f  spearman=%+.3f  best=%.3f  sig=%s",
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


def analyze(
    bundle: MetricsBundle,
    max_lag: int = MAX_LAG_MINUTES,
    base_step_seconds: int = 60,
) -> CorrelationReport:
    n = len(bundle.business)
    logger.info(
        "Correlation analysis: %d raw points  max_lag=%d min  base_step=%ds",
        n, max_lag, base_step_seconds,
    )

    if n < MIN_POINTS:
        logger.warning(
            "Only %d data points (minimum %d) — results may be unreliable.",
            n, MIN_POINTS,
        )

    biz_raw = _extract_values(bundle.business)

    biz_std     = float(np.std(biz_raw))
    is_constant = biz_std < CONSTANT_STD_THRESHOLD
    if is_constant:
        logger.error(
            "Business metric has near-zero variance (std=%.2e). "
            "Check the PromQL formula — the series appears constant.",
            biz_std,
        )

    raw = {
        "biz":     biz_raw,
        "cpu":     _extract_values(bundle.cpu),
        "ram_gb":  _extract_values(bundle.ram_gb),
        "ram_pct": _extract_values(bundle.ram_percent),
        "net":     _extract_values(bundle.network),
        "disk":    _extract_values(bundle.disk),
    }

    biz_thr = float(biz_raw.max()) * 0.10
    cpu_thr = float(raw["cpu"].max()) * 0.10
    active  = (biz_raw > biz_thr) & (raw["cpu"] > cpu_thr)
    n_active = int(active.sum())
    if n_active >= MIN_POINTS:
        raw_for_corr = {k: v[active] for k, v in raw.items()}
        logger.info(
            "Idle filter: kept %d/%d points (biz>%.2f AND cpu>%.2f)",
            n_active, n, biz_thr, cpu_thr,
        )
    else:
        raw_for_corr = raw
        logger.warning(
            "Idle filter: only %d active points after filtering — "
            "using full dataset for correlation analysis", n_active,
        )

    best_step = _select_best_step(raw_for_corr, base_step_seconds, max_lag)

    resampled   = _resample_bundle(raw_for_corr, base_step_seconds, best_step)
    max_lag_st  = max_lag * 60 // best_step   # lag in steps at best resolution

    biz_d   = _first_difference(resampled["biz"])
    cpu_d   = _first_difference(resampled["cpu"])
    rgb_d   = _first_difference(resampled["ram_gb"])
    rpt_d   = _first_difference(resampled["ram_pct"])
    net_d   = _first_difference(resampled["net"])
    dsk_d   = _first_difference(resampled["disk"])

    cpu_res = _analyze_pair(biz_d, cpu_d, "cpu",         max_lag_st)
    rgb_res = _analyze_pair(biz_d, rgb_d, "ram_gb",      max_lag_st)
    rpt_res = _analyze_pair(biz_d, rpt_d, "ram_percent", max_lag_st)
    net_res = _analyze_pair(biz_d, net_d, "network",     max_lag_st)
    dsk_res = _analyze_pair(biz_d, dsk_d, "disk",        max_lag_st)

    step_min = best_step / 60
    for res in [cpu_res, rgb_res, rpt_res, net_res, dsk_res]:
        res.lag_minutes = round(res.lag_minutes * step_min)

    report = CorrelationReport(
        cpu=cpu_res, ram_gb=rgb_res, ram_percent=rpt_res,
        network=net_res, disk=dsk_res,
        n_points=len(resampled["biz"]),
        is_business_constant=is_constant,
        best_step_seconds=best_step,
        _base_step_seconds=base_step_seconds,
    )

    sig_count = sum(1 for r in report.all_results() if r.is_significant)
    logger.info(
        "Analysis complete: %d/%d significant  best_lag=%d min  "
        "best_step=%ds  constant_biz=%s",
        sig_count, len(ALL_TARGETS), report.best_lag(),
        best_step, is_constant,
    )
    for r in report.all_results():
        logger.info(
            "  %-12s  lag=%2d min  best_r=%.3f  sig=%s",
            r.target_metric, r.lag_minutes, r.best_r, r.is_significant,
        )

    return report
