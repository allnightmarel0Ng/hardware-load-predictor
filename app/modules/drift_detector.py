from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

PSI_STABLE   = 0.10
PSI_MODERATE = 0.20

N_BINS       = 10
MIN_SAMPLES  = 30


@dataclass
class DriftResult:
    psi: float
    level: str
    n_reference: int
    n_current: int
    is_drifted: bool
    bin_edges: list[float]
    reference_freqs: list[float]
    current_freqs: list[float]


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = N_BINS) -> DriftResult:
    reference = reference[np.isfinite(reference)]
    current   = current[np.isfinite(current)]

    if len(reference) < MIN_SAMPLES or len(current) < MIN_SAMPLES:
        logger.debug(
            "PSI skipped: reference=%d samples, current=%d (need %d)",
            len(reference), len(current), MIN_SAMPLES,
        )
        return DriftResult(
            psi=0.0, level="stable",
            n_reference=len(reference), n_current=len(current),
            is_drifted=False,
            bin_edges=[], reference_freqs=[], current_freqs=[],
        )

    percentiles = np.linspace(0, 100, bins + 1)
    bin_edges   = np.unique(np.percentile(reference, percentiles))

    if len(bin_edges) < 3:
        bin_edges = np.linspace(reference.min(), reference.max(), bins + 1)

    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current,   bins=bin_edges)

    eps = 1e-6
    ref_freq = (ref_counts / len(reference)) + eps
    cur_freq = (cur_counts / len(current))   + eps

    psi_values = (cur_freq - ref_freq) * np.log(cur_freq / ref_freq)
    psi_total  = float(np.sum(psi_values))

    if psi_total < PSI_STABLE:
        level = "stable"
    elif psi_total < PSI_MODERATE:
        level = "moderate"
    else:
        level = "significant"

    return DriftResult(
        psi=round(psi_total, 4),
        level=level,
        n_reference=len(reference),
        n_current=len(current),
        is_drifted=(psi_total >= PSI_MODERATE),
        bin_edges=bin_edges.tolist(),
        reference_freqs=ref_freq.tolist(),
        current_freqs=cur_freq.tolist(),
    )


def compute_reference_distribution(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    percentiles = np.linspace(0, 100, N_BINS + 1)
    bin_edges   = np.unique(np.percentile(values, percentiles))

    if len(bin_edges) < 3:
        bin_edges = np.linspace(values.min(), values.max(), N_BINS + 1)

    counts, _ = np.histogram(values, bins=bin_edges)
    eps = 1e-6
    freqs = (counts / len(values)) + eps

    return {
        "n_samples":  int(len(values)),
        "mean":       float(np.mean(values)),
        "std":        float(np.std(values)),
        "min":        float(np.min(values)),
        "max":        float(np.max(values)),
        "bin_edges":  bin_edges.tolist(),
        "freqs":      freqs.tolist(),
    }


def check_drift_from_snapshot(
    reference_snapshot: dict,
    current_values: np.ndarray,
) -> DriftResult:
    current = current_values[np.isfinite(current_values)]

    if len(current) < MIN_SAMPLES:
        return DriftResult(
            psi=0.0, level="stable",
            n_reference=reference_snapshot.get("n_samples", 0),
            n_current=len(current),
            is_drifted=False,
            bin_edges=[], reference_freqs=[], current_freqs=[],
        )

    bin_edges = np.array(reference_snapshot["bin_edges"])
    ref_freq  = np.array(reference_snapshot["freqs"])

    cur_counts, _ = np.histogram(current, bins=bin_edges)
    eps = 1e-6
    cur_freq = (cur_counts / len(current)) + eps

    min_len  = min(len(ref_freq), len(cur_freq))
    ref_freq = ref_freq[:min_len]
    cur_freq = cur_freq[:min_len]

    psi_values = (cur_freq - ref_freq) * np.log(cur_freq / ref_freq)
    psi_total  = float(np.sum(psi_values))

    if psi_total < PSI_STABLE:
        level = "stable"
    elif psi_total < PSI_MODERATE:
        level = "moderate"
    else:
        level = "significant"

    return DriftResult(
        psi=round(psi_total, 4),
        level=level,
        n_reference=reference_snapshot.get("n_samples", 0),
        n_current=len(current),
        is_drifted=(psi_total >= PSI_MODERATE),
        bin_edges=bin_edges.tolist(),
        reference_freqs=ref_freq.tolist(),
        current_freqs=cur_freq.tolist(),
    )
