"""
Drift Detector — concept drift detection using Population Stability Index
──────────────────────────────────────────────────────────────────────────────
Detects when the distribution of the business metric (the model's input)
has shifted significantly from what it looked like at training time.

Why PSI?
  PSI is the industry standard for monitoring model input drift in production
  (originally from credit scoring, now used in all ML monitoring). Unlike a
  pure R² threshold which only flags degraded outputs, PSI flags distribution
  shift in the *inputs* — allowing preemptive retraining before accuracy drops.

  PSI = Σ (P_actual - P_reference) × ln(P_actual / P_reference)

  Interpretation (standard thresholds):
    PSI < 0.10  — no significant drift, model is stable
    0.10 ≤ PSI < 0.20  — moderate drift, monitor closely
    PSI ≥ 0.20  — significant drift, retraining strongly recommended

How it's used here:
  After each accuracy evaluation, the monitor also checks whether the recent
  business metric distribution (last 24h of ForecastResult.business_metric_value)
  has drifted from the training-time distribution stored in TrainedModel.parameters.

  If PSI ≥ 0.20 the system logs a warning and can optionally trigger retraining
  even if R² is still acceptable — because the model was not trained on data
  that looks like what it's currently receiving.

  The training distribution is stored as a histogram (10 bins, bin edges +
  frequencies) in TrainedModel.parameters["input_distribution"] when the model
  is trained.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# PSI thresholds (industry standard)
PSI_STABLE   = 0.10   # below this: stable
PSI_MODERATE = 0.20   # above this: significant drift → retrain recommended

N_BINS       = 10     # number of histogram bins for PSI computation
MIN_SAMPLES  = 30     # minimum samples needed to compute a reliable PSI


@dataclass
class DriftResult:
    psi: float
    level: str          # "stable" | "moderate" | "significant"
    n_reference: int    # number of samples in reference distribution
    n_current: int      # number of samples in current window
    is_drifted: bool    # True if PSI >= PSI_MODERATE
    bin_edges: list[float]
    reference_freqs: list[float]
    current_freqs: list[float]


# ── Core PSI computation ──────────────────────────────────────────────────────

def _psi(reference: np.ndarray, current: np.ndarray, bins: int = N_BINS) -> DriftResult:
    """
    Compute the Population Stability Index between reference and current samples.

    Both arrays are 1-D float arrays of the same feature (business metric values).
    Bins are defined by the reference distribution's percentiles so the PSI is
    not sensitive to absolute scale — only to relative distributional shift.
    """
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

    # Build bin edges from reference percentiles — robust to outliers
    percentiles = np.linspace(0, 100, bins + 1)
    bin_edges   = np.unique(np.percentile(reference, percentiles))

    # Ensure we have enough unique edges; fall back to min/max range if not
    if len(bin_edges) < 3:
        bin_edges = np.linspace(reference.min(), reference.max(), bins + 1)

    # Compute frequencies (proportions) in each bin
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    cur_counts, _ = np.histogram(current,   bins=bin_edges)

    # Convert to proportions; add small epsilon to avoid log(0)
    eps = 1e-6
    ref_freq = (ref_counts / len(reference)) + eps
    cur_freq = (cur_counts / len(current))   + eps

    # PSI formula
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


# ── Distribution snapshot (stored with model at training time) ────────────────

def compute_reference_distribution(values: np.ndarray) -> dict:
    """
    Compute a compact reference distribution snapshot to be stored in
    TrainedModel.parameters["input_distribution"].

    Stores enough information to reconstruct the histogram for future PSI
    comparisons without keeping the full training dataset.
    """
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
    """
    Compute PSI between a stored distribution snapshot and a current array.
    Used by the accuracy monitor to check drift without the full training data.
    """
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

    # Align lengths in case histogram bins differ (edge case with very skewed data)
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
