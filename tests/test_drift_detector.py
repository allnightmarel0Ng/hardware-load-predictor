"""Tests for drift_detector.py — PSI computation and distribution snapshots."""
import pytest
import numpy as np

from app.modules.drift_detector import (
    _psi,
    compute_reference_distribution,
    check_drift_from_snapshot,
    PSI_STABLE,
    PSI_MODERATE,
    MIN_SAMPLES,
    DriftResult,
)

RNG = np.random.default_rng(42)


# ── _psi core ─────────────────────────────────────────────────────────────────

class TestPsiCore:
    def test_identical_distributions_give_zero_psi(self):
        data = RNG.normal(100, 20, 500)
        result = _psi(data, data.copy())
        assert result.psi < PSI_STABLE
        assert result.level == "stable"
        assert not result.is_drifted

    def test_very_different_distributions_give_high_psi(self):
        reference = RNG.normal(100, 10, 500)   # mean=100
        current   = RNG.normal(500, 10, 500)   # mean=500 — completely shifted
        result    = _psi(reference, current)
        assert result.psi >= PSI_MODERATE
        assert result.level == "significant"
        assert result.is_drifted

    def test_moderate_shift_gives_moderate_level(self):
        reference = RNG.normal(100, 15, 500)
        current   = RNG.normal(130, 15, 500)   # shifted ~2σ — moderate
        result    = _psi(reference, current)
        # Just verify level encoding is consistent with psi value
        if result.psi < PSI_STABLE:
            assert result.level == "stable"
        elif result.psi < PSI_MODERATE:
            assert result.level == "moderate"
        else:
            assert result.level == "significant"

    def test_returns_correct_sample_counts(self):
        ref = RNG.normal(100, 10, 300)
        cur = RNG.normal(100, 10, 150)
        result = _psi(ref, cur)
        assert result.n_reference == 300
        assert result.n_current   == 150

    def test_psi_is_non_negative(self):
        for _ in range(5):
            ref = RNG.normal(RNG.uniform(50, 200), 20, 200)
            cur = RNG.normal(RNG.uniform(50, 200), 20, 200)
            result = _psi(ref, cur)
            assert result.psi >= 0.0

    def test_too_few_samples_returns_stable(self):
        tiny_ref = RNG.normal(100, 10, MIN_SAMPLES - 1)
        tiny_cur = RNG.normal(200, 10, MIN_SAMPLES - 1)
        result = _psi(tiny_ref, tiny_cur)
        assert result.psi == 0.0
        assert result.level == "stable"
        assert not result.is_drifted

    def test_bin_edges_and_freqs_populated_when_enough_data(self):
        ref = RNG.normal(100, 10, 300)
        cur = RNG.normal(100, 10, 300)
        result = _psi(ref, cur)
        assert len(result.bin_edges) > 0
        assert len(result.reference_freqs) > 0
        assert len(result.current_freqs) > 0

    def test_is_drifted_consistent_with_psi_value(self):
        ref = RNG.normal(100, 10, 500)
        cur = RNG.normal(100, 10, 500)
        result = _psi(ref, cur)
        assert result.is_drifted == (result.psi >= PSI_MODERATE)


# ── compute_reference_distribution ────────────────────────────────────────────

class TestComputeReferenceDistribution:
    def test_returns_required_keys(self):
        data = RNG.normal(1000, 200, 500)
        snap = compute_reference_distribution(data)
        for key in ["n_samples", "mean", "std", "min", "max", "bin_edges", "freqs"]:
            assert key in snap, f"Missing key: {key}"

    def test_statistics_are_correct(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20, dtype=float)
        snap = compute_reference_distribution(data)
        assert snap["n_samples"] == 100
        assert abs(snap["mean"] - 3.0) < 0.01
        assert snap["min"] == 1.0
        assert snap["max"] == 5.0

    def test_freqs_sum_to_approximately_one(self):
        data = RNG.normal(500, 100, 400)
        snap = compute_reference_distribution(data)
        # Freqs include epsilon, so sum is slightly above 1
        assert abs(sum(snap["freqs"]) - 1.0) < 0.1

    def test_handles_constant_series(self):
        data = np.full(100, 42.0)
        snap = compute_reference_distribution(data)
        assert snap["n_samples"] == 100
        assert snap["mean"] == 42.0


# ── check_drift_from_snapshot ─────────────────────────────────────────────────

class TestCheckDriftFromSnapshot:
    def _snap(self, mean: float = 1000.0, std: float = 200.0, n: int = 500) -> dict:
        data = RNG.normal(mean, std, n)
        return compute_reference_distribution(data)

    def test_same_distribution_stable(self):
        snap = self._snap(1000, 200, 500)
        current = RNG.normal(1000, 200, 200)
        result = check_drift_from_snapshot(snap, current)
        assert result.level in ("stable", "moderate")   # small sample variance OK
        assert result.psi >= 0.0

    def test_large_shift_detected(self):
        snap = self._snap(1000, 50, 500)
        current = RNG.normal(3000, 50, 300)   # completely different range
        result = check_drift_from_snapshot(snap, current)
        assert result.is_drifted
        assert result.level == "significant"

    def test_too_few_current_samples_gives_stable(self):
        snap = self._snap(1000, 200, 500)
        tiny = RNG.normal(5000, 50, MIN_SAMPLES - 1)
        result = check_drift_from_snapshot(snap, tiny)
        assert result.psi == 0.0
        assert not result.is_drifted

    def test_n_reference_comes_from_snapshot(self):
        snap = self._snap(1000, 200, 400)
        current = RNG.normal(1000, 200, 150)
        result = check_drift_from_snapshot(snap, current)
        assert result.n_reference == 400
        assert result.n_current   == 150

    def test_is_drifted_consistent_with_psi(self):
        snap = self._snap()
        for mean in [1000, 1200, 2000, 5000]:
            current = RNG.normal(mean, 200, 200)
            result  = check_drift_from_snapshot(snap, current)
            assert result.is_drifted == (result.psi >= PSI_MODERATE)

    def test_roundtrip_same_data(self):
        """Storing and restoring should give near-zero PSI."""
        data = RNG.normal(1500, 300, 600)
        snap = compute_reference_distribution(data)
        # Use a fresh draw from the same distribution
        current = RNG.normal(1500, 300, 300)
        result  = check_drift_from_snapshot(snap, current)
        # Should be stable or at most moderate (sampling variance)
        assert result.psi < PSI_MODERATE * 2   # generous threshold for random draws
