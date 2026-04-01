"""Unit tests for Module 3 — Correlation Analyzer and Module 4 — Model Trainer."""
import os
import pytest
import numpy as np

from app.modules.correlation_analyzer import (
    analyze, CorrelationReport, CorrelationResult,
    _pearson_at_lag, _spearman_at_lag, _first_difference,
    _best_lag_and_coeffs, SIGNIFICANCE_THRESHOLD, ALL_TARGETS,
)
from app.modules.data_collector import fetch_historical_data
from app.modules.model_trainer import train_model, get_latest_ready_model, _build_features
from app.modules.config_manager import create_config
from app.models.db_models import ModelStatus
from app.schemas.schemas import ForecastingConfigCreate


# ── helpers ───────────────────────────────────────────────────────────────────

def _bundle(lookback_days: int = 7):
    return fetch_historical_data("host", 9090, "orders_metric", lookback_days=lookback_days)


def _cfg_data(name: str = "trainer-test") -> ForecastingConfigCreate:
    return ForecastingConfigCreate(
        name=name,
        host="prometheus.internal",
        port=9090,
        business_metric_name="orders_per_minute",
        business_metric_formula="sum(rate(orders_total[1m]))",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestFirstDifference:
    def test_output_length(self):
        x = np.array([1.0, 3.0, 6.0, 10.0])
        assert len(_first_difference(x)) == len(x) - 1

    def test_correct_differences(self):
        x = np.array([1.0, 4.0, 9.0])
        np.testing.assert_array_almost_equal(_first_difference(x), [3.0, 5.0])


class TestPearsonAtLag:
    def test_perfect_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(_pearson_at_lag(x, x, 0) - 1.0) < 1e-10

    def test_perfect_anti_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(_pearson_at_lag(x, -x, 0) + 1.0) < 1e-10

    def test_zero_variance_returns_zero(self):
        x = np.array([5.0, 5.0, 5.0, 5.0])
        assert _pearson_at_lag(x, np.array([1.0, 2.0, 3.0, 4.0]), 0) == 0.0

    def test_result_in_valid_range(self):
        rng = np.random.default_rng(0)
        x, y = rng.random(100), rng.random(100)
        for lag in [0, 5, 10]:
            assert -1.0 <= _pearson_at_lag(x, y, lag) <= 1.0

    def test_lag_sensitivity(self):
        n, k = 100, 7
        x = np.sin(np.linspace(0, 4 * np.pi, n))
        y = np.zeros(n); y[k:] = x[:-k]
        assert abs(_pearson_at_lag(x, y, k)) > abs(_pearson_at_lag(x, y, 0))


class TestSpearmanAtLag:
    def test_perfect_monotone(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(_spearman_at_lag(x, x, 0) - 1.0) < 1e-9

    def test_result_in_valid_range(self):
        rng = np.random.default_rng(1)
        x, y = rng.random(80), rng.random(80)
        for lag in [0, 3, 7]:
            assert -1.0 <= _spearman_at_lag(x, y, lag) <= 1.0

    def test_captures_nonlinear_monotone(self):
        x = np.linspace(-2, 2, 50)
        assert _spearman_at_lag(x, x ** 3, 0) > 0.99


class TestBestLagAndCoeffs:
    def test_recovers_known_lag(self):
        rng = np.random.default_rng(42)
        n = 200
        x = np.sin(np.linspace(0, 4 * np.pi, n)) + rng.normal(0, 0.1, n)
        y = np.roll(x, 5); y[:5] = 0.0
        lag, p, s = _best_lag_and_coeffs(x, y, max_lag=20)
        assert lag == 5

    def test_returns_correct_types(self):
        rng = np.random.default_rng(7)
        lag, p, s = _best_lag_and_coeffs(rng.random(100), rng.random(100), 10)
        assert isinstance(lag, int)
        assert isinstance(p, float) and isinstance(s, float)


# ══════════════════════════════════════════════════════════════════════════════
# Correlation Analyzer — public API — five targets
# ══════════════════════════════════════════════════════════════════════════════

class TestCorrelationAnalyzer:
    def test_returns_correlation_report(self):
        assert isinstance(analyze(_bundle()), CorrelationReport)

    def test_report_has_five_targets(self):
        report = analyze(_bundle())
        for attr in ("cpu", "ram_gb", "ram_percent", "network", "disk"):
            result = getattr(report, attr)
            assert isinstance(result, CorrelationResult), f"missing/wrong type: {attr}"

    def test_all_results_returns_five(self):
        report = analyze(_bundle())
        results = report.all_results()
        assert len(results) == 5
        names = {r.target_metric for r in results}
        assert names == set(ALL_TARGETS)

    def test_pearson_in_valid_range(self):
        for r in analyze(_bundle()).all_results():
            assert -1.0 <= r.pearson_r <= 1.0, f"{r.target_metric}: {r.pearson_r}"

    def test_spearman_in_valid_range(self):
        for r in analyze(_bundle()).all_results():
            assert -1.0 <= r.spearman_r <= 1.0, f"{r.target_metric}: {r.spearman_r}"

    def test_best_r_equals_max_of_both(self):
        for r in analyze(_bundle()).all_results():
            expected = max(abs(r.pearson_r), abs(r.spearman_r))
            assert abs(r.best_r - expected) < 1e-6

    def test_lag_is_non_negative(self):
        for r in analyze(_bundle()).all_results():
            assert r.lag_minutes >= 0

    def test_significance_consistent_with_threshold(self):
        for r in analyze(_bundle()).all_results():
            if r.is_significant:
                assert r.best_r >= SIGNIFICANCE_THRESHOLD
            else:
                assert r.best_r < SIGNIFICANCE_THRESHOLD

    def test_any_significant_reflects_results(self):
        report = analyze(_bundle())
        expected = any(r.is_significant for r in report.all_results())
        assert report.any_significant == expected

    def test_best_lag_returns_non_negative_int(self):
        lag = analyze(_bundle()).best_lag()
        assert isinstance(lag, int) and lag >= 0

    def test_best_lag_zero_when_nothing_significant(self):
        """All insignificant → best_lag() should return 0 (train with no lag)."""
        report = CorrelationReport(
            cpu=        CorrelationResult("cpu",         5, 0.1, 0.1, 0.1, False),
            ram_gb=     CorrelationResult("ram_gb",      3, 0.1, 0.1, 0.1, False),
            ram_percent=CorrelationResult("ram_percent", 2, 0.1, 0.1, 0.1, False),
            network=    CorrelationResult("network",     8, 0.1, 0.1, 0.1, False),
            disk=       CorrelationResult("disk",        4, 0.1, 0.1, 0.1, False),
            n_points=100,
        )
        assert report.best_lag() == 0
        assert not report.any_significant

    def test_per_target_lag_zero_for_insignificant(self):
        """Insignificant targets should get lag=0 from per_target_lag()."""
        report = CorrelationReport(
            cpu=        CorrelationResult("cpu",         5, 0.8, 0.75, 0.8, True),   # sig
            ram_gb=     CorrelationResult("ram_gb",      3, 0.1, 0.1,  0.1, False),  # not sig
            ram_percent=CorrelationResult("ram_percent", 2, 0.1, 0.1,  0.1, False),
            network=    CorrelationResult("network",     8, 0.7, 0.65, 0.7, True),   # sig
            disk=       CorrelationResult("disk",        4, 0.1, 0.1,  0.1, False),
            n_points=200,
        )
        lags = report.per_target_lag()
        assert lags["cpu"]         == 5   # significant → use discovered lag
        assert lags["ram_gb"]      == 0   # not significant → use 0
        assert lags["ram_percent"] == 0
        assert lags["network"]     == 8   # significant → use discovered lag
        assert lags["disk"]        == 0

    def test_is_business_constant_false_for_normal_data(self):
        report = analyze(_bundle())
        assert report.is_business_constant is False

    def test_constant_business_series_flagged(self):
        """A constant business series should set is_business_constant=True."""
        bundle = _bundle()
        # Override business with a constant series
        const_val = 100.0
        bundle.business = [{"timestamp": p["timestamp"], "value": const_val}
                           for p in bundle.business]
        report = analyze(bundle)
        assert report.is_business_constant is True

    def test_training_not_blocked_by_insignificant_results(self):
        """
        analyze() should NEVER raise even when all correlations are weak.
        The caller decides whether to proceed.
        """
        bundle = _bundle(lookback_days=1)
        # Should not raise
        report = analyze(bundle)
        assert isinstance(report, CorrelationReport)

    def test_n_points_matches_bundle(self):
        bundle = _bundle(lookback_days=3)
        assert analyze(bundle).n_points == len(bundle.business)

    def test_synthetic_stub_produces_some_significant(self):
        """
        The stub generator creates correlated system metrics —
        at least one target should be significant after 14 days.
        """
        report = analyze(_bundle(lookback_days=14))
        assert report.any_significant, (
            "Expected ≥1 significant target in stub data. Got: "
            + ", ".join(f"{r.target_metric}={r.best_r:.3f}" for r in report.all_results())
        )


# ══════════════════════════════════════════════════════════════════════════════
# Model Trainer
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildFeatures:
    def test_output_shapes(self):
        bundle = _bundle(lookback_days=3)
        X, y = _build_features(bundle, lag=5)
        assert X.ndim == 2 and y.ndim == 2
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 9   # 9 features
        assert y.shape[1] == 5   # 5 targets: cpu, ram_gb, ram_pct, net, disk

    def test_lag_zero_still_produces_features(self):
        X, y = _build_features(_bundle(lookback_days=2), lag=0)
        assert X.shape[0] > 0

    def test_larger_lag_fewer_rows(self):
        bundle = _bundle(lookback_days=3)
        X0, _ = _build_features(bundle, lag=0)
        X30, _ = _build_features(bundle, lag=30)
        assert X0.shape[0] >= X30.shape[0]

    def test_no_nans(self):
        bundle = _bundle(lookback_days=5)
        X, y = _build_features(bundle, lag=5)
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()


class TestModelTrainer:
    def test_train_creates_ready_model(self, db):
        cfg = create_config(db, _cfg_data("train-ready"))
        model = train_model(db, cfg, _bundle(), analyze(_bundle()))
        assert model.id is not None
        assert model.status == ModelStatus.READY

    def test_artifact_is_joblib_file(self, db):
        cfg = create_config(db, _cfg_data("train-artifact"))
        model = train_model(db, cfg, _bundle(), analyze(_bundle()))
        assert model.artifact_path is not None
        assert model.artifact_path.endswith(".joblib")
        assert os.path.exists(model.artifact_path)

    def test_artifact_loadable_with_five_outputs(self, db):
        import joblib
        cfg = create_config(db, _cfg_data("train-loadable"))
        model = train_model(db, cfg, _bundle(), analyze(_bundle()))
        art = joblib.load(model.artifact_path)
        assert "model" in art and "scaler" in art
        # Verify model outputs 5 targets
        from app.modules.model_trainer import _build_features
        X, _ = _build_features(_bundle(), lag=model.lag_minutes or 0)
        pred = art["model"].predict(art["scaler"].transform(X[:1]))
        assert pred.shape == (1, 5), f"expected (1,5), got {pred.shape}"

    def test_model_has_five_target_metrics(self, db):
        cfg = create_config(db, _cfg_data("train-metrics"))
        model = train_model(db, cfg, _bundle(), analyze(_bundle()))
        m = model.metrics
        assert m is not None
        for key in ["mae_cpu", "rmse_cpu", "r2_cpu",
                    "mae_ram_gb", "r2_ram_gb",
                    "mae_ram_pct", "r2_ram_pct",
                    "mae_net", "r2_net",
                    "mae_disk", "r2_disk",
                    "mape_overall"]:
            assert key in m, f"Missing metric: {key}"
            assert isinstance(m[key], float)

    def test_algorithm_stored(self, db):
        cfg = create_config(db, _cfg_data("train-algo"))
        model = train_model(db, cfg, _bundle(), analyze(_bundle()))
        assert model.algorithm in ("gradient_boosting", "ridge")

    def test_params_stored(self, db):
        cfg = create_config(db, _cfg_data("train-params"))
        model = train_model(db, cfg, _bundle(), analyze(_bundle()))
        assert model.parameters is not None
        assert model.parameters["n_features"] == 9
        assert "feature_names" in model.parameters

    def test_version_increments(self, db):
        cfg = create_config(db, _cfg_data("train-versioning"))
        bundle, report = _bundle(), analyze(_bundle())
        m1 = train_model(db, cfg, bundle, report)
        m2 = train_model(db, cfg, bundle, report)
        assert m2.version == m1.version + 1

    def test_lag_minutes_stored(self, db):
        cfg = create_config(db, _cfg_data("train-lag"))
        model = train_model(db, cfg, _bundle(), analyze(_bundle()))
        assert model.lag_minutes is not None
        assert model.lag_minutes >= 0

    def test_get_latest_ready_returns_newest(self, db):
        cfg = create_config(db, _cfg_data("train-latest"))
        bundle, report = _bundle(), analyze(_bundle())
        m1 = train_model(db, cfg, bundle, report)
        m2 = train_model(db, cfg, bundle, report)
        assert get_latest_ready_model(db, cfg.id).id == m2.id

    def test_get_latest_ready_none_when_no_model(self, db):
        cfg = create_config(db, _cfg_data("train-none"))
        assert get_latest_ready_model(db, cfg.id) is None

    def test_trains_even_when_no_significant_correlation(self, db):
        """
        Training must proceed even if correlation is below threshold for all targets.
        The model will predict mean values (R² ≈ 0) — still a valid baseline.
        """
        cfg = create_config(db, _cfg_data("train-no-sig"))
        bundle = _bundle()
        # Force all results to be insignificant
        report = CorrelationReport(
            cpu=        CorrelationResult("cpu",         0, 0.1, 0.1, 0.1, False),
            ram_gb=     CorrelationResult("ram_gb",      0, 0.1, 0.1, 0.1, False),
            ram_percent=CorrelationResult("ram_percent", 0, 0.1, 0.1, 0.1, False),
            network=    CorrelationResult("network",     0, 0.1, 0.1, 0.1, False),
            disk=       CorrelationResult("disk",        0, 0.1, 0.1, 0.1, False),
            n_points=len(bundle.business),
        )
        # Should not raise; training proceeds with lag=0
        model = train_model(db, cfg, bundle, report)
        assert model.status == ModelStatus.READY
        assert model.lag_minutes == 0

    def test_r2_cpu_non_negative_with_correlated_data(self, db):
        """With 14 days of correlated synthetic data, model beats the mean."""
        cfg = create_config(db, _cfg_data("train-r2"))
        bundle = _bundle(lookback_days=14)
        model = train_model(db, cfg, bundle, analyze(bundle))
        assert model.metrics["r2_cpu"] >= 0.0, (
            f"R² cpu negative: {model.metrics['r2_cpu']:.3f}"
        )
