"""Unit tests for Module 3 — Correlation Analyzer and Module 4 — Model Trainer."""
import os
import pytest
import numpy as np

from app.modules.correlation_analyzer import (
    analyze, CorrelationReport, CorrelationResult,
    _pearson_at_lag, _spearman_at_lag, _first_difference,
    _best_lag_and_coeffs, SIGNIFICANCE_THRESHOLD,
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
# Correlation Analyzer — unit tests on internal functions
# ══════════════════════════════════════════════════════════════════════════════

class TestFirstDifference:
    def test_output_length(self):
        x = np.array([1.0, 3.0, 6.0, 10.0])
        d = _first_difference(x)
        assert len(d) == len(x) - 1

    def test_correct_differences(self):
        x = np.array([1.0, 4.0, 9.0])
        d = _first_difference(x)
        np.testing.assert_array_almost_equal(d, [3.0, 5.0])


class TestPearsonAtLag:
    def test_lag_zero_perfect_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = _pearson_at_lag(x, x, lag=0)
        assert abs(r - 1.0) < 1e-10

    def test_lag_zero_perfect_negative(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = _pearson_at_lag(x, -x, lag=0)
        assert abs(r + 1.0) < 1e-10

    def test_lag_shifts_correctly(self):
        # x = [0,1,2,3,4], y = [1,2,3,4,5] → y = x shifted by 1
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r_lag0 = _pearson_at_lag(x, y, lag=0)
        r_lag1 = _pearson_at_lag(x, y, lag=1)
        # At lag=0 they're already nearly identical; at lag=1 x[:-1] aligns with y[1:]
        assert abs(r_lag0) > 0.99    # both should be ~1 for linear series
        assert abs(r_lag1) > 0.99

    def test_zero_variance_returns_zero(self):
        x = np.array([5.0, 5.0, 5.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        r = _pearson_at_lag(x, y, lag=0)
        assert r == 0.0

    def test_result_in_minus_one_to_one(self):
        rng = np.random.default_rng(0)
        x = rng.random(100)
        y = rng.random(100)
        for lag in [0, 5, 10]:
            r = _pearson_at_lag(x, y, lag)
            assert -1.0 <= r <= 1.0


class TestSpearmanAtLag:
    def test_perfect_monotone(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = _spearman_at_lag(x, x, lag=0)
        assert abs(r - 1.0) < 1e-9

    def test_result_in_minus_one_to_one(self):
        rng = np.random.default_rng(1)
        x = rng.random(80)
        y = rng.random(80)
        for lag in [0, 3, 7]:
            r = _spearman_at_lag(x, y, lag)
            assert -1.0 <= r <= 1.0

    def test_captures_nonlinear_monotone(self):
        # y = x^3 is non-linear but monotone → Spearman ≈ 1
        x = np.linspace(-2, 2, 50)
        y = x ** 3
        r = _spearman_at_lag(x, y, lag=0)
        assert r > 0.99


class TestBestLagAndCoeffs:
    def test_recovers_known_lag(self):
        """Synthetic series: y = x shifted by 5 steps + small noise."""
        rng = np.random.default_rng(42)
        n = 200
        x = np.sin(np.linspace(0, 4 * np.pi, n)) + rng.normal(0, 0.1, n)
        y = np.roll(x, 5)   # shift by 5 — y[5:] matches x[:-5]
        y[:5] = 0.0
        lag, p, s = _best_lag_and_coeffs(x, y, max_lag=20)
        assert lag == 5

    def test_returns_floats(self):
        rng = np.random.default_rng(7)
        x = rng.random(100)
        y = rng.random(100)
        lag, p, s = _best_lag_and_coeffs(x, y, max_lag=10)
        assert isinstance(lag, int)
        assert isinstance(p, float)
        assert isinstance(s, float)


# ══════════════════════════════════════════════════════════════════════════════
# Correlation Analyzer — public API tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCorrelationAnalyzer:
    def test_returns_correlation_report(self):
        report = analyze(_bundle())
        assert isinstance(report, CorrelationReport)

    def test_report_has_three_results(self):
        report = analyze(_bundle())
        assert isinstance(report.cpu, CorrelationResult)
        assert isinstance(report.ram, CorrelationResult)
        assert isinstance(report.network, CorrelationResult)

    def test_pearson_r_in_valid_range(self):
        report = analyze(_bundle())
        for result in [report.cpu, report.ram, report.network]:
            assert -1.0 <= result.pearson_r <= 1.0

    def test_spearman_r_in_valid_range(self):
        report = analyze(_bundle())
        for result in [report.cpu, report.ram, report.network]:
            assert -1.0 <= result.spearman_r <= 1.0

    def test_best_r_is_max_of_both(self):
        report = analyze(_bundle())
        for result in [report.cpu, report.ram, report.network]:
            expected = max(abs(result.pearson_r), abs(result.spearman_r))
            assert abs(result.best_r - expected) < 1e-6

    def test_lag_is_non_negative(self):
        report = analyze(_bundle())
        for result in [report.cpu, report.ram, report.network]:
            assert result.lag_minutes >= 0

    def test_any_significant_reflects_results(self):
        report = analyze(_bundle())
        expected = any(r.is_significant for r in [report.cpu, report.ram, report.network])
        assert report.any_significant == expected

    def test_significance_flag_consistent_with_threshold(self):
        report = analyze(_bundle())
        for result in [report.cpu, report.ram, report.network]:
            if result.is_significant:
                assert result.best_r >= SIGNIFICANCE_THRESHOLD
            else:
                assert result.best_r < SIGNIFICANCE_THRESHOLD

    def test_best_lag_returns_integer(self):
        lag = analyze(_bundle()).best_lag()
        assert isinstance(lag, int)
        assert lag >= 0

    def test_best_lag_zero_when_no_significant(self):
        report = CorrelationReport(
            cpu=CorrelationResult("cpu", 5, 0.3, 0.2, 0.3, False),
            ram=CorrelationResult("ram", 3, 0.2, 0.1, 0.2, False),
            network=CorrelationResult("network", 8, 0.1, 0.05, 0.1, False),
            n_points=100,
        )
        assert report.best_lag() == 0

    def test_n_points_matches_bundle(self):
        bundle = _bundle(lookback_days=3)
        report = analyze(bundle)
        assert report.n_points == len(bundle.business)

    def test_detects_correlation_in_synthetic_data(self):
        """
        The stub data generator creates system metrics that are lagged
        and correlated with business metrics — the analyzer should
        find significant correlations.
        """
        bundle = _bundle(lookback_days=14)
        report = analyze(bundle)
        assert report.any_significant, (
            f"Expected significant correlations in synthetic data, got: "
            f"cpu={report.cpu.best_r:.3f} ram={report.ram.best_r:.3f} "
            f"net={report.network.best_r:.3f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Model Trainer
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildFeatures:
    def test_output_shapes(self):
        bundle = _bundle(lookback_days=3)
        X, y = _build_features(bundle, lag=5)
        assert X.ndim == 2
        assert y.ndim == 2
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 9    # 9 features defined in model_trainer
        assert y.shape[1] == 3    # cpu, ram, net

    def test_lag_zero_still_produces_features(self):
        bundle = _bundle(lookback_days=2)
        X, y = _build_features(bundle, lag=0)
        assert X.shape[0] > 0

    def test_larger_lag_fewer_rows(self):
        bundle = _bundle(lookback_days=3)
        X0, _ = _build_features(bundle, lag=0)
        X30, _ = _build_features(bundle, lag=30)
        assert X0.shape[0] >= X30.shape[0]

    def test_no_nans_in_features(self):
        bundle = _bundle(lookback_days=5)
        X, y = _build_features(bundle, lag=5)
        assert not np.isnan(X).any(), "NaN found in feature matrix"
        assert not np.isnan(y).any(), "NaN found in target matrix"


class TestModelTrainer:
    def test_train_creates_ready_model(self, db):
        cfg = create_config(db, _cfg_data("train-ready"))
        bundle = _bundle()
        report = analyze(bundle)
        model = train_model(db, cfg, bundle, report)
        assert model.id is not None
        assert model.status == ModelStatus.READY

    def test_artifact_is_joblib_file(self, db):
        cfg = create_config(db, _cfg_data("train-artifact"))
        model = train_model(db, cfg, _bundle(), analyze(_bundle()))
        assert model.artifact_path is not None
        assert model.artifact_path.endswith(".joblib")
        assert os.path.exists(model.artifact_path)

    def test_artifact_loadable(self, db):
        """joblib.load should return the {model, scaler} dict."""
        import joblib
        cfg = create_config(db, _cfg_data("train-loadable"))
        model = train_model(db, cfg, _bundle(), analyze(_bundle()))
        artifact = joblib.load(model.artifact_path)
        assert "model" in artifact
        assert "scaler" in artifact

    def test_model_has_real_metrics(self, db):
        cfg = create_config(db, _cfg_data("train-metrics"))
        model = train_model(db, cfg, _bundle(), analyze(_bundle()))
        m = model.metrics
        assert m is not None
        for key in ["mae_cpu", "rmse_cpu", "r2_cpu",
                    "mae_ram", "rmse_ram", "r2_ram",
                    "mae_net", "rmse_net", "r2_net",
                    "mape_overall"]:
            assert key in m, f"Missing metric key: {key}"
            assert isinstance(m[key], float), f"{key} is not a float"

    def test_algorithm_name_stored(self, db):
        cfg = create_config(db, _cfg_data("train-algo"))
        model = train_model(db, cfg, _bundle(), analyze(_bundle()))
        assert model.algorithm in ("gradient_boosting", "ridge")

    def test_model_params_stored(self, db):
        cfg = create_config(db, _cfg_data("train-params"))
        model = train_model(db, cfg, _bundle(), analyze(_bundle()))
        assert model.parameters is not None
        assert "n_features" in model.parameters
        assert model.parameters["n_features"] == 9
        assert "feature_names" in model.parameters

    def test_model_version_increments(self, db):
        cfg = create_config(db, _cfg_data("train-versioning"))
        bundle = _bundle()
        report = analyze(bundle)
        m1 = train_model(db, cfg, bundle, report)
        m2 = train_model(db, cfg, bundle, report)
        assert m2.version == m1.version + 1

    def test_model_has_lag_minutes(self, db):
        cfg = create_config(db, _cfg_data("train-lag"))
        model = train_model(db, cfg, _bundle(), analyze(_bundle()))
        assert model.lag_minutes is not None
        assert model.lag_minutes >= 0

    def test_get_latest_ready_returns_newest(self, db):
        cfg = create_config(db, _cfg_data("train-latest"))
        bundle = _bundle()
        report = analyze(bundle)
        m1 = train_model(db, cfg, bundle, report)
        m2 = train_model(db, cfg, bundle, report)
        assert get_latest_ready_model(db, cfg.id).id == m2.id

    def test_get_latest_ready_returns_none_when_no_model(self, db):
        cfg = create_config(db, _cfg_data("train-none"))
        assert get_latest_ready_model(db, cfg.id) is None

    def test_r2_cpu_is_reasonable(self, db):
        """
        With 14 days of correlated synthetic data, R² should be non-negative
        (model beats the mean-predictor baseline).
        """
        cfg = create_config(db, _cfg_data("train-r2"))
        model = train_model(db, cfg, _bundle(lookback_days=14), analyze(_bundle(lookback_days=14)))
        assert model.metrics["r2_cpu"] >= 0.0, (
            f"R² cpu is negative ({model.metrics['r2_cpu']:.3f}), "
            "model is worse than predicting the mean."
        )

