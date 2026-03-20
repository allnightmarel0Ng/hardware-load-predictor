"""Unit tests for Module 3 — Correlation Analyzer and Module 4 — Model Trainer."""
import pytest

from app.modules.correlation_analyzer import analyze, CorrelationReport, CorrelationResult
from app.modules.data_collector import fetch_historical_data
from app.modules.model_trainer import train_model, get_latest_ready_model
from app.modules.config_manager import create_config
from app.models.db_models import ModelStatus
from app.schemas.schemas import ForecastingConfigCreate


# ── helpers ───────────────────────────────────────────────────────────────────

def _bundle():
    return fetch_historical_data("host", 9090, "orders_metric", lookback_days=7)


def _cfg_data(name: str = "trainer-test") -> ForecastingConfigCreate:
    return ForecastingConfigCreate(
        name=name,
        host="prometheus.internal",
        port=9090,
        business_metric_name="orders_per_minute",
        business_metric_formula="sum(rate(orders_total[1m]))",
    )


# ── Correlation Analyzer ──────────────────────────────────────────────────────

class TestCorrelationAnalyzer:
    def test_returns_correlation_report(self):
        bundle = _bundle()
        report = analyze(bundle)
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

    def test_lag_is_non_negative(self):
        report = analyze(_bundle())
        for result in [report.cpu, report.ram, report.network]:
            assert result.lag_minutes >= 0

    def test_any_significant_reflects_results(self):
        report = analyze(_bundle())
        expected = any(
            r.is_significant for r in [report.cpu, report.ram, report.network]
        )
        assert report.any_significant == expected

    def test_best_lag_returns_integer(self):
        report = analyze(_bundle())
        lag = report.best_lag()
        assert isinstance(lag, int)
        assert lag >= 0

    def test_best_lag_zero_when_no_significant(self):
        """If no correlations are significant, best_lag should be 0."""
        report = CorrelationReport(
            cpu=CorrelationResult("cpu", 5, 0.3, False),
            ram=CorrelationResult("ram", 3, 0.2, False),
            network=CorrelationResult("network", 8, 0.1, False),
        )
        assert report.best_lag() == 0


# ── Model Trainer ─────────────────────────────────────────────────────────────

class TestModelTrainer:
    def test_train_creates_ready_model(self, db):
        cfg = create_config(db, _cfg_data("train-ready"))
        bundle = _bundle()
        report = analyze(bundle)
        model = train_model(db, cfg, bundle, report)
        assert model.id is not None
        assert model.status == ModelStatus.READY

    def test_model_has_artifact_path(self, db):
        cfg = create_config(db, _cfg_data("train-artifact"))
        model = train_model(db, cfg, _bundle(), analyze(_bundle()))
        assert model.artifact_path is not None
        import os
        assert os.path.exists(model.artifact_path)

    def test_model_has_metrics(self, db):
        cfg = create_config(db, _cfg_data("train-metrics"))
        model = train_model(db, cfg, _bundle(), analyze(_bundle()))
        assert model.metrics is not None
        assert "r2_cpu" in model.metrics
        assert "mae_cpu" in model.metrics

    def test_model_has_parameters(self, db):
        cfg = create_config(db, _cfg_data("train-params"))
        model = train_model(db, cfg, _bundle(), analyze(_bundle()))
        assert model.parameters is not None
        assert "cpu_coeff" in model.parameters
        assert "ram_coeff" in model.parameters
        assert "net_coeff" in model.parameters

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
        latest = get_latest_ready_model(db, cfg.id)
        assert latest.id == m2.id

    def test_get_latest_ready_returns_none_when_no_model(self, db):
        cfg = create_config(db, _cfg_data("train-none"))
        result = get_latest_ready_model(db, cfg.id)
        assert result is None
