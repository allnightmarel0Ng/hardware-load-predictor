"""Unit tests for Module 5 — Forecasting Engine."""
import pytest

from app.modules.config_manager import create_config
from app.modules.data_collector import fetch_historical_data
from app.modules.correlation_analyzer import analyze
from app.modules.model_trainer import train_model
from app.modules.forecasting_engine import forecast
from app.models.db_models import ForecastResult
from app.schemas.schemas import ForecastingConfigCreate


# ── helpers ───────────────────────────────────────────────────────────────────

def _prepare_config_and_model(db, name: str):
    """Create a config and train a model on it. Returns (config, model)."""
    cfg = create_config(
        db,
        ForecastingConfigCreate(
            name=name,
            host="prometheus.internal",
            port=9090,
            business_metric_name="orders_per_minute",
            business_metric_formula="sum(rate(orders_total[1m]))",
        ),
    )
    bundle = fetch_historical_data(cfg.host, cfg.port, cfg.business_metric_formula)
    report = analyze(bundle)
    model = train_model(db, cfg, bundle, report)
    return cfg, model


# ── tests ─────────────────────────────────────────────────────────────────────

class TestForecastingEngine:
    def test_returns_forecast_result(self, db):
        cfg, _ = _prepare_config_and_model(db, "fc-basic")
        result = forecast(db, cfg, business_metric_value=1000.0)
        assert isinstance(result, ForecastResult)

    def test_result_is_persisted(self, db):
        cfg, _ = _prepare_config_and_model(db, "fc-persist")
        result = forecast(db, cfg, business_metric_value=500.0)
        from_db = db.get(ForecastResult, result.id)
        assert from_db is not None
        assert from_db.business_metric_value == 500.0

    def test_cpu_is_between_0_and_100(self, db):
        cfg, _ = _prepare_config_and_model(db, "fc-cpu-range")
        result = forecast(db, cfg, business_metric_value=800.0)
        assert 0.0 <= result.predicted_cpu_percent <= 100.0

    def test_ram_is_positive(self, db):
        cfg, _ = _prepare_config_and_model(db, "fc-ram-positive")
        result = forecast(db, cfg, business_metric_value=800.0)
        assert result.predicted_ram_gb >= 0.0

    def test_network_is_positive(self, db):
        cfg, _ = _prepare_config_and_model(db, "fc-net-positive")
        result = forecast(db, cfg, business_metric_value=800.0)
        assert result.predicted_network_mbps >= 0.0

    def test_stores_correct_business_value(self, db):
        cfg, _ = _prepare_config_and_model(db, "fc-biz-val")
        biz_val = 1234.56
        result = forecast(db, cfg, business_metric_value=biz_val)
        assert result.business_metric_value == pytest.approx(biz_val)

    def test_higher_load_predicts_higher_resources(self, db):
        """Linear model: higher business value → higher predicted load."""
        cfg, _ = _prepare_config_and_model(db, "fc-monotone")
        low  = forecast(db, cfg, business_metric_value=100.0)
        high = forecast(db, cfg, business_metric_value=2000.0)
        assert high.predicted_cpu_percent >= low.predicted_cpu_percent
        assert high.predicted_ram_gb >= low.predicted_ram_gb
        assert high.predicted_network_mbps >= low.predicted_network_mbps

    def test_raises_value_error_without_model(self, db):
        cfg = create_config(
            db,
            ForecastingConfigCreate(
                name="fc-no-model",
                host="prometheus.internal",
                port=9090,
                business_metric_name="orders_per_minute",
                business_metric_formula="orders_total",
            ),
        )
        with pytest.raises(ValueError, match="No ready model"):
            forecast(db, cfg, business_metric_value=500.0)

    def test_uses_latest_model_after_retrain(self, db):
        cfg, first_model = _prepare_config_and_model(db, "fc-retrain")
        # Retrain
        bundle = fetch_historical_data(cfg.host, cfg.port, cfg.business_metric_formula)
        second_model = train_model(db, cfg, bundle, analyze(bundle))
        # Forecast should use the second model
        result = forecast(db, cfg, business_metric_value=500.0)
        assert result.model_id == second_model.id
