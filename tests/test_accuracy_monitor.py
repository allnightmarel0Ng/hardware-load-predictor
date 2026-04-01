"""
Tests for Module 6 — Accuracy Monitor.

Strategy: monkeypatch _fetch_actuals_from_prometheus so all tests run
without a real Prometheus instance.  Everything else — DB writes, metric
computation, retraining trigger, scheduler integration — is real.
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from app.modules.accuracy_monitor import (
    ActualValues,
    _compute_post_deployment_metrics,
    _needs_retraining,
    _backfill_actuals,
    _evaluate_model,
    get_accuracy_status,
    force_evaluate,
    MIN_EVAL_SAMPLES,
)
from app.models.db_models import (
    ForecastResult,
    ForecastingConfig,
    ModelEvaluation,
    ModelStatus,
    TrainedModel,
)
from app.modules.config_manager import create_config
from app.modules.correlation_analyzer import analyze
from app.modules.data_collector import fetch_historical_data
from app.modules.model_trainer import train_model
from app.schemas.schemas import ForecastingConfigCreate


# ── fixtures ──────────────────────────────────────────────────────────────────

def _cfg_data(name: str) -> ForecastingConfigCreate:
    return ForecastingConfigCreate(
        name=name, host="prometheus.internal", port=9090,
        business_metric_name="orders_per_minute",
        business_metric_formula="sum(rate(orders_total[1m]))",
    )


def _make_trained_model(db, name: str) -> tuple[ForecastingConfig, TrainedModel]:
    cfg = create_config(db, _cfg_data(name))
    bundle = fetch_historical_data("h", 9090, "orders", lookback_days=14)
    report = analyze(bundle)
    model = train_model(db, cfg, bundle, report)
    return cfg, model


def _insert_forecast_result(
    db,
    config_id: int,
    model_id: int,
    *,
    predicted_cpu: float = 50.0,
    predicted_ram: float = 10.0,
    predicted_net: float = 100.0,
    minutes_ago: int = 30,
    with_actuals: bool = False,
    actual_cpu: float = 48.0,
    actual_ram: float = 10.5,
    actual_net: float = 105.0,
) -> ForecastResult:
    fr = ForecastResult(
        config_id=config_id,
        model_id=model_id,
        business_metric_value=1000.0,
        predicted_cpu_percent=predicted_cpu,
        predicted_ram_gb=predicted_ram,
        predicted_network_mbps=predicted_net,
        created_at=datetime.utcnow() - timedelta(minutes=minutes_ago),
    )
    if with_actuals:
        fr.actual_cpu_percent  = actual_cpu
        fr.actual_ram_gb       = actual_ram
        fr.actual_network_mbps = actual_net
        fr.actuals_fetched_at  = datetime.utcnow()
    db.add(fr)
    db.commit()
    db.refresh(fr)
    return fr


GOOD_ACTUALS = ActualValues(cpu_percent=48.0, ram_gb=10.5, network_mbps=105.0)
BAD_ACTUALS  = ActualValues(cpu_percent=95.0, ram_gb=60.0, network_mbps=900.0)


# ══════════════════════════════════════════════════════════════════════════════
# Unit tests — pure metric computation
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeMetrics:
    def _rows(self, n: int, cpu_err: float = 2.0) -> list:
        rows = []
        for i in range(n):
            fr = MagicMock()
            fr.predicted_cpu_percent  = 50.0
            fr.predicted_ram_gb       = 10.0
            fr.predicted_network_mbps = 100.0
            fr.actual_cpu_percent     = 50.0 + cpu_err
            fr.actual_ram_gb          = 10.0 + 0.5
            fr.actual_network_mbps    = 100.0 + 5.0
            rows.append(fr)
        return rows

    def test_returns_all_metric_keys(self):
        rows = self._rows(10)
        m = _compute_post_deployment_metrics(rows)
        for key in ["mae_cpu", "mae_ram", "mae_net",
                    "rmse_cpu", "rmse_ram", "rmse_net",
                    "r2_cpu", "r2_ram", "r2_net", "mape_overall"]:
            assert key in m, f"Missing: {key}"

    def test_perfect_predictions_give_r2_one(self):
        """When predictions == actuals exactly, R² should be 1.0."""
        rows = []
        for v in [30.0, 40.0, 50.0, 60.0, 70.0]:
            fr = MagicMock()
            fr.predicted_cpu_percent  = v
            fr.actual_cpu_percent     = v
            fr.predicted_ram_gb       = 10.0
            fr.actual_ram_gb          = 10.0
            fr.predicted_network_mbps = 100.0
            fr.actual_network_mbps    = 100.0
            rows.append(fr)
        m = _compute_post_deployment_metrics(rows)
        assert m["mae_cpu"]  == 0.0
        assert m["rmse_cpu"] == 0.0
        assert m["r2_cpu"]   == 1.0

    def test_large_errors_give_low_r2(self):
        rows = self._rows(20, cpu_err=40.0)   # 40% constant error
        m = _compute_post_deployment_metrics(rows)
        assert m["mae_cpu"] == pytest.approx(40.0, abs=0.01)
        assert m["r2_cpu"] <= 0.0   # constant-error model can't explain variance

    def test_mape_overall_is_non_negative(self):
        rows = self._rows(10, cpu_err=5.0)
        m = _compute_post_deployment_metrics(rows)
        assert m["mape_overall"] >= 0.0


class TestNeedsRetraining:
    def test_good_r2_does_not_trigger(self):
        m = {"r2_cpu": 0.92, "r2_ram": 0.89, "r2_net": 0.91}
        assert not _needs_retraining(m, threshold=0.85)

    def test_bad_r2_triggers(self):
        m = {"r2_cpu": 0.60, "r2_ram": 0.55, "r2_net": 0.70}
        assert _needs_retraining(m, threshold=0.85)

    def test_mixed_r2_uses_average(self):
        # avg = (0.90 + 0.50 + 0.90) / 3 = 0.767 — below 0.85 threshold
        m = {"r2_cpu": 0.90, "r2_ram": 0.50, "r2_net": 0.90}
        assert _needs_retraining(m, threshold=0.85)

    def test_empty_metrics_no_retrain(self):
        assert not _needs_retraining({}, threshold=0.85)


# ══════════════════════════════════════════════════════════════════════════════
# Integration tests — DB + Prometheus mock
# ══════════════════════════════════════════════════════════════════════════════

class TestBackfillActuals:
    def test_fetches_and_writes_actuals(self, db):
        cfg, model = _make_trained_model(db, "backfill-write")
        fr = _insert_forecast_result(db, cfg.id, model.id, minutes_ago=30)
        assert fr.actuals_fetched_at is None

        with patch(
            "app.modules.accuracy_monitor._fetch_actuals_from_prometheus",
            return_value=GOOD_ACTUALS,
        ):
            filled = _backfill_actuals(db, model)

        assert len(filled) == 1
        db.refresh(fr)
        assert fr.actual_cpu_percent  == GOOD_ACTUALS.cpu_percent
        assert fr.actual_ram_gb       == GOOD_ACTUALS.ram_gb
        assert fr.actual_network_mbps == GOOD_ACTUALS.network_mbps
        assert fr.actuals_fetched_at  is not None

    def test_skips_recent_forecasts(self, db):
        """Forecasts created < LAG_FETCH_BUFFER_MINUTES ago should be skipped."""
        cfg, model = _make_trained_model(db, "backfill-recent")
        _insert_forecast_result(db, cfg.id, model.id, minutes_ago=2)

        with patch(
            "app.modules.accuracy_monitor._fetch_actuals_from_prometheus",
            return_value=GOOD_ACTUALS,
        ) as mock_prom:
            filled = _backfill_actuals(db, model)

        assert len(filled) == 0
        mock_prom.assert_not_called()

    def test_skips_already_fetched(self, db):
        """Rows that already have actuals_fetched_at set should not be re-fetched."""
        cfg, model = _make_trained_model(db, "backfill-skip")
        _insert_forecast_result(db, cfg.id, model.id, minutes_ago=30, with_actuals=True)

        with patch(
            "app.modules.accuracy_monitor._fetch_actuals_from_prometheus",
            return_value=GOOD_ACTUALS,
        ) as mock_prom:
            filled = _backfill_actuals(db, model)

        assert len(filled) == 0
        mock_prom.assert_not_called()

    def test_handles_prometheus_returning_none(self, db):
        """If Prometheus is unreachable, the row stays pending for next run."""
        cfg, model = _make_trained_model(db, "backfill-prom-fail")
        fr = _insert_forecast_result(db, cfg.id, model.id, minutes_ago=30)

        with patch(
            "app.modules.accuracy_monitor._fetch_actuals_from_prometheus",
            return_value=None,
        ):
            filled = _backfill_actuals(db, model)

        assert len(filled) == 0
        db.refresh(fr)
        assert fr.actuals_fetched_at is None   # still pending

    def test_multiple_rows_all_filled(self, db):
        cfg, model = _make_trained_model(db, "backfill-multi")
        for _ in range(3):
            _insert_forecast_result(db, cfg.id, model.id, minutes_ago=60)

        with patch(
            "app.modules.accuracy_monitor._fetch_actuals_from_prometheus",
            return_value=GOOD_ACTUALS,
        ):
            filled = _backfill_actuals(db, model)

        assert len(filled) == 3


class TestEvaluateModel:
    def _setup_evaluated_model(self, db, name: str, n: int, cpu_err: float = 2.0):
        """Helper: create a model with n forecast-actual pairs already populated."""
        cfg, model = _make_trained_model(db, name)
        for i in range(n):
            _insert_forecast_result(
                db, cfg.id, model.id,
                minutes_ago=60 + i,
                with_actuals=True,
                actual_cpu=50.0 + cpu_err,
                actual_ram=10.5,
                actual_net=105.0,
            )
        return cfg, model

    def test_creates_evaluation_record(self, db):
        _, model = self._setup_evaluated_model(db, "eval-create", n=MIN_EVAL_SAMPLES)
        with patch("app.modules.accuracy_monitor._fetch_actuals_from_prometheus",
                   return_value=None):
            ev = _evaluate_model(db, model)
        assert ev is not None
        assert ev.model_id == model.id
        assert ev.n_samples == MIN_EVAL_SAMPLES

    def test_evaluation_metrics_populated(self, db):
        _, model = self._setup_evaluated_model(db, "eval-metrics", n=MIN_EVAL_SAMPLES)
        with patch("app.modules.accuracy_monitor._fetch_actuals_from_prometheus",
                   return_value=None):
            ev = _evaluate_model(db, model)
        assert ev.r2_cpu   is not None
        assert ev.mae_cpu  is not None
        assert ev.mape_overall is not None

    def test_skips_if_too_few_samples(self, db):
        _, model = self._setup_evaluated_model(db, "eval-few", n=MIN_EVAL_SAMPLES - 1)
        with patch("app.modules.accuracy_monitor._fetch_actuals_from_prometheus",
                   return_value=None):
            ev = _evaluate_model(db, model)
        assert ev is None

    def test_triggers_retrain_on_bad_accuracy(self, db):
        # cpu_err=50 forces R² ≈ 0 (constant error, no variance explained)
        _, model = self._setup_evaluated_model(
            db, "eval-retrain", n=MIN_EVAL_SAMPLES + 5, cpu_err=50.0
        )
        retrain_called = []

        def fake_retrain(db, config_id):
            retrain_called.append(config_id)

        with patch("app.modules.accuracy_monitor._fetch_actuals_from_prometheus",
                   return_value=None), \
             patch("app.modules.accuracy_monitor._trigger_retrain", fake_retrain):
            ev = _evaluate_model(db, model)

        assert ev is not None
        assert ev.triggered_retrain is True
        assert len(retrain_called) == 1

    def test_no_retrain_on_good_accuracy(self, db):
        # cpu_err=1.0 → small error, high R² (constant predictions, variance from setup)
        _, model = self._setup_evaluated_model(
            db, "eval-good", n=MIN_EVAL_SAMPLES + 5, cpu_err=1.0
        )
        retrain_called = []

        with patch("app.modules.accuracy_monitor._fetch_actuals_from_prometheus",
                   return_value=None), \
             patch("app.modules.accuracy_monitor._trigger_retrain",
                   side_effect=lambda db, cid: retrain_called.append(cid)):
            ev = _evaluate_model(db, model)

        assert len(retrain_called) == 0

    def test_backfill_happens_before_evaluation(self, db):
        """
        If rows start with no actuals, _evaluate_model should fetch them first
        then use them for metric computation.
        """
        cfg, model = _make_trained_model(db, "eval-backfill-first")
        for _ in range(MIN_EVAL_SAMPLES):
            _insert_forecast_result(db, cfg.id, model.id, minutes_ago=60)

        with patch(
            "app.modules.accuracy_monitor._fetch_actuals_from_prometheus",
            return_value=GOOD_ACTUALS,
        ):
            ev = _evaluate_model(db, model)

        assert ev is not None
        assert ev.n_samples == MIN_EVAL_SAMPLES


class TestAccuracyStatusEndpoint:
    def test_no_evaluations_is_unhealthy(self, db):
        _, model = _make_trained_model(db, "status-none")
        status = get_accuracy_status(db, model.id)
        assert status["is_healthy"] is False
        assert "No evaluations" in status["health_reason"]

    def test_healthy_with_good_metrics(self, db):
        _, model = _make_trained_model(db, "status-good")
        ev = ModelEvaluation(
            model_id=model.id, config_id=model.config_id,
            n_samples=10,
            r2_cpu=0.92, r2_ram=0.88, r2_net=0.90,
            mae_cpu=2.0, mae_ram=0.5, mae_net=8.0,
            rmse_cpu=3.0, rmse_ram=0.8, rmse_net=12.0,
            mape_overall=5.0, triggered_retrain=False,
        )
        db.add(ev); db.commit()
        status = get_accuracy_status(db, model.id)
        assert status["is_healthy"] is True
        assert status["health_reason"] is None
        assert status["n_evaluations"] == 1

    def test_unhealthy_with_low_r2(self, db):
        _, model = _make_trained_model(db, "status-bad")
        ev = ModelEvaluation(
            model_id=model.id, config_id=model.config_id,
            n_samples=10,
            r2_cpu=0.50, r2_ram=0.45, r2_net=0.55,
            mae_cpu=15.0, mae_ram=5.0, mae_net=80.0,
            rmse_cpu=20.0, rmse_ram=7.0, rmse_net=100.0,
            mape_overall=35.0, triggered_retrain=True,
        )
        db.add(ev); db.commit()
        status = get_accuracy_status(db, model.id)
        assert status["is_healthy"] is False
        assert "R²" in status["health_reason"]

    def test_returns_empty_dict_for_missing_model(self, db):
        result = get_accuracy_status(db, 999_999)
        assert result == {}

    def test_counts_total_samples_across_evals(self, db):
        _, model = _make_trained_model(db, "status-total")
        for n in [5, 10, 8]:
            db.add(ModelEvaluation(
                model_id=model.id, config_id=model.config_id,
                n_samples=n, r2_cpu=0.9, r2_ram=0.9, r2_net=0.9,
                mae_cpu=2.0, mae_ram=0.5, mae_net=8.0,
                rmse_cpu=3.0, rmse_ram=0.8, rmse_net=12.0,
                mape_overall=5.0, triggered_retrain=False,
            ))
        db.commit()
        status = get_accuracy_status(db, model.id)
        assert status["n_evaluations"] == 3
        assert status["n_samples_total"] == 23


class TestForceEvaluate:
    def test_returns_none_for_missing_model(self, db):
        result = force_evaluate(db, 999_999)
        assert result is None

    def test_returns_evaluation_when_enough_samples(self, db):
        cfg, model = _make_trained_model(db, "force-eval-ok")
        for _ in range(MIN_EVAL_SAMPLES):
            _insert_forecast_result(db, cfg.id, model.id,
                                    minutes_ago=60, with_actuals=True)
        with patch("app.modules.accuracy_monitor._fetch_actuals_from_prometheus",
                   return_value=None):
            result = force_evaluate(db, model.id)
        assert result is not None
        assert isinstance(result, ModelEvaluation)

    def test_returns_none_when_too_few_samples(self, db):
        cfg, model = _make_trained_model(db, "force-eval-few")
        _insert_forecast_result(db, cfg.id, model.id,
                                minutes_ago=60, with_actuals=True)
        with patch("app.modules.accuracy_monitor._fetch_actuals_from_prometheus",
                   return_value=None):
            result = force_evaluate(db, model.id)
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# API endpoint tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAccuracyAPIEndpoints:
    def _setup(self, client, name: str):
        cfg_r = client.post("/configs/", json={
            "name": name, "host": "h", "port": 9090,
            "business_metric_name": "orders",
            "business_metric_formula": "orders_total",
        })
        config_id = cfg_r.json()["id"]
        train_r = client.post(f"/configs/{config_id}/train/", json={"lookback_days": 14})
        model_id = train_r.json()["model_id"]
        return config_id, model_id

    def test_get_accuracy_status_404_for_missing(self, client):
        r = client.get("/models/999999/accuracy")
        assert r.status_code == 404

    def test_get_accuracy_status_returns_unhealthy_initially(self, client):
        _, model_id = self._setup(client, "api-status-new")
        r = client.get(f"/models/{model_id}/accuracy")
        assert r.status_code == 200
        body = r.json()
        assert body["model_id"] == model_id
        assert body["is_healthy"] is False
        assert body["n_evaluations"] == 0

    def test_force_evaluate_409_without_samples(self, client):
        _, model_id = self._setup(client, "api-force-empty")
        with patch("app.modules.accuracy_monitor._fetch_actuals_from_prometheus",
                   return_value=None):
            r = client.post(f"/models/{model_id}/accuracy/evaluate")
        assert r.status_code == 409

    def test_get_accuracy_history_empty_list(self, client):
        _, model_id = self._setup(client, "api-history-empty")
        r = client.get(f"/models/{model_id}/accuracy/history")
        assert r.status_code == 200
        assert r.json() == []
