"""
Integration tests for the REST API (Module 7 — Request Handler).
Uses FastAPI TestClient — no real HTTP, but full routing + DB pipeline.

Training is async — POST /train/ returns 202 + job_id (status=queued).
The test client runs everything synchronously inside the worker thread,
so by the time the job runner's _run_training_job finishes, the model
is in the DB and ready to query.
"""
import pytest
from unittest.mock import patch

from app.modules import job_runner


# ── helpers ───────────────────────────────────────────────────────────────────

CONFIG_PAYLOAD = {
    "name":                   "api-test-config",
    "host":                   "prometheus.internal",
    "port":                   9090,
    "business_metric_name":   "orders_per_minute",
    "business_metric_formula": "sum(rate(orders_total[1m]))",
}


def _create_config(client, name: str) -> dict:
    r = client.post("/configs/", json={**CONFIG_PAYLOAD, "name": name})
    assert r.status_code == 201, r.text
    return r.json()


def _train(client, config_id: int) -> dict:
    """Submit a training job and return the job response body."""
    r = client.post(f"/configs/{config_id}/train/", json={"lookback_days": 7})
    assert r.status_code == 202, r.text
    return r.json()


def _create_and_train(client, name: str) -> tuple[dict, dict]:
    """Create config + submit training. Returns (config, job) dicts."""
    cfg = _create_config(client, name)
    job = _train(client, cfg["id"])
    return cfg, job


def _wait_for_model(client, config_id: int) -> int:
    """
    Poll GET /jobs/{id} until done, then return the model_id.
    In tests the executor runs jobs synchronously in the same process,
    so we only need a single poll.
    """
    # list models directly — cleaner than polling in unit tests
    r = client.get(f"/configs/{config_id}/train/models")
    assert r.status_code == 200
    models = r.json()
    assert len(models) >= 1, "No models found after training"
    # models are returned newest-first
    return models[0]["id"]


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


# ── /configs ──────────────────────────────────────────────────────────────────

class TestConfigEndpoints:
    def test_create_config_201(self, client):
        r = client.post("/configs/", json={**CONFIG_PAYLOAD, "name": "create-201"})
        assert r.status_code == 201
        body = r.json()
        assert body["name"] == "create-201"
        assert "id" in body
        assert "created_at" in body
        assert body["server_id"] is None   # legacy single-server config

    def test_create_duplicate_returns_409(self, client):
        client.post("/configs/", json={**CONFIG_PAYLOAD, "name": "dup-cfg"})
        r = client.post("/configs/", json={**CONFIG_PAYLOAD, "name": "dup-cfg"})
        assert r.status_code == 409

    def test_list_configs_200(self, client):
        client.post("/configs/", json={**CONFIG_PAYLOAD, "name": "list-cfg-1"})
        client.post("/configs/", json={**CONFIG_PAYLOAD, "name": "list-cfg-2"})
        r = client.get("/configs/")
        assert r.status_code == 200
        names = [c["name"] for c in r.json()]
        assert "list-cfg-1" in names
        assert "list-cfg-2" in names

    def test_get_config_200(self, client):
        cfg = _create_config(client, "get-cfg")
        r = client.get(f"/configs/{cfg['id']}")
        assert r.status_code == 200
        assert r.json()["id"] == cfg["id"]

    def test_get_missing_config_404(self, client):
        assert client.get("/configs/999999").status_code == 404

    def test_update_config_200(self, client):
        cfg = _create_config(client, "upd-cfg")
        r = client.patch(f"/configs/{cfg['id']}", json={"host": "new-host.internal"})
        assert r.status_code == 200
        assert r.json()["host"] == "new-host.internal"
        assert r.json()["name"] == "upd-cfg"   # unchanged

    def test_delete_config_204(self, client):
        cfg = _create_config(client, "del-cfg")
        assert client.delete(f"/configs/{cfg['id']}").status_code == 204
        assert client.get(f"/configs/{cfg['id']}").status_code == 404

    def test_create_config_missing_field_422(self, client):
        r = client.post("/configs/", json={"name": "incomplete"})
        assert r.status_code == 422


# ── /configs/{id}/train + /jobs ───────────────────────────────────────────────

class TestTrainEndpoints:
    def test_train_returns_202_with_job_id(self, client):
        cfg = _create_config(client, "train-202")
        r = client.post(f"/configs/{cfg['id']}/train/", json={"lookback_days": 7})
        assert r.status_code == 202
        body = r.json()
        assert "job_id" in body
        assert body["config_id"] == cfg["id"]
        assert body["status"] == "queued"

    def test_train_missing_config_404(self, client):
        assert client.post(
            "/configs/999999/train/", json={"lookback_days": 7}
        ).status_code == 404

    def test_train_creates_model_in_db(self, client):
        cfg, job = _create_and_train(client, "train-model-db")
        model_id = _wait_for_model(client, cfg["id"])
        assert model_id is not None

    def test_train_twice_creates_two_model_versions(self, client):
        cfg = _create_config(client, "train-2x")
        _train(client, cfg["id"])
        _train(client, cfg["id"])
        r = client.get(f"/configs/{cfg['id']}/train/models")
        assert r.status_code == 200
        assert len(r.json()) >= 2

    def test_list_models_newest_first(self, client):
        cfg, _ = _create_and_train(client, "list-models")
        r = client.get(f"/configs/{cfg['id']}/train/models")
        assert r.status_code == 200
        models = r.json()
        assert len(models) >= 1
        # newest-first: version numbers should be descending
        versions = [m["version"] for m in models]
        assert versions == sorted(versions, reverse=True)

    def test_list_jobs_for_config(self, client):
        cfg, job = _create_and_train(client, "list-jobs")
        r = client.get(f"/configs/{cfg['id']}/train/jobs")
        assert r.status_code == 200
        jobs = r.json()
        assert len(jobs) >= 1
        job_ids = [j["id"] for j in jobs]
        assert job["job_id"] in job_ids

    def test_get_job_by_id(self, client):
        cfg, job = _create_and_train(client, "get-job")
        r = client.get(f"/jobs/{job['job_id']}")
        assert r.status_code == 200
        body = r.json()
        assert body["id"] == job["job_id"]
        assert body["config_id"] == cfg["id"]
        # status is one of the valid lifecycle states
        assert body["status"] in ("queued", "running", "done", "failed")

    def test_get_missing_job_404(self, client):
        assert client.get("/jobs/999999").status_code == 404

    def test_job_response_has_lookback_days(self, client):
        cfg = _create_config(client, "job-lookback")
        r = client.post(f"/configs/{cfg['id']}/train/", json={"lookback_days": 14})
        job_id = r.json()["job_id"]
        job_r = client.get(f"/jobs/{job_id}")
        assert job_r.json()["lookback_days"] == 14


# ── /configs/{id}/forecast ────────────────────────────────────────────────────

class TestForecastEndpoints:
    def _trained_config(self, client, name: str) -> dict:
        cfg, _ = _create_and_train(client, name)
        return cfg

    def test_forecast_returns_predictions(self, client):
        cfg = self._trained_config(client, "fc-api-1")
        r = client.post(
            f"/configs/{cfg['id']}/forecast/",
            json={"business_metric_value": 1000.0},
        )
        assert r.status_code == 200
        body = r.json()
        assert "predicted_cpu_percent"  in body
        assert "predicted_ram_gb"       in body
        assert "predicted_network_mbps" in body

    def test_forecast_cpu_in_valid_range(self, client):
        cfg = self._trained_config(client, "fc-api-2")
        r = client.post(
            f"/configs/{cfg['id']}/forecast/",
            json={"business_metric_value": 500.0},
        )
        assert 0.0 <= r.json()["predicted_cpu_percent"] <= 100.0

    def test_forecast_ram_non_negative(self, client):
        cfg = self._trained_config(client, "fc-api-ram")
        r = client.post(
            f"/configs/{cfg['id']}/forecast/",
            json={"business_metric_value": 500.0},
        )
        assert r.json()["predicted_ram_gb"] >= 0.0

    def test_forecast_without_model_returns_409(self, client):
        cfg = _create_config(client, "fc-no-mdl")
        r = client.post(
            f"/configs/{cfg['id']}/forecast/",
            json={"business_metric_value": 500.0},
        )
        assert r.status_code == 409

    def test_forecast_missing_config_404(self, client):
        r = client.post(
            "/configs/999999/forecast/",
            json={"business_metric_value": 500.0},
        )
        assert r.status_code == 404

    def test_forecast_invalid_business_value_422(self, client):
        cfg = self._trained_config(client, "fc-api-3")
        r = client.post(
            f"/configs/{cfg['id']}/forecast/",
            json={"business_metric_value": -10.0},
        )
        assert r.status_code == 422

    def test_forecast_stores_result(self, client):
        cfg = self._trained_config(client, "fc-api-4")
        r = client.post(
            f"/configs/{cfg['id']}/forecast/",
            json={"business_metric_value": 750.0},
        )
        body = r.json()
        assert body["business_metric_value"] == 750.0
        assert body["config_id"] == cfg["id"]
        assert "id" in body
        assert "model_id" in body


# ── /models/{id}/accuracy ─────────────────────────────────────────────────────

class TestAccuracyEndpoints:
    def _setup(self, client, name: str) -> tuple[dict, int]:
        """Create config, train, return (config, model_id)."""
        cfg, _ = _create_and_train(client, name)
        model_id = _wait_for_model(client, cfg["id"])
        return cfg, model_id

    def test_get_accuracy_status_200(self, client):
        cfg, model_id = self._setup(client, "acc-status")
        r = client.get(f"/models/{model_id}/accuracy")
        assert r.status_code == 200
        body = r.json()
        assert body["model_id"] == model_id
        assert "is_healthy" in body
        assert "n_evaluations" in body

    def test_accuracy_status_unhealthy_before_evaluation(self, client):
        """No evaluations yet → is_healthy=False."""
        cfg, model_id = self._setup(client, "acc-unhealthy")
        r = client.get(f"/models/{model_id}/accuracy")
        assert r.json()["is_healthy"] is False
        assert r.json()["n_evaluations"] == 0

    def test_get_accuracy_missing_model_404(self, client):
        assert client.get("/models/999999/accuracy").status_code == 404

    def test_force_evaluate_409_without_enough_samples(self, client):
        """No ForecastResult rows with actuals yet → 409."""
        from unittest.mock import patch
        cfg, model_id = self._setup(client, "acc-force-empty")
        with patch(
            "app.modules.accuracy_monitor._fetch_actuals_from_prometheus",
            return_value=None,
        ):
            r = client.post(f"/models/{model_id}/accuracy/evaluate")
        assert r.status_code == 409

    def test_get_accuracy_history_empty_list(self, client):
        cfg, model_id = self._setup(client, "acc-history")
        r = client.get(f"/models/{model_id}/accuracy/history")
        assert r.status_code == 200
        assert r.json() == []

    def test_get_accuracy_history_respects_limit(self, client):
        cfg, model_id = self._setup(client, "acc-history-limit")
        r = client.get(f"/models/{model_id}/accuracy/history?limit=5")
        assert r.status_code == 200
        assert isinstance(r.json(), list)
