"""
Integration tests for the REST API (Module 7 — Request Handler).
Uses FastAPI TestClient — no real HTTP, but full routing + DB pipeline.
"""
import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

CONFIG_PAYLOAD = {
    "name": "api-test-config",
    "host": "prometheus.internal",
    "port": 9090,
    "business_metric_name": "orders_per_minute",
    "business_metric_formula": "sum(rate(orders_total[1m]))",
}


def _create_and_train(client, name: str = "api-test-config") -> dict:
    """Convenience: create config + trigger training, return config dict."""
    payload = {**CONFIG_PAYLOAD, "name": name}
    r = client.post("/configs/", json=payload)
    assert r.status_code == 201
    config = r.json()

    train_r = client.post(
        f"/configs/{config['id']}/train/",
        json={"lookback_days": 7},
    )
    assert train_r.status_code == 202
    return config


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
        create_r = client.post("/configs/", json={**CONFIG_PAYLOAD, "name": "get-cfg"})
        config_id = create_r.json()["id"]
        r = client.get(f"/configs/{config_id}")
        assert r.status_code == 200
        assert r.json()["id"] == config_id

    def test_get_missing_config_404(self, client):
        r = client.get("/configs/999999")
        assert r.status_code == 404

    def test_update_config_200(self, client):
        create_r = client.post("/configs/", json={**CONFIG_PAYLOAD, "name": "upd-cfg"})
        config_id = create_r.json()["id"]
        r = client.patch(f"/configs/{config_id}", json={"host": "new-host.internal"})
        assert r.status_code == 200
        assert r.json()["host"] == "new-host.internal"
        assert r.json()["name"] == "upd-cfg"  # unchanged

    def test_delete_config_204(self, client):
        create_r = client.post("/configs/", json={**CONFIG_PAYLOAD, "name": "del-cfg"})
        config_id = create_r.json()["id"]
        r = client.delete(f"/configs/{config_id}")
        assert r.status_code == 204
        assert client.get(f"/configs/{config_id}").status_code == 404

    def test_create_config_missing_field_422(self, client):
        r = client.post("/configs/", json={"name": "incomplete"})
        assert r.status_code == 422


# ── /configs/{id}/train ───────────────────────────────────────────────────────

class TestTrainEndpoints:
    def test_train_returns_202(self, client):
        r_cfg = client.post("/configs/", json={**CONFIG_PAYLOAD, "name": "train-202"})
        config_id = r_cfg.json()["id"]
        r = client.post(f"/configs/{config_id}/train/", json={"lookback_days": 7})
        assert r.status_code == 202
        body = r.json()
        assert "model_id" in body
        assert body["status"] == "ready"

    def test_train_missing_config_404(self, client):
        r = client.post("/configs/999999/train/", json={"lookback_days": 7})
        assert r.status_code == 404

    def test_train_twice_creates_two_versions(self, client):
        r_cfg = client.post("/configs/", json={**CONFIG_PAYLOAD, "name": "train-2x"})
        config_id = r_cfg.json()["id"]
        r1 = client.post(f"/configs/{config_id}/train/", json={"lookback_days": 7})
        r2 = client.post(f"/configs/{config_id}/train/", json={"lookback_days": 7})
        assert r1.json()["model_id"] != r2.json()["model_id"]

    def test_list_models_returns_trained(self, client):
        r_cfg = client.post("/configs/", json={**CONFIG_PAYLOAD, "name": "list-models"})
        config_id = r_cfg.json()["id"]
        client.post(f"/configs/{config_id}/train/", json={"lookback_days": 7})
        r = client.get(f"/configs/{config_id}/train/models")
        assert r.status_code == 200
        assert len(r.json()) >= 1


# ── /configs/{id}/forecast ────────────────────────────────────────────────────

class TestForecastEndpoints:
    def test_forecast_returns_predictions(self, client):
        config = _create_and_train(client, "fc-api-1")
        r = client.post(
            f"/configs/{config['id']}/forecast/",
            json={"business_metric_value": 1000.0},
        )
        assert r.status_code == 200
        body = r.json()
        assert "predicted_cpu_percent" in body
        assert "predicted_ram_gb" in body
        assert "predicted_network_mbps" in body

    def test_forecast_cpu_in_valid_range(self, client):
        config = _create_and_train(client, "fc-api-2")
        r = client.post(
            f"/configs/{config['id']}/forecast/",
            json={"business_metric_value": 500.0},
        )
        cpu = r.json()["predicted_cpu_percent"]
        assert 0.0 <= cpu <= 100.0

    def test_forecast_without_model_returns_409(self, client):
        r_cfg = client.post("/configs/", json={**CONFIG_PAYLOAD, "name": "fc-no-mdl"})
        config_id = r_cfg.json()["id"]
        r = client.post(
            f"/configs/{config_id}/forecast/",
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
        config = _create_and_train(client, "fc-api-3")
        r = client.post(
            f"/configs/{config['id']}/forecast/",
            json={"business_metric_value": -10.0},
        )
        assert r.status_code == 422

    def test_forecast_stores_result(self, client):
        config = _create_and_train(client, "fc-api-4")
        r = client.post(
            f"/configs/{config['id']}/forecast/",
            json={"business_metric_value": 750.0},
        )
        assert r.json()["business_metric_value"] == 750.0
        assert r.json()["config_id"] == config["id"]
