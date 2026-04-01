"""
Tests for multi-server support:
  - ServerGroup CRUD (server_group_manager)
  - Server CRUD
  - Config provisioning
  - Cluster forecast (cluster_forecaster)
  - API endpoints
"""
import pytest
from unittest.mock import patch, MagicMock

from app.models.db_models import ForecastingConfig, Server, ServerGroup
from app.modules.server_group_manager import (
    create_group, get_group, list_groups, update_group, delete_group,
    add_server, get_server, list_servers, update_server, remove_server,
    provision_group_configs,
)
from app.modules.cluster_forecaster import (
    ClusterForecast, ServerForecast, forecast_cluster,
)
from app.schemas.schemas import (
    ServerCreate, ServerGroupCreate, ServerGroupUpdate, ServerUpdate,
)
from fastapi import HTTPException


# ── fixtures ──────────────────────────────────────────────────────────────────

def _group_data(name: str = "test-cluster") -> ServerGroupCreate:
    return ServerGroupCreate(
        name=name,
        description="Test cluster",
        business_metric_name="orders_per_minute",
        business_metric_formula="sum(rate(orders_total[1m]))",
        metrics_host="prometheus.internal",
        metrics_port=9090,
    )

def _server_data(name: str = "node-1", host: str = "10.0.0.1") -> ServerCreate:
    return ServerCreate(name=name, host=host, port=9090,
                        tags={"datacenter": "eu-west-1"})


# ══════════════════════════════════════════════════════════════════════════════
# ServerGroup CRUD
# ══════════════════════════════════════════════════════════════════════════════

class TestServerGroupCRUD:
    def test_create_group(self, db):
        g = create_group(db, _group_data())
        assert g.id is not None
        assert g.name == "test-cluster"
        assert g.metrics_host == "prometheus.internal"
        assert g.business_metric_formula == "sum(rate(orders_total[1m]))"

    def test_duplicate_name_raises_409(self, db):
        create_group(db, _group_data("dup"))
        with pytest.raises(HTTPException) as exc:
            create_group(db, _group_data("dup"))
        assert exc.value.status_code == 409

    def test_get_group_returns_correct(self, db):
        g = create_group(db, _group_data("get-me"))
        fetched = get_group(db, g.id)
        assert fetched.id == g.id

    def test_get_missing_group_raises_404(self, db):
        with pytest.raises(HTTPException) as exc:
            get_group(db, 999_999)
        assert exc.value.status_code == 404

    def test_list_groups(self, db):
        create_group(db, _group_data("list-a"))
        create_group(db, _group_data("list-b"))
        groups = list_groups(db)
        names = [g.name for g in groups]
        assert "list-a" in names and "list-b" in names

    def test_update_group(self, db):
        g = create_group(db, _group_data("upd-group"))
        updated = update_group(db, g.id, ServerGroupUpdate(description="new desc"))
        assert updated.description == "new desc"
        assert updated.name == "upd-group"   # unchanged

    def test_delete_group(self, db):
        g = create_group(db, _group_data("del-group"))
        delete_group(db, g.id)
        with pytest.raises(HTTPException):
            get_group(db, g.id)

    def test_delete_cascades_to_servers(self, db):
        g = create_group(db, _group_data("cascade-group"))
        add_server(db, g.id, _server_data("cascade-node"))
        delete_group(db, g.id)
        # Server should be gone too
        assert db.query(Server).filter_by(group_id=g.id).count() == 0


# ══════════════════════════════════════════════════════════════════════════════
# Server CRUD
# ══════════════════════════════════════════════════════════════════════════════

class TestServerCRUD:
    def test_add_server(self, db):
        g = create_group(db, _group_data("srv-add"))
        s = add_server(db, g.id, _server_data())
        assert s.id is not None
        assert s.group_id == g.id
        assert s.name == "node-1"
        assert s.host == "10.0.0.1"
        assert s.tags == {"datacenter": "eu-west-1"}
        assert s.is_active is True

    def test_duplicate_server_name_in_group_raises_409(self, db):
        g = create_group(db, _group_data("srv-dup"))
        add_server(db, g.id, _server_data("dup-node"))
        with pytest.raises(HTTPException) as exc:
            add_server(db, g.id, _server_data("dup-node"))
        assert exc.value.status_code == 409

    def test_same_name_different_groups_is_ok(self, db):
        g1 = create_group(db, _group_data("grp-1"))
        g2 = create_group(db, _group_data("grp-2"))
        s1 = add_server(db, g1.id, _server_data("shared-name"))
        s2 = add_server(db, g2.id, _server_data("shared-name"))
        assert s1.id != s2.id

    def test_get_server(self, db):
        g = create_group(db, _group_data("srv-get"))
        s = add_server(db, g.id, _server_data())
        fetched = get_server(db, g.id, s.id)
        assert fetched.id == s.id

    def test_get_server_wrong_group_raises_404(self, db):
        g1 = create_group(db, _group_data("srv-wg1"))
        g2 = create_group(db, _group_data("srv-wg2"))
        s = add_server(db, g1.id, _server_data())
        with pytest.raises(HTTPException) as exc:
            get_server(db, g2.id, s.id)   # server belongs to g1, not g2
        assert exc.value.status_code == 404

    def test_list_servers_all(self, db):
        g = create_group(db, _group_data("srv-list"))
        add_server(db, g.id, _server_data("n1", "10.0.0.1"))
        add_server(db, g.id, _server_data("n2", "10.0.0.2"))
        servers = list_servers(db, g.id)
        assert len(servers) == 2

    def test_list_servers_active_only(self, db):
        g = create_group(db, _group_data("srv-active"))
        s1 = add_server(db, g.id, _server_data("active", "10.0.0.1"))
        s2 = add_server(db, g.id, _server_data("inactive", "10.0.0.2"))
        update_server(db, g.id, s2.id, ServerUpdate(is_active=False))
        active = list_servers(db, g.id, active_only=True)
        assert len(active) == 1
        assert active[0].id == s1.id

    def test_update_server(self, db):
        g = create_group(db, _group_data("srv-upd"))
        s = add_server(db, g.id, _server_data())
        updated = update_server(db, g.id, s.id,
                                ServerUpdate(host="10.0.99.99", is_active=False))
        assert updated.host == "10.0.99.99"
        assert updated.is_active is False
        assert updated.name == "node-1"   # unchanged

    def test_remove_server(self, db):
        g = create_group(db, _group_data("srv-rm"))
        s = add_server(db, g.id, _server_data())
        remove_server(db, g.id, s.id)
        with pytest.raises(HTTPException):
            get_server(db, g.id, s.id)

    def test_add_server_to_missing_group_raises_404(self, db):
        with pytest.raises(HTTPException) as exc:
            add_server(db, 999_999, _server_data())
        assert exc.value.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
# Config provisioning
# ══════════════════════════════════════════════════════════════════════════════

class TestProvisionGroupConfigs:
    def test_creates_config_per_server(self, db):
        g = create_group(db, _group_data("prov-group"))
        add_server(db, g.id, _server_data("n1", "10.0.0.1"))
        add_server(db, g.id, _server_data("n2", "10.0.0.2"))
        configs = provision_group_configs(db, g.id)
        assert len(configs) == 2

    def test_config_inherits_group_formula(self, db):
        g = create_group(db, _group_data("prov-formula"))
        add_server(db, g.id, _server_data())
        configs = provision_group_configs(db, g.id)
        assert configs[0].business_metric_formula == g.business_metric_formula

    def test_config_uses_server_host_port(self, db):
        g = create_group(db, _group_data("prov-host"))
        s = add_server(db, g.id, _server_data("n1", "192.168.1.1"))
        configs = provision_group_configs(db, g.id)
        assert configs[0].host == "192.168.1.1"
        assert configs[0].server_id == s.id

    def test_idempotent_does_not_duplicate(self, db):
        g = create_group(db, _group_data("prov-idem"))
        add_server(db, g.id, _server_data())
        provision_group_configs(db, g.id)   # first call
        created = provision_group_configs(db, g.id)  # second call
        assert len(created) == 0   # nothing new

    def test_skips_inactive_servers(self, db):
        g = create_group(db, _group_data("prov-inactive"))
        s = add_server(db, g.id, _server_data())
        update_server(db, g.id, s.id, ServerUpdate(is_active=False))
        configs = provision_group_configs(db, g.id)
        assert len(configs) == 0

    def test_config_name_format(self, db):
        g = create_group(db, _group_data("my-cluster"))
        add_server(db, g.id, _server_data("api-1"))
        configs = provision_group_configs(db, g.id)
        assert configs[0].name == "my-cluster::api-1"


# ══════════════════════════════════════════════════════════════════════════════
# Cluster Forecaster
# ══════════════════════════════════════════════════════════════════════════════

class TestClusterForecaster:
    def _setup_group_with_models(self, db, name: str, n_servers: int = 2):
        """Create a group with n servers, each with a provisioned config."""
        from app.modules.config_manager import create_config
        from app.modules.data_collector import fetch_historical_data
        from app.modules.correlation_analyzer import analyze
        from app.modules.model_trainer import train_model

        g = create_group(db, _group_data(name))
        for i in range(n_servers):
            s = add_server(db, g.id, _server_data(f"node-{i}", f"10.0.0.{i+1}"))
            # Create config linked to this server
            from app.schemas.schemas import ForecastingConfigCreate
            from app.modules.config_manager import create_config
            cfg = db.query(ForecastingConfig).filter_by(server_id=s.id).first()
            if cfg is None:
                from app.schemas.schemas import ForecastingConfigCreate
                cfg_data = ForecastingConfigCreate(
                    name=f"{name}::node-{i}",
                    host=s.host, port=9090,
                    business_metric_name=g.business_metric_name,
                    business_metric_formula=g.business_metric_formula,
                )
                cfg = create_config(db, cfg_data)
                cfg.server_id = s.id
                db.commit()
            # Train a model for this config
            bundle = fetch_historical_data(cfg.host, cfg.port, cfg.business_metric_formula, lookback_days=7)
            report = analyze(bundle)
            train_model(db, cfg, bundle, report)
        return g

    def test_returns_cluster_forecast(self, db):
        g = self._setup_group_with_models(db, "cf-basic", n_servers=2)
        result = forecast_cluster(db, g.id, 1000.0)
        assert isinstance(result, ClusterForecast)
        assert result.n_servers == 2
        assert len(result.servers) == 2

    def test_per_server_predictions_in_range(self, db):
        g = self._setup_group_with_models(db, "cf-range", n_servers=2)
        result = forecast_cluster(db, g.id, 1000.0)
        for sf in result.servers:
            assert 0.0 <= sf.predicted_cpu_percent <= 100.0
            assert sf.predicted_ram_gb >= 0.0
            assert sf.predicted_network_mbps >= 0.0

    def test_cluster_aggregates_computed(self, db):
        g = self._setup_group_with_models(db, "cf-agg", n_servers=3)
        result = forecast_cluster(db, g.id, 1000.0)
        expected_avg = sum(s.predicted_cpu_percent for s in result.servers) / len(result.servers)
        expected_ram = sum(s.predicted_ram_gb for s in result.servers)
        expected_net = sum(s.predicted_network_mbps for s in result.servers)
        assert abs(result.cluster_cpu_avg_percent - expected_avg) < 0.01
        assert abs(result.cluster_ram_total_gb    - expected_ram) < 0.01
        assert abs(result.cluster_network_total_mbps - expected_net) < 0.01

    def test_empty_group_returns_zero_servers(self, db):
        g = create_group(db, _group_data("cf-empty"))
        result = forecast_cluster(db, g.id, 1000.0)
        assert result.n_servers == 0
        assert result.servers == []

    def test_server_without_model_is_skipped(self, db):
        g = create_group(db, _group_data("cf-skip"))
        s = add_server(db, g.id, _server_data("no-model"))
        # Provision config but do NOT train
        provision_group_configs(db, g.id)
        result = forecast_cluster(db, g.id, 1000.0)
        assert result.n_servers == 0
        assert len(result.skipped_servers) == 1
        assert "no-model" in result.skipped_servers[0]

    def test_server_names_in_results(self, db):
        g = self._setup_group_with_models(db, "cf-names", n_servers=2)
        result = forecast_cluster(db, g.id, 1000.0)
        names = {sf.server_name for sf in result.servers}
        assert "node-0" in names and "node-1" in names

    def test_group_name_in_result(self, db):
        g = self._setup_group_with_models(db, "cf-gname", n_servers=1)
        result = forecast_cluster(db, g.id, 500.0)
        assert result.group_name == "cf-gname"
        assert result.business_metric_value == 500.0


# ══════════════════════════════════════════════════════════════════════════════
# API endpoint tests
# ══════════════════════════════════════════════════════════════════════════════

GROUP_PAYLOAD = {
    "name": "api-test-group",
    "description": "Test group",
    "business_metric_name": "orders_per_minute",
    "business_metric_formula": "sum(rate(orders_total[1m]))",
    "metrics_host": "prometheus.internal",
    "metrics_port": 9090,
}
SERVER_PAYLOAD = {"name": "node-1", "host": "10.0.0.1", "port": 9090}


class TestGroupAPIEndpoints:
    def _create_group(self, client, name: str = "api-test-group") -> dict:
        r = client.post("/groups/", json={**GROUP_PAYLOAD, "name": name})
        assert r.status_code == 201
        return r.json()

    def test_create_group_201(self, client):
        r = client.post("/groups/", json={**GROUP_PAYLOAD, "name": "api-create"})
        assert r.status_code == 201
        body = r.json()
        assert body["name"] == "api-create"
        assert body["metrics_host"] == "prometheus.internal"

    def test_create_duplicate_group_409(self, client):
        client.post("/groups/", json={**GROUP_PAYLOAD, "name": "api-dup"})
        r = client.post("/groups/", json={**GROUP_PAYLOAD, "name": "api-dup"})
        assert r.status_code == 409

    def test_list_groups(self, client):
        self._create_group(client, "list-g1")
        self._create_group(client, "list-g2")
        r = client.get("/groups/")
        assert r.status_code == 200
        names = [g["name"] for g in r.json()]
        assert "list-g1" in names and "list-g2" in names

    def test_get_group_200(self, client):
        g = self._create_group(client, "get-g")
        r = client.get(f"/groups/{g['id']}")
        assert r.status_code == 200
        assert r.json()["id"] == g["id"]

    def test_get_missing_group_404(self, client):
        assert client.get("/groups/999999").status_code == 404

    def test_update_group(self, client):
        g = self._create_group(client, "upd-g")
        r = client.patch(f"/groups/{g['id']}", json={"description": "updated"})
        assert r.status_code == 200
        assert r.json()["description"] == "updated"

    def test_delete_group_204(self, client):
        g = self._create_group(client, "del-g")
        assert client.delete(f"/groups/{g['id']}").status_code == 204
        assert client.get(f"/groups/{g['id']}").status_code == 404

    def test_add_server(self, client):
        g = self._create_group(client, "add-srv-g")
        r = client.post(f"/groups/{g['id']}/servers/", json=SERVER_PAYLOAD)
        assert r.status_code == 201
        body = r.json()
        assert body["name"] == "node-1"
        assert body["group_id"] == g["id"]
        assert body["is_active"] is True

    def test_list_servers(self, client):
        g = self._create_group(client, "list-srv-g")
        client.post(f"/groups/{g['id']}/servers/", json={"name":"n1","host":"10.0.0.1","port":9090})
        client.post(f"/groups/{g['id']}/servers/", json={"name":"n2","host":"10.0.0.2","port":9090})
        r = client.get(f"/groups/{g['id']}/servers/")
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_update_server_active(self, client):
        g = self._create_group(client, "upd-srv-g")
        s = client.post(f"/groups/{g['id']}/servers/", json=SERVER_PAYLOAD).json()
        r = client.patch(f"/groups/{g['id']}/servers/{s['id']}", json={"is_active": False})
        assert r.status_code == 200
        assert r.json()["is_active"] is False

    def test_remove_server(self, client):
        g = self._create_group(client, "rm-srv-g")
        s = client.post(f"/groups/{g['id']}/servers/", json=SERVER_PAYLOAD).json()
        assert client.delete(f"/groups/{g['id']}/servers/{s['id']}").status_code == 204
        assert client.get(f"/groups/{g['id']}/servers/{s['id']}").status_code == 404

    def test_provision_creates_configs(self, client):
        g = self._create_group(client, "prov-g")
        client.post(f"/groups/{g['id']}/servers/", json={"name":"n1","host":"10.0.0.1","port":9090})
        client.post(f"/groups/{g['id']}/servers/", json={"name":"n2","host":"10.0.0.2","port":9090})
        r = client.post(f"/groups/{g['id']}/provision")
        assert r.status_code == 200
        body = r.json()
        assert body["configs_created"] == 2
        assert len(body["config_ids"]) == 2

    def test_provision_idempotent(self, client):
        g = self._create_group(client, "prov-idem-g")
        client.post(f"/groups/{g['id']}/servers/", json=SERVER_PAYLOAD)
        client.post(f"/groups/{g['id']}/provision")
        r2 = client.post(f"/groups/{g['id']}/provision")
        assert r2.json()["configs_created"] == 0  # already provisioned

    def test_group_train_submits_jobs(self, client):
        g = self._create_group(client, "train-g")
        client.post(f"/groups/{g['id']}/servers/", json={"name":"n1","host":"10.0.0.1","port":9090})
        client.post(f"/groups/{g['id']}/provision")
        with patch("app.modules.job_runner._executor") as mock_exec:
            mock_exec.submit = lambda fn, jid: None
            r = client.post(f"/groups/{g['id']}/train/", json={"lookback_days": 7})
        assert r.status_code == 202
        jobs = r.json()
        assert len(jobs) == 1
        assert jobs[0]["status"] == "queued"

    def test_group_train_no_servers_422(self, client):
        g = self._create_group(client, "train-empty-g")
        r = client.post(f"/groups/{g['id']}/train/", json={"lookback_days": 7})
        assert r.status_code == 422

    def test_group_train_no_configs_422(self, client):
        g = self._create_group(client, "train-noprov-g")
        client.post(f"/groups/{g['id']}/servers/", json=SERVER_PAYLOAD)
        # Note: NOT calling /provision — so no configs exist
        r = client.post(f"/groups/{g['id']}/train/", json={"lookback_days": 7})
        assert r.status_code == 422

    def test_cluster_forecast_no_models_409(self, client):
        g = self._create_group(client, "fc-no-models-g")
        client.post(f"/groups/{g['id']}/servers/", json=SERVER_PAYLOAD)
        client.post(f"/groups/{g['id']}/provision")
        r = client.post(
            f"/groups/{g['id']}/forecast/",
            json={"business_metric_value": 1000.0},
        )
        assert r.status_code == 409
