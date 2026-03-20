"""Unit tests for Module 1 — Configuration Manager."""
import pytest
from fastapi import HTTPException

from app.modules.config_manager import (
    create_config,
    get_config,
    list_configs,
    update_config,
    delete_config,
)
from app.schemas.schemas import ForecastingConfigCreate, ForecastingConfigUpdate


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_config(name: str = "test-config") -> ForecastingConfigCreate:
    return ForecastingConfigCreate(
        name=name,
        host="prometheus.internal",
        port=9090,
        business_metric_name="orders_per_minute",
        business_metric_formula="sum(rate(orders_total[1m]))",
    )


# ── tests ─────────────────────────────────────────────────────────────────────

class TestCreateConfig:
    def test_creates_and_returns_config(self, db):
        cfg = create_config(db, _make_config())
        assert cfg.id is not None
        assert cfg.name == "test-config"
        assert cfg.host == "prometheus.internal"
        assert cfg.port == 9090

    def test_duplicate_name_raises_409(self, db):
        create_config(db, _make_config("dup"))
        with pytest.raises(HTTPException) as exc_info:
            create_config(db, _make_config("dup"))
        assert exc_info.value.status_code == 409

    def test_different_names_both_created(self, db):
        c1 = create_config(db, _make_config("alpha"))
        c2 = create_config(db, _make_config("beta"))
        assert c1.id != c2.id


class TestGetConfig:
    def test_returns_existing(self, db):
        created = create_config(db, _make_config("get-me"))
        fetched = get_config(db, created.id)
        assert fetched.id == created.id

    def test_missing_raises_404(self, db):
        with pytest.raises(HTTPException) as exc_info:
            get_config(db, 999_999)
        assert exc_info.value.status_code == 404


class TestListConfigs:
    def test_returns_all(self, db):
        create_config(db, _make_config("list-a"))
        create_config(db, _make_config("list-b"))
        result = list_configs(db)
        names = [c.name for c in result]
        assert "list-a" in names
        assert "list-b" in names

    def test_respects_limit(self, db):
        for i in range(5):
            create_config(db, _make_config(f"lim-{i}"))
        result = list_configs(db, limit=2)
        assert len(result) <= 2


class TestUpdateConfig:
    def test_updates_host(self, db):
        cfg = create_config(db, _make_config("upd-me"))
        updated = update_config(db, cfg.id, ForecastingConfigUpdate(host="new-host"))
        assert updated.host == "new-host"
        assert updated.name == "upd-me"  # unchanged

    def test_partial_update_leaves_others_intact(self, db):
        cfg = create_config(db, _make_config("partial"))
        updated = update_config(db, cfg.id, ForecastingConfigUpdate(port=9999))
        assert updated.port == 9999
        assert updated.host == "prometheus.internal"

    def test_update_missing_raises_404(self, db):
        with pytest.raises(HTTPException) as exc_info:
            update_config(db, 999_999, ForecastingConfigUpdate(host="x"))
        assert exc_info.value.status_code == 404


class TestDeleteConfig:
    def test_deletes_successfully(self, db):
        cfg = create_config(db, _make_config("del-me"))
        delete_config(db, cfg.id)
        with pytest.raises(HTTPException) as exc_info:
            get_config(db, cfg.id)
        assert exc_info.value.status_code == 404

    def test_delete_missing_raises_404(self, db):
        with pytest.raises(HTTPException) as exc_info:
            delete_config(db, 999_999)
        assert exc_info.value.status_code == 404
