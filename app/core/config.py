from __future__ import annotations

import os
import re
from pathlib import Path

import yaml


def _interpolate(value: str) -> str:
    def _replace(match: re.Match) -> str:
        var = match.group(1)
        result = os.environ.get(var)
        if result is None:
            defaults = {
                "DATABASE_URL": "sqlite:///./test.db",
                "METRICS_SOURCE_URL": "http://localhost:9090",
            }
            if var in defaults:
                return defaults[var]
            raise RuntimeError(
                f"config.yaml references ${{{var}}} but the environment "
                f"variable {var!r} is not set."
            )
        return result
    return re.sub(r"\$\{([^}]+)\}", _replace, value)


def _interpolate_recursive(obj: object) -> object:
    if isinstance(obj, str):
        return _interpolate(obj)
    if isinstance(obj, dict):
        return {k: _interpolate_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_recursive(i) for i in obj]
    return obj


def _load_yaml() -> dict:
    config_path = Path(os.environ.get("CONFIG_PATH", "config.yaml"))
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path.resolve()}. "
            "Make sure config.yaml is present in the project root."
        )
    with config_path.open() as fh:
        raw = yaml.safe_load(fh)
    return _interpolate_recursive(raw)  # type: ignore[return-value]


_CPU_QUERY     = '100 - avg(rate(node_cpu_seconds_total{{mode="idle",{instance}}}[5m])) * 100'
_RAM_GB_QUERY  = "(node_memory_MemTotal_bytes{{{instance}}} - node_memory_MemAvailable_bytes{{{instance}}}) / 1073741824"
_RAM_PCT_QUERY = "(1 - node_memory_MemAvailable_bytes{{{instance}}} / node_memory_MemTotal_bytes{{{instance}}}) * 100"
_NET_QUERY     = "sum(rate(node_network_receive_bytes_total{{{instance}}}[5m])) * 8 / 1048576"
_DISK_QUERY    = "avg(rate(node_disk_io_time_seconds_total{{{instance}}}[5m])) * 100"


class Settings:
    def __init__(self, data: dict) -> None:
        app  = data.get("app", {})
        db   = data.get("database", {})
        prom = data.get("prometheus", {})

        self.database_url:           str   = db.get("url", "")
        self.metrics_source_url:     str   = prom.get("url", "http://localhost:9090")
        self.model_storage_path:     str   = app.get("model_storage_path", "./models")
        self.accuracy_threshold:     float = float(app.get("accuracy_threshold", 0.85))
        self.retrain_interval_hours: int   = int(app.get("retrain_interval_hours", 24))
        self.log_level:              str   = app.get("log_level", "INFO")
        self.use_prometheus_stub:    bool  = bool(app.get("use_prometheus_stub", False))
        self.step_seconds:           int   = int(app.get("step_seconds", 300))

        self.prometheus_cpu_query:     str = _CPU_QUERY
        self.prometheus_ram_gb_query:  str = _RAM_GB_QUERY
        self.prometheus_ram_pct_query: str = _RAM_PCT_QUERY
        self.prometheus_net_query:     str = _NET_QUERY
        self.prometheus_disk_query:    str = _DISK_QUERY


settings = Settings(_load_yaml())
