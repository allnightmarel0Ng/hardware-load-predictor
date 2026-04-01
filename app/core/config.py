from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://predictor:predictor@localhost:5432/predictor"
    metrics_source_url: str = "http://localhost:9090"
    model_storage_path: str = "./models"
    accuracy_threshold: float = 0.85
    retrain_interval_hours: int = 24
    log_level: str = "INFO"

    # ── Prometheus query expressions for system metrics ───────────────────────
    # Override these in .env if your node_exporter labels differ.
    prometheus_cpu_query: str = (
        '100 - avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100'
    )
    prometheus_ram_gb_query: str = (
        "(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / 1073741824"
    )
    prometheus_ram_pct_query: str = (
        "(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100"
    )
    prometheus_net_query: str = (
        "sum(irate(node_network_receive_bytes_total[5m])) * 8 / 1048576"
    )
    prometheus_disk_query: str = (
        "avg(irate(node_disk_io_time_seconds_total[5m])) * 100"
    )

    # ── Stub mode ─────────────────────────────────────────────────────────────
    # Set USE_PROMETHEUS_STUB=true in .env to keep synthetic data even when
    # Prometheus is configured (useful for local development without Prometheus).
    use_prometheus_stub: bool = False

    class Config:
        env_file = ".env"


settings = Settings()
