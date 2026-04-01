"""
Cluster Forecaster
────────────────────────────────────────────────────────────────────────────
Produces predictions for every active server in a ServerGroup from a single
business metric value.

This is the multi-server extension of the forecasting engine.  Instead of
forecasting for one config at a time, the engineer passes a group_id and a
business metric value and gets back a prediction per server plus aggregated
cluster-level totals.

Aggregation:
  - cluster_cpu_avg_percent     = mean of per-server CPU predictions
  - cluster_ram_total_gb        = sum of per-server RAM predictions
  - cluster_network_total_mbps  = sum of per-server network predictions

These aggregates are useful for cluster-level capacity planning: the SRE
team can ask "if we expect 5,000 orders/min, does the cluster have enough
headroom?" without inspecting every server individually.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.models.db_models import ForecastingConfig, ForecastResult, Server, ServerGroup
from app.modules.forecasting_engine import forecast as _single_forecast
from app.modules.server_group_manager import get_group, list_servers

logger = logging.getLogger(__name__)


@dataclass
class ServerForecast:
    """Prediction for a single server."""
    server_id:                int
    server_name:              str
    host:                     str
    config_id:                int
    predicted_cpu_percent:    float
    predicted_ram_gb:         float
    predicted_ram_percent:    float
    predicted_network_mbps:   float
    predicted_disk_io_percent: float
    forecast_result_id:       int


@dataclass
class ClusterForecast:
    """Aggregated cluster-level prediction across all active servers."""
    group_id:              int
    group_name:            str
    business_metric_value: float
    n_servers:             int
    servers:               list[ServerForecast]
    # Aggregates
    cluster_cpu_avg_percent:     float = field(default=0.0)
    cluster_ram_total_gb:        float = field(default=0.0)
    cluster_ram_avg_percent:     float = field(default=0.0)
    cluster_network_total_mbps:  float = field(default=0.0)
    cluster_disk_avg_io_percent: float = field(default=0.0)
    skipped_servers:             list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.servers:
            n = len(self.servers)
            self.cluster_cpu_avg_percent     = round(sum(s.predicted_cpu_percent     for s in self.servers) / n,  2)
            self.cluster_ram_total_gb        = round(sum(s.predicted_ram_gb          for s in self.servers),      2)
            self.cluster_ram_avg_percent     = round(sum(s.predicted_ram_percent     for s in self.servers) / n,  2)
            self.cluster_network_total_mbps  = round(sum(s.predicted_network_mbps    for s in self.servers),      2)
            self.cluster_disk_avg_io_percent = round(sum(s.predicted_disk_io_percent for s in self.servers) / n,  2)


def forecast_cluster(
    db: Session,
    group_id: int,
    business_metric_value: float,
) -> ClusterForecast:
    """
    Run inference for every active server in the group and return a
    ClusterForecast with per-server results and cluster-level aggregates.

    Servers whose ForecastingConfig has no READY model are skipped and
    listed in ClusterForecast.skipped_servers — this allows partial
    responses when some servers have not been trained yet.

    Args:
        db:                    SQLAlchemy session.
        group_id:              ServerGroup.id.
        business_metric_value: Expected value of the shared business metric.

    Returns:
        ClusterForecast with per-server and cluster aggregates.
    """
    group = get_group(db, group_id)
    active_servers = list_servers(db, group_id, active_only=True)

    if not active_servers:
        logger.warning("Group %d has no active servers.", group_id)
        return ClusterForecast(
            group_id=group_id,
            group_name=group.name,
            business_metric_value=business_metric_value,
            n_servers=0,
            servers=[],
            skipped_servers=[],
        )

    logger.info(
        "Cluster forecast for group %d (%s): %d servers  biz_value=%.2f",
        group_id, group.name, len(active_servers), business_metric_value,
    )

    server_results: list[ServerForecast] = []
    skipped: list[str] = []

    for server in active_servers:
        # Find the ForecastingConfig for this server
        config = db.query(ForecastingConfig).filter_by(server_id=server.id).first()

        if config is None:
            logger.warning(
                "No ForecastingConfig for server %d (%s) — run POST /groups/%d/provision first.",
                server.id, server.name, group_id,
            )
            skipped.append(f"{server.name} (no config)")
            continue

        try:
            result: ForecastResult = _single_forecast(db, config, business_metric_value)
            server_results.append(ServerForecast(
                server_id=server.id,
                server_name=server.name,
                host=server.host,
                config_id=config.id,
                predicted_cpu_percent=result.predicted_cpu_percent,
                predicted_ram_gb=result.predicted_ram_gb,
                predicted_ram_percent=result.predicted_ram_percent,
                predicted_network_mbps=result.predicted_network_mbps,
                predicted_disk_io_percent=result.predicted_disk_io_percent,
                forecast_result_id=result.id,
            ))
        except ValueError as exc:
            # No READY model for this server yet
            logger.warning(
                "Skipping server %d (%s): %s",
                server.id, server.name, exc,
            )
            skipped.append(f"{server.name} (no ready model)")

    cluster = ClusterForecast(
        group_id=group_id,
        group_name=group.name,
        business_metric_value=business_metric_value,
        n_servers=len(server_results),
        servers=server_results,
        skipped_servers=skipped,
    )

    logger.info(
        "Cluster forecast: %d/%d servers  "
        "avg_cpu=%.1f%%  total_ram=%.1f GB  avg_ram=%.1f%%  "
        "total_net=%.1f Mbps  avg_disk=%.1f%%",
        len(server_results), len(active_servers),
        cluster.cluster_cpu_avg_percent,
        cluster.cluster_ram_total_gb,
        cluster.cluster_ram_avg_percent,
        cluster.cluster_network_total_mbps,
        cluster.cluster_disk_avg_io_percent,
    )
    return cluster
