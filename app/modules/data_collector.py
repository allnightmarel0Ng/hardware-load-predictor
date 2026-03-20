"""
Module 2 — Historical Data Collector
Fetches time-series data from the configured metrics source (Prometheus).

The public interface is `fetch_historical_data()`.  The actual HTTP call is
isolated in `_query_prometheus()` so it can be monkeypatched in tests without
touching the network.

NOTE: In production wire `_query_prometheus` to your real Prometheus instance.
      The stub returns synthetic sinusoidal data so the rest of the pipeline
      works end-to-end without an external service.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

type Timeseries = list[dict]
# Each point: {"timestamp": datetime, "value": float}


class MetricsBundle:
    """Container for aligned business + system time-series."""

    def __init__(
        self,
        business: Timeseries,
        cpu: Timeseries,
        ram: Timeseries,
        network: Timeseries,
    ) -> None:
        self.business = business
        self.cpu = cpu
        self.ram = ram
        self.network = network

    def __repr__(self) -> str:
        return (
            f"<MetricsBundle business={len(self.business)} pts "
            f"cpu={len(self.cpu)} pts>"
        )


# ---------------------------------------------------------------------------
# Internal Prometheus query (stubbed)
# ---------------------------------------------------------------------------

def _query_prometheus(
    host: str,
    port: int,
    formula: str,
    start: datetime,
    end: datetime,
    step_seconds: int = 60,
) -> Timeseries:
    """
    Query Prometheus range API for a given PromQL formula.

    Real implementation:
        url = f"http://{host}:{port}/api/v1/query_range"
        params = {"query": formula, "start": start.timestamp(),
                  "end": end.timestamp(), "step": step_seconds}
        resp = httpx.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()["data"]["result"][0]["values"]
        return [{"timestamp": datetime.utcfromtimestamp(float(t)), "value": float(v)}
                for t, v in raw]

    Stub: returns synthetic data so the pipeline works without Prometheus.
    """
    logger.debug("Querying Prometheus stub for formula=%r (%s → %s)", formula, start, end)

    total_minutes = int((end - start).total_seconds() / step_seconds)
    result: Timeseries = []
    for i in range(total_minutes):
        ts = start + timedelta(seconds=i * step_seconds)
        # Sine wave + small noise to mimic realistic load patterns
        base = 100 + 50 * math.sin(2 * math.pi * i / (24 * 60))  # daily cycle
        noise = (hash((formula, i)) % 20) - 10
        result.append({"timestamp": ts, "value": max(0.0, base + noise)})

    return result


def _generate_system_stub(
    base_value: float,
    amplitude: float,
    business_series: Timeseries,
    lag_minutes: int = 5,
) -> Timeseries:
    """
    Generate a synthetic system metric that correlates with the business
    series with a fixed lag — used only by the stub path.
    """
    result: Timeseries = []
    n = len(business_series)
    for i, point in enumerate(business_series):
        src_idx = max(0, i - lag_minutes)
        biz_val = business_series[src_idx]["value"]
        # linear mapping + noise
        noise = (hash((base_value, i)) % 10) - 5
        value = base_value + amplitude * (biz_val / 150) + noise
        result.append({"timestamp": point["timestamp"], "value": max(0.0, value)})
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_historical_data(
    host: str,
    port: int,
    business_formula: str,
    lookback_days: int = 30,
) -> MetricsBundle:
    """
    Collect aligned historical data for a given config.

    Returns a MetricsBundle with four aligned time-series:
      - business metric
      - CPU load (%)
      - RAM usage (GB)
      - Network traffic (Mbps)
    """
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)

    logger.info(
        "Fetching historical data: host=%s port=%d lookback=%d days",
        host, port, lookback_days,
    )

    business = _query_prometheus(host, port, business_formula, start, end)

    # System metrics — in production these would be separate PromQL queries,
    # e.g. node_cpu_seconds_total, node_memory_MemAvailable_bytes, etc.
    cpu = _generate_system_stub(base_value=30.0, amplitude=40.0, business_series=business)
    ram = _generate_system_stub(base_value=8.0,  amplitude=8.0,  business_series=business)
    net = _generate_system_stub(base_value=50.0, amplitude=100.0, business_series=business)

    bundle = MetricsBundle(business=business, cpu=cpu, ram=ram, network=net)
    logger.info("Fetched %s", bundle)
    return bundle
