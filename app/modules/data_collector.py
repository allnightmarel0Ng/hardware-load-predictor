"""
Module 2 — Historical Data Collector
Fetches time-series data from Prometheus and returns it as a MetricsBundle
ready for correlation analysis and model training.

Real vs stub:
  By default the module makes real HTTP calls to the Prometheus range API.
  Set USE_PROMETHEUS_STUB=true in .env (or settings.use_prometheus_stub=True)
  to fall back to synthetic sinusoidal data — useful for local dev without
  a live Prometheus, and for the test suite (which monkeypatches
  _query_prometheus directly).

Prometheus queries used:
  - Business metric : the PromQL formula from ForecastingConfig
  - CPU %           : settings.prometheus_cpu_query   (node_exporter default)
  - RAM GB          : settings.prometheus_ram_query   (node_exporter default)
  - Network Mbps    : settings.prometheus_net_query   (node_exporter default)

All four are queried against the same host:port (the server's Prometheus).
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Any

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

Timeseries = list[dict[str, Any]]


# ── Public data type ──────────────────────────────────────────────────────────

class MetricsBundle:
    """Container for aligned business + system time-series (five targets)."""

    def __init__(
        self,
        business:    Timeseries,
        cpu:         Timeseries,
        ram_gb:      Timeseries,
        ram_percent: Timeseries,
        network:     Timeseries,
        disk:        Timeseries,
    ) -> None:
        self.business    = business
        self.cpu         = cpu
        self.ram_gb      = ram_gb
        self.ram_percent = ram_percent
        self.network     = network
        self.disk        = disk

    def __repr__(self) -> str:
        return (
            f"<MetricsBundle business={len(self.business)} pts "
            f"cpu={len(self.cpu)} pts>"
        )


# ── Prometheus range query ────────────────────────────────────────────────────

class PrometheusQueryError(RuntimeError):
    """Raised when Prometheus returns an unexpected response."""


def _query_prometheus(
    host: str,
    port: int,
    formula: str,
    start: datetime,
    end: datetime,
    step_seconds: int = 60,
) -> Timeseries:
    """
    Query the Prometheus range API for a PromQL expression.

    GET /api/v1/query_range
        query = <formula>
        start = <unix timestamp>
        end   = <unix timestamp>
        step  = <resolution in seconds>

    Returns a list of {"timestamp": datetime, "value": float} dicts,
    sorted ascending by timestamp.

    If the formula returns multiple series (e.g. per-instance metrics),
    values are averaged across all series at each timestamp.

    Raises:
        PrometheusQueryError  if the response is malformed or status != "success"
        httpx.HTTPError       if the HTTP request itself fails
    """
    url = f"http://{host}:{port}/api/v1/query_range"
    params = {
        "query": formula,
        "start": start.timestamp(),
        "end":   end.timestamp(),
        "step":  step_seconds,
    }

    logger.debug(
        "Prometheus query: host=%s:%d formula=%r step=%ds",
        host, port, formula, step_seconds,
    )

    with httpx.Client(timeout=30.0) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()

    body = resp.json()

    if body.get("status") != "success":
        raise PrometheusQueryError(
            f"Prometheus returned status={body.get('status')!r} "
            f"for formula={formula!r}: {body.get('error', 'no error message')}"
        )

    result_list = body.get("data", {}).get("result", [])
    if not result_list:
        logger.warning(
            "Prometheus returned no time-series for formula=%r (%s to %s)",
            formula, start.isoformat(), end.isoformat(),
        )
        return []

    if len(result_list) == 1:
        raw_values = result_list[0]["values"]
    else:
        # Multiple series — average across all at each timestamp
        logger.debug(
            "Formula %r returned %d series — averaging", formula, len(result_list)
        )
        merged: dict[float, list[float]] = {}
        for series in result_list:
            for ts_raw, val_raw in series["values"]:
                ts = float(ts_raw)
                merged.setdefault(ts, []).append(float(val_raw))
        raw_values = [
            [ts, sum(vals) / len(vals)]
            for ts, vals in sorted(merged.items())
        ]

    timeseries: Timeseries = [
        {
            "timestamp": datetime.utcfromtimestamp(float(ts_raw)),
            "value":     float(val_raw),
        }
        for ts_raw, val_raw in raw_values
    ]

    logger.debug(
        "Prometheus: %d points returned for formula=%r", len(timeseries), formula
    )
    return timeseries


# ── Stub fallback (synthetic data) ───────────────────────────────────────────

def _query_prometheus_stub(
    host: str,
    port: int,
    formula: str,
    start: datetime,
    end: datetime,
    step_seconds: int = 60,
) -> Timeseries:
    """
    Synthetic stub — daily sinusoidal pattern + deterministic noise.
    Used when USE_PROMETHEUS_STUB=true or when monkeypatched in tests.
    """
    logger.debug("Using Prometheus STUB for formula=%r", formula)
    total_steps = int((end - start).total_seconds() / step_seconds)
    result: Timeseries = []
    for i in range(total_steps):
        ts    = start + timedelta(seconds=i * step_seconds)
        base  = 100 + 50 * math.sin(2 * math.pi * i / (24 * 3600 / step_seconds))
        noise = (hash((formula, i)) % 20) - 10
        result.append({"timestamp": ts, "value": max(0.0, base + noise)})
    return result


def _generate_system_stub(
    base_value: float,
    amplitude:  float,
    business_series: Timeseries,
    lag_steps: int = 5,
) -> Timeseries:
    """
    Synthetic system metric correlated with business_series.
    Only used in full stub mode.
    """
    result: Timeseries = []
    for i, point in enumerate(business_series):
        src_idx = max(0, i - lag_steps)
        biz_val = business_series[src_idx]["value"]
        noise   = (hash((base_value, i)) % 10) - 5
        value   = base_value + amplitude * (biz_val / 150) + noise
        result.append({"timestamp": point["timestamp"], "value": max(0.0, value)})
    return result


# ── Series alignment ──────────────────────────────────────────────────────────

def _align_series(*series_list: Timeseries) -> tuple[Timeseries, ...]:
    """
    Align multiple time-series to their common timestamps.

    Prometheus range queries for different metrics should return identical
    timestamps, but scrape gaps and recording rule delays can cause slight
    mismatches.  Keeps only timestamps present in ALL series.
    """
    if not series_list:
        return ()

    ts_sets = [{p["timestamp"] for p in s} for s in series_list]
    common  = ts_sets[0]
    for ts_set in ts_sets[1:]:
        common &= ts_set

    if not common:
        logger.warning(
            "No common timestamps across %d series — returning unaligned data",
            len(series_list),
        )
        return series_list  # type: ignore[return-value]

    drop = sum(len(s) for s in series_list) - len(common) * len(series_list)
    if drop > 0:
        logger.debug(
            "Alignment: dropped %d mismatched points across %d series",
            drop, len(series_list),
        )

    return tuple(
        sorted([p for p in s if p["timestamp"] in common], key=lambda p: p["timestamp"])
        for s in series_list
    )


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_historical_data(
    host: str,
    port: int,
    business_formula: str,
    lookback_days: int = 30,
    step_seconds: int = 60,
) -> MetricsBundle:
    """
    Fetch aligned historical time-series for correlation analysis and training.

    In production (USE_PROMETHEUS_STUB=false, the default) makes four real
    Prometheus range queries — one for the business metric and three for
    system metrics (CPU, RAM, network) using the PromQL expressions from
    settings.  All queries target the same host:port.

    In stub mode (USE_PROMETHEUS_STUB=true) all four series are synthetic.

    Args:
        host:             Prometheus host (from ForecastingConfig).
        port:             Prometheus port.
        business_formula: PromQL expression for the business metric.
        lookback_days:    History window to fetch.
        step_seconds:     Query resolution (default 60 s = 1-min data points).

    Returns:
        MetricsBundle with four aligned Timeseries.
    """
    end   = datetime.utcnow()
    start = end - timedelta(days=lookback_days)

    logger.info(
        "Fetching data: host=%s:%d  lookback=%dd  stub=%s",
        host, port, lookback_days, settings.use_prometheus_stub,
    )

    if settings.use_prometheus_stub:
        # Full synthetic path — business + correlated system metrics
        business    = _query_prometheus_stub(host, port, business_formula, start, end, step_seconds)
        cpu         = _generate_system_stub(30.0,  40.0,  business)
        ram_gb      = _generate_system_stub(8.0,   8.0,   business)
        ram_percent = _generate_system_stub(55.0,  20.0,  business)
        network     = _generate_system_stub(50.0,  100.0, business)
        disk        = _generate_system_stub(15.0,  30.0,  business)

    else:
        # Real path — six separate Prometheus queries
        business    = _fetch_series(host, port, business_formula,                  start, end, step_seconds)
        cpu         = _fetch_series(host, port, settings.prometheus_cpu_query,     start, end, step_seconds)
        ram_gb      = _fetch_series(host, port, settings.prometheus_ram_gb_query,  start, end, step_seconds)
        ram_percent = _fetch_series(host, port, settings.prometheus_ram_pct_query, start, end, step_seconds)
        network     = _fetch_series(host, port, settings.prometheus_net_query,     start, end, step_seconds)
        disk        = _fetch_series(host, port, settings.prometheus_disk_query,    start, end, step_seconds)

        business, cpu, ram_gb, ram_percent, network, disk = _align_series(
            business, cpu, ram_gb, ram_percent, network, disk
        )

    bundle = MetricsBundle(
        business=business, cpu=cpu,
        ram_gb=ram_gb, ram_percent=ram_percent,
        network=network, disk=disk,
    )
    logger.info("Collected %s", bundle)
    return bundle
