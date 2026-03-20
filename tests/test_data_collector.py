"""Unit tests for Module 2 — Data Collector."""
from datetime import datetime, timedelta

import pytest

from app.modules.data_collector import (
    MetricsBundle,
    _generate_system_stub,
    _query_prometheus,
    fetch_historical_data,
)


class TestQueryPrometheus:
    def test_returns_list_of_dicts(self):
        end = datetime.utcnow()
        start = end - timedelta(hours=1)
        result = _query_prometheus("host", 9090, "my_metric", start, end)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_each_point_has_timestamp_and_value(self):
        end = datetime.utcnow()
        start = end - timedelta(minutes=5)
        result = _query_prometheus("host", 9090, "metric", start, end)
        for point in result:
            assert "timestamp" in point
            assert "value" in point
            assert isinstance(point["timestamp"], datetime)
            assert isinstance(point["value"], float)

    def test_values_are_non_negative(self):
        end = datetime.utcnow()
        start = end - timedelta(hours=2)
        result = _query_prometheus("host", 9090, "metric", start, end)
        for point in result:
            assert point["value"] >= 0

    def test_points_are_chronologically_ordered(self):
        end = datetime.utcnow()
        start = end - timedelta(hours=1)
        result = _query_prometheus("host", 9090, "metric", start, end)
        timestamps = [p["timestamp"] for p in result]
        assert timestamps == sorted(timestamps)

    def test_different_formulas_produce_different_data(self):
        end = datetime.utcnow()
        start = end - timedelta(hours=1)
        r1 = _query_prometheus("host", 9090, "metric_a", start, end)
        r2 = _query_prometheus("host", 9090, "metric_b", start, end)
        values1 = [p["value"] for p in r1]
        values2 = [p["value"] for p in r2]
        # Hash-based noise differs between formulas
        assert values1 != values2


class TestGenerateSystemStub:
    def test_output_length_matches_input(self):
        business = [
            {"timestamp": datetime.utcnow(), "value": float(i)} for i in range(100)
        ]
        result = _generate_system_stub(10.0, 5.0, business, lag_minutes=3)
        assert len(result) == len(business)

    def test_values_are_non_negative(self):
        business = [
            {"timestamp": datetime.utcnow(), "value": 50.0} for _ in range(20)
        ]
        result = _generate_system_stub(0.0, 1.0, business)
        for point in result:
            assert point["value"] >= 0


class TestFetchHistoricalData:
    def test_returns_metrics_bundle(self):
        bundle = fetch_historical_data("host", 9090, "metric", lookback_days=1)
        assert isinstance(bundle, MetricsBundle)

    def test_all_series_have_same_length(self):
        bundle = fetch_historical_data("host", 9090, "metric", lookback_days=1)
        lengths = {
            len(bundle.business),
            len(bundle.cpu),
            len(bundle.ram),
            len(bundle.network),
        }
        assert len(lengths) == 1  # all equal

    def test_bundle_has_data(self):
        bundle = fetch_historical_data("host", 9090, "metric", lookback_days=1)
        assert len(bundle.business) > 0

    def test_longer_lookback_produces_more_data(self):
        b1 = fetch_historical_data("host", 9090, "m", lookback_days=1)
        b7 = fetch_historical_data("host", 9090, "m", lookback_days=7)
        assert len(b7.business) > len(b1.business)
