"""
Tests for Module 2 — Data Collector.

Strategy:
  - _query_prometheus()      is the real HTTP function — tested via monkeypatch
  - _query_prometheus_stub() is synthetic — tested directly (no network)
  - fetch_historical_data()  routes to real or stub — tested both paths
  - _align_series()          pure logic — tested directly
"""
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest

from app.modules.data_collector import (
    MetricsBundle,
    PrometheusQueryError,
    _align_series,
    _generate_system_stub,
    _query_prometheus,
    _query_prometheus_stub,
    fetch_historical_data,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_prom_response(n: int = 10, base: float = 50.0) -> dict:
    """Build a minimal Prometheus range-query success response."""
    now = datetime.utcnow()
    values = [
        [str((now + timedelta(minutes=i)).timestamp()), str(base + i * 0.5)]
        for i in range(n)
    ]
    return {
        "status": "success",
        "data": {"resultType": "matrix", "result": [{"metric": {}, "values": values}]},
    }

def _make_series(n: int, base: float = 50.0) -> list[dict]:
    """Build a simple timeseries list."""
    now = datetime.utcnow()
    return [
        {"timestamp": now + timedelta(minutes=i), "value": base + float(i)}
        for i in range(n)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# _query_prometheus_stub — synthetic path (no network)
# ══════════════════════════════════════════════════════════════════════════════

class TestQueryPrometheusStub:
    def test_returns_non_empty_list(self):
        end   = datetime.utcnow()
        start = end - timedelta(hours=1)
        result = _query_prometheus_stub("host", 9090, "metric", start, end)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_each_point_has_timestamp_and_value(self):
        end   = datetime.utcnow()
        start = end - timedelta(minutes=5)
        for point in _query_prometheus_stub("h", 9090, "m", start, end):
            assert "timestamp" in point and isinstance(point["timestamp"], datetime)
            assert "value" in point and isinstance(point["value"], float)

    def test_values_are_non_negative(self):
        end   = datetime.utcnow()
        start = end - timedelta(hours=2)
        for point in _query_prometheus_stub("h", 9090, "m", start, end):
            assert point["value"] >= 0.0

    def test_chronologically_ordered(self):
        end   = datetime.utcnow()
        start = end - timedelta(hours=1)
        ts = [p["timestamp"] for p in _query_prometheus_stub("h", 9090, "m", start, end)]
        assert ts == sorted(ts)

    def test_different_formulas_give_different_values(self):
        end   = datetime.utcnow()
        start = end - timedelta(hours=1)
        v1 = [p["value"] for p in _query_prometheus_stub("h", 9090, "metric_a", start, end)]
        v2 = [p["value"] for p in _query_prometheus_stub("h", 9090, "metric_b", start, end)]
        assert v1 != v2

    def test_longer_window_produces_more_points(self):
        end = datetime.utcnow()
        r1  = _query_prometheus_stub("h", 9090, "m", end - timedelta(hours=1), end)
        r24 = _query_prometheus_stub("h", 9090, "m", end - timedelta(hours=24), end)
        assert len(r24) > len(r1)


# ══════════════════════════════════════════════════════════════════════════════
# _query_prometheus — real HTTP path (monkeypatched)
# ══════════════════════════════════════════════════════════════════════════════

class TestQueryPrometheusReal:
    def _mock_get(self, response_body: dict):
        """Return a context-manager mock for httpx.Client.get()."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = response_body
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        return mock_client

    def test_parses_single_series_correctly(self):
        end   = datetime.utcnow()
        start = end - timedelta(hours=1)
        with patch("app.modules.data_collector.httpx.Client",
                   return_value=self._mock_get(_make_prom_response(10, 42.0))):
            result = _query_prometheus("h", 9090, "metric", start, end)
        assert len(result) == 10
        assert all("timestamp" in p and "value" in p for p in result)
        assert result[0]["value"] == pytest.approx(42.0, abs=0.1)

    def test_raises_on_error_status(self):
        end   = datetime.utcnow()
        start = end - timedelta(hours=1)
        body  = {"status": "error", "error": "bad query"}
        with patch("app.modules.data_collector.httpx.Client",
                   return_value=self._mock_get(body)):
            with pytest.raises(PrometheusQueryError, match="bad query"):
                _query_prometheus("h", 9090, "bad{query}", start, end)

    def test_returns_empty_list_when_no_results(self):
        end   = datetime.utcnow()
        start = end - timedelta(hours=1)
        body  = {"status": "success", "data": {"result": []}}
        with patch("app.modules.data_collector.httpx.Client",
                   return_value=self._mock_get(body)):
            result = _query_prometheus("h", 9090, "metric", start, end)
        assert result == []

    def test_averages_multiple_series(self):
        """If Prometheus returns N series, values should be averaged."""
        end   = datetime.utcnow()
        start = end - timedelta(hours=1)
        ts    = str(end.timestamp())
        body  = {
            "status": "success",
            "data": {"result": [
                {"metric": {"instance": "a"}, "values": [[ts, "100.0"]]},
                {"metric": {"instance": "b"}, "values": [[ts, "200.0"]]},
            ]},
        }
        with patch("app.modules.data_collector.httpx.Client",
                   return_value=self._mock_get(body)):
            result = _query_prometheus("h", 9090, "metric", start, end)
        assert len(result) == 1
        assert result[0]["value"] == pytest.approx(150.0, abs=0.01)

    def test_raises_on_http_error(self):
        import httpx
        end   = datetime.utcnow()
        start = end - timedelta(hours=1)
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__  = MagicMock(return_value=False)
        mock_client.get.side_effect = httpx.ConnectError("refused")
        with patch("app.modules.data_collector.httpx.Client", return_value=mock_client):
            with pytest.raises(httpx.ConnectError):
                _query_prometheus("h", 9090, "metric", start, end)


# ══════════════════════════════════════════════════════════════════════════════
# _generate_system_stub
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateSystemStub:
    def test_output_length_matches_input(self):
        business = _make_series(100)
        result   = _generate_system_stub(10.0, 5.0, business, lag_steps=3)
        assert len(result) == 100

    def test_values_are_non_negative(self):
        business = [{"timestamp": datetime.utcnow(), "value": 50.0} for _ in range(20)]
        for point in _generate_system_stub(0.0, 1.0, business):
            assert point["value"] >= 0.0

    def test_larger_base_gives_larger_values(self):
        business = [{"timestamp": datetime.utcnow(), "value": 100.0} for _ in range(50)]
        low  = [p["value"] for p in _generate_system_stub(10.0, 5.0, business)]
        high = [p["value"] for p in _generate_system_stub(100.0, 5.0, business)]
        assert sum(high) > sum(low)


# ══════════════════════════════════════════════════════════════════════════════
# _align_series
# ══════════════════════════════════════════════════════════════════════════════

class TestAlignSeries:
    def test_identical_timestamps_unchanged(self):
        s1 = _make_series(10, 10.0)
        s2 = _make_series(10, 20.0)
        # Give them exactly the same timestamps
        for i in range(10):
            s2[i]["timestamp"] = s1[i]["timestamp"]
        a1, a2 = _align_series(s1, s2)
        assert len(a1) == 10 and len(a2) == 10

    def test_drops_non_common_timestamps(self):
        now = datetime.utcnow()
        s1  = [{"timestamp": now + timedelta(minutes=i), "value": float(i)}
               for i in range(5)]
        s2  = [{"timestamp": now + timedelta(minutes=i), "value": float(i * 2)}
               for i in range(3, 8)]   # overlaps only at minutes 3, 4
        a1, a2 = _align_series(s1, s2)
        assert len(a1) == 2 and len(a2) == 2

    def test_empty_input_returns_empty(self):
        result = _align_series()
        assert result == ()

    def test_single_series_returns_unchanged(self):
        s = _make_series(5)
        (aligned,) = _align_series(s)
        assert len(aligned) == 5

    def test_result_is_sorted_ascending(self):
        now = datetime.utcnow()
        s1  = [{"timestamp": now + timedelta(minutes=i), "value": float(i)}
               for i in range(5)]
        s2  = s1[:]
        a1, _ = _align_series(s1, s2)
        timestamps = [p["timestamp"] for p in a1]
        assert timestamps == sorted(timestamps)


# ══════════════════════════════════════════════════════════════════════════════
# fetch_historical_data — stub mode (default in tests)
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchHistoricalDataStub:
    """
    In tests settings.use_prometheus_stub is True (set in conftest.py)
    so no real HTTP calls are made.
    """

    def test_returns_metrics_bundle(self):
        bundle = fetch_historical_data("h", 9090, "metric", lookback_days=1)
        assert isinstance(bundle, MetricsBundle)

    def test_all_four_series_non_empty(self):
        bundle = fetch_historical_data("h", 9090, "metric", lookback_days=1)
        assert len(bundle.business) > 0
        assert len(bundle.cpu)      > 0
        assert len(bundle.ram)      > 0
        assert len(bundle.network)  > 0

    def test_all_series_same_length(self):
        bundle = fetch_historical_data("h", 9090, "metric", lookback_days=1)
        lengths = {len(bundle.business), len(bundle.cpu),
                   len(bundle.ram),      len(bundle.network)}
        assert len(lengths) == 1

    def test_longer_lookback_more_data(self):
        b1 = fetch_historical_data("h", 9090, "m", lookback_days=1)
        b7 = fetch_historical_data("h", 9090, "m", lookback_days=7)
        assert len(b7.business) > len(b1.business)


# ══════════════════════════════════════════════════════════════════════════════
# fetch_historical_data — real path (monkeypatched)
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchHistoricalDataReal:
    """Test the real Prometheus path by patching _query_prometheus."""

    def _fake_query(self, host, port, formula, start, end, step_seconds=60):
        return _query_prometheus_stub(host, port, formula, start, end, step_seconds)

    def test_calls_prometheus_four_times(self):
        call_count = []

        def counting_query(*args, **kwargs):
            call_count.append(args[2])   # formula
            return _query_prometheus_stub(*args, **kwargs)

        with patch("app.modules.data_collector.settings") as mock_settings:
            mock_settings.use_prometheus_stub = False
            mock_settings.prometheus_cpu_query = "cpu_q"
            mock_settings.prometheus_ram_query = "ram_q"
            mock_settings.prometheus_net_query = "net_q"
            with patch("app.modules.data_collector._query_prometheus",
                       side_effect=counting_query):
                fetch_historical_data("h", 9090, "biz_formula", lookback_days=1)

        assert len(call_count) == 4
        assert "biz_formula" in call_count
        assert "cpu_q" in call_count
        assert "ram_q" in call_count
        assert "net_q" in call_count

    def test_alignment_applied_in_real_mode(self):
        """Real mode should call _align_series to harmonise timestamps."""
        with patch("app.modules.data_collector.settings") as mock_settings:
            mock_settings.use_prometheus_stub = False
            mock_settings.prometheus_cpu_query = "cpu_q"
            mock_settings.prometheus_ram_query = "ram_q"
            mock_settings.prometheus_net_query = "net_q"
            with patch("app.modules.data_collector._query_prometheus",
                       side_effect=self._fake_query):
                with patch("app.modules.data_collector._align_series",
                           wraps=_align_series) as mock_align:
                    fetch_historical_data("h", 9090, "biz", lookback_days=1)
                    mock_align.assert_called_once()
