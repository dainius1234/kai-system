"""Tests for D124: Service Watchdog — agentic/service_watchdog.py."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))

from service_watchdog import (
    ServiceWatchdog,
    ServiceProfile,
    CheckResult,
    get_watchdog,
    reset_watchdog,
    _FAILURE_THRESHOLD,
    _FSM_EVENT_SERVICE_DOWN,
    _FSM_EVENT_SERVICE_RESTORED,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _watchdog(tmp_path: Path | None = None) -> ServiceWatchdog:
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp()) / "watchdog"
    return ServiceWatchdog(data_dir=tmp_path)


def _mock_httpx_ok(status: int = 200, latency: float = 0.0):
    resp = MagicMock()
    resp.status_code = status
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get = MagicMock(return_value=resp)
    return client


def _services(*names_critical: tuple) -> list:
    return [
        {"name": n, "url": f"http://svc-{n}:9000", "critical": c}
        for n, c in names_critical
    ]


# ── ServiceProfile ────────────────────────────────────────────────────────────

def test_health_url_appends_path():
    p = ServiceProfile(name="svc", url="http://example.com:8080")
    assert p.health_url == "http://example.com:8080/health"


def test_health_url_no_double_slash():
    p = ServiceProfile(name="svc", url="http://example.com/")
    assert not p.health_url.endswith("//health")


# ── ping() ────────────────────────────────────────────────────────────────────

def test_ping_healthy_on_2xx(tmp_path):
    w = _watchdog(tmp_path)
    client = _mock_httpx_ok(200)
    with patch("service_watchdog.httpx.Client", return_value=client):
        result = w.ping("svc", "http://example.com")
    assert result.healthy is True
    assert result.status_code == 200
    assert result.error is None


def test_ping_healthy_on_301(tmp_path):
    w = _watchdog(tmp_path)
    client = _mock_httpx_ok(301)
    with patch("service_watchdog.httpx.Client", return_value=client):
        result = w.ping("svc", "http://example.com")
    assert result.healthy is True


def test_ping_unhealthy_on_500(tmp_path):
    w = _watchdog(tmp_path)
    client = _mock_httpx_ok(500)
    with patch("service_watchdog.httpx.Client", return_value=client):
        result = w.ping("svc", "http://example.com")
    assert result.healthy is False
    assert result.status_code == 500


def test_ping_unhealthy_on_network_error(tmp_path):
    w = _watchdog(tmp_path)
    with patch("service_watchdog.httpx.Client", side_effect=Exception("connection refused")):
        result = w.ping("svc", "http://example.com")
    assert result.healthy is False
    assert result.status_code == 0
    assert result.error is not None


def test_ping_records_latency(tmp_path):
    w = _watchdog(tmp_path)
    client = _mock_httpx_ok(200)
    with patch("service_watchdog.httpx.Client", return_value=client):
        result = w.ping("svc", "http://example.com")
    assert result.latency_ms >= 0.0


def test_ping_propagates_critical_flag(tmp_path):
    w = _watchdog(tmp_path)
    client = _mock_httpx_ok(200)
    with patch("service_watchdog.httpx.Client", return_value=client):
        result = w.ping("broker", "http://broker:8080", critical=True)
    assert result.critical is True


# ── check_all() ───────────────────────────────────────────────────────────────

def test_check_all_returns_results_per_service(tmp_path):
    w = _watchdog(tmp_path)
    client = _mock_httpx_ok(200)
    svcs = _services(("svc1", False), ("svc2", False))
    with patch("service_watchdog.httpx.Client", return_value=client):
        results, _ = w.check_all(services=svcs)
    assert len(results) == 2
    names = {r.name for r in results}
    assert "svc1" in names and "svc2" in names


def test_check_all_no_fsm_events_when_all_healthy(tmp_path):
    w = _watchdog(tmp_path)
    client = _mock_httpx_ok(200)
    svcs = _services(("broker", True), ("calendar", False))
    with patch("service_watchdog.httpx.Client", return_value=client):
        _, events = w.check_all(services=svcs)
    assert _FSM_EVENT_SERVICE_DOWN not in events


def test_check_all_no_fsm_event_below_threshold(tmp_path):
    """Single failure below threshold should not emit service_down."""
    w = _watchdog(tmp_path)
    svcs = _services(("broker", True))
    with patch("service_watchdog.httpx.Client", side_effect=Exception("down")):
        _, events = w.check_all(services=svcs)
    # Only 1 failure — below _FAILURE_THRESHOLD (2), no event yet
    assert _FSM_EVENT_SERVICE_DOWN not in events


def test_check_all_fires_service_down_after_threshold(tmp_path):
    w = _watchdog(tmp_path)
    svcs = _services(("broker", True))
    # First failure — below threshold
    with patch("service_watchdog.httpx.Client", side_effect=Exception("down")):
        w.check_all(services=svcs)
    # Second failure — at threshold, event fires
    with patch("service_watchdog.httpx.Client", side_effect=Exception("down")):
        _, events = w.check_all(services=svcs)
    assert _FSM_EVENT_SERVICE_DOWN in events


def test_check_all_no_service_down_for_non_critical(tmp_path):
    w = _watchdog(tmp_path)
    svcs = _services(("weather", False))
    # Two consecutive failures on non-critical service
    for _ in range(_FAILURE_THRESHOLD):
        with patch("service_watchdog.httpx.Client", side_effect=Exception("down")):
            w.check_all(services=svcs)
    _, events = w.check_all(services=[{"name": "weather", "url": "http://w:8080", "critical": False}])
    assert _FSM_EVENT_SERVICE_DOWN not in events


def test_check_all_fires_service_restored_after_recovery(tmp_path):
    w = _watchdog(tmp_path)
    svcs = _services(("broker", True))
    # Fail twice to set was_down
    for _ in range(_FAILURE_THRESHOLD):
        with patch("service_watchdog.httpx.Client", side_effect=Exception("down")):
            w.check_all(services=svcs)
    # Now recover
    client = _mock_httpx_ok(200)
    with patch("service_watchdog.httpx.Client", return_value=client):
        _, events = w.check_all(services=svcs)
    assert _FSM_EVENT_SERVICE_RESTORED in events


def test_check_all_persists_to_disk(tmp_path):
    w = _watchdog(tmp_path)
    client = _mock_httpx_ok(200)
    svcs = _services(("svc1", False))
    with patch("service_watchdog.httpx.Client", return_value=client):
        w.check_all(services=svcs)
    f = tmp_path / "status.json"
    assert f.exists()
    data = json.loads(f.read_text())
    assert "services" in data
    assert data["last_checked_at"] > 0


def test_check_all_empty_services_returns_empty(tmp_path):
    w = _watchdog(tmp_path)
    results, events = w.check_all(services=[])
    assert results == []
    assert events == []


def test_check_all_updates_last_checked_at(tmp_path):
    w = _watchdog(tmp_path)
    before = w._last_checked_at
    client = _mock_httpx_ok(200)
    svcs = _services(("svc", False))
    with patch("service_watchdog.httpx.Client", return_value=client):
        w.check_all(services=svcs)
    assert w._last_checked_at > before


# ── status() ─────────────────────────────────────────────────────────────────

def test_status_structure(tmp_path):
    w = _watchdog(tmp_path)
    s = w.status()
    assert "total" in s
    assert "healthy_count" in s
    assert "unhealthy_count" in s
    assert "critical_down" in s
    assert "services" in s


def test_status_reflects_check_results(tmp_path):
    w = _watchdog(tmp_path)
    client = _mock_httpx_ok(200)
    svcs = _services(("svc1", False), ("svc2", False))
    with patch("service_watchdog.httpx.Client", return_value=client):
        w.check_all(services=svcs)
    s = w.status()
    assert s["total"] == 2
    assert s["healthy_count"] == 2
    assert s["unhealthy_count"] == 0


def test_status_lists_critical_down(tmp_path):
    w = _watchdog(tmp_path)
    svcs = _services(("broker", True))
    for _ in range(_FAILURE_THRESHOLD):
        with patch("service_watchdog.httpx.Client", side_effect=Exception("down")):
            w.check_all(services=svcs)
    s = w.status()
    assert "broker" in s["critical_down"]


def test_status_seconds_since_check_is_none_before_first_check(tmp_path):
    w = _watchdog(tmp_path)
    s = w.status()
    assert s["seconds_since_check"] is None


# ── CheckResult ───────────────────────────────────────────────────────────────

def test_check_result_to_dict():
    r = CheckResult(name="svc", url="http://x", healthy=True, status_code=200,
                    latency_ms=10.0, consecutive_failures=0, critical=False)
    d = r.to_dict()
    assert d["name"] == "svc"
    assert d["healthy"] is True


# ── Singleton ─────────────────────────────────────────────────────────────────

def test_singleton_returns_same_instance(tmp_path):
    reset_watchdog()
    w1 = get_watchdog(tmp_path)
    w2 = get_watchdog(tmp_path)
    assert w1 is w2
    reset_watchdog()


def test_reset_clears_singleton(tmp_path):
    reset_watchdog()
    w1 = get_watchdog(tmp_path)
    reset_watchdog()
    w2 = get_watchdog(tmp_path)
    assert w1 is not w2
    reset_watchdog()
