"""Sysmetrics service tests — psutil is stubbed so tests run without the real library."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


# ── Module loader ─────────────────────────────────────────────────────────────

def _make_psutil_stub():
    stub = types.ModuleType("psutil")

    def cpu_percent(interval=None):
        return 12.5

    def cpu_count(logical=True):
        return 8 if logical else 4

    freq = MagicMock()
    freq.current = 3200.0

    def cpu_freq():
        return freq

    vm = MagicMock()
    vm.total = 16_000_000_000
    vm.used  = 4_000_000_000
    vm.available = 12_000_000_000
    vm.percent = 25.0

    def virtual_memory():
        return vm

    part = MagicMock()
    part.mountpoint = "/"
    part.fstype = "ext4"

    def disk_partitions(all=False):
        return [part]

    usage = MagicMock()
    usage.total = 500_000_000_000
    usage.used  = 100_000_000_000
    usage.free  = 400_000_000_000
    usage.percent = 20.0

    def disk_usage(path):
        return usage

    net = MagicMock()
    net.bytes_sent = 1_000_000
    net.bytes_recv = 5_000_000
    net.packets_sent = 1000
    net.packets_recv = 5000

    def net_io_counters():
        return net

    def getloadavg():
        return (0.5, 0.4, 0.3)

    proc1 = MagicMock()
    proc1.info = {
        "pid": 1,
        "name": "python",
        "cpu_percent": 5.0,
        "memory_info": MagicMock(rss=50_000_000),
        "status": "running",
    }
    proc2 = MagicMock()
    proc2.info = {
        "pid": 2,
        "name": "idle",
        "cpu_percent": 0.0,
        "memory_info": MagicMock(rss=1_000_000),
        "status": "sleeping",
    }

    def process_iter(attrs=None):
        return [proc1, proc2]

    stub.cpu_percent = cpu_percent
    stub.cpu_count = cpu_count
    stub.cpu_freq = cpu_freq
    stub.virtual_memory = virtual_memory
    stub.disk_partitions = disk_partitions
    stub.disk_usage = disk_usage
    stub.net_io_counters = net_io_counters
    stub.getloadavg = getloadavg
    stub.process_iter = process_iter
    stub.NoSuchProcess = Exception
    stub.AccessDenied = Exception
    return stub


def _load_module(monkeypatch, with_psutil=True):
    mod_name = "sysmetrics_app"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    if with_psutil:
        sys.modules["psutil"] = _make_psutil_stub()
    elif "psutil" in sys.modules:
        del sys.modules["psutil"]

    spec = importlib.util.spec_from_file_location(
        mod_name,
        Path(__file__).parent.parent / "sysmetrics" / "app.py",
    )
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setenv("PORT", "8035")
    monkeypatch.setenv("TOP_PROCESSES", "20")
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def client(monkeypatch):
    mod = _load_module(monkeypatch, with_psutil=True)
    return TestClient(mod.app), mod


@pytest.fixture()
def stub_client(monkeypatch):
    """Client where psutil is unavailable (stub mode)."""
    mod = _load_module(monkeypatch, with_psutil=False)
    mod._PSUTIL_OK = False
    return TestClient(mod.app), mod


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health_ok(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["psutil"] is True
    assert body["uptime_seconds"] >= 0


def test_health_stub_mode(stub_client):
    c, mod = stub_client
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_metrics_returns_dict(client):
    c, _ = client
    r = c.get("/metrics")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_snapshot_cpu(client):
    c, _ = client
    r = c.get("/snapshot")
    assert r.status_code == 200
    body = r.json()
    assert "cpu" in body
    assert body["cpu"]["percent"] == 12.5
    assert body["cpu"]["count_logical"] == 8
    assert body["cpu"]["count_physical"] == 4
    assert body["cpu"]["freq_mhz"] == 3200.0


def test_snapshot_memory(client):
    c, _ = client
    r = c.get("/snapshot")
    body = r.json()
    assert "memory" in body
    mem = body["memory"]
    assert mem["percent"] == 25.0
    assert mem["total_mb"] > 0
    assert mem["used_mb"] > 0
    assert mem["available_mb"] > 0


def test_snapshot_disk(client):
    c, _ = client
    r = c.get("/snapshot")
    body = r.json()
    assert "disk" in body
    assert len(body["disk"]) >= 1
    disk = body["disk"][0]
    assert disk["mountpoint"] == "/"
    assert disk["percent"] == 20.0


def test_snapshot_network(client):
    c, _ = client
    r = c.get("/snapshot")
    body = r.json()
    assert "network" in body
    net = body["network"]
    assert net["bytes_sent_mb"] > 0
    assert net["bytes_recv_mb"] > 0
    assert net["packets_sent"] == 1000


def test_snapshot_load_avg(client):
    c, _ = client
    r = c.get("/snapshot")
    body = r.json()
    assert "load_avg_1_5_15" in body
    assert body["load_avg_1_5_15"] == [0.5, 0.4, 0.3]


def test_snapshot_uptime(client):
    c, _ = client
    r = c.get("/snapshot")
    body = r.json()
    assert "uptime_seconds" in body
    assert body["uptime_seconds"] >= 0


def test_snapshot_stub_mode(stub_client):
    c, _ = stub_client
    r = c.get("/snapshot")
    assert r.status_code == 200
    body = r.json()
    assert body.get("stub") is True


def test_processes_returns_list(client):
    c, _ = client
    r = c.get("/processes")
    assert r.status_code == 200
    body = r.json()
    assert "processes" in body
    assert isinstance(body["processes"], list)


def test_processes_sorted_by_cpu(client):
    c, _ = client
    r = c.get("/processes")
    procs = r.json()["processes"]
    assert len(procs) >= 1
    assert procs[0]["cpu_percent"] >= (procs[-1]["cpu_percent"] if len(procs) > 1 else 0)


def test_processes_fields(client):
    c, _ = client
    r = c.get("/processes")
    procs = r.json()["processes"]
    for p in procs:
        assert "pid" in p
        assert "name" in p
        assert "cpu_percent" in p
        assert "mem_mb" in p
        assert "status" in p


def test_processes_stub_mode(stub_client):
    c, _ = stub_client
    r = c.get("/processes")
    assert r.status_code == 200
    body = r.json()
    assert body["processes"] == []
