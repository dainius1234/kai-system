"""Tests for monitor-service/app.py."""
import importlib.util
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_SVC = Path(__file__).resolve().parents[1] / "monitor-service" / "app.py"


def _load():
    common_stub = MagicMock()
    common_stub.setup_json_logger.return_value = MagicMock()
    common_stub.ErrorBudget = MagicMock(
        return_value=MagicMock(snapshot=MagicMock(return_value={}))
    )
    sys.modules.setdefault("common", MagicMock())
    sys.modules["common.runtime"] = common_stub

    spec = importlib.util.spec_from_file_location("monitor_app", _SVC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load()
client = TestClient(_mod.app)

_RULE = {
    "name": "BTC Price Alert",
    "source": {"type": "http", "url": "https://api.example.com/price", "extract": "price"},
    "condition": {"op": "gt", "value": 70000},
    "actions": ["notify"],
    "interval_seconds": 60,
    "cooldown_seconds": 300,
}


def _clean():
    _mod._rules.clear()
    _mod._last_check.clear()
    _mod._last_value.clear()
    _mod._last_fired.clear()
    _mod._check_errors.clear()
    _mod._alert_history.clear()
    _mod._fire_counts.clear()


# ── basic endpoints ──────────────────────────────────────────────────

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "rules" in data and "alerts" in data


def test_metrics():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_status_empty():
    _clean()
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["rules_total"] == 0
    assert data["rules_enabled"] == 0


# ── rule CRUD ────────────────────────────────────────────────────────

def test_add_rule():
    _clean()
    resp = client.post("/rules", json=_RULE)
    assert resp.status_code == 201
    data = resp.json()
    assert data["ok"] is True
    assert "id" in data


def test_list_rules():
    _clean()
    client.post("/rules", json=_RULE)
    resp = client.get("/rules")
    assert resp.status_code == 200
    assert resp.json()["total"] == 1
    assert resp.json()["rules"][0]["name"] == "BTC Price Alert"


def test_add_rule_with_explicit_id():
    _clean()
    rule = {**_RULE, "id": "my-rule"}
    resp = client.post("/rules", json=rule)
    assert resp.status_code == 201
    assert resp.json()["id"] == "my-rule"
    assert "my-rule" in _mod._rules


def test_add_rule_duplicate_id():
    _clean()
    rule = {**_RULE, "id": "dup"}
    client.post("/rules", json=rule)
    resp = client.post("/rules", json=rule)
    assert resp.status_code == 409


def test_update_rule():
    _clean()
    resp = client.post("/rules", json={**_RULE, "id": "upd"})
    client.put("/rules/upd", json={"interval_seconds": 120})
    assert _mod._rules["upd"]["interval_seconds"] == 120


def test_update_rule_not_found():
    resp = client.put("/rules/nope", json={"enabled": False})
    assert resp.status_code == 404


def test_delete_rule():
    _clean()
    client.post("/rules", json={**_RULE, "id": "del-me"})
    resp = client.delete("/rules/del-me")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert "del-me" not in _mod._rules


def test_delete_nonexistent_rule():
    resp = client.delete("/rules/ghost")
    assert resp.status_code == 200  # idempotent


def test_enable_disable_rule():
    _clean()
    client.post("/rules", json={**_RULE, "id": "tog", "enabled": True})
    resp = client.post("/rules/tog/disable")
    assert resp.status_code == 200
    assert _mod._rules["tog"]["enabled"] is False
    resp = client.post("/rules/tog/enable")
    assert _mod._rules["tog"]["enabled"] is True


def test_enable_missing_rule():
    resp = client.post("/rules/missing/enable")
    assert resp.status_code == 404


def test_manual_check_trigger():
    _clean()
    client.post("/rules", json={**_RULE, "id": "chk"})
    resp = client.post("/rules/chk/check")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_manual_check_missing():
    resp = client.post("/rules/ghost/check")
    assert resp.status_code == 404


# ── alert history ────────────────────────────────────────────────────

def test_alerts_empty():
    _clean()
    resp = client.get("/alerts")
    assert resp.status_code == 200
    assert resp.json()["alerts"] == []
    assert resp.json()["total"] == 0


def test_alerts_populated():
    _clean()
    _mod._alert_history.appendleft({"rule_id": "r1", "rule_name": "Test", "value": "100", "timestamp": time.time(), "message": "Test: 100"})
    resp = client.get("/alerts?limit=10")
    assert len(resp.json()["alerts"]) == 1
    assert resp.json()["alerts"][0]["rule_id"] == "r1"


def test_clear_alerts():
    _clean()
    _mod._alert_history.appendleft({"rule_id": "r1", "message": "x", "timestamp": 0, "value": "1", "rule_name": "x"})
    resp = client.delete("/alerts")
    assert resp.status_code == 200
    assert len(_mod._alert_history) == 0


# ── condition evaluation ─────────────────────────────────────────────

def test_evaluate_gt_true():
    assert _mod._evaluate({"op": "gt", "value": 100}, 150, None) is True


def test_evaluate_gt_false():
    assert _mod._evaluate({"op": "gt", "value": 100}, 50, None) is False


def test_evaluate_lt():
    assert _mod._evaluate({"op": "lt", "value": 100}, 50, None) is True


def test_evaluate_gte_equal():
    assert _mod._evaluate({"op": "gte", "value": 100}, 100, None) is True


def test_evaluate_lte():
    assert _mod._evaluate({"op": "lte", "value": 50}, 50, None) is True


def test_evaluate_eq():
    assert _mod._evaluate({"op": "eq", "value": 42}, 42, None) is True
    assert _mod._evaluate({"op": "eq", "value": 42}, 43, None) is False


def test_evaluate_ne():
    assert _mod._evaluate({"op": "ne", "value": 42}, 43, None) is True


def test_evaluate_contains():
    assert _mod._evaluate({"op": "contains", "text": "whale"}, "large whale order", None) is True
    assert _mod._evaluate({"op": "contains", "text": "whale"}, "no fish here", None) is False


def test_evaluate_not_contains():
    assert _mod._evaluate({"op": "not_contains", "text": "error"}, "all good", None) is True


def test_evaluate_changed():
    assert _mod._evaluate({"op": "changed"}, "new", "old") is True
    assert _mod._evaluate({"op": "changed"}, "same", "same") is False
    assert _mod._evaluate({"op": "changed"}, "x", None) is False  # no previous


def test_evaluate_increased_pct():
    cond = {"op": "increased_pct", "percent": 5}
    assert _mod._evaluate(cond, 105, 100) is True
    assert _mod._evaluate(cond, 104, 100) is False


def test_evaluate_decreased_pct():
    cond = {"op": "decreased_pct", "percent": 10}
    assert _mod._evaluate(cond, 90, 100) is True
    assert _mod._evaluate(cond, 95, 100) is False


# ── field extraction ─────────────────────────────────────────────────

def test_extract_field_simple():
    assert _mod._extract_field({"price": "65000"}, "price") == "65000"


def test_extract_field_nested():
    data = {"data": [{"last": 65000}]}
    assert _mod._extract_field(data, "data.0.last") == 65000


def test_extract_field_empty_path():
    data = {"x": 1}
    assert _mod._extract_field(data, "") == data


# ── status reflects state ────────────────────────────────────────────

def test_status_counts_enabled():
    _clean()
    client.post("/rules", json={**_RULE, "id": "e1", "enabled": True})
    client.post("/rules", json={**_RULE, "id": "e2", "enabled": False})
    resp = client.get("/status")
    data = resp.json()
    assert data["rules_total"] == 2
    assert data["rules_enabled"] == 1
