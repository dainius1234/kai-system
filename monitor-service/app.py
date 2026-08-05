"""Monitor service — background rule-based alerting.

Rules define a data source (HTTP API or browser scrape), a condition to check,
and actions to fire (desktop notify, TTS) when the condition is met.
Each rule runs on its own interval with a cooldown to avoid alert storms.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager, suppress
from typing import Any, Dict, List, Optional

import httpx
from fastapi import Body, Depends, FastAPI, HTTPException

import sys as _sys, os as _os
_repo = _os.path.dirname(_os.path.abspath(__file__))
while _repo != _os.path.dirname(_repo) and not _os.path.isdir(_os.path.join(_repo, 'common')):
    _repo = _os.path.dirname(_repo)
if _repo not in _sys.path:
    _sys.path.insert(0, _repo)
from common.http_hygiene import pooled_client
from common.service_auth import require_service_auth
from pydantic import BaseModel, Field

try:
    from common.runtime import ErrorBudget, setup_json_logger
    logger = setup_json_logger("monitor", os.getenv("LOG_PATH", "/tmp/monitor.json.log"))
    _budget: Any = ErrorBudget(window_seconds=300)
except Exception:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("monitor")
    _budget = None

BROWSER_AGENT_URL = os.getenv("BROWSER_AGENT_URL", "http://browser-agent:8040")
NOTIFY_URL = os.getenv("NOTIFY_SERVICE_URL", "http://notify-service:8031")
TTS_URL = os.getenv("TTS_SERVICE_URL", "http://tts-service:8030")
RULES_FILE = os.getenv("RULES_FILE", "")
MAX_ALERTS = int(os.getenv("MONITOR_MAX_ALERTS", "200"))

# ── runtime state ────────────────────────────────────────────────────
_rules: Dict[str, dict] = {}
_last_check: Dict[str, float] = {}
_last_value: Dict[str, Any] = {}
_last_fired: Dict[str, float] = {}
_check_errors: Dict[str, str] = {}
_alert_history: deque = deque(maxlen=MAX_ALERTS)
_fire_counts: Dict[str, int] = {}


# ── Pydantic models ──────────────────────────────────────────────────

class RuleSource(BaseModel):
    type: str = "http"          # "http" | "scrape"
    url: str
    extract: str = ""           # dot-path into JSON: "price" or "data.0.last"
    selector: str = "body"      # CSS selector for scrape source type


class RuleCondition(BaseModel):
    op: str = "gt"              # gt lt gte lte eq ne contains not_contains changed increased_pct decreased_pct
    value: Optional[float] = None
    text: Optional[str] = None  # for contains / not_contains
    percent: Optional[float] = None  # for increased_pct / decreased_pct


class RuleIn(BaseModel):
    id: Optional[str] = None
    name: str
    source: RuleSource
    condition: RuleCondition
    actions: List[str] = ["notify"]   # "notify" | "tts"
    interval_seconds: int = Field(60, ge=5)
    cooldown_seconds: int = Field(300, ge=0)
    message: str = "{name}: {value}"
    urgency: str = "normal"     # low | normal | critical
    enabled: bool = True


# ── helpers ──────────────────────────────────────────────────────────

def _extract_field(data: Any, path: str) -> Any:
    """Traverse a JSON object via dot-separated path: 'data.0.price'."""
    if not path:
        return data
    for key in path.split("."):
        if isinstance(data, dict):
            data = data[key]
        elif isinstance(data, list):
            data = data[int(key)]
        else:
            break
    return data


def _evaluate(condition: dict, current: Any, previous: Any) -> bool:
    op = condition.get("op", "gt")
    threshold = condition.get("value")
    text_match = condition.get("text", "")
    percent = condition.get("percent")

    # Numeric ops
    try:
        fval = float(current)
        if op == "gt" and threshold is not None: return fval > float(threshold)
        if op == "lt" and threshold is not None: return fval < float(threshold)
        if op == "gte" and threshold is not None: return fval >= float(threshold)
        if op == "lte" and threshold is not None: return fval <= float(threshold)
        if op == "eq" and threshold is not None: return fval == float(threshold)
        if op == "ne" and threshold is not None: return fval != float(threshold)
        if op == "increased_pct" and previous is not None:
            fprev = float(previous)
            return fprev != 0 and ((fval - fprev) / abs(fprev) * 100) >= float(percent or 0)
        if op == "decreased_pct" and previous is not None:
            fprev = float(previous)
            return fprev != 0 and ((fprev - fval) / abs(fprev) * 100) >= float(percent or 0)
    except (TypeError, ValueError):
        pass

    # String / existence ops
    sval = str(current).lower()
    smatch = str(text_match).lower()
    if op == "contains": return smatch in sval
    if op == "not_contains": return smatch not in sval
    if op == "changed": return previous is not None and str(current) != str(previous)
    return False


def _format_message(rule: dict, value: Any) -> str:
    try:
        return rule.get("message", "{name}: {value}").format(
            name=rule["name"], value=value, rule_id=rule["id"],
        )
    except Exception:
        return f"{rule['name']}: {value}"


def _save_rules() -> None:
    if not RULES_FILE:
        return
    try:
        with open(RULES_FILE, "w") as f:
            json.dump(list(_rules.values()), f, indent=2)
    except Exception as exc:
        logger.error("Failed to save rules to %s: %s", RULES_FILE, exc)


# ── core async tasks ─────────────────────────────────────────────────

async def _fetch_value(source: dict) -> Any:
    stype = source.get("type", "http")
    if stype == "http":
        async with pooled_client(timeout=10.0) as client:
            resp = await client.get(source["url"])
            resp.raise_for_status()
            data = resp.json()
            return _extract_field(data, source.get("extract", ""))
    elif stype == "scrape":
        async with pooled_client(timeout=30.0) as client:
            resp = await client.post(
                f"{BROWSER_AGENT_URL}/scrape",
                json={"url": source["url"], "selector": source.get("selector", "body")},
            )
            resp.raise_for_status()
            return resp.json().get("text", "")
    else:
        raise ValueError(f"Unknown source type: {stype}")


async def _fire_actions(rule: dict, value: Any) -> None:
    message = _format_message(rule, value)
    actions = rule.get("actions", ["notify"])
    async with pooled_client(timeout=10.0) as client:
        if "notify" in actions:
            with suppress(Exception):
                await client.post(f"{NOTIFY_URL}/notify", json={
                    "title": rule["name"],
                    "body": message,
                    "urgency": rule.get("urgency", "normal"),
                })
        if "tts" in actions:
            with suppress(Exception):
                await client.post(f"{TTS_URL}/synthesize", json={"text": message})
    logger.info("Rule %s fired: %s", rule["id"], message)


async def _check_rule(rule: dict) -> None:
    rule_id = rule["id"]
    try:
        value = await _fetch_value(rule["source"])
        previous = _last_value.get(rule_id)
        _last_value[rule_id] = value
        _check_errors.pop(rule_id, None)

        if _evaluate(rule["condition"], value, previous):
            now = time.time()
            cooldown = rule.get("cooldown_seconds", 300)
            if now - _last_fired.get(rule_id, 0) >= cooldown:
                _last_fired[rule_id] = now
                _fire_counts[rule_id] = _fire_counts.get(rule_id, 0) + 1
                await _fire_actions(rule, value)
                _alert_history.appendleft({
                    "rule_id": rule_id,
                    "rule_name": rule["name"],
                    "value": str(value),
                    "timestamp": now,
                    "message": _format_message(rule, value),
                })
    except Exception as exc:
        _check_errors[rule_id] = str(exc)
        logger.error("Rule %s check failed: %s", rule_id, exc)


async def _watch_loop() -> None:
    """Main background loop — ticks every second, fires rule checks when due."""
    while True:
        now = time.time()
        for rule in list(_rules.values()):
            if not rule.get("enabled", True):
                continue
            interval = rule.get("interval_seconds", 60)
            if now - _last_check.get(rule["id"], 0) >= interval:
                _last_check[rule["id"]] = now
                asyncio.create_task(_check_rule(rule))
        await asyncio.sleep(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load persisted rules on startup
    if RULES_FILE and os.path.exists(RULES_FILE):
        try:
            with open(RULES_FILE) as f:
                for rule in json.load(f):
                    _rules[rule["id"]] = rule
            logger.info("Loaded %d rules from %s", len(_rules), RULES_FILE)
        except Exception as exc:
            logger.error("Failed to load rules from %s: %s", RULES_FILE, exc)
    task = asyncio.create_task(_watch_loop())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="Monitor Service", version="1.0.0", lifespan=lifespan)


# ── rule CRUD endpoints ──────────────────────────────────────────────

@app.get("/rules")
async def list_rules():
    return {"rules": list(_rules.values()), "total": len(_rules)}


@app.post("/rules", status_code=201,
          dependencies=[Depends(require_service_auth("monitor_rule_create"))])
async def add_rule(rule_in: RuleIn):
    rule = rule_in.model_dump()
    if not rule.get("id"):
        rule["id"] = str(uuid.uuid4())[:8]
    if rule["id"] in _rules:
        raise HTTPException(409, f"Rule ID already exists: {rule['id']}")
    _rules[rule["id"]] = rule
    _save_rules()
    return {"ok": True, "id": rule["id"]}


@app.put("/rules/{rule_id}",
         dependencies=[Depends(require_service_auth("monitor_rule_update"))])
async def update_rule(rule_id: str, updates: dict = Body(...)):
    if rule_id not in _rules:
        raise HTTPException(404, f"Rule not found: {rule_id}")
    _rules[rule_id].update(updates)
    _save_rules()
    return {"ok": True}


@app.delete("/rules/{rule_id}",
            dependencies=[Depends(require_service_auth("monitor_rule_delete"))])
async def delete_rule(rule_id: str):
    _rules.pop(rule_id, None)
    _last_check.pop(rule_id, None)
    _last_value.pop(rule_id, None)
    _last_fired.pop(rule_id, None)
    _check_errors.pop(rule_id, None)
    _fire_counts.pop(rule_id, None)
    _save_rules()
    return {"ok": True}


@app.post("/rules/{rule_id}/enable",
          dependencies=[Depends(require_service_auth("monitor_rule_enable"))])
async def enable_rule(rule_id: str):
    if rule_id not in _rules:
        raise HTTPException(404, f"Rule not found: {rule_id}")
    _rules[rule_id]["enabled"] = True
    _save_rules()
    return {"ok": True}


@app.post("/rules/{rule_id}/disable",
          dependencies=[Depends(require_service_auth("monitor_rule_disable"))])
async def disable_rule(rule_id: str):
    if rule_id not in _rules:
        raise HTTPException(404, f"Rule not found: {rule_id}")
    _rules[rule_id]["enabled"] = False
    _save_rules()
    return {"ok": True}


@app.post("/rules/{rule_id}/check",
          dependencies=[Depends(require_service_auth("monitor_rule_check"))])
async def manual_check(rule_id: str):
    if rule_id not in _rules:
        raise HTTPException(404, f"Rule not found: {rule_id}")
    asyncio.create_task(_check_rule(_rules[rule_id]))
    return {"ok": True, "message": "Check triggered"}


# ── alert + status endpoints ─────────────────────────────────────────

@app.get("/alerts")
async def get_alerts(limit: int = 50):
    alerts = list(_alert_history)[:max(1, min(limit, MAX_ALERTS))]
    return {"alerts": alerts, "total": len(_alert_history)}


@app.delete("/alerts",
            dependencies=[Depends(require_service_auth("monitor_alerts_clear"))])
async def clear_alerts():
    _alert_history.clear()
    return {"ok": True}


@app.get("/status")
async def status():
    return {
        "rules_total": len(_rules),
        "rules_enabled": sum(1 for r in _rules.values() if r.get("enabled", True)),
        "alerts_total": len(_alert_history),
        "last_value": {rid: str(v) for rid, v in _last_value.items()},
        "last_check": _last_check,
        "fire_counts": _fire_counts,
        "errors": _check_errors,
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rules": len(_rules),
        "alerts": len(_alert_history),
    }


@app.get("/metrics")
async def metrics():
    snap = _budget.snapshot() if _budget else {}
    return {"status": "ok", "error_budget": snap}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8033)
