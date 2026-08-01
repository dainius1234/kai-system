"""Dispatch handlers for mutating actuators (tiers 2-8).

Completes UH tracker gap G-09.  Unlike the tier-1 read handlers, these
cause real side effects, so three things differ:

  - every call is a **POST/DELETE** carrying the capability's parameters;
  - each handler declares the ``side_effects`` it produces, which the
    receipt records for audit;
  - the shared service token is attached, because the endpoints they call
    are now authenticated and would otherwise 503.

The registry still gates all of this: nothing here can be invoked without
a consumed, audience-matched capability, and the actuator must have
reached VERIFIED or ACTIVE.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

# actuator → (env var, default URL, {action: (method, path, side_effects)})
MUTATING_ENDPOINTS: Dict[str, Tuple[str, str, Dict[str, Tuple[str, str, List[str]]]]] = {
    # ── Tier 2: isolated local/test operations ───────────────────────
    "shell-sandbox": (
        "SHELL_SANDBOX_URL", "http://shell-sandbox:8055",
        {"sandbox_shell_read": ("POST", "/run", ["subprocess"])},
    ),
    "executor-sandbox": (
        "EXECUTOR_URL", "http://executor:8003",
        {
            "sandbox_python_eval": ("POST", "/execute", ["sandboxed_eval"]),
            "noop": ("POST", "/execute", []),
        },
    ),
    # ── Tier 3: document and browser reads ───────────────────────────
    "web-scout": (
        "AGENTIC_URL", "http://agentic:8001",
        {
            "web_search": ("POST", "/scout/search", ["outbound_http"]),
            "web_fetch": ("POST", "/scout/fetch", ["outbound_http"]),
        },
    ),
    "document-parser": (
        "DOCUMENT_PARSER_URL", "http://document-parser:8032",
        {"document_parse": ("POST", "/parse", ["temp_file"])},
    ),
    "browser-reader": (
        "BROWSER_AGENT_URL", "http://browser-agent:8040",
        {
            "browser_scrape": ("POST", "/scrape", ["browser_navigation"]),
            "browser_screenshot": ("POST", "/screenshot", ["browser_navigation"]),
        },
    ),
    "screen-capture": (
        "SCREEN_CAPTURE_URL", "http://screen-capture:8021",
        {
            "screen_capture": ("POST", "/capture", ["disk_write"]),
            "screen_ocr": ("POST", "/ocr", ["disk_write"]),
        },
    ),
    # ── Tier 4: notifications / draft creation ───────────────────────
    "notify-service": (
        "NOTIFY_URL", "http://notify-service:8031",
        {"notify_desktop": ("POST", "/notify", ["desktop_notification"])},
    ),
    "tts-service": (
        "TTS_URL", "http://tts:8005",
        {"tts_speak": ("POST", "/speak", ["audio_output"])},
    ),
    "monitor-service": (
        "MONITOR_URL", "http://monitor-service:8033",
        {
            "monitor_alert": ("POST", "/rules/{rule_id}/check", ["notification"]),
            "monitor_rule_write": ("POST", "/rules", ["persisted_rule"]),
        },
    ),
    "screen-watcher": (
        "SCREEN_WATCHER_URL", "http://screen-watcher:8036",
        {"screen_change_alert": ("POST", "/check", ["notification"])},
    ),
    # ── Tier 5: file mutations ───────────────────────────────────────
    "vault-sync": (
        "VAULT_SYNC_URL", "http://vault-sync:8047",
        {
            "vault_export": ("POST", "/export", ["vault_file_write"]),
            "vault_ingest": ("POST", "/ingest", ["knowledge_graph_write"]),
        },
    ),
    "checkpoint-manager": (
        "AGENTIC_URL", "http://agentic:8001",
        {
            "checkpoint_create": ("POST", "/checkpoint", ["disk_write"]),
            "checkpoint_restore": (
                "POST", "/checkpoint/{checkpoint_id}/restore",
                ["breaker_state_change"],
            ),
        },
    ),
    "memory-writer": (
        "MEMU_URL", "http://memu-core:8001",
        {"memory_write": ("POST", "/memory", ["persisted_memory"])},
    ),
    # ── Tier 6: calendar / external messages ─────────────────────────
    "telegram-bot": (
        "TELEGRAM_BOT_URL", "http://telegram-bot:8046",
        {
            "telegram_send": ("POST", "/alert", ["external_message"]),
            "telegram_alert": ("POST", "/alert", ["external_message"]),
        },
    ),
    "calendar-writer": (
        "CALENDAR_SERVICE_URL", "http://calendar-service:8043",
        {"calendar_write": ("POST", "/events", ["external_calendar_write"])},
    ),
    # ── Tier 7: recovery / admin ─────────────────────────────────────
    "backup-service": (
        "BACKUP_SERVICE_URL", "http://backup-service:8048",
        {"backup_create": ("POST", "/backup", ["disk_write", "subprocess"])},
    ),
    "ledger-worker": (
        "LEDGER_WORKER_URL", "http://ledger-worker:8049",
        {
            "ledger_archive": ("POST", "/archive", ["disk_write"]),
            "ledger_verify": ("POST", "/verify", []),
        },
    ),
    "supervisor": (
        "SUPERVISOR_URL", "http://supervisor:8050",
        {
            "service_recover": ("POST", "/recover", ["service_restart"]),
            "breaker_reset": ("POST", "/breakers/reset", ["breaker_state_change"]),
        },
    ),
    "heartbeat": (
        "HEARTBEAT_URL", "http://heartbeat:8004",
        {
            "auto_sleep": ("POST", "/sleep", ["memory_mutation"]),
            "memory_compress": ("POST", "/compress", ["memory_mutation"]),
            "memory_decay": ("POST", "/decay", ["memory_mutation"]),
        },
    ),
    # ── Tier 8: financial / destructive / self-modifying ─────────────
    "paper-trader": (
        "AGENTIC_URL", "http://agentic:8001",
        {
            "paper_trade_open": ("POST", "/paper/open", ["simulated_position"]),
            "paper_trade_close": ("POST", "/paper/close", ["simulated_position"]),
        },
    ),
    "executor-shell": (
        "EXECUTOR_URL", "http://executor:8003",
        {
            "shell_exec": ("POST", "/execute", ["subprocess", "irreversible"]),
            "script_exec": ("POST", "/execute", ["subprocess", "irreversible"]),
        },
    ),
    "browser-actor": (
        "BROWSER_AGENT_URL", "http://browser-agent:8040",
        {
            "browser_click": ("POST", "/click", ["web_interaction", "irreversible"]),
            "browser_type": ("POST", "/type", ["web_interaction", "irreversible"]),
        },
    ),
    "db-restore": (
        "BACKUP_SERVICE_URL", "http://backup-service:8048",
        {"db_restore": (
            "POST", "/restore/postgres",
            ["database_overwrite", "irreversible"],
        )},
    ),
}


class MutatingHandlerError(Exception):
    pass


def _base_url(actuator: str) -> str:
    env_key, default, _ = MUTATING_ENDPOINTS[actuator]
    return os.getenv(env_key, default).rstrip("/")


def _auth_headers() -> Dict[str, str]:
    """Bearer header for the now-authenticated downstream endpoints."""
    try:
        from common.service_auth import _load_token
        token = _load_token()
    except Exception:
        token = os.getenv("KAI_SERVICE_TOKEN", "")
    return {"Authorization": f"Bearer {token}"} if token else {}


def side_effects_for(actuator: str, action_type: str) -> List[str]:
    _, _, actions = MUTATING_ENDPOINTS[actuator]
    entry = actions.get(action_type)
    return list(entry[2]) if entry else []


def build_mutating_handler(
    actuator: str,
    http_post: Optional[Callable[[str, Dict[str, Any], Dict[str, str]], Any]] = None,
    timeout: float = 30.0,
) -> Callable[[Dict[str, Any], str], Dict[str, Any]]:
    """Build a capability-gated handler for one mutating actuator."""
    if actuator not in MUTATING_ENDPOINTS:
        raise MutatingHandlerError(
            f"'{actuator}' is not a registered mutating actuator"
        )

    def _default_post(url: str, body: Dict[str, Any], headers: Dict[str, str]) -> Any:
        import httpx
        response = httpx.post(url, json=body, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()

    poster = http_post or _default_post

    def handler(parameters: Dict[str, Any], action_type: str) -> Dict[str, Any]:
        _, _, actions = MUTATING_ENDPOINTS[actuator]
        entry = actions.get(action_type)
        if entry is None:
            raise MutatingHandlerError(
                f"'{actuator}' has no endpoint for action '{action_type}'"
            )
        method, template, effects = entry

        params = dict(parameters or {})
        try:
            path = template.format(**params)
        except KeyError as exc:
            raise MutatingHandlerError(
                f"'{action_type}' requires parameter {exc} for '{actuator}'"
            ) from exc

        # Path parameters are consumed by the URL, not resent in the body.
        body = {k: v for k, v in params.items() if "{" + k + "}" not in template}
        url = f"{_base_url(actuator)}{path}"

        try:
            payload = poster(url, body, _auth_headers())
        except Exception as exc:
            return {
                "actuator": actuator,
                "action_type": action_type,
                "method": method,
                "url": url,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "side_effects": effects,
                # An error after a POST may still have caused the effect.
                # Recording it is what makes reconciliation possible.
                "effect_uncertain": bool(effects),
            }

        return {
            "actuator": actuator,
            "action_type": action_type,
            "method": method,
            "url": url,
            "ok": True,
            "data": payload,
            "side_effects": effects,
            "effect_uncertain": False,
        }

    return handler


def attach_mutating_handlers(
    registry,
    http_post: Optional[Callable] = None,
) -> int:
    """Attach handlers to every registered mutating actuator."""
    attached = 0
    for actuator in MUTATING_ENDPOINTS:
        if registry.get(actuator) is None:
            continue
        registry.set_handler(
            actuator, build_mutating_handler(actuator, http_post)
        )
        attached += 1
    return attached


def attach_all_handlers(registry, http_get=None, http_post=None) -> int:
    """Attach read and mutating handlers across the whole catalogue."""
    from common.actuator_registry.handlers import attach_read_handlers
    return (
        attach_read_handlers(registry, http_get)
        + attach_mutating_handlers(registry, http_post)
    )
