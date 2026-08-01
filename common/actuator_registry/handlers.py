"""Dispatch handlers for read-only (tier-1) actuators.

Closes the tier-1 half of UH tracker gap G-01.  Each handler performs the
actual service call behind a consumed capability, so "migrated" means
traffic genuinely flows through the capability pipeline rather than the
registry merely knowing the actuator exists.

Every handler here is a **read**.  They issue GET requests, take no
parameters that could mutate remote state, and return the decoded
response.  A handler that needed to write would belong to a higher
migration tier and would not be eligible yet.

The HTTP client is injectable so tests exercise dispatch without live
services.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, Tuple

# actuator identity → (env var, default URL, path per action_type)
READ_ONLY_ENDPOINTS: Dict[str, Tuple[str, str, Dict[str, str]]] = {
    "broker-bridge": (
        "BROKER_BRIDGE_URL", "http://broker-bridge:8034",
        {
            "market_ticker_read": "/ticker/{symbol}",
            "account_balance_read": "/balance",
            "orderbook_read": "/depth/{symbol}",
        },
    ),
    "alpha-signals": (
        "ALPHA_SIGNALS_URL", "http://agentic:8001",
        {"alpha_signal_read": "/alpha/{symbol}/composite"},
    ),
    "market-data": (
        "MARKET_DATA_URL", "http://agentic:8001",
        {"market_data_read": "/market-data/prices"},
    ),
    "docker-watcher": (
        "DOCKER_WATCHER_URL", "http://docker-watcher:8041",
        {"container_status_read": "/summary"},
    ),
    "git-watcher": (
        "GIT_WATCHER_URL", "http://git-watcher:8044",
        {"git_status_read": "/summary"},
    ),
    "email-reader": (
        "EMAIL_READER_URL", "http://email-reader:8037",
        {"email_read": "/inbox"},
    ),
    "calendar-service": (
        "CALENDAR_SERVICE_URL", "http://calendar-service:8043",
        {"calendar_read": "/summary"},
    ),
    "news-feed": (
        "NEWS_FEED_URL", "http://news-feed:8038",
        {"news_read": "/articles"},
    ),
    "sysmetrics": (
        "SYSMETRICS_URL", "http://sysmetrics:8035",
        {"system_metrics_read": "/snapshot"},
    ),
    "weather-service": (
        "WEATHER_SERVICE_URL", "http://weather-service:8039",
        {"weather_read": "/summary"},
    ),
    "service-watchdog": (
        "SERVICE_WATCHDOG_URL", "http://agentic:8001",
        {"service_health_read": "/watchdog/status"},
    ),
}


class HandlerError(Exception):
    pass


def _base_url(actuator: str) -> str:
    env_key, default, _ = READ_ONLY_ENDPOINTS[actuator]
    return os.getenv(env_key, default).rstrip("/")


def _resolve_path(actuator: str, action_type: str, parameters: Dict[str, Any]) -> str:
    _, _, paths = READ_ONLY_ENDPOINTS[actuator]
    template = paths.get(action_type)
    if template is None:
        raise HandlerError(
            f"'{actuator}' has no endpoint for action '{action_type}'"
        )
    try:
        return template.format(**parameters)
    except KeyError as exc:
        raise HandlerError(
            f"'{action_type}' requires parameter {exc} for '{actuator}'"
        ) from exc


def build_read_handler(
    actuator: str,
    http_get: Optional[Callable[[str], Any]] = None,
    timeout: float = 10.0,
) -> Callable[[Dict[str, Any], str], Dict[str, Any]]:
    """Build a dispatch handler for one read-only actuator.

    ``http_get`` is injectable; the default uses httpx.  It receives a
    fully-resolved URL and returns decoded JSON.
    """
    if actuator not in READ_ONLY_ENDPOINTS:
        raise HandlerError(f"'{actuator}' is not a registered read-only actuator")

    def _default_get(url: str) -> Any:
        import httpx
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()

    getter = http_get or _default_get

    def handler(parameters: Dict[str, Any], action_type: str) -> Dict[str, Any]:
        path = _resolve_path(actuator, action_type, parameters or {})
        url = f"{_base_url(actuator)}{path}"
        try:
            payload = getter(url)
        except Exception as exc:
            # A read that fails returns an explicit unavailable result
            # rather than raising: the receipt should record that the
            # read was attempted and did not succeed.
            return {
                "actuator": actuator,
                "action_type": action_type,
                "url": url,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {
            "actuator": actuator,
            "action_type": action_type,
            "url": url,
            "ok": True,
            "data": payload,
        }

    return handler


def attach_read_handlers(
    registry,
    http_get: Optional[Callable[[str], Any]] = None,
) -> int:
    """Attach handlers to every registered read-only actuator.

    Returns the number attached.  Actuators absent from the registry are
    skipped rather than raising, so a partial catalogue still works.
    """
    attached = 0
    for actuator in READ_ONLY_ENDPOINTS:
        if registry.get(actuator) is None:
            continue
        registry.set_handler(actuator, build_read_handler(actuator, http_get))
        attached += 1
    return attached
