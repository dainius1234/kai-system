"""Perception adapters — convert raw sensor HTTP responses to PerceptionEvents.

Each adapter function takes the raw dict from a sensor service endpoint and
returns a fully constructed PerceptionEvent.  Adapters:

  - compute a raw_hash from the payload for dedup
  - set source_timestamp when the sensor provides timing
  - set appropriate confidence based on available data
  - tag events for downstream filtering

All adapters are pure functions with no side effects.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from common.contracts.base import Principal, Provenance, RiskTier
from common.contracts.perception import EventSource, PerceptionEvent


def _hash_payload(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _base_kwargs(
    source: str,
    principal: Principal,
    purpose: str = "perception",
) -> Dict[str, Any]:
    return {
        "principal": principal,
        "purpose": purpose,
        "provenance": Provenance(source=source),
    }


def adapt_weather(
    data: Dict[str, Any], principal: Principal
) -> Optional[PerceptionEvent]:
    if not data:
        return None
    summary = data.get("summary", "")
    if not summary:
        return None
    payload = {"summary": str(summary)}
    for key in ("temp_c", "feels_like_c", "wind_kph", "weathercode"):
        if key in data:
            payload[key] = data[key]
    return PerceptionEvent(
        event_type="weather_reading",
        source_type=EventSource.WEATHER,
        payload=payload,
        raw_hash=_hash_payload(payload),
        tags=["environment", "weather"],
        **_base_kwargs("weather-service", principal),
    )


def adapt_calendar(
    data: Dict[str, Any], principal: Principal
) -> Optional[PerceptionEvent]:
    if not data:
        return None
    summary = data.get("summary", "")
    if not summary:
        return None
    payload = {"summary": str(summary)}
    events = data.get("events", [])
    if events:
        payload["event_count"] = len(events)
    return PerceptionEvent(
        event_type="calendar_reading",
        source_type=EventSource.CALENDAR,
        payload=payload,
        raw_hash=_hash_payload(payload),
        tags=["schedule", "calendar"],
        **_base_kwargs("calendar-service", principal),
    )


def adapt_docker(
    data: Dict[str, Any], principal: Principal
) -> Optional[PerceptionEvent]:
    if not data:
        return None
    summary = data.get("summary", "")
    if not summary:
        return None
    payload = {"summary": str(summary)}
    for key in ("containers_running", "unhealthy_count"):
        if key in data:
            payload[key] = data[key]
    return PerceptionEvent(
        event_type="docker_status",
        source_type=EventSource.DOCKER,
        payload=payload,
        raw_hash=_hash_payload(payload),
        tags=["infrastructure", "docker"],
        **_base_kwargs("docker-watcher", principal),
    )


def adapt_git(
    data: Dict[str, Any], principal: Principal
) -> Optional[PerceptionEvent]:
    if not data:
        return None
    summary = data.get("summary", "")
    if not summary:
        return None
    payload = {"summary": str(summary)}
    for key in ("repo_count", "dirty_count"):
        if key in data:
            payload[key] = data[key]
    return PerceptionEvent(
        event_type="git_status",
        source_type=EventSource.GIT,
        payload=payload,
        raw_hash=_hash_payload(payload),
        tags=["development", "git"],
        **_base_kwargs("git-watcher", principal),
    )


def adapt_system_metrics(
    data: Dict[str, Any], principal: Principal
) -> Optional[PerceptionEvent]:
    if not data:
        return None
    payload: Dict[str, Any] = {}
    if "cpu_percent" in data:
        payload["cpu_percent"] = data["cpu_percent"]
    mem = data.get("memory", {})
    if isinstance(mem, dict) and "percent" in mem:
        payload["memory_percent"] = mem["percent"]
    disk = data.get("disk", {})
    if isinstance(disk, dict) and "percent" in disk:
        payload["disk_percent"] = disk["percent"]
    if not payload:
        return None
    return PerceptionEvent(
        event_type="system_metrics",
        source_type=EventSource.SYSTEM,
        payload=payload,
        raw_hash=_hash_payload(payload),
        tags=["infrastructure", "system"],
        **_base_kwargs("sysmetrics", principal),
    )


def adapt_screen(
    data: Dict[str, Any], principal: Principal
) -> Optional[PerceptionEvent]:
    if not data:
        return None
    if not data.get("watching"):
        return None
    diff = data.get("last_diff_score", 0)
    payload = {
        "watching": True,
        "diff_score": diff,
        "active": diff > 0.05,
    }
    return PerceptionEvent(
        event_type="screen_activity",
        source_type=EventSource.SCREEN,
        payload=payload,
        raw_hash=_hash_payload(payload),
        tags=["user_activity", "screen"],
        **_base_kwargs("screen-watcher", principal),
    )


def adapt_clipboard(
    data: Dict[str, Any], principal: Principal
) -> Optional[PerceptionEvent]:
    if not data:
        return None
    content = data.get("content", "").strip()
    if not content:
        return None
    payload = {
        "content_length": len(content),
        "content_preview": content[:120],
    }
    clip_id = data.get("id", "")
    if clip_id:
        payload["clip_id"] = clip_id
    return PerceptionEvent(
        event_type="clipboard_update",
        source_type=EventSource.CLIPBOARD,
        payload=payload,
        raw_hash=_hash_payload(payload),
        tags=["user_activity", "clipboard"],
        **_base_kwargs("clipboard-service", principal),
    )


def adapt_email(
    data: Dict[str, Any], principal: Principal
) -> Optional[PerceptionEvent]:
    if not data:
        return None
    unread = data.get("unread_count", data.get("count", 0))
    payload: Dict[str, Any] = {"unread_count": unread}
    subjects = data.get("subjects", [])
    if subjects:
        payload["latest_subjects"] = subjects[:5]
    if not unread and not subjects:
        return None
    return PerceptionEvent(
        event_type="email_check",
        source_type=EventSource.EMAIL,
        payload=payload,
        raw_hash=_hash_payload(payload),
        tags=["communication", "email"],
        **_base_kwargs("email-reader", principal),
    )


def adapt_news(
    data: Dict[str, Any], principal: Principal
) -> Optional[PerceptionEvent]:
    if not data:
        return None
    articles = data.get("articles", [])
    if not articles:
        return None
    payload = {
        "article_count": len(articles),
        "headlines": [a.get("title", "") for a in articles[:5]],
    }
    return PerceptionEvent(
        event_type="news_update",
        source_type=EventSource.NEWS,
        payload=payload,
        raw_hash=_hash_payload(payload),
        tags=["information", "news"],
        **_base_kwargs("news-feed", principal),
    )


def adapt_telegram(
    data: Dict[str, Any], principal: Principal
) -> Optional[PerceptionEvent]:
    if not data:
        return None
    text = data.get("text", data.get("message", "")).strip()
    if not text:
        return None
    payload = {
        "text_length": len(text),
        "chat_id": data.get("chat_id", ""),
    }
    if data.get("is_voice"):
        payload["is_voice"] = True
    return PerceptionEvent(
        event_type="telegram_message",
        source_type=EventSource.TELEGRAM,
        payload=payload,
        raw_hash=_hash_payload(payload),
        tags=["communication", "telegram"],
        **_base_kwargs("telegram-bot", principal),
    )


def adapt_market(
    data: Dict[str, Any], principal: Principal
) -> Optional[PerceptionEvent]:
    if not data:
        return None
    payload: Dict[str, Any] = {}
    for key in ("symbol", "price", "bid", "ask", "volume", "change_pct"):
        if key in data:
            payload[key] = data[key]
    if not payload:
        return None
    return PerceptionEvent(
        event_type="market_tick",
        source_type=EventSource.MARKET,
        payload=payload,
        raw_hash=_hash_payload(payload),
        tags=["market", "financial"],
        **_base_kwargs("broker-bridge", principal),
    )


ADAPTER_REGISTRY: Dict[str, Any] = {
    "weather": adapt_weather,
    "calendar": adapt_calendar,
    "docker": adapt_docker,
    "git": adapt_git,
    "system": adapt_system_metrics,
    "screen": adapt_screen,
    "clipboard": adapt_clipboard,
    "email": adapt_email,
    "news": adapt_news,
    "telegram": adapt_telegram,
    "market": adapt_market,
}
