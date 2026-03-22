"""External World Anchor — offline proxy for real-world context.

Provides date/time awareness, local file-based news/notes, and adaptive
nudge context so Kai can ground its advice in reality.

All data is local/offline — no cloud APIs.  News and events come from
local files in DATA_DIR that the operator can update manually or via
a cron job with a proxy fetch.

Source: OpenClaw "world-anchor" skill pattern
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI

app = FastAPI(title="World Anchor", version="0.2.0")

DATA_DIR = Path(os.getenv("WORLD_ANCHOR_DATA_DIR", "/tmp/world-anchor"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
NEWS_FILE = DATA_DIR / "news.json"
EVENTS_FILE = DATA_DIR / "events.json"
MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")

# Seed files if missing
if not NEWS_FILE.exists():
    NEWS_FILE.write_text("[]", encoding="utf-8")
if not EVENTS_FILE.exists():
    EVENTS_FILE.write_text("[]", encoding="utf-8")


def _load_json(path: Path) -> List[Dict[str, Any]]:
    """Safely load a JSON array from a file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_json(path: Path, data: List) -> None:
    """Safely write a JSON array to a file."""
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════
# DATE / TIME CONTEXT
# ═══════════════════════════════════════════════════════════════════════

def _date_context() -> Dict[str, Any]:
    """Rich date/time context for nudge adaptation."""
    now = datetime.now()
    tomorrow = now + timedelta(days=1)

    # Day type
    is_weekend = now.weekday() >= 5
    is_monday = now.weekday() == 0
    is_friday = now.weekday() == 4

    # Time of day
    hour = now.hour
    if hour < 6:
        time_of_day = "early_morning"
    elif hour < 12:
        time_of_day = "morning"
    elif hour < 17:
        time_of_day = "afternoon"
    elif hour < 21:
        time_of_day = "evening"
    else:
        time_of_day = "night"

    # Adaptive suggestions
    suggestions: List[str] = []
    if is_monday:
        suggestions.append("It's Monday — plan the week?")
    if is_friday:
        suggestions.append("It's Friday — review what got done this week?")
    if time_of_day == "night":
        suggestions.append("It's late — consider wrapping up for the day")
    if time_of_day == "early_morning":
        suggestions.append("You're up early — focused deep work window")
    if is_weekend:
        suggestions.append("Weekend mode — lighter workload okay")

    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_name": now.strftime("%A"),
        "tomorrow": tomorrow.strftime("%A"),
        "week_number": now.isocalendar()[1],
        "is_weekend": is_weekend,
        "is_monday": is_monday,
        "is_friday": is_friday,
        "time_of_day": time_of_day,
        "hour": hour,
        "suggestions": suggestions,
    }


# ═══════════════════════════════════════════════════════════════════════
# LOCAL NEWS / NOTES FEED
# ═══════════════════════════════════════════════════════════════════════

def _recent_news(limit: int = 10) -> List[Dict[str, Any]]:
    """Read recent news items from local file."""
    items = _load_json(NEWS_FILE)
    # Sort by timestamp descending
    items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return items[:limit]


# ═══════════════════════════════════════════════════════════════════════
# EVENTS / CALENDAR
# ═══════════════════════════════════════════════════════════════════════

def _upcoming_events(days: int = 7) -> List[Dict[str, Any]]:
    """Read events from local file, filter to upcoming N days."""
    events = _load_json(EVENTS_FILE)
    now = datetime.now()
    cutoff = now + timedelta(days=days)
    upcoming = []
    for ev in events:
        try:
            ev_date = datetime.fromisoformat(ev.get("date", ""))
            if now <= ev_date <= cutoff:
                upcoming.append(ev)
        except (ValueError, TypeError):
            continue
    upcoming.sort(key=lambda x: x.get("date", ""))
    return upcoming


# ═══════════════════════════════════════════════════════════════════════
# COMBINED WORLD CONTEXT
# ═══════════════════════════════════════════════════════════════════════

def _world_context() -> Dict[str, Any]:
    """Full world anchor snapshot for consumption by nudge engine."""
    date_ctx = _date_context()
    news = _recent_news(5)
    events = _upcoming_events(7)
    return {
        "date": date_ctx,
        "news": news,
        "news_count": len(news),
        "events": events,
        "events_count": len(events),
    }


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "world-anchor"}


@app.get("/context")
async def get_context() -> Dict[str, Any]:
    """Full world context snapshot (date + news + events)."""
    return {"status": "ok", **_world_context()}


@app.get("/date")
async def get_date() -> Dict[str, Any]:
    """Date/time context with adaptive suggestions."""
    return {"status": "ok", **_date_context()}


@app.get("/news")
async def get_news(limit: int = 10) -> Dict[str, Any]:
    """Recent local news/notes."""
    limit = min(max(limit, 1), 50)
    items = _recent_news(limit)
    return {"status": "ok", "items": items, "count": len(items)}


@app.post("/news")
async def add_news(item: Dict[str, Any]) -> Dict[str, Any]:
    """Add a news item to the local feed."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "title": str(item.get("title", ""))[:200],
        "summary": str(item.get("summary", ""))[:500],
        "source": str(item.get("source", "manual"))[:100],
    }
    items = _load_json(NEWS_FILE)
    items.append(entry)
    # Keep last 200 items
    if len(items) > 200:
        items = items[-200:]
    _save_json(NEWS_FILE, items)
    return {"status": "ok", "entry": entry}


@app.get("/events")
async def get_events(days: int = 7) -> Dict[str, Any]:
    """Upcoming events within N days."""
    days = min(max(days, 1), 90)
    events = _upcoming_events(days)
    return {"status": "ok", "events": events, "count": len(events)}


@app.post("/events")
async def add_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Add an event to the local calendar."""
    entry = {
        "date": str(event.get("date", datetime.now().isoformat()))[:30],
        "title": str(event.get("title", ""))[:200],
        "description": str(event.get("description", ""))[:500],
        "category": str(event.get("category", "general"))[:50],
    }
    events = _load_json(EVENTS_FILE)
    events.append(entry)
    if len(events) > 500:
        events = events[-500:]
    _save_json(EVENTS_FILE, events)
    return {"status": "ok", "entry": entry}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8055")))
