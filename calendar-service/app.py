"""Calendar service — CalDAV event polling for upcoming schedule awareness.

Set CALDAV_URL, CALDAV_USER, CALDAV_PASS to connect (Google Calendar, Nextcloud, iCloud).
Runs in stub mode (no crash) when unconfigured.

Endpoints:
  GET /health           → {status, configured, last_poll, uptime_seconds}
  GET /metrics          → error budget
  GET /events/today     → events starting today
  GET /events/upcoming  ?days=7  → events in next N days
  GET /summary          → one-sentence summary for agentic context
  POST /refresh         → force-poll now
"""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query

try:
    from common.runtime import setup_json_logger, ErrorBudget
    logger = setup_json_logger("calendar-service", os.getenv("LOG_PATH", "/tmp/calendar-service.json.log"))
    budget = ErrorBudget(window_seconds=300)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("calendar-service")
    class ErrorBudget:
        def __init__(self, **_): pass
        def record(self, *_, **__): pass
        def snapshot(self): return {}
    budget = ErrorBudget()

try:
    import caldav
    from icalendar import Calendar as iCalendar
    _CALDAV_OK = True
except ImportError:
    _CALDAV_OK = False
    logger.info("caldav/icalendar not available — calendar-service in stub mode")

PORT = int(os.getenv("PORT", "8043"))
CALDAV_URL  = os.getenv("CALDAV_URL", "")
CALDAV_USER = os.getenv("CALDAV_USER", "")
CALDAV_PASS = os.getenv("CALDAV_PASS", "")
REFRESH_INTERVAL = int(os.getenv("CALDAV_REFRESH_SECONDS", "300"))
LOOK_AHEAD_DAYS = int(os.getenv("CALDAV_LOOK_AHEAD_DAYS", "14"))

_start = time.time()
_events: List[dict] = []
_last_poll: float = 0.0
_poll_error: Optional[str] = None
_poll_task: Optional[asyncio.Task] = None


def _dt_to_iso(dt) -> str:
    if isinstance(dt, datetime):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    if isinstance(dt, date):
        return dt.isoformat()
    return str(dt)


def _poll_caldav() -> List[dict]:
    if not _CALDAV_OK or not CALDAV_URL or not CALDAV_USER or not CALDAV_PASS:
        return []
    client = caldav.DAVClient(url=CALDAV_URL, username=CALDAV_USER, password=CALDAV_PASS)
    principal = client.principal()
    events = []
    now = datetime.now(timezone.utc)
    end = now + timedelta(days=LOOK_AHEAD_DAYS)
    for cal in principal.calendars():
        try:
            results = cal.date_search(start=now, end=end, expand=True)
            for vevent in results:
                try:
                    comp = iCalendar.from_ical(vevent.data)
                    for component in comp.walk():
                        if component.name != "VEVENT":
                            continue
                        dtstart = component.get("DTSTART")
                        dtend = component.get("DTEND")
                        events.append({
                            "uid": str(component.get("UID", "")),
                            "summary": str(component.get("SUMMARY", "(no title)")),
                            "start": _dt_to_iso(dtstart.dt) if dtstart else None,
                            "end": _dt_to_iso(dtend.dt) if dtend else None,
                            "location": str(component.get("LOCATION", "")),
                            "description": str(component.get("DESCRIPTION", ""))[:300],
                            "calendar": str(cal.name),
                        })
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("calendar %s fetch error: %s", cal.name, exc)
    events.sort(key=lambda e: e.get("start") or "")
    return events


async def _poll_loop():
    global _events, _last_poll, _poll_error
    while True:
        try:
            loop = asyncio.get_running_loop()
            evts = await loop.run_in_executor(None, _poll_caldav)
            _events = evts
            _last_poll = time.time()
            _poll_error = None
            logger.info("calendar polled: %d upcoming events", len(evts))
        except Exception as exc:
            _poll_error = str(exc)
            logger.warning("calendar poll error: %s", exc)
        await asyncio.sleep(REFRESH_INTERVAL)


@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _poll_task
    configured = bool(CALDAV_URL and CALDAV_USER and CALDAV_PASS)
    if configured and _CALDAV_OK:
        _poll_task = asyncio.create_task(_poll_loop())
    else:
        logger.info("calendar-service stub mode (CALDAV_URL/USER/PASS not set or caldav not installed)")
    yield
    if _poll_task:
        _poll_task.cancel()


app = FastAPI(title="calendar-service", version="0.1.0", lifespan=_lifespan)


def _today_events() -> List[dict]:
    today = date.today().isoformat()
    return [e for e in _events if (e.get("start") or "").startswith(today)]


def _upcoming_events(days: int = 7) -> List[dict]:
    cutoff = (date.today() + timedelta(days=days)).isoformat()
    today = date.today().isoformat()
    return [e for e in _events if today <= (e.get("start") or "") <= cutoff]


@app.get("/health")
def health():
    configured = bool(CALDAV_URL and CALDAV_USER and CALDAV_PASS)
    return {
        "status": "ok",
        "configured": configured,
        "caldav_available": _CALDAV_OK,
        "last_poll": _last_poll,
        "poll_error": _poll_error,
        "event_count": len(_events),
        "uptime_seconds": round(time.time() - _start, 1),
    }


@app.get("/metrics")
def metrics():
    return budget.snapshot()


@app.get("/events/today")
def events_today():
    return {"date": date.today().isoformat(), "events": _today_events()}


@app.get("/events/upcoming")
def events_upcoming(days: int = Query(7, ge=1, le=90)):
    return {"days": days, "events": _upcoming_events(days)}


@app.get("/summary")
def summary():
    today_evts = _today_events()
    upcoming = _upcoming_events(7)
    if not (CALDAV_URL and CALDAV_USER and CALDAV_PASS):
        return {"summary": "Calendar not configured (set CALDAV_URL, CALDAV_USER, CALDAV_PASS)."}
    if not today_evts and not upcoming:
        return {"summary": "No upcoming events in the next 7 days."}
    parts = []
    if today_evts:
        titles = ", ".join(e["summary"] for e in today_evts[:3])
        parts.append(f"Today: {titles}")
    if upcoming:
        next_evt = upcoming[0]
        parts.append(f"Next: {next_evt['summary']} on {(next_evt.get('start') or '')[:10]}")
    return {"summary": ". ".join(parts) + "."}


@app.post("/refresh")
async def refresh():
    global _events, _last_poll, _poll_error
    if not (CALDAV_URL and CALDAV_USER and CALDAV_PASS):
        raise HTTPException(503, "CalDAV credentials not configured")
    if not _CALDAV_OK:
        raise HTTPException(503, "caldav library not installed")
    try:
        loop = asyncio.get_running_loop()
        evts = await loop.run_in_executor(None, _poll_caldav)
        _events = evts
        _last_poll = time.time()
        _poll_error = None
        return {"ok": True, "events": len(evts)}
    except Exception as exc:
        _poll_error = str(exc)
        raise HTTPException(502, f"Poll failed: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
