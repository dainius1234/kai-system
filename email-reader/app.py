"""Email-reader service — IMAP read-only polling.

Endpoints:
  GET /health                     → {status, connected}
  GET /metrics                    → error budget
  GET /inbox                      ?limit=20&folder=INBOX  → recent emails
  GET /unread                     ?folder=INBOX  → unread count + sample
  POST /refresh                   → force poll now
"""
from __future__ import annotations

import asyncio
import email as email_lib
import imaplib
import os
import time
from contextlib import asynccontextmanager, suppress
from email.header import decode_header
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query

try:
    from common.runtime import setup_json_logger, ErrorBudget
    logger = setup_json_logger("email-reader", os.getenv("LOG_PATH", "/tmp/email-reader.json.log"))
    budget = ErrorBudget(window_seconds=300)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("email-reader")

    class ErrorBudget:
        def __init__(self, **_): pass
        def record(self, *_, **__): pass
        def snapshot(self): return {}
    budget = ErrorBudget()

PORT = int(os.getenv("PORT", "8037"))
MAIL_HOST = os.getenv("MAIL_HOST", "")
MAIL_PORT = int(os.getenv("MAIL_PORT", "993"))
MAIL_USER = os.getenv("MAIL_USER", "")
MAIL_PASS = os.getenv("MAIL_PASS", "")
POLL_INTERVAL = int(os.getenv("EMAIL_POLL_INTERVAL_SECONDS", "120"))
MAX_FETCH = int(os.getenv("EMAIL_MAX_FETCH", "50"))

_start = time.time()
_inbox_cache: List[dict] = []
_unread_count: int = 0
_last_poll: float = 0.0
_poll_error: Optional[str] = None
_poll_task: Optional[asyncio.Task] = None


def _decode_header_value(raw) -> str:
    parts = decode_header(raw or "")
    decoded = []
    for part, charset in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return " ".join(decoded)


def _get_body(msg) -> str:
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if ct == "text/plain" and "attachment" not in disp:
                payload = part.get_payload(decode=True)
                if payload:
                    body = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                    break
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            body = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
    return body[:1000]


def _poll_imap(folder: str = "INBOX") -> tuple[List[dict], int]:
    if not MAIL_HOST or not MAIL_USER or not MAIL_PASS:
        return [], 0
    conn = imaplib.IMAP4_SSL(MAIL_HOST, MAIL_PORT)
    try:
        conn.login(MAIL_USER, MAIL_PASS)
        conn.select(folder, readonly=True)
        _, data = conn.search(None, "ALL")
        all_ids = data[0].split() if data[0] else []
        recent_ids = all_ids[-MAX_FETCH:] if len(all_ids) > MAX_FETCH else all_ids
        _, unread_data = conn.search(None, "UNSEEN")
        unread_count = len(unread_data[0].split()) if unread_data[0] else 0
        messages = []
        for msg_id in reversed(recent_ids):
            _, msg_data = conn.fetch(msg_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email_lib.message_from_bytes(response_part[1])
                    messages.append({
                        "id": msg_id.decode(),
                        "subject": _decode_header_value(msg.get("Subject", "")),
                        "from": _decode_header_value(msg.get("From", "")),
                        "date": msg.get("Date", ""),
                        "snippet": _get_body(msg),
                        "read": "\\Seen" in (msg.get("Flags", "") or ""),
                    })
        return messages, unread_count
    finally:
        with suppress(Exception):
            conn.logout()


async def _poll_loop():
    global _inbox_cache, _unread_count, _last_poll, _poll_error
    while True:
        try:
            loop = asyncio.get_running_loop()
            msgs, unread = await loop.run_in_executor(None, _poll_imap)
            _inbox_cache = msgs
            _unread_count = unread
            _last_poll = time.time()
            _poll_error = None
            logger.info("email poll: %d messages, %d unread", len(msgs), unread)
        except Exception as exc:
            _poll_error = str(exc)
            logger.warning("email poll error: %s", exc)
        await asyncio.sleep(POLL_INTERVAL)


@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _poll_task
    if MAIL_HOST and MAIL_USER and MAIL_PASS:
        _poll_task = asyncio.create_task(_poll_loop())
    else:
        logger.info("MAIL_HOST/USER/PASS not set — email-reader in stub mode")
    yield
    if _poll_task:
        _poll_task.cancel()


app = FastAPI(title="email-reader", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
def health():
    configured = bool(MAIL_HOST and MAIL_USER and MAIL_PASS)
    return {
        "status": "ok",
        "configured": configured,
        "last_poll": _last_poll,
        "poll_error": _poll_error,
        "uptime_seconds": round(time.time() - _start, 1),
    }


@app.get("/metrics")
def metrics():
    return budget.snapshot()


@app.get("/inbox")
def inbox(limit: int = Query(20, ge=1, le=200), folder: str = Query("INBOX")):
    return {"folder": folder, "messages": _inbox_cache[:limit], "total": len(_inbox_cache)}


@app.get("/unread")
def unread(folder: str = Query("INBOX")):
    unread_msgs = [m for m in _inbox_cache if not m.get("read")]
    return {
        "folder": folder,
        "unread_count": _unread_count,
        "sample": unread_msgs[:5],
        "last_poll": _last_poll,
    }


@app.post("/refresh")
async def refresh(folder: str = Query("INBOX")):
    global _inbox_cache, _unread_count, _last_poll, _poll_error
    if not MAIL_HOST or not MAIL_USER or not MAIL_PASS:
        raise HTTPException(503, "email credentials not configured")
    try:
        loop = asyncio.get_running_loop()
        msgs, unread = await loop.run_in_executor(None, lambda: _poll_imap(folder))
        _inbox_cache = msgs
        _unread_count = unread
        _last_poll = time.time()
        _poll_error = None
        return {"ok": True, "messages": len(msgs), "unread": unread}
    except Exception as exc:
        _poll_error = str(exc)
        raise HTTPException(502, f"Poll failed: {exc}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
