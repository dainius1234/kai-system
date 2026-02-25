"""Telegram Bot — Kai's phone.

Gives Kai a Telegram interface so the operator can chat from anywhere.

Pipeline:
  Text msg  → langgraph /chat → streaming response → Telegram reply
  Voice msg → download audio → audio-service STT → transcript
              → langgraph /chat → response → TTS → voice + text reply

Requires TELEGRAM_BOT_TOKEN from @BotFather.
Set ALLOWED_CHAT_IDS to restrict access (comma-separated, empty = allow all).

Endpoints:
  GET /health   — bot status
  GET /metrics  — message counts
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from common.runtime import detect_device, sanitize_string, setup_json_logger

logger = setup_json_logger(
    "telegram-bot",
    os.getenv("LOG_PATH", "/tmp/telegram-bot.json.log"),
)
DEVICE = detect_device()

# ── config ──────────────────────────────────────────────────────────
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://langgraph:8007")
AUDIO_SERVICE_URL = os.getenv("AUDIO_SERVICE_URL", "http://audio-service:8021")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://tts-service:8030")
TTS_ENABLED = os.getenv("TTS_ENABLED", "true").lower() == "true"
DEFAULT_MODE = os.getenv("DEFAULT_MODE", "PUB")
ALLOWED_CHAT_IDS = os.getenv("ALLOWED_CHAT_IDS", "")  # empty = allow all

TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}" if BOT_TOKEN else ""
TG_FILE = f"https://api.telegram.org/file/bot{BOT_TOKEN}" if BOT_TOKEN else ""

_bot_running = False
_message_count = 0
_voice_count = 0
_last_ts = 0.0
# per-chat mode overrides  {chat_id: "PUB"|"WORK"}
_chat_modes: Dict[int, str] = {}


# ── helpers ─────────────────────────────────────────────────────────
def _is_allowed(chat_id: int) -> bool:
    if not ALLOWED_CHAT_IDS:
        return True
    allowed = set()
    for cid in ALLOWED_CHAT_IDS.split(","):
        cid = cid.strip()
        if cid.lstrip("-").isdigit():
            allowed.add(int(cid))
    return chat_id in allowed


async def _tg(method: str, **kwargs) -> Dict[str, Any]:
    """Call Telegram Bot API."""
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.post(f"{TG_API}/{method}", **kwargs)
        r.raise_for_status()
        return r.json()


async def _send_text(chat_id: int, text: str, reply_to: int = 0):
    """Send a text message (falls back to plain if Markdown fails)."""
    payload: Dict[str, Any] = {
        "chat_id": chat_id,
        "text": text[:4096],
        "parse_mode": "Markdown",
    }
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    try:
        await _tg("sendMessage", json=payload)
    except Exception:
        payload.pop("parse_mode", None)
        await _tg("sendMessage", json=payload)


async def _send_voice(chat_id: int, audio: bytes, caption: str = ""):
    """Send a voice note."""
    data: Dict[str, str] = {"chat_id": str(chat_id)}
    if caption:
        data["caption"] = caption[:1024]
    files = {"voice": ("kai.mp3", audio, "audio/mpeg")}
    await _tg("sendVoice", data=data, files=files)


async def _send_typing(chat_id: int):
    try:
        await _tg("sendChatAction", json={"chat_id": chat_id, "action": "typing"})
    except Exception:
        pass


async def _download_file(file_id: str) -> bytes:
    """Download a Telegram file by file_id."""
    info = await _tg("getFile", json={"file_id": file_id})
    fpath = info["result"]["file_path"]
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.get(f"{TG_FILE}/{fpath}")
        r.raise_for_status()
        return r.content


# ── service calls ───────────────────────────────────────────────────
async def _chat_kai(text: str, session_id: str, mode: str) -> str:
    """Stream /chat and collect full response."""
    try:
        timeout = httpx.Timeout(pool=180.0, connect=10.0, read=120.0)
        async with httpx.AsyncClient(timeout=timeout) as c:
            async with c.stream(
                "POST",
                f"{LANGGRAPH_URL}/chat",
                json={"message": text, "session_id": session_id, "mode": mode},
            ) as resp:
                resp.raise_for_status()
                tokens = []
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "token" in chunk:
                                tokens.append(chunk["token"])
                        except json.JSONDecodeError:
                            continue
                return "".join(tokens) or "[no response]"
    except Exception as e:
        logger.error("chat failed: %s", e)
        return f"[Kai is thinking... error: {str(e)[:100]}]"


async def _transcribe(audio_bytes: bytes, filename: str) -> str:
    """Send audio to audio-service for STT."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            files = {"file": (filename, audio_bytes, "audio/ogg")}
            r = await c.post(f"{AUDIO_SERVICE_URL}/capture/file", files=files)
            r.raise_for_status()
            return r.json().get("transcript", "[transcription failed]")
    except Exception as e:
        logger.error("STT failed: %s", e)
        return "[could not transcribe voice message]"


async def _synthesize(text: str) -> Optional[bytes]:
    """Get TTS audio bytes from TTS service."""
    if not TTS_ENABLED:
        return None
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.post(
                f"{TTS_SERVICE_URL}/synthesize",
                json={"text": text[:2000], "voice": "kai-default"},
            )
            r.raise_for_status()
            if len(r.content) > 100:
                return r.content
            return None
    except Exception as e:
        logger.warning("TTS failed: %s", e)
        return None


# ── update handler ──────────────────────────────────────────────────
async def _handle(update: Dict[str, Any]):
    global _message_count, _voice_count, _last_ts

    msg = update.get("message")
    if not msg:
        return

    chat_id = msg.get("chat", {}).get("id")
    if not chat_id or not _is_allowed(chat_id):
        return

    session_id = f"tg-{chat_id}"
    mode = _chat_modes.get(chat_id, DEFAULT_MODE)
    _last_ts = time.time()

    text = msg.get("text", "")

    # ── commands ────────────────────────────────────────────────────
    if text.startswith("/start"):
        await _send_text(
            chat_id,
            "*Hey! I'm Kai* — Kind And Intelligent.\n\n"
            "Send me a text message to chat.\n"
            "Send a voice note and I'll listen + respond with voice.\n\n"
            "*Commands:*\n"
            "/mode pub — Casual pub mode (uncensored)\n"
            "/mode work — Professional mode\n"
            "/status — My current state\n"
            "/voice — Toggle voice replies on/off",
        )
        return

    if text.startswith("/mode"):
        parts = text.split()
        if len(parts) > 1 and parts[1].upper() in ("PUB", "WORK"):
            _chat_modes[chat_id] = parts[1].upper()
            await _send_text(chat_id, f"Switched to *{parts[1].upper()}* mode.")
        else:
            await _send_text(chat_id, "Usage: `/mode pub` or `/mode work`")
        return

    if text.startswith("/status"):
        await _send_text(
            chat_id,
            f"*Kai Status*\n"
            f"Mode: {mode}\n"
            f"Messages: {_message_count}\n"
            f"Voice msgs: {_voice_count}\n"
            f"TTS: {'ON' if TTS_ENABLED else 'OFF'}\n"
            f"Device: {DEVICE}",
        )
        return

    if text.startswith("/"):
        return  # ignore unknown commands

    # ── voice message ───────────────────────────────────────────────
    voice = msg.get("voice")
    if voice:
        _voice_count += 1
        _message_count += 1
        await _send_typing(chat_id)

        audio_bytes = await _download_file(voice["file_id"])
        transcript = await _transcribe(audio_bytes, "voice.ogg")

        if transcript.startswith("["):
            await _send_text(chat_id, f"Heard you but: {transcript}")
            return

        await _send_text(chat_id, f"_🎤 {transcript}_")
        await _send_typing(chat_id)

        response = await _chat_kai(transcript, session_id, mode)
        await _send_text(chat_id, response)

        # voice reply
        audio = await _synthesize(response)
        if audio:
            await _send_voice(chat_id, audio)
        return

    # ── text message ────────────────────────────────────────────────
    if text:
        _message_count += 1
        await _send_typing(chat_id)

        response = await _chat_kai(sanitize_string(text), session_id, mode)
        await _send_text(chat_id, response)

        # short responses also get voice
        if TTS_ENABLED and len(response) < 500:
            audio = await _synthesize(response)
            if audio:
                await _send_voice(chat_id, audio)


# ── polling loop ────────────────────────────────────────────────────
async def _poll():
    global _bot_running

    if not BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set — bot disabled")
        return

    _bot_running = True
    offset = 0
    logger.info("Telegram bot polling started (mode=%s)", DEFAULT_MODE)

    while _bot_running:
        try:
            timeout = httpx.Timeout(pool=60.0, connect=10.0, read=45.0)
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.get(
                    f"{TG_API}/getUpdates",
                    params={
                        "offset": offset,
                        "timeout": 30,
                        "allowed_updates": '["message"]',
                    },
                )
                r.raise_for_status()
                for upd in r.json().get("result", []):
                    offset = upd["update_id"] + 1
                    try:
                        await _handle(upd)
                    except Exception as e:
                        logger.error("handle error: %s", e)
        except httpx.ReadTimeout:
            continue
        except Exception as e:
            logger.error("poll error: %s", e)
            await asyncio.sleep(5)


# ── FastAPI app ─────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app):
    task = asyncio.create_task(_poll())
    yield
    global _bot_running
    _bot_running = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="Telegram Bot — Kai's Phone", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok" if _bot_running else "no_token",
        "bot_token_set": bool(BOT_TOKEN),
        "tts_enabled": TTS_ENABLED,
        "messages": _message_count,
        "voice_messages": _voice_count,
        "device": DEVICE,
    }


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    return {
        "messages": _message_count,
        "voice_messages": _voice_count,
        "last_message_ts": _last_ts,
        "bot_running": _bot_running,
        "tts_enabled": TTS_ENABLED,
        "chat_modes": {str(k): v for k, v in _chat_modes.items()},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8025")))
