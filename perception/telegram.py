from __future__ import annotations

import os
import time

import httpx

from common.runtime import detect_device, sanitize_string, setup_json_logger

logger = setup_json_logger("perception-telegram", os.getenv("LOG_PATH", "/tmp/perception-telegram.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

TELEGRAM_API = os.getenv("TELEGRAM_API", "http://telegram-bridge:9000")
LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://langgraph:8007/run")
BRIDGE_SHARED_SECRET = os.getenv("BRIDGE_SHARED_SECRET", "")


def poll_loop() -> None:
    while True:
        try:
            with httpx.Client(timeout=5.0) as client:
                headers = {"x-bridge-secret": BRIDGE_SHARED_SECRET} if BRIDGE_SHARED_SECRET else {}
                resp = client.get(f"{TELEGRAM_API}/messages", headers=headers)
                resp.raise_for_status()
                payload = resp.json()
                for msg in payload.get("messages", []):
                    client.post(
                        LANGGRAPH_URL,
                        json={
                            "user_input": sanitize_string(str(msg.get("text", ""))),
                            "session_id": sanitize_string(str(msg.get("session_id", "telegram"))),
                            "device": DEVICE,
                        },
                    )
        except Exception:
            logger.warning("telegram poll failed")
        time.sleep(5)


if __name__ == "__main__":
    poll_loop()
