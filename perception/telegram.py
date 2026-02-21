from __future__ import annotations

import logging
import os
import time
from logging.handlers import RotatingFileHandler

import httpx

LOG_PATH = os.getenv("LOG_PATH", "/tmp/perception-telegram.json.log")
handler = RotatingFileHandler(LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=30)
handler.setFormatter(logging.Formatter('{"time":"%(asctime)s","level":"%(levelname)s","service":"%(name)s","msg":"%(message)s"}'))
logger = logging.getLogger("perception-telegram")
logger.setLevel(logging.INFO)
logger.handlers = [handler]
logger.propagate = False

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
logger.info("Running on %s.", DEVICE)

TELEGRAM_API = os.getenv("TELEGRAM_API", "http://telegram-bridge:9000")
LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://langgraph:8007/run")


def poll_loop() -> None:
    while True:
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{TELEGRAM_API}/messages")
                resp.raise_for_status()
                payload = resp.json()
                for msg in payload.get("messages", []):
                    client.post(
                        LANGGRAPH_URL,
                        json={
                            "user_input": str(msg.get("text", ""))[:1024],
                            "session_id": str(msg.get("session_id", "telegram"))[:1024],
                            "device": DEVICE,
                        },
                    )
        except Exception:
            logger.warning("telegram poll failed")
        time.sleep(5)


if __name__ == "__main__":
    poll_loop()
