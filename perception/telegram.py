from __future__ import annotations

import os
import subprocess
import tempfile
import time
from typing import Dict

import httpx

from common.runtime import detect_device, sanitize_string, setup_json_logger

logger = setup_json_logger("perception-telegram", os.getenv("LOG_PATH", "/tmp/perception-telegram.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

TELEGRAM_API = os.getenv("TELEGRAM_API", "http://telegram-bridge:9000")
LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://langgraph:8007/run")
BRIDGE_SHARED_SECRET = os.getenv("BRIDGE_SHARED_SECRET", "")
KEEPER_TELEGRAM_USER_ID = os.getenv("KEEPER_TELEGRAM_USER_ID", "")
KAI_TPM_SIGNING_KEY_CTX = os.getenv("KAI_TPM_SIGNING_KEY_CTX", "/etc/kai/keeper_signing.ctx")
KAI_KEEPER_PUBKEY_CTX = os.getenv("KAI_KEEPER_PUBKEY_CTX", "/etc/kai/keeper_pub.ctx")
KAI_DOCKER_CMD = os.getenv("KAI_DOCKER_CMD", "docker")


def _tpm_verified(challenge: str) -> bool:
    try:
        subprocess.run(["tpm2_createprimary", "-Q", "-C", "o", "-g", "sha256", "-G", "rsa", "-c", "transient.ctx"], check=False, timeout=5)
        with tempfile.TemporaryDirectory() as td:
            msg = os.path.join(td, "challenge.txt")
            sig = os.path.join(td, "challenge.sig")
            with open(msg, "w", encoding="utf-8") as f:
                f.write(challenge)
            subprocess.run(["tpm2_sign", "-Q", "-c", KAI_TPM_SIGNING_KEY_CTX, "-g", "sha256", "-d", msg, "-o", sig], check=True, timeout=5)
            verify = subprocess.run(["tpm2_verifysignature", "-Q", "-c", KAI_KEEPER_PUBKEY_CTX, "-g", "sha256", "-m", msg, "-s", sig], check=False, timeout=5)
            return verify.returncode == 0
    except Exception:
        return False


def _send_reply(client: httpx.Client, session_id: str, text: str, headers: Dict[str, str]) -> None:
    client.post(f"{TELEGRAM_API}/send", headers=headers, json={"session_id": session_id, "text": text}, timeout=5.0)


def _handle_kai_stop(client: httpx.Client, msg: Dict[str, str], headers: Dict[str, str]) -> None:
    if str(msg.get("user_id", "")) != KEEPER_TELEGRAM_USER_ID:
        return
    challenge = f"kai_stop:{msg.get('session_id','telegram')}:{time.time()}"
    if not _tpm_verified(challenge):
        _send_reply(client, str(msg.get("session_id", "telegram")), "Denied", headers)
        return
    subprocess.run([KAI_DOCKER_CMD, "update", "--restart=no", "executor"], check=False)
    subprocess.run([KAI_DOCKER_CMD, "stop", "executor"], check=False)
    subprocess.run([KAI_DOCKER_CMD, "network", "disconnect", "sovereign-net", "executor"], check=False)
    _send_reply(client, str(msg.get("session_id", "telegram")), "Kai isolated", headers)


def _handle_kai_go(client: httpx.Client, msg: Dict[str, str], headers: Dict[str, str]) -> None:
    if str(msg.get("user_id", "")) != KEEPER_TELEGRAM_USER_ID:
        return
    challenge = f"kai_go:{msg.get('session_id','telegram')}:{time.time()}"
    if not _tpm_verified(challenge):
        _send_reply(client, str(msg.get("session_id", "telegram")), "Denied", headers)
        return
    subprocess.run([KAI_DOCKER_CMD, "network", "connect", "sovereign-net", "executor"], check=False)
    subprocess.run([KAI_DOCKER_CMD, "update", "--restart=always", "executor"], check=False)
    subprocess.run([KAI_DOCKER_CMD, "start", "executor"], check=False)
    _send_reply(client, str(msg.get("session_id", "telegram")), "Kai resumed", headers)


def poll_loop() -> None:
    while True:
        try:
            with httpx.Client(timeout=5.0) as client:
                headers = {"x-bridge-secret": BRIDGE_SHARED_SECRET} if BRIDGE_SHARED_SECRET else {}
                resp = client.get(f"{TELEGRAM_API}/messages", headers=headers)
                resp.raise_for_status()
                payload = resp.json()
                for msg in payload.get("messages", []):
                    text = sanitize_string(str(msg.get("text", ""))).strip()
                    if text == "/kai_stop":
                        _handle_kai_stop(client, msg, headers)
                        continue
                    if text == "/kai_go":
                        _handle_kai_go(client, msg, headers)
                        continue
                    client.post(
                        LANGGRAPH_URL,
                        json={
                            "user_input": text,
                            "session_id": sanitize_string(str(msg.get("session_id", "telegram"))),
                            "device": DEVICE,
                        },
                    )
        except Exception:
            logger.warning("telegram poll failed")
        time.sleep(5)


if __name__ == "__main__":
    poll_loop()
