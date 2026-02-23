from __future__ import annotations

import base64
import json
import os
import secrets
import time
from pathlib import Path

STATE_PATH = Path(os.getenv("ED25519_STATE_PATH", "security/ed25519_rotation_state.json"))
ROTATE_SECONDS = int(os.getenv("ED25519_ROTATE_SECONDS", "604800"))


def _new_keypair() -> dict:
    # lightweight offline placeholder key material (ed25519-ready envelope)
    priv = base64.b64encode(secrets.token_bytes(32)).decode("ascii")
    pub = base64.b64encode(secrets.token_bytes(32)).decode("ascii")
    kid = f"k{int(time.time())}"
    return {"key_id": kid, "private": priv, "public": pub}


def load_state() -> dict:
    if not STATE_PATH.exists():
        kp = _new_keypair()
        data = {"current": kp["key_id"], "previous": None, "keys": {kp["key_id"]: kp}, "rotated_at": time.time(), "mode": "single"}
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return data
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def main() -> None:
    state = load_state()
    now = time.time()
    age = now - float(state.get("rotated_at", 0))
    if age < ROTATE_SECONDS:
        print(json.dumps({"rotated": False, "age_s": int(age), "next_in_s": int(ROTATE_SECONDS - age), "mode": state.get("mode", "single")}, indent=2))
        return

    new = _new_keypair()
    prev = state.get("current")
    state.setdefault("keys", {})[new["key_id"]] = new
    state["previous"] = prev
    state["current"] = new["key_id"]
    state["rotated_at"] = now
    state["mode"] = "dual_sign"

    # on next run after another interval, drop old to complete migration
    if prev and state.get("drop_previous_on_next", False):
        state["keys"].pop(prev, None)
        state["previous"] = None
        state["mode"] = "single"
        state["drop_previous_on_next"] = False
    else:
        state["drop_previous_on_next"] = True

    save_state(state)
    print(json.dumps({"rotated": True, "current": state["current"], "previous": state.get("previous"), "mode": state["mode"]}, indent=2))


if __name__ == "__main__":
    main()
