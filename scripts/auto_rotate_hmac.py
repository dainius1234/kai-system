from __future__ import annotations

import json
import os
import secrets
import time
from pathlib import Path

STATE_PATH = Path(os.getenv("HMAC_ROTATION_STATE", "output/hmac_rotation_state.json"))
ROTATE_SECONDS = int(os.getenv("HMAC_ROTATE_SECONDS", str(7 * 24 * 3600)))


def _load() -> dict:
    if not STATE_PATH.exists():
        return {"current": "v1", "secrets": {"v1": secrets.token_hex(32)}, "revoked": [], "rotated_at": time.time()}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def _save(data: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    data = _load()
    now = time.time()
    age = now - float(data.get("rotated_at", 0))
    if age < ROTATE_SECONDS:
        print(json.dumps({"rotated": False, "age_s": int(age), "next_in_s": int(ROTATE_SECONDS - age)}, indent=2))
        return

    current = str(data.get("current", "v1"))
    next_id = f"v{int(current.lstrip('v') or '1') + 1}"
    data.setdefault("secrets", {})[next_id] = secrets.token_hex(32)
    data.setdefault("revoked", []).append(current)
    data["current"] = next_id
    data["rotated_at"] = now
    _save(data)

    prev_secret = data["secrets"].get(current, "")
    print(json.dumps({
        "rotated": True,
        "current_key_id": next_id,
        "prev_key_id": current,
        "env": {
            "INTERSERVICE_HMAC_KEY_ID": next_id,
            "INTERSERVICE_HMAC_SECRET": data["secrets"][next_id],
            "INTERSERVICE_HMAC_SECRET_PREV": prev_secret,
            "INTERSERVICE_HMAC_REVOKED_IDS": ",".join(data.get("revoked", [])),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
