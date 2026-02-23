from __future__ import annotations

import json
import os
import time
from pathlib import Path

from scripts.kai_control import APP_DIR, KeeperRecoveryManager

OUT_DIR = Path(os.getenv("PAPER_BACKUP_DIR", str(APP_DIR / "paper_backups")))
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    manager = KeeperRecoveryManager()
    recovery = manager.generate_paper_recovery()
    vault_root = os.getenv("VAULT_ROOT_TOKEN", "")
    snapshot = {
        "generated_at": int(time.time()),
        "payload": recovery["payload"],
        "words": recovery["words"],
        "qr_path": recovery.get("qr_path", ""),
        "vault_root_token": vault_root,
        "notes": "Print and store in two physical locations. Keep offline.",
    }
    out = OUT_DIR / f"paper_recovery_{time.strftime('%Y%m%d')}.json"
    out.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "backup": str(out), "qr": recovery.get("qr_path", "")}, indent=2))


if __name__ == "__main__":
    main()
