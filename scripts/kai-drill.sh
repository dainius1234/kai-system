#!/usr/bin/env sh
set -eu

# Suggested cron: 0 */4 * * * /workspace/kai-system/scripts/kai-drill.sh
export KAI_CONTROL_TEST_MODE="${KAI_CONTROL_TEST_MODE:-false}"
export KAI_DRILL_TEST_MODE="${KAI_DRILL_TEST_MODE:-false}"
export PYTHONPATH="${PYTHONPATH:-}:/workspace/kai-system"

ALERT_URL="${TELEGRAM_ALERT_URL:-http://perception-telegram:9000/alert}"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
LOG_DIR="${KAI_LOG_DIR:-/logs}"
STATUS_PATH="${KAI_MEDIC_STATUS_PATH:-/tmp/kai_medic_status.json}"
mkdir -p "$LOG_DIR" "${KAI_QUARANTINE_DIR:-/quarantine}"

# Proactive medic sweep (offline tools + heuristic quarantine)
python /workspace/kai-system/tools/medic_scan.py >/dev/null 2>&1 || true
if [ -f "$STATUS_PATH" ]; then
  if python - <<PY
import json
from pathlib import Path
s=json.loads(Path("$STATUS_PATH").read_text())
raise SystemExit(0 if s.get("unsafe") else 1)
PY
  then
    MSG="Unsafe state — medic flagged system. Quarantine/scan review required before execution."
    curl -sS -X POST "$ALERT_URL" -H 'content-type: application/json' -d "{\"text\":\"${MSG} (${TS})\"}" >/dev/null 2>&1 || true
  fi
fi

if python - <<'PY'
from pathlib import Path
import os
import tempfile
from common.runtime import AuditStream
import scripts.kai_control as kc

if os.getenv("KAI_DRILL_TEST_MODE", "false").lower() == "true":
    mem = {}
    if kc.AESGCM is None:
        class _DummyAESGCM:
            def __init__(self, key: bytes) -> None:
                self.key = key
            def encrypt(self, nonce: bytes, data: bytes, aad: bytes) -> bytes:
                return data + self.key[:16]
            def decrypt(self, nonce: bytes, data: bytes, aad: bytes) -> bytes:
                return data[:-16]
        kc.AESGCM = _DummyAESGCM
    def _write(path: str, value: str) -> None:
        mem[path] = value
    def _read(path: str):
        return mem.get(path)
    kc.vault_write = _write
    kc.vault_read = _read

    with tempfile.TemporaryDirectory() as td:
        usb1 = Path(td) / "usb1"
        usb2 = Path(td) / "usb2"
        usb1.mkdir(); usb2.mkdir()

        kc.list_usb_mounts = lambda: [usb1, usb2]
        mgr = kc.KeeperRecoveryManager()
        mgr.seal_primary(usb1)
        mgr.seal_backup(usb2, usb1)
        presence = mgr.key_presence()
        if not presence["primary"] or not presence["backup"]:
            raise SystemExit("drill test key check failed")
        rec = mgr.generate_paper_recovery()
        dummy = Path(td) / "dummy"
        dummy.mkdir()
        mgr.restore_from_paper(rec["payload"], rec["words"], dummy)
        if not (dummy / "kai-primary.pub").exists():
            raise SystemExit("drill test restore failed")

    print("drill ok test")
else:
    mgr = kc.KeeperRecoveryManager()
    presence = mgr.key_presence()
    if not presence["primary"]:
        raise SystemExit("primary key check failed")
    if not presence["backup"]:
        raise SystemExit("backup key check failed")

    recovery = mgr.generate_paper_recovery()
    with tempfile.TemporaryDirectory() as td:
        dummy = Path(td) / "dummy-usb"
        dummy.mkdir()
        mgr.restore_from_paper(recovery["payload"], recovery["words"], dummy)
        if not (dummy / "kai-primary.pub").exists():
            raise SystemExit("paper restore simulation failed")

stream = AuditStream("kai-drill", redis_url="redis://redis:6379", required=False)
stream.log("info", "drill passed by keeper")
print("drill ok")
PY
then
  echo "drill passed by keeper, ts=${TS}"
  exit 0
fi

curl -sS -X POST "$ALERT_URL" -H 'content-type: application/json' -d "{\"text\":\"Drill failed — check keys (${TS})\"}" >/dev/null 2>&1 || true
exit 1
