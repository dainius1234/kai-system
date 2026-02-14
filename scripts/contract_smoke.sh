#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/var/log/sovereign"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/contract_smoke.log"

resp_tg=$(curl -fsS -X POST http://localhost:8000/gate/request \
  -H 'content-type: application/json' \
  -d '{"tool":"shell","params":{"cmd":"echo hi"},"confidence":0.95,"actor_did":"smoke","request_source":"contract_smoke"}')

echo "$resp_tg" | tee -a "$LOG_FILE"
echo "$resp_tg" | python - <<'PY'
import json,sys
obj=json.load(sys.stdin)
for k in ["approved","reason_code","ledger_hash","policy_version"]:
    assert k in obj, f"missing {k}"
print("tool-gate contract ok")
PY

resp_memu=$(curl -fsS -X POST http://localhost:8001/route \
  -H 'content-type: application/json' \
  -d '{"query":"need policy plan","session_id":"smoke","timestamp":"now"}')

echo "$resp_memu" | tee -a "$LOG_FILE"
echo "$resp_memu" | python - <<'PY'
import json,sys
obj=json.load(sys.stdin)
assert "specialist" in obj
cp=obj.get("context_payload",{})
assert "device" in cp
assert "metadata" in cp and "session_id" in cp["metadata"]
print("memu route contract ok")
PY

resp_dash=$(curl -fsS http://localhost:8080/)
echo "$resp_dash" | tee -a "$LOG_FILE"
echo "$resp_dash" | python - <<'PY'
import json,sys
obj=json.load(sys.stdin)
for k in ["core_ready","alive_nodes","ledger_size","memory_count","policy_mode","device_summary"]:
    assert k in obj, f"missing {k}"
print("dashboard contract ok")
PY

echo "contract smoke complete" | tee -a "$LOG_FILE"
