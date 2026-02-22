#!/usr/bin/env bash
set -euo pipefail

TOOL_GATE_URL=${TOOL_GATE_URL:-http://127.0.0.1:8000}
MEMU_URL=${MEMU_URL:-http://127.0.0.1:8001}
DASHBOARD_URL=${DASHBOARD_URL:-http://127.0.0.1:8080}
SESSION_ID=${SESSION_ID:-bootstrap-token-1}

check_keys() {
  local json="$1"
  shift
  python - "$json" "$@" <<'PY'
import json,sys
payload=json.loads(sys.argv[1])
for key in sys.argv[2:]:
    if key not in payload:
        raise SystemExit(f"missing key: {key}")
print("ok")
PY
}

route_resp=$(curl -fsS -X POST "$MEMU_URL/route" -H 'content-type: application/json' \
  -d '{"query":"status check","session_id":"smoke-session"}')
check_keys "$route_resp" specialist context_payload

ledger_verify=$(curl -fsS "$TOOL_GATE_URL/ledger/verify")
check_keys "$ledger_verify" status valid

dash_resp=$(curl -fsS "$DASHBOARD_URL/")
check_keys "$dash_resp" core_ready alive_nodes ledger_size memory_count policy_mode device_summary

ready_resp=$(curl -fsS "$DASHBOARD_URL/readiness")
check_keys "$ready_resp" status core_ready


echo "contract smoke passed"
