#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=${LOG_DIR:-/var/log/sovereign}
LOG_FILE="$LOG_DIR/health_sweep.log"
mkdir -p "$LOG_DIR"

TOOL_GATE_URL=${TOOL_GATE_URL:-http://127.0.0.1:8000}
MEMU_URL=${MEMU_URL:-http://127.0.0.1:8001}
EXECUTOR_URL=${EXECUTOR_URL:-http://127.0.0.1:8002}
DASHBOARD_URL=${DASHBOARD_URL:-http://127.0.0.1:8080}

endpoints=(
  "$TOOL_GATE_URL/health"
  "$MEMU_URL/health"
  "$EXECUTOR_URL/health"
  "$DASHBOARD_URL/health"
  "$DASHBOARD_URL/readiness"
)

status=0
for url in "${endpoints[@]}"; do
  if curl -fsS "$url" >/dev/null; then
    echo "$(date -Is) OK $url" | tee -a "$LOG_FILE"
  else
    echo "$(date -Is) FAIL $url" | tee -a "$LOG_FILE"
    status=1
  fi
done

exit $status
