#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/var/log/sovereign"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/health_sweep.log"

check() {
  local url="$1"
  echo "[check] $url" | tee -a "$LOG_FILE"
  curl -fsS "$url" | tee -a "$LOG_FILE"
  echo | tee -a "$LOG_FILE"
}

check "http://localhost:8000/health"
check "http://localhost:8080/health"
check "http://localhost:8000/ledger/stats"
check "http://localhost:8000/ledger/verify"
check "http://localhost:8001/health"
check "http://localhost:8001/memory/stats"
check "http://localhost:8001/memory/diagnostics"
check "http://localhost:8002/health"
check "http://localhost:8007/health"
check "http://localhost:8010/status"

echo "health sweep complete" | tee -a "$LOG_FILE"
