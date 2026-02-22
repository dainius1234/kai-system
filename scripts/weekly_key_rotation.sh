#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=${LOG_DIR:-/var/log/sovereign}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/weekly_key_rotation.log"

ROTATE_SECONDS=${ROTATE_SECONDS:-604800} PYTHONPATH=. python scripts/auto_rotate_hmac.py | tee -a "$LOG_FILE"
