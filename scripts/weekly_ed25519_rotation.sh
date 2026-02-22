#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=${LOG_DIR:-/var/log/sovereign}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/weekly_ed25519_rotation.log"

ED25519_ROTATE_SECONDS=${ED25519_ROTATE_SECONDS:-604800} python scripts/auto_rotate_ed25519.py | tee -a "$LOG_FILE"
