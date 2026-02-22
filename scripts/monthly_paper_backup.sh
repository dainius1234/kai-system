#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=${LOG_DIR:-/var/log/sovereign}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/monthly_paper_backup.log"

PYTHONPATH=. python scripts/monthly_paper_backup.py | tee -a "$LOG_FILE"
