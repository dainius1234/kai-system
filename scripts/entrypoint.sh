#!/usr/bin/env bash
set -euo pipefail

TS_IP="${TAILSCALE_IP:-}"
if [[ -z "$TS_IP" ]]; then
  echo "TAILSCALE_IP is required"
  exit 1
fi

if ! command -v ufw >/dev/null 2>&1; then
  echo "ufw not found"
  exit 1
fi

ufw allow from "$TS_IP" to any port 22 proto tcp || true
ufw --force enable

exec "$@"
