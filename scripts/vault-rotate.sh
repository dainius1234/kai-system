#!/usr/bin/env sh
set -eu
export VAULT_ADDR="${VAULT_ADDR:-http://vault:8200}"
[ -n "${VAULT_ROOT_TOKEN:-}" ] || { echo "VAULT_ROOT_TOKEN required" >&2; exit 1; }
vault login "${VAULT_ROOT_TOKEN}" >/dev/null
while true; do
  vault kv patch secret/sovereign/executor rotated_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)" >/dev/null
  sleep 2592000
done
