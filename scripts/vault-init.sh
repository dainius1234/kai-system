#!/usr/bin/env sh
set -eu

export VAULT_ADDR="${VAULT_ADDR:-http://127.0.0.1:8200}"
ROOT_TOKEN="${VAULT_ROOT_TOKEN:-devroot}"
UNSEAL_KEY_FALLBACK="${VAULT_UNSEAL_KEY:-}"

if [ -n "${TPM_UNSEAL_CMD:-}" ]; then
  UNSEAL_KEY="$(sh -c "$TPM_UNSEAL_CMD")"
else
  UNSEAL_KEY="$UNSEAL_KEY_FALLBACK"
fi

vault login "$ROOT_TOKEN" >/dev/null 2>&1 || true

vault kv put secret/sovereign/executor \
  tool_gate_url="${TOOL_GATE_URL:-http://tool-gate:8000}" \
  heartbeat_url="${HEARTBEAT_URL:-http://heartbeat:8010}" >/dev/null

vault kv put secret/sovereign/memuco \
  database_url="${MEMU_DATABASE_URL:-postgresql://keeper:localdev@postgres:5432/memu_db}" \
  vector_store="${VECTOR_STORE:-pgvector}" >/dev/null

# rotation metadata stamp (real rotation should be handled by external scheduler)
vault kv put secret/sovereign/rotation last_rotated="$(date -u +%Y-%m-%dT%H:%M:%SZ)" interval_days="30" >/dev/null

echo "Vault init complete."
