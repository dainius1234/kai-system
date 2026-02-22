#!/usr/bin/env sh
set -eu

export VAULT_ADDR="${VAULT_ADDR:-http://127.0.0.1:8200}"
if [ -n "${TPM_UNSEAL_CMD:-}" ]; then
  VAULT_RUNTIME_TOKEN="$(sh -c "$TPM_UNSEAL_CMD")"
elif [ -n "${VAULT_ROOT_TOKEN:-}" ]; then
  VAULT_RUNTIME_TOKEN="${VAULT_ROOT_TOKEN}"
else
  echo "VAULT_ROOT_TOKEN or TPM_UNSEAL_CMD is required" >&2
  exit 1
fi

vault login "$VAULT_RUNTIME_TOKEN" >/dev/null

vault kv put secret/sovereign/executor \
  tool_gate_url="${TOOL_GATE_URL:-http://tool-gate:8000}" \
  heartbeat_url="${HEARTBEAT_URL:-http://heartbeat:8010}" \
  telegram_bot_token="${TELEGRAM_BOT_TOKEN:-}" >/dev/null

vault kv put secret/sovereign/memuco \
  database_url="${MEMU_DATABASE_URL:-postgresql://keeper:localdev@postgres:5432/memu_db}" \
  vector_store="${VECTOR_STORE:-pgvector}" >/dev/null

vault kv get secret/sovereign/executor >/dev/null

echo "Vault init complete."
