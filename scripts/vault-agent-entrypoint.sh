#!/bin/sh
set -eu

if [ -n "${VAULT_ADDR:-}" ] && [ -n "${VAULT_ROLE:-}" ] && [ -n "${VAULT_SECRET_PATH:-}" ]; then
  echo "[vault-agent] fetching startup secrets for role=${VAULT_ROLE} path=${VAULT_SECRET_PATH}"
  if command -v curl >/dev/null 2>&1; then
    TOKEN="${VAULT_TOKEN:-${VAULT_DEV_ROOT_TOKEN_ID:-}}"
    if [ -n "${TOKEN:-}" ]; then
      curl -sS -H "X-Vault-Token: ${TOKEN}" "${VAULT_ADDR}/v1/${VAULT_SECRET_PATH}" >/tmp/vault-secrets.json || true
    fi
  fi
fi

exec "$@"
