#!/usr/bin/env sh
set -eu

usage() {
  echo "Usage: $0 [--challenge FILE --signature FILE] [--rollback-to COMMIT | --kill-executor | --unlock-vault]" >&2
  exit 1
}

CHALLENGE_FILE=""
SIGNATURE_FILE=""
ROLLBACK_TO=""
DO_KILL="false"
DO_UNLOCK="false"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --challenge) CHALLENGE_FILE="${2:-}"; shift 2 ;;
    --signature) SIGNATURE_FILE="${2:-}"; shift 2 ;;
    --rollback-to=*) ROLLBACK_TO="${1#*=}"; shift ;;
    --kill-executor) DO_KILL="true"; shift ;;
    --unlock-vault) DO_UNLOCK="true"; shift ;;
    *) usage ;;
  esac
done

[ -n "$ROLLBACK_TO" ] || [ "$DO_KILL" = "true" ] || [ "$DO_UNLOCK" = "true" ] || usage

KAI_KEEPER_PUBKEY_CTX="${KAI_KEEPER_PUBKEY_CTX:-/etc/kai/keeper_pub.ctx}"
KAI_TPM_SIGNING_KEY_CTX="${KAI_TPM_SIGNING_KEY_CTX:-/etc/kai/keeper_signing.ctx}"
KAI_AUDIT_REDIS_URL="${KAI_AUDIT_REDIS_URL:-redis://127.0.0.1:6379}"
KAI_DOCKER_CMD="${KAI_DOCKER_CMD:-docker}"
MEMU_LOCAL_URL="${MEMU_LOCAL_URL:-http://127.0.0.1:8001}"
VAULT_ADDR_LOCAL="${VAULT_ADDR_LOCAL:-http://127.0.0.1:8200}"
VAULT_UNSEAL_KEY_FILE="${VAULT_UNSEAL_KEY_FILE:-/etc/kai/vault_unseal.key}"

for bin in tpm2_createprimary tpm2_sign tpm2_verifysignature redis-cli; do
  command -v "$bin" >/dev/null 2>&1 || { echo "missing binary: $bin" >&2; exit 1; }
done

[ -f "$KAI_TPM_SIGNING_KEY_CTX" ] || { echo "missing TPM signing context: $KAI_TPM_SIGNING_KEY_CTX" >&2; exit 1; }
[ -f "$KAI_KEEPER_PUBKEY_CTX" ] || { echo "missing keeper public context: $KAI_KEEPER_PUBKEY_CTX" >&2; exit 1; }

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

challenge="${CHALLENGE_FILE:-$workdir/challenge.bin}"
if [ -z "$CHALLENGE_FILE" ]; then
  printf 'kai-emergency:%s:%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${ROLLBACK_TO:-$DO_KILL$DO_UNLOCK}" > "$challenge"
fi

sig="${SIGNATURE_FILE:-$workdir/challenge.sig}"
if [ -z "$SIGNATURE_FILE" ]; then
  tpm2_sign -Q -c "$KAI_TPM_SIGNING_KEY_CTX" -g sha256 -d "$challenge" -o "$sig"
fi

if ! tpm2_verifysignature -Q -c "$KAI_KEEPER_PUBKEY_CTX" -g sha256 -m "$challenge" -s "$sig" >/dev/null 2>&1; then
  exit 1
fi

signed_ts_file="$workdir/ts.txt"
printf '%s' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$signed_ts_file"
ts_sig="$workdir/ts.sig"
tpm2_sign -Q -c "$KAI_TPM_SIGNING_KEY_CTX" -g sha256 -d "$signed_ts_file" -o "$ts_sig"
ts_sig_b64="$(base64 "$ts_sig" | tr -d '\n')"

log_event() {
  msg="$1"
  redis-cli -u "$KAI_AUDIT_REDIS_URL" XADD audit:logs '*' service kai-emergency level critical msg "$msg" signed_ts "$(cat "$signed_ts_file")" signed_ts_sig "$ts_sig_b64" >/dev/null
}

if [ "$DO_KILL" = "true" ]; then
  "$KAI_DOCKER_CMD" update --restart=no executor >/dev/null || true
  "$KAI_DOCKER_CMD" stop executor >/dev/null
  "$KAI_DOCKER_CMD" network disconnect sovereign-net executor >/dev/null || true
  log_event "keeper emergency kill-switch activated"
fi

if [ -n "$ROLLBACK_TO" ]; then
  curl -fsS -X POST "$MEMU_LOCAL_URL/memory/revert?version=$ROLLBACK_TO" >/dev/null
  "$KAI_DOCKER_CMD" restart sovereign-memu-core executor >/dev/null
  log_event "emergency rollback by keeper, version=$ROLLBACK_TO"
fi

if [ "$DO_UNLOCK" = "true" ]; then
  [ -f "$VAULT_UNSEAL_KEY_FILE" ] || { echo "missing vault unseal key file" >&2; exit 1; }
  curl -fsS -X PUT "$VAULT_ADDR_LOCAL/v1/sys/unseal" -d "{\"key\":\"$(cat "$VAULT_UNSEAL_KEY_FILE")\"}" >/dev/null
  log_event "keeper emergency vault unlock"
fi

exit 0
