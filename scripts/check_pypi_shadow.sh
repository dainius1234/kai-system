#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLOCKLIST_FILE="${ROOT_DIR}/scripts/.pypi_shadow_blocklist"

if [[ ! -f "${BLOCKLIST_FILE}" ]]; then
  echo "ERROR: Missing blocklist file: ${BLOCKLIST_FILE}" >&2
  exit 1
fi

# langgraph/ is a permanent backward-compat shim (symlinks into agentic/,
# kept so pre-rename references don't break) — not transitional debt, so
# it's allowed by default rather than via a one-off env var.
declare -A allow_map=([langgraph]=1)
if [[ -n "${KAI_SHADOW_ALLOW:-}" ]]; then
  IFS=',' read -ra allow_entries <<<"${KAI_SHADOW_ALLOW}"
  for entry in "${allow_entries[@]}"; do
    name="$(echo "${entry}" | xargs)"
    [[ "${name}" =~ ^[a-zA-Z0-9._-]+$ ]] || continue
    [[ -n "${name}" ]] && allow_map["${name}"]=1
  done
fi

violations=()
while IFS= read -r name; do
  name="${name%%#*}"
  name="$(echo "${name}" | xargs)"
  [[ -z "${name}" ]] && continue
  [[ "${name}" =~ ^[a-zA-Z0-9._-]+$ ]] || continue

  allowed_entry="${allow_map["${name}"]:-}"
  if [[ -d "${ROOT_DIR}/${name}" ]] && [[ -z "${allowed_entry}" ]]; then
    violations+=("${name}")
  fi
done < "${BLOCKLIST_FILE}"

if (( ${#violations[@]} > 0 )); then
  echo "ERROR: Repo-root folders shadow common PyPI package names:" >&2
  for folder in "${violations[@]}"; do
    echo "  - ${folder}/" >&2
  done
  echo >&2
  echo "Temporary unblock: KAI_SHADOW_ALLOW=<name1,name2,...>" >&2
  exit 1
fi

echo "OK: no blocked PyPI-shadow folders found at repo root."
