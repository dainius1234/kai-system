#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLOCKLIST_FILE="${ROOT_DIR}/scripts/.pypi_shadow_blocklist"

if [[ ! -f "${BLOCKLIST_FILE}" ]]; then
  echo "ERROR: Missing blocklist file: ${BLOCKLIST_FILE}" >&2
  exit 1
fi

declare -A allow_map=()
if [[ -n "${KAI_SHADOW_ALLOW:-}" ]]; then
  IFS=',' read -ra allow_entries <<<"${KAI_SHADOW_ALLOW}"
  for entry in "${allow_entries[@]}"; do
    name="$(echo "${entry}" | xargs)"
    [[ -n "${name}" ]] && allow_map["${name}"]=1
  done
fi

violations=()
while IFS= read -r name; do
  name="${name%%#*}"
  name="$(echo "${name}" | xargs)"
  [[ -z "${name}" ]] && continue

  if [[ -d "${ROOT_DIR}/${name}" ]] && [[ -z "${allow_map["${name}"]:-}" ]]; then
    violations+=("${name}")
  fi
done < "${BLOCKLIST_FILE}"

if (( ${#violations[@]} > 0 )); then
  echo "ERROR: Repo-root folders shadow common PyPI package names:" >&2
  for folder in "${violations[@]}"; do
    echo "  - ${folder}/" >&2
  done
  echo >&2
  echo "Temporary unblock (cleanup-only): KAI_SHADOW_ALLOW=langgraph" >&2
  echo "Do not keep long-term allows after rename lands." >&2
  exit 1
fi

echo "OK: no blocked PyPI-shadow folders found at repo root."
