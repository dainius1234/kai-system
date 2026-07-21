#!/usr/bin/env bash
# backup_offsite.sh — Encrypt and archive Kai's critical data for off-site storage.
#
# What it backs up:
#   1. CIS financial records  (/data/finance/)
#   2. Letta agent memory     (/data/letta/)
#   3. Decision log           (kai-pm/DECISIONS.md)
#   4. Changelog              (CHANGELOG.md)
#   5. Full Postgres dump     (via backup-service API)
#
# Encryption: GPG symmetric AES-256 (requires BACKUP_PASSPHRASE env var).
# Output:     A single .tar.gz.gpg archive in BACKUP_DEST (default: /data/backup/offsite/).
#
# Usage:
#   BACKUP_PASSPHRASE=mysecret bash scripts/backup_offsite.sh
#
# Optional env vars:
#   BACKUP_DEST          — where to write the archive (default: /data/backup/offsite)
#   BACKUP_SERVICE_URL   — backup-service base URL (default: http://localhost:8054)
#   FINANCE_ROOT         — CIS records root (default: /data/finance)
#   LETTA_BASE_PATH      — Letta SQLite root (default: /data/letta)
#   BACKUP_SKIP_ENCRYPT  — set to "true" to skip GPG encryption (dev/test only)

set -euo pipefail

BACKUP_DEST="${BACKUP_DEST:-/data/backup/offsite}"
BACKUP_SERVICE_URL="${BACKUP_SERVICE_URL:-http://localhost:8054}"
FINANCE_ROOT="${FINANCE_ROOT:-/data/finance}"
LETTA_BASE_PATH="${LETTA_BASE_PATH:-/data/letta}"
BACKUP_SKIP_ENCRYPT="${BACKUP_SKIP_ENCRYPT:-false}"

TS=$(date -u +%Y%m%d-%H%M%S)
WORK_DIR=$(mktemp -d)
ARCHIVE_NAME="kai-offsite-${TS}"
trap 'rm -rf "$WORK_DIR"' EXIT

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "=== Kai off-site backup starting — ${TS} ==="

# ── 1. Trigger Postgres backup via backup-service ─────────────────────────────
log "Requesting Postgres dump from backup-service..."
if curl -sf -X POST "${BACKUP_SERVICE_URL}/backup/postgres" \
    -o "${WORK_DIR}/backup_postgres_response.json" \
    --connect-timeout 10 --max-time 120; then
  log "Postgres dump complete: $(cat "${WORK_DIR}/backup_postgres_response.json" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("path","?"), d.get("size_bytes","?")+"B")' 2>/dev/null || true)"
else
  log "WARNING: backup-service not reachable — skipping Postgres dump"
fi

# ── 2. Stage critical files ───────────────────────────────────────────────────
STAGE="${WORK_DIR}/stage"
mkdir -p "${STAGE}"

# CIS finance records
if [ -d "${FINANCE_ROOT}" ]; then
  log "Staging finance records from ${FINANCE_ROOT}..."
  cp -r "${FINANCE_ROOT}" "${STAGE}/finance"
else
  log "WARNING: FINANCE_ROOT ${FINANCE_ROOT} not found — skipping"
fi

# Letta agent memory (SQLite)
if [ -d "${LETTA_BASE_PATH}" ]; then
  log "Staging Letta memory from ${LETTA_BASE_PATH}..."
  cp -r "${LETTA_BASE_PATH}" "${STAGE}/letta"
else
  log "WARNING: LETTA_BASE_PATH ${LETTA_BASE_PATH} not found — skipping"
fi

# Decision log + changelog (source control files — always present)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cp "${REPO_ROOT}/kai-pm/DECISIONS.md" "${STAGE}/DECISIONS.md" 2>/dev/null || true
cp "${REPO_ROOT}/CHANGELOG.md"        "${STAGE}/CHANGELOG.md"  2>/dev/null || true

# ── 3. Create manifest ────────────────────────────────────────────────────────
cat > "${STAGE}/MANIFEST.txt" << MANIFEST
Kai Off-site Backup
Timestamp: ${TS}
Host: $(hostname)
Contents:
$(find "${STAGE}" -type f | sed "s|${STAGE}/||" | sort)
MANIFEST

# ── 4. Create archive ─────────────────────────────────────────────────────────
mkdir -p "${BACKUP_DEST}"
TARBALL="${WORK_DIR}/${ARCHIVE_NAME}.tar.gz"
log "Creating archive ${ARCHIVE_NAME}.tar.gz..."
tar -czf "${TARBALL}" -C "${WORK_DIR}/stage" .

TARBALL_SIZE=$(du -sh "${TARBALL}" | cut -f1)
TARBALL_SHA=$(sha256sum "${TARBALL}" | cut -d' ' -f1)
log "Archive: ${TARBALL_SIZE}, SHA-256: ${TARBALL_SHA}"

# ── 5. Encrypt (unless skipped) ───────────────────────────────────────────────
if [ "${BACKUP_SKIP_ENCRYPT}" = "true" ]; then
  log "Encryption skipped (BACKUP_SKIP_ENCRYPT=true)"
  FINAL="${BACKUP_DEST}/${ARCHIVE_NAME}.tar.gz"
  cp "${TARBALL}" "${FINAL}"
else
  if [ -z "${BACKUP_PASSPHRASE:-}" ]; then
    log "ERROR: BACKUP_PASSPHRASE not set — cannot encrypt. Set it or use BACKUP_SKIP_ENCRYPT=true."
    exit 1
  fi
  if ! command -v gpg &>/dev/null; then
    log "ERROR: gpg not found — install gnupg or use BACKUP_SKIP_ENCRYPT=true."
    exit 1
  fi
  FINAL="${BACKUP_DEST}/${ARCHIVE_NAME}.tar.gz.gpg"
  log "Encrypting archive with GPG (AES-256)..."
  echo "${BACKUP_PASSPHRASE}" | gpg \
    --batch \
    --yes \
    --passphrase-fd 0 \
    --symmetric \
    --cipher-algo AES256 \
    --output "${FINAL}" \
    "${TARBALL}"
  FINAL_SHA=$(sha256sum "${FINAL}" | cut -d' ' -f1)
  log "Encrypted archive: $(du -sh "${FINAL}" | cut -f1), SHA-256: ${FINAL_SHA}"
fi

# ── 6. Write checksum sidecar ─────────────────────────────────────────────────
sha256sum "${FINAL}" > "${FINAL}.sha256"

# ── 7. Prune archives older than 90 days ─────────────────────────────────────
find "${BACKUP_DEST}" -name "kai-offsite-*.tar.gz*" -mtime +90 -delete 2>/dev/null || true

log "=== Backup complete → ${FINAL} ==="
log ""
log "To restore:"
log "  gpg --decrypt ${FINAL} | tar -xzf - -C /restore/path"
log ""
log "To copy off-site (example with rsync):"
log "  rsync -av ${BACKUP_DEST}/ user@remote-host:/backups/kai/"
