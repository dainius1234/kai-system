#!/bin/bash
# kai-handoff READ mode, run automatically at session start, resume,
# /clear, compaction and fork. Plain stdout from a SessionStart hook is
# added to the session's context, so the session begins from MEASURED,
# SOURCED state instead of a lossy summary.
#
# Design rules (checked before this was written):
#   - READ-ONLY. It runs `handoff.py verify` and `handoff.py check`, and
#     nothing else. `python3 -B` so no __pycache__ is written.
#   - NEVER BLOCKS A SESSION. Always exits 0. Only exit 0 and exit 2 are
#     documented for SessionStart; a failure here is REPORTED in stdout,
#     never turned into a non-zero exit.
#   - BOUNDED. The remote comparison (`git ls-remote`) is time-limited; on
#     timeout it re-runs without the remote and says so. Nothing is
#     silently dropped (R10/R11): every fallback is announced.
#   - The log is NON-AUTHORITATIVE working memory. This hook grants
#     nothing and changes nothing.
set -u

REMOTE_TIMEOUT="${KAI_HANDOFF_REMOTE_TIMEOUT:-30}"
LOCAL_TIMEOUT="${KAI_HANDOFF_LOCAL_TIMEOUT:-20}"

ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
TOOL="$ROOT/.claude/skills/kai-handoff/handoff.py"

# The event JSON arrives on stdin; read the trigger if present.
SOURCE="unknown"
if [ ! -t 0 ]; then
  INPUT="$(cat || true)"
  SOURCE="$(printf '%s' "$INPUT" | sed -n 's/.*"source"[[:space:]]*:[[:space:]]*"\([a-z]*\)".*/\1/p' | head -1)"
  SOURCE="${SOURCE:-unknown}"
fi

echo "=== kai-handoff READ (automatic, SessionStart: ${SOURCE}) ==="
echo "Non-authoritative working memory with sources. Grants nothing."

if ! command -v python3 >/dev/null 2>&1; then
  echo "HOOK: python3 not found — READ mode NOT run. Run it by hand:"
  echo "  python3 -B .claude/skills/kai-handoff/handoff.py verify"
  exit 0
fi
if [ ! -f "$TOOL" ]; then
  echo "HOOK: $TOOL not found — READ mode NOT run."
  exit 0
fi

cd "$ROOT" || { echo "HOOK: cannot cd to $ROOT — READ mode NOT run."; exit 0; }

timeout "$REMOTE_TIMEOUT" python3 -B "$TOOL" verify 2>&1
rc=$?
if [ "$rc" -eq 124 ]; then
  echo "HOOK: verify with remote timed out after ${REMOTE_TIMEOUT}s — re-running WITHOUT the remote comparison (remote heads NOT verified):"
  timeout "$LOCAL_TIMEOUT" python3 -B "$TOOL" verify --no-remote 2>&1
  rc=$?
fi
[ "$rc" -ne 0 ] && echo "HOOK: verify exited $rc — the state above is INCOMPLETE; report this before any work."

timeout "$LOCAL_TIMEOUT" python3 -B "$TOOL" check 2>&1
crc=$?
[ "$crc" -ne 0 ] && echo "HOOK: check exited $crc — the handoff log FAILS its rules (see findings above); report this before any work."

cat <<'EOF'
=== Before any work (CLAUDE.md, session start): report to the operator
    (1) the DIFFERS/UNMEASURED lines above, (2) every ⚠ UNBANKED ruling in
    the last entry of kai-pm/HANDOFF_LOG.md, (3) the next authorised step
    quoted with its source. The repository wins for facts. ===
EOF
exit 0
