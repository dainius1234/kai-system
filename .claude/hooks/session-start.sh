#!/bin/bash
# kai-handoff READ mode, run automatically at session start, resume,
# /clear, compaction and fork. Plain stdout from a SessionStart hook is
# added to the session's context, so the session begins from MEASURED,
# SOURCED state instead of a lossy summary.
#
# Design rules (checked before this was written):
#   - READ-ONLY. It runs `handoff.py verify`, `check` and `due`, and
#     nothing else. `python3 -B` so no __pycache__ is written.
#   - NEVER BLOCKS A SESSION. Always exits 0. Only exit 0 and exit 2 are
#     documented for SessionStart; a failure here is REPORTED in stdout,
#     never turned into a non-zero exit.
#   - BOUNDED. The remote comparison (`git ls-remote`) is time-limited; on
#     timeout it re-runs without the remote and says so. Nothing is
#     silently dropped (R10/R11): every fallback is announced.
#   - The log is NON-AUTHORITATIVE working memory. This hook grants
#     nothing and changes nothing.
#   - ANY BRANCH. A session started from `main` (the app's default) has
#     no handoff tool or log. The live branch is named in ONE place,
#     .claude/handoff-branch; the hook fetches it (time-bounded), runs READ
#     in a temporary detached worktree, and removes the worktree.
#   - NEVER MAKES A FULL CLONE SHALLOW. `git fetch --depth` on a full
#     clone converts the WHOLE repository to shallow and cuts history
#     (measured 60 -> 50 commits; Orion measured 60 -> 40). --depth=50 is
#     passed only when the clone is already shallow (cloud sessions).
#     The only side effects are Git's fetch refs and that transient worktree.
#     This file is byte-identical on every branch that carries it, so
#     merging branches never conflicts on it.
set -u

REMOTE_TIMEOUT="${KAI_HANDOFF_REMOTE_TIMEOUT:-30}"
LOCAL_TIMEOUT="${KAI_HANDOFF_LOCAL_TIMEOUT:-20}"
FETCH_TIMEOUT="${KAI_HANDOFF_FETCH_TIMEOUT:-45}"

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
cd "$ROOT" || { echo "HOOK: cannot cd to $ROOT — READ mode NOT run."; exit 0; }

LIVE="$(head -n1 "$ROOT/.claude/handoff-branch" 2>/dev/null | tr -d '[:space:]')"
CUR="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"

# READ in directory $1: verify (bounded; the no-remote fallback is
# announced), check, due. Every failure is printed as a HOOK: line.
read_in() {
  local dir="$1" tool="$1/.claude/skills/kai-handoff/handoff.py" rc crc
  cd "$dir" || { echo "HOOK: cannot cd to $dir — READ mode NOT run."; return; }
  timeout "$REMOTE_TIMEOUT" python3 -B "$tool" verify 2>&1
  rc=$?
  if [ "$rc" -eq 124 ]; then
    echo "HOOK: verify with remote timed out after ${REMOTE_TIMEOUT}s — re-running WITHOUT the remote comparison (remote heads NOT verified):"
    timeout "$LOCAL_TIMEOUT" python3 -B "$tool" verify --no-remote 2>&1
    rc=$?
  fi
  [ "$rc" -ne 0 ] && echo "HOOK: verify exited $rc — the state above is INCOMPLETE; report this before any work."
  timeout "$LOCAL_TIMEOUT" python3 -B "$tool" check 2>&1
  crc=$?
  [ "$crc" -ne 0 ] && echo "HOOK: check exited $crc — the handoff log FAILS its rules (see findings above); report this before any work."
  timeout "$LOCAL_TIMEOUT" python3 -B "$tool" due 2>&1
  rc=$?
  [ "$rc" -gt 1 ] && echo "HOOK: due exited $rc — whether a WRITE is owed is NOT measured (that copy of handoff.py may predate 'due')."
  cd "$ROOT" || true
}

if [ -f "$TOOL" ] && [ -f "$ROOT/kai-pm/HANDOFF_LOG.md" ]; then
  read_in "$ROOT"
  if [ -n "$LIVE" ] && [ "$CUR" != "$LIVE" ]; then
    echo "HOOK: this session is on '$CUR', NOT the live handoff branch '$LIVE' [FILE .claude/handoff-branch]. The READ above is this branch's copy of the log. An entry committed here reaches the live log only when merged into '$LIVE'."
  fi
elif [ -z "$LIVE" ]; then
  echo "HOOK: no handoff tool/log on '$CUR' and no .claude/handoff-branch pointer — READ mode NOT run. Start the session on the live handoff branch."
else
  echo "HOOK: this session is on '$CUR', which has no handoff tool/log. The live handoff branch is '$LIVE' [FILE .claude/handoff-branch]. Reading it from there (fetch, temporary worktree):"
  git worktree prune 2>/dev/null
  DEPTH=""
  [ "$(git rev-parse --is-shallow-repository 2>/dev/null)" = "true" ] && DEPTH="--depth=50"
  timeout "$FETCH_TIMEOUT" git fetch --quiet $DEPTH origin "refs/heads/$LIVE" 2>&1
  frc=$?
  if [ "$frc" -ne 0 ]; then
    if [ "$frc" -eq 124 ]; then
      echo "HOOK: fetching '$LIVE' timed out after ${FETCH_TIMEOUT}s — READ mode NOT run. Run by hand: git fetch origin $LIVE, then read kai-pm/HANDOFF_LOG.md there."
    else
      echo "HOOK: fetching '$LIVE' failed (git exit $frc) — READ mode NOT run. Run by hand: git fetch origin $LIVE, then read kai-pm/HANDOFF_LOG.md there."
    fi
  else
    SHA="$(git rev-parse FETCH_HEAD)"
    TMPD="$(mktemp -d "${TMPDIR:-/tmp}/kai-handoff-live.XXXXXX")"
    if git worktree add --detach --quiet "$TMPD/live" "$SHA" 2>&1; then
      echo "(live branch '$LIVE' at ${SHA:0:12}; 'branch' reads HEAD because the worktree is detached)"
      read_in "$TMPD/live"
      git worktree remove --force "$TMPD/live" 2>/dev/null
    else
      echo "HOOK: could not create a worktree for '$LIVE' — READ mode NOT run."
    fi
    rm -rf "$TMPD"
    echo "HOOK: WORK HERE LANDS ON '$CUR', NOT ON '$LIVE'. Commits from this session are not on the live line. To work on the live line, start the session on '$LIVE'."
  fi
fi

cat <<'EOF'
=== Before any work (CLAUDE.md, session start): report to the operator
    (1) the DIFFERS/UNMEASURED lines above, (2) every ⚠ UNBANKED ruling in
    the last entry of kai-pm/HANDOFF_LOG.md, (3) the next authorised step
    quoted with its source, (4) every HOOK: line. The repository wins for
    facts. WRITE-DUE: DUE means the previous session left commits that no
    entry covers. ===
EOF
exit 0
