#!/bin/bash
# kai-handoff WRITE reminder: the Stop and PreCompact hooks.
#
# READ is automatic (session-start.sh). WRITE needs judgement, so no hook
# can write it; what a hook CAN do is make an owed WRITE impossible to
# miss. `handoff.py due` decides, from Git alone, whether commits exist
# that no entry covers. `handoff.py hook <event>` then:
#   stop        adds Stop-hook feedback ONCE per HEAD; never inside its own
#               continuation (stop_hook_active), so it cannot loop
#   precompact  blocks a MANUAL /compact ONCE per HEAD (a second /compact
#               proceeds). An AUTO compaction is NEVER blocked. The hooks
#               docs: "If compaction was triggered to recover from a
#               context-limit error already returned by the API, the
#               underlying error surfaces and the current request fails."
#               A hook cannot tell that case from a proactive one, so it
#               never blocks auto. SessionStart:compact reports WRITE-DUE.
#
# Blind spot, stated: a ruling made only in conversation leaves no trace
# in Git. This hook cannot see it.
#
# Never breaks a session: absent tool, absent log, absent python3, a
# timeout or an internal error all exit 0. Only the deliberate
# PreCompact block exits 2. Writes nothing in the repository; its
# once-per-state marker lives under $TMPDIR/kai-handoff-hooks/.
set -u
EVENT="${1:-}"
ROOT="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
TOOL="$ROOT/.claude/skills/kai-handoff/handoff.py"

case "$EVENT" in stop|precompact) ;; *) exit 0 ;; esac
[ -f "$TOOL" ] && [ -f "$ROOT/kai-pm/HANDOFF_LOG.md" ] || exit 0
command -v python3 >/dev/null 2>&1 || exit 0
cd "$ROOT" || exit 0

timeout 20 python3 -B "$TOOL" hook "$EVENT"
rc=$?
[ "$rc" -eq 2 ] && exit 2
exit 0
