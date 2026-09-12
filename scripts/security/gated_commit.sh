#!/usr/bin/env bash
#
# TESTED TREE SHA == COMMITTED TREE SHA.
#
# The defect this closes
# ----------------------
#
#     GATED-STATE / COMMITTED-STATE DRIFT
#
# The ordinary sequence is `make prepush && git add -A && git commit`.
# The gate runs against the working tree at time T; `git add -A` collects
# whatever exists at time T+n. Nothing connects them. An edit made while
# the gate runs is committed having never been tested, and the record
# says GREEN. It is a time-of-check/time-of-use problem, the failure is
# silent, and it is permanent: nothing afterwards reveals the divergence.
#
# On 2026-08-11 the only thing preventing this was me noticing mid-run
# and choosing not to type. That is not a control.
#
# What this does
# --------------
#
#   1. stage the intended candidate
#   2. capture its immutable tree object      (git write-tree)
#   3. refuse if any non-ignored file is unstaged or untracked
#   4. run the gate in an ISOLATED WORKTREE built from that tree
#   5. capture the gate's own exit status, from the gate
#   6. recompute the tree; refuse if it changed
#   7. commit that exact tree
#   8. verify the commit's tree SHA equals the gated tree SHA
#
# Step 3 is not garnish. `git write-tree` describes the INDEX, and the
# tests read the FILESYSTEM — so an untracked file is invisible to the
# hash and fully visible to the tests. Without step 3 the invariant is
# defeatable by one untracked module.
#
# Step 4 matters for the same reason from the other side: gating the
# mutable working directory would let a concurrent edit change what runs
# after the hash was taken. The worktree is a private copy of the
# candidate, so nothing outside it can alter what was measured.
#
# R-VERDICT-INTEGRITY: the gate's status is captured with the
# conditional form, never after a pipe and never bare under `set -e`,
# so a failing gate arrives as a failing gate.
#
set -uo pipefail

REPO="$(git rev-parse --show-toplevel)"
cd "$REPO"

GATE="${GATE:-make prepush}"
MESSAGE_FILE="${1:-}"

fail() { printf '\nREFUSING: %s\n' "$1" >&2; exit 1; }

# ── 1. stage the candidate ──────────────────────────────────────────
git add -A

# ── 3. nothing may be readable by the tests that is not in the tree ──
#
# Checked BEFORE the hash so the message names the real problem. Ignored
# paths are permitted only because they are ignored — that is asserted
# below rather than assumed, since a gate artefact that is NOT ignored
# would silently change the tree and make this guard look broken.
untracked="$(git status --porcelain --untracked-files=all | grep -v '^[MADRC]' || true)"
[ -z "$untracked" ] || fail "files exist that the gate would read but the
tree does not describe:
$untracked
Stage them or ignore them. An untracked file is invisible to
git write-tree and fully visible to the tests."

# ── 2. the immutable candidate ──────────────────────────────────────
TREE_BEFORE="$(git write-tree)"
printf 'candidate tree: %s\n' "$TREE_BEFORE"

# ── 4. gate that exact tree, in a private copy ──────────────────────
CANDIDATE="$(git commit-tree "$TREE_BEFORE" -p HEAD -m 'gated candidate (not for history)')"
WORKTREE="$(mktemp -d)"
cleanup() { git worktree remove --force "$WORKTREE" >/dev/null 2>&1 || true; }
trap cleanup EXIT
git worktree add --detach "$WORKTREE" "$CANDIDATE" >/dev/null 2>&1 \
  || fail "could not create the isolated worktree"

# THE SANDBOX MUST RESEMBLE THE WORKING DIRECTORY IT STANDS IN FOR.
#
# `mktemp -d` creates 0700 root-owned -- the right default for secrets,
# and wrong for a source checkout. The real repository is 0755, and a
# gate that runs tests as an unprivileged user (test_container_proof_
# harness does, deliberately, because as root a 0444 file is still
# writable and its read-only assertion would be untestable) then fails
# for a reason that exists only inside the sandbox.
#
# Found by this guard's first real use: 12 assertions failed in the
# worktree and none on the real tree. A gate whose environment differs
# from the one that ships produces verdicts about the wrong environment.
# A checkout of committed content carries nothing secret, so this
# matches the real repository rather than relaxing anything.
chmod 755 "$WORKTREE" || fail "could not match the worktree's permissions
to the repository's"

printf 'gating %s in %s\n\n' "$CANDIDATE" "$WORKTREE"

# ── 5. the gate's own verdict, conditional so `-e` cannot eat it ────
if ( cd "$WORKTREE" && eval "$GATE" ) > "$WORKTREE/../gate.log" 2>&1; then
  gate_rc=0
else
  gate_rc=$?
fi
printf 'GATE EXIT CODE: %s  (producer=gate)\n' "$gate_rc"
tail -25 "$WORKTREE/../gate.log" || true

[ "$gate_rc" -eq 0 ] || fail "the gate failed (exit $gate_rc). Nothing is
committed. The full log is at $WORKTREE/../gate.log"

# ── 6. did the candidate change while the gate ran? ─────────────────
git add -A
TREE_AFTER="$(git write-tree)"
[ "$TREE_BEFORE" = "$TREE_AFTER" ] || fail "THE TREE CHANGED WHILE THE
GATE RAN.
  gated:     $TREE_BEFORE
  now:       $TREE_AFTER
Committing now would record a GREEN verdict against bytes that were
never tested. Re-run the gate."

# ── 7 & 8. commit, then prove the commit carries the gated tree ─────
[ -n "$MESSAGE_FILE" ] || fail "no commit message file given.
Usage: gated_commit.sh <message-file>
Refusing rather than opening an editor or inventing a message: this
runs unattended, and `git commit` with no message aborts with a
diagnosis about the message that reads like a gate failure."
[ -f "$MESSAGE_FILE" ] || fail "commit message file $MESSAGE_FILE does not exist"
git commit -q -F "$MESSAGE_FILE" || fail "git commit refused the candidate"

COMMITTED_TREE="$(git rev-parse 'HEAD^{tree}')"
[ "$COMMITTED_TREE" = "$TREE_BEFORE" ] || fail "the commit's tree
($COMMITTED_TREE) is not the tree that passed the gate ($TREE_BEFORE)."

printf '\nTESTED TREE SHA == COMMITTED TREE SHA  (%s)\n' "$TREE_BEFORE"
printf 'commit %s\n' "$(git rev-parse --short HEAD)"
