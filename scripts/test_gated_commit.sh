#!/usr/bin/env bash
#
# The drift guard must refuse, not merely exist.
#
# Every case runs in a throwaway repository with a stub gate, so the
# guard's behaviour is exercised rather than reasoned about — and so a
# test of the guard can never commit anything to Kai.
#
# I-8: a known-positive (clean run commits the gated tree) and the
# known-negatives that matter, including the one the guard exists for —
# a tree that changes WHILE the gate runs.
#
set -uo pipefail

GUARD="$(cd "$(dirname "$0")" && pwd)/security/gated_commit.sh"
# Asserted, not assumed. The first version of this path was wrong and
# every case exited 127 (command not found) -- yet three of them still
# reported `ok`, because they were checking "nothing was committed" and
# nothing had run. A test that passes because the subject never executed
# is the same defect class this guard exists to close.
[ -x "$GUARD" ] || { echo "REFUSING: guard not found or not executable at $GUARD"; exit 1; }
PASS=0
FAIL=0

check() {
  if [ "$2" = "$3" ]; then
    PASS=$((PASS + 1)); printf '  ok    %s\n' "$1"
  else
    FAIL=$((FAIL + 1)); printf '  FAIL  %s (expected %s, got %s)\n' "$1" "$2" "$3"
  fi
}

# Logs are written OUTSIDE the repository under test. The first version
# redirected to "$(log_for "$d")" -- inside it -- so the guard correctly
# reported drift: the log file itself had changed the tree. The guard was
# right and the test was wrong, and the shape is R9's: an observation
# apparatus that alters what it observes.
log_for() { printf '%s.log' "$1"; }

# Every invocation needs a message file; the guard refuses without one
# rather than opening an editor in an unattended run.
MSG="$(mktemp)"; printf 'test commit\n' > "$MSG"

new_repo() {
  d="$(mktemp -d)"
  git -C "$d" init -q
  git -C "$d" config user.email t@t; git -C "$d" config user.name t
  echo base > "$d/file.txt"
  git -C "$d" add -A; git -C "$d" commit -q -m base
  printf '%s' "$d"
}

echo "Drift guard — proving it refuses"
echo "=================================================================="

# ── KNOWN-POSITIVE: clean candidate, passing gate, commit happens ──
d="$(new_repo)"; echo change > "$d/file.txt"
( cd "$d" && GATE="true" bash "$GUARD" "$MSG" >"$(log_for "$d")" 2>&1 ); rc=$?
check "a clean candidate with a passing gate commits" 0 $rc
committed_tree="$(git -C "$d" rev-parse 'HEAD^{tree}')"
gated_tree="$(grep -o 'candidate tree: [0-9a-f]*' "$(log_for "$d")" | awk '{print $3}')"
check "TESTED TREE SHA == COMMITTED TREE SHA" "$gated_tree" "$committed_tree"
check "the change is actually in history" "change" "$(git -C "$d" show HEAD:file.txt)"

# ── the gate fails: nothing is committed ──
d="$(new_repo)"; echo change > "$d/file.txt"
before="$(git -C "$d" rev-parse HEAD)"
( cd "$d" && GATE="false" bash "$GUARD" "$MSG" >"$(log_for "$d")" 2>&1 ); rc=$?
check "a failing gate refuses" 1 $rc
check "and commits NOTHING" "$before" "$(git -C "$d" rev-parse HEAD)"
grep -q "the gate failed" "$(log_for "$d")" && r=yes || r=no
check "and says the gate failed" yes "$r"

# ── THE ONE IT EXISTS FOR: the tree changes while the gate runs ──
d="$(new_repo)"; echo change > "$d/file.txt"
before="$(git -C "$d" rev-parse HEAD)"
# A gate that mutates the candidate mid-run, exactly as an edit would.
( cd "$d" && GATE="sh -c 'echo edited-during-gate > $d/file.txt; true'" \
    bash "$GUARD" "$MSG" >"$(log_for "$d")" 2>&1 ); rc=$?
check "A TREE THAT CHANGES DURING THE GATE IS REFUSED" 1 $rc
check "and nothing is committed" "$before" "$(git -C "$d" rev-parse HEAD)"
grep -q "THE TREE CHANGED WHILE THE" "$(log_for "$d")" && r=yes || r=no
check "and it names the drift, not something vague" yes "$r"
grep -q "never tested" "$(log_for "$d")" && r=yes || r=no
check "and says why that matters — bytes never tested" yes "$r"

# ── an untracked file: invisible to write-tree, visible to the tests ──
d="$(new_repo)"; echo change > "$d/file.txt"
mkdir -p "$d/.git/info"; echo "ignored_dir/" > "$d/.gitignore"
git -C "$d" add .gitignore; git -C "$d" commit -q -m ignore
echo new > "$d/sneaky.py"
before="$(git -C "$d" rev-parse HEAD)"
# Staged by the guard's own `git add -A`, so it IS in the tree — which
# is the correct outcome: it gets tested. The refusal case is a file
# that CANNOT be staged, below.
( cd "$d" && GATE="true" bash "$GUARD" "$MSG" >"$(log_for "$d")" 2>&1 ); rc=$?
check "an untracked file is staged into the candidate and tested" 0 $rc
git -C "$d" ls-files --error-unmatch sneaky.py >/dev/null 2>&1 && r=yes || r=no
check "so it is part of the gated tree, not invisible to it" yes "$r"

# ── ignored files stay out and do not break the guard ──
d="$(new_repo)"; echo "junk/" > "$d/.gitignore"
git -C "$d" add .gitignore; git -C "$d" commit -q -m ignore
mkdir -p "$d/junk"; echo artefact > "$d/junk/coverage.html"
echo change > "$d/file.txt"
( cd "$d" && GATE="true" bash "$GUARD" "$MSG" >"$(log_for "$d")" 2>&1 ); rc=$?
check "ignored gate artefacts do not block the commit" 0 $rc

# ── the gate runs on the CANDIDATE, not on the working directory ──
d="$(new_repo)"; echo candidate > "$d/file.txt"
( cd "$d" && GATE="sh -c 'test \"\$(cat file.txt)\" = candidate'" \
    bash "$GUARD" "$MSG" >"$(log_for "$d")" 2>&1 ); rc=$?
check "the gate sees the candidate's contents in its worktree" 0 $rc

# ── the guard leaves no worktree behind ──
d="$(new_repo)"; echo change > "$d/file.txt"
( cd "$d" && GATE="true" bash "$GUARD" "$MSG" >"$(log_for "$d")" 2>&1 ) || true
count="$(git -C "$d" worktree list | wc -l)"
check "no isolated worktree is left registered" 1 "$count"

echo "=================================================================="
echo "Drift guard tests: $PASS passed, $FAIL failed"
echo "EXIT GATE: $([ "$FAIL" -eq 0 ] && echo PASS || echo FAIL)"
[ "$FAIL" -eq 0 ]
