#!/usr/bin/env bash
set -euo pipefail

# Rebuild a clean PR branch from latest main and replay the PR#32 commit.
# Usage:
#   scripts/rescue_pr32.sh <remote> <base-branch> <new-branch> <commit-sha>
# Example:
#   scripts/rescue_pr32.sh origin main rescue/pr32 e6e2692

REMOTE="${1:-origin}"
BASE_BRANCH="${2:-main}"
NEW_BRANCH="${3:-rescue/pr32}"
COMMIT_SHA="${4:-e6e2692}"

printf '[1/7] Fetching %s/%s\n' "$REMOTE" "$BASE_BRANCH"
git fetch "$REMOTE" "$BASE_BRANCH"

printf '[2/7] Creating clean branch %s from %s/%s\n' "$NEW_BRANCH" "$REMOTE" "$BASE_BRANCH"
git checkout -B "$NEW_BRANCH" "$REMOTE/$BASE_BRANCH"

printf '[3/7] Cherry-picking commit %s\n' "$COMMIT_SHA"
if ! git cherry-pick "$COMMIT_SHA"; then
  echo "Cherry-pick conflict detected. Aborting and generating patch fallback..."
  git cherry-pick --abort || true
  git format-patch -1 "$COMMIT_SHA" --stdout > "/tmp/${NEW_BRANCH//\//_}.patch"
  echo "Patch fallback written to /tmp/${NEW_BRANCH//\//_}.patch"
  echo "Try: git apply --3way /tmp/${NEW_BRANCH//\//_}.patch"
  exit 2
fi

printf '[4/7] Running quick static checks\n'
python -m py_compile tool-gate/app.py heartbeat/app.py scripts/ocr_receipt.py scripts/market_price_cache.py common/market_cache.py scripts/auto_rotate_ed25519.py scripts/kai_control.py common/self_emp_advisor.py

printf '[5/7] Running merge gate\n'
make merge-gate

printf '[6/7] Pushing %s to %s\n' "$NEW_BRANCH" "$REMOTE"
git push -u "$REMOTE" "$NEW_BRANCH"

printf '[7/7] Done. Open PR: %s -> %s\n' "$NEW_BRANCH" "$BASE_BRANCH"
