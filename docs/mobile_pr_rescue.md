# Mobile-Friendly PR Rescue (when GitHub says conflicts are too complex)

If GitHub web editor says **"conflicts are too complex to resolve in a web editor"**, use this flow from any desktop terminal.

## One-command rescue

```bash
scripts/rescue_pr32.sh origin main rescue/pr32 e6e2692
```

This will:
1. fetch latest `origin/main`,
2. create a clean branch,
3. replay the PR commit,
4. run static checks + `make merge-gate`,
5. push the new branch.

Then open PR: `rescue/pr32 -> main`.

## If cherry-pick still conflicts
The script auto-generates a fallback patch under `/tmp/rescue_pr32.patch` (name varies by branch).

Run:

```bash
git apply --3way /tmp/rescue_pr32.patch
```

Then run:

```bash
make merge-gate
git add -A
git commit -m "rescue: replay pr32 onto latest main"
git push -u origin rescue/pr32
```

## Why this works
This bypasses GitHub web conflict UI and rebuilds a fresh PR branch from latest `main`, minimizing conflict scope.
