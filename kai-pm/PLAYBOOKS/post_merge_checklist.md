# Playbook: Post-Merge Checklist

Run this checklist after **every** PR merge to `main`.

1. Confirm CI passed on the merge commit.
2. Refresh `kai-pm/STATUS.md` (in-flight PRs, current focus, last-updated date).
3. If a decision was made, append to `kai-pm/DECISIONS.md` (D{n}, never edit prior entries).
4. If risk profile changed, edit `kai-pm/RISKS.md`.
5. If test count / LOC / milestone count changed, refresh `kai-pm/METRICS.md`.
6. Update `CHANGELOG.md` `[Unreleased]` if a user-visible change shipped.
7. Run the diff-vs-README ritual — `README.md` must not lie about current state.
8. Delete merged PR source branch unless it is `main` or a long-lived branch.
