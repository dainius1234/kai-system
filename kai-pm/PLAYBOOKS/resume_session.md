# Resume Session Playbook

When the operator says **"resume"**, re-hydrate in this order:

1. Read `kai-pm/SESSION_BOOTSTRAP.md` (single-source fast context)
2. Read `kai-pm/STATUS.md` (live state and next actions)
3. Read `kai-pm/SEQUENCE.md` (locked dependencies and blockers)
4. Read tail of `kai-pm/DECISIONS.md` (latest policy decisions)
5. Check open PRs and open issues on GitHub
6. Confirm whether any blocked item has changed unlock status
7. Respond with: current state, immediate next action, and any blockers

Do not start implementation work until this sequence is complete.
