# Playbook: Dispatch a J-Series Jewel

Use this recipe when handing a single J-series jewel (for example J2 wake-word) to a coding agent.

## 1) Pre-flight checks

1. Confirm the jewel ID exists in `kai-pm/SEQUENCE.md` (or is explicitly tracked as DONE there).
2. Confirm all dependencies are marked DONE before dispatch.
3. Confirm CI on `main` is green before dispatching implementation work.

If any pre-flight check fails, stop and resolve that first.

## 2) Draft the problem statement

Include all of the following, explicitly:

- **Goal:** one clear outcome for this jewel only.
- **Scope limits:** what files/areas are in-bounds and out-of-bounds.
- **Acceptance criteria:** objective pass/fail checks.
- **File + test surface:** exact files expected to change and tests to run.

Keep the statement deterministic and parallel-safe.

## 3) Dispatch execution

- **Base branch:** `main` unless an integration branch is explicitly approved.
- **Agent:** assign one coding agent to one jewel.
- **Parallel-safety check:** verify no overlap with other active PRs before dispatch.

Suggested dispatch note:
- Jewel ID
- Dependency confirmation
- Parallel-safe confirmation
- Required validations

## 4) Post-merge updates

After merge to `main`:

1. Update `kai-pm/STATUS.md`.
2. Append to `kai-pm/DECISIONS.md` if an architectural decision was made (append-only).
3. Refresh `kai-pm/METRICS.md` test count when it changed.

Do not retro-edit historical decision entries.
