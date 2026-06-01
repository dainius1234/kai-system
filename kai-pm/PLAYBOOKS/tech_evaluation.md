# Playbook: Tech Evaluation (Model / Library / Tool)

Use this before adding a new model, library, or external tool.

## Step 1 — Sovereignty check

- Is it offline-capable in real operation?
- Does it phone home or require cloud callbacks?
- Is the license compatible with this repository's usage model?

## Step 2 — Footprint check

- VRAM / RAM impact under realistic load.
- Dependency weight and install complexity.
- Transitive dependency risk and maintenance burden.

## Step 3 — Drop-in test

Run an A/B test against the current solution on a representative task:

- quality delta
- latency delta
- reliability delta
- operational complexity delta

If results are mixed or unverified, mark them `(unverified)`.

## Step 4 — Rollback plan

Define how to revert in **< 1 commit** if regression appears:

- exact files to revert
- feature flag fallback path (if present)
- validation to confirm rollback success

## Step 5 — Document verdict

Append a verdict entry to `kai-pm/TECH_WATCH.md` with date and rationale:

- `ADOPT`
- `TRIAL`
- `HOLD`
- `REJECT`
