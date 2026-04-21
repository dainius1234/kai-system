# Dispatch Jewel Playbook

## Purpose
Dispatch one coding agent to deliver a single jewel safely and reviewably.

## Steps
1. Confirm jewel scope and locked sequence step from `kai-pm/SEQUENCE.md`.
2. Confirm prerequisites and dependency unlocks are met.
3. Open/identify target branch (base: `main` unless an approved integration branch is specified).
4. Send the agent a complete problem statement using the template below.
5. Require: minimal surgical changes, tests updated where relevant, docs synced.
6. After delivery, verify changed file scope matches requested jewel.
7. Validate tests/checks that cover touched areas.
8. Ensure PR description includes sequence step + jewel + risk note.

## Problem statement template
- **Repository:** `dainius1234/kai-system`
- **Base branch:** `main`
- **Jewel:** `Jx`
- **Sequence step:** `Sxx`
- **Goal:** (single-sentence functional objective)
- **Must change:** (explicit files/areas)
- **Must not change:** (out-of-scope areas)
- **Validation required:** (test targets + manual checks)
- **Docs required:** (`kai-pm/STATUS.md`, `CHANGELOG.md`, README if interface changed)

## Verify after dispatch
- PR file list is jewel-scoped and minimal.
- Sequence row and jewel are declared in PR body.
- Status and changelog are updated (or explicitly N/A with rationale).
