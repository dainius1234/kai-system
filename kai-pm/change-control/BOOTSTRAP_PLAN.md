# CAI v1.0 — bootstrap plan

Tranche `CAI-BOOT-001`. Branch `claude/cai-v1-bootstrap`, created from
exactly `main` = `194db0a0c13b4d5b322997fc1ceb33bdd21a77bc`
(tree `11712eb38e0cf3ffd5a934e755f40520dcb148de`). Nothing is taken from
the D379 branch (`0af5d320…`) or PR #122.

## Authority for this tranche

Dainius's explicit instruction, relayed in Kai's implementation order.
CAI does not exist yet, so it cannot authorise itself (order §4.3).

## Leg 1 — source, calibration, design files (THIS LEG)

1. contracts and schemas;
2. deterministic verifiers under `scripts/security/check_cai_*.py`;
3. hostile calibration A1–A9, S1–S7, H1–H4, V1–V5, D1–D8, N1–N4,
   executed against real Git repositories and real OpenPGP signatures;
4. `.github/workflows/cai-authority-verifier.yml` and **proposed**
   rulesets under `rulesets/` — not applied;
5. consumer inventory and evidence capture.

**Stop: return to Kai for IV&V.** No live ruleset. No admission tag. No
WA tag. No ACTIVE claim.

## Leg 2 — live bootstrap, only after Kai rules READY FOR BOOTSTRAP

1. Dainius provides the authority public key; its fingerprint is pinned.
2. Apply exactly the reviewed ruleset configuration. Any admin-only
   operation Orion cannot perform: stop at that exact operation and give
   Dainius the minimum manual action.
3. Capture the actual server state (rulesets, bypass actors, branch
   protection) and run real hostile probes against the server — no
   simulation substitutes for a server-side test.

## Activation exit gate

Order §26, items 1–15, each independently verified by Kai. Only then
may "CAI ACTIVE" be proposed to Dainius. No self-certification.

## Known integration dependency outside the authorised surface

`scripts/security/check_gate_registry.py --gate` discovers every
`scripts/security/*.py` containing `def main(` and every script a
workflow runs with an enforcing exit code, and fails invariant I-4 for
any that is not declared in `scripts/security/gate_registry.py`. That
file and the `Makefile` `policy-check` target are **not** on the
authorised surface. The exact entries CAI needs are in `REGISTRY_INTEGRATION.patch`
(98 lines, unapplied). Measured in a throwaway worktree: with it the
gate PASSES, 49 declared / 49 on disk, I-1..I-7 hold, and
`check_cai_authority` / `check_cai_scope` are reported as declared but
not yet enforced until leg 2. `git apply --check` passes on this
branch. Relocating or renaming the
verifiers to escape discovery would bypass an existing control and is
not done.
