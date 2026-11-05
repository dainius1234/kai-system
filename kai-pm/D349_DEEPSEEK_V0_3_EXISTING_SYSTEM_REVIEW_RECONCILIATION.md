# D349 — DeepSeek v0.3 Existing-System Review Reconciliation

> **STATUS: GOVERNANCE / ARCHITECTURE REVIEW CHECKPOINT — NOT IMPLEMENTATION AUTHORITY, NOT PROGRAMME EXECUTION AUTHORITY, NOT FINAL CANON.**

## Trigger

DeepSeek returned the requested adversarial review of `KINGSMAN_EXISTING_KAI_EVOLUTION_MASTER_PLAN_V0_3.md` and gave `APPROVE WITH CHANGES`.

The review successfully attacked the existing-system migration rather than producing a blank-sheet architecture.

Kai then independently checked the material claims against the current repository before accepting them.

## Reconciliation artifact

`KAI_RECONCILIATION_DEEPSEEK_EXISTING_KAI_EVOLUTION_V0_3.md`

Creation commit:

`89fdd1b820e245550fe2574d26ef17e6651f4dec`

## Main result

DeepSeek review is useful but is **not accepted verbatim**.

Kai classification:

- 9 findings materially supported;
- 5 supported but require narrowing/correction;
- 1 mainly a phasing recommendation rather than a current defect.

## Most important repo-backed correction

DeepSeek correctly sensed a final-hand activation problem, but proposed the wrong universal gate.

`KAI_AUTONOMY_ENFORCE=true` must **not** become a prerequisite for every operator-approved mutating action.

Autonomy and execution authority are distinct:

- manually approved consequential action needs exact proposal/policy/approval/capability and final-hand validation;
- autonomous consequential action needs the same path **plus** valid scoped autonomy delegation.

## New repo-backed blocker KAI-REV-016

Current `ActuatorRegistry` requires/consumes an `ActionCapability` before dispatch.

But current mutating handlers call the downstream side-effecting service with parameters plus service auth/workload signature. The exact one-time `ActionCapability` is not propagated to and atomically consumed by the actual service performing the effect.

Therefore current implementation is centrally capability-gated but not yet literally final-hand capability enforcement.

Required target:

`exact capability / derived one-use execution credential → actual actuator validates audience+operation+parameters+expiry+single-use → atomic consume → side effect`.

Workload identity proves who presents the authority; it is not the authority itself.

## New repo-backed blocker KAI-REV-017

Current legacy-path source verifier can consider several routes “closed” once service authentication is present.

This is insufficient for the final architecture because:

`AUTHENTICATED DIRECT PATH != FINAL-HAND CAPABILITY GATE`.

A shared-token holder may still bypass the central registry by calling an authenticated route directly.

Runtime negative bypass proof plus final-hand capability validation is required before high-risk legacy retirement can be claimed.

## New repo-backed major KAI-REV-018

Current autonomy preflight creates a fresh in-memory `AutonomyAuthority` and inspects that object for active grants.

Grant state is process-local today.

Durable grant/approval/capability state and real runtime-state inspection are required before `KAI_AUTONOMY_ENFORCE` readiness is meaningful.

## DeepSeek security correction

F-04 overstated missing signed-request controls.

Current repo already contains:

- per-workload Ed25519 identity;
- principal derived from verifying key;
- signed destination/method/path/body hash;
- timestamp skew enforcement;
- nonce replay protection with persisted cache/restart handling;
- revoked-key handling;
- no fallback after a bad signature;
- signed identity mandatory on grant-gated endpoints.

Remaining problem is the deliberate transition window where some class-B routes can still accept shared membership token as unverified identity.

Do not reimplement controls already present.

## Carry-forward architecture corrections

Next candidate/reconciliation must include:

1. E0 reuses/extends existing House/Census and repo scanners rather than inventing a new census system.
2. Cortex World-State steady-state fallback becomes explicit DEGRADED/UNKNOWN rather than silent legacy truth restoration.
3. final-hand capability propagation/atomic consumption becomes an explicit named migration gate.
4. manual and autonomous authority lanes remain separate.
5. legacy retirement requires runtime negative bypass proof.
6. signed-identity migration closes remaining class-B shared-token paths without duplicating existing replay/nonce/timestamp machinery.
7. durable authority state evolves behind existing Tool Gate-compatible interfaces.
8. Postgres workflow/outbox is the first justified durable workflow candidate around existing ActuatorRegistry; no permanent technology ban is created.
9. flags remain simple selectors; evidence-bound migration/release records govern promotion.
10. target/egress constraints become part of policy/exact capability for network-capable hands.
11. memu/proactivity/model-runtime physical homes remain provisional until exact current reader/writer/dependency map.
12. all existing capability families remain protected from silent deletion.

## Programme authority unchanged

D349 authorises no runtime implementation, refactor, service merge/delete, H2 v1.1, 048 change, Item8 execution, A-4/Future-A4 execution, succession, autonomous finance or uncontrolled self-modification.

**ITEM 8 BEFORE A4** remains standing.

## THREAD RECOVERY BLOCK

**CURRENT PRIMARY ARCHITECTURE SUBJECT:** Existing-Kai Evolution v0.3 plus D349 reconciliation; v0.4/final canon not yet authored/frozen.

**CURRENT KAI RECONCILIATION:** `KAI_RECONCILIATION_DEEPSEEK_EXISTING_KAI_EVOLUTION_V0_3.md` at `89fdd1b820e245550fe2574d26ef17e6651f4dec`.

**NEW BLOCKERS:** KAI-REV-016 final-hand capability propagation/consumption; KAI-REV-017 authenticated-direct-path closure is insufficient.

**NEW MAJOR:** KAI-REV-018 durable autonomy state/preflight subject binding.

**NEXT ARCHITECTURE ACTION:** incorporate supported/corrected findings into a v0.4 change set only after current exact component/authority map and no-loss review; do not implement from DeepSeek prose directly.
