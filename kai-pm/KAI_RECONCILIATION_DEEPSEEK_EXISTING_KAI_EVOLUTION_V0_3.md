# Kai Reconciliation — DeepSeek Existing-Kai Evolution v0.3 Review

> **STATUS: KAI-VERIFIED ARCHITECTURE REVIEW / RECONCILIATION INPUT — NOT IMPLEMENTATION AUTHORITY, NOT PROGRAMME EXECUTION AUTHORITY, NOT FINAL CANON.**
>
> DeepSeek had no repo access. Its findings are therefore review input until reconciled against the current repository. This file records that reconciliation.

## Standing role clarification — DeepSeek is Kai's analysis instrument, not a project authority

This is not a temporary limitation caused only by lack of GitHub access. It is the intended operating model.

DeepSeek is used by Kai as an **external specialist analysis instrument** for:

- adversarial review;
- alternative hypotheses;
- architecture attack;
- coding/technical second opinion;
- identifying hidden failure modes;
- proposing discriminating tests;
- challenging Kai's assumptions.

DeepSeek does **not** independently own or determine:

- Dainius's philosophy or primary mission;
- Kai's identity/lineage intent;
- the accumulated architectural history;
- current repository truth;
- current programme order;
- authority/freeze state;
- product capability retention;
- final architecture;
- implementation authority.

DeepSeek output is therefore never accepted "in full" merely because it is technically persuasive. Kai is responsible for using it as raw analytical input, reconciling it against the actual repository, prior decisions, project philosophy and Dainius's intended destination, and rejecting or modifying anything that does not fit.

The durable workflow is:

`KAI DEFINES THE QUESTION + SUPPLIES QUALIFIED CONTEXT`
→ `DEEPSEEK ATTACKS / CHALLENGES / PROPOSES`
→ `KAI CHECKS REPO + HISTORY + PHILOSOPHY + ARCHITECTURE`
→ `KAI ACCEPTS / MODIFIES / REJECTS EACH POINT`
→ `ORION MAY VERIFY FEASIBILITY OR EXECUTE APPROVED WORK`
→ `DAINIUS RETAINS FINAL PROJECT AUTHORITY`.

DeepSeek is therefore analogous to a specialist analytical subsystem available to Kai, **not a fourth architect sitting beside Kai and Dainius**.

The standing law remains:

> **MULTIPLE MINDS — ONE EVIDENCE STANDARD — ONE ARCHITECTURAL RECONCILIATION RESPONSIBILITY.**

Kai owns that reconciliation responsibility.

## Executive result

DeepSeek's overall direction is useful and substantially better than a blank-sheet review, but it must **not** be adopted verbatim.

Current reconciliation:

- 9 findings materially supported;
- 5 supported but require correction/narrowing;
- 1 is primarily a phasing recommendation rather than a current defect;
- Kai found a stronger final-hand authority defect that DeepSeek identified only indirectly.

The most important correction is:

> **`KAI_AUTONOMY_ENFORCE=true` is NOT the universal prerequisite for every mutating actuator. Autonomy and execution authority are different controls. The real blocker is that the one-time `ActionCapability` is consumed centrally by `ActuatorRegistry`, while the downstream side-effecting service receives parameters plus service authentication/signature, not the exact one-time capability. Final-hand enforcement is therefore not yet literally at the hand.**

Manual operator-approved actions must be able to execute without an autonomy grant; autonomous initiation must additionally satisfy scoped autonomy.

---

## Finding reconciliation

| DeepSeek finding | Kai classification | Reconciliation |
|---|---|---|
| F-01 machine-generated E0 census | **SUPPORTED GOAL / IMPLEMENTATION MODIFIED** | Exact machine census is required, but do not invent another census system. Reuse/extend House/Census, existing security reports, compose/gate scanners and current repo instrumentation into the component/dependency/authority map. |
| F-02 autonomy enforcement hard-gates all mutating actuator ACTIVE | **PARTIALLY SUPPORTED — PROPOSED FIX REJECTED/REPLACED** | The final-hand blocker is real. But `KAI_AUTONOMY_ENFORCE` governs autonomous initiation, not all operator-approved execution. Correct hard gate: final-hand exact capability + workload identity + runtime legacy-bypass denial. Autonomous invocation additionally requires scoped autonomy grant/enforcement. |
| F-03 Cortex silent World-State fallback | **SUPPORTED MAJOR** | `resolve_cortex_state()` silently returns polled state when selected World State is empty. Cold-start compatibility is useful but steady-state fallback can mask a failed canonical path. Need explicit COLD_START/DEGRADED/UNKNOWN semantics and alerting; no silent steady-state dual truth. |
| F-04 shared-token→signed identity downgrade | **PARTIALLY SUPPORTED** | Remaining transition-window shared-token acceptance is real. However DeepSeek proposed controls already implemented: signed requests bind destination/method/path/body, timestamp and nonce; nonce cache survives restart; bad signature never falls back; grant-gated endpoints require verified identity. Do not reimplement these. Migrate remaining class-B routes to verified identity and prove coverage. |
| F-05 source-only legacy-path proof | **SUPPORTED, STRONGER THAN STATED** | `legacy_verification.py` can call an authenticated route “closed”; for many routes this means shared-token auth, not one-time final-hand capability. Existing live mutating verification proves routes are reachable, not that old direct bypasses are denied. Need runtime negative proof and capability-at-hand enforcement. |
| F-06 minimal/full/sovereign reconciliation | **SUPPORTED MAJOR** | Current compose populations differ. One canonical component/profile model is required. Exact rendering technique (generated YAML vs base+overlays) remains an implementation choice. |
| F-07 memu-core authoritative hot memory | **PARTIALLY SUPPORTED / CURRENT OWNERSHIP UNVERIFIED** | No new parallel Memory service is preferred. But exact authoritative/derived role assignment must wait for reader/writer/state-owner census. Treat memu-core-authoritative mapping as hypothesis until proven. |
| F-08 proactivity in agentic + World State | **PARTIALLY SUPPORTED / HOME NOT FROZEN** | Consolidating current detectors under Goal/Watch/Attention semantics is correct. Agentic + World State is plausible but must be earned by mapping monitor-service/Cortex/calendar/Supervisor/current persistence. No new proactivity service by default. |
| F-09 reduce/postpone Runtime Manager | **SUPPORTED AS PHASING REFINEMENT, NOT REMOVAL OF TARGET RESPONSIBILITY** | Do not create a wrapper service now. Add model identity/digest/resource/qualification data around current runtime first. Keep Model Runtime Manager as future responsibility when multi-runtime/resource admission becomes real. |
| F-10 durable authority behind Tool Gate | **SUPPORTED MAJOR** | `ApprovalGate`, `CapabilityBridge` and `AutonomyAuthority` currently hold records/nonces/grants in process-local collections. Existing Tool Gate is the control point and already has persistent ledger/idempotency work. Evolve durable authority behind compatible APIs rather than create a second Authority service. |
| F-11 Postgres workflow/outbox first | **SUPPORTED DIRECTION** | `ActionWorkflow` contract exists but the UH vertical slice constructs workflow state in memory. Existing Postgres makes a transactional workflow/outbox/fencing implementation the first justified option. Do not turn “no Temporal/NATS/Kafka now” into an eternal technology ban; re-evaluate only if measured scale/complexity earns it. |
| F-12 Doctor/Supervisor/resilience separation | **SUPPORTED** | Keep House Doctor=diagnosis, Supervisor=recovery/health, Doctor teammate=interactive cognitive specialist, verifier=independent outcome truth. Repairs go through normal authority/workflow. Physical merges later require failure-domain/independence evidence. |
| F-13 feature flags become evidence-bound cutovers | **SUPPORTED OBJECTIVE / MECHANISM MODIFIED** | Keep flags simple path selectors. Put shadow/soak/canary/active/retire evidence in a migration/release record checked by preflight/CI; do not overload environment flags into another authority database. |
| F-14 manual autonomy bootstrap/no self-grant | **SUPPORTED MAJOR WITH SCOPE NUANCE** | Current grants are in-memory; `granted_by` is a string, not verified operator authority. A1/A2 do not require human confirmation. Initial bootstrap and any widening of autonomy must be authenticated/operator-governed. Future renewals inside an explicitly delegated envelope may be separately qualified; do not conflate per-action approval with autonomy grant. |
| F-15 egress target policy | **SUPPORTED MAJOR** | Browser navigate accepts arbitrary URLs and egress network placement is reachability, not target authority. Add target/egress constraints to policy + exact capability and enforce again at final hand. No separate egress service by default. |

---

## New Kai finding KAI-REV-016 — Final-hand capability is not yet at the actual hand

**Severity:** BLOCKER for claiming UH-INV-06/final-hand cutover complete.

### Evidence

`common/actuator_registry/registry.py` requires a consumed `ActionCapability` before dispatch.

But `common/actuator_registry/mutating_handlers.py` then calls the downstream real service with:

- action parameters;
- signed workload-identity headers where available;
- shared bearer token fallback during migration.

It does **not** pass the one-time `ActionCapability` to the downstream side-effecting service for atomic validation/consumption there.

Many downstream endpoints, e.g. browser `/click` and `/type`, currently enforce service membership/authentication rather than the exact capability.

### Consequence

The current path is:

`Capability → central Registry consumes → HTTP handler → service auth/identity → side effect`.

The target invariant requires:

`Capability → transport-bound execution credential → actual actuator validates exact audience/action/parameters/expiry/single-use → atomic consume → side effect`.

A service able to call the downstream endpoint directly can bypass the central consumed capability if the endpoint only requires membership/static identity grant.

### Correct fix direction

Do not gate every mutating action on `KAI_AUTONOMY_ENFORCE`.

Instead:

1. propagate an exact execution capability or cryptographically derived one-use execution credential to the downstream actuator;
2. bind actuator audience, method/path, parameters/body digest, proposal/approval/workflow, expiry and nonce/consumption ID;
3. downstream service validates and atomically consumes it before side effect;
4. workload identity proves **who is presenting the capability**, not permission by itself;
5. autonomous requests additionally require a valid scoped autonomy grant;
6. operator-approved requests do not need an autonomy grant merely because they are mutating.

---

## New Kai finding KAI-REV-017 — Current legacy-closure definition can close the wrong property

**Severity:** BLOCKER for high-risk actuator retirement claims.

`common/actuator_registry/legacy_verification.py` deliberately treats several routes as “legacy closed” when they require service authentication.

That was a real improvement over unauthenticated direct routes, but it proves **membership protection**, not the stronger final-hand capability invariant.

For a shared-token route, any service holding the token can still directly call the route. Therefore:

> `AUTHENTICATED DIRECT PATH != CAPABILITY-GATED FINAL HAND != LEGACY AUTHORITY PATH DEAD`.

Required closure evidence must include runtime negative bypass tests and exact capability enforcement at the actuator.

---

## New Kai finding KAI-REV-018 — Current autonomy preflight cannot prove persistent runtime grants

**Severity:** MAJOR.

`scripts/preflight_deploy.py` checks `KAI_AUTONOMY_ENFORCE=true` by constructing a new `AutonomyAuthority` and asking that new object for active grants.

`AutonomyAuthority` currently stores grants in an in-memory dictionary.

Therefore the current preflight does not inspect durable runtime grant state; persistent grant storage/inspection is a prerequisite to meaningful enforcement readiness.

---

## Additional test/instrumentation note

`scripts/verify_live_mutating.py` proves that selected handlers can reach running routes, including authentication/parameter plumbing. It explicitly does **not** prove retired legacy routes are denied. Several high-consequence actions are skipped rather than passed. This is correct reporting, but its evidence claim must not be widened into final-hand/legacy-retirement proof.

---

## Corrected architecture cutover laws

### Authority separation

`OPERATOR/POLICY APPROVAL` and `SCOPED AUTONOMY` are separate dimensions.

- Manual consequential action: exact proposal/policy/approval/capability/final-hand validation.
- Autonomous consequential action: all of the above **plus** valid scoped autonomy delegation.

### Identity separation

`MEMBERSHIP != IDENTITY != AUTHORITY != ONE-TIME EXECUTION CAPABILITY`.

- shared token = membership;
- signed key = workload identity;
- static operation grant = role/scope authorisation;
- `ActionCapability` = exact one-time operation authority;
- autonomy grant = permission for Kai to initiate within a bounded envelope.

None substitutes for another.

### Legacy retirement

A legacy path is not retired merely because:

- a flag says disabled;
- source scan sees authentication;
- a new path works;
- an endpoint is reachable;
- a caller is signed.

Retirement means the old weaker authority path is **unusable at runtime** and cannot be restored by ordinary rollback without an explicit governed migration reversal.

---

## Revised v0.3 changes to carry into the next candidate

Do **not** edit runtime now. Future v0.4/master-canon candidate should incorporate:

1. E0 = extend existing House/Census tooling into exact machine component/dependency/authority inventory.
2. E2 = fix explicit World-State degradation before retiring legacy polling.
3. Add **Final-Hand Capability Propagation/Consumption** as a named mandatory shim/gate.
4. Replace F-02's universal autonomy-enforce gate with separate manual-vs-autonomous authority lanes.
5. Add runtime negative legacy-bypass proof.
6. Finish signed-identity migration without reimplementing timestamp/nonce/replay features already present.
7. Durable authority/grant/approval/capability state behind existing Tool Gate-compatible APIs.
8. Postgres-backed durable workflow/outbox around existing actuator registry as first implementation candidate.
9. Evidence-bound migration/release record controlling when simple feature flags may advance.
10. Add target/egress policy to exact capabilities for network-capable hands.
11. Keep memu/proactivity/model-runtime target homes provisional until E0 current ownership map proves them.
12. Preserve all existing product capability families; simplify boundaries only after evidence.

---

## DeepSeek review disposition

**USE:** architecture attack, gap discovery, migration-order refinement, alternative hypotheses and discriminating-test generation.

**DO NOT USE AS AUTHORITY:** repo-current claims, exact component ownership, project philosophy, Dainius's vision, existing security-feature absence, service merge decisions, final architecture, final programme order or implementation authority.

DeepSeek recommendations are never promoted directly into the plan. They are inputs to Kai's analysis.

Every recommendation remains subject to:

`DeepSeek analysis input → Kai repo/history/philosophy reconciliation → Orion exact feasibility/evidence when useful → Dainius final authority`.
