# D345 — Kingsman Candidate Architecture v0.1 Review Checkpoint

> **STATUS: GOVERNANCE / DESIGN CHECKPOINT — CANDIDATE RETURNED FOR EXTERNAL ADVERSARIAL REVIEW. NOT FINAL CANON. NOT IMPLEMENTATION AUTHORITY.**
>
> This file is a durable checkpoint because the canonical `kai-pm/DECISIONS.md` remains unsafe to append through the current large-file accessor. D344/D344A and D345 must later be appended byte-safely to the canonical ledger without reconstructing historical content.

---

## 1. What was done

Following Dainius's direction to review all major Kai parts again and produce an actual professional-grade architecture for DeepSeek review, Kai:

1. re-read the rooted mission/identity/lineage architecture;
2. inspected current architecture/product plans;
3. inspected current implementation seeds including contracts, perception spine, event journal, world state, service identity, capability bridge, autonomy authority, Supervisor, House Doctor, backup service and current deployment topology;
4. researched current external reference standards/practices for workload identity, event envelopes, provenance, telemetry, policy separation, durable workflows, transactional outbox, software attestations and Strix Halo/ROCm support;
5. produced one integrated architecture candidate;
6. produced an exact DeepSeek adversarial-review packet.

No frozen experiment/runtime code was changed.

---

## 2. Candidate architecture

Primary subject:

`kai-pm/KINGSMAN_CANDIDATE_ARCHITECTURE_V0_1_DEEPSEEK_REVIEW.md`

Creation commit:

`905130f7210c203e3ce287ea896748d94ed5d571`

Review handoff:

`kai-pm/DEEPSEEK_REVIEW_PACKET_KINGSMAN_ARCHITECTURE_V0_1.md`

Creation commit:

`56fbcb929baf5a5523d139a716dd908c770b1b99`

---

## 3. Main architectural recommendation

Candidate architecture style:

> **MODULAR ORGANISM + EARNED PROCESS/SERVICE BOUNDARIES + DURABLE SHARED TRUTH + ISOLATED AUTHORITY + ISOLATED HANDS.**

Reject both:

- one giant shared-fate monolith;
- service-per-idea/microservice soup.

A separate process/service must be justified by fault isolation, trust/security boundary, hardware/resource requirement, independent deployment lifecycle, durable workflow semantics, provider boundary, scaling or privileged OS/device access.

---

## 4. Proposed first-production physical domains

1. Operator / Edge Gateway
2. Kai Core
3. Authority Service
4. Model Compute Plane
5. Memory / Knowledge Plane
6. Execution Zone
7. Assurance / Health / Recovery Plane
8. Durable Data / Continuity Plane

Logical organs remain finer-grained, but not every logical organ becomes a container.

---

## 5. Proposed main runtime flow

`OBSERVE`
→ `VALIDATE / PROVENANCE`
→ `DURABLE EVENT`
→ `EVIDENCE QUALIFICATION`
→ `WORLD STATE`
→ `MEMORY + GOALS / OBLIGATIONS`
→ `ATTENTION / TASK FRAME`
→ `COGNITIVE WORKSPACE / SPECIALISTS`
→ `PROPOSAL`
→ `POLICY / AUTHORITY`
→ `HUMAN APPROVAL or SCOPED AUTONOMY where valid`
→ `SINGLE-USE CAPABILITY`
→ `DURABLE WORKFLOW`
→ `NARROW ACTUATOR / EGRESS CONTROL`
→ `RECEIPT`
→ `INDEPENDENT OUTCOME VERIFICATION`
→ `WORLD STATE / VERIFIED LEARNING / DIAGNOSIS`

No model or specialist receives a parallel action-authority path.

---

## 6. Major current strengths identified for reuse

- typed contract system;
- perception ingress validation/bounding;
- crash-conscious event journal concept;
- immutable/scoped world-state/conflict semantics;
- proposal workspace/Unified Hunter concepts;
- audience-bound capability concept;
- scoped/expiring/revocable/evidence-earned autonomy direction;
- per-service Ed25519 principal derivation direction;
- actuator registry;
- resilience primitives;
- House Doctor diagnosis concept;
- Supervisor health/recovery concept;
- existing real backup functions;
- PostgreSQL/Redis/network-segmented deployment foundations;
- memory/graph and financial-awareness organs.

Candidate explicitly recommends **reworking/re-homing these rather than replacing everything**.

---

## 7. Major missing / under-specified organs identified

### P0 architecture-critical

- Goal / Obligation / Watch / Attention subsystem;
- durable authority state;
- Model Runtime Manager;
- Structure / Dependency Graph;
- production durable workflow engine + durable timers;
- unified telemetry plane;
- lineage / restore identity mechanism;
- egress / target-control boundary;
- durable audit anchoring;
- schema / contract registry + compatibility policy;
- protected operator approval surface;
- data classification/key lifecycle enforcement across stores.

### P1 long-horizon maturity

- dependency/provider EOL and migration registry;
- automated restore drills;
- Financial Sustainability / Runway plane;
- succession state machine + legal/trust binding;
- machine-readable component maturity/capability registry;
- release/attestation registry.

### Explicitly NOT required now

- multi-node HA;
- full SPIRE deployment;
- NATS/JetStream backbone;
- Temporal deployment;
- autonomous financial execution;
- automatic succession;
- NPU always-on runtime before exact qualification.

These are options, not architectural prerequisites.

---

## 8. Main technology recommendations — candidates, not mandates

- **PostgreSQL + transactional outbox** as first authoritative event/state backbone.
- encrypted content-addressed object store for large raw evidence/data.
- vector/graph stores as reconstructable projections where feasible.
- Redis demoted to cache/ephemeral coordination, not sole authority/audit.
- CloudEvents envelope semantics as reference for events.
- W3C PROV semantics as reference for provenance.
- OpenTelemetry as telemetry standard.
- existing Ed25519 workload identity for near-term, with a provider interface and future SPIFFE/mTLS evaluation.
- OPA evaluated behind a Policy Decision Port, not automatically adopted.
- Postgres-backed workflow engine first; Temporal retained as later candidate if complexity earns it.
- in-toto/SLSA-style subject-bound release attestations.
- Strix Halo treated as current body generation behind Model Runtime Manager; GPU/CPU primary, NPU role only after exact measurement/qualification.

---

## 9. Important architectural separations

- Kai ≠ model/framework/service/device.
- Memory ≠ current fact.
- Telemetry ≠ qualified Evidence Plane truth.
- Evidence ≠ action authority.
- Identity ≠ membership ≠ authority.
- Model reasoning ≠ approval.
- Actuator receipt ≠ verified outcome.
- Diagnosis ≠ recovery authority.
- Self-sufficiency ≠ uncontrolled self-preservation.
- Temporary operator absence ≠ succession.

---

## 10. DeepSeek review requirement

DeepSeek must classify findings as:

- BLOCKER
- MAJOR
- MINOR
- QUESTION

and return, among other things:

- overall verdict;
- top 10 changes before freeze;
- missing organs;
- recommended logical/physical/data/model/resilience layout;
- simplify-by-30% version;
- top five cascading failures;
- experiments/spikes needed before freeze;
- strongest parts and hidden assumptions.

DeepSeek has no repository authority. Repo-dependent claims from its review remain `REVIEW INPUT / HYPOTHESIS` until independently verified.

---

## 11. What happens after DeepSeek returns

1. Kai classifies each DeepSeek point:
   - `SUPPORTED`
   - `PARTIALLY_SUPPORTED`
   - `CONFLICTS_WITH_REPO`
   - `UNVERIFIED`
   - `REJECTED`
2. Verify every repo-dependent premise.
3. For real disagreements, define cheapest discriminating test/spike.
4. Produce Candidate v0.2.
5. Orion maps v0.2 onto actual source/services/contracts and estimates migration/qualification work.
6. Dainius reviews the reconciled visual/product architecture.
7. Only after material issues are resolved: create/freeze exact `KINGSMAN_MASTER_CANON_v1`.
8. Execution still follows the latest valid D-numbered programme sequence.

---

## 12. Programme authority — unchanged

This checkpoint does **not** authorise:

- H2 v1.1 execution;
- modification of frozen House/Census evidence;
- KAI-GATE-048 scope expansion;
- Item 8 execution beyond existing authority;
- A-4 provenance execution;
- FUTURE A4 implementation;
- architecture implementation/refactor;
- succession implementation;
- autonomous financial execution;
- uncontrolled self-modification/self-preservation.

**ITEM 8 BEFORE A4 remains standing.**

---

## 13. THREAD RECOVERY BLOCK

**REPORTING COMMIT:** commit that creates this D345 file; recover from repository history.

**ARCHITECTURE SUBJECT:** `KINGSMAN_CANDIDATE_ARCHITECTURE_V0_1_DEEPSEEK_REVIEW.md` at creation commit `905130f7210c203e3ce287ea896748d94ed5d571`.

**EXTERNAL REVIEW PACKET:** `DEEPSEEK_REVIEW_PACKET_KINGSMAN_ARCHITECTURE_V0_1.md` at `56fbcb929baf5a5523d139a716dd908c770b1b99`.

**CURRENT WORKSTREAM:** architecture/canon candidate review only; programme execution remains under latest canonical D authority.

**LAST PROVEN STATE:** candidate and review packet banked; no runtime/experiment claim added.

**AUTHORISED NEXT ARCHITECTURE ACTION:** obtain DeepSeek review, verify/reconcile, then Orion feasibility map and Dainius review before canon freeze.

**EXPLICITLY NOT AUTHORISED:** implementation of candidate architecture or change to frozen programme sequence.

**OPEN:** DeepSeek review; Kai reconciliation; any discriminating spikes; Orion feasibility map; Candidate v0.2; Dainius approval; master-canon freeze; safe append of D344/D344A/D345 to canonical `DECISIONS.md` once exact ledger bytes can be retrieved.

**OPERATOR VISIBILITY:** architecture candidate now has explicit Mermaid diagrams and gap matrix. Full mission-control UI/README remains future Phase-2 work and therefore operator visibility remains **INCOMPLETE but materially improved**.
