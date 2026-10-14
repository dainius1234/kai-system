# Kingsman Engineering Architecture Visual Standard

> **STATUS: STANDING DESIGN / OPERATOR-VISIBILITY EXTENSION — NOT IMPLEMENTATION AUTHORITY.**
>
> This standard exists because a generated architecture infographic was reviewed against the actual repository and found materially below the engineering maturity of the system. It was visually polished but collapsed real services, contracts, trust boundaries, authority stages, network segmentation, evidence semantics, current/target distinctions and failure domains into generic boxes. That output is **REJECTED AS AN AUTHORITATIVE ARCHITECTURE VIEW**.

## 1. Core rule

> **AN ARCHITECTURE VISUAL IS AN ENGINEERING MODEL, NOT DECORATION.**

A box asserts that a component/boundary exists or is planned. An arrow asserts a relationship. A colour/status asserts currentness or maturity. Therefore every consequential visual statement must be recoverable to repository evidence or explicitly labelled as target/planned/unknown.

## 2. Authoritative visual source form

The authoritative architecture must be stored as deterministic text source that can be diffed, reviewed, regenerated and bound to a repository subject.

Preferred forms:

- Mermaid for repository-native flow/dependency/sequence views;
- Graphviz/C4-as-code where automated extraction/layout becomes useful;
- generated SVG/PNG/PDF only as renderings of reviewed source.

**Do not use generative-image output as the authoritative technical diagram.** A generative image may later be used as cover art only, never as the source of system truth.

## 3. Mandatory status vocabulary

Every component or relationship shown in a mixed current/target view must be classifiable as one of:

- `CURRENT — VERIFIED/PRESENT FOR SUBJECT`
- `PRESENT — NOT CUT OVER / TRANSITIONAL`
- `TARGET — APPROVED DESIGN INPUT, NOT LIVE`
- `HISTORICAL / SUPERSEDED`
- `UNKNOWN / NOT YET QUALIFIED`

Never infer `LIVE` from file existence or Compose declaration alone.

## 4. Mandatory engineering dimensions

The complete Kingsman architecture drawing set must expose, without forcing Dainius or a reviewer to infer them:

1. **system/container/process boundary**;
2. **Docker/network/trust zone** where relevant;
3. **state ownership and durable stores**;
4. **typed contract crossing the boundary** where material;
5. **workload identity/authentication boundary**;
6. **policy/approval/capability authority boundary**;
7. **data/evidence flow**;
8. **control/authority flow**;
9. **execution/side-effect flow**;
10. **independent verification flow**;
11. **health/degradation/recovery flow**;
12. **external egress/provider boundary**;
13. **current versus target mapping**;
14. **failure domain / expected blast radius** for major organs;
15. **exact subject/currentness stamp**.

## 5. Required drawing set

A single poster is insufficient. The professional architecture set contains separate views with one concern per view.

### A — Current deployment topology

Derived from current Compose/runtime configuration.

Must show:

- actual named services/processes;
- actual networks/trust zones;
- persistent volumes/stores;
- declared dependencies;
- external egress points;
- profiles/optional components where material.

It is a view of **what is declared/present**, not proof that every service is live.

### B — Target physical deployment topology

Shows the intended Kingsman failure/trust/resource domains and which current services/modules migrate into each.

### C — Authority and consequential-action sequence

Must preserve the exact distinction:

`Perception / World State`
→ `Proposal Workspace`
→ `Constraint Assessment`
→ `Policy Decision`
→ `Human Approval or Scoped Autonomy`
→ `Single-use ActionCapability`
→ `Durable Workflow`
→ `Final-hand Actuator Validation/Consumption`
→ `ActuatorReceipt`
→ `Independent Verifier`
→ `VerifiedOutcome`
→ `Learning/Calibration`

Generic `decide → act` is unacceptable.

### D — Evidence / world-state / memory data flow

Must distinguish:

`Observation`
→ `PerceptionEvent`
→ `Journal/Event Subject`
→ `EvidenceRecord / GradedEvidence`
→ `Claim`
→ `WorldStateSnapshot`
→ `Proposal/Reasoning`

and show that memory is context, not automatically current fact.

### E — Identity and trust-boundary view

Must show:

- operator identity/approval boundary;
- service/workload identity;
- current Ed25519 direction where applicable;
- trust zones/networks;
- secrets/key ownership;
- distinction `MEMBERSHIP ≠ IDENTITY ≠ AUTHORITY`.

### F — Resilience / degradation / contingency view

Must show:

`Telemetry/Health`
→ `Dependency/Structure Graph`
→ `Diagnosis`
→ `Contingency Matching`
→ `Policy/Authority`
→ `Contain/Recover Workflow`
→ `Independent Verification`

and explicitly state what remains operational when a major organ fails.

### G — Current → target migration view

For each current service/module, show one disposition:

- RETAIN
- REWORK
- MERGE
- SPLIT
- REHOME
- SUPERSEDE
- ARCHIVE/HISTORICAL
- UNKNOWN — MORE EVIDENCE

No service is deleted merely because a target diagram has fewer boxes.

### H — Operator mission-control view

Shows:

- whole organism;
- current programme stage;
- component maturity S0–S5;
- current health/degradation;
- open defects/unknowns;
- approvals needed;
- current evidence subject;
- drill-down routes.

This is the operator/control-room view, not a substitute for A–G.

## 6. Arrow taxonomy

Where practical the diagram source or legend must distinguish:

- `DATA` — ordinary payload/data transfer;
- `EVIDENCE` — qualified/provenance-bearing evidence transfer;
- `PROPOSAL` — non-authoritative cognitive output;
- `AUTHORITY` — approval/grant/capability transfer;
- `EXECUTION` — side-effect dispatch;
- `VERIFY` — independent observation/outcome verification;
- `HEALTH` — telemetry/health/degradation signal;
- `CONTROL` — configuration/release/recovery control.

An unlabeled arrow across a trust/authority boundary is incomplete.

## 7. Subject stamp

Each authoritative drawing set must carry at minimum:

```text
Repository: dainius1234/kai-system
Branch/ref: <exact ref>
Architecture candidate: <file + blob/commit>
Current topology source: <compose/config file + blob/commit>
Generated/reviewed: <date>
Status: CURRENT / TARGET / MIXED
```

Where the view is machine-generated, include generator revision.

## 8. Current known repository complexity that visuals must not erase

The current full Compose already declares separate `agent-net`, `control-net`, `data-net`, `edge-net`, `egress-net`, `execution-net`, `observability-net` and `sensor-net` boundaries. Major declared components include PostgreSQL, Redis, Tool Gate, memory, introspection, heartbeat, dashboard, Supervisor, verifier, fusion, agentic cognition, executor, graph/memory extensions, financial awareness, ledger/metrics, sensors, model runtime, external interfaces, workspace/skills, House Doctor and backup/recovery components.

The contract layer also already distinguishes perception events, world-state claims/evidence/snapshots, proposals, assessments, policy decisions, approvals, capabilities, workflows, actuator receipts, verified outcomes, learning updates, autonomy evidence/grants and independent verifier identities.

Any visual that collapses those distinctions into generic `brain`, `tools` and `act` boxes is an orientation sketch only, not a professional architecture model.

## 9. Review gate

Before a visual is called Kingsman-grade, reviewer must answer:

1. Can every CURRENT box be mapped to an exact repository subject?
2. Can every TARGET box be mapped to an accepted design obligation?
3. Does every authority-crossing arrow name the contract/authority class?
4. Are state owners/stores visible?
5. Are trust/failure domains visible?
6. Is independent verification visibly separate from execution?
7. Are UNKNOWN/DEGRADED/PRESENT-NOT-CUT-OVER states representable?
8. Can the visual be regenerated without an image model inventing or losing text?
9. Would removal of underlying evidence invalidate the displayed status?
10. Can Dainius understand the whole machine and drill down without reverse-engineering it from source?

If not, report:

`ARCHITECTURE VISUAL — NOT KINGSMAN GRADE`

## 10. Standing correction

The previously generated infographic associated with D345 is **not an architecture authority artefact and must not be supplied to DeepSeek as the technical architecture**.

DeepSeek review should use the Markdown candidate plus the deterministic engineering drawing set.
