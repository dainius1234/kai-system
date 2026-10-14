# D346 — Engineering Architecture Visual Correction

> **STATUS: GOVERNANCE / DESIGN CORRECTION — NO RUNTIME OR PROGRAMME EXECUTION AUTHORISED.**

## 1. Trigger

After D345, a generative-image architecture infographic was produced as a visual representation of the candidate system.

Dainius reviewed it against the actual repository and correctly rejected it as toy/clickbait-level rather than professional-grade engineering architecture.

Kai independently compared it with the current Compose topology, contract layer, identity/authority code, proposal workspace, world-state/evidence semantics, resilience/recovery components and standing operator-visibility doctrine.

**Conclusion: operator criticism is SUPPORTED.**

The generated infographic is **REJECTED AS AN AUTHORITATIVE ARCHITECTURE VIEW**.

## 2. Why it failed

The image materially compressed or omitted:

- eight existing Docker network/trust zones;
- named current services/processes;
- durable stores/volumes/secrets;
- current versus target state;
- proposal versus policy versus approval versus capability;
- final-hand actuator enforcement;
- independent verifier separation;
- evidence versus memory versus current world state;
- service/workload identity versus authority;
- failure/degraded modes;
- current House Doctor/Supervisor/resilience machinery;
- current→target migration/disposition.

It therefore functioned as orientation/presentation art, not an evidence-bearing engineering model.

## 3. Repository evidence that makes the failure clear

The current repository already contains distinct engineering primitives including:

- `PerceptionEvent`;
- validated/deduplicated/staleness-aware ingress;
- crash-conscious append/replay journal;
- `Claim`, `EvidenceRecord`, `WorldStateSnapshot`;
- proposal-only workspace that explicitly cannot issue capabilities or execute;
- `PolicyDecision`;
- digest-bound `ApprovalRecord`;
- audience-bound single-use `ActionCapability`;
- `ActionWorkflow`;
- `ActuatorReceipt`;
- independent `VerifiedOutcome`;
- graded evidence and scoped autonomy;
- per-service Ed25519 identity direction;
- capability-gated actuator registry;
- independent verifier registry;
- circuit breakers, health, watchdog/healing primitives;
- Supervisor and House Doctor prototypes;
- real backup/restore functionality.

A professional visual must preserve these distinctions rather than reduce them to generic `brain / policy / tools / act` concepts.

## 4. Correction banked

### Engineering visual standard

`kai-pm/KINGSMAN_ENGINEERING_ARCHITECTURE_VISUAL_STANDARD.md`

Creation commit:

`6a4d89a2f3b583cd3643aceceb1de7254024a0bb`

Standing rule:

> **AN ARCHITECTURE VISUAL IS AN ENGINEERING MODEL, NOT DECORATION.**

Authoritative diagrams should be deterministic/diffable Mermaid, Graphviz or C4-as-code source; generated bitmap/SVG/PDF is only a rendering of reviewed source.

Generative-image output is not an authoritative technical diagram.

### Engineering drawing set v0.1

`kai-pm/KINGSMAN_ENGINEERING_ARCHITECTURE_DRAWING_SET_V0_1.md`

Creation commit:

`a07312ebb190328cb5f5dec9981994d5cef58c1a`

Views:

A. current repository network/deployment topology;
B. target physical trust/failure domains;
C. exact consequential-action authority sequence;
D. evidence/world-state/memory flow;
E. workload identity/authority boundary;
F. resilience/diagnosis/contingency flow;
G. current→target migration/disposition;
H. adversarial review matrix.

### DeepSeek correction supplement

`kai-pm/DEEPSEEK_REVIEW_PACKET_SUPPLEMENT_D346_ENGINEERING_DRAWINGS.md`

Creation commit:

`b57e5ea677dd8591b78ae57e327d86b83127a50f`

DeepSeek is explicitly instructed to ignore the rejected infographic and review deterministic engineering source.

## 5. Visual doctrine tightened

Authoritative architecture views must now show or explicitly scope out:

- exact subject/currentness;
- current/present/target/historical/unknown state;
- process/container boundary;
- trust/network boundary;
- state ownership;
- typed contract where material;
- identity boundary;
- evidence/data path;
- authority path;
- execution path;
- independent verification;
- health/degradation/recovery;
- external egress;
- current→target disposition.

An unlabeled arrow across an authority/trust boundary is incomplete.

## 6. Architecture candidate status

D346 **does not reject D345's underlying architecture specification merely because its generated visual failed**.

The specification and engineering logic remain candidate review inputs. The correction is that the architecture must be reviewed through precise engineering drawings and repo mappings, not through presentation art.

The candidate remains:

`NOT FROZEN / NOT IMPLEMENTATION AUTHORITY / OPEN TO DEEPSEEK + KAI + ORION + DAINIUS REVIEW.`

## 7. Next review order

1. Dainius reviews whether the corrected drawing-set level is appropriate.
2. DeepSeek attacks the candidate + engineering drawing set.
3. Kai verifies every repo-dependent DeepSeek premise and reconciles findings.
4. Orion later performs complete current-service/dependency/state-owner mapping against the reconciled target.
5. Required discriminating spikes are run before any disputed architectural choice is frozen.
6. Candidate v0.2 is produced.
7. Dainius reviews/finalises the master canon.

## 8. Programme authority unchanged

D346 does not authorise:

- runtime refactor;
- service consolidation/deletion;
- frozen House/048/Item8 changes;
- A-4 provenance execution;
- Future A4 implementation;
- financial autonomy;
- succession implementation.

**ITEM 8 BEFORE A4 remains standing.**

## 9. THREAD RECOVERY BLOCK

**CURRENT WORKSTREAM:** Kingsman master-canon architecture review/correction only.

**LAST PROVEN STATE:** D345 candidate remains a review subject; presentation infographic rejected; deterministic visual standard, drawing set and DeepSeek supplement banked.

**AUTHORISED NEXT ARCHITECTURE ACTION:** DeepSeek adversarial review of candidate + deterministic drawing set, followed by Kai repo-fact reconciliation.

**EXPLICITLY NOT AUTHORISED:** implementation/refactor or programme-sequence change.

**OPEN:** DeepSeek review, complete Orion current→target feasibility/dependency map, any discriminating spikes, Candidate v0.2, Dainius review, final master-canon freeze.

**OPERATOR VISIBILITY:** materially improved, but still `INCOMPLETE` until the full current service/dependency population and final target are mechanically reconciled and the mission-control surface is built.
