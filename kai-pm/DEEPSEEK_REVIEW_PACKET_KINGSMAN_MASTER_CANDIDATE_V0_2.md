# DeepSeek Adversarial Review Packet — Kingsman Master Candidate v0.2

> **REVIEW SUBJECT:** `kai-pm/KINGSMAN_MASTER_ARCHITECTURE_AND_PROFESSIONALISATION_CANDIDATE_V0_2.md`
>
> **TRACEABILITY SUBJECT:** `kai-pm/KINGSMAN_MASTER_CANON_INPUT_MANIFEST_V0_2.md`
>
> **ENGINEERING DRAWINGS:** `kai-pm/KINGSMAN_ENGINEERING_ARCHITECTURE_DRAWING_SET_V0_1.md` remains the current deterministic drawing set but must be interpreted through v0.2 and will be revised after this review.
>
> **STATUS:** external adversarial review input only. No implementation or programme execution is authorised by this packet.

## 1. What this review is for

Kai is a private, long-lived personal intelligence organism, not a commercial agent product and not a wrapper around one LLM.

The system is intended to:

- be proactive rather than prompt-only;
- preserve identity/lineage while replacing models/frameworks/hardware;
- use multiple cognitive specialists without parallel authority;
- maintain qualified world state and memory;
- enforce one deterministic policy/authority/capability path for consequential action;
- independently verify outcomes;
- contain local failure rather than collapse as one monolith;
- diagnose itself and use qualified contingency/recovery knowledge;
- learn/grow through controlled releases rather than uncontrolled self-modification;
- remain technically/economically viable over years/decades;
- eventually survive the original operator under explicit succession/legal/governance design;
- protect operator/family interests rather than reinterpret them as fuel for self-preservation.

Your task is to **try to break the architecture before we freeze it**.

Do not flatter it. Do not redesign merely for novelty. Where the candidate is sound, say so precisely.

## 2. Important role/authority context

- **Dainius** — final operator/programme authority.
- **Kai** — architecture/repo review/reconciliation/continuity.
- **Orion/Claude** — repo-side investigator/executor.
- **DeepSeek** — external adversarial technical/coding reviewer; no repo access/authority.

Repo-dependent claims in your response remain `HYPOTHESIS / REVIEW INPUT` until Kai/Orion verify them.

## 3. Non-negotiable programme constraints

Do not recommend opportunistic mutation of current frozen work.

Standing programme constraints include:

- House-in-Order current sequence remains separately governed;
- KAI-GATE-048 remains separately governed;
- Item 8 is an assurance/build workstream, not a runtime agent/team;
- **ITEM 8 BEFORE A4**;
- `A-4 PROVENANCE` is distinct from `FUTURE A4 SELF-DIAGNOSIS`;
- this architecture does not authorise H2 v1.1, 048 changes, Item8 builds, A-4 execution, Future A4 implementation, succession or autonomous finance.

## 4. Current repo facts already inspected by Kai

Treat these as review context, not proof that target architecture is live:

- typed Pydantic contracts already separate perception/world/action/authority/outcome concepts;
- perception ingress validates, bounds, deduplicates and marks stale events;
- file EventJournal has fsync/replay/digest logic but is not a production multi-writer backbone;
- world-state code has scoped immutable snapshots/conflict/freshness semantics but persistence remains prototype-level;
- proposal workspace explicitly cannot issue capabilities or execute;
- policy engine fails closed;
- approval is digest-bound, expiring, replay-protected and revocable;
- capability concept is audience-bound and single-use but critical authority state is in memory;
- autonomy grants are scoped, bounded, expiring, revocable and evidence-earned but in memory;
- per-service Ed25519 request identity derives principal from verifying key rather than caller assertion; complete rollout still requires qualification;
- actuator registry is capability-gated and migration-aware;
- verifier registry rejects self/same-independence-group verification;
- graded evidence prevents self-generated model/simulation output from laundering itself into trust;
- release bundles bind capability/autonomy to code revision;
- resilience primitives include retry, circuit-breaker, deep health, watchdog and heuristic healing;
- Supervisor mixes fleet health/recovery with proactive nudging;
- House Doctor is currently a rule/string v0.1 diagnosis service;
- backup-service performs real backups/restores but not complete lineage/restore qualification;
- model registry is a hard-coded card/keyword-router sketch, not a real resource/runtime manager;
- current Compose has separate agent/control/data/edge/egress/execution/observability/sensor networks and many historical services.

## 5. Review dimensions

### A — organism coherence

1. Is Kai correctly modelled as organism above replaceable components?
2. Are any organs actually duplicate sovereign orchestrators?
3. Does the architecture accidentally create a second source of truth or authority?
4. Which conceptual organs should be merged?
5. Which missing organ/primitives have still not been identified?

### B — service/process boundaries

6. Is the proposed eight-domain physical layout appropriate for one powerful local machine?
7. Is `Kai Core` still too broad? Identify exact modules that should remain in-process versus isolated.
8. Which current microservices should become modules/libraries?
9. Which current modules genuinely deserve stronger process isolation?
10. Where will synchronous dependencies cause shared-fate cascades?

### C — data / evidence / world state

11. Is PostgreSQL + transactional outbox a sound first authoritative backbone?
12. Where would a dedicated event broker be necessary rather than fashionable?
13. Is Observation → Evidence → Claim → WorldState the correct semantic layering?
14. What is the minimum useful structured Claim model?
15. How should memory be referenced without becoming current fact?
16. How should immutable provenance/audit coexist with privacy/erasure?
17. What storage/index should be authoritative versus rebuildable projection?

### D — proactivity

18. Is Goal / Obligation / Commitment / Watch + Time + Attention the correct missing abstraction?
19. What is the minimal durable scheduler/watch design that survives restart/time-zone/clock issues?
20. How should attention/interruption be calibrated for usefulness without becoming spam?
21. How should proactive learning improve timing without silently increasing authority?
22. What failure mode occurs when proactive observers are blind/stale?

### E — cognition / models

23. Is Cognitive Workspace / Unified Hunter correctly separated from Model Runtime Manager?
24. What robust routing/admission mechanism should replace keyword voting?
25. What qualification data should each model/role carry?
26. What stops model-council deliberation from exploding in latency/cost/non-termination?
27. How should unified-memory/KV-cache/preemption be managed on Strix Halo?
28. Which workloads realistically belong on CPU/GPU/NPU today rather than theoretically?
29. What hardware/runtime spikes must precede canon freeze?

### F — identity / policy / authority

30. Is current Ed25519 workload identity a sensible v1, or should mTLS/SPIFFE semantics be adopted sooner?
31. What exact data must an ActionCapability bind at final hand?
32. How should single-use capability consumption be persistent/atomic under concurrency?
33. Should policy remain custom, use OPA/Rego, or another deterministic engine?
34. What protected human approval mechanism is appropriate for a local personal AI?
35. Find any route where cognition, memory, evidence, proactivity, Doctor or recovery could accidentally become authority.
36. How should authority survive restart without risking stale/replayed permissions?

### G — workflow / hands / verification

37. Postgres-backed workflow engine vs Temporal for first production generation — recommend one with operational reasoning.
38. What workflow states/semantics are missing?
39. How should non-idempotent external actions reconcile after timeout/unknown outcome?
40. Is a separate egress/target-control boundary justified?
41. Which actuators require independent containers/sandboxes?
42. What does target-specific independent verification require beyond a generic verifier service?

### H — immune system / self-diagnosis

43. Is Component/Dependency/Authority Graph a correct first-class primitive?
44. How should House Doctor, Future A4, Supervisor and Contingency Library divide responsibility?
45. How should qualified contingencies be versioned/applicability-bound?
46. How should conflicting contingencies compose without becoming a new orchestrator?
47. What fault-injection matrix proves blast-radius containment?
48. What remains operational when each major organ fails independently?
49. How do we stop automatic recovery from repeatedly masking a structural defect?

### I — evolution / release / learning

50. Is the proposed skill/release/probation lifecycle sufficient to prevent uncontrolled self-modification?
51. What evidence should be required before a learned behaviour/skill gains production authority?
52. What should be immutable/attested in a release bundle?
53. How should build-time assurance machinery inform runtime without shipping brittle scripts unchanged?

### J — long horizon / succession / self-sufficiency

54. What minimum lineage manifest proves a restored/migrated system is the intended Kai lineage?
55. What backup topology is realistic for decades on a personal system?
56. How should root/key recovery avoid one catastrophic master secret?
57. Which succession primitives should be architected now and which must wait for legal design?
58. How should operator temporary absence be separated from permanent succession?
59. Is Financial Sustainability best a plane, capability family or external subsystem?
60. What controls prevent operating-cost survival from risking protected family assets?
61. What safe archive/read-only mode should exist when authority/funding/identity cannot be established?

### K — operator control / documentation

62. Is the mission-control information architecture sufficient for Dainius to govern the organism?
63. Which status fields should be machine-derived versus human-reviewed?
64. How should architecture diagrams/status ticks be evidence-bound and invalidated when evidence disappears?
65. What is the smallest useful operator dashboard that does not become another stale truth source?

### L — migration / professionalisation

66. Review W0–W9 dependency order. What is missing or misordered?
67. Which current services should clearly be retired rather than professionalised?
68. Which existing prototype code is stronger than the target proposal gives it credit for?
69. Where will dual old/new authority exist during migration and how should cutover prevent it?
70. Which work packages need explicit migration rollback/compatibility windows?

### M — adversarial simplification

71. If forced to remove/merge **30% of the proposed boxes**, what do you remove/merge without losing invariants?
72. What are the top five cascading-failure paths?
73. What are the top five hidden assumptions based on project history rather than architectural necessity?
74. Which external standards/frameworks are unnecessary complexity?
75. What must be experimentally measured before any architect can responsibly decide?

## 6. Finding format

For every material issue use:

```text
ID:
SEVERITY: BLOCKER / MAJOR / MINOR / QUESTION
SECTION / ORGAN:
CLAIM / PROBLEM:
WHY IT MATTERS:
PROPOSED CHANGE:
WHAT IT SIMPLIFIES / ADDS:
NEW RISKS CREATED:
REPO FACTS IT DEPENDS ON:
TEST / MEASUREMENT THAT WOULD DISCRIMINATE:
CONFIDENCE:
```

Severity definitions:

- `BLOCKER` — breaks mission/identity/truth/authority/failure-containment invariant.
- `MAJOR` — direction viable but likely unsafe/incorrect/fragile.
- `MINOR` — worthwhile improvement not required to approve direction.
- `QUESTION` — insufficient evidence; test/repo inspection needed.

## 7. Required final sections

Finish with exactly:

1. `OVERALL VERDICT — APPROVE DIRECTION / APPROVE WITH CHANGES / REJECT DIRECTION`
2. `TOP 15 CHANGES BEFORE CANON FREEZE`
3. `MISSING ORGANS / SYSTEM PRIMITIVES`
4. `RECOMMENDED LOGICAL ARCHITECTURE`
5. `RECOMMENDED PHYSICAL DEPLOYMENT`
6. `RECOMMENDED DATA / EVENT / EVIDENCE ARCHITECTURE`
7. `RECOMMENDED STRIX HALO MODEL-RUNTIME ARCHITECTURE`
8. `RECOMMENDED AUTHORITY / WORKFLOW / ACTUATION ARCHITECTURE`
9. `RECOMMENDED RESILIENCE / SELF-DIAGNOSIS ARCHITECTURE`
10. `RECOMMENDED LONG-HORIZON / SUCCESSION PREPARATION`
11. `SIMPLIFY BY 30%`
12. `TOP 5 CASCADING FAILURE PATHS`
13. `EXPERIMENTS / SPIKES REQUIRED BEFORE FREEZE`
14. `TOP 5 STRONGEST PARTS`
15. `TOP 5 HIDDEN ASSUMPTIONS`
16. `CURRENT-REPO PIECES YOU WOULD RETAIN / MERGE / RETIRE`
17. `FINAL QUESTIONS FOR DAINIUS`

## 8. Review discipline

Do not assume a framework is better merely because it is standard. If recommending OPA, SPIRE, Temporal, NATS, Kubernetes or another platform, state:

- which current Kai component/problem it replaces;
- what new operational dependency it creates;
- why a smaller local mechanism is insufficient;
- what discriminating test would justify adoption.

Do not turn this personal organism into generic SaaS/cloud architecture by habit.
