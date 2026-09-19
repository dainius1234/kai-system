# D348 — Existing-Kai Evolution Correction

> **STATUS: GOVERNANCE / ARCHITECTURE CORRECTION — v0.3 BECOMES PRIMARY REVIEW SUBJECT. NOT IMPLEMENTATION AUTHORITY. NOT PROGRAMME EXECUTION AUTHORITY.**

## 1. Trigger

Dainius rejected the v0.2 DeepSeek handoff because, although technically coherent, it still described the future system in a way that invited a blank-sheet redesign.

The criticism was specific and correct:

- it under-described the architecture that already exists in the repository;
- it did not make Unified Hunter's already-built migration machinery the centre of the evolution plan;
- it did not enumerate the existing compatibility/cutover shims strongly enough;
- it risked DeepSeek proposing new Event/Memory/Authority/Workflow/Doctor systems beside components that already implement those responsibilities;
- it described missing future primitives without sufficiently distinguishing **new code** from **professionalisation of existing code**;
- it did not show enough of the existing cognitive, proactive, identity, relationship, sensor, output, recovery and growth capabilities that must survive simplification.

This was a design-handoff defect, not merely a wording preference.

## 2. Repo comparison performed

Kai re-read current repository subjects including:

- `docker-compose.minimal.yml`;
- `docker-compose.full.yml`;
- `docker-compose.sovereign.yml`;
- `README.md` as a stale-but-useful capability census, not authority;
- `KAI_UNIFIED_HUNTER_ARCHITECTURE_AND_ROADMAP.md`;
- `UH_PROGRESS_TRACKER.md`;
- perception shadow/active runner;
- Cortex world-state source adapter;
- scoped-autonomy legacy bridge;
- actuator migration driver and legacy verifier;
- service/workload identity transition;
- browser/notify/monitor/screen-watcher examples.

The comparison confirmed that current Kai already contains a serious migration skeleton and many capability families that v0.2 could make look newly invented.

## 3. Key corrected fact — Unified Hunter is already built behind the existing system

Current UH tracker records completed/tested work including:

- UH-1 canonical contracts;
- UH-2 perception spine;
- UH-3 scoped World State;
- UH-4 proposal-only workspace;
- UH-5 policy / approval / capability;
- UH-6 paper-trade vertical slice;
- UH-7 actuator registry + migration;
- UH-8 autonomy requalification;
- payload bounds;
- Ohana/assessment separation;
- rollback guards;
- concurrency/clock/fencing;
- service authentication;
- erasure lineage;
- legacy trust bridge;
- full 34-actuator catalogue migration/test coverage;
- legacy path source verification;
- feature flags exercised together;
- live endpoint/mutating-handler verification;
- dashboard identity/scopes/degraded semantics;
- architecture-rule gates.

The current operational state is **BUILT, NOT CUT OVER BY DEFAULT**.

Existing controlled transition points include:

- `KAI_PERCEPTION_MODE=shadow|active`;
- `KAI_CORTEX_SOURCE=poll|world_state`;
- `KAI_AUTONOMY_ENFORCE=false|true`;
- actuator migration state + source-verified legacy closure;
- existing feature flags and preflight controls.

Therefore the correct architecture programme is to **finish/evolve this migration**, not create another Hunter/control pipeline beside it.

## 4. Existing shims now elevated as architectural assets

D348 explicitly recognises and preserves:

1. **Perception Shadow/Active Runner** — current sensor→PerceptionEvent→Journal→optional WorldState bridge.
2. **Cortex World-State Adapter** — compatibility projection with deliberate fallback to old polling.
3. **LegacyTrustBridge** — old scalar trust versus scoped autonomy comparison/enforcement transition.
4. **Actuator Migration Driver + Legacy Verification** — risk-tiered migration, handler requirement, source-proven legacy closure.
5. **Shared Service Auth → Signed Workload Identity transition** — current membership/auth compatibility evolving toward receiver-verifiable service identity.
6. **Dashboard credential/degraded-response shims** — current UI transition/security machinery.
7. **Feature flags as cutover controls** — never evidence of qualification by themselves.
8. **Evidence/release/erasure/verifier foundations** — future platform inputs, not disposable experiment code.

Standing migration principle:

> **SHADOW / COMPARE / SOAK → CUT OVER → PROVE OLD AUTHORITY PATH DEAD → RETIRE.**

## 5. New primary master review subject

`kai-pm/KINGSMAN_EXISTING_KAI_EVOLUTION_MASTER_PLAN_V0_3.md`

Creation commit:

`98dc2560c0204dd9c58d823dbd8c07754704276a`

v0.3 corrects v0.2 by structuring the architecture as:

`CURRENT COMPONENT / PATH`
→ `QUALIFY REALITY`
→ `PRESERVE INTENT`
→ `SHIM / ADAPTER`
→ `TARGET RESPONSIBILITY / CONTRACT`
→ `SHADOW / SOAK`
→ `VERIFIED CUTOVER`
→ `PROVE OLD AUTHORITY PATH DEAD`
→ `RETIRE / REHOME LEGACY IMPLEMENTATION`.

Standing rule:

> **NO NEW BOX WITHOUT A CURRENT-TO-TARGET LINEAGE.**

## 6. New DeepSeek review subject

`kai-pm/DEEPSEEK_REVIEW_PACKET_EXISTING_KAI_EVOLUTION_V0_3.md`

Creation commit:

`1d189de2cfd7b20499b7b7fc8bb8981915672350`

The packet is self-contained because DeepSeek has no repository access. It explicitly requires every recommendation to name:

- current component/path;
- what already works and must survive;
- current defect/limit;
- rework/split/merge;
- shim/adapter;
- target home;
- new code actually required;
- whether a new process/service is justified;
- shadow/soak method;
- cutover test;
- old-path retirement condition;
- rollback risk.

Blank-sheet advice without this mapping is rejected.

## 7. Existing capability families explicitly protected from accidental deletion

v0.3 makes explicit that architecture simplification must retain/map:

### Inner life / identity / relationship

- emotional memory/mood arcs;
- self-reflection/epistemic humility/confession;
- narrative identity/autobiography/legacy time capsules;
- imagination/theory of mind/counterfactuals;
- conscience/confirmed values;
- gratitude/relationship continuity;
- operator model/cognitive fingerprint;
- Obsidian Brain.

### Cognitive depth

- Socratic questioning;
- hypothesis engine;
- temporal projection;
- dialectical/analogical/concept-blending/synthetic/transitive future modules;
- causal world model;
- Global Workspace;
- reasoning FSM;
- persistent teammates;
- adversary/conviction/swarm/reputation.

### Proactive/world-awareness depth

- world context injection;
- proactive observer;
- anomaly detection;
- cross-sensor correlation;
- pattern learning;
- proactive scheduling;
- ritual discovery;
- capability-gap logging;
- monitor/screen/system watchers.

### Growth and interfaces

- skill-hunter/provenance/probation;
- Agent-Evolver/Dream/curiosity/workspaces;
- dashboard/voice/wake/STT/TTS/notify/avatar/Telegram/browser/documents/files/vault.

Simplification may consolidate implementation boundaries. It may not silently delete product capability.

## 8. Missing shims now explicitly identified

v0.3 currently proposes these missing joints for review:

- M01 Contract v1→v2 adapter layer;
- M02 EventJournal backend adapter / dual-write verifier;
- M03 World-State consumer compatibility projections;
- M04 Tool Gate compatibility facade;
- M05 shared-token→signed-workload-identity dual stack;
- M06 ActuatorRegistry→DurableWorkflow adapter;
- M07 Ollama→ModelRuntimeManager adapter;
- M08 memu compatibility facade;
- M09 proactivity consolidation adapter;
- M10 Supervisor split facade;
- M11 House Doctor structured-diagnosis adapter;
- M12 Dashboard→MissionControl read adapter;
- M13 Backup→LineageManifest wrapper;
- M14 finance separation adapter;
- M15 growth/release bridge;
- M16 telemetry compatibility adapter.

These are candidate shims for DeepSeek/Kai/Orion review, not implementation decisions.

## 9. Correct evolution sequence

v0.3 replaces a target-first W0–W9 interpretation with a more repo-grounded E0–E10 evolution sequence:

- E0 exact current component/connection census;
- E1 reconcile minimal/full/sovereign topology;
- E2 finish the Unified Hunter cutover already built;
- E3 persist existing authority/workflow behind compatible APIs;
- E4 consolidate World State/memory consumers;
- E5 consolidate proactivity around existing detectors;
- E6 professionalise current cognition/model runtime;
- E7 unify health/Doctor/recovery;
- E8 evolve current backup into continuity/lineage;
- E9 add sustainability/succession layers over existing identity/finance/authority;
- E10 evolve Dashboard/docs/repo into Mission Control and one professional release structure.

This remains architecture dependency order only. Current D-numbered programme authority still controls when implementation may happen.

## 10. v0.2 disposition

`KINGSMAN_MASTER_ARCHITECTURE_AND_PROFESSIONALISATION_CANDIDATE_V0_2.md`

is now:

> **SUPERSEDED AS PRIMARY REVIEW SUBJECT — RETAINED AS TARGET-DESIGN INPUT / REVIEW LINEAGE.**

It is not deleted because its target architecture, invariants and missing-primitives analysis remain useful. v0.3 corrects the migration framing and binds those target responsibilities to current Kai.

The v0.2 DeepSeek packet is likewise superseded for current review.

## 11. Programme authority unchanged

D348 does not authorise:

- H2 v1.1;
- mutation of House/Census evidence;
- KAI-GATE-048 scope change;
- Item 8 execution beyond current authority;
- A-4 provenance execution;
- Future A4 implementation;
- architecture refactor;
- service merge/delete;
- succession implementation;
- autonomous finance;
- uncontrolled self-modification/self-preservation.

**ITEM 8 BEFORE A4 remains standing.**

## 12. THREAD RECOVERY BLOCK

**PRIMARY MASTER REVIEW SUBJECT:** `KINGSMAN_EXISTING_KAI_EVOLUTION_MASTER_PLAN_V0_3.md` at `98dc2560c0204dd9c58d823dbd8c07754704276a`.

**DEEPSEEK REVIEW SUBJECT:** `DEEPSEEK_REVIEW_PACKET_EXISTING_KAI_EVOLUTION_V0_3.md` at `1d189de2cfd7b20499b7b7fc8bb8981915672350`.

**SUPERSEDED PRIMARY SUBJECT:** v0.2 remains retained target-design/history input.

**CURRENT WORKSTREAM:** architecture evolution review only; current canonical programme authority unchanged.

**LAST PROVEN STATE:** existing UH migration/shims/topology were re-read; v0.3 evolution plan and self-contained review packet are durably banked. No implementation/runtime change is claimed.

**AUTHORISED NEXT ARCHITECTURE ACTION:** DeepSeek adversarial review of **existing-Kai evolution v0.3**, followed by Kai repo-fact reconciliation and Orion exact current→target map when appropriate.

**EXPLICITLY NOT AUTHORISED:** blank-sheet implementation, service consolidation/deletion, frozen-programme mutation or high-consequence future autonomy.

**OPEN:** complete repo component/connection census; DeepSeek review; Kai reconciliation; Orion mapping; missing-shim discovery; discriminating spikes; deterministic drawings v0.3; final Dainius review/freeze; canonical DECISIONS append.

**OPERATOR VISIBILITY:** corrected review framing exists, but final current→target visual/control-room map remains OPEN.
