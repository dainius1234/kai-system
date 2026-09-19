# DeepSeek Adversarial Review Packet — EXISTING KAI → KINGSMAN Evolution v0.3

> **THIS IS NOT A BLANK-SHEET DESIGN EXERCISE.**
>
> DeepSeek has no GitHub access. Everything needed to understand the architecture/migration problem is included here. Treat supplied repo facts as `KAI-VERIFIED CONTEXT`; do not claim you independently inspected them.
>
> Review subject: evolve the **existing Kai** into the Kingsman production architecture with minimum destructive rewrite, explicit shims, verified cutover and zero capability loss.

---

## 1. The architectural mistake this review must avoid

Do **not** respond by inventing a fresh architecture consisting of generic boxes such as:

- Event Bus
- Memory Plane
- Agent Orchestrator
- Policy Service
- Workflow Engine
- Self-Healing Agent
- Dashboard

without showing exactly how they evolve current Kai.

Kai already has working/prototype versions of most of these responsibilities and a serious migration programme behind the existing system.

For every recommendation you must state:

```text
CURRENT COMPONENT(S):
CURRENT ROLE / PATH:
WHAT IS ALREADY GOOD / MUST SURVIVE:
CURRENT DEFECT / LIMIT:
SHIM / ADAPTER:
TARGET RESPONSIBILITY:
NEW CODE ACTUALLY REQUIRED:
WHY A NEW PROCESS/SERVICE IS OR IS NOT REQUIRED:
SHADOW / SOAK / COMPATIBILITY METHOD:
CUTOVER EVIDENCE:
OLD PATH RETIRE CONDITION:
ROLLBACK:
```

Advice that does not map current→target is too generic to use.

---

## 2. Product purpose and identity

Kai is intended to be a private, proactive, long-lived personal intelligence organism.

Primary purpose:

- grow with Dainius;
- understand, assist, challenge, protect and care for him;
- preserve memory, relationship, values, history and knowledge;
- proactively notice important change rather than wait for prompts;
- gain useful skills/capabilities through controlled learning;
- maintain technical continuity;
- eventually become economically self-sufficient under bounded governance;
- survive replacement of models/frameworks/hardware/providers;
- eventually survive beyond Dainius under explicit succession/legal/governance design;
- continue appropriate stewardship for his daughter without treating inherited purpose as unlimited authority.

Identity law:

`KAI IS THE WHOLE ORGANISM.`

Kimi, DeepSeek, GLM, Dolphin, Ollama, Unified Hunter, memu, House Doctor, hardware, databases etc. are replaceable organs/resources.

The model is not Kai.

---

## 3. Existing architecture population — current Kai is already large

The repo contains roughly 60 service/application families plus shared architecture modules.

### Core

- PostgreSQL / pgvector
- Redis
- Tool Gate
- memu-core
- memu-core-introspect
- agentic
- heartbeat
- dashboard
- Ollama + pull/init

### Perception / awareness

- audio-service
- camera-service
- vision-service
- wake-service
- screen-capture
- screen-watcher
- clipboard-service
- files-service
- document-parser
- email-reader
- news-feed
- weather-service
- airquality-service
- calendar-service
- calendar-sync
- git-watcher
- docker-watcher
- sysmetrics
- Cortex
- monitor-service
- agentic proactive observer

### Memory / identity / context

- memu-core
- memu-core-introspect
- memu-graph
- TurboVec / pgvector paths
- Letta archival memory
- memory-compressor
- Obsidian/vault-sync
- emotional memory
- narrative identity
- operator model
- cognitive fingerprint

### Cognition

- agentic reasoning service
- deterministic reasoning FSM
- Socratic questioning
- Scout / Sage / Doctor / Oracle teammates
- swarm assembly and reputation/conflict resolver
- adversary engine
- conviction scoring
- hypothesis engine
- temporal projection
- causal world model
- Global Workspace prototype/stub
- GPU-era stubs: dialectical synthesis, analogical reasoning, concept blending, synthetic experience, transitive reasoning
- Kai Advisor

### Hands / outputs

- executor
- browser-agent
- notify-service
- TTS
- avatar
- Telegram
- vault export
- calendar actions
- backup/restore actions
- broker bridge / paper trader
- actuator registry with 34 actuator identities across 8 risk tiers

### Health / recovery

- heartbeat
- Supervisor
- House Doctor
- conversational Doctor teammate
- common resilience library
- verifier
- fusion-engine
- metrics-gateway
- introspection
- Docker/system watchers
- system FSM DEGRADED/RECOVERING

### Growth

- skill-hunter
- workspace-manager
- Agent-Evolver
- Dream/introspection cold path
- skill provenance/probation/auto-disable
- capability-gap logging
- ritual discovery

### Finance

- financial-awareness
- broker-bridge
- paper-trade vertical slice
- market/opportunity/strategy analysis modules

### Security/operations

- minimal/full/sovereign Compose variants
- Docker trust/network segmentation
- service auth bridge
- newer Ed25519 workload identity
- dashboard roles/scopes
- Vault/Vault rotator
- Tailscale
- Prometheus/Alertmanager/Grafana
- gVisor/AppArmor executor hardening
- CI/security gates
- release-bundle/evidence primitives
- backup system

Do not discard a capability merely because its current implementation is prototype/stub/poorly placed.

---

## 4. Existing deployment reality is inconsistent and must be reconciled

There are three Compose definitions.

### Minimal

Broad daily-driver topology. Includes many services such as:

browser-agent, notify, document-parser, monitor, broker, sysmetrics, screen-watcher, email, news, weather, docker-watcher, airquality, calendar, git, skill-hunter, House Doctor, vault-sync, Cortex, wake, Supervisor, verifier, plus core.

Networks include:

`agent-net`
`control-net`
`data-net`
`edge-net`
`egress-net`
`observability-net`
`sensor-net`

### Full

Not currently a simple superset of minimal.

Adds/contains heavy/alternate components such as:

agentic-introspect, executor, fusion-engine, memory-compressor, memu-graph, Letta, financial-awareness, ledger-worker, metrics-gateway, camera/audio, avatar, screen-capture, backup, calendar-sync, Kai Advisor, Telegram, workspace-manager, parakeet.

But it currently omits several of the minimal services.

### Sovereign

Production-hardening direction with:

- read-only/cap-drop/no-new-privileges defaults;
- Vault / rotation;
- Tailscale;
- Prometheus/Alertmanager/Grafana;
- hardened executor/gVisor/AppArmor direction;
- pgvector storage;
- stronger control/data/execution separation.

**Question:** How should these become one canonical topology/profile model without rewriting the services?

---

## 5. Unified Hunter migration is ALREADY BUILT behind existing Kai

This is the most important context.

Existing Unified Hunter work reports complete/tested:

- UH-1 canonical contracts
- UH-2 perception spine
- UH-3 scoped World State
- UH-4 proposal-only workspace
- UH-5 policy / approval / capability
- UH-6 paper-trade vertical slice
- UH-7 actuator registry + migration
- UH-8 autonomy requalification

Also built/tested:

- payload bounds
- assessment/Ohana separation
- rollback guards
- concurrency/clock/fencing
- service authentication
- erasure lineage
- legacy trust bridge
- 34 actuator migration catalogue / handlers
- legacy-path source verification
- migration flags exercised together
- live endpoint verification
- mutating handler verification
- dashboard auth/scopes/degraded response
- architecture-rule gates

The tracker says the architecture is **BUILT, NOT CUT OVER**.

New machinery sits behind existing runtime; migration flags default to legacy/safe paths.

Therefore DO NOT propose rebuilding Unified Hunter from scratch.

---

## 6. Existing canonical UH law

Every consequential action is intended to follow:

`Perception`
→ `World State`
→ `Proposal`
→ `Policy`
→ `Approval`
→ `Capability`
→ `Execution`
→ `Observation`
→ `Verification`
→ `Learning`.

Critical current invariants already designed/tested:

- specialists do not self-authorise;
- proposal workspace cannot issue capabilities or execute;
- exact operation approval/capability binding;
- capability enforced at final hand;
- immutable/scoped versioned World State;
- UNKNOWN/conflict first class;
- workflows durable/idempotent/fenced in target semantics;
- outcome learning requires independent verification;
- human approval is exact/authenticated for high consequence;
- free-form text cannot carry control authority;
- no big-bang rewrite;
- legacy paths disabled only as replacements become verified.

This is the skeleton to evolve, not replace.

---

## 7. Existing shims / migration mechanisms

### E01 — Perception Shadow Runner

Current:

`KAI_PERCEPTION_MODE=shadow` default.

It polls existing sensors, adapts readings to `PerceptionEvent`, validates/journals them and does not affect current consumers.

`active` additionally reduces accepted events into World State.

Legacy polling remains during migration.

REVIEW:
Should this remain the standard sensor cutover pattern? What needs improving?

### E02 — Cortex Source Adapter

Current:

`KAI_CORTEX_SOURCE=poll` default.

Optional:

`KAI_CORTEX_SOURCE=world_state`.

Adapter renders canonical World State back into the shape current Cortex/agentic code consumes.

If new state is empty, it deliberately falls back to existing polled state.

REVIEW:
Should equivalent compatibility projections migrate other current consumers?

### E03 — Legacy Trust / Scoped Autonomy Bridge

Current old system has scalar `TrustLevel` semantics and new scoped grants.

`LegacyTrustBridge` exists.

Default advisory mode compares decisions but legacy behaviour stands.

`KAI_AUTONOMY_ENFORCE=true` makes new scoped authority binding.

Rule:
legacy authority may only subtract, never widen new scoped authority.

Current blocker:
no grants means enforcement would deny all gated capabilities, so preflight blocks premature activation.

REVIEW:
How should this be completed safely without replacing it with another autonomy system?

### E04 — Actuator Migration Driver

34 actuator identities across 8 risk tiers are registered/migrated in tests.

Migration driver:

- lower risk tiers first;
- refuses ACTIVE without handler;
- refuses VERIFIED if legacy path remains open;
- allows soak at VERIFIED before ACTIVE.

Legacy verifier checks source tree to prove legacy path is actually closed.

For many routes “legacy closed” means authentication/capability added to the existing route, not route deletion.

REVIEW:
Should this migration state machine become the general pattern for other subsystem cutovers?

### E05 — Service Auth → Workload Identity transition

Current:

shared `KAI_SERVICE_TOKEN` fails closed on several mutating routes.

Problem:
shared secret proves possession/membership, not which service called.

Newer Ed25519 path exists:
receiver derives caller principal from the public key that verifies the request.

Cortex already has mixed transition semantics where one class of route uses shared token while another uses signed identity.

REVIEW:
Design the minimum dual-stack migration from shared token to per-workload identity without changing every caller simultaneously.

### E06 — Dashboard Credential / Degraded Shim

Current dashboard remediation includes:

- browser credential shim;
- route roles/scopes;
- fail-closed protected routes;
- explicit degraded envelope so dependency outage cannot look like normal response.

REVIEW:
How should Mission Control evolve from Dashboard rather than become a second dashboard?

### E07 — Feature Flags

Current migration flags include:

`KAI_PERCEPTION_MODE`
`KAI_CORTEX_SOURCE`
`KAI_AUTONOMY_ENFORCE`
plus many `FF_*` capability flags.

Rule:
flag selects a path; flag does not prove that path safe.

### E08 — Evidence / release / verification foundations

Current system already has:

- graded evidence;
- independent verifier registry;
- release bundle/code revision binding;
- erasure lineage;
- architecture/security CI gates.

REVIEW:
Which should evolve into runtime platform primitives versus remain build/assurance-only?

---

## 8. Existing capability → target responsibility map

For each row, recommend KEEP / REWORK / SPLIT / MERGE / REHOME / RETIRE and the migration shim.

### Perception

CURRENT:
sensor services + Cortex + monitor/watchers + perception adapters + shadow runner.

TARGET RESPONSIBILITY:
one typed/provenance-bound perception path feeding qualified World State.

DO NOT:
replace sensor services simply because a “Perception Plane” is drawn.

### Event journal

CURRENT:
file-backed crash-conscious EventJournal with append/replay/digest semantics.

TARGET:
durable production journal/outbox.

QUESTION:
keep same interface and swap backend, or introduce broker? What evidence would justify a broker?

### World State

CURRENT:
scoped/immutable/conflict/freshness semantics plus Cortex compatibility adapter.

TARGET:
one durable qualified current-state model.

QUESTION:
how should existing consumers be migrated one by one?

### Memory

CURRENT:
memu-core, introspect, graph, Letta, compressor, Obsidian/vault, emotional/narrative/operator state.

TARGET:
coherent memory/continuity organ with authoritative records and derived indexes clearly distinguished.

QUESTION:
what should remain one service/module and what should split? How to avoid a new “Memory Plane” beside memu?

### Proactivity

CURRENT:
agentic proactive observer, monitor rules, Cortex, calendar scheduling, anomaly/correlation, screen watcher, rituals, capability-gap logging, Supervisor nudges.

TARGET:
first-class Goals / Obligations / Commitments / Watches / Time / Attention semantics.

QUESTION:
what is the smallest semantic owner that consolidates these existing detectors without rewriting them?

### Cognition

CURRENT:
agentic FSM, Socratic questioning, swarm, teammates, adversary, conviction, causal/forecast/hypothesis modules, proposal workspace, Global Workspace concepts/stubs.

TARGET:
Unified Hunter/Cognitive Workspace where models are role-qualified cognitive organs and proposal-only.

QUESTION:
which current orchestration should remain; which overlaps should merge; which stubs are retained for future activation?

### Model runtime

CURRENT:
Ollama + static model registry/flags.

TARGET:
resource/qualification/admission manager for GPU/CPU/NPU and replaceable models.

QUESTION:
should first runtime manager simply wrap Ollama/current API rather than replace it?

### Authority

CURRENT:
Tool Gate + policy bridge + approval + capability + scoped autonomy + legacy bridge.

TARGET:
durable deterministic authority, workload identity, final-hand capability.

QUESTION:
what internals can evolve behind current Tool Gate API so dozens of consumers do not change together?

### Hands

CURRENT:
executor + actuator registry + 34 handlers + browser/notify/vault/backup/broker/calendar/etc.

TARGET:
durable workflow + privilege-separated actuators + egress controls + independent verification.

QUESTION:
how should workflow state wrap current registry rather than create a parallel actuator system?

### Health / diagnosis

CURRENT:
heartbeat, Supervisor, House Doctor, Doctor teammate, resilience library, verifier/fusion, sysmetrics/Docker/introspection.

TARGET:
Telemetry → Structure Graph → Diagnosis → Contingency → Authority → Recovery → Verification.

QUESTION:
how to consolidate without a new Doctor and without deleting useful current health/recovery functions?

### Growth

CURRENT:
skill-hunter, Dream, Agent-Evolver, curiosity, probation/error disable, workspace manager.

TARGET:
candidate → sandbox → evidence → approval → probation → release → rollback.

QUESTION:
how to wrap existing skills/evolution outputs in this lifecycle?

### Backup / lifetime continuity

CURRENT:
backup-service + sovereign hardening + release/evidence pieces.

TARGET:
backup manifests, off-device copies, restore drills, lineage verification, hardware migration.

QUESTION:
what metadata wrapper can make current backups lineage-aware before redesigning storage?

### Finance

CURRENT:
financial-awareness + broker bridge + paper trading + strategy/opportunity modules.

TARGET:
financial observation/analysis/proposal separated from execution; later operating-runway/sustainability layer.

QUESTION:
how to preserve existing finance work while preventing it becoming Kai's self-preservation authority?

### Operator UI

CURRENT:
Dashboard + Grafana + PM/UH status/evidence + notifications/interfaces.

TARGET:
Mission Control.

QUESTION:
how to evolve Dashboard and machine status source rather than build a separate control room?

---

## 9. Current capabilities that MUST NOT disappear in architecture simplification

### Inner life / relationship

- emotional memory
- mood arcs
- self-reflection
- epistemic humility
- confession/mistake surfacing
- narrative identity / autobiography / legacy time capsules
- imagination / theory of mind / counterfactual thinking
- conscience / confirmed values
- gratitude / relationship continuity
- operator model
- cognitive fingerprint
- Obsidian Brain

These are identity/relationship/learned-state capabilities, not security authority.

### Cognitive depth

- Socratic questioning
- hypothesis engine
- temporal projection
- dialectical synthesis stub
- analogical reasoning stub
- concept blending stub
- synthetic experience stub
- transitive reasoning stub
- causal world model
- policy memory
- Global Workspace
- reasoning FSM
- persistent teammates
- adversary
- conviction
- swarm/reputation

These should become specialist modules/roles, not dozens of new services.

### Proactivity depth

- world context injection
- proactive observer
- anomaly detection
- cross-sensor correlation
- pattern learning
- proactive scheduling
- ritual discovery
- capability gap logging
- monitor rules
- screen watcher
- system watchers

### Growth

- skill hunter
- skill provenance/probation
- Agent-Evolver
- Dream
- curiosity/hypothesis research
- workspace manager

### Interfaces

- dashboard
- voice/wake/STT/TTS
- notifications
- avatar
- Telegram
- browser
- documents/files/vault

Simplification may change implementation boundaries. It may not silently delete product capability.

---

## 10. Missing shims we currently think are required

Review each and identify missing ones.

### M01 Contract v1→v2 adapters

Existing services cannot all change contracts in one commit.

Need bounded dual-schema compatibility and explicit retirement.

### M02 EventJournal backend adapter

Keep current EventJournal API/semantics while shadowing candidate durable backend; compare order/digest/replay before reader cutover.

### M03 World-State consumer compatibility projections

Generalise the existing Cortex adapter pattern for other point-to-point consumers.

### M04 Tool Gate compatibility façade

Keep current routes while authority internals become durable/cleaner. Do not create competing Authority API unless isolation requires it.

### M05 Shared-token → signed workload identity dual stack

Legacy auth may remain compatibility/membership while signed identity takes identity-sensitive operations; measure coverage then retire identity use of shared token.

### M06 Actuator registry → durable workflow adapter

Workflow should wrap current 34-actuator catalogue/handlers and preserve migration verification.

### M07 Ollama → Model Runtime Manager adapter

First Runtime Manager backend manages the current Ollama path/API; later add other runtime adapters.

### M08 memu compatibility façade

Keep current consumer API while separating authoritative memory, derived vector/graph, archival/compression/identity responsibilities internally.

### M09 Proactivity consolidation adapter

Normalize existing observer/monitor/Cortex/calendar/anomaly/ritual/Supervisor signals into Watch/Goal/Attention semantics; shadow decisions before retiring loops.

### M10 Supervisor split façade

Keep existing endpoints while health/recovery and proactive attention ownership are separated internally.

### M11 House Doctor structured-diagnosis adapter

Translate existing rule results into evidence-bound Diagnosis/Differential/BlastRadius/Contingency candidate schema.

### M12 Dashboard → Mission Control adapter

One machine current-state model feeding current Dashboard/new panels; no second truth/dashboard.

### M13 Backup → Lineage Manifest wrapper

Add exact release/schema/store/hash/restore metadata around existing backup operations.

### M14 Finance separation adapter

Preserve finance modules while enforcing Observation→Proposal→Policy→Execution→Verification semantics and separate operating/family assets.

### M15 Growth/release bridge

Existing skill/evolver outputs become CandidateCapability records entering sandbox/probation/release lifecycle.

### M16 Telemetry adapter

Normalize current heartbeat/metrics/sysmetrics/Prometheus/introspection incrementally; do not rewrite every service merely to adopt OpenTelemetry.

---

## 11. Proposed evolution sequence

This is architecture migration dependency order only. Current programme gates still control when implementation is allowed.

### E0 — exact current component/connection census

Map all services/modules/readers/writers/ports/networks/volumes/flags/direct-action paths/current status.

NO REFACTOR.

### E1 — reconcile minimal/full/sovereign topology

Create one canonical component/deployment registry and explicit profiles.

Retain best sovereign hardening.

### E2 — finish existing Unified Hunter cutover

Preserve current sequence:

1. perception active soak;
2. World State/Cortex source cutover;
3. scoped grants/disagreement evidence;
4. autonomy enforcement only when ready;
5. activate final-hand actuator path by risk tier;
6. prove old direct paths closed;
7. retire legacy polling only after soak.

### E3 — persist current authority/workflow

Add durable backing behind existing APIs; wrap existing actuator dispatch with workflow state; do not change all callers.

### E4 — consolidate World State / memory consumers

Migrate point-to-point reads one consumer at a time; clarify authoritative vs derived memory.

### E5 — consolidate proactivity

Add Goal/Watch/Attention semantics over current detectors; shadow and measure usefulness before retiring loops.

### E6 — professionalise current cognition/model runtime

Keep working cognitive modules; unify orchestration; wrap current model runtime; activate stubs only after prerequisites.

### E7 — unify health/Doctor/recovery

Structure graph + normalized telemetry + structured Doctor + qualified contingency + normal authority/workflow recovery.

### E8 — evolve backup into continuity/lineage

Manifest wrappers, restore drills, off-device copies, migration proof.

### E9 — add long-horizon sustainability/succession layers

Use existing finance/identity/authority; no premature autonomous money/succession.

### E10 — evolve Dashboard/docs into Mission Control and professional repo release structure

Machine-derived current truth, current→target map, S0–S5 maturity, risks/unknowns, exact subjects.

---

## 12. Review questions

### Existing-system fit

1. Which target responsibilities already have an adequate current home?
2. Which v0.2 “new boxes” should disappear because an existing component can evolve?
3. Which current services should become modules/libraries?
4. Which current modules/services need stronger process isolation?
5. Which current capability has no target home yet?
6. Which current capability is represented twice/three times and should be consolidated?

### Shims/cutover

7. Are E01–E08 existing migration shims conceptually sound?
8. Which of M01–M16 are unnecessary?
9. Which missing shim have we overlooked?
10. Where can dual authority arise during cutover?
11. Which old paths must be proven unreachable versus merely authenticated?
12. What evidence should each cutover require?
13. Where is rollback likely to restore weaker legacy authority?

### Deployment

14. How should minimal/full/sovereign become one canonical topology/profile model?
15. Which Docker networks should remain distinct?
16. Where are network/service boundaries currently accidental rather than justified?
17. Where should sovereign security controls become common defaults?

### Perception/world/memory

18. Can existing perception spine/EventJournal/WorldState become production foundations without replacement?
19. What backend/storage changes are actually necessary?
20. How should legacy point-to-point consumers migrate?
21. How should memu/graph/Letta/Obsidian/compression responsibilities be rationalized without losing memory capability?

### Proactivity

22. Is Goal/Obligation/Watch/Attention a useful semantic layer over existing proactive mechanisms?
23. Should it be a new service, an agentic module, a world-state subsystem, or another existing home?
24. What existing loops should remain detectors versus be retired after consolidation?
25. How do we test proactivity usefulness/anti-spam?

### Cognition/model runtime

26. Which current agentic FSM/swarm/teammate/GlobalWorkspace functions should form the final Hunter?
27. What is duplicate orchestration versus complementary cognitive layer?
28. How should Runtime Manager wrap existing Ollama first?
29. Which GPU-era stubs should remain future dormant modules rather than be removed?

### Authority/hands

30. How should Tool Gate/policy_bridge/autonomy/actuator registry be consolidated without changing dozens of clients at once?
31. What durable authority/workflow state is missing?
32. What final-hand enforcement is still needed?
33. What current shared-token routes need signed identity first?
34. Which actuator families need isolation beyond existing service boundaries?

### Doctor/resilience

35. How should Supervisor/House Doctor/Doctor teammate/resilience/verifier/fusion divide roles?
36. What should remain automatic containment versus approval-required recovery?
37. What current health/watch signals should populate a generated dependency graph?

### Continuity

38. What is the smallest Lineage Manifest wrapper that makes current backups useful for long-horizon restore proof?
39. What must remain outside the main machine/failure domain?
40. What current identity/narrative/release artifacts should contribute to lineage?

### Simplification

41. Simplify current+target architecture by 30% through **boundary consolidation**, not capability deletion.
42. Name every service you would merge and why.
43. Name every new proposed service you would refuse to create and where its logic belongs instead.
44. Name any existing service whose isolation is so valuable it should definitely remain separate.

---

## 13. Required finding format

For every material recommendation:

```text
ID:
SEVERITY: BLOCKER / MAJOR / MINOR / QUESTION
CURRENT COMPONENT(S):
CURRENT ROLE / PATH:
WHAT MUST SURVIVE:
PROBLEM:
PROPOSED REWORK / SPLIT / MERGE:
SHIM / ADAPTER:
TARGET HOME:
NEW CODE REQUIRED:
NEW SERVICE REQUIRED? YES/NO + WHY:
SHADOW / SOAK:
CUTOVER TEST:
LEGACY RETIRE CONDITION:
ROLLBACK RISK:
CONFIDENCE:
```

---

## 14. Required final response

Finish with:

1. `OVERALL VERDICT — EVOLUTION PLAN APPROVE / APPROVE WITH CHANGES / REJECT`
2. `WHAT WE ALREADY HAVE THAT SHOULD NOT BE REBUILT`
3. `MISSING CURRENT COMPONENTS / CAPABILITIES IN OUR MAP`
4. `SHIMS TO KEEP`
5. `SHIMS WE ARE MISSING`
6. `SERVICES TO KEEP SEPARATE`
7. `SERVICES TO MERGE INTO MODULES / OTHER ORGANS`
8. `TARGET RESPONSIBILITY MAP — CURRENT → TARGET`
9. `MINIMAL/FULL/SOVEREIGN RECONCILIATION`
10. `UNIFIED HUNTER CUTOVER PLAN`
11. `PROACTIVITY CONSOLIDATION PLAN`
12. `MEMORY / IDENTITY CONSOLIDATION PLAN`
13. `COGNITION / MODEL-RUNTIME CONSOLIDATION PLAN`
14. `AUTHORITY / WORKFLOW / ACTUATOR CONSOLIDATION PLAN`
15. `DOCTOR / SUPERVISOR / RESILIENCE CONSOLIDATION PLAN`
16. `CONTINUITY / LINEAGE EVOLUTION PLAN`
17. `SIMPLIFY PROCESS BOUNDARIES BY 30% WITHOUT DELETING CAPABILITY`
18. `TOP 10 CUTOVER RISKS`
19. `TOP 10 DISCRIMINATING TESTS / SPIKES`
20. `PROPOSED REVISED EVOLUTION ORDER`
21. `FINAL QUESTIONS FOR DAINIUS`

Do not return a blank-sheet architecture. Return an **evolution of the existing Kai**.
