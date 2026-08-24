# KAI KINGSMAN — Existing-Kai Master Architecture Plan v0.4

> **STATUS: PRIMARY WHOLE-SYSTEM ARCHITECTURE / PROFESSIONALISATION CANDIDATE — REPO-GROUNDED, POST-DEEPSEEK/KAI RECONCILIATION — NOT FROZEN, NOT IMPLEMENTATION AUTHORITY, NOT PROGRAMME EXECUTION AUTHORITY.**
>
> **Subject basis:** branch `claude/project-rework-plan-pgvp35`; architecture state recovered through D349 and current recovery pointer at commit `8eb7fd4740c12c90e15c3631beecee564121d830`.
>
> This document is the first candidate intended to function as the **actual professional master plan** for evolving the existing Kai. It does not start from a blank page. It binds the accumulated product vision, the current repository, Unified Hunter's already-built migration machinery, existing shims, the Kingsman root mission/identity/resilience doctrines, DeepSeek's adversarial review, and Kai's repo-backed D349 reconciliation into one architecture and migration programme.
>
> **Governing law:**
>
> `CURRENT SUBJECT → INTENT → CURRENT REALITY → GAP → SHIM → TARGET RESPONSIBILITY → SHADOW/SOAK → VERIFIED CUTOVER → PROVE WEAKER PATH DEAD → RETIRE/REHOME → BANK EVIDENCE`.
>
> **No new box without current-to-target lineage. No capability deletion merely to simplify a diagram.**

---

# 1. Executive architecture decision

Kai is already a substantial organism. The programme does **not** need to build a new AI beside it. It needs to finish the migration and professionalisation of the organism that already exists.

The target is therefore:

> **ONE KAI — ONE QUALIFIED WORLD MODEL — ONE GOVERNED CONSEQUENTIAL-ACTION PATH — MANY REPLACEABLE COGNITIVE/SENSORY/EXECUTION ORGANS — PROACTIVE AWARENESS — BOUNDED FAILURE — VERIFIED LEARNING — OPERATOR-VISIBLE TRUTH — PRESERVED LINEAGE — LONG-HORIZON STEWARDSHIP.**

The architecture is deliberately **logical first, physical second**. A logical responsibility does not automatically become a Docker service.

The core engineering style is:

- preserve working capabilities;
- recover original intent before changing implementation;
- consolidate duplicate responsibility, not useful capability;
- preserve stable interfaces while internals migrate;
- use compatibility shims and shadow/soak rather than big-bang replacement;
- strengthen authority and truth at the actual enforcement point;
- prove old weaker paths dead before calling migration complete;
- isolate failures where isolation earns its cost;
- keep the system simple enough for a private long-lived organism to remain maintainable;
- design growth so future models/hardware/backends can replace organs without architectural amnesia.

---

# 2. Root purpose and architectural identity

## 2.1 Primary mission

Kai exists to grow with Dainius, preserve continuity, proactively assist/challenge/protect/care within earned authority, develop controlled capabilities, remain technically and economically sustainable, survive component/hardware/provider replacement, and eventually survive beyond the original operator through explicit succession/governance while preserving intended stewardship for his daughter.

This root purpose is not a feature backlog item. It determines the architecture.

## 2.2 Kai is the organism

`KAI ≠ Kimi / DeepSeek / GLM / Dolphin / Ollama / CrewAI / Unified Hunter / memu / House Doctor / one machine / one repo snapshot`.

Models, services, databases, frameworks and hardware are organs/substrates.

Working identity:

`KAI = MISSION + IDENTITY/LINEAGE + MEMORY/CONTINUITY + QUALIFIED WORLD STATE/EVIDENCE + COGNITION + RELATIONSHIPS/VALUES + GOVERNANCE/AUTHORITY + CAPABILITIES + LEARNING/HISTORY`.

## 2.3 Three change classes

### Core invariants — high-authority change only

- mission;
- identity/lineage principles;
- truth/evidence semantics;
- operator sovereignty/stewardship;
- authority separation;
- UNKNOWN/non-fabrication rules;
- protected family-asset boundary;
- controlled self-development.

### Evolvable organs

- models/model runtimes;
- memory backends;
- sensors;
- cognitive specialists;
- databases/indexes;
- tools/actuators;
- diagnostic mechanisms;
- workflow implementation;
- external providers;
- hardware.

### Learned state

- memories;
- relationship history;
- operator-confirmed preferences/values;
- skills;
- trust/calibration;
- incident lessons;
- behavioural adaptation;
- standing instructions.

---

# 3. Architectural constitution

These are the candidate non-negotiable laws for master-canon freeze.

1. **One Kai / one governed consequential-action path.**
2. **Truth outranks fluency.** UNKNOWN, UNOBSERVED, STALE, CONFLICTING and DEGRADED are real states.
3. **Observation ≠ evidence ≠ current fact ≠ memory ≠ proposal.**
4. **Present ≠ executed ≠ enforced.**
5. **Models reason/propose; they do not create authority.**
6. **Membership ≠ identity ≠ static scope ≠ one-time execution authority ≠ autonomy delegation.**
7. **Manual operator-approved action and autonomous initiation are separate authority lanes.**
8. **Exact one-time capability must be enforced at the actual side-effecting hand.**
9. **A central dispatcher consuming authority is insufficient if downstream direct mutation can bypass it.**
10. **No silent authority fallback.**
11. **No silent truth fallback.** Cold-start compatibility may degrade explicitly; steady-state canonical-path failure must not silently restore an old truth source.
12. **Independent outcome verification is separate from actuator receipt.**
13. **Failure stops at the narrowest safe boundary.**
14. **Fallback that hides failure is not resilience.**
15. **New service/process boundaries must be earned by security, failure, resource, lifecycle or scaling needs.**
16. **Historical capability is preserved until intent and current reality are qualified.**
17. **A stub may represent retained future intent; absence of current implementation quality does not erase the concept.**
18. **Operator legibility is part of governance.**
19. **Every green status, arrow or tick is an evidence-bearing claim.**
20. **Self-sufficiency serves stewardship and never becomes unrestricted self-preservation.**
21. **Temporary operator silence is not succession.**
22. **Build/assurance mechanisms may become platform primitives only after semantic requalification.**
23. **Programme execution order is separate from architecture dependency order.**
24. **No real known defect becomes optional because it is outside the local task.**
25. **Multiple analytical minds; one evidence standard; Kai reconciles architecture; Dainius retains final project authority.**

---

# 4. Current Kai — existing organism to preserve and professionalise

The repository already contains three overlapping realities: capability population, multiple deployment definitions and a built-but-not-default-cut-over Unified Hunter migration layer.

## 4.1 Existing capability families

### Core/control/data

PostgreSQL/pgvector, Redis, Tool Gate, memu-core, memu introspection, agentic, heartbeat, Dashboard, Ollama/runtime setup.

### Perception/world awareness

audio, camera, vision, wake, screen capture/watcher, clipboard, files/doc parsing, email/news/weather/air-quality/calendar, git/docker/system watchers, Cortex, monitor-service, proactive observer.

### Memory/identity/continuity

memu hot path, introspection/maintenance, memu-graph, pgvector/TurboVec paths, Letta archive, memory compressor, Obsidian/vault sync, emotional memory, narrative identity, operator model, cognitive fingerprint, relationship/history data.

### Cognition

agentic reasoning FSM, Socratic mechanisms, Scout/Sage/Doctor/Oracle, swarm/reputation/conflict resolver, adversary, conviction, hypothesis, temporal/causal reasoning, Global Workspace concepts, dormant GPU/graph-era higher-cognition modules, Kai Advisor.

### Hands/outputs

executor, browser-agent, notify, TTS, avatar, Telegram, file/vault/calendar actions, backup/recovery, broker/paper trading, 34 registered actuator identities across 8 risk tiers.

### Health/diagnosis/resilience

heartbeat, Supervisor, House Doctor, cognitive Doctor teammate, common resilience primitives, verifier, fusion, metrics, introspection, system/Docker watchers, DEGRADED/RECOVERING FSM concepts.

### Growth/self-development

skill-hunter, workspace manager, Agent-Evolver, Dream/introspection cold path, skill provenance/probation/disable, curiosity/hypothesis research, capability-gap and ritual discovery.

### Finance/sustainability seeds

financial-awareness, broker bridge, paper-trade vertical slice, market/strategy/opportunity analysis.

### Security/operations

minimal/full/sovereign Compose variants, network segmentation, shared service auth, Ed25519 workload identity, Dashboard roles/scopes, Vault/rotation direction, Tailscale, Prometheus/Alertmanager/Grafana, gVisor/AppArmor executor hardening, CI/security gates, release/evidence/erasure machinery and backups.

**Product rule:** simplification may consolidate services/modules but must not silently delete these capabilities.

---

# 5. Current transitional control architecture — preserve and finish it

Unified Hunter UH-1→UH-8 is not a new proposal. It is an already-built migration skeleton behind legacy/default paths.

Existing built/tested responsibility chain:

`Perception → World State → Proposal-only Workspace → Policy / Approval / Capability → Actuator Registry → Observation / Verification → Learning`

plus scoped autonomy, legacy trust bridging, service authentication, erasure lineage, full actuator catalogue migration and dashboard/security remediation.

Current cutover shims include:

- `KAI_PERCEPTION_MODE=shadow|active`;
- `KAI_CORTEX_SOURCE=poll|world_state`;
- `KAI_AUTONOMY_ENFORCE=false|true`;
- actuator migration state machine + legacy verifier;
- shared service-auth compatibility + newer signed workload identity;
- Dashboard credentials/degraded envelope;
- feature flags/preflight/CI gates.

These shims are architectural assets and should be generalized, not discarded.

---

# 6. D349 correction — the authority chain must end at the actual hand

This is now a master-plan blocker, not a footnote.

## 6.1 Current implementation gap

Current central `ActuatorRegistry` requires/consumes an `ActionCapability` before it dispatches a handler. However the mutating handler then invokes the downstream service using parameters plus service authentication/workload signature. The downstream side-effecting service does not yet validate and atomically consume the exact one-use `ActionCapability` itself.

Current effective path:

`proposal/policy/approval → ActionCapability → central registry consumes → downstream authenticated route → side effect`.

Target path:

`proposal/policy/approval → one-use execution capability → workflow → downstream actuator validates exact capability at final hand → atomic consume → side effect`.

## 6.2 Correct authority lanes

### Manual/operator-approved action

`Proposal → deterministic policy → exact authenticated operator approval → exact one-use capability → final-hand consume → effect → independent verification`.

No autonomy grant is required merely because the action mutates something.

### Autonomous action

`Proposal → deterministic policy → valid scoped autonomy grant → exact one-use capability → final-hand consume → effect → independent verification`.

Autonomy is the right to **initiate within a bounded envelope**, not a substitute for one-use execution authority.

## 6.3 Required new shim: M17 Final-Hand Execution Capability

This is a genuinely missing joint attaching to existing Tool Gate/policy/actuator machinery.

Candidate requirements:

- exact actuator audience;
- exact operation/method/path;
- parameter/body digest;
- proposal/approval/workflow binding;
- expiry;
- nonce/consumption identity;
- presenter workload identity;
- one-time atomic consumption at actual actuator;
- no shared-token-only bypass;
- audit link to receipt/outcome.

This does **not** require a new “Capability Service” by default. It should evolve behind current Tool Gate/policy/capability interfaces and current actuator endpoints.

---

# 7. D349 correction — legacy closure must prove the weaker path is unusable

Current source-based verifier can classify some direct routes as closed after service authentication is added. That proves a meaningful security improvement, but it does not prove the target invariant.

`AUTHENTICATED DIRECT PATH ≠ FINAL-HAND CAPABILITY PATH ≠ LEGACY AUTHORITY PATH DEAD`.

Final migration closure must use both:

1. static/source/config proof of expected enforcement; and
2. runtime negative bypass proof showing the weaker direct path is rejected.

A migration flag, endpoint reachability, signed caller or shared-token requirement is not enough by itself.

---

# 8. D349 correction — authority state must become durable behind existing control APIs

Current approval/capability/autonomy records include process-local state. Current autonomy preflight creates a fresh authority object, so it cannot prove persistent runtime grant readiness.

Target evolution:

- retain Tool Gate/current policy APIs as compatibility facade;
- introduce durable transactional records for approval, grants, capabilities, revocation, consumption and workflow correlation;
- make restart/concurrency/replay semantics testable;
- migrate one state family at a time;
- do not introduce a parallel Authority service unless isolation evidence later earns it.

First production persistence candidate: existing PostgreSQL, with transactional operations and an outbox/event record where needed.

---

# 9. Logical final organism — responsibilities, not service mandates

## 9.1 Mission / identity / lineage

Preserves constitutional mission, operator authority lineage, release/migration identity and long-horizon continuity.

Current seeds:

- primary-mission/identity doctrine;
- narrative identity;
- operator model;
- release bundles;
- service identity;
- backups/history.

Genuinely missing joint:

- product-level Lineage Manifest/Registry and restore qualification semantics.

## 9.2 Perception / observation

Current seeds:

- sensor services;
- watchers;
- Cortex polling;
- perception adapters;
- shadow runner.

Target:

one typed/provenance-bound ingress into EventJournal/evidence/world state.

Keep acquisition services; migrate consumers, not hardware drivers.

## 9.3 Event / evidence / world state

Current seeds:

- EventJournal;
- PerceptionIngress;
- graded evidence;
- WorldStateSnapshot/conflict/freshness;
- release/evidence/erasure machinery.

Target:

single qualified current-state path with explicit UNKNOWN/STALE/CONFLICT/SOURCE_UNAVAILABLE.

Important change:

Cortex/world-state compatibility fallback becomes explicit `COLD_START` or `DEGRADED`, not silent steady-state old polling.

## 9.4 Memory / relationship / continuity

Current seeds:

memu-core, graph/vector stores, Letta, compressor, Obsidian/vault, emotional/narrative/operator/relationship data.

Target responsibility:

- authoritative memory records;
- derived retrieval/index projections;
- archival;
- human-readable mirror;
- maintenance/compression;
- relationship/identity learned state.

**Physical owner remains provisional until E0 reader/writer/state-owner census.** Do not declare memu-core authoritative merely by architectural preference until actual data flows are qualified.

## 9.5 Proactivity / goals / time / attention

Current seeds:

proactive observer, monitor rules, Cortex, calendar, anomaly/correlation, screen watcher, rituals, gap logging, Supervisor nudges, notifications.

Target semantic objects:

- Goal;
- Obligation;
- Commitment;
- Watch;
- Timer;
- AttentionCandidate.

Target decisions:

`IGNORE / STORE / WATCH / PREPARE / PROPOSE / NOTIFY / ACT IF ALREADY AUTHORISED`.

No new proactivity service by default. Agentic + World State is a plausible home but remains **provisional until E0**.

## 9.6 Cognitive workspace / Unified Hunter

Current seeds:

agentic FSM, teammates, swarm, adversary, causal/forecast/hypothesis modules, proposal-only workspace, Global Workspace/future cognitive modules.

Target:

one cognitive workspace that:

- frames tasks;
- retrieves world/memory;
- selects specialist roles;
- preserves disagreement;
- runs adversarial/fact/causal review;
- produces proposals/explanations only;
- cannot mint authority or directly execute.

Models remain replaceable role resources.

## 9.7 Model runtime/resource qualification

Current seeds:

Ollama + current model registry/flags.

Near-term:

add exact model identity/digest/runtime/resource/qualification data to current serving path.

Do **not** create a wrapper service simply to satisfy the label “Model Runtime Manager”.

Future responsibility becomes a full runtime manager only when multi-runtime, memory admission, load/unload/preemption or hardware scheduling becomes a real measured need.

## 9.8 Values / constraints / policy / authority

Current seeds:

Ohana/conscience/preferences, Tool Gate, policy bridge, approval, capability, autonomy, LegacyTrustBridge, signed identity.

Target:

one deterministic control responsibility behind compatible current APIs:

- verified workload identity;
- deterministic constraints/policy;
- authenticated operator approvals;
- scoped autonomy delegations;
- exact one-use execution capabilities;
- durable revocation/consumption/audit.

No second independent policy/autonomy system.

## 9.9 Workflow / execution / hands

Current seeds:

executor, ActuatorRegistry, 34 actuator identities, browser/notify/files/calendar/backup/broker/etc. handlers.

Target:

Postgres-backed durable workflow/outbox as first candidate around the existing registry:

`WorkflowRecord → outbox/fencing → existing handler → final-hand capability consume → receipt → verifier`.

No Temporal/NATS/Kafka is justified now; they remain future options if measured complexity later earns them.

## 9.10 Egress / target authority

Current seeds:

network segmentation, browser-agent, egress-net, Tool Gate/policy/capability.

Missing joint:

explicit target constraints for network-capable hands.

Add target/domain/method/classification/budget constraints to policy/exact capability and enforce again at the actual hand. No separate egress service by default.

## 9.11 Verification / learning

Current seeds:

verifier, verifier registry, fusion, graded evidence, paper-trade reconciliation, release evidence.

Target:

`ActuatorReceipt` remains only execution evidence; a target-specific independent observation produces `VerifiedOutcome`; only qualified outcomes feed learning/trust.

## 9.12 Health / diagnosis / recovery

Current seeds:

heartbeat, metrics, watchers, Supervisor, House Doctor, Doctor teammate, resilience library, verifier/fusion, FSM.

Target responsibility split:

- telemetry/health = observations;
- component/dependency/authority graph = structure;
- House Doctor/Future A4 = structured diagnosis;
- Doctor teammate = cognitive/interactive diagnostic specialist;
- contingency library = qualified response knowledge;
- Supervisor = narrow service/recovery execution coordinator;
- policy/workflow/actuator = authority and hands;
- verifier = recovery outcome truth.

No third Doctor and no self-authorising repair agent.

## 9.13 Growth / evolution / skills

Current seeds:

skill-hunter, Dream, Agent-Evolver, curiosity, probation/disable, workspace manager.

Target:

`CandidateCapability → sandbox/tests → evidence → approval/release → probation → promotion → monitor → rollback/retire`.

Existing skills enter the lifecycle through metadata/adapters; no rewrite solely for compliance.

## 9.14 Backup / continuity / lineage

Current seeds:

backup-service, sovereign hardening, release/evidence, databases/memory stores.

Target:

existing backup + Lineage Manifest wrapper + off-device/offline copy + isolated restore drills + hardware migration qualification + provider/EOL/credential watches.

## 9.15 Financial sustainability / long horizon

Current seeds:

financial-awareness, broker/paper-trade, market/strategy/opportunity modules.

Target separation:

`Financial Observation → Analysis/Proposal → Policy/Risk → exact capability → execution → reconciliation`.

Long-horizon operating-cost/runway planning consumes these capabilities but does not create a “survival agent”. Protected operator/family assets remain a separate trust domain.

## 9.16 Operator Mission Control

Current seeds:

Dashboard, Grafana, UH tracker, PM/evidence files, notifications, voice/interfaces.

Target:

one machine-state model feeding an evolved Dashboard/Mission Control. No second dashboard/truth source.

---

# 10. Physical / process boundary candidate

The final physical layout must be derived from E0/E1, not imposed upfront. However the following boundary classes are currently justified candidates.

## Definitely separate / independently protected candidates

- PostgreSQL/authoritative data;
- Redis/cache/ephemeral coordination where retained;
- Tool Gate/control boundary;
- model host (Ollama/current runtime);
- privileged browser/executor classes;
- hardware-facing camera/audio/wake where device/runtime isolation warrants;
- Vault/secrets when sovereign controls are active;
- backup execution/restore environment;
- independent verification where real independence requires separate failure/credential group.

## Likely module/consolidation candidates pending E0

- git/docker/sysmetrics/screen/monitor watcher logic into observer/telemetry adapters rather than one service per simple watcher;
- Scout/Sage/Oracle/Advisor as cognitive specialist modules, not separately sovereign agents;
- skill-hunter/Evolver/Dream/curiosity/gap logic as growth modules around one release lifecycle;
- memory compressor/vault sync as memory/continuity maintenance modules;
- financial analysis subcomponents under finance capability family;
- output adapters (notify/TTS/avatar/Telegram) may share orchestration while preserving channel-specific connectors/credentials;
- verifier/fusion code may share a verification framework while verifier independence groups remain logically enforceable.

**No merge is authorised until E0 proves current consumers/state/failure boundaries.**

---

# 11. Canonical deployment topology strategy

Current `minimal`, `full` and `sovereign` Compose files have drifted into partly different service populations.

Target E1:

one machine-readable component registry with fields such as:

- component ID;
- implementation path;
- role/responsibility;
- profile membership;
- networks;
- dependencies;
- state ownership;
- credentials/secrets;
- privileges;
- health source;
- current maturity/status;
- lifecycle owner;
- current evidence subject.

Then represent:

- `minimal` = daily-driver baseline profile;
- `full` = baseline + heavy/optional capability set;
- `sovereign` = security/hardening overlay/profile, not a competing architecture tree;
- optional hardware/model capabilities = explicit profiles.

Implementation choice remains open between generated Compose output and base+override profiles. The invariant is **one component truth**, not a specific YAML generator.

---

# 12. Machine-readable current architecture census — E0

DeepSeek correctly demanded a machine census, but the implementation must reuse existing House/Census/security/reporting machinery rather than start another inventory project.

E0 target output:

`CURRENT_KAI_COMPONENT_DEPENDENCY_AUTHORITY_MAP`.

Required dimensions:

- services/modules/files;
- Compose/profile membership;
- ports/networks/volumes;
- routes/endpoints;
- readers/writers/state owners;
- direct side-effect paths;
- feature flags/shims;
- service identity/auth mode;
- authority/capability path;
- consumers/callers;
- health source;
- status: LIVE / PRESENT-NOT-CUT-OVER / STUB / HISTORICAL / UNKNOWN;
- exact repo tree/evidence subject.

E0 is read/measure/map work, not refactor.

---

# 13. Current + missing migration shims

## Existing shims to keep

- E01 perception shadow/active runner;
- E02 Cortex World-State compatibility projection;
- E03 LegacyTrustBridge;
- E04 Actuator migration driver + legacy verifier;
- E05 service-auth→signed-workload-identity transition;
- E06 Dashboard credential/degraded-response compatibility;
- E07 feature flags/preflight controls;
- E08 release/evidence/verification/erasure foundations.

## Corrected/new joints

### M01 Contract compatibility adapters

Bounded v1→future-v2 migration only where semantic change requires it; do not invent a full schema service before need.

### M02 EventJournal durable-backend adapter

Preserve EventJournal interface; compare order/digest/replay before switching backend. Broker only if measured requirements later demand it.

### M03 World-State consumer projections

Generalize Cortex adapter pattern; explicit COLD_START/DEGRADED states; no silent steady-state old truth.

### M04 Tool Gate compatibility facade

Persist/harden authority behind current callers instead of a parallel Authority API.

### M05 Shared-token→verified workload identity

Finish class-B migration; retain existing timestamp/body/path/destination/nonce/replay/revocation protections; do not reimplement them.

### M06 ActuatorRegistry→DurableWorkflow

Postgres WorkflowRecord/outbox/fencing around current registry and handlers.

### M07 Current model registry enrichment

Add exact model/runtime/resource/qualification identity now; defer separate Runtime Manager process until earned.

### M08 memu compatibility/ownership tracing

Trace source/derived/archival/mirror/maintenance roles before physical consolidation.

### M09 proactivity semantic adapter

Normalize existing detectors into Watch/Goal/Attention candidates; shadow old/new decisions.

### M10 Supervisor responsibility split

Preserve interface while separating health/recovery from operator-attention decisions.

### M11 House Doctor structured-diagnosis adapter

Translate current heuristic outputs into evidence-bound diagnosis/differential/blast-radius/contingency candidates.

### M12 Dashboard→Mission Control machine-state adapter

One derived operator truth model; legacy panels migrate gradually.

### M13 Backup→Lineage Manifest wrapper

Keep backup operations; add subject/hashes/schema/store/offset/key/restore qualification metadata.

### M14 finance separation contracts

Observation→analysis/proposal→policy→capability→execution→reconciliation with protected-asset boundaries.

### M15 growth/release bridge

Existing skill/evolver outputs enter CandidateCapability/probation/release lifecycle.

### M16 telemetry normalization adapter

Reuse heartbeat/sysmetrics/Prometheus/introspection; introduce OpenTelemetry-compatible semantics incrementally only if useful.

### M17 Final-Hand Execution Capability — NEW MANDATORY

Propagate/derive exact one-use execution authority to actual actuator and atomically consume before side effect.

### M18 Runtime Legacy-Bypass Probe — NEW MANDATORY

Automated negative tests prove retired weaker routes cannot mutate through direct/shared-token/bypass paths.

### M19 Durable Authority State Adapter — NEW MANDATORY

Persist approval/grant/capability/revocation/consumption state behind current Tool Gate-compatible API.

### M20 Scoped-Autonomy Grant Bootstrap — NEW MANDATORY

Authenticated operator-governed initial/widening grant flow; no self-grant. Future bounded renewal is a separate later design problem.

### M21 Egress/Target Constraint Adapter — NEW MANDATORY

Add destination/operation/data/budget constraints to policy and exact execution capability; enforce at network-capable hand.

### M22 Evidence-Bound Migration Record — NEW

Keep flags simple. Record `SHADOW / QUALIFIED / CANARY / ACTIVE / RETIRED`, evidence subject, soak, rollback and legacy-closure proof outside the environment flag itself.

---

# 14. Evolution programme — exact architecture dependency plan

This is a dependency plan for future professionalisation after current governed prerequisites permit implementation. It does **not** reorder House/048/Item8/A-4 programme authority.

## E0 — Current machine census / connection map

**Goal:** know the actual organism before surgery.

Work:

- reuse/extend House/Census + security/compose/route scanners;
- enumerate components/routes/dependencies/readers/writers/state owners/side effects/auth modes/shims;
- classify live/present/stub/historical/unknown;
- bind exact repo subject;
- produce initial Mission Control architecture inventory.

Exit:

closed-enough denominator and no major architecture component whose current role is unknown.

## E1 — Canonical deployment/profile model

**Goal:** minimal/full/sovereign become profiles of one architecture.

Work:

- reconcile component sets;
- define one component registry;
- preserve network/trust/hardening semantics;
- identify accidental services vs earned isolation;
- choose generated Compose or base/overlays after spike.

Exit:

all deployment variants derive from one declared component truth.

## E2 — Complete UH truth-path cutover safely

**Prerequisites:** E0 map for perception/Cortex consumers.

Sequence:

1. retain shadow ingestion;
2. qualify active reduction;
3. make World-State fallback explicit COLD_START/DEGRADED;
4. cut consumers individually through projections;
5. soak/compare;
6. prove no steady-state legacy polling is load-bearing;
7. retire old polling only after negative proof.

Exit:

one qualified current-state path for migrated consumers.

## E3 — Complete identity / authority / final-hand chain

**This is the critical security/control work package.**

Sequence:

1. inventory remaining shared-token-only mutating/class-B routes;
2. finish verified workload identity migration without duplicating existing nonce/timestamp/replay machinery;
3. persist approvals/grants/capabilities/revocations/consumption;
4. implement M20 authenticated autonomy grant bootstrap;
5. implement M17 exact final-hand execution capability;
6. implement M21 egress/target constraints for network-capable hands;
7. implement M18 runtime bypass negative tests;
8. requalify actuator risk tiers;
9. retire weaker routes only after static + runtime proof.

Exit:

`membership → identity → policy/approval/autonomy → exact one-use capability → final hand → verified outcome` is enforceable with no direct weaker bypass for migrated actions.

## E4 — Durable workflow around existing hands

Sequence:

- WorkflowRecord/Postgres state;
- transactional outbox/dispatch;
- fencing/idempotency/unknown-outcome reconciliation;
- existing ActuatorRegistry handlers retained;
- receipts linked to capability/workflow;
- independent verifiers linked to expected outcome;
- crash/restart/duplicate tests.

Exit:

restart-safe consequential workflow without parallel executor framework.

## E5 — World State / memory ownership consolidation

Sequence:

- use E0 read/write data;
- determine authoritative memory records vs derived projections;
- migrate direct sensor/current-state reads behind World-State projections;
- preserve graph/vector/Letta/Obsidian/compressor functions;
- introduce compatibility facade only where needed;
- remove duplicate truth after consumers migrate.

Exit:

one current-state truth path and explicit memory-source/projection ownership.

## E6 — Proactivity / goals / attention consolidation

Sequence:

- define smallest semantic schema starting Watch/Timer;
- map existing monitor/Cortex/calendar/anomaly/screen/Supervisor/proactive-observer outputs;
- shadow decision comparator;
- extend to Obligation/Commitment/Goal after evidence;
- measure usefulness, timeliness, misses and spam;
- retire duplicate loops only after better/equivalent behaviour proven.

Exit:

proactivity is durable and coherent without a second orchestrator.

## E7 — Cognition / model resource professionalisation

Sequence:

- map every cognitive module to role/maturity/dependencies;
- preserve dormant future modules;
- remove duplicated orchestration only after mapping;
- enrich model registry with exact artifact/runtime/resource/qualification fields;
- measure Strix Halo model residency, context/KV, latency, power;
- introduce stronger runtime management only when measurements show need.

Exit:

models are replaceable role-qualified organs and cognition remains proposal-only.

## E8 — Health / Doctor / contingency / recovery

Sequence:

- normalize current health signals;
- generate Component/Dependency/Authority graph from E0/component registry;
- structure House Doctor outputs;
- build qualified contingency records from existing resilience/recovery knowledge;
- keep Supervisor narrow;
- route recovery through normal authority/workflow/final-hand path;
- fault-inject major organs and verify blast radius/recovery truth.

Exit:

self-diagnosis can explain failure, propose containment/recovery and independently prove outcome without self-authority.

## E9 — Continuity / lineage / restore

Sequence:

- wrap current backup with Lineage Manifest;
- identify authoritative stores/RPO/RTO;
- off-device/offline copy;
- isolated restore drills;
- hardware migration rehearsal;
- key/provider/credential/EOL watches;
- preservation/read-only mode.

Exit:

restored/migrated Kai can prove intended lineage and authority state, not merely boot.

## E10 — Sustainability / succession scaffolding

Sequence:

- operating-cost/runway state;
- renewal obligations;
- proposal-only sustainability planner using existing finance data;
- operating-capital/family-asset boundaries;
- succession state model/legal/human dependencies;
- no automatic succession/financial expansion.

Exit:

long-horizon risks visible and architecturally bounded without premature autonomy.

## E11 — Mission Control / docs / professional release structure

Sequence:

- one machine operator-state schema;
- evolve Dashboard/Grafana/PM views;
- full current architecture + target overlay;
- programme phase, risks, approvals, degradation, continuity, maturity;
- evidence-bound ticks;
- README/docs generated/derived where appropriate;
- historical architecture supersession/index;
- branch/release/attestation/professional repo consolidation;
- final long-duration/fault/security qualification.

Exit:

first S5 Kingsman-compliant production baseline.

---

# 15. Migration gate template

Every material migration must answer and prove:

1. exact current subject;
2. original intent;
3. current maturity/status;
4. readers/writers/callers/consumers;
5. state ownership;
6. authority currently held;
7. failure domain/dependents;
8. target responsibility;
9. compatibility shim;
10. new code genuinely required;
11. whether a new process boundary is earned;
12. shadow/dual-read comparison;
13. positive/negative/boundary/adversarial tests;
14. soak/current runtime evidence where required;
15. state migration integrity;
16. exact new authority path;
17. runtime proof weaker old path is dead;
18. rollback cannot silently restore weaker authority;
19. operator view/docs updated;
20. exact evidence/commit/tree banked;
21. unrelated known defects remain tracked.

Allowed dispositions:

`RETAIN / REWORK / SPLIT / MERGE / REHOME / SUPERSEDE / ARCHIVE-HISTORICAL / DELETE / UNKNOWN-MORE-EVIDENCE`.

There is no disposition called forgotten.

---

# 16. Fault-containment and degraded-operation requirements

Every major organ/component family needs:

- criticality;
- dependencies;
- dependents;
- timeout/budget;
- retry policy;
- circuit-breaker/degraded behaviour;
- safe rollback;
- health signal;
- recovery authority requirement;
- independent verification;
- operator-visible status;
- intended blast radius.

First-class states:

`HEALTHY / DEGRADED / RECOVERING / UNAVAILABLE / QUARANTINED / UNKNOWN-UNMEASURED`.

Examples:

- one specialist model down → missing viewpoint explicit; cognition continues if safe;
- memory down → reduced-context mode, no invented continuity;
- sensor/provider down → dependent claim UNKNOWN/unavailable, unrelated cognition continues;
- House Doctor down → diagnosis degraded, not “healthy”;
- authority down → consequential actions fail closed, cognition remains available;
- verifier down → outcome remains unverified;
- actuator unknown outcome → reconcile before retry;
- PostgreSQL down → only explicitly designed degraded/read-only functions continue;
- Dashboard down → Kai core may continue, but operator governance visibility is degraded and high-risk approvals may become unavailable.

---

# 17. Operator Mission Control information architecture

Mission Control is the human governance surface for the whole organism.

## View A — Whole Kai

- root mission/identity;
- current organs/components;
- current status: LIVE / PRESENT-NOT-CUT-OVER / STUB / HISTORICAL / UNKNOWN;
- S0–S5 maturity;
- health/degradation;
- exact evidence subject;
- target overlay and migration stage.

## View B — Programme / migration

- current governed programme phase;
- current architecture work package;
- authorised next action;
- blocked/unauthorised work;
- current branch/commit/tree;
- closure evidence.

## View C — Decisions / attention / autonomy

- pending operator approvals;
- active autonomy grants and expiry/scope;
- prepared proposals;
- active Watches/Obligations;
- suppressed/deferred items;
- policy denies.

## View D — Resilience / authority

- active incident/degradation;
- impacted components;
- expected blast radius;
- blind observers;
- contingency/recovery state;
- final-hand authority status;
- runtime bypass-test status;
- independent verification result.

## View E — Continuity / lifetime

- backup age;
- latest isolated restore result;
- lineage manifest/release identity;
- key/credential expiry;
- provider/EOL risks;
- hardware state;
- operating runway;
- succession readiness state.

Mission Control must derive volatile status from qualified machine data; Markdown remains governance/design evidence, not the live source of truth.

---

# 18. Maturity model

`S0 SKETCH`
`S1 PROTOTYPE`
`S2 WORKING`
`S3 QUALIFIED`
`S4 PRODUCTION-GRADE`
`S5 KINGSMAN-COMPLIANT`.

S5 requires, where applicable:

- mission fit;
- contract/state ownership clarity;
- identity/authority boundaries;
- exact evidence/currentness;
- final-hand enforcement;
- independent outcome verification;
- failure/degraded behaviour;
- replaceability/migration;
- continuity/lineage;
- operator legibility;
- adversarial/fault qualification;
- docs/architecture synchronization;
- no weaker legacy authority path.

---

# 19. Open decisions / discriminating spikes before final canon freeze

These are explicitly **not frozen** yet.

1. generated Compose vs base/overlay profile implementation;
2. precise authoritative memory owner(s) after E0;
3. exact home of Goal/Watch/Attention state/decision logic after E0;
4. whether Tool Gate remains one process or internally split components after fault/security measurement;
5. capability-at-hand transport/atomic-consumption design;
6. Postgres workflow/outbox exact schema/worker model;
7. whether any event broker is needed after measured fan-out/load;
8. exact independent-verifier placement for each actuator family;
9. browser/egress allow/deny policy granularity;
10. model registry enrichment vs later full runtime-manager boundary;
11. Strix Halo exact GPU/CPU/NPU role measurements;
12. lineage manifest minimum invariant set;
13. audit retention/privacy/cryptographic-erasure semantics;
14. future autonomy-grant renewal inside an already delegated envelope;
15. succession/key custody/legal binding;
16. which simple watchers should merge into modules versus remain isolated services.

Every unresolved choice gets a cheapest discriminating test rather than architectural preference masquerading as fact.

---

# 20. Programme authority — architecture does not jump the queue

This v0.4 plan does **not** change the separately governed execution sequence.

Standing programme order remains subject to latest valid canonical D-numbered authority, including:

1. current House-in-Order authorised/frozen sequence;
2. KAI-GATE-048 return under its frozen experiment rules;
3. Item 8 under its separately frozen authority;
4. **ITEM 8 BEFORE A4**;
5. `A-4 PROVENANCE` repair/review/freeze/hash;
6. assurance integration;
7. professionalisation / Evidence Plane / Kingsman implementation toward the eventually frozen canon.

`FUTURE A4 SELF-DIAGNOSIS` remains distinct from `A-4 PROVENANCE`.

This plan authorises none of H2 v1.1, 048 scope changes, Item8 builds, A-4 execution, runtime refactor, service merge/delete, succession, autonomous finance or uncontrolled self-modification.

---

# 21. Review / freeze workflow

DeepSeek is used as Kai's external analysis instrument, not architecture authority.

Final review loop:

`Kai current repo/plan synthesis`
→ `DeepSeek targeted adversarial attack where useful`
→ `Kai reconciles every point against repo/history/philosophy`
→ `Orion exact feasibility/current mapping and discriminating tests`
→ `Kai incorporates evidence`
→ `Dainius final review`
→ `exact-byte KINGSMAN_MASTER_CANON_v1 + drawings + manifest freeze`.

No external model recommendation enters canon without Kai reconciliation and Dainius authority.

---

# 22. Freeze criteria

Do not freeze `KINGSMAN_MASTER_CANON_v1` until:

- [ ] E0 gives a sufficiently closed current component/connection/authority population;
- [ ] every major current capability has a target home or explicit unresolved disposition;
- [ ] no target box lacks current-to-target lineage or explicit genuinely-new-joint justification;
- [ ] final-hand authority architecture is resolved;
- [ ] manual vs autonomous authority lanes are explicit;
- [ ] legacy retirement semantics require runtime negative proof;
- [ ] World-State fallback semantics are explicit and non-silent;
- [ ] authority persistence/workflow strategy is credible;
- [ ] deployment profile strategy is reconciled;
- [ ] memory/proactivity/model-runtime physical homes are evidence-based;
- [ ] resilience/Doctor/Supervisor boundaries are non-overlapping;
- [ ] continuity/lineage requirements are explicit;
- [ ] Mission Control information model is explicit;
- [ ] existing capability preservation has zero-loss review;
- [ ] DeepSeek/Kai findings are reconciled;
- [ ] Orion feasibility/current map is reconciled;
- [ ] material open decisions have decisions or discriminating tests;
- [ ] Dainius approves;
- [ ] deterministic engineering drawings match the exact architecture subject;
- [ ] exact master subject is hashed/frozen with change-control rules.

---

# 23. Plain-language architecture

Kai already has most of the organs. Some are rough, some overlap, some are not cut over, and some responsibilities are in the wrong place.

We are **not replacing the organism**.

We are doing the engineering equivalent of taking a powerful prototype machine apart one subsystem at a time, documenting what each part really does, keeping the useful machinery running through adapters, moving each responsibility into the correct governed path, proving the old unsafe path is actually dead, and only then removing or rehoming the obsolete implementation.

The final organism should therefore be recognisably the Kai that has already been built — just professionally integrated, evidence-bound, resilient, replaceable, visible, and capable of growing for decades without turning into bolt-on soup.
