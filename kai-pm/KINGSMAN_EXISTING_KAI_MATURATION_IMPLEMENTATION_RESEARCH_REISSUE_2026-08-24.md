# Kingsman Existing-Kai Maturation — Implementation Research Reissue

**Date:** 24 August 2026  
**Repository:** `dainius1234/kai-system`  
**Branch reviewed:** `claude/project-rework-plan-pgvp35`  
**Status:** **DEEP-RESEARCH / IMPLEMENTATION GUIDANCE — NOT ARCHITECTURE AUTHORITY, NOT IMPLEMENTATION AUTHORITY, NOT PROGRAMME EXECUTION AUTHORITY**

> **Governing posture:** **REFIT, HARDEN, RATIONALISE, MATURE — DO NOT REINVENT KAI.**
>
> This is a reissue after a second adversarial research pass. It reconciles current external production guidance with what Kai has already designed in the repository. It deliberately does **not** produce another architecture version or another parallel programme.

---

# 1. Direct conclusion

The second-pass research changes the recommendation in one important way:

> **Kai already contains, or already has planned, most of the production-grade patterns that the external research recommends. The work is not to import a modern “agent architecture”; it is to reconcile, finish and qualify the existing P1–P4 remediation, Unified Hunter, House-in-Order, Item 8, A-4 provenance, Evidence Plane, resilience, continuity and operator-control work as one maturity programme.**

The strongest existing plans are more mature than the first research pass gave them credit for:

- **P1 Security Foundation** already specifies verified human/workload identity, exact operation digests, one-use audience-bound capabilities and enforcement at the final side-effecting service.
- **P2 Isolation & Integrity** already specifies hostile-content isolation, controlled egress, untrusted-content boundaries, data scoping, provenance, derivative lineage, transactional multi-store memory mutation and verifiable deletion/supersession.
- **P3 Reliability/Audit/Privacy/Recovery** already specifies typed uncertain/partial-effect states, durable operation identities, transactional shared state, leases/fencing, signed audit, data lifecycle, authorised recovery and isolated restore qualification.
- **P4 Capability Requalification** already specifies exact model/runtime identity, no stub impersonation, proposition-level evidence, source independence, calibrated uncertainty, outcome-linked trust and attack-chain release qualification.
- **House-in-Order Phase 2** already defines the correct professionalisation method: preserve intent, verify current reality, map dependencies, rework/merge/rehome only where evidence supports it, test/mutate/verify, synchronize docs and bank known-good state.

Therefore the mature implementation programme should **reuse those plans as first-class design inputs**, not rediscover the same controls under new Kingsman terminology.

The second pass identifies only a small number of genuinely new or under-specified cross-cutting requirements:

1. **Evidence-contract integration** — maturation instruments must consume/produce the eventual A-4/Evidence Plane provenance/qualification model instead of creating a second truth system.
2. **E0 instrument qualification** — the architecture/current-organism census itself must have a declared denominator, calibration, known positives/negatives, mutation/can-fail proof and exact subject binding.
3. **Generated Mission Control** — operator architecture/status views must be projections from qualified machine records, not another hand-maintained status document.
4. **Time authority semantics** — event time, ingest time, decision time and monotonic deadline/lease time must be distinguished; caller wall clock is never authority.
5. **Cross-organ resource budgets** — not just model context limits: CPU/GPU/NPU memory, token/context, network bytes, parser bytes, retries, queues, time and consequence budgets need explicit admission/degradation semantics.
6. **Data-classification propagation** — principal/purpose/classification/provenance must travel into retrieval, model-context assembly, remote-provider use, tool requests and egress decisions, not stop at storage metadata.
7. **Secure update / anti-rollback / freeze resistance** — a valid signature is insufficient if a restored or compromised machine can install an older vulnerable but validly signed release.
8. **Cryptographic agility** — Ed25519 is appropriate now, but identity/release/backup schemas must name algorithm/key versions and permit controlled algorithm transition over Kai's lifetime.

These are **strengthening joints**, not new mini-systems.

---

# 2. What changed after the second research pass

## 2.1 First-pass mistake avoided

The first implementation research risked describing industry best practice as if Kai needed to invent it.

The repo comparison shows that many of those recommendations are already explicitly designed in:

- `CODE_AUDIT_P1_SECURITY_FOUNDATION_PLAN.md`
- `CODE_AUDIT_P2_ISOLATION_AND_INTEGRITY_PLAN.md`
- `CODE_AUDIT_P3_RELIABILITY_AUDIT_PRIVACY_RECOVERY_PLAN.md`
- `CODE_AUDIT_P4_CAPABILITY_REQUALIFICATION_PLAN.md`
- `KAI_FINAL_PRODUCT_ARCHITECTURE_SPECIFICATION.md`
- `KAI_UNIFIED_HUNTER_ARCHITECTURE_AND_ROADMAP.md`
- `UH_PROGRESS_TRACKER.md`
- `HOUSE_IN_ORDER_PHASE2_PROFESSIONALISATION.md`
- `A4_SELF_DIAGNOSIS_EVOLUTION.md`
- D349/D351/D352 governance corrections.

The correct reissue therefore asks:

> **Which existing design owns this requirement, what is actually implemented today, what remains unqualified, and what is the smallest maturity move required?**

not:

> “What modern component should we add?”

## 2.2 DeepSeek review incorporated, but not promoted to authority

The DeepSeek review produced six useful discipline improvements. They survive with Kai's reconciliation:

- Evidence Plane compatibility — **accepted**, without reordering the frozen programme or making Evidence Plane a superior architecture authority.
- migration/readiness table ≠ authority — **accepted explicitly**.
- reuse existing service-identity registry — **accepted**; no duplicate workload-key authority.
- side effect succeeded but receipt lost → `OUTCOME_UNKNOWN` and reconcile before retry — **accepted**.
- E0 generator must itself be calibrated/can-fail — **accepted**.
- Mission Control generated from machine evidence — **accepted**.

Standing relationship:

`DeepSeek analysis input → Kai repo/history/philosophy reconciliation → Orion exact feasibility/evidence when useful → Dainius final authority`.

---

# 3. Current programme relationship — no new parallel programme

The research does **not** change current programme authority.

Architecture/professionalisation dependency relationship:

```text
House / Census truth qualification
        ↓
KAI-GATE-048 under its frozen design
        ↓
ITEM 8 under its separate frozen authority
        ↓
A-4 PROVENANCE
        ↓
Assurance / Evidence qualification contracts
        ↓
Kingsman maturation instruments consume qualified truth
        ↓
Existing Kai organs are hardened / finished / merged / moved / split where justified
```

**ITEM 8 BEFORE A4 remains standing.**

`A-4 PROVENANCE` remains distinct from `FUTURE A4 SELF-DIAGNOSIS`.

Kingsman maturation is **not “under” Evidence Plane as another authority hierarchy**. Evidence Plane supplies qualified observations/provenance. Kai's product architecture and D-numbered decisions govern what those observations mean for the organism.

---

# 4. Coverage / gap matrix after two research passes

| Area | Existing Kai ownership | External benchmark | Assessment | Required action |
|---|---|---|---|---|
| Human/workload identity | P1 + current Ed25519 `service_identity` | NIST Zero Trust, RFC 9421 | **Strong design / partial migration** | FINISH current identity migration; do not create second registry |
| One-use capability at final hand | P1 + D349 KAI-REV-016 | capability/PoP patterns, complete mediation | **Designed earlier; current final-hand implementation incomplete** | HARDEN existing Tool Gate/Actuator path |
| Legacy bypass retirement | UH migration + D349 KAI-REV-017 | zero trust / negative security tests | **Under-proven** | ADD runtime deny proof to existing verifier/migration machinery |
| Durable authority | P1/P3 + Tool Gate | PostgreSQL transactional patterns | **Design present / current process-local gaps** | HARDEN behind existing APIs using Postgres first |
| Durable workflow | P3 + UH `ActionWorkflow` | transactional outbox/idempotency/fencing | **Design present / implementation immature** | FINISH around current ActuatorRegistry; no new workflow platform now |
| Hostile content / prompt injection | P2 + memory/prompt audits | OWASP Agent/LLM/RAG guidance | **Strong existing design, underweighted in first research** | CARRY P2 boundary into all retrieval/memory/proactivity/tool paths |
| Egress/SSRF/browser isolation | P2 + current browser/network work | OWASP SSRF, Playwright isolation | **Strong existing design / incomplete qualification** | FINISH, no separate egress service by default |
| Memory provenance/lineage | P2 + P3 + memu audits | provenance/RAG integrity | **Strong plan / current implementation highly immature** | TRACE-FIRST, transactional maturation; no new memory system |
| Unknown/degraded semantics | UH + P3 + D349 | SRE graceful degradation | **Good doctrine, inconsistent implementation** | HARDEN fallbacks so degradation never becomes weaker truth/authority |
| Model identity/qualification | P4 + final product spec | SLSA-style exact artifact identity, AI TEVV | **Strong existing plan** | FINISH exact model/runtime registry; do not prematurely add wrapper runtime service |
| Proactivity/Goals/Watches | existing observers/monitor/Cortex/agentic | event-driven agent practice | **Capability present; semantic ownership fragmented** | REWORK/MERGE after E0; no new proactivity service by default |
| Doctor/Supervisor/recovery | P3 + House Doctor + Future A4 | SRE/fault isolation | **Strong conceptual split; current implementation fragmented** | FINISH split responsibilities + evidence-bound contingency/verification |
| Self-improvement/skills | P4/growth/Dream/Evolver/probation | AI supply-chain/sandbox practice | **Existing concepts strong; release controls need maturation** | HARDEN candidate→sandbox→evidence→approval→probation→rollback lifecycle |
| Backup/restore/lineage | P3 + backup-service + final product spec | NIST recovery / signed manifests | **Strong design / current restore gaps** | FINISH manifest + isolated restore + off-device copy |
| Evidence/provenance | House/A-4/Evidence research | W3C PROV, SLSA, in-toto | **Active programme foundation** | CONSUME once frozen; no parallel truth model |
| E0 component census | House/Census future extension | CycloneDX/OTel identities as references | **Concept exists; instrument not yet final** | EXTEND existing instrumentation + calibration/can-fail proof |
| Mission Control | Dashboard/operator doctrine/doc sync work | SRE control-room patterns | **Vision exists; machine projection under-specified** | GENERATE from qualified current-state records |
| Time semantics | UH leases/fencing + key/memory audit lessons | NIST AU-8/time integrity | **Partially addressed, fragmented** | ADD shared semantic contract, not necessarily a new service |
| Resource/consumption budgets | model/runtime/rate-limit audits + hardware plan | OWASP Unbounded Consumption, Google SRE | **Partially addressed, fragmented** | ADD common budget vocabulary/admission/degradation semantics |
| Data-class propagation | P2/P3 + final product FP-INV-06 | NIST AI RMF / third-party GAI controls | **Storage design strong; end-to-end propagation under-specified** | PROPAGATE labels through context/model/tool/egress paths |
| Secure software update | release/provenance pieces exist | TUF rollback/freeze resistance | **True/important long-horizon gap or under-specified area** | ADD requirements to continuity/release design; no TUF dependency automatically |
| Crypto agility | current Ed25519 + key rotation work | NIST CSWP 39upd1 | **True long-horizon requirement** | ADD algorithm/key metadata and controlled transition capability |

---

# 5. Existing Kai organism — implementation direction by organ

The professional implementation view should preserve the organism/product language.

## 5.1 Soul / Identity / Inner Life

### Existing assets

Narrative identity, emotional memory, operator model, cognitive fingerprint, confirmed values/conscience concepts, release/lineage records.

### Maturation

- KEEP these as identity/relationship learned state.
- Bind durable identity-affecting writes to authenticated principal/source/provenance.
- Never allow retrieved/external content or Kai's own generated reply to become operator value/identity authority automatically.
- Distinguish constitutional mission/lineage from learned preferences and temporary mood/context.
- Version any high-authority identity/mission record and make supersession explicit/auditable.

### Important security boundary

**External content is data, not identity instruction.**

OWASP's current agentic guidance treats persistent memory/context poisoning as a separate production risk. P2 already has the correct Kai invariant: web/document/email/news/screen/clipboard/camera/audio/model output is untrusted evidence and cannot be promoted directly into operator preferences, trusted memory or system instructions.

Action: **carry P2's rule through every memory, retrieval and prompt-assembly path rather than inventing a new “memory guard service.”**

---

## 5.2 Memory / Relationship / Continuity

### Existing assets

memu-core, memu-graph/vector paths, Letta, Obsidian/vault-sync, memory compressor, emotional/narrative/operator memory.

### Repo reality that matters

The memu hot-path audit already found critical/high weaknesses including anonymous authority, keeper impersonation, verifier fail-open, missing source provenance, process-local state, partial state/memory transactions, graph side-channel divergence and untrusted memory text formatted as ready-to-inject LLM context.

### Maturation

Do **not** declare a new memory architecture from filenames.

Sequence:

1. E0 traces real readers/writers/source-of-truth relationships.
2. Establish authenticated principal/source/purpose/classification/provenance on each durable record.
3. Establish authoritative source record versus derived vector/graph/archive/mirror.
4. Make derivative lineage rebuildable and observable.
5. Make memory + graph/vector mutation a durable operation with explicit partial/repair state.
6. Make supersession atomic from retrieval-authority perspective.
7. Prevent retrieved content from gaining instruction authority when assembled into model context.
8. Bound retrieval size/tokens and preserve source attribution/trust labels.
9. Restore authoritative records first; rebuild or validate derived projections afterward.

Default disposition: **KEEP + HARDEN + REWORK INTERNALLY**, not replace memu with another memory product.

---

## 5.3 Senses / World Awareness

### Existing assets

Perception services, watchers, Cortex, PerceptionIngress, EventJournal, shadow runner, World State transition.

### Maturation

- KEEP current sensors/adapters.
- Reuse the existing UH shadow/active migration.
- Preserve event identity, source, schema and source-event time separately from ingest time.
- Use explicit idempotency/deduplication at the ingestion boundary.
- Reducer/materialized-state failure must be visible as `DEGRADED/UNKNOWN`, not silently hidden by legacy polling.
- Consumer projections may preserve old interfaces during cutover.
- Old polling retires only after runtime evidence proves it is no longer load-bearing.

CloudEvents is a useful semantic reference for `id/source/type/schema/time`, but adopting CloudEvents wire format or an event broker is **not required**.

---

## 5.4 Intelligence / Reasoning / Cognitive Depth

### Existing assets

agentic FSM, Unified Hunter proposal-only workspace, Scout/Sage/Doctor/Oracle, swarm/reputation/conflict, adversary, conviction, hypothesis, temporal/causal reasoning, Global Workspace concepts, dormant higher-cognition modules, Ollama/current model registry.

### Maturation

- KEEP the existing Hunter/agentic cognitive centre.
- Map every specialist to role, exact model identity, maturity, evidence and resource requirements.
- Do not give specialist modules authority merely because they produce high-confidence language.
- P4 exact model identity requirements should become the qualification standard: artifact/provider revision, tokenizer/context, quantization/build, prompt/policy/tool-schema revision, runtime/image, workload identity and readiness/capacity evidence.
- Preserve disagreement rather than manufacturing consensus.
- Treat unavailable/stub/fallback specialist views as missing, not votes.
- Enforce aggregate prompt/context budgets per exact model/runtime.

Do **not** build a separate “Runtime Manager service” merely to satisfy an abstraction. First mature the existing registry/serving path. A stronger manager boundary is earned only when multiple runtimes/residency/preemption/resource scheduling require it.

---

## 5.5 Proactivity / Goals / Attention

### Existing assets

agentic proactive observer, monitor-service, Cortex, calendar, anomaly/correlation, screen watcher, rituals, capability gaps, Supervisor nudges, notifications.

### Maturation

The missing part is primarily a **shared semantic model**, not another service.

Candidate durable concepts:

- `Goal`
- `Obligation`
- `Commitment`
- `Watch`
- `Timer`
- `AttentionCandidate`

Detector output should remain observation/candidate data. Decision output should normally be:

`IGNORE / STORE / WATCH / PREPARE / PROPOSE / NOTIFY`

and only reach `ACT` through the existing authority/capability path.

Qualification:

- historical replay;
- useful-intervention precision/recall;
- deduplication;
- interruption cost;
- spam rate;
- stale-watch expiry;
- escalation behaviour;
- no prompt-injected external content becoming a durable Goal/Commitment without qualified promotion.

---

## 5.6 Governed Hands / Capabilities

### Existing assets

Tool Gate, policy bridge, approval/capability, LegacyTrustBridge/scoped autonomy, service identity, ActuatorRegistry, 34 actuator identities, executor/browser/notify/files/calendar/backup/finance handlers.

### Maturation

P1 already defines the correct invariant and D349 proves it remains incomplete in current implementation:

> **The actual final side-effect endpoint must validate and atomically consume a one-use audience-bound capability for the exact operation.**

Correct semantic stack:

```text
membership
≠ workload identity
≠ static operation scope
≠ policy/approval
≠ one-use execution capability
≠ scoped autonomy delegation
```

Manual lane:

`proposal → policy → authenticated exact operator approval → one-use capability → final-hand consume → effect → independent verification`

Autonomous lane:

`proposal → policy → valid scoped autonomy delegation → one-use capability → final-hand consume → effect → independent verification`

Autonomy allows bounded **initiation**. It does not replace final execution authority.

### Current implementation direction

- FINISH existing Ed25519 identity migration; no new identity registry.
- Persist approval/grant/capability/revocation/consumption behind current Tool Gate-compatible APIs.
- Use existing PostgreSQL first.
- For final-hand request binding, reuse the current signed-request infrastructure and add the execution-capability binding rather than adopting OAuth/DPoP wholesale.
- RFC 9421 and RFC 9449 are **design references** for request/time/key/nonce/method/URI/token binding, not mandates to replace Kai's protocol.
- Add runtime negative bypass tests; source/authentication checks alone cannot prove the weaker direct route is dead.

---

## 5.7 Durable workflow / exact external effects

### Existing assets

UH `ActionWorkflow`, ActuatorRegistry, executor/handlers, P3 reliability requirements.

### First implementation choice

Use PostgreSQL before adding Temporal/NATS/Kafka.

PostgreSQL 18 explicitly documents `FOR UPDATE ... SKIP LOCKED` as appropriate for queue-like tables, and `INSERT ... ON CONFLICT DO UPDATE` provides atomic insert-or-update semantics under concurrency.

Candidate workflow properties:

- immutable operation ID/digest;
- state transitions in one durable table;
- transactional outbox;
- worker lease/fencing;
- idempotency key;
- bounded retry policy;
- exact capability reference;
- actuator receipt;
- independent verification reference;
- compensation/reconciliation state.

### Critical uncertain-outcome rule

Both of these are `OUTCOME_UNKNOWN`:

1. capability consumed → process crashes before side effect;
2. side effect succeeds → process crashes before receipt is durably recorded.

Therefore:

> **absence of a receipt is never permission to retry a consequential operation. Observe/reconcile target state first.**

Only if reconciliation proves the side effect did not happen may retry become eligible.

---

## 5.8 Immune System / Doctor / Resilience

### Existing assets

heartbeat/metrics/watchers, common resilience, Supervisor, House Doctor, Doctor teammate, verifier/fusion, system FSM, Future A4 self-diagnosis design.

### Correct split

- Telemetry/watchers = observations.
- Component/dependency/authority map = structure.
- House Doctor/Future A4 = structured diagnosis.
- Doctor teammate = interactive cognitive diagnostic specialist.
- Contingency knowledge = qualified response options.
- Supervisor = bounded recovery coordinator.
- Tool Gate/workflow/actuator = authority and hands.
- Independent verifier = recovery truth.

No component should diagnose, approve, repair and certify itself end-to-end.

### Resilience discipline

Google SRE/AWS guidance reinforces existing Kai doctrine:

- load-test to failure;
- bounded queue/admission;
- exponential backoff + jitter;
- retry budgets;
- avoid retry multiplication at multiple layers;
- shed/degrade workload explicitly;
- regularly exercise degraded modes;
- fallback must not hide the original failure or silently change authority/truth.

This directly supports the D349 Cortex correction: legacy polling may be an explicit cold-start compatibility path, but not a silent steady-state truth substitution.

---

## 5.9 Growth / Dream / Evolution / Skills

### Existing assets

skill-hunter, Agent-Evolver, Dream/introspection, capability-gap discovery, workspace manager, probation/disable mechanisms.

### Maturation

Retain the existing concepts and add a common governed lifecycle:

`candidate → provenance → sandbox → tests → evidence → operator/release decision → probation → promote → monitor → auto-disable/rollback/retire`.

Important AI-specific controls:

- external documentation/repository/tool descriptions are untrusted input;
- skill dependencies and remote references must be version/hash-pinned where feasible;
- generated code cannot certify its own tests/evidence;
- skills do not self-grant new tools/network/data access;
- release subject binds exact source/build/runtime/policy/tool schema;
- supply-chain evidence is generated by qualified mechanisms, not copied from the candidate's claims.

Do not create a competing self-modification/self-healing agent.

---

## 5.10 Continuity / Stewardship / Survivability

### Existing assets

backup-service, release/evidence work, identity/narrative continuity, sovereign hardening, P3 restore requirements.

### Maturation already required by P3

- coherent backup manifest;
- exact release/schema/store identities;
- checksums;
- key references;
- isolated restore drill;
- application-level postconditions;
- off-device/offline encrypted copies;
- migration to replacement hardware;
- operator-visible RPO/RTO/restore qualification.

### New long-horizon strengthening requirement — secure update/anti-rollback

A restored Kai must not accept an old vulnerable release merely because it was once validly signed.

The Update Framework (TUF) is the external reference pattern, not an automatic dependency. The required properties are:

- signed root of update trust;
- target artifact hashes/sizes;
- monotonic version/release metadata;
- expiry/freshness metadata;
- protection against rollback, freeze and mix-and-match states;
- key separation/threshold/offline root options appropriate to Kai's private scale;
- explicit emergency recovery procedure;
- software update selection separate from installation authority.

Kai can initially implement the **requirements** in its release/lineage manifest without importing a full TUF deployment if the simpler design can prove equivalent required properties for the local/private threat model.

### New long-horizon strengthening requirement — cryptographic agility

Current Ed25519 is appropriate and should be finished, not replaced now.

But long-lived records should not assume one algorithm forever. Store:

- `algorithm_id`
- `key_id`
- key purpose/role
- activation/revocation/retirement state
- signature/proof version
- verification-policy revision

and design key/algorithm transitions so old data can remain verifiable while new operations move to stronger algorithms later.

NIST CSWP 39upd1 (June 2026) explicitly frames crypto agility as the ability to replace/adapt cryptographic algorithms while maintaining security and operations.

---

## 5.11 Operator Relationship / Mission Control

### Existing assets

Dashboard, Grafana, PM/UH trackers, operator-visibility doctrine, doc-sync work.

### Maturation

Mission Control is a **projection**, not a new truth authority.

Target data sources:

```text
qualified component/current-state registry
+ runtime telemetry/health
+ programme/D-number state
+ authority/grants/approvals
+ workflow/action state
+ evidence/provenance/currentness
+ memory/world-state freshness
+ model/resource state
+ backup/lineage/update state
        ↓
Mission Control projection
        ↓
Dashboard panels / README current-status regions / architecture maps
```

A green tick/arrow/status must disappear or degrade when its evidence becomes stale/invalid.

The operator view should lead with the organism:

- WHO KAI IS
- KAI TODAY
- KAI MATURED
- WHAT CHANGES
- CURRENT PROGRAMME POSITION
- RISKS/BLOCKERS/DECISIONS

Technical evidence is drill-down underneath.

---

# 6. E0 — current organism census must itself meet the House evidence standard

E0 is not a hand-maintained architecture document and not a magical source scanner.

It should **extend existing House/Census instrumentation** and combine independent evidence classes such as:

- `docker compose config` normalized service/profile/network/volume/env declarations;
- AST/import/static route/client analysis;
- FastAPI/route schemas where mechanically discoverable;
- Alembic/PostgreSQL schema declarations;
- model/service/capability registries;
- tests/fixtures/CI target declarations;
- runtime telemetry/service identity where available;
- explicit manual declarations only for semantic relationships that cannot be safely inferred.

Useful standards can inform the output:

- CycloneDX can contribute component/service/dependency/completeness and identity-evidence concepts;
- OpenTelemetry can contribute stable service/instance/version/criticality resource identity;
- neither replaces Kai's authority/reader/writer/semantic relationship model.

## Mandatory E0 instrument qualification

Before E0 becomes CI-blocking:

1. declare the intended population/denominator;
2. known-positive fixture;
3. known-negative fixture;
4. boundary/ambiguous fixture;
5. deliberate architecture drift mutation;
6. prove the detector flips only the intended result;
7. remove mutation and prove restoration;
8. prove deterministic regeneration on the same exact tree;
9. bind report to exact commit/tree/instrument version;
10. distinguish `NO_PROVEN_*` from proven absence;
11. preserve UNKNOWN where the search boundary is not closed.

Example mutation:

> add one temporary service declaration with no component identity/owner; E0 must detect exactly the new violation, and the finding must disappear when the mutation is removed.

This is the Item8/House discipline applied to the architecture instrument itself.

---

# 7. Evidence Plane relationship — explicit and non-duplicative

E0, migration evidence, runtime health and Mission Control should eventually consume/produce the same core evidence concepts as A-4/Evidence Plane:

```text
observation
→ evidence identity
→ exact subject
→ claim
→ provenance
→ qualification
→ applicability
→ uncertainty/currentness
→ downstream use decision
→ action
→ verified outcome
```

W3C PROV provides useful generic concepts around entities, activities and agents. SLSA/in-toto provide useful artifact/build/source attestations. Kai keeps its own semantics for subject applicability, contamination, UNKNOWN, claim-scoped negatives and operator authority.

Do not conflate three different kinds of provenance:

1. **observation/runtime provenance** — why a current claim is believed;
2. **source/build provenance** — where a software artifact came from and how it was produced;
3. **continuity/restore lineage** — which Kai release/state/memory/authority set a restored organism descends from.

They should link, not collapse into one overloaded table.

---

# 8. Migration/readiness evidence must never become authority

A migration record may say:

```text
SHADOW → COMPARED → QUALIFIED → CANARY → ACTIVE → LEGACY_DENY_PROVEN → RETIRED
```

but that describes **readiness/evidence state**.

It does not grant action authority.

Minimum separation:

```text
migration_evidence
  subject
  phase
  evidence_refs
  instrument_revision
  soak/result
  qualified_at

operator_or_programme_decision
  decision_id
  exact_subject
  decision
  authority_identity
  scope
  expiry/change-control
```

A deployment flag may request a path. A qualified migration record may prove it is eligible. A valid authority decision determines whether that cutover is permitted.

---

# 9. Time authority — shared semantics, not a new time service

Kai already uses monotonic time correctly in at least the perception lease/fencing path, and the audit has found multiple wall-clock weaknesses elsewhere.

Define common time semantics:

- `event_time` — when source claims the event happened; may be untrusted/uncertain;
- `observed_at` / `ingested_at` — trusted local observation/receipt timestamp;
- `decided_at` — policy/approval decision time;
- `created_at` / `expires_at` — signed authority metadata using synchronized wall time;
- `deadline` / lease TTL / retry backoff — monotonic duration where possible;
- `verified_at` — verifier observation time.

Rules:

- caller-supplied wall time never controls authority, retention or lease validity without qualification;
- signed-request clock skew is explicitly bounded;
- audit timestamps use UTC/offset-aware system time;
- monotonic clocks control local elapsed-duration safety;
- clock synchronization/degradation becomes an observable dependency for expiry-based controls.

This strengthens current code rather than adding a dedicated clock service.

---

# 10. Cross-organ resource and consequence budgets

Kai's local hardware and autonomous/proactive nature make unbounded consumption a product risk, not only a DoS risk.

Define a shared **budget vocabulary**, while each organ enforces the budget it owns:

- request/operation wall time;
- model input/output/context tokens;
- CPU time;
- GPU/NPU memory residency;
- system RAM/headroom;
- parser/upload expanded bytes;
- network request/response bytes;
- number of external calls;
- retry attempts and retry budget;
- queue depth/in-flight work;
- background/dream batch quota;
- notification/interruption budget;
- financial/resource consequence budget;
- storage growth/retention budget.

Principles:

- admission before expensive work where possible;
- budget travels with durable operation/workflow;
- retries consume budget;
- degraded mode reduces work explicitly;
- an exhausted budget cannot silently switch to a less-governed implementation;
- resource status becomes visible in Mission Control.

OWASP's Unbounded Consumption guidance and Google SRE cascading-failure guidance support this approach.

---

# 11. Data classification and purpose must propagate, not stop at storage

The final product spec already requires durable records to carry principal, tenant, purpose, classification, provenance, revision and lifecycle metadata.

The maturation programme must propagate those controls through:

`record → retrieval → prompt/context → specialist/model → tool proposal → capability → egress/external provider → audit/outcome`.

Examples:

- a local-only/private memory cannot be sent to a remote model merely because that model was selected as a specialist;
- a retrieved untrusted web document cannot become `operator_preference` because a model summarizes it confidently;
- an externally sourced claim retains its provenance/trust class when transformed into an AttentionCandidate;
- a browser/email/tool capability carries the data classes it may disclose as part of egress policy;
- telemetry/logging applies separate audience/retention/redaction policy rather than becoming a shadow database.

No new “data-classification service” is required by default. The classification metadata and policy must be enforced by the existing storage/context/policy/egress boundaries.

---

# 12. Deployment profiles — reconcile, do not maintain three architectures

Docker Compose supports service profiles within one application model. This fits Kai's requirement better than independent `minimal/full/sovereign` files drifting into different topologies.

Maturation target:

- one qualified component inventory;
- core components always present where required;
- optional/heavy/debug/hardware components activated by profiles/overlays;
- sovereign hardening represented as compatible security/resource/network configuration, not a competing product definition;
- CI renders each intended profile and checks dependency/health/network/volume/secret invariants;
- generated/rendered output may remain committed for operator inspection if deterministic.

Do not force a particular YAML-generation framework before E0/E1 proves the simplest workable layout.

---

# 13. Hardware-aware maturation — Strix Halo

AMD's current official specifications for Ryzen AI Max+ 395 confirm:

- 16 Zen 5 cores / 32 threads;
- Radeon 8060S, 40 RDNA 3.5 CUs;
- XDNA 2 NPU, up to 50 TOPS;
- up to 128GB LPDDR5x platform configurations;
- 45–120W configurable TDP.

Current AMD Ryzen AI Software supports NPU/iGPU inference through ONNX Runtime/Vitis AI paths; current ROCm documentation includes Ryzen AI Max+ 395 support, subject to exact OS/kernel/ROCm compatibility.

Implementation rule:

- **GPU/iGPU** = default heavy local generative/multimodal compute candidate;
- **NPU** = only workloads demonstrated to be supported/accurate/power-beneficial through the current AMD runtime; do not assume arbitrary LLM portability;
- **CPU** = deterministic control/data/policy/DB/coordination and workloads where CPU is the right measured choice;
- unified memory is a shared budget, not “free VRAM”.

Qualification per workload:

`exact model → exact runtime/version → placement → memory → latency/throughput → power/thermal → output accuracy → fallback/degraded behaviour`.

The hardware manager should emerge from measured admission/residency requirements; do not create a large scheduler service before those measurements exist.

---

# 14. Secure update / release / restore chain — proposed new canon obligation

This is the clearest true long-horizon addition from the second pass.

Current provenance/release/backup work answers much of **“what is this artifact/state and where did it come from?”**

Kai also needs to answer:

> **“Is this still an allowed release to run, or is somebody replaying a valid but obsolete/vulnerable version?”**

Candidate minimum release/update metadata:

```text
release_id
source_commit/tree
build_provenance_ref
artifact/image/model digests
configuration/schema revisions
minimum_compatible_lineage_version
monotonic_release_sequence
created_at
expires_or_review_by
signing_role/key_id/algorithm_id
allowed_previous_release_for_rollback
rollback_reason/authority if used
supersedes
revoked_release_ids
```

Required tests:

- valid newest release accepted;
- tampered artifact rejected;
- older but validly signed vulnerable release rejected by default;
- frozen metadata detected after freshness threshold;
- mixed metadata/artifacts from two releases rejected;
- compromised/retired signing key no longer authorizes new release;
- emergency authorized rollback is explicit, bounded and visible;
- restored old backup does not silently reset the “highest trusted release seen” state.

Whether this eventually uses TUF directly is a later implementation decision. The **security properties** are the requirement.

---

# 15. No-new-system list after the second pass

External research does **not** justify adding these now:

- no new memory service beside memu;
- no new orchestration framework beside Unified Hunter/agentic;
- no new Authority service beside Tool Gate/control path;
- no new autonomy system beside scoped autonomy/LegacyTrustBridge;
- no new generic actuator framework beside ActuatorRegistry;
- no new workflow platform such as Temporal yet;
- no new Kafka/NATS event backbone yet;
- no new proactivity service by default;
- no new Doctor/self-healing agent;
- no second Dashboard/Mission Control truth source;
- no parallel identity registry;
- no blanket SPIFFE/SPIRE migration now;
- no mandatory TUF deployment now;
- no mandatory CloudEvents wire-format conversion now;
- no Kubernetes migration merely for architecture fashion.

Every future new dependency must prove that current Kai cannot meet the required property safely/maintainably in place, and must state the operational/failure cost it introduces.

---

# 16. Evidence / testing bar for every maturity change

A retained capability is not production-ready because the new code exists or a unit test passes.

For each material control/migration:

1. exact current subject/tree/build/runtime;
2. original intent and current consumers;
3. positive test;
4. negative test;
5. boundary test;
6. known unavailable/degraded condition;
7. deliberate mutation/can-fail proof for load-bearing detectors;
8. restart/concurrency/replay where stateful;
9. crash-before-effect;
10. effect-succeeded-but-receipt-lost;
11. stale/rollback/clock-change where time/version matters;
12. bypass/adversarial test where authority matters;
13. fault injection and expected blast radius where resilience matters;
14. shadow/compare/canary/soak where migration affects live semantics;
15. prove the weaker old path is unusable before retirement;
16. rollback itself cannot silently restore weaker authority;
17. operator-visible current state updates from machine evidence;
18. bank exact evidence/provenance.

This is the same truth discipline already learned through House and Item 8.

---

# 17. Suggested future Phase-2 maturation sequence — only after upstream authority permits

This is **not current execution authority**. It is the implementation dependency map to apply when House/048/Item8/A-4/Assurance prerequisites permit.

### M0 — recover exact current organism

Extend House/Census into component/capability/reader/writer/state/authority/runtime map; calibrate the instrument before CI enforcement.

### M1 — reconcile existing P1–P4 + UH + current code

For each major organ, identify:

`already implemented / planned but not implemented / implemented differently / obsolete / superseded / true gap`.

This step prevents duplicate engineering.

### M2 — complete identity / final-hand / durable authority foundations

Finish current Ed25519 route migration; persist authority state; close final-hand capability gap; add runtime bypass proof.

### M3 — complete isolation / hostile-content / egress boundaries

Apply P2 to browser/parser/memory/context/tool paths with data-class propagation.

### M4 — durable workflow / reliability / audit / unknown-outcome semantics

Apply P3 using existing PostgreSQL first; qualify retry/fencing/reconciliation/recovery.

### M5 — world-state / memory / proactivity ownership maturation

Trace first, then merge/move/split only where actual ownership/consumer evidence requires it.

### M6 — capability/model requalification

Apply P4 exact artifact/runtime/evidence/calibration requirements; preserve dormant future capabilities without pretending they are live.

### M7 — self-diagnosis / resilience consolidation

Feed qualified component/evidence/runtime maps into House Doctor/Future A4; retain operator authority and independent verification.

### M8 — continuity / secure update / crypto-agile lineage

Finish P3 backup/restore and add anti-rollback/freeze-resistant release semantics + crypto-agility requirements.

### M9 — Mission Control / README / diagrams generated from truth

Only after enough machine truth exists, produce the professional organism-first front layer and keep volatile status generated/verified.

This broadly aligns with the already banked `HOUSE_IN_ORDER_PHASE2_PROFESSIONALISATION.md`; it is a reconciliation, not a replacement roadmap.

---

# 18. Open discriminating questions — research cannot answer these from standards alone

These must be resolved from exact repo/runtime evidence or controlled spikes:

1. Which current memory store is authoritative for each memory class and which are projections/mirrors?
2. Which simple watchers genuinely need process isolation versus becoming modules/adapters?
3. Which current routes still accept shared-token-only identity and which are consequential?
4. What is the minimal final-hand capability transport/consume design that fits current `service_identity` and ActuatorRegistry without duplicate crypto?
5. What exact Postgres schema/transaction boundaries best fit current Tool Gate and workflow code?
6. Which current Compose topology should become the base application model after exact component census?
7. Which proactivity records already exist and can carry Watch/Timer/Goal semantics without a new store?
8. Which current model-registry/runtime features are live versus static metadata/stubs?
9. Which NPU workloads on the actual Flow Z13 hardware produce a measured power/performance win without accuracy loss?
10. Which release/update metadata already exists in current release-bundle code and what remains for anti-rollback/freeze resistance?
11. What exact Evidence Plane/A-4 contract is frozen when professionalisation actually starts?
12. Which operator Mission Control statuses can be mechanically derived immediately and which still require a governed human declaration?

These are the places to use Orion inspection and targeted DeepSeek attack—not broad architecture invention.

---

# 19. Diminishing-return stop test

The second research pass is stopped here because additional broad searching is no longer producing material new architecture requirements.

New searches are now predominantly confirming already identified controls or finding alternative implementations of the same properties.

The remaining uncertainty is primarily **Kai-specific current-state evidence**, not lack of external best practice.

Therefore the next high-value work is:

> **repo/current-runtime reconciliation of this research against exact existing code and plans**, then organism-first implementation/maturity planning when programme authority permits.

---

# 20. Primary external source ledger

The external sources below were selected as standards/official documentation or mature engineering guidance. They inform requirements; they do not override Kai's repo/programme authority.

## Provenance / software supply chain

- W3C PROV Overview — https://www.w3.org/TR/prov-overview/
- SLSA v1.2 specification — https://slsa.dev/spec/v1.2/
- SLSA v1.2 Provenance — https://slsa.dev/spec/v1.2/provenance
- in-toto Getting Started / layout-link model — https://in-toto.io/docs/getting-started/

## Secure update / crypto agility

- NIST CSWP 39upd1, *Considerations for Achieving Crypto Agility: Strategies and Practices* — https://csrc.nist.gov/pubs/cswp/39/upd1/considerations-for-achieving-crypto-agility/final
- The Update Framework roles/metadata — https://theupdateframework.io/docs/metadata/
- The Update Framework security properties — https://theupdateframework.io/docs/security/

## Identity / request binding

- RFC 9421, HTTP Message Signatures — https://www.rfc-editor.org/rfc/rfc9421.html
- RFC 9449, OAuth 2.0 Demonstrating Proof of Possession — https://www.rfc-editor.org/rfc/rfc9449.html

## Durable state / workflow primitives

- PostgreSQL 18 `SELECT` / locking / `SKIP LOCKED` — https://www.postgresql.org/docs/18/sql-select.html
- PostgreSQL 18 `INSERT ... ON CONFLICT` — https://www.postgresql.org/docs/18/sql-insert.html

## AI/agent security / hostile content

- OWASP LLM01 Prompt Injection — https://genai.owasp.org/llmrisk/llm01-prompt-injection/
- OWASP AI Agent Security Cheat Sheet — https://cheatsheetseries.owasp.org/cheatsheets/AI_Agent_Security_Cheat_Sheet.html
- OWASP RAG Security Cheat Sheet — https://cheatsheetseries.owasp.org/cheatsheets/RAG_Security_Cheat_Sheet.html
- OWASP LLM10 Unbounded Consumption — https://owasp.org/www-project-top-10-for-large-language-model-applications/2_0_vulns/LLM10_UnboundedConsumption
- OWASP memory/context poisoning discussion (2026) — https://genai.owasp.org/2026/05/13/memory-is-a-feature-it-is-also-an-attack-surface/
- NIST AI 600-1 Generative AI Profile — https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence

## Resilience

- Google SRE, Addressing Cascading Failures — https://sre.google/sre-book/addressing-cascading-failures/
- Google SRE, Production Services Best Practices — https://sre.google/sre-book/service-best-practices/
- AWS Well-Architected, Control and limit retry calls — https://docs.aws.amazon.com/wellarchitected/2022-03-31/framework/rel_mitigate_interaction_failure_limit_retries.html

## Component / runtime identity and deployment

- CycloneDX Authoritative Guide to SBOM — https://cyclonedx.org/guides/sbom/lifecycle_phases/
- OpenTelemetry service semantic conventions — https://opentelemetry.io/docs/specs/semconv/resource/service/
- Docker Compose profiles — https://docs.docker.com/compose/how-tos/profiles/

## Hardware

- AMD Ryzen AI Max+ 395 official specifications — https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html
- AMD Ryzen AI Software 1.8 documentation — https://ryzenai.docs.amd.com/
- AMD ROCm/Ryzen Linux support matrix — https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html

## Time

- NIST SP 800-53 Rev. 5.1, AU-8 Time Stamps — https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final

---

# 21. Final research verdict

**The project does not need another architecture invention round.**

The best implementation path now visible is:

> **recover the exact existing organism → reconcile P1–P4/UH/House/A-4/Evidence against current code → finish the controls already designed → add only the few genuinely missing long-horizon joints → qualify every migration with the same evidence discipline → generate the operator view from that truth.**

The external research supports the philosophy D351/D352 established rather than contradicting it.

The technically strongest move is now **integration and qualification**, not novelty.
