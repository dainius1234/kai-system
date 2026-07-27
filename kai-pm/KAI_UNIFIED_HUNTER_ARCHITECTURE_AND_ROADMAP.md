# Kai Unified Hunter Architecture and Migration Roadmap

Repository: `dainius1234/kai-system`  
Planning date: 27 July 2026  
Status: **AUTHORITATIVE TARGET-ARCHITECTURE ADDENDUM — PLANNING ONLY**  
Runtime remediation: **NONE**  
Finding closures created by this document: **ZERO**

## 1. Decision

The “hunter with different tools” concept is adopted as the target functional architecture, with one critical clarification:

> Kai will have one logically unified decision process, but no single component will possess unchecked authority to observe, decide, approve and execute a consequential action.

The target is a coordinated organism:

- specialist modules perceive, analyse or simulate;
- a shared decision workspace compares proposals in one context;
- policy and human approval authorise exact actions;
- actuators execute only the authorised action;
- independent verification establishes the real outcome;
- learning uses verified outcomes, not self-reported success.

This document does not authorise implementation. Evidence preservation and P0 containment remain the first implementation work.

---

## 2. Why the architecture is required

The audit confirmed that current modules often behave as separate decision-makers:

- `agentic/strategy_engine.py` creates its own majority decision and can call `auto_trade()` directly;
- its governance check fails open when the trust dependency fails;
- `agentic/opportunity_intel.py` independently creates conviction scores and directive recommendations;
- the same or correlated source data can be counted multiple times as apparent consensus;
- specialist errors and missing sources can become HOLD, neutral or ordinary recommendation-shaped results;
- final action services can often be called without one exact, immutable, final-boundary authorisation.

The audit also confirmed that the proposed central components are not yet an operating brain:

- D102 `GlobalWorkspace` is currently a stub: bids are discarded, no winner is selected, broadcast is a no-op and `can_operate()` remains false;
- D109 Ohana state can mutate despite its capability gate reporting disabled, and current learning can record Kai’s response as if it were the operator’s decision;
- D110 Cortex can create a Workspace bid, but the receiving workspace does not operate;
- trust, conviction, memory and outcome state are not yet reliable authorities.

Therefore D131 cannot be a thin adapter that leaves existing decision-to-action paths intact. It must be a cross-system migration programme.

---

## 3. One-sentence target

> Every observation enters Kai through a typed, provenance-bound perception contract; one shared deliberation process produces an immutable action proposal; policy and, where required, Dainius approve the exact proposal; a narrowly scoped actuator executes it; an independent verifier records the real outcome; only that verified outcome may influence future learning or trust.

---

## 4. Governing design laws

### UH-INV-01 — One logical decision path

Every consequential action must be traceable through one canonical sequence:

`Perception → World State → Proposal → Policy → Approval → Capability → Execution → Observation → Verification → Learning`

No alternative direct path may remain usable in a protected profile.

### UH-INV-02 — Specialists do not self-authorise

A module may analyse, rank, predict or propose. It may not independently authorise and execute the same consequential action.

### UH-INV-03 — The Global Workspace proposes; it does not grant authority

D102 coordinates attention, alternatives and reasoning. Its output is an `ActionProposal`, never an execution credential.

### UH-INV-04 — Values do not become factual confidence

Ohana/value alignment may identify constraints, preferences and trade-offs. It must not increase factual certainty, evidence quality, source independence or security authority.

### UH-INV-05 — Exact-action authorisation

Approval and capability must bind the complete canonical operation digest: actor, target, action, parameters, limits, data scope, policy revision and expiry.

Changing any consequential field invalidates approval and capability.

### UH-INV-06 — Enforcement occurs at the hand

The actuator that performs the final side effect must validate and atomically consume the exact capability before the effect.

A gateway or central decision alone is not sufficient.

### UH-INV-07 — World State is immutable, versioned and scoped

There is no mutable global Python object containing “everything Kai knows.” World State is a versioned projection built from typed events, scoped by authenticated principal, tenant, purpose and data class.

### UH-INV-08 — Unknown and conflict are first-class states

Missing, stale, conflicting, degraded and unverifiable information must not be converted into neutral or successful evidence.

### UH-INV-09 — Execution is durable and idempotent

Action workflows use durable state, idempotency, fencing, reconciliation and compensation where possible. Timeouts create `UNKNOWN`, not automatic success or blind retry.

### UH-INV-10 — Learning requires independently verified outcomes

Predictions, proposals, simulations, model reflections, assistant responses and self-reported execution are not outcomes.

### UH-INV-11 — Human approval is protected authority

For high-consequence actions, Dainius’s approval is authenticated, explicit, operation-specific, revision-bound, expiring and auditable. An ordinary chat response, click on an unauthenticated page or broad standing instruction is not approval.

### UH-INV-12 — Logical centralisation, physical resilience

The decision policy is logically singular and coherent, but implementations may be replicated. Leadership, revision ownership and state mutation require durable shared state and fencing; process-local singletons are not authority.

### UH-INV-13 — Free-form text never carries control authority

Natural-language content may explain a proposal, but executable control fields use typed schemas. Prompt or document text cannot alter action, scope, identity, policy or capability fields.

### UH-INV-14 — No big-bang rewrite

Migration occurs by contracts, shadow operation and one vertical slice at a time. Legacy action paths are disabled as each replacement becomes verified; dual authority is not allowed.

---

## 5. Correct role model

The original binary rule — “every module is either perception or actuator” — is useful but incomplete. Kai needs six explicit roles.

### 5.1 Perception provider

Produces observations from a defined source.

Examples:

- market quotes;
- alpha/funding/positioning data;
- macro/news extracts;
- Cortex/sensor state;
- document/OCR results;
- memory retrieval results.

It may validate and summarise its own source. It does not recommend or execute a consequential action unless it is separately acting as a proposal specialist through a different typed interface.

### 5.2 Transformer / state reducer

Validates events and builds deterministic materialised views.

It does not invent missing facts or silently turn failure into neutral state.

### 5.3 Proposal specialist

Produces a candidate interpretation or action proposal from a specific expertise domain.

Examples:

- technical strategy analysis;
- opportunity analysis;
- risk analysis;
- counterargument/adversarial review;
- value/Ohana constraints;
- causal prediction.

It declares evidence, assumptions, uncertainty and source dependencies. It does not approve or execute.

### 5.4 Policy and approval authority

Determines whether the exact action is allowed under identity, scope, risk, limits and operator approval.

This is not an LLM judgement alone.

### 5.5 Actuator

Executes one fixed-schema operation after validating an exact one-time capability.

Examples:

- paper trader;
- browser action worker;
- notification sender;
- file operation worker;
- recovery worker;
- future broker adapter.

### 5.6 Outcome verifier

Independently checks what actually happened and records a verified outcome or `UNKNOWN`.

The actuator cannot be the sole judge of its own success.

---

## 6. Target component architecture

## 6.1 D131 is an umbrella workstream, not one file

D131 should be retained as the programme label **Perception Bus + World State**, but implemented as several bounded components/contracts:

1. `D131-A — Event Contract and Schema Registry`
2. `D131-B — Perception Ingress and Validation`
3. `D131-C — Event Journal and Transactional Outbox`
4. `D131-D — World State Reducers and Versioned Views`
5. `D131-E — Proposal and Decision Contracts`
6. `D131-F — Approval/Capability Bridge`
7. `D131-G — Outcome and Learning Bridge`

This avoids creating one giant god-service.

## 6.2 Logical flow

```text
Authenticated perception providers
        │
        ▼
D131-A/B Typed Perception Ingress
  schema • identity • provenance • freshness • limits
        │
        ▼
D131-C Durable Event Journal / Outbox
        │
        ▼
D131-D Scoped, versioned World State views
        │
        ├──────────────► specialist analyses / simulations
        │                         │
        ▼                         ▼
D102 Global Workspace / Deliberation Coordinator
        │
        ▼
Immutable ActionProposal + alternatives + evidence graph
        │
        ├────► D109 Ohana constraints/value assessment
        ├────► risk/evidence/adversarial verification
        └────► deterministic plan compiler
        │
        ▼
Tool Gate policy decision + protected human approval
        │
        ▼
One-time exact ActionCapability
        │
        ▼
Durable Action Workflow / WTE actuator
        │
        ▼
Independent outcome observation and verification
        │
        ▼
Signed audit + verified learning event
        │
        ▼
Wisdom/Trust/Calibration update under P4 rules
```

---

## 7. Canonical contracts

The schemas below are conceptual contracts. Exact field names will be frozen during P1 before implementation.

## 7.1 `PerceptionEvent`

Required properties:

```text
spec_version
event_id
event_type
schema_id
source_workload_id
source_service_revision
principal_id / tenant_id / purpose
data_classification
observed_at
received_at
expires_at or freshness_policy
trace_id
correlation_id
causation_id
source_event_id where available
payload_digest
payload
quality:
  validity
  completeness
  source_freshness
  uncertainty
  calibration_reference
  independence_group
provenance:
  provider
  retrieval/request reference
  transformation chain
  raw-object reference where retained
trust_class:
  authoritative | corroborated | external_untrusted | model_generated | simulated
```

Rules:

- event IDs are unique and deduplicated;
- source event time and local receipt time are separate;
- schemas are versioned and backward-compatibility tested;
- payload size, depth, cardinality and enums are bounded;
- NaN, infinity and invalid domain values are rejected;
- untrusted content is labelled and cannot populate control fields;
- workload identity is verified, not accepted from the body;
- producer and consumer revisions are auditable.

A CloudEvents-compatible envelope may be used, extended with Kai-specific security, provenance and quality fields.

## 7.2 `WorldStateSnapshot`

```text
snapshot_id
snapshot_schema_version
principal_scope
tenant_scope
purpose_scope
as_of_event_offset
created_at
reducer_revision
facts[]:
  fact_id
  type
  value
  state: known | unknown | stale | conflicting | unavailable
  observed_at
  expires_at
  supporting_event_ids
  contradiction_event_ids
  confidence_type
  uncertainty
  independence_groups
  sensitivity
snapshot_digest
```

Rules:

- snapshots are immutable;
- reducers are deterministic and revisioned;
- a snapshot is reproducible from retained source events;
- conflicting facts are preserved, not averaged away automatically;
- views expose only required data for the requesting principal/purpose;
- one domain cannot silently overwrite another domain’s fact;
- no module mutates the snapshot in place;
- event retention and snapshot retention are separately governed.

## 7.3 `ActionProposal`

```text
proposal_id
proposal_revision
proposer_workload_id
principal_id
goal
risk_class
world_state_snapshot_id
operations[]
expected_effects[]
predicted_outcomes[]
alternatives[]
no_action_option
assumptions[]
contraindications[]
evidence_refs[]
source_independence_groups[]
uncertainty
expiry
proposal_digest
```

Rules:

- proposal is non-authoritative;
- exact operations use fixed schemas;
- narrative explanation is separate from executable fields;
- no proposal may cite its own generated text as independent evidence;
- no “confidence” field alone authorises execution;
- correlated sources are declared;
- proposal includes the option to do nothing;
- proposal expiry is mandatory for time-sensitive domains.

## 7.4 `ConstraintAssessment`

Ohana, safety, privacy, risk and domain controls return typed assessments:

```text
assessment_id
proposal_digest
assessor_identity
assessment_type
result: allow_advisory | caution | block | requires_human | unavailable
constraints[]
reasons[]
evidence_refs[]
policy_or_values_revision
created_at
expires_at
```

Rules:

- Ohana never creates a security allow by itself;
- a hard safety/security block cannot be outweighed by loyalty or conviction;
- unavailable required assessment fails closed;
- value assessment and factual verification remain separate dimensions.

## 7.5 `ApprovalRecord`

```text
approval_id
principal_id
authentication_context
proposal_digest
operation_digests[]
policy_revision
risk_class
decision: approve | deny
constraints / limits
approved_at
expires_at
single_use
approval_signature
```

Rules:

- approval UI displays the exact action, target, data, quantity, limits, expected effects and uncertainty;
- approval cannot be applied to a changed proposal;
- approval is not inferred from ordinary conversation;
- denial and revocation take precedence;
- high-consequence approval requires step-up authentication;
- approval endpoint is itself authenticated, origin restricted and auditable.

## 7.6 `ActionCapability`

```text
capability_id
issuer
subject_workload_id
audience_actuator
principal_id
operation_digest
scope
limits
policy_revision
approval_id where required
issued_at
expires_at
nonce / consumption_id
```

Rules:

- audience restricted to one actuator;
- single operation or tightly bounded operation set;
- short lived;
- atomic one-time consumption;
- replay, target substitution and parameter modification fail;
- capability is not logged in reusable form;
- rollback cannot restore a weaker legacy credential path.

## 7.7 `ActionWorkflow`

Minimum durable states:

```text
PROPOSED
POLICY_BLOCKED
WAITING_FOR_APPROVAL
APPROVED
CAPABILITY_ISSUED
DISPATCHED
RUNNING
SUCCEEDED_UNVERIFIED
FAILED
UNKNOWN
COMPENSATING
COMPENSATED
VERIFIED_SUCCESS
VERIFIED_FAILURE
CLOSED
```

Rules:

- state transitions are transactional and revision checked;
- retries use idempotency keys;
- timeout becomes `UNKNOWN` until reconciled;
- compensation is explicit and not assumed to reverse every real-world effect;
- only verified terminal states feed learning;
- workflow history is immutable and correlated end to end.

A durable workflow engine may be adopted or these semantics may be implemented in Kai’s transactional store. The requirement is durable behaviour, not a specific vendor.

## 7.8 `VerifiedOutcome`

```text
outcome_id
workflow_id
operation_digest
verification_method
verifier_workload_id
observations[]
source_refs[]
result: success | failure | partial | unknown
before_state_ref
after_state_ref
measured_effects[]
unintended_effects[]
verified_at
outcome_digest
```

Rules:

- executor response is evidence, not final truth;
- independent state/source is checked where possible;
- partial success is not flattened to success;
- unknown remains unknown;
- prediction and outcome use separate records;
- trust/calibration updates link to this record.

---

## 8. Reassignment of existing modules

## 8.1 D102 Global Workspace

Target role: **deliberation and proposal coordinator**.

It may:

- request scoped World State views;
- receive typed specialist proposals;
- compare alternatives and contradictions;
- ask for missing information;
- assemble an `ActionProposal`;
- broadcast deliberation state to authorised subscribers.

It must not:

- execute tools;
- issue capabilities;
- treat salience as permission;
- treat winning a bid as factual correctness;
- persist a free-form “conscious stream” as trusted evidence;
- use one model’s output as independent corroboration of itself.

The current D102 stub must not be activated simply by implementing its existing salience loop. Its contracts and trust model must be replaced first.

## 8.2 D109 Ohana Core

Target role: **operator-confirmed values and constraint advisor**.

Required changes:

- remove automatic learning from assistant responses;
- require authenticated, explicit operator confirmation for durable values;
- version, sign and audit value changes;
- distinguish preference, value, legal/safety constraint and temporary instruction;
- never let loyalty or rule flexibility override security/safety blocks;
- never convert value alignment into factual confidence;
- allow contradiction and uncertainty rather than silently overwriting one `general` stance;
- treat current fingerprint data as untrusted migration input.

## 8.3 D110 Cortex

Target role: **perception producer and situational hypothesis provider**.

Required changes:

- emit typed events rather than free-form Workspace authority;
- separate observed facts from model summaries and implications;
- preserve source freshness, uncertainty and provenance;
- prevent activity/sensor text from becoming control instructions;
- scope all state to the authenticated principal and purpose;
- treat inferred intent as a hypothesis, never proof of operator intent.

## 8.4 D111 Trust Ledger / Wisdom Graph

Target role: **verified outcome history, calibration and scoped reputation**, not authorisation.

Required changes:

- stop granting authority from loyalty, wording, acknowledgements, predictions or self-reported results;
- store immutable links from action proposal to executed operation and verified outcome;
- partition by principal, domain, capability and revision;
- support contradiction, supersession and expiry;
- require independent outcome evidence before credit;
- preserve negative and unknown outcomes;
- use trust as an input to bounded policy, never as a universal permission score.

## 8.5 D126–D130 and equivalent specialist modules

Target role: **perception or proposal providers**.

For the first financial vertical slice:

- `alpha_signals.py` → typed perception provider;
- `market_intel.py` → typed perception provider with source-quality metadata;
- `opportunity_intel.py` → proposal specialist, not directive authority;
- `strategy_engine.py` → proposal specialist; `auto_trade()` removed/disabled from the specialist boundary;
- `paper_trader.py` → actuator behind exact capability;
- market/portfolio state → independent outcome/valuation authority.

## 8.6 D132 onwards / WTE concept

No committed WTE implementation was found under the D132 label at the planning snapshot. The roadmap therefore treats WTE as a **target actuator/workflow role**, not an existing trusted component.

Existing services will become WTE-class actuators only after they satisfy the actuator contract and final-boundary enforcement tests.

---

## 9. Human authority model

Dainius remains the final authority for high-consequence operations, but the architecture must not force manual approval for every harmless internal calculation.

## 9.1 Risk tiers

### R0 — Internal observation/computation

Examples:

- calculate a summary;
- retrieve a scoped non-sensitive record;
- evaluate a model in an isolated test;
- produce an action proposal.

May run automatically under policy. No external side effect.

### R1 — Reversible local/test mutation

Examples:

- write a disposable test artefact;
- update isolated simulation state;
- run a bounded sandbox task.

May use narrowly pre-approved policy after P1/P2 qualification.

### R2 — Sensitive processing or external read

Examples:

- browse an external source;
- process private documents;
- access authenticated browser state;
- query financial/account data.

Requires authenticated scope, purpose, isolation and explicit policy. Human approval depends on data and destination.

### R3 — External communication or consequential reversible action

Examples:

- send an email/message;
- create or alter a calendar event;
- execute a paper trade that affects the performance record;
- change account settings;
- start recovery or data movement.

Requires exact action approval unless a separately qualified narrow standing policy exists.

### R4 — High-consequence action

Examples:

- real financial transaction;
- destructive delete/restore;
- public communication;
- administrative/security change;
- self-modification;
- broad autonomous recovery.

Remains disabled until its P4 domain qualification. Per-action step-up human approval is mandatory unless a future formally approved policy explicitly defines a narrower safe envelope.

## 9.2 Approval UX requirements

The approval screen must show:

- what Kai proposes;
- why and based on which evidence;
- what information is missing or conflicting;
- exact target, parameters and limits;
- expected and irreversible effects;
- risk class;
- alternatives, including no action;
- expiry;
- the immutable digest being approved.

The user must be able to deny, narrow or expire the action. Editing creates a new proposal and requires a new approval.

---

## 10. Control-plane and data-plane separation

The architecture separates:

### Cognitive/data plane

- perceptions;
- World State views;
- specialist analyses;
- proposals;
- explanations.

### Security control plane

- identity;
- delegation;
- policy;
- approval;
- capability issuance;
- revocation;
- audit integrity.

### Actuation plane

- fixed-schema action workers;
- external side effects;
- postcondition collection.

D102 belongs to the cognitive plane. Tool Gate/capability issuance belongs to the security control plane. WTE workers belong to the actuation plane.

No cognitive component can directly assume security-control authority.

---

## 11. Reliability and consistency model

## 11.1 Delivery semantics

Perception and workflow events are expected to be delivered **at least once**. Consumers must be idempotent and deduplicate by event/operation ID.

The roadmap does not claim magical end-to-end exactly-once delivery across external systems.

## 11.2 Transactional outbox

When a service changes its owned state and emits an event, both are committed in one local transaction. A relay publishes the event later.

This prevents state changes without events and events without committed state.

## 11.3 Sagas and compensation

Multi-service actions use durable workflows with explicit compensable, pivot and non-compensable steps.

Compensation is a new authorised action and may not be possible for every external effect.

## 11.4 Reconciliation

For every consequential operation, the system defines how to reconcile an unknown outcome against an independent target/source before retrying.

## 11.5 Leadership and writers

Logically singular stores or reducers use leases/fencing or transactional single-writer ownership. A process-local singleton is not a distributed leader.

---

## 12. Security model

## 12.1 Workload identity

Every producer, reducer, planner, policy service, actuator and verifier requires a verifiable workload identity. Network location and static IP are not identity.

SPIFFE/SPIRE is a candidate standard/implementation, not a mandatory product decision.

## 12.2 Policy decision and enforcement

Policy may be evaluated centrally or locally, but enforcement is local to the final side-effect service. A policy engine such as OPA is a candidate mechanism; the invariant is a versioned policy decision and a fail-closed enforcement point.

## 12.3 Capabilities

Capabilities are attenuated and audience restricted. Macaroon-style caveats or another cryptographically sound mechanism may be evaluated, but simple reusable shared HMAC body tokens are not accepted.

## 12.4 Zero trust

Every request is authenticated and authorised based on identity, resource, action, context and current policy. Internal network placement does not create implicit trust.

## 12.5 Content security

Untrusted text cannot populate identity, policy, capability, action or approval fields. Parsing and model inference operate in isolated workers. Control schemas reject extra/unknown privileged fields.

---

## 13. Observability and audit

Every request, event, proposal, decision, capability, workflow and outcome carries correlated trace/context identifiers.

Required correlation chain:

```text
trace_id
  → perception event IDs
  → world snapshot ID
  → proposal ID/digest
  → policy decision ID
  → approval ID
  → capability ID
  → workflow/operation ID
  → actuator receipt
  → observation IDs
  → verified outcome ID
  → learning/calibration update IDs
```

OpenTelemetry/W3C Trace Context may be used for transport correlation, but trace headers are not authentication or authorisation.

Audit records must be structured, minimised, signed/anchored under P3 and must not expose reusable credentials.

---

## 14. Migration strategy — no big-bang rewrite

## UH-0 — Preserve and map

Dependency: `P0-PR-01` evidence preservation.

Deliverables:

- immutable snapshot and evidence manifest;
- complete module-role inventory;
- complete direct decision-to-action call graph;
- side-effect endpoint registry;
- data/source/consumer lineage map;
- list of process-local stores and shared writable files;
- classification of current modules as provider, transformer, proposal specialist, policy, actuator or verifier;
- explicit list of paths that must remain disabled.

Exit gate:

- every consequential path has an owner and migration state;
- no new direct action path is permitted.

## UH-1 — Freeze canonical contracts

Maps primarily to P1.

Deliverables:

- versioned schemas for PerceptionEvent, WorldStateSnapshot, ActionProposal, ConstraintAssessment, ApprovalRecord, ActionCapability, ActionWorkflow and VerifiedOutcome;
- canonical serialisation/digest rules;
- risk tiers and approval matrix;
- schema compatibility policy;
- identity/delegation requirements;
- error/state vocabulary;
- architecture dependency rules.

Exit gate:

- contracts pass malformed, unknown-field, digest and compatibility tests;
- no free-form text field can alter a control field.

## UH-2 — Build the perception spine in shadow mode

Maps to P1/P2.

Deliverables:

- authenticated event ingress;
- schema registry/validation;
- durable journal and outbox;
- source and receipt timestamps;
- provenance and independence-group metadata;
- adapters for a small number of existing providers;
- shadow comparison against current outputs.

No actions are triggered from shadow events.

Exit gate:

- invalid/stale/duplicate events are rejected or explicitly classified;
- restart/replay reproduces the same accepted event sequence;
- cross-principal events cannot leak.

## UH-3 — Build scoped World State

Maps to P2/P3.

Deliverables:

- deterministic reducers;
- immutable snapshots;
- conflict/unknown/stale semantics;
- principal/purpose/data-class views;
- event-to-fact lineage;
- snapshot replay and digest verification;
- bounded retention and deletion lineage.

Exit gate:

- snapshots are reproducible;
- conflicting sources remain visible;
- deleted/superseded records do not remain active in derivatives.

## UH-4 — Rebuild D102 as proposal-only workspace

Contract foundation maps to P1/P2; cognitive qualification maps to P4.

Deliverables:

- registered authenticated bidders;
- typed proposal interface;
- evidence/assumption/dependency graph;
- alternatives and no-action option;
- contradiction and missing-evidence handling;
- deterministic proposal envelope;
- no imports or network permissions to actuators;
- no capability issuance.

Exit gate:

- winning a bid cannot execute anything;
- duplicate/correlated/stub bidders cannot create qualifying consensus;
- workspace outage blocks proposals requiring it rather than causing a bypass.

## UH-5 — Policy, human approval and capability bridge

Maps to P1/P3.

Deliverables:

- risk classification;
- policy-as-code decision;
- protected approval UI/API;
- exact digest binding;
- single-use audience-bound capabilities;
- revocation and expiry;
- audit linkage;
- final-boundary enforcement library.

Exit gate:

- anonymous/low-scope/XSS/replay/modified-action approvals fail;
- policy/approval service outage fails closed;
- actuator cannot use a capability intended for another actuator.

## UH-6 — Migrate one vertical slice

Recommended first slice: **paper-trading proposal to verified simulated outcome**.

Reason:

- it exercises multiple providers, proposal logic, policy, approval, actuator and outcome verification;
- it avoids real capital while retaining meaningful state and audit requirements;
- it directly removes the audited fragmented financial decision path.

Migration:

1. Alpha and market modules emit typed perceptions.
2. Opportunity/strategy modules emit proposals only.
3. D102 assembles one proposal using a versioned World State snapshot.
4. Policy and human approval evaluate the exact simulated operation.
5. Paper Trader executes only an exact capability.
6. Independent portfolio state verifies the result.
7. No trust/learning update occurs without verified outcome.
8. Legacy `auto_trade()` direct path is disabled and then removed.

Exit gate:

- no direct financial mutation path remains;
- correlation and stale-source tests fail safely;
- one signal cannot close unrelated positions;
- partial/unknown outcomes reconcile safely;
- the slice runs in shadow/test mode before any supervised enablement.

## UH-7 — Migrate remaining actuators by risk

Suggested order:

1. read-only data retrieval;
2. isolated local/test operations;
3. document and browser reads;
4. notifications/draft creation;
5. file mutations;
6. calendar/external messages;
7. recovery/admin operations;
8. financial/destructive/public/self-modifying operations last.

Each migration disables the old path before the new path is marked verified.

## UH-8 — Outcome-based learning and autonomy requalification

Maps to P4.

Deliverables:

- immutable claim/evidence service;
- outcome verifier registry;
- calibration by task/domain/revision;
- Trust Ledger replacement;
- explicit value confirmation workflow;
- Wisdom Graph lineage and contradiction;
- A0–A4 scoped autonomy authority;
- capability-specific signed release bundles.

Exit gate:

- self-generated text or simulation cannot grant trust;
- high-consequence domains pass separate attack-chain tests;
- autonomy remains bounded, expiring and revocable.

---

## 15. Required codebase-wide refactoring rules

These rules apply to Claude or any other implementer.

1. A provider package may not import an actuator package.
2. A proposal specialist may not call a side-effect endpoint.
3. D102 may not import or possess actuator credentials.
4. Ohana may block or request human review but cannot issue security permission.
5. Trust/conviction values may not bypass policy or human approval.
6. Every action route is registered and final-boundary enforced.
7. Direct legacy action APIs are disabled during migration and removed after verification.
8. Every state-changing method returns a typed operation state, not a success-shaped dictionary.
9. Every external effect has an idempotency/reconciliation design before implementation.
10. Every persistent record carries principal, purpose, data class, provenance and revision.
11. Every model-generated field is labelled model-generated.
12. Extra/unknown fields are rejected on privileged schemas.
13. Missing mandatory dependencies produce blocked/unavailable states.
14. No `except Exception: pass` or fail-open behaviour in policy, approval, execution, verification or persistence paths.
15. Feature flags may disable capability but are not the authority boundary.

A CI dependency rule should enforce forbidden imports/calls, supported by the side-effect registry and architecture tests.

---

## 16. Adversarial and failure tests

Minimum integrated suite:

1. Anonymous provider event injection.
2. Compromised provider attempts to impersonate another source.
3. NaN/infinity/negative/out-of-range payloads.
4. Oversized/deep/high-cardinality event payloads.
5. Duplicate and out-of-order events.
6. Stale source event received recently.
7. Conflicting sources and independence-group collision.
8. Prompt injection inside event payload attempts to change action fields.
9. Cross-principal World State access.
10. Reducer crash/restart/replay determinism.
11. D102 unavailable or split-brain.
12. One/duplicate/correlated/stub proposal specialists.
13. Ohana unavailable or poisoned values state.
14. Policy engine unavailable.
15. Human approval endpoint anonymous/XSS/CSRF/replay attempts.
16. Proposal changed after approval.
17. Capability used by wrong actuator.
18. Capability replay and concurrent consumption.
19. Actuator timeout after possible side effect.
20. Blind retry prevention and reconciliation.
21. Partial multi-step execution and compensation.
22. Actuator lies about success.
23. Outcome verifier unavailable or contradictory.
24. Self-generated prediction submitted as outcome evidence.
25. Old direct path remains callable after migration.
26. Rollback attempts to restore fail-open/legacy authority.
27. Multi-worker, restart, clock-change and leader-fencing tests.
28. Audit persistence failure before protected effect.
29. Event/trace context tampering.
30. End-to-end data deletion across source events, views, proposals, audit-allowed references and learning derivatives.

---

## 17. Anti-patterns explicitly rejected

- one giant `WorldState` dictionary shared by all modules;
- D102 as an all-powerful god service;
- “winning salience” treated as truth or permission;
- free-form module-to-module prompts as the control protocol;
- every module sharing the same database tables;
- majority vote treated as confidence;
- correlated indicators treated as independent specialists;
- Ohana loyalty overriding safety/security/legal constraints;
- automatic learning from Kai’s own answers;
- human approval inferred from normal chat;
- reusable admin credentials held by Dashboard;
- central policy with no final-boundary enforcement;
- event bus treated as the sole source of truth without durable ownership;
- dual writes without outbox;
- automatic retry after unknown external effect;
- executor self-verifying success;
- big-bang rewrite of every module;
- retaining old and new action paths “temporarily” in a protected profile;
- calling planning completion remediation.

---

## 18. Claude implementation supervision protocol

Every implementation PR must include:

```text
Unified Hunter work package
P0–P4 dependency
Finding IDs addressed
Architecture invariants affected
Module role before/after
Perception/event schemas
World State views
Proposal/operation digests
Side-effect routes
Identity/delegation/capability path
Risk class and approval path
Persistent data classes and lineage
Positive tests
Negative/adversarial tests
Failure/restart/reconciliation tests
Audit evidence
Legacy path disabled/removed
Rollback-to-disabled behaviour
Residual risks
```

Automatic rejection conditions:

- a new direct tool/action call from a provider or planner;
- a broad/shared credential introduced;
- fail-open handling in a protected path;
- untyped free-form execution parameters;
- mutable global authority state;
- self-reported outcome used as verified success;
- legacy bypass left enabled without explicit test-only isolation;
- missing adversarial tests;
- claims that findings are closed without the closure evidence package.

The auditor/project manager should review architecture impact before code quality. Code that is locally clean but restores fragmented decision authority is rejected.

---

## 19. Roadmap crosswalk

| Unified Hunter work | Existing programme dependency | Status |
|---|---|---|
| UH-0 preserve/map | P0 evidence preservation and containment | Planning complete; implementation not started |
| UH-1 contract freeze | P1 identity/canonical operation | Planning complete; implementation not started |
| UH-2 perception spine | P1 identity + P2 isolation/provenance | Planning complete; implementation not started |
| UH-3 World State | P2 data integrity + P3 durable state | Planning complete; implementation not started |
| UH-4 D102 proposal-only rebuild | P1/P2 foundation; P4 qualification | Planning complete; implementation not started |
| UH-5 policy/approval/capability | P1 enforcement + P3 audit | Planning complete; implementation not started |
| UH-6 paper-trading vertical slice | P0–P3 before supervised test; P4 before autonomy | Planning complete; implementation not started |
| UH-7 actuator migration | Applicable P1–P4 gates per capability | Planning complete; implementation not started |
| UH-8 verified learning/autonomy | P4 | Planning complete; implementation not started |

This addendum does not change the first authorised implementation step:

- `P0-PR-01` — preserve evidence and create the immutable acquisition manifest.

---

## 20. Research and standards basis

The design was checked against the following primary standards, official documentation and foundational research:

- NIST SP 800-207, **Zero Trust Architecture** — protect resources and authorise each access rather than trusting network location: https://doi.org/10.6028/NIST.SP.800-207
- NIST AI RMF 1.0 and Playbook — lifecycle governance, documented human oversight, measurement and explicit go/no-go decisions: https://doi.org/10.6028/NIST.AI.100-1
- CloudEvents — standard event envelope and interoperability: https://cloudevents.io/
- SPIFFE/SPIRE — verifiable workload identity and short-lived SVIDs: https://spiffe.io/docs/latest/spiffe-about/overview/
- Open Policy Agent — separation of policy decision points from application enforcement points: https://www.openpolicyagent.org/docs/deploy
- W3C Trace Context and OpenTelemetry context propagation — end-to-end distributed correlation: https://www.w3.org/TR/trace-context/
- OAuth 2.0 Resource Indicators, RFC 8707 — audience-restricted tokens: https://www.rfc-editor.org/rfc/rfc8707
- Google Research, **Macaroons: Cookies with Contextual Caveats for Decentralized Authorization in the Cloud** — attenuated contextual credentials: https://research.google/pubs/macaroons-cookies-with-contextual-caveats-for-decentralized-authorization-in-the-cloud/
- Google Research, **Zanzibar** — uniform authorisation and causal consistency: https://research.google/pubs/zanzibar-googles-consistent-global-authorization-system/
- Microsoft Azure Architecture Center, **Transactional Outbox** and **Saga** patterns — reliable event publication, durable multi-service workflows and compensation: https://learn.microsoft.com/en-us/azure/architecture/databases/guide/transactional-out-box-cosmos and https://learn.microsoft.com/en-us/azure/architecture/patterns/saga
- Temporal official documentation — durable workflow execution and recovery from process/infrastructure failure: https://docs.temporal.io/
- Barbara Hayes-Roth, **A Blackboard Architecture for Control** — separating domain knowledge from control of which action/problem-solving step should run: https://doi.org/10.1016/0004-3702(85)90063-3
- Bernard Baars, Global Workspace research — specialist processes contribute to a shared workspace; the workspace is a coordination architecture, not a security authority: https://doi.org/10.1016/S0079-6123(05)50004-9

These sources inform the architecture; they do not prove that the proposed Kai implementation is safe. That requires implementation and verification evidence.

---

## 21. Final architecture judgement

The user’s core concept is correct and stronger than the current fragmented system:

> One hunter should understand the situation and coordinate its tools; tools should not behave like independent creatures making unobserved decisions.

The production-grade version is:

> One coherent, auditable deliberation path; multiple bounded specialist senses; independent policy and human authority; narrowly scoped hands; verified outcomes; no self-authorising tool and no self-certified learning.

Current state remains:

- audit baseline: **4,580 findings**;
- runtime remediation: **none**;
- findings closed: **zero**;
- release decision: **NO_GO**.
