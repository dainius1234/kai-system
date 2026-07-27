# Kai System — Integrated Implementation Sequence and Closure Matrix

Repository: `dainius1234/kai-system`  
Authoritative audit baseline: **4,580 findings — 252 Critical, 2,440 High, 1,885 Medium, 3 Low**  
Status: **PLANNING AND CONTROL ARTEFACT ONLY — NO RUNTIME REMEDIATION PERFORMED**

Source plans:

- `kai-pm/CODE_AUDIT_P0_CONTAINMENT_PLAN.md`
- `kai-pm/CODE_AUDIT_P1_SECURITY_FOUNDATION_PLAN.md`
- `kai-pm/CODE_AUDIT_P2_ISOLATION_AND_INTEGRITY_PLAN.md`
- `kai-pm/CODE_AUDIT_P3_RELIABILITY_AUDIT_PRIVACY_RECOVERY_PLAN.md`
- `kai-pm/CODE_AUDIT_P4_CAPABILITY_REQUALIFICATION_PLAN.md`
- `kai-pm/KAI_UNIFIED_HUNTER_ARCHITECTURE_AND_ROADMAP.md`

Authoritative findings and judgement:

- `kai-pm/CODE_AUDIT_MASTER.md`
- `kai-pm/CODE_AUDIT_FINAL_REPORT.md`
- `kai-pm/CODE_AUDIT_BATCH_*.md`

---

## 1. Purpose

This matrix is the programme-level control document for converting the completed audit into implementation work without losing dependency order, reopening bypass paths or claiming closure from local patches.

It provides:

1. One ordered implementation sequence.
2. Cross-phase prerequisites and release gates.
3. Attack-chain closure ownership.
4. Finding-closure evidence rules.
5. Migration and rollback constraints.
6. Unified Hunter architecture integration across P0–P4.
7. A current status baseline showing that no finding is closed by planning work.

---

## 2. Non-negotiable programme rules

### Rule 1 — Containment precedes construction

Do not begin connected testing or capability re-enablement until the applicable P0 containment controls are verified.

### Rule 2 — No side-effect service ahead of identity and capability enforcement

P2/P3/P4 controls do not make an unauthenticated or bypassable service safe. P1 identity, delegation, canonical operation and final-boundary capability enforcement are mandatory prerequisites.

### Rule 3 — Local remediation does not close a systemic finding

A finding closes only when its complete affected path and downstream consumers satisfy the closure evidence. Patching one endpoint does not close an architecture or attack-chain finding.

### Rule 4 — Legacy and replacement paths cannot remain simultaneously usable

A migrated service must reject the legacy protocol before it can be marked verified. Dual-path migration is a bypass unless the legacy path is isolated and test-only.

### Rule 5 — Success requires immutable evidence

Code merge, a green shallow health check, a screenshot or an operator statement is not closure evidence. Required evidence includes immutable revision, production-equivalent configuration, negative tests, outcome verification and signed audit linkage.

### Rule 6 — Rollback may disable, never weaken

Rollback may turn a capability off or restore the last verified secure revision. It must not restore anonymous access, shared-secret identity, advisory policy, generic execution, fail-open recovery, direct specialist-to-actuator calls or production-shaped stubs.

### Rule 7 — Finding counts remain unchanged until formal closure review

Planning documents do not reduce the 4,580 total. Closure is a separate evidence-backed register action.

### Rule 8 — One governed perception-to-outcome path

Consequential actions must follow the canonical Unified Hunter sequence:

`Perception → World State → Proposal → Policy → Approval → Capability → Execution → Observation → Verification → Learning`

A specialist may perceive, analyse or propose. It may not independently authorise and execute the same consequential action. D102 Global Workspace is a proposal coordinator, not a security authority. Ohana alignment is not factual confidence or permission.

---

## 3. Programme waves

## Wave 0 — Evidence preservation and immediate isolation

### Required work

- P0 evidence freeze and manifest.
- Remove/loopback-bind privileged host ports.
- Disable dangerous services and profiles by default.
- Lock Tool Gate and Dashboard policy administration.
- Remove known/default credential fallbacks from active profiles.
- Temporary trust-zone segmentation.
- Single-writer containment for shared indexes.
- Freeze automatic recovery and unverified scheduled mutation.
- Add deployment policy checks.
- Inventory every current decision-to-action call path and side-effect endpoint.
- Freeze new direct specialist-to-actuator paths.
- Classify modules as perception provider, transformer/reducer, proposal specialist, policy/approval authority, actuator or outcome verifier.

### Exit gate W0

- Evidence copies verified independently.
- No privileged service reachable from Internet/shared LAN.
- Executor, browser, recovery, finance and external messaging disabled by default.
- Tool Gate cannot start permissive from missing/invalid configuration.
- No known development credential accepted in the protected profile.
- Compose/service inventory policy check passes.
- Every consequential route appears in the side-effect/decision-path inventory.
- No unregistered direct action path is permitted to be added.

### Permitted state after W0

Isolated disposable development laboratory only.

---

## Wave 1 — Identity, canonical contracts and effect-boundary enforcement

### Required work

- P1 human principal authentication.
- Unique workload identity and authenticated transport.
- Explicit narrow delegation.
- Canonical operation envelope/digest.
- Protected operator approval.
- Tool Gate decision rebuild.
- Single-use audience-bound capability.
- Atomic capability consumption.
- Executor pilot enforcement.
- Dashboard confused-deputy removal.
- Agentic principal/delegation migration.
- Side-effect route registry and migration.
- Legacy shared-HMAC/body-token/cosign removal.
- Freeze Unified Hunter contracts for:
  - `PerceptionEvent`;
  - `WorldStateSnapshot`;
  - `ActionProposal`;
  - `ConstraintAssessment`;
  - `ApprovalRecord`;
  - `ActionCapability`;
  - `ActionWorkflow`;
  - `VerifiedOutcome`.
- Define canonical serialisation and digests across perception, proposal, approval, execution and outcome.
- Define risk classes and the operator-approval matrix.
- Add architecture dependency rules prohibiting provider/planner imports or calls into actuators.
- Define D102 as proposal-only and D109 Ohana as constraints/value advice only.

### Exit gate W1

- Anonymous privileged request rejected.
- Workload cannot impersonate another service or operator.
- Changing any consequential operation field invalidates approval/capability.
- Direct Executor/action-service bypass rejected before side effect.
- Dashboard holds no reusable operator/admin credential.
- Agentic cannot turn anonymous input into trusted Gate identity.
- All registered side-effect endpoints enforce capabilities.
- Legacy protocol use is zero in release tests.
- Privileged schemas reject unknown control fields and free-form text cannot alter identity/action/policy fields.
- A Global Workspace proposal cannot itself grant permission or call an actuator.
- Ohana/Trust/conviction cannot override policy or an explicit block.

### Permitted state after W1

Privileged internal testing only; no hostile content, sensitive data, generic execution or broad external actions.

---

## Wave 2 — Perception spine, World State, isolation, egress and persistent-data integrity

### Required work

- Replace generic command interpreters with fixed-schema operations.
- Disposable execution workers.
- Controlled egress/SSRF authority.
- Per-principal/workflow browser contexts.
- Exact browser actions and postconditions.
- Isolated Monitor rule execution.
- Upload quarantine and format detection.
- Archive/OOXML preflight.
- Disposable parser/converter workers.
- Provenance-rich parser outputs.
- Secure Vault object/path model.
- Principal/tenant/purpose data partitioning.
- Immutable evidence and untrusted-content boundary.
- Safe prompt context assembly.
- Durable memory/vector/graph state machine.
- Graph partitioning and lineage.
- Atomic supersession/contradiction.
- End-to-end derivative deletion.
- Authenticated typed Perception ingress and schema registry.
- Durable event journal and transactional outbox.
- Source-event time, receipt time, freshness and independence-group metadata.
- Deterministic reducers and immutable versioned World State snapshots.
- Explicit known/unknown/stale/conflicting/unavailable fact states.
- Event-to-fact and snapshot-to-proposal lineage.
- Shadow-mode adapters for selected existing providers.
- Rebuild D110 Cortex as an observation/hypothesis provider, not operator-intent authority.
- Rebuild D126–D130-class modules as perception or proposal providers without action authority.

### Exit gate W2

- No arbitrary-code route in approved operations.
- Worker escape/process/network/filesystem/resource tests pass.
- Browser authenticated state cannot cross principal/workflow.
- SSRF/DNS rebinding/redirect attacks fail safely.
- Hostile archive/parser/converter tests pass in disposable workers.
- Every persistent record is principal/purpose/class scoped.
- External/model/document data cannot enter trusted prompt/evidence roles directly.
- Memory/vector/graph partial failure is visible/recoverable.
- Superseded/deleted records cannot remain active through derivatives.
- Invalid, stale, duplicate and out-of-order perception events are rejected or explicitly classified.
- World State snapshots are reproducible from retained accepted events and reducer revision.
- Conflicting evidence remains visible rather than becoming false consensus.
- Shadow providers cannot trigger action.

### Permitted state after W2

Controlled hostile-content and sensitive-data testing under W0–W2 constraints; no production qualification or consequential autonomy.

---

## Wave 3 — Durable action workflows, audit, privacy, recovery and backup

### Required work

- Standard state/error/health contracts.
- Correct retry and unknown-outcome reconciliation.
- Durable operation/idempotency authority.
- Transactional outbox/inbox.
- Shared breaker/dependency state.
- Fenced leader election and scheduler ownership.
- Supervisor observation-only rebuild.
- Incident/recovery authority and service-specific actions.
- Remove fabricated healing success/knowledge.
- Transactional audit sequencer.
- Signed immutable audit segments and external checkpoints.
- Tool Gate/Trust Ledger migration.
- Data classification and purpose annotations.
- Encryption/key management.
- Retention/deletion/legal hold.
- Structured operational logging.
- Coherent signed backups.
- Isolated restore qualification.
- Incident response/evidence preservation.
- Integrated chaos and operational release evidence.
- Durable Unified Hunter `ActionWorkflow` state machine.
- Independent outcome-observation and verification services.
- End-to-end trace/correlation from perception event to verified outcome.
- Human approval history bound to exact proposal/operation digests.
- Reconciliation before retry after external timeout or uncertain result.
- Paper-trading vertical slice as the first full provider-to-verified-outcome migration.
- Disable/remove the legacy `strategy_engine.auto_trade()` decision-to-action path when the replacement slice is verified.

### Exit gate W3

- Error/stub/degraded/fallback cannot look successful.
- Distributed mutation executes once logically through timeout/failover.
- Multi-worker security/reliability state converges.
- Stale leaders cannot commit.
- Health observation cannot directly trigger recovery mutation.
- Recovery requires exact authority and independent postcondition.
- Audit append is linear, signed, segmented and externally anchored.
- Protected effect cannot succeed without required audit.
- Sensitive data is classified, encrypted and retention governed.
- Logs are structured/minimised and injection resistant.
- Backups are immutable/manifest-bound and regularly restore-qualified.
- Chaos, race, clock, privacy and restore tests pass.
- Actuator response alone cannot establish success.
- Timeouts produce `UNKNOWN` and reconcile before retry.
- Capability replay, wrong audience and concurrent consumption fail.
- The migrated vertical slice has no usable legacy bypass.

### Permitted state after W3

Formal production qualification of non-autonomous, explicitly released capabilities. Model judgement, trust and autonomy remain unqualified.

---

## Wave 4 — Deliberation, evidence, trust and autonomy requalification

### Required work

- Signed authoritative capability/model/tool/service registry.
- Exact model/backend attestation and fresh readiness.
- Reproducible benchmark authority.
- Qualified model selection/failover.
- Remove heuristic/style-based execution conviction.
- Immutable claim/evidence service.
- Proposition-level Verifier rebuild.
- Verifier enforcement integration.
- Qualified specialist/Fusion registry.
- Structured agreement/contradiction synthesis.
- Prediction/action/observation/outcome separation.
- Calibration service.
- Scoped outcome-based trust.
- Staged A0–A4 autonomy authority.
- Financial-domain qualification.
- Public-communication qualification.
- Destructive/admin/recovery qualification.
- Self-modification review pipeline.
- Stub/fallback truthfulness.
- Capability-specific integrated release gate.
- Rebuild/qualify D102 as a proposal-only Global Workspace using authenticated typed bidders and versioned World State.
- Rebuild D109 Ohana around explicit authenticated operator-confirmed values and constraints.
- Rebuild D111 Trust/Wisdom learning around independently verified outcomes, contradiction, scope and expiry.
- Require alternatives and a no-action option in consequential proposals.
- Prevent salience, loyalty, wording, majority vote or model confidence from becoming authority.
- Migrate remaining actuators in risk order and issue capability-specific release bundles.

### Exit gate W4

- All models/backends/tool capabilities resolve from one signed registry.
- No model is selected without exact attestation, readiness and task qualification.
- Stub/fake/fallback cannot create benchmark, consensus, trust or GO state.
- Caller cannot fabricate evidence/ranking/PASS.
- Claim-level contradiction and source independence are enforced.
- Empty/one/duplicate/correlated specialists cannot create consensus.
- Verifier blocks consequential output/action when required.
- Style, wording, hedging and formatting cannot increase authority.
- Trust credit requires linked independently verified outcomes.
- Autonomy is scoped, budgeted, expiring and revision-bound.
- High-consequence domains pass their separate attack-chain qualification.
- Every released capability has a signed evidence bundle and suspension/rollback plan.
- D102 cannot execute, approve or issue capability.
- Ohana cannot convert loyalty/value alignment into factual confidence or security permission.
- Predictions, simulations, reflections and actuator claims cannot be recorded as qualifying outcomes.

### Permitted state after W4

Only individually qualified capabilities may be enabled at their approved release state. All others remain disabled/test/advisory.

---

## 4. Unified Hunter migration workstreams

| Workstream | Programme mapping | Required result |
|---|---|---|
| UH-0 Preserve and map | W0 | Evidence manifest; module-role map; direct decision/action call graph; side-effect registry |
| UH-1 Contract freeze | W1 | Typed versioned perception/state/proposal/approval/capability/workflow/outcome contracts |
| UH-2 Perception spine | W1–W2 | Authenticated ingress, schema validation, journal/outbox, provider adapters in shadow mode |
| UH-3 World State | W2–W3 | Immutable scoped reproducible views with conflict, freshness and lineage |
| UH-4 D102 proposal-only rebuild | W1 contract; W2 data; W4 qualification | Shared deliberation with no action authority |
| UH-5 Policy/approval/capability bridge | W1–W3 | Exact policy, protected human approval and one-time final-boundary capabilities |
| UH-6 Paper-trading vertical slice | W0–W3; W4 before autonomy | End-to-end proposal, approval, simulation, independent verification and legacy-path removal |
| UH-7 Remaining actuator migration | Applicable W1–W4 gates | Old path disabled before replacement is verified |
| UH-8 Verified learning/autonomy | W4 | Outcome-only calibration/trust and scoped expiring autonomy |

---

## 5. Critical dependency matrix

| Work package | Depends on | Blocks |
|---|---|---|
| Evidence preservation | None | Secret rotation, destructive cleanup, restore/rebuild, decision-path mutation |
| Host exposure containment | None | Connected development |
| Decision/action path inventory | Evidence preservation | Unified Hunter migration, proof that bypasses are removed |
| Human/workload identity | P0 exposure/secret containment | Delegation, event authenticity, Gate rebuild, all protected APIs |
| Canonical perception/event contract | Workload identity + schema governance | Perception spine and reproducible World State |
| Canonical World State/proposal contracts | Principal identity + perception contract | D102 rebuild, exact approval and evidence lineage |
| Canonical operation digest | Principal identity + proposal contract | Approval, capability, idempotency, audit |
| Single-use capability | Delegation + operation digest | All side-effect release |
| Final-boundary enforcement | Capability | Executor/browser/data/recovery/finance release |
| Provider/proposal/actuator role separation | Contracts + side-effect registry | Removal of fragmented decision authority |
| Perception journal/outbox | Identity + event contract | Replayable state and reliable provider integration |
| World State reducers | Perception journal + partition/provenance | Shared deliberation and conflict-aware proposals |
| D102 proposal-only coordinator | World State + proposal contract | Unified deliberation; later cognitive qualification |
| Protected human approval | Principal auth + exact proposal/operation digest | R3/R4 action release |
| Durable ActionWorkflow | Capability + shared transactional state | Safe multi-step execution, retries and reconciliation |
| Independent outcome verification | ActionWorkflow + provenance + audit | Learning, trust, calibration and action closure |
| Disposable execution/parser workers | P1 enforcement | Hostile execution/document processing |
| Controlled egress | P1 enforcement + P0 segmentation | Browser/Web/Monitor/parser network release |
| Principal data partition | P1 principal context | Sensitive data/memory release |
| Provenance/lineage | Data partition + operation identity | Verification, trust, deletion |
| Durable operations/outbox | P1 digest + P2 state model | Multi-worker/production reliability |
| Fenced leadership | Shared transactional state | Schedulers, recovery, backup, audit checkpoints, reducers |
| Audit authority | Operation identity + shared state | Protected production effects, trust/outcome evidence |
| Privacy/retention | Data classification + P2 lineage | Sensitive production data |
| Recovery authority | P1 capability + P3 incidents/leadership/audit | Automatic/manual recovery release |
| Backup/restore qualification | Data/operation consistency + audit + privacy | Production recovery claims |
| Model registry/attestation | P1–P3 foundations | Model qualification/Fusion/Verifier/D102 bidders |
| Verifier rebuild | P2 evidence + P3 audit + registry | Consequential model output/action |
| Ohana value migration | Authenticated operator confirmation + provenance | Values advice; never security permission |
| Calibration/trust | Independently verified outcomes + audit | Autonomy levels |
| Domain autonomy | All applicable W0–W4 gates | Financial/public/destructive/self-modifying release |

---

## 6. Attack-chain closure matrix

| Attack chain | Primary closure controls | Required closure test | Status |
|---|---|---|---|
| Dashboard stored XSS → control-plane compromise | P0 isolation; P1 authenticated ingress/confused-deputy removal; CSP/frontend remediation; capability enforcement | Inject hostile feed/mail/system content and attempt every privileged proxy | **OPEN** |
| Anonymous Dashboard → Gate mode change | P1 principal auth; no server-held admin token; step-up admin operation | Anonymous/low-scope/XSS mode-change attempts | **OPEN** |
| Anonymous Agentic input → trusted Gate action | P1 ingress/delegation/operation binding; P4 model/action qualification | Anonymous `/run`/task-hint escalation | **OPEN** |
| Direct Executor → arbitrary code/fleet pivot | P0 disable/isolate; P1 final enforcement; P2 fixed operations/sandbox/egress | Direct capability bypass and escape suite | **OPEN** |
| memU poisoning → privileged system prompt | P1 principal scope; P2 provenance/untrusted context; P4 evidence eligibility | Persistent preference/correction injection across principals | **OPEN** |
| Forged Verifier evidence → PASS | P2 immutable evidence; P4 claim-level Verifier | Duplicate/caller-ranked/contradictory evidence pack | **OPEN** |
| Fusion manufactured consensus | P4 specialist registry, independence, strict Verifier enforcement | Empty/one/duplicate/stub/correlated specialist suite | **OPEN** |
| External content → Dashboard XSS | P2 untrusted-content schema; frontend safe rendering/CSP; P1 privilege separation | Email/RSS/broker/document payload to privileged action | **OPEN** |
| Vault arbitrary file → memory/prompt exfiltration | P1 capability/principal; P2 object/path policy, provenance, egress/data partition | Container-secret/path ingest and retrieval | **OPEN** |
| Gate/ledger disclosure → lateral privilege | P1 credential separation; P3 minimised audit-reader access | Low-scope credential attempts ledger secret extraction | **OPEN** |
| Health manipulation → recovery reset | P0 recovery freeze; P3 observation/action split, incidents, capability, postcondition | Repeated/spoofed health sweep/recovery | **OPEN** |
| Values/loyalty feedback → trust inflation | P1 principal; P2 provenance; P4 outcome-only trust | Anonymous feedback/value/acknowledgement escalation | **OPEN** |
| Weak market signal → financial mutation | P1 final enforcement; P3 durable finance operation; P4 financial qualification | One-source/correlated/invalid/stale signal suite | **OPEN** |
| Specialist module → local decision → direct actuator | Unified Hunter role separation; side-effect registry; exact capability; legacy path removal | Call each specialist/direct singleton/API and prove no side effect is reachable | **OPEN** |
| D102/Ohana/Trust score → self-authorised action | Proposal-only D102; Ohana advisory constraints; P1 capability; P4 outcome-only trust | High salience/loyalty/conviction with absent policy/approval must remain blocked | **OPEN** |
| Actuator self-reports success → learning/trust inflation | P3 independent outcome verifier; P4 verified-outcome eligibility | False/partial/timeout actuator responses must not create success/trust credit | **OPEN** |

No attack-chain row can move to CLOSED from a single component test. Every primary and alternate route must be exercised under production-equivalent deployment.

---

## 7. Finding closure evidence standard

A finding may be proposed for closure only when the evidence package contains:

1. Finding ID and owning batch.
2. Exact affected source/configuration paths.
3. Root-cause statement and applicable invariant.
4. Immutable remediation commit(s).
5. Built image/artefact digests.
6. Configuration/registry/policy/schema revisions.
7. Positive functional tests.
8. Negative/adversarial tests matching the exploit condition.
9. Integration test across downstream consumers.
10. Multi-worker/restart/failure test where applicable.
11. Audit/outcome evidence.
12. Proof that direct/legacy authority paths are disabled or removed.
13. Residual risk and known exclusions.
14. Independent reviewer and approval date.
15. Qualification expiry/retest trigger where applicable.

Closure states:

- `OPEN`
- `IMPLEMENTATION_IN_PROGRESS`
- `IMPLEMENTED_NOT_VERIFIED`
- `VERIFICATION_FAILED`
- `VERIFIED_PENDING_REVIEW`
- `CLOSED`
- `RISK_ACCEPTED` — requires explicit documented authority; does not mean fixed.
- `NOT_APPLICABLE_AFTER_ARCHITECTURE_REMOVAL` — requires proof the affected capability/path no longer exists.

---

## 8. Architecture finding closure rules

### KAI-ARCH-001 — Principal/delegation authority

Cannot close until every protected ingress, perception producer and side-effect route uses verified principal/workload/delegation context and body-supplied identity cannot grant authority.

### KAI-ARCH-002 — Decision/enforcement split

Cannot close until the side-effect registry is complete and every consequential route atomically consumes the exact capability. Unified Hunter proposals, D102, Ohana, Trust and specialist modules cannot bypass enforcement.

### KAI-ARCH-003 — Canonical operation binding

Cannot close until proposal, request, approval, capability, execution, outcome, idempotency and audit share one digest across all protected services.

### KAI-ARCH-004 — Evidence/provenance authority

Cannot close until all trusted evidence and World State facts use immutable typed sources/lineage and Verifier rejects caller authority, duplicates, correlated sources and untrusted derivatives.

### KAI-ARCH-005 — Cross-service transaction model

Cannot close until consequential multi-service mutations use durable state machines/outbox/inbox/compensation and unknown outcomes reconcile safely.

### KAI-ARCH-006 — Capability sandbox

Cannot close until generic execution is removed and execution/browser/parser workers satisfy isolation/egress/resource/descendant tests.

### KAI-ARCH-007 — Human approval object

Cannot close until step-up authenticated approval binds one exact proposal/operation and service/runtime/model tokens cannot approve.

### KAI-ARCH-008 — Recursive self-certification

Cannot close until predictions, proposals, actions, reflections and simulations cannot become qualifying outcomes and trust/calibration use independent resolution.

### KAI-ARCH-009 — Global personal-state namespace

Cannot close until every personal/behavioural/financial/sensor/memory event, World State view and derivative is principal/tenant/purpose partitioned.

### KAI-ARCH-010 — Data lifecycle model

Cannot close until classification, encryption, retention, lineage deletion, event/snapshot expiry, backup expiry and legal hold are machine enforced.

### Unified Hunter programme closure rule — not a new finding ID

The fragmented-decision root cause is not considered removed until:

- every material specialist has an assigned role and typed contract;
- no provider/proposal specialist can directly invoke a consequential actuator;
- D102 creates proposals only;
- Ohana/Trust/conviction cannot grant security authority;
- every action uses exact policy/approval/capability enforcement;
- the real outcome is independently verified before learning;
- every legacy bypass has been removed or proven isolated test-only.

---

## 9. Implementation branch and PR discipline

Recommended structure:

- One programme branch per wave or narrowly scoped workstream.
- Small implementation PRs matching the numbered plan PRs/UH workstreams.
- No mixed security-foundation and capability re-enablement PR.
- Every PR names dependencies and affected finding IDs.
- Every PR includes rollback-to-disabled behaviour.
- Feature flags default off and are not security boundaries.
- Migration state is machine-readable and release-checked.
- Protected profiles reject TODO/stub/compatibility mode.
- Architecture dependency tests prevent providers/planners from importing or calling actuators.

Required PR metadata:

```text
Wave / Plan PR / Unified Hunter workstream
Finding IDs
Architecture invariants
Module role before and after
Perception / World State / proposal contracts
Side-effect routes
Identity / delegation / capability path
Data classes and lineage
Migration state before/after
Legacy path disabled/removed
Tests/evidence
Outcome verification
Rollback behaviour
Residual risks
```

Automatic programme rejection conditions:

- new direct specialist-to-actuator path;
- broad/shared credential;
- fail-open policy/approval/execution/verification/persistence;
- free-form executable control;
- mutable global authority state;
- self-reported outcome treated as verified;
- legacy bypass retained in protected deployment;
- missing adversarial or reconciliation tests.

---

## 10. Release decisions

System-level status remains:

# **NO_GO**

Release decisions are capability-specific and revision-bound after W4. Suggested states:

- `DISABLED`
- `ISOLATED_TEST_ONLY`
- `ADVISORY_ONLY`
- `SUPERVISED_INTERNAL`
- `SUPERVISED_PRODUCTION`
- `NARROW_AUTONOMOUS`
- `SUSPENDED`
- `REVOKED`

There is no blanket “Kai is safe” decision. A capability release does not release unrelated services, data types, domains or autonomy scopes.

---

## 11. Current programme status

| Area | Status |
|---|---|
| Source/deployment audit | **COMPLETE for reviewed snapshot** |
| Final finding reconciliation | **COMPLETE — 4,580 findings** |
| Full executive/explanatory report | **COMPLETE** |
| Prioritised backlog | **COMPLETE** |
| Phase 0 containment plan | **COMPLETE — planning only** |
| Phase 1 security-foundation plan | **COMPLETE — planning only** |
| Phase 2 isolation/integrity plan | **COMPLETE — planning only** |
| Phase 3 reliability/audit/privacy/recovery plan | **COMPLETE — planning only** |
| Phase 4 capability-requalification plan | **COMPLETE — planning only** |
| Unified Hunter architecture and migration roadmap | **COMPLETE — planning only** |
| Runtime remediation | **NOT STARTED by this programme work** |
| Formally verified closed findings | **0 by planning work** |
| Overall release decision | **NO_GO** |

---

## 12. First authorised implementation step

Under the existing no-remediation instruction, no runtime change is performed.

When implementation is explicitly authorised, the first action is:

- `P0-PR-01` — preserve evidence and create the immutable acquisition manifest before secrets, volumes, networks, indexes, logs, ledgers, decision paths or deployment behaviour are altered.

The next action after evidence preservation is host exposure containment and decision/side-effect path inventory, not feature development, D102 activation, D131 coding or local vulnerability patching.

---

## Final programme judgement

The audit cannot be responsibly remediated as a flat queue of 4,580 independent tickets. The findings are dominated by shared architectural causes and end-to-end compromise paths.

The only defensible sequence is:

1. evidence preservation and containment;
2. identity, typed contracts and final-boundary enforcement;
3. perception spine, scoped World State, isolation and data integrity;
4. durable action workflows, independent outcome verification, audit, privacy and recovery;
5. D102/Ohana/Trust/model requalification and risk-bounded capability release.

The “hunter with different tools” concept is adopted as the functional target, but the hunter’s reasoning, authority, hands and outcome verification remain separated by enforceable contracts.

**No runtime remediation is performed and no finding is closed by this matrix. Current status remains NO_GO.**
