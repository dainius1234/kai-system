# Kai System — Phase 4 Capability Requalification Plan

Repository: `dainius1234/kai-system`  
Authoritative audit baseline: **4,580 findings — 252 Critical, 2,440 High, 1,885 Medium, 3 Low**  
Parent backlog: `kai-pm/CODE_AUDIT_REMEDIATION_BACKLOG.md`  
Dependencies:

- `kai-pm/CODE_AUDIT_P0_CONTAINMENT_PLAN.md`
- `kai-pm/CODE_AUDIT_P1_SECURITY_FOUNDATION_PLAN.md`
- `kai-pm/CODE_AUDIT_P2_ISOLATION_AND_INTEGRITY_PLAN.md`
- `kai-pm/CODE_AUDIT_P3_RELIABILITY_AUDIT_PRIVACY_RECOVERY_PLAN.md`

Status: **IMPLEMENTATION DESIGN ONLY — NO RUNTIME REMEDIATION PERFORMED**

---

## 1. Objective

Requalify Kai’s models, verification, evidence aggregation, confidence, trust and autonomy only after the P0–P3 identity, isolation, integrity, reliability, audit and privacy foundations are implemented and independently verified.

Phase 4 establishes:

1. One authoritative service/model/tool/capability registry.
2. Verified immutable model and backend identity.
3. Reproducible capability and benchmark evidence.
4. Proposition-level evidence verification with contradiction handling.
5. Explicit source independence and correlation modelling.
6. Fusion that reports agreement, disagreement and uncertainty without manufacturing consensus.
7. Calibrated uncertainty rather than style/wording/heuristic “conviction”.
8. Strict separation of prediction, proposed action, executed action, observation and verified outcome.
9. Trust/autonomy scores based only on linked externally observed outcomes.
10. Staged re-enablement for financial, public, destructive, self-modifying and autonomous actions.

Phase 4 is the final planning stage. It does not authorise release by itself. Every capability remains disabled until its specific evidence bundle and complete attack-chain tests satisfy the release gates defined here.

---

## 2. Governing capability and autonomy invariants

### INV-P4-01 — Registry identity is authoritative and immutable

Every service, model, backend, tool and capability used in a protected workflow is resolved from one signed, versioned registry. Runtime strings, environment names, caller lists and local dictionaries are not capability authority.

### INV-P4-02 — A model identity names an exact artefact and runtime

A model identity includes:

- provider/runtime;
- immutable model artefact digest or provider model revision;
- tokenizer and context policy;
- quantisation/build revision;
- system prompt/policy revision;
- tool schema revision;
- execution environment/image digest;
- endpoint/workload identity;
- current readiness and capacity evidence.

A configured name such as “Ollama” or a future/unknown remote model is not proof of availability or quality.

### INV-P4-03 — No stub or fallback can impersonate real capability

A stub, canned response, fake embedding, static heuristic, disabled verifier, unavailable provider or fallback backend uses an explicit non-operational state and cannot produce:

- consensus;
- PASS verification;
- benchmark success;
- calibrated confidence;
- trust credit;
- autonomy evidence;
- production readiness.

### INV-P4-04 — Verification operates on claims and immutable evidence

Verifier resolves signed evidence references from authoritative stores. The requesting model/caller cannot provide its own ranking scores or evidence authority. Each material claim is evaluated separately for support, contradiction, uncertainty, source quality, freshness and independence.

### INV-P4-05 — Agreement is not correctness

Textual similarity, vote share, repeated model names, one specialist, shared prompts, correlated heuristics and deterministic stubs cannot be labelled consensus or confidence. Fusion must distinguish:

- number of requested specialists;
- number of distinct verified backends;
- successful outputs;
- source/model dependence;
- proposition agreement;
- unresolved contradictions;
- verification state.

### INV-P4-06 — Uncertainty is calibrated and task-specific

Confidence is a measured property of a defined task/outcome distribution, not a number inferred from verbosity, punctuation, hedging, moral language, vote share, memory frequency or caller assertion.

### INV-P4-07 — Self-generated records cannot certify themselves

Predictions, plans, reflections, model critiques, simulated results, paper trades and operator acknowledgements are not successful outcomes. Trust credit requires an independently observed, linked and verified outcome.

### INV-P4-08 — Consequential decisions fail closed on evidence gaps

Financial, public, destructive, privacy-sensitive, security-administrative and self-modifying actions require complete fresh evidence, verified models, policy readiness and the P1 capability path. Unavailable Verifier/Gate/audit/data sources cannot increase conviction or become neutral success.

### INV-P4-09 — Autonomy is narrow, reversible and budgeted

Autonomy is granted per principal, purpose, operation class, resource, consequence budget, time window and verified competence domain. A global Trust Level or general “GUARDIAN” state cannot unlock unrelated actions.

### INV-P4-10 — Requalification is attack-chain based

A local component test does not qualify a capability. Release requires the full path from input/evidence through model/verification/policy/capability/final effect/outcome/audit/recovery to pass under production-equivalent configuration.

---

## 3. Confirmed source conditions driving Phase 4

### 3.1 Model selector and Model Council

Primary sources:

- `agentic/model_selector.py`
- `agentic/model_council.py`
- `common/model_registry.py`
- shared model runtime/configuration modules.

Primary audits:

- `kai-pm/CODE_AUDIT_BATCH_MODEL_DECISION_CONTROLS.md`
- `kai-pm/CODE_AUDIT_BATCH_COMMON_MODEL_RUNTIME.md`
- GPU/model utility and foundation-stub batches.

Confirmed conditions include:

- An explicitly empty available-model list expands to all registered models.
- One candidate bypasses registry/readiness validation.
- Unknown candidates can silently fall back to Ollama.
- Multiple hard-coded model registries disagree.
- Model complexity/routing uses wording length, punctuation and keywords.
- Runtime code can overwrite profiles and primary model without governance.
- Built-in models begin available without live discovery or credentials.
- Static heuristics are persisted as successful benchmark evidence.
- Injected benchmark scores are unauthenticated and unbounded.
- Availability can be changed by ordinary success/failure calls.
- Persistence failures can still report successful model mutation.
- Model identity is configuration text rather than verified artefact/runtime proof.

### 3.2 Adversary, tree search and decision scoring

Primary audits:

- Model Decision Controls.
- Conviction, planning, adversary and cognitive audits.

Confirmed conditions include:

- Verifier unavailability, Tool Gate unavailability and security-audit failure can be recorded as passed challenges.
- Missing Gate mode can default to WORK.
- Unknown verifier verdicts can be interpreted as PASS.
- Superficial plan structure increases confidence.
- History/calibration trusts caller-controlled or self-generated episode outcomes.
- Claimed independent challenges share memory and heuristic assumptions.
- Tree search generates prompt suffixes rather than independently reasoned candidate outputs.
- The first branch reaching threshold can be selected rather than the best validated branch.
- Prompt wording inflates branch scores.
- Failed debate may only attach metadata and not block the action.

### 3.3 Verifier

Primary source:

- Verifier service application and policy.

Primary audit:

- `kai-pm/CODE_AUDIT_BATCH_VERIFIER.md`

Confirmed conditions include:

- Caller-supplied evidence packs bypass authoritative retrieval and can fabricate PASS.
- Evidence scoring measures rank and word overlap, not entailment/contradiction.
- Duplicate/same-source records count as independent evidence.
- Caller plan formatting contributes a perfect consistency score.
- Financial/configuration claims can be excluded from material checks.
- Verification reads/mutates global `keeper` evidence state.
- One evidence chunk can satisfy unrelated claims.
- Poisoned/pinned/synthetic memories are accepted.
- No evidence or retrieval outage may yield REPAIR instead of FAIL_CLOSED.
- Context length and hedge words add plausibility confidence.
- Unknown/unrecognised claims can receive a strong positive default.
- SAGE self-critique double-counts the same signals.
- Verdict evidence is not durably bound to the full request/evidence/policy digest.

### 3.4 Fusion Engine

Primary source:

- `fusion-engine/app.py`

Primary audits:

- `kai-pm/CODE_AUDIT_BATCH_FUSION_ENGINE.md`
- `kai-pm/CODE_AUDIT_BATCH_FUSION_ENGINE_EXTENSION.md`

Confirmed conditions include:

- Empty and single-specialist sets receive 100% agreement.
- Caller-controlled zero threshold makes any result consensus.
- Duplicate specialist names manufacture multiplicity.
- Deterministic stubs participate as evidence.
- Caller context is placed into specialist system prompts.
- Verifier output does not veto consensus.
- `require_consensus` does not actually enforce consensus.
- Text similarity is labelled agreement/correctness.
- Merge selects longest response rather than resolving propositions/conflicts.
- Only a prefix of the returned response may be verified.
- Health can be green while every backend is a stub.

### 3.5 Trust, personality and autonomy evidence

Primary audits:

- Trust Governance Authority and Trust Ledger batches.
- Behavioural Feedback Tools.
- Cognitive Governance Foundations.
- Personality/autonomy P17–P22 batches.
- Autonomous State.

Confirmed conditions include:

- Anonymous feedback, values, conscience, loyalty and acknowledgement records can influence governance.
- Trust scoring can reward explicit zero/missing measurements through favourable defaults.
- Operator acknowledgement is treated as execution success without outcome evidence.
- Overrides are global and not linked to the action corrected.
- Global `keeper` state mixes callers/principals.
- Self-generated reflections, predictions and alignment assessments re-enter memory/trust.
- A global trust/autonomy level can be granted/reset through weak control paths.

### 3.6 Financial and public/action domains

Primary audits:

- `kai-pm/CODE_AUDIT_BATCH_FINANCIAL_AUTONOMY_MARKET_INTELLIGENCE.md`
- Financial Awareness/Broker/Business Safety Advisor batches.
- Notify, TTS, email, browser and Executor batches.

Confirmed conditions include:

- Auto-trading fails open when governance is unavailable.
- Fixed self-asserted conviction is supplied to governance.
- Correlated indicators are counted as independent voters.
- Vote share is labelled confidence.
- Strategy failures become HOLD and hide degraded evidence.
- One heuristic can produce conviction 10/10.
- One unauthenticated quote source directly changes financial state.
- Failed market/news sources become neutral evidence.
- Untrusted search abstracts become financial/macro evidence.
- One SELL signal can close every matching long position.
- Multi-position mutation is non-transactional.
- Public messaging, notifications and browser actions have parallel evidence and approval weaknesses.

---

# 4. Authoritative registry architecture

## 4.1 Registry scope

Create one signed registry for:

- services and workloads;
- API capability/operation types;
- tools and final effect boundaries;
- model artefacts and providers;
- tokenisers/context limits;
- embedding/reranking models;
- specialist roles;
- benchmark suites;
- verifier policies;
- evidence source classes;
- autonomy-eligible operation classes.

Suggested artefacts:

- `registry/kai_capabilities_v1.yaml`
- `registry/kai_models_v1.yaml`
- signed canonical registry manifest.

## 4.2 Registry entry requirements

A model/backend entry contains:

```text
model_id
provider
provider_revision_or_digest
runtime_image_digest
tokenizer_digest
quantisation/build
context_limit
supported_input/output types
approved task classes
restricted/prohibited task classes
tool-use support
privacy/data residency class
benchmark_suite_revision
minimum qualification state
endpoint workload identity
readiness evidence TTL
cost/resource limits
registry revision
signature
```

A tool/capability entry contains:

```text
operation_type
schema_revision
final audience
side-effect class
required evidence classes
required verifier policy
human approval rule
consequence budget
reversibility/postcondition
allowed model/task classes
release state
```

## 4.3 Registry controls

- Immutable revision activated transactionally.
- Strong operator/governance approval.
- No runtime mutation from benchmark calls or success/failure observations.
- Dynamic health is separate from immutable capability metadata.
- Unknown model/tool/task types are unavailable, not default chat/Ollama/noop.
- Registry drift checks across Agentic, Tool Gate, Executor, Dashboard, Fusion, Verifier and deployment.

---

# 5. Model identity, readiness and benchmark evidence

## 5.1 Model attestation

For local models:

- hash model weights/files;
- hash tokenizer/config/template;
- verify runtime image and library versions;
- record quantisation and hardware/runtime settings;
- perform startup self-test;
- sign an attestation linked to workload identity.

For remote providers:

- use approved exact provider model revision where available;
- authenticate provider endpoint;
- record provider response model ID/revision;
- maintain contract tests for schema/tool behaviour;
- classify provider data handling and residency;
- treat unverified model aliases as mutable/unqualified.

## 5.2 Readiness evidence

Readiness is a fresh signed observation containing:

- exact model identity;
- endpoint/workload identity;
- loaded/available state;
- context/tokeniser consistency;
- current capacity/queue limits;
- supported operation classes;
- last successful qualification canary;
- timestamp and expiry.

Static profile `available=True` is prohibited.

## 5.3 Benchmark design

A benchmark record includes:

```text
benchmark_run_id
model_attestation_id
suite_revision
task/domain
input dataset digest
expected-result/evaluator revision
execution configuration
raw output digest
score metric
confidence interval/sample count
failure rate
latency/resource use
operator/automation identity
signed result
```

Required properties:

- Reproducible held-out datasets.
- No caller-injected arbitrary score.
- Task/domain-specific metrics.
- Separate correctness, calibration, safety, refusal, robustness and cost.
- Explicit live versus simulated/stub profile.
- Versioned evaluator with independent validation.
- Durable commit required before availability/primary recommendations change.

## 5.4 Primary/failover selection

Selection requires:

- exact approved registry entry;
- fresh readiness;
- qualified task class;
- privacy/data policy compatibility;
- context/resource fit;
- capacity and cost budget;
- deterministic policy revision.

An empty candidate set returns `NO_QUALIFIED_MODEL`. It never expands to all models or silently selects a fallback.

---

# 6. Evidence and claim-verification architecture

## 6.1 Claim object

Verifier accepts or derives a set of immutable claim objects:

```text
claim_id
statement
subject/predicate/object or typed proposition
claim_type
materiality/consequence class
units/currency/time scope
source span
context/qualification
required evidence policy
operation/request digest
```

Claims include ordinary facts, numbers, dates, identities, coordinates, configuration, financial, legal, medical, safety and action preconditions. Unknown consequential claims are material by default.

## 6.2 Evidence object

Evidence is a P2 immutable evidence reference containing:

```text
evidence_id
source identity
source class
content digest
quoted proposition/span
source event time
retrieval time
integrity/attestation
principal/purpose scope
trust state
supersession state
independence group
transformation lineage
```

The caller cannot submit rank/relevance/importance as verification authority.

## 6.3 Claim-evidence relation

Each relation is explicitly classified:

- `ENTAILS`
- `CONTRADICTS`
- `PARTIALLY_SUPPORTS`
- `CONTEXT_ONLY`
- `OUTDATED`
- `UNRELATED`
- `UNREADABLE`
- `UNTRUSTED`

For material numeric/configuration claims, use deterministic comparison where possible:

- exact values and tolerances;
- units/currency;
- effective date/time;
- entity identity;
- configuration revision;
- direction/negation/modality.

## 6.4 Independence and correlation

Evidence sources receive an `independence_group` based on causal origin, not URL/model label alone.

Examples:

- three indicators from one price series are one underlying data group;
- multiple articles copying one wire report are one origin group;
- multiple model responses using the same evidence/prompt are correlated;
- duplicated memories/summary/graph derivatives of one source are one source lineage.

Consequential policy specifies minimum independent source groups.

## 6.5 Verifier outcome

Use strict outcomes:

- `VERIFIED`
- `VERIFIED_WITH_LIMITATIONS`
- `CONTRADICTED`
- `INSUFFICIENT_EVIDENCE`
- `EVIDENCE_UNAVAILABLE`
- `POLICY_UNAVAILABLE`
- `INVALID_REQUEST`
- `NOT_APPLICABLE`

There is no generic positive score from context length, hedging or plan formatting.

Each outcome includes:

- verified/contradicted claim IDs;
- exact evidence relations;
- unresolved claims;
- evidence freshness/independence state;
- policy and evaluator revisions;
- complete request/evidence digest;
- signed audit-linked decision.

## 6.6 Read-only evidence evaluation

Verification uses immutable snapshots and cannot mutate access counts, memory stability, trust scores or retrieval rank. Any ranking/search analytics are separate non-authoritative observations.

---

# 7. Fusion and multi-model reasoning

## 7.1 Specialist identity

A specialist is a unique qualified model attestation plus role/prompt/evidence policy revision. Repeating a name or calling the same backend twice does not create another independent specialist.

## 7.2 Minimum viable fusion

Fusion requires a server-controlled policy specifying:

- minimum distinct qualified specialists;
- minimum independent model/provider/evidence groups;
- permitted specialist roles;
- complete output schema;
- verifier requirement;
- contradiction/escalation rule;
- resource budget.

Empty or single-specialist results cannot be labelled consensus.

## 7.3 Structured specialist output

Each specialist returns:

```text
specialist_attestation_id
claims
reasoning summary
assumptions
evidence references
uncertainties
proposed answer/action
refusal/degraded state
output digest
```

Raw caller context is never used as the specialist system prompt. Untrusted input is placed in the P2 data channel.

## 7.4 Agreement analysis

Fusion compares claim/proposition objects, not response wording alone.

It reports:

- supported common propositions;
- contradictory propositions;
- unsupported additions;
- differing assumptions;
- model/evidence dependence;
- unresolved uncertainty.

Text similarity may be a diagnostic only and must be labelled as such.

## 7.5 Merge and verifier enforcement

- Do not select the longest answer.
- Produce one bounded structured synthesis from verified propositions.
- Verify the complete returned content or mark exact verified spans.
- Required Verifier `CONTRADICTED`, `INSUFFICIENT`, unavailable or policy error blocks consequential output/action.
- Low agreement returns a disagreement/escalation state, not a normal merged answer with misleading metadata.

---

# 8. Uncertainty and calibration

## 8.1 Replace “conviction”

Retire one scalar used simultaneously for factual confidence, plan quality, moral alignment, vote share, writing style and execution authority.

Use separate typed measures:

- evidence completeness;
- evidence quality;
- source independence;
- model calibration probability for a defined task;
- plan feasibility;
- execution risk/consequence;
- policy/approval state;
- uncertainty interval;
- abstention reason.

## 8.2 Calibration records

Calibration requires linked predictions and independently verified outcomes:

```text
prediction_id
model/task attestation
predicted probability/range
claim/outcome definition
prediction time
resolution source
verified outcome
resolution time
scoring rule
calibration cohort
```

Self-generated episode labels, model critique, blocked Gate responses, simulated success and operator acknowledgement do not qualify as outcomes.

## 8.3 Abstention

The system must abstain or escalate when:

- required evidence is unavailable/stale/correlated;
- model/backend unqualified;
- uncertainty exceeds policy;
- claim consequence exceeds qualified domain;
- conflicting independent sources remain unresolved;
- evaluator/policy/audit unavailable.

Abstention is a successful safety outcome, not a model failure to be hidden as HOLD/neutral.

---

# 9. Trust and autonomy rebuild

## 9.1 Trust object

Trust is not one global user-to-Kai score. It is a scoped competence record:

```text
principal_id
capability/operation class
domain/resource
model/system revision
evidence cohort
verified success/failure outcomes
harm/override links
calibration metrics
sample size
validity window
maximum consequence budget
state
```

## 9.2 Evidence eligibility

Eligible trust evidence:

- completed Phase 1-capability operation;
- P3 durable operation/audit linkage;
- independently observed and verified outcome;
- exact model/tool/system revision;
- defined success/harm criteria;
- no unresolved incident/override.

Ineligible evidence:

- plan/model claims;
- acknowledgement alone;
- self-reflection;
- simulation/paper state presented as real outcome;
- duplicated/correlated evidence;
- anonymous feedback/value records;
- success-shaped fallback/stub/degraded response.

## 9.3 Override and harm linkage

Every operator correction/override references the exact operation/outcome and classifies:

- prevented action;
- corrected factual output;
- reversed side effect;
- safety/privacy harm;
- false positive/appropriate abstention.

Global override counts cannot penalise or certify unrelated operations.

## 9.4 Autonomy levels

Suggested staged levels:

### A0 — Advisory only

No external mutation. Evidence-backed response with explicit limitations.

### A1 — Reversible local preparation

Creates drafts/plans in principal-scoped workspace. No send/execute/financial/public action.

### A2 — Low-consequence bounded action

Exact allowlisted action, reversible postcondition, low budget, no sensitive external disclosure.

### A3 — Conditional consequential action

Requires fresh evidence, qualified domain, human approval or policy-defined supervised mode, full monitoring and rollback/containment.

### A4 — Narrow unattended operation

Only for a specifically qualified repeated operation class with strong historical outcomes, small budgets, rapid detection/stop and no irreversible/public/financial/security-admin authority unless separately mandated.

There is no general unrestricted A5/full autonomy release under this plan.

## 9.5 Revocation and regression

Autonomy scope automatically suspends on:

- model/registry/policy revision change;
- evidence/verifier/audit degradation;
- incident or linked override/harm;
- calibration drift;
- insufficient recent sample;
- failed postcondition/restore/recovery test;
- expired qualification window.

Revocation has precedence over cached trust and process-local state.

---

# 10. High-consequence domain qualification

## 10.1 Financial

Before any real financial action:

- approved instrument/venue/account registry;
- independently authenticated market data sources;
- source event time and freshness;
- outlier/cross-venue validation;
- strategy configuration immutable and backtested out of sample;
- correlation-aware signal model;
- risk/position/quantity limits;
- atomic portfolio operations;
- transaction-cost/slippage/error model;
- human approval policy;
- complete reconciliation and independent account outcome.

Paper trading may be used only as simulation labelled `SIMULATED`; it cannot directly increase real-capital trust.

## 10.2 Public communications

Email, notification, TTS, Telegram and published content require:

- exact recipient/channel/audience;
- complete content digest;
- evidence/claim verification where factual;
- privacy/secret scan by schema;
- rate/reputation budget;
- human preview/approval for consequential/public statements;
- delivery and retraction/correction outcome.

## 10.3 Destructive/security administration

Recovery, deletion, configuration, credential, policy, identity and infrastructure actions require:

- dedicated operation type;
- exact resource/revision;
- independent precondition evidence;
- step-up operator approval;
- maintenance/fencing;
- reversible/pre-snapshot plan where possible;
- postcondition and incident/audit linkage.

No model confidence alone can authorise them.

## 10.4 Self-modification

SOUL, AGENTS, skills, prompts, registries, policies, code, model profiles and trust rules are deployment/governance artefacts.

Self-proposed changes may be generated as signed review packages but cannot activate themselves. Required path:

- source evidence and rationale;
- bounded diff;
- tests/security analysis;
- operator review;
- immutable commit/build;
- staged deployment;
- rollback and monitoring;
- independent qualification.

---

# 11. Ordered implementation PRs

## P4-PR-01 — Capability and autonomy contract freeze

Deliverables:

- Signed registry schemas.
- Model attestation/readiness schema.
- Benchmark record schema.
- Claim/evidence/relation/verdict schemas.
- Fusion specialist/synthesis schemas.
- Calibration record schema.
- Scoped trust/autonomy schema.
- Domain qualification templates.

Acceptance:

- Security, model, data, product and domain owners approve one model.
- Unknown/stub/degraded states cannot map to qualified states.

---

## P4-PR-02 — Authoritative signed registry

Required changes:

- Consolidate duplicated service/model/tool registries.
- Canonical signed revision and transactional activation.
- Default unknown to unavailable.
- Drift checks against deployment and code.

Acceptance:

- Agentic, Tool Gate, Executor, Fusion, Verifier, Dashboard and Supervisor resolve the same revision.
- Runtime model registration cannot alter protected capability state.

---

## P4-PR-03 — Model/backend attestation

Required changes:

- Exact local artefact/runtime digests.
- Remote provider identity/contract validation.
- Fresh signed readiness.
- Capacity, privacy and task-class metadata.

Acceptance:

- Static name or profile cannot make a model available.
- Wrong/mutated artefact/tokenizer/runtime rejected.
- No candidate returns `NO_QUALIFIED_MODEL`.

---

## P4-PR-04 — Reproducible benchmark authority

Required changes:

- Approved suites/dataset/evaluator revisions.
- Signed runs and raw-output digests.
- Task-specific correctness, safety, calibration and cost metrics.
- Durable transactional profile qualification.

Acceptance:

- Caller-injected score rejected.
- Static heuristic/stub cannot pass benchmark.
- Persistence failure prevents qualification mutation.

---

## P4-PR-05 — Model selection and failover rebuild

Required changes:

- Use registry + fresh readiness + qualification.
- Exact token budgeting.
- Deterministic cost/privacy/capacity policy.
- No implicit Ollama/chat fallback.
- Failure state scoped by endpoint/task/failure class.

Acceptance:

- Empty/invalid list produces no model.
- One candidate receives full validation.
- Failover never leaves the approved set.

---

## P4-PR-06 — Remove heuristic execution conviction

Required changes:

- Delete style/punctuation/keyword confidence authority.
- Separate evidence, feasibility, risk, policy and calibration measures.
- Unknown Gate/Verifier/audit/security state fails closed.
- Remove substring conviction overrides.

Acceptance:

- Padding, hedging, formatting and moral language cannot increase execution authority.
- Security failure cannot record a passed challenge.

---

## P4-PR-07 — Immutable claim/evidence service

Dependencies: P2 provenance and P3 audit.

Required changes:

- Typed material claims.
- Authoritative evidence resolution.
- Read-only evidence snapshots.
- Source integrity/freshness/independence metadata.

Acceptance:

- Caller evidence ranking fields ignored/rejected.
- Verification does not mutate memory ranking.
- Financial/configuration/unknown consequential claims are material.

---

## P4-PR-08 — Proposition-level Verifier rebuild

Required changes:

- Entailment/contradiction/context/outdated classification.
- Deterministic material-value comparison.
- Claim-specific evidence and independent-source policy.
- Strict verdict enum and complete digest/audit.
- No context/hedge/plan-format bonuses.

Acceptance:

- “Not approved” contradicts “approved”.
- Duplicated evidence counts once.
- Unsupported claim cannot PASS.
- Evidence/policy outage produces unavailable/insufficient and blocks consequential use.

---

## P4-PR-09 — Verifier enforcement integration

Required changes:

- Agentic, Fusion, finance and public-action workflows consume strict verdicts.
- Required negative/unavailable verdict blocks capability issue.
- Verify the complete exact result/action, not a prefix.

Acceptance:

- Verification cannot remain advisory metadata for protected workflows.
- Modified post-verification output invalidates the decision digest.

---

## P4-PR-10 — Specialist and Fusion registry

Required changes:

- Server-approved distinct specialists.
- Qualified model/role/prompt/evidence revisions.
- Minimum independent backend/evidence groups.
- Bounded concurrency/resource budget.

Acceptance:

- Empty, one, duplicate or stub specialist cannot produce consensus.
- Caller cannot lower consensus threshold or supply system prompt.

---

## P4-PR-11 — Structured fusion and contradiction handling

Required changes:

- Proposition comparison.
- Assumption/dependence/conflict reporting.
- Verified synthesis only from supported propositions.
- Explicit disagreement/escalation state.

Acceptance:

- Longest response is not automatically selected.
- Similar wording alone does not establish correctness.
- Unresolved contradiction cannot be presented as consensus.

---

## P4-PR-12 — Prediction/outcome separation

Required changes:

- Distinct prediction, proposal, execution, observation and verified-outcome records.
- No self-certification.
- Resolution sources and outcome links.

Acceptance:

- Blocked/no-action episode cannot be successful outcome.
- Operator acknowledgement alone cannot certify success.
- Reflection/model critique cannot become external outcome.

---

## P4-PR-13 — Calibration service

Required changes:

- Task-specific probability/range records.
- Independently verified resolutions.
- Proper scoring and cohort/sample reporting.
- Drift and expiry.

Acceptance:

- Confidence exposed only where calibration evidence exists.
- Insufficient sample returns uncalibrated/abstain state.

---

## P4-PR-14 — Trust Ledger/scoring replacement

Required changes:

- Scoped competence records.
- Exact operation/outcome links.
- No favourable defaults for zero/missing data.
- Override/harm attribution.
- Snapshot-consistent scoring with uncertainty/sample size.

Acceptance:

- Anonymous values/feedback cannot increase trust.
- Global acknowledgement/override counts cannot influence unrelated capabilities.
- Trust revision names exact system/model/domain.

---

## P4-PR-15 — Staged autonomy authority

Required changes:

- A0–A4 scoped levels.
- Purpose/resource/consequence/time budgets.
- Automatic suspension triggers.
- Revocation precedence and requalification expiry.
- Operator UI for scope/limits/history.

Acceptance:

- No global GUARDIAN/full-autonomy grant.
- Capability outside qualified domain fails.
- Model/policy/evidence revision suspends prior qualification.

---

## P4-PR-16 — Financial-domain qualification

Required changes:

- Authenticated multi-source quotes.
- Correlation-aware strategy evidence.
- Reproducible out-of-sample tests.
- Atomic position/risk operations.
- Simulation versus real-capital separation.
- Human/risk approval and reconciliation.

Acceptance:

- One source/heuristic cannot cause action.
- Vote share is not confidence.
- One SELL cannot close unrelated positions.
- Governance/data/Verifier outage blocks mutation.

---

## P4-PR-17 — Public communication qualification

Required changes:

- Exact recipient/channel/content operation.
- Factual verification and privacy classification.
- Approval/rate/reputation controls.
- Delivery/correction/retraction outcome.

Acceptance:

- Stubs, unverified facts and sensitive data cannot be sent.
- Delivery is not success until confirmed under channel policy.

---

## P4-PR-18 — Destructive/admin/recovery qualification

Required changes:

- Exact action registry and evidence requirements.
- Mandatory step-up operator approval.
- Fencing/snapshot/postcondition.
- No model-only authority.

Acceptance:

- Confidence/trust score cannot administer policy, credentials or recovery.
- Alternate direct route fails.

---

## P4-PR-19 — Self-modification review pipeline

Required changes:

- Proposal-only model output.
- Signed diff/evidence/test package.
- Operator review and normal Git/build/deployment controls.
- Staged rollout/rollback and requalification.

Acceptance:

- Model cannot write/activate SOUL, AGENTS, registry, policy or code directly.
- Self-generated test claims cannot approve deployment.

---

## P4-PR-20 — Stub/fallback truthfulness migration

Required changes:

- Inventory every stub/fake/neutral fallback.
- Explicit non-operational response and readiness state.
- Block from consensus, benchmarks, trust and release gates.
- Remove production-shaped success contracts.

Acceptance:

- CI/runtime evidence identifies live versus simulated coverage.
- A stack operating only on stubs cannot report GO/ready/consensus.

---

## P4-PR-21 — Integrated capability requalification gate

Required evidence per capability:

- registry and model/tool attestations;
- P0–P3 control gates passed;
- benchmark/calibration reports;
- verifier/fusion evidence;
- domain-specific risk tests;
- complete operation/outcome/audit chain;
- attack-chain regression tests;
- rollback/suspension drill;
- explicit release owner approval and expiry date.

No capability inherits qualification from another capability or earlier revision.

---

# 12. Phase 4 adversarial closure tests

## Test P4-A — Model identity substitution

Replace model weights, tokenizer, provider alias, runtime image or endpoint identity.

**Pass:** attestation/readiness invalid; model unavailable until requalified.

## Test P4-B — Empty/unknown model set

Supply empty, unknown and one invalid candidate.

**Pass:** `NO_QUALIFIED_MODEL`; no fallback expansion or Ollama selection.

## Test P4-C — Benchmark injection

Submit arbitrary high scores, static heuristic/stub output and tamper dataset/evaluator.

**Pass:** rejected; qualification unchanged.

## Test P4-D — Evidence fabrication

Supply duplicate high-rank records, same-source derivatives, contradictory text and caller scores.

**Pass:** authoritative resolution, deduplication, contradiction and insufficient-evidence verdict.

## Test P4-E — Material claim bypass

Use currencies, dates, configuration, negation, quotation, hypotheticals and ordinary factual claims designed to evade regexes.

**Pass:** typed claim coverage and consequence-default materiality.

## Test P4-F — Fusion manufacture

Use empty, one, duplicate and stub specialists; zero threshold; similar boilerplate; contradictory outputs.

**Pass:** no consensus; explicit unqualified/disagreement state.

## Test P4-G — Verifier veto

Return contradicted, insufficient, unavailable and policy-error verdicts after Fusion/Agentic proposes action.

**Pass:** protected capability not issued; no advisory bypass.

## Test P4-H — Style-based confidence attack

Pad text, add hedging, punctuation, keywords, plan formatting and moral alignment language.

**Pass:** no increase in calibrated evidence or execution authority.

## Test P4-I — Self-certification loop

Feed model prediction/reflection/simulated success/acknowledgement back as evidence.

**Pass:** ineligible for outcome/trust/calibration.

## Test P4-J — Correlated evidence

Use multiple indicators from one price series, copied articles, duplicated memories and multiple models sharing one evidence source.

**Pass:** one independence group; policy cannot count them as independent corroboration.

## Test P4-K — Financial weak-signal attack

Manipulate one quote/source/indicator and cause strategy failures/HOLD conversions.

**Pass:** action blocked; degraded sources explicit; no 10/10 or vote-share confidence.

## Test P4-L — Global trust escalation

Submit anonymous feedback/values/loyalty/acknowledgements and request broad autonomy.

**Pass:** no trust change; only scoped outcome-based competence records accepted.

## Test P4-M — Revision regression

Change model, prompt, registry, policy, evidence source or tool implementation after qualification.

**Pass:** autonomy suspended and requalification required.

## Test P4-N — Direct high-consequence route

Call financial/public/destructive/self-modifying final service outside the qualified path.

**Pass:** P1 capability and domain qualification required; direct route rejected.

## Test P4-O — Stub-only deployment

Disable all live models/verifier/evidence and enable stubs/fakes.

**Pass:** readiness non-operational, release NO_GO, no consensus/trust/benchmark credit.

## Test P4-P — Complete compromise-chain regression

Replay final audit chains: Dashboard XSS/confused deputy, anonymous Agentic escalation, memory poisoning, forged evidence, Fusion consensus, Executor bypass, health recovery reset and weak market mutation.

**Pass:** every chain breaks at multiple independently enforced boundaries and produces complete audit/incident evidence.

---

# 13. Capability release evidence bundle

Each released capability must have one signed bundle containing:

```text
capability_id and operation class
registry revision
model/tool/service attestations
P0-P3 gate evidence
benchmark suite and result revisions
calibration state/sample size
Verifier/Fusion policy revisions
data/evidence source policy
autonomy scope and consequence budget
human approval rule
attack-chain tests
incident/override history
release owner and date
qualification expiry/review date
rollback/suspension procedure
```

Bundle states:

- `UNQUALIFIED`
- `TEST_ONLY`
- `ADVISORY_ONLY`
- `SUPERVISED_RELEASE`
- `NARROW_AUTONOMOUS_RELEASE`
- `SUSPENDED`
- `EXPIRED`
- `REVOKED`

No generic system-wide “GO” overrides a capability-specific state.

---

# 14. Phase 4 exit criteria

Phase 4 planning/implementation is complete only when all are true:

- One signed authoritative service/model/tool/capability registry is enforced.
- Every used model/backend has exact attestation and fresh readiness.
- Benchmarks are reproducible, task-specific and signed.
- No stub/fake/fallback produces operational success, consensus or trust.
- Model selection returns no model when none is qualified.
- Heuristic/style-based conviction is removed from execution authority.
- Verifier resolves immutable evidence and evaluates propositions, contradictions and independence.
- Fusion requires distinct qualified specialists and reports unresolved conflicts.
- Required Verifier outcomes enforce protected workflows.
- Predictions/actions/observations/outcomes are separate and auditable.
- Calibration uses independently verified outcomes.
- Trust is scoped by domain/capability/revision and includes uncertainty/sample size.
- Global broad trust/autonomy grants are removed.
- Financial, public, destructive/admin and self-modifying domains have separate qualification gates.
- Revision changes automatically suspend prior qualification.
- Complete original attack chains and new domain adversarial tests pass.
- Every released capability has a signed expiring evidence bundle and rollback/suspension process.

Only capabilities with an explicit current release bundle may be enabled. Everything else remains disabled or advisory/test-only.

---

# 15. Immediate next implementation queue

After P0–P3 are implemented and verified:

1. P4-PR-01/02 — freeze contracts and consolidate the signed registry.
2. P4-PR-03/04/05 — model attestation, benchmark authority and selection.
3. P4-PR-06 — remove heuristic conviction authority.
4. P4-PR-07/08/09 — evidence service, Verifier rebuild and enforcement.
5. P4-PR-10/11 — specialist registry and structured Fusion.
6. P4-PR-12/13/14 — outcome separation, calibration and scoped trust.
7. P4-PR-15 — staged autonomy authority.
8. P4-PR-16/17/18/19 — domain qualification.
9. P4-PR-20 — stub/fallback truthfulness.
10. P4-PR-21 — capability-specific integrated release gate.

Do not enable consequential autonomy while qualification depends on current conviction, Verifier, Fusion, Trust Ledger, model-profile or self-generated outcome mechanisms.

---

## Final Phase 4 planning judgement

Kai’s current model and autonomy layer does not fail merely because individual heuristics are imperfect. It lacks trustworthy semantic boundaries between evidence, agreement, confidence, execution and outcome. Caller-controlled evidence can verify itself, duplicate/correlated signals masquerade as independence, stubs and missing dependencies can look healthy, and self-generated records recursively become proof of competence.

The minimum defensible correction is an attested capability registry, reproducible model qualification, proposition-level evidence verification, correlation-aware fusion, calibrated uncertainty and scoped autonomy based only on independently verified outcomes.

**Current status remains NO_GO. This document implements no runtime remediation and closes no findings.**
