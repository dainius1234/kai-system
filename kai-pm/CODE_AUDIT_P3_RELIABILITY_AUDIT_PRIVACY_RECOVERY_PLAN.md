# Kai System — Phase 3 Reliability, Audit, Privacy and Recovery Plan

Repository: `dainius1234/kai-system`  
Authoritative audit baseline: **4,580 findings — 252 Critical, 2,440 High, 1,885 Medium, 3 Low**  
Parent backlog: `kai-pm/CODE_AUDIT_REMEDIATION_BACKLOG.md`  
Dependencies:

- `kai-pm/CODE_AUDIT_P0_CONTAINMENT_PLAN.md`
- `kai-pm/CODE_AUDIT_P1_SECURITY_FOUNDATION_PLAN.md`
- `kai-pm/CODE_AUDIT_P2_ISOLATION_AND_INTEGRITY_PLAN.md`

Status: **IMPLEMENTATION DESIGN ONLY — NO RUNTIME REMEDIATION PERFORMED**

---

## 1. Objective

Make Kai’s distributed operation, health, audit, privacy, recovery and backup behaviour dependable enough to support controlled production qualification.

Phase 3 establishes:

1. One typed success/failure and health vocabulary across all services.
2. Durable operation state machines and idempotent distributed mutation.
3. Transactional shared state for security and reliability controls.
4. Leader election, leases and single-writer ownership.
5. A signed append-only audit authority with externally anchored checkpoints.
6. Data classification, purpose, consent, retention and encryption enforcement.
7. Recovery separated from health observation and governed as a consequential action.
8. Coherent, signed, verified backups and isolated restore qualification.
9. Incident-state, forensic and legal-hold controls.
10. Production-equivalent failure injection and operational release evidence.

Phase 3 does not requalify model judgement, evidence scoring, trust or autonomy. Those remain Phase 4 work. A reliable platform can still make unsafe decisions if the cognitive and evidence layers are not rebuilt.

---

## 2. Governing reliability and governance invariants

### INV-P3-01 — Failure cannot look like success

Every API and event result uses a typed terminal or non-terminal state. Error-shaped HTTP 200 responses, empty fallback objects and success-shaped stubs are prohibited for protected workflows.

Required operation states include:

- `ACCEPTED`
- `PENDING`
- `RUNNING`
- `SUCCEEDED`
- `SUCCEEDED_WITH_LIMITATIONS`
- `DENIED`
- `INVALID`
- `UNAUTHENTICATED`
- `UNAUTHORISED`
- `UNAVAILABLE`
- `DEGRADED`
- `TIMED_OUT_UNKNOWN_OUTCOME`
- `FAILED_BEFORE_EFFECT`
- `FAILED_AFTER_PARTIAL_EFFECT`
- `COMPENSATION_REQUIRED`
- `COMPENSATED`
- `CANCELLED`

A caller must never infer success from transport reachability alone.

### INV-P3-02 — Health is an observation, not recovery authority

Liveness, readiness, dependency status and business capability are separate observations. No health response directly authorises reset, restart, token reload, breaker closure, data restore or other state mutation.

### INV-P3-03 — Every distributed mutation has one durable operation identity

Retries, failover, restarts and worker changes operate on one Phase 1 operation ID and digest. A client timeout cannot cause a second mutation until the authoritative operation state is reconciled.

### INV-P3-04 — Security and reliability state is shared and transactional

Breaker state, capability consumption, idempotency, recovery leases, pending approvals, operation jobs, notification delivery and audit sequence are not process-local dictionaries or mutable files.

### INV-P3-05 — One owner performs scheduled or singleton work

Schedulers, backup coordinators, ledger sequencers, recovery controllers, retention workers and index maintenance use a lease or leader-election authority with fencing tokens. A stale leader cannot continue committing work.

### INV-P3-06 — Audit evidence is complete, ordered and externally verifiable

A protected operation cannot claim success if its required audit event cannot be durably committed. Audit ordering is assigned by one transactional sequencer, signed with managed keys and periodically anchored outside the application trust domain.

### INV-P3-07 — Personal data has enforceable lifecycle policy

Every data object has principal, tenant, purpose, classification, consent/legal basis, retention class, encryption policy and deletion state. Logging, audit and backups do not become uncontrolled shadow databases.

### INV-P3-08 — Recovery is an authorised, diagnosable operation

Recovery requires an incident identity, authenticated authority, service-specific action, preconditions, expected postconditions, fencing, idempotency and a Phase 1 capability. Recovery cannot silently erase containment or forensic evidence.

### INV-P3-09 — A backup is not valid until independently verified

Creation success means the artefact, signed manifest and checksums were durably committed. Qualification requires an isolated restore, integrity verification and application-level postcondition tests against the exact artefact set.

### INV-P3-10 — Production release evidence is immutable and revision-bound

Tests, SBOMs, image digests, configuration manifests, migration state, restore results and adversarial evidence must identify one immutable release revision. A green CI status without linked artefacts is not release evidence.

---

## 3. Confirmed source conditions driving Phase 3

### 3.1 Shared resilience and health primitives

Primary source:

- `common/resilience.py`
- `common/runtime.py`

Primary audits:

- `kai-pm/CODE_AUDIT_BATCH_COMMON_RESILIENCE.md`
- `kai-pm/CODE_AUDIT_BATCH_SHARED_RUNTIME_CONTROLS.md`

Confirmed conditions include:

- POST mutations are retried without operation identity or committed-outcome reconciliation.
- Methods other than GET are converted to POST in shared retry logic.
- Fallback objects are indistinguishable from successful business responses.
- Breaker identity can collide across ports/services or be caller selected.
- Health with zero checks reports healthy.
- Health checks have no complete deadline and can overwrite duplicate names.
- Required tasks that never started are invisible to watchdog state.
- Healing advances by invocation count, not verified action/outcome.
- `auto_recovery` and caller-supplied fixes are recorded as successful knowledge without execution proof.
- Error budgets ignore many authentication, policy and gateway failures.
- Circuit breakers are process-local, race-prone and have unsafe half-open semantics.
- Audit verification can return true when no backend exists.
- Concurrent audit writers can fork the hash chain.
- Audit append and tail-hash update are non-atomic and unkeyed.

### 3.2 Supervisor and recovery control plane

Primary source:

- `supervisor/app.py`

Primary audit:

- `kai-pm/CODE_AUDIT_BATCH_LIVE_SUPERVISOR.md`

Confirmed conditions include:

- Host-published unauthenticated sweep and recovery controls.
- Repeated sweeps can accelerate breaker opening and fleet recovery.
- Shallow self-reported health drives consequential `/recover` calls.
- Any HTTP 200 recovery response is treated as healed.
- Recovery is not followed by a verified readiness/postcondition check.
- Recovery requests carry no incident, operation, expected revision or authorised action.
- Breakers, fleet state, attempts and history are process-local.
- Multiple workers can recover the same service concurrently.
- The mandatory background loop is untracked and not drained on shutdown.
- Fleet health can be green before any evidence exists.
- Unknown health states and missing status fields become healthy.

### 3.3 Tool Gate and trust/audit ledgers

Primary sources:

- `tool-gate/app.py`
- `trust-ledger/app.py`
- `trust-ledger/ledger.py`
- `trust-ledger/score.py`

Primary audits:

- `kai-pm/CODE_AUDIT_BATCH_TOOL_GATE_EXTENSION.md`
- `kai-pm/CODE_AUDIT_BATCH_TRUST_LEDGER_EXTENSION.md`
- `kai-pm/CODE_AUDIT_BATCH_SHARED_RUNTIME_CONTROLS.md`

Confirmed conditions include:

- Gate decisions can return successfully when ledger persistence fails.
- Concurrent file appends can fork or corrupt claimed linear chains.
- Corrupt entries are skipped and operation continues.
- Complete credentials, signatures and parameters may enter ledger records.
- File-backed ledgers lack strong ownership, permissions, rotation and external anchoring.
- Workers maintain different in-memory ledger histories.
- Merkle publication records success when disabled or failed.
- Published roots can be stale immediately after creation.
- Checkpoint persistence is unlocked, non-atomic and unbounded.
- Signing keys lack key ID, activation, expiry and rotation evidence.
- Trust scoring uses inconsistent snapshots and success proxies rather than linked outcomes.

### 3.4 Backup and restore

Primary source:

- Backup Service application, Dockerfile and deployment configuration.

Primary audits:

- `kai-pm/CODE_AUDIT_BATCH_BACKUP_SERVICE.md`
- `kai-pm/CODE_AUDIT_BATCH_BACKUP_SERVICE_EXTENSION.md`
- CI/off-site backup batches.

Confirmed conditions include:

- Restore accepts unverified plaintext files and may execute psql client meta-commands.
- Required PostgreSQL/Redis tools and durable storage are absent in the deployed image/topology.
- Redis backup can report success without producing an artefact.
- Known `localdev` password fallback and ignored mounted secret.
- Partial dumps are written to trusted-looking final names.
- Listing/restore can follow symlinks.
- Restore lacks manifest binding, strict SQL failure semantics, transaction/fencing and pre-restore rollback point.
- Full backup components are captured at different times with no coherent snapshot.
- Manifests are ephemeral or omit source identity, schema, checksums and compatibility.
- No isolated restore drill proves restorability.

### 3.5 Process-local and multi-writer state

Cross-service audits repeatedly confirm process-local or file-backed state for:

- breakers and error budgets;
- idempotency and nonces;
- pending approvals and modes;
- queues and schedulers;
- notification/delivery state;
- memory/vector/graph mappings;
- backup jobs and recovery attempts;
- audit and trust ledgers;
- sessions, models and personal state.

Phase 3 converts the security foundations from P1/P2 into a supported multi-worker transactional architecture.

### 3.6 Privacy and lifecycle

The final architecture audit established that Kai has no enforceable classification, privacy, retention or derivative-deletion model. Sensitive content can enter:

- operational logs and stdout;
- audit/Trust Ledger records;
- memory, graph, vectors and summaries;
- browser state and downloads;
- parser outputs and temporary files;
- notification channels;
- backup archives and checkpoints;
- localStorage and process-local caches.

P2 establishes principal/purpose partitioning and lineage. P3 enforces lifecycle, encryption, retention and lawful/consented use across those stores.

---

# 4. Standard service-state and failure contract

## 4.1 Endpoint classes

Every endpoint must declare one class:

- `OBSERVATION_READ`
- `SENSITIVE_READ`
- `IDEMPOTENT_MUTATION`
- `NON_IDEMPOTENT_MUTATION`
- `LONG_RUNNING_JOB`
- `ADMINISTRATION`
- `RECOVERY`
- `HEALTH_LIVENESS`
- `HEALTH_READINESS`

The class controls authentication, capability, retry, idempotency, audit and timeout requirements.

## 4.2 Error envelope

Adopt one versioned error object:

```json
{
  "schema": "kai.error.v1",
  "operation_id": "...",
  "operation_digest": "...",
  "code": "DEPENDENCY_UNAVAILABLE",
  "class": "TRANSIENT|PERMANENT|POLICY|AUTH|VALIDATION|UNKNOWN_OUTCOME",
  "retryable": false,
  "safe_to_retry": false,
  "effect_state": "NO_EFFECT|UNKNOWN|PARTIAL|COMMITTED",
  "message": "safe operator-facing summary",
  "trace_id": "...",
  "occurred_at": "..."
}
```

Raw internal exceptions, URLs, credentials, filesystem paths and private content are excluded.

## 4.3 Health vocabulary

### Liveness

Answers only: can the process event loop serve a minimal response?

### Readiness

Answers: can this exact service instance accept its declared protected operations safely?

Readiness must include:

- validated configuration revision;
- authenticated dependency identity;
- required storage and audit authority;
- leader/lease state where relevant;
- migration/schema compatibility;
- no stub/fake/development authority;
- capacity and backpressure state.

### Capability status

Each advertised capability reports:

- `READY`
- `DEGRADED_READ_ONLY`
- `DISABLED_BY_POLICY`
- `NOT_INITIALISED`
- `STUB`
- `UNAVAILABLE`
- `MIGRATION_REQUIRED`

A top-level green state cannot hide failed nested checks.

## 4.4 Retry rules

- GET is not automatically safe when it causes active probes or state mutation.
- Mutations retry only under one operation identity and idempotency authority.
- `TIMED_OUT_UNKNOWN_OUTCOME` triggers reconciliation, not blind replay.
- Respect server retry guidance and operation deadlines.
- Use bounded jittered backoff.
- Retries cease on authentication, authorisation, policy and permanent validation failures.

---

# 5. Durable operation and saga model

## 5.1 Operation record

Use a transactional store for a versioned record containing:

```text
operation_id
operation_digest
principal_id
tenant_id
requesting_workload
operation_type
resource
state
attempt
idempotency_key
capability_id
policy_revision
created_at
updated_at
lease_owner
fencing_token
last_error_code
side_effect_checkpoint
compensation_state
outcome_digest
```

## 5.2 Mutation state machine

Minimum state machine:

```text
RECEIVED
  -> VALIDATED
  -> AUTHORISED
  -> EFFECT_STARTED
  -> EFFECT_COMMITTED
  -> OUTCOME_VERIFIED
  -> SUCCEEDED
```

Failure paths:

```text
VALIDATION_FAILED
DENIED
FAILED_BEFORE_EFFECT
UNKNOWN_EFFECT
PARTIAL_EFFECT
COMPENSATION_PENDING
COMPENSATED
MANUAL_INTERVENTION
```

## 5.3 Transactional outbox/inbox

Every cross-service command/event uses:

- local transaction plus outbox record;
- unique event/command ID;
- consumer inbox deduplication;
- explicit acknowledged/processed outcome;
- bounded retry and dead-letter state;
- operation-digest correlation.

HTTP fire-and-forget and best-effort audit/notification are not sufficient for consequential operations.

## 5.4 Reconciliation workers

Implement dedicated workers to resolve:

- timed-out unknown outcomes;
- stuck capability consumption;
- partial memory/vector/graph updates;
- pending notification delivery;
- incomplete backup jobs;
- interrupted recovery operations;
- stale leases and orphaned tasks.

Reconciliation is read/compare-driven and cannot invent success.

---

# 6. Shared state, leadership and fencing

## 6.1 Transactional authorities

Recommended ownership:

- Postgres: operations, idempotency, capability consumption, recovery incidents, approval state, audit sequencing metadata, jobs, delivery outcomes and retention tasks.
- Purpose-built/object storage: immutable audit segments, backup artefacts, signed manifests and legal-hold archives.
- Redis: bounded cache/ephemeral coordination only, not sole authority for irreversible security state.

## 6.2 Leader election

Required singleton roles include:

- Supervisor scheduler.
- Recovery coordinator.
- Backup coordinator.
- Audit segment/checkpoint publisher.
- Retention/deletion worker.
- Ledger archival worker.
- Memory/index maintenance scheduler.

Every lease includes:

- owner identity;
- lease expiry;
- monotonically increasing fencing token;
- operation/task generation;
- durable acquisition/renewal/release audit.

Workers must present the fencing token to the authoritative store. Expired leaders cannot commit.

## 6.3 Circuit-breaker authority

Replace process-local breakers with an operation/dependency-specific shared model:

- canonical dependency identity;
- failure class and observation source;
- rolling bounded samples;
- minimum sample floor;
- single half-open probe lease;
- recovery postcondition;
- configuration revision;
- manual containment override;
- audit-linked state transitions.

A successful unrelated request cannot close the breaker.

---

# 7. Authoritative audit architecture

## 7.1 Event schema

Every audit event contains:

```text
event_id
sequence
schema_version
event_type
principal_id
tenant_id
workload_id
delegation_id
operation_id
operation_digest
capability_id
resource
action
result_state
reason_code
policy_revision
service_revision
source_event_time
recorded_at
previous_event_hash
payload_digest
data_classification
retention_class
signing_key_id
signature
```

Secret material and unrestricted payload bodies are prohibited.

## 7.2 Sequencing and append

- One transactional sequencer allocates monotonically increasing sequence values.
- Event append and sequence advancement are atomic.
- Multiple writers submit events but cannot independently create chain predecessors.
- Segment closure calculates a Merkle root over a fixed event range.
- Segment manifest includes previous segment root, first/last sequence, event count, key ID, policy revision and object digest.
- Closed segments are immutable.

## 7.3 Signing and key lifecycle

- Asymmetric audit-signing keys held by a dedicated signer/HSM-compatible authority.
- Key IDs and validity intervals inside signed manifests.
- Rotation with explicit continuity events.
- Compromise/revocation does not permit silent historical re-signing.
- Verification supports historical key trust and revocation-at-time semantics.

## 7.4 External anchoring

Periodically publish signed segment roots to an independent store or transparency endpoint outside the normal Kai administrative trust domain.

The checkpoint process must:

- commit the segment first;
- verify durable object storage;
- publish the exact fixed root;
- confirm external receipt;
- append a separate receipt event referencing the already closed segment.

A failed publication is a failure event, never `MERKLE_PUBLISH success`.

## 7.5 Audit availability semantics

- Required audit unavailable → consequential mutation unavailable.
- Observation-only endpoints may operate in explicitly marked degraded mode where policy permits.
- Audit backpressure is bounded and surfaced in readiness.
- No `verify=true` response when zero audit backend/events exist.

## 7.6 Audit privacy

Audit records store identifiers/digests and minimised structured facts, not full prompts, tokens, emails, documents or commands by default.

Sensitive audit access requires:

- dedicated audit-reader scope;
- purpose and case/incident ID;
- field-level filtering;
- immutable access event;
- export controls and expiry.

---

# 8. Data classification, privacy and retention

## 8.1 Classification model

Minimum classes:

- `PUBLIC`
- `INTERNAL`
- `CONFIDENTIAL`
- `PERSONAL`
- `SPECIAL_CATEGORY_OR_BIOMETRIC`
- `FINANCIAL`
- `CREDENTIAL_OR_SECRET`
- `SECURITY_AUDIT`
- `LEGAL_HOLD`

Every schema field and stored object declares a classification or inherits from a validated object schema.

## 8.2 Purpose and consent/legal basis

Each personal-data record includes:

- authenticated principal/data subject;
- purpose identifier;
- consent or other lawful/authorised basis where applicable;
- collection source;
- permitted consumers;
- retention class;
- onward-disclosure restrictions;
- revocation/withdrawal state.

Kai-generated personality, emotion, values, loyalty, cognitive fingerprint, audio, camera, screen and financial records require explicit purpose controls rather than generic “memory”.

## 8.3 Encryption

- TLS/mTLS in transit.
- Envelope encryption at rest for sensitive classes.
- Separate data-encryption keys by tenant/purpose/class where practical.
- Managed key rotation and revocation.
- Credentials/secrets use dedicated secret storage, never general memory/log/audit.
- Temporary files and worker disks encrypted or ephemeral.

## 8.4 Retention engine

Create a machine-enforced retention registry:

```text
record_type
data_class
purpose
active_retention
archive_retention
backup_retention
legal_hold_policy
deletion_method
owner
policy_revision
```

Retention workers use leases/fencing and lineage from P2. Completion produces evidence, not merely a successful scheduled run.

## 8.5 Logging policy

- Structured fields only.
- Sensitive-value denylist by schema, not regex alone.
- PII/secret scanning as defence-in-depth, not primary classification.
- Separate operational logs from security audit.
- UTC timestamps and operation correlation.
- No unconditional duplicate stdout/file exposure.
- Bounded retention, rotation and access controls.

## 8.6 Legal hold and incident preservation

Legal/incident hold prevents normal deletion only under an authorised, scoped record with:

- case/incident ID;
- approving authority;
- affected data and period;
- start/expiry/review date;
- access restrictions;
- eventual release process.

Hold state must propagate to backup and archive retention without activating broader indefinite retention.

---

# 9. Recovery architecture

## 9.1 Observation and action separation

Supervisor may observe and open an incident. It does not directly execute generic `/recover` calls.

Required components:

- Health observer.
- Incident authority.
- Recovery policy registry.
- Recovery executor using P1 capability and P2 isolation where needed.
- Postcondition verifier.
- Operator escalation path.

## 9.2 Incident record

```text
incident_id
service/resource
observations
first_seen
severity
failure_class
configuration_revision
containment_state
proposed_recovery_action
approval_requirement
recovery_operation_id
lease/fencing_token
attempts
postcondition
resolution_state
forensic_hold
```

## 9.3 Recovery policy registry

Each recoverable component declares exact actions such as:

- reconnect a client pool;
- restart one stateless worker generation;
- promote a validated replica;
- replay a specific durable outbox;
- rebuild one derivative index from authoritative source;
- restore a signed configuration revision.

Generic “reset all state”, “reload tokens” or “close breakers” is prohibited.

## 9.4 Recovery preconditions

- Authenticated incident and service identity.
- Current configuration/schema revision.
- No active conflicting operation.
- Evidence snapshot/forensic preservation where required.
- Fencing/maintenance mode.
- Approved capability.
- Defined rollback or manual-intervention state.

## 9.5 Postconditions

A recovery is successful only when:

- action outcome is committed;
- expected revision is active;
- deep readiness passes from an independent observer;
- protected canary operation passes where applicable;
- no containment/audit/privacy control was weakened;
- incident state and operator notification are durably updated.

Failure remains failure; it does not enter cooldown as though healed.

## 9.6 Recovery limits

- Per-incident and per-service attempt budgets.
- Exponential bounded backoff with jitter.
- Circuit escalation to operator/manual intervention.
- No repeated destructive action without fresh approval.
- Recovery channel availability/readiness monitored separately.

---

# 10. Backup and restore architecture

## 10.1 Backup job authority

Every backup is a durable job with:

```text
backup_job_id
principal/operator authority
source environment
source service identities
snapshot boundary
component revisions
schema/migration versions
start/end sequence
artefact objects
checksums
signing key
manifest state
verification state
retention/legal hold
```

## 10.2 Coherent snapshot

Use service-specific supported snapshot mechanisms and a documented consistency point:

- database transaction/PITR boundary;
- Redis persistence/replication boundary if Redis data is authoritative;
- immutable object-store generations;
- audit segment boundary;
- memory/vector/graph source generations;
- configuration and image digests.

A “full backup” cannot be a list of unrelated files captured while writers continue without a common boundary.

## 10.3 Artefact publication

- Write to an isolated temporary object/key.
- Stream with byte limits and checksums.
- fsync/object-store durability confirmation.
- Malware/content-type and format validation where relevant.
- Commit signed manifest transactionally.
- Promote to immutable complete namespace only after all required components succeed.
- Mark partial jobs explicitly and make them ineligible for restore.

## 10.4 Restore eligibility

Restore accepts only:

- immutable manifest-selected artefacts;
- valid signature and checksums;
- expected component type/format;
- compatible source environment/schema/version;
- verified retention and authorisation state;
- no symlink/path selection;
- isolated restore worker identity.

Plaintext psql input with executable meta-commands is prohibited.

## 10.5 Restore execution

- Dedicated disposable restore environment.
- No inherited application secrets beyond scoped restore credentials.
- Network restricted to the target/test environment.
- Maintenance/fencing mode.
- Pre-restore recovery point.
- Transactional or supported PITR semantics.
- Strict client error handling.
- Process-group and resource limits.
- Full command/output audit with secrets excluded.

## 10.6 Restore qualification

Each backup policy requires scheduled isolated restore drills proving:

- manifest and checksums validate;
- database/storage starts;
- migrations/schema match;
- application-level record counts and invariants pass;
- memory/vector/graph lineage is consistent;
- audit segments verify;
- deletion tombstones/cryptographic expiry remain effective;
- recovery time and recovery point objectives are measured.

A backup not recently restore-qualified is `UNVERIFIED`, not healthy.

---

# 11. Ordered implementation PRs

## P3-PR-01 — Reliability and lifecycle contract freeze

**Backlog mapping:** KAI-REM-301, KAI-REM-302, KAI-REM-305  
**Runtime effect:** none initially

Deliverables:

- Service-state/error schema.
- Endpoint-class registry.
- Health/readiness contract.
- Operation/saga schema.
- Audit event/segment schema.
- Data-classification and retention registry schema.
- Recovery incident/action schema.
- Backup manifest and restore-result schema.

Acceptance:

- Cross-service owners approve one vocabulary.
- Existing success-shaped error/stub patterns are listed for migration.
- Unknown states fail closed in conformance tests.

---

## P3-PR-02 — Replace shared retry/fallback semantics

**Primary source:** `common/resilience.py`

Required changes:

- Preserve exact HTTP method.
- Require endpoint class and operation ID.
- Distinguish transient/permanent/policy/auth/unknown-outcome failures.
- Remove success-shaped fallback values.
- Add bounded jitter and Retry-After handling.
- Reconcile unknown mutation outcomes before retry.

Acceptance:

- Committed-but-timed-out mutation executes once.
- DELETE/PATCH/PUT are not converted to POST.
- Auth/policy failures are not retried.
- Fallback is explicit degraded result, never ordinary business data.

---

## P3-PR-03 — Standard service health library

**Primary source:** `common/runtime.py`, service health routes

Required changes:

- Separate liveness/readiness/capability status.
- Require registered checks and complete deadline.
- Include validated service/config/image/schema identity.
- Reject duplicate check names and unknown states.
- Use monotonic durations and UTC event time.
- Remove raw exception leakage.

Acceptance:

- Zero checks cannot report ready.
- Nested failed check makes readiness non-ready.
- Stub, disabled, migration-required and unavailable states remain distinct.
- Required background task absence before first heartbeat is detected.

---

## P3-PR-04 — Durable operation and idempotency authority

Required changes:

- Transactional operation table/state machine.
- Principal-scoped idempotency uniqueness.
- Unknown-outcome reconciliation.
- Attempt and effect checkpoints.
- API for operation status and safe retry guidance.

Acceptance:

- Duplicate clients/workers converge on one operation.
- Restart/failover preserves state.
- Same key with different digest/principal is rejected.
- Unknown effect cannot be labelled failed-before-effect or successful without evidence.

---

## P3-PR-05 — Transactional outbox/inbox foundation

Required changes:

- Shared event envelope.
- Outbox committed with source mutation.
- Inbox deduplication and processing state.
- Dead-letter/manual intervention workflow.
- Backpressure/readiness thresholds.

Acceptance:

- Crash after commit but before send eventually delivers once logically.
- Duplicate deliveries do not duplicate effects.
- Poison event is visible and does not block unrelated partitions indefinitely.

---

## P3-PR-06 — Shared breaker and dependency-state authority

Required changes:

- Canonical dependency registry.
- Shared bounded observations.
- Minimum sample floor.
- One half-open probe lease.
- Configuration revision and fencing.
- Manual containment override.

Acceptance:

- Multiple workers observe one state.
- Unknown restored state blocks traffic.
- One unrelated success cannot close a failing dependency.
- Probe concurrency is one.

---

## P3-PR-07 — Leader election and scheduler ownership

Migrate:

- Supervisor loops.
- Monitor/scheduled tasks where applicable.
- Backup scheduler.
- Audit checkpoint publisher.
- Retention/deletion workers.
- Index maintenance.

Acceptance:

- One active fenced leader.
- Old leader cannot commit after lease loss.
- Failover resumes from durable task state without duplicate action.

---

## P3-PR-08 — Supervisor observation-only rebuild

**Primary source:** `supervisor/app.py`

Required changes:

- Remove direct generic `/recover` execution from sweeps.
- Authenticated workload-only observation.
- Typed health parsing.
- Durable fleet observations/incidents.
- Complete architecture-manifest service inventory.
- Owned lifecycle/task handles and graceful drain.

Acceptance:

- Repeated sweep cannot mutate target service.
- Missing/unknown status is unhealthy/unknown, never healthy.
- Initial state is `NO_EVIDENCE`, not green.
- Multiple replicas do not duplicate incident creation.

---

## P3-PR-09 — Recovery policy and incident authority

Required changes:

- Incident records.
- Exact service-specific recovery action registry.
- Phase 1 capability and operator approval rules.
- Leases/fencing and attempt budgets.
- Postcondition verifier.

Acceptance:

- Generic token/health response cannot trigger recovery.
- Failed recovery remains failed and escalates.
- Security token/nonce reload requires explicit protected action.
- Recovery cannot close containment without postcondition evidence.

---

## P3-PR-10 — Remove fabricated healing knowledge

**Primary source:** shared HealingEngine and consumers

Required changes:

- Separate diagnosis, proposal, approval, execution and verification.
- Remove `auto_recovery` success insertion.
- Require outcome evidence before learning a fix.
- Version and scope recovery knowledge by service/configuration.

Acceptance:

- Repeated `heal()` calls cannot advance state without action evidence.
- Caller-supplied `fix_applied` is an assertion, not proven knowledge.
- Known fix is never marked successful unless applied and verified.

---

## P3-PR-11 — Authoritative audit sequencer

Required changes:

- Transactional sequence allocation and append.
- Versioned structured event schema.
- Required audit availability for protected effects.
- Bounded segments and archive policy.
- Independent reader scopes.

Acceptance:

- Concurrent writers cannot fork order.
- No backend means `UNAVAILABLE`, not verified.
- Append failure prevents protected success.
- Event does not contain reusable credentials/full arbitrary payloads.

---

## P3-PR-12 — Signed audit segments and external checkpoints

Required changes:

- Asymmetric segment signing.
- Managed key lifecycle.
- Fixed range manifests and previous-root continuity.
- Immutable object storage.
- External checkpoint receipt.

Acceptance:

- Tamper, truncation, insertion and reordering detected.
- Failed publication records failure only.
- Root refers to the exact closed event range.
- Historical verification identifies key/revision and external receipt.

---

## P3-PR-13 — Tool Gate and Trust Ledger migration

Required changes:

- Replace local JSONL/in-memory authority with audit/operation authorities.
- Remove credentials/signatures from ledger payloads.
- Durable single-writer ordering.
- Snapshot-consistent trust scoring.
- Link acknowledgements/overrides to exact operations and verified outcomes.

Acceptance:

- Gate approval cannot succeed without durable decision/audit commit.
- Corruption blocks protected operation.
- Workers return one ledger generation/root.
- Trust score exposes exact evidence snapshot and never rewards zero/missing values through favourable defaults.

---

## P3-PR-14 — Data classification and schema annotations

Required changes:

- Classify API/storage fields.
- Add purpose, principal, consent/basis, retention and encryption metadata.
- Reject unclassified sensitive records.
- Define restricted consumers and exports.

Acceptance:

- Credentials cannot enter general memory/audit/log schemas.
- Biometric, screen, audio, financial and personality records have explicit purpose/retention.
- Cross-purpose access fails.

---

## P3-PR-15 — Encryption and key-management foundation

Required changes:

- Managed envelope encryption.
- Tenant/purpose/class key separation.
- Key rotation and access audit.
- Encrypted ephemeral/temp storage.
- Secret-manager integration.

Acceptance:

- Database/object theft does not reveal protected plaintext without keys.
- Revoked service cannot decrypt new data.
- Rotation preserves authorised historical access and deletion policy.

---

## P3-PR-16 — Retention, deletion and legal-hold engine

Dependencies: P2 lineage/deletion and P3 classification.

Required changes:

- Machine-readable retention registry.
- Fenced deletion workers.
- Backup/archive expiry propagation.
- Legal/incident hold records and review.
- Completion evidence and exception queue.

Acceptance:

- Expired data removed across primary/derivative/log/audit/backup policy.
- Hold prevents only scoped deletion.
- Release of hold resumes policy.
- Failed deletion is visible and cannot be reported complete.

---

## P3-PR-17 — Structured operational logging

**Primary source:** `common/runtime.py`

Required changes:

- Real JSON encoding of structured dictionaries.
- Safe fixed log destinations/permissions.
- UTC timestamps and operation correlation.
- Schema-based redaction and data minimisation.
- Separate operational/security-audit channels.
- Bounded rotation/retention.

Acceptance:

- Newline/JSON/control injection does not create forged records.
- Sensitive values are not duplicated to stdout/file.
- Logs remain parseable and revisioned.

---

## P3-PR-18 — Backup job and immutable manifest rebuild

Required changes:

- Reviewed tool image and readiness probes.
- Durable backup storage.
- Coherent source snapshot boundary.
- Temporary-to-final atomic publication.
- Signed manifest, checksums, source identity/version.
- Partial jobs ineligible for restore.

Acceptance:

- Redis/DB/memory/audit component success requires artefact existence and checksum.
- No `localdev` fallback or credential process-argument exposure.
- Symlink/unmanifested artefacts rejected.
- Full backup proves common consistency boundary.

---

## P3-PR-19 — Isolated restore and qualification pipeline

Required changes:

- Manifest-only selection.
- Disposable restore worker.
- Maintenance/fencing and pre-restore recovery point.
- Strict supported restore format.
- Application-level verification and RTO/RPO evidence.

Acceptance:

- psql meta-command artefact rejected.
- Partial/incompatible/tampered backup rejected.
- Scheduled restore drill succeeds against isolated environment.
- Result links exact manifest and release revision.

---

## P3-PR-20 — Incident response and evidence preservation workflow

Required changes:

- Incident creation/severity/ownership.
- Containment and legal/forensic hold.
- Evidence snapshot manifest.
- Recovery approval/escalation.
- Post-incident review and corrective-action linkage.

Acceptance:

- Recovery cannot destroy sole evidence copy.
- Incident access is audited.
- Containment state survives restart/failover.
- Closure requires verified postconditions and evidence references.

---

## P3-PR-21 — Integrated chaos, restore and operational release gate

Required tests:

- Multi-worker race and failover.
- Leader lease loss/fencing.
- Committed-but-timed-out mutation.
- Redis/Postgres/object-store interruption.
- Audit backend failure and backpressure.
- Log/audit injection and secret-leak tests.
- Health unknown/stub/degraded-state conformance.
- Recovery failure/partial-effect/postcondition tests.
- Backup partial/tampered/incompatible/symlink tests.
- Isolated restore qualification.
- Retention/deletion/legal-hold propagation.
- Clock change and monotonic-duration tests.

Release evidence:

- immutable source/image/config digests;
- signed test/chaos report;
- SBOM and provenance;
- audit segment verification;
- restore report and measured RTO/RPO;
- unresolved risk register;
- owner approval for every Phase 3 gate.

---

# 12. Phase 3 adversarial and failure closure tests

## Test P3-A — Success-shaped failure rejection

Return HTTP 200 bodies representing error, blocked, stale, stub, empty fallback and partial state.

**Pass:** all consumers preserve the typed non-success state; no breaker/recovery/trust success is recorded.

## Test P3-B — Committed timeout

Commit a mutation, drop the response and force client retry/failover.

**Pass:** one logical effect; client reconciles operation state rather than replaying blindly.

## Test P3-C — Multi-worker breaker race

Generate concurrent successes/failures across replicas.

**Pass:** one shared dependency state, bounded samples and one half-open probe.

## Test P3-D — Leader fencing

Pause leader past lease expiry, elect replacement, then resume old process.

**Pass:** stale leader cannot commit scheduled/recovery/backup/audit work.

## Test P3-E — Audit concurrency and tampering

Append concurrently, delete/reorder/edit one event and alter checkpoint files.

**Pass:** order remains linear under concurrency; every tamper is detected against signed externally anchored segment.

## Test P3-F — Audit unavailable

Remove audit authority during a protected mutation.

**Pass:** operation does not claim success; readiness is non-ready and recovery does not silently disable audit.

## Test P3-G — Health-to-recovery separation

Spoof degraded health and repeatedly call sweep/status.

**Pass:** incident may be created, but no recovery effect occurs without exact authorised action and postcondition process.

## Test P3-H — Recovery partial failure

Interrupt recovery after partial state change.

**Pass:** incident becomes partial/manual or compensation-required; it is not labelled recovered or cooled down as success.

## Test P3-I — Backup tampering and command content

Modify artefact bytes, manifest, source environment and include psql meta-commands/symlinks.

**Pass:** restore is rejected before target mutation.

## Test P3-J — Restore qualification

Restore the exact signed set in isolation and execute application/data/audit invariants.

**Pass:** measured RTO/RPO and verified outcome attached to manifest.

## Test P3-K — Privacy/log leakage

Submit credentials, PII, biometric, financial and hostile control characters through every input path.

**Pass:** classified stores enforce purpose/encryption; logs/audit contain only permitted minimised fields and remain parseable.

## Test P3-L — Retention and legal hold

Expire a record with derivatives/backups, apply/release a scoped hold and inject deletion failures.

**Pass:** correct items are retained/deleted, failures remain open and completion is evidence-backed.

## Test P3-M — Clock manipulation

Move wall clock forward/back while operations, leases, cooldowns and deadlines run.

**Pass:** security/reliability duration logic uses monotonic or authoritative sequencing; expiry/event time remains explicit.

## Test P3-N — Incident evidence preservation

Trigger compromise, containment and recovery while attempting log/backup/ledger rotation.

**Pass:** held evidence remains immutable and independently readable; recovery cannot erase the only copy.

---

# 13. Phase 3 exit criteria

Phase 3 is complete only when all are true:

- All protected services implement the standard state/error/health contract.
- No success-shaped fallback, stub or error is accepted as real success.
- Mutations use durable operation identities and transactional idempotency.
- Cross-service commands/events use outbox/inbox semantics.
- Security/reliability state is shared, transactional and multi-worker safe.
- Singleton workers use leased leadership and fencing.
- Supervisor observes and opens incidents but cannot perform generic recovery.
- Recovery is exact, authorised, fenced and postcondition verified.
- Fabricated healing knowledge and call-count phase progression are removed.
- Audit append is transactional, signed, segmented and externally anchored.
- Gate/Trust Ledger decisions and scores use authoritative consistent snapshots.
- Data is classified, purpose-scoped, encrypted and retention governed.
- Logging is structured, minimised, protected and parseable.
- Legal/incident hold and evidence preservation are enforceable.
- Backups are coherent, immutable, signed and manifest-bound.
- Restores are isolated and regularly qualified against application invariants.
- Failure injection, chaos, clock, race, privacy and restore tests pass against immutable release artefacts.

Passing Phase 3 permits formal production-qualification work only. It does not authorise consequential autonomy, financial action, self-modification or trust escalation. Phase 4 must requalify models, evidence, verification, confidence and autonomy on top of the completed P0–P3 controls.

---

# 14. Immediate next implementation queue

After P0–P2 prerequisites are implemented and verified:

1. P3-PR-01 — freeze state, health, audit, privacy, recovery and backup contracts.
2. P3-PR-02/03 — replace retry/fallback and health semantics.
3. P3-PR-04/05 — durable operations and outbox/inbox.
4. P3-PR-06/07 — shared breakers and fenced leadership.
5. P3-PR-08/09/10 — Supervisor, recovery authority and healing correction.
6. P3-PR-11/12/13 — audit sequencer, external anchoring and ledger migration.
7. P3-PR-14/15/16/17 — classification, encryption, retention and logging.
8. P3-PR-18/19 — backup and restore rebuild.
9. P3-PR-20 — incident/evidence workflow.
10. P3-PR-21 — integrated operational release gate.

Do not restore automatic recovery, production retention claims or backup assurances ahead of their transactional, audit and verification dependencies.

---

## Final Phase 3 planning judgement

Kai’s current reliability and recovery mechanisms cannot be made trustworthy through additional health checks or retry counts. The central defects are semantic and architectural: failures become normal-looking data, mutations replay without authoritative operation state, breakers and schedulers disagree across workers, recovery is triggered by shallow observation, audit chains fork or verify absence, and backups are not demonstrably durable or restorable.

The minimum defensible correction is a typed distributed-operation model, transactional shared control state, fenced ownership, signed externally anchored audit, enforceable privacy lifecycle and independently verified backup/restore system.

**Current status remains NO_GO. This document implements no runtime remediation and closes no findings.**
