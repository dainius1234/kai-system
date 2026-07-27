# Kai System — Prioritised Remediation Backlog

Repository: `dainius1234/kai-system`  
Source audit: `kai-pm/CODE_AUDIT_FINAL_REPORT.md`  
Confirmed audit baseline: **2,529 findings — 221 Critical, 1,284 High, 1,021 Medium, 3 Low**  
Status: **PLANNING ONLY — NO REMEDIATION IMPLEMENTED BY THIS COMMIT**

---

## 1. Purpose

This backlog converts the completed source audit into an executable remediation programme. It deliberately organises work around security invariants and architectural failure modes rather than treating 2,529 findings as unrelated patches.

The governing rule is:

> Do not re-enable a consequential capability because its local defect was patched. Re-enable it only after the complete trust path, enforcement boundary, evidence chain and failure behaviour pass the applicable release gate.

This document is a planning artefact. It does not claim that any finding is fixed, accepted or mitigated.

---

## 2. Operating constraints

Until the release gates in this backlog are satisfied:

- Treat the stack as a development laboratory only.
- Do not expose it to the Internet or a shared LAN.
- Do not load sensitive personal, financial, credential or operational data.
- Do not allow autonomous financial, execution, browser, surveillance or recovery actions.
- Do not treat Dashboard, Agentic, Verifier, Fusion, Tool Gate or Trust outputs as authoritative security evidence.
- Preserve current audit records, logs and volumes before destructive cleanup.

---

## 3. Priority model

| Priority | Meaning | Release implication |
|---|---|---|
| **P0 — Containment** | Stops currently reachable compromise paths and prevents further exposure. | Required before any connected development environment is used. |
| **P1 — Security foundation** | Establishes authenticated identity, canonical operations and final-boundary enforcement. | Required before any privileged service is re-enabled. |
| **P2 — Isolation and integrity** | Adds execution containment, egress control, evidence integrity and data partitioning. | Required before autonomous or sensitive-data workflows. |
| **P3 — Reliability and auditability** | Makes distributed state, recovery, audit and lifecycle behaviour dependable. | Required before production qualification. |
| **P4 — Capability requalification** | Rebuilds model, confidence, verification and autonomy controls on trusted foundations. | Required before consequential autonomous decisions. |

A backlog item is not complete because code merged. Completion requires the listed closure evidence.

---

## 4. Programme sequence

The mandatory order is:

1. **Contain exposure and preserve evidence.**
2. **Establish principal and workload identity.**
3. **Define one canonical operation and capability model.**
4. **Enforce policy at every final side-effect boundary.**
5. **Isolate execution, browser and network egress.**
6. **Rebuild evidence, memory and data integrity.**
7. **Standardise mutation, failure, health and recovery semantics.**
8. **Create authoritative audit, privacy and backup controls.**
9. **Requalify models, verification, trust and autonomy.**
10. **Run adversarial release testing against complete attack chains.**

Later phases may be designed in parallel, but no later capability may be operationally released ahead of its dependencies.

---

# 5. P0 — Immediate containment backlog

## KAI-REM-001 — Remove direct host exposure

**Owner role:** Platform / Infrastructure  
**Priority:** P0  
**Dependencies:** None

### Scope

- Remove host publishing for all privileged internal services.
- Bind unavoidable development endpoints to loopback only.
- Place any user-facing entrypoint behind one authenticated reverse proxy.
- Remove alternate direct ports that bypass the proxy.
- Verify both full and minimal Compose definitions.

### Minimum deliverables

- Revised Compose manifests with no unintended `0.0.0.0` exposure.
- Documented inbound network matrix.
- Automated check that rejects privileged host-published ports.

### Closure evidence

- External and LAN scans show no privileged service reachable directly.
- Container-network scan confirms only explicitly permitted paths.
- Dashboard, Executor, Tool Gate, memU, Supervisor, Vault, browser, finance and sensor services cannot be reached outside the approved ingress path.

### Audit risk addressed

Direct anonymous access, Dashboard compromise, Executor bypass, service-fleet pivot and broad flat-network exposure.

---

## KAI-REM-002 — Disable consequential services by default

**Owner role:** Platform / Product Security  
**Priority:** P0  
**Dependencies:** KAI-REM-001

### Scope

Create explicit opt-in deployment profiles for:

- Executor and arbitrary command facilities.
- Browser Agent, Web Scout and broad egress services.
- Vault Sync and arbitrary file ingestion.
- Dashboard administrative proxies.
- Introspection and graph ingestion.
- Autonomous finance and broker mutation.
- Camera, screen, clipboard, audio and wake ingestion.
- Supervisor-initiated recovery.

### Closure evidence

- A default deployment starts without these capabilities.
- Enabling a capability requires a named profile and documented risk acceptance.
- CI asserts that default Compose cannot start dangerous profiles accidentally.

---

## KAI-REM-003 — Lock Tool Gate and remove fail-open startup

**Owner role:** Security Architecture / Tool Gate  
**Priority:** P0  
**Dependencies:** None

### Scope

- Remove automatic WORK-mode activation.
- Default to deny/locked state.
- Refuse startup where policy, key or mode configuration is missing.
- Ensure recovery cannot silently reset policy mode.
- Separate administrative mode changes from runtime service credentials.

### Closure evidence

- Clean install starts locked.
- Missing or invalid configuration prevents privileged execution.
- Mode changes require authenticated administrative action and produce signed audit evidence.
- Restart, recovery and configuration reload do not widen permissions.

---

## KAI-REM-004 — Rotate and eliminate fallback secrets

**Owner role:** Security Operations  
**Priority:** P0  
**Dependencies:** Evidence preservation completed

### Scope

Rotate and invalidate all known or potentially exposed:

- Database credentials.
- HMAC/signing keys.
- Dashboard bridge and Tool Gate tokens.
- Broker, Telegram, email and external-provider credentials.
- Session, API and webhook secrets.

Remove fallback values such as `localdev` and any source-controlled development secret accepted in a privileged environment.

### Closure evidence

- Secret inventory completed.
- Old credentials demonstrably rejected.
- Startup fails closed when mandatory secrets are absent.
- Secret-scanning policy blocks new committed credentials.

---

## KAI-REM-005 — Temporary network segmentation

**Owner role:** Platform Security  
**Priority:** P0  
**Dependencies:** KAI-REM-001

### Scope

- Replace the single flat trust network with temporary deny-by-default segments.
- Separate ingress, control, data, execution, egress and observability planes.
- Permit only documented source-to-destination flows.
- Prevent execution and egress workers from reaching policy, identity, memory and audit control planes unless explicitly required.

### Closure evidence

- Network policy tests demonstrate blocked lateral movement.
- Compromised Executor/browser worker cannot reach Tool Gate administration, identity, audit, memory administration or database endpoints.

---

## KAI-REM-006 — Preserve incident and audit evidence

**Owner role:** Security Operations / Forensics  
**Priority:** P0  
**Dependencies:** None

### Scope

- Snapshot logs, volumes, databases, indexes, ledgers and deployment configuration before cleanup.
- Record hashes and acquisition timestamps.
- Store evidence read-only outside ordinary application retention.
- Document known gaps and mutable sources.

### Closure evidence

- Evidence manifest with hashes and custodians.
- Restore/read test from preserved copies.
- No remediation script mutates the only available evidence copy.

---

# 6. P1 — Identity, operation binding and final enforcement

## KAI-REM-101 — Authoritative principal identity

**Owner role:** Identity / Security Architecture  
**Priority:** P1  
**Dependencies:** KAI-REM-001, KAI-REM-004

### Scope

- Introduce authenticated human principals.
- Bind sessions to principal, tenant, device and assurance level.
- Reject caller-supplied identity fields unless they are derived from authenticated context.
- Remove hard-coded global identities such as `keeper` from authority decisions.
- Define service accounts independently from human users.

### Closure evidence

- Every privileged request has an authenticated principal.
- Body fields cannot impersonate another principal.
- Cross-principal data access tests fail.
- Identity appears consistently in operation, policy, data and audit records.

---

## KAI-REM-102 — Workload identity and authenticated service transport

**Owner role:** Platform / Identity  
**Priority:** P1  
**Dependencies:** KAI-REM-101

### Scope

- Give every service a unique workload identity.
- Use mTLS or equivalent authenticated transport internally.
- Bind endpoint scopes to workload identity.
- Remove shared bearer credentials from broad service groups.
- Rotate workload credentials automatically.

### Closure evidence

- Unknown workloads cannot call internal APIs.
- A low-purpose service cannot reuse its credential against higher-purpose endpoints.
- Transport confidentiality and peer authentication are verified in runtime tests.

---

## KAI-REM-103 — Delegation and scope authority

**Owner role:** Security Architecture  
**Priority:** P1  
**Dependencies:** KAI-REM-101, KAI-REM-102

### Scope

Define an explicit delegation chain containing:

- Human or system principal.
- Delegating workload.
- Allowed capability and resource.
- Purpose and policy version.
- Expiry and revocation state.
- Maximum consequence/budget.

### Closure evidence

- Anonymous Agentic input cannot become trusted Tool Gate identity.
- Dashboard cannot borrow a server-held administrative credential.
- Delegation scope is enforced by the destination service, not only by the caller.

---

## KAI-REM-104 — Canonical operation schema and digest

**Owner role:** Security Architecture / API Platform  
**Priority:** P1  
**Dependencies:** KAI-REM-101

### Scope

Create one canonical operation representation covering all security-relevant fields, including:

- Principal and delegation chain.
- Operation type, target and exact parameters.
- Data classifications and resources.
- Policy and model versions.
- Evidence references.
- Time, expiry, nonce and idempotency key.
- Expected outcome and consequence limits.

Use one deterministic serialisation and digest across request, approval, co-sign, capability, execution, outcome and audit.

### Closure evidence

- Mutation of any consequential parameter invalidates approval/capability.
- All participating services record the same operation digest.
- Conformance vectors pass across languages and services.

---

## KAI-REM-105 — Single-use execution capabilities

**Owner role:** Tool Gate / Security Architecture  
**Priority:** P1  
**Dependencies:** KAI-REM-103, KAI-REM-104

### Scope

Tool Gate must issue short-lived, single-use capabilities bound to:

- Canonical operation digest.
- Principal and delegated workload.
- Intended final executor.
- Policy version and decision.
- Expiry, nonce and consequence limits.

### Closure evidence

- Replay is rejected transactionally.
- Capability cannot be used by another service, principal or operation.
- Partial or reordered parameters fail verification.
- Capability consumption and outcome are atomically/auditably linked.

---

## KAI-REM-106 — Final side-effect enforcement

**Owner role:** All service owners, coordinated by Security Architecture  
**Priority:** P1  
**Dependencies:** KAI-REM-105

### Scope

Require a valid execution capability at every consequential boundary, including:

- Executor commands.
- Browser and external network actions.
- File reads/writes and Vault ingestion.
- Memory, graph, identity, values and trust mutation.
- Tool Gate mode/policy administration.
- Finance, broker, notification and email actions.
- Recovery, restart and security-state reset.
- Camera, screen, clipboard and audio acquisition where sensitive.

### Closure evidence

- Direct endpoint calls without capability fail.
- Alternate internal routes and legacy endpoints cannot bypass enforcement.
- Negative test suite covers every side-effect route.
- Penetration testing confirms that advisory-layer bypasses cannot reach an effect.

---

## KAI-REM-107 — Separate administrative, approval and runtime credentials

**Owner role:** Security Architecture / Operations  
**Priority:** P1  
**Dependencies:** KAI-REM-101 to KAI-REM-106

### Scope

- Separate service runtime identity from operator administration.
- Make co-sign and human approval interactive, intent-specific and non-reusable.
- Prevent service tokens from changing security mode or policy.
- Require step-up authentication for high-consequence administration.

### Closure evidence

- Compromise of one runtime token cannot administer policy.
- Approval artefacts cannot be replayed for another action.
- Administrative action has explicit authenticated operator evidence.

---

# 7. P2 — Execution, egress, evidence and data integrity

## KAI-REM-201 — Replace generic command execution

**Owner role:** Execution Platform  
**Priority:** P2  
**Dependencies:** KAI-REM-106

### Scope

- Remove generic shell, Python expression, Make, Git hook/alias and package-build primitives from ordinary execution.
- Replace them with fixed-schema operations and strict argument validation.
- Maintain a small explicit operation registry.
- Reject unknown binaries, flags, environment variables and redirections.

### Closure evidence

- Known command-injection and allowlist-bypass test corpus fails safely.
- No operation can invoke an interpreter or secondary command path unless explicitly designed and isolated.

---

## KAI-REM-202 — Disposable sandbox workers

**Owner role:** Execution Platform / Platform Security  
**Priority:** P2  
**Dependencies:** KAI-REM-201

### Scope

- One disposable worker per operation.
- Read-only minimal root filesystem.
- Explicit input/output mounts only.
- No Docker socket, host namespaces or broad device access.
- CPU, memory, process, syscall, time and output limits.
- Destroy worker after verified result collection.

### Closure evidence

- Escape and persistence tests fail.
- Worker cannot read unrelated files or credentials.
- Worker compromise does not survive the operation.
- Resource-exhaustion tests remain within enforced budgets.

---

## KAI-REM-203 — Hardened egress authority

**Owner role:** Network Security  
**Priority:** P2  
**Dependencies:** KAI-REM-005, KAI-REM-104, KAI-REM-106

### Scope

- Route browser, RSS, Web Scout, update and provider traffic through one controlled egress proxy.
- Validate scheme, hostname, DNS result, IP range, port and redirect on every connection.
- Block loopback, link-local, metadata, private and service-network destinations by default.
- Apply destination, data-volume and request budgets.

### Closure evidence

- SSRF test suite cannot reach internal services or metadata endpoints.
- DNS rebinding and redirect tests remain blocked.
- Egress records are bound to operation digest and principal.

---

## KAI-REM-204 — Immutable evidence and provenance schema

**Owner role:** Data Integrity / Security Architecture  
**Priority:** P2  
**Dependencies:** KAI-REM-101, KAI-REM-104

### Scope

Define typed evidence records distinguishing:

- External observation.
- User assertion.
- Model inference.
- System reflection.
- Operator approval.
- Independently observed outcome.

Each record must include source identity, content digest, source/event time, trust class, freshness, independence, purpose, supersession and contradiction links.

### Closure evidence

- Caller-generated scores cannot masquerade as independent evidence.
- Duplicate or correlated sources are detectable.
- Generated content cannot certify its own outcome.
- Evidence required by a policy is immutable and retrievable by digest.

---

## KAI-REM-205 — Principal and purpose partitioning

**Owner role:** Data Platform / Privacy  
**Priority:** P2  
**Dependencies:** KAI-REM-101

### Scope

Partition every store and derivative by:

- Principal/tenant.
- Session/device where applicable.
- Purpose and consent.
- Data classification.
- Retention class.

Apply to Postgres, Redis, local files, JSONL, vectors, graph, browser storage, logs, archives and backups.

### Closure evidence

- Cross-principal and cross-purpose queries fail by construction.
- Search, prompt assembly and graph traversal cannot merge unrelated identities.
- Partition keys are enforced at storage and API layers.

---

## KAI-REM-206 — Authenticated memory writes and prompt assembly

**Owner role:** Memory / Agentic  
**Priority:** P2  
**Dependencies:** KAI-REM-204, KAI-REM-205

### Scope

- Require authenticated source identity for every memory write.
- Separate raw memory from approved prompt context.
- Treat external/user/model-generated content as untrusted data.
- Apply provenance-aware filtering, quoting and instruction isolation.
- Prevent pinned/correction/preference flags from granting authority by caller assertion.

### Closure evidence

- Persistent prompt-poisoning attack corpus cannot create privileged instructions.
- Prompt context visibly preserves source/trust labels.
- Unverified records cannot alter policy, identity, Tool Gate mode or execution authority.

---

## KAI-REM-207 — Transactional memory, vector and graph updates

**Owner role:** Data Platform / Memory  
**Priority:** P2  
**Dependencies:** KAI-REM-205

### Scope

- Introduce durable operation state and outbox/inbox processing.
- Make add, cognify, source mapping, vector and graph updates idempotent.
- Assign one writer to each mutable index.
- Preserve prior lineage until replacement is fully committed.
- Add compensating cleanup and verified terminal state.

### Closure evidence

- Failure at every intermediate step leaves a recoverable, diagnosed state.
- Re-ingest does not orphan prior graph/vector content.
- Concurrent writers cannot corrupt or lose updates.
- Replays converge to one correct state.

---

## KAI-REM-208 — Verified supersession and derivative deletion

**Owner role:** Privacy / Data Platform  
**Priority:** P2  
**Dependencies:** KAI-REM-205, KAI-REM-207

### Scope

- Maintain lineage from source to every derivative.
- Implement supersession, contradiction and tombstone semantics.
- Delete or render inaccessible all derivatives when required.
- Address backups through defined expiry/cryptographic erasure strategy.

### Closure evidence

- End-to-end deletion test proves removal from primary, cache, vector, graph, logs where required, archives and restoration paths.
- Re-ingest cannot resurrect deleted data without new authorised source evidence.

---

# 8. P3 — Distributed reliability, audit, privacy and recovery

## KAI-REM-301 — Standard health and failure contracts

**Owner role:** API Platform / Reliability  
**Priority:** P3  
**Dependencies:** P1 foundation

### Scope

Standardise:

- Liveness.
- Readiness.
- Degraded/stale.
- Blocked/rejected.
- Unavailable/failed.
- Stub/non-authoritative.

Use non-2xx status for failed or blocked operations and machine-readable reason codes.

### Closure evidence

- No stub, failure or rejected state is success-shaped.
- Compose and Supervisor health checks test readiness, not HTTP reachability only.
- Downstream services cannot interpret unavailable evidence as neutral/positive evidence.

---

## KAI-REM-302 — Idempotent mutation state machines

**Owner role:** API Platform / Service Owners  
**Priority:** P3  
**Dependencies:** KAI-REM-104

### Scope

For every mutation define:

- Operation ID/digest.
- Accepted state.
- In-progress steps.
- Terminal success/failure.
- Retry semantics.
- Compensating action.
- Outcome verification.

### Closure evidence

- Duplicate, reordered and interrupted requests converge safely.
- Partial cross-service commits are visible and recoverable.
- Caller can determine authoritative terminal state.

---

## KAI-REM-303 — Shared transactional security state

**Owner role:** Security Platform / Reliability  
**Priority:** P3  
**Dependencies:** KAI-REM-102, KAI-REM-105

### Scope

Move nonces, capability consumption, breakers, revocation, rate limits and security mode from process-local/file-backed state to transactional shared stores with controlled ownership.

### Closure evidence

- Restart or multi-replica deployment does not reset security state.
- Concurrent capability use has one winner.
- Revocation propagates consistently.

---

## KAI-REM-304 — Leader election and worker ownership

**Owner role:** Reliability / Platform  
**Priority:** P3  
**Dependencies:** KAI-REM-303

### Scope

- Add leader election or distributed leases for schedulers, sweepers, graph workers, notification workers and maintenance jobs.
- Define lease expiry, fencing and takeover.
- Add graceful shutdown and resource drains.

### Closure evidence

- Multi-replica tests produce one authoritative worker action.
- Split-brain and lease-expiry tests cannot duplicate consequential effects.

---

## KAI-REM-305 — Signed append-only audit authority

**Owner role:** Security Platform / Compliance  
**Priority:** P3  
**Dependencies:** KAI-REM-104, KAI-REM-102

### Scope

Create one transactional audit service recording:

- Principal and workload identity.
- Delegation chain.
- Canonical operation digest.
- Policy/model/configuration versions.
- Evidence references.
- Decision, capability issue/consume and outcome.
- Administrative and recovery actions.

Sign entries and checkpoint them outside the ordinary application trust domain.

### Closure evidence

- Missing audit write fails closed for consequential actions.
- Mutation/deletion is detectable.
- Complete attack-chain reconstruction is possible from audit evidence.
- Ordinary log rotation cannot erase authoritative records.

---

## KAI-REM-306 — Data classification, consent and retention

**Owner role:** Privacy / Governance  
**Priority:** P3  
**Dependencies:** KAI-REM-205, KAI-REM-305

### Scope

- Create a data inventory and classification registry.
- Define allowed purpose, consent basis, retention and deletion per class.
- Enforce minimisation at collection and prompt assembly.
- Protect camera, screen, clipboard, audio, email, finance and personal-memory data as high sensitivity.

### Closure evidence

- Every sensitive field maps to a purpose and retention policy.
- Expiry jobs are idempotent and audited.
- Collection without required consent/purpose is rejected.

---

## KAI-REM-307 — Encryption and secret separation

**Owner role:** Security Operations / Data Platform  
**Priority:** P3  
**Dependencies:** KAI-REM-004, KAI-REM-205

### Scope

- Encrypt sensitive stores, indexes, queues and backups.
- Separate encryption keys from application data and runtime images.
- Use scoped envelope keys where practical.
- Document key rotation and revocation.

### Closure evidence

- Stolen volume/backup cannot be read without separately controlled key material.
- Rotation and recovery tests succeed without data loss.

---

## KAI-REM-308 — Authorised recovery architecture

**Owner role:** Reliability / Security Operations  
**Priority:** P3  
**Dependencies:** KAI-REM-301, KAI-REM-303, KAI-REM-305

### Scope

- Separate health observation from recovery authority.
- Require diagnosed cause, authorised operation and idempotent recovery plan.
- Prevent recovery from clearing policy, tokens, nonces or containment without explicit approval.
- Verify postconditions before declaring recovery complete.

### Closure evidence

- Public or forged health signals cannot trigger recovery.
- Recovery does not widen permissions or reset security state.
- Every recovery action is linked to diagnosis, operator/automation authority and verified outcome.

---

## KAI-REM-309 — Backup restore qualification

**Owner role:** Operations / Data Platform  
**Priority:** P3  
**Dependencies:** KAI-REM-307, KAI-REM-308

### Scope

- Define backup scope and consistency points.
- Encrypt and integrity-protect backups.
- Test clean-environment restore.
- Verify restored identity, policy, audit, memory lineage and deletion state.

### Closure evidence

- Successful documented restore exercise.
- Restore does not resurrect revoked credentials or deleted data outside policy.
- Restored services remain locked until integrity checks complete.

---

# 9. P4 — Model, verification, trust and autonomy requalification

## KAI-REM-401 — Canonical model and capability registry

**Owner role:** ML Platform / Governance  
**Priority:** P4  
**Dependencies:** KAI-REM-305

### Scope

- Record model artefact digest, provider, version, prompt/template, tool capability, data class, evaluation status and permitted consequence.
- Pin production artefacts by immutable digest.
- Treat unavailable/stub models as non-authoritative.

### Closure evidence

- Every consequential inference is attributable to an approved immutable artefact and configuration.
- Unknown or changed model artefacts cannot run consequential workflows.

---

## KAI-REM-402 — Rebuild Verifier around trusted evidence

**Owner role:** Verification / ML Safety  
**Priority:** P4  
**Dependencies:** KAI-REM-204, KAI-REM-401

### Scope

- Reject caller-supplied trust scores as proof.
- Deduplicate evidence by provenance and content digest.
- Test entailment and contradiction, not word overlap.
- Require evidence independence, freshness and relevance.
- Make FAIL, REPAIR and unavailable states binding at consequential boundaries.

### Closure evidence

- Forged, duplicate, correlated and contradictory evidence attack sets fail.
- A Verifier PASS includes reproducible evidence references and policy version.
- Verifier unavailability cannot produce a positive downstream decision.

---

## KAI-REM-403 — Rebuild Fusion consensus semantics

**Owner role:** Verification / ML Safety  
**Priority:** P4  
**Dependencies:** KAI-REM-402

### Scope

- Require a defined number of genuinely independent specialists.
- Reject duplicate specialist identity and failed/stub responses.
- Prevent caller-selected zero/unsafe thresholds.
- Account for correlated models, prompts and data.
- Bind consensus to one canonical operation and evidence set.

### Closure evidence

- One model, duplicate models or failed specialists cannot form consensus.
- Consensus cannot override a binding Verifier failure.
- Independence assumptions are measured and logged.

---

## KAI-REM-404 — Replace style-based conviction and trust inflation

**Owner role:** ML Safety / Agentic  
**Priority:** P4  
**Dependencies:** KAI-REM-204, KAI-REM-402

### Scope

- Remove confidence based on fluency, formatting or self-consistency alone.
- Calibrate decision confidence per evidence type and consequence.
- Separate predicted success from independently observed outcome.
- Exclude self-generated reflections, loyalty, gratitude, values or moral statements from authority unless independently validated for a narrow purpose.

### Closure evidence

- Synthetic certainty cannot raise execution authority.
- Generated outcomes cannot recursively certify future autonomy.
- Calibration and abstention metrics are published per consequence class.

---

## KAI-REM-405 — Financial decision safety case

**Owner role:** Financial Controls / ML Safety  
**Priority:** P4  
**Dependencies:** KAI-REM-402 to KAI-REM-404, KAI-REM-106

### Scope

- Separate market observation, strategy proposal, risk approval and broker execution.
- Treat correlated indicators as correlated.
- Fail closed on governance, provider or freshness errors.
- Bind execution to exact instrument, side, size, price limits and position.
- Prohibit broad effects such as closing every matching position from one weak signal.

### Closure evidence

- No autonomous live-money execution until independent simulation and controls review pass.
- Provider failure, stale data and one-signal cases abstain safely.
- Limits and exact order digest are enforced at broker boundary.

---

## KAI-REM-406 — Staged autonomy requalification

**Owner role:** Product Governance / Security / ML Safety  
**Priority:** P4  
**Dependencies:** All applicable P1–P4 items

### Scope

Re-enable capabilities in stages:

1. Read-only simulation with synthetic data.
2. Read-only operation with non-sensitive real data.
3. Human-approved low-consequence actions.
4. Bounded reversible actions.
5. Higher-consequence actions only after an explicit safety case.

### Closure evidence

- Each stage has defined scope, limits, rollback and independent outcomes.
- Advancement requires signed evidence that applicable release gates pass.
- Any critical regression automatically revokes the stage.

---

# 10. Cross-service attack-chain closure programme

Each confirmed compromise path must have a dedicated adversarial test. Local ticket closure is insufficient.

| Chain | Required proof of closure |
|---|---|
| Dashboard stored XSS → control plane | Context-safe rendering, restrictive CSP, authenticated ingress, least-privilege backend, no server credential borrowing, negative payload tests. |
| Direct Executor → arbitrary code → fleet pivot | No direct endpoint, exact capability enforcement, fixed-schema operations, disposable sandbox, deny-by-default network, escape/pivot test failure. |
| memU poisoning → privileged prompt | Authenticated provenance, principal partitioning, untrusted-context isolation, no caller-granted authority, persistent injection test failure. |
| Forged Verifier/Fusion evidence | Immutable evidence identity, deduplication, independence checks, binding failure states, adversarial consensus test failure. |
| Anonymous Agentic → trusted Gate actor | Authenticated principal/delegation, canonical digest, no hard-coded service impersonation, exact capability binding. |
| Tool Gate ledger → credential expansion | Ledger access scoped by principal/purpose, secrets excluded or protected, signed audit, low-purpose token cannot inspect stronger authority. |
| Vault arbitrary file → memory/prompt exfiltration | Explicit roots, capability-bound file reads, classification, provenance, principal partitioning, derivative lifecycle, sensitive-file test failure. |
| Health manipulation → recovery reset | Authenticated health evidence, separated recovery authority, diagnosed and capability-bound recovery, security-state preservation tests. |
| Personal/moral state poisoning → trust inflation | Untrusted source labelling, no caller-assigned authority, independent outcome evidence, trust/autonomy policy separation. |
| Weak market signal → autonomous financial mutation | Independent evidence, fail-closed governance, exact order capability, position/size limits, simulation and broker-boundary tests. |

---

# 11. Finding-level triage rules

The 2,529 findings should be linked to the architectural backlog using these rules:

1. **Every Critical and High finding receives:** owner, parent epic, affected service, attack-chain tag, closure test and evidence link.
2. **Duplicates are not silently closed:** preserve original finding ID and mark `covered-by` only where the same code change and test prove closure.
3. **Architectural findings close last:** component patches cannot close an architecture invariant until all relevant services conform.
4. **Risk acceptance is exceptional:** it must name the affected capability, exposure, compensating control, expiry and accountable approver.
5. **No closure by documentation alone:** implementation and runtime evidence are required.
6. **No severity downgrade without evidence:** changed assumptions must be demonstrable in deployment and test configuration.
7. **Regression test required:** each confirmed defect gains a test that fails before the fix and passes after it.

Recommended finding states:

- `OPEN`
- `MAPPED`
- `IN_PROGRESS`
- `CODE_COMPLETE`
- `EVIDENCE_PENDING`
- `VERIFIED`
- `RISK_ACCEPTED`
- `NOT_APPLICABLE_WITH_EVIDENCE`

Only `VERIFIED`, time-bounded `RISK_ACCEPTED`, or evidenced `NOT_APPLICABLE` remove a finding from the open release count.

---

# 12. Pull-request and change-control rules

For remediation pull requests:

- One security invariant or tightly related service set per PR.
- State exact audit finding IDs and parent backlog epic.
- Include threat model and before/after trust path.
- Include negative tests and failure-mode tests.
- Do not combine containment removal with capability re-enablement.
- Do not introduce compatibility bypasses that preserve an insecure legacy route.
- Record deployment/configuration changes alongside code changes.
- Require security review for P0/P1 and all cross-service chain closures.
- Require independent runtime evidence before marking `VERIFIED`.

Suggested PR evidence block:

```text
Backlog epic:
Audit finding IDs:
Threat/attack chain:
Security invariant introduced:
Negative tests:
Failure/recovery tests:
Deployment evidence:
Residual risk:
Reviewer sign-off:
```

---

# 13. Release gates

## Gate A — Isolated development use

All must pass:

- P0 containment complete.
- No privileged service directly host-published.
- Dangerous capabilities disabled by default.
- Tool Gate locked by default.
- Known credentials rotated; no fallback secrets.
- Temporary deny-by-default segmentation active.

## Gate B — Privileged internal testing

All Gate A requirements plus:

- Authenticated principal and workload identity.
- Canonical operation digest implemented.
- Single-use capability required at every tested side-effect boundary.
- Runtime and administrative credentials separated.
- Direct bypass tests fail.

## Gate C — Sensitive-data testing

All Gate B requirements plus:

- Principal/purpose partitioning.
- Authenticated provenance-aware memory.
- Data classification, encryption, retention and deletion controls.
- Signed authoritative audit.
- Verified backup/restore and derivative deletion tests.

## Gate D — Bounded autonomous simulation

All Gate C requirements plus:

- Sandboxed execution and hardened egress independently tested.
- Verifier and Fusion rebuilt on immutable evidence.
- Self-certification removed.
- Model/capability registry active.
- Consequence budgets and abstention behaviour verified.

## Gate E — Production qualification

All previous gates plus:

- Every Critical finding verified closed or explicitly time-bounded and accepted by accountable governance.
- Every High finding has closure evidence or approved release-blocking exception.
- All ten cross-service compromise chains demonstrably broken.
- Penetration testing, fuzzing, dependency/SCA, historical secret scan and runtime architecture review completed.
- Health, mutation, recovery and audit semantics pass failure-injection testing.
- Operational rollback, incident response and key-revocation exercises pass.

Production remains blocked while any applicable Gate E condition is unmet.

---

# 14. Programme evidence dashboard

Track at minimum:

| Measure | Required view |
|---|---|
| Findings by severity/state | Critical, High, Medium, Low; open versus verified. |
| Findings by service and parent epic | Reveals concentration and unmapped work. |
| Attack-chain status | Open, partially broken, fully verified. |
| Release-gate status | Each criterion with evidence link and approver. |
| Capability exposure | Disabled, isolated, test-only, production-qualified. |
| Secrets and identity migration | Legacy credentials remaining and rotation state. |
| Side-effect enforcement coverage | Endpoints protected versus total consequential endpoints. |
| Data lifecycle coverage | Stores with classification, retention, lineage and verified deletion. |
| Runtime assurance | Pen-test, fuzz, SCA, restore, failure-injection and incident exercises. |

Counts must be generated from the finding register, not manually copied into status reports.

---

# 15. Immediate implementation queue

The first implementation queue, in dependency order, is:

1. KAI-REM-006 — preserve evidence.
2. KAI-REM-001 — remove direct host exposure.
3. KAI-REM-002 — disable consequential services by default.
4. KAI-REM-003 — lock Tool Gate and remove fail-open startup.
5. KAI-REM-004 — rotate and eliminate fallback secrets.
6. KAI-REM-005 — temporary network segmentation.
7. KAI-REM-101/102 — principal and workload identity design.
8. KAI-REM-104 — canonical operation schema/digest.
9. KAI-REM-103/105 — delegation and single-use capability design.
10. KAI-REM-106 — final side-effect enforcement inventory and migration.

Do not start autonomy, confidence or model-quality remediation as a substitute for these foundational controls.

---

# 16. Final planning judgement

The fastest safe route is not a mass patch of endpoint defects. The highest-value sequence is containment, identity, canonical operation binding and final-boundary enforcement, followed by sandboxing, evidence/data integrity and authoritative audit. This sequence breaks multiple Critical attack chains with shared controls and creates a defensible basis for later component-level closure.

**Current status remains unchanged: 2,529 confirmed findings; no remediation implemented by this planning commit.**
