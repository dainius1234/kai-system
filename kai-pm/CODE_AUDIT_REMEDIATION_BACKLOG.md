# Kai System — Prioritised Remediation Backlog

Repository: `dainius1234/kai-system`  
Source audit: `kai-pm/CODE_AUDIT_FINAL_REPORT.md`  
Confirmed audit baseline: **4,580 findings — 252 Critical, 2,440 High, 1,885 Medium, 3 Low**  
Status: **PLANNING ONLY — NO REMEDIATION IMPLEMENTED BY THIS COMMIT**

---

## 1. Purpose

This backlog converts the completed source, deployment and architecture audit into an executable remediation programme. It organises work around security invariants and systemic failure modes rather than treating **4,580 findings** as unrelated patches.

The governing rule is:

> Do not re-enable a consequential capability because its local defect was patched. Re-enable it only after the complete identity path, trust decision, final enforcement boundary, evidence chain, failure behaviour and recovery path pass the applicable release gate.

This document is a planning artefact. It does not claim that any finding is fixed, accepted, mitigated or safe to defer.

---

## 2. Operating constraints

Until all applicable release gates are satisfied:

- Treat the stack as an isolated disposable development laboratory only.
- Do not expose it to the Internet or a shared LAN.
- Do not load sensitive personal, financial, credential, biometric or operational data.
- Do not permit autonomous execution, browser actions, recovery, financial decisions or external messaging.
- Do not treat Dashboard, Agentic, Verifier, Fusion, Tool Gate, Trust, self-audit or health output as authoritative evidence.
- Preserve current audit files, logs, databases, volumes, indexes and ledgers before destructive cleanup.
- Treat existing memory, preference, feedback, confidence, trust, personality and world-context records as untrusted.
- Treat current backups, checkpoints and ledgers as unverified until independently restored or validated.

---

## 3. Priority model

| Priority | Meaning | Release implication |
|---|---|---|
| **P0 — Containment** | Stops reachable compromise paths and prevents further exposure. | Required before any connected development environment is used. |
| **P1 — Security foundation** | Establishes authenticated identity, canonical operations and final-boundary enforcement. | Required before any privileged service is re-enabled. |
| **P2 — Isolation and integrity** | Adds execution containment, egress control, evidence integrity and data partitioning. | Required before sensitive-data or autonomous workflows. |
| **P3 — Reliability and auditability** | Makes distributed state, recovery, backup, audit and lifecycle behaviour dependable. | Required before production qualification. |
| **P4 — Capability requalification** | Rebuilds verification, models, confidence, trust and autonomy on trusted foundations. | Required before consequential autonomous decisions. |

A backlog item is not complete because code merged. Completion requires the listed closure evidence against an immutable tested revision.

---

## 4. Mandatory programme order

1. Contain exposure and preserve evidence.
2. Establish human, workload and service identity.
3. Define one canonical operation and capability model.
4. Enforce policy at every final side-effect boundary.
5. Isolate execution, browser, parser and network egress.
6. Rebuild memory, evidence and data integrity.
7. Standardise mutation, failure, health and recovery semantics.
8. Create authoritative audit, privacy, backup and restore controls.
9. Requalify models, verification, trust and autonomy.
10. Run adversarial release testing against complete cross-service attack chains.

Later phases may be designed in parallel, but no later capability may be operationally released ahead of its dependencies.

---

# 5. P0 — Immediate containment

## KAI-REM-001 — Remove direct host exposure

**Owner:** Platform / Infrastructure  
**Dependencies:** None

### Scope

- Remove host publishing for all privileged internal services.
- Bind unavoidable development endpoints to loopback only.
- Place the single user-facing entry point behind authenticated ingress.
- Remove alternate ports that bypass ingress.
- Verify both full and minimal Compose profiles.

### Closure evidence

- External and LAN scans show no privileged service reachable directly.
- Container-network tests prove only approved source-to-destination paths.
- Dashboard, Executor, Tool Gate, memU, Supervisor, Vault, browser, finance and sensor services cannot be reached outside approved ingress.

---

## KAI-REM-002 — Disable consequential services by default

**Owner:** Platform / Product Security  
**Dependencies:** KAI-REM-001

### Scope

Create explicit opt-in profiles for:

- Executor and arbitrary command facilities.
- Browser Agent, Web Scout and broad egress services.
- Vault/file ingestion and destructive storage operations.
- Dashboard administrative proxies.
- Introspection, self-improvement and graph mutation.
- Autonomous finance and broker mutation.
- Camera, screen, clipboard, audio, vision and wake ingestion.
- Supervisor-initiated recovery.
- Monitoring actions, Notify and TTS delivery.

### Closure evidence

- Default deployment starts without consequential capabilities.
- Enabling a capability requires a named profile and documented risk acceptance.
- CI rejects accidental activation in the default profile.

---

## KAI-REM-003 — Lock Tool Gate and remove fail-open startup

**Owner:** Security Architecture / Tool Gate  
**Dependencies:** None

### Scope

- Default to deny/locked state.
- Refuse startup when policy, key, token or mode configuration is missing or invalid.
- Remove automatic WORK-mode activation.
- Separate administrative mode changes from runtime service credentials.
- Ensure restart and recovery cannot widen permissions.

### Closure evidence

- Clean install starts locked.
- Missing or invalid security configuration prevents privileged execution.
- Mode changes require authenticated administrative action and signed audit evidence.

---

## KAI-REM-004 — Rotate and eliminate fallback secrets

**Owner:** Security Operations  
**Dependencies:** Evidence preservation completed

### Scope

Rotate and invalidate all known or potentially exposed:

- Database credentials.
- HMAC and signing keys.
- Dashboard bridge and Tool Gate tokens.
- Broker, Telegram, email and external-provider credentials.
- Session, API and webhook secrets.

Remove `localdev`, source-known development secrets and committed predictable tokens from privileged profiles.

### Closure evidence

- Complete secret inventory and owner recorded.
- Old credentials demonstrably rejected.
- Startup fails closed when mandatory secrets are absent.
- Secret scanning blocks new committed credentials.

---

## KAI-REM-005 — Temporary deny-by-default segmentation

**Owner:** Platform Security  
**Dependencies:** KAI-REM-001

### Scope

Separate ingress, control, data, execution, egress and observability planes. Permit only documented flows.

### Closure evidence

- Compromised Executor/browser/parser worker cannot reach policy administration, identity, audit, memory administration or databases except through explicit approved interfaces.
- Network-policy tests prove blocked lateral movement.

---

## KAI-REM-006 — Preserve incident and audit evidence

**Owner:** Security Operations / Forensics  
**Dependencies:** None

### Scope

- Snapshot logs, volumes, databases, indexes, ledgers and deployment configuration.
- Record hashes, acquisition times and custodians.
- Store evidence read-only outside normal retention.
- Document mutable or incomplete sources.

### Closure evidence

- Evidence manifest exists with verified hashes.
- Preserved copies can be read/restored independently.
- No remediation process mutates the only evidence copy.

---

# 6. P1 — Identity, operation binding and final enforcement

## KAI-REM-101 — Authoritative human principal identity

**Owner:** Identity / Security Architecture  
**Dependencies:** KAI-REM-001, KAI-REM-004

### Scope

- Authenticate human principals.
- Bind sessions to principal, tenant, device and assurance level.
- Remove body-supplied authority fields such as `keeper`, `dainius`, requester and actor strings.
- Separate human identity from service accounts.

### Closure evidence

- Every privileged request has an authenticated principal.
- Body fields cannot impersonate another principal.
- Cross-principal data-access tests fail.

---

## KAI-REM-102 — Workload identity and authenticated service transport

**Owner:** Platform / Identity  
**Dependencies:** KAI-REM-101

### Scope

- Give every service a unique workload identity.
- Use mTLS or equivalent authenticated transport.
- Bind endpoint scopes to workload identity.
- Replace broad shared bearer credentials.
- Rotate workload credentials automatically.

### Closure evidence

- Unknown workloads cannot call internal APIs.
- A low-purpose service cannot reuse its credential against administrative endpoints.
- Transport confidentiality and peer identity pass runtime tests.

---

## KAI-REM-103 — Delegation and scope authority

**Owner:** Security Architecture  
**Dependencies:** KAI-REM-101, KAI-REM-102

Define an explicit delegation chain containing principal, workload, capability, resource, purpose, policy revision, expiry, revocation and consequence budget.

### Closure evidence

- Anonymous Agentic input cannot become trusted Tool Gate identity.
- Dashboard cannot borrow a server-held administrative credential.
- Destinations enforce delegation scope independently.

---

## KAI-REM-104 — Canonical operation schema and digest

**Owner:** Security Architecture / API Platform  
**Dependencies:** KAI-REM-101

Create one deterministic representation for:

- Principal and delegation chain.
- Operation type, target and exact parameters.
- Classified data/resources.
- Policy, model and evidence revisions.
- Expiry, nonce and idempotency key.
- Expected outcome and consequence limits.

### Closure evidence

- Changing any consequential field invalidates approval.
- Request, approval, execution, result and audit contain the same operation digest.
- Conformance vectors pass across all services.

---

## KAI-REM-105 — Single-use execution capabilities

**Owner:** Tool Gate / Security Architecture  
**Dependencies:** KAI-REM-103, KAI-REM-104

Tool Gate must issue a short-lived, single-use capability bound to the exact operation, principal, delegated workload, final executor, policy revision, expiry and consequence budget.

### Closure evidence

- Replay is rejected transactionally.
- Capability cannot be used by another service, principal or operation.
- Consumption and outcome are durably linked.

---

## KAI-REM-106 — Final side-effect enforcement

**Owner:** All service owners  
**Dependencies:** KAI-REM-105

Require the exact capability at every consequential boundary, including:

- Executor commands and scripts.
- Browser and external network actions.
- File, Vault and backup mutation.
- Memory, graph, identity, values, preferences and trust mutation.
- Tool Gate mode and policy administration.
- Finance, broker, email, notification and TTS actions.
- Recovery, restart and security-state reset.
- Sensitive camera, screen, clipboard, audio and vision acquisition.

### Closure evidence

- Direct calls without capability fail.
- Legacy and alternate routes cannot bypass enforcement.
- Negative tests enumerate every side-effect route.

---

## KAI-REM-107 — Separate administrative, approval and runtime credentials

**Owner:** Security Architecture / Operations  
**Dependencies:** KAI-REM-101 to KAI-REM-106

### Closure evidence

- Compromise of one runtime credential cannot administer policy.
- Human approval cannot be replayed for another action.
- High-consequence administration requires step-up authentication and explicit operator evidence.

---

# 7. P2 — Isolation, egress, evidence and data integrity

## KAI-REM-201 — Replace generic command execution

**Owner:** Execution Platform  
**Dependencies:** KAI-REM-106

- Remove generic shell, Python expression, Make, Git-hook/alias and package-build primitives.
- Replace them with fixed-schema operations and strict argument validation.

### Closure evidence

- No arbitrary-code path through allowed operations.
- Every operation has bounded inputs, outputs, resources and postconditions.

---

## KAI-REM-202 — Isolate execution workloads

Use disposable, read-only, resource-limited sandboxes with no inherited secrets, no broad network and process-group termination.

### Closure evidence

- Escape, fork, timeout, descendant, filesystem and network tests pass.
- Cancellation leaves no surviving process or persistent side effect.

---

## KAI-REM-203 — Browser principal isolation

- One isolated browser context per principal/workflow.
- Clear cookies, storage, workers, downloads, popups and permissions on completion.
- Bind every action to page identity and verified postcondition.

### Closure evidence

- Cross-caller authenticated-state access fails.
- Monitor cannot scrape another workflow’s page.
- Popups/downloads/background traffic are bounded and auditable.

---

## KAI-REM-204 — Controlled egress and SSRF prevention

- Use one hardened egress proxy.
- Validate DNS and every redirect hop.
- Deny loopback, private, link-local and metadata destinations.
- Enforce destination, byte, request and content-type budgets.

### Closure evidence

- SSRF, DNS-rebinding and redirect tests fail safely.
- Parser, browser and Executor services cannot reach unauthorised internal endpoints.

---

## KAI-REM-205 — Parser and upload isolation

- Stream and bound uploads before materialisation.
- Validate magic bytes, archive expansion, OOXML containers and filenames.
- Run OCR/CAD/document converters in disposable sandboxes.
- Mark extracted content as untrusted evidence.

### Closure evidence

- Archive/decompression bombs, malformed multipart, symlink and converter-escape tests pass.
- Rejected uploads cause no downstream or local side effect.

---

## KAI-REM-206 — Principal-partitioned data model

Partition memory, sessions, finance, email, calendar, browser, sensor, personality, ledger and operator-model data by authenticated principal, tenant, purpose and retention policy.

### Closure evidence

- No hard-coded global `keeper` authority remains.
- Cross-user reads/writes fail at storage and API layers.
- Export, correction and deletion operate per principal and purpose.

---

## KAI-REM-207 — Memory provenance and poisoning control

Every memory must carry authenticated source, immutable evidence reference, trust state, verification state, lifecycle state and user/purpose scope.

### Closure evidence

- Duplicate/poisoned/self-generated records cannot strengthen evidence.
- Retrieval is read-only and does not silently increase authority.
- Corrections and supersession are atomic and auditable.

---

## KAI-REM-208 — Verifier rebuild

- Resolve evidence internally from signed references.
- Map evidence proposition-by-proposition.
- Model contradiction, independence, freshness and source trust.
- Require exact PASS for consequential use.

### Closure evidence

- Caller-fabricated, duplicate, negated and unrelated evidence cannot produce PASS.
- Missing or unavailable evidence fails closed.

---

## KAI-REM-209 — Fusion independence and enforcement

- Require a minimum number of unique live independent models/sources.
- Exclude stubs, duplicates and failures.
- Make Verifier PASS mandatory.
- Persist the full panel and decision evidence.

### Closure evidence

- One, failed, duplicated or stub specialist never produces consensus.
- Correlated-provider tests reduce decision authority.

---

## KAI-REM-210 — Sensitive-data privacy controls

Apply classification, consent, minimisation, encryption, no-store responses, retention and deletion to personal, financial, email, clipboard, biometric, audio, screen and operator-model data.

### Closure evidence

- Sensitive APIs require purpose-bound access.
- Caches/logs do not retain prohibited fields.
- Retention and deletion tests pass.

---

# 8. P3 — Transactional state, audit, recovery and assurance

## KAI-REM-301 — Shared transactional security state

Move modes, nonces, idempotency, pending approvals, breakers, trust and critical configuration to shared transactional stores with compare-and-swap revisions.

### Closure evidence

- Multi-worker and restart tests show one authoritative state.
- Concurrent updates do not lose or widen policy.

---

## KAI-REM-302 — Distributed mutation and saga semantics

Define idempotent operation records, durable steps, compensation and reconciliation for multi-service mutations.

### Closure evidence

- Timeout/retry after partial commit does not duplicate or conceal side effects.
- Every partial operation reaches completed, compensated or quarantined state.

---

## KAI-REM-303 — Standard typed failure and readiness contracts

Standardise `ready`, `degraded`, `blocked`, `unavailable`, `failed` and `completed` responses. Never encode failure as success-shaped HTTP 200 without a strict typed state.

### Closure evidence

- Health fails when mandatory capability is absent.
- Stubs and fallbacks are explicitly non-ready.
- Supervisors and release gates validate semantics, not key presence.

---

## KAI-REM-304 — Immutable audit chain

Record authenticated actor, delegation, operation digest, policy/model revisions, source evidence, before/after state, result and postcondition. Protect with append integrity and external anchoring.

### Closure evidence

- Audit cannot be silently skipped or rewritten.
- No credentials or reusable signatures are stored.
- Incident reconstruction succeeds from the audit trail alone.

---

## KAI-REM-305 — Service-specific recovery

Replace generic `/recover` with narrowly scoped idempotent operations requiring authenticated incident authority and verified postconditions.

### Closure evidence

- Recovery cannot reset containment without dependency proof.
- Failed recovery does not enter a false cooldown/success state.
- Supervisor cannot invoke unsupported recovery actions.

---

## KAI-REM-306 — Verified backup and restore

- Create coherent versioned snapshots.
- Sign manifests and checksums.
- Use isolated restoration workers.
- Reject symlinks, arbitrary files and psql meta-commands.
- Run regular restore drills.

### Closure evidence

- A selected backup restores into an isolated environment and passes integrity/application checks.
- Partial, tampered or incompatible backups are rejected.

---

## KAI-REM-307 — Lifecycle and workload governance

Every service must own clients, queues, background tasks, subprocess groups and shutdown drain through one lifespan contract. Add bounded concurrency, timeouts and quotas.

### Closure evidence

- No untracked tasks remain.
- Shutdown leaves no in-flight side effects or orphan processes.
- Load tests prove bounded memory, thread, process and connection growth.

---

## KAI-REM-308 — Production-equivalent CI profiles

- Remove global dev-secret and fake-embedding defaults.
- Test each service in its declared dependency environment.
- Run real authenticated integration paths.
- Upload signed reports, SBOMs and image digests.

### Closure evidence

- Secure missing-secret paths are tested.
- Mock versus live coverage is explicit.
- Built images, not only source trees, are scanned and exercised.

---

## KAI-REM-309 — Fail-closed release gate

The go/no-go authority must require authenticated semantic readiness, exact tested commit/image/configuration digests and complete critical-chain tests.

### Closure evidence

- Dashboard outage, malformed result or missing evidence produces NO_GO.
- Fatal JavaScript, Compose, Dockerfile, policy and secret defects are included.
- Final report is signed and retained.

---

## KAI-REM-310 — Adversarial resilience programme

Run isolated chaos, restart, multi-worker, clock-change, maintenance-race and dependency-failure tests against the actual target deployment.

### Closure evidence

- Tests use disposable stores and unique credentials/ports.
- Recovery time, data integrity and SLO evidence are measured.
- No check can pass by contacting an unrelated stack.

---

# 9. P4 — Capability requalification

## KAI-REM-401 — Rebuild conviction and confidence

Replace verbosity, keyword, repetition and boilerplate scoring with calibrated evidence-based confidence. Consequential actions require explicit evidence floors.

---

## KAI-REM-402 — Rebuild model registry and hardware authority

Use one signed model/artefact registry with live readiness, exact digests, token/context limits, cost and device capacity. Remove independent static registries and fail-open GPU detection.

---

## KAI-REM-403 — Requalify cognitive modules

Permanent stubs and no-op foundations must remain explicitly unavailable. Re-enable only after real implementation, provenance, bounded workloads and independent validation.

---

## KAI-REM-404 — Requalify financial and advisory logic

- Use Decimal/currency-aware models.
- Version legal/tax rules with effective dates and official sources.
- Validate date windows, future records and source evidence.
- Require human review for tax, invoice, broker and RAMS outputs.

---

## KAI-REM-405 — Requalify autonomous actions individually

Each autonomous capability needs a separate threat model, evidence contract, approval policy, sandbox, rollback/postcondition and release test.

No blanket trust level or system-wide promotion may activate multiple capabilities.

---

# 10. Cross-cutting release gates

No production or sensitive-data release until all applicable conditions are true:

- No unintended privileged host ports.
- All requests carry authenticated principal and workload identities.
- All side effects require a valid single-use exact-operation capability.
- Executor/browser/parser isolation passes adversarial tests.
- Egress is deny-by-default and SSRF-resistant.
- Memory and evidence are principal-partitioned and provenance-bound.
- Verifier rejects fabricated/duplicate/contradictory evidence.
- Fusion requires independent live sources and authoritative PASS.
- Recovery is narrow, idempotent and postcondition-verified.
- Backups pass isolated restoration and integrity verification.
- Health and release checks fail closed.
- CI uses production-equivalent secret, model and storage profiles.
- Multi-worker, restart, clock-change and concurrent-maintenance tests pass.
- Immutable audit evidence links every consequential operation end to end.

---

## 11. Programme governance

For every backlog item record:

- Accountable owner and reviewers.
- Threat model and affected audit IDs.
- Design decision record.
- Tested source, image and configuration digests.
- Negative and adversarial tests.
- Closure evidence and independent sign-off.
- Rollback plan and monitoring.

A finding may be marked resolved only when its full attack chain is broken at an authoritative boundary. Local mitigation without end-to-end enforcement remains open.

---

## 12. Final planning statement

This backlog is aligned to the final reconciled audit baseline:

- **4,580 total findings**
- **252 Critical**
- **2,440 High**
- **1,885 Medium**
- **3 Low**

The current release decision remains **NO_GO**.

No remediation was implemented by this backlog update.
