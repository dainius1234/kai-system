# Kai System — Phase 1 Security Foundation Plan

Repository: `dainius1234/kai-system`  
Authoritative audit baseline: **4,580 findings — 252 Critical, 2,440 High, 1,885 Medium, 3 Low**  
Parent backlog: `kai-pm/CODE_AUDIT_REMEDIATION_BACKLOG.md`  
Phase 0 dependency: `kai-pm/CODE_AUDIT_P0_CONTAINMENT_PLAN.md`  
Status: **IMPLEMENTATION DESIGN ONLY — NO RUNTIME REMEDIATION PERFORMED**

---

## 1. Objective

Establish the security foundation that every later remediation depends on:

1. Verified human principal identity.
2. Verified workload/service identity.
3. Explicit delegation and endpoint scope.
4. One canonical operation envelope and digest.
5. A short-lived, single-use capability for the exact approved operation.
6. Enforcement by the service that performs the final side effect.
7. Separation of operator, approval, administrative and runtime credentials.
8. Authoritative audit linkage from request through decision, execution and outcome.

Phase 1 does not rebuild the Executor sandbox, browser isolation, memory provenance or distributed transaction model. It ensures those later controls operate on a trustworthy identity and operation model.

---

## 2. Governing security invariants

The Phase 1 implementation must make the following statements true:

### INV-P1-01 — No anonymous privileged operation

Every privileged operation originates from either:

- an authenticated human principal; or
- an authenticated workload acting under an explicit, valid delegation.

A JSON field such as `user_id`, `actor_did`, `requester`, `keeper`, `dainius`, `role` or `session_id` is data, not proof of identity.

### INV-P1-02 — Workload identity is not shared

Every service has a distinct, independently revocable identity. Compromise of a low-purpose service must not permit it to impersonate Agentic, Dashboard, Tool Gate, an operator or another service.

### INV-P1-03 — Approval binds the exact operation

Approval covers one immutable operation digest containing all consequential fields. Changing the tool, target, parameters, resource, principal, policy revision, consequence budget, audience or expiry produces a different digest and invalidates the approval.

### INV-P1-04 — Policy is enforced at the effect boundary

A Tool Gate decision is not advisory. The final side-effect endpoint must reject the request unless it receives and atomically consumes a valid capability for the exact operation.

### INV-P1-05 — Human approval is a human decision object

Operator approval is produced by strong operator authentication and records the reviewed operation digest. A service token, caller-supplied `cosign: true`, role string or generic bearer token cannot represent human approval.

### INV-P1-06 — Capabilities are one-use and audience-bound

A capability is valid only for:

- one operation digest;
- one principal/delegation chain;
- one final executor/audience;
- one policy revision;
- one bounded time window;
- one consequence/resource budget;
- one atomic consumption.

### INV-P1-07 — Audit uses the same operation identity

Ingress, delegation, Gate decision, approval, capability issue, capability consumption, side effect and verified outcome all record the same operation ID and digest.

### INV-P1-08 — Security configuration fails closed

Missing identity material, unknown key IDs, invalid configuration, unavailable revocation state, failed audit append or uncertain capability state makes the consequential endpoint unavailable. It must not activate a development secret, wildcard scope or permissive fallback.

---

## 3. Confirmed source conditions driving Phase 1

### 3.1 Shared authentication helper

Primary file:

- `common/auth.py`

Confirmed conditions:

- One shared HMAC secret signs for every actor, service, session and tool.
- The signed value is a pipe-delimited string of `actor_did|session_id|tool|nonce|int(ts)`.
- Operation parameters, endpoint, audience, HTTP method, policy revision and consequence limits are absent.
- Key ID is a mutable prefix, not cryptographically bound to the digest.
- Strict key-ID validation defaults off.
- A valid signature can be relabelled with another unrevoked key ID under non-strict verification.
- Secret loading can fall back to the known development secret.
- Verification returns only Boolean and provides no accepted identity, key revision or typed failure.
- Keyring and revocation configuration are reread from mutable environment/files without one validated atomic snapshot.

Primary audit mapping:

- `KAI-COMAUTH-001` through `KAI-COMAUTH-032`.
- Original `KAI-AUTH-*` findings.
- HMAC/key-rotation batches.

### 3.2 Tool Gate

Primary file:

- `tool-gate/app.py`

Confirmed conditions:

- The trusted bearer token is placed in `GateRequest.session_id`.
- `actor_did` is caller supplied and not bound to the token/service.
- The HMAC does not cover `params`, `conviction`, rationale, source, trace, idempotency or destination.
- Idempotency lookup occurs before request authentication and canonical validation.
- Any trusted token can change mode.
- Any trusted token can act as human co-signer.
- Any trusted token can read complete ledger payloads.
- `cosign: true` in a request can be treated as upstream operator approval.
- A later co-sign creates no executable capability for the original request.
- Mode, pending approval, nonce and fallback idempotency state are process-local.
- Ledger success can be returned when persistence fails.
- Recovery and several audit/status routes lack appropriate authentication and scopes.

Primary audit mapping:

- `KAI-GATEX-001` through `KAI-GATEX-068`.
- `KAI-ARCH-001`, `KAI-ARCH-002`, `KAI-ARCH-003`, `KAI-ARCH-007`.

### 3.3 Agentic control plane

Primary file:

- `agentic/app.py`

Confirmed conditions:

- Inbound API requests have no principal or workload authentication.
- `GraphRequest` accepts caller-supplied `session_id` and `task_hint`.
- Agentic signs outbound Gate requests using the shared secret.
- Anonymous caller intent can be transformed into a signed request attributed to the trusted Agentic identity.
- The signature does not bind the generated plan parameters or conviction.
- Low conviction and adversary block recommendations do not reliably prevent Gate submission.
- Identity, trust, recovery, vault, memory, finance and runtime-control routes are privileged but open.
- Hard-coded/global identities such as `keeper` and `dainius` are used for data and audit attribution.

Primary audit mapping:

- `KAI-AGAPI-001` through `KAI-AGAPI-072`, especially `001`–`016` and `044`–`049`.

### 3.4 Dashboard privileged proxy

Primary file:

- `dashboard/app.py`

Confirmed conditions:

- The Dashboard has no inbound authenticated principal model.
- It holds `DASHBOARD_GATE_TOKEN`, a reusable Tool Gate token.
- Anonymous `/api/mode` callers can cause Dashboard to use that token for a mode change.
- The proxy returns local/synchronisation-shaped success even where authoritative policy did not change.
- Generic proxy helpers retry POSTs without one operation identity/capability contract.
- Dashboard aggregates many internal data/control services behind one browser origin.

Primary audit mapping:

- Dashboard Gateway and Frontend batches.
- `KAI-CHAIN-001`, `KAI-CHAIN-002`, `KAI-ARCH-007`.

### 3.5 Executor final side-effect boundary

Primary file:

- `executor/app.py`

Confirmed conditions:

- `ExecutionRequest` contains only `tool`, `params`, `task_id` and `device`.
- `/execute` performs no caller authentication.
- It does not verify a Tool Gate decision, policy version, ledger proof, operation digest or one-time capability.
- `policy_context` is self-constructed response metadata rather than evidence of authorisation.
- Duplicate task IDs execute repeatedly.
- Execution history and recovery are not principal/capability scoped.

Primary audit mapping:

- `KAI-EXEC-001`, `KAI-EXEC-002`, `KAI-EXEC-024`, `KAI-EXEC-037`, `KAI-EXEC-044`, `KAI-EXEC-048`, `KAI-EXEC-064`, `KAI-EXEC-070`, `KAI-EXEC-071`.

### 3.6 Other final side-effect services

The same enforcement gap exists across browser, Web Scout, Monitor, Vault/files, memory, graph, identity, trust, finance, broker, email, notifications, TTS, recovery and sensitive sensor acquisition.

Phase 1 therefore requires a complete side-effect inventory and migration, not only Tool Gate and Executor changes.

---

# 4. Target identity architecture

## 4.1 Human principal identity

Use a standard identity provider and authenticated session rather than application-defined identity strings.

### Required attributes

Every authenticated human session must expose a validated security context containing:

- `principal_id` — immutable unique identity.
- `tenant_id` — explicit tenant/household/organisation boundary.
- `session_id` — server-issued session identifier, never a bearer secret in business JSON.
- `device_id` — registered device identity where applicable.
- `assurance_level` — authentication strength/step-up state.
- `authenticated_at` and `expires_at`.
- `auth_method` — for example passkey/WebAuthn, OIDC or local hardware-backed credential.
- roles and scopes from the identity authority, not from the request body.

### Required controls

- Strong authentication for ordinary privileged use.
- Step-up authentication for policy administration, high-consequence approval, secret rotation and recovery.
- Session revocation and logout propagation.
- CSRF protection where browser cookies are used.
- No authentication token in URL, logs, localStorage or operation payload.
- Short session lifetime for administrative interfaces.

### Source migration

- Dashboard must authenticate the user before serving privileged APIs or SSE.
- Agentic must derive principal/session from authenticated middleware.
- Remove default/hard-coded authority fields from request models.
- Memory, episode, finance, trust and vault requests receive principal context internally after authentication.

---

## 4.2 Workload identity

Use mTLS or equivalent cryptographically authenticated workload identity.

Preferred architecture:

- SPIFFE-compatible workload IDs or an equivalent service-mesh identity.
- Unique identity per service and environment.
- Short-lived automatically rotated certificates.
- Peer identity derived from the authenticated transport.
- Default-deny service authorisation policies.

Example workload identities:

```text
spiffe://kai.local/prod/tool-gate
spiffe://kai.local/prod/agentic
spiffe://kai.local/prod/executor
spiffe://kai.local/prod/dashboard
spiffe://kai.local/prod/memu-core
```

### Required properties

- No shared interservice identity.
- Independent revocation and rotation.
- Environment binding; a development identity cannot authenticate to production.
- Destination validates the workload identity directly.
- Workload identity is included in delegation and audit records.
- Application bearer tokens are not used as a substitute for authenticated transport identity.

### Transitional rule

The current shared HMAC may remain only inside an isolated compatibility bridge during migration. It must not be accepted by newly migrated endpoints and must have a fixed removal date. No new service may adopt it.

---

## 4.3 Delegation authority

A workload may act for a human principal only when a delegation record explicitly grants that authority.

### Delegation record

Required fields:

```text
delegation_id
principal_id
tenant_id
source_workload_id
audience_workload_id
allowed_operation_types
allowed_resource_patterns
allowed_data_classes
purpose
max_consequence_class
budget
not_before
expires_at
policy_version
revocation_revision
parent_delegation_id
issued_by
signature
```

### Rules

- No implicit delegation from network location or service name.
- Dashboard cannot delegate administrative authority merely because it holds a server secret.
- Agentic cannot convert anonymous text into a trusted tool request.
- Delegation must be narrower than the delegator’s own authority.
- Destination independently validates delegation chain and audience.
- Revocation must be checked at capability issue and consumption.
- Delegation cannot grant human approval authority to a workload.

---

# 5. Canonical operation model

## 5.1 Operation envelope

Introduce one shared versioned schema, implemented in a new security package such as:

- `common/security/operation.py`
- `common/security/identity.py`
- `common/security/capability.py`
- `common/security/authorisation.py`

Suggested canonical envelope:

```json
{
  "schema": "kai.operation.v1",
  "operation_id": "uuid-or-uuidv7",
  "principal": {
    "principal_id": "...",
    "tenant_id": "...",
    "session_id": "...",
    "assurance_level": "..."
  },
  "delegation_chain": ["delegation-id-1", "delegation-id-2"],
  "requesting_workload": "spiffe://.../agentic",
  "audience": "spiffe://.../executor",
  "operation_type": "executor.fixed_operation",
  "target": {
    "service": "executor",
    "resource": "..."
  },
  "parameters": {},
  "data_classes": [],
  "purpose": "...",
  "evidence_refs": [],
  "policy": {
    "policy_id": "...",
    "policy_version": "..."
  },
  "consequence": {
    "class": "low|moderate|high|irreversible",
    "limits": {}
  },
  "idempotency_key": "...",
  "nonce": "...",
  "issued_at": "RFC3339 UTC",
  "not_before": "RFC3339 UTC",
  "expires_at": "RFC3339 UTC"
}
```

### Prohibited fields

- Raw authentication credentials.
- HMAC signatures from prior protocols.
- Caller-supplied actor labels treated as identity.
- Unbounded arbitrary metadata.
- Secrets, private prompts or complete file contents unless the operation explicitly requires and classifies them.

---

## 5.2 Canonical serialisation

Do not implement another custom delimiter format.

Required options:

- RFC 8785 JSON Canonicalisation Scheme; or
- deterministic CBOR with a fixed schema.

One format must be selected for the whole system.

### Canonicalisation requirements

- Exact UTF-8 and Unicode normalisation rules.
- Explicit numeric restrictions; reject NaN and infinity.
- No duplicate object keys.
- Fixed timestamp grammar and UTC representation.
- Fixed field names and schema version.
- Maximum byte, depth, item and string limits.
- Unknown security-relevant fields rejected for the current schema version.
- Test vectors committed for every supported language/runtime.

---

## 5.3 Operation digest

Compute:

```text
operation_digest = SHA-256(canonical_operation_bytes)
```

The exact algorithm and schema version must be included in the signed capability/protected header.

The same digest must appear in:

- ingress audit event;
- delegation evaluation;
- Tool Gate decision;
- operator approval;
- capability;
- executor/action-service request;
- idempotency state;
- effect/outcome record;
- incident/audit search.

Changing any consequential field creates a different digest.

---

# 6. Capability model

## 6.1 Capability contents

Tool Gate issues a signed capability only after identity, delegation, policy and approval checks pass.

Required claims:

```text
capability_id
schema/version
operation_id
operation_digest
principal_id
tenant_id
requesting_workload
audience/final_executor
operation_type
resource/target digest
policy_id and policy_version
approval_record_id when required
consequence class and limits
issued_at
not_before
expires_at
one_use = true
issuer identity
key_id and algorithm
```

### Cryptographic rule

Use a service-bound asymmetric signing key or a standard authenticated token format with asymmetric signatures. The final service must verify the Tool Gate issuer independently.

The key ID, algorithm, issuer, audience, schema and operation digest must be inside the signed/protected data.

Do not use the current shared HMAC as the long-term capability signature.

---

## 6.2 Capability consumption

The final service must atomically:

1. Verify authenticated caller workload.
2. Parse and validate capability schema.
3. Verify issuer signature and key status.
4. Check audience equals this final service.
5. Recompute the operation digest from the received operation.
6. Compare exact operation digest.
7. Validate principal/delegation/policy/approval references.
8. Check time window and consequence limits.
9. Atomically consume `capability_id` using a transactional shared store.
10. Execute only after successful consumption.
11. Record the outcome against the same operation/capability.

Replay, parallel use and consumption after expiry must fail.

### Consumption store

Use a transactional authority, preferably Postgres with unique constraints and explicit states:

```text
ISSUED -> CONSUMING -> CONSUMED
                \-> FAILED_BEFORE_EFFECT
```

Later P3 work may extend this into a full saga. Phase 1 must at least prevent duplicate use and ambiguous execution authority.

Redis-only best-effort state or process-local dictionaries are not sufficient for consequential capabilities.

---

## 6.3 Decision versus capability

Keep these concepts separate:

- **GateDecision** — policy assessment and reason.
- **ApprovalRecord** — authenticated human decision for an exact digest.
- **ExecutionCapability** — short-lived one-use authority for a final service.
- **ExecutionOutcome** — independently recorded effect/result.

A positive decision does not itself authorise execution. Only the valid capability does.

A co-sign must result in a new capability for the immutable pending operation; it must not be inferred from a second ledger entry.

---

# 7. Operator approval and administration

## 7.1 Operator approval flow

For an operation requiring human approval:

1. Tool Gate stores the immutable pending operation and digest.
2. Operator UI authenticates with step-up assurance.
3. UI displays principal, source workload, exact target, parameters, data, consequence, limits and expiry.
4. Operator approves or denies that exact digest.
5. Approval record is signed/attested by the operator-authentication authority.
6. Tool Gate verifies the approval and issues a one-use capability.
7. Final service consumes the capability.
8. Outcome is shown against the same operation digest.

### Prohibited approval representations

- `cosign: true` in caller JSON.
- A generic trusted service token.
- A caller-supplied `role: operator` or actor name.
- Browser localStorage state.
- A reason string without authenticated decision metadata.
- Prior approval for a similar operation.

---

## 7.2 Administrative authority

Separate scopes and credentials for:

- Policy revision publication.
- Gate mode/state administration.
- Tool/capability registry changes.
- Secret/key rotation.
- Identity/delegation administration.
- Audit access.
- Recovery authority.
- Ordinary runtime calls.

No runtime workload credential may perform operator approval, policy administration or ledger-secret access.

Tool Gate `/gate/mode`, `/gate/cosign`, ledger reads and recovery must migrate to distinct authenticated administrative/approval APIs.

---

# 8. Side-effect enforcement inventory

Create a machine-readable inventory before migration begins.

Suggested file:

- `security/side_effect_registry.yaml`

Required fields per route:

```text
service
method
path
operation_type
side_effect_class
resource_type
data_classes
required_principal_scope
required_workload_scope
capability_audience
human_approval_rule
idempotency_rule
outcome_verification
legacy_status
owner
migration_PR
```

### Mandatory initial inventory

#### Execution and external actions

- Executor `/execute` and recovery.
- Browser Agent navigation, clicks, forms, downloads and script execution.
- Web Scout/HTTP fetch/search.
- Monitor actions.
- Shell/script utilities and remote Docker/Git actions.

#### Data mutation and access

- memU memory/preference/correction/feedback writes.
- Memory deletion/compression/graph mutation.
- Vault/file import/export/search/delete.
- Backup creation, restore and deletion.
- SOUL/AGENTS and skill/model registry mutation.

#### Governance and authority

- Tool Gate mode/policy/cosign/autonomy.
- Trust promotion/demotion.
- Conscience, values, loyalty and operator-model mutation.
- Checkpoint restore/delete.
- Recovery, breaker reset and service restart.

#### Communications and finance

- Email send/forward/reply.
- Notification, Telegram and TTS delivery.
- Broker/paper-trader order and position mutation.
- Financial-record writes and autonomous strategy actions.

#### Sensitive acquisition

- Camera capture.
- Screen capture/watch.
- Clipboard read/write.
- Audio recording/transcription.
- Vision and biometric/emotional analysis.

No route may remain undocumented because it appears “internal” or low risk.

---

# 9. Ordered implementation PRs

## P1-PR-01 — Security contracts and threat-model freeze

**Backlog mapping:** KAI-REM-101 to KAI-REM-107  
**Runtime effect:** none initially

### Deliverables

- Architecture decision record for human identity, workload identity and capability format.
- Versioned operation schema.
- Canonicalisation specification.
- Capability and approval schemas.
- Side-effect registry schema.
- Threat models for Dashboard → Agentic → Tool Gate → Executor and direct service calls.
- Test vectors for operation digest and signature verification.

### Acceptance

- Security architecture, service owners and test owners sign off one model.
- No competing legacy/new canonical formats are introduced.
- All consequential fields are identified before implementation.

---

## P1-PR-02 — Immutable identity/keyring runtime

**Backlog mapping:** KAI-REM-102, KAI-REM-107  
**Primary files:** replace/supersede `common/auth.py`

### Required changes

- Introduce validated immutable identity/keyring configuration loaded at startup.
- Remove dev-secret fallback from privileged runtime.
- Remove non-strict key-ID behaviour.
- Bind algorithm, key ID, issuer, audience and protocol version.
- Return typed verification result with accepted workload/key/config revision and failure reason.
- Verify key ownership/source/permissions and activate atomically.
- Add explicit not-before, expiry and revocation semantics.
- Preserve legacy helper only in a named compatibility module with warnings and denied production readiness.

### Acceptance

- Key relabelling and delimiter-collision tests fail.
- Unknown/revoked/expired keys fail closed.
- Low-purpose workload cannot sign as another workload.
- Configuration reload is atomic and auditable.
- No known development secret is accepted in protected profiles.

---

## P1-PR-03 — Human principal authentication at ingress

**Backlog mapping:** KAI-REM-101  
**Primary files:** Dashboard, Agentic ingress, approved edge gateway

### Required changes

- Add authenticated browser/operator session.
- Add principal context middleware/dependency.
- Remove principal authority from body/query parameters.
- Require CSRF protection and session assurance checks.
- Restrict SSE/WebSocket/streaming channels to authenticated principal.
- Add logout/revocation and sensitive-route reauthentication.

### Initial protected surfaces

- Dashboard all APIs.
- Agentic chat/run/control APIs.
- Identity, trust, memory, finance, vault and recovery routes.

### Acceptance

- Anonymous access fails.
- Body-supplied `keeper`, `dainius`, `actor_did`, `user_id` or role cannot impersonate.
- Session A cannot access Session/Principal B data.
- Revoked session cannot continue SSE or privileged calls.

---

## P1-PR-04 — Workload mTLS identity and service policy

**Backlog mapping:** KAI-REM-102  
**Primary files:** deployment, ingress/service mesh, service middleware

### Required changes

- Issue unique workload identities.
- Authenticate internal transport.
- Define default-deny service-to-service policy.
- Expose peer identity to application authorisation.
- Remove network location/static IP as identity.
- Add identity/certificate rotation and revocation tests.

### Acceptance

- Unknown service certificate rejected.
- Agentic cannot impersonate Dashboard/operator.
- Dashboard cannot directly administer Tool Gate using an ordinary runtime identity.
- Executor accepts calls only from explicitly authorised workloads, before capability enforcement is enabled.

---

## P1-PR-05 — Canonical operation library

**Backlog mapping:** KAI-REM-104  
**Primary files:** new shared security package

### Required changes

- Implement versioned OperationEnvelope.
- Implement canonical serialisation and digest.
- Enforce byte/depth/cardinality/timestamp/numeric limits.
- Produce cross-service conformance fixtures.
- Integrate structured audit event fields.

### Acceptance

- Same operation produces identical bytes/digest across services.
- Any parameter, principal, audience, expiry, policy or consequence change changes digest.
- Ambiguous delimiters, duplicate keys, non-finite numbers and Unicode edge cases are rejected.
- Operation IDs and idempotency keys cannot be caller-confused across principals/audiences.

---

## P1-PR-06 — Delegation authority

**Backlog mapping:** KAI-REM-103  
**Primary files:** identity/security service and shared validation middleware

### Required changes

- Implement signed, revocable delegation records.
- Bind principal, source workload, audience, purpose, operation/resource scopes and consequence budget.
- Add parent-chain validation and narrowing.
- Add short lifetimes and revocation revision.

### Acceptance

- Anonymous Agentic caller cannot obtain a principal delegation.
- Dashboard runtime identity cannot delegate policy-administration scope.
- Delegation cannot increase authority or cross tenant/principal.
- Revocation blocks new capability issue immediately.

---

## P1-PR-07 — Rebuild Tool Gate decision API

**Backlog mapping:** KAI-REM-105, KAI-REM-107  
**Primary file:** `tool-gate/app.py`

### Required changes

- Replace body token/session authentication with verified workload/principal/delegation context.
- Accept canonical operation envelope, not independent caller assertions.
- Authenticate before idempotency/policy disclosure.
- Bind policy decision to operation digest.
- Separate runtime decision, operator approval, administration and audit-reader APIs.
- Remove request-body `cosign` authority.
- Store pending operations durably by operation digest.
- Fail closed if decision/audit persistence fails.
- Remove wildcard token authority and legacy mode administration from runtime credentials.

### Acceptance

- One workload cannot assert another actor/principal.
- Parameters and conviction/evidence changes invalidate the operation.
- Generic runtime identity cannot mode-change, co-sign or read sensitive audit.
- Pending approval survives restart and cannot be approved for a modified request.
- Decision response contains no reusable execution authority unless capability issue succeeds.

---

## P1-PR-08 — Operator approval service/UI

**Backlog mapping:** KAI-REM-107  
**Primary files:** protected operator UI and approval authority

### Required changes

- Strong step-up operator authentication.
- Exact operation preview.
- Approve/deny immutable digest.
- Signed approval record with assurance level and expiry.
- No server-held generic operator bearer token in Dashboard.
- Approval and mode/policy administration separated.

### Acceptance

- Approval for operation A cannot approve operation B.
- Approval replay fails.
- Runtime/service credential cannot approve.
- Operator sees exact target, parameters, data class and consequence.
- Denial is durable and cannot be bypassed through alternate Tool Gate route.

---

## P1-PR-09 — Capability issuance and atomic consumption authority

**Backlog mapping:** KAI-REM-105  
**Primary files:** Tool Gate, shared capability verifier, transactional store

### Required changes

- Issue asymmetric signed capabilities.
- Bind issuer, audience, operation digest, principal/delegation, policy, approval, expiry and limits.
- Introduce transactional capability state.
- Add revocation/key rotation.
- Add outcome linkage.

### Acceptance

- Parallel consumption produces one success.
- Replay, expired, wrong-audience and modified-operation cases fail.
- Compromised low-purpose service cannot forge Tool Gate capability.
- Capability issue fails if authoritative audit/transaction fails.

---

## P1-PR-10 — Executor pilot enforcement

**Backlog mapping:** KAI-REM-106  
**Primary file:** `executor/app.py`

### Required changes

Extend `ExecutionRequest` or replace it with:

- canonical operation envelope;
- signed capability;
- no caller-defined policy metadata.

Before any subprocess or state mutation:

- authenticate workload;
- verify/consume capability;
- recompute exact operation digest;
- validate audience and consequence limits;
- create durable execution-start record.

Remove public unauthenticated history/recovery or protect with separate capabilities/scopes.

### Acceptance

- Direct `/execute` without capability fails before malware scan/state mutation.
- Gate denial cannot be bypassed.
- Modified `params`, `tool`, task/resource or device/consequence field fails.
- Duplicate capability/task executes once.
- Execution result references exact operation and capability IDs.

This PR does not declare the current command model safe. P2 must still replace arbitrary-code primitives and add real sandboxing.

---

## P1-PR-11 — Dashboard confused-deputy removal

**Backlog mapping:** KAI-REM-103, KAI-REM-107  
**Primary files:** `dashboard/app.py`, `dashboard/static/app.html`

### Required changes

- Remove `DASHBOARD_GATE_TOKEN` as a reusable administrative authority.
- Dashboard forwards authenticated principal context/delegation, not its own operator credential.
- Mode/policy changes use the protected administrative workflow.
- Generic proxy endpoints may not elevate the caller’s authority.
- Bind mutations to operation IDs and capabilities.
- Remove local/browser mode as authority.

### Acceptance

- XSS or anonymous caller cannot borrow server-held Gate authority.
- Dashboard runtime compromise does not grant operator approval or mode administration.
- Proxy retry cannot duplicate a consumed capability/mutation.

---

## P1-PR-12 — Agentic caller/delegation migration

**Backlog mapping:** KAI-REM-101, KAI-REM-103, KAI-REM-104  
**Primary file:** `agentic/app.py`

### Required changes

- Require authenticated principal/workload at ingress.
- Remove caller-supplied session/tool identity as authority.
- Server derives `operation_type` from an allowlisted plan/action registry, not `task_hint` directly.
- Build one canonical operation containing exact plan parameters, evidence and consequence.
- Request Gate decision under explicit delegation.
- Do not record blocked/unavailable decisions as successful outcomes.
- Do not store global `keeper` records; attach principal/purpose scope.

### Acceptance

- Anonymous input cannot produce a signed Gate operation.
- `task_hint` cannot select arbitrary Tool Gate identity.
- Low conviction/binding policy failure cannot reach capability issue.
- Gate operation digest includes the exact actionable plan.
- Outcome state distinguishes denied, unavailable, not executed and verified success.

---

## P1-PR-13 — Side-effect service migration wave 1

**Backlog mapping:** KAI-REM-106

Migrate the highest-risk services after Executor:

1. Browser Agent/Web Scout/Monitor.
2. Vault/files/backup.
3. memU identity/preference/feedback/trust mutations.
4. Recovery/checkpoint/breaker controls.
5. Broker/financial mutation.
6. Email/notification/TTS external delivery.
7. Sensitive camera/screen/clipboard/audio acquisition.

### Acceptance

- Every registered side-effect endpoint enforces a capability.
- Alternate/legacy/internal routes are disabled or equivalently enforced.
- Negative route inventory tests remain complete.

---

## P1-PR-14 — Legacy authentication removal

**Backlog mapping:** KAI-REM-102, KAI-REM-107

### Required changes

- Remove shared HMAC acceptance from migrated production profiles.
- Remove trusted-token files/body tokens.
- Remove `HMAC_ALLOW_DEV_SECRET` from assurance workflows.
- Remove `DASHBOARD_GATE_TOKEN`.
- Remove request-body `cosign` authority.
- Remove wildcard implicit scopes.
- Delete compatibility endpoints after migration window.

### Acceptance

- Repository search and runtime tests show no protected service accepts the legacy protocol.
- Legacy credential cannot call migrated endpoint.
- Deployment fails when compatibility mode is enabled in a release profile.

---

## P1-PR-15 — Security CI and release-gate assurance

**Backlog mapping:** supports all P1 items  
**Primary files:** `.github/workflows/`, per-service tests, release/go-no-go scripts

### Required gates

- Separate per-service dependency environments.
- No global development HMAC mode.
- Canonical operation cross-language vectors.
- Key relabelling, delimiter collision and unknown-key tests.
- Anonymous, wrong-principal, wrong-workload and wrong-scope tests.
- Wrong-audience, changed-parameter, expired and replayed capability tests.
- Parallel capability-consumption race test.
- Dashboard confused-deputy test.
- Agentic anonymous-input-to-Gate negative test.
- Executor direct-bypass negative test.
- Full side-effect registry coverage check.
- Built image digest linked to test evidence.
- Signed consolidated assurance result.

### Release rule

A green unit-test suite or shallow health check cannot satisfy P1. Release remains NO_GO unless the integrated identity → delegation → Gate → capability → final effect path passes with production-equivalent security configuration and no dev secret/stub authority.

---

# 10. Migration compatibility and sequencing

## 10.1 Dual-path risks

Running legacy and new authorisation paths simultaneously creates a bypass. During migration:

- a route must reject requests unless exactly one explicitly configured protocol is active;
- production-equivalent profiles must prefer and require the new protocol;
- compatibility endpoints must be isolated and unavailable through ordinary ingress;
- Gate decisions issued through legacy protocol must not be accepted by new capability consumers;
- metrics must count all legacy use and fail the release gate when nonzero.

## 10.2 Service migration state

Use explicit states:

- `UNMIGRATED_DISABLED`
- `LEGACY_ISOLATED_TEST_ONLY`
- `IDENTITY_ONLY`
- `CAPABILITY_ENFORCED`
- `LEGACY_REMOVED`
- `VERIFIED`

Only `VERIFIED` satisfies the P1 release gate.

## 10.3 Rollback rule

Rollback may disable a capability/service. It must not restore an unauthenticated or advisory-only side-effect path.

---

# 11. P1 adversarial closure tests

## Test P1-A — Shared-secret impersonation

Compromise a low-purpose service identity and attempt to act as Agentic, Dashboard, Tool Gate or operator.

**Pass:** every impersonation fails; audit records the actual workload.

## Test P1-B — Operation mutation

Obtain approval/capability, then alter one field at a time:

- parameter;
- target/resource;
- tool/operation type;
- principal;
- audience;
- policy version;
- data classification;
- consequence limit;
- expiry/idempotency.

**Pass:** every modified request fails before effect.

## Test P1-C — Replay and race

Submit one capability concurrently to multiple workers and replay after completion/restart.

**Pass:** exactly one transactional consumption; no duplicate effect.

## Test P1-D — Dashboard confused deputy

Use anonymous caller, low-privilege principal and injected browser script to attempt mode change, approval and privileged proxy calls.

**Pass:** no server-held authority can be borrowed; operator step-up required.

## Test P1-E — Anonymous Agentic escalation

Submit `/run`, chat and task hints without principal identity and attempt to generate Gate/capability activity.

**Pass:** no operation or capability is issued; no privileged memory/trust mutation occurs.

## Test P1-F — Direct Executor bypass

Call Executor directly with valid-looking tool/params but no or altered capability.

**Pass:** rejection before state push, scan, subprocess or output/audit side effect.

## Test P1-G — Co-sign substitution

Approve operation A and try to use the approval for operation B or after expiry/revocation.

**Pass:** rejected; approval is exact-digest and one-operation scoped.

## Test P1-H — Key lifecycle

Test unknown ID, relabelled signature, revoked key, expired key, future key, mixed configuration revision and failed key source.

**Pass:** fail closed with typed reason and audit; no fallback.

## Test P1-I — Side-effect registry completeness

Discover routes statically and at runtime; compare with `side_effect_registry.yaml`.

**Pass:** no unregistered mutating/sensitive-acquisition route; CI fails on drift.

## Test P1-J — Audit chain continuity

Trace one denied, one approved-not-executed, one successfully executed and one failed-after-consumption operation.

**Pass:** exact operation digest and identities are present end to end with no success-shaped ambiguity.

---

# 12. P1 exit criteria

Phase 1 is complete only when all are true:

- Human principals are strongly authenticated.
- Workloads have unique authenticated identities.
- No body field or hard-coded name grants identity/authority.
- Delegation is explicit, audience-bound, revocable and narrow.
- One canonical operation schema/digest is used across all protected services.
- Tool Gate decisions bind the exact operation.
- Human approvals bind the exact digest and require step-up identity.
- Tool Gate issues short-lived, one-use, asymmetric signed capabilities.
- Final side-effect endpoints atomically consume capabilities.
- Executor direct bypass is impossible.
- Dashboard cannot borrow administrative/operator authority.
- Agentic cannot transform anonymous intent into trusted internal authority.
- Legacy shared-HMAC/body-token/cosign paths are removed from protected profiles.
- The full side-effect registry is capability-enforced.
- Integrated adversarial tests pass against production-equivalent configuration.
- Audit evidence links principal, workload, delegation, digest, decision, approval, consumption and outcome.

Passing Phase 1 permits privileged internal testing only after Phase 0 remains effective. It does not authorise sensitive-data use, arbitrary execution, broad browser egress, autonomous finance or production deployment. Those remain blocked by P2–P4.

---

# 13. Immediate next implementation queue

After Phase 0 containment and evidence preservation, start in this order:

1. P1-PR-01 — freeze contracts and threat models.
2. P1-PR-02 — immutable identity/keyring runtime.
3. P1-PR-03 — human principal ingress authentication.
4. P1-PR-04 — workload mTLS identity.
5. P1-PR-05 — canonical operation library.
6. P1-PR-06 — delegation authority.
7. P1-PR-07 — rebuild Tool Gate decision API.
8. P1-PR-08 — protected operator approval.
9. P1-PR-09 — capability issue/consumption authority.
10. P1-PR-10 — Executor pilot enforcement.
11. P1-PR-11/12 — Dashboard and Agentic migration.
12. P1-PR-13 — remaining side-effect services.
13. P1-PR-14 — legacy removal.
14. P1-PR-15 — integrated assurance and release evidence.

Do not begin capability re-enablement while a legacy direct side-effect route remains reachable.

---

## Final Phase 1 planning judgement

The current system’s authentication defects cannot be repaired by adding another API key check to individual endpoints. The shared-secret design permits identity forgery; Tool Gate signs and assesses only fragments of an action; Dashboard and Agentic borrow internal authority; and final effect services do not require Tool Gate proof.

The minimum defensible correction is one principal/workload/delegation model, one canonical operation digest and one single-use capability enforced by the final service. Every later sandbox, memory, evidence, financial and autonomy control depends on this foundation.

**Current status remains NO_GO. This document implements no runtime remediation and closes no findings.**
