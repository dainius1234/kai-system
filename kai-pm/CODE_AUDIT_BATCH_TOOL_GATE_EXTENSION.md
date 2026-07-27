# Kai Code Audit — Tool Gate Policy, Co-sign and Ledger Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings beyond the existing `KAI-AUTH-001` through `KAI-AUTH-004` entries in `CODE_AUDIT_REGISTER.md` (full-request HMAC binding, pre-auth idempotency, nonce persistence and request-body co-sign assertion). Those four findings are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-GATEX-001 | CRITICAL | Any trusted token can change Gate mode regardless of its configured tool scopes |
| KAI-GATEX-002 | CRITICAL | Any trusted token can approve or deny every pending co-sign request |
| KAI-GATEX-003 | CRITICAL | Any trusted token can read complete ledger payloads containing credentials, signatures, parameters and rationale |
| KAI-GATEX-004 | CRITICAL | Unauthenticated `/recover` reloads trusted tokens and replay state |
| KAI-GATEX-005 | CRITICAL | Pending co-sign approval is not converted into a new executable GateDecision for the original request |
| KAI-GATEX-006 | CRITICAL | Ledger append acknowledges Gate decisions even when durable persistence fails |
| KAI-GATEX-007 | CRITICAL | Concurrent ledger appends can create sibling hashes and corrupt the claimed linear chain |
| KAI-GATEX-008 | CRITICAL | Ledger tampering/corruption is skipped at startup and the Gate continues operating instead of failing closed |
| KAI-GATEX-009 | HIGH | Trusted token identity is carried in the `session_id` body field rather than an authentication channel |
| KAI-GATEX-010 | HIGH | A trusted token is not cryptographically bound to `actor_did` |
| KAI-GATEX-011 | HIGH | The shared HMAC secret is not bound to one token, actor or service identity |
| KAI-GATEX-012 | HIGH | Tool allowlist and token validity are checked before signature verification, creating policy and token oracles |
| KAI-GATEX-013 | HIGH | Token comparisons use ordinary set membership rather than constant-time verification |
| KAI-GATEX-014 | HIGH | Token files are trusted without ownership, permission, symlink, signature or revision validation |
| KAI-GATEX-015 | HIGH | Token reload replaces live security state without locking or atomic snapshot activation |
| KAI-GATEX-016 | HIGH | A token line without explicit scopes receives wildcard authority |
| KAI-GATEX-017 | HIGH | Mode and co-sign endpoints ignore `TOKEN_SCOPES` entirely |
| KAI-GATEX-018 | HIGH | Ledger records persist full Gate request payloads including trusted tokens and signatures |
| KAI-GATEX-019 | HIGH | Ledger files are plaintext and lack permission hardening, encryption or external integrity anchoring |
| KAI-GATEX-020 | HIGH | Ledger replay materialises the complete unbounded file and in-memory history at startup |
| KAI-GATEX-021 | HIGH | Malformed ledger fields can crash startup because replay catches only JSON syntax errors |
| KAI-GATEX-022 | HIGH | Ledger verification and Merkle calculation scan the complete history on every call |
| KAI-GATEX-023 | HIGH | Ledger stats, verification and Merkle-root endpoints are unauthenticated |
| KAI-GATEX-024 | HIGH | The Merkle root is self-computed, unsigned and not anchored to an external trusted checkpoint |
| KAI-GATEX-025 | HIGH | Manual mode changes are bearer-token-only and lack nonce, timestamp, signature and replay protection |
| KAI-GATEX-026 | HIGH | Co-sign actions are bearer-token-only and lack nonce, timestamp, signature and replay protection |
| KAI-GATEX-027 | HIGH | Manual mode override is worker-local, restart-volatile and not restored from the ledger |
| KAI-GATEX-028 | HIGH | Scheduled mode uses the server host timezone rather than an operator-configured timezone |
| KAI-GATEX-029 | HIGH | Invalid scheduled modes receive a zero conviction offset, equivalent to permissive WORK behaviour |
| KAI-GATEX-030 | HIGH | Schedule entries, hours, days and modes are not schema/range validated |
| KAI-GATEX-031 | HIGH | Overnight schedules cannot be represented correctly by the simple start/end comparison |
| KAI-GATEX-032 | HIGH | Health reports configured `policy.mode`, not the effective scheduled/manual mode actually enforced |
| KAI-GATEX-033 | HIGH | Health reports `ok` with no trusted tokens loaded |
| KAI-GATEX-034 | HIGH | Health treats ledger-directory existence as proof of ledger writeability and integrity |
| KAI-GATEX-035 | HIGH | Recovery always reports tokens/nonces recovered even when files are absent or persistence failed |
| KAI-GATEX-036 | HIGH | Pending co-sign state is process-local and disappears on restart |
| KAI-GATEX-037 | HIGH | Multiple workers maintain different pending requests, modes, nonces, ledgers and idempotency fallbacks |
| KAI-GATEX-038 | HIGH | Co-sign removes the pending request before verifying durable approval-ledger persistence |
| KAI-GATEX-039 | HIGH | Co-sign approval does not update or evict a cached blocked idempotency decision |
| KAI-GATEX-040 | HIGH | Expired co-sign requests are cleaned only during later requests and may remain indefinitely |
| KAI-GATEX-041 | HIGH | Co-sign notification delivery blocks async Gate requests with synchronous HTTP |
| KAI-GATEX-042 | HIGH | Notification HTTP status is ignored and operator delivery is not part of pending-state semantics |
| KAI-GATEX-043 | HIGH | Hard-coded tool allowlist diverges from shared policy, model routing and deployed capabilities |
| KAI-GATEX-044 | HIGH | Default irreversible taxonomy classifies only `shell`; other destructive/public/financial tools remain ordinary |
| KAI-GATEX-045 | HIGH | Environment-controlled irreversible categories can be malformed, overlapping or semantically unsafe |
| KAI-GATEX-046 | HIGH | An explicitly empty irreversible taxonomy cannot be configured because `{}` silently restores defaults |
| KAI-GATEX-047 | HIGH | Gate parameters, rationale, trace fields and signature lists have no aggregate-size or depth limits |
| KAI-GATEX-048 | MEDIUM | Signature candidate lists are unbounded and amplify HMAC verification work |
| KAI-GATEX-049 | MEDIUM | Non-finite timestamps can trigger internal conversion failures instead of a typed rejection |
| KAI-GATEX-050 | MEDIUM | Nonce cache restore accepts arbitrary keys/timestamps without finite or range validation |
| KAI-GATEX-051 | MEDIUM | Restored stale nonces are not pruned until a later request or recovery call |
| KAI-GATEX-052 | MEDIUM | Nonce persistence does not create/validate its parent directory or file permissions |
| KAI-GATEX-053 | MEDIUM | Idempotency state is written to both Redis and process memory and can diverge silently |
| KAI-GATEX-054 | MEDIUM | Redis idempotency failures are silently ignored and downgrade to worker-local behaviour |
| KAI-GATEX-055 | MEDIUM | Client-selected idempotency keys have no length, namespace or cardinality bound |
| KAI-GATEX-056 | MEDIUM | In-memory idempotency pruning occurs only on writes, allowing stale growth during read-heavy periods |
| KAI-GATEX-057 | MEDIUM | Mode-change reasons, actor fields, rationale and trace metadata are caller assertions stored as audit evidence |
| KAI-GATEX-058 | MEDIUM | Sanitisation truncates/changes security identifiers after an unauthenticated idempotency lookup |
| KAI-GATEX-059 | MEDIUM | Token strings containing `:` cannot be represented unambiguously in the token-file format |
| KAI-GATEX-060 | MEDIUM | Token reload through a Python signal handler races request evaluation and mutable set replacement |
| KAI-GATEX-061 | MEDIUM | Ledger tail limits accept negative and arbitrarily large values |
| KAI-GATEX-062 | MEDIUM | Public metrics reveal Gate request/error behaviour without authentication |
| KAI-GATEX-063 | MEDIUM | Public mode endpoint reveals schedules and override lifetime |
| KAI-GATEX-064 | MEDIUM | Public autonomy requests allow unauthenticated ledger spam and behavioural-policy claims |
| KAI-GATEX-065 | MEDIUM | Autonomy-request success claims the ledger entry was written even when ledger persistence failed |
| KAI-GATEX-066 | MEDIUM | Audit logging is optional and records no authenticated actor or canonical request digest |
| KAI-GATEX-067 | MEDIUM | Configuration floats/integers and cross-threshold relationships are not validated at startup |
| KAI-GATEX-068 | MEDIUM | Gate service has no owned lifespan resources, graceful persistence flush or multi-process safety contract |

---

## Critical privilege and integrity findings

### KAI-GATEX-001 — CRITICAL — Token scopes do not constrain mode administration
**Issue:** `/gate/mode` checks only `token in TRUSTED_TOKENS`. It does not call `_is_tool_allowed()` or require an administrative scope.  
**Risk:** A token intended for one low-risk tool can move the entire Gate between PUB and WORK.  
**Recommendation:** define separate immutable administrative identities/scopes and require a body-bound signed operator event.  
**Status:** OPEN — immediate remediation required

### KAI-GATEX-002 — CRITICAL — Any token is an operator co-signer
**Issue:** `/gate/cosign` likewise checks only membership in the trusted-token set.  
**Risk:** Every service token becomes human operator approval for destructive, public or financial actions.  
**Recommendation:** accept co-sign only from a distinct strongly authenticated operator credential referencing the immutable original request digest.  
**Status:** OPEN — immediate remediation required

### KAI-GATEX-003 — CRITICAL — Full ledger secret disclosure
**Issue:** `/ledger/tail` permits any trusted token and returns complete payloads. Gate payloads contain `session_id` (the trusted token), signatures/signature list, nonce, params, rationale and identity metadata.  
**Risk:** One low-scope token can retrieve other credentials/signatures and sensitive action parameters, enabling lateral privilege expansion and replay analysis.  
**Recommendation:** never store credentials/signatures in the ledger; redact secrets and require dedicated audit-reader scope with field-level filtering.  
**Status:** OPEN — immediate remediation required

### KAI-GATEX-004 — CRITICAL — Open security-state recovery
**Issue:** `POST /recover` is unauthenticated and reconnects Redis, reloads trusted tokens and prunes replay state.  
**Risk:** Anonymous or Supervisor-triggered calls can activate changed token files and alter security/replay behaviour during an incident.  
**Recommendation:** protect recovery with authenticated incident authority, signed configuration revision and postcondition verification.  
**Status:** OPEN — immediate remediation required

### KAI-GATEX-005 — CRITICAL — Co-sign cannot safely authorise the original request
**Issue:** A pending GateDecision is returned as blocked/pending. Later `/gate/cosign` appends a second ledger entry but creates no replacement GateDecision, approval capability, one-time execution token or immutable request-hash grant for the original caller.  
**Risk:** Implementations may either ignore co-sign because no executable decision exists, or infer approval from loosely related ledger/state and execute the wrong request.  
**Recommendation:** issue one short-lived single-use approval bound to the exact original body digest, actor and executor.  
**Status:** OPEN — immediate remediation required

### KAI-GATEX-006 — CRITICAL — Gate success precedes durable ledger success
**Issue:** `PersistentLedger.append()` adds the entry to memory and returns it even when `_persist_entry()` catches a write failure. Gate/mode/co-sign/autonomy endpoints report success using that hash.  
**Risk:** Approved decisions can execute without the claimed immutable durable audit evidence.  
**Recommendation:** fail closed until a fsynced/transactional append succeeds and is externally anchored.  
**Status:** OPEN — immediate remediation required

### KAI-GATEX-007 — CRITICAL — Concurrent hash-chain corruption
**Issue:** `prev_hash` selection, in-memory append and file append are unsynchronised. Concurrent requests can read the same predecessor and create sibling entries.  
**Risk:** The chain becomes invalid under normal load, or one writer’s evidence is lost/interleaved across workers.  
**Recommendation:** use one transactional shared append authority with a uniqueness/CAS constraint on predecessor and sequence.  
**Status:** OPEN — immediate remediation required

### KAI-GATEX-008 — CRITICAL — Corruption fails open
**Issue:** Startup replay logs and skips invalid JSON or chain mismatches, then continues operating from the last accepted hash. It does not quarantine the file or fail readiness.  
**Risk:** Ledger tampering or disk corruption removes evidence while the Gate continues approving actions and reports a shorter valid chain.  
**Recommendation:** halt consequential approvals on any integrity gap and require audited recovery from a trusted checkpoint.  
**Status:** OPEN — immediate remediation required

---

## High-severity identity, token and ledger findings

### KAI-GATEX-009 — HIGH — Token in business body
The trusted token is supplied as `session_id` inside JSON rather than an authentication header/transport identity, increasing logging, caching and accidental-propagation exposure.

### KAI-GATEX-010 — HIGH — Actor spoofing
A valid token may assert any `actor_did`; token configuration has no actor/service binding.

### KAI-GATEX-011 — HIGH — Shared signing authority
`verify_gate_signature` uses a shared HMAC authority and does not select a key/identity based on token or actor. Any service with that secret can sign as another actor for any known trusted token.

### KAI-GATEX-012 — HIGH — Pre-signature policy/token oracle
Tool allowlisting, trusted-token membership and token scope are evaluated before signature verification, yielding distinguishable 400/401/403 responses to unauthenticated probes.

### KAI-GATEX-013 — HIGH — Non-constant-time token checks
Secrets are stored as Python strings in sets and compared by ordinary equality/membership.

### KAI-GATEX-014 — HIGH — Unverified token source
The token file may be replaced, symlinked or world-readable; ownership, mode, signature, trusted mount and revision are not checked.

### KAI-GATEX-015 — HIGH — Non-atomic token activation
Reload empties and repopulates global token/scope objects without a lock or complete temporary validation. Concurrent requests can see partial/empty state.

### KAI-GATEX-016 — HIGH — Implicit wildcard privilege
Any token line without `:` automatically receives `{"*"}`.

### KAI-GATEX-017 — HIGH — Administrative scope collapse
Mode, co-sign and ledger-read endpoints use no scopes, even though scopes are available for tool requests.

### KAI-GATEX-018 — HIGH — Ledger stores credentials and secrets
`request.model_dump()` includes session token, signatures, nonce, arbitrary params and potentially secret rationale/context.

### KAI-GATEX-019 — HIGH — Ungoverned plaintext audit storage
Ledger content is ordinary JSONL with no encryption, permission hardening, rotation, retention, secure deletion or signed checkpoint.

### KAI-GATEX-020 — HIGH — Unbounded startup replay
The complete file is read as one string, split into lines, parsed and retained as objects.

### KAI-GATEX-021 — HIGH — Malformed-field startup crash
After JSON parsing, required keys and types are accessed directly; `KeyError`, `TypeError` and malformed hash inputs are not caught per line.

### KAI-GATEX-022 — HIGH — Unbounded live verification
`ledger_verify()` and `ledger_merkle()` traverse every retained entry on demand and run synchronously on the async worker.

### KAI-GATEX-023 — HIGH — Public ledger integrity metadata
Stats, verify and Merkle endpoints expose history size, validity and roots without authentication, aiding incident/replay timing and enabling workload abuse.

### KAI-GATEX-024 — HIGH — Self-attested Merkle root
The root is computed from the same mutable local in-memory entries and has no signature, external timestamp or transparency-log anchor.

### KAI-GATEX-025 — HIGH — Replayable mode administration
Mode changes use a static bearer token only; no timestamp, nonce, body signature, CSRF/origin or one-time approval is required.

### KAI-GATEX-026 — HIGH — Replayable co-sign administration
The co-sign decision uses the same static bearer-only pattern and can be replayed/raced from any holder.

### KAI-GATEX-027 — HIGH — Volatile manual override
Override deadline and current mode are mutable process state; restart/other workers revert to schedule/environment despite a ledger entry claiming the change.

### KAI-GATEX-028 — HIGH — Host-time policy
Mode scheduling uses `datetime.now()` without an operator timezone, DST policy or authoritative clock source.

### KAI-GATEX-029 — HIGH — Invalid mode fails permissive
`_mode_conviction_offset()` returns zero for unknown modes. A malformed schedule mode therefore receives no PUB safety offset.

### KAI-GATEX-030 — HIGH — Unsafe schedule schema
Environment JSON entries may contain arbitrary modes, days/types and negative/extreme/non-numeric hours.

### KAI-GATEX-031 — HIGH — Overnight schedule failure
A rule such as 18:00–08:00 never matches because the implementation requires `start <= hour < end`.

### KAI-GATEX-032 — HIGH — Misreported enforced mode
Health returns `policy.mode`, while actual evaluation calls `_effective_mode()` and may use schedule/PUB instead.

### KAI-GATEX-033 — HIGH — No-token health remains green
`warn: no tokens` does not start with `fail`; health reports `status: ok` even though no Gate request or admin operation can authenticate.

### KAI-GATEX-034 — HIGH — False ledger health
The only ledger check is whether the parent directory exists; it does not test write/fsync, replay integrity, permissions or chain state.

### KAI-GATEX-035 — HIGH — False recovery success
Recovery appends `tokens` and `nonces` to the recovered list unconditionally after operations that may load nothing or fail persistence.

### KAI-GATEX-036 — HIGH — Pending approvals disappear
`_pending_cosign` is an in-memory dictionary and is not restored from the ledger/Redis.

### KAI-GATEX-037 — HIGH — Worker security-state divergence
Mode override, pending co-signs, nonces, ledger entries and idempotency fallback are not shared transactionally across workers.

### KAI-GATEX-038 — HIGH — Pop-before-persist co-sign loss
The pending request is removed before the approval/denial ledger append is proven durable.

### KAI-GATEX-039 — HIGH — Stale idempotency decision after approval
A previously cached pending/blocked GateDecision is not replaced or evicted when the operator later co-signs it.

### KAI-GATEX-040 — HIGH — Lazy expiry only
Pending approvals expire only when another request calls `_cleanup_pending`; dormant expired requests remain in memory and listings.

### KAI-GATEX-041 — HIGH — Blocking notification in Gate path
Policy evaluation calls synchronous `httpx.Client.post()` while serving an async request.

### KAI-GATEX-042 — HIGH — Notification result ignored
Transport success but HTTP rejection is treated as delivered; pending state does not indicate that the operator was never notified.

### KAI-GATEX-043 — HIGH — Independent hard-coded tool policy
`allowed_tools={shell,qgis,n8n,noop,speak}` is another authority disconnected from `security/policy.yml`, feature flags, executor capabilities and model routing.

### KAI-GATEX-044 — HIGH — Incomplete irreversible defaults
Only shell is destructive by default; qgis/n8n and future financial/public tools receive ordinary threshold treatment unless deployment configuration is perfect.

### KAI-GATEX-045 — HIGH — Unsafe irreversible configuration
Category/tool values are not enum/schema checked and tools may appear in multiple categories; first dictionary iteration match silently wins.

### KAI-GATEX-046 — HIGH — Empty taxonomy cannot be intentional
`json.loads(... ) or defaults` converts an explicitly empty dictionary into the default taxonomy.

### KAI-GATEX-047 — HIGH — Unbounded security payload
Params may be arbitrarily deep/large; rationale, signatures, trace/source and identifiers have no strict aggregate body limits before hashing/ledger storage.

---

## Medium-severity operational findings

### KAI-GATEX-048 — MEDIUM — Signature verification amplification
Every string in an unbounded `signatures` list triggers a full HMAC verification.

### KAI-GATEX-049 — MEDIUM — Non-finite timestamp failure
Float constraints do not explicitly require finiteness; NaN/infinity can produce comparison/conversion exceptions in timestamp/signature logic.

### KAI-GATEX-050 — MEDIUM — Unvalidated restored nonces
Nonce JSON values are converted to float but not checked for finiteness, future time or acceptable age.

### KAI-GATEX-051 — MEDIUM — Stale restore memory
Restored entries remain until another Gate/recover call invokes cleanup.

### KAI-GATEX-052 — MEDIUM — Weak nonce-file lifecycle
Parent/permissions/owner are not established or verified, and complete-file writes are non-atomic.

### KAI-GATEX-053 — MEDIUM — Dual idempotency truth
Redis and local caches are both written; one may fail, expire or contain a different value without reconciliation.

### KAI-GATEX-054 — MEDIUM — Silent Redis downgrade
All Redis get/set/delete exceptions are suppressed and worker-local semantics continue without a degraded result.

### KAI-GATEX-055 — MEDIUM — Unbounded idempotency key
Caller keys directly form Redis keys and local dictionary entries with no length/namespace/cardinality constraint.

### KAI-GATEX-056 — MEDIUM — Lazy local-cache pruning
Expired entries are pruned during `_idem_set`, not periodically or on every lookup across all keys.

### KAI-GATEX-057 — MEDIUM — Caller assertions become audit metadata
Mode reasons, actor DID, rationale, request source, trace and device are not independently verified but are stored as decision evidence.

### KAI-GATEX-058 — MEDIUM — Security identifier mutation
Generic sanitisation strips characters and truncates tool/actor/session/nonce/signature fields after the pre-auth idempotency cache check, creating inconsistent request identities.

### KAI-GATEX-059 — MEDIUM — Ambiguous token-file grammar
The first colon separates token and scopes, preventing transparent support for tokens containing colons and making malformed lines easy to misinterpret.

### KAI-GATEX-060 — MEDIUM — Reload race through signal handler
SIGHUP calls live file I/O/global-set replacement during normal request processing without synchronisation.

### KAI-GATEX-061 — MEDIUM — Unsafe ledger tail limit
Negative values use Python’s surprising negative slicing; huge values return complete secret-bearing history.

### KAI-GATEX-062 — MEDIUM — Public metrics
Error-budget state is exposed without authentication and measures HTTP statuses rather than security decision quality.

### KAI-GATEX-063 — MEDIUM — Public schedule/override disclosure
`GET /gate/mode` returns the complete mode schedule and remaining manual override.

### KAI-GATEX-064 — MEDIUM — Anonymous autonomy-request spam
The feature-flagged endpoint requires no token/signature and writes one ledger record per request.

### KAI-GATEX-065 — MEDIUM — False autonomy ledger acknowledgement
The response always says `ledger_entry: written`, even when `_persist_entry()` logged a write failure.

### KAI-GATEX-066 — MEDIUM — Weak optional audit
`AUDIT_REQUIRED` defaults false; method/path/status logs omit actor, token scope, body digest, decision ID and operation revision.

### KAI-GATEX-067 — MEDIUM — Unvalidated configuration relationships
TTL/skew, conviction thresholds/offsets, irreversible floor, idempotency TTL and schedule configuration can be negative, non-finite, impossible or internally contradictory.

### KAI-GATEX-068 — MEDIUM — Missing lifecycle/multi-process contract
There is no lifespan-owned Redis/client/persistence resource, graceful flush, ledger lock, shared leader or supported-worker validation.

---

## Batch totals

- Findings: **68**
- Critical: **8**
- High: **39**
- Medium: **21**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,792**
- Critical: **156**
- High: **864**
- Medium: **769**
- Low: **3**

## Files materially reviewed

`tool-gate/app.py`, with existing authentication findings reconciled against `common/auth.py`, `common/rate_limit.py`, `common/runtime.py`, `security/policy.yml` and Dashboard/Supervisor integration paths.
