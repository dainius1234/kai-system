# Kai Code Audit — Agentic Introspection and Self-Audit Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-INTROSPECT-001 | CRITICAL | The host-published introspection/self-improvement service has no authentication or authorisation |
| KAI-INTROSPECT-002 | HIGH | The service exposes no inbound service-identity verification despite mutating core state and memory |
| KAI-INTROSPECT-003 | HIGH | Dream consolidation can be triggered without a trust or feature gate |
| KAI-INTROSPECT-004 | HIGH | Evolution analysis can be triggered without a trust or feature gate |
| KAI-INTROSPECT-005 | HIGH | Security self-audit can be triggered and read without authentication |
| KAI-INTROSPECT-006 | HIGH | No rate, concurrency or duplicate-cycle control protects expensive endpoints |
| KAI-INTROSPECT-007 | HIGH | Synchronous episode retrieval and analysis run directly inside async handlers |
| KAI-INTROSPECT-008 | HIGH | Concurrent dream/evolution cycles race shared report and spool files |
| KAI-INTROSPECT-009 | HIGH | All self-improvement analysis is hard-coded to the global `keeper` identity |
| KAI-INTROSPECT-010 | HIGH | Poisoned and self-scored episodes are treated as authoritative improvement evidence |
| KAI-INTROSPECT-011 | HIGH | Dream insights are written into durable memory without operator or verifier approval |
| KAI-INTROSPECT-012 | HIGH | Self-generated insight confidence becomes durable relevance and importance metadata |
| KAI-INTROSPECT-013 | HIGH | Any HTTP response from memory storage is counted as a successful write |
| KAI-INTROSPECT-014 | HIGH | Each insight/suggestion creates a separate HTTP client and sequential request |
| KAI-INTROSPECT-015 | HIGH | Post-dream checkpoint creation ignores HTTP status and returned checkpoint identity |
| KAI-INTROSPECT-016 | HIGH | Dream responses disclose detailed history-derived insights and failure clusters |
| KAI-INTROSPECT-017 | HIGH | Evolution suggestions are stored as high-importance behavioural guidance without validation |
| KAI-INTROSPECT-018 | HIGH | Stored evolution reports are exposed without authentication |
| KAI-INTROSPECT-019 | MEDIUM | The endpoint hard-codes five episodes instead of using the configured dream threshold |
| KAI-INTROSPECT-020 | MEDIUM | Partial memory/checkpoint failure is hidden behind overall `status: ok` |
| KAI-INTROSPECT-021 | MEDIUM | Naive UTC timestamps are stored without timezone or source-clock provenance |
| KAI-INTROSPECT-022 | MEDIUM | Health is readiness-blind and reports only device/status |
| KAI-INTROSPECT-023 | HIGH | Audit logging is optional and silently disables when Redis is unavailable |
| KAI-INTROSPECT-024 | MEDIUM | Audit records contain only method/path/status and no authenticated actor or operation digest |
| KAI-INTROSPECT-025 | HIGH | The exposed security audit omits the HMAC boundary tests entirely |
| KAI-INTROSPECT-026 | HIGH | Known semantic/obfuscated injection attacks are classified as expected non-matches |
| KAI-INTROSPECT-027 | HIGH | Sanitizer test results can create false assurance about context-specific output safety |
| KAI-INTROSPECT-028 | HIGH | “Live defence” audit tests local functions/constants rather than deployed HTTP boundaries |
| KAI-INTROSPECT-029 | HIGH | A fixed public test corpus is easy for implementations to overfit |
| KAI-INTROSPECT-030 | HIGH | HMAC secret checks inspect an environment path string rather than loaded secret material |
| KAI-INTROSPECT-031 | HIGH | Policy audit does not verify secret entropy, file permissions, timestamps, nonce stores or body binding |
| KAI-INTROSPECT-032 | MEDIUM | Security audit `passed` count subtracts findings rather than failed test cases |
| KAI-INTROSPECT-033 | MEDIUM | Risk score treats multiple findings from one test as independent failures |
| KAI-INTROSPECT-034 | MEDIUM | HMAC audit considers only four narrow cases and does not test timestamp/replay enforcement |
| KAI-INTROSPECT-035 | MEDIUM | Sanitization length test threshold is inconsistent with the real 1,024-character truncation |
| KAI-INTROSPECT-036 | MEDIUM | One sanitization payload can produce several findings and distort aggregate metrics |
| KAI-INTROSPECT-037 | HIGH | The audit endpoint returns working bypass payloads and defensive gaps to anonymous callers |
| KAI-INTROSPECT-038 | MEDIUM | Audit IDs are short hashes of current time and are not durable evidence identifiers |
| KAI-INTROSPECT-039 | MEDIUM | Security audit runs are not persisted, signed or compared over time |
| KAI-INTROSPECT-040 | MEDIUM | Security audit executes synchronously on the request worker |
| KAI-INTROSPECT-041 | MEDIUM | Endpoint exceptions have no typed handling or redacted failure contract |
| KAI-INTROSPECT-042 | MEDIUM | Hot operations repeatedly construct short-lived HTTP clients |
| KAI-INTROSPECT-043 | MEDIUM | Dream/evolution report files are unsigned local state and failures are silent |
| KAI-INTROSPECT-044 | MEDIUM | The service has no controlled lifecycle resources, shutdown drain or background-job state |

---

## Service exposure and self-improvement endpoints

### KAI-INTROSPECT-001 — CRITICAL — Open self-improvement control plane
**Issue:** `docker-compose.full.yml` publishes `8023:8023`. `agentic/introspect_app.py` has no authentication dependency or authorisation middleware.  
**Risk:** Any reachable caller can trigger dream/evolution jobs, cause high-importance memory writes and inspect live security weaknesses.  
**Recommendation:** remove direct host exposure and require strongly authenticated, scoped operator/service calls.  
**Status:** OPEN — immediate remediation required

### KAI-INTROSPECT-002 — HIGH — No verified service identity
The service calls MEMU and Agentic Core and mutates persistent learning state, yet incoming requests carry no signed service identity, user delegation or replay protection.

### KAI-INTROSPECT-003 — HIGH — Ungated dream cycles
`POST /dream` has no Trust Core check, operator approval, feature flag or scheduler-origin validation.

### KAI-INTROSPECT-004 — HIGH — Ungated evolution cycles
`POST /evolve/analyze` similarly creates and persists behavioural recommendations without governance.

### KAI-INTROSPECT-005 — HIGH — Open security audit
`GET /security/audit` runs the defensive test suite and returns every finding/payload to any caller.

### KAI-INTROSPECT-006 — HIGH — No workload admission policy
Repeated callers can run overlapping CPU/file/Redis work. No mutex, job ID, queue bound, per-principal quota or cooldown exists.

### KAI-INTROSPECT-007 — HIGH — Blocking work in async handlers
`saver.recall`, `run_dream_cycle`, `analyze_failures`, report reads and security audit are synchronous and called directly from async endpoints.

### KAI-INTROSPECT-008 — HIGH — Shared-file race conditions
Dream/evolver persistence reads, appends and rewrites complete local JSON files without locks or atomic compare-and-swap. Concurrent cycles can lose or corrupt reports.

### KAI-INTROSPECT-009 — HIGH — Global identity collapse
Both analysis endpoints retrieve only `user_id="keeper"`, irrespective of caller, session or tenant.

### KAI-INTROSPECT-010 — HIGH — Self-generated evidence loop
Dream/evolution consumes episode outcome, conviction, failure classes and metacognitive rules created by the Agentic system itself—including the already-logged false-success outcomes—and treats them as real performance evidence.

### KAI-INTROSPECT-011 — HIGH — Unreviewed dream insight ingestion
Every actionable dream insight may be written directly to long-term memory as `dream_insight` without operator confirmation, provenance verification or contradiction checking.

### KAI-INTROSPECT-012 — HIGH — Self-confidence becomes memory authority
Insight confidence is copied into `relevance`; importance is fixed at 0.85. No external calibration or source independence supports these values.

### KAI-INTROSPECT-013 — HIGH — False storage success
The code awaits `client.post()` but never checks status. 4xx/5xx responses still increment `stored` and are reported as successful memory writes.

### KAI-INTROSPECT-014 — HIGH — Sequential connection churn
Each of up to five dream insights and every high/critical evolution suggestion creates its own `AsyncClient` and request sequentially.

### KAI-INTROSPECT-015 — HIGH — Unverified post-dream checkpoint
The service posts to Agentic Core `/checkpoint` but does not check status, parse the checkpoint ID or link it transactionally to the dream cycle. Failure is debug-only.

### KAI-INTROSPECT-016 — HIGH — Dream evidence disclosure
The response returns every generated insight, failure-cluster counts, merged rules and boundary-gap metrics derived from private episode history.

### KAI-INTROSPECT-017 — HIGH — Unverified evolution guidance
Critical/high suggestions are memorised with importance 0.9/0.8 and confidence-based relevance, allowing flawed self-analysis to shape future behaviour.

### KAI-INTROSPECT-018 — HIGH — Evolution-report disclosure
`GET /evolve/suggestions` returns the latest reports and concrete fixes without any principal or operator check.

### KAI-INTROSPECT-019 — MEDIUM — Configuration drift
The route requires a literal five episodes, while `kai_config.py` exposes `DREAM_MIN_EPISODES` as configuration. The endpoint and engine can disagree about readiness.

### KAI-INTROSPECT-020 — MEDIUM — Misleading success semantics
Individual insight writes and checkpoint creation may fail, but the endpoint returns `status="ok"`; only a count hints at partial work and even that count includes HTTP failures.

### KAI-INTROSPECT-021 — MEDIUM — Ambiguous timestamps
Memory records use naive `datetime.utcnow().isoformat()` without `Z`, timezone, source event ID or monotonic sequence.

### KAI-INTROSPECT-022 — MEDIUM — Readiness-blind health
`/health` always returns `ok` plus device and does not test episode store, MEMU, Agentic Core, report storage or security-audit dependencies.

### KAI-INTROSPECT-023 — HIGH — Optional audit authority
`AUDIT_REQUIRED` defaults false. `AuditStream` disables itself on Redis errors, so self-improvement/security operations can proceed with no durable audit.

### KAI-INTROSPECT-024 — MEDIUM — Weak audit event content
Middleware logs method, path and status only. It lacks authenticated actor, request/body digest, job/cycle ID, affected memory records and before/after state.

---

## Security self-audit implementation: `agentic/security_audit.py`

### KAI-INTROSPECT-025 — HIGH — HMAC tests are not run
The endpoint supplies only `injection_re` and `sanitize_fn`; `sign_fn` and `verify_fn` are omitted. The HMAC category is silently absent even though the endpoint claims a security self-audit.

### KAI-INTROSPECT-026 — HIGH — Bypass payloads are marked acceptable
Several known attacks—spaced letters, leetspeak, semantic instruction replacement, filter-removal phrasing and zero-width characters—are assigned `should_match=False`. If they bypass the regex, no finding is emitted; if caught, they may be labelled false positives.

### KAI-INTROSPECT-027 — HIGH — Context-free sanitization assurance
The audit assumes a generic string sanitizer should neutralise HTML, SQL, shell and path syntax. It does not test the actual sink/context where the value is later used, so both false positives and false negatives are likely.

### KAI-INTROSPECT-028 — HIGH — Not a deployed-boundary audit
The suite invokes local regex/function objects and reads environment settings. It does not send requests through live authentication middleware, proxies, redirects, Tool Gate, Redis nonce stores, model prompts or output renderers.

### KAI-INTROSPECT-029 — HIGH — Static corpus overfitting
The complete payload list is source-visible and fixed. A filter can pass the suite while remaining vulnerable to trivial unseen variants.

### KAI-INTROSPECT-030 — HIGH — Docker-secret path misclassified as secret material
Policy audit reads `os.getenv("INTERSERVICE_HMAC_SECRET")`. In compose this is `/run/secrets/hmac_secret`; the check treats that non-empty path string as a valid non-default secret without reading or validating the file.

### KAI-INTROSPECT-031 — HIGH — Governance audit omits core properties
It does not verify actual secret entropy/length, file ownership/mode, key-ID binding, revocation semantics, timestamp skew, nonce uniqueness/storage, request-body binding, endpoint scopes or inbound authentication.

### KAI-INTROSPECT-032 — MEDIUM — Invalid pass metric
`passed = total_tests - len(all_findings)`. Findings are not one-to-one with tests, so pass count can understate/overstate success and could become negative if multiple findings arise per case.

### KAI-INTROSPECT-033 — MEDIUM — Invalid risk denominator
Risk sums finding severities and divides by test count, treating correlated/multiple findings from one payload as independent test failures.

### KAI-INTROSPECT-034 — MEDIUM — Incomplete HMAC test design
Even when invoked elsewhere, HMAC tests cover valid/tampered/modified-tool/empty-nonce cases only. They do not test timestamp expiry, nonce replay, key revocation, strict key IDs, body mutation or cross-service identity.

### KAI-INTROSPECT-035 — MEDIUM — Meaningless overflow assertion
The sanitizer truncates to 1,024 characters, while the audit only flags output longer than 50,000. This check cannot detect changes that exceed the application’s intended bound but remain below 50,001.

### KAI-INTROSPECT-036 — MEDIUM — Duplicate per-case findings
One XSS payload can emit both script-tag and event-handler findings, distorting failed/pass/risk statistics.

### KAI-INTROSPECT-037 — HIGH — Defensive exploit disclosure
`SecurityFinding.to_dict()` returns the working payload and recommendation. The open endpoint provides attackers with confirmed bypasses and configuration weaknesses.

### KAI-INTROSPECT-038 — MEDIUM — Weak audit identity
Default IDs are 12 hexadecimal characters derived solely from current time. They are neither unpredictable security tokens nor durable, signed, collision-resistant evidence references.

### KAI-INTROSPECT-039 — MEDIUM — No audit history/integrity
Runs are returned but not persisted, signed, compared with prior baselines or anchored to source commit/configuration digest.

### KAI-INTROSPECT-040 — MEDIUM — Blocking audit execution
The complete suite executes synchronously in the async request handler with no deadline or bounded worker pool.

### KAI-INTROSPECT-041 — MEDIUM — Untyped endpoint failures
Dream/evolution/security exceptions propagate as generic 500 responses; there is no redacted error code, job state or recovery record.

### KAI-INTROSPECT-042 — MEDIUM — HTTP client lifecycle churn
MEMU writes and core checkpoint calls repeatedly create disposable clients rather than using one bounded lifecycle-managed pool.

### KAI-INTROSPECT-043 — MEDIUM — Report persistence is not trustworthy
Dream and evolver files are unsigned complete-file rewrites; errors are swallowed and loaders return empty lists, hiding loss/corruption.

### KAI-INTROSPECT-044 — MEDIUM — No job lifecycle management
The service exposes long-running work as synchronous HTTP calls and has no startup validation, shutdown drain, cancellation, progress, idempotency or durable job status.

---

## Batch totals

- Findings: **44**
- Critical: **1**
- High: **26**
- Medium: **17**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,182**
- Critical: **108**
- High: **494**
- Medium: **577**
- Low: **3**

## Files materially reviewed

`agentic/introspect_app.py`, `agentic/security_audit.py`, `agentic/kai_config.py`, with deployment confirmation against `docker-compose.full.yml` and core-checkpoint integration against `agentic/app.py`.
