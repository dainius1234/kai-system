# Kai Code Audit — memU Core Hot Path Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers the main memory write, retrieval, routing, session and operator-preference paths in `memu-core/app.py`. Destructive maintenance findings are recorded separately in `CODE_AUDIT_BATCH_MEMU_INTROSPECTION.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-MEMCORE-001 | CRITICAL | The host-published memU Core memory authority has no inbound authentication or authorisation |
| KAI-MEMCORE-002 | CRITICAL | Anonymous callers can impersonate `keeper` and create pinned/high-authority memories |
| KAI-MEMCORE-003 | CRITICAL | The “require verdict PASS” gate stores memories when the verifier is unreachable, returns REPAIR or returns an unknown verdict |
| KAI-MEMCORE-004 | CRITICAL | `/route` discloses cross-user memory embedding vectors and internal model-routing metadata |
| KAI-MEMCORE-005 | CRITICAL | Arbitrary-user retrieval and evidence-pack endpoints expose full private memory records |
| KAI-MEMCORE-006 | CRITICAL | Anonymous session callers can read, inject system-role messages into, or delete any session |
| KAI-MEMCORE-007 | CRITICAL | Anonymous callers can create pinned operator preferences that are injected into future plans |
| KAI-MEMCORE-008 | CRITICAL | Forced contradiction override leaves the superseded memory active and retrievable |
| KAI-MEMCORE-009 | HIGH | No inbound service identity, end-user delegation or replay protection exists |
| KAI-MEMCORE-010 | HIGH | Policy-load failure disables verifier gating |
| KAI-MEMCORE-011 | HIGH | Log-only mode stores exact FAIL_CLOSED memories |
| KAI-MEMCORE-012 | HIGH | A successful verifier verdict is not persisted into the record’s trust tier |
| KAI-MEMCORE-013 | HIGH | Stored memories omit authenticated source provenance |
| KAI-MEMCORE-014 | HIGH | Caller-controlled timestamps govern ordering, retention and deletion |
| KAI-MEMCORE-015 | HIGH | PII redaction applies only to result text and not nested metrics/state/category/source fields |
| KAI-MEMCORE-016 | HIGH | Caller-supplied relevance and importance values are not bounded or checked for finiteness |
| KAI-MEMCORE-017 | HIGH | Event type, category and user identity are weakly validated strings |
| KAI-MEMCORE-018 | HIGH | State delta may commit before memory insertion and leave a partial operation |
| KAI-MEMCORE-019 | HIGH | Persistent-store state is process-local rather than database authoritative |
| KAI-MEMCORE-020 | HIGH | State duplicate checks and updates race across requests and workers |
| KAI-MEMCORE-021 | HIGH | Anonymous callers can create arbitrary persistent-looking state keys and values |
| KAI-MEMCORE-022 | HIGH | Generic sanitisation silently truncates and strips data, creating collisions and semantic corruption |
| KAI-MEMCORE-023 | HIGH | Synchronous embedding generation executes inside async request handlers |
| KAI-MEMCORE-024 | HIGH | Graph-ingest tasks are fire-and-forget and untracked |
| KAI-MEMCORE-025 | HIGH | Graph-ingest HTTP failure/status is not part of write success semantics |
| KAI-MEMCORE-026 | HIGH | Unverified caller identity and metadata are propagated into the graph side channel |
| KAI-MEMCORE-027 | HIGH | Records with an empty or missing user ID are visible to every requested user |
| KAI-MEMCORE-028 | HIGH | Memory retrieval is a state-changing operation |
| KAI-MEMCORE-029 | HIGH | Repeated anonymous retrieval makes selected memories increasingly permanent and prominent |
| KAI-MEMCORE-030 | HIGH | Retrieval interval calculation always falls back to one day because `last_accessed` is overwritten before comparison |
| KAI-MEMCORE-031 | HIGH | Concurrent background access-count/stability updates lose increments |
| KAI-MEMCORE-032 | HIGH | Semantic similarity is an unnormalised dot product |
| KAI-MEMCORE-033 | HIGH | Ranking combines manipulable relevance, importance, pin and access fields as authoritative evidence |
| KAI-MEMCORE-034 | HIGH | Unknown trust tiers receive a neutral ranking weight rather than a restrictive one |
| KAI-MEMCORE-035 | HIGH | Malformed timestamps are treated as freshly created memories |
| KAI-MEMCORE-036 | HIGH | Automatic record trimming can delete pinned preferences and protected memories |
| KAI-MEMCORE-037 | HIGH | TurboVec trimming deletes Postgres rows without removing corresponding vector IDs |
| KAI-MEMCORE-038 | HIGH | Record trimming orders caller-controlled timestamp text lexicographically |
| KAI-MEMCORE-039 | HIGH | TurboVec query failures silently degrade to latest-record retrieval |
| KAI-MEMCORE-040 | HIGH | Stale vector IDs missing from Postgres are silently dropped from results |
| KAI-MEMCORE-041 | HIGH | Record upsert updates only a subset of security and provenance fields |
| KAI-MEMCORE-042 | HIGH | Legitimate zero relevance, importance and stability values are replaced with elevated defaults on read |
| KAI-MEMCORE-043 | HIGH | Database configuration has a known `keeper:localdev` fallback credential |
| KAI-MEMCORE-044 | HIGH | Missing LakeFS dependency silently substitutes non-durable in-process version control |
| KAI-MEMCORE-045 | HIGH | In-memory store mode is volatile, worker-local and presented through the same API contract |
| KAI-MEMCORE-046 | HIGH | Fake hash embeddings can be enabled without a degraded health/result marker |
| KAI-MEMCORE-047 | HIGH | `/route` ignores the caller’s timestamp and returns current-state metadata instead |
| KAI-MEMCORE-048 | HIGH | Specialist routing uses hard-coded substring rules and unverified model identities |
| KAI-MEMCORE-049 | HIGH | Sanitised session IDs can collide after character removal and truncation |
| KAI-MEMCORE-050 | HIGH | Redis failure silently creates split-brain per-process session state |
| KAI-MEMCORE-051 | HIGH | In-memory fallback permits unbounded unique session IDs |
| KAI-MEMCORE-052 | HIGH | Session message content is unbounded and stored without sanitisation or provenance |
| KAI-MEMCORE-053 | HIGH | Session clear reports success even when Redis deletion fails |
| KAI-MEMCORE-054 | HIGH | Session context always retrieves long-term memory for global user `keeper` |
| KAI-MEMCORE-055 | HIGH | Untrusted memory text is formatted as ready-to-inject LLM context |
| KAI-MEMCORE-056 | HIGH | Anonymous recovery mutates the database pool and records fabricated perfect conscience alignment |
| KAI-MEMCORE-057 | MEDIUM | Session TTL and turn-limit configuration is not validated |
| KAI-MEMCORE-058 | MEDIUM | Memory, candidate and state limits are weakly validated at startup |
| KAI-MEMCORE-059 | MEDIUM | Numerous endpoints scan up to 10,000 complete records per request |
| KAI-MEMCORE-060 | MEDIUM | Preference retrieval limit is not range validated |
| KAI-MEMCORE-061 | MEDIUM | Memory list/search surfaces lack cursor-based, revision-consistent pagination |
| KAI-MEMCORE-062 | MEDIUM | `/route` can return a large payload of raw embedding vectors |
| KAI-MEMCORE-063 | MEDIUM | Version branch names use second-resolution timestamps and can collide |
| KAI-MEMCORE-064 | MEDIUM | Regex PII redaction is context-incomplete and can over-redact or miss sensitive data |
| KAI-MEMCORE-065 | MEDIUM | Category classification is manipulable substring counting |
| KAI-MEMCORE-066 | MEDIUM | Importance scoring rewards verbosity and trigger-word stuffing |
| KAI-MEMCORE-067 | MEDIUM | Keeper authority is derived from a caller-controlled string comparison |
| KAI-MEMCORE-068 | MEDIUM | Boundary, proactive, silence and tempo analysis aggregate all users together |
| KAI-MEMCORE-069 | MEDIUM | Malformed proactive timestamps are retained as recent candidates |
| KAI-MEMCORE-070 | MEDIUM | Silence analysis silently ignores invalid timestamps and emits misleading inactivity results |
| KAI-MEMCORE-071 | MEDIUM | Tempo modelling treats all memory records as operator interactions |
| KAI-MEMCORE-072 | MEDIUM | Source-event clocks and timestamp formats have no canonical validation |
| KAI-MEMCORE-073 | MEDIUM | State validation errors disclose caller-supplied key names |
| KAI-MEMCORE-074 | MEDIUM | No global rate limiting, workload admission or principal quotas protect the memory authority |
| KAI-MEMCORE-075 | MEDIUM | Synchronous database and CPU-heavy work runs on async request workers |
| KAI-MEMCORE-076 | MEDIUM | Background persistence and graph tasks are not owned by a lifespan manager |
| KAI-MEMCORE-077 | MEDIUM | Fire-and-forget task admission is unbounded and has no shutdown drain |
| KAI-MEMCORE-078 | MEDIUM | Specialist inventories and routing claims can drift from model registries and live backends |
| KAI-MEMCORE-079 | MEDIUM | Audit records lack authenticated actor, request/body digest and affected-record provenance |
| KAI-MEMCORE-080 | MEDIUM | Full `MemoryRecord` responses leak embeddings and internal ranking/security fields |

---

## Exposure and authority

### KAI-MEMCORE-001 — CRITICAL — Open memory authority
**Issue:** `docker-compose.full.yml` publishes `8001:8001`. `memu-core/app.py` defines no authentication or authorisation middleware.  
**Risk:** Any reachable caller can write, retrieve, pin, supersede, route, recover and manipulate the system’s core memory and session state.  
**Recommendation:** remove direct host exposure and require principal-scoped authenticated requests, service identity and endpoint-specific authorisation.  
**Status:** OPEN — immediate remediation required

### KAI-MEMCORE-002 — CRITICAL — Keeper impersonation and pinned poisoning
**Issue:** `user_id` and `pin` are caller-controlled. A caller setting `user_id="keeper"` receives keeper importance/pinning behaviour; quick notes and preferences can be pinned permanently.  
**Risk:** Anonymous input receives operator authority and preferential future retrieval.  
**Recommendation:** derive principal and pin rights from verified credentials, never from body fields.  
**Status:** OPEN — immediate remediation required

### KAI-MEMCORE-003 — CRITICAL — “Require PASS” is not enforced
**Issue:** When `REQUIRE_VERDICT_PASS` is enabled, only exact `FAIL_CLOSED` blocks. `VERIFIER_UNREACHABLE`, `REPAIR`, unknown/malformed verdicts and other non-PASS values continue to storage.  
**Risk:** The system claims verifier-gated promotion while storing unverified or explicitly repair-needed claims.  
**Recommendation:** accept only a strict authenticated PASS result; every other outcome must block or quarantine.  
**Status:** OPEN — immediate remediation required

### KAI-MEMCORE-004 — CRITICAL — Cross-user embedding disclosure through routing
**Issue:** `/route` calls `store.search(top_k=50)` with no query or user filter and returns every selected record’s full embedding vector, plus device and specialist inventory.  
**Risk:** Anonymous callers can extract behavioural/vector representations across users and infer storage/model configuration.  
**Recommendation:** remove embeddings from API responses and enforce authenticated user-scoped retrieval.  
**Status:** OPEN — immediate remediation required

### KAI-MEMCORE-005 — CRITICAL — Arbitrary-user memory exfiltration
**Issue:** `/memory/retrieve` and `/memory/evidence-pack` accept caller-selected `user_id` and return full `MemoryRecord` content, embeddings, trust/rank/source fields and metadata.  
**Risk:** Private operator memories and evidential data can be enumerated remotely.  
**Recommendation:** derive user scope from authentication and minimise response fields.  
**Status:** OPEN — immediate remediation required

### KAI-MEMCORE-006 — CRITICAL — Session takeover and system-role injection
**Issue:** GET, POST and DELETE session routes accept any caller-selected session ID. Append permits role `system`, although the request model describes user/assistant turns.  
**Risk:** An attacker can read private conversations, insert privileged prompt instructions and erase working memory.  
**Recommendation:** use server-generated principal-bound sessions and prohibit externally supplied system-role messages.  
**Status:** OPEN — immediate remediation required

### KAI-MEMCORE-007 — CRITICAL — Operator-preference poisoning
**Issue:** `POST /memory/preferences` lets any caller create a pinned record with relevance 1.0 and importance 0.95. Agentic planning retrieves these preferences before plans.  
**Risk:** Anonymous instructions become durable high-authority operator preferences.  
**Recommendation:** require explicit authenticated operator confirmation and signed preference schemas.  
**Status:** OPEN — immediate remediation required

### KAI-MEMCORE-008 — CRITICAL — Superseded memory remains active
**Issue:** `force=true` bypasses contradiction review. The old record receives only a `quarantine_reason`; `poisoned` is not set and retrieval excludes only poisoned records.  
**Risk:** Both contradictory records continue to influence answers while the API falsely reports the old one as superseded.  
**Recommendation:** perform an atomic versioned replacement with explicit active/superseded state and review evidence.  
**Status:** OPEN — immediate remediation required

---

## Write path and verifier integration

### KAI-MEMCORE-009 — HIGH — No trusted incoming identity
No HMAC, mTLS, API token, user delegation, nonce or replay control protects memory requests.

### KAI-MEMCORE-010 — HIGH — Policy failure disables protection
Any exception loading the memory policy sets `REQUIRE_VERDICT_PASS=False` and `LOG_ONLY_MODE=False`, silently removing verifier enforcement.

### KAI-MEMCORE-011 — HIGH — FAIL_CLOSED log-only promotion
When `LOG_ONLY_MODE` is enabled, exact verifier FAIL_CLOSED records are stored normally rather than quarantined.

### KAI-MEMCORE-012 — HIGH — Verdict is not durable trust state
The API returns the verifier verdict, but the new `MemoryRecord` does not set `trust_tier`; it remains `unverified` even after PASS and cannot be audited consistently later.

### KAI-MEMCORE-013 — HIGH — Missing source provenance
New notes/assertions/memorised events usually leave `source_id=None`; body `user_id` is not an authenticated source.

### KAI-MEMCORE-014 — HIGH — Caller timestamp authority
Arbitrary timestamp strings are stored and later drive recency, sorting, trimming, cleanup, proactive nudges and behavioural analysis.

### KAI-MEMCORE-015 — HIGH — Nested PII bypass
`redact_pii()` covers only `result_raw`/note text. Arbitrary PII can remain in metrics, state deltas, category, event type, context and other nested fields.

### KAI-MEMCORE-016 — HIGH — Unbounded ranking values
`relevance` and explicit `importance` lack finite 0–1 validation. Negative, NaN, infinity and huge values can distort ranking and JSON.

### KAI-MEMCORE-017 — HIGH — Weak identity/category/type validation
User IDs, event types and explicit categories are ordinary strings with generic truncation but no canonical enum, ownership or namespace checks.

### KAI-MEMCORE-018 — HIGH — Partial state/memory transaction
`state_delta` is applied before the memory record is embedded/inserted. Later failure leaves state committed without the corresponding memory event.

### KAI-MEMCORE-019 — HIGH — State is not persistent authoritative data
PG/Turbo `apply_state_delta()` modifies process-local `_state` and local version client; it does not store the state in Postgres/shared Redis.

### KAI-MEMCORE-020 — HIGH — State races
Duplicate-key checking and update are separate, unsynchronised operations against worker-local dictionaries.

### KAI-MEMCORE-021 — HIGH — Anonymous state mutation
Any caller may submit arbitrary nested state keys/values within per-item size limits; no allowed-key schema or authorisation exists.

### KAI-MEMCORE-022 — HIGH — Destructive generic sanitisation
`sanitize_string()` strips `;|&` and truncates to 1,024 characters. This silently alters legitimate facts/IDs and can make distinct input collide; it does not make values safe for all sinks.

### KAI-MEMCORE-023 — HIGH — Blocking embedding generation
SentenceTransformer embedding executes synchronously in async endpoints for every note, assertion, preference, memorise, retrieve and category query.

### KAI-MEMCORE-024 — HIGH — Untracked graph tasks
Writes use raw `asyncio.create_task()` for graph ingestion; no task registry, bounded admission, cancellation or shutdown drain exists.

### KAI-MEMCORE-025 — HIGH — Graph write not acknowledged
Graph ingestion does not check response status and failures cannot affect memory-write success, producing silent vector/graph divergence.

### KAI-MEMCORE-026 — HIGH — Unverified graph provenance
Caller-supplied user/event/category metadata is propagated into graph ingestion as though it were source attribution.

---

## Retrieval and storage consistency

### KAI-MEMCORE-027 — HIGH — Missing-user records bypass isolation
Retrieval filters only when both requested and stored user IDs are non-empty. Records with no user ID are returned to every user scope.

### KAI-MEMCORE-028 — HIGH — Reads mutate memory
Every returned record has access count, last-accessed and stability changed. A supposedly read-only API therefore changes ranking and retention state.

### KAI-MEMCORE-029 — HIGH — Retrieval amplification
Repeated queries increase stability multiplicatively up to a one-year cap, causing selected memories to dominate future recall and resist forgetting.

### KAI-MEMCORE-030 — HIGH — Rehearsal interval bug
The code assigns `record.last_accessed = now_iso` before comparing it with `now_iso`; the previous access timestamp is lost and interval defaults to one day every time.

### KAI-MEMCORE-031 — HIGH — Lost update races
Background updates read-modify-write absolute access/stability values. Concurrent retrievals can overwrite one another rather than atomically incrementing.

### KAI-MEMCORE-032 — HIGH — Invalid similarity metric
`_similarity()` is a raw dot product. The code does not explicitly normalise embeddings, so vector magnitude can affect score independently of semantic angle.

### KAI-MEMCORE-033 — HIGH — Manipulable rank authority
Caller-controlled relevance, importance, pin, user identity, access count and malformed timestamps contribute directly to evidence ranking.

### KAI-MEMCORE-034 — HIGH — Unknown trust defaults neutral
Any unrecognised trust-tier string receives zero adjustment rather than quarantine/deny weighting.

### KAI-MEMCORE-035 — HIGH — Bad timestamps become newest
`_recency_weight()` catches parse errors and sets age to zero, giving malformed records maximum freshness.

### KAI-MEMCORE-036 — HIGH — Pinned records can be auto-evicted
PG/Turbo insertion trims oldest rows above `MAX_MEMORY_RECORDS` without excluding pinned/legal-hold records.

### KAI-MEMCORE-037 — HIGH — Orphan TurboVec vectors
The automatic Postgres trim does not remove deleted rows’ `int_id` values from the index or rewrite the index file.

### KAI-MEMCORE-038 — HIGH — Lexical age ordering
Trim uses `ORDER BY timestamp ASC` on a text field containing caller-controlled formats, not a validated timestamp column.

### KAI-MEMCORE-039 — HIGH — Semantic outage hides as recency
TurboVec search catches all index exceptions and returns newest records, with no degraded marker to the caller.

### KAI-MEMCORE-040 — HIGH — Stale IDs silently reduce recall
Vector IDs that no longer exist in Postgres are omitted from `ordered`; no consistency error or rebuild is triggered.

### KAI-MEMCORE-041 — HIGH — Partial upsert semantics
On ID conflict, content/embedding/relevance/importance/category update, but timestamp, event type, pin, trust, source, poison and quarantine fields remain old.

### KAI-MEMCORE-042 — HIGH — Zero values are inflated
Row conversion uses expressions such as `r[6] or 1.0`, `r[7] or 0.5` and `r[10] or 1.0`; valid zero values become elevated defaults.

### KAI-MEMCORE-043 — HIGH — Known fallback database credentials
PG/Turbo default to `postgresql://keeper:localdev@postgres:5432/sovereign` when deployment secrets are missing.

### KAI-MEMCORE-044 — HIGH — Fictional version-control fallback
If `lakefs_client` is unavailable, an in-memory stub creates commit IDs and accepts revert calls without durable external versioning.

### KAI-MEMCORE-045 — HIGH — Volatile store contract
`VECTOR_STORE=memory` exposes the same success-shaped API while all records/state/version history are process-local and lost on restart.

### KAI-MEMCORE-046 — HIGH — Fake embeddings lack runtime provenance
When explicitly enabled, deterministic 8-dimensional hash vectors are used, but health/responses do not identify semantic retrieval as degraded/fake.

---

## Routing, sessions and operator models

### KAI-MEMCORE-047 — HIGH — Ignored routing timestamp
`MemoryRequest.timestamp` is required but unused. The response inserts a new server time, obscuring request/event chronology.

### KAI-MEMCORE-048 — HIGH — Static substring specialist routing
Model selection checks a short list of substrings and returns hard-coded model names without live capability/readiness verification.

### KAI-MEMCORE-049 — HIGH — Session-ID canonicalisation collisions
Removing `;|&` and truncating to 1,024 can map distinct attacker/user IDs to the same Redis key.

### KAI-MEMCORE-050 — HIGH — Session split-brain fallback
Redis failures are swallowed and operations fall back to one worker’s in-memory dictionaries. Other workers and later Redis recovery see different histories.

### KAI-MEMCORE-051 — HIGH — Unbounded fallback session cardinality
Appending does not clean expired sessions first, and arbitrary new IDs create process-memory entries without a global cap.

### KAI-MEMCORE-052 — HIGH — Unbounded raw message content
`SessionMessage.content` has no byte limit and is stored directly in Redis/in-memory buffers.

### KAI-MEMCORE-053 — HIGH — False session-clear success
Redis delete exceptions are ignored and the endpoint always returns `cleared: true`, even if the authoritative Redis session remains.

### KAI-MEMCORE-054 — HIGH — Session context hard-codes keeper memory
`/session/{id}/context` always calls `retrieve_ranked(..., "keeper")`, independent of session owner or caller.

### KAI-MEMCORE-055 — HIGH — Memory prepared as prompt text
The endpoint formats untrusted memory content into strings intended for direct LLM prompt injection, including attacker-created pinned preferences/notes.

### KAI-MEMCORE-056 — HIGH — Recovery fabricates ethical success
Unauthenticated `/recover` may replace the DB pool, suppresses failures, then writes a conscience event with alignment score 1.0 and verdict `fully_aligned` even when nothing was healed.

---

## Additional operational findings

### KAI-MEMCORE-057 — MEDIUM — Session configuration validation
`SESSION_MAX_TURNS` and TTL are direct integer environment parses without positive/safe-range validation.

### KAI-MEMCORE-058 — MEDIUM — Limit configuration validation
`MAX_MEMORY_RECORDS`, candidate counts, state sizes, graph timeout and other limits can be negative/extreme or fail startup.

### KAI-MEMCORE-059 — MEDIUM — Full-store scans
Preferences, categories, stats, proactive, silence, tempo, reflection, maintenance and other routes repeatedly request up to 10,000 records.

### KAI-MEMCORE-060 — MEDIUM — Preference limit semantics
`top_k` is not constrained; negative slicing and large values create surprising results after a full-store scan.

### KAI-MEMCORE-061 — MEDIUM — No consistent pagination
Large retrieval/list endpoints lack cursors tied to a snapshot/revision, making responses expensive and inconsistent during concurrent mutation.

### KAI-MEMCORE-062 — MEDIUM — Large vector payload
`/route` returns 50 complete embedding vectors even though the caller normally needs contextual records or a specialist decision.

### KAI-MEMCORE-063 — MEDIUM — Version branch collision
Branch names use `int(time.time())`, so concurrent updates within one second can produce identical names.

### KAI-MEMCORE-064 — MEDIUM — Regex PII limitations
The scanner supports a small pattern set, may classify ordinary numeric strings as phone data, misses many secret/identity formats and does not handle structured-field context.

### KAI-MEMCORE-065 — MEDIUM — Weak category classification
Category is selected by raw substring hit count; words inside unrelated text and caller-provided explicit categories bypass semantic validation.

### KAI-MEMCORE-066 — MEDIUM — Importance keyword gaming
Length, words such as “critical/urgent/safety” and current global emotion increase importance without evidence of actual significance.

### KAI-MEMCORE-067 — MEDIUM — String-based keeper authority
Importance and pin defaults use `user_id == "keeper"`, which is a caller-controlled string rather than a verified principal.

### KAI-MEMCORE-068 — MEDIUM — Cross-user behavioural aggregation
Knowledge boundary, proactive nudges, silence signals and tempo modelling aggregate the entire memory store without user partition.

### KAI-MEMCORE-069 — MEDIUM — Invalid proactive dates remain eligible
Proactive scanning catches timestamp parse errors and continues without excluding the record, allowing malformed entries into the current window.

### KAI-MEMCORE-070 — MEDIUM — Invalid silence dates distort absence
Silence analysis increments category totals before parsing timestamps; invalid dates can make categories look historically active but recently silent.

### KAI-MEMCORE-071 — MEDIUM — Tempo does not measure interactions
It treats every recent memory timestamp—including system reflections, imports and background insights—as an operator interaction.

### KAI-MEMCORE-072 — MEDIUM — No canonical event clock
Timestamp fields accept ISO-like, epoch-like and arbitrary strings with no timezone, clock-skew or source-event validation.

### KAI-MEMCORE-073 — MEDIUM — State-key disclosure
Oversized-state errors include the caller-supplied key name in public detail.

### KAI-MEMCORE-074 — MEDIUM — No admission control
No service-wide rate limit, per-user quota, write budget, embedding concurrency or retrieval cost limit protects the memory authority.

### KAI-MEMCORE-075 — MEDIUM — Blocking async workers
Synchronous psycopg2, TurboVec, filesystem and embedding calls execute directly inside async endpoints.

### KAI-MEMCORE-076 — MEDIUM — Unowned lifecycle tasks
Health persistence and graph/retrieval updates are created ad hoc rather than through an application lifespan/task manager.

### KAI-MEMCORE-077 — MEDIUM — Unbounded background work
Repeated writes/reads can create unlimited graph and persistence tasks; shutdown does not drain them.

### KAI-MEMCORE-078 — MEDIUM — Model-inventory drift
`SPECIALISTS` and `select_specialist()` are another independent model registry that can disagree with Agentic registries and deployed models.

### KAI-MEMCORE-079 — MEDIUM — Weak audit evidence
Audit events record method/path/status only and can disable unless required; no authenticated actor, memory ID list, body digest or before/after revision is included.

### KAI-MEMCORE-080 — MEDIUM — Internal schema leakage
Returning `MemoryRecord` exposes embeddings, access counts, stability, trust tier, source, poison/quarantine fields and ranking scores beyond ordinary memory content.

---

## Batch totals

- Findings: **80**
- Critical: **8**
- High: **48**
- Medium: **24**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,322**
- Critical: **120**
- High: **578**
- Medium: **621**
- Low: **3**

## Files materially reviewed

`memu-core/app.py`, `common/runtime.py`, and main memU deployment definitions in `docker-compose.full.yml`.
