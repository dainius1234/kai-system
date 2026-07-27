# Kai Code Audit — memU Introspection and Memory Maintenance Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers `memu-core/introspect_app.py` and the maintenance handlers it re-registers from `memu-core/app.py`. General hot-path memory API findings will be recorded separately.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-MEMINT-001 | CRITICAL | The host-published memory-maintenance control plane has no authentication or authorisation |
| KAI-MEMINT-002 | CRITICAL | memU Core and Introspection concurrently mutate the same TurboVec index file from separate processes |
| KAI-MEMINT-003 | CRITICAL | Focus compression deletes original memories before replacement without a transaction or recovery journal |
| KAI-MEMINT-004 | CRITICAL | Anonymous callers can delete, revert, quarantine or restore long-term memories |
| KAI-MEMINT-005 | HIGH | No inbound service identity or signed operator delegation protects maintenance calls |
| KAI-MEMINT-006 | HIGH | Importing `app.py` constructs a second complete memory engine, model and store instance |
| KAI-MEMINT-007 | HIGH | Each process holds a stale independent TurboVec index object after the other process writes |
| KAI-MEMINT-008 | HIGH | Postgres mutations and TurboVec index mutations are not atomic |
| KAI-MEMINT-009 | HIGH | Deployed weekly compression is a no-op inherited from `PGVectorStore.compress()` |
| KAI-MEMINT-010 | HIGH | `/memory/compress` reports success and zero work under the deployed TurboVec backend |
| KAI-MEMINT-011 | HIGH | Persistent-store revert restores neither records nor TurboVec vectors |
| KAI-MEMINT-012 | HIGH | Revert operates on process-local commit history that excludes live Core writes |
| KAI-MEMINT-013 | HIGH | The returned commit-chain hash is not an integrity proof |
| KAI-MEMINT-014 | HIGH | Cleanup compares unvalidated text timestamps lexicographically |
| KAI-MEMINT-015 | HIGH | Cleanup has no maximum age bound, preview, confirmation or deletion cap |
| KAI-MEMINT-016 | HIGH | Focus compression merges memories across users and source principals |
| KAI-MEMINT-017 | HIGH | Merged memories discard source, trust tier, quarantine and evidential provenance |
| KAI-MEMINT-018 | HIGH | Merged text reuses one original embedding instead of embedding the merged claim |
| KAI-MEMINT-019 | HIGH | Merge rules inherit maximum relevance/importance/stability and summed access counts |
| KAI-MEMINT-020 | HIGH | Token budgeting uses an English character heuristic rather than the deployed model tokenizer |
| KAI-MEMINT-021 | HIGH | Focus compression may remain over budget while still returning success |
| KAI-MEMINT-022 | HIGH | Greedy keyword clustering is order-dependent, non-transitive and semantically weak |
| KAI-MEMINT-023 | HIGH | Focus compression does not reconcile graph-memory nodes for deleted or merged records |
| KAI-MEMINT-024 | HIGH | Reflection includes quarantined/poisoned memories in its source evidence |
| KAI-MEMINT-025 | HIGH | Reflection memories feed later reflection cycles and create self-reinforcing synthetic evidence |
| KAI-MEMINT-026 | HIGH | Reflection combines every user’s memories into one global behavioural summary |
| KAI-MEMINT-027 | HIGH | Repeated retrieval can manipulate “most revisited” reflection evidence |
| KAI-MEMINT-028 | HIGH | Reflection infers themes from raw category and word frequency without provenance or correctness |
| KAI-MEMINT-029 | HIGH | Self-generated reflection summaries are stored with high relevance and importance |
| KAI-MEMINT-030 | HIGH | The `half_life_days` decay parameter is accepted and returned but never used |
| KAI-MEMINT-031 | HIGH | Decay changes relevance for all users without principal scope |
| KAI-MEMINT-032 | HIGH | Decay is a sequence of independent updates with no transactional completion state |
| KAI-MEMINT-033 | HIGH | Decay/prune thresholds are environment-controlled and not range validated |
| KAI-MEMINT-034 | HIGH | Persistent `update_record()` reports success even when no record was updated |
| KAI-MEMINT-035 | HIGH | Clear-quarantine can remotely restore poisoned memory to retrieval |
| KAI-MEMINT-036 | HIGH | Quarantine reasons are unbounded caller text written to state and logs |
| KAI-MEMINT-037 | HIGH | Quarantine inventory is disclosed without authentication |
| KAI-MEMINT-038 | HIGH | `/memory/state` exposes a process-local state object that is not the live Core state |
| KAI-MEMINT-039 | HIGH | Category search returns full cross-user memory records without user filtering |
| KAI-MEMINT-040 | HIGH | Stats, categories, diagnostics and commit metadata are exposed without authentication |
| KAI-MEMINT-041 | MEDIUM | Maintenance repeatedly scans up to 10,000 complete records |
| KAI-MEMINT-042 | MEDIUM | Category-search query text bypasses `sanitize_string()` before embedding/search |
| KAI-MEMINT-043 | MEDIUM | No rate limit, job queue, concurrency cap or maintenance cooldown exists |
| KAI-MEMINT-044 | MEDIUM | Synchronous database, embedding, compression and filesystem work runs inside async handlers |
| KAI-MEMINT-045 | MEDIUM | Destructive maintenance operations have no shared mutual-exclusion lock |
| KAI-MEMINT-046 | MEDIUM | The weekly compression task is created without retaining its task handle |
| KAI-MEMINT-047 | MEDIUM | Shutdown does not cancel or await the weekly loop |
| KAI-MEMINT-048 | MEDIUM | Deprecated startup event hooks are used instead of owned lifespan resources |
| KAI-MEMINT-049 | MEDIUM | Health always returns `ok` without checking Postgres, TurboVec or index consistency |
| KAI-MEMINT-050 | MEDIUM | Audit logging is optional and can silently disable on Redis failure |
| KAI-MEMINT-051 | MEDIUM | Audit events omit actor identity, parameters, affected record IDs and operation digest |
| KAI-MEMINT-052 | MEDIUM | Maintenance and index configuration values are parsed without complete startup validation |
| KAI-MEMINT-053 | MEDIUM | Two revert route aliases expand the same unauthenticated destructive surface |
| KAI-MEMINT-054 | MEDIUM | Revert returns raw `KeyError` detail to callers |
| KAI-MEMINT-055 | MEDIUM | Merged summary timestamps are selected by lexical string maximum |
| KAI-MEMINT-056 | MEDIUM | Poisoned records are omitted from pre-compression token accounting |
| KAI-MEMINT-057 | MEDIUM | Post-compression accounting includes a different record set and may produce misleading savings |
| KAI-MEMINT-058 | MEDIUM | Singleton candidates are deleted and reinserted even when not merged |
| KAI-MEMINT-059 | MEDIUM | Version-control and state history are worker-local rather than shared deployment state |
| KAI-MEMINT-060 | MEDIUM | List and diagnostic endpoints lack pagination and bounded response contracts |

---

## Exposure and split-store architecture

### KAI-MEMINT-001 — CRITICAL — Open destructive memory control plane
**Issue:** `memu-core-introspect` is published on `8009:8009`. The FastAPI service has no authentication dependency or authorisation middleware, yet exposes compression, cleanup, decay, revert and quarantine mutation.  
**Risk:** Any reachable caller can irreversibly alter the system’s long-term memory and evidence base.  
**Recommendation:** remove host publication and require authenticated, scoped, reviewed maintenance jobs.  
**Status:** OPEN — immediate remediation required

### KAI-MEMINT-002 — CRITICAL — Shared TurboVec file corruption race
**Issue:** Both `memu-core` and `memu-core-introspect` use `VECTOR_STORE=turbovec` and mount `/data/turbovec/memories.tv`. Each process loads its own `IdMapIndex` and calls `.write()` to the same path without inter-process locking or atomic versioning.  
**Risk:** Concurrent insert/delete/rebuild writes can corrupt the file, lose vectors or publish an index inconsistent with Postgres.  
**Recommendation:** give index ownership to one authoritative process and perform atomic versioned snapshots/read-side reloads.  
**Status:** OPEN — immediate remediation required

### KAI-MEMINT-003 — CRITICAL — Delete-before-replace compression
**Issue:** `focus_compress()` deletes every compress-candidate ID, then inserts summaries/singletons one by one. There is no transaction, staging table, verified snapshot or rollback.  
**Risk:** Any exception, crash or index failure after deletion permanently loses original memories.  
**Recommendation:** build and validate a complete new snapshot, commit database/index/graph changes atomically, and retain a verified rollback generation.  
**Status:** OPEN — immediate remediation required

### KAI-MEMINT-004 — CRITICAL — Anonymous destructive memory operations
**Issue:** Cleanup, revert, quarantine and clear-quarantine routes are remotely callable without identity or confirmation.  
**Risk:** Attackers can delete history, roll state backward, suppress evidence or restore known poisoned records.  
**Recommendation:** enforce strong operator identity, reason, dry-run, signed approval and immutable audit for every mutation.  
**Status:** OPEN — immediate remediation required

### KAI-MEMINT-005 — HIGH — No trusted caller identity
No HMAC/mTLS/user token, delegation chain, nonce or endpoint scope is verified before maintenance work.

### KAI-MEMINT-006 — HIGH — Full engine duplicated by import
`introspect_app.py` imports handlers and `store` from `app.py`. Import executes embedding-model load, Redis setup, database/index initialisation, FastAPI construction and all module-level state in a second process.

### KAI-MEMINT-007 — HIGH — Stale independent vector views
After startup, each process’s `_index` changes only through its own operations. Postgres may contain records whose vector IDs are absent/stale in the other process’s in-memory index.

### KAI-MEMINT-008 — HIGH — Database/index split commits
TurboVec insert commits Postgres first, then mutates and writes the index. Delete commits Postgres before index removal/write. Either side can fail independently.

### KAI-MEMINT-009 — HIGH — Scheduled compaction does nothing
`TurboVecStore` inherits `PGVectorStore.compress()`, which returns fixed zero statistics and mutates nothing. The weekly loop therefore reports no exception while performing no compaction.

### KAI-MEMINT-010 — HIGH — False successful `/memory/compress`
The route wraps the no-op result with `status="ok"`, making deployed maintenance appear complete.

### KAI-MEMINT-011 — HIGH — Persistent revert cannot restore memory
`PGVectorStore.revert()` and inherited TurboVec behaviour call the LakeFS client and only assign `_state`; they do not restore Postgres records or rebuild/reload TurboVec.

### KAI-MEMINT-012 — HIGH — Wrong commit universe
The default LakeFS fallback is an in-process list. Introspection sees only commits created by its own store instance, not Core’s history.

### KAI-MEMINT-013 — HIGH — Self-hash is not integrity
`memory_revert()` hashes the current local commit-list JSON after the operation. There is no signed chain, prior trusted root or proof that records/index match that list.

### KAI-MEMINT-014 — HIGH — Text timestamp deletion
Persistent cleanup uses `timestamp < cutoff` on a text column. Non-canonical, timezone-varied or malformed timestamps compare lexically rather than chronologically.

### KAI-MEMINT-015 — HIGH — Unbounded destructive cleanup scope
Only `max_age_days >= 1` is checked. One anonymous request can remove nearly all non-pinned history, with no count cap, preview or second approval.

---

## Focus compression and reflection

### KAI-MEMINT-016 — HIGH — Cross-user memory merging
Compression groups by category only. Records from different `content.user_id` values are clustered together, and the merged record inherits the first/best user ID.

### KAI-MEMINT-017 — HIGH — Provenance destruction
Merged records default trust tier/source/quarantine fields and retain only abbreviated snippets, count and categories. Original IDs, sources, evidence links and complete claims disappear.

### KAI-MEMINT-018 — HIGH — Embedding/content mismatch
A merged summary reuses `best.embedding`, although its text represents up to ten different memories.

### KAI-MEMINT-019 — HIGH — Synthetic authority inflation
Merged records take maximum relevance, importance and stability and sum access counts, making a lossy synthetic summary rank more strongly than its inputs.

### KAI-MEMINT-020 — HIGH — Invalid token budget model
Token cost is `len(text)//4`, ignoring the deployed model, language, code, metadata and message framing.

### KAI-MEMINT-021 — HIGH — Budget may remain violated
Pinned/top-K focus records are never truncated. If they alone exceed the budget, the function still applies changes and returns `status="ok"` without an unmet-budget error.

### KAI-MEMINT-022 — HIGH — Weak/order-dependent clustering
Greedy Jaccard overlap of four-character words uses the first cluster member as the only comparison. Ordering changes cluster membership, while negation, numbers and causal meaning are ignored.

### KAI-MEMINT-023 — HIGH — Graph/vector divergence
Focus compression calls store delete/insert directly and does not issue graph forget/ingest operations for originals or summaries.

### KAI-MEMINT-024 — HIGH — Poisoned evidence enters reflection
`reflect()` does not exclude records marked `poisoned`; malformed timestamps are also included as recent.

### KAI-MEMINT-025 — HIGH — Recursive synthetic evidence
Reflection records are normal high-scoring memories and are included in later reflection scans, allowing generated summaries to reinforce and multiply themselves.

### KAI-MEMINT-026 — HIGH — Cross-user behavioural inference
Reflection aggregates category counts, access counts and keywords across the complete store, with no user/source partition.

### KAI-MEMINT-027 — HIGH — Retrieval-count manipulation
“Most revisited” topics depend on `access_count`, which grows whenever retrieval returns a record. Repeated queries can force selected text into reflection summaries.

### KAI-MEMINT-028 — HIGH — Unsupported theme inference
Raw word/category frequency is labelled focus areas and emerging themes without source quality, independence, chronology or semantic verification.

### KAI-MEMINT-029 — HIGH — Generated insight promotion
Each reflection string is stored at relevance 0.9 and importance 0.85 without operator/verifier approval.

---

## Decay, quarantine and disclosure

### KAI-MEMINT-030 — HIGH — Unused half-life control
`apply_spaced_repetition_decay(half_life_days)` validates and returns the parameter, but `_recency_weight()` does not receive or use it.

### KAI-MEMINT-031 — HIGH — Global decay scope
All non-pinned/non-poisoned records are modified irrespective of user, source, legal hold or retention policy.

### KAI-MEMINT-032 — HIGH — Partial decay commits
Each record update is independent. Mid-loop failure leaves an unrecorded partially transformed memory store while the endpoint has no resumable job state.

### KAI-MEMINT-033 — HIGH — Unsafe thresholds
`DECAY_FADE_THRESHOLD`, `MARS_PRUNE_THRESHOLD`, focus budgets, top-K and weekly interval are parsed from environment without complete finite/range/cross-field validation.

### KAI-MEMINT-034 — HIGH — False update success
Persistent `update_record()` commits and returns `True` without checking `rowcount`. Quarantine/clear can claim success for a nonexistent ID.

### KAI-MEMINT-035 — HIGH — Poison restoration endpoint
`clear_quarantine()` removes poisoned status without re-verification, source review or operator identity.

### KAI-MEMINT-036 — HIGH — Unbounded quarantine reason
Reason text is not sanitised/bounded and is written to database and info logs, enabling log injection/sensitive-data retention.

### KAI-MEMINT-037 — HIGH — Quarantine disclosure
The list endpoint returns IDs, reasons, timestamps, event types and categories for every poisoned record.

### KAI-MEMINT-038 — HIGH — False live-state view
The separate TurboVec store’s `_state` is process-local and not loaded from Core; `/memory/state` can return `ok` with an empty/stale state.

### KAI-MEMINT-039 — HIGH — Cross-user category search
`search_by_category()` filters category only and returns complete `MemoryRecord` objects, including content and embeddings, for all users.

### KAI-MEMINT-040 — HIGH — Memory metadata disclosure
Stats/categories/diagnostics expose record/event counts, limits and local commit metadata without authentication.

---

## Operational findings

### KAI-MEMINT-041 — MEDIUM — Full-store scans
Reflection, focus compression, decay, quarantine listing, categories, stats and diagnostics repeatedly request up to 10,000 complete records.

### KAI-MEMINT-042 — MEDIUM — Unsanitised category query
The optional query is passed to store search and embedding generation without the sanitisation applied to category.

### KAI-MEMINT-043 — MEDIUM — No maintenance admission control
No rate limit, job queue, per-operation mutex, principal quota or cooldown protects expensive/destructive jobs.

### KAI-MEMINT-044 — MEDIUM — Blocking request execution
Most handlers call synchronous Postgres, TurboVec, embedding, compression and filesystem methods directly from async endpoints.

### KAI-MEMINT-045 — MEDIUM — No shared maintenance lock
Cleanup, decay, focus compression, revert, weekly compression and Core writes can overlap against the same records/index.

### KAI-MEMINT-046 — MEDIUM — Untracked weekly task
Startup creates `_weekly_compress_loop()` with raw `asyncio.create_task()` and discards the handle.

### KAI-MEMINT-047 — MEDIUM — No graceful loop shutdown
There is no shutdown handler to cancel/await the weekly task or finish an in-progress index write.

### KAI-MEMINT-048 — MEDIUM — Deprecated lifecycle API
The service uses `@app.on_event("startup")` rather than an owned lifespan context.

### KAI-MEMINT-049 — MEDIUM — Readiness-blind health
Health always returns `ok` plus device and never checks Postgres, index load, index/Postgres ID consistency or weekly-loop state.

### KAI-MEMINT-050 — MEDIUM — Optional audit
`AUDIT_REQUIRED` defaults false and the audit client can disable itself if Redis is unavailable.

### KAI-MEMINT-051 — MEDIUM — Inadequate mutation audit
Middleware logs only method/path/status, not actor, parameters, before/after counts, record IDs, snapshot/index generation or body digest.

### KAI-MEMINT-052 — MEDIUM — Weak configuration validation
Port, index bits/path, candidate limits, thresholds, record limits and intervals can fail startup or create unsafe behaviour without a validated configuration report.

### KAI-MEMINT-053 — MEDIUM — Duplicate destructive aliases
Both `/memory/revert` and `/revert` expose the same rollback operation.

### KAI-MEMINT-054 — MEDIUM — Error detail disclosure
A revert `KeyError` is copied into an HTTP 404 detail.

### KAI-MEMINT-055 — MEDIUM — Lexical summary timestamp
`max(r.timestamp)` selects the merged record timestamp as a string, not a parsed normalised instant.

### KAI-MEMINT-056 — MEDIUM — Incomplete pre-budget accounting
Poisoned records are skipped from `tokens_before` while remaining in the store.

### KAI-MEMINT-057 — MEDIUM — Inconsistent post-budget accounting
`tokens_after` scans all final records, including records excluded from the initial total; reported savings can therefore be misleading or negative for reasons unrelated to compression.

### KAI-MEMINT-058 — MEDIUM — Unnecessary singleton replacement
Every non-focus singleton is deleted and reinserted even when no merge occurred, changing database/index identity and increasing failure exposure.

### KAI-MEMINT-059 — MEDIUM — Local version/state history
Each process has its own LakeFS fallback client and `_state`, so maintenance status/commit listings cannot represent the deployment’s authoritative history.

### KAI-MEMINT-060 — MEDIUM — Unpaged response surfaces
Quarantine, categories, search, stats and diagnostics have fixed large scans and no cursor/revision-consistent pagination contract.

---

## Batch totals

- Findings: **60**
- Critical: **4**
- High: **36**
- Medium: **20**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,242**
- Critical: **112**
- High: **530**
- Medium: **597**
- Low: **3**

## Files materially reviewed

`memu-core/introspect_app.py`, maintenance/store implementations in `memu-core/app.py`, and shared deployment definitions in `docker-compose.full.yml`.
