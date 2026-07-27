# Kai Code Audit — Vault Sync Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-VAULT-001 | CRITICAL | Unauthenticated manual ingest can read files outside the vault root |
| KAI-VAULT-002 | CRITICAL | Vault export authorisation trusts a caller-supplied conviction score |
| KAI-VAULT-003 | HIGH | Vault export writes are non-atomic and can overwrite existing notes |
| KAI-VAULT-004 | HIGH | Failed graph deletion still removes the local mapping |
| KAI-VAULT-005 | HIGH | Mapping persistence failures are silently discarded |
| KAI-VAULT-006 | HIGH | Watcher-thread callbacks attempt to discover an event loop from the wrong thread |
| KAI-VAULT-007 | HIGH | Background queue workers are not retained, supervised or restarted |
| KAI-VAULT-008 | MEDIUM | Corrupt mapping state silently resets to empty |
| KAI-VAULT-009 | MEDIUM | Diagnostic endpoints expose mappings, paths and raw backend errors |
| KAI-VAULT-010 | MEDIUM | Request fields and search limits lack bounded validation |
| KAI-VAULT-011 | MEDIUM | Watcher startup failure leaves the service healthy and active |
| KAI-VAULT-012 | CRITICAL | Unauthenticated callers can search and retrieve synced vault memory |
| KAI-VAULT-013 | CRITICAL | Automatic watcher ingestion follows vault symlinks to files outside the vault |
| KAI-VAULT-014 | HIGH | Export path validation has a check-then-write symlink race |
| KAI-VAULT-015 | HIGH | Export reports success even when immediate memory ingestion fails |
| KAI-VAULT-016 | HIGH | Startup performs no initial vault scan, leaving existing notes unsynchronised |
| KAI-VAULT-017 | HIGH | Non-canonical filepath aliases can create duplicate graph nodes and broken deletion mappings |
| KAI-VAULT-018 | HIGH | Ingest and delete queues are unbounded and filesystem event storms have no backpressure |
| KAI-VAULT-019 | HIGH | Mapping file replacement is non-atomic and can corrupt the sole reconciliation state |
| KAI-VAULT-020 | MEDIUM | memu-core responses are unbounded and weakly schema-validated |
| KAI-VAULT-021 | MEDIUM | Note parsing has read/stat races and silently accepts malformed frontmatter state |
| KAI-VAULT-022 | MEDIUM | HTTP clients and connection pools are recreated for each ingest, delete and search operation |

---

## Vault sync: `vault-sync/app.py`, `vault-sync/mapper.py`, `vault-sync/parser.py`, `vault-sync/watcher.py`

### KAI-VAULT-001 — CRITICAL — Unauthenticated ingest reads arbitrary files
**Issue:** `POST /ingest` passes the caller-provided absolute or relative filepath directly to `parse_note`. Unlike export, ingest does not resolve the path beneath `VAULT_PATH`, require a `.md` extension or reject symlinks. The parser reads any existing regular file accessible to the service and forwards its content/metadata to memu-core.  
**Risk:** A reachable caller can import container/vault-mounted files into searchable persistent memory, creating a direct sensitive-file disclosure and persistence path.  
**Recommendation:** Accept vault-relative identifiers only, securely resolve beneath the configured root without following symlinks and require authenticated ingestion authority.  
**Status:** OPEN — immediate remediation required

### KAI-VAULT-002 — CRITICAL — Caller-supplied conviction bypasses write control
**Issue:** `POST /export` permits a write when `req.conviction` exceeds the configured threshold. The score and requester are ordinary caller-controlled fields; `TOOL_GATE_URL` is declared but never called, and no signed decision or authenticated identity is verified.  
**Risk:** Any reachable caller can submit a high score, overwrite/create vault notes and immediately poison the knowledge graph.  
**Recommendation:** Require an immutable replay-protected Tool Gate approval bound to authenticated identity, canonical path and exact content hash.  
**Status:** OPEN — immediate remediation required

### KAI-VAULT-003 — HIGH — Export overwrites notes non-atomically
**Issue:** Export creates parent directories and calls `target.write_text` directly with no exclusive-create/update distinction, revision check, backup, temporary file, fsync or atomic replacement.  
**Risk:** Existing human-authored notes can be silently replaced, and interruption can leave truncated content.  
**Recommendation:** Use explicit create/update semantics, optimistic revision checks and atomic durable replacement.  
**Status:** OPEN

### KAI-VAULT-004 — HIGH — Failed graph deletion discards reconciliation mapping
**Issue:** `_handle_delete` does not call `raise_for_status()` on memu-core deletion. Transport failures are logged, but the mapper entry is unconditionally removed afterward.  
**Risk:** Orphaned graph records remain while the service discards the only local relationship needed to identify/retry their deletion.  
**Recommendation:** Remove mappings only after confirmed idempotent deletion and use a durable retry/outbox state.  
**Status:** OPEN

### KAI-VAULT-005 — HIGH — Mapping persistence failures are invisible
**Issue:** `VaultMapper._save` catches every exception and does nothing while callers proceed as though state was durably updated.  
**Risk:** Sync state can be lost on restart, causing duplicate ingestion, missed deletion and graph/vault divergence without any visible failure.  
**Recommendation:** Propagate persistence failure, degrade readiness and use transactional durable storage.  
**Status:** OPEN

### KAI-VAULT-006 — HIGH — Watcher events are dropped from the wrong thread
**Issue:** `_enqueue_change` and `_enqueue_delete`, called from watchdog’s thread, invoke `asyncio.get_event_loop()` before `call_soon_threadsafe`. A non-main watcher thread generally has no current event loop, causing the exception path and dropping events.  
**Risk:** Automatic file changes/deletions fail to enter the queues, leaving the graph silently stale.  
**Recommendation:** Capture the running application loop at startup and use that exact loop from callbacks.  
**Status:** OPEN

### KAI-VAULT-007 — HIGH — Queue workers are unmanaged
**Issue:** Startup creates ingest/delete worker tasks without retaining references, completion callbacks, restart policy or shutdown cancellation.  
**Risk:** Unexpected worker termination permanently stops synchronisation while health remains ok; shutdown abandons queued work.  
**Recommendation:** Own all tasks through lifespan, supervise mandatory workers and expose durable queue/job state.  
**Status:** OPEN

### KAI-VAULT-008 — MEDIUM — Corrupt mapping silently resets
**Issue:** Mapping load catches every exception and replaces state with `{}` without preserving/quarantining the damaged file or exposing degradation.  
**Risk:** Existing node relationships disappear, preventing reliable update/deletion reconciliation and concealing tampering/corruption.  
**Recommendation:** Fail closed, quarantine damaged state and run explicit reconciliation.  
**Status:** OPEN

### KAI-VAULT-009 — MEDIUM — Diagnostic data is public
**Issue:** `/health` returns the vault path; `/mapping` returns every filepath-to-node mapping; `/export` returns an absolute path; `/search` exposes raw backend errors. No access control is present.  
**Risk:** Reachable callers gain knowledge-structure, filesystem and dependency diagnostics.  
**Recommendation:** Restrict all non-liveness endpoints and minimise/redact paths and identifiers.  
**Status:** OPEN

### KAI-VAULT-010 — MEDIUM — Inputs are unbounded
**Issue:** Filepaths, note content, requester, query, folder filter and search limit have no strict length/range constraints or global body quota.  
**Risk:** Oversized writes, parsing work, memory payloads and extreme search result requests can exhaust memory/storage.  
**Recommendation:** Add strict schemas, content quotas and bounded pagination.  
**Status:** OPEN

### KAI-VAULT-011 — MEDIUM — Health ignores disabled watcher/readiness
**Issue:** Missing watchdog or observer startup errors are only logged. `/health` always returns `status: ok`, even when sync is disabled, watcher is not running or workers are dead.  
**Risk:** Orchestration treats the primary change-detection capability as operational when it is absent.  
**Recommendation:** Separate liveness, enabled state, watcher/worker readiness and reconciliation health.  
**Status:** OPEN

### KAI-VAULT-012 — CRITICAL — Public vault-memory search
**Issue:** `GET /search` requires no authentication and proxies arbitrary queries/folder filters to `memu-core /memory/vault/search`, returning the complete JSON response.  
**Risk:** Any reachable caller can search and retrieve personal/operational content ingested from the Obsidian vault—including arbitrary files imported through KAI-VAULT-001.  
**Recommendation:** Require owner-scoped authentication, enforce result redaction/limits and isolate vault memory from general service networks.  
**Status:** OPEN — immediate remediation required

### KAI-VAULT-013 — CRITICAL — Automatic symlink escape
**Issue:** The watcher accepts any non-hidden path ending in `.md`. `parse_note` uses `Path.is_file()` and `read_text()`, both following symlinks, and automatic `_ingest_note` performs no canonical vault-root check.  
**Risk:** A symlink inside the watched vault can point to any readable external file with a `.md` link name; modifying/creating it causes automatic persistence of external contents into memu-core.  
**Recommendation:** Reject symlinks and use secure descriptor-based traversal constrained beneath the vault root.  
**Status:** OPEN — immediate remediation required

### KAI-VAULT-014 — HIGH — Export validation is TOCTOU-raceable
**Issue:** Export resolves/checks `target` beneath the vault, then separately creates parents and calls `write_text`. Vault directories/files can change between check and write; a symlink introduced after validation can redirect the write outside the vault.  
**Risk:** A local vault writer racing an unauthenticated export can cause arbitrary writable-file overwrite outside the intended root.  
**Recommendation:** Use descriptor-based no-follow creation beneath a pre-opened vault directory and atomic same-directory replacement.  
**Status:** OPEN

### KAI-VAULT-015 — HIGH — Export acknowledges unsynchronised writes
**Issue:** After writing, export calls `_ingest_note`. That function silently returns when parsing or memu-core ingestion fails. Export then always returns `status: ok` without confirming mapping or graph persistence.  
**Risk:** Callers/operators believe the vault and memory graph are synchronised when only the filesystem write succeeded, suppressing retries and reconciliation.  
**Recommendation:** Return separate durable vault-write and graph-ingest states; fail or queue retry when ingestion is not confirmed.  
**Status:** OPEN

### KAI-VAULT-016 — HIGH — Existing notes are never initially reconciled
**Issue:** Startup loads the mapping and starts the filesystem observer/workers but performs no scan of existing `.md` files and no comparison against graph/mapping state. Watchdog only receives future events.  
**Risk:** Notes present before service startup—or changes made while the service was down—remain permanently absent/stale until manually touched or ingested.  
**Recommendation:** Run a bounded versioned initial reconciliation before declaring readiness.  
**Status:** OPEN

### KAI-VAULT-017 — HIGH — Path aliases fragment identity
**Issue:** Mapper keys use the filepath string exactly as supplied. Manual relative paths, absolute watcher paths, `..` aliases and symlink aliases can refer to the same file under different keys/checksums. No canonical relative identity is enforced.  
**Risk:** The same note can create duplicate graph nodes; updates/deletions can target one alias while orphaning another.  
**Recommendation:** Canonicalise every note to one vault-relative no-symlink identifier before parsing, mapping or API response.  
**Status:** OPEN

### KAI-VAULT-018 — HIGH — Event queues are unbounded
**Issue:** `_ingest_queue` and `_delete_queue` are default unbounded `asyncio.Queue` instances. Watchdog debounce is per path but has no aggregate pending limit; event callbacks use `put_nowait`.  
**Risk:** Large filesystem churn or many unique files can grow pending dictionaries/queues without backpressure, exhausting memory while one worker processes sequentially.  
**Recommendation:** Use bounded queues, coalescing, per-path generation IDs and overflow/reconciliation fallback.  
**Status:** OPEN

### KAI-VAULT-019 — HIGH — Mapping writes are non-atomic
**Issue:** `_save` serialises the entire mapping and calls `Path.write_text` directly. The lock protects only one process/thread; no temporary file, fsync, atomic rename or revision exists.  
**Risk:** Crash/interruption can truncate the sole reconciliation state; multiple workers/processes can overwrite one another despite the local lock.  
**Recommendation:** Use transactional shared storage or locked atomic versioned replacement.  
**Status:** OPEN

### KAI-VAULT-020 — MEDIUM — Downstream payloads are weakly bounded/validated
**Issue:** Ingest/search materialise complete memu-core responses and parse JSON without byte/nesting/schema limits. The mapper trusts returned node/concept identifiers and types.  
**Risk:** Malformed/oversized responses can exhaust memory or corrupt mapping state.  
**Recommendation:** Enforce response caps and strict endpoint-specific schemas.  
**Status:** OPEN

### KAI-VAULT-021 — MEDIUM — Parser races and malformed frontmatter are hidden
**Issue:** `parse_note` checks existence/is-file, reads content, then calls `path.stat()` outside the read exception handler. Concurrent deletion/replacement can raise after content was read. YAML/frontmatter parsing failures are silently ignored and represented as ordinary body content/empty metadata.  
**Risk:** Manual requests can produce unstructured 500 errors; malformed metadata is ingested without an explicit degraded state.  
**Recommendation:** Open/stat through one stable descriptor and return typed parser errors.  
**Status:** OPEN

### KAI-VAULT-022 — MEDIUM — HTTP connection churn
**Issue:** Delete, ingest and search operations each create a new `httpx.AsyncClient`.  
**Risk:** Continuous vault changes repeatedly create DNS/TCP connection pools and increase latency/socket pressure.  
**Recommendation:** Reuse lifecycle-managed clients with bounded pools.  
**Status:** OPEN

---

## Batch totals

- Findings: **22**
- Critical: **4**
- High: **11**
- Medium: **7**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **878**
- Critical: **89**
- High: **314**
- Medium: **472**
- Low: **3**

## Files materially reviewed in this batch

`vault-sync/app.py`, `vault-sync/mapper.py`, `vault-sync/parser.py`, `vault-sync/watcher.py`, and the relevant deployment definition in `docker-compose.minimal.yml`.
