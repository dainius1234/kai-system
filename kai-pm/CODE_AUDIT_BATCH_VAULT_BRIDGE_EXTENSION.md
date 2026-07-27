# Kai Code Audit — Vault Bridge Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only additional findings not already present in `CODE_AUDIT_BATCH_VAULT_SYNC.md`. The earlier 22 Vault Sync findings are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-VAULTX-001 | CRITICAL | Non-finite conviction values bypass the vault export threshold |
| KAI-VAULTX-002 | CRITICAL | memU’s vault-delete route can delete any memory record by caller-supplied ID |
| KAI-VAULTX-003 | CRITICAL | Tampering the mapping file can turn a vault file deletion into arbitrary memU memory deletion |
| KAI-VAULTX-004 | HIGH | Vault export can overwrite `.vault-sync` metadata and other hidden control files |
| KAI-VAULTX-005 | HIGH | The requester identity is caller-supplied and accepted only for attribution/logging |
| KAI-VAULTX-006 | HIGH | Unbounded file writes execute synchronously on the async request worker |
| KAI-VAULTX-007 | HIGH | Filesystem write, memU ingestion and mapper update have no atomic transaction |
| KAI-VAULTX-008 | HIGH | Note parsing materialises the complete file before any size enforcement |
| KAI-VAULTX-009 | HIGH | YAML frontmatter parsing has no byte, alias, nesting or structural-work budget |
| KAI-VAULTX-010 | HIGH | Wikilink and tag extraction has no item-count or aggregate-size bounds |
| KAI-VAULTX-011 | HIGH | memU retains complete vault-note bodies in unbounded process memory |
| KAI-VAULTX-012 | HIGH | memU trusts the caller-provided checksum instead of hashing received content |
| KAI-VAULTX-013 | HIGH | The memU “knowledge graph” ingest creates no graph nodes or relationships |
| KAI-VAULTX-014 | HIGH | memU’s vault filepath index is volatile, worker-local state |
| KAI-VAULTX-015 | HIGH | Restart or worker change loses filepath identity and creates duplicate orphan memories |
| KAI-VAULTX-016 | HIGH | memU suppresses vault-record deletion errors and returns success |
| KAI-VAULTX-017 | HIGH | The mapping file is an unsigned, writable authority for downstream deletion identity |
| KAI-VAULTX-018 | HIGH | Mapper lookups expose mutable internal dictionaries outside the lock |
| KAI-VAULTX-019 | HIGH | Queue-processing failures permanently drop sync events without retry or dead-letter state |
| KAI-VAULTX-020 | MEDIUM | Debounce timer threads are not cancelled or joined during shutdown |
| KAI-VAULTX-021 | MEDIUM | File moves generate independent delete and ingest operations that can race |
| KAI-VAULTX-022 | MEDIUM | Deprecated startup/shutdown event hooks weaken lifecycle ownership |
| KAI-VAULTX-023 | MEDIUM | No rate limit, concurrency cap, workload quota or idempotency key protects vault operations |
| KAI-VAULTX-024 | MEDIUM | memU vault search becomes empty or stale after restart and differs across workers |
| KAI-VAULTX-025 | MEDIUM | Tag-derived concept IDs are unsanitised and collision-prone |
| KAI-VAULTX-026 | MEDIUM | Frontmatter, wikilinks and modified time are accepted but discarded by memU ingestion |
| KAI-VAULTX-027 | MEDIUM | Logs retain full filesystem paths, spoofed requester identity and conviction values |
| KAI-VAULTX-028 | MEDIUM | No durable audit links actor, file revision, checksum, memory ID and mapper revision |
| KAI-VAULTX-029 | MEDIUM | An empty search query matches and enumerates every process-local vault entry |
| KAI-VAULTX-030 | MEDIUM | Folder filtering is a raw string-prefix test rather than canonical path containment |

---

## Critical findings

### KAI-VAULTX-001 — CRITICAL — Non-finite conviction bypass
**Issue:** Export blocks only when `req.conviction < VAULT_WRITE_CONVICTION_THRESHOLD`. IEEE NaN compares false, so a non-finite value passes this test. The field has no finite 0–10 validation and is not a signed Gate outcome.  
**Risk:** The sole write threshold can be bypassed without supplying the configured score.  
**Recommendation:** Reject every non-finite/out-of-range value and authorise only from a replay-protected Gate result bound to the exact path and content digest.  
**Status:** OPEN — immediate remediation required

### KAI-VAULTX-002 — CRITICAL — Vault delete is an arbitrary memory-delete primitive
**Issue:** `DELETE /memory/vault/{note_node_id}` calls `store.delete_record(note_node_id)` without verifying that the ID belongs to `_vault_notes`, has event type `vault_note`, or is owned by the initiating principal.  
**Risk:** A caller who learns any memory record ID can delete that non-vault memory through the vault route.  
**Recommendation:** Resolve deletion from an authenticated canonical filepath-to-vault-record mapping and require the stored record to match the expected vault type, owner and revision.  
**Status:** OPEN — immediate remediation required

### KAI-VAULTX-003 — CRITICAL — Mapping tampering becomes arbitrary memory deletion
**Issue:** Vault Sync trusts `note_node_id` loaded from writable `.vault-sync/mapping.json`. When a mapped file is deleted, that ID is sent to memU’s unrestricted vault-delete route.  
**Risk:** Modifying the mapping file to reference another memory ID, then deleting the mapped note, causes deletion of the targeted memory record.  
**Recommendation:** Sign/version mapping entries, verify the referenced record is the matching vault note, and perform deletion through an authenticated idempotent operation.  
**Status:** OPEN — immediate remediation required

---

## Additional high-severity findings

### KAI-VAULTX-004 — HIGH — Export can overwrite sync-control files
The root-containment check permits paths such as `.vault-sync/mapping.json` and other hidden metadata below the vault root. Export can therefore corrupt the service’s own reconciliation authority, not only ordinary notes.

### KAI-VAULTX-005 — HIGH — Spoofed requester attribution
`requester` is an unrestricted body string. It is logged as though it identifies the actor but is never authenticated or used in authorisation.

### KAI-VAULTX-006 — HIGH — Blocking unbounded writes
`content` has no byte limit and `Path.write_text()` executes synchronously inside an async endpoint, allowing large requests to block the event loop and consume disk.

### KAI-VAULTX-007 — HIGH — Split file/memory/mapping transaction
The file may be written, memU insertion may fail, and mapping may remain unchanged; conversely memU may insert while mapper persistence fails. No operation ID, staging state or compensating rollback exists.

### KAI-VAULTX-008 — HIGH — Whole-file materialisation
`parse_note()` calls `read_text()` before any size check and then duplicates data during checksum, YAML, regex and HTTP payload construction.

### KAI-VAULTX-009 — HIGH — Unbounded YAML work
Although `yaml.safe_load` prevents arbitrary Python-object construction, it does not impose application limits on aliases, nesting, scalar lengths or total parse work.

### KAI-VAULTX-010 — HIGH — Unbounded extracted metadata
Every wikilink and tag match is accumulated and serialised. One note can create arbitrarily large lists and downstream payloads.

### KAI-VAULTX-011 — HIGH — Full-note process-memory retention
memU truncates searchable `memory_text` to 2,000 content characters but stores the complete request body in `_vault_notes`. Unique filepaths allow that dictionary to grow without a cardinality or byte cap.

### KAI-VAULTX-012 — HIGH — Unverified checksum provenance
The memU endpoint stores `req.checksum` directly. It does not recompute a digest from the title/content/frontmatter/tags it actually received.

### KAI-VAULTX-013 — HIGH — Graph ingest is an interface fiction
The endpoint returns generated concept IDs but never calls the graph service, persists wikilink edges, or records tag relationships. The resulting state is a memory record plus a process-local dictionary, not a knowledge-graph sync.

### KAI-VAULTX-014 — HIGH — Volatile worker-local vault index
`_vault_notes` is not persisted and is not rebuilt from Postgres. Each memU worker has a different search/delete filepath view.

### KAI-VAULTX-015 — HIGH — Duplicate/orphan creation after restart
After `_vault_notes` is lost, ingesting the same filepath generates a new node ID. The old Postgres/TurboVec record remains, creating duplicate searchable notes and an orphan that normal filepath deletion cannot identify.

### KAI-VAULTX-016 — HIGH — False successful memU deletion
The memU route catches every `store.delete_record` exception and still returns `status: ok`, preventing Vault Sync from detecting or retrying storage failure.

### KAI-VAULTX-017 — HIGH — Unsigned deletion authority
The mapping file controls which memU node is deleted but has no MAC/signature, trusted ownership/mode verification or monotonic revision.

### KAI-VAULTX-018 — HIGH — Mutable mapper state escapes its lock
`get_by_filepath()` returns the live internal entry dictionary. Callers may mutate it after the lock is released without persistence or synchronisation.

### KAI-VAULTX-019 — HIGH — Failed events are discarded
Queue workers log any ingest/delete exception and always call `task_done()`. There is no retry count, durable outbox, quarantine or reconciliation job.

---

## Additional medium-severity findings

### KAI-VAULTX-020 — MEDIUM — Debounce timer lifecycle leak
`FileWatcher.stop()` stops only the watchdog observer. Any active daemon `threading.Timer` and pending callback map are not cancelled or joined.

### KAI-VAULTX-021 — MEDIUM — Move ordering race
A move schedules destination ingestion and immediately emits source deletion. Separate queues/workers can process them in either order, producing transient or permanent mapping/node inconsistencies.

### KAI-VAULTX-022 — MEDIUM — Deprecated lifecycle API
The service uses `@app.on_event("startup"/"shutdown")` rather than a lifespan context that owns the observer, event loop reference, clients and worker tasks.

### KAI-VAULTX-023 — MEDIUM — Missing workload controls
Manual ingest/export/search have no global rate limit, per-principal quota, concurrent parser/write bound or idempotency key.

### KAI-VAULTX-024 — MEDIUM — Stale process-local search
`/memory/vault/search` searches `_vault_notes` only. Existing Postgres records vanish from search after restart, and different workers expose different results.

### KAI-VAULTX-025 — MEDIUM — Unsafe concept identifiers
Concept IDs replace `/` in tags but otherwise retain spaces, punctuation, Unicode variants and delimiters, allowing collisions and invalid downstream identities.

### KAI-VAULTX-026 — MEDIUM — Accepted metadata is discarded
Frontmatter, wikilinks and `modified_at` are accepted and transmitted but not stored in the memory record or represented as graph relations.

### KAI-VAULTX-027 — MEDIUM — Sensitive operational logging
Logs include complete source/target filesystem paths, spoofed requester values, node IDs and conviction scores.

### KAI-VAULTX-028 — MEDIUM — Missing end-to-end audit chain
No immutable event links authenticated actor, source file descriptor/path, content digest, filesystem revision, memU record ID, mapper revision and graph status.

### KAI-VAULTX-029 — MEDIUM — Empty-query enumeration
In memU vault search, `q_lower == ""` is a substring of every title/content/tag, so an empty query returns all process-local entries subject only to the weak slice limit.

### KAI-VAULTX-030 — MEDIUM — Lexical folder containment
`fp.startswith(folder_filter)` treats similarly prefixed paths as the same folder and does not canonicalise separators, relative components or path segments.

---

## Batch totals

- Findings: **30**
- Critical: **3**
- High: **16**
- Medium: **11**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,482**
- Critical: **125**
- High: **686**
- Medium: **668**
- Low: **3**

## Files materially reviewed

`vault-sync/app.py`, `vault-sync/mapper.py`, `vault-sync/parser.py`, `vault-sync/watcher.py`, memU vault endpoints in `memu-core/app.py`, and vault deployment in `docker-compose.minimal.yml`.
