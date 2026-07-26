# Kai Code Audit — Vault Sync Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

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
| KAI-VAULT-009 | MEDIUM | Diagnostic and search endpoints expose mappings, paths and raw backend errors |
| KAI-VAULT-010 | MEDIUM | Request fields and search limits lack bounded validation |
| KAI-VAULT-011 | MEDIUM | Watcher startup failure leaves the service healthy and active |

---

### KAI-VAULT-001 — CRITICAL — Unauthenticated manual ingest can read files outside the vault root
**Issue:** `POST /ingest` passes the caller-provided absolute or relative filepath directly to `parse_note`. Unlike export, ingest does not resolve the path beneath `VAULT_PATH`. The parser reads any existing regular file accessible to the service and forwards its content and metadata to memu-core. No authentication dependency is present.  
**Risk:** A reachable caller can import host/container files into searchable persistent memory, creating a direct sensitive-file disclosure and persistence path.  
**Recommendation:** Accept vault-relative identifiers only, canonicalise beneath the configured root, reject symlinks and require authenticated ingestion authority.  
**Status:** OPEN — immediate remediation required

### KAI-VAULT-002 — CRITICAL — Vault export authorisation trusts caller-supplied conviction
**Issue:** `POST /export` permits a write when `req.conviction` exceeds a threshold. The score and requester are ordinary caller-controlled request fields; no signed Tool Gate decision or authenticated identity is verified.  
**Risk:** Any reachable caller can submit a high score, overwrite or create vault notes and immediately poison the knowledge graph.  
**Recommendation:** Require an immutable, replay-protected approval capability bound to the exact path, content hash and requester identity.  
**Status:** OPEN — immediate remediation required

### KAI-VAULT-003 — HIGH — Vault export is non-atomic and overwrites existing notes
**Issue:** Export creates parent directories and calls `target.write_text` directly with no exclusive-create option, version check, backup or atomic replacement.  
**Risk:** Existing human-authored notes can be silently replaced, and interruption can leave truncated content.  
**Recommendation:** Use explicit create/update semantics, optimistic revision checks and atomic durable replacement.  
**Status:** OPEN

### KAI-VAULT-004 — HIGH — Failed graph deletion still removes the local mapping
**Issue:** `_handle_delete` logs memu-core deletion failure but unconditionally removes the mapper entry afterward. It also does not validate successful HTTP status before removing the mapping.  
**Risk:** Orphaned graph records remain while the service discards the only local relationship needed to identify and retry their deletion.  
**Recommendation:** Remove mappings only after confirmed idempotent deletion and use a durable retry/outbox state for failures.  
**Status:** OPEN

### KAI-VAULT-005 — HIGH — Mapping persistence failures are silently discarded
**Issue:** `VaultMapper._save` catches every exception and does nothing, while callers proceed as though the mapping was durably updated.  
**Risk:** Synchronisation state can be lost on restart, causing duplicate ingestion, missed deletion and graph/vault divergence without any visible failure.  
**Recommendation:** Propagate persistence failure, mark readiness degraded and write through transactional or atomic durable storage.  
**Status:** OPEN

### KAI-VAULT-006 — HIGH — Watcher callbacks obtain the event loop from the watcher thread
**Issue:** `_enqueue_change` and `_enqueue_delete`, called by the watchdog thread, use `asyncio.get_event_loop()` before `call_soon_threadsafe`. In current Python thread semantics, a non-main watcher thread generally has no current loop, causing the exception path and dropping the event.  
**Risk:** Automatic file changes and deletions can fail to enter the queues, leaving the graph silently stale.  
**Recommendation:** Capture the running application loop during startup and use that specific loop from watcher callbacks.  
**Status:** OPEN

### KAI-VAULT-007 — HIGH — Queue workers are not supervised
**Issue:** Startup calls `asyncio.create_task` for ingest and delete workers but does not retain task references, inspect completion or restart failed workers.  
**Risk:** An unexpected worker termination can permanently stop synchronisation while the process remains healthy.  
**Recommendation:** Retain and supervise lifecycle tasks, expose worker readiness and restart or fail the service when a mandatory worker exits.  
**Status:** OPEN

### KAI-VAULT-008 — MEDIUM — Corrupt mapping state silently resets to empty
**Issue:** Mapping load catches every exception and replaces state with `{}` without preserving the damaged file or exposing degradation.  
**Risk:** Existing node relationships disappear from service state, preventing reliable update and deletion reconciliation.  
**Recommendation:** Quarantine corrupt state, block mutation and run an explicit reconciliation procedure.  
**Status:** OPEN

### KAI-VAULT-009 — MEDIUM — Diagnostics expose mappings, paths and backend errors
**Issue:** `/health` returns the vault path; `/mapping` returns every filepath-to-node mapping; `/export` returns an absolute path; and `/search` returns raw backend exception text. No access control is visible.  
**Risk:** Reachable callers gain sensitive knowledge-structure and filesystem information.  
**Recommendation:** Minimise public health data and restrict diagnostics to authenticated operators with redaction.  
**Status:** OPEN

### KAI-VAULT-010 — MEDIUM — Request fields lack bounded validation
**Issue:** Filepaths, note content, requester, search query, folder filter and search limit have no strict length/range constraints.  
**Risk:** Oversized writes, parsing load and extreme backend result requests can exhaust memory or storage.  
**Recommendation:** Add strict schemas, global body limits, content quotas and finite pagination caps.  
**Status:** OPEN

### KAI-VAULT-011 — MEDIUM — Watcher failure does not affect service readiness
**Issue:** Missing watchdog or observer startup errors are only logged. `/health` still returns `status: ok`, even when `watcher_running` is false and automatic synchronisation is unavailable.  
**Risk:** Orchestration and operators can treat the service as operational while its primary change-detection function is disabled.  
**Recommendation:** Separate liveness and readiness and fail readiness when enabled mandatory watcher or workers are unavailable.  
**Status:** OPEN

---

## Batch totals

- Findings: **11**
- Critical: **2**
- High: **5**
- Medium: **4**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **145**
- Critical: **21**
- High: **64**
- Medium: **59**
- Low: **1**

## Files materially reviewed in this batch

`vault-sync/app.py`, `vault-sync/mapper.py`, `vault-sync/parser.py`, `vault-sync/watcher.py`.
