# Kai Code Audit — File-backed Persistence Batch 1

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

This batch records confirmed persistence and data-integrity findings from the world-anchor, financial-awareness and market-cache paths.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-PERS-001 | HIGH | World-anchor mutation endpoints are unauthenticated |
| KAI-PERS-002 | HIGH | World-anchor JSON persistence is non-atomic and concurrency-unsafe |
| KAI-PERS-003 | HIGH | Financial record persistence is non-atomic and concurrency-unsafe |
| KAI-PERS-004 | HIGH | Corrupt financial records silently become an empty dataset |
| KAI-PERS-005 | MEDIUM | World-anchor corrupt JSON silently becomes an empty dataset |
| KAI-PERS-006 | MEDIUM | World-anchor data defaults to ephemeral `/tmp` storage |
| KAI-PERS-007 | MEDIUM | Financial health output discloses the internal storage path |
| KAI-PERS-008 | MEDIUM | Financial persistence errors expose raw exception text |
| KAI-PERS-009 | MEDIUM | Financial string and date inputs lack consistent bounded validation |
| KAI-PERS-010 | HIGH | Market-cache failures silently substitute hard-coded values as current data |
| KAI-PERS-011 | MEDIUM | Market-cache persistence is non-atomic |
| KAI-PERS-012 | MEDIUM | Corrupt market cache is overwritten without preserving forensic evidence |

---

## World anchor: `calendar-sync/app.py`

### KAI-PERS-001 — HIGH — World-anchor mutation endpoints are unauthenticated
**Issue:** `POST /news` and `POST /events` accept and persist caller-supplied content without an authentication or authorisation dependency.  
**Risk:** Any caller with network reachability can poison the contextual data used to ground downstream advice and nudges.  
**Recommendation:** Require authenticated operator or trusted-ingestion identity, source provenance and per-operation authorisation.  
**Status:** OPEN

### KAI-PERS-002 — HIGH — World-anchor JSON persistence is non-atomic and concurrency-unsafe
**Issue:** Mutations perform an unlocked read-modify-write cycle and overwrite the entire JSON file with `Path.write_text`.  
**Risk:** Concurrent requests can lose entries, and interruption during a rewrite can corrupt the complete dataset.  
**Recommendation:** Use transactional storage or locked atomic temp-file replacement with fsync and optimistic version checks.  
**Status:** OPEN

### KAI-PERS-005 — MEDIUM — Corrupt world-anchor JSON silently becomes an empty dataset
**Issue:** `_load_json` catches every exception and returns `[]`.  
**Risk:** Corruption or permission failure is indistinguishable from a genuinely empty feed and subsequent writes can permanently replace recoverable data.  
**Recommendation:** Fail visibly, preserve the damaged file and expose degraded readiness until repaired.  
**Status:** OPEN

### KAI-PERS-006 — MEDIUM — World-anchor data defaults to ephemeral storage
**Issue:** `WORLD_ANCHOR_DATA_DIR` defaults to `/tmp/world-anchor`.  
**Risk:** Events and contextual records can disappear on restart, image replacement or ordinary temporary-directory cleanup.  
**Recommendation:** Require an explicitly mounted persistent data path outside development mode.  
**Status:** OPEN

---

## Financial awareness: `financial-awareness/app.py`

### KAI-PERS-003 — HIGH — Financial record persistence is non-atomic and concurrency-unsafe
**Issue:** CIS records are loaded, appended and the complete JSON array is rewritten without locking, a transaction or atomic replacement.  
**Risk:** Simultaneous submissions can overwrite one another; interruption can destroy the sole financial-record file.  
**Recommendation:** Store records in an ACID database with unique immutable IDs and transactional inserts, or implement rigorously locked atomic replacement.  
**Status:** OPEN

### KAI-PERS-004 — HIGH — Corrupt financial records silently become an empty dataset
**Issue:** `_load_records` catches every exception and returns `[]`.  
**Risk:** Tax, CIS and VAT summaries can report zero records after corruption or access failure, creating materially false financial outputs; a later write can overwrite the original evidence.  
**Recommendation:** Treat parse/read failure as a hard degraded state, preserve the original file and block mutation until recovery.  
**Status:** OPEN

### KAI-PERS-007 — MEDIUM — Financial health output discloses the storage path
**Issue:** `/health` returns `finance_root` as a filesystem path.  
**Risk:** Unauthenticated callers gain internal deployment and sensitive-data location information.  
**Recommendation:** Keep health responses minimal and place diagnostics behind authenticated operator endpoints.  
**Status:** OPEN

### KAI-PERS-008 — MEDIUM — Financial persistence errors expose raw exception text
**Issue:** Record-save failures return `detail=f"Failed to persist record: {exc}"`.  
**Risk:** Responses can disclose filesystem paths, permission details and runtime internals.  
**Recommendation:** Return stable public error codes and keep exception detail in protected logs with a trace ID.  
**Status:** OPEN

### KAI-PERS-009 — MEDIUM — Financial inputs lack consistent bounded validation
**Issue:** Contractor names, UTRs, descriptions, addresses, invoice references and date strings have no explicit length bounds; payment and invoice dates are persisted without consistent schema-level ISO-date validation.  
**Risk:** Oversized or malformed values can bloat the single-file store, degrade rendering and silently disappear from tax-year calculations.  
**Recommendation:** Add strict lengths, structured identifiers and typed date validation before persistence.  
**Status:** OPEN

---

## Market cache: `common/market_cache.py`

### KAI-PERS-010 — HIGH — Failed external data is silently replaced with hard-coded current-looking values
**Issue:** Fetch exceptions are suppressed and the cache retains defaults such as a petrol price and `"+0.05 tomorrow"`, then stamps the payload with the current time.  
**Risk:** Downstream logic can treat synthetic fallback values as freshly observed market data. The fabricated trend can influence financial or planning advice without provenance.  
**Recommendation:** Represent each source as `unavailable` or explicitly `synthetic`, retain the last verified observation with its original timestamp and never relabel fallback data as freshly fetched.  
**Status:** OPEN

### KAI-PERS-011 — MEDIUM — Market-cache persistence is non-atomic
**Issue:** The cache is rewritten directly with `Path.write_text`.  
**Risk:** Process interruption or concurrent refreshes can leave invalid JSON or lose a newer result.  
**Recommendation:** Use locked atomic replacement and version/timestamp conflict handling.  
**Status:** OPEN

### KAI-PERS-012 — MEDIUM — Corrupt market cache is overwritten without forensic preservation
**Issue:** Any read or parse exception calls `refresh_cache`, which rewrites the same cache path.  
**Risk:** Evidence of corruption is destroyed and synthetic defaults can immediately replace previously valid data.  
**Recommendation:** Quarantine the damaged file, emit an alert and require an explicit recovery path.  
**Status:** OPEN

---

## Batch totals

- Findings: **12**
- Critical: **0**
- High: **5**
- Medium: **7**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **134**
- Critical: **19**
- High: **59**
- Medium: **55**
- Low: **1**

## Files materially reviewed in this batch

`calendar-sync/app.py`, `financial-awareness/app.py`, `common/market_cache.py`.
