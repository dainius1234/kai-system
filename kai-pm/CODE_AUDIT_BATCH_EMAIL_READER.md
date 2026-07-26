# Kai Code Audit — Email Reader Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-MAIL-001 | CRITICAL | Mailbox content is exposed through unauthenticated endpoints |
| KAI-MAIL-002 | HIGH | Forced mailbox polling is unauthenticated and caller-selectable |
| KAI-MAIL-003 | HIGH | Message read status is derived from the wrong data source |
| KAI-MAIL-004 | HIGH | Stale mailbox contents remain available indefinitely after polling failure |
| KAI-MAIL-005 | HIGH | IMAP operations have no explicit network timeout |
| KAI-MAIL-006 | MEDIUM | Health reports `ok` when unconfigured or polling has failed |
| KAI-MAIL-007 | MEDIUM | Raw IMAP errors are returned to clients and exposed in health |
| KAI-MAIL-008 | MEDIUM | Folder parameters on read endpoints do not correspond to cached folder state |
| KAI-MAIL-009 | MEDIUM | Cache and unread count are process-local and non-durable |
| KAI-MAIL-010 | MEDIUM | Poll worker is not supervised for unexpected termination |
| KAI-MAIL-011 | MEDIUM | Configuration limits are not validated |
| KAI-MAIL-012 | MEDIUM | Error-budget telemetry is exposed but never populated |

---

## Email reader: `email-reader/app.py`

### KAI-MAIL-001 — CRITICAL — Unauthenticated mailbox disclosure
**Issue:** `GET /inbox` and `GET /unread` return email subjects, sender identities, dates and up to 1,000 characters of message body without authentication or authorisation.  
**Risk:** Any caller with network access can read sensitive personal, commercial or security-related email content.  
**Recommendation:** Require strong user/service authentication, mailbox-scoped authorisation and strict network isolation before returning any message metadata or content.  
**Status:** OPEN — immediate remediation required

### KAI-MAIL-002 — HIGH — Caller-controlled forced mailbox polling
**Issue:** `POST /refresh` is unauthenticated and accepts an arbitrary `folder` value that is passed to `conn.select`.  
**Risk:** Reachable callers can trigger repeated IMAP connections, enumerate or poll other accessible folders and consume mailbox/server resources.  
**Recommendation:** Restrict refresh to authenticated operators, allowlist folders and rate-limit polling.  
**Status:** OPEN

### KAI-MAIL-003 — HIGH — Read status is calculated incorrectly
**Issue:** Each message sets `read` from `msg.get("Flags")`. IMAP flags are response metadata, not ordinary RFC822 message headers returned by `email.message_from_bytes`.  
**Risk:** Messages are generally treated as unread regardless of actual `\\Seen` state. `/unread` samples therefore conflict with the server unread count and can drive false alerts or duplicate actions.  
**Recommendation:** Request and parse IMAP FLAGS from the fetch response or perform a dedicated flag query keyed by stable UID.  
**Status:** OPEN

### KAI-MAIL-004 — HIGH — Stale email remains exposed indefinitely
**Issue:** Poll failure updates `_poll_error` but preserves the previous inbox cache and unread count. Read endpoints expose it without age limit or stale marker.  
**Risk:** Downstream logic can treat outdated mailbox state as current during prolonged authentication or network failure.  
**Recommendation:** Publish freshness explicitly and reject or degrade stale data after a defined maximum age.  
**Status:** OPEN

### KAI-MAIL-005 — HIGH — IMAP calls can block indefinitely
**Issue:** `imaplib.IMAP4_SSL` is created without an explicit timeout, and login, select, search and per-message fetch operations have no bounded deadline. Running them in an executor prevents event-loop blocking but does not stop stuck worker threads.  
**Risk:** A slow or malicious server can exhaust executor threads, stall refresh calls and degrade the service indefinitely.  
**Recommendation:** Apply socket and operation deadlines, cap total poll duration and isolate blocking work in a bounded worker pool.  
**Status:** OPEN

### KAI-MAIL-006 — MEDIUM — Health is not readiness-aware
**Issue:** `/health` always returns `status: ok`, including stub mode, before first poll and after polling errors.  
**Risk:** Watchdogs report a working email service when no mailbox data can be obtained.  
**Recommendation:** Separate liveness, credential configuration, connection readiness and data freshness.  
**Status:** OPEN

### KAI-MAIL-007 — MEDIUM — Internal errors are disclosed
**Issue:** Raw exception strings are stored in `_poll_error`, returned through `/health`, and included directly in 502 responses from `/refresh`.  
**Risk:** IMAP host, TLS, authentication and mailbox details may be exposed.  
**Recommendation:** Return stable error codes and protected trace identifiers only.  
**Status:** OPEN

### KAI-MAIL-008 — MEDIUM — Folder parameters misrepresent cache contents
**Issue:** Background polling always populates the cache from `INBOX`. `GET /inbox?folder=...` and `GET /unread?folder=...` merely echo the requested folder while returning the same INBOX-derived cache. A forced refresh can replace the global cache with another folder, after which later read calls can label it as any folder.  
**Risk:** Consumers cannot determine which mailbox folder the returned messages actually came from.  
**Recommendation:** Key cache state by canonical folder and return the actual source folder/version.  
**Status:** OPEN

### KAI-MAIL-009 — MEDIUM — State is worker-local and non-durable
**Issue:** Inbox cache, unread count and poll timestamps are module-level variables.  
**Risk:** Multiple workers expose inconsistent mailbox snapshots, and all state disappears on restart.  
**Recommendation:** Use a shared protected cache with explicit snapshot identity and retention controls.  
**Status:** OPEN

### KAI-MAIL-010 — MEDIUM — Poll task is not supervised
**Issue:** The lifespan retains the task only for cancellation. It does not detect unexpected exit, restart it or fail readiness.  
**Risk:** Polling can stop permanently while cached messages remain available.  
**Recommendation:** Supervise mandatory background tasks and expose their state.  
**Status:** OPEN

### KAI-MAIL-011 — MEDIUM — Configuration lacks bounds
**Issue:** Poll interval, maximum fetch count and port are parsed directly with no range validation.  
**Risk:** Invalid values can crash startup, create tight polling loops or cause excessive mailbox downloads.  
**Recommendation:** Validate startup configuration against explicit minimum and maximum values.  
**Status:** OPEN

### KAI-MAIL-012 — MEDIUM — Error-budget metrics are inert
**Issue:** `ErrorBudget` is exposed through `/metrics`, but no poll or endpoint result is recorded.  
**Risk:** Monitoring lacks a reliable signal of mailbox polling availability.  
**Recommendation:** Record classified successes, failures and latency.  
**Status:** OPEN

---

## Batch totals

- Findings: **12**
- Critical: **1**
- High: **4**
- Medium: **7**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **257**
- Critical: **30**
- High: **111**
- Medium: **114**
- Low: **2**

## Files materially reviewed in this batch

`email-reader/app.py`.
