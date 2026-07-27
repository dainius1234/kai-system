# Kai Code Audit — Email Reader Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-MAIL-001 | CRITICAL | Mailbox content is exposed through unauthenticated endpoints |
| KAI-MAIL-002 | CRITICAL | Unauthenticated callers can poll arbitrary IMAP folders and replace the shared mailbox cache |
| KAI-MAIL-003 | HIGH | Message read status is derived from the wrong data source |
| KAI-MAIL-004 | HIGH | Stale mailbox contents remain available indefinitely after polling failure |
| KAI-MAIL-005 | HIGH | IMAP operations have no explicit network timeout |
| KAI-MAIL-006 | HIGH | Configurable IMAP destination receives mailbox credentials without host validation |
| KAI-MAIL-007 | HIGH | Complete RFC822 messages are fetched and decoded before the snippet limit is applied |
| KAI-MAIL-008 | HIGH | Forced mailbox polling is unauthenticated and resource-unbounded |
| KAI-MAIL-009 | HIGH | Background and manual polls can race and overwrite one another’s folder results |
| KAI-MAIL-010 | MEDIUM | Folder parameters on read endpoints do not correspond to cached folder state |
| KAI-MAIL-011 | MEDIUM | Every message ID in the selected folder is materialised before the fetch limit is applied |
| KAI-MAIL-012 | MEDIUM | Header and MIME parsing complexity and field lengths are unbounded |
| KAI-MAIL-013 | MEDIUM | Cache and unread count are process-local and non-durable |
| KAI-MAIL-014 | MEDIUM | Poll task shutdown and supervision are incomplete |
| KAI-MAIL-015 | MEDIUM | Health reports `ok` when unconfigured, never polled or failed |
| KAI-MAIL-016 | MEDIUM | Raw IMAP errors are returned to clients and exposed in health |
| KAI-MAIL-017 | MEDIUM | Error-budget telemetry is exposed but never populated |
| KAI-MAIL-018 | MEDIUM | Configuration values and limits are not validated |

---

## Email reader: `email-reader/app.py`

### KAI-MAIL-001 — CRITICAL — Unauthenticated mailbox disclosure
**Issue:** `GET /inbox` and `GET /unread` return message IDs, subjects, sender identities, dates and up to 1,000 characters of body content without authentication or authorisation.  
**Risk:** Any caller with network access can read sensitive personal, commercial or security-related email content, including one-time codes and confidential correspondence.  
**Recommendation:** Require strong owner-scoped authentication, mailbox-specific authorisation and strict data minimisation before returning message metadata or content.  
**Status:** OPEN — immediate remediation required

### KAI-MAIL-002 — CRITICAL — Arbitrary-folder polling poisons the shared cache
**Issue:** `POST /refresh?folder=...` accepts any folder string and calls `_poll_imap(folder)`. The result replaces the single global `_inbox_cache` and `_unread_count`; no source folder is stored with the cache.  
**Risk:** Any reachable caller can enumerate or guess private folders, force their contents into the shared cache and retrieve them through `/inbox` or `/unread`. One caller also changes the mailbox view for every other user and downstream consumer.  
**Recommendation:** Remove public folder selection, allow only authenticated approved folders and partition all snapshots by owner and canonical folder.  
**Status:** OPEN — immediate remediation required

### KAI-MAIL-003 — HIGH — Read status is calculated incorrectly
**Issue:** Each message sets `read` from `msg.get("Flags")`. IMAP flags are response metadata, not ordinary RFC822 headers returned by `email.message_from_bytes`.  
**Risk:** Messages are generally treated as unread regardless of actual `\Seen` state. `/unread` samples therefore conflict with the server unread count and can drive false alerts or duplicate actions.  
**Recommendation:** Request and parse IMAP FLAGS explicitly alongside stable UIDs.  
**Status:** OPEN

### KAI-MAIL-004 — HIGH — Stale email remains exposed indefinitely
**Issue:** Poll failure updates `_poll_error` but preserves the previous inbox cache and unread count. Read endpoints expose the stale content without age limit, error or freshness marker.  
**Risk:** Downstream logic can treat outdated mailbox state as current during prolonged authentication or network failure, while old private content remains publicly readable.  
**Recommendation:** Publish snapshot age/error state on every response and expire sensitive cached content after a short maximum age.  
**Status:** OPEN

### KAI-MAIL-005 — HIGH — IMAP calls can block indefinitely
**Issue:** `imaplib.IMAP4_SSL` is created without a timeout, and login, select, search, fetch and logout have no bounded operation deadline. Executor use prevents event-loop blocking but does not terminate stuck threads.  
**Risk:** A slow or malicious server can exhaust executor threads, stall refresh requests and permanently degrade polling.  
**Recommendation:** Apply strict socket and total-poll deadlines and isolate IMAP work in a bounded dedicated worker pool.  
**Status:** OPEN

### KAI-MAIL-006 — HIGH — Credential-bearing destination is configuration-controlled
**Issue:** `MAIL_HOST`, port, username and password are accepted directly from environment configuration and passed to `IMAP4_SSL` without an application allowlist or pinned provider identity. The minimal Compose deployment supplies the password as a normal environment variable.  
**Risk:** Compromised or mistaken deployment configuration can transmit mailbox credentials to an unintended IMAP server and exposes the secret to process-environment readers.  
**Recommendation:** Pin approved TLS hosts/certificates and use secret-managed least-privilege credentials rather than ordinary environment interpolation.  
**Status:** OPEN

### KAI-MAIL-007 — HIGH — Message limit is applied after full download and parsing
**Issue:** `conn.fetch(msg_id, "(RFC822)")` downloads the complete message; `message_from_bytes`, MIME traversal and payload decoding process the full content before `_get_body` slices the resulting text to 1,000 characters.  
**Risk:** Large messages or MIME bombs can consume excessive bandwidth, memory and CPU despite the small returned snippet.  
**Recommendation:** Fetch bounded headers and partial body ranges, reject excessive message sizes and enforce MIME depth/part limits.  
**Status:** OPEN

### KAI-MAIL-008 — HIGH — Public forced polling is resource-unbounded
**Issue:** `/refresh` is unauthenticated and performs login, folder selection, full ID search, unread search and one complete RFC822 fetch per selected message. There is no rate limit, quota, cooldown or global concurrency limit.  
**Risk:** Repeated callers can consume mailbox/server quotas, worker threads and network capacity while continuously replacing shared state.  
**Recommendation:** Restrict refresh to a protected scheduler/operator and enforce strict rate and concurrency controls.  
**Status:** OPEN

### KAI-MAIL-009 — HIGH — Poll generations race
**Issue:** The background loop and any number of manual refresh requests can execute concurrently. Each publishes cache, unread count, timestamp and error state independently with no lock or generation identity.  
**Risk:** A slower older poll or different-folder refresh can overwrite a newer intended result, producing inconsistent and mislabelled mailbox state.  
**Recommendation:** Use one serialised polling coordinator and atomically publish folder-scoped versioned snapshots.  
**Status:** OPEN

### KAI-MAIL-010 — MEDIUM — Folder labels misrepresent cache contents
**Issue:** `GET /inbox?folder=...` and `GET /unread?folder=...` merely echo the requested folder while returning the same global cache. The cache may contain INBOX or the last manually refreshed folder.  
**Risk:** Consumers cannot determine which mailbox folder the returned messages actually came from.  
**Recommendation:** Remove the parameter or return only authenticated canonical folder-specific snapshots with actual source metadata.  
**Status:** OPEN

### KAI-MAIL-011 — MEDIUM — Folder search materialises every message ID
**Issue:** `conn.search(None, "ALL")` returns and stores all message IDs before only the final `MAX_FETCH` values are selected.  
**Risk:** Very large folders consume server/client processing and memory even when only a small recent sample is required.  
**Recommendation:** Use server-side bounded/sorted UID searches or incremental checkpoints.  
**Status:** OPEN

### KAI-MAIL-012 — MEDIUM — Header and MIME complexity is unbounded
**Issue:** Subjects and sender fields have no length cap; header decoding, `message_from_bytes` and `msg.walk()` process arbitrary header counts, encodings, nesting and MIME parts.  
**Risk:** Malformed or intentionally complex messages can consume parser resources and inflate API responses.  
**Recommendation:** Enforce strict aggregate message, header, MIME-depth, part-count and field limits.  
**Status:** OPEN

### KAI-MAIL-013 — MEDIUM — State is worker-local and non-durable
**Issue:** Inbox cache, unread count, poll errors and timestamps are module-level variables.  
**Risk:** Multiple workers expose inconsistent mailbox snapshots, duplicate pollers and lose all state on restart.  
**Recommendation:** Use one polling authority and encrypted shared snapshot storage with short retention.  
**Status:** OPEN

### KAI-MAIL-014 — MEDIUM — Poll lifecycle is incomplete
**Issue:** Lifespan shutdown cancels `_poll_task` without awaiting it. Executor-backed IMAP work cannot be cancelled by cancelling the coroutine. Unexpected task termination is not restarted or reflected in readiness.  
**Risk:** Polling can stop permanently while stale messages remain available, and network work can continue after shutdown begins.  
**Recommendation:** Supervise the task, await shutdown and manage a dedicated executor/client lifecycle.  
**Status:** OPEN

### KAI-MAIL-015 — MEDIUM — Health is not readiness-aware
**Issue:** `/health` always returns `status: ok`, including stub mode, before the first successful poll and after polling errors.  
**Risk:** Watchdogs report a working email source when no current mailbox data can be obtained.  
**Recommendation:** Separate liveness, credential configuration, connection readiness, task state and snapshot freshness.  
**Status:** OPEN

### KAI-MAIL-016 — MEDIUM — Internal errors are disclosed
**Issue:** Raw exception strings are stored in `_poll_error`, returned through `/health` and included directly in 502 responses from `/refresh`.  
**Risk:** IMAP host, TLS, authentication, mailbox and parser details may be exposed.  
**Recommendation:** Return stable error codes and protected trace identifiers.  
**Status:** OPEN

### KAI-MAIL-017 — MEDIUM — Error-budget metrics are inert
**Issue:** `ErrorBudget` is exposed through `/metrics`, but no poll or endpoint result is recorded.  
**Risk:** Monitoring lacks a reliable signal of mailbox polling availability.  
**Recommendation:** Record classified request and poll successes, failures and latency.  
**Status:** OPEN

### KAI-MAIL-018 — MEDIUM — Configuration lacks bounds
**Issue:** Service/mail ports, poll interval and maximum fetch count are parsed directly without safe ranges. Zero/negative intervals can create tight loops; negative fetch values alter slicing semantics.  
**Risk:** Invalid values can crash startup, create uncontrolled polling or produce unexpected mailbox selection.  
**Recommendation:** Validate typed startup configuration against explicit minimum and maximum values.  
**Status:** OPEN

---

## Batch totals

- Findings: **18**
- Critical: **2**
- High: **7**
- Medium: **9**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **781**
- Critical: **86**
- High: **276**
- Medium: **416**
- Low: **3**

## Files materially reviewed in this batch

`email-reader/app.py` and the relevant `email-reader` deployment definition in `docker-compose.minimal.yml`.
