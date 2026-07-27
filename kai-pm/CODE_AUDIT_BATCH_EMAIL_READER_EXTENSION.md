# Kai Code Audit — Email Reader Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_EMAIL_READER.md`. The existing 18 findings are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-MAILX-001 | HIGH | The service uses unstable IMAP sequence numbers rather than persistent UIDs |
| KAI-MAILX-002 | HIGH | UIDVALIDITY is never captured, so mailbox resets and reused identifiers cannot be detected |
| KAI-MAILX-003 | HIGH | “Recent” messages are selected by sequence position rather than authenticated message date or internal date |
| KAI-MAILX-004 | HIGH | IMAP SELECT, SEARCH and FETCH status codes are ignored |
| KAI-MAILX-005 | HIGH | A failed folder selection can flow into later commands instead of producing a typed folder error |
| KAI-MAILX-006 | HIGH | Every poll repeatedly downloads the same complete recent messages with no incremental UID checkpoint |
| KAI-MAILX-007 | HIGH | Message sender identity is accepted from the unauthenticated `From` header without SPF, DKIM or DMARC provenance |
| KAI-MAILX-008 | HIGH | Subjects, sender names, dates and snippets retain untrusted control characters and markup |
| KAI-MAILX-009 | HIGH | Mail content is exposed to Dashboard rendering without a canonical safe-text or provenance schema |
| KAI-MAILX-010 | HIGH | Header charset lookup failures can abort the complete mailbox poll |
| KAI-MAILX-011 | HIGH | Attachment exclusion is case-sensitive and can include attachment text as the displayed body snippet |
| KAI-MAILX-012 | HIGH | Body decoding occurs before the 1,000-character snippet limit and can expand compressed/encoded content substantially |
| KAI-MAILX-013 | HIGH | Manual refreshes use the shared default executor with no bounded IMAP-work queue |
| KAI-MAILX-014 | HIGH | Cancelling the poll coroutine cannot stop an in-progress blocking IMAP thread |
| KAI-MAILX-015 | HIGH | IMAP connections are recreated and reauthenticated for every poll and refresh |
| KAI-MAILX-016 | HIGH | Mailbox snapshots contain no source UID, UIDVALIDITY, message hash or immutable revision |
| KAI-MAILX-017 | HIGH | No audit trail records who requested a folder poll or which messages were exposed |
| KAI-MAILX-018 | MEDIUM | One RFC822 FETCH response can append duplicate message entries when multiple tuple parts are returned |
| KAI-MAILX-019 | MEDIUM | Message dates are returned as raw unparsed header strings |
| KAI-MAILX-020 | MEDIUM | The service does not validate IMAP server capabilities or expected mailbox semantics |
| KAI-MAILX-021 | MEDIUM | `/unread` combines a server-wide unread count with a sample inferred from a different cached subset |
| KAI-MAILX-022 | MEDIUM | No per-message size, MIME-part size or decoded-body byte budget is reported in the snapshot |
| KAI-MAILX-023 | MEDIUM | Empty or HTML-only messages are represented as empty snippets without a content-state marker |
| KAI-MAILX-024 | MEDIUM | The cached snapshot has no ETag, generation number or compare-and-swap publication contract |
| KAI-MAILX-025 | MEDIUM | Polling uses a fixed interval with no failure backoff, jitter or mailbox-rate-limit handling |
| KAI-MAILX-026 | MEDIUM | Read endpoints provide no per-message authentication, phishing or remote-content risk metadata |
| KAI-MAILX-027 | MEDIUM | Missing shared runtime imports silently replace structured telemetry with no-op fallbacks |
| KAI-MAILX-028 | MEDIUM | The service has no dedicated IMAP executor, graceful thread drain or credential-rotation lifecycle |

---

## High-severity findings

### KAI-MAILX-001 — HIGH — Sequence numbers are exposed as identities
**Issue:** `conn.search()` and `conn.fetch()` use message sequence numbers; the returned `id` is `msg_id.decode()`. Sequence numbers change when messages are expunged or the mailbox changes.  
**Risk:** A cached or downstream ID can later refer to a different email, undermining deduplication, attribution and incident evidence.  
**Recommendation:** use IMAP UIDs and bind them to the selected mailbox’s UIDVALIDITY.  
**Status:** OPEN

### KAI-MAILX-002 — HIGH — Mailbox identity resets are invisible
The service never requests/stores UIDVALIDITY, UIDNEXT or a mailbox revision.

### KAI-MAILX-003 — HIGH — Sequence position is labelled recency
The last sequence numbers are assumed to be recent. They are not sorted by parsed Date/INTERNALDATE and may be unrelated to actual chronology.

### KAI-MAILX-004 — HIGH — IMAP command failures are not checked
Return status values from `select`, both `search` calls and every `fetch` are discarded. The code proceeds directly into response-data indexing/parsing.

### KAI-MAILX-005 — HIGH — Folder failure lacks a safe boundary
A nonexistent, inaccessible or malformed folder does not produce a specific stop immediately after `select`; subsequent failures collapse into a generic poll error while stale data remains.

### KAI-MAILX-006 — HIGH — Complete repeated mailbox downloads
Every two-minute poll performs `SEARCH ALL` and downloads full RFC822 content for up to `MAX_FETCH` messages, even when no message changed.

### KAI-MAILX-007 — HIGH — Sender spoofing presented as identity
The returned `from` field is decoded directly from the RFC822 header. The service does not expose authenticated envelope sender, SPF, DKIM, DMARC or trusted-source status.

### KAI-MAILX-008 — HIGH — Untrusted presentation characters
Decoded headers and body snippets are not stripped of NUL, bidi, newline or other control characters and may contain HTML/Markdown-like text.

### KAI-MAILX-009 — HIGH — No safe downstream mail schema
Dashboard consumes these fields as ordinary display content; Email Reader supplies no plain-text guarantee, encoding marker, trusted-link policy or provenance flags.

### KAI-MAILX-010 — HIGH — Unknown charset aborts the generation
`part.decode(charset or "utf-8")` and header decoding do not catch `LookupError` for unknown/malformed charset labels, so one message can fail the entire poll.

### KAI-MAILX-011 — HIGH — Attachment text can enter snippets
The check is `"attachment" not in disp` and is case-sensitive. A disposition such as `Attachment` may be treated as ordinary body text.

### KAI-MAILX-012 — HIGH — Limit applies after decoding
Transfer decoding and charset decoding materialise the whole selected body part before slicing to 1,000 characters.

### KAI-MAILX-013 — HIGH — Unbounded shared-executor IMAP admission
Each manual refresh submits a blocking poll to the event loop’s default executor. There is no semaphore or dedicated worker-count/queue bound.

### KAI-MAILX-014 — HIGH — Blocking work survives cancellation
Lifespan cancels `_poll_task`, but an IMAP operation already running in its executor thread continues until the socket/library returns.

### KAI-MAILX-015 — HIGH — Reauthentication churn
Every scheduled/manual poll creates a new TLS connection, logs in, selects/searches/fetches and logs out rather than using a controlled connection/session strategy.

### KAI-MAILX-016 — HIGH — Snapshots lack immutable provenance
Records contain only sequence ID and raw display fields; no mailbox identity, UID, UIDVALIDITY, content hash, poll generation or internal date is returned.

### KAI-MAILX-017 — HIGH — Missing access audit
There is no tamper-evident record of caller identity, requested folder, poll generation, exposed message IDs or downstream reads.

---

## Medium-severity findings

### KAI-MAILX-018 — MEDIUM — Duplicate tuple handling
The code appends a message for every tuple in `msg_data`; it does not ensure exactly one RFC822 payload per requested ID.

### KAI-MAILX-019 — MEDIUM — Raw date semantics
The Date header is returned without parsing, timezone normalisation, invalid-date state or distinction from server INTERNALDATE.

### KAI-MAILX-020 — MEDIUM — Capability assumptions
The service does not verify IMAP4rev capability, UID support, folder encoding or response features before relying on command semantics.

### KAI-MAILX-021 — MEDIUM — Mismatched unread evidence
`unread_count` comes from server `UNSEEN` across the folder; the sample is derived from the recent cache’s broken local `read` field.

### KAI-MAILX-022 — MEDIUM — No decoded-size telemetry
Snapshots omit original/decoded sizes, MIME-part count and whether content was truncated or skipped for safety.

### KAI-MAILX-023 — MEDIUM — Empty-content ambiguity
HTML-only, encrypted, malformed or attachment-only messages all look like an ordinary empty snippet.

### KAI-MAILX-024 — MEDIUM — No snapshot generation contract
Consumers cannot request/compare one atomic revision or detect that data changed between reads.

### KAI-MAILX-025 — MEDIUM — Weak retry cadence
All failures wait exactly the normal poll interval; there is no jitter, transient/permanent classification, server throttling or credential-failure lockout.

### KAI-MAILX-026 — MEDIUM — No mail-risk metadata
The API does not indicate suspicious sender mismatch, external links, remote images, attachment presence, encrypted content or authentication results.

### KAI-MAILX-027 — MEDIUM — Silent runtime downgrade
If common runtime imports fail, logging becomes basic and ErrorBudget no-op while health remains normal.

### KAI-MAILX-028 — MEDIUM — Missing IMAP lifecycle ownership
No dedicated executor/connection manager, shutdown drain, credential rotation or one-authoritative-poller contract exists.

---

## Batch totals

- Findings: **28**
- Critical: **0**
- High: **17**
- Medium: **11**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,183**
- Critical: **189**
- High: **1,086**
- Medium: **905**
- Low: **3**

## Files materially reviewed

`email-reader/app.py`, the existing Email Reader audit, Email Reader deployment and Dashboard/mailbox integration.
