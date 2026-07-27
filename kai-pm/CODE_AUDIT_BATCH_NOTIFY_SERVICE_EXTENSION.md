# Kai Code Audit — Notify Service Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_NOTIFY_SERVICE.md`. The existing 11 findings are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-NOTIFYX-001 | HIGH | The deployed container has no host D-Bus/display integration, so desktop notifications are normally non-functional |
| KAI-NOTIFYX-002 | HIGH | User-controlled title/body are passed to `notify-send` without an end-of-options delimiter |
| KAI-NOTIFYX-003 | HIGH | No rate limit or global subprocess-concurrency policy protects notification dispatch |
| KAI-NOTIFYX-004 | HIGH | Notification work is submitted to the shared default executor without a bounded application queue |
| KAI-NOTIFYX-005 | HIGH | Health executes `notify-send` synchronously on the async event-loop thread |
| KAI-NOTIFYX-006 | HIGH | Queue fallback returns `ok: true` without durable persistence or operator acknowledgement |
| KAI-NOTIFYX-007 | HIGH | Full titles are logged and may contain secrets, PII, newlines or control characters |
| KAI-NOTIFYX-008 | HIGH | Desktop and Dashboard surfaces receive untrusted markup/control text without a safe canonical format |
| KAI-NOTIFYX-009 | HIGH | Dispatch has no idempotency key or duplicate-suppression policy |
| KAI-NOTIFYX-010 | HIGH | Notification creation, fallback and acknowledgement have no immutable provenance audit |
| KAI-NOTIFYX-011 | HIGH | The service has no source-service, recipient, purpose or notification-class scope |
| KAI-NOTIFYX-012 | MEDIUM | Public health exposes delivery-channel availability and pending queue depth |
| KAI-NOTIFYX-013 | MEDIUM | Public metrics expose request telemetry without administrative authentication |
| KAI-NOTIFYX-014 | MEDIUM | Missing shared-runtime imports silently replace structured logging and metrics with no-op fallbacks |
| KAI-NOTIFYX-015 | MEDIUM | Notification timestamps use wall-clock floats without a source event or sequence |
| KAI-NOTIFYX-016 | MEDIUM | `unread_only=false` exposes already dismissed notification content |
| KAI-NOTIFYX-017 | MEDIUM | Dismissal marks entries read but does not remove or securely erase their content |
| KAI-NOTIFYX-018 | MEDIUM | `dismiss_all` reports total queue length rather than the number newly dismissed |
| KAI-NOTIFYX-019 | MEDIUM | Pending notifications have no expiry, retention or staleness policy |
| KAI-NOTIFYX-020 | MEDIUM | Pending retrieval has no pagination or aggregate response-byte limit |
| KAI-NOTIFYX-021 | MEDIUM | Unknown urgency values silently become normal |
| KAI-NOTIFYX-022 | MEDIUM | Invalid timeout values are silently clamped/defaulted without caller feedback |
| KAI-NOTIFYX-023 | MEDIUM | The service has no lifespan-owned executor, shutdown drain or pending-queue persistence |
| KAI-NOTIFYX-024 | MEDIUM | `notify-send` inherits the complete service environment |
| KAI-NOTIFYX-025 | MEDIUM | Captured `notify-send` stdout/stderr are discarded and cannot explain delivery failure |

---

### KAI-NOTIFYX-001 — HIGH — Desktop delivery is not connected
**Issue:** The image installs `libnotify-bin`, but Compose mounts no user-session D-Bus socket and configures no `DBUS_SESSION_BUS_ADDRESS` or display integration.  
**Risk:** `notify-send` normally cannot reach the host desktop, so the service silently becomes an in-memory Dashboard queue rather than the documented primary OS notification channel.  
**Recommendation:** either implement an authenticated host notification bridge or explicitly disable/mark desktop delivery unavailable.  
**Status:** OPEN

### KAI-NOTIFYX-002 — HIGH — `notify-send` option injection
**Issue:** Attacker-controlled title/body follow command options without a `--` end-of-options marker. GNU option parsing may interpret option-like values as flags/hints rather than content.  
**Risk:** A caller can alter app name, icon, hints or notification presentation beyond the declared API fields.  
**Recommendation:** insert `--` before user content and validate a safe presentation schema.  
**Status:** OPEN

### KAI-NOTIFYX-003 — HIGH — Missing dispatch admission control
Every reachable caller can initiate notification processing with no rate, quota or semaphore.

### KAI-NOTIFYX-004 — HIGH — Unbounded executor backlog
`run_in_executor(None, ...)` uses the process default executor; the service adds no bounded work queue before it.

### KAI-NOTIFYX-005 — HIGH — Blocking health probe
`subprocess.run()` executes directly inside the async health handler.

### KAI-NOTIFYX-006 — HIGH — Queueing is reported as delivery success
A volatile fallback append returns `ok:true` even though the message may be lost on restart, silent eviction or a zero-capacity queue.

### KAI-NOTIFYX-007 — HIGH — Sensitive/log-injection title logging
Titles are logged verbatim on both desktop success and queue fallback.

### KAI-NOTIFYX-008 — HIGH — No canonical safe-notification rendering
The same untrusted title/body may be interpreted by libnotify/Desktop and Dashboard HTML/notification views without one provenance/escaping format.

### KAI-NOTIFYX-009 — HIGH — Duplicate delivery ambiguity
Repeated callers and committed-but-uncertain subprocess outcomes cannot be reconciled against an idempotent operation ID.

### KAI-NOTIFYX-010 — HIGH — Missing provenance audit
No tamper-evident event identifies source service/actor, body digest, channel outcome, queue ID and acknowledgement actor.

### KAI-NOTIFYX-011 — HIGH — No routing authority
All senders share one global desktop/queue channel; the service cannot restrict safety, financial, system or personal notification classes by origin/recipient.

### KAI-NOTIFYX-012 — MEDIUM — Public channel health
Health reveals binary availability and queue depth.

### KAI-NOTIFYX-013 — MEDIUM — Public telemetry
Metrics requires no administrative identity.

### KAI-NOTIFYX-014 — MEDIUM — Silent runtime downgrade
If `common.runtime` cannot import, local basic logging and a no-op ErrorBudget are substituted while health remains normal.

### KAI-NOTIFYX-015 — MEDIUM — Weak timestamp identity
`time.time()` has no triggering event ID, timezone, trace or monotonic sequence.

### KAI-NOTIFYX-016 — MEDIUM — Dismissed data remains exposed
The optional query returns read entries with full content.

### KAI-NOTIFYX-017 — MEDIUM — Dismissal is not erasure
Only `entry["read"]` changes.

### KAI-NOTIFYX-018 — MEDIUM — Misleading clear count
The endpoint returns the length of the queue, including entries already marked read.

### KAI-NOTIFYX-019 — MEDIUM — No retention lifecycle
Entries persist until maxlen eviction, dismissal-state retention or restart.

### KAI-NOTIFYX-020 — MEDIUM — Unpaged queue response
Every retained entry is returned in one response; item sizes remain unbounded under the original finding.

### KAI-NOTIFYX-021 — MEDIUM — Urgency coercion
Unknown values are converted to `normal` rather than rejected/audited.

### KAI-NOTIFYX-022 — MEDIUM — Timeout coercion
Values below/above limits or null/zero are silently changed.

### KAI-NOTIFYX-023 — MEDIUM — Missing lifecycle ownership
No lifespan controls executor work, queue persistence or graceful completion.

### KAI-NOTIFYX-024 — MEDIUM — Child environment exposure
The subprocess inherits all environment variables available to Notify Service.

### KAI-NOTIFYX-025 — MEDIUM — Lost delivery diagnostics
Captured stdout/stderr are ignored even when the return code is nonzero.

---

## Batch totals

- Findings: **25**
- Critical: **0**
- High: **11**
- Medium: **14**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,155**
- Critical: **189**
- High: **1,069**
- Medium: **894**
- Low: **3**

## Files materially reviewed

`output/notify/app.py`, `output/notify/Dockerfile`, the existing Notify audit and deployment/integration with Dashboard, Monitor, Supervisor and House Doctor.
