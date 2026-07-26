# Kai Code Audit — Notify Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-NOTIFY-001 | CRITICAL | Unauthenticated callers can display arbitrary desktop notifications |
| KAI-NOTIFY-002 | HIGH | Pending notification content is exposed without authentication |
| KAI-NOTIFY-003 | HIGH | Unauthenticated callers can dismiss individual or all pending notifications |
| KAI-NOTIFY-004 | HIGH | Notification title and body lengths are unbounded |
| KAI-NOTIFY-005 | MEDIUM | Queue overflow silently discards older notifications |
| KAI-NOTIFY-006 | MEDIUM | Queue and identifiers are process-local and volatile |
| KAI-NOTIFY-007 | MEDIUM | Notification ID allocation is not synchronised |
| KAI-NOTIFY-008 | MEDIUM | Desktop delivery success is inferred only from process exit code |
| KAI-NOTIFY-009 | MEDIUM | Health can report notify-send available when the command failed or timed out |
| KAI-NOTIFY-010 | MEDIUM | Error-budget recording passes a Boolean rather than the HTTP status code |
| KAI-NOTIFY-011 | MEDIUM | Configuration limits are not validated |

---

## Notify service: `output/notify/app.py`

### KAI-NOTIFY-001 — CRITICAL — Unauthenticated desktop notification injection
**Issue:** `POST /notify` requires no authentication or authorisation. Caller-controlled title, body, urgency and timeout are passed directly as arguments to the host `notify-send` command.  
**Risk:** Any network-reachable caller can create trusted-looking desktop pop-ups, impersonate system alerts, socially engineer the operator and repeatedly interrupt the workstation. Although argument-list invocation avoids shell expansion, the user-visible content remains fully attacker-controlled.  
**Recommendation:** Restrict notification creation to authenticated, authorised service identities and label messages with verified provenance.  
**Status:** OPEN — immediate remediation required

### KAI-NOTIFY-002 — HIGH — Pending messages are publicly readable
**Issue:** `GET /pending` exposes notification titles, bodies, urgency, timestamps and read state without authentication.  
**Risk:** Operational alerts, personal reminders and sensitive message content can be read by any reachable caller.  
**Recommendation:** Require user-scoped access and minimise retained notification content.  
**Status:** OPEN

### KAI-NOTIFY-003 — HIGH — Notification suppression is unauthenticated
**Issue:** `DELETE /pending/{notification_id}` and `DELETE /pending` require no authentication.  
**Risk:** A caller can hide one or all dashboard notifications before the operator sees them, suppressing security or operational warnings.  
**Recommendation:** Require authenticated acknowledgement and preserve an immutable delivery/audit record.  
**Status:** OPEN

### KAI-NOTIFY-004 — HIGH — Message sizes are unbounded
**Issue:** `NotifyRequest.title` and `body` have no maximum lengths. They are passed to a subprocess, retained in memory, logged by title and returned through the queue.  
**Risk:** Oversized concurrent requests can consume memory, exceed operating-system argument limits, flood logs and degrade desktop or dashboard rendering.  
**Recommendation:** Enforce strict per-field and aggregate request limits at the API boundary.  
**Status:** OPEN

### KAI-NOTIFY-005 — MEDIUM — Queue overflow silently loses alerts
**Issue:** `_pending` is a `deque(maxlen=MAX_PENDING)`. Appending after capacity is reached automatically drops the oldest entry without warning, audit record or delivery escalation.  
**Risk:** Notification floods or normal bursts can silently evict important unread alerts.  
**Recommendation:** Use durable prioritised delivery with explicit overflow handling and metrics.  
**Status:** OPEN

### KAI-NOTIFY-006 — MEDIUM — Queue state is worker-local and non-durable
**Issue:** Pending notifications, read state and the counter are module-level memory only.  
**Risk:** Restart erases messages, multiple workers expose different queues and duplicate IDs, and acknowledgement state is inconsistent.  
**Recommendation:** Use a shared durable notification store or enforce a single authoritative dispatcher.  
**Status:** OPEN

### KAI-NOTIFY-007 — MEDIUM — Identifier allocation races
**Issue:** `_counter += 1` and queue insertion are not protected by a lock. Multiple async requests complete executor work and can interleave around the shared counter and queue.  
**Risk:** IDs can become inconsistent across concurrent requests or workers, undermining targeted dismissal and auditability.  
**Recommendation:** Allocate IDs transactionally in shared storage.  
**Status:** OPEN

### KAI-NOTIFY-008 — MEDIUM — Delivery acknowledgement is weak
**Issue:** Desktop delivery is considered successful solely when `notify-send` exits with code zero. There is no confirmation that a notification server displayed or retained the message.  
**Risk:** The API returns `ok: true` and does not queue the message even when no operator-visible notification was actually delivered.  
**Recommendation:** Use an acknowledgement-capable notification transport or retain every message until operator acknowledgement.  
**Status:** OPEN

### KAI-NOTIFY-009 — MEDIUM — Health check can produce false readiness
**Issue:** `/health` runs `notify-send --version` but does not inspect its return code. It also fails to catch `subprocess.TimeoutExpired`.  
**Risk:** A failing binary can be reported as available, while a timeout produces an unhandled 500 response.  
**Recommendation:** Validate return code and handle all bounded subprocess outcomes.  
**Status:** OPEN

### KAI-NOTIFY-010 — MEDIUM — Error-budget input type is inconsistent
**Issue:** Middleware calls `budget.record(response.status_code >= 500)`, passing `True` or `False` rather than the response status code used by other services and implied by the runtime interface. Exceptions raised before a response are also not recorded because the middleware lacks an exception branch.  
**Risk:** Reliability metrics can classify outcomes incorrectly or omit failed requests.  
**Recommendation:** Record the actual status code and explicitly capture exceptions.  
**Status:** OPEN

### KAI-NOTIFY-011 — MEDIUM — Configuration lacks bounds
**Issue:** `NOTIFY_MAX_PENDING`, default timeout and port are parsed directly. Zero/negative queue sizes or invalid values can fail startup or create unexpected queue behaviour.  
**Risk:** Misconfiguration disables retention, crashes the service or changes notification behaviour silently.  
**Recommendation:** Validate a typed startup configuration with explicit ranges.  
**Status:** OPEN

---

## Batch totals

- Findings: **11**
- Critical: **1**
- High: **3**
- Medium: **7**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **352**
- Critical: **38**
- High: **140**
- Medium: **171**
- Low: **3**

## Files materially reviewed in this batch

`output/notify/app.py`.
