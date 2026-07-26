# Kai Code Audit Register — Continued Findings 2

Repository: `dainius1234/kai-system`  
Parent registers: `kai-pm/CODE_AUDIT_REGISTER.md`, `kai-pm/CODE_AUDIT_REGISTER_CONTINUED.md`  
Status: ACTIVE CONTINUATION  
Started: 26 July 2026

This file continues the numbered audit register from cumulative finding 52. It will be consolidated into the final full defect register at audit completion.

---

## Supervisor, recovery and watchdog: `supervisor/app.py`

### KAI-SUP-001 — CRITICAL — Manual recovery endpoint is unauthenticated

**Issue:** `POST /recover/{service_name}` can trigger a recovery action for any registered service without visible authentication, authorisation, CSRF protection, request signing or operator approval.

**Risk:** Any caller that can reach the supervisor may repeatedly invoke service recovery hooks, flush caches, reconnect dependencies or trigger implementation-specific restart behaviour. This creates a direct operational-control surface and denial-of-service path.

**Recommendation:** Restrict the endpoint to an authenticated administrative principal, require a signed and replay-protected request, apply per-service scopes and rate limits, and record immutable recovery audit events.

**Status:** OPEN — immediate remediation required

### KAI-SUP-002 — HIGH — Due reminders and tasks are marked fired without confirming notification delivery

**Issue:** `_fire_due_items()` posts to Telegram and then calls the corresponding `.../fire` endpoint without checking the Telegram response status or calling `raise_for_status()`.

**Risk:** A Telegram 4xx or 5xx response can still cause the reminder or task to be marked fired, permanently losing a notification that was never delivered.

**Recommendation:** Use a durable outbox with delivery state. Mark an item fired only after an accepted provider response, and retain retryable failures with idempotency keys and attempt metadata.

**Status:** OPEN

### KAI-SUP-003 — HIGH — Reminder and scheduled-task delivery is not transactionally idempotent

**Issue:** Notification delivery and state transition are separate network operations with no shared idempotency key or transactional boundary. A crash after Telegram accepts the message but before the `fire` update will resend the same item on the next sweep.

**Risk:** Users may receive duplicate reminders or scheduled-task alerts, while the inverse failure path can lose notifications entirely.

**Recommendation:** Introduce a durable outbox/inbox pattern, stable event IDs and provider-level idempotency. Record `pending → sent → acknowledged` transitions atomically where possible.

**Status:** OPEN

### KAI-SUP-004 — HIGH — Background loop task is not retained, supervised or restarted

**Issue:** Startup calls `asyncio.create_task(_background_loop())` without retaining the task reference, attaching a completion callback or implementing restart policy.

**Risk:** An unexpected uncaught exception or cancellation can terminate the only supervisory loop permanently. The process remains alive, but health checks merely report a stale heartbeat and no component restarts the loop.

**Recommendation:** Retain the task, monitor completion, expose failure reason, restart with bounded backoff or terminate the process so the orchestrator can replace it. Cancel and await it during shutdown.

**Status:** OPEN

### KAI-SUP-005 — MEDIUM — Watchdog can report healthy before any loop heartbeat exists

**Issue:** `TaskWatchdog.frozen()` only evaluates registered heartbeat entries. Before the first `_background_loop()` heartbeat, the watchdog contains no `main_loop` entry and `/health` returns `ok`.

**Risk:** A startup failure that prevents the loop from beginning can produce a false healthy status indefinitely.

**Recommendation:** Pre-register required tasks as `starting`, enforce a startup deadline and report degraded or not-ready until each mandatory loop has produced its first heartbeat.

**Status:** OPEN

### KAI-SUP-006 — MEDIUM — Watchdog staleness conflates long legitimate work with a frozen loop

**Issue:** The main-loop heartbeat occurs before and after a long serial chain of sweep, proactive, camera, reminders, escalation, greeting and forecast operations. With a staleness threshold of only three check intervals, slow but progressing work can exceed the threshold.

**Risk:** Health checks can falsely report the supervisor as frozen, causing unnecessary restarts or operator alarms during dependency slowness.

**Recommendation:** Track heartbeat/progress per subtask, use monotonic timestamps and record active operation plus deadline. Separate `busy`, `late` and `frozen` states.

**Status:** OPEN

### KAI-SUP-007 — MEDIUM — Recovery success is accepted from HTTP 200 without post-recovery verification

**Issue:** `_attempt_recovery()` treats any HTTP 200 from `/recover` as success and immediately sends a success notification. It does not validate the response body or rerun the service health check.

**Risk:** A no-op, partial or falsely successful recovery endpoint can be reported as healed while the service remains unhealthy.

**Recommendation:** Require a typed recovery response and independently verify deep health after a settling period before declaring success or closing related incidents.

**Status:** OPEN

### KAI-SUP-008 — MEDIUM — Recovery cooldown is consumed before the recovery outcome is known

**Issue:** `_recovery_attempts[name]` is updated before making the recovery request. A transient connection failure or immediate request error therefore suppresses all retries for the full cooldown period.

**Risk:** Recoverable failures remain untreated for two minutes or longer after a failed attempt, slowing restoration and reducing availability.

**Recommendation:** Track attempt start, result and next eligible time separately. Use shorter retry backoff for transport failures and longer cooldown only after a completed recovery action.

**Status:** OPEN

### KAI-SUP-009 — MEDIUM — Per-service predictive trend uses current breaker state for every historical sample

**Issue:** `predict_per_service()` iterates historical snapshots but derives `is_open` from the breaker object's current state on every iteration rather than from state stored in each snapshot.

**Risk:** The generated historical series is fabricated from present state, making slopes, R² and trend labels statistically invalid.

**Recommendation:** Persist per-service health and breaker state in every fleet snapshot and compute forecasts solely from immutable historical observations.

**Status:** OPEN

### KAI-SUP-010 — MEDIUM — On-demand sweep can race the background sweep and recovery flow

**Issue:** `POST /sweep` calls `_sweep()` directly with no lock or single-flight guard while the background loop may be running the same operation.

**Risk:** Concurrent sweeps can duplicate notifications and recovery attempts, race shared breaker and status state, and create inconsistent fleet history.

**Recommendation:** Serialise sweeps with an async lock or queue; return the current in-flight result to duplicate callers and make recovery dispatch idempotent.

**Status:** OPEN

### KAI-SUP-011 — MEDIUM — Proactive mode lookup fails open to `PUB`

**Issue:** `_get_current_mode()` returns `PUB` whenever Tool Gate is unreachable or returns an unexpected response.

**Risk:** Mode-restricted memory nudges may be processed under a permissive default precisely when the authoritative policy service is unavailable.

**Recommendation:** Fail closed to a restricted mode or suppress proactive delivery until policy state is available and fresh.

**Status:** OPEN

---

## Continuation summary

- New findings in this file: 11
- Critical: 1
- High: 3
- Medium: 7
- Cumulative findings across all registers: 63
- Cumulative Critical: 9
- Cumulative High: 28
- Cumulative Medium: 25
- Cumulative Low: 1
- Additional file materially reviewed: `supervisor/app.py`
- Current security posture: HIGH RISK / NOT READY FOR EXTERNAL EXPOSURE
- Audit state: IN PROGRESS
