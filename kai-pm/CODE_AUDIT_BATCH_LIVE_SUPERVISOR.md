# Kai Code Audit — Live Supervisor and Self-Heal Control Plane Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers the deployed `supervisor/app.py`. The earlier `CODE_AUDIT_BATCH_SELF_AUDIT_SUPERVISOR.md` covers separate scripts and is not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-SUP-001 | CRITICAL | The host-published Supervisor control plane has no authentication or authorisation |
| KAI-SUP-002 | CRITICAL | Anonymous callers can manually trigger internal service recovery actions |
| KAI-SUP-003 | CRITICAL | Anonymous sweeps can accelerate breaker opening and trigger fleet recovery actions |
| KAI-SUP-004 | CRITICAL | Automatic recovery invokes consequential `/recover` endpoints from shallow self-reported health |
| KAI-SUP-005 | HIGH | Incoming requests have no verified user/service identity or replay protection |
| KAI-SUP-006 | HIGH | Health and recovery calls do not authenticate the target service or Supervisor identity |
| KAI-SUP-007 | HIGH | Every 200 status except exact `degraded` is classified healthy |
| KAI-SUP-008 | HIGH | A 200 response without `status` defaults to healthy |
| KAI-SUP-009 | HIGH | Failed dependency checks are ignored when top-level status says `ok` |
| KAI-SUP-010 | HIGH | Fleet status reports healthy before any sweep has produced evidence |
| KAI-SUP-011 | HIGH | Supervisor health ignores fleet health and required dependency readiness |
| KAI-SUP-012 | HIGH | Fleet remains “healthy” through sub-threshold consecutive service failures |
| KAI-SUP-013 | HIGH | Open circuit breakers do not stop Supervisor from continuing health calls |
| KAI-SUP-014 | HIGH | Manual and background sweeps can overlap and race breaker/recovery state |
| KAI-SUP-015 | HIGH | Recovery has no distributed mutex, idempotency key or per-incident operation identity |
| KAI-SUP-016 | HIGH | Every registered service is assigned a `/recover` action whether it supports safe recovery or not |
| KAI-SUP-017 | HIGH | Any HTTP 200 recovery response is treated as successful healing |
| KAI-SUP-018 | HIGH | Recovery success is not followed by a verified health/readiness probe |
| KAI-SUP-019 | HIGH | Recovery requests contain no incident, expected state, revision or authorised action body |
| KAI-SUP-020 | HIGH | Failed recovery attempts enter cooldown before the request outcome is known |
| KAI-SUP-021 | HIGH | Open-breaker notifications repeat every sweep despite recovery cooldown |
| KAI-SUP-022 | HIGH | Recovery notifications are silently disabled by the default empty `NOTIFY_URL` |
| KAI-SUP-023 | HIGH | Synchronous notification HTTP blocks the async Supervisor loop |
| KAI-SUP-024 | HIGH | Notification HTTP status is ignored |
| KAI-SUP-025 | HIGH | Breakers, recovery history and fleet state are process-local and restart-volatile |
| KAI-SUP-026 | HIGH | Multiple workers or replicas independently recover and notify for the same incident |
| KAI-SUP-027 | HIGH | The mandatory background loop task is created without retaining its handle |
| KAI-SUP-028 | HIGH | Shutdown does not cancel or await the recovery/proactive loop |
| KAI-SUP-029 | HIGH | A loop failure before its first heartbeat leaves `/health` green indefinitely |
| KAI-SUP-030 | HIGH | Supervisor’s fixed inventory omits many actively deployed full-stack services |
| KAI-SUP-031 | HIGH | Status, breaker, history, prediction and watchdog internals are publicly exposed |
| KAI-SUP-032 | HIGH | Public `/status` calls actively probe memU quarantine and Verifier metrics |
| KAI-SUP-033 | HIGH | No rate limit, admission control or caller quota protects sweeps, recovery or probes |
| KAI-SUP-034 | HIGH | Service health response bodies and nested `checks` are unbounded |
| KAI-SUP-035 | HIGH | Raw service connection/parsing errors are exposed in public fleet status |
| KAI-SUP-036 | HIGH | Tool Gate mode failure defaults proactive behaviour to permissive PUB mode |
| KAI-SUP-037 | HIGH | Failure of the filtered proactive endpoint falls back to unfiltered nudge feeds |
| KAI-SUP-038 | HIGH | Only five nudges are delivered but every queued nudge is marked sent |
| KAI-SUP-039 | HIGH | Escalated nudges are resent every loop with no Supervisor-side deduplication |
| KAI-SUP-040 | HIGH | Untrusted, unbounded nudge content is inserted into Telegram Markdown |
| KAI-SUP-041 | HIGH | Anonymous high-frequency sweeps manipulate predictive-failure history |
| KAI-SUP-042 | HIGH | Per-service prediction fabricates history from each service’s current breaker state |
| KAI-SUP-043 | MEDIUM | Aggregate forecasts alert without a minimum R² or confidence threshold |
| KAI-SUP-044 | MEDIUM | Forecast alert cooldown is set before confirming notification delivery |
| KAI-SUP-045 | MEDIUM | Forecast history records 200-response “recoveries” as genuine recovery evidence |
| KAI-SUP-046 | MEDIUM | Predicted unhealthy counts are not capped to the fleet size |
| KAI-SUP-047 | MEDIUM | An already-breached failure threshold suppresses the predictive warning branch |
| KAI-SUP-048 | MEDIUM | OLS is applied to irregular and caller-influenced sample intervals without validation |
| KAI-SUP-049 | MEDIUM | Constant-series R² handling can present misleadingly strong model fit |
| KAI-SUP-050 | MEDIUM | Sequential recovery calls can stall the main loop beyond its watchdog threshold |
| KAI-SUP-051 | MEDIUM | Each service recovery constructs a new HTTP client/pool |
| KAI-SUP-052 | MEDIUM | Every public status call constructs separate memU and Verifier clients |
| KAI-SUP-053 | MEDIUM | Every notification constructs a new synchronous client/pool |
| KAI-SUP-054 | MEDIUM | Proactive nudge deduplication state grows without a retention bound |
| KAI-SUP-055 | MEDIUM | First-50-character fallback deduplication can suppress distinct messages |
| KAI-SUP-056 | MEDIUM | Wall-clock changes can bypass or extend recovery, nudge and forecast cooldowns |
| KAI-SUP-057 | MEDIUM | Sweep, recovery, prediction and cooldown configuration lacks safe range validation |
| KAI-SUP-058 | MEDIUM | Extra service names and URLs are weakly parsed and unvalidated |
| KAI-SUP-059 | MEDIUM | Duplicate service names overwrite breaker entries while remaining duplicated in the service list |
| KAI-SUP-060 | MEDIUM | Unknown manual recovery returns HTTP 200 with an error-shaped body |
| KAI-SUP-061 | MEDIUM | Cooldown suppression and actual recovery failure share the same `ok: false` outcome |
| KAI-SUP-062 | MEDIUM | Fleet trend and recovery history disappear on restart |
| KAI-SUP-063 | MEDIUM | Supervisor API ErrorBudget measures caller HTTP responses, not fleet health |
| KAI-SUP-064 | MEDIUM | Mutable fleet/status structures are read and written without snapshot consistency |
| KAI-SUP-065 | MEDIUM | Private proactive and greeting content is written to operational logs |
| KAI-SUP-066 | MEDIUM | Service errors and topology are retained in logs/status without protected trace IDs |
| KAI-SUP-067 | MEDIUM | Required watchdog task names are not registered before the loop starts |
| KAI-SUP-068 | MEDIUM | A watchdog served by the same event loop cannot observe a full event-loop deadlock |
| KAI-SUP-069 | MEDIUM | Deprecated startup events are used instead of an owned lifespan context |
| KAI-SUP-070 | MEDIUM | Recovery, notification and prediction actions have no immutable audit trail |

---

## Critical recovery-control findings

### KAI-SUP-001 — CRITICAL — Open Supervisor control plane
**Issue:** `docker-compose.minimal.yml` publishes `8051:8051`. `supervisor/app.py` has no inbound authentication or authorisation middleware.  
**Risk:** Any reachable caller can trigger sweeps/recovery and inspect internal fleet state.  
**Recommendation:** remove host publication and require scoped, replay-protected operator/service identity.  
**Status:** OPEN — immediate remediation required

### KAI-SUP-002 — CRITICAL — Anonymous internal recovery trigger
**Issue:** `POST /recover/{service_name}` invokes the registered internal `/recover` endpoint with no caller identity or approval.  
**Risk:** Remote callers can reset breakers, reconnect pools, reload security tokens/nonces and invoke other state-changing service recovery handlers.  
**Recommendation:** require authenticated incident authority and an exact approved recovery action bound to service/revision/reason.  
**Status:** OPEN — immediate remediation required

### KAI-SUP-003 — CRITICAL — Anonymous sweep-to-recovery chain
**Issue:** `POST /sweep` runs health checks, records breaker failures and automatically recovers every service whose breaker becomes open. Concurrent/repeated calls can reach the failure threshold faster than the normal 15-second cadence.  
**Risk:** Anonymous traffic can force fleet-wide recovery mutations and alert storms.  
**Recommendation:** make sweeps internal/read-only, serialize them, and separate health observation from authorised recovery execution.  
**Status:** OPEN — immediate remediation required

### KAI-SUP-004 — CRITICAL — Shallow health drives consequential recovery
**Issue:** Three failed/degraded self-reported health responses automatically cause POST requests to fixed `/recover` routes. Confirmed recovery handlers include Agentic breaker reset, memU pool/state mutation, Tool Gate token/nonce reload and Executor state recovery. No root-cause or action-risk authority intervenes.  
**Risk:** Transient, spoofed or non-recoverable dependency problems trigger state-changing control actions that may erase containment or alter security state.  
**Recommendation:** require service-specific recovery policy, authenticated health evidence, diagnosis, approval and postcondition checks.  
**Status:** OPEN — immediate remediation required

---

## Health, breaker and recovery findings

### KAI-SUP-005 — HIGH — Missing incoming identity
No API key, mTLS, HMAC, user delegation, nonce or replay prevention is checked.

### KAI-SUP-006 — HIGH — Unauthenticated internal control traffic
Health and recovery use ordinary HTTP and verify neither the expected service identity/version nor Supervisor identity at the target.

### KAI-SUP-007 — HIGH — Unknown failure states become healthy
Only exact `status == "degraded"` is unhealthy. `error`, `failed`, `disabled`, `inactive`, `stub`, `not_ready` and arbitrary values close the breaker as success.

### KAI-SUP-008 — HIGH — Missing status defaults healthy
A 200 JSON body with no `status` uses `ok`.

### KAI-SUP-009 — HIGH — Nested failure ignored
A response may report failed `checks` while top-level status remains `ok`; Supervisor records success and publishes those checks as healthy evidence.

### KAI-SUP-010 — HIGH — Evidence-free initial green state
Before `_last_status` or history contains any sweep, `/status` reports fleet healthy because no breaker is open.

### KAI-SUP-011 — HIGH — Readiness-blind Supervisor health
`/health` checks only watchdog staleness and device. It ignores fleet availability, notification channel, recovery loop readiness and internal dependencies.

### KAI-SUP-012 — HIGH — Failures hidden until threshold
Fleet status is based on open-breaker count, so one or two consecutive failures per service still produce `fleet: healthy`.

### KAI-SUP-013 — HIGH — Breaker is not enforced
`_check_service()` never calls `cb.allow()`. Open services continue receiving probes on every sweep, contrary to the stated cascade-containment role.

### KAI-SUP-014 — HIGH — Sweep races
Background and manual sweeps can overlap, concurrently increment/reset breakers, append history and invoke recovery.

### KAI-SUP-015 — HIGH — No recovery mutual exclusion
Cooldown is only a timestamp check. Concurrent recoveries can both pass before the timestamp is updated across workers/processes, and there is no idempotency key.

### KAI-SUP-016 — HIGH — Generic recovery assumption
Every configured service receives `base + /recover`, even when recovery is unsupported, destructive, differently authenticated or semantically inappropriate.

### KAI-SUP-017 — HIGH — HTTP 200 equals healed
The response body is not schema-checked. Partial/no-op/error-shaped 200 responses count as successful self-healing.

### KAI-SUP-018 — HIGH — No postcondition verification
Recovery success is announced before a fresh deep-health/readiness check demonstrates that the incident is resolved.

### KAI-SUP-019 — HIGH — Context-free recovery request
The POST carries no incident ID, observed failure, intended recovery operation, expected revision, idempotency token or authenticated actor.

### KAI-SUP-020 — HIGH — Failure enters cooldown
`_recovery_attempts[name]` is set before the network call. A timeout or immediate failure suppresses another attempt for the full cooldown.

### KAI-SUP-021 — HIGH — Repeated open-circuit alerts
Each sweep calls `_send_notification("circuit OPEN")` before cooldown logic, creating repeated alerts every check interval.

### KAI-SUP-022 — HIGH — Alerts disabled by default
`NOTIFY_URL` defaults to empty and minimal Compose does not configure it. Recovery success/failure and predictive alerts are silently skipped.

### KAI-SUP-023 — HIGH — Notification blocks control loop
`_send_notification()` creates a synchronous `httpx.Client` and can block the event loop for five seconds per alert.

### KAI-SUP-024 — HIGH — Delivery success unverified
Notification response status/body is ignored; only transport exceptions produce a warning.

### KAI-SUP-025 — HIGH — Volatile containment state
Breakers, attempts, history and last status are ordinary process dictionaries/lists and reset on restart.

### KAI-SUP-026 — HIGH — Replica duplication
No leader election/shared lock exists. Multiple Supervisor workers/replicas independently sweep, recover and notify.

### KAI-SUP-027 — HIGH — Unsupervised mandatory loop
Startup calls `asyncio.create_task(_background_loop())` and discards the task handle.

### KAI-SUP-028 — HIGH — No graceful loop shutdown
There is no shutdown handler to cancel/await the loop or in-flight recoveries/notifications.

### KAI-SUP-029 — HIGH — Pre-heartbeat death is invisible
TaskWatchdog knows only tasks that have called `heartbeat()`. If the loop fails before its first beat, `frozen()` remains empty forever.

### KAI-SUP-030 — HIGH — Incomplete fleet inventory
The fixed registry monitors ten services but the full deployment contains many additional control, perception, messaging, financial and memory services. No manifest coverage check exists.

---

## Public observability and proactive-delivery findings

### KAI-SUP-031 — HIGH — Public operational intelligence
`/status`, `/breakers`, `/fleet/history`, `/predict`, `/predict/per-service` and `/watchdog` disclose internal service names, failures, breaker states, trends, recovery times and watchdog state.

### KAI-SUP-032 — HIGH — Status is an active internal probe
Every public `/status` request creates calls to memU quarantine and Verifier metrics, amplifying internal load and exposing their data.

### KAI-SUP-033 — HIGH — No admission controls
Sweep, status, prediction and recovery endpoints have no rate limit, queue bound, per-principal quota or concurrency limit.

### KAI-SUP-034 — HIGH — Unbounded health payload retention
Complete health JSON `checks` is retained in `_last_status` and returned publicly without response byte/depth/schema limits.

### KAI-SUP-035 — HIGH — Error disclosure
Transport/DNS/TLS/parser exception strings are stored and returned in service status.

### KAI-SUP-036 — HIGH — Gate failure becomes permissive mode
`_get_current_mode()` returns PUB whenever Tool Gate is unavailable or malformed, selecting the broader/lower-threshold proactive configuration.

### KAI-SUP-037 — HIGH — Policy/anti-annoyance bypass on failure
If `/memory/proactive/filtered` is unavailable, Supervisor falls back to `/full` and then `/proactive`, bypassing mode filtering and anti-annoyance semantics.

### KAI-SUP-038 — HIGH — Undelivered nudges marked sent
The Telegram message includes `to_send[:5]`, but after a 200 response the code marks every item in `to_send` as sent.

### KAI-SUP-039 — HIGH — Escalation spam loop
Every 15-second loop sends all level-3/4 ladder entries again. No sent marker/cooldown/dedup exists in `_check_escalations()`.

### KAI-SUP-040 — HIGH — Telegram markup/content injection
Nudge category/message and escalation target are inserted directly into Markdown-formatted Telegram text. Message length and control characters are not bounded here.

### KAI-SUP-041 — HIGH — Forecast history is caller-manipulable
Anonymous `/sweep` calls append snapshots at arbitrary frequency, changing regression slope and alert timing.

### KAI-SUP-042 — HIGH — Per-service forecast has no historical service data
For every historical snapshot, `predict_per_service()` reads the service’s current breaker state. Each generated series is therefore based on one current value repeated across history, not the state at each timestamp.

---

## Medium-severity forecasting and operational findings

### KAI-SUP-043 — MEDIUM — No forecast confidence gate
Warnings depend on positive slope and threshold crossing; R² can be zero and still trigger an authoritative prediction.

### KAI-SUP-044 — MEDIUM — Failed alert suppresses retries
`_last_forecast_alert` is updated before `_send_notification()`, including when notifications are disabled or fail.

### KAI-SUP-045 — MEDIUM — False recovery evidence
A service enters the snapshot’s `recovered` list based solely on recovery HTTP 200, contaminating later predictions.

### KAI-SUP-046 — MEDIUM — Impossible predicted counts
Regression output is lower-bounded at zero but not upper-bounded by `len(SERVICES)`.

### KAI-SUP-047 — MEDIUM — Prediction omits active breach
The warning branch requires `current < threshold`; once the threshold is already reached, the forecast returns no warning.

### KAI-SUP-048 — MEDIUM — Weak time-series assumptions
Manual/background sweeps create irregular/correlated samples, but ordinary OLS is used without sampling validation, autocorrelation handling or uncertainty intervals.

### KAI-SUP-049 — MEDIUM — Misleading constant-series fit
`ss_tot` is replaced with 1.0 for a constant series, allowing an exact constant fit to report R² 1.0 despite no predictive information.

### KAI-SUP-050 — MEDIUM — Recovery stalls watchdog loop
Open services are recovered sequentially, each with up to ten seconds. Several failures can exceed `CHECK_INTERVAL * 3` and make the healthy loop appear frozen.

### KAI-SUP-051 — MEDIUM — Recovery client churn
A new `AsyncClient` is created for each individual service recovery.

### KAI-SUP-052 — MEDIUM — Status client churn
Separate short-lived clients are created for memU and Verifier on every status call.

### KAI-SUP-053 — MEDIUM — Notification client churn
Every notification creates and closes a synchronous client.

### KAI-SUP-054 — MEDIUM — Unbounded dedup state
`_nudges_sent` grows one key per unique memory/message and is never pruned.

### KAI-SUP-055 — MEDIUM — Dedup collision
Nudges without memory IDs use only the first 50 message characters, suppressing distinct messages with the same prefix.

### KAI-SUP-056 — MEDIUM — Wall-clock cooldown errors
Recovery, nudge, greeting and forecast cooldowns use `time.time()`; clock changes can prematurely permit or indefinitely defer actions.

### KAI-SUP-057 — MEDIUM — Unsafe configuration ranges
Failure threshold, recovery/check/proactive/cooldown intervals and prediction horizon/threshold parse directly from environment without safe cross-field validation.

### KAI-SUP-058 — MEDIUM — Weak extra-service parsing
Comma-separated `name=url` entries accept empty/duplicate names, arbitrary schemes/hosts and unlimited entries.

### KAI-SUP-059 — MEDIUM — Duplicate identity ambiguity
Duplicate names overwrite the `breakers` dictionary entry while both service rows remain in `SERVICES`, causing shared breaker/recovery/status identity.

### KAI-SUP-060 — MEDIUM — Wrong HTTP semantics
Unknown manual services return HTTP 200 with `{ok:false}` rather than a typed 404/validation error.

### KAI-SUP-061 — MEDIUM — Ambiguous recovery false result
Cooldown suppression and an attempted-but-failed recovery both return false, preventing callers from distinguishing state.

### KAI-SUP-062 — MEDIUM — Forecast evidence resets
Fleet history and recovery attempts are not persisted, so restart erases the predictive baseline and incident chronology.

### KAI-SUP-063 — MEDIUM — Mislabelled metrics
The ErrorBudget records HTTP responses served by Supervisor, not health-check/recovery outcomes, yet `/metrics` is presented alongside fleet operations.

### KAI-SUP-064 — MEDIUM — Inconsistent snapshots
Concurrent sweeps/status requests mutate/read dictionaries, breakers and history without a revision or snapshot lock.

### KAI-SUP-065 — MEDIUM — Private content logging
Greeting, reminder, task, escalation and proactive message text is written to ordinary logs.

### KAI-SUP-066 — MEDIUM — Weak diagnostic provenance
Service errors/topology are logged or exposed directly without stable redacted codes and protected trace IDs.

### KAI-SUP-067 — MEDIUM — Watchdog has no required registry
No expected task list is configured, so absence is indistinguishable from not-yet-registered until a heartbeat exists.

### KAI-SUP-068 — MEDIUM — Same-loop deadlock blind spot
If the event loop is fully blocked, the watchdog timestamps become stale but the same loop cannot serve `/health`; detection depends on an external monitor.

### KAI-SUP-069 — MEDIUM — Deprecated lifecycle hook
The background loop is launched with `@app.on_event("startup")` rather than a lifespan context.

### KAI-SUP-070 — MEDIUM — Missing immutable operations audit
There is no tamper-evident event containing actor, health evidence, breaker revision, chosen recovery, response digest, postcondition and notification outcome.

---

## Batch totals

- Findings: **70**
- Critical: **4**
- High: **38**
- Medium: **28**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,552**
- Critical: **129**
- High: **724**
- Medium: **696**
- Low: **3**

## Files materially reviewed

`supervisor/app.py`, `common/resilience.py`, `common/runtime.py`, relevant `/recover` implementations in `tool-gate/app.py`, `agentic/app.py`, `memu-core/app.py`, `executor/app.py`, and Supervisor deployment in `docker-compose.minimal.yml`.
