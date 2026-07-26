# Kai Code Audit — Screen Watcher Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-SCREENWATCH-001 | CRITICAL | Unauthenticated callers can start continuous screen capture |
| KAI-SCREENWATCH-002 | CRITICAL | Latest screenshot bytes are exposed without authentication |
| KAI-SCREENWATCH-003 | HIGH | Unauthenticated callers can configure zero-threshold persistent alert generation |
| KAI-SCREENWATCH-004 | HIGH | Screenshot response size is unbounded before caching |
| KAI-SCREENWATCH-005 | HIGH | Change detection hashes compressed screenshot bytes rather than visual content |
| KAI-SCREENWATCH-006 | MEDIUM | Stop cancels the task without awaiting termination |
| KAI-SCREENWATCH-007 | MEDIUM | Application shutdown does not stop or await the watcher task |
| KAI-SCREENWATCH-008 | MEDIUM | Notification HTTP failures and status codes are ignored |
| KAI-SCREENWATCH-009 | MEDIUM | Fire-and-forget notification tasks are untracked |
| KAI-SCREENWATCH-010 | MEDIUM | A new HTTP client is created for every capture and alert request |
| KAI-SCREENWATCH-011 | MEDIUM | Service state and surveillance timings are exposed publicly |
| KAI-SCREENWATCH-012 | MEDIUM | Watch state and cached screenshot are process-local and volatile |
| KAI-SCREENWATCH-013 | MEDIUM | Error-budget telemetry is instantiated but never populated |
| KAI-SCREENWATCH-014 | MEDIUM | Configuration values are not validated at startup |
| KAI-SCREENWATCH-015 | MEDIUM | Declared TTS integration is unused |

---

## Screen watcher: `screen-watcher/app.py`

### KAI-SCREENWATCH-001 — CRITICAL — Unauthenticated continuous screen capture
**Issue:** `POST /watch/start` requires no authentication or authorisation. It starts `_watch_loop`, which repeatedly calls the screen-capture service and retains the latest screenshot.  
**Risk:** Any reachable caller can activate ongoing monitoring of the operator’s display, creating persistent privacy intrusion and resource use.  
**Recommendation:** Require authenticated local-device authority, explicit user consent and visible active-capture indication.  
**Status:** OPEN — immediate remediation required

### KAI-SCREENWATCH-002 — CRITICAL — Screenshot disclosure
**Issue:** `GET /snapshot` returns the complete cached screenshot without authentication.  
**Risk:** Callers can retrieve sensitive screen contents including messages, credentials, documents and operational data whenever watching has captured a frame.  
**Recommendation:** Remove public screenshot retrieval or require tightly scoped authenticated access with audit logging.  
**Status:** OPEN — immediate remediation required

### KAI-SCREENWATCH-003 — HIGH — Alert flooding through threshold control
**Issue:** Unauthenticated callers can set `threshold` to `0.0`. After the first baseline frame, every subsequent capture satisfies `diff >= 0.0`, causing a notification task each interval. The minimum interval is two seconds.  
**Risk:** A caller can generate persistent desktop alert spam and consume capture, hashing and notification resources until another caller stops the watcher.  
**Recommendation:** Restrict configuration changes, enforce safe non-zero thresholds and rate-limit deduplicated alerts.  
**Status:** OPEN

### KAI-SCREENWATCH-004 — HIGH — Screenshot cache allocation is unbounded
**Issue:** `_capture_screen` accepts `resp.content` without checking content length, MIME type or image validity, then stores the full byte string in `_last_screenshot`.  
**Risk:** A compromised or misconfigured screen-capture service can return an arbitrarily large payload and exhaust memory.  
**Recommendation:** Stream with strict response-size, type and image-dimension limits before caching.  
**Status:** OPEN

### KAI-SCREENWATCH-005 — HIGH — Change score is not perceptual
**Issue:** `_image_hash` samples raw encoded PNG bytes and computes MD5. `_diff_score` compares the hexadecimal digest characters. This measures differences in compressed file representation, metadata and encoding, not visual pixel change.  
**Risk:** Identical-looking screenshots can trigger large changes, while meaningful screen changes may not correlate reliably with the score. Alerts and downstream conclusions are therefore materially misleading.  
**Recommendation:** Decode images and use a validated perceptual or pixel-difference method with calibrated thresholds.  
**Status:** OPEN

### KAI-SCREENWATCH-006 — MEDIUM — Task cancellation is not awaited
**Issue:** `/watch/stop` calls `_watch_task.cancel()` and immediately drops the reference without awaiting task completion.  
**Risk:** In-flight capture or notification work may continue after the API reports monitoring stopped, and cancellation exceptions/resources are not observed.  
**Recommendation:** Await cancellation and confirm termination before returning success.  
**Status:** OPEN

### KAI-SCREENWATCH-007 — MEDIUM — Shutdown lifecycle is empty
**Issue:** The FastAPI lifespan handler only yields. It does not stop, cancel or await `_watch_task` during application shutdown.  
**Risk:** Reloads and shutdowns can terminate active surveillance tasks abruptly and leak in-flight network operations.  
**Recommendation:** Implement explicit startup/shutdown ownership of the watcher task.  
**Status:** OPEN

### KAI-SCREENWATCH-008 — MEDIUM — Alert delivery failures are ignored
**Issue:** `_send_notify` does not inspect the HTTP response status or body, and suppresses every exception.  
**Risk:** Change alerts can be silently lost while timestamps indicate detection occurred.  
**Recommendation:** Validate acknowledgements and record durable delivery state.  
**Status:** OPEN

### KAI-SCREENWATCH-009 — MEDIUM — Notification tasks are untracked
**Issue:** Each detected change launches `asyncio.create_task(_send_notify(...))` without retaining, bounding or supervising the task.  
**Risk:** Frequent changes can accumulate concurrent tasks; shutdown abandons them and failures are not attributable.  
**Recommendation:** Use a bounded alert queue with supervised workers.  
**Status:** OPEN

### KAI-SCREENWATCH-010 — MEDIUM — HTTP connection churn
**Issue:** Every capture, notification and TTS call constructs a new `httpx.AsyncClient`.  
**Risk:** Continuous watching repeatedly creates sockets and connection pools, increasing latency and resource pressure.  
**Recommendation:** Reuse lifecycle-managed clients with bounded pools.  
**Status:** OPEN

### KAI-SCREENWATCH-011 — MEDIUM — Surveillance state is public
**Issue:** `/health` and `/status` expose whether monitoring is active, interval, threshold, last capture/change times and latest difference score without authentication.  
**Risk:** Callers can infer operator activity and tune attacks around monitoring state.  
**Recommendation:** Require scoped operational access.  
**Status:** OPEN

### KAI-SCREENWATCH-012 — MEDIUM — State is worker-local and volatile
**Issue:** Watching state, task reference, cached screenshot, previous hash and timestamps are module-level memory.  
**Risk:** Multiple workers can run independent watchers and expose different snapshots; restart erases state while external callers may believe monitoring persists.  
**Recommendation:** Enforce a single authoritative watcher and shared state, or externalise scheduling and storage.  
**Status:** OPEN

### KAI-SCREENWATCH-013 — MEDIUM — Error budget is never recorded
**Issue:** `budget` is created and `/metrics` returns its snapshot, but no middleware or endpoint calls `budget.record`.  
**Risk:** Metrics appear available while containing no request outcome data.  
**Recommendation:** Record actual response statuses and exceptions consistently.  
**Status:** OPEN

### KAI-SCREENWATCH-014 — MEDIUM — Startup configuration lacks validation
**Issue:** Port, service URLs, default interval and threshold are parsed directly. Invalid text crashes startup; negative/default threshold and interval values can create unsafe runtime behaviour before request-level clamping occurs.  
**Risk:** Misconfiguration can disable the service, create tight loops or cause continuous alerts.  
**Recommendation:** Validate typed configuration with explicit safe ranges at startup.  
**Status:** OPEN

### KAI-SCREENWATCH-015 — MEDIUM — TTS integration is dead code
**Issue:** `TTS_URL` and `_send_tts` are defined, and the module documentation states it fires notify/TTS alerts, but `_watch_loop` never calls `_send_tts`.  
**Risk:** Operators and documentation overstate alert channels; expected spoken warnings never occur.  
**Recommendation:** Remove the claim/dead path or implement and verify the intended delivery channel.  
**Status:** OPEN

---

## Batch totals

- Findings: **15**
- Critical: **2**
- High: **3**
- Medium: **10**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **456**
- Critical: **48**
- High: **169**
- Medium: **236**
- Low: **3**

## Files materially reviewed in this batch

`screen-watcher/app.py`.
