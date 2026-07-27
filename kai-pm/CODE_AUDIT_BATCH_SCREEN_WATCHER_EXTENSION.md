# Kai Code Audit — Screen Watcher Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_SCREEN_WATCHER.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-SCREENWATCHX-001 | HIGH | Screen Watcher calls a nonexistent Screen Capture `/screenshot` endpoint |
| KAI-SCREENWATCHX-002 | HIGH | Screen Capture exposes JSON OCR results rather than the raw screenshot bytes Screen Watcher requires |
| KAI-SCREENWATCHX-003 | HIGH | Minimal Compose points Screen Watcher to port 8020 while Screen Capture listens on 8059 |
| KAI-SCREENWATCHX-004 | HIGH | Minimal Compose does not deploy the Screen Capture service that Screen Watcher depends on |
| KAI-SCREENWATCHX-005 | HIGH | Full Compose does not deploy Screen Watcher although Agentic/Dashboard code advertises the capability |
| KAI-SCREENWATCHX-006 | HIGH | Watching remains active and health-green through unlimited consecutive capture failures |
| KAI-SCREENWATCHX-007 | HIGH | Capture failure has no counter, last-error, degraded state or automatic stop policy |
| KAI-SCREENWATCHX-008 | HIGH | The loop retries a permanently broken dependency every interval without backoff or circuit breaking |
| KAI-SCREENWATCHX-009 | HIGH | A dead watcher task can leave `_watching=true` and make future start requests return `already_watching` |
| KAI-SCREENWATCHX-010 | HIGH | Starting monitoring commits state before a successful baseline capture is obtained |
| KAI-SCREENWATCHX-011 | HIGH | Stopping monitoring retains the last sensitive screenshot indefinitely |
| KAI-SCREENWATCHX-012 | HIGH | Restarting monitoring exposes the previous session’s screenshot until a new capture succeeds |
| KAI-SCREENWATCHX-013 | HIGH | Cached screenshots have no age/TTL, clear operation or principal/session ownership |
| KAI-SCREENWATCHX-014 | HIGH | Snapshot responses always claim `image/png` without validating the cached media bytes |
| KAI-SCREENWATCHX-015 | HIGH | Snapshot responses lack `Cache-Control: no-store` and freshness/provenance headers |
| KAI-SCREENWATCHX-016 | HIGH | Ordinary screen-change alerts bypass Tool Gate and any operator approval policy |
| KAI-SCREENWATCHX-017 | HIGH | Continuous real changes can generate one alert every interval with no cooldown or deduplication |
| KAI-SCREENWATCHX-018 | HIGH | Change time is committed before notification delivery is acknowledged |
| KAI-SCREENWATCHX-019 | HIGH | Alerts contain no screenshot digest, baseline identity, event ID or evidence reference |
| KAI-SCREENWATCHX-020 | HIGH | The sampled-byte hash can completely miss changes located outside selected encoded-byte positions |
| KAI-SCREENWATCHX-021 | MEDIUM | Hash-difference scores have only 1/32 increments despite arbitrary decimal thresholds |
| KAI-SCREENWATCHX-022 | MEDIUM | MD5 hash collisions can produce a zero change score even for different sampled data |
| KAI-SCREENWATCHX-023 | MEDIUM | Screenshot metadata/encoding changes can reset the baseline despite no visual change |
| KAI-SCREENWATCHX-024 | MEDIUM | The service retains no alert history or delivery outcome ledger |
| KAI-SCREENWATCHX-025 | MEDIUM | Status does not expose screenshot age, source, digest or last capture error |
| KAI-SCREENWATCHX-026 | MEDIUM | Snapshot returns no ETag/content digest or immutable capture identifier |
| KAI-SCREENWATCHX-027 | MEDIUM | `interval_seconds` accepts Boolean values and has no safe upper bound |
| KAI-SCREENWATCHX-028 | MEDIUM | Threshold input accepts Boolean and non-finite values with surprising clamp behaviour |
| KAI-SCREENWATCHX-029 | MEDIUM | Starting an already-running watcher silently ignores requested interval and threshold changes |
| KAI-SCREENWATCHX-030 | MEDIUM | Stop returns success even when the task reference is absent or already dead |
| KAI-SCREENWATCHX-031 | MEDIUM | Task completion exceptions are never consumed or surfaced |
| KAI-SCREENWATCHX-032 | MEDIUM | Effective capture cadence is request latency plus interval, not the reported interval |
| KAI-SCREENWATCHX-033 | MEDIUM | Watch and change timestamps use wall-clock time without a monotonic event sequence |
| KAI-SCREENWATCHX-034 | MEDIUM | Public metrics expose an empty/misleading telemetry object without administrative access control |
| KAI-SCREENWATCHX-035 | MEDIUM | Shared-runtime import failure silently replaces structured telemetry with no-op fallbacks |
| KAI-SCREENWATCHX-036 | MEDIUM | Service dependencies and the Python base image use non-reproducible version ranges/tags |
| KAI-SCREENWATCHX-037 | MEDIUM | No structured audit records who started/stopped monitoring or which alert was delivered |
| KAI-SCREENWATCHX-038 | MEDIUM | No tests were found that verify the real Screen Capture endpoint, deployment port or alert delivery contract |
| KAI-SCREENWATCHX-039 | MEDIUM | The service has no authoritative single-watcher leadership or shared state across replicas |

---

## High-severity findings

### KAI-SCREENWATCHX-001 — HIGH — Nonexistent capture route
**Issue:** `_capture_screen()` posts to `${SCREEN_CAPTURE_URL}/screenshot`, but `screen-capture/app.py` defines only `/capture` and `/capture/file`.  
**Risk:** Every normal watch cycle receives 404 or connection failure and never establishes a baseline.  
**Recommendation:** Use one versioned raw-image API contract and verify it in integration tests/readiness.  
**Status:** OPEN

### KAI-SCREENWATCHX-002 — HIGH — Incompatible service contract
Even `/capture` cannot substitute directly: it returns a JSON `CaptureResult` containing OCR text, not raw screenshot bytes.

### KAI-SCREENWATCHX-003 — HIGH — Wrong deployment port
Minimal Compose configures `http://screen-capture:8020`; Screen Capture’s application and full deployment use port 8059.

### KAI-SCREENWATCHX-004 — HIGH — Missing dependency in minimal topology
Minimal Compose defines Screen Watcher but has no Screen Capture service entry.

### KAI-SCREENWATCHX-005 — HIGH — Missing watcher in full topology
The full stack deploys Screen Capture but not Screen Watcher, while other modules retain Screen Watcher URLs/capability claims.

### KAI-SCREENWATCHX-006 — HIGH — False active health
The loop converts every capture error to `None`, sleeps and retries. `_watching` remains true and `/health` stays `ok` indefinitely.

### KAI-SCREENWATCHX-007 — HIGH — No failure state
There is no consecutive-failure count, last error, last successful capture, dependency readiness or maximum outage duration.

### KAI-SCREENWATCHX-008 — HIGH — Permanent retry storm
A broken route/DNS/service is contacted at the fixed interval forever, with no exponential backoff or circuit breaker.

### KAI-SCREENWATCHX-009 — HIGH — Dead-task lockout
No done callback/finally clears `_watching`. An unexpected task exception leaves state active and prevents a new watcher from being created.

### KAI-SCREENWATCHX-010 — HIGH — Start acknowledges before evidence
`_watching=true` and a background task are returned immediately; no baseline capture or dependency check must succeed first.

### KAI-SCREENWATCHX-011 — HIGH — Stop does not erase surveillance data
The cached image, hash and timestamps remain available through `/snapshot` and `/status` after stop.

### KAI-SCREENWATCHX-012 — HIGH — Cross-session stale image
A new start clears only `_prev_hash`; the old screenshot remains public until replaced.

### KAI-SCREENWATCHX-013 — HIGH — Indefinite screenshot retention
There is no maximum age, explicit clear, source/session boundary or inactivity purge.

### KAI-SCREENWATCHX-014 — HIGH — Forced PNG labelling
Arbitrary upstream bytes are returned as `image/png` without magic-byte/decode verification.

### KAI-SCREENWATCHX-015 — HIGH — Cacheable screen content
The snapshot endpoint provides no no-store, age or capture-ID headers.

### KAI-SCREENWATCHX-016 — HIGH — Alert action bypasses governance
Screen change directly calls Notify Service; no Tool Gate decision, operator policy or action capability is checked.

### KAI-SCREENWATCHX-017 — HIGH — Normal-change alert spam
Beyond the zero-threshold defect already logged, any continuously changing screen above a reasonable threshold sends an alert every cycle because there is no event cooldown/hysteresis/deduplication.

### KAI-SCREENWATCHX-018 — HIGH — Detection is confused with delivery
`_last_change_ts` is updated and retained before the untracked notification task succeeds.

### KAI-SCREENWATCHX-019 — HIGH — Alert lacks evidence identity
The notification contains only a percentage and cannot be linked to the before/after screenshot hashes or watcher session.

### KAI-SCREENWATCHX-020 — HIGH — Deterministic blind spots
The hash samples at most 1,024 encoded-byte positions. Changes outside those positions are invisible even before considering compression effects.

---

## Medium-severity findings

### KAI-SCREENWATCHX-021 — MEDIUM — Coarse score resolution
The MD5 hex strings contain 32 characters, so possible scores are multiples of 3.125%; a configured 5% threshold actually requires at least 6.25% hash-character change.

### KAI-SCREENWATCHX-022 — MEDIUM — Collision-prone equality
MD5 is not collision resistant; a collision in sampled bytes produces no detected change.

### KAI-SCREENWATCHX-023 — MEDIUM — Baseline depends on file encoding
PNG compressor/version/metadata changes alter encoded bytes and may generate unrelated baselines.

### KAI-SCREENWATCHX-024 — MEDIUM — No alert ledger
Only the most recent timestamps/diff survive; no notification attempt/result/history is retained.

### KAI-SCREENWATCHX-025 — MEDIUM — Incomplete status
Consumers cannot determine whether the cached image is current, which source produced it or why captures stopped succeeding.

### KAI-SCREENWATCHX-026 — MEDIUM — Missing snapshot identity
There is no digest, ETag, capture timestamp header or immutable ID.

### KAI-SCREENWATCHX-027 — MEDIUM — Weak interval validation
JSON booleans satisfy Pydantic integer typing; huge intervals are accepted and make monitoring appear active while effectively dormant.

### KAI-SCREENWATCHX-028 — MEDIUM — Weak threshold validation
Boolean/non-finite floats are not explicitly rejected; nested min/max semantics can produce unexpected values.

### KAI-SCREENWATCHX-029 — MEDIUM — Configuration update ignored while active
A caller receives `already_watching` after the code may update global interval/threshold fields, but the response does not return the effective changed configuration or distinguish update from no-op.

### KAI-SCREENWATCHX-030 — MEDIUM — Ambiguous stop success
Stop always returns `ok`, including when no active task exists.

### KAI-SCREENWATCHX-031 — MEDIUM — Unobserved task failure
The watcher task has no done callback or exception retrieval.

### KAI-SCREENWATCHX-032 — MEDIUM — Misstated interval
Each loop waits for capture completion and then sleeps; slow ten-second requests make the real period `_capture latency + interval`.

### KAI-SCREENWATCHX-033 — MEDIUM — Weak chronology
Wall-clock adjustments affect uptime and event ordering; no session sequence is used.

### KAI-SCREENWATCHX-034 — MEDIUM — Public misleading metrics
The already-logged unpopulated ErrorBudget is returned publicly without indicating telemetry disabled.

### KAI-SCREENWATCHX-035 — MEDIUM — Silent runtime downgrade
Missing shared runtime imports yield basic logging/no-op metrics while health remains ordinary.

### KAI-SCREENWATCHX-036 — MEDIUM — Non-reproducible runtime
Requirements use broad lower bounds and the Python image is tag-based.

### KAI-SCREENWATCHX-037 — MEDIUM — Missing surveillance audit
No immutable event identifies start/stop actor, configuration, captures, change event and delivery result.

### KAI-SCREENWATCHX-038 — MEDIUM — Missing integration tests
Repository search found no dedicated Screen Watcher test covering route/port/schema and Notify acknowledgement.

### KAI-SCREENWATCHX-039 — MEDIUM — No watcher leadership
Multiple processes/replicas can independently run monitoring loops and alerts without election or shared state.

---

## Batch totals

- Findings: **39**
- Critical: **0**
- High: **20**
- Medium: **19**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,598**
- Critical: **191**
- High: **1,305**
- Medium: **1,099**
- Low: **3**

## Files materially reviewed

`screen-watcher/app.py`, `screen-watcher/Dockerfile`, `screen-watcher/requirements.txt`, both Compose topologies, Screen Capture’s actual API contract, Notify/TTS integration and the existing Screen Watcher audit.
