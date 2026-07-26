# Kai Code Audit — Monitor Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-MONITOR-001 | CRITICAL | Unauthenticated callers can create persistent arbitrary-URL HTTP monitoring rules |
| KAI-MONITOR-002 | CRITICAL | Unauthenticated scrape rules can drive the browser-agent to arbitrary URLs |
| KAI-MONITOR-003 | CRITICAL | Unauthenticated callers can create persistent desktop/TTS alert automation |
| KAI-MONITOR-004 | HIGH | Rule updates bypass all Pydantic validation |
| KAI-MONITOR-005 | HIGH | Rule check tasks are unbounded and can overlap indefinitely |
| KAI-MONITOR-006 | HIGH | Persisted rules are loaded without schema validation |
| KAI-MONITOR-007 | HIGH | Rule, alert, value and error data are exposed without authentication |
| KAI-MONITOR-008 | HIGH | Unauthenticated callers can delete, disable, enable and manually trigger rules |
| KAI-MONITOR-009 | MEDIUM | Action delivery status is ignored while alerts are recorded as fired |
| KAI-MONITOR-010 | MEDIUM | Cooldown and fire count are committed before delivery succeeds |
| KAI-MONITOR-011 | MEDIUM | Rule persistence is non-atomic and failure does not roll back memory state |
| KAI-MONITOR-012 | MEDIUM | Rule identifiers and text fields are weakly bounded |
| KAI-MONITOR-013 | MEDIUM | HTTP response size and JSON complexity are not bounded |
| KAI-MONITOR-014 | MEDIUM | New HTTP clients are created for every check and action batch |
| KAI-MONITOR-015 | MEDIUM | Background check tasks are not tracked or cancelled on shutdown |
| KAI-MONITOR-016 | MEDIUM | Alert history overflow silently discards older records |
| KAI-MONITOR-017 | MEDIUM | Error strings may expose internal network and endpoint details |
| KAI-MONITOR-018 | MEDIUM | Health is dependency- and scheduler-blind |
| KAI-MONITOR-019 | MEDIUM | Error-budget telemetry is never populated |
| KAI-MONITOR-020 | MEDIUM | Runtime state is process-local and inconsistent across workers |

---

## Monitor service: `monitor-service/app.py`

### KAI-MONITOR-001 — CRITICAL — Persistent arbitrary-URL HTTP monitoring
**Issue:** `POST /rules` requires no authentication or authorisation. A caller can create an enabled rule whose `source.type` is `http` and whose URL is arbitrary. The background loop repeatedly performs server-side GET requests to that URL at intervals as low as five seconds.  
**Risk:** This is persistent server-side request forgery. Callers can probe internal services, cloud metadata endpoints, loopback interfaces and network-restricted resources, with repeated scheduled requests surviving through `RULES_FILE` when configured.  
**Recommendation:** Require administrative authentication and restrict destinations using canonical URL parsing, DNS/IP resolution checks, approved schemes/domains and egress policy.  
**Status:** OPEN — immediate remediation required

### KAI-MONITOR-002 — CRITICAL — Browser-agent arbitrary navigation
**Issue:** A rule with `source.type: scrape` sends its arbitrary URL and caller-controlled CSS selector to `browser-agent /scrape`.  
**Risk:** The monitor becomes a persistent remote control layer for browser navigation and scraping, inheriting browser-agent access to internal web applications, authenticated sessions and local network resources.  
**Recommendation:** Enforce a destination allowlist and authenticated policy at both monitor and browser-agent boundaries.  
**Status:** OPEN — immediate remediation required

### KAI-MONITOR-003 — CRITICAL — Persistent alert automation
**Issue:** Unauthenticated rule creation can configure `notify` and `tts` actions, custom messages, critical urgency, five-second intervals and zero cooldown.  
**Risk:** A caller can establish durable desktop notification and speech spam that continues in the background, impersonates trusted monitoring and can survive restarts when persistence is enabled.  
**Recommendation:** Restrict rule/action creation to authorised operators, impose minimum cooldowns and require provenance-labelled approved templates.  
**Status:** OPEN — immediate remediation required

### KAI-MONITOR-004 — HIGH — Update endpoint bypasses validation
**Issue:** `PUT /rules/{rule_id}` accepts an arbitrary dictionary and applies `_rules[rule_id].update(updates)` directly. It does not re-run `RuleIn` validation.  
**Risk:** Callers can set intervals below five seconds or negative, invalid source/action/condition structures, arbitrary IDs/fields and malformed values that bypass all creation-time constraints. This materially expands SSRF, alert flooding and scheduler failure paths.  
**Recommendation:** Validate a complete replacement or a typed patch model and reject unknown fields.  
**Status:** OPEN

### KAI-MONITOR-005 — HIGH — Unbounded overlapping checks
**Issue:** `_watch_loop` launches `asyncio.create_task(_check_rule(rule))` whenever a rule is due and does not track whether the previous check is still running. Update validation bypass permits intervals of zero or negative values.  
**Risk:** Slow endpoints create overlapping checks every second, accumulating network requests and tasks without a concurrency bound.  
**Recommendation:** Maintain one in-flight task per rule with bounded global concurrency, deadlines and backoff.  
**Status:** OPEN

### KAI-MONITOR-006 — HIGH — Persisted rules bypass schema validation
**Issue:** Startup loads JSON objects directly into `_rules` using their embedded IDs. No `RuleIn` validation, type checking, duplicate handling or security review occurs.  
**Risk:** A modified/corrupt rules file can inject unsafe intervals, arbitrary structures and destinations that would not pass the creation schema.  
**Recommendation:** Validate and migrate every persisted rule before activation; quarantine invalid entries.  
**Status:** OPEN

### KAI-MONITOR-007 — HIGH — Sensitive monitoring state is publicly readable
**Issue:** `/rules`, `/alerts` and `/status` expose complete source URLs, selectors, conditions, actions, messages, last values, timestamps, fire counts and raw errors without authentication.  
**Risk:** Callers can discover internal endpoints, business thresholds, monitored assets, returned values and network diagnostics.  
**Recommendation:** Require scoped read access and redact credentials/query strings and sensitive values.  
**Status:** OPEN

### KAI-MONITOR-008 — HIGH — Rule control is unauthenticated
**Issue:** Delete, enable, disable and manual-check endpoints require no authentication.  
**Risk:** Callers can suppress legitimate monitoring, erase rules, reactivate dangerous rules or trigger immediate SSRF/browser/action checks.  
**Recommendation:** Require authenticated administrative authority and immutable audit events.  
**Status:** OPEN

### KAI-MONITOR-009 — MEDIUM — Delivery outcomes are ignored
**Issue:** `_fire_actions` suppresses exceptions and never checks notify/TTS status codes. It logs the rule as fired regardless.  
**Risk:** Alerts are recorded as delivered when downstream services rejected or failed them.  
**Recommendation:** Record separate condition, attempted, accepted and delivered states.  
**Status:** OPEN

### KAI-MONITOR-010 — MEDIUM — Cooldown is committed before success
**Issue:** `_last_fired` and `_fire_counts` are updated before `_fire_actions` runs.  
**Risk:** Failed delivery still starts the cooldown and increments counts, suppressing a legitimate retry.  
**Recommendation:** Commit cooldown after confirmed delivery or model retry state explicitly.  
**Status:** OPEN

### KAI-MONITOR-011 — MEDIUM — Persistence is non-atomic and inconsistent
**Issue:** `_save_rules` writes the complete JSON file directly and catches errors without propagating them. In-memory changes are returned as successful even if persistence failed.  
**Risk:** A crash can truncate the file; restart can lose accepted changes while callers were told they succeeded.  
**Recommendation:** Use atomic write-rename, fsync and transactional success semantics.  
**Status:** OPEN

### KAI-MONITOR-012 — MEDIUM — Text and identifier fields lack practical bounds
**Issue:** Rule ID, name, URL, extract path, selector, message, urgency and action list lengths are not bounded.  
**Risk:** Oversized rules consume memory, persistence, logs and notification/TTS capacity.  
**Recommendation:** Apply strict per-field, list and aggregate schema limits.  
**Status:** OPEN

### KAI-MONITOR-013 — MEDIUM — HTTP response allocation is unbounded
**Issue:** Direct HTTP checks call `resp.json()` without limiting response bytes, nesting depth or object complexity. Scrape responses are also accepted without output-size enforcement.  
**Risk:** A monitored endpoint can return a huge payload and exhaust memory or CPU.  
**Recommendation:** Stream with strict byte limits and validate bounded JSON/extracted output.  
**Status:** OPEN

### KAI-MONITOR-014 — MEDIUM — HTTP clients are recreated repeatedly
**Issue:** Each fetch and action batch constructs a new `httpx.AsyncClient`.  
**Risk:** Persistent rules cause unnecessary socket/TLS churn and resource pressure.  
**Recommendation:** Reuse lifecycle-managed clients with bounded connection pools.  
**Status:** OPEN

### KAI-MONITOR-015 — MEDIUM — Check tasks outlive scheduler ownership
**Issue:** The lifespan handler cancels only `_watch_loop`. Tasks created for `_check_rule` are not retained, awaited or cancelled.  
**Risk:** In-flight SSRF/browser/action requests can continue during shutdown and failures are not supervised.  
**Recommendation:** Track all child tasks in a task group and cancel/await them on shutdown.  
**Status:** OPEN

### KAI-MONITOR-016 — MEDIUM — Alert history overflow is silent
**Issue:** `_alert_history` is a bounded deque that automatically drops old entries.  
**Risk:** Flooding rules erase earlier alert evidence without a dropped-record metric.  
**Recommendation:** Persist alerts or record explicit overflow counts and retention policy.  
**Status:** OPEN

### KAI-MONITOR-017 — MEDIUM — Raw errors disclose internals
**Issue:** `_check_errors` stores `str(exc)` and `/status` returns it publicly. Logs also include raw error text.  
**Risk:** Internal URLs, DNS results, connection details and parser/index failures are exposed.  
**Recommendation:** Store stable error codes publicly and protect detailed diagnostics.  
**Status:** OPEN

### KAI-MONITOR-018 — MEDIUM — Health does not verify operation
**Issue:** `/health` always reports ok and only counts rules/alerts. It does not verify the scheduler task, HTTP/browser dependencies, persistence writability or action services.  
**Risk:** The service is considered ready while no checks or notifications can complete.  
**Recommendation:** Separate liveness, scheduler readiness and dependency health.  
**Status:** OPEN

### KAI-MONITOR-019 — MEDIUM — Error budget is not recorded
**Issue:** `_budget` is instantiated and exposed but no middleware calls `_budget.record`.  
**Risk:** Metrics appear authoritative while containing no request outcome data.  
**Recommendation:** Record actual status codes and exceptions consistently.  
**Status:** OPEN

### KAI-MONITOR-020 — MEDIUM — Runtime state is worker-local
**Issue:** Rules, values, timers, errors, alerts and fire counts are module-level memory.  
**Risk:** Multiple workers duplicate scheduler execution and alerts, expose inconsistent state and race on the shared rules file.  
**Recommendation:** Run one scheduler authority backed by transactional shared storage and leader election where needed.  
**Status:** OPEN

---

## Batch totals

- Findings: **20**
- Critical: **3**
- High: **5**
- Medium: **12**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **476**
- Critical: **51**
- High: **174**
- Medium: **248**
- Low: **3**

## Files materially reviewed in this batch

`monitor-service/app.py`.
