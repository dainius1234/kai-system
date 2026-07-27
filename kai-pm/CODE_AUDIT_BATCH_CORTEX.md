# Kai Code Audit — Cortex Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-CORTEX-001 | CRITICAL | Unauthenticated conversation observations can poison agent context and learned operator preferences |
| KAI-CORTEX-002 | CRITICAL | Untrusted clipboard and sensor content is promoted into agent-consumed Cortex state |
| KAI-CORTEX-003 | HIGH | Cortex state exposes sensitive clipboard, calendar, Git, system and diagnostic context without authentication |
| KAI-CORTEX-004 | HIGH | Session identifiers are accepted but ignored, merging all callers into one global behavioural model |
| KAI-CORTEX-005 | HIGH | Deterministic heuristics issue operational and wellbeing recommendations from weak/unverified sensor text |
| KAI-CORTEX-006 | HIGH | Background refresh task is untracked and has no shutdown lifecycle |
| KAI-CORTEX-007 | MEDIUM | Stable sensor readings are incorrectly downgraded as potentially stale |
| KAI-CORTEX-008 | MEDIUM | Sensor failures are silently converted into absence of evidence |
| KAI-CORTEX-009 | MEDIUM | Shared state and behavioural accumulators are unsynchronised and worker-local |
| KAI-CORTEX-010 | MEDIUM | Message, session and specialist fields are unbounded |
| KAI-CORTEX-011 | MEDIUM | Activity-hour inference uses UTC rather than the operator’s local timezone |
| KAI-CORTEX-012 | MEDIUM | New HTTP clients are created for every sensor request |
| KAI-CORTEX-013 | MEDIUM | Health reports ok when Cortex is disabled or has never refreshed |
| KAI-CORTEX-014 | MEDIUM | Refresh interval, URLs and Boolean configuration are weakly validated |
| KAI-CORTEX-015 | MEDIUM | Deprecated startup event handling obscures task ownership and restart behaviour |

---

## Cortex: `cortex/app.py`

### KAI-CORTEX-001 — CRITICAL — Behavioural/context poisoning through unauthenticated turns
**Issue:** `POST /observe_turn` requires no authentication or caller verification. Arbitrary `user_message` content updates `_topic_history`, bridge state, message-length statistics and hourly activity counts used to derive tacit operator rules.  
**Risk:** Any reachable caller can fabricate conversation history, force context-shift signals and teach the system false preferences such as response style or alert timing. These outputs are then returned in Cortex state for agentic context assembly.  
**Recommendation:** Accept only signed observations from the authenticated conversation service, bind them to a verified user/session and isolate behavioural models per principal.  
**Status:** OPEN — immediate remediation required

### KAI-CORTEX-002 — CRITICAL — Poisoned sensor data is elevated into trusted context
**Issue:** The refresh loop retrieves unauthenticated service outputs, including full clipboard content from `/latest`, and inserts them into `level1_facts`, tags, summaries, implications and intent hypotheses. No provenance, trust boundary or untrusted-data treatment is applied.  
**Risk:** The previously confirmed clipboard injection path and compromised sensor services become a direct agent-context poisoning chain. Attacker-controlled text can be presented as operator/environment fact and influence recommendations or later reasoning.  
**Recommendation:** Require signed provenance for every sensor event, apply source-specific trust policy and represent external text as untrusted quoted observations rather than instructions/facts.  
**Status:** OPEN — immediate remediation required

### KAI-CORTEX-003 — HIGH — Sensitive situational state is publicly readable
**Issue:** `GET /state` and `/health` require no authentication. State can contain clipboard excerpts, calendar shape, Git branch/work status, system load, Docker health, diagnoses, intent inferences and learned behavioural rules.  
**Risk:** Callers can reconstruct the operator’s current work, schedule pressure, system condition and behavioural profile.  
**Recommendation:** Require user-scoped access and minimise/redact raw facts and inferred personal data.  
**Status:** OPEN

### KAI-CORTEX-004 — HIGH — Global cross-session behavioural contamination
**Issue:** `TurnObservation.session_id` is parsed but never used. All observations update one global topic history and tacit model.  
**Risk:** Different users, sessions, tests and attackers contaminate each other’s inferred preferences and context boundaries. A single hostile caller changes the model for every legitimate interaction.  
**Recommendation:** Partition all topic/tacit state by authenticated user and session with explicit retention.  
**Status:** OPEN

### KAI-CORTEX-005 — HIGH — Weak heuristics produce authoritative recommendations
**Issue:** Regex and keyword rules infer critical system state, operator sprinting, deadlines, cognitive-load risk and recommendations such as restarting services, postponing meetings, opening windows or committing work. Inputs are unverified summaries and crude numeric extraction.  
**Risk:** False or poisoned signals generate operational and personal recommendations presented without calibrated uncertainty or source evidence.  
**Recommendation:** Preserve source evidence/confidence, require corroboration and label heuristic outputs as tentative rather than actionable conclusions.  
**Status:** OPEN

### KAI-CORTEX-006 — HIGH — Refresh task has no lifecycle ownership
**Issue:** Startup calls `asyncio.create_task(_refresh_loop())` without retaining the task. No shutdown handler cancels or awaits it.  
**Risk:** Reloads/tests can create duplicate refresh loops; shutdown abandons network work and failures outside `_refresh` are not supervised.  
**Recommendation:** Manage the task through FastAPI lifespan with one retained task, cancellation and awaited termination.  
**Status:** OPEN

### KAI-CORTEX-007 — MEDIUM — Stable readings are treated as stale
**Issue:** `_update_credibility` reduces credibility to 0.5 whenever the same string appears for three cycles. Many valid sensors naturally remain unchanged, such as weather summaries, quiet systems, calendar shape or clipboard content.  
**Risk:** Correct stable evidence is systematically discounted while changing attacker-controlled text retains full credibility.  
**Recommendation:** Use timestamps/heartbeats and source-specific freshness semantics rather than value equality.  
**Status:** OPEN

### KAI-CORTEX-008 — MEDIUM — Sensor errors disappear silently
**Issue:** `_fetch` catches every exception and non-200 response, returning `None`; `asyncio.gather` exceptions are also converted to `None`. State does not expose which sensors failed or how stale prior evidence is.  
**Risk:** Absence of facts is misread as normality, allowing “signals within normal range” or “calm” summaries while critical sensors are unavailable.  
**Recommendation:** Represent unavailable, stale and failed sources explicitly and prevent normal-state conclusions when required evidence is missing.  
**Status:** OPEN

### KAI-CORTEX-009 — MEDIUM — Shared mutable state is unsafe and non-authoritative
**Issue:** `_state`, histories and counters are module-level mutable objects updated by refresh and request handlers without locks. Multiple workers maintain separate models.  
**Risk:** Concurrent updates can be lost or inconsistent; users receive different Cortex states depending on worker, and restart erases learning.  
**Recommendation:** Use transactional per-user storage or a single authoritative state worker with atomic snapshots.  
**Status:** OPEN

### KAI-CORTEX-010 — MEDIUM — Observation fields are unbounded
**Issue:** `session_id`, `user_message`, `specialist` and timestamp strings have no maximum lengths. Keyword extraction processes the complete message before any bound.  
**Risk:** Oversized submissions consume memory and CPU and distort long-term message-length statistics.  
**Recommendation:** Enforce strict schema and body-size limits.  
**Status:** OPEN

### KAI-CORTEX-011 — MEDIUM — Operator activity model uses the wrong clock basis
**Issue:** Hourly activity is recorded using `datetime.now(timezone.utc).hour`, not the operator’s configured/local timezone.  
**Risk:** The derived rule “Most active around HH:00” can be offset from the user’s real schedule and is then used to calibrate alert thresholds.  
**Recommendation:** Store timezone-aware event timestamps and infer activity in the authenticated user’s configured timezone.  
**Status:** OPEN

### KAI-CORTEX-012 — MEDIUM — Sensor polling recreates clients
**Issue:** Every `_fetch` call creates a new `httpx.AsyncClient`; nine are created each refresh cycle.  
**Risk:** Continuous polling creates unnecessary socket and connection-pool churn.  
**Recommendation:** Reuse lifecycle-managed clients with bounded pools.  
**Status:** OPEN

### KAI-CORTEX-013 — MEDIUM — Health is readiness-blind
**Issue:** `/health` always returns `status: ok`, including when `FF_CORTEX=false`, refresh count is zero or every sensor request fails.  
**Risk:** Orchestration treats disabled or empty Cortex output as ready situational intelligence.  
**Recommendation:** Separate liveness, enabled state, refresh freshness and required-sensor readiness.  
**Status:** OPEN

### KAI-CORTEX-014 — MEDIUM — Configuration is weakly validated
**Issue:** Refresh interval and all service URLs are accepted directly. Boolean settings silently interpret any non-`true` value as false. Zero/negative refresh intervals can create tight loops or runtime errors.  
**Risk:** Misconfiguration causes excessive polling, silent disablement or routing to unintended services.  
**Recommendation:** Validate typed startup configuration with safe ranges and approved service destinations.  
**Status:** OPEN

### KAI-CORTEX-015 — MEDIUM — Startup mechanism obscures modern lifecycle guarantees
**Issue:** The service uses `@app.on_event("startup")` for an untracked background task rather than one lifespan owner.  
**Risk:** Test clients, reloads and future framework changes can produce inconsistent startup/shutdown behaviour and duplicated tasks.  
**Recommendation:** Consolidate startup and shutdown into a single lifespan context.  
**Status:** OPEN

---

## Batch totals

- Findings: **15**
- Critical: **2**
- High: **4**
- Medium: **9**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **564**
- Critical: **63**
- High: **200**
- Medium: **298**
- Low: **3**

## Files materially reviewed in this batch

`cortex/app.py`.
