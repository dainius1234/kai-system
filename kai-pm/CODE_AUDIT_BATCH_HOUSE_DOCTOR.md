# Kai Code Audit — House Doctor Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-DOCTOR-001 | CRITICAL | Unauthenticated observations can create trusted medical-report memories |
| KAI-DOCTOR-002 | CRITICAL | Unauthenticated observations can trigger fabricated WARNING and CRITICAL system notifications |
| KAI-DOCTOR-003 | HIGH | Rule output recommends privileged and potentially disruptive host commands without execution safeguards |
| KAI-DOCTOR-004 | HIGH | Diagnosis classification uses broad substring matching and can be trivially manipulated |
| KAI-DOCTOR-005 | HIGH | Recent diagnoses expose source observations and system-health history without authentication |
| KAI-DOCTOR-006 | HIGH | All callers share one global diagnosis stream consumed by agentic/Cortex workflows |
| KAI-DOCTOR-007 | HIGH | No rate limit, cooldown or deduplication prevents repeated memory and alert flooding |
| KAI-DOCTOR-008 | MEDIUM | `report_written` reports intent rather than confirmed persistence |
| KAI-DOCTOR-009 | MEDIUM | Memory and notification HTTP failures are not validated by status code |
| KAI-DOCTOR-010 | MEDIUM | Observation and world-state inputs are unbounded |
| KAI-DOCTOR-011 | MEDIUM | Caller-supplied `world_state` is accepted but ignored |
| KAI-DOCTOR-012 | MEDIUM | Full diagnosis rules and treatment commands are exposed publicly |
| KAI-DOCTOR-013 | MEDIUM | A new HTTP client is created for every diagnosis report |
| KAI-DOCTOR-014 | MEDIUM | Diagnosis history is process-local, unsynchronised and lost on restart |
| KAI-DOCTOR-015 | MEDIUM | Health reports ok without validating memory or notification readiness |
| KAI-DOCTOR-016 | MEDIUM | Service URLs have no startup validation |

---

## House Doctor: `house-doctor/app.py`

### KAI-DOCTOR-001 — CRITICAL — Trusted memory injection through fabricated observations
**Issue:** `POST /diagnose` requires no authentication or authorisation. Caller-controlled observation strings are classified, converted into a `medical_report`, appended locally and posted to `memu-core /memory/memorize` as user `keeper`.  
**Risk:** Any reachable caller can create persistent false system-health memories represented as House Doctor diagnoses. Those reports can be consumed by later agentic and Cortex cycles as trusted historical evidence.  
**Recommendation:** Accept only signed observations from approved telemetry services, preserve source provenance and require corroboration before durable diagnosis storage.  
**Status:** OPEN — immediate remediation required

### KAI-DOCTOR-002 — CRITICAL — Fabricated WARNING/CRITICAL alert injection
**Issue:** When manipulated observations match WARNING or CRITICAL rules, the unauthenticated endpoint posts a system notification containing the generated diagnosis.  
**Risk:** Callers can impersonate the system-health authority, generate urgent warnings and repeatedly disrupt the operator with false critical events.  
**Recommendation:** Authenticate diagnosis requests, sign alert provenance and require verified multi-sensor evidence for urgent notifications.  
**Status:** OPEN — immediate remediation required

### KAI-DOCTOR-003 — HIGH — Disruptive command recommendations lack safeguards
**Issue:** Rule treatments recommend commands/actions including killing processes, restarting services and `sync; echo 3 > /proc/sys/vm/drop_caches`. They are returned as treatment guidance without privilege, workload or safety checks.  
**Risk:** Operators or downstream automation may follow disruptive host-level actions derived from weak heuristics, causing outages, data-loss risk or severe performance degradation.  
**Recommendation:** Remove privileged command prescriptions from automatic output; require evidence-backed runbooks, approval and execution safeguards.  
**Status:** OPEN

### KAI-DOCTOR-004 — HIGH — Classification is trivially manipulable
**Issue:** Tags are assigned by broad substring tests. Any CPU/RAM mention containing `%`, `high`, `anomaly` or `pressure` is treated as elevated; any `aqi`, `pm2.5`, calendar, meeting or schedule mention triggers corresponding tags. No numeric thresholds or source identity are checked.  
**Risk:** Benign text and attacker-crafted observations can force diagnoses, including the CRITICAL system-wide anomaly rule.  
**Recommendation:** Use typed telemetry schemas with validated numeric thresholds, source authentication and temporal correlation.  
**Status:** OPEN

### KAI-DOCTOR-005 — HIGH — Diagnostic history is publicly disclosed
**Issue:** `GET /diagnoses/recent` requires no authentication and returns severity, active tags, diagnosis, treatment, all matched rules and up to ten original source observations.  
**Risk:** Callers can inspect recent operational incidents, system-health conditions and potentially sensitive observation text.  
**Recommendation:** Require scoped operational access and redact raw observations by default.  
**Status:** OPEN

### KAI-DOCTOR-006 — HIGH — One global diagnosis stream contaminates every workflow
**Issue:** `_recent_diagnoses` is a single module-level deque with no user, session, source or tenant partition. The module states that agentic reads it at the start of proactive cycles, and Cortex polls `/diagnoses/recent`.  
**Risk:** One caller’s fabricated diagnosis becomes shared context for all subsequent users and autonomous workflows.  
**Recommendation:** Partition state by authenticated principal and source, and expose only provenance-verified diagnoses to downstream agents.  
**Status:** OPEN

### KAI-DOCTOR-007 — HIGH — Repeated report/alert flooding is unrestricted
**Issue:** Every matching `/diagnose` request creates another local report and attempts memory plus notification delivery. There is no rate limit, cooldown, event identity, deduplication or one-in-flight control.  
**Risk:** A caller can flood long-term memory, evict genuine entries from the 20-item ring buffer and generate sustained alert spam.  
**Recommendation:** Enforce authenticated quotas, event deduplication, cooldowns and bounded queues.  
**Status:** OPEN

### KAI-DOCTOR-008 — MEDIUM — `report_written` is a false persistence acknowledgement
**Issue:** The response sets `report_written` to `len(diagnoses) > 0`, regardless of whether memu-core accepted the report. `_write_medical_report` suppresses memory exceptions.  
**Risk:** Callers and monitoring systems are told a durable report exists when only an in-process deque append may have occurred.  
**Recommendation:** Return separate local-recorded, memory-accepted and notification-accepted states.  
**Status:** OPEN

### KAI-DOCTOR-009 — MEDIUM — Downstream status codes are ignored
**Issue:** Memory and notification calls do not inspect HTTP status or response content. Only transport exceptions are caught.  
**Risk:** Rejected requests are silently treated as completed attempts, while local state and API responses imply success.  
**Recommendation:** Validate acknowledgements and record durable delivery outcomes.  
**Status:** OPEN

### KAI-DOCTOR-010 — MEDIUM — Inputs are unbounded
**Issue:** `observations` has no maximum item count or string length. `world_state` accepts arbitrary nested dictionaries with no size/depth limit.  
**Risk:** Oversized requests consume parsing, classification, memory payload and response capacity.  
**Recommendation:** Enforce strict body, list, field and nesting limits.  
**Status:** OPEN

### KAI-DOCTOR-011 — MEDIUM — `world_state` is ignored
**Issue:** `DiagnosisRequest.world_state` is accepted but never read during classification or diagnosis.  
**Risk:** Callers may believe contextual evidence is considered when diagnosis is based solely on substring matches in observations.  
**Recommendation:** Remove the field or implement a typed, evidence-backed use with explicit output traceability.  
**Status:** OPEN

### KAI-DOCTOR-012 — MEDIUM — Rule internals are public
**Issue:** `GET /rules` exposes every trigger pattern, severity, diagnosis, treatment and differential without authentication.  
**Risk:** Callers can precisely craft inputs to trigger desired diagnoses and alerts, and privileged command recommendations are broadly disclosed.  
**Recommendation:** Restrict rule access and publish only non-sensitive capability metadata.  
**Status:** OPEN

### KAI-DOCTOR-013 — MEDIUM — HTTP client churn
**Issue:** Each report creates a new `httpx.AsyncClient` for memory and notification requests.  
**Risk:** Repeated diagnoses cause unnecessary socket and connection-pool churn.  
**Recommendation:** Reuse a lifecycle-managed client with bounded pools.  
**Status:** OPEN

### KAI-DOCTOR-014 — MEDIUM — History is volatile and worker-local
**Issue:** The diagnosis deque is process memory without locking or shared storage. Multiple workers expose different histories; restart erases all local reports.  
**Risk:** Downstream agent context is inconsistent and genuine incidents can disappear while memory persistence may also have failed.  
**Recommendation:** Use transactional shared storage with explicit retention and event IDs.  
**Status:** OPEN

### KAI-DOCTOR-015 — MEDIUM — Health is readiness-blind
**Issue:** `/health` always returns ok and only reports rule/history counts. It does not verify memu-core or notify-service connectivity, delivery schemas or storage readiness.  
**Risk:** Orchestration treats a diagnosis service as ready when it cannot persist or alert.  
**Recommendation:** Separate liveness from memory and notification readiness.  
**Status:** OPEN

### KAI-DOCTOR-016 — MEDIUM — Service destinations are unvalidated
**Issue:** `MEMU_URL` and `NOTIFY_URL` are accepted directly from environment configuration without scheme/host policy.  
**Risk:** Misconfiguration can route sensitive diagnosis reports and system alerts to unintended destinations.  
**Recommendation:** Validate approved internal service URLs at startup.  
**Status:** OPEN

---

## Batch totals

- Findings: **16**
- Critical: **2**
- High: **5**
- Medium: **9**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **658**
- Critical: **74**
- High: **228**
- Medium: **353**
- Low: **3**

## Files materially reviewed in this batch

`house-doctor/app.py`.
