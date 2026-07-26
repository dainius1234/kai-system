# Kai Code Audit — House Doctor Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-DOCTOR-001 | CRITICAL | Unauthenticated observations can create trusted medical-report memories |
| KAI-DOCTOR-002 | HIGH | Unauthenticated observations can trigger WARNING and CRITICAL notifications |
| KAI-DOCTOR-003 | HIGH | Rule output recommends privileged and potentially disruptive host commands without execution safeguards |
| KAI-DOCTOR-004 | HIGH | Diagnosis classification uses broad substring matching and can be trivially manipulated |
| KAI-DOCTOR-005 | MEDIUM | `report_written` reports intent rather than confirmed persistence |
| KAI-DOCTOR-006 | MEDIUM | Memory and notification HTTP failures are not validated by status code |
| KAI-DOCTOR-007 | MEDIUM | Observation and world-state inputs are weakly bounded |
| KAI-DOCTOR-008 | MEDIUM | Recent diagnoses and full rule treatments are exposed without authentication |
| KAI-DOCTOR-009 | MEDIUM | Diagnosis history is process-local and lost on restart |
| KAI-DOCTOR-010 | MEDIUM | Duplicate observations generate repeated reports and alerts without deduplication |
| KAI-DOCTOR-011 | MEDIUM | World-state input is accepted but ignored |
| KAI-DOCTOR-012 | MEDIUM | Health reports ok without checking memory or notification dependencies |

---

## House Doctor: `house-doctor/app.py`

### KAI-DOCTOR-001 — CRITICAL — Unauthenticated medical-memory injection
**Issue:** `POST /diagnose` requires no authentication. Caller-supplied observation strings are classified and, when any rule matches, written to memu-core as `medical_report` memory under trusted user ID `keeper`. Source observations are included in metadata.  
**Risk:** Any reachable caller can insert fabricated system-health diagnoses and attacker-controlled observations into durable trusted memory, influencing future reasoning and operational decisions.  
**Recommendation:** Accept only authenticated, signed observations from approved collectors and preserve verifiable provenance through diagnosis and storage.  
**Status:** OPEN — immediate remediation required

### KAI-DOCTOR-002 — HIGH — Unauthenticated alert generation
**Issue:** The same unauthenticated diagnosis request can match WARNING or CRITICAL rules and send messages through `notify-service`.  
**Risk:** Callers can create false emergencies, notification spam and alert fatigue under the trusted House Doctor identity.  
**Recommendation:** Authenticate sources, deduplicate events and require evidence-backed severity transitions.  
**Status:** OPEN

### KAI-DOCTOR-003 — HIGH — Treatment text includes hazardous privileged commands
**Issue:** Rules return operational treatment instructions directly to callers and memory. D003 recommends killing/restarting processes; D008 recommends `sync; echo 3 > /proc/sys/vm/drop_caches`; other rules recommend container restarts. No privilege, workload, platform or change-control context is attached.  
**Risk:** Downstream autonomous or human consumers may execute disruptive host-level actions based on heuristic string matches rather than verified diagnosis.  
**Recommendation:** Separate observational diagnosis from executable remediation, classify hazardous actions and require authorised human/change-control approval.  
**Status:** OPEN

### KAI-DOCTOR-004 — HIGH — Diagnosis rules are easily manipulated
**Issue:** Classification relies on broad substring tests such as any observation containing `cpu` plus `%`, `high` or `anomaly`, and any text containing `meeting`, `calendar`, `event in` or `schedule`. No structured sensor values, thresholds, source identity or freshness are required.  
**Risk:** Ordinary or adversarial prose can manufacture tags and trigger serious diagnoses unrelated to actual system state.  
**Recommendation:** Consume typed, signed measurements and evaluate explicit thresholds and temporal correlation.  
**Status:** OPEN

### KAI-DOCTOR-005 — MEDIUM — Persistence success is falsely reported
**Issue:** `/diagnose` returns `report_written: true` whenever diagnoses exist, regardless of whether the memu-core POST succeeded.  
**Risk:** Callers believe a report is durable when it may only exist in the local ring buffer.  
**Recommendation:** Return separate attempted, locally-buffered and durably-acknowledged states.  
**Status:** OPEN

### KAI-DOCTOR-006 — MEDIUM — HTTP response failures are ignored
**Issue:** Memory and notification POSTs do not call `raise_for_status` or inspect response bodies. Only transport exceptions are caught.  
**Risk:** 4xx/5xx failures are silently treated as success.  
**Recommendation:** Validate acknowledgements and use durable idempotent delivery.  
**Status:** OPEN

### KAI-DOCTOR-007 — MEDIUM — Inputs are insufficiently bounded
**Issue:** `observations` has no maximum item count or per-item length. `world_state` accepts an arbitrary dictionary with no size or schema constraints.  
**Risk:** Oversized requests consume memory, CPU, response capacity and downstream memory metadata storage.  
**Recommendation:** Apply strict typed schemas and aggregate size limits.  
**Status:** OPEN

### KAI-DOCTOR-008 — MEDIUM — Internal diagnosis data is publicly readable
**Issue:** `/diagnoses/recent` and `/rules` expose recent source observations, diagnoses, treatment commands, differentials and complete rule logic without authentication.  
**Risk:** Callers can obtain sensitive operational history and tune malicious observations to trigger specific rules.  
**Recommendation:** Require scoped operational access and redact source details.  
**Status:** OPEN

### KAI-DOCTOR-009 — MEDIUM — History is worker-local and volatile
**Issue:** Recent diagnoses are stored only in a module-level deque.  
**Risk:** Multiple workers expose inconsistent histories and restart erases the local diagnostic record.  
**Recommendation:** Use a shared versioned store or explicitly treat the endpoint as non-authoritative cache.  
**Status:** OPEN

### KAI-DOCTOR-010 — MEDIUM — No event identity or deduplication
**Issue:** Repeated identical requests generate new reports and notifications every time.  
**Risk:** Retries, loops or malicious callers can flood memory and alerts with duplicate diagnoses.  
**Recommendation:** Require source event IDs and idempotency windows.  
**Status:** OPEN

### KAI-DOCTOR-011 — MEDIUM — Accepted world state is ignored
**Issue:** `DiagnosisRequest` accepts `world_state`, but diagnosis and reporting never read it.  
**Risk:** Callers and documentation may assume diagnoses are grounded in structured world state when they are based solely on observation strings.  
**Recommendation:** Remove the unused field or incorporate validated evidence explicitly.  
**Status:** OPEN

### KAI-DOCTOR-012 — MEDIUM — Health is dependency-blind
**Issue:** `/health` always returns ok and does not test memu-core or notify-service readiness.  
**Risk:** The service is advertised as functional when it cannot persist reports or deliver critical alerts.  
**Recommendation:** Separate liveness from dependency and delivery readiness.  
**Status:** OPEN

---

## Batch totals

- Findings: **12**
- Critical: **1**
- High: **3**
- Medium: **8**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **341**
- Critical: **37**
- High: **137**
- Medium: **164**
- Low: **3**

## Files materially reviewed in this batch

`house-doctor/app.py`.
