# Kai Code Audit — Notify Service Build Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_NOTIFY_SERVICE.md` or `CODE_AUDIT_BATCH_NOTIFY_SERVICE_EXTENSION.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-NOTIFYB-001 | HIGH | The configured Docker build cannot locate `requirements.txt` from the repository-root build context |
| KAI-NOTIFYB-002 | HIGH | The configured Docker build cannot locate `app.py` from the repository-root build context |
| KAI-NOTIFYB-003 | HIGH | `COPY ../../common` attempts to copy outside the Docker build context |
| KAI-NOTIFYB-004 | MEDIUM | The configured `NOTIFY_SEND_TIMEOUT` value is never used |

---

### KAI-NOTIFYB-001 — HIGH — Missing requirements source
**Issue:** Compose builds Notify with repository root as the Docker context, but the Dockerfile executes `COPY requirements.txt ./`. No repository-root `requirements.txt` exists; the service dependency file is `output/notify/requirements.txt`.  
**Risk:** The image cannot build from the committed Compose/Dockerfile pair.  
**Recommendation:** Copy the explicit service path and make image construction a mandatory CI check.  
**Status:** OPEN

### KAI-NOTIFYB-002 — HIGH — Missing application source
**Issue:** The Dockerfile executes `COPY app.py ./`, but no repository-root `app.py` exists. The service source is `output/notify/app.py`.  
**Risk:** Image construction fails before the application can be installed or started.  
**Recommendation:** Use `COPY output/notify/app.py ./app.py` or a service-directory build context.  
**Status:** OPEN

### KAI-NOTIFYB-003 — HIGH — Invalid out-of-context COPY
**Issue:** Docker `COPY` sources resolve relative to the build context. With repository root as context, `COPY ../../common /app/common` traverses outside the context and is rejected.  
**Risk:** The checked-in image definition is not reproducibly deployable.  
**Recommendation:** Copy `common/` directly from the root context and verify the resulting source digest.  
**Status:** OPEN

### KAI-NOTIFYB-004 — MEDIUM — Dead timeout configuration
**Issue:** `NOTIFY_SEND_TIMEOUT` is parsed from `NOTIFY_SEND_TIMEOUT_MS`, but `_try_notify_send()` always passes a hard-coded three-second subprocess timeout.  
**Risk:** Operators believe the timeout is configurable while deployment changes have no effect.  
**Recommendation:** Validate and use one canonical timeout setting, separately from notification display expiry.  
**Status:** OPEN

---

## Batch totals

- Findings: **4**
- Critical: **0**
- High: **3**
- Medium: **1**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,187**
- Critical: **189**
- High: **1,089**
- Medium: **906**
- Low: **3**

## Files materially reviewed

`output/notify/Dockerfile`, `output/notify/app.py`, `output/notify/requirements.txt`, Notify Compose deployment, and the two existing Notify audit batches.
