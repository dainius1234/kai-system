# Kai Code Audit — Orchestration Stubs and Health Sweep Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-STUB-001 | HIGH | Deployed workspace-manager reports healthy while implementing no workspace capability |
| KAI-STUB-002 | HIGH | Deprecated orchestrator reports `status: ok` despite implementing no orchestration or risk authority |
| KAI-STUB-003 | HIGH | Health sweep treats every HTTP 200 response as `UP` regardless of semantic readiness |
| KAI-STUB-004 | HIGH | Health sweep lists an orchestrator service that is absent from the full Compose deployment |
| KAI-STUB-005 | HIGH | Health sweep omits multiple actively deployed services and therefore cannot represent stack health |
| KAI-STUB-006 | HIGH | Documented `--host` targeting is not implemented and checks always use localhost |
| KAI-STUB-007 | MEDIUM | Health sweep does not verify service identity, allowing any HTTP server on the port to satisfy the check |
| KAI-STUB-008 | MEDIUM | Workspace and orchestrator health endpoints expose no capability/readiness detail |
| KAI-STUB-009 | MEDIUM | Workspace manager service identity is environment-controlled and weakly validated |
| KAI-STUB-010 | MEDIUM | Scorecard write failures are silently suppressed |
| KAI-STUB-011 | MEDIUM | Raw connection and parser errors are persisted in the scorecard |
| KAI-STUB-012 | MEDIUM | Service ports and process ports are parsed without validation |

---

## Active placeholder: `workspace-manager/app.py`

### KAI-STUB-001 — HIGH — Active no-op service reports healthy
**Issue:** `workspace-manager` is published on host port 8060 and has a Compose healthcheck, but its application implements only `/health`. The endpoint always returns `status: ok`; no workspace create, inspect, isolate, persist, delete or lifecycle operation exists.  
**Risk:** Deployment, monitoring and operators can treat workspace isolation/management as present when the capability is entirely absent. Any downstream design relying on this boundary has no implemented enforcement.  
**Recommendation:** Remove the service from active deployment until implemented, or return an explicit non-ready/not-implemented state that fails the healthcheck.  
**Status:** OPEN

## Deprecated placeholder: `orchestrator/app.py`

### KAI-STUB-002 — HIGH — Non-functional authority reports ready
**Issue:** The module declares itself deprecated and states it does nothing beyond `/health`, yet that endpoint returns `status: ok` and identifies the service as orchestrator. The comments describe the intended role as a potential final-risk authority.  
**Risk:** If launched or discovered by health tooling, the service can be mistaken for an active orchestration/risk-control layer even though it performs no validation, routing or decision enforcement.  
**Recommendation:** Remove the runtime artefact or make its endpoint return a machine-readable disabled/deprecated non-ready state.  
**Status:** OPEN

## Health sweep: `scripts/health_sweep.py`

### KAI-STUB-003 — HIGH — HTTP 200 is equated with health
**Issue:** `check_health` classifies every HTTP 200 response as `status: UP`; it only copies the response body’s `status` into a free-text detail field. It does not treat `degraded`, `disabled`, stub mode, inactive schedulers or false readiness as failure.  
**Risk:** Services repeatedly confirmed in this audit as returning `status: ok` while non-functional are counted green, creating a materially misleading system scorecard.  
**Recommendation:** Define service-specific readiness contracts and require validated capability state, dependency state and freshness.  
**Status:** OPEN

### KAI-STUB-004 — HIGH — Inventory includes a non-deployed service
**Issue:** The sweep checks `orchestrator` on port 8050, but `docker-compose.full.yml` does not define or publish the orchestrator service.  
**Risk:** The documented full-stack sweep generates a guaranteed down result for a service not present in the deployment, obscuring genuine failures and encouraging operators to ignore the scorecard.  
**Recommendation:** Generate the inventory from the deployed Compose model and explicitly distinguish intentionally absent/deprecated components.  
**Status:** OPEN

### KAI-STUB-005 — HIGH — Inventory omits active services
**Issue:** The manually maintained `SERVICES` list excludes multiple services actively defined in the full Compose stack, including memu-core-introspect, telegram-bot, skill-hunter, house-doctor, letta-agent, financial-awareness and others.  
**Risk:** The sweep can return “ALL GREEN” while active security-sensitive services are unavailable or compromised.  
**Recommendation:** Discover services from a versioned deployment manifest and require coverage checks that fail when deployed services are missing.  
**Status:** OPEN

### KAI-STUB-006 — HIGH — Documented remote host option is ignored
**Issue:** The usage text advertises `--host 172.20.0.3`, but the script never parses command-line arguments. `HOST` remains the hard-coded string `localhost`.  
**Risk:** Operators can believe they tested a remote deployment while the script actually probed local ports, producing a false health conclusion for the target system.  
**Recommendation:** Implement strict argument parsing and print/record the resolved target from parsed input.  
**Status:** OPEN

### KAI-STUB-007 — MEDIUM — Service identity is not verified
**Issue:** A response from the expected port is accepted without checking a service identifier, version or signed instance identity.  
**Risk:** A different process, stale container or generic HTTP responder can satisfy the health check and be attributed to the intended service.  
**Recommendation:** Validate expected service name, deployment/version identity and authenticated health provenance.  
**Status:** OPEN

### KAI-STUB-008 — MEDIUM — Placeholder health lacks readiness semantics
**Issue:** Workspace-manager and orchestrator health responses contain no implementation state, required dependency status, capability list or readiness evidence.  
**Risk:** Consumers cannot distinguish operational functionality from a process that merely started.  
**Recommendation:** Publish structured liveness/readiness/capability status and fail readiness when required behaviour is absent.  
**Status:** OPEN

### KAI-STUB-009 — MEDIUM — Workspace identity is configuration-controlled
**Issue:** Workspace-manager uses `SERVICE_NAME` from the environment as the FastAPI title and returned service name, defaulting to the generic value `service`. No allowlist or consistency check exists.  
**Risk:** Misconfiguration can cause misleading identity in health responses and monitoring without affecting the HTTP status.  
**Recommendation:** Hard-code the authoritative service identity or validate it against deployment metadata.  
**Status:** OPEN

### KAI-STUB-010 — MEDIUM — Scorecard persistence failure is invisible
**Issue:** All exceptions while writing `output/health_scorecard.json` are suppressed with `except Exception: pass`.  
**Risk:** The sweep can print a successful result while no audit artefact was saved, and automation has no indication that scorecard persistence failed.  
**Recommendation:** Report the write failure and return a non-zero exit code when durable output is required.  
**Status:** OPEN

### KAI-STUB-011 — MEDIUM — Raw errors enter the scorecard
**Issue:** Connection, JSON parsing and other exception strings are truncated and stored as `detail` in the generated JSON.  
**Risk:** Network addresses, library diagnostics and local environment details are persisted and potentially published with CI artefacts.  
**Recommendation:** Store stable error codes and protected trace references.  
**Status:** OPEN

### KAI-STUB-012 — MEDIUM — Port configuration is unvalidated
**Issue:** Workspace-manager and orchestrator parse `PORT` directly with `int`; the health sweep hard-codes ports without validating range, collisions or consistency with Compose.  
**Risk:** Invalid or drifted port configuration causes startup failure or checks the wrong process while monitoring logic remains unaware of the mismatch.  
**Recommendation:** Validate a single generated service/port manifest at startup and in health tooling.  
**Status:** OPEN

---

## Batch totals

- Findings: **12**
- Critical: **0**
- High: **6**
- Medium: **6**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **726**
- Critical: **81**
- High: **254**
- Medium: **388**
- Low: **3**

## Files materially reviewed in this batch

`workspace-manager/app.py`, `orchestrator/app.py`, `scripts/health_sweep.py`, and the relevant deployment definitions in `docker-compose.full.yml`.
