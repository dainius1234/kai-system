# Kai Code Audit — Docker Watcher Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-DWATCH-001 | CRITICAL | Host Docker daemon socket is mounted into a network-published service; `:ro` does not make Docker API operations read-only |
| KAI-DWATCH-002 | HIGH | Container identities, images, states and lifecycle metadata are exposed without authentication |
| KAI-DWATCH-003 | HIGH | Failed polls preserve and serve stale container state as current |
| KAI-DWATCH-004 | HIGH | Health reports `ok` before any successful poll and after Docker access fails |
| KAI-DWATCH-005 | HIGH | SDK polling lists only running containers, making stopped/exited containers invisible to `/unhealthy` |
| KAI-DWATCH-006 | HIGH | Restart counts are read from the wrong Docker inspect location and remain falsely zero |
| KAI-DWATCH-007 | HIGH | Configurable Docker endpoint is accepted without destination or transport validation |
| KAI-DWATCH-008 | MEDIUM | Docker SDK clients are created repeatedly and never closed |
| KAI-DWATCH-009 | MEDIUM | Docker polling has no service-level total deadline or bounded dedicated worker pool |
| KAI-DWATCH-010 | MEDIUM | Container cache and poll state are process-local and unsynchronised |
| KAI-DWATCH-011 | MEDIUM | Refresh-task cancellation is not awaited during shutdown |
| KAI-DWATCH-012 | MEDIUM | Raw Docker and subprocess errors are exposed through health |
| KAI-DWATCH-013 | MEDIUM | Error-budget telemetry is exposed but never populated |
| KAI-DWATCH-014 | MEDIUM | Untrusted container names are inserted into Cortex-consumed natural-language summaries |
| KAI-DWATCH-015 | MEDIUM | Container result count and metadata lengths are not bounded |
| KAI-DWATCH-016 | MEDIUM | Poll interval, Docker endpoint and port configuration are not validated |

---

## Docker watcher: `docker-watcher/app.py`, deployment configuration

### KAI-DWATCH-001 — CRITICAL — Docker daemon control boundary is mounted into the service
**Issue:** `docker-compose.minimal.yml` mounts `/var/run/docker.sock` into the host-published Docker Watcher container. The mount is labelled `:ro`, but a Unix socket is an API transport: read-only filesystem mounting does not restrict which Docker Engine API methods a connected process may send. The application creates a full Docker SDK client rather than a capability-limited proxy.  
**Risk:** Any code-execution compromise inside this unauthenticated network service can use the daemon connection to create privileged containers, mount host filesystems and take control of the host wherever socket permissions permit the intended polling. This collapses container isolation.  
**Recommendation:** Never mount the Docker socket directly. Use a narrowly scoped authenticated read-only metrics proxy with an explicit endpoint allowlist, or collect required telemetry outside the application trust domain.  
**Status:** OPEN — immediate remediation required

### KAI-DWATCH-002 — HIGH — Public container inventory disclosure
**Issue:** `GET /containers`, `/unhealthy`, `/summary` and `/health` require no authentication. Records expose short container IDs, names, image names/tags, runtime state, health, start time, exit code and alleged restart counts. The service is published on host port 8041.  
**Risk:** Any reachable caller can map stack components, versions, deployment naming, operational incidents and restart behaviour for targeted exploitation.  
**Recommendation:** Require scoped operational authentication and return privacy-minimised aggregate state by default.  
**Status:** OPEN

### KAI-DWATCH-003 — HIGH — Stale container state survives poll failure
**Issue:** On any polling exception, only `_poll_error` changes. `_containers` and `_last_poll` retain the prior successful snapshot, while `/containers`, `/unhealthy` and `/summary` do not expose an error or freshness state.  
**Risk:** Operators, dashboard and Cortex can treat outdated topology and health as current during prolonged Docker-daemon failure. A previously healthy snapshot can conceal a live outage.  
**Recommendation:** Attach generation time/error/freshness to every response and expire the snapshot when it exceeds a strict age.  
**Status:** OPEN

### KAI-DWATCH-004 — HIGH — Health is readiness-blind
**Issue:** `/health` always returns `status: ok`, including before the first poll, after repeated poll errors or when the mounted socket is inaccessible to the non-root container user.  
**Risk:** Compose and external health tooling keep a non-observing Docker monitor in service and downstream consumers continue trusting its cached output.  
**Recommendation:** Separate process liveness from verified Docker access, successful-poll freshness and refresh-task state.  
**Status:** OPEN

### KAI-DWATCH-005 — HIGH — Stopped containers cannot be detected
**Issue:** `_poll_via_sdk` calls `client.containers.list(all=False)`, which returns running containers only. `/unhealthy` is described as showing containers not in running/healthy state, but stopped, exited and failed containers never enter `_containers`.  
**Risk:** The service can report the fleet normal while expected services have exited or failed to start.  
**Recommendation:** Query all expected/deployed containers and compare them against an authoritative service inventory.  
**Status:** OPEN

### KAI-DWATCH-006 — HIGH — Restart monitoring reads the wrong field
**Issue:** The code reads `state.get("RestartCount", 0)` from `attrs["State"]`. Docker inspect exposes restart count as a top-level container attribute, not within `State`. The resulting value therefore remains zero in normal responses.  
**Risk:** `/unhealthy` and `/summary` cannot detect restart loops using the advertised `restarts > 3` rule, creating false normality.  
**Recommendation:** Parse the validated Docker inspect schema and test restart-loop detection against real fixtures.  
**Status:** OPEN

### KAI-DWATCH-007 — HIGH — Docker destination is configuration-controlled
**Issue:** `DOCKER_HOST` is accepted directly from environment configuration and Docker SDK `from_env()` consumes Docker connection environment. No allowlist of Unix socket path, host, scheme, TLS material or daemon identity is enforced.  
**Risk:** Compromised configuration can redirect privileged daemon polling to an unintended local or remote Docker endpoint and potentially expose configured client credentials/certificates.  
**Recommendation:** Pin one approved telemetry endpoint with mutual authentication and no general Docker control capability.  
**Status:** OPEN

### KAI-DWATCH-008 — MEDIUM — Docker clients are leaked
**Issue:** `_poll_via_sdk` constructs `docker_sdk.from_env()` on every poll and never calls `client.close()`.  
**Risk:** Repeated polling can leak connection-pool and file-descriptor resources and produces unnecessary daemon connection churn.  
**Recommendation:** Create one lifecycle-managed client and close it during shutdown.  
**Status:** OPEN

### KAI-DWATCH-009 — MEDIUM — Poll work lacks a bounded execution domain
**Issue:** Polling is delegated to the default executor. The application defines no total poll deadline, dedicated worker count or cancellation mechanism for SDK calls; cancelling the coroutine does not stop blocked executor work.  
**Risk:** Slow/unresponsive daemon operations can occupy shared executor threads and accumulate across restarts or failures.  
**Recommendation:** Use a bounded dedicated worker with strict Docker client and total-operation timeouts.  
**Status:** OPEN

### KAI-DWATCH-010 — MEDIUM — State is volatile and worker-local
**Issue:** Container cache, timestamps, errors and task reference are module-level process memory.  
**Risk:** Multiple workers run separate daemon pollers and return inconsistent snapshots; restart erases history.  
**Recommendation:** Run one watcher authority and publish immutable timestamped snapshots to shared telemetry storage.  
**Status:** OPEN

### KAI-DWATCH-011 — MEDIUM — Shutdown does not await task termination
**Issue:** Lifespan shutdown calls `_refresh_task.cancel()` but does not await it or close the Docker client/executor work.  
**Risk:** Polling may continue after shutdown begins and failures/resources are not observed or released cleanly.  
**Recommendation:** Await cancellation and close all client/worker resources within a bounded lifespan shutdown.  
**Status:** OPEN

### KAI-DWATCH-012 — MEDIUM — Internal errors are public
**Issue:** Complete exception strings from Docker SDK or subprocess polling are retained in `_poll_error` and returned by `/health`.  
**Risk:** Callers learn socket paths, permissions, daemon endpoints, executable and network diagnostics.  
**Recommendation:** Return stable error codes and protected trace identifiers.  
**Status:** OPEN

### KAI-DWATCH-013 — MEDIUM — Error-budget metrics are inert
**Issue:** `budget` is created and returned from `/metrics`, but no request or polling outcome is recorded.  
**Risk:** Monitoring receives empty reliability telemetry for a critical fleet-observation component.  
**Recommendation:** Record classified poll/request outcomes and latency.  
**Status:** OPEN

### KAI-DWATCH-014 — MEDIUM — Container names become untrusted agent context
**Issue:** `/summary` inserts container names directly into natural-language text for high-restart warnings. Cortex polls this endpoint and promotes the summary into broader agent context without provenance or instruction/data separation.  
**Risk:** Attacker- or operator-controlled container names can inject misleading text into system context, while stale or incorrect health claims gain additional authority.  
**Recommendation:** Return typed identifiers and metrics through an authenticated channel; do not insert raw names into privileged prompts.  
**Status:** OPEN

### KAI-DWATCH-015 — MEDIUM — Inventory volume and metadata are unbounded
**Issue:** Every running container returned by the daemon is accumulated and exposed. Names, image tags and timestamp strings have no per-field or aggregate response limits.  
**Risk:** A host with many containers or oversized metadata can consume memory, response bandwidth and Cortex context capacity.  
**Recommendation:** Enforce expected-service allowlists, page results and bound all metadata.  
**Status:** OPEN

### KAI-DWATCH-016 — MEDIUM — Configuration lacks validation
**Issue:** Refresh interval and port are parsed directly; `DOCKER_HOST` is unrestricted. Zero/negative intervals can create tight loops or runtime errors, and invalid ports fail at startup.  
**Risk:** Misconfiguration produces uncontrolled polling, false readiness or unsafe daemon routing.  
**Recommendation:** Validate typed startup configuration with strict ranges and one approved telemetry destination.  
**Status:** OPEN

---

## Batch totals

- Findings: **16**
- Critical: **1**
- High: **6**
- Medium: **9**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **797**
- Critical: **87**
- High: **282**
- Medium: **425**
- Low: **3**

## Files materially reviewed in this batch

`docker-watcher/app.py`, `docker-watcher/Dockerfile`, `docker-watcher/requirements.txt`, and the relevant `docker-watcher` deployment definition in `docker-compose.minimal.yml`.
