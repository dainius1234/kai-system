# Kai Code Audit — Git and Docker Watchers Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-WATCH-001 | HIGH | Git repository metadata and working-tree state are exposed without authentication |
| KAI-WATCH-002 | HIGH | Docker inventory and health state are exposed without authentication |
| KAI-WATCH-003 | HIGH | Watcher services serve stale host state indefinitely after polling failure |
| KAI-WATCH-004 | HIGH | Docker watcher requires access to the privileged Docker control socket |
| KAI-WATCH-005 | MEDIUM | Git polling loop can terminate permanently on an uncaught inspection failure |
| KAI-WATCH-006 | MEDIUM | Docker SDK polling has no explicit request timeout and does not close the client |
| KAI-WATCH-007 | MEDIUM | Health endpoints report `ok` before successful polling and during failures |
| KAI-WATCH-008 | MEDIUM | Raw poll errors and filesystem/network details are exposed |
| KAI-WATCH-009 | MEDIUM | Watcher state is process-local and inconsistent across workers |
| KAI-WATCH-010 | MEDIUM | Configuration intervals and watch paths are not validated |
| KAI-WATCH-011 | MEDIUM | Error-budget metrics are exposed but polling outcomes are never recorded |
| KAI-WATCH-012 | MEDIUM | Missing Git upstream is represented as zero ahead and zero behind |
| KAI-WATCH-013 | LOW | Docker subprocess fallback cannot determine health, restart or exit state accurately |

---

## Git watcher: `git-watcher/app.py`

### KAI-WATCH-001 — HIGH — Unauthenticated repository-state disclosure
**Issue:** `/repos`, `/repos/{index}`, `/dirty` and `/summary` require no authentication. Responses expose absolute filesystem paths, branch names, commit hashes, commit messages, commit authors, commit dates, stash counts, ahead/behind state and counts of modified and untracked files.  
**Risk:** Any network-reachable caller can map development directories, active work, personnel identities and release state, providing valuable reconnaissance and exposing confidential project activity.  
**Recommendation:** Restrict the service to authenticated operational consumers and minimise returned metadata by scope.  
**Status:** OPEN

### KAI-WATCH-005 — MEDIUM — Poll loop can die permanently
**Issue:** `_poll_loop` has no outer exception handler. Although `_inspect_repo` catches many Git failures, an unexpected executor, configuration, cancellation-adjacent or result-processing exception exits the background task.  
**Risk:** Repository state stops updating while the API remains available and reports cached data as healthy.  
**Recommendation:** Supervise the worker, expose task state and restart or fail readiness on termination.  
**Status:** OPEN

### KAI-WATCH-012 — MEDIUM — Unknown upstream state is reported as synchronised
**Issue:** When no upstream exists or `rev-list` fails, `ahead` and `behind` remain initialised to zero.  
**Risk:** Consumers cannot distinguish “synchronised” from “not measurable,” leading to false clean/safe conclusions.  
**Recommendation:** Represent unavailable measurements explicitly as `null` or a classified state.  
**Status:** OPEN

---

## Docker watcher: `docker-watcher/app.py`

### KAI-WATCH-002 — HIGH — Unauthenticated container-state disclosure
**Issue:** `/containers`, `/unhealthy` and `/summary` expose container IDs, names, image tags, status, health, restarts, start times and exit codes without authentication.  
**Risk:** Callers can enumerate architecture, service names, versions and operational weaknesses.  
**Recommendation:** Require authenticated operational read scopes and redact unnecessary identifiers.  
**Status:** OPEN

### KAI-WATCH-004 — HIGH — Service depends on the Docker control socket
**Issue:** The service is designed to mount `/var/run/docker.sock`, which normally grants broad Docker daemon control equivalent to host-level administrative capability if the process or dependency is compromised.  
**Risk:** A vulnerability in this internet-facing-style FastAPI process, Docker SDK, parser or dependency can become a host/container escape path through the mounted socket.  
**Recommendation:** Replace direct socket access with a narrowly scoped metrics exporter or hardened socket proxy exposing only required read operations.  
**Status:** OPEN

### KAI-WATCH-006 — MEDIUM — Docker client lifecycle and timeout are unbounded
**Issue:** `docker_sdk.from_env()` is created on every poll without an explicit timeout or `close()`.  
**Risk:** Daemon stalls can occupy executor threads and repeated polling can leak transport resources.  
**Recommendation:** Use a retained client with bounded request deadlines and deterministic closure/reconnection.  
**Status:** OPEN

### KAI-WATCH-013 — LOW — Subprocess fallback presents incomplete state as real values
**Issue:** The CLI fallback hard-codes health=`none`, restarts=`0` and exit_code=`0` for every running container.  
**Risk:** `/unhealthy` and summaries can miss restart storms and health failures when the SDK is unavailable.  
**Recommendation:** Mark unavailable fields as unknown or query inspect data explicitly.  
**Status:** OPEN

---

## Shared watcher findings

### KAI-WATCH-003 — HIGH — Stale state is served indefinitely
**Issue:** Docker polling failures retain the old container cache. Git task failure similarly leaves the last snapshot available. Neither service enforces maximum age or clearly marks stale responses.  
**Risk:** Agentic context and operators can rely on obsolete repository or container state during a prolonged failure.  
**Recommendation:** Attach freshness and snapshot identity, reject data past a bounded age and expose degraded readiness.  
**Status:** OPEN

### KAI-WATCH-007 — MEDIUM — Health is not readiness-aware
**Issue:** Both `/health` endpoints always return `status: ok`, including before the first successful poll. Docker also reports ok while `poll_error` is populated.  
**Risk:** Watchdogs treat unavailable host-state collection as functional.  
**Recommendation:** Separate liveness, dependency readiness, worker state and data freshness.  
**Status:** OPEN

### KAI-WATCH-008 — MEDIUM — Internal diagnostics are exposed
**Issue:** Docker returns `_poll_error` directly. Git repository records expose error strings and configured paths.  
**Risk:** Filesystem, daemon, permissions and runtime details leak to callers.  
**Recommendation:** Return stable error categories and protected trace IDs only.  
**Status:** OPEN

### KAI-WATCH-009 — MEDIUM — Snapshot state is worker-local
**Issue:** Both services use module-level lists and timestamps.  
**Risk:** Multiple workers expose inconsistent snapshots and restart clears all state.  
**Recommendation:** Use a single collector with shared versioned snapshots or enforce one worker explicitly.  
**Status:** OPEN

### KAI-WATCH-010 — MEDIUM — Configuration is not validated
**Issue:** Refresh intervals are directly parsed without positive minimum/maximum checks. Git watch paths are accepted as arbitrary colon-separated paths.  
**Risk:** Invalid values can crash startup, create tight loops or expose unintended mounted directories.  
**Recommendation:** Validate deployment configuration and allowlist permitted roots.  
**Status:** OPEN

### KAI-WATCH-011 — MEDIUM — Error-budget telemetry is inert
**Issue:** Both services instantiate and expose `ErrorBudget` but never record poll or endpoint outcomes.  
**Risk:** Metrics cannot represent actual collection health.  
**Recommendation:** Record classified successes, failures, latency and freshness violations.  
**Status:** OPEN

---

## Batch totals

- Findings: **13**
- Critical: **0**
- High: **4**
- Medium: **8**
- Low: **1**

## Provisional repository totals after all logged batches

- Findings: **270**
- Critical: **30**
- High: **115**
- Medium: **122**
- Low: **3**

## Files materially reviewed in this batch

`git-watcher/app.py`, `docker-watcher/app.py`.
