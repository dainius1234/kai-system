# Kai Code Audit — Sysmetrics Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-SYSMET-001 | HIGH | Process, memory, disk, network, temperature and battery telemetry is exposed without authentication |
| KAI-SYSMET-002 | HIGH | Container-local telemetry is represented and consumed as host/system health |
| KAI-SYSMET-003 | HIGH | Unauthenticated callers can repeatedly force full process and filesystem enumeration |
| KAI-SYSMET-004 | HIGH | Snapshot processing can block worker threads and scale linearly with mounts/processes |
| KAI-SYSMET-005 | HIGH | Internal mountpoints, process names and PIDs are disclosed |
| KAI-SYSMET-006 | HIGH | Unverified telemetry is promoted into Cortex operational and wellbeing recommendations |
| KAI-SYSMET-007 | MEDIUM | Per-process CPU percentages are generally unsampled/zero and produce unreliable ranking |
| KAI-SYSMET-008 | MEDIUM | `uptime_seconds` is service-process uptime rather than host uptime |
| KAI-SYSMET-009 | MEDIUM | Disk permission failures are silently discarded, producing incomplete snapshots |
| KAI-SYSMET-010 | MEDIUM | Snapshot components are collected at different instants without a coherent sample timestamp |
| KAI-SYSMET-011 | MEDIUM | CPU frequency is queried twice and can produce inconsistent or avoidable work |
| KAI-SYSMET-012 | MEDIUM | Health reports ok in stub mode and does not validate telemetry capability |
| KAI-SYSMET-013 | MEDIUM | Error-budget telemetry is exposed but never populated |
| KAI-SYSMET-014 | MEDIUM | Negative or extreme `TOP_PROCESSES` values alter slicing and response volume |
| KAI-SYSMET-015 | MEDIUM | Sensor/process labels and collection volume are not bounded |
| KAI-SYSMET-016 | MEDIUM | Port and process-limit configuration are not validated at startup |

---

## Sysmetrics: `sysmetrics/app.py`, deployment configuration

### KAI-SYSMET-001 — HIGH — Public system/process telemetry
**Issue:** `/snapshot`, `/processes`, `/temperature`, `/battery`, `/health` and `/metrics` require no authentication. They expose CPU topology/frequency/load, memory capacity/use, disk mountpoints/capacity, network counters, process IDs/names/resource use, temperature sensors and battery state. The service is published on host port 8035.  
**Risk:** Any reachable caller can profile runtime capacity, workloads, storage layout and operational state for reconnaissance and targeted denial of service.  
**Recommendation:** Require scoped operational authentication and expose minimised aggregate telemetry only.  
**Status:** OPEN

### KAI-SYSMET-002 — HIGH — Container view is misrepresented as system health
**Issue:** The minimal Compose service runs in an ordinary container without host PID namespace, host filesystem mounts or host network namespace. `psutil` therefore observes primarily the container/cgroup namespace, but the service/module and downstream consumers describe the output as system health.  
**Risk:** Host CPU, memory, disks, processes and sensors can be absent or materially different, while automation and operators trust container-local values as authoritative machine state.  
**Recommendation:** Define the measurement scope explicitly. Collect host telemetry through a dedicated least-privilege host agent or label all fields as container-local.  
**Status:** OPEN

### KAI-SYSMET-003 — HIGH — Public expensive enumeration
**Issue:** Every `/snapshot` request enumerates disk partitions and queries usage for each mount; every `/processes` request iterates all visible processes and reads multiple attributes. No authentication, rate limit, cache or concurrency bound exists.  
**Risk:** Repeated callers can consume worker threads, procfs/sysfs operations and filesystem-stat capacity.  
**Recommendation:** Scrape on a protected schedule, cache bounded snapshots and enforce caller quotas.  
**Status:** OPEN

### KAI-SYSMET-004 — HIGH — Collection blocks request workers
**Issue:** Endpoints are synchronous and perform `psutil.cpu_percent(interval=0.2)`, full mount iteration, disk usage calls, process iteration and sensor reads inline. FastAPI executes sync handlers in its threadpool, so each request occupies a worker until collection completes.  
**Risk:** Concurrent unauthenticated requests can exhaust the threadpool and deny health/telemetry service.  
**Recommendation:** Use one bounded sampler and serve immutable cached snapshots.  
**Status:** OPEN

### KAI-SYSMET-005 — HIGH — Runtime topology disclosure
**Issue:** `/processes` returns PID, process name, memory and status; `/snapshot` returns exact mountpoint paths and capacities.  
**Risk:** Callers can identify running software, deployment structure, storage targets and potential high-value processes.  
**Recommendation:** Return only aggregate resource metrics and keep process/mount detail behind administrative authorisation.  
**Status:** OPEN

### KAI-SYSMET-006 — HIGH — Weak telemetry becomes agent authority
**Issue:** Cortex polls `/snapshot` and uses CPU, memory and related values to infer system pressure, cognitive load and recommendations. Sysmetrics provides no authentication, provenance, host/container scope or snapshot confidence.  
**Risk:** Incomplete or container-local measurements are elevated into operational and wellbeing guidance as if they described the operator’s machine/environment.  
**Recommendation:** Require signed provenance, explicit scope and corroborated thresholds before telemetry can influence agent decisions.  
**Status:** OPEN

### KAI-SYSMET-007 — MEDIUM — Process CPU ranking is unreliable
**Issue:** `process_iter(["cpu_percent", ...])` reads instantaneous/cached per-process percentages without establishing a measurement interval for each process. Fresh values are commonly zero or stale, yet results are sorted as the top CPU processes.  
**Risk:** The endpoint presents an authoritative ranking that may not reflect actual load.  
**Recommendation:** Maintain sampled deltas across timed collection cycles and include sample window/timestamp.  
**Status:** OPEN

### KAI-SYSMET-008 — MEDIUM — Uptime label is misleading
**Issue:** `uptime_seconds` is calculated from module import time (`time.time() - _start`), not system boot time.  
**Risk:** Consumers can interpret service-process age as host uptime or stability.  
**Recommendation:** Rename it `service_uptime_seconds` and separately report validated host boot time when available.  
**Status:** OPEN

### KAI-SYSMET-009 — MEDIUM — Missing disks are hidden
**Issue:** `PermissionError` from `disk_usage` is silently ignored. No missing/inaccessible mount list or completeness flag is returned.  
**Risk:** Critical full or inaccessible filesystems can disappear from snapshots while the service appears healthy.  
**Recommendation:** Report per-mount collection status and aggregate completeness.  
**Status:** OPEN

### KAI-SYSMET-010 — MEDIUM — Snapshot is not temporally coherent
**Issue:** CPU sampling waits 200 ms, then memory, each disk, network and load are collected sequentially. No observation timestamp or start/end window accompanies the result.  
**Risk:** Fields can describe different moments during rapidly changing load but are treated as one atomic state.  
**Recommendation:** Include sample interval/generation timestamps and use a scheduled coherent collector.  
**Status:** OPEN

### KAI-SYSMET-011 — MEDIUM — CPU frequency is queried twice
**Issue:** `psutil.cpu_freq()` is called once in the condition and again to obtain `.current`.  
**Risk:** The two reads can differ or the second can fail/return `None`; it also performs unnecessary system calls.  
**Recommendation:** Capture one result and validate it before use.  
**Status:** OPEN

### KAI-SYSMET-012 — MEDIUM — Health is readiness-blind
**Issue:** `/health` always returns `status: ok`, including when `psutil` is unavailable and all telemetry endpoints operate in stub/error mode. It does not verify that expected sensors/namespaces are accessible.  
**Risk:** Monitoring treats a non-observing or incorrectly scoped service as ready.  
**Recommendation:** Separate liveness, psutil capability and required telemetry readiness/scope.  
**Status:** OPEN

### KAI-SYSMET-013 — MEDIUM — Error budget is inert
**Issue:** `budget` is returned by `/metrics`, but no endpoint records successes, failures or latency.  
**Risk:** Reliability telemetry appears implemented but contains no operational evidence.  
**Recommendation:** Record all collection outcomes and partial failures.  
**Status:** OPEN

### KAI-SYSMET-014 — MEDIUM — Process-limit slicing can expand output
**Issue:** `TOP_PROCESSES` is parsed directly and used as `procs[:TOP_PROCESSES]`. A negative value returns all but the final N records rather than rejecting the configuration; extreme positive values expose the complete process table.  
**Risk:** Misconfiguration defeats the intended response limit and increases disclosure/resource use.  
**Recommendation:** Enforce a small positive bounded range.  
**Status:** OPEN

### KAI-SYSMET-015 — MEDIUM — Collection/result sizes are unbounded
**Issue:** All visible processes, disk partitions and sensor entries are collected before any process slicing; names, mountpoints and sensor labels have no length bounds.  
**Risk:** Large namespaces or hostile labels can consume memory and inflate responses/Cortex context.  
**Recommendation:** Bound collection counts and field lengths before allocation/publication.  
**Status:** OPEN

### KAI-SYSMET-016 — MEDIUM — Startup configuration lacks validation
**Issue:** Port and `TOP_PROCESSES` are parsed directly without safe ranges.  
**Risk:** Invalid values crash startup or silently alter disclosure and load characteristics.  
**Recommendation:** Validate typed startup configuration and fail with explicit diagnostics.  
**Status:** OPEN

---

## Batch totals

- Findings: **16**
- Critical: **0**
- High: **6**
- Medium: **10**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **813**
- Critical: **87**
- High: **288**
- Medium: **435**
- Low: **3**

## Files materially reviewed in this batch

`sysmetrics/app.py` and the relevant `sysmetrics` deployment definition in `docker-compose.minimal.yml`.
