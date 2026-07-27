# Kai Code Audit — Sysmetrics Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_SYSMETRICS_WORLD_ANCHOR.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-SYSMX-001 | HIGH | Host-wide CPU and memory values are presented without cgroup/container quota context |
| KAI-SYSMX-002 | HIGH | Process and network data may be container-namespace scoped while CPU/memory/load are host scoped |
| KAI-SYSMX-003 | HIGH | `uptime_seconds` is service-process uptime rather than operating-system uptime |
| KAI-SYSMX-004 | HIGH | Snapshot components are collected at different instants but returned as one coherent state |
| KAI-SYSMX-005 | HIGH | No snapshot timestamp, generation or collector-scope identity is returned |
| KAI-SYSMX-006 | HIGH | Every request performs a fresh full collection with no cache, rate limit or workload admission |
| KAI-SYSMX-007 | HIGH | `/processes` enumerates every visible process before applying the top-N response limit |
| KAI-SYSMX-008 | HIGH | Missing/denied disks and processes are silently omitted without a completeness marker |
| KAI-SYSMX-009 | HIGH | Temperature and battery endpoints trust raw sensor values without range or finiteness validation |
| KAI-SYSMX-010 | HIGH | Health does not verify permissions, collectors, cgroup scope or successful recent collection |
| KAI-SYSMX-011 | HIGH | Multiple workers independently perform expensive sampling and expose inconsistent process observations |
| KAI-SYSMX-012 | HIGH | No immutable audit links caller, collection scope, snapshot generation and returned host details |
| KAI-SYSMX-013 | MEDIUM | CPU frequency is queried twice and can produce inconsistent/avoidable work |
| KAI-SYSMX-014 | MEDIUM | Logical/physical CPU counts may be null but no unavailable state is attached |
| KAI-SYSMX-015 | MEDIUM | Load averages are exposed without normalisation by CPU quota or host/container scope |
| KAI-SYSMX-016 | MEDIUM | Disk sizes use decimal units while field names do not identify GB convention |
| KAI-SYSMX-017 | MEDIUM | Network counters are cumulative since namespace startup but no interval/rate basis is given |
| KAI-SYSMX-018 | MEDIUM | Process names, statuses, mountpoints and sensor labels are returned without control-character normalisation |
| KAI-SYSMX-019 | MEDIUM | Temperature sensor groups and entry counts have no response cardinality limit |
| KAI-SYSMX-020 | MEDIUM | Battery `secs_left` semantics omit the source timestamp and discharge/charge direction context |
| KAI-SYSMX-021 | MEDIUM | Public metrics expose telemetry without administrative authentication |
| KAI-SYSMX-022 | MEDIUM | Missing shared-runtime imports silently replace telemetry with no-op fallbacks |
| KAI-SYSMX-023 | MEDIUM | Collection failures have no stable partial-snapshot schema or protected trace ID |
| KAI-SYSMX-024 | MEDIUM | Wall-clock service uptime can move backwards or forwards after clock changes |
| KAI-SYSMX-025 | MEDIUM | No history or baseline distinguishes transient readings from sustained resource pressure |
| KAI-SYSMX-026 | MEDIUM | The service has no lifecycle-owned collector, cached sampling cadence or graceful metrics drain |

---

## High-severity findings

### KAI-SYSMX-001 — HIGH — Host values lack cgroup context
**Issue:** psutil CPU count, memory and load commonly reflect host-kernel resources, while the container may be limited to a smaller CPU/memory quota.  
**Risk:** Consumers calculate utilisation/headroom against the wrong capacity and may miss container exhaustion or falsely report safety.  
**Recommendation:** report both cgroup limits/usage and host values with explicit scope.  
**Status:** OPEN

### KAI-SYSMX-002 — HIGH — Mixed measurement scopes
Visible processes/network interfaces may be namespaced to the container while CPU/memory/load are host-wide; the API labels all as one system snapshot.

### KAI-SYSMX-003 — HIGH — Uptime is mislabeled
`_start` records module import time. The result is service uptime, not system boot uptime implied by the service contract.

### KAI-SYSMX-004 — HIGH — Non-atomic snapshot
CPU waits 200 ms, then memory, disks, network and load are collected sequentially with no common measurement instant.

### KAI-SYSMX-005 — HIGH — No collection identity
The response has no `collected_at`, duration, generation, host/container ID or namespace/cgroup scope.

### KAI-SYSMX-006 — HIGH — Unmetered collection
Every anonymous request launches complete synchronous collection; no cache, semaphore or quota protects it.

### KAI-SYSMX-007 — HIGH — Full process enumeration
`process_iter` walks all visible processes and builds a complete list before sorting/slicing.

### KAI-SYSMX-008 — HIGH — Partial evidence appears complete
Permission and disappearance exceptions are swallowed, but no omitted counts or partial/degraded flag is returned.

### KAI-SYSMX-009 — HIGH — Raw sensor trust
Null, NaN, infinity, negative or implausible temperature/battery values can raise or be returned as operational facts.

### KAI-SYSMX-010 — HIGH — Readiness-blind health
Health checks only whether psutil imported and always reports ok; it does not run/validate any collection.

### KAI-SYSMX-011 — HIGH — Worker duplication/divergence
Each worker samples independently and sees a different instantaneous/process state.

### KAI-SYSMX-012 — HIGH — Missing reconnaissance audit
No tamper-evident event records the actor, purpose and exact sensitive host/process snapshot exposed.

---

## Medium-severity findings

### KAI-SYSMX-013 — MEDIUM — Duplicate frequency calls
`psutil.cpu_freq()` is invoked once for truthiness and again for `.current`.

### KAI-SYSMX-014 — MEDIUM — Null topology ambiguity
CPU counts may be unavailable, but values are returned without source/error context.

### KAI-SYSMX-015 — MEDIUM — Unnormalised load
Raw 1/5/15-minute load averages are not divided by available/cgroup CPUs.

### KAI-SYSMX-016 — MEDIUM — Ambiguous size units
Values divide by `1e9`/`1e6` but field names do not distinguish decimal GB/MB from binary GiB/MiB.

### KAI-SYSMX-017 — MEDIUM — Counter/rate ambiguity
Network totals have no interface list, namespace start, sample duration or rate calculation.

### KAI-SYSMX-018 — MEDIUM — Raw operating-system text
Names, paths and labels may include control/confusable characters and are displayed/logged downstream.

### KAI-SYSMX-019 — MEDIUM — Unbounded sensor cardinality
Every sensor group/entry is serialised with no field/item/byte cap.

### KAI-SYSMX-020 — MEDIUM — Battery-time ambiguity
Seconds-left lacks timestamp, whether charging/discharging and confidence/source state.

### KAI-SYSMX-021 — MEDIUM — Public telemetry
`/metrics` has no administrative access control.

### KAI-SYSMX-022 — MEDIUM — Silent telemetry downgrade
If shared runtime import fails, basic logging/no-op ErrorBudget are used with normal health.

### KAI-SYSMX-023 — MEDIUM — Weak failure contract
Unhandled platform errors become generic 500 responses rather than a versioned partial snapshot with protected diagnostics.

### KAI-SYSMX-024 — MEDIUM — Wall-clock uptime
`time.time() - _start` is not monotonic.

### KAI-SYSMX-025 — MEDIUM — No trend evidence
Consumers receive one sample and cannot distinguish sustained pressure, spikes or sampling noise.

### KAI-SYSMX-026 — MEDIUM — Missing collector lifecycle
No lifespan/background collector owns sampling, cache publication, shutdown or worker leadership.

---

## Batch totals

- Findings: **26**
- Critical: **0**
- High: **12**
- Medium: **14**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,345**
- Critical: **189**
- High: **1,171**
- Medium: **982**
- Low: **3**

## Files materially reviewed

`sysmetrics/app.py`, existing Sysmetrics audit findings and Dashboard/Supervisor/Metrics Gateway consumption paths.
