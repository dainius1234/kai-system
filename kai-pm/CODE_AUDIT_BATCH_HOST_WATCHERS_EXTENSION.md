# Kai Code Audit — Git and Docker Watchers Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_HOST_WATCHERS.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-WATCHX-001 | HIGH | Docker Watcher lists only running containers, so stopped, exited and dead services disappear from monitoring |
| KAI-WATCHX-002 | HIGH | Docker restart counts are read from `State.RestartCount` instead of the top-level inspect field |
| KAI-WATCHX-003 | HIGH | Docker unhealthy detection does not independently evaluate container status |
| KAI-WATCHX-004 | HIGH | Containers without Docker healthchecks are treated as normal rather than unverified |
| KAI-WATCHX-005 | HIGH | Docker summary can claim every container is normal while required containers are stopped or absent |
| KAI-WATCHX-006 | HIGH | The watcher has no expected-service inventory and cannot detect a missing container |
| KAI-WATCHX-007 | HIGH | Container names and image tags are inserted into downstream natural-language context without a safe provenance boundary |
| KAI-WATCHX-008 | HIGH | A read-only Docker socket mount does not make Docker API access read-only |
| KAI-WATCHX-009 | HIGH | Docker polling uses the shared default executor with no bounded daemon-call queue |
| KAI-WATCHX-010 | HIGH | Git status polling can execute repository-configured fsmonitor hooks |
| KAI-WATCHX-011 | HIGH | Git subprocesses inherit global/system/user repository configuration and the full service environment |
| KAI-WATCHX-012 | HIGH | Git command stdout/stderr are fully buffered before parsing or limits |
| KAI-WATCHX-013 | HIGH | One repository can consume roughly the sum of many independent ten-second Git command deadlines |
| KAI-WATCHX-014 | HIGH | Repository polling is sequential and has no whole-cycle deadline |
| KAI-WATCHX-015 | HIGH | Git summary reports errored repositories as clean when change counts remain zero |
| KAI-WATCHX-016 | HIGH | `/dirty` silently omits repositories that could not be inspected |
| KAI-WATCHX-017 | HIGH | A single errored repository can be described as “no uncommitted changes” with an empty branch |
| KAI-WATCHX-018 | HIGH | Repository-controlled commit messages, authors and branch text enter Dashboard/Agentic context without normalisation |
| KAI-WATCHX-019 | HIGH | Repository indexes are unstable identifiers and can refer to different repositories after configuration/order changes |
| KAI-WATCHX-020 | HIGH | Neither watcher records an immutable engine/repository snapshot generation or source identity |
| KAI-WATCHX-021 | HIGH | Multiple workers independently poll Docker/Git and expose divergent snapshots while amplifying host load |
| KAI-WATCHX-022 | HIGH | Watcher summaries have no authenticated principal, purpose or downstream-context audit |
| KAI-WATCHX-023 | MEDIUM | Docker short IDs are returned without full immutable IDs or daemon identity |
| KAI-WATCHX-024 | MEDIUM | Docker `started_at`, status and image fields are trusted without typed schema validation |
| KAI-WATCHX-025 | MEDIUM | Docker CLI fallback parses tab-delimited human output rather than a versioned machine schema |
| KAI-WATCHX-026 | MEDIUM | Docker fallback ignores the configured `DOCKER_HOST` identity in response provenance |
| KAI-WATCHX-027 | MEDIUM | No Docker event stream is used, creating up to one refresh interval of detection delay |
| KAI-WATCHX-028 | MEDIUM | Zero containers is indistinguishable from not-yet-polled or an empty/failed daemon result in summaries |
| KAI-WATCHX-029 | MEDIUM | Git uses shortened commit hashes without reporting their abbreviation length or collision state |
| KAI-WATCHX-030 | MEDIUM | Git status counts do not preserve filenames, rename pairs, submodule state or ignored-state completeness |
| KAI-WATCHX-031 | MEDIUM | Untracked-directory collapsing and Git configuration can make file counts incomparable between repositories |
| KAI-WATCHX-032 | MEDIUM | Stash counting buffers and splits the complete stash list even though only a count is required |
| KAI-WATCHX-033 | MEDIUM | Git command failures are selectively swallowed, producing partial snapshots without field-level availability flags |
| KAI-WATCHX-034 | MEDIUM | Per-repository last-success time and poll duration are not returned |
| KAI-WATCHX-035 | MEDIUM | Repository paths are not canonicalised to stable logical IDs before indexing and display |
| KAI-WATCHX-036 | MEDIUM | Git and Docker snapshots use wall-clock times without monotonic sequence numbers |
| KAI-WATCHX-037 | MEDIUM | Public metrics expose telemetry without administrative authentication |
| KAI-WATCHX-038 | MEDIUM | Missing shared-runtime imports silently replace telemetry with no-op fallbacks |
| KAI-WATCHX-039 | MEDIUM | Shutdown cancels watcher tasks without awaiting executor/Git/Docker work completion |
| KAI-WATCHX-040 | MEDIUM | No rate limit or caller response budget protects repeated full snapshot reads |
| KAI-WATCHX-041 | MEDIUM | Snapshot caches have no immutable digest, ETag or compare-and-swap publication contract |
| KAI-WATCHX-042 | MEDIUM | The services retain no historical state to distinguish transient and sustained failures |
| KAI-WATCHX-043 | MEDIUM | No durable audit links watched source/configuration, collector result and downstream summary use |
| KAI-WATCHX-044 | MEDIUM | The services have no authoritative collector leadership or lifecycle-managed Docker/Git worker pools |

---

## High-severity findings

### KAI-WATCHX-001 — HIGH — Stopped containers are invisible
**Issue:** Docker SDK uses `containers.list(all=False)` and CLI uses `docker ps` without `-a`.  
**Risk:** A crashed/stopped required service disappears from the cache, unhealthy endpoint and summary instead of being reported failed.  
**Recommendation:** collect all expected containers and compare actual state against a governed service inventory.  
**Status:** OPEN

### KAI-WATCHX-002 — HIGH — Restart detection is non-functional
Docker inspect places `RestartCount` at the top level, but code reads `state.get("RestartCount",0)`, normally yielding zero.

### KAI-WATCHX-003 — HIGH — Status is not part of unhealthy logic
The unhealthy filter checks health, exit code and restart count but not `status` itself.

### KAI-WATCHX-004 — HIGH — No-healthcheck equals acceptable
`health="none"` is explicitly excluded from unhealthy classification and summary counts.

### KAI-WATCHX-005 — HIGH — False all-normal summary
Only visible running containers are counted; absent/stopped services cannot create an issue.

### KAI-WATCHX-006 — HIGH — Missing-container detection absent
No required container names/images/replica counts are configured.

### KAI-WATCHX-007 — HIGH — Container metadata becomes context
Names/images are operator or image-controlled strings and can contain misleading instruction-like text in summaries/UI.

### KAI-WATCHX-008 — HIGH — Read-only socket misconception
Mount mode `:ro` protects the socket filesystem object, not Docker API methods available after connecting; a compromised process can issue mutating daemon requests.

### KAI-WATCHX-009 — HIGH — Unbounded Docker executor admission
Every worker poll uses the default executor and Docker operations may block indefinitely under the existing timeout defect.

### KAI-WATCHX-010 — HIGH — Git read polling may execute code
`git status` can invoke a repository-configured fsmonitor hook. A mounted repository can therefore execute code inside the watcher container during automatic polling.

### KAI-WATCHX-011 — HIGH — Git configuration/environment not isolated
No `GIT_CONFIG_NOSYSTEM`, controlled HOME, disabled hooks/fsmonitor or reduced environment is set.

### KAI-WATCHX-012 — HIGH — Unbounded Git outputs
`capture_output=True` stores complete status, log, stash and error output before any count/field use.

### KAI-WATCHX-013 — HIGH — Per-repository cumulative timeout
The inspector executes numerous commands, each with its own ten-second timeout; a hostile/slow repo can occupy a worker for many tens of seconds.

### KAI-WATCHX-014 — HIGH — Serial repository cycle
Paths are inspected one after another and no global deadline bounds the cycle.

### KAI-WATCHX-015 — HIGH — Errors become clean summaries
Change counts default zero and many failures leave only an `error` field; summary ignores it when computing clean state.

### KAI-WATCHX-016 — HIGH — Dirty view hides uninspectable repos
Repositories with errors but zero default counts are excluded.

### KAI-WATCHX-017 — HIGH — Single-repo false assurance
The one-repo branch can report empty branch plus no uncommitted changes despite `error`.

### KAI-WATCHX-018 — HIGH — Repository text is privileged context
Commit subject/author, branch and path are returned/raw-formatted without control or prompt-injection separation.

### KAI-WATCHX-019 — HIGH — Index endpoint identity drift
`/repos/{index}` identifies by current list position rather than a canonical repository ID.

### KAI-WATCHX-020 — HIGH — No snapshot integrity identity
Neither service returns source daemon/repo identity, generation/digest or collector version.

### KAI-WATCHX-021 — HIGH — Replica divergence/amplification
Each worker launches its own loop and maintains independent caches.

### KAI-WATCHX-022 — HIGH — Missing sensitive-context audit
No actor/purpose record exists for host/repository reads or the exact summaries injected downstream.

---

## Medium-severity findings

### KAI-WATCHX-023 — MEDIUM — Short Docker identity
Only short container IDs are exposed, with no daemon ID, Compose project or full immutable ID.

### KAI-WATCHX-024 — MEDIUM — Weak Docker schema handling
Nested inspect fields and types are trusted through `.get` and direct output.

### KAI-WATCHX-025 — MEDIUM — Human-output fallback
The CLI parser depends on tab-separated `docker ps` formatting and status strings.

### KAI-WATCHX-026 — MEDIUM — Endpoint identity omitted
Responses do not state which Docker daemon/DOCKER_HOST produced them.

### KAI-WATCHX-027 — MEDIUM — Polling delay
No Docker events subscription provides immediate start/stop/health transitions.

### KAI-WATCHX-028 — MEDIUM — Empty-state ambiguity
No data, no containers and failed/not-yet polling produce similar summary states.

### KAI-WATCHX-029 — MEDIUM — Abbreviated Git identity
`--short` hashes are not durable cross-repository/global commit identifiers.

### KAI-WATCHX-030 — MEDIUM — Lossy Git status model
Only aggregate changed/untracked counts remain.

### KAI-WATCHX-031 — MEDIUM — Count comparability depends on Git config
Untracked directory handling and repository settings affect status line counts.

### KAI-WATCHX-032 — MEDIUM — Full stash materialisation
All stash text is returned by Git and split solely to count lines.

### KAI-WATCHX-033 — MEDIUM — Partial fields look valid
Many exceptions are swallowed with default empty/zero values and no field-level unavailable reason.

### KAI-WATCHX-034 — MEDIUM — Missing per-repo freshness
Only one global poll time exists.

### KAI-WATCHX-035 — MEDIUM — Path identity ambiguity
Raw configured path spelling is used as identity and display.

### KAI-WATCHX-036 — MEDIUM — Weak chronology
Poll/uptime use wall-clock values without sequence.

### KAI-WATCHX-037 — MEDIUM — Public telemetry
Metrics endpoints are unauthenticated.

### KAI-WATCHX-038 — MEDIUM — Silent runtime downgrade
Missing runtime import yields no-op metrics/basic logging.

### KAI-WATCHX-039 — MEDIUM — Incomplete shutdown
Cancelled coroutines do not await active executor/subprocess/SDK work.

### KAI-WATCHX-040 — MEDIUM — Unmetered reads
Repeated callers can serialise all cached sensitive state without quotas.

### KAI-WATCHX-041 — MEDIUM — No snapshot revision
Caches lack digest/ETag/generation.

### KAI-WATCHX-042 — MEDIUM — No history
Only the latest snapshot remains, preventing incident chronology/restart-rate reconstruction.

### KAI-WATCHX-043 — MEDIUM — Missing end-to-end audit
No immutable link connects configuration/source snapshot to downstream output.

### KAI-WATCHX-044 — MEDIUM — Missing collector lifecycle authority
No leader election, dedicated bounded executor/client, or graceful reconciliation exists.

---

## Batch totals

- Findings: **44**
- Critical: **0**
- High: **22**
- Medium: **22**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,389**
- Critical: **189**
- High: **1,193**
- Medium: **1,004**
- Low: **3**

## Files materially reviewed

`docker-watcher/app.py`, `git-watcher/app.py`, the existing Host Watchers audit, deployment mounts and Agentic/Dashboard/Supervisor integrations.
