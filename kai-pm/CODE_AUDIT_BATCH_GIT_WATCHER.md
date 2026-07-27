# Kai Code Audit — Git Watcher Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-GITWATCH-001 | HIGH | Repository paths, branches, commits, authors and working-state metadata are exposed without authentication |
| KAI-GITWATCH-002 | HIGH | Repositories with inspection errors can be reported as clean |
| KAI-GITWATCH-003 | HIGH | Git subprocess output is fully buffered without byte limits |
| KAI-GITWATCH-004 | HIGH | Poll duration scales sequentially across every path and multiple subprocesses without a global deadline |
| KAI-GITWATCH-005 | HIGH | Poll-task failure or stalled polling leaves stale repository state while health remains ok |
| KAI-GITWATCH-006 | HIGH | Untrusted branch names and configured paths are inserted into agent-consumed natural-language summaries |
| KAI-GITWATCH-007 | MEDIUM | Repository state and task metadata are process-local and unsynchronised |
| KAI-GITWATCH-008 | MEDIUM | Refresh-task cancellation is not awaited during shutdown |
| KAI-GITWATCH-009 | MEDIUM | The Git executable is resolved through inherited `PATH` rather than a pinned binary |
| KAI-GITWATCH-010 | MEDIUM | No aggregate poll error or completeness state is retained |
| KAI-GITWATCH-011 | MEDIUM | Each repository poll launches up to seven separate Git subprocesses |
| KAI-GITWATCH-012 | MEDIUM | Commit metadata, branch names and paths are not length-bounded |
| KAI-GITWATCH-013 | MEDIUM | Watch paths are weakly parsed and not canonicalised or allowlisted |
| KAI-GITWATCH-014 | MEDIUM | Health reports ok before first poll and after repository failures |
| KAI-GITWATCH-015 | MEDIUM | Error-budget telemetry is exposed but never populated |
| KAI-GITWATCH-016 | MEDIUM | Deployment hard-codes one host repository path and can silently monitor an absent/incorrect directory |

---

## Git watcher: `git-watcher/app.py`, deployment configuration

### KAI-GITWATCH-001 — HIGH — Public repository-state disclosure
**Issue:** `/repos`, `/repos/{index}`, `/dirty`, `/summary` and `/health` require no authentication. Repository records include absolute paths, branch, short commit hash, latest commit message, author, date, dirty/untracked counts, ahead/behind counts, stash count and error details. The service is published on host port 8044.  
**Risk:** Any reachable caller can map development activity, identities, work-in-progress state, repository layout and deployment cadence.  
**Recommendation:** Require owner-scoped operational authentication and expose only minimised aggregate state.  
**Status:** OPEN

### KAI-GITWATCH-002 — HIGH — Failed repositories are represented as clean
**Issue:** `_inspect_repo` returns a dictionary with `error` and zero change counts when a path is absent, not a repository or branch inspection fails. `/summary` and `/dirty` ignore `error`; zero counts cause the repository to be described as clean/no uncommitted changes.  
**Risk:** A missing mount, permission failure or corrupt repository can produce a reassuring clean-state statement precisely when no valid inspection occurred.  
**Recommendation:** Treat any inspection error as unknown/degraded and block clean-state conclusions until all required checks succeed.  
**Status:** OPEN

### KAI-GITWATCH-003 — HIGH — Subprocess output allocation is unbounded
**Issue:** `_run_git` uses `capture_output=True` and materialises complete stdout/stderr for `status --short`, `stash list`, log and other commands. Counting/truncation occurs only after full output exists.  
**Risk:** Repositories with very large working trees, untracked sets, stash histories or malicious metadata can consume excessive memory before the snapshot is built.  
**Recommendation:** Stream bounded output, use count-oriented Git plumbing where possible and terminate commands at a strict byte cap.  
**Status:** OPEN

### KAI-GITWATCH-004 — HIGH — Poll work is serial and lacks an overall deadline
**Issue:** Every configured path is inspected one at a time. Each inspection invokes several Git commands, each with a 10-second timeout, but no total per-repository or full-cycle deadline exists. The number of watch paths is unrestricted.  
**Risk:** Slow repositories or many configured paths can make a poll take minutes, leaving state stale and occupying executor capacity.  
**Recommendation:** Enforce a small path allowlist, bounded concurrency and strict per-repo/global deadlines.  
**Status:** OPEN

### KAI-GITWATCH-005 — HIGH — Stale state and task failure are invisible
**Issue:** `_poll_loop` has no outer exception handling or supervision. An exception escaping executor invocation or assignment terminates the task. Health does not inspect `_refresh_task` and always returns ok; prior `_repos` and `_last_poll` remain.  
**Risk:** Repository monitoring can stop permanently while stale state continues to be served as current.  
**Recommendation:** Supervise/restart the task, retain a poll error and enforce freshness in health/data endpoints.  
**Status:** OPEN

### KAI-GITWATCH-006 — HIGH — Repository-controlled text becomes agent context
**Issue:** `/summary` inserts branch names and configured filesystem paths directly into natural-language text. Agentic/Cortex consume Git summaries as situational context. Git branch names and deployment paths are not trusted or delimited.  
**Risk:** Crafted branch/path strings can inject misleading statements into agent context, and incorrect clean-state conclusions gain privileged situational authority.  
**Recommendation:** Return typed signed repository facts and never concatenate raw identifiers into privileged prompts.  
**Status:** OPEN

### KAI-GITWATCH-007 — MEDIUM — State is worker-local
**Issue:** Repository snapshots, last-poll timestamp and task reference are module-level process memory.  
**Risk:** Multiple workers run duplicate polls and expose inconsistent states; restart erases freshness history.  
**Recommendation:** Use one watcher authority and shared immutable versioned snapshots.  
**Status:** OPEN

### KAI-GITWATCH-008 — MEDIUM — Shutdown does not await polling termination
**Issue:** Lifespan shutdown cancels `_refresh_task` without awaiting it; executor-backed Git subprocesses are not terminated by coroutine cancellation.  
**Risk:** Git commands can continue after shutdown begins and task failures/resources are not observed.  
**Recommendation:** Await cancellation and manage subprocess/executor shutdown explicitly.  
**Status:** OPEN

### KAI-GITWATCH-009 — MEDIUM — Git binary identity is not pinned
**Issue:** Subprocesses invoke the bare command `git`, resolved through the inherited `PATH`.  
**Risk:** A compromised/writable PATH directory can replace the expected binary with arbitrary code under the watcher’s filesystem access.  
**Recommendation:** Use a pinned absolute executable verified at startup.  
**Status:** OPEN

### KAI-GITWATCH-010 — MEDIUM — Poll completeness is not represented
**Issue:** Individual command failures are frequently suppressed, and no aggregate snapshot completeness/error list is returned. Only some early failures populate the repository `error` field.  
**Risk:** Partial metadata—such as zero ahead/behind, zero stash or no commit details—is indistinguishable from a successful zero result.  
**Recommendation:** Return per-check success/error state and a required-check completeness flag.  
**Status:** OPEN

### KAI-GITWATCH-011 — MEDIUM — Excess subprocess fan-out
**Issue:** A normal repository inspection starts separate processes for git-dir validation, branch, hash, message, author, date, status, upstream counts and stash list.  
**Risk:** Periodic monitoring creates avoidable process and filesystem churn proportional to repository count.  
**Recommendation:** Use fewer structured Git plumbing calls or a library with bounded reads.  
**Status:** OPEN

### KAI-GITWATCH-012 — MEDIUM — Metadata lengths are unbounded
**Issue:** Paths, branch names, commit messages, authors, dates and stderr are retained without per-field length limits and returned through the API.  
**Risk:** Malicious or oversized repository metadata can inflate memory, responses and agent context.  
**Recommendation:** Enforce strict field and aggregate snapshot limits.  
**Status:** OPEN

### KAI-GITWATCH-013 — MEDIUM — Watch paths are weakly governed
**Issue:** `GIT_WATCH_PATHS` is split on `:` and stripped. Paths are not canonicalised, required to be beneath approved roots, checked for symlinks or deduplicated.  
**Risk:** Configuration can inspect unintended repositories/directories and duplicate expensive work. Colon parsing is also incompatible with some path forms.  
**Recommendation:** Use a typed list of canonical allowlisted roots and deduplicate before polling.  
**Status:** OPEN

### KAI-GITWATCH-014 — MEDIUM — Health is readiness-blind
**Issue:** `/health` always reports `status: ok`, including before first poll, with zero valid repositories or when every repository has an error.  
**Risk:** Monitoring treats a non-observing watcher as ready.  
**Recommendation:** Separate liveness, task state, required repository validity and snapshot freshness.  
**Status:** OPEN

### KAI-GITWATCH-015 — MEDIUM — Error-budget telemetry is inert
**Issue:** `budget` is exposed through `/metrics`, but no request, subprocess or poll outcome is recorded.  
**Risk:** Reliability metrics provide no evidence of repository-monitoring success.  
**Recommendation:** Record classified checks, poll outcomes and latency.  
**Status:** OPEN

### KAI-GITWATCH-016 — MEDIUM — Deployment path is brittle and misleading
**Issue:** The minimal Compose file hard-codes `/home/user/kai-system:/workspace:ro`, while watcher health does not verify that the intended host repository is actually mounted and valid.  
**Risk:** On a different host the path can be absent/incorrect, yet the service starts and reports healthy while summarising an invalid repository as clean.  
**Recommendation:** Require an explicit existing canonical deployment path and fail readiness when the expected repository identity/hash is not present.  
**Status:** OPEN

---

## Batch totals

- Findings: **16**
- Critical: **0**
- High: **6**
- Medium: **10**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **867**
- Critical: **87**
- High: **308**
- Medium: **469**
- Low: **3**

## Files materially reviewed in this batch

`git-watcher/app.py` and the relevant `git-watcher` deployment definition in `docker-compose.minimal.yml`.
