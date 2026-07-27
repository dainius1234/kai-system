# Kai Code Audit — Executor Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-EXEC-001 | CRITICAL | The host-published execution engine has no authentication or authorisation |
| KAI-EXEC-002 | CRITICAL | Executor never verifies a Tool Gate decision, ledger hash, policy version or signed request digest |
| KAI-EXEC-003 | CRITICAL | Allowlisted `python3` permits arbitrary code through `python3 -c` and module execution |
| KAI-EXEC-004 | CRITICAL | Allowlisted `find` permits arbitrary command execution through `-exec`/`-execdir` |
| KAI-EXEC-005 | CRITICAL | Allowlisted `make` permits arbitrary shell execution through makefiles, `--eval` and command variables |
| KAI-EXEC-006 | CRITICAL | Allowlisted `pip` executes arbitrary package build/install code and can persist attacker-controlled modules |
| KAI-EXEC-007 | CRITICAL | Allowlisted Git supports shell-command aliases, hooks, SSH commands and destructive repository operations |
| KAI-EXEC-008 | CRITICAL | Allowlisted curl permits unrestricted internal-network access, local-file reads and file writes |
| KAI-EXEC-009 | CRITICAL | Python-expression validation is bypassed through `__builtins__[...]` subscript calls |
| KAI-EXEC-010 | CRITICAL | Any executable named like an allowlisted command can be invoked by absolute path |
| KAI-EXEC-011 | CRITICAL | Timeout kills only the direct process and does not terminate spawned descendants/process groups |
| KAI-EXEC-012 | CRITICAL | “Rollback” only pops a metadata entry and cannot reverse filesystem, network, package, Docker or process effects |
| KAI-EXEC-013 | CRITICAL | Failure rollback can pop another concurrent task’s state rather than the failed task |
| KAI-EXEC-014 | CRITICAL | Malware detection pops the prior unrelated state before the current execution state is pushed |
| KAI-EXEC-015 | HIGH | The command scanner inspects a sanitised/truncated string but executes the original unsanitised parameters |
| KAI-EXEC-016 | HIGH | ClamAV is not installed by the Executor image and malware scanning deterministically degrades to `engine: none` |
| KAI-EXEC-017 | HIGH | ClamAV scan errors and return codes above one are treated as safe |
| KAI-EXEC-018 | HIGH | Malware scan covers only stringified request parameters, not scripts, downloaded files or runtime behaviour |
| KAI-EXEC-019 | HIGH | Subprocess output is fully buffered before the configured output limit is applied |
| KAI-EXEC-020 | HIGH | Successful stderr is returned without any size limit |
| KAI-EXEC-021 | HIGH | Request parameters, command strings, script arguments, expressions and task IDs are unbounded |
| KAI-EXEC-022 | HIGH | CPU-heavy subprocess and malware-scan operations run synchronously on the async event loop |
| KAI-EXEC-023 | HIGH | No execution concurrency cap, queue, per-caller quota or rate limit exists |
| KAI-EXEC-024 | HIGH | Duplicate task IDs execute repeatedly because no idempotency or operation-state authority exists |
| KAI-EXEC-025 | HIGH | Shell commands retain unrestricted outbound network access |
| KAI-EXEC-026 | HIGH | Shell commands can read every container-readable absolute filesystem path |
| KAI-EXEC-027 | HIGH | Shell commands can write arbitrary accessible paths through command-specific options |
| KAI-EXEC-028 | HIGH | Docker CLI can target an attacker-selected remote Docker API endpoint |
| KAI-EXEC-029 | HIGH | Git can target arbitrary repositories and paths through `-C`, URLs and configuration options |
| KAI-EXEC-030 | HIGH | Python expressions can allocate excessive CPU/memory through comprehensions, ranges and large arithmetic |
| KAI-EXEC-031 | HIGH | Script arguments are unrestricted and may activate destructive behaviour inside an approved script |
| KAI-EXEC-032 | HIGH | Approved scripts are trusted by filename pattern rather than immutable digest, owner and revision |
| KAI-EXEC-033 | HIGH | The deployed Executor image does not copy or mount the configured scripts directory |
| KAI-EXEC-034 | HIGH | Script subprocesses inherit the full service environment |
| KAI-EXEC-035 | HIGH | Security-blocking HTTPExceptions leave the already-pushed execution state in history |
| KAI-EXEC-036 | HIGH | State history stores complete raw parameters including commands, expressions, paths and possible secrets |
| KAI-EXEC-037 | HIGH | Execution history is exposed without authentication |
| KAI-EXEC-038 | HIGH | Available tools, command allowlist, script patterns and blocked builtins are exposed without authentication |
| KAI-EXEC-039 | HIGH | Execution output and stderr may disclose service configuration, files, network responses and secrets |
| KAI-EXEC-040 | HIGH | Raw internal exception strings are returned in HTTP 500 responses |
| KAI-EXEC-041 | HIGH | Heartbeat failure notifications can include sensitive exception and subprocess details |
| KAI-EXEC-042 | HIGH | Heartbeat delivery failure does not affect execution/error semantics |
| KAI-EXEC-043 | HIGH | The state store is unbounded, process-local and concurrency-unsafe |
| KAI-EXEC-044 | HIGH | Public recovery deletes matching temporary files without caller identity or operation ownership |
| KAI-EXEC-045 | HIGH | Recovery claims to reset state but never clears or reconciles the StateStore |
| KAI-EXEC-046 | HIGH | Health does not verify Tool Gate connectivity, approval enforcement, command binaries, scripts or malware scanner |
| KAI-EXEC-047 | HIGH | `alive` always returns healthy regardless of execution readiness |
| KAI-EXEC-048 | HIGH | The result’s `policy_context` is self-constructed metadata rather than evidence of policy approval |
| KAI-EXEC-049 | HIGH | Caller-controlled device is accepted as policy context although execution is unchanged |
| KAI-EXEC-050 | HIGH | The container filesystem is writable and no explicit read-only root, tmpfs sandbox or capability-drop policy is configured |
| KAI-EXEC-051 | HIGH | no-new-privileges and a non-root user do not isolate network, process, CPU, memory or writable-filesystem effects |
| KAI-EXEC-052 | HIGH | Command-specific destructive options bypass the small regex denylist |
| KAI-EXEC-053 | HIGH | Dangerous-pattern checks are case-, spacing- and syntax-sensitive and operate on the raw command string only |
| KAI-EXEC-054 | MEDIUM | Command allowlist parsing uses whitespace split while execution uses `shlex.split`, creating inconsistent interpretation |
| KAI-EXEC-055 | MEDIUM | Dangerous-pattern matches expose the internal regex to callers |
| KAI-EXEC-056 | MEDIUM | Command rejections disclose the complete allowlist |
| KAI-EXEC-057 | MEDIUM | Script rejections disclose approved filenames and path expectations |
| KAI-EXEC-058 | MEDIUM | Execution timeout and output-size environment values are not range validated |
| KAI-EXEC-059 | MEDIUM | Negative output limits produce surprising slicing and truncation behaviour |
| KAI-EXEC-060 | MEDIUM | Text-mode subprocess decoding can fail on non-UTF-8 output |
| KAI-EXEC-061 | MEDIUM | Output truncation counts characters rather than encoded bytes |
| KAI-EXEC-062 | MEDIUM | Duration uses wall-clock time and is vulnerable to clock adjustments |
| KAI-EXEC-063 | MEDIUM | The post-completion timeout check can reject an already-completed command and pop unrelated state |
| KAI-EXEC-064 | MEDIUM | State entries have no completion, failure, output, actor or Gate-decision linkage |
| KAI-EXEC-065 | MEDIUM | History limits accept negative and arbitrarily large values |
| KAI-EXEC-066 | MEDIUM | Recovery silently ignores every file-deletion error and still returns success |
| KAI-EXEC-067 | MEDIUM | Recovery glob can remove files created by other work sharing the prefix |
| KAI-EXEC-068 | MEDIUM | Health treats disk-stat failure as `unknown` without degrading the service |
| KAI-EXEC-069 | MEDIUM | Health does not check actual free memory/CPU/process limits or subprocess creation |
| KAI-EXEC-070 | MEDIUM | Audit logging is optional and records no authenticated actor, request digest or Gate decision |
| KAI-EXEC-071 | MEDIUM | Successful execution has no durable append-only execution ledger |
| KAI-EXEC-072 | MEDIUM | Failed/security-blocked executions are not consistently recorded in the audit stream |
| KAI-EXEC-073 | MEDIUM | Executor creates no owned lifespan resources, graceful task drain or process cleanup |
| KAI-EXEC-074 | MEDIUM | Tool definitions and implementation capabilities drift from Tool Gate’s independent hard-coded allowlist |

---

## Critical execution-boundary findings

### KAI-EXEC-001 — CRITICAL — Open execution engine
**Issue:** `docker-compose.full.yml` publishes `8002:8002`. `/execute` has no authentication, user/service identity, token, signature or caller scope.  
**Risk:** Any reachable caller can invoke shell, Python, scripts and network-capable command-line tools.  
**Recommendation:** remove host publication and accept only one-time body-bound execution grants from Tool Gate through an authenticated private channel.  
**Status:** OPEN — immediate remediation required

### KAI-EXEC-002 — CRITICAL — Gate is not enforced at the action boundary
**Issue:** `TOOL_GATE_URL` is configured in Compose but Executor never reads or calls it. `ExecutionRequest` contains only tool, params, task ID and device.  
**Risk:** Tool Gate can deny an action while a caller sends the same action directly to Executor.  
**Recommendation:** require a single-use Gate capability containing exact body digest, actor, tool, params, expiry, policy version and ledger proof.  
**Status:** OPEN — immediate remediation required

### KAI-EXEC-003 — CRITICAL — `python3` shell escape
An execution such as `python3 -c <code>` passes the base-command allowlist and runs arbitrary Python with the shell environment, filesystem and network access.

### KAI-EXEC-004 — CRITICAL — `find -exec` shell escape
GNU/POSIX find can execute arbitrary binaries through `-exec`, `-execdir` and perform destructive operations such as `-delete`; the denylist does not model command options.

### KAI-EXEC-005 — CRITICAL — Makefile shell escape
`make` executes recipe shells from an attacker-selected file/path and supports configuration/evaluation mechanisms that create commands.

### KAI-EXEC-006 — CRITICAL — Package-install code execution
`pip` downloads/builds/installs attacker-selected packages; build backends/setup hooks execute code and installations can persist modules in writable locations.

### KAI-EXEC-007 — CRITICAL — Git command execution
Git supports `alias.<name>=!<shell command>`, `core.sshCommand`, hooks/helpers and attacker-selected repository paths/URLs, allowing arbitrary command execution and destructive changes.

### KAI-EXEC-008 — CRITICAL — Curl SSRF/file primitive
Curl accepts arbitrary schemes/hosts/options, including internal service URLs, metadata-like targets, `file://` reads and output-file writes.

### KAI-EXEC-009 — CRITICAL — Python AST sandbox bypass
The validator blocks direct calls where `node.func` is an `ast.Name`, but permits a call whose function is a subscript. `__builtins__['__import__']('os').system(...)` therefore bypasses every blocked-name check.  
**Status:** OPEN — immediate remediation required

### KAI-EXEC-010 — CRITICAL — Basename allowlist bypass
`Path(parts[0]).name` accepts `/tmp/python3`, `/tmp/git` or any attacker-created executable whose basename matches an allowlisted command.

### KAI-EXEC-011 — CRITICAL — Descendant-process escape
`subprocess.run(... timeout=...)` terminates only the direct child. Allowed Python/Git/Make/scripts can spawn detached or background descendants that survive request timeout/completion.

### KAI-EXEC-012 — CRITICAL — Rollback is fictitious
`StateStore.revert_last_state()` removes one dictionary from memory. It does not undo files, packages, repositories, network requests, child processes, remote Docker actions or command output.

### KAI-EXEC-013 — CRITICAL — Cross-task rollback corruption
The store is global and unlocked. Any failure pops the most recently pushed state, which may belong to a different concurrently executing request.

### KAI-EXEC-014 — CRITICAL — Malware block removes previous state
Malware scanning occurs before the current `store.push()`. On detection, `revert_last_state()` deletes the prior execution record.

---

## High-severity sandbox, integrity and disclosure findings

### KAI-EXEC-015 — HIGH — Scan/execution payload mismatch
The scan input is `sanitize_string(str(request.params))`, which strips/truncates data, while handlers execute the original dictionary.

### KAI-EXEC-016 — HIGH — Claimed ClamAV is absent
The Dockerfile installs no ClamAV package. The normal deployed result is `engine: none, code: 0`, indistinguishable from a clean scan to the execution path.

### KAI-EXEC-017 — HIGH — Scanner errors fail open
Only return code exactly one blocks. ClamAV operational errors use other nonzero codes and are executed anyway.

### KAI-EXEC-018 — HIGH — Inadequate malware subject
The scanner sees only request text, not the approved script bytes, Git/package content, files downloaded by curl or processes launched at runtime.

### KAI-EXEC-019 — HIGH — Post-buffer output cap
`capture_output=True` accumulates complete stdout/stderr in memory before stdout is sliced.

### KAI-EXEC-020 — HIGH — Unbounded successful stderr
`ExecutionResult.stderr` returns complete stderr even when stdout is truncated.

### KAI-EXEC-021 — HIGH — Unbounded request inputs
Pydantic fields have no command/expression/task/argument/body depth or item limits.

### KAI-EXEC-022 — HIGH — Event-loop blocking
ClamAV and all subprocess executions are synchronous calls inside the async route.

### KAI-EXEC-023 — HIGH — No workload admission
Unlimited callers can launch overlapping 30-second processes and 10-second scans/Python evaluations.

### KAI-EXEC-024 — HIGH — No idempotency
`task_id` is caller metadata only; replaying it executes the action again.

### KAI-EXEC-025 — HIGH — No network isolation
Shell/script/Python child processes use the container network and can reach internal/external services.

### KAI-EXEC-026 — HIGH — No filesystem read sandbox
Absolute paths are permitted for command arguments and Python code.

### KAI-EXEC-027 — HIGH — No filesystem write sandbox
Curl, Python, Pip, Git, Make and scripts can write any path accessible to the service user.

### KAI-EXEC-028 — HIGH — Remote Docker control
Even without a mounted local socket, `docker -H <endpoint>` can target any reachable unauthenticated/credentialed remote Docker API.

### KAI-EXEC-029 — HIGH — Arbitrary Git target paths
`git -C` and repository arguments permit operations against any accessible working tree/path.

### KAI-EXEC-030 — HIGH — Python resource exhaustion
Allowed expressions can construct huge lists/strings/integers or expensive calculations. A timeout does not provide a per-process memory limit.

### KAI-EXEC-031 — HIGH — Script argument authority
Filename approval says nothing about safe combinations of unrestricted arguments.

### KAI-EXEC-032 — HIGH — Filename-only script trust
No checksum, immutable image manifest, file owner/mode or symlink check binds the executed bytes to an approved revision.

### KAI-EXEC-033 — HIGH — Script tool deployment failure
The Executor Dockerfile copies only Executor and common code; no `/workspaces/kai-system/scripts` directory is copied or mounted in the shown deployment.

### KAI-EXEC-034 — HIGH — Script environment leakage
Unlike shell/Python paths, script execution passes the complete inherited service environment.

### KAI-EXEC-035 — HIGH — Rejected state retained
State is pushed before the handler. Handler-raised HTTP security/validation exceptions are re-raised without a pop or failure status.

### KAI-EXEC-036 — HIGH — Secret-bearing state history
Raw params are stored, including credentials in URLs, command arguments, script arguments and Python expressions.

### KAI-EXEC-037 — HIGH — Public history
`GET /history` has no authentication.

### KAI-EXEC-038 — HIGH — Public attack-surface inventory
`GET /tools` reveals command names, script patterns and blocked builtins.

### KAI-EXEC-039 — HIGH — Result exfiltration
The direct caller receives complete command output/stderr up to large limits, enabling deliberate file/environment/internal-service extraction.

### KAI-EXEC-040 — HIGH — Exception disclosure
Unexpected exception text is placed directly in HTTP detail.

### KAI-EXEC-041 — HIGH — Sensitive heartbeat text
Exception/subprocess failure reason is forwarded to Heartbeat and may enter its logs/alerts.

### KAI-EXEC-042 — HIGH — Unverified heartbeat delivery
Shared resilience can classify 4xx as success; notification failure never changes execution failure semantics or durable evidence.

### KAI-EXEC-043 — HIGH — Volatile unbounded state
The list has no cap/persistence/lock and differs across workers/restarts.

### KAI-EXEC-044 — HIGH — Public cleanup mutation
`POST /recover` is unauthenticated and deletes matching files.

### KAI-EXEC-045 — HIGH — False state reset
The route documentation says “reset state” but never clears/reconciles `store`.

### KAI-EXEC-046 — HIGH — Readiness-blind health
Health checks only `/tmp` writeability and approximate free disk.

### KAI-EXEC-047 — HIGH — Unconditional alive
`/alive` always returns ok.

### KAI-EXEC-048 — HIGH — Fabricated policy context
The result repeats caller device/tool and static limits; no approval ID/hash is present.

### KAI-EXEC-049 — HIGH — Caller-labelled device
CPU/CUDA changes no handler but appears as policy evidence.

### KAI-EXEC-050 — HIGH — Writable container root
Compose/Dockerfile do not make the root filesystem read-only or provide dedicated isolated work volumes.

### KAI-EXEC-051 — HIGH — Insufficient container hardening
Running non-root with `no-new-privileges` does not restrict network destinations, subprocess counts, child persistence or writable user-accessible files.

### KAI-EXEC-052 — HIGH — Command-option bypass
Operations such as `find -delete/-exec`, `git clean`, Pip installation and Curl writes are dangerous without matching the small patterns.

### KAI-EXEC-053 — HIGH — Weak denylist syntax
Regexes are case-sensitive and match narrow whitespace/text arrangements; command-specific equivalent forms evade them.

---

## Medium-severity operational findings

### KAI-EXEC-054 — MEDIUM — Parser mismatch
Allowlist base extraction uses `.split()`, while execution uses shell-aware `shlex.split()`.

### KAI-EXEC-055 — MEDIUM — Regex disclosure
Dangerous-pattern responses reveal the exact matched defensive expression.

### KAI-EXEC-056 — MEDIUM — Allowlist disclosure
Rejection includes every allowed command.

### KAI-EXEC-057 — MEDIUM — Script-policy disclosure
Responses reveal approved patterns and expected script names.

### KAI-EXEC-058 — MEDIUM — Unsafe configuration
Negative/zero/extreme timeout and output-size values are not validated at startup.

### KAI-EXEC-059 — MEDIUM — Negative truncation semantics
A negative `MAX_OUTPUT_SIZE` slices from the end and sets nonsensical policy context.

### KAI-EXEC-060 — MEDIUM — Output decoding failure
Text-mode decoding uses locale assumptions and can throw on arbitrary binary output.

### KAI-EXEC-061 — MEDIUM — Character-based output limit
Unicode output bytes can greatly exceed the configured character count.

### KAI-EXEC-062 — MEDIUM — Non-monotonic duration
Duration uses `time.time()` rather than `time.monotonic()`.

### KAI-EXEC-063 — MEDIUM — Late timeout misclassification
A process that returns after the threshold is already complete, but the route then reports timeout and pops global state.

### KAI-EXEC-064 — MEDIUM — Incomplete state model
History entries never record completion/error/exit/duration/Gate identity.

### KAI-EXEC-065 — MEDIUM — Unsafe history limit
Negative and huge limits expose surprising or complete state.

### KAI-EXEC-066 — MEDIUM — False recovery success
Every deletion failure is ignored.

### KAI-EXEC-067 — MEDIUM — Prefix-based cleanup collision
The glob is not linked to Executor-created files/task IDs and may delete unrelated files with that prefix.

### KAI-EXEC-068 — MEDIUM — Disk-stat failure stays green
`unknown` does not cause degraded health.

### KAI-EXEC-069 — MEDIUM — Missing resource readiness
Health does not test process creation, actual commands, scripts, network policy, memory headroom or child cleanup.

### KAI-EXEC-070 — MEDIUM — Weak optional audit
Audit defaults optional and records only method/path/status plus a success message without actor/body/Gate proof.

### KAI-EXEC-071 — MEDIUM — No durable execution ledger
Successful actions exist only in the HTTP response, optional audit and volatile StateStore.

### KAI-EXEC-072 — MEDIUM — Inconsistent failure evidence
Many HTTPExceptions/security rejections are captured only by generic middleware status, not structured tool/task/actor evidence.

### KAI-EXEC-073 — MEDIUM — Missing lifecycle ownership
No lifespan manager owns process groups, shared clients, shutdown cancellation or temporary artefact reconciliation.

### KAI-EXEC-074 — MEDIUM — Tool-policy drift
Executor supports shell/script/python/noop, while Tool Gate’s independent allowlist includes qgis/n8n/speak but omits script/python.

---

## Batch totals

- Findings: **74**
- Critical: **14**
- High: **39**
- Medium: **21**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,866**
- Critical: **170**
- High: **903**
- Medium: **790**
- Low: **3**

## Files materially reviewed

`executor/app.py`, `executor/Dockerfile`, Executor deployment in `docker-compose.full.yml`, and integration against Tool Gate, Heartbeat, common resilience/runtime primitives.
