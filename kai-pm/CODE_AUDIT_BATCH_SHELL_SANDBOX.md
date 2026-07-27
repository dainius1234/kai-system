# Kai Code Audit — Shell Sandbox Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-SHELL-001 | CRITICAL | `/proc/self` allowlisting permits direct disclosure of the sandbox process environment |
| KAI-SHELL-002 | CRITICAL | `abspath` path checks do not resolve symlinks, permitting reads outside safe directories |
| KAI-SHELL-003 | CRITICAL | Option-embedded file-list arguments bypass path validation and can read arbitrary files |
| KAI-SHELL-004 | HIGH | Unauthenticated callers can execute allowlisted host subprocesses |
| KAI-SHELL-005 | HIGH | Unrestricted `ps` options can expose process command lines and environment variables |
| KAI-SHELL-006 | HIGH | Command binaries are resolved through the inherited `PATH` rather than pinned absolute executables |
| KAI-SHELL-007 | HIGH | Output caps are applied only after complete stdout/stderr buffering |
| KAI-SHELL-008 | HIGH | Blocking subprocess execution runs directly inside the async request handler |
| KAI-SHELL-009 | HIGH | No rate limit or concurrency bound protects subprocess capacity |
| KAI-SHELL-010 | HIGH | Configurable safe-directory prefixes are not canonicalised or constrained |
| KAI-SHELL-011 | MEDIUM | Command-specific option semantics are not validated |
| KAI-SHELL-012 | MEDIUM | Allowlist and filesystem prefixes are exposed without authentication |
| KAI-SHELL-013 | MEDIUM | Full submitted commands are returned to callers |
| KAI-SHELL-014 | MEDIUM | Execution and parsing errors disclose internal diagnostics |
| KAI-SHELL-015 | MEDIUM | Text decoding failures are not handled as typed execution errors |
| KAI-SHELL-016 | MEDIUM | Health reports ok without validating binaries, paths or subprocess execution |
| KAI-SHELL-017 | MEDIUM | Timeout, output and port configuration are not safely validated |
| KAI-SHELL-018 | MEDIUM | Commands inherit the service working directory and process environment |

---

## Shell sandbox: `sandboxes/shell/app.py`

### KAI-SHELL-001 — CRITICAL — Direct environment-secret disclosure
**Issue:** `SAFE_DIRS` includes `/proc/self`. Path commands therefore accept `cat /proc/self/environ`, `head /proc/self/environ`, `tail /proc/self/environ` and `wc /proc/self/environ`. The module documentation explicitly claims `/proc/*/environ` is prevented, but the configured prefix permits it directly.  
**Risk:** Any reachable caller can retrieve environment variables held by the sandbox process, including service URLs, tokens, credentials and deployment secrets supplied through environment configuration.  
**Recommendation:** Remove `/proc/self` from readable paths and run with a minimal secret-free environment; explicitly deny all procfs/sysfs paths.  
**Status:** OPEN — immediate remediation required

### KAI-SHELL-002 — CRITICAL — Symlink escape from allowed directories
**Issue:** `_validate_path_args` uses `os.path.abspath`, despite its comment claiming it resolves without following symlinks. `abspath` only normalises the string; it does not resolve symlinks. A path such as `/tmp/link` passes the prefix check, while `cat` follows the link to an arbitrary target such as `/etc/passwd` or a secret mount.  
**Risk:** Any process able to create a symlink in an allowed directory can convert the public sandbox into arbitrary readable-file exfiltration outside the sandbox policy.  
**Recommendation:** Open files securely using descriptor-based traversal (`openat2`/`O_NOFOLLOW`) under dedicated immutable roots and verify the final inode path; do not delegate path resolution to general commands.  
**Status:** OPEN — immediate remediation required

### KAI-SHELL-003 — CRITICAL — File-list options bypass path inspection
**Issue:** Every argument beginning with `-` is skipped as a flag. Commands such as GNU `wc --files0-from=/tmp/list` or `du --files0-from=/tmp/list` can read a permitted list file whose contents name arbitrary files outside `SAFE_DIRS`. The actual secondary file paths are never validated.  
**Risk:** Callers can indirectly access arbitrary filesystem targets despite all direct positional paths passing the prefix check.  
**Recommendation:** Define exact per-command argument grammars and reject all options that introduce paths, response files, recursion or indirect input.  
**Status:** OPEN — immediate remediation required

### KAI-SHELL-004 — HIGH — Public subprocess execution surface
**Issue:** `POST /run` requires no authentication or authorisation and launches a host/container subprocess for every accepted command.  
**Risk:** Callers can perform system reconnaissance, consume process capacity and repeatedly inspect host/container state. The allowlist reduces scope but does not constitute an identity or policy boundary.  
**Recommendation:** Require authenticated capability-scoped requests and isolate execution in disposable resource-limited sandboxes.  
**Status:** OPEN

### KAI-SHELL-005 — HIGH — `ps` can disclose process secrets
**Issue:** `ps` is allowlisted without argument restrictions. Options such as environment-inclusive/wide process listings can expose command lines and environment data for accessible processes. `_validate_path_args` applies no checks to `ps`.  
**Risk:** API keys, passwords and confidential arguments passed to other processes may be returned to unauthenticated callers.  
**Recommendation:** Remove `ps` or expose a fixed redacted process-summary operation rather than arbitrary command options.  
**Status:** OPEN

### KAI-SHELL-006 — HIGH — Executables are selected through inherited PATH
**Issue:** `subprocess.run(parts, shell=False)` receives only the bare command name. The operating system searches the service’s inherited `PATH`; the allowlist validates the string but not the resolved executable inode/hash.  
**Risk:** A writable or compromised PATH directory can replace an allowed name with arbitrary code, turning an information command into code execution.  
**Recommendation:** Map each allowed operation to a pinned absolute executable verified at startup, or implement the operation directly in Python.  
**Status:** OPEN

### KAI-SHELL-007 — HIGH — Output limit does not limit allocation
**Issue:** `capture_output=True` buffers complete stdout and stderr in memory. Slicing to `MAX_OUTPUT_BYTES` occurs only after the process exits.  
**Risk:** Reading a very large permitted file, recursively listing a large tree or producing extensive command output can exhaust memory despite the nominal cap.  
**Recommendation:** Stream through bounded pipes, terminate the process when combined output reaches the limit and report truncation explicitly.  
**Status:** OPEN

### KAI-SHELL-008 — HIGH — Subprocess blocks the event loop
**Issue:** `subprocess.run` executes synchronously inside an async FastAPI endpoint for up to `EXECUTION_TIMEOUT`.  
**Risk:** One slow command blocks the event-loop worker; repeated callers can deny health and execution service.  
**Recommendation:** Use an isolated bounded worker/process pool with asynchronous cancellation and hard resource limits.  
**Status:** OPEN

### KAI-SHELL-009 — HIGH — Subprocess concurrency is unrestricted
**Issue:** There is no authentication, rate limit, per-caller quota, semaphore or global process limit. Each request can start another subprocess.  
**Risk:** Concurrent callers can exhaust process IDs, CPU, memory and file-descriptor capacity even using individually read-only commands.  
**Recommendation:** Enforce strict global and per-principal concurrency/throughput limits.  
**Status:** OPEN

### KAI-SHELL-010 — HIGH — Safe roots are configuration-controlled prefixes
**Issue:** `SANDBOX_SAFE_DIRS` accepts arbitrary comma-separated strings. Values are not required to be absolute, canonical, existing, owned, read-only or free of symlinks. Prefix checks use those raw values.  
**Risk:** Misconfiguration can silently allow broad filesystem trees or create inconsistent/bypassable comparisons.  
**Recommendation:** Pin a minimal immutable root set and verify canonical ownership, permissions and mount properties at startup.  
**Status:** OPEN

### KAI-SHELL-011 — MEDIUM — Arbitrary flags remain available
**Issue:** Apart from path-looking arguments, all command flags are accepted. Recursive traversal, alternate formatting, environment display, response-file behaviour and expensive modes are not controlled per binary.  
**Risk:** Callers can expand data exposure and resource use beyond the intended simple read-only operations.  
**Recommendation:** Replace general command parsing with fixed typed operations and approved option enums.  
**Status:** OPEN

### KAI-SHELL-012 — MEDIUM — Security policy is publicly enumerated
**Issue:** `GET /allowlist` returns every command, path-restricted command and safe-directory prefix without authentication. Rejection messages also repeat the policy.  
**Risk:** Callers can tailor bypass attempts precisely to enabled binaries and readable roots.  
**Recommendation:** Restrict detailed policy disclosure and return generic denial responses publicly.  
**Status:** OPEN

### KAI-SHELL-013 — MEDIUM — Command strings are reflected
**Issue:** The complete submitted command is returned in `ShellResult.command`.  
**Risk:** Sensitive path names or arguments are unnecessarily duplicated into responses, proxies and logs.  
**Recommendation:** Return an operation ID and normalised operation metadata rather than raw command text.  
**Status:** OPEN

### KAI-SHELL-014 — MEDIUM — Internal diagnostics are exposed
**Issue:** `shlex` errors, OSError strings, timeout values, missing binary names and policy details are returned directly in HTTP responses.  
**Risk:** Callers learn runtime, filesystem and executable-resolution information useful for reconnaissance.  
**Recommendation:** Return stable error codes and protected trace identifiers.  
**Status:** OPEN

### KAI-SHELL-015 — MEDIUM — Output decoding can escape error handling
**Issue:** `text=True` uses default decoding/error behaviour. A permitted file containing invalid bytes can cause decode exceptions not covered by the listed exception handlers.  
**Risk:** Requests can generate unstructured server errors and bypass the intended ShellResult/error contract.  
**Recommendation:** Capture bytes and decode with an explicit bounded safe policy.  
**Status:** OPEN

### KAI-SHELL-016 — MEDIUM — Health is readiness-blind
**Issue:** `/health` always returns ok and does not verify required binaries, safe-root integrity, subprocess capability or resource capacity.  
**Risk:** Orchestration treats the sandbox as ready while every command may fail or policy roots may be unsafe.  
**Recommendation:** Separate liveness from verified execution-policy readiness.  
**Status:** OPEN

### KAI-SHELL-017 — MEDIUM — Numeric configuration lacks validation
**Issue:** Timeout, output limit and port are parsed directly. Zero/negative output limits produce misleading slicing; non-positive/extreme timeouts and invalid ports fail or behave unexpectedly.  
**Risk:** Misconfiguration removes effective limits or disables operation only at runtime.  
**Recommendation:** Validate strict safe ranges during startup.  
**Status:** OPEN

### KAI-SHELL-018 — MEDIUM — Commands inherit process context
**Issue:** No explicit `cwd`, minimal environment or environment scrub is supplied to subprocesses. Commands inherit the service working directory and all environment variables; `pwd`, `ps` and procfs expose aspects of that context.  
**Risk:** Deployment paths and secrets become reachable through multiple command channels, and behaviour varies by launch environment.  
**Recommendation:** Use an empty/minimal environment, fixed isolated cwd and no secret mounts/variables in the execution process.  
**Status:** OPEN

---

## Batch totals

- Findings: **18**
- Critical: **3**
- High: **7**
- Medium: **8**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **775**
- Critical: **85**
- High: **273**
- Medium: **414**
- Low: **3**

## Files materially reviewed in this batch

`sandboxes/shell/app.py`.
