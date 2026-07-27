# Kai Code Audit — Shell Sandbox Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_SHELL_SANDBOX.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-SHELLX-001 | CRITICAL | `/proc/self/root` provides a direct arbitrary-readable-filesystem escape inside the default safe prefix |
| KAI-SHELLX-002 | CRITICAL | Allowlisted `date --file` reads arbitrary files because `date` is not path-restricted |
| KAI-SHELLX-003 | HIGH | `df` accepts unrestricted filesystem path operands outside the safe-directory policy |
| KAI-SHELLX-004 | HIGH | Command output can contain terminal/ANSI/OSC control sequences and is returned without a safe text contract |
| KAI-SHELLX-005 | HIGH | Nonzero command exits are returned with HTTP 200 and can be misclassified as successful operations |
| KAI-SHELLX-006 | HIGH | The output limit is applied independently to stdout and stderr, permitting nearly twice the advertised response budget |
| KAI-SHELLX-007 | HIGH | Output truncation uses Unicode characters rather than encoded bytes |
| KAI-SHELLX-008 | MEDIUM | `/proc/self/cwd` and `/proc/self/fd/*` expose symlinked process resources inside the allowed prefix |
| KAI-SHELLX-009 | MEDIUM | Recursive `ls` and `du` options can traverse extremely large allowed trees with no operation-specific work budget |
| KAI-SHELLX-010 | MEDIUM | Named pipes and special files inside safe directories can hold workers until timeout |
| KAI-SHELLX-011 | MEDIUM | Command responses contain no executable digest, environment revision or policy version |
| KAI-SHELLX-012 | MEDIUM | No immutable audit links caller, normalised command, resolved executable, files accessed and result digest |

---

## Critical findings

### KAI-SHELLX-001 — CRITICAL — `/proc/self/root` bypasses the filesystem boundary
**Issue:** `/proc/self` is a default safe directory. Linux exposes `/proc/self/root` as a symlink to the process root filesystem. A request such as `cat /proc/self/root/etc/passwd` passes the lexical prefix check and reads the target through procfs.  
**Risk:** The default policy directly exposes any container-readable file, not only environment data.  
**Recommendation:** prohibit all procfs/sysfs paths and use descriptor-based access beneath dedicated immutable data roots.  
**Status:** OPEN — immediate remediation required

### KAI-SHELLX-002 — CRITICAL — `date --file` bypasses path controls
**Issue:** GNU `date` accepts `-f FILE` and `--file=FILE`, reading date strings from a caller-selected file. `date` is allowlisted but absent from `_PATH_ARG_COMMANDS`, so no file argument is checked.  
**Risk:** Callers can make the process open arbitrary readable files outside safe directories and obtain parsed content/errors/timing information.  
**Recommendation:** replace command execution with fixed typed operations and reject every option/operand not explicitly modelled.  
**Status:** OPEN — immediate remediation required

---

## High-severity findings

### KAI-SHELLX-003 — HIGH — `df` path operands bypass the root policy
`df` accepts arbitrary files/directories and reveals their backing filesystem, mountpoint, capacity and existence without the safe-directory check.

### KAI-SHELLX-004 — HIGH — Unsafe terminal-control output
Allowed commands can emit colour, hyperlink and other escape/control sequences through options or file contents. The service returns them as ordinary strings for logs, terminals and web clients.

### KAI-SHELLX-005 — HIGH — Failed commands use success transport semantics
Any completed nonzero exit becomes `ShellResult(status="error")` with HTTP 200. Shared proxies/resilience code can treat it as a successful execution.

### KAI-SHELLX-006 — HIGH — Combined output budget is doubled
Both stdout and stderr are independently sliced to `MAX_OUTPUT_BYTES`; the response may contain approximately twice the configured maximum plus metadata.

### KAI-SHELLX-007 — HIGH — Character limit is not a byte limit
The field/configuration is named bytes, but slicing decoded Python strings counts Unicode code points. Encoded response size can substantially exceed the intended limit.

---

## Medium-severity findings

### KAI-SHELLX-008 — MEDIUM — Procfs symlink resource exposure
`/proc/self/cwd`, `/proc/self/exe` and `/proc/self/fd/*` are symlinked process resources reachable through the allowed prefix and reveal/open paths outside it.

### KAI-SHELLX-009 — MEDIUM — Command-specific traversal budgets absent
Recursive or summary options can traverse very large `/tmp` or log trees; only one coarse wall-clock timeout exists.

### KAI-SHELLX-010 — MEDIUM — Special-file blocking
FIFOs/devices/sockets reachable under configured roots are not rejected as non-regular files and can block/read unpredictably until timeout.

### KAI-SHELLX-011 — MEDIUM — Missing execution provenance
Results do not state the resolved executable inode/hash, environment/cwd digest, safe-root revision or policy version.

### KAI-SHELLX-012 — MEDIUM — Missing operation audit
There is no tamper-evident record of principal, exact normalised arguments, canonical file targets, output digest and completion state.

---

## Batch totals

- Findings: **12**
- Critical: **2**
- High: **5**
- Medium: **5**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,472**
- Critical: **191**
- High: **1,238**
- Medium: **1,040**
- Low: **3**

## Files materially reviewed

`sandboxes/shell/app.py` and the existing Shell Sandbox audit.
