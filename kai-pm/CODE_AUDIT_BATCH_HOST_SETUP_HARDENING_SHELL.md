# Kai Code Audit — Host Setup and Hardening Shell Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers previously unlogged findings in `scripts/setup.sh`, `scripts/entrypoint.sh`, `scripts/randomize_ssh_port.sh`, `scripts/check_pypi_shadow.sh` and `scripts/setup_fail2ban.sh`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-HOSTSH-001 | HIGH | Setup does not change to or verify the repository root before creating files and running Compose/Make |
| KAI-HOSTSH-002 | HIGH | Docker Compose plugin detection uses `command -v docker compose` and fails on normal plugin-only installations |
| KAI-HOSTSH-003 | HIGH | Minimal `.env` generation writes a known `change-me-to-a-random-secret` HMAC value |
| KAI-HOSTSH-004 | HIGH | Setup continues to build and reports completion while known placeholder secrets remain |
| KAI-HOSTSH-005 | HIGH | Generated `.env` permissions are not restricted |
| KAI-HOSTSH-006 | HIGH | Generated configuration defaults the system to WORK mode |
| KAI-HOSTSH-007 | HIGH | Existing `.env` is trusted without checking required keys, placeholders, permissions or syntax |
| KAI-HOSTSH-008 | HIGH | `.env.example` is copied without validating that its values are safe for the current deployment |
| KAI-HOSTSH-009 | HIGH | The go/no-go result is explicitly non-fatal and stderr is suppressed |
| KAI-HOSTSH-010 | HIGH | Setup reports complete even after the validation command fails |
| KAI-HOSTSH-011 | HIGH | The documented next-step URL points to deprecated Orchestrator port 8050 rather than the active Dashboard gateway |
| KAI-HOSTSH-012 | HIGH | Docker image builds execute repository Dockerfiles without source or dependency integrity verification |
| KAI-HOSTSH-013 | MEDIUM | Tool presence is checked but minimum/supported versions are not validated |
| KAI-HOSTSH-014 | MEDIUM | Docker daemon reachability and caller permissions are not checked before the build |
| KAI-HOSTSH-015 | MEDIUM | No host disk, memory, CPU, architecture or operating-system readiness is checked |
| KAI-HOSTSH-016 | MEDIUM | Setup does not start services or verify runtime health despite reporting completion |
| KAI-HOSTSH-017 | MEDIUM | Existing generated files are not backed up or revisioned |
| KAI-HOSTSH-018 | MEDIUM | `.env` creation is not atomic and can leave a partial configuration |
| KAI-HOSTSH-019 | MEDIUM | The script does not verify Git working-tree state or expected repository revision |
| KAI-HOSTSH-020 | MEDIUM | Compose command is retained as a whitespace-split string rather than an argv array |
| KAI-HOSTSH-021 | MEDIUM | Build output and resulting image digests are not recorded in a setup report |
| KAI-HOSTSH-022 | MEDIUM | Setup has no rollback or cleanup when the build or later check fails |
| KAI-HOSTSH-023 | HIGH | `TAILSCALE_IP` accepts broad networks such as `0.0.0.0/0` and can expose SSH globally |
| KAI-HOSTSH-024 | HIGH | UFW rule-creation failure is ignored with `|| true` |
| KAI-HOSTSH-025 | HIGH | The application is started after a failed allow rule if UFW enable succeeds |
| KAI-HOSTSH-026 | HIGH | The entrypoint always opens port 22 and ignores the randomised/configured SSH port |
| KAI-HOSTSH-027 | HIGH | Enabling UFW during application startup can lock out the operator |
| KAI-HOSTSH-028 | HIGH | Repeated starts accumulate obsolete Tailscale source rules without removal or reconciliation |
| KAI-HOSTSH-029 | HIGH | The script does not verify that UFW rules actually protect the host rather than an isolated container namespace |
| KAI-HOSTSH-030 | HIGH | Arbitrary `exec` arguments become the final process without an approved command policy |
| KAI-HOSTSH-031 | MEDIUM | IP/CIDR syntax, address family and expected Tailscale range are not validated |
| KAI-HOSTSH-032 | MEDIUM | Existing firewall policy and required service ports are not inspected before enabling UFW |
| KAI-HOSTSH-033 | MEDIUM | No post-enable UFW status or effective-rule check is performed |
| KAI-HOSTSH-034 | MEDIUM | The entrypoint requires root-like firewall privileges but does not verify the execution context |
| KAI-HOSTSH-035 | MEDIUM | Firewall mutations have no rollback if the application command later fails |
| KAI-HOSTSH-036 | MEDIUM | Firewall changes have no durable actor, source revision or before/after audit record |
| KAI-HOSTSH-037 | MEDIUM | An empty final command is not validated before `exec` |
| KAI-HOSTSH-038 | HIGH | SSH port randomisation does not verify that the selected port is unused |
| KAI-HOSTSH-039 | HIGH | The random range can collide with KAI, database, monitoring and other host services |
| KAI-HOSTSH-040 | HIGH | Generic `.env` key `PORT` is overwritten even though many services may interpret it differently |
| KAI-HOSTSH-041 | HIGH | The `.env` target depends on the caller’s working directory |
| KAI-HOSTSH-042 | HIGH | `.env` and `sshd_config` edits are non-atomic and lack locks or backups |
| KAI-HOSTSH-043 | HIGH | SSH configuration is changed without running `sshd -t` validation first |
| KAI-HOSTSH-044 | HIGH | SSH restart failure is ignored and the script still prints a successful new port |
| KAI-HOSTSH-045 | HIGH | A missing SSH configuration file still produces “SSH port set” success |
| KAI-HOSTSH-046 | HIGH | Firewall, Fail2ban, Tailscale ACLs and client connection instructions are not updated with the new port |
| KAI-HOSTSH-047 | HIGH | The current SSH session can be cut off with no connectivity test or rollback timer |
| KAI-HOSTSH-048 | HIGH | Every invocation changes the port again with no stable desired-state or idempotency policy |
| KAI-HOSTSH-049 | HIGH | Existing multiple or commented SSH Port directives are not reconciled safely |
| KAI-HOSTSH-050 | HIGH | Partial failure can leave `.env` and SSH daemon configuration on different ports |
| KAI-HOSTSH-051 | MEDIUM | The required `shuf` command is not checked before use |
| KAI-HOSTSH-052 | MEDIUM | The script does not explicitly require root before editing system SSH configuration |
| KAI-HOSTSH-053 | MEDIUM | No systemd/SSH unit discovery is performed before trying two hard-coded service names |
| KAI-HOSTSH-054 | MEDIUM | Randomness source and selected-port event are not recorded with host/configuration identity |
| KAI-HOSTSH-055 | MEDIUM | Generated `.env` permissions and ownership are not hardened |
| KAI-HOSTSH-056 | MEDIUM | SSH configuration permissions and owner are not verified before modification |
| KAI-HOSTSH-057 | MEDIUM | IPv4/IPv6 listener behaviour is not checked after the change |
| KAI-HOSTSH-058 | MEDIUM | No connection test confirms the daemon listens on the selected port |
| KAI-HOSTSH-059 | MEDIUM | The script does not detect port-policy restrictions or reserved administrative ranges |
| KAI-HOSTSH-060 | MEDIUM | The selected port is printed to ordinary terminal/automation logs without a protected configuration record |
| KAI-HOSTSH-061 | MEDIUM | `sed -i` behaviour is platform-dependent and no supported OS is enforced |
| KAI-HOSTSH-062 | MEDIUM | No trap restores prior files when an intermediate command fails |
| KAI-HOSTSH-063 | HIGH | PyPI-shadow detection relies on a manually maintained finite blocklist |
| KAI-HOSTSH-064 | HIGH | Only repository-root directories are checked |
| KAI-HOSTSH-065 | HIGH | Shadowing Python files such as `requests.py` or `json.py` are not detected |
| KAI-HOSTSH-066 | HIGH | Modules in `scripts/` and other early `sys.path` directories are not checked |
| KAI-HOSTSH-067 | HIGH | Distribution-name normalisation for hyphens, underscores and case is not applied |
| KAI-HOSTSH-068 | HIGH | `KAI_SHADOW_ALLOW` can bypass any blocklisted package name with no approval or expiry |
| KAI-HOSTSH-069 | HIGH | `langgraph` is permanently exempted even though it intentionally shadows a package namespace |
| KAI-HOSTSH-070 | HIGH | The check does not resolve actual Python imports or installed package origins |
| KAI-HOSTSH-071 | HIGH | Blocklist contents have no integrity signature, source or update mechanism |
| KAI-HOSTSH-072 | MEDIUM | Invalid blocklist entries are silently ignored |
| KAI-HOSTSH-073 | MEDIUM | Invalid allowlist entries are silently ignored without a failed hardening result |
| KAI-HOSTSH-074 | MEDIUM | Symlink targets and namespace-package behaviour are not inspected |
| KAI-HOSTSH-075 | MEDIUM | Package names declared in project metadata are not compared with filesystem/import names |
| KAI-HOSTSH-076 | MEDIUM | The external `xargs` dependency is not checked |
| KAI-HOSTSH-077 | MEDIUM | Successful output does not list the blocklist/allowlist revision or scanned paths |
| KAI-HOSTSH-078 | MEDIUM | Environment-based exceptions are not recorded as an auditable waiver |
| KAI-HOSTSH-079 | MEDIUM | The scan is point-in-time only and provides no runtime import protection |
| KAI-HOSTSH-080 | MEDIUM | Case-sensitive filesystem tests produce different results across platforms |
| KAI-HOSTSH-081 | HIGH | Fail2ban `PORT` accepts arbitrary text and newline characters from the environment |
| KAI-HOSTSH-082 | HIGH | A multiline `PORT` value can inject additional Fail2ban directives or jail sections |
| KAI-HOSTSH-083 | HIGH | The generic `PORT` variable may not represent the actual SSH daemon port |
| KAI-HOSTSH-084 | HIGH | The jail file is overwritten directly without backup, lock or atomic replacement |
| KAI-HOSTSH-085 | HIGH | Fail2ban configuration is not validated before service restart |
| KAI-HOSTSH-086 | HIGH | Package installation uses mutable current repository metadata without a pinned version |
| KAI-HOSTSH-087 | HIGH | The script runs privileged package-manager and systemd mutations with no explicit operator confirmation |
| KAI-HOSTSH-088 | HIGH | Existing Fail2ban customisations in the same file are destroyed |
| KAI-HOSTSH-089 | HIGH | No rollback restores prior package/configuration state after restart failure |
| KAI-HOSTSH-090 | HIGH | SSH log backend, journal availability and jail match effectiveness are not verified |
| KAI-HOSTSH-091 | HIGH | Fail2ban is configured independently from firewall and SSH-port randomisation state |
| KAI-HOSTSH-092 | MEDIUM | Numeric port range and single-port syntax are not checked |
| KAI-HOSTSH-093 | MEDIUM | Root privileges are assumed rather than verified with a controlled error |
| KAI-HOSTSH-094 | MEDIUM | The script supports only apt/systemd Debian-like hosts despite broader setup claims |
| KAI-HOSTSH-095 | MEDIUM | `apt-get update` and installation have no timeout or retry budget |
| KAI-HOSTSH-096 | MEDIUM | Static retry/findtime/bantime values are not linked to current threat or access policy |
| KAI-HOSTSH-097 | MEDIUM | Service enable/restart actions are not tied to an immutable configuration revision |
| KAI-HOSTSH-098 | MEDIUM | No test login/failure event confirms that banning works |
| KAI-HOSTSH-099 | MEDIUM | Configuration file permissions and ownership are not explicitly enforced |
| KAI-HOSTSH-100 | MEDIUM | Setup changes have no structured report or immutable host-hardening audit record |

---

## One-command setup — `scripts/setup.sh`

### KAI-HOSTSH-001 — HIGH — Working-directory confusion
Every `.env`, Compose and Make operation is relative to the invocation directory.

### KAI-HOSTSH-002 — HIGH — Broken plugin detection
`command -v docker compose` tests two command names; the Compose plugin is not normally a standalone `compose` executable.

### KAI-HOSTSH-003 — HIGH — Known default secret
The generated HMAC value is committed in source and predictable.

### KAI-HOSTSH-004 — HIGH — Placeholder is non-blocking
The script immediately proceeds to image build and completion.

### KAI-HOSTSH-005 — HIGH — Secret-file mode absent
`cp`/redirection use the caller’s umask.

### KAI-HOSTSH-006 — HIGH — Execution mode default
New installations start configuration in WORK mode.

### KAI-HOSTSH-007 — HIGH — Existing environment trusted
No placeholder or required-variable scan occurs.

### KAI-HOSTSH-008 — HIGH — Example copied as deployment truth
No environment-specific validation follows.

### KAI-HOSTSH-009 — HIGH — Failed gate suppressed
`make go_no_go` failure only emits a warning; stderr is discarded.

### KAI-HOSTSH-010 — HIGH — False completion
The final banner is unconditional after the non-fatal gate.

### KAI-HOSTSH-011 — HIGH — Wrong application URL
Port 8050 belongs to the deprecated Orchestrator stub, while the active Dashboard uses 8080.

### KAI-HOSTSH-012 — HIGH — Unattested build
No image/source/dependency digest is checked or recorded.

### KAI-HOSTSH-013 — MEDIUM — Version compatibility absent
Any installed Docker/Python/Git version passes.

### KAI-HOSTSH-014 — MEDIUM — Daemon permission untested
Only CLI presence/version is queried.

### KAI-HOSTSH-015 — MEDIUM — Capacity checks absent
No minimum resources are verified.

### KAI-HOSTSH-016 — MEDIUM — No runtime postcondition
Images are built but services are not started/checked.

### KAI-HOSTSH-017 — MEDIUM — Existing-file safety absent
No backup/versioning is created.

### KAI-HOSTSH-018 — MEDIUM — Partial `.env` risk
Direct redirection can leave incomplete content.

### KAI-HOSTSH-019 — MEDIUM — Repository identity absent
Dirty/wrong revisions are not detected.

### KAI-HOSTSH-020 — MEDIUM — Command-string execution
`$COMPOSE` relies on shell word splitting.

### KAI-HOSTSH-021 — MEDIUM — Missing setup artefact
No machine-readable build report exists.

### KAI-HOSTSH-022 — MEDIUM — No rollback
Partial setup state remains.

---

## Firewall entrypoint — `scripts/entrypoint.sh`

### KAI-HOSTSH-023 — HIGH — Broad-source exposure
Any UFW-compatible source, including all networks, is accepted.

### KAI-HOSTSH-024 — HIGH — Rule failure ignored
The allow command cannot block startup.

### KAI-HOSTSH-025 — HIGH — False secured state
UFW may be enabled without the required allow rule.

### KAI-HOSTSH-026 — HIGH — Port-state mismatch
The hard-coded 22 conflicts with randomisation and Fail2ban scripts.

### KAI-HOSTSH-027 — HIGH — Lockout risk
Firewall enable is immediate and untested.

### KAI-HOSTSH-028 — HIGH — Rule accumulation
Old source addresses are never removed.

### KAI-HOSTSH-029 — HIGH — Namespace ambiguity
Container UFW may not enforce host ingress.

### KAI-HOSTSH-030 — HIGH — Unconstrained final command
Any argv supplied by image/runtime is executed.

### KAI-HOSTSH-031 — MEDIUM — Address validation absent
No Tailscale range or IP version check.

### KAI-HOSTSH-032 — MEDIUM — Existing policy ignored
The script does not preserve/verify other necessary access.

### KAI-HOSTSH-033 — MEDIUM — No effective status verification
Only command exit is used.

### KAI-HOSTSH-034 — MEDIUM — Privilege assumption
Root/network capability is implicit.

### KAI-HOSTSH-035 — MEDIUM — No rollback
Later application failure leaves firewall mutation.

### KAI-HOSTSH-036 — MEDIUM — Missing audit
No before/after rule evidence.

### KAI-HOSTSH-037 — MEDIUM — Empty exec unhandled
No usage/command check exists.

---

## SSH randomisation — `scripts/randomize_ssh_port.sh`

### KAI-HOSTSH-038 — HIGH — Port conflict unchecked
Random selection is not tested with socket/system service state.

### KAI-HOSTSH-039 — HIGH — Shared-service collision
The whole unprivileged range is used.

### KAI-HOSTSH-040 — HIGH — Generic configuration overwrite
`PORT` is not SSH-specific.

### KAI-HOSTSH-041 — HIGH — CWD configuration confusion
`.env` may be created outside the repository.

### KAI-HOSTSH-042 — HIGH — Unsafe dual-file mutation
No atomicity/locking/backup.

### KAI-HOSTSH-043 — HIGH — No SSH syntax test
The daemon is restarted directly.

### KAI-HOSTSH-044 — HIGH — Restart failure masked
Both restart failures are followed by success output.

### KAI-HOSTSH-045 — HIGH — Missing config masked
The success message is unconditional.

### KAI-HOSTSH-046 — HIGH — Dependent controls stale
Firewall/Fail2ban/client state is not reconciled.

### KAI-HOSTSH-047 — HIGH — Remote lockout
No rollback timer or second-session check.

### KAI-HOSTSH-048 — HIGH — Non-idempotent desired state
Every call picks a new port.

### KAI-HOSTSH-049 — HIGH — Directive ambiguity
Only exact active `Port ` lines are replaced.

### KAI-HOSTSH-050 — HIGH — Partial-state mismatch
Environment update precedes system-config/restart success.

### KAI-HOSTSH-051 — MEDIUM — `shuf` prerequisite absent
Failure is not explained by preflight.

### KAI-HOSTSH-052 — MEDIUM — Root check absent
System write fails midway for ordinary users.

### KAI-HOSTSH-053 — MEDIUM — Service manager assumptions
Only two unit names/systemctl are attempted.

### KAI-HOSTSH-054 — MEDIUM — Randomisation provenance absent
No operation ID/host/source record.

### KAI-HOSTSH-055 — MEDIUM — `.env` hardening absent
Mode/owner are not set.

### KAI-HOSTSH-056 — MEDIUM — SSH config trust absent
Existing file owner/mode/symlink is not checked.

### KAI-HOSTSH-057 — MEDIUM — Listener-family state unknown
No IPv4/IPv6 verification.

### KAI-HOSTSH-058 — MEDIUM — Postcondition absent
Listening port is not probed.

### KAI-HOSTSH-059 — MEDIUM — Policy constraints absent
Reserved/approved ranges are not enforced.

### KAI-HOSTSH-060 — MEDIUM — Unprotected output
The port appears only as terminal text, not controlled configuration evidence.

### KAI-HOSTSH-061 — MEDIUM — Platform-specific edit
`sed -i` differs across systems.

### KAI-HOSTSH-062 — MEDIUM — No restoration trap
Intermediate changes survive failure.

---

## PyPI-shadow check — `scripts/check_pypi_shadow.sh`

### KAI-HOSTSH-063 — HIGH — Finite manual detection
Only listed names are protected.

### KAI-HOSTSH-064 — HIGH — Root-directory-only scan
Other import locations are ignored.

### KAI-HOSTSH-065 — HIGH — Module files omitted
A single `.py` file can shadow a package.

### KAI-HOSTSH-066 — HIGH — Script-directory imports omitted
Python places the executing script directory early on `sys.path`.

### KAI-HOSTSH-067 — HIGH — Name normalisation omitted
Equivalent distribution/import spellings are not compared.

### KAI-HOSTSH-068 — HIGH — Environment waiver
One variable removes protection for any named entry.

### KAI-HOSTSH-069 — HIGH — Permanent namespace exception
The known shim remains an accepted shadow.

### KAI-HOSTSH-070 — HIGH — No import-origin verification
The Python resolver is never queried.

### KAI-HOSTSH-071 — HIGH — Blocklist authority untrusted
No signed/versioned source exists.

### KAI-HOSTSH-072 — MEDIUM — Malformed entries disappear
Invalid blocklist lines are skipped.

### KAI-HOSTSH-073 — MEDIUM — Malformed waivers disappear
No failed configuration state.

### KAI-HOSTSH-074 — MEDIUM — Symlink/namespace semantics absent
Targets/package portions are not resolved.

### KAI-HOSTSH-075 — MEDIUM — Project metadata omitted
Declared package namespaces are not checked.

### KAI-HOSTSH-076 — MEDIUM — Dependency assumption
`xargs` is required.

### KAI-HOSTSH-077 — MEDIUM — Scan evidence absent
The success result lacks manifest/revision.

### KAI-HOSTSH-078 — MEDIUM — Waiver evidence absent
No owner/reason/expiry.

### KAI-HOSTSH-079 — MEDIUM — No runtime enforcement
Later generated files/path changes bypass the check.

### KAI-HOSTSH-080 — MEDIUM — Platform inconsistency
Case sensitivity changes results.

---

## Fail2ban setup — `scripts/setup_fail2ban.sh`

### KAI-HOSTSH-081 — HIGH — Unsanitised configuration value
`PORT` is expanded verbatim into a root-owned config file.

### KAI-HOSTSH-082 — HIGH — Heredoc directive injection
Newlines can add arbitrary Fail2ban settings/sections.

### KAI-HOSTSH-083 — HIGH — Ambiguous port source
The generic variable may belong to another service.

### KAI-HOSTSH-084 — HIGH — Destructive config write
Existing file content is replaced directly.

### KAI-HOSTSH-085 — HIGH — No pre-restart config test
Invalid config can break protection.

### KAI-HOSTSH-086 — HIGH — Unpinned package
Current apt metadata determines installed code.

### KAI-HOSTSH-087 — HIGH — Privileged mutation without confirmation
Package and service changes begin immediately.

### KAI-HOSTSH-088 — HIGH — Existing policy overwritten
No merge/preservation.

### KAI-HOSTSH-089 — HIGH — No rollback
Prior working state is not restored.

### KAI-HOSTSH-090 — HIGH — Detection backend unverified
The jail can be active but ineffective.

### KAI-HOSTSH-091 — HIGH — Independent security-state drift
Other scripts change the port/firewall separately.

### KAI-HOSTSH-092 — MEDIUM — Port schema absent
No numeric/range check.

### KAI-HOSTSH-093 — MEDIUM — Privilege assumption
Root is implicit.

### KAI-HOSTSH-094 — MEDIUM — Narrow platform support
apt/systemd only.

### KAI-HOSTSH-095 — MEDIUM — Unbounded package commands
No timeout/retry control.

### KAI-HOSTSH-096 — MEDIUM — Static ban policy
No deployment/threat calibration.

### KAI-HOSTSH-097 — MEDIUM — Revision link absent
Service state is not tied to a config digest.

### KAI-HOSTSH-098 — MEDIUM — Functional test absent
No generated failed login proves operation.

### KAI-HOSTSH-099 — MEDIUM — File hardening omitted
Mode/owner not explicitly set.

### KAI-HOSTSH-100 — MEDIUM — Audit artefact absent
No structured host-hardening record.

---

## Batch totals

- Findings: **100**
- Critical: **0**
- High: **55**
- Medium: **45**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,456**
- Critical: **181**
- High: **1,225**
- Medium: **1,047**
- Low: **3**

## Files materially reviewed

`scripts/setup.sh`, `scripts/entrypoint.sh`, `scripts/randomize_ssh_port.sh`, `scripts/check_pypi_shadow.sh`, `scripts/setup_fail2ban.sh`, with port/service/deployment references checked against current Compose and Orchestrator/Dashboard sources.
