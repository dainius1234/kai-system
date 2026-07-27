# Kai System — Phase 0 Source-Specific Containment Plan

Repository: `dainius1234/kai-system`  
Parent backlog: `kai-pm/CODE_AUDIT_REMEDIATION_BACKLOG.md`  
Source audit: `kai-pm/CODE_AUDIT_FINAL_REPORT.md`  
Status: **IMPLEMENTATION PLAN ONLY — NO RUNTIME CODE OR CONFIGURATION CHANGED**

---

## 1. Objective

Break the currently reachable compromise paths before feature remediation begins. This plan maps the Phase 0 backlog to concrete repository files and an ordered pull-request sequence.

Phase 0 does not make Kai production-safe. It creates a contained development baseline from which the P1 identity and enforcement architecture can be built.

---

## 2. Confirmed source hotspots

### Deployment topology

Primary files:

- `docker-compose.full.yml`
- `docker-compose.minimal.yml`

Confirmed conditions in both deployment definitions include:

- One flat `sovereign-net` bridge with static addresses in `172.20.0.0/16`.
- Broad `ports:` publication for privileged services.
- `restart: unless-stopped` inherited widely.
- Mutable image/model tags.
- health checks based mainly on HTTP reachability.
- inconsistent full/minimal inventories and identities.

### Tool Gate startup and development authentication

Primary files:

- `docker-compose.full.yml`
- `docker-compose.minimal.yml`
- `tool-gate/` application and authentication modules
- `common/auth.py` or the current shared secret loader

Confirmed conditions:

- `MODE: "WORK"` is configured at deployment startup.
- Minimal Compose sets `HMAC_ALLOW_DEV_SECRET: "true"` for Tool Gate and Agentic.
- Tool Gate is directly host-published.
- Runtime services can reach it over unauthenticated plaintext service transport.

### Dashboard automatic mode transition

Primary file:

- `dashboard/static/app.html`

Confirmed condition:

- Page initialisation calls `loadMode()`; absent browser state defaults to `WORK` and invokes `setMode()`, posting the mode through the Dashboard backend.

Related gateway file:

- `dashboard/app.py`

The Dashboard must not retain or borrow a privileged Tool Gate administrative credential during containment.

### Database and Redis secrets

Primary files:

- `docker-compose.full.yml`
- `docker-compose.minimal.yml`
- environment examples, setup scripts and CI workflows that supply `DB_PASSWORD` or dev HMAC flags

Confirmed conditions:

- Postgres and service connection strings use `${DB_PASSWORD:-localdev}`.
- Full Compose declares a `db_password` Docker secret but the displayed database configuration still uses the environment fallback path.
- Redis starts without authentication or TLS.
- Multiple services receive complete database URIs through environment variables.

### Shared TurboVec writer

Primary files:

- `docker-compose.full.yml`
- `docker-compose.minimal.yml`
- `memu-core/`

Confirmed condition:

- `memu-core` and `memu-core-introspect` mount `turbovec_data` and point at the same `/data/turbovec/memories.tv` with separate process-local state and write paths.

### Consequential services enabled as ordinary fleet members

Primary files:

- `docker-compose.full.yml`
- `docker-compose.minimal.yml`

Containment applies at minimum to:

- Dashboard administrative gateway.
- Executor.
- Browser Agent / Web Scout / external browse paths.
- Vault Sync and arbitrary file ingestion.
- memU introspection and graph ingestion.
- Supervisor recovery actions.
- Broker and autonomous financial actions.
- Camera, screen, clipboard, audio and wake ingestion.
- Monitor, Docker watcher, Git watcher and other host-observation services.

---

# 3. Ordered implementation PRs

## P0-PR-01 — Evidence freeze and deployment manifest capture

**Backlog mapping:** KAI-REM-006  
**Code effect:** none; operational evidence and documentation only

### Files to add/update

- Add an evidence acquisition script under `scripts/security/`.
- Add a generated-manifest schema under `kai-pm/evidence/` or a protected external location reference.
- Update incident/operations documentation.

### Required capture

- Current Git commit and dirty state.
- Resolved output of both Compose definitions.
- Container image IDs/digests.
- Running container configuration and network membership.
- Environment-key names without publishing secret values.
- Volume inventory and immutable snapshots where supported.
- Postgres, Redis, JSONL, audit ledger, TurboVec and graph snapshots.
- Hashes, acquisition time, host identity and operator.

### Acceptance tests

- Acquisition is repeatable and read-only.
- Evidence hashes verify after copying.
- One protected copy exists outside application-managed retention.
- Secret values are not added to Git.

### Merge/release rule

Complete before credential rotation, volume cleanup, network changes or restart-heavy work.

---

## P0-PR-02 — Edge lockdown and host-port removal

**Backlog mapping:** KAI-REM-001  
**Primary files:** `docker-compose.full.yml`, `docker-compose.minimal.yml`

### Required changes

1. Remove `ports:` from every privileged internal service.
2. Keep only one approved ingress endpoint.
3. Bind any temporary development ingress explicitly to loopback, for example `127.0.0.1:<host>:<container>`, never an implicit all-interface bind.
4. Remove Ollama and data-plane host publication unless an explicit local-only requirement exists.
5. Ensure full and minimal definitions use the same ingress rule.
6. Add a machine check that fails CI when a disallowed service publishes a port.

### Services that must not be directly published

At minimum:

- Tool Gate.
- memU Core and Introspection.
- Agentic and Agentic Introspection.
- Executor.
- Verifier and Fusion.
- Supervisor and Heartbeat.
- Browser, files, Vault, monitor, broker and watcher services.
- Sensors and perception services.
- Databases, Redis, model servers and graph backends.

### Acceptance tests

- `docker compose config` shows no disallowed host bindings.
- LAN scan reaches no privileged service.
- Direct calls to former service ports fail from the host and another LAN machine.
- The approved local ingress remains functional only on the intended interface.

### Rollback constraint

Rollback may restore a local-only diagnostic ingress but must not restore broad host publication.

---

## P0-PR-03 — Dangerous capability profiles and default-off fleet

**Backlog mapping:** KAI-REM-002  
**Primary files:** both Compose definitions; README/Makefile/start scripts

### Required changes

Create explicit Compose profiles, for example:

- `core-readonly`
- `operator-ui`
- `execution-lab`
- `external-egress`
- `sensors`
- `finance-sim`
- `recovery-admin`

The exact names may differ, but default startup must exclude consequential services.

### Default-disabled services/capabilities

- Executor and generic subprocess execution.
- Browser Agent, Web Scout and broad external egress.
- Vault Sync and arbitrary file ingestion.
- Introspection and graph mutation.
- Broker/live finance mutation.
- Camera, screen, clipboard, audio and wake ingestion.
- Supervisor-initiated recovery.
- Host Docker/Git/process watchers.
- Dashboard administrative mutation routes.

### Required behavioural changes

- Default `docker compose up` starts only the minimum contained core.
- Makefile and setup scripts must not silently add dangerous profiles.
- Profile activation must be visible in resolved configuration and startup output.
- CI must test the true default profile, not only a fully enabled stack.

### Acceptance tests

- Default deployment contains no listed consequential service.
- Each dangerous profile can be enumerated before activation.
- Enabling one profile does not implicitly enable unrelated profiles.
- Profile-disabled services have no residual published endpoint or scheduled worker.

---

## P0-PR-04 — Tool Gate locked startup and Dashboard mode containment

**Backlog mapping:** KAI-REM-003  
**Primary files:**

- `docker-compose.full.yml`
- `docker-compose.minimal.yml`
- `dashboard/static/app.html`
- `dashboard/app.py`
- Tool Gate mode/configuration modules

### Required changes

1. Replace startup `MODE: "WORK"` with an explicit locked/restricted state.
2. Remove any code that changes server enforcement mode during page rendering or initialisation.
3. Remove `WORK` as the browser default when localStorage is empty.
4. Treat browser mode as display state only; fetch the authoritative server mode.
5. Remove Dashboard possession of a reusable Tool Gate administrative credential.
6. Require explicit authenticated operator action for a later mode transition.
7. Ensure restart, health recovery and configuration reload preserve the restrictive state.
8. Reject unknown mode strings at client, gateway and Tool Gate layers.

### Immediate containment behaviour

Until P1 identity and capability work exists, mode-changing routes should be disabled or restricted to a local, manual administrative process outside the ordinary Dashboard session.

### Acceptance tests

- Fresh browser profile cannot alter Tool Gate mode.
- Modified localStorage cannot alter Tool Gate mode.
- Opening or refreshing Dashboard produces zero mode-mutation requests.
- Tool Gate starts locked with absent, invalid or partial configuration.
- Restart/recovery cannot transition to WORK.
- Runtime service credential cannot perform mode administration.

---

## P0-PR-05 — Fail-closed secrets and credential rotation support

**Backlog mapping:** KAI-REM-004  
**Primary files:**

- `docker-compose.full.yml`
- `docker-compose.minimal.yml`
- shared authentication/secret loader
- setup scripts
- `.env` examples
- CI workflows and integration tests

### Required changes

1. Remove every `${DB_PASSWORD:-localdev}` fallback.
2. Make database secret absence a startup error.
3. Mount the declared database secret where it is actually consumed.
4. Stop embedding complete database credentials in broadly distributed environment URIs where a file/credential provider can be used.
5. Remove `HMAC_ALLOW_DEV_SECRET: "true"` from deployment definitions.
6. Ensure development secret fallback is impossible outside an explicit isolated test harness.
7. Add Redis authentication/ACL and protected transport or keep Redis entirely inside a tightly restricted temporary segment pending P1.
8. Separate database users by service and minimum privilege.
9. Add credential-rotation runbook and dual-key migration only where unavoidable.
10. Add secret scanning for current tree and, separately, historical Git objects.

### Acceptance tests

- Stack fails to start with absent database/HMAC/bridge secrets.
- Literal `localdev` is rejected as a privileged credential.
- Old rotated credentials fail.
- Low-purpose service credential cannot access another service's data authority.
- Secret values are absent from resolved diagnostic output where practical.
- Redis rejects unauthenticated clients.

### Rotation order

After evidence capture:

1. Bridge/Dashboard administrative credentials.
2. Tool Gate and interservice HMAC material.
3. Database credentials and service users.
4. Redis credentials.
5. Broker, Telegram, email and provider credentials.
6. Session and webhook secrets.

---

## P0-PR-06 — Temporary trust-zone segmentation

**Backlog mapping:** KAI-REM-005  
**Primary files:** both Compose definitions and host firewall/start scripts

### Required network zones

At minimum:

- `edge-net` — approved ingress only.
- `control-net` — identity/policy/Tool Gate, no external egress.
- `data-net` — Postgres, Redis, vector/graph stores.
- `agent-net` — Agentic/planning/verification.
- `execution-net` — Executor workers, denied access to control/data by default.
- `egress-net` — browser/feed/provider proxy only.
- `sensor-net` — camera/screen/audio/clipboard collection.
- `observability-net` — health/metrics with no administrative mutation authority.

### Required rules

- A service joins only the networks required by its data flow.
- Static IPs are removed as identity signals.
- Execution and egress zones cannot reach Tool Gate administration, identity, audit administration or direct data stores.
- Sensors cannot directly mutate memory, trust, identity or policy.
- Observability may read standard health/metrics but cannot invoke recovery.
- External egress is available only through a controlled proxy path.

### Acceptance tests

Use a source/destination matrix to assert every allowed and denied path. Required negative cases include:

- Executor → Tool Gate administration: denied.
- Executor → Postgres/Redis/memU administration: denied.
- Browser/egress worker → private service ranges: denied.
- Sensor service → memory write or Agentic execution: denied unless routed through a later authenticated policy path.
- Metrics/health observer → `/recover` or mode endpoint: denied.

---

## P0-PR-07 — Single-writer TurboVec containment

**Backlog mapping:** immediate part of KAI-REM-207  
**Primary files:**

- `docker-compose.full.yml`
- `docker-compose.minimal.yml`
- `memu-core/`
- introspection startup/configuration

### Required changes

1. Assign one process as the only writer to `memories.tv`.
2. Prevent Introspection from mounting the live index read-write.
3. Prefer an API/read model or immutable versioned snapshot for Introspection.
4. If a temporary shared mount remains, make the reader mount read-only and add generation/version checks.
5. Add startup lock/ownership validation so two writers cannot start.
6. Do not treat this containment as the final transactional memory design; P2 still requires durable outbox and derivative lineage.

### Acceptance tests

- Second writer refuses startup.
- Introspection cannot modify the live index.
- Concurrent read/load cannot corrupt index state.
- Index generation mismatch is reported as degraded/unavailable, not healthy.
- Failure during snapshot publication leaves the prior valid generation readable.

---

## P0-PR-08 — Restart, health and recovery containment

**Backlog mapping:** containment portion of KAI-REM-301 and KAI-REM-308  
**Primary files:** both Compose definitions, Supervisor, service health routes

### Required changes

1. Remove broad `restart: unless-stopped` from services where invalid security configuration must remain stopped.
2. Use a no-restart/fail-visible policy for missing secrets, invalid policy, corrupt index and incompatible model configuration.
3. Disable Supervisor recovery calls during Phase 0.
4. Ensure health observers cannot mutate service state.
5. Distinguish liveness from readiness and locked/degraded/stub states.
6. Prevent Dashboard/fleet summaries from representing a stub or locked service as operationally ready.
7. Add bounded logging to preserve disk and evidence.

### Acceptance tests

- Missing secret produces one visible failed start, not a restart loop.
- Locked Tool Gate is live but not execution-ready.
- Stub model/service is not reported ready for consequential use.
- Forged or public health input cannot trigger recovery.
- Restart does not clear mode, nonce, capability or containment state.

---

## P0-PR-09 — Compose convergence and policy-as-code checks

**Backlog mapping:** supports KAI-REM-001 to KAI-REM-005  
**Primary files:**

- `docker-compose.full.yml`
- `docker-compose.minimal.yml`
- `.github/workflows/`
- `scripts/`
- `Makefile`

### Required changes

Add automated validation for:

- Disallowed host-published ports.
- Missing profiles on dangerous services.
- `MODE: WORK` or equivalent execution-capable default.
- development secret flags.
- fallback passwords/secrets.
- flat-network membership and forbidden cross-zone joins.
- mutable privileged image tags.
- shared read-write sensitive volumes.
- health checks that do not distinguish readiness.
- drift between full and minimal architecture manifests.

Create one machine-readable service manifest containing:

- Service identity.
- Trust zone.
- Allowed callers and destinations.
- External egress requirement.
- Data classifications.
- Published ingress status.
- Required secrets.
- Health contract.
- Dangerous-capability profile.
- Volume ownership/read-write mode.

### Acceptance tests

A deliberately unsafe fixture must fail each policy check. Both Compose files must be generated from or validated against the same manifest.

---

# 4. Phase 0 pull-request boundaries

Do not combine all containment changes into one unreviewable PR.

Recommended boundaries:

1. Evidence capture only.
2. Host-port removal and ingress rule.
3. Dangerous-service profiles.
4. Tool Gate/Dashboard locked-mode behaviour.
5. Secret fallback removal and rotation support.
6. Network segmentation.
7. TurboVec single-writer containment.
8. Health/restart/recovery containment.
9. CI policy-as-code and Compose convergence.

Each PR must include:

- Exact audit finding IDs.
- Before/after resolved Compose excerpts.
- Negative tests.
- Rollback behaviour.
- Evidence that the change does not re-enable another dangerous route.

---

# 5. Finding mapping for Phase 0

Primary findings covered by this implementation sequence include:

- `KAI-ORCH-001` through `KAI-ORCH-018` where applicable.
- `KAI-ORCH-024`, `KAI-ORCH-026`, `KAI-ORCH-028`, `KAI-ORCH-030`, `KAI-ORCH-032`, `KAI-ORCH-034`, `KAI-ORCH-035`.
- `KAI-DASHUI-002`, `KAI-DASHUI-016`, `KAI-DASHUI-017`, `KAI-DASHUI-018`.
- Tool Gate findings covering default mode, reusable credentials, dev-secret fallback, replay/security state and direct endpoint access.
- Executor, Dashboard gateway, Supervisor, Vault, browser, sensor, finance and memory findings that rely on direct exposure or flat-network reachability.
- Cross-service compromise chains involving Dashboard, Executor, memU, Tool Gate ledger, Vault and recovery.

Finding IDs remain OPEN until implementation and closure evidence are reviewed. Mapping a finding to a PR is not closure.

---

# 6. Phase 0 exit criteria

Phase 0 is complete only when all are true:

- Evidence was preserved before destructive changes.
- No privileged service is directly published to host/LAN.
- Default deployment excludes consequential services.
- Tool Gate starts locked and Dashboard cannot change mode during load.
- No privileged fallback password or development HMAC secret is accepted.
- Redis and databases are not anonymously reachable.
- Temporary deny-by-default trust zones block lateral pivot paths.
- Exactly one TurboVec writer exists.
- Health observation cannot trigger recovery or widen permission.
- Invalid security configuration remains visibly failed instead of restart-looping.
- Full and minimal Compose definitions pass the same policy-as-code controls.
- External/LAN and container-network scans confirm the intended exposure matrix.

Passing Phase 0 permits contained development only. It does not permit production, sensitive data, autonomous execution or financial action.

---

# 7. Next dependency after Phase 0

The next mandatory work is P1:

- authoritative human and workload identity;
- explicit delegation;
- canonical operation serialisation and digest;
- single-use Tool Gate capabilities;
- enforcement at every final side-effect endpoint;
- separation of runtime, operator and approval credentials.

Do not use temporary network containment as a substitute for P1 identity and final-boundary enforcement.

**Current repository status remains: audit complete, planning advanced, no remediation implemented by this document.**
