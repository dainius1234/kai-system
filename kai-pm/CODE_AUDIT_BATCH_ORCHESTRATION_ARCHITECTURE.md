# Kai Code Audit — Orchestration and Deployment Control Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This phase covers Compose topology, service exposure, startup ordering, health semantics, secret distribution, isolation and fleet ownership. Component-level endpoint defects remain in their original batches.

## Consolidated batch index

| ID | Severity | Orchestration finding |
|---|---|---|
| KAI-ORCH-001 | CRITICAL | Nearly every privileged service is published directly on a host port instead of being reachable only through an authenticated gateway |
| KAI-ORCH-002 | CRITICAL | All services share one flat bridge network with no trust-zone or egress segmentation |
| KAI-ORCH-003 | CRITICAL | Full deployment starts Tool Gate in WORK mode by default |
| KAI-ORCH-004 | CRITICAL | Production database credentials fall back to the known password `localdev` |
| KAI-ORCH-005 | CRITICAL | memU Core and Introspection mount and mutate the same TurboVec index from separate processes |
| KAI-ORCH-006 | HIGH | Internal service traffic is plaintext HTTP with no mTLS service identity |
| KAI-ORCH-007 | HIGH | Health checks treat any HTTP-success `/health` response as readiness without validating semantic state |
| KAI-ORCH-008 | HIGH | `depends_on` frequently waits only for container start, not dependency readiness |
| KAI-ORCH-009 | HIGH | Several dependencies explicitly bypass their own connection/readiness tests |
| KAI-ORCH-010 | HIGH | memU graph ingest is enabled by default while graph backend access control is disabled |
| KAI-ORCH-011 | HIGH | Financial context is enabled by default and injected into Agentic before principal authorisation exists |
| KAI-ORCH-012 | HIGH | Secrets are distributed through environment-expanded connection strings to multiple services |
| KAI-ORCH-013 | HIGH | Redis runs without authentication while holding sessions, idempotency, audit and personal-state data |
| KAI-ORCH-014 | HIGH | Static container IPs create brittle identity assumptions and collision risk across alternate Compose stacks |
| KAI-ORCH-015 | HIGH | Minimal and full Compose definitions expose different service sets, ports and identities |
| KAI-ORCH-016 | HIGH | Running minimal and full stacks together creates host-port and network ownership conflicts |
| KAI-ORCH-017 | HIGH | Dangerous services are not separated behind Compose profiles or explicit operator activation |
| KAI-ORCH-018 | HIGH | Restart-unless-stopped can create uncontrolled restart loops for invalid security or model configuration |
| KAI-ORCH-019 | HIGH | External images and model tags are not pinned to immutable digests |
| KAI-ORCH-020 | HIGH | Model-pull completion is treated as model readiness without verifying exact model digest/capability |
| KAI-ORCH-021 | HIGH | Health start periods are not tied to actual model, graph or database warm-up duration |
| KAI-ORCH-022 | HIGH | No deployment-wide leader election exists for schedulers, supervisors, pollers or maintenance loops |
| KAI-ORCH-023 | HIGH | No central admission or resource budget coordinates expensive LLM, embedding, browser, graph and subprocess workloads |
| KAI-ORCH-024 | HIGH | Service recovery ownership is duplicated between Compose restart policy, Supervisor and service-local recovery endpoints |
| KAI-ORCH-025 | HIGH | Backup and persistence volumes have no deployment-level encryption, integrity or restore verification policy |
| KAI-ORCH-026 | HIGH | The fleet inventory used by Supervisor, Dashboard, health sweep and Metrics Gateway is inconsistent |
| KAI-ORCH-027 | MEDIUM | Compose `version: '3.8'` is obsolete metadata and can hide assumptions about runtime feature support |
| KAI-ORCH-028 | MEDIUM | Shared defaults do not enforce a non-root user, read-only root filesystem or dropped Linux capabilities |
| KAI-ORCH-029 | MEDIUM | Services that spawn child processes lack a deployment-wide init/reaping policy |
| KAI-ORCH-030 | MEDIUM | No logging-driver rotation or central retention policy protects disk capacity and audit continuity |
| KAI-ORCH-031 | MEDIUM | No explicit stop-grace periods coordinate in-flight writes, queues, model calls and index flushes |
| KAI-ORCH-032 | MEDIUM | Static `/16` addressing grants every container broad reachability to the complete service subnet |
| KAI-ORCH-033 | MEDIUM | Docker health checks depend on Python/urllib availability inside each image rather than a uniform probe contract |
| KAI-ORCH-034 | MEDIUM | Named volumes are shared by logical role rather than immutable generation or service ownership |
| KAI-ORCH-035 | MEDIUM | Deployment files contain extensive operational comments/workarounds but no machine-validated architecture manifest |

---

## Critical orchestration findings

### KAI-ORCH-001 — CRITICAL — Direct host exposure is the default
**Issue:** Full/minimal Compose publish ports for Tool Gate, memU, Executor, Agentic, Dashboard, Supervisor, Verifier, Fusion, introspection, sensors, finance, browser, monitoring, notifications and other services. Most of those applications have no incoming authentication.  
**Risk:** Network reachability bypasses the intended gateway and trust architecture; a single host/LAN exposure grants direct control-plane access.  
**Recommendation:** publish only one strongly authenticated edge, keep all privileged services internal and apply host firewall/loopback binding by default.  
**Status:** OPEN — immediate remediation required

### KAI-ORCH-002 — CRITICAL — Flat network trust collapse
**Issue:** All services join `sovereign-net`. No separate data, execution, perception, model, finance or administration zones and no egress restrictions exist.  
**Risk:** Compromise of any service—especially Executor, Browser, Dashboard or feed services—provides network access to every other control/data service.  
**Recommendation:** segment by trust function, allow only explicit one-way flows and route external egress through controlled proxies.  
**Status:** OPEN — immediate remediation required

### KAI-ORCH-003 — CRITICAL — Default WORK mode
**Issue:** Full Compose sets Tool Gate `MODE: "WORK"`. Dashboard frontend also attempts WORK mode on first load.  
**Risk:** The most execution-capable mode is the deployment default rather than an explicit authenticated transition after readiness.  
**Recommendation:** default to a locked/restricted mode and require verified operator activation tied to a deployment revision.  
**Status:** OPEN — immediate remediation required

### KAI-ORCH-004 — CRITICAL — Known database fallback password
**Issue:** Postgres and PG connection strings use `${DB_PASSWORD:-localdev}`.  
**Risk:** Missing secret configuration silently starts a predictable credential used by multiple services.  
**Recommendation:** fail startup when the secret is absent; mount a rotated secret and use separate least-privilege database users.  
**Status:** OPEN — immediate remediation required

### KAI-ORCH-005 — CRITICAL — Shared mutable vector index
**Issue:** memU Core and memU Introspection mount `turbovec_data` and point at the same `memories.tv`, while each process owns an independent in-memory index and write path.  
**Risk:** Concurrent writes can corrupt the index; either service can continue serving stale vectors inconsistent with Postgres.  
**Recommendation:** assign single-writer ownership and publish immutable versioned snapshots to readers.  
**Status:** OPEN — immediate remediation required

---

## High-severity orchestration findings

### KAI-ORCH-006 — HIGH — No authenticated internal transport
Service URLs use `http://service:port`; DNS names are treated as identity and no mTLS, request signature or network policy authenticates most calls.

### KAI-ORCH-007 — HIGH — HTTP reachability masquerades as readiness
Compose probes only whether `/health` returns success. Many audited health handlers return `ok` in stub, stale, no-token, no-model or failed-dependency states.

### KAI-ORCH-008 — HIGH — Incomplete dependency conditions
Many `depends_on` entries use list syntax and start dependants before the dependency health check passes.

### KAI-ORCH-009 — HIGH — Readiness tests deliberately bypassed
memu-graph sets `COGNEE_SKIP_CONNECTION_TEST=true`, while its own health does not initialise Cognee or verify Ollama/embedding capability.

### KAI-ORCH-010 — HIGH — Graph mutation default-on without access control
memU sets `FF_GRAPH_INGEST=true`; memu-graph explicitly sets `ENABLE_BACKEND_ACCESS_CONTROL=false` and is host-published.

### KAI-ORCH-011 — HIGH — Financial context default-on
Agentic sets `FF_FINANCIAL_CONTEXT=true` and reads Financial Awareness into privileged prompts despite the open service/principal model.

### KAI-ORCH-012 — HIGH — Secret proliferation through environment
Database credentials appear inside complete PG URIs passed to multiple containers, expanding exposure through process environment, diagnostics and child processes.

### KAI-ORCH-013 — HIGH — Unauthenticated Redis authority
Redis has no password/TLS/user ACL in Compose while storing cross-service coordination and sensitive state.

### KAI-ORCH-014 — HIGH — Static IP identity brittleness
Hard-coded addresses couple security/monitoring to deployment order and can collide when alternate stacks or changed services share the subnet.

### KAI-ORCH-015 — HIGH — Divergent deployment definitions
Minimal and full Compose use different services, ports, environment flags, IPs and dependencies. Testing one does not validate the other.

### KAI-ORCH-016 — HIGH — Concurrent-stack conflicts
Both files bind many identical host ports and use the same subnet/service names, so accidental simultaneous deployment causes partial startup and unpredictable routing.

### KAI-ORCH-017 — HIGH — No dangerous-service profiles
Executor, browser, finance, introspection, recovery and surveillance services start as ordinary fleet members rather than opt-in profiles.

### KAI-ORCH-018 — HIGH — Restart-loop amplification
`restart: unless-stopped` applies broadly. A bad secret, broken Dockerfile, missing model or invalid index can produce continuous rebuild/restart/health traffic and repeated side effects.

### KAI-ORCH-019 — HIGH — Mutable supply-chain identities
Postgres/Redis/base images and Ollama model tags are referenced by tags, not verified digests or signed manifests.

### KAI-ORCH-020 — HIGH — Pull completion is not capability proof
`ollama-pull` completion satisfies dependencies even when the selected model lacks embeddings, wrong dimensions or incompatible runtime behaviour.

### KAI-ORCH-021 — HIGH — Fixed startup timing
Health `start_period` values are static seconds and do not reflect CPU-only model load, graph extension installation, migration or large-index recovery.

### KAI-ORCH-022 — HIGH — Duplicate active schedulers
Every replica can start refresh loops, Supervisor sweeps, memory compression, introspection, polling and delivery workers without leader election.

### KAI-ORCH-023 — HIGH — No global compute/workload governor
Per-service limits do not coordinate concurrent LLM inference, embeddings, browser work, graph cognification, IMAP/RSS polls and subprocess execution.

### KAI-ORCH-024 — HIGH — Competing recovery authorities
Docker restart policy, Supervisor `/recover`, service-local recovery and health checks can race, reset containment or repeat non-idempotent recovery actions.

### KAI-ORCH-025 — HIGH — Persistence is not a verified recovery system
Named volumes are present, but no encryption, snapshot cadence, content integrity, restore drill or generation relationship is enforced by Compose.

### KAI-ORCH-026 — HIGH — Fleet inventory mismatch
Supervisor, Dashboard, Metrics Gateway, health scripts and Compose enumerate different subsets and service identities; unmonitored services can remain exposed and “healthy” fleet counts can be incomplete.

---

## Medium-severity orchestration findings

### KAI-ORCH-027 — MEDIUM — Obsolete Compose version metadata
Modern Compose ignores/deprecates the top-level version, while contributors may assume v3.8 semantics and Swarm-only `deploy` behaviour.

### KAI-ORCH-028 — MEDIUM — Incomplete shared hardening
`no-new-privileges` and resource hints are applied, but defaults do not enforce user, `read_only`, `cap_drop: ALL`, tmpfs or writable-path allowlists.

### KAI-ORCH-029 — MEDIUM — Missing init/process reaping
Subprocess-heavy services do not uniformly set `init: true`, increasing orphan/zombie handling risk.

### KAI-ORCH-030 — MEDIUM — No log rotation contract
Compose does not define bounded logging options or central storage; verbose service errors can exhaust disk and overwrite evidence.

### KAI-ORCH-031 — MEDIUM — No coordinated graceful stop
Default stop behaviour does not give explicit time for queue drains, ledger fsync, vector/index flushes or child-process termination.

### KAI-ORCH-032 — MEDIUM — Oversized flat subnet
A `/16` bridge is unnecessary for the fleet and broadens discoverability/reachability.

### KAI-ORCH-033 — MEDIUM — Inconsistent probe implementation
Each image must contain working Python/urllib and a compatible health route; probe code is not a versioned shared readiness schema.

### KAI-ORCH-034 — MEDIUM — Shared volume ownership ambiguity
Volumes such as `soul_data` and `turbovec_data` are mounted by multiple writers without immutable generation or ownership rules.

### KAI-ORCH-035 — MEDIUM — No machine-readable architecture manifest
Important compatibility assumptions live in comments and duplicated environment variables. No validated manifest ties service identity, port, model, capability, trust zone, data class and required health contract.

---

## Batch totals

- Findings: **35**
- Critical: **5**
- High: **21**
- Medium: **9**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,289**
- Critical: **209**
- High: **1,147**
- Medium: **930**
- Low: **3**

## Files and evidence used

Current `docker-compose.full.yml`, `docker-compose.minimal.yml`, service Dockerfiles/health routes, Supervisor/Dashboard/Metrics inventory code and all confirmed component audit batches.
