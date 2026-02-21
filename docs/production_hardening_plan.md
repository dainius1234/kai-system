# Production Hardening Plan (Post Core-Stability)

## Objective
Promote the validated core stack to a professional, production-ready sovereign deployment with auditable controls, robust observability, and deterministic operations.

## Workstreams

## 1) Security Hardening
**Owner:** Security Lead  
**Contributors:** Platform + Core API owners

### Tasks
- Replace in-memory ledger with PostgreSQL-backed hash chain and tamper checks.
- Enforce egress denial at network/policy layer.
- Run all services as non-root, read-only where safe, with cap-drop all.
- Add signed mode transitions for tool-gate (`PUB` ↔ `WORK`).

### Acceptance criteria
- External egress attempts fail by policy.
- Ledger verification endpoint returns valid chain after 24h soak.
- All containers pass least-privilege audit checklist.

---

## 2) Reliability & Lifecycle
**Owner:** Platform Lead  
**Contributors:** SRE

### Tasks
- Add service healthchecks and restart policies.
- Add dependency readiness gates (not only startup order).
- Add graceful shutdown and startup timeout handling.
- Add backup/restore runbooks and recurring snapshot jobs.

### Acceptance criteria
- Core services recover from single container failure automatically.
- Restore drill from backup completes in under target RTO.

---

## 3) Observability
**Owner:** SRE Lead

### Tasks
- Deploy Prometheus/Grafana/Loki/Promtail.
- Instrument all service endpoints with request latencies, error counts.
- Add dashboard SLO panels:
  - Core API availability
  - Tool-gate decision throughput
  - memu-core memory growth

### Acceptance criteria
- P95 latency and error-rate dashboards live.
- Alert rules trigger and route to on-call channel.

---

## 4) API Governance & Contracts
**Owner:** Core API Lead

### Tasks
- Introduce API versioning (`/v1/*`).
- Publish OpenAPI snapshots per service.
- Add schema contract tests in CI.

### Acceptance criteria
- breaking schema changes blocked unless version bumped.
- OpenAPI docs generated in CI artifact bundle.

---

## 5) Data Governance
**Owner:** Data/Memory Lead

### Tasks
- Move memu-core from in-memory to pgvector persistence.
- Add retention policy and memory compression jobs.
- Add migration strategy for schema evolution.

### Acceptance criteria
- memu-core survives restarts without data loss.
- compression jobs run on schedule and preserve retrieval quality thresholds.

---

## 6) Release Engineering
**Owner:** Release Manager

### Tasks
- Add build pipeline (lint, unit tests, contract tests, image scans).
- Sign and tag releases (semantic versioning).
- Define promotion gates: dev → staging → prod.

### Acceptance criteria
- reproducible image build with immutable digest pinning.
- release candidate cannot promote unless all gates pass.

---

## 7) GPU Enablement Plan (Deferred)
**Owner:** ML Platform Lead

### Tasks
- Keep GPU services commented until core SLOs are stable.
- Add runtime switch `GPU_ENABLED` and CPU fallback in all model-serving code.
- Validate CUDA path via canary deployment only.

### Acceptance criteria
- core behavior unchanged with GPU disabled.
- GPU canary passes without regressions before full enablement.

---

## Readiness Gate to Exit Hardening
All workstreams must satisfy acceptance criteria plus:
- 72h stability soak with no sev-1 incidents.
- successful restore drill.
- security sign-off and architecture review sign-off.
