# First Implementation Plan (Core Sovereign Stack)

## Scope
This plan executes the minimum viable core stack for local-only validation:
- postgres
- tool-gate
- memu-core
- executor
- langgraph
- redis
- dashboard
- heartbeat

Perception/output/GPU-heavy services are out-of-scope for initial go-live.

## Owners
- **Platform Owner**: Compose runtime, container policy, startup sequencing
- **Core API Owner**: tool-gate + memu-core + executor API contracts
- **Orchestration Owner**: langgraph integration and routing behavior
- **Ops Owner**: dashboard visibility, health sweep, runbook logging

## Entry Criteria
- Docker + docker compose available on target host
- `.env` present with non-placeholder values for DB_PASSWORD
- No host ports 8000/8080 occupied

---

## Step 1 — Preflight

### Command
```bash
mkdir -p /var/log/sovereign
docker --version && docker compose version
```

### Expected output
- Docker and compose versions print successfully.

### Failure conditions
- `docker: command not found` → install Docker engine + compose plugin.
- permission denied on Docker socket → add user to docker group or use sudo.

---

## Step 2 — Validate compose config

### Command
```bash
docker compose -f docker-compose.sovereign.yml config > /var/log/sovereign/compose.rendered.yml
```

### Expected output
- Exit code `0`.
- Rendered compose written to `/var/log/sovereign/compose.rendered.yml`.

### Failure conditions
- YAML parse errors → fix indentation or invalid fields.
- Undefined variables → define in `.env`.

---

## Step 3 — Bring up core stack

### Command
```bash
docker compose -f docker-compose.sovereign.yml up -d postgres tool-gate memu-core redis executor langgraph heartbeat dashboard
```

### Expected output
- All listed services transition to `running`.

### Failure conditions
- Any container exits repeatedly → inspect logs and block progression.

---

## Step 4 — Startup ordering checks

### Command
```bash
docker compose -f docker-compose.sovereign.yml ps
```

### Expected output
- `tool-gate` starts only after `postgres`.
- `memu-core` starts after `postgres` and `tool-gate`.
- `executor` starts after `tool-gate` and `memu-core`.

### Failure conditions
- Incorrect startup order or repeated restarts → fix `depends_on` and service readiness assumptions.

---

## Step 5 — Endpoint health sweep

### Command
```bash
set -euo pipefail
for ep in \
  "http://localhost:8000/health" \
  "http://localhost:8080/health"; do
  echo "checking $ep"
  curl -fsS "$ep" | tee -a /var/log/sovereign/health_sweep.log
  echo
 done

# Internal endpoints via docker network
for svc in memu-core executor langgraph heartbeat; do
  docker compose -f docker-compose.sovereign.yml exec -T dashboard \
    sh -lc "wget -qO- http://$svc:$(case $svc in memu-core) echo 8001;; executor) echo 8002;; langgraph) echo 8007;; heartbeat) echo 8010;; esac)/health"
done
```

### Expected output
- All responses return JSON and success status.

### Failure conditions
- `curl`/`wget` non-zero or malformed JSON → fail fast, capture logs.

---

## Step 6 — Functional contract tests

### Command
```bash
# tool-gate
curl -fsS -X POST http://localhost:8000/gate/request \
  -H 'content-type: application/json' \
  -d '{"tool":"shell","params":{"cmd":"echo hi"},"confidence":0.9,"actor_did":"test"}' \
  | tee -a /var/log/sovereign/contract.log

# memu-core route
curl -fsS -X POST http://localhost:8001/route \
  -H 'content-type: application/json' \
  -d '{"query":"status check","session_id":"smoke","timestamp":"now"}' \
  | tee -a /var/log/sovereign/contract.log

# dashboard root snapshot
curl -fsS http://localhost:8080/ | tee -a /var/log/sovereign/dashboard_snapshot.json
```

### Expected output
- tool-gate returns decision JSON with `approved`/`blocked`.
- dashboard returns `alive_nodes`, `ledger_size`, `memory_count`.

### Failure conditions
- 4xx/5xx where success is expected.
- Missing required fields in response payload.

---

## Step 7 — Evidence capture + teardown

### Command
```bash
docker compose -f docker-compose.sovereign.yml logs --no-color > /var/log/sovereign/compose.log
# optional
# docker compose -f docker-compose.sovereign.yml down
```

### Expected output
- Logs captured for audit/debugging.

### Failure conditions
- inability to collect logs indicates runtime instability or permissions issue.

## Exit Criteria
- All core services healthy.
- Dashboard root shows live status and counts.
- Contract test responses match schema expectations.
- Logs stored in `/var/log/sovereign/`.
