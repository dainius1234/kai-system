# 14-Day Core Stabilization Board

This board is intentionally **non-generic**: it focuses only on the currently wired core services in `docker-compose.sovereign.yml`.

## Scope Lock (Day 0)
- No new services for 14 days.
- Allowed work: reliability, security, observability, rollback drills.
- Run only core stack by default:
  - `docker compose -f docker-compose.sovereign.yml up -d`
- Optional stacks:
  - Ops networking: `docker compose -f docker-compose.sovereign.yml --profile ops up -d`
  - Perception: `docker compose -f docker-compose.sovereign.yml --profile perception up -d`

## Go / No-Go Gate
Promotion requires all four to pass:
1. Core containers healthy.
2. Health endpoints return 200.
3. One rollback drill succeeds.
4. 24h soak has no critical restarts.

## Daily Checklist

### Days 1-2: Build + Boot Determinism
- `docker compose -f docker-compose.sovereign.yml build tool-gate memu-core executor langgraph dashboard heartbeat`
- `docker compose -f docker-compose.sovereign.yml up -d`
- Verify health:
  - `curl -fsS http://localhost:8000/health`
  - `curl -fsS http://localhost:8001/health`
  - `curl -fsS http://localhost:8002/health`
  - `curl -fsS http://localhost:8007/health`
  - `curl -fsS http://localhost:8010/health`
  - `curl -fsS http://localhost:8080/health`

### Days 3-4: Tool-Gate Policy Hardening
- Validate trusted token path mounted and non-empty.
- Send one trusted and one untrusted request to `/gate/request`.
- Reload tokens with `SIGHUP` and re-test.

### Days 5-6: Executor Failure Drills
- Simulate timeout and confirm rollback path executes.
- Simulate malware-scan block path and confirm heartbeat event posting.
- Check `/metrics` error budget counters increase as expected.

### Days 7-8: Memory Service Controls
- Trigger `POST /memory/compress` and verify response + logs.
- Confirm compression schedule guard does not loop.

### Days 9-10: Dashboard + Langgraph Contract Checks
- Verify dashboard dependency URLs resolve.
- Send representative request payload through langgraph route and inspect structured logs.

### Days 11-12: Restart & Recovery
- Restart each core service one by one and verify no cascading failures.
- Validate compose restart policies recover from intentional process kill.

### Days 13-14: 24h Soak + Promotion Decision
- Collect restart counts, health history, and top error logs.
- Decide go/no-go strictly on the gate above.

## Evidence to Keep
- `docker compose ps`
- `docker compose logs --since 24h`
- Health endpoint responses (timestamped)
- One rollback drill transcript
