# Compose Drift Map

> Prep doc for Cleanup Sprint Week 2.2. This is an inventory of what currently differs across `/tmp/workspace/dainius1234/kai-system/docker-compose.minimal.yml`, `/tmp/workspace/dainius1234/kai-system/docker-compose.full.yml`, and `/tmp/workspace/dainius1234/kai-system/docker-compose.sovereign.yml`.

## Current files and roles

| File | Current role | Service count |
|---|---|---|
| `docker-compose.minimal.yml` | smallest runnable dev/core stack | 9 |
| `docker-compose.full.yml` | feature-complete local stack | 27 |
| `docker-compose.sovereign.yml` | hardened / sovereign-oriented stack with vault + monitoring | 17 |

## Biggest drift points

### 1) Minimal stack count drifts from README

- README still describes the minimal stack as **8 services**.
- `docker-compose.minimal.yml` currently includes **9 services** because `wake-service` is part of the file.
- That means the public story and the real compose entry point disagree.

### 2) Core service presence is not aligned

Common to all three files:

- `postgres`
- `redis`
- `tool-gate`
- `memu-core`
- `heartbeat`
- `dashboard`

Present in minimal + full, but not sovereign:

- `supervisor`
- `verifier`
- `wake-service`

Present in full + sovereign, but not minimal:

- `agentic`
- `executor`
- `audio-service`
- `camera-service`

### 3) Shared services use different wiring conventions

Examples:

- `memu-core`
  - minimal/full use `PG_URI`
  - sovereign uses `DATABASE_URL`
- `tool-gate`
  - minimal uses `NONCE_TTL_SECONDS` + `HMAC_ALLOW_DEV_SECRET`
  - full uses `REDIS_URL`
  - sovereign uses `DATABASE_URL`, `AUDIT_REDIS_URL`, `TRUSTED_TOKENS_PATH`
- `dashboard`
  - minimal has no `LANGGRAPH_URL` / `EXECUTOR_URL`
  - full adds both
  - sovereign swaps in `LEDGER_URL` and omits `SUPERVISOR_URL` / `WAKE_URL`

### 4) Healthcheck and build style drift

- minimal/full mostly use `python -c ... urllib.request.urlopen(...)`
- sovereign mostly uses `wget -qO- ...`
- minimal/full mostly use `build: { context: ., dockerfile: ... }`
- sovereign mostly uses `build: ./service`

### 5) Security posture is encoded differently instead of layered

- minimal/full share a lighter `x-service-defaults`
- sovereign adds `read_only`, `tmpfs`, `cap_drop`, `user`, Vault, Tailscale, Prometheus, Grafana
- the hardening is real, but it is duplicated rather than composed from a shared base

### 6) Naming drift leaks into service topology

- full uses `telegram-bot`
- sovereign uses `perception-telegram`
- full includes `ollama`, `orchestrator`, `metrics-gateway`, `backup-service`, `workspace-manager`
- sovereign replaces that emphasis with network/security/monitoring services

## Recommended reconciliation direction

1. Define a **canonical shared core** for:
   - `postgres`
   - `redis`
   - `tool-gate`
   - `memu-core`
   - `heartbeat`
   - `dashboard`

2. Decide the truth for the **minimal stack**:
   - either keep README at 8 and make `wake-service` optional there,
   - or accept 9 as the new truth and update README/docs to match.

3. Normalize shared service wiring:
   - one database env convention
   - one redis env convention
   - one healthcheck style
   - one build syntax

4. Make the files additive:
   - `minimal` = shared core + only essential operator path
   - `full` = minimal + feature services
   - `sovereign` = full or core + hardening/ops overlays, not a partially separate topology

## Suggested PR order

1. Settle the minimal-stack truth (`8` vs `9`, especially `wake-service`).
2. Extract shared core service blocks.
3. Normalize env names for `tool-gate`, `memu-core`, and `dashboard`.
4. Align service naming (`telegram-bot` vs `perception-telegram`) or document why they intentionally differ.
5. Only then decide whether sovereign should be a variant of full or a separate hardened profile.
