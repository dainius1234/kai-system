# Compose Drift Analysis

**Generated:** 2026-07-21
**Files compared:**
- `docker-compose.minimal.yml` — minimal dev stack (13 services)
- `docker-compose.sovereign.yml` — hardened self-hosted profile (19 services)
- `docker-compose.full.yml` — full feature stack (33 services)

---

## 1. Service Matrix

| Service | minimal | sovereign | full |
|---|:---:|:---:|:---:|
| postgres | ✓ | ✓ | ✓ |
| redis | ✓ | ✓ | ✓ |
| tool-gate | ✓ | ✓ | ✓ |
| memu-core | ✓ | ✓ | ✓ |
| memu-core-introspect | ✓ | ✓ | ✓ |
| heartbeat | ✓ | ✓ | ✓ |
| dashboard | ✓ | ✓ | ✓ |
| agentic | ✓ | ✓ | ✓ |
| agentic-introspect | — | ✓ | ✓ |
| executor | — | ✓ | ✓ |
| camera-service | — | ✓ | ✓ |
| audio-service | — | ✓ | ✓ |
| ollama | ✓ | — | ✓ |
| ollama-pull | ✓ | — | ✓ |
| wake-service | ✓ | — | ✓ |
| supervisor | ✓ | — | ✓ |
| verifier | ✓ | — | ✓ |
| tailscale | — | ✓ | — |
| vault (profile: dev) | — | ✓ | — |
| vault-rotator (profile: dev) | — | ✓ | — |
| prometheus | — | ✓ | — |
| alertmanager | — | ✓ | — |
| grafana | — | ✓ | — |
| perception-telegram | — | ✓ | — |
| fusion-engine | — | — | ✓ |
| memory-compressor | — | — | ✓ |
| memu-graph | — | — | ✓ |
| letta-agent | — | — | ✓ |
| financial-awareness | — | — | ✓ |
| ledger-worker | — | — | ✓ |
| metrics-gateway | — | — | ✓ |
| kai-advisor | — | — | ✓ |
| tts-service | — | — | ✓ |
| avatar-service | — | — | ✓ |
| screen-capture | — | — | ✓ |
| backup-service | — | — | ✓ |
| calendar-sync | — | — | ✓ |
| telegram-bot | — | — | ✓ |
| workspace-manager | — | — | ✓ |
| parakeet-server (profile: parakeet) | — | — | ✓ |

---

## 2. Critical Divergences

### D1 — postgres image: pgvector vs plain postgres

| File | Image |
|---|---|
| minimal, full | `pgvector/pgvector:pg15` |
| sovereign | `postgres:15-alpine` |

Sovereign uses plain postgres without pgvector. Sovereign sets `VECTOR_STORE: postgres` for memu-core, implying vector queries against postgres — but pgvector extension is absent.

### D2 — memu-core VECTOR_STORE and database name

| File | VECTOR_STORE | DB connection key | DB name |
|---|---|---|---|
| minimal, full | `turbovec` | `PG_URI` | `sovereign` |
| sovereign | `postgres` | `DATABASE_URL` | `memu_db` |

Three differences at once: the store type, the env var name, and the database name.

### D3 — tool-gate MODE

| File | MODE |
|---|---|
| minimal, full | `"WORK"` |
| sovereign | `"PUB"` |

Behavioral divergence — PUB vs WORK gating applies different tool execution rules.

### D4 — tool-gate auth mechanism (three different approaches)

| File | Auth |
|---|---|
| minimal | `HMAC_ALLOW_DEV_SECRET: "true"` (dev bypass) |
| sovereign | `TRUSTED_TOKENS_PATH: /config/trusted_tokens.txt` (file-based tokens) |
| full | `INTERSERVICE_HMAC_SECRET` via Docker secrets |

### D5 — agentic service is a different application in sovereign

Sovereign's agentic has no OLLAMA vars, no LLM timeout, no wake routing. Instead it carries self-employment accounting vars (`MTD_START`, `VAT_THRESHOLD`, `MILEAGE_RATE`, `SELF_EMP_ROOT`). This is a different use case from the LLM orchestrator in minimal/full.

### D6 — agentic OLLAMA_MODEL: hardcoded in full, parameterized in minimal

| File | Value |
|---|---|
| minimal | `"${OLLAMA_MODEL:-qwen2.5:0.5b}"` (overridable) |
| full | `qwen2.5:0.5b` (literal, not overridable via env) |

### D7 — IP address schema

Only three services have consistent IPs across all files that include them:
- `postgres` → 172.20.0.2
- `tool-gate` → 172.20.0.3
- `memu-core` → 172.20.0.5

Everything else differs. The three files cannot be combined or merged at the networking layer without conflicts.

### D8 — executor is fundamentally different between sovereign and full

| Attribute | sovereign | full |
|---|---|---|
| runtime | gvisor | standard |
| restart policy | on-failure (max 5) | unless-stopped |
| healthcheck path | `/alive` | `/health` |
| healthcheck tool | `wget` | `python urllib` |
| Vault integration | present | absent |

### D9 — ollama-pull pulls different model sets

| File | Models pulled |
|---|---|
| minimal | `${OLLAMA_MODEL:-qwen2.5:0.5b}` |
| full | `${OLLAMA_MODEL:-qwen2.5:0.5b}` AND `${EMBEDDING_OLLAMA_MODEL:-all-minilm}` |

Minimal is stale — it does not pull the embedding model that full added.

### D10 — agentic depends_on conditions (startup ordering guarantee)

| File | depends_on condition for memu-core + redis |
|---|---|
| minimal | `service_healthy` |
| full | `service_started` |

Minimal waits for health; full does not.

---

## 3. Copy-Pasted Blocks (Duplication to Reduce)

### A. `x-service-defaults` anchor — identical in minimal and full

Both files have the same `&defaults` YAML anchor (restart policy, resource limits, `no-new-privileges`). Sovereign uses a different anchor (`&service_defaults`) with additional hardening (`user`, `read_only`, `cap_drop`, `tmpfs`).

### B. Network definition — identical in minimal and full

Same `sovereign-net` bridge with subnet `172.20.0.0/16`. Sovereign adds `internal: true` and explicit gateway.

### C. Python urllib healthcheck pattern — ~25 verbatim copies

Every custom-built service in minimal and full uses this identical template (only port number varies):
```yaml
healthcheck:
  test: ["CMD-SHELL", "python -c \"import urllib.request; urllib.request.urlopen('http://localhost:PORT/health')\""]
  interval: 30s
  timeout: 5s
  retries: 3
  start_period: 10s
```
Sovereign uses `wget -qO- ... || exit 1` instead.

### D. wake-service — byte-for-byte identical in minimal and full

All 5 env vars, healthcheck, port, and Dockerfile path are identical. The only differences are the network IP and `container_name` (minimal only).

### E. verifier — near-verbatim in minimal and full

Same env, healthcheck, port, Dockerfile, and network IP.

### F. postgres environment block — verbatim in minimal and full

Same 3 env vars, volume mount, and network IP.

### G. redis healthcheck — identical in minimal and full; absent in sovereign

```yaml
healthcheck:
  test: ["CMD", "redis-cli", "ping"]
  interval: 10s
  timeout: 5s
  retries: 5
```

---

## 4. Other Inconsistencies

| # | Description |
|---|---|
| I1 | `container_name` missing from all full services; inconsistently present in minimal and sovereign |
| I2 | Build format: minimal/full use `{context: ., dockerfile: service/Dockerfile}`; sovereign uses shorthand `build: ./service-name` |
| I3 | postgres volume `:rw` mode suffix present in minimal/full, absent in sovereign |
| I4 | postgres healthcheck `start_period: 10s` present in minimal/full, absent in sovereign; interval 10s vs 30s |
| I5 | `agentic-introspect` healthcheck format differs: sovereign uses CMD list form, full uses CMD-SHELL with quoting |
| I6 | `heartbeat` in sovereign has 5 extra env vars (`TOOL_GATE_URL`, `EXECUTOR_LOG_PATH`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `AUDIT_REDIS_URL`) absent from minimal/full |
| I7 | `redis` in sovereign has `command: redis-server --appendonly yes --save ""` (AOF on, snapshot off); minimal/full use redis defaults |
| I8 | `dashboard` env vars differ substantially: sovereign adds `LEDGER_URL`, `AGENTIC_INTROSPECT_URL`, `EXECUTOR_URL`, `AUDIT_REDIS_URL`; minimal/full have `MEMU_INTROSPECT_URL`, `SUPERVISOR_URL`, `WAKE_URL` |
| I9 | camera-service and audio-service: sovereign defines these without ports/env; full adds env, ports, and healthchecks |
| I10 | `perception-telegram` in sovereign uses inline `pip install` in the command field — no image build |
| I11 | `supervisor` env keys differ between minimal (`SUPERVISOR_CHECK_INTERVAL: "15"`) and full (`SWEEP_INTERVAL: "30"`) — same concept, different keys and values |

---

## 5. What to Extract to a Base File

These blocks are strong candidates for a `docker-compose.base.yml` (using YAML anchors or `extends:`):

| Candidate | Files | Notes |
|---|---|---|
| `x-service-defaults` anchor | minimal + full | Byte-for-byte identical |
| `networks:` block | minimal + full | Identical; sovereign adds `internal: true` as override |
| `postgres` service | minimal + full | Same env, same healthcheck template; sovereign overrides `image` |
| `redis` service | minimal + full | Same image, same healthcheck; sovereign overrides with `command` |
| `heartbeat` core env (3 vars) | all three | `MEMU_INTROSPECT_URL`, `CHECK_INTERVAL: 60`, `ALERT_WINDOW: 300` are identical |
| Python urllib healthcheck template | minimal + full | YAML anchor saves ~6 lines per service |
| `wake-service` | minimal + full | Entire service is verbatim except IP |
| `verifier` | minimal + full | Near-verbatim |
| `dashboard` core 4 env vars | all three | `TOOL_GATE_URL`, `MEMU_URL`, `HEARTBEAT_URL`, `LANGGRAPH_URL` identical |
| `agentic` core 3 env vars | all three | `MEMU_URL`, `TOOL_GATE_URL`, `REDIS_URL` identical |

**Not extractable:** `tool-gate` (different auth per file), `executor` (gvisor vs standard), `supervisor` (different env schema), sovereign's `agentic` (different application).

---

## 6. Recommended Next Steps

Ordered by impact:

1. **Fix D1** — Add `pgvector/pgvector:pg15` to sovereign postgres (or remove `VECTOR_STORE: postgres` if TurboVec is intended).
2. **Fix D2** — Align sovereign's memu-core env var name (`DATABASE_URL` → `PG_URI`) and database name (`memu_db` → `sovereign`).
3. **Fix D6** — Parameterize `OLLAMA_MODEL` in full to match minimal.
4. **Fix D9** — Add embedding model pull to minimal's `ollama-pull` service.
5. **Fix D10** — Align `depends_on` conditions (both to `service_healthy` or both to `service_started`).
6. **Extract base** — Once D1–D5 divergences are resolved or documented as intentional, extract the shared blocks listed in §5 to reduce drift surface.

Items D3, D4, D5, D7, D8 are **intentional architectural differences** between profiles (not bugs) — document them as design decisions before changing.
