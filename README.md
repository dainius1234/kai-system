


## 📦 Repo Structure
orchestrator/       # Final risk authority before execution
supervisor/         # Watchdog and circuit-breaker control loop
fusion-engine/      # Multi-signal consensus and conviction gating
verifier/           # Fact-checking and signal cross-validation
executor/           # Execution bridge and order-routing stubs
dashboard/          # Operator console (Dash)
memu-core/          # Memory/compression and operator state helpers
tool-gate/          # Tool access policy and local gatekeeping
langgraph/          # Graph/runtime app integration layer
data/               # Seed datasets and local advisor inputs
scripts/            # Operational scripts and validation checks
docs/               # Implementation plans and hardening runbooks

---

## 🚀 Roadmap

### Phase 0 – Core Skeleton  
- Repo scaffold, EventBus, config loader, Docker + CI.  
- Kraken adapter + BookBuilder.  
- Basic Dash UI with kill-switch.  

### Phase 1 – Microstructure Battlefield  
- Multi-venue L2 heatmaps.  
- Iceberg/spoof detection.  
- Stop clusters + BOS/CHOCH.  
- Persistent storage for replay.  

---

## 🧠 Sovereign AI Minimal Core Stack

The latter half of this repository implements a completely separate project: **Sovereign AI (Local-Only)**, a self-sovereign, air-gapped personal intelligence platform.  The documentation below now matches the stack actually built in the previous iteration of development; nothing has been left as a stale placeholder – every listed service can be started, tested and has accompanying unit/integration scripts.

A lightweight development stack is provided by `docker-compose.minimal.yml`, which starts the six essential services:

1. `postgres` – immutable ledger and vector store
2. `redis` – short-term memory spool
3. `tool-gate` – execution choke point with human co-sign
4. `memu-core` – in-memory memory & routing engine
5. `heartbeat` – system pulse and auto-sleep controller
6. `dashboard` – health UI and go/no-go report

When you graduate to the complete prototype, `docker-compose.full.yml` layers on additional stubs and placeholders: `langgraph`, `executor`, `perception/audio`, `perception/camera`, `kai-advisor`, `tts-service`, `avatar-service` (plus later sandbox services). These extra containers currently just expose health endpoints and simple behaviours but they bring the full network topology into play.  The set of LLM specialist services backing LangGraph and Kai-Advisor is still to be finalised – we expect at least three different models (e.g. DeepSeek‑V4, Kimi‑2.5, Qwen‑VL or equivalents) but the exact vendors/configuration will be decided later.

Run the stack:

```bash
# build images (includes audio/camera in full stack)

docker compose -f docker-compose.minimal.yml build

# optionally initialise the database for pgvector persistence
make init-memu-db

# bring the core up

docker compose -f docker-compose.minimal.yml up -d

# validate that services are alive

# also probes executor, langgraph, audio, camera, kai-advisor if they are running
python3 scripts/smoke_core.py

# exercise unit tests across the core services (memu-core, dashboard, audio, camera, executor, langgraph, kai-advisor)
make test-core

> **Note:** In restricted environments (e.g. GitHub Codespaces) Docker containers may
not be reachable on `localhost` due to networking limitations.  If the smoke
script reports connection failures, try running the same commands from within a
container or on a machine where the ports are exposed.

```

### Vector store configuration

By default the memory core keeps episodes in memory. To enable
PostgreSQL/pgvector persistence, set the following environment variables
before starting the stack or running tests:

```bash
export VECTOR_STORE=postgres
export PG_URI=postgresql://keeper:localdev@localhost:5432/sovereign
```

The helper script `scripts/init_memu_db.py` will create the necessary
extension and table.  The unit test `scripts/test_memu_pgvector.py` will
execute when `PG_URI` is defined and silently skip otherwise.

Stop the stack with `docker compose -f docker-compose.minimal.yml down`.

Once the services are running you can open a browser to `http://localhost:8080/ui` to view the
simple HTML/JS dashboard stub. It polls the core services every 5 seconds and
includes a placeholder "Toggle Gate Mode" button.

When you're ready to bring up the **complete** stack including language graph,
executor, perception and output services, use the `docker-compose.full.yml` file:

```bash
# build everything and start

docker compose -f docker-compose.full.yml build

docker compose -f docker-compose.full.yml up -d
```

This full composition is primarily useful for later development; the extra
services initially act as stubs but they establish the networking and
configuration for future expansion.  After the containers are up you can run
an end-to-end smoke/integration script:

```bash
python3 scripts/test_core_integration.py
```

The smoke test script polls each service endpoint and prints the health status.  These components form the hardened foundation; additional features and scripts are developed on top of them as the system grows.

For further architecture details, see `docs/sovereign_ai_spec.md` and the Phase‑1 patch set in `docs/phase1_patch_set.md`.

### Phase 2 – Flow & Arb Engines  
- CVD + delta footprint + absorption.  
- Cross-venue divergence (Kraken, Coinbase, Bitstamp).  
- Spot↔Spot & triangular arbitrage engines.  
- Trade logging → Accounting Squad.  

### Phase 3 – Institutional Intelligence  
- Accumulation/distribution detector.  
- Funding & OI dashboard (multi-venue).  
- ETF flow tracker (BTC, Gold, SPY/QQQ).  
- Cross-asset correlations (BTC, ETH, Gold, DXY, SPX).  
- Macro calendar overlay (auto-throttle).  

### Phase 4 – Playbook & Autonomy  
- Strategy toggle panel.  
- Alt rotation detector (majors → mid-caps → small-caps).  
- Replay mode.  
- Verifier + LLM advisor integration.  

### Phase 5 – Institutional Polish  
- Liquidity shock index.  
- Whale inflows/outflows (on-chain).  
- Options skew (Deribit, CME).  
- Ubuntu desktop launcher, operator presets.  

---

## 🔒 Design Principles

- 100% event-driven & modular – no monolith rewrites.  
- Offline-capable, sovereign, GDPR-compliant.  
- Backtest/live parity – same engine.  
- Extensible: drop in new squad or venue adapter.  
- Autonomous with safeguards: kill-switch, circuit breakers, verifier.  
- Institutional discipline: ledger, attribution, tax-ready.  

---

⚠️ **Disclaimer**  
For research, backtesting, and paper trading only.  
Live trading requires explicit configuration and operator consent.  
Derivatives trading may be restricted in your jurisdiction. Use responsibly.

---

## Sovereign Implementation Planning Docs
- `docs/first_implementation_plan.md` — step-by-step first implementation runbook (commands, expected outputs, failure conditions)
- `docs/phase1_patch_set.md` — concrete Phase-1 patch set aligned to current repo layout
- `docs/production_hardening_plan.md` — production-grade hardening plan with owners and acceptance criteria


## Kai Control Offline Triple-Recovery
- `scripts/kai_control.py` provides a standalone local keeper console (USB primary, USB backup, paper restore).
- Build binary: `make build-kai-control` (PyInstaller one-file output).
- Local self-check: `make kai-control-selftest`.
- Host egress policy helper: `scripts/enforce_egress.sh`.
- Run `kai-control` as normal user (no sudo).
- Monthly drill helper: `scripts/kai-drill.sh` (cron suggested: `0 0 1 * *`).
- HMAC migration readiness advisor: `make hmac-migration-advice` (decides when to leave shared-secret auth).
- HMAC prepare-now hardening: `TOOL_GATE_DUAL_SIGN=true` and then `INTERSERVICE_HMAC_STRICT_KEY_ID=true` after overlap stabilizes.


## Self-Employment Advisor Mode (Offline, UK-focused)
- Preloaded folders: `data/self-emp/{Accounting,Legal,Coding,Engineering,Social}`.
- Skill map: `data/self-emp/skill_map.yml`.
- Advisor rules use local thresholds: `MTD_START`, `VAT_THRESHOLD`, `MILEAGE_RATE`.
- Kai Control has **Advisor Mode** button (`Дайниус, что на уме?`) for strategic suggestions from local income/expense logs.
- Run `kai-control` as normal user (no sudo).
