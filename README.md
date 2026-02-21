# Regal Industries – Institutional-Grade Autonomous Market Maker

**Regal Industries** is the institutional-grade trading core of the Kai System.  
It is a sovereign, modular, event-driven market-making and arbitrage engine designed to rival professional desks.  
No retail “indicator soup” – pure liquidity, flow, and macro intelligence.  
It is built to be **self-sufficient, autonomous, and sovereign AI-powered infrastructure**.

---

## 🔧 Core Components

### 🏦 Trading Squads
- **Market Maker Core** – adaptive spreads, inventory control, rebate capture.  
- **Liquidity Grab Hunter** – fades stop runs and fake breakouts.  
- **Iceberg Ride** – detects absorption, trades with institutions.  
- **Funding/OI Divergence** – exploits leverage imbalances.  
- **Session Expansion** – trades Asia/EU/US flow patterns.  
- **Arbitrage Engines** – spot ↔ spot, triangular, spot vs perp, ETF vs BTC.  

### ⚙️ Engineering Squad
- Order book tools: multi-venue L2 heatmaps, iceberg/spoof detection.  
- Flow analytics: CVD, delta footprint, absorption/exhaustion.  
- Pattern recognition: stop clusters, BOS/CHOCH, FVG, OB.  
- Institutional metrics: OI, funding, ETF flows, basis spreads.  
- Simulation: fill probability models, spread efficiency tests.  
- System: replay, DuckDB/Parquet storage, latency monitors.  

### 📒 Accounting Squad
- Trade ledger: all fills, hedges, arb legs, fees.  
- PnL attribution: spread capture, rebates, taker fees, funding, inventory.  
- Equity & risk book: leverage ratios, margin stress, equity curve.  
- Tax/compliance: FIFO/LIFO, annual reports, CSV/Excel exports.  
- Treasury functions: capital allocation, stablecoin/fiat tracking.  

### 🛡️ Risk & Governance
- **Orchestrator** – final authority before execution, risk checks.  
- **Supervisor** – watchdog, circuit breakers, auto-restart adapters.  
- **Junior (Self-Healer)** – repairs state, resyncs order books, crash recovery.  
- **Verifier (Fact Checker)** – cross-checks signals against raw tape.  
- **Fusion Engine** – consensus of multiple squads + optional LLM advisor.  

### 🖥️ Operator Console (Dash App)
- Chart with multi-venue liquidity heatmaps, iceberg markers, BOS/CHOCH overlays.  
- Flow panels: CVD, VPIN, delta footprint.  
- Arbitrage dashboard: cross-venue spreads, triangular arb cycles.  
- Institutional panel: OI, funding, basis, ETF flows, macro events.  
- Accounting panel: ledger, equity curve, tax reports.  
- Replay mode: study past engineered moves.  
- Kill-switch & hotkeys for safety.  

---

## 📦 Repo Structure
core/         # Strategy engines, risk manager
engineering/  # Orderbook, flow, pattern, institutional tools
accounting/   # Ledger, pnl, tax, treasury
data/         # Venue adapters, book builder
arb/          # Arbitrage engines
risk/         # Risk guards, supervisor, self-healing
storage/      # DuckDB, replay, compression
ui/           # Dash operator console
agents/       # LLM advisors + verifier
scripts/      # Run scripts, Docker, launchers
tests/        # Unit + integration tests

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
