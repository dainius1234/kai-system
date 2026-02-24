# Copilot Instructions for kai-system

This repository implements **Sovereign AI** — a self-sovereign, air-gapped personal intelligence platform built primarily in Python. It orchestrates a set of microservices that run as Docker containers and communicate via HTTP.

## Repository Structure

- `orchestrator/` — final risk authority before execution
- `supervisor/` — watchdog and circuit-breaker control loop
- `fusion-engine/` — multi-signal consensus and conviction gating
- `verifier/` — fact-checking and signal cross-validation
- `executor/` — execution bridge and order-routing stubs
- `dashboard/` — operator console (Dash/Flask)
- `memu-core/` — memory/compression and operator state helpers
- `tool-gate/` — tool access policy and local gatekeeping
- `langgraph/` — graph/runtime app integration layer
- `kai-advisor/` — self-employment advisor (offline, UK-focused)
- `perception/` — audio and camera capture services
- `heartbeat/` — system pulse and auto-sleep controller
- `scripts/` — operational scripts and validation checks
- `data/` — seed datasets and local advisor inputs
- `docs/` — implementation plans and hardening runbooks
- `common/` — shared utilities and helpers
- `security/` — HMAC/auth hardening helpers
- `sandboxes/` — ephemeral sandbox environments

## Development Flow

### Build

```bash
# Build the minimal stack (core 6 services)
docker compose -f docker-compose.minimal.yml build

# Build the full stack
docker compose -f docker-compose.full.yml build
```

### Run

```bash
# Start the minimal sovereign AI core stack
make core-up          # docker-compose.minimal.yml up -d --build

# Stop the core stack
make core-down

# Start the full stack
make full-up
make full-down
```

### Test

```bash
# Run all core unit tests
make test-core

# Run individual service tests
make test-phase-b-memu       # memu-core unit tests
make test-dashboard-ui       # dashboard UI tests
make test-audio              # audio service smoke test
make test-camera             # camera service smoke test
make test-executor           # executor service smoke test
make test-langgraph          # langgraph service smoke test
make test-kai-advisor        # kai-advisor unit tests
make test-tts                # TTS service smoke test
make test-avatar             # avatar service smoke test

# Integration / smoke tests (requires running stack)
make core-smoke              # python3 scripts/smoke_core.py
make test-integration        # python3 scripts/test_core_integration.py

# Full merge-gate check (all validation steps)
make merge-gate
```

### Lint / Static Analysis

```bash
# Syntax check key service entry points
make go_no_go

# Flake8 (CI runs this automatically)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

### Validate Before Committing

Run `make go_no_go` before committing changes to Python service entry points to catch syntax errors early.

## Coding Standards

1. **Python version**: target Python 3.11 (used in the core CI workflow). The linting workflow currently targets 3.10; keep both workflows in sync when upgrading.
2. **Line length**: max 127 characters (flake8 configured).
3. **Imports**: follow PEP 8; stdlib first, then third-party, then local.
4. **Service entry points**: each service exposes at least a `/health` HTTP endpoint returning `{"status": "ok"}`.
5. **Tests**: place unit tests in `scripts/test_<service>.py`; service-level tests in `<service>/test_<service>.py`. Use `pytest` or plain `unittest`.
6. **Secrets**: never commit real credentials; use `.env` files (see `.env.example`) and environment variables. The `.gitignore` excludes `.env`.
7. **Docker**: each service has its own `Dockerfile`. Multi-service compositions use `docker-compose.minimal.yml` (core 6) or `docker-compose.full.yml` (full stack).
8. **HMAC auth**: inter-service calls use HMAC signing. Set `TOOL_GATE_DUAL_SIGN=true` and later `INTERSERVICE_HMAC_STRICT_KEY_ID=true` after overlap stabilises.

## Key Guidelines

- Always add or update the relevant `requirements.txt` when introducing new Python dependencies.
- When modifying a service's `app.py`, re-run `make go_no_go` to catch syntax errors.
- Prefer small, focused pull requests. Use `make merge-gate` to confirm all checks pass locally before opening a PR.
- Document significant changes in `docs/` and update `README.md` if the public interface changes.
- For PostgreSQL/pgvector persistence, set `VECTOR_STORE=postgres` and `PG_URI=<connection-string>` before running tests.
- Kai Control (`scripts/kai_control.py`) must run without `sudo`. Keep it dependency-light and offline-capable.
