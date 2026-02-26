# Sovereign AI — kai-system

Self-sovereign, air-gapped personal intelligence platform.

## 📝 Simple System Overview (For Everyone)

Kai System is like a secure AI brain made of different parts, each with a job:

- **Tool-Gate & Orchestrator:** The main gatekeepers. They check every action for safety and give the final OK before anything important happens.
- **Memu-Core:** The memory. It remembers everything and gives context to help the AI make better decisions.
- **LangGraph, AutoGen, CrewAI, OpenAgents:** The planners and thinkers. They help the AI plan, reason, and work as a team of smart assistants.
- **Executor & Sandboxes:** The doers. They safely run actions that have been approved.
- **Dashboard & Output:** The display and feedback. They show you what’s happening and let you interact.

**How it works:**
1. You (or another service) make a request.
2. Tool-Gate and Orchestrator check it for safety.
3. The planners (LangGraph, etc.) figure out what to do, using Memu-Core for memory.
4. If approved, Executor runs the action in a safe sandbox.
5. Results are saved in Memu-Core and shown on the Dashboard.

**In short:**
Kai is a secure, multi-part AI system. Everything is checked, logged, and controlled for safety and privacy.

---

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


---

## 🧠 Sovereign AI Minimal Core Stack

 **Sovereign AI (Local-Only)**, a self-sovereign, air-gapped personal intelligence platform.  The documentation below now matches the stack actually built in the previous iteration of development; nothing has been left as a stale placeholder – every listed service can be started, tested and has accompanying unit/integration scripts.

A lightweight development stack is provided by `docker-compose.minimal.yml`, which starts the eight core services:

1. `postgres` – immutable ledger and vector store
2. `redis` – session buffer and short-term memory spool
3. `tool-gate` – execution choke point with human co-sign
4. `memu-core` – memory engine (vector search, session buffer, auto-classification)
5. `heartbeat` – system pulse and auto-sleep controller
6. `dashboard` – health UI and go/no-go report
7. `supervisor` – watchdog with circuit breakers and health sweeps
8. `verifier` – fact-checking and signal cross-validation

The full stack (`docker-compose.full.yml`) adds production services: `fusion-engine`, `langgraph`, `executor`, `perception/audio`, `perception/camera`, `kai-advisor`, `tts-service`, `avatar-service`. Three local LLM backends are supported: DeepSeek-V4 (reasoning/code), Kimi-2.5 (general/multimodal), Dolphin (uncensored PUB mode).

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

---

## 🤖 Self-Audit & Feedback

- Run `make self-audit` to:
  - Review recent test, lint, and health check results
  - Summarize system health and recurring issues
  - Propose actionable improvements
  - Log lessons/incidents to memu-core (if running)
  - Output a summary and full audit log to `output/self_audit_log.json`
- This is the first step toward a self-reflective, self-improving AI partner.

---

## 🗣️ Operator Feedback Channel

- Submit feedback, suggestions, or goals directly to the system:
  - `python3 scripts/operator_feedback.py "<your message>"`
- Feedback is logged as a structured event in memu-core (if running), or saved locally for later ingestion.
- This builds a persistent memory of operator guidance, enabling the system to learn and adapt from your input.

---

## 📝 Feedback & Memory Summary

- Run `python3 scripts/feedback_summary.py` to:
  - Retrieve and summarize recent operator feedback, lessons, and system actions from memory
  - Fallback to local logs if memu-core is unavailable
  - Surface actionable insights for both operator and AI review

---

## 🛡️ CI & Automation Coverage (2026)

- **All scripts and Makefile targets** are now enforced in CI via the `merge-gate` target.
- Every operational, utility, and maintenance script is tested or invoked automatically on every PR and push.
- The new `scripts/quality_gate.py` blocks stubs, TODOs, and missing docstrings from merging.
- Linting, unit tests, integration tests, and system health checks are all run in CI.
- Maintenance and rotation scripts (e.g., paper-backup, weekly-key-rotate) are included in the automation pipeline.
- Failures in CI will block merges, ensuring only high-quality, production-ready code is accepted.

### Operator Workflow

- To validate the system locally, run:
  - `make merge-gate` — runs all quality, test, and maintenance checks as in CI
  - `make test-core` — runs all core unit and integration tests
  - `make health-sweep` — probes all service health endpoints
  - `make contract-smoke` — runs contract-level smoke tests
- All scripts in `scripts/` are now subject to automated quality checks and must have docstrings, no stubs, and no TODOs.
- For new scripts or features, ensure they are added to the Makefile and covered by tests or invoked in CI.

---

## 🦾 Kai Supervisor Agent (Prototype)

- Run `python3 scripts/kai_supervisor.py` to:
  - Review recent memory for patterns, recurring issues, and actionable improvements
  - Suggest or draft improvements (e.g., docstrings, stub removal)
  - Log its own actions and suggestions as system_action events in memory
  - (Future) Auto-apply safe changes or request operator approval for higher-impact actions

---

## ⚠️ Agentic Integration Notes (Feb 2026)

- **Namespace Clashes:**
  - Local file `langgraph/config.py` was renamed to avoid shadowing the installed `langgraph` package. Always avoid naming local modules after installed packages.
  - If you need the local config, import it as `kai_langgraph_config.py`.
- **OpenAgents API:**
  - The class `AgentContainer` is not available at the top level. Check OpenAgents documentation for the correct import path or usage.
- **Best Practices:**
  - Never name local files/folders after installed packages.
  - Always check for API changes in fast-evolving agentic frameworks.
  - Log and document all integration issues and fixes in the README for future reference.

---

## 🤖 Agentic Framework Integration (2026)

### Integrated Frameworks
- **LangGraph**: For agent orchestration and stateful multi-actor workflows. Use `from langgraph.graph import StateGraph` for graph construction.
- **AutoGen**: For multi-agent orchestration and self-reflection. Use `from autogen import AssistantAgent, UserProxyAgent`.
- **CrewAI**: For collaborative, role-based agent tasking. Use `from crewai import Crew, Task, Agent`.
- **OpenAgents**: For multi-agent protocol and containerization. Use `from openagents.container.agent_container import AgentContainer` (not top-level import).

### Integration Test
- See `scripts/agentic_integration_test.py` for a working example that exercises all four frameworks in a minimal, side-effect-free way.
- The test verifies:
  - LangGraph node orchestration
  - AutoGen agent instantiation
  - CrewAI crew/task setup
  - OpenAgents agent container creation

### Troubleshooting & Lessons Learned
- **Namespace Clashes**: Never name local files/folders after installed packages (e.g., avoid `langgraph/config.py`). This caused import errors and shadowed the installed package. Local config was renamed.
- **OpenAgents API**: The `AgentContainer` class is not available at the top level. Use `from openagents.container.agent_container import AgentContainer`.
- **API Drift**: Agentic frameworks evolve rapidly. Always check the latest documentation and inspect installed packages if imports fail.
- **Testing**: Always run `python3 scripts/agentic_integration_test.py` after changes to agentic dependencies or imports.
- **Documentation**: Log all integration issues, fixes, and best practices in the README for future maintainers.

### Best Practices
- Use explicit, non-conflicting names for all local modules.
- Document all architectural decisions and integration issues.
- Prefer minimal, side-effect-free integration tests for new frameworks.
- Update requirements and test scripts with every new agentic dependency.
- Review and update this section as frameworks evolve.

---

## 🏛️ Architectural Analysis (2026)

### System Overview
Kai System is a modular, air-gapped, self-sovereign AI platform. It is designed with strict local-only, no-egress security, layered defense, and composable microservices. The architecture is documented in detail in `docs/sovereign_ai_spec.md` (see layered Mermaid diagram and service breakdown).

#### Layered Architecture
- **L0: Hardware Root** — TPM, encrypted storage, direct device access (camera, mic, GPU)
- **L1: Sovereign Core** — Tool Gate (execution choke), Ledger (Postgres+pgvector), Dashboard (operator UI)
- **L2: Intelligence** — Memu-Core (memory), LangGraph (agent orchestration), LLM Pool, Kai Advisor
- **L3: Perception** — Audio, Camera, Screen Capture pipelines
- **L4: Awareness** — Heartbeat, Calendar sync
- **L5: Execution** — Executor, Sandboxes (QGIS, n8n, shell)
- **L6: Output** — TTS, Avatar, Display overlay

#### Agentic Integration Points
- **LangGraph**: Orchestrates agent workflows, routes context and tasks between memory, LLMs, and advisors.
- **AutoGen**: Enables multi-agent, self-reflective reasoning and user proxying.
- **CrewAI**: Coordinates collaborative, role-based agent teams for complex tasks.
- **OpenAgents**: Provides protocol and containerization for multi-agent systems.

#### Security Invariants
- All execution requests pass through Tool Gate (with HMAC co-sign).
- No network egress; all inter-service traffic is on a private bridge network.
- All persistent storage is local and encrypted where possible.
- Sandboxes are isolated (network_mode: none, read-only, dropped caps).

#### Best Practices & Lessons
- **Namespace Hygiene**: Never shadow installed packages with local files (e.g., langgraph/config.py).
- **Explicit Imports**: Use explicit, version-checked imports for agentic frameworks (see Integration section above).
- **API Drift Management**: Regularly audit and test all agentic integrations; document all breakages and fixes.
- **Minimal, Isolated Tests**: Use scripts/agentic_integration_test.py as a template for future framework integrations.
- **Documentation**: All architectural and integration decisions, issues, and fixes are logged in this README and in docs/.

#### References
- See `docs/sovereign_ai_spec.md` for full technical specification, layered diagrams, and service details.
- See `scripts/agentic_integration_test.py` for a working, minimal agentic integration test.

---
