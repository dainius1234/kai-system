# Kai Tech Watch (May 2026)

Monthly review reminder is automated via `.github/workflows/tech-watch-reminder.yml`.

## Adopt

| Tech | Verdict | Date assessed | Why | Re-evaluate |
|---|---|---|---|---|
| GitHub Actions workflows for PM guardrails | Adopt | 2026-04-21 | Native automation for drift prevention and repeatable PM checks | 2026-07-01 |
| Keep a Changelog format | Adopt | 2026-04-21 | Consistent human-readable release history already in use | 2026-10-01 |
| Mermaid for sequence dependency graph | Adopt | 2026-04-21 | Renders directly on GitHub and clarifies locked order | 2026-08-01 |

## Trial

| Tech | Verdict | Date assessed | Why | Re-evaluate |
|---|---|---|---|---|
| qwen2.5:7b (target upgrade from qwen2:0.5b) | Trial | 2026-04-21 | Needed for stronger reasoning once GPU arrives; currently hardware-blocked | 2026-06-01 |
| sentence-transformers for richer agreement scoring | Trial | 2026-04-21 | Improves semantic checks over lexical overlap where resources allow | 2026-06-15 |
| parakeet.cpp (GGML port of NVIDIA NeMo Parakeet, incl. Nemotron-3.5-ASR-streaming-0.6b) | Trial | 2026-06-18 | Verified real (429 stars, MIT, active releases). Correct serving path for Nemotron ASR — Ollama cannot serve NeMo/ASR models, but parakeet-server exposes an OpenAI-compatible HTTP transcription API. CPU-capable, fits audio-service's existing faster-whisper fallback pattern | 2026-08-01 |
| ASI-Evolve (GAIR-NLP/ASI-Evolve) | Trial | 2026-06-18 | Verified real (757 stars, Apache-2.0, active). Cloned and read `utils/llm.py`: LLM client is a plain `openai` SDK wrapper, no tool-calling/exotic params — confirmed Ollama-compatible via OpenAI-compatible `base_url`. Scoped narrowly to bounded, scored optimization loops (e.g. tuning conviction-scoring against logged outcomes), not a Skill Forge replacement. Must set `logging.wandb.enabled: false` before use — defaults to phoning home, violates zero-telemetry principle | 2026-08-01 |
| Cognee (memory controller candidate) | Trial | 2026-06-18 | Verified real and substantial (17.9k stars, Apache-2.0, actively released v1.1.3 same day as assessment). Supports fully local/self-hosted deployment, Kuzu graph DB + vector storage. No corrections needed to existing "Unchanged" assumption | 2026-09-01 |
| Graphiti (memory controller candidate) | Trial | 2026-06-18 | Verified real and substantial (27.6k stars, Apache-2.0, 876 commits). Confirmed local/offline-capable: FalkorDB embedded mode, documented support for Ollama/vLLM/llama.cpp as the LLM backend. No corrections needed | 2026-09-01 |

## Assess

| Tech | Verdict | Date assessed | Why | Re-evaluate |
|---|---|---|---|---|
| MCP surface refactor approach | Assess | 2026-04-21 | Planned parallel scope after J2 sequencing; needs compatibility review | 2026-05-15 |
| Nightly benchmark bot (Bench-Bot concept) | Assess | 2026-04-21 | Potential value, but currently deferred under D12 | 2026-07-01 |
| TurboVec / TurboQuant | Assess | 2026-06-18 | Verified real (11.9k stars, MIT, ICLR-2026-backed), 16x compression at 2-bit. Standalone in-process index with its own file format — no documented pgvector integration. Needs an explicit architectural decision before adoption: store compressed blobs as Postgres bytea+metadata, or replace pgvector similarity search with TurboVec's own index entirely | 2026-08-01 |
| OpenHands (All-Hands-AI, Skill Forge code-gen candidate) | Assess | 2026-06-18 | Verified real and mature (77.7k stars, beta) but architected as a standalone, always-on autonomous coding control plane (Node.js 22.12+, `uv`, optional Docker sandboxing — full filesystem access without it), not a lightweight embeddable call-out as first assumed. Needs full skill-security-gauntlet + irreversible-action review before any adoption decision | 2026-08-01 |
| Letta (formerly MemGPT) | Assess | 2026-06-18 | Verified real (23.4k stars, Apache-2.0) but Ollama/full-offline compatibility unconfirmed — docs emphasize Claude Opus/GPT-5.2 and API-key examples despite "model-agnostic" claim. Needs a follow-up check of Letta's model-providers docs before any decision | 2026-07-15 |
| NVIDIA LocateAnything-3B | Assess | 2026-06-18 | Unverified — HuggingFace page returned HTTP 403 on every fetch attempt (3x across two sessions). No license/runtime data gathered; do not treat as confirmed until a successful fetch | 2026-07-01 |

## Hold

| Tech | Verdict | Date assessed | Why | Re-evaluate |
|---|---|---|---|---|
| 70B-class local models | Hold | 2026-04-21 | Not practical on current hardware budget/power envelope | 2027-01-01 |
| Full-time autonomous sub-agent fleet | Hold | 2026-04-21 | Coordination overhead exceeds current benefit before unlock triggers | 2026-09-01 |
| DeerFlow (bytedance/deer-flow) | Hold | 2026-06-18 | Verified real and mature (71.5k stars) but built on LangGraph/LangChain, which `agentic` already uses directly — adopting it would stack a redundant orchestration layer. Its workspace UI only shows its own internal task/thread state, confirmed no capability to observe or dashboard an external LangGraph app's state, ruling out the "read-only swarm dashboard" role it was proposed for | 2026-10-01 |
| CrewAI / AutoGen | Hold | 2026-06-18 | Already evaluated previously: exist only as `pytest.skip()`-guarded import-smoke-tests in `scripts/agentic_integration_test.py`, never wired into real orchestration. Adopting now would duplicate LangGraph, which is already load-bearing in `agentic` | 2026-10-01 |
| ASI-Arch (GAIR-NLP/ASI-Arch) | Hold | 2026-06-18 | Verified real (1.2k stars, Apache-2.0) but narrowly scoped to discovering novel neural network architectures from scratch — out of scope, since KAI only consumes existing pretrained models via Ollama | 2027-01-01 |
