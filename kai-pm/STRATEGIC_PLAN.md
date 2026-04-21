# Sovereign AI Strategic Plan

> Canonical 5-phase roadmap for Kai. Supersedes any prior "11-step sequence" references.
> Full body to be supplied by operator in a follow-up commit. This file is the agreed canonical location.

## Phases (summary)
- Phase 0 — Pre-GPU Hardening (ACTIVE)
- Phase 1 — Local LLM Integration (BLOCKED on GPU)
- Phase 2 — Multi-Specialist Routing (BLOCKED on GPU)
- Phase 3 — Memory & Reflection Hardening (partial)
- Phase 4 — Avatar / Voice / Multimodal (BLOCKED on GPU)
- Phase 5 — Production Hardening & Self-Improvement (BLOCKED on GPU)

## Known correction flags (to be addressed when full plan body lands)
- J1–J7 are DONE, not queued
- RTX 5080 16GB VRAM cannot fit llama3.3:70b at usable quant — plan must reflect 8B/13B class for primary and use 70B only via remote/cloud fallback
- Ollama speculative decoding support is partial / version-dependent — verify before claiming
- P29 placement TBD — confirm phase
- All cited external benchmarks need source links
