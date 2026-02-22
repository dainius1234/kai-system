# Next-Level Reliability Roadmap (Project Lead Notes)

This document explains **why** the latest hardening direction was chosen.

## Why these 4 tracks first
1. **State continuity** (breaker + nonce persistence): without this, restart resets safety context.
2. **Service authenticity** (signed gate requests): token checks alone are not enough for inter-service trust.
3. **Operational truth** (game-day scorecard): if we cannot measure drill quality, stability work is guesswork.
4. **Deployment separation** (compose profiles): dev defaults must never leak into production posture.

## Decision principles
- Prefer deterministic degraded behavior over optimistic retries.
- Prefer deny-by-default when trust assertions are missing.
- Prefer measurable operations over narrative confidence.
- Prefer explicit environment profiles over mutable "remember to change this" steps.

## What to do next
- Add persistent storage for breaker and nonce state in Redis/Vault-backed channel with integrity hash.
- Add key rotation for inter-service HMAC secret and dual-key verification window.
- Expand game-day scorecard with SLO targets (p95 latency, degraded serve %, false-allow/false-block).
- Add strict CI policy: `docker compose --profile dev config` and `docker compose config` must both pass.
