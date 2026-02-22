# Next-Level Reliability Roadmap (Project Lead Notes)

This document explains **why** the hardening direction was chosen, plus decision options.

## Why these tracks first
1. **State continuity** (breaker + nonce persistence): restart should not reset safety context.
2. **Service authenticity** (signed gate requests): token checks alone are insufficient for inter-service trust.
3. **Operational truth** (game-day scorecard): stability requires measurable evidence.
4. **Deployment separation** (compose profiles): dev defaults must not leak into production posture.

## Big decision #1: Request signing scheme
- **Chosen now:** HMAC-SHA256 with primary+previous secret (rotation window).
- **Why:** lowest operational overhead and immediate upgrade from unsigned/flat token trust.
- **Alternative:** ed25519 per-service keys.
  - Pros: asymmetric trust and better key isolation.
  - Cons: key distribution and rotation complexity.

## Big decision #2: Safety-state persistence medium
- **Chosen now:** local file persistence (`/tmp`) for breaker/nonce continuity.
- **Why:** no external dependency added; works offline immediately.
- **Alternative:** Redis/Vault-backed state with integrity hash.
  - Pros: multi-instance coherence and stronger tamper model.
  - Cons: introduces dependency coupling and migration complexity.

## Big decision #3: Operational acceptance policy
- **Chosen now:** scorecard with pass-rate + total-duration SLO thresholds.
- **Why:** easy to run repeatedly; provides objective go/no-go signal.
- **Alternative:** full synthetic load + latency percentile budget gating.
  - Pros: closer to production reality.
  - Cons: heavier runtime and infrastructure cost.

## What to do next
- Move safety-state persistence to Redis/Vault-backed channel with integrity hashing.
- Introduce dual-key rotation runbook for `INTERSERVICE_HMAC_SECRET` / `_PREV`.
- Add CI policy that validates both compose modes (`default` and `--profile dev`) in an environment with Docker.
- Add per-service p95 latency and degraded-serve ratio targets to scorecard.
