# Kai System — Project Backlog

> Living document. Updated each session. Prioritised by what makes Kai
> more *organic* — able to learn, perceive, reason, and remember — not
> just more features.

**Vision:** As close to organic AI as possible. A system that learns from
every interaction, builds understanding over time, perceives its environment,
and earns the right to act through verified reasoning. Not a chatbot.
Not an agent framework. A sovereign intelligence that grows.

**Hardware constraint:** No local GPU until RTX 5080 arrives. All LLM
backends are stubs. System is designed so GPU arrival = 3 env vars changed.

**Last updated:** 2026-02-25 — session: v7 test coverage + perception wiring

---

## Current State

| Metric | Value |
|---|---|
| Services | 23 (all with /health) |
| Tests | 76 passing |
| Lines of Python | ~10,500 |
| Compose files | 3 (minimal/full/sovereign) |
| Stack actually runs as containers? | **NO — untested** |
| Real LLM wired? | No (stubs) |
| Real persistence? | No (in-memory default) |
| Real input channel? | No (telegram is a dead polling loop) |
| Can Kai learn right now? | Yes — memu-core stores, ranks, decays, but only via API calls in tests |

---

## Priority Tiers

### P0 — Give it a body (stack must run)
*Without this, everything else is fiction.*

- [ ] **Docker compose build** — build all services, fix whatever breaks.
      Every Dockerfile must produce a working image.
- [ ] **Health sweep automation** — `docker compose up` → script hits every
      `/health` → green/red scorecard. Must pass before any PR merges.
- [ ] **Postgres persistence** — switch memu-core default from in-memory to
      postgres+pgvector (compose already has postgres service). Memories
      survive restarts. This is fundamental to "organic" — an organism
      that forgets everything on restart isn't learning.
- [ ] **Redis for shared state** — rate-limit windows, idempotency cache,
      and session state need to survive process restarts. Compose already
      has Redis.

### P1 — Give it senses (perception)
*An organism needs input channels. These all work without GPU.*

- [ ] **Telegram bot (real)** — Wire python-telegram-bot or aiogram.
      Commands: /ask, /remember, /status, /gate. Webhook to langgraph.
      This is the operator interface — you talk to Kai from your phone.
      **Single highest-value feature for daily use.**
- [ ] **Screen capture → OCR → memorize pipeline** — The app.py is wired
      (mss + pytesseract). Needs: actually test it in Docker with X11/Xvfb,
      or run headless with a screenshot file input mode.
- [ ] **Audio capture → VAD → memorize pipeline** — app.py has sounddevice
      + VAD. Needs: test in Docker, fallback for headless (file input mode).

### P2 — Give it a voice (output)
*Kai should be able to respond, not just process.*

- [ ] **TTS wiring** — output/tts/app.py exists as stub. Wire piper-tts
      or edge-tts (both CPU-only). Kai can speak responses back.
- [ ] **Telegram response** — after /ask, send the response back via
      Telegram message. Closes the loop: human asks → Kai reasons → Kai
      responds.

### P3 — Make memory organic (the core differentiator)
*This is what separates Kai from every other agent framework.*

- [ ] **Spaced repetition enforcement** — memu-core has Ebbinghaus decay
      and access_count. Wire it so frequently-accessed memories strengthen,
      neglected ones fade. The retrieve_ranked() function already does this
      but nobody exercises the decay path in production.
- [ ] **Proactive memory surfacing** — Kai should surface relevant memories
      *without being asked*. E.g. "You mentioned a site inspection on
      Friday — that's tomorrow." Needs a background loop in memu-core or
      supervisor that scans upcoming events/deadlines.
- [ ] **Learning from corrections** — when the verifier gives REPAIR or
      FAIL_CLOSED, Kai should learn *why*. Store the correction as a
      memory with high importance. Next time a similar claim comes in,
      the verifier has the correction in its evidence pack.
- [ ] **Cross-session context** — memories from session A should inform
      session B. Currently works via memu-core but nobody tests the
      multi-session path. Write tests that: memorize in session 1,
      retrieve in session 2, verify context carries.
- [ ] **Category-aware retrieval** — memu-core auto-classifies into 8 UK
      construction categories. Use this for domain-specific retrieval
      boosting (setting-out memories score higher when the current query
      is about setting-out).

### P4 — Wire the brain (when RTX 5080 arrives)
*These are blocked on hardware. Don't touch until GPU is here.*

- [ ] **Ollama integration** — wire common/llm.py LLMRouter to real
      endpoints. DeepSeek-V4 for reasoning, Kimi-2.5 for general,
      Dolphin for uncensored PUB mode.
- [ ] **Multi-LLM consensus** — fusion-engine asks 2+ models the same
      question, verifier checks agreement. Real "organic" reasoning.
- [ ] **Whisper transcription** — audio service captures → whisper.cpp
      transcribes → memu-core stores. Full audio memory pipeline.
- [ ] **Vision model** — screen capture → local vision model (LLaVA or
      similar) instead of just OCR. Kai *sees* what's on screen.

### P5 — Production hardening
*Important but not urgent. Do after P0-P3 are solid.*

- [ ] **CI docker build** — GitHub Actions builds all images on every PR.
      Catches Dockerfile regressions before merge.
- [ ] **Integration test in CI** — compose up → smoke test → compose down.
      Runs on every push.
- [ ] **Secrets management** — move from .env to Docker secrets or Vault.
      sovereign compose has Vault config but it's not wired.
- [ ] **Backup-service validation** — backup-service/app.py exists but
      is a stub. Wire real backup of memu-core postgres to local disk.
- [ ] **Log aggregation** — all services write JSON logs. Collect them
      somewhere queryable (Loki, or just a shared volume with grep).
- [ ] **HMAC key rotation in production** — scripts exist and work.
      Need to test in a running stack.

### P6 — Nice-to-have / future
*Park these. Don't think about them until P0-P4 are done.*

- [ ] Calendar sync — calendar-sync/app.py stub. Wire to CalDAV.
- [ ] Workspace manager — workspace-manager stub. Wire to file system ops.
- [ ] Avatar service — output/avatar/app.py stub. Visual presence.
- [ ] gVisor/AppArmor — sovereign compose has config. Only matters when
      executing real tools.
- [ ] Prometheus/Grafana dashboards — compose has the config. Only matters
      when the stack runs in production.
- [ ] Distributed rate limiting — move from per-process dicts to Redis.
      Only matters with multiple replicas.
- [ ] Mobile app / web UI — replace Dash dashboard with something real.

---

## Completed

### v7 hardening (2026-02-25)
- [x] Policy-as-code (security/policy.yml + common/policy.py)
- [x] Verifier rewrite (PASS/REPAIR/FAIL_CLOSED, material claims)
- [x] Evidence packs (memu-core /memory/evidence-pack)
- [x] Circuit breakers wired into fusion-engine
- [x] Quarantine spine (poisoned flag, quarantine endpoints)
- [x] Rate limiting (common/rate_limit.py, wired into tool-gate)
- [x] Idempotency (tool-gate idempotency_key with TTL cache)
- [x] Memorize verdict gating (verifier PASS before store)
- [x] Watermark-driven compressor
- [x] Ledger-worker pagination
- [x] Dashboard policy/breaker/quarantine display
- [x] Supervisor quarantine/verifier integration
- [x] Prometheus config (12 scrape jobs)
- [x] Alert rules (8 rules)
- [x] Alertmanager config (grouping, inhibition)

### v7 test coverage (2026-02-25)
- [x] 28 verifier tests
- [x] 9 quarantine/evidence tests
- [x] 17 policy + rate limit tests
- [x] 4 idempotency tests
- [x] 3 integration chain tests
- [x] Screen capture wired (mss + pytesseract)
- [x] Audio capture wired (sounddevice + VAD)
- [x] datetime.utcnow() bugfix in memu-core

### Earlier phases (P0-P5)
- [x] Git cleanup and repo structure
- [x] Common utilities (auth, runtime, llm, policy)
- [x] All 23 service stubs with /health
- [x] HMAC auth hardening
- [x] UK construction domain (8 categories)
- [x] 5-signal memory ranking
- [x] Ebbinghaus decay model
- [x] Tool-gate with co-sign flow
- [x] Ledger with hash-chain integrity

---

## Session Notes

### 2026-02-25
- v7 plan from GPT/Grok/DeepSeek was ~70% accurate, 30% inflated
- Adapted into 12 PRs, all completed in 2 commits
- Test count: 15 → 76 (5x)
- Key insight: architecture is solid but runs against in-memory stubs.
  The system has never been started as containers.
- Next session priority: **P0 — Docker compose build validation**
- User saving for RTX 5080. All LLM work parked.
- Vision: organic AI that learns. Not an agent framework.
