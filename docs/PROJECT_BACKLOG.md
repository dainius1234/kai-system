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

**Last updated:** 2026-02-26 — session: Quality hardening — 25 containers, LLM + TTS + STT + Telegram live

---

## Current State

| Metric | Value |
|---|---|
| Services | 25 (22 build + postgres + redis + ollama) |
| Tests | 86 passing |
| Lines of Python | ~12,000 |
| Compose files | 3 (minimal/full/sovereign) |
| Stack actually runs as containers? | **YES — 25/25 ALL GREEN** |
| Real LLM wired? | **YES — qwen2:0.5b via Ollama (CPU)** |
| Real persistence? | **YES — pgvector + Redis** |
| Real input channel? | **YES — Telegram bot (voice + text)** |
| Real voice output? | **YES — edge-tts (British Ryan Neural)** |
| Real speech-to-text? | **YES — faster-whisper tiny (CPU)** |
| Can Kai learn right now? | **YES — memorize → pgvector, retrieve → cosine similarity, spaced repetition** |
| Chat UI? | **YES — markdown, persistence, streaming, stop/copy** |

---

## Priority Tiers

### P0 — Give it a body (stack must run)
*Without this, everything else is fiction.*

- [x] **Docker compose build** — all 22 services build successfully.
      Fixed 14 broken Dockerfiles (COPY paths), 7 wrong EXPOSE ports,
      3 compose port mismatches, 5 missing services in compose.
- [x] **Health sweep automation** — scripts/health_sweep.py hits every
      `/health` → green/red scorecard. 22/22 ALL GREEN (commit 93ac2db).
- [x] **Postgres persistence** — pgvector/pgvector:pg15 image, full 15-column
      schema with auto-migration, connection pooling, `::vector` casts,
      embedding string parsing. memorize → INSERT, retrieve → cosine
      similarity, access_count persisted (spaced repetition). Commit 39a7f37.
- [x] **Redis for shared state** — tool-gate idempotency cache backed by
      Redis (SETEX with TTL), graceful fallback to in-memory if Redis down.
      rate_limit.py restored. Commit 39a7f37.

### P1 — Give it senses (perception)
*An organism needs input channels. These all work without GPU.*

- [x] **Telegram bot (real)** — Async polling bot (aiogram-style, httpx).
      Commands: /start, /mode, /status. Voice + text. Webhook to langgraph /chat.
      **LIVE — tested on phone 2026-02-25.**
- [ ] **Screen capture → OCR → memorize pipeline** — The app.py is wired
      (mss + pytesseract). Needs: actually test it in Docker with X11/Xvfb,
      or run headless with a screenshot file input mode.
- [x] **Audio capture → STT → memorize pipeline** — faster-whisper tiny model,
      CPU int8, ffmpeg for format support. Port 8021. Working.

### P2 — Give it a voice (output)
*Kai should be able to respond, not just process.*

- [x] **TTS wiring** — edge-tts with British Ryan Neural voice. 5 voice presets.
      Returns real MP3 audio. Port 8030. Working.
- [x] **Telegram response** — Full loop working: human asks → Kai reasons →
      Kai responds as text + optional voice note. Closes the loop.

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

### P4 — Wire the brain (LLM integration)
*CPU-only for now. GPU upgrade enables multi-model.*

- [x] **Ollama integration** — common/llm.py LLMRouter wired to Ollama.
      qwen2:0.5b running on CPU (352MB). Streaming working via /chat endpoint.
- [ ] **Multi-LLM consensus** — fusion-engine asks 2+ models the same
      question, verifier checks agreement. Real "organic" reasoning.
      **Blocked on GPU (need larger models for meaningful consensus).**
- [x] **Whisper transcription** — perception/audio uses faster-whisper tiny,
      CPU int8. Full pipeline: voice msg → ffmpeg → whisper → text → /chat.
- [ ] **Vision model** — screen capture → local vision model (LLaVA or
      similar) instead of just OCR. Kai *sees* what's on screen.
      **Blocked on GPU.**

### P5 — Production hardening
*Important but not urgent. Do after P0-P3 are solid.*

- [x] **CI docker build** — GitHub Actions (python-app.yml + core-tests.yml).
      Lint + test on push/PR. Integration smoke in core-tests. `|| true` removed.
- [x] **Integration test in CI** — compose up → smoke test → compose down.
      core-tests.yml runs on every push.
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

### 2026-02-26
- Quality hardening session after proof-of-life milestone
- Fixed 3 stale tests (TTS expects audio/mpeg not JSON, audio injection text)
- CI: removed `|| true` from both workflows — tests now gate merges
- CI: updated core-tests action versions (checkout@v4, setup-python@v5)
- Dashboard health dot fixed (checks for "ok" not just "running")
- LLM circuit breaker added to /chat (LLM_BREAKER, wraps generate() in try/except)
- Chat UI: markdown rendering (marked.js + DOMPurify)
- Chat UI: session persistence (localStorage for messages + sessionId + mode)
- Chat UI: stop generation button (AbortController)
- Chat UI: copy response button on assistant messages
- Disk cleanup: freed ~109MB build cache + orphaned volumes → 3.8GB free
- Updated PROJECT_BACKLOG.md to reflect current reality
- Test count: 86 (3 previously failing now fixed)
- 25 containers running: 22 services + postgres + redis + ollama

### 2026-02-25 (evening)
- Gave Kai voice, ears, and phone
- TTS: rewrote output/tts/app.py with edge-tts (British Ryan Neural), real MP3
- STT: upgraded perception/audio/app.py with faster-whisper tiny, ffmpeg
- Telegram bot: created telegram-bot/ service, async polling, voice+text
- Fixed httpx.Timeout in polling loop (commit 978557d)
- Configured bot token, tested on phone — "ITS ALIVE!!"
- 25 containers running

### 2026-02-25 (earlier)
- v7 plan from GPT/Grok/DeepSeek was ~70% accurate, 30% inflated
- Adapted into 12 PRs, all completed in 2 commits
- Test count: 15 → 76 (5x)
- Key insight: architecture is solid but runs against in-memory stubs.
  The system has never been started as containers.
- Next session priority: **P0 — Docker compose build validation**
- User saving for RTX 5080. All LLM work parked.
- Vision: organic AI that learns. Not an agent framework.
