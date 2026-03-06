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

**Last updated:** 2026-03-06 — session: Production hardening (Redis pubsub, Docker secrets, backup service, HMAC rotation drill) — **48 targets, 366 tests**

---

## Current State

| Metric | Value |
|---|---|
| Services | 25 (22 build + postgres + redis + ollama) |
| Test targets | 48 (make test-core) |
| Individual tests | 366 passing, 0 failures |
| Lines of Python | ~14,000 |
| Compose files | 3 (minimal/full/sovereign) |
| Stack actually runs as containers? | **YES — 25/25 ALL GREEN** |
| Real LLM wired? | **YES — qwen2:0.5b via Ollama (CPU)** |
| Real persistence? | **YES — pgvector + Redis** |
| Real input channel? | **YES — Telegram bot (voice + text)** |
| Real voice output? | **YES — edge-tts (British Ryan Neural)** |
| Real speech-to-text? | **YES — faster-whisper tiny (CPU)** |
| Can Kai learn right now? | **YES — memorize → pgvector, retrieve → cosine similarity, spaced repetition** |
| Chat UI? | **YES — markdown, persistence, streaming, stop/copy, PUB/WORK modes** |
| Dream state? | **YES — 6-phase memory consolidation, boundary recalibration** |
| Security self-hacking? | **YES — 4 audit categories, 34 payloads, 6 adversary challenges** |
| Thinking dashboard? | **YES — 6 visualization cards (conviction, tempo, boundary, silence, dream, security)** |

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
- [x] **Secrets management** — Docker secrets support via `load_secret()`
      in `common/auth.py`. Reads from `/run/secrets/` convention files.
      `docker-compose.full.yml` has secrets blocks for tool-gate, langgraph, backup-service.
- [x] **Backup-service validation** — Full rewrite: postgres, redis, memory,
      ledger backup + restore. SHA-256 checksums, filename sanitization.
- [ ] **Log aggregation** — all services write JSON logs. Collect them
      somewhere queryable (Loki, or just a shared volume with grep).
- [x] **HMAC key rotation in production** — 3-phase lifecycle drill
      (single → overlap → retire) with 14 unittest tests.

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

### P7 — Agentic patterns (2026-02-28)
- [x] Episode saver with fallback (memory, disk, spool)
- [x] Episode spool integrity with checksum verification
- [x] Error budget circuit breaker (window-based, auto-recovery)
- [x] Specialist router (8-category classification + LLM fallback)
- [x] Memory-driven planner (gap analysis, recent-weighted, priority queue)
- [x] Invoice generator (memu-core /invoice endpoint)
- [x] Memu retrieval ranking (recency, importance, access boost)

### P8 — Dashboard Thinking Pathways (2026-03-01)
- [x] Thinking Pathways page (thinking.html)
- [x] 4 visualization cards (conviction, tempo, boundary, silence)
- [x] Real-time data from fusion-engine + langgraph endpoints
- [x] Chart.js gauge + radar visualizations

### P9 — Security Self-Hacking (2026-03-04)
- [x] Security audit engine (langgraph/security_audit.py)
- [x] 4 audit categories: injection, auth_bypass, data_leak, resource_abuse
- [x] 21 + 13 test payloads (34 total)
- [x] 6 adversary challenges (including challenge_security)
- [x] Dashboard security audit card + runSecurityAudit() JS
- [x] 23 unit tests

### P10 — memu-core retrieval hardening (2026-02-28)
- [x] Multi-signal retrieval ranking
- [x] Recency + importance + access boost scoring

### P11 — Specialist router (2026-02-28)
- [x] 8-category keyword classification
- [x] LLM fallback for ambiguous queries

### P12 — Memory-driven planner (2026-02-28)
- [x] Gap analysis from memory
- [x] Priority queue with recent-weighted scoring

### P13 — Error budget breaker (2026-02-28)
- [x] Window-based error tracking
- [x] Auto-recovery on success streak

### P14 — Adversary challenge engine (2026-02-28)
- [x] 6 challenge types (hallucination, confidence, evidence, boundary, planning, security)
- [x] ChallengeResult with modifier + confidence

### P15 — Dream State (2026-03-04)
- [x] 6-phase dream cycle (ENTER → CONSOLIDATE → PRUNE → CONNECT → RECALIBRATE → WAKE)
- [x] Memory consolidation (cluster, merge, boost high-importance)
- [x] Boundary recalibration (gap detection, drift detection)
- [x] DreamInsight generation
- [x] Dashboard dream card + triggerDream() JS
- [x] 26 unit tests

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

### 2026-03-04
- P9 Security Self-Hacking + P15 Dream State — built and committed (d52d12c)
- README updated with PUB/WORK modes section
- Fixed `has_gap` → `is_gap` bug in kai_config.py boundary recalibration
- Test suite: 51 targets, 347 individual tests, zero failures
- Adversary challenges: 6 (challenge_security added as #6)

### 2026-03-05
- Full documentation hardening pass
- Created `docs/gaps_and_hardening.md` — honest status on all gap items
- Added Hardware Performance Track (HP1-HP6) to `unfair_advantages.md`
- Updated continuation notes in `unfair_advantages.md` (47/270 tests, 6 challenges)
- Updated this backlog — P7-P15 all marked completed, test count 86 → 270

### 2026-03-05 (continued)
- **Gaps Sprint:** JSON stdout logging (common/runtime.py), vector cleanup endpoint
  (memu-core /memory/cleanup), ledger stats proxy (dashboard /api/ledger-stats).
  10 tests added. Committed `052c913`.
- **HP2: MoE Model Selector** — `langgraph/model_selector.py` (4 model profiles,
  complexity estimation, scoring algorithm). Wired into /chat and /run. 20 tests.
- **HP4: CoT Tree Search** — `langgraph/tree_search.py` (branch generation,
  conviction pruning, multi-depth search). Wired as /run rethink fallback. 14 tests.
- **HP5: Priority Queue** — `langgraph/priority_queue.py` (4 priority levels,
  semaphore concurrency, singleton). /queue/stats endpoint added. 12 tests.
- New endpoints: GET /models, GET /queue/stats
- Test count: 280 → 347 (51 targets)

---

## Hardware Performance Track (Future — GPU Arrival)

See `docs/unfair_advantages.md` for full details. Summary:

| ID | Feature | Impact | Effort |
|---|---|---|---|
| HP1 | Speculative Decoding | High (2-3× speed) | Low |
| HP1 | Speculative Decoding | High (2-3× speed) | Low |
| HP2 | MoE Model Selection | High (expert routing) | Zero (use model) | ✅ DONE |
| HP3 | VRAM Watchdog + Adaptive Quantization | Medium (stability) | Medium |
| HP4 | CoT Tree Search + Conviction Pruning | Med-High (quality) | Medium | ✅ DONE |
| HP5 | Priority Queue (latency-sensitive first) | Medium (UX) | Low | ✅ DONE |
| HP6 | Partial Layer Loading + NVMe Offload | Low-Med (flexibility) | Low |

---

## Remaining Open Items

### P3 — Organic Memory (still open)
- [ ] Spaced repetition enforcement (production decay path)
- [ ] Proactive memory surfacing (background loop)
- [ ] Learning from corrections (verifier → memory feedback)
- [ ] Category-aware retrieval boosting

### P5 — Production Hardening (still open)
- [x] Docker secrets / Vault wiring — ✅ DONE (load_secret + compose secrets)
- [x] Backup-service real implementation — ✅ DONE (pg/redis/memory/ledger + restore)
- [x] JSON structured logging — ✅ DONE (gaps sprint: stdout + file)
- [x] HMAC rotation test in running stack — ✅ DONE (14-test drill)

### System Gaps (see `docs/gaps_and_hardening.md`)
- [x] Vector cleanup job (90-day TTL) — ✅ DONE (POST /memory/cleanup)
- [x] Ledger stats dashboard endpoint — ✅ DONE (GET /api/ledger-stats)
- [x] Redis pubsub for real-time dashboard — ✅ DONE (SSE /api/events + 4 channels)
- [ ] TPM verify (hardware-blocked)
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
