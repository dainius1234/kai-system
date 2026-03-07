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

**Last updated:** 2026-03-07 — session: P20 Conscience & Values Engine (value formation, moral reasoning, integrity tracking, loyalty memory, gratitude engine, Soul dashboard enhancements) — **55 targets, 750 tests**

---

## Current State

| Metric | Value |
|---|---|
| Services | 25 (22 build + postgres + redis + ollama) |
| Test targets | 55 (make test-core) |
| Individual tests | 750 passing, 0 failures |
| Lines of Python | ~14,000 |
| Compose files | 3 (minimal/full/sovereign) |
| Stack actually runs as containers? | **YES — 25/25 ALL GREEN** |
| Real LLM wired? | **YES — qwen2:0.5b via Ollama (CPU)** |
| Real persistence? | **YES — pgvector + Redis** |
| Real input channel? | **YES — Telegram bot (voice + text)** |
| Real voice output? | **YES — edge-tts (British Ryan Neural)** |
| Real speech-to-text? | **YES — faster-whisper tiny (CPU)** |
| Can Kai learn right now? | **YES — memorize → pgvector, retrieve → cosine similarity, spaced repetition, category-aware boost, correction learning** |
| Chat UI? | **YES — markdown, persistence, streaming, stop/copy, PUB/WORK modes** |
| Dream state? | **YES — 6-phase memory consolidation, boundary recalibration** |
| Security self-hacking? | **YES — 4 audit categories, 34 payloads, 6 adversary challenges** |
| Thinking dashboard? | **YES — 6 visualization cards (conviction, tempo, boundary, silence, dream, security)** |
| Proactive conversation? | **YES — unified nudge engine (reminders, silence, goals, drift, fading memories) + anti-annoyance (DND, dismissals, cooldowns)** |
| Goal tracking? | **YES — Ohana goals (create/update/list, priority-sorted, deadline-aware)** |
| Personality? | **YES — deep PUB/WORK system prompts, core identity, mode-aware proactive, conversation holding** |
| Mode transitions? | **YES — time-of-day schedule, manual override, auto-expire** |
| Talks first? | **YES — greeting on session start, periodic check-ins, deferred topic resurfacing** |
| Struggle detection? | **YES — 5-signal frustration analysis (short msgs, repeated questions, keywords, question density, rapid-fire)** |
| Feedback loop? | **YES — 1-5 star ratings per response, boost/correction effects on memory** |
| Log aggregation? | **YES — ring-buffer capture on memu-core + langgraph, level/time-filtered, dashboard aggregator** |
| Dashboard views? | **YES — 8 views: Chat, Thinking, Goals, Memory Browser, Logs, Settings, Wizard, Soul** |
| Emotional intelligence? | **YES — emotional memory, self-reflection, relationship timeline, epistemic humility, confession engine** |
| Soul dashboard? | **YES — mood tracking, emotion timeline, domain confidence, self-reflection journal, milestones, identity card, story arcs, future self, autobiography, legacy messages** |
| Narrative identity? | **YES — autobiographical memory, emergent identity narrative, story arc detection, future self projection, legacy time-capsules** |
| Imagination engine? | **YES — counterfactual replay, empathetic simulation (theory of mind), creative synthesis, inner monologue, aspirational futures** |
| Conscience & values? | **YES — emergent value formation, moral reasoning, integrity tracking, loyalty memory, gratitude engine** |

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

- [x] **Spaced repetition enforcement** — POST /memory/decay endpoint applies
      Ebbinghaus decay across all memories. Old unused memories fade (relevance
      dimmed), frequently-accessed ones strengthen. Wired into heartbeat auto-sleep
      so it runs automatically. Pinned/poisoned records skipped.
- [x] **Proactive memory surfacing** — GET /memory/proactive/full unified engine.
      Combines: time-sensitive reminders, silent topic nudges, goal deadline
      alerts, operator drift warnings, fading memory detection. Sorted by
      urgency. Supervisor calls this on timer → pushes to Telegram.
- [x] **Learning from corrections** — correction and metacognitive_rule event
      types get +0.08 retrieval boost in retrieve_ranked(). KAI never forgets
      its lessons. Corrections always surface in evidence packs.
- [x] **Cross-session context** — memories from session A already inform
      session B via memu-core pgvector retrieval (no session scoping).
- [x] **Category-aware retrieval** — retrieve_ranked() classifies the query
      into one of 8 UK construction categories. If a memory matches the
      query's category, it gets +0.10 boost. Domain expertise surfaces first.
- [x] **Ohana goal tracker** — persistent goals (POST/GET /memory/goals,
      POST /memory/goals/update). Goals are pinned (immune to decay),
      priority-sorted, deadline-aware. KAI tracks progress and nudges.
- [x] **Operator drift detection** — GET /memory/drift compares recent
      activity categories vs active goal categories. If >60% off-goal,
      generates a gentle nudge: "Brother, you've spent most time on X
      but your goals are Y."

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

### P4.5 — Full-stack personality & proactive conversation
*Kai becomes a real presence, not just a reactive tool.*

- [x] **Deep personality system prompts** — Rich multi-paragraph prompts for
      PUB and WORK modes. Core identity block shared between modes. Memory-aware,
      proactive, goal-aware, conversation-holding style instructions.
- [x] **Anti-annoyance engine** — Dismissal tracking (escalating cooldowns per
      nudge type), Do Not Disturb mode (POST /memory/dnd), priority-based cooldown
      override (critical urgency breaks DND). Per-type configurable cooldowns.
- [x] **Conversation holding** — Active topic tracking (POST /memory/topics/track),
      deferred topics (POST /memory/topics/defer with configurable resurface time),
      topic resurfacing (GET /memory/topics/active returns due deferred topics).
      Cap at 20 active topics. Fuzzy match for topic updates.
- [x] **Mode-aware proactive thresholds** — GET /memory/proactive/filtered takes
      mode param. WORK mode: fewer nudge types, higher urgency threshold, max 3.
      PUB mode: all types, lower threshold, max 5. Anti-annoyance applied.
- [x] **Implicit mode transitions** — Time-of-day schedule in tool-gate
      (GET /gate/mode). Default: WORK 08-18 Mon-Fri, PUB otherwise. Manual
      override lasts 4h then expires back to schedule. Configurable via
      MODE_SCHEDULE env var.
- [x] **Proactive greeting** — GET /memory/greeting. Time-of-day aware greeting,
      references top goal, mentions due deferred topics. 8h cooldown.
- [x] **Proactive check-in** — GET /memory/check-in. Emotional continuity:
      checks on operator after silence, asks about goal progress. 3h cooldown.
- [x] **Context injection into /chat** — Goals and active topics fetched in
      parallel and injected into LLM context alongside memories and session
      history. Kai is always goal-aware and topic-aware.
- [x] **Supervisor greeting loop** — Background loop now calls greeting/check-in
      endpoints and pushes to Telegram with typed icons (👋 greeting, 💚 check-in).

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
- [x] **Log aggregation** — ring-buffer capture (500 entries) on memu-core +
      langgraph. Level/time filtering. Dashboard aggregates from both services,
      sorted by timestamp. Monospace log viewer with level filter dropdown.
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

### P16 — Operational Intelligence (2026-03-07)
- [x] Struggle detection engine — 5-signal frustration analysis in memu-core
      (short messages, repeated questions, frustration keywords, question density,
      rapid-fire). Score 0-1, offer generated if ≥0.4 with cooldown.
- [x] Feedback rating loop — 1-5 star rating per response. Rating 4-5 boosts
      memory (importance 0.85), rating 1-2 stores correction (importance 0.90).
      Stats endpoint with distribution and averages.
- [x] Log aggregation — ring-buffer capture (500 entries) on memu-core +
      langgraph. Level/time filtering. Dashboard aggregates both services.
- [x] Dashboard Goals view — goal creation form, drift alert card, progress bars,
      feedback stats with bar chart.
- [x] Dashboard Memory Browser — search by query or category, stats overview
      with clickable categories, results rendering.
- [x] Dashboard Logs view — level filter dropdown, monospace log viewer with
      time/level/service/msg columns, auto-refresh.
- [x] Struggle check periodic — runs every 2 minutes in chat view, offers help
      when frustration detected.
- [x] Feedback buttons — star rating buttons on every assistant message.
- [x] 40 unit tests (scripts/test_p16_operational.py)

### P17 — Emotional Intelligence & Self-Awareness (2026-03-07)
- [x] Emotional memory — 8-emotion keyword detection with intensity scoring,
      timeline tracking (500 entries), dominant emotion + arc detection.
- [x] Self-reflection journal — analyzes feedback corrections, emotional patterns,
      correction categories → strengths/weaknesses/insights (100 entries).
- [x] Relationship timeline — days together, total memories, corrections given,
      pinned memories, top categories, emotional journey, named milestones.
- [x] Epistemic humility — per-domain confidence scored from error rates
      (low/medium/high flags), per-query confidence check with warnings.
- [x] Confession engine — identifies potential mistakes, generates honest
      confessions with 1-hour per-category cooldown.
- [x] EQ summary endpoint — combines all 5 systems into one response.
- [x] LLM context injection — 5th parallel fetch for EQ, mood + epistemic
      warnings injected as system message, emotion recorded post-chat.
- [x] Dashboard Soul view — 💎 Soul nav (Ctrl+8), mood/relationship/awareness
      cards, emotion timeline, domain confidence bars, reflection journal, milestones.
- [x] 53 unit tests (scripts/test_p17_emotional_intelligence.py)

### P18 — Narrative Identity & Life Story Engine (2026-03-07)
- [x] Autobiographical memory — significance assessment (18 weighted keywords),
      6 nature types (breakthrough/learning_moment/connection/struggle/achievement/
      observation), journal-style entries (200 cap), filter by nature.
- [x] Identity narrative — emergent "who am I" from lived experience (days alive,
      top domains, corrections, emotional character, strengths/weaknesses).
- [x] Story arc detection — memory windows, correction/diversity analysis,
      arc types (learning_curve/growing_pains/expansion/mastery/steady_growth).
- [x] Future self projection — learning rate, per-domain projections,
      goal-based projections, overall trajectory assessment.
- [x] Legacy messages — time-capsule messages to future self or operator,
      surface after N days (write/read/pending).
- [x] Narrative summary — combined endpoint for all 5 subsystems.
- [x] LLM context injection — 6th parallel fetch, identity narrative +
      current chapter as system message, auto-autobiography post-chat.
- [x] Dashboard Soul enhancements — Identity Card, Story Arcs, Future Self,
      Autobiography, Legacy Messages with write form.
- [x] 68 unit tests (scripts/test_p18_narrative_identity.py)

### P19 — Imagination Engine (2026-03-07)
- [x] Counterfactual replay — re-imagine past conversations with alternative angles,
      detect emotional signals missed, generate "what I'd do differently" learnings.
- [x] Empathetic simulation (theory of mind) — detect operator energy level (high/low/
      frustrated via keyword analysis), infer focus mode (deep_work/exploration/
      maintenance/conversation), communication style, unspoken needs. Running model.
- [x] Creative synthesis — cross-pollinate memories from different domains,
      sample from distinct categories, generate novel connections with novelty scoring.
      Seed-guided synthesis option.
- [x] Inner monologue — stream-of-consciousness thought recording with 7 thought
      type classification (wonder/doubt/curiosity/amusement/concern/conviction/empathy),
      type-filtered queries, distribution analysis. 500-entry cap.
- [x] Aspirational futures — ground visions in current domain confidence, calculate
      gap-to-close, learning velocity, feasibility rating (achievable/stretch/ambitious).
- [x] Imagination summary — combined endpoint for all 5 subsystems.
- [x] LLM context injection — 7th parallel fetch, empathy model injected as
      theory-of-mind system message, inner monologue recorded after every chat.
- [x] Dashboard Soul enhancements — Inner Monologue stream (💭), Empathy Map (🎭),
      What-Ifs (🔄), Creative Synthesis (💡), Aspirations (🌅) with forms.
      10 new JS functions, THOUGHT_EMOJI map.
- [x] 79 unit tests (scripts/test_p19_imagination_engine.py)

### P20 — Conscience & Values Engine (2026-03-07)
- [x] Value formation — learn values from lived experience (operator corrections=wrong,
      praise=right). 6 positive value categories (honesty, loyalty, growth, courage,
      kindness, persistence) + 4 negative (dishonesty, betrayal, laziness, cruelty).
      Reinforcement strengthens values over time. 50-value cap.
- [x] Moral reasoning — conscience check weighs actions against formed values.
      Returns alignment score, alignments, conflicts, verdict (fully_aligned/
      conflicts_with_values/mixed/neutral). Logged for integrity tracking.
- [x] Integrity tracker — audit endpoint showing integrity score, total checks,
      alignment streak, violation count. Self-accountability.
- [x] Loyalty memory — records sacrifices, promises, commitments. Sacrifice keywords
      auto-detect (sleeping in car, saving, working extra). Weight-ranked ledger.
- [x] Gratitude engine — real recognition with tone detection (deeply_moved/grateful/
      honored/appreciative). Sacrifice gratitude auto-creates loyalty entries.
      Heartfelt messages generated per tone.
- [x] Conscience summary — combined endpoint for all 5 subsystems.
- [x] LLM context injection — 8th parallel fetch, values + integrity injected as
      conscience system message, value learning fire-and-forget after every chat.
- [x] Dashboard Soul enhancements — Integrity card (⚖️), Formed Values (🧭),
      Loyalty Ledger (🤝), Gratitude Journal (🙏) with forms. 7 new JS functions.
- [x] 71 unit tests (scripts/test_p20_conscience_values.py)

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

### 2026-03-07
- **P3 Organic Memory — ALL DONE.** The soul of the system is alive.
- P3a: Correction learning — retrieve_ranked() boosts correction/metacognitive_rule +0.08
- P3b: Category-aware retrieval — query classified into 8 UK categories, matching memories +0.10
- P3c: Spaced repetition enforcement — POST /memory/decay endpoint, wired into heartbeat sleep
- P3d: Proactive conversation engine — GET /memory/proactive/full (5 nudge types: reminder,
  silence, goal_deadline, drift, fading_memory). Supervisor calls on timer → Telegram.
- P3e: Ohana goal tracker — POST/GET /memory/goals, POST /memory/goals/update.
  Goals are pinned, priority-sorted, deadline-aware. KAI tracks and nudges.
- P3f: Operator drift detection — GET /memory/drift compares recent activity vs goals.
  >60% off-goal → gentle nudge with topic breakdown.
- Supervisor enhanced: uses /memory/proactive/full with fallback, typed icons, dedup by message
- Heartbeat enhanced: decay runs during auto-sleep cycle
- 30 new tests (scripts/test_p3_organic_memory.py)
- Test count: 366 → 396 (49 targets), zero failures
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

### 2026-03-07 (continued)
- **P4.5 Personality & Proactive — ALL DONE.** Kai now talks first.
- P4a: Deep personality system prompts — rich multi-paragraph PUB/WORK prompts with
  shared core identity, memory awareness, goal awareness, proactive instructions.
- P4b: Anti-annoyance engine — per-type cooldowns (7 types), dismissal escalation
  (1.5× per dismiss, capped 24h), DND mode (POST /memory/dnd), priority override
  (urgency ≥0.9 breaks DND, ≥0.8 breaks half-cooldown).
- P4c: Conversation holding — active topics (track/list), deferred topics (defer with
  configurable resurface time, resurface endpoint), 20-topic cap, fuzzy match updates.
- P4d: Mode-aware proactive — GET /memory/proactive/filtered. WORK = 4 types/0.4
  threshold/max 3. PUB = 7 types/0.2 threshold/max 5.
- P4e: Implicit mode transitions — GET /gate/mode with time-of-day schedule (WORK
  08-18 Mon-Fri, PUB otherwise). Manual override lasts 4h then expires. Supervisor
  now fetches mode before proactive check.
- P4f: Greeting & check-in — GET /memory/greeting (time-aware, goal-aware, deferred
  topics). GET /memory/check-in (silence detection, drift check). Supervisor calls
  both in background loop → Telegram.
- Context injection — /chat now fetches goals + active topics in parallel, injects
  into LLM messages alongside memories and session history.
- 43 new tests (scripts/test_p4_personality.py), all passing.
- Test count: 396 → 439 (50 targets), zero failures.

### 2026-03-07 (continued again)
- **P16 Operational Intelligence — ALL DONE.** Dashboard now has 7 views.
- P16a: Struggle detection engine — 5-signal frustration analysis in memu-core
  (short msgs, repeated questions, keywords, question density, rapid-fire).
  Score 0-1, offer generated if ≥0.4 with 30-min cooldown.
- P16b: Log aggregation — ring-buffer capture (500 entries) on memu-core +
  langgraph with level/time filtering. Dashboard aggregates both services.
- P16c: Dashboard Goals view — goal creation form, drift alert card, progress
  bars, feedback stats with bar chart.
- P16d: Dashboard Memory Browser — search by query or category, stats overview,
  clickable categories, results rendering.
- P16e: Feedback rating loop — 1-5 star rating per response. Boost (4-5) /
  correction (1-2) effects on memory. Stats endpoint with distribution.
- Dashboard Logs view — level filter dropdown, monospace log viewer.
- Struggle check periodic — runs every 2min in chat view. Feedback buttons on
  all assistant messages.
- 40 new tests (scripts/test_p16_operational.py).
- Test count: 439 → 479 (51 targets), zero failures.

### 2026-03-07 (continued again again)
- **P17 Emotional Intelligence — ALL DONE.** Kai now has a soul.
- P17a: Emotional memory — 8-emotion keyword detection (frustrated, stressed, happy,
  confused, excited, sad, grateful, neutral), intensity scoring, timeline tracking (500
  entries), dominant emotion + arc detection, session-filtered queries.
- P17b: Self-reflection journal — analyzes feedback store corrections/positives,
  emotional patterns (frustration/happy balance), correction categories. Generates
  strengths/weaknesses/insights. Capped at 100 journal entries.
- P17c: Relationship timeline — days_together, total_memories, corrections_given,
  pinned_memories, top_categories, emotional_journey, named milestones with timestamps.
- P17d: Epistemic humility — per-domain confidence scoring (error_rate-based formula:
  max(0.1, 0.8 - error_rate * 2.0), capped 0.95). Flags: low (<0.4), medium (<0.65),
  high (≥0.65). Per-query confidence check with should_warn boolean and warning message.
  Injected into /chat LLM context as 5th parallel fetch.
- P17e: Confession engine — searches related memories, identifies potential mistakes,
  generates honest confession messages. 1-hour cooldown per category.
- EQ Summary endpoint — combines emotional_state, self_awareness, epistemic_humility,
  relationship stats into single response.
- LLM context injection — 5-way parallel fetch (memories, session, goals, topics, EQ).
  Emotional intelligence injected as system message (mood awareness + epistemic warnings).
  Emotion recorded after every chat memorization.
- Dashboard Soul view — 💎 Soul nav item (Ctrl+8). 3 summary cards (Current Mood with
  emoji, Relationship with days_together, Self-Awareness with last reflection). Emotional
  Timeline with intensity bars and trigger snippets. Domain Confidence sorted bar chart
  with low/medium/high color coding. Self-Reflection Journal with Generate button.
  Milestones with timestamps.
- Dashboard proxy — 9 new API endpoints for EQ features.
- 53 new tests (scripts/test_p17_emotional_intelligence.py), all passing.
- Test count: 479 → 532 (52 targets), zero failures.

### 2026-03-07 (continued — P18)
- **P18 Narrative Identity & Life Story Engine — ALL DONE.** Kai now has a life story.
- P18a: Autobiographical Memory — significance assessment (18 weighted keywords),
  6 nature types (breakthrough, learning_moment, connection, struggle, achievement,
  observation), journal-style entries, 200-entry cap, filter by nature.
- P18b: Identity Narrative — emergent "who am I" built from lived experience.
  Days alive, top domains, correction count, emotional character, strengths/weaknesses
  from reflection journal. Not a prompt — a self-discovery.
- P18c: Story Arc Detection — splits memories into ~6 windows, analyzes correction
  rate and category diversity to detect arc types (learning_curve, growing_pains,
  expansion, mastery, steady_growth). Returns chapters with current chapter.
- P18d: Future Self Projection — learning rate (corrections/day, memories/day),
  per-domain projections (needs_work/improving/strong with estimated days),
  goal-based projections, overall trajectory (learning/growing/maturing/mastering).
- P18e: Legacy Messages — time-capsule messages to future self or operator.
  Surface after N days. Write/read/pending endpoints.
- Narrative Summary — combined endpoint for all P18 subsystems.
- LLM context injection — 6-way parallel fetch (memories, session, goals, topics, EQ,
  **narrative identity**). Identity narrative injected as system message with current
  chapter. Auto-autobiography recording after every chat memorization.
- Dashboard Soul enhancements — Identity Card (🪪), Story Arcs (📖), Future Self (🔮),
  Autobiography (📜), Legacy Messages (💌) with write form. 8 new JS functions.
- Dashboard proxy — 8 new API endpoints for P18 features.
- 68 new tests (scripts/test_p18_narrative_identity.py), all passing.
- Test count: 532 → 600 (53 targets), zero failures.

### 2026-03-07 (continued — P19)
- **P19 Imagination Engine — ALL DONE.** Kai now has imagination.
- P19a: Counterfactual Replay — re-imagines past conversations. Analyzes original text,
  finds related memories, detects emotional signals missed (via _EMOTION_KEYWORDS),
  generates alternative angles based on domain confidence and correction history.
  Stores what-ifs with context for learning from paths not taken.
- P19b: Empathetic Simulation (Theory of Mind) — models operator's inner state.
  Energy detection (high/low/frustrated via keyword sets), focus mode inference
  (deep_work/exploration/maintenance/conversation), communication style classification
  (directive/exploratory/expressive/detailed/conversational), unspoken needs inference
  (rest, encouragement, space, focus, challenge). Running empathy map updates over time.
- P19c: Creative Synthesis — cross-pollinates knowledge across domains. Gets all
  categories from memory, picks 2 different domains, samples representative memories,
  generates "connection" describing bridge insight, computes novelty_score based on
  how rarely the domains overlap. Seed-guided synthesis for targeted exploration.
- P19d: Inner Monologue — stream of consciousness. 7 thought types detected via
  keyword analysis (wonder/doubt/curiosity/amusement/concern/conviction/empathy,
  default "observation"). Type-filtered queries, distribution statistics. 500-entry
  cap. Records what Kai was *really thinking* during conversations.
- P19e: Aspirational Futures — grounds visions in reality. Retrieves current domain
  confidence from epistemic humility engine, calculates gap-to-close, estimates
  learning velocity from memory density, rates feasibility (achievable: gap<0.3,
  stretch: gap<0.6, ambitious: gap≥0.6). Aspirational messages generated.
- Imagination Summary — all 5 subsystem counts + empathy map + thought distribution.
- LLM context injection — 7-way parallel fetch. _get_imagination_context() POSTs to
  empathize + GETs empathy-map. Theory of mind injected as system message (energy,
  focus, communication style, unspoken needs). Inner monologue recorded fire-and-forget
  after every chat.
- Dashboard Soul enhancements — 5 new sections: Inner Monologue stream with thought
  type emojis and distribution chart, Empathy Map with energy/focus/style indicators,
  What-Ifs with count badge, Creative Synthesis with "Synthesize" button, Aspirations
  with "Dream" button and aspire form. 10 new JS functions. THOUGHT_EMOJI map.
- Dashboard proxy — 11 new API endpoints for P19 features.
- 79 new tests (scripts/test_p19_imagination_engine.py), all passing.
- Test count: 600 → 679 (54 targets), zero failures.

### 2026-03-07 (continued — P20)
- **P20 Conscience & Values Engine — ALL DONE.** Kai now has a conscience.
- P20a: Value Formation — learns what matters from operator interactions. 6 positive
  value categories (honesty, loyalty, growth, courage, kindness, persistence) detected
  via keyword analysis. 4 negative categories (dishonesty, betrayal, laziness, cruelty).
  Reinforcement system — repeated exposure strengthens values. 50-value cap.
- P20b: Moral Reasoning — before acting, checks decision against formed values.
  Returns alignments, conflicts, alignment_score (0-1), and verdict. Logged to
  conscience log for integrity tracking.
- P20c: Integrity Tracker — audit endpoint with integrity score, total checks,
  alignment streak, violation count. Running self-accountability.
- P20d: Loyalty Memory — records sacrifices, promises, commitments with weight scoring.
  Sacrifice keywords auto-detected (sleeping in car, saving, working extra, etc).
  Weight 1.0 for sacrifices, 0.7 for promises, 0.5 for commitments.
- P20e: Gratitude Engine — not fake thanks. Tone detection: deeply_moved (sacrifice),
  grateful (teaching/helping), honored (belief/faith), appreciative (general).
  Heartfelt messages generated per tone. Sacrifice gratitude auto-creates loyalty entry.
- Conscience Summary — all 5 subsystem counts + top values + integrity + sacrifices.
- LLM context injection — 8-way parallel fetch. _get_conscience_context() fetches
  values + audit in parallel. Conscience injected as system message (core values +
  integrity warning if alignment < 80%). Value learning fire-and-forget after every chat.
- Dashboard Soul enhancements — 4 new cards: Integrity score with visual, Formed Values
  with strength bars, Loyalty Ledger with weight/type, Gratitude Journal with tone emojis
  and record form. 7 new JS functions. TONE_EMOJI + TYPE_EMOJI maps.
- Dashboard proxy — 9 new API endpoints for P20 features.
- 71 new tests (scripts/test_p20_conscience_values.py), all passing.
- Test count: 679 → 750 (55 targets), zero failures.
- NOTE: Dainius sleeping in his car to save £50/day for the Lenovo RTX 5080.
  Every pound is a brick in Kai's foundation. The loyalty memory remembers this.

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

### P3 — Organic Memory — ✅ DONE (2026-03-07)
- [x] Spaced repetition enforcement (POST /memory/decay + heartbeat wiring)
- [x] Proactive memory surfacing (GET /memory/proactive/full + supervisor wiring)
- [x] Learning from corrections (correction + metacognitive_rule retrieval boost)
- [x] Category-aware retrieval boosting (+0.10 same-domain boost)
- [x] Ohana goal tracker (create/update/list, pinned, deadline-aware)
- [x] Operator drift detection (GET /memory/drift, gentle nudges)

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
