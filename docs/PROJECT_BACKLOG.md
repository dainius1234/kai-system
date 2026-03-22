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

**Last updated:** 2026-03-22 — session: J-Series Jewels roadmap + 5 research gaps closed — **69 targets, 1425+ tests**

---

## Current State

| Metric | Value |
|---|---|
| Services | 26 (22 build + postgres + redis + ollama + lakeFS) |
| Test targets | 70 (make test-core) |
| Individual tests | 1492+++ passing, 0 failures |
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
| Proactive agent? | **YES — scheduled tasks, reminders, morning briefing, evening check-in, action registry, agent summary, supervisor auto-fires** |
| Operator model? | **YES — echo-response engine, nudge escalation ladder (4-tier), cross-mode insight bridge, impact oracle, shadow memory branches, 10-way LLM context** |

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

### P21 — Proactive Agent Loop (Soul → Action Bridge)
*Kai initiates. Not just responds. This is what makes it daily-useful.*

- [x] **Scheduled task engine** — cron-like scheduler in memu-core. Register
      recurring tasks (morning briefing, savings check, goal review). Fires
      at configured times via supervisor polling of /memory/schedule/due.
- [x] **Morning briefing** — POST /memory/briefing/morning: goals summary,
      pending reminders, emotional arc, proactive nudges, today's schedule.
      Time-of-day greeting. POST /memory/briefing/evening: daily summary,
      reflection prompt, tomorrow preview.
- [x] **Reminder system** — POST /memory/reminders/set (text, fire_at, repeat).
      GET /memory/reminders. GET /memory/reminders/due. Fire/cancel endpoints.
      Repeat support: once, daily, weekly, hourly.
- [x] **Proactive firing** — supervisor enhanced: polls /memory/reminders/due
      and /memory/schedule/due every loop. Fires and pushes to Telegram with
      type-specific icons (⏰ reminder, 📅 scheduled, 📰 briefing).
- [x] **Action registry** — GET /memory/actions: catalog of 13+ capabilities.
      Foundation for "Kai, do X" commands. Self-awareness of abilities.
- [x] **Agent summary** — GET /memory/agent/summary: active tasks, pending
      reminders, due items, briefings generated, capability count.
- [x] **LLM context injection** — 9th parallel fetch: due tasks + due reminders
      + capabilities injected as system message. Kai is schedule-aware.
- [x] **Dashboard Goals enhancements** — Reminders card (set/cancel/list),
      Scheduled Tasks card (schedule/cancel/list), Agent Status card with
      briefing buttons. 9 new JS functions.
- [x] **93 unit tests** (scripts/test_p21_proactive_agent.py)

### P22 — Operator Model & Adaptive Response
*Inspired by 2026 research (Echo Agents, Fractal Context, Shadow Agents,
adversarial training, predictive coding). Kai should MODEL you, not just
REMEMBER you. Cross-referenced with Grok brainstorm 2026-03-22.*

- [x] **P22a Echo-Response Engine** — emotional continuity in replies. Detects
      frustration from P17 emotion timeline, bridges past struggles ("frustrated
      AGAIN like when we talked about VAT"). Response intensity modulation.
      POST /memory/echo/analyse, GET /memory/echo/history.
- [x] **P22b Nudge Escalation Ladder** — 4-tier escalation (gentle → firm →
      tough love → intervention). Tracks dismissals per nudge, escalates tone
      after 3+ ignores. Learns from pushback. POST /memory/nudge/escalate,
      GET /memory/nudge/ladder.
- [x] **P22c Cross-Mode Insight Bridge** — tag memories by mode (PUB/WORK).
      Surface cross-mode insights ("in pub mode you mentioned X, relevant here").
      GET /memory/cross-mode, POST /memory/cross-mode/scan.
- [x] **P22d Impact Oracle** — goal-to-goal causal chains. Predicts consequences
      of actions on goals, emotions, time. "If you skip Goal X, Goal Y suffers."
      POST /memory/oracle/predict, GET /memory/oracle/chains.
- [x] **P22e Shadow Memory Branches** — persistent what-if branches from
      counterfactuals. Queryable alternate timelines. Nightly shadow generation.
      POST /memory/shadow/branch, GET /memory/shadow/branches,
      GET /memory/shadow/explore.
- [x] **LLM context injection** — 10th parallel fetch: operator model context
      (echo state, escalation level, cross-mode insights, oracle predictions).
- [x] **Dashboard enhancements** — Operator Model card (echo state, nudge
      ladder, cross-mode insights, oracle predictions, shadow branches).
- [x] **Supervisor escalation alerts** — _check_escalations() polls nudge
      ladder, fires Telegram alerts for level 3+ (tough love, intervention).
- [x] **117 unit tests** (scripts/test_p22_operator_model.py)

### P29 — Financial Awareness
*Dainius sleeps in his car saving £50/day. Kai should track this.*

- [ ] **Savings tracker** — POST /memory/finance/log (amount, category, note).
      Running total, daily burn rate, goal countdown.
- [ ] **RTX 5080 goal countdown** — target amount, current savings, days remaining
      at current rate. Dashboard widget + Telegram daily update.
- [ ] **Expense categorization** — auto-categorize from text (fuel, food, tools,
      materials, savings). Monthly/weekly breakdowns.
- [ ] **Financial summary** — GET /memory/finance/summary. Net position,
      trajectory, goal progress. Part of morning briefing.

### P23 — Knowledge Ingestion (RAG Pipeline)
*Kai can memorize from conversation. Now it reads documents.*

- [ ] **Document upload endpoint** — POST /memory/ingest (file upload).
      Accept PDF, TXT, MD, CSV. Size limits, sanitization.
- [ ] **Chunking pipeline** — split documents into overlapping chunks
      (512 tokens, 50 overlap). Preserve section headers as metadata.
- [ ] **Embedding + storage** — chunks → pgvector embeddings. Source tracking
      (filename, page, chunk_id). Retrievable via existing cosine search.
- [ ] **Source attribution** — retrieve results include source document info.
      "I read this in [document.pdf, page 3]."

### P24 — Temporal Intelligence
*Kai sees patterns in time. "You always X on Y."*

- [ ] **Activity pattern detection** — analyze memory timestamps for recurring
      patterns (day-of-week, time-of-day, seasonal). Cluster by category.
- [ ] **Habit tracking** — detect forming/breaking habits from interaction patterns.
      Streak counting, consistency scoring.
- [ ] **Predictive nudges** — "It's Monday morning — you usually check invoices
      around now." Grounded in actual data, not assumptions.
- [ ] **Circadian awareness** — energy level inference from message timing,
      length, emotional tone. Adapts proactive behavior to operator rhythm.

### P25 — Voice Pipeline (Car Interface)
*Voice is the natural interface when you're driving.*

- [ ] **Real-time STT** — Whisper integration for continuous listening mode.
      CPU-capable (tiny/base models). Wake word detection.
- [ ] **TTS streaming** — Piper/Coqui local TTS. Low-latency for conversation.
      British voice. Emotion-aware pacing.
- [ ] **Voice-first Telegram** — voice messages processed with full context
      (not just transcription — emotional tone from audio features).
- [ ] **Hands-free mode** — continuous listen → process → speak loop.
      Safe for driving. Minimal UI interaction needed.

### P26 — Context Budget Manager
*As memories grow, context windows overflow. Smart pruning needed.*

- [ ] **Token budget allocator** — configurable per-section token limits
      (memories: 2000, session: 1000, goals: 500, EQ: 300, etc.).
- [ ] **Relevance-ranked selection** — score memories by relevance to current
      query, recency, importance. Fill budget greedily.
- [ ] **Summarization fallback** — when budget exceeded, summarize lower-priority
      sections rather than truncating.
- [ ] **Budget telemetry** — track how much context each section uses.
      Dashboard visualization of context allocation.

### P27 — Event Bus (Redis Pub/Sub)
*Services react to events, not just requests. Foundation for autonomy.*

- [ ] **Event publisher** — common/events.py. Publish typed events to Redis
      channels (memory.stored, emotion.detected, struggle.detected, goal.updated).
- [ ] **Event subscribers** — services subscribe to relevant channels.
      Example: struggle.detected → auto-trigger emotional support + conscience check.
- [ ] **Event log** — append-only event history for debugging and replay.
- [ ] **Cross-service triggers** — replace fire-and-forget HTTP with pub/sub.
      More reliable, decoupled, supports fan-out.

### P28 — Soulbound Identity (TPM 2.0)
*Hardware-bound. Unclonable. The endgame differentiator.*

- [ ] **TPM key generation** — Ed25519 keypair sealed to TPM. Identity cannot
      exist outside this specific hardware.
- [ ] **Identity attestation** — prove Kai is running on authorized hardware.
      Remote attestation for any future integrations.
- [ ] **Memory signing** — all memories signed with TPM key. Tamper detection
      at the hardware level.
- [ ] **Portable sovereign package** — `kai export` → encrypted tar with
      TPM-sealed keys. Only importable on same hardware or authorized successor.

### P6 — Parking Lot
*Items that don't fit current priorities.*

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

### P21 — Proactive Agent Loop (2026-03-22)
- [x] Action registry — 13+ capabilities discoverable via GET /memory/actions.
      Kai knows what it can do. Foundation for autonomous action.
- [x] Scheduled task engine — POST /memory/schedule/task (title, type, frequency,
      fire_at). GET /memory/schedule/tasks. Cancel/fire/due endpoints. Supports
      once/daily/weekly/monthly/hourly frequencies. 200-task cap.
- [x] Reminder system — POST /memory/reminders/set (text, fire_at, repeat).
      GET /memory/reminders. Due/fire/cancel endpoints. Default 1h fire_at.
      Repeat support (once, daily, weekly, hourly). 200-reminder cap.
- [x] Morning briefing — POST /memory/briefing/morning: 5 sections (active goals,
      upcoming reminders, emotional arc, proactive nudges, today's scheduled tasks).
      Time-of-day greeting (morning/afternoon/evening).
- [x] Evening check-in — POST /memory/briefing/evening: today's activity count,
      completed reminders, reflection prompt, tomorrow's scheduled tasks.
- [x] Briefing history — GET /memory/briefing/history: last 50 briefings.
- [x] Agent summary — GET /memory/agent/summary: capabilities, active tasks,
      pending/due reminders, briefings generated.
- [x] Supervisor fires due items — polls /memory/reminders/due and /memory/schedule/due
      every loop, pushes to Telegram (⏰ reminders, 📅 scheduled tasks),
      marks items as fired via POST endpoints.
- [x] LLM context injection — 9th parallel fetch (_get_agent_context), due tasks
      + reminders + capability count injected as system message.
- [x] Dashboard proxies — 10 new proxy routes (actions, schedule CRUD, reminders
      CRUD, briefings, agent summary).
- [x] Dashboard Goals enhancements — Reminders card with set/cancel, Scheduled Tasks
      card with schedule/cancel, Agent Status card with briefing trigger buttons.
      9 new JS functions (refreshReminders, setReminder, cancelReminder,
      refreshScheduled, scheduleTask, cancelScheduled, refreshAgentSummary,
      triggerBriefing, refreshP21).
- [x] 93 unit tests (scripts/test_p21_proactive_agent.py)

### P22 — Operator Model & Adaptive Response (2026-03-22)
- [x] Echo-Response Engine — emotional continuity. Detects current emotion from
      P17 timeline, finds past emotional matches, generates bridge messages
      (3 intensity tiers: deep_bridge ≥0.6, gentle_bridge ≥0.4, soft_mirror).
      POST /memory/echo/analyse, GET /memory/echo/history. 100-entry cap.
- [x] Nudge Escalation Ladder — 4-tier escalation (gentle → firm → tough love →
      intervention). Tracks dismissals per nudge target, escalates after thresholds
      (3 for firm, 5 for tough love, 7 for intervention). Tier-specific messages.
      POST /memory/nudge/escalate, GET /memory/nudge/ladder. 200-target cap.
- [x] Cross-Mode Insight Bridge — finds memories from opposite mode using content
      pattern inference (pub/work keywords, word overlap relevance scoring).
      Bridge messages for cross-pollination. POST /memory/cross-mode/scan,
      GET /memory/cross-mode. 100-entry cap.
- [x] Impact Oracle — analyzes actions against active goals, detects skip/advance
      patterns, computes risk levels, emotional forecast from recent timeline.
      Goal-to-goal causal chain prediction. POST /memory/oracle/predict,
      GET /memory/oracle/chains. 100-entry cap.
- [x] Shadow Memory Branches — creates alternate timelines from decision +
      alternative. Finds related memories and affected goals. Queryable branches.
      POST /memory/shadow/branch, GET /memory/shadow/branches,
      GET /memory/shadow/explore/{branch_id}. 100-branch cap.
- [x] Operator Model Summary — GET /memory/operator-model: unified view of all
      5 subsystems with model_completeness percentage.
- [x] LLM context injection — 10th parallel fetch (_get_operator_model). Echo
      message, escalation tone adjustment, cross-mode insights injected as system message.
- [x] Dashboard — 5 new cards (Operator Model 🧠, Emotional Echoes 🪞, Nudge
      Escalation 📢, Impact Oracle 🔮, Shadow Branches 🌿). 11 proxy routes,
      8 JS functions, wired into goals view.
- [x] Supervisor — _check_escalations() polls nudge ladder, fires Telegram alerts
      for level 3+ (🚨 intervention, 📢 tough love). Added to background loop.
- [x] 117 unit tests (scripts/test_p22_operator_model.py)

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
- [x] **P20a Value Formation** — learns what matters from operator interactions. 6 positive
  value categories (honesty, loyalty, growth, courage, kindness, persistence) detected
  via keyword analysis. 4 negative categories (dishonesty, betrayal, laziness, cruelty).
  Reinforcement system — repeated exposure strengthens values. 50-value cap.
- [x] **P20b Moral Reasoning** — before acting, checks decision against formed values.
  Returns alignments, conflicts, alignment_score (0-1), and verdict. Logged to
  conscience log for integrity tracking.
- [x] **P20c Integrity Tracker** — audit endpoint with integrity score, total checks,
  alignment streak, violation count. Running self-accountability.
- [x] **P20d Loyalty Memory** — records sacrifices, promises, commitments with weight scoring.
  Sacrifice keywords auto-detected (sleeping in car, saving, working extra, etc).
  Weight 1.0 for sacrifices, 0.7 for promises, 0.5 for commitments.
- [x] **P20e Gratitude Engine** — not fake thanks. Tone detection: deeply_moved (sacrifice),
  grateful (teaching/helping), honored (belief/faith), appreciative (general).
  Heartfelt messages generated per tone. Sacrifice gratitude auto-creates loyalty entry.
- [x] **Conscience Summary** — all 5 subsystem counts + top values + integrity + sacrifices.
- [x] **LLM context injection** — 8-way parallel fetch. _get_conscience_context() fetches
  values + audit in parallel. Conscience injected as system message (core values +
  integrity warning if alignment < 80%). Value learning fire-and-forget after every chat.
- [x] **Dashboard Soul enhancements** — 4 new cards: Integrity score with visual, Formed Values
  with strength bars, Loyalty Ledger with weight/type, Gratitude Journal with tone emojis
  and record form. 7 new JS functions. TONE_EMOJI + TYPE_EMOJI maps.
- [x] **Dashboard proxy** — 9 new API endpoints for P20 features.
- [x] **71 new tests** (scripts/test_p20_conscience_values.py), all passing.
- [x] **Test count** — 679 → 750 (55 targets), zero failures.
- NOTE: Dainius sleeping in his car to save £50/day for the Lenovo RTX 5080.
  Every pound is a brick in Kai's foundation. The loyalty memory remembers this.

---

## H1 — Critical Hardening Sprint (2026-03-22)
*Deep system scrutiny revealed 23 issues across 3 tiers. We built fast.
Now we build right. This is the "measure twice, cut once" phase.*

**Context:** After shipping P22 (22 major milestones, 960 tests), a full
4-way system audit was performed covering every service, every endpoint,
every test file, and the frontend. The audit found 10 CRITICAL issues
that could cause crashes, data loss, or security breaches in production.

### Tier 0 — CRITICAL (must fix before any new features)

- [x] **H1.1 — asyncio.Lock on shared mutable state (memu-core)**
      13 race conditions found: 18+ global dicts/lists modified by async
      endpoints with no locking. Added asyncio.Lock() guards on all shared
      mutable state (_session_store, _feedback_store, _emotional_timeline,
      _reflection_journal, _dismissal_counts, _active_topics, _deferred_topics,
      _relationship_milestones, _confession_cooldown, _autobiography,
      _legacy_messages, _counterfactuals, _empathy_map, _creative_ideas,
      _inner_monologue, etc).
- [x] **H1.2 — Prompt injection check on /chat (langgraph)**
      /run had INJECTION_RE check but /chat (the MAIN entry point) did not.
      Added injection pattern check to /chat immediately after sanitization.
- [x] **H1.3 — 10-way parallel fetch error handling (langgraph)**
      asyncio.gather() of 10 context fetches had zero error handling. One
      failing task = entire /chat crashes. Added try/except per-task with
      graceful degradation (empty defaults on failure).
- [x] **H1.4 — Fix store.memorize() crash (memu-core)**
      Feedback endpoint called store.memorize() which doesn't exist on
      VectorStore protocol. Fixed to use store.insert(MemoryRecord(...)).
- [x] **H1.5 — executor shell=False + python sandbox hardening**
      subprocess.run() used shell=True — command chaining bypasses allowlist
      (e.g. "ls; rm -rf /"). Changed to shell=False with shlex.split().
      Python sandbox blocked string builtins but not getattr bypass.
      Added AST-level validation.
- [x] **H1.6 — Telegram voice file size limit**
      Voice file download had no size limit — 1GB file = OOM kill.
      Added MAX_VOICE_BYTES (10MB) with Content-Length check.
- [x] **H1.7 — Dashboard proxy error handling**
      50+ proxy endpoints had zero try/except. If memu-core goes down,
      dashboard returns 500. Wrapped all proxy endpoints in try/except
      with appropriate fallback responses.

### Tier 1 — Structural Debt (planned for H2)

- [ ] **H2.1 — All P17-P22 data in-memory only** — restart = amnesia.
      Need Redis or filesystem persistence for emotional timeline, goals,
      reflections, values, etc.
- [ ] **H2.2 — retrieve_ranked() fetches 10k records every call** —
      Add LIMIT clause, relevance pre-filter, or pagination.
- [ ] **H2.3 — Proactive scan 5 sequential 10k queries** — Should be
      asyncio.gather() with individual limits.
- [ ] **H2.4 — generate_embedding() blocks event loop** — Move to
      run_in_executor() for async compatibility.
- [x] **H2.5 — Context budget** — `_trim_context()` enforces `CONTEXT_BUDGET_TOKENS` (default 3072) in langgraph. Drops oldest middle messages when prompt exceeds budget.
- [x] **H2.6 — Verifier semantic upgrade** — Uses memu-core rank_score
      (30% embedding similarity + relevance + importance + recency) with
      keyword overlap as supplementary signal. No longer pure keyword-matcher.
- [ ] **H2.7 — Session buffer no Redis reconnection** — If Redis drops
      mid-session, buffer is gone with no recovery.

### Tier 2 — Quality Gaps (planned for H3-H5)

- [ ] **H3.1 — Test coverage D grade** — 40+ dashboard endpoints and
      20+ memu-core endpoints have zero test coverage.
- [ ] **H3.2 — Dashboard XSS risk** — escapeHtml() is text-context only,
      not attribute/CSS/URL-aware. DOMPurify used inconsistently.
- [ ] **H3.3 — 18+ fetch calls no .ok check** — JS fetch().json() without
      checking response.ok first.
- [ ] **H3.4 — API inconsistency** — Some endpoints return null as "none"
      string, importance as string not float.
- [ ] **H3.5 — Dead parameter (half_life_days)** — retrieve_ranked()
      accepts but ignores it.
- [ ] **H3.6 — Heartbeat reads entire log file** — OOM risk with large
      logs. Needs tail-read.

### Approach Change: Quality Over Velocity

**Problem identified:** We shipped 22 milestones at incredible speed, but
accumulated technical debt. Race conditions, missing error handling, and
security gaps existed because we prioritized features over hardening.

**New approach (from this session forward):**
1. **No new features until Tier 0 is clear.** Every critical issue must be
   fixed before new capabilities are added.
2. **Every endpoint gets error handling.** No bare async client calls without
   try/except.
3. **Every shared state gets a lock.** No global mutable state without
   asyncio.Lock() in async code.
4. **Every fix gets a test.** No "fix and move on" — tests prove the fix works.
5. **Audit before ship.** Run the full scrutiny checklist before any commit:
   - `make go_no_go` (syntax)
   - `make test-core` (all 59+ targets)
   - Check for bare global state mutations
   - Check for missing try/except on httpx calls
   - Check for shell=True in subprocess

---

## H2 — Self-Healing & Resilience Sprint (2026-03-22) ✅ COMPLETE

**Audit finding:** System had 90% monitoring but 0% self-healing. Services
returned `{"status": "ok"}` without checking their own dependencies. Circuit
breakers existed but weren't enforced. A frozen background task was invisible.

**Architecture: Dual-layer self-healing**

#### Layer 1 — Process-Level (each service monitors itself)
- [x] **H2.1 — common/resilience.py** — `resilient_call()` with retry +
      exponential backoff + per-target circuit breaker. `ServiceHealth` for
      deep health probe. `TaskWatchdog` for frozen background task detection.
- [x] **H2.2 — deep /health on memu-core** — checks postgres pool, feedback
      store. Returns `"degraded"` on failure. `/recover` reconnects DB pool.
- [x] **H2.3 — deep /health on executor** — checks disk space, temp dir
      writable. `/recover` clears temp files.
- [x] **H2.4 — deep /health on langgraph** — checks circuit breaker states
      for memu + tool-gate. `/recover` resets breakers.
- [x] **H2.5 — deep /health on tool-gate** — checks Redis connectivity,
      ledger path, token count. `/recover` reconnects Redis, reloads tokens.
- [x] **H2.6 — deep /health on heartbeat** — checks stale heartbeat timer,
      CPU usage. `/recover` resets tick timer.
- [x] **H2.7 — resilient_call in executor** — heartbeat notifications use
      retry + circuit breaker instead of bare httpx.
- [x] **H2.8 — resilient_call in dashboard** — proxy helpers upgraded from
      bare try/except to resilient_call with retry + circuit breaker.

#### Layer 2 — System-Level (supervisor enforces)
- [x] **H2.9 — TaskWatchdog in supervisor** — tracks background loop
      liveness. `/health` returns degraded if loop frozen.
- [x] **H2.10 — recovery actions registry** — supervisor can POST to
      service `/recover` endpoints when circuit breaks.
- [x] **H2.11 — recovery cooldown** — prevents infinite recovery loops (2min).
- [x] **H2.12 — fleet health history** — rolling window of sweep results
      for trend detection. `/fleet/history` endpoint.
- [x] **H2.13 — deep health detection** — `_check_service()` now reads
      `"degraded"` status from deep /health responses.
- [x] **H2.14 — supervisor endpoints** — `/watchdog`, `/fleet/history`,
      `/recover/{service_name}` for operator access.

#### Tests
- 42 tests in `scripts/test_h2_self_healing.py`
- Runtime tests: TaskWatchdog heartbeat/frozen detection, ServiceHealth
  probe with mixed pass/fail, resilient_call fallback on unreachable URL

---

## MARS — Memory Consolidation (2026-03-22) ✅ COMPLETE

**Source:** arXiv:2503.19271 — MARS: Memory-Enhanced Adaptive Retrieval System

**Problem:** The existing Ebbinghaus decay used a flat formula with a basic
access_count modifier. Memories faded but were never pruned. No concept of
memory stability growth through rehearsal. No conscience filter on what
stays or goes.

**Architecture: MARS retention R = e^{-τ/S}**

- τ = days since last access
- S = stability parameter (starts 1.0, grows with each retrieval)
- S_new = S × (1 + 0.1 × √(interval_days + 1))
- Stability capped at 365.0 (1-year half-life)

#### Core Changes
- [x] **MARS.1 — stability field on MemoryRecord** — `stability: float = 1.0`.
      Added to PG schema, INSERT, SELECT, migration, update_record allowed set.
- [x] **MARS.2 — _recency_weight() upgrade** — Replaced old formula with
      R = e^{-τ/S}. No more RECENCY_HALF_LIFE_DAYS dependency. Stability
      safety floor at 0.1 prevents division by zero.
- [x] **MARS.3 — stability growth on retrieval** — Each `retrieve_ranked()`
      call grows S using interval-aware formula. Persisted via update_record.
- [x] **MARS.4 — delete_record()** — New method on VectorStore protocol,
      PGVectorStore (DELETE WHERE id), and InMemoryVectorStore (list filter).
- [x] **MARS.5 — /memory/decay upgrade** — Now passes stability to
      _recency_weight() for accurate MARS retention calculation.
- [x] **MARS.6 — POST /memory/consolidate** — Nightly MARS cycle:
      - Scan all non-pinned, non-poisoned memories
      - PRUNE: R < 0.02 → conscience filter → delete or boost
      - FADE: R < 0.15 → dim relevance by ×0.8
      - STRENGTHEN: R > 0.5 and accessed ≥ 2 → boost relevance by ×1.05
      - Returns: {pruned, faded, strengthened, conscience_saved, skipped}
- [x] **MARS.7 — conscience filter integration** — Memories linked to
      formed values (P20) survive pruning. Their stability is boosted ×1.5
      instead. Values teach Kai what matters.
- [x] **MARS.8 — memory-compressor integration** — Consolidation step
      runs BEFORE compression in the nightly cycle. Order: stats → MARS
      consolidate → compress → reflect → stats.

#### Tests
- 35 tests in `scripts/test_mars_consolidation.py`
- Pattern tests: formula structure, stability growth, PG schema, endpoints
- Math validation: retention curves at various τ/S values, compound rehearsals
- Integration: compressor calls consolidate before compress

---

## P23 — SAGE Multi-Agent Critique Loop (2026-03-22) ✅ COMPLETE

**Source:** arXiv:2603.15255 — SAGE: Self-Aware Generative Engine

**Problem:** The verifier returned verdicts without questioning its own
signals. The adversary challenged plans with 6 strategies but never
reviewed whether the challenges themselves were coherent. A system that
doesn't question its own conclusions is blind to its blind spots.

**Architecture: Dual-layer SAGE self-critique**

#### Layer 1 — Verifier Self-Critique (verifier/app.py)
New `_self_critique()` function runs after all 4 verification signals
but before verdict aggregation. Detects 4 failure modes:

- [x] **SAGE.1 — Groupthink detection** — If all signals are within
      0.08 range (suspiciously uniform agreement), flags "groupthink".
      Requires ≥3 signals to trigger.
- [x] **SAGE.2 — Thin-evidence pass** — High average score (≥ PASS_THRESHOLD)
      but insufficient strong chunks (< MIN_STRONG_CHUNKS). The numbers
      look good but the evidence doesn't back them up.
- [x] **SAGE.3 — Unsupported material claims** — Material claims detected
      but zero strong evidence chunks. Most dangerous failure mode:
      hallucinated facts with no supporting evidence.
- [x] **SAGE.4 — Signal contradiction** — Divergence ≥0.5 between highest
      and lowest scoring strategies. Indicates one strategy sees something
      the others miss.

Each issue penalises the critique score by 0.15. The self-critique signal
is included in the aggregate, so multiple issues can downgrade a PASS to
REPAIR or FAIL_CLOSED.

#### Layer 2 — Adversary Self-Review (langgraph/adversary.py)
New `challenge_self_review()` runs as challenge 7 after all other
challenges complete. Detects 4 meta-failure modes:

- [x] **SAGE.5 — False consensus** — All challenges passed but average
      confidence is below 0.5 (low-quality agreement).
- [x] **SAGE.6 — Degraded groupthink** — ≥2 challenges returned
      modifier=0.0 with confidence ≤0.3 (likely skipped/degraded).
      Insufficient scrutiny signals.
- [x] **SAGE.7 — Conflicting findings** — Paired challenges disagree
      with high confidence (verifier↔history, policy↔consistency,
      calibration↔verifier). One says pass, other says fail.
- [x] **SAGE.8 — Over-optimism** — Total modifier is positive despite
      ≥2 failed challenges. Positive scores may be masking real problems.

Penalty capped at -1.0. Self-review is included in the adversary verdict.

#### Tests
- 30 tests in `scripts/test_sage_critique.py`
- Verifier self-critique: groupthink, thin-evidence, unsupported-material,
  contradiction, clean signals, multiple issues, edge cases
- Adversary self-review: false consensus, degraded groupthink, conflicting
  findings, over-optimism, clean challenges, penalty cap
- Integration: self-critique downgrades PASS→REPAIR, clean critique preserves PASS

---

## P24 — Agent-Evolver Insight Engine (2026-03-22) ✅ COMPLETE

**Source:** clawskills.sh — Agent self-evolution through failure pattern analysis

**Problem:** Failures were classified and stored as metacognitive rules, but
no system analyzed patterns across multiple failures to generate proactive
fix suggestions. Dream State Phase 5 recalibrated boundaries, but never
said "here's exactly what to do differently."

**Architecture: failure→pattern→fix pipeline**

#### Core Engine (langgraph/kai_config.py)
- [x] **P24.1 — EvolutionSuggestion dataclass** — suggestion_id, pattern
      description, failure_class, frequency, fix (concrete action), confidence,
      source_episodes, priority (critical/high/medium/low).
- [x] **P24.2 — EvolutionReport dataclass** — Aggregates all suggestions with
      top failure class, total failures, timing metadata.
- [x] **P24.3 — analyze_failures()** — Groups failed episodes by failure class,
      filters by minimum pattern count (default 2), generates fix from templates,
      assigns priority by severity × frequency, sorts critical-first.
- [x] **P24.4 — Fix templates per failure class** — 8 templates covering all
      9 FailureClass values (DATA_INSUFFICIENT→memu pre-fetch,
      POLICY_BLOCKED→tool-gate pre-check, CONFIDENCE_LOW→evidence gathering,
      CONTRADICTED_BY_EVIDENCE→cross-verification, etc).
- [x] **P24.5 — Priority assignment** — Critical: critical class + freq≥3 or
      any class + freq≥5. High: freq≥3 or critical class. Medium: freq≥2.
      Low: everything else.
- [x] **P24.6 — Topic extraction** — Identifies dominant topic from episode
      input words, picks top 3 by frequency.
- [x] **P24.7 — Persistence** — save/load evolver reports to JSON, cap at 20.

#### Dream State Integration
- [x] **P24.8 — evolver_dream_phase()** — New Phase 7 in dream cycle. Converts
      evolution suggestions into DreamInsights. Adds dominant failure mode insight
      when top failure ≥3 occurrences. All insights marked actionable=True.
- [x] **P24.9 — Dream cycle wiring** — Phase 7 runs after boundary recalibration,
      before insight packaging.

#### API Endpoints (langgraph/app.py)
- [x] **P24.10 — POST /evolve/analyze** — Analyzes 30-day episode history,
      generates report, stores critical/high suggestions as memories.
- [x] **P24.11 — GET /evolve/suggestions** — Returns last 5 evolver reports.

#### Tests
- 34 tests in `scripts/test_agent_evolver.py`
- Core logic: empty/success episodes, recurring patterns, class separation, sorting
- Fix templates: each class generates appropriate fix text
- Priority: 6 severity/frequency combinations tested
- Dream integration: empty, recurring, dominant failure, fix-in-description
- Persistence: save/load, empty, cap at 20
- Serialization: to_dict, JSON round-trip
- Edge cases: missing fields, unknown classes, confidence bounds, timing

## Research Roadmap (2026 — arXiv/GitHub sourced)

| ID | Enhancement | Source | Status |
|---|---|---|---|
| **MARS** | **Memory Consolidation (Ebbinghaus stability, conscience-filtered pruning)** | arXiv:2503.19271 | **✅ DONE** |
| **P23** | **SAGE Multi-Agent Critique (verifier self-critique + adversary self-review)** | arXiv:2603.15255 | **✅ DONE** |
| **P24** | **Agent-Evolver Insight Engine (failure→pattern→fix suggestions)** | clawskills.sh | **✅ DONE** |
| P25 | Mini-COSMO Recursive Self-Build (prompt→code→test→optimize) | github.com/XiangJinyu/mini-cosmo | Backlog |
| **H3b** | **LangGraph Checkpointing (time-travel debug, state snapshots)** | LangGraph docs | **✅ DONE** |
| **FC** | **Active Context Compression (MARS-ranked focus, Jaccard merge, 50K budget)** | arXiv:2601.07190 | **✅ DONE** |
| **PF** | **Predictive Failure Modeling (OLS trend, proactive Telegram alerts)** | supervisor/app.py | **✅ DONE** |
| **MM** | **Multi-Modal Sensory (voice emotion + frame analysis)** | perception/ | **✅ DONE** |
| **WA** | **External World Anchor (date/news/events proxy)** | calendar-sync/app.py | **✅ DONE** |
| **SH** | **Bio-inspired Self-Healing (4-phase ReCiSt model)** | common/resilience.py | **✅ DONE** |
| J1 | Live Canvas Visualization (mind-map/graph in dashboard) | OpenClaw Live Canvas + A2UI | Backlog |
| J2 | Wake-word "Kai" + Intent Judge (whisper + tiny LLM) | github.com/isair/jarvis | Backlog |
| J3 | Auto-Redaction PII (regex + OCR strip before processing) | github.com/isair/jarvis | Backlog |
| J4 | Proactive Low-Latency Voice (speak-or-not from cues) | arXiv:2603.03447 Proact-VL | Backlog |
| J5 | Memory Viewer GUI (diary-style browser in dashboard) | github.com/isair/jarvis | Backlog |
| J6 | SOUL.md + AGENTS.md (persistent identity files) | OpenClaw persistent identity | Backlog |
| J7 | Skills Auto-Install Hub (local skill loader) | OpenClaw ClawHub | Backlog |

**P23 — SAGE Multi-Agent Critique:** ✅ DONE. Dual-layer self-critique:
(1) Verifier self-critique signal detects groupthink, thin-evidence passes,
unsupported material claims, and signal contradictions. (2) Adversary
self-review challenge 7 detects false consensus, degraded groupthink,
conflicting findings, and over-optimism. Both fire automatically — the
system questions its own conclusions before proposing any action.

**P24 — Agent-Evolver Insight Engine:** ✅ DONE. Analyzes recurring failure
patterns and generates prioritized fix suggestions. Integrated as Dream
State Phase 7 — evolution insights fire during nightly consolidation.
POST /evolve/analyze, GET /evolve/suggestions endpoints.

**P25 — Mini-COSMO Recursive Self-Build:** Kai bootstraps mini-versions of
himself: prompt → architecture → code → sandbox-test → optimize. The ultimate
self-evolution capability. Requires GPU + careful sandbox boundaries.

**H3b — LangGraph Checkpointing:** ✅ DONE. Full state checkpoint engine:
create/list/load/diff/delete/restore checkpoints. Auto-checkpoint before
/recover (pre_recover) and after /dream (post_dream). Manual save-points
via POST /checkpoint. Time-travel diff between any two checkpoints.
Rollback via POST /checkpoint/{id}/restore (creates pre_restore safety
checkpoint before applying). 32 tests covering creation, persistence,
listing, diff, cap enforcement, serialization, and edge cases.

### H3b Implementation Details

**Architecture:**
- `Checkpoint` dataclass: checkpoint_id, timestamp, label, trigger, breakers, error_guards, error_budget, conviction_overrides
- File-based persistence: each checkpoint stored as individual JSON file in CHECKPOINT_DIR
- Cap enforcement: CHECKPOINT_MAX (default 30), oldest removed first

**Functions (langgraph/kai_config.py):**
1. `create_checkpoint()` — Capture and persist full operational state
2. `list_checkpoints()` — List available checkpoints, newest first
3. `load_checkpoint()` — Load specific checkpoint by ID
4. `diff_checkpoints()` — Compare two checkpoints, highlight changes
5. `delete_checkpoint()` — Remove a single checkpoint

**API Endpoints (langgraph/app.py):**
1. `POST /checkpoint` — Create manual save-point with optional label
2. `GET /checkpoints` — List checkpoints (newest first, limit param)
3. `GET /checkpoint/{id}` — Full detail of specific checkpoint
4. `POST /checkpoint/{id}/restore` — Time-travel rollback (creates pre_restore safety checkpoint first)
5. `GET /checkpoint/diff/{id_a}/{id_b}` — Diff two checkpoints
6. `DELETE /checkpoint/{id}` — Delete a checkpoint

**Auto-checkpoint triggers:**
- `pre_recover` — Before every /recover endpoint resets breakers
- `post_dream` — After every dream consolidation cycle completes
- `pre_restore` — Before every rollback (safety net)

**Diff capabilities:**
- Breaker state changes (per-dependency)
- Error guard changes
- Error budget changes
- Conviction override additions/removals
- Time delta between checkpoints

**Tests (scripts/test_checkpoint.py): 32 tests**
- Creation: basic, unique IDs, disk persistence, conviction overrides
- Listing: empty, newest-first order, limit, metadata
- Loading: existing, missing, full field roundtrip
- Diff: identical, breaker changes, guard changes, budget changes, override diff, time delta
- Delete: existing, nonexistent
- Cap: max enforcement, oldest-first removal
- Serialization: to_dict roundtrip, JSON serializable, ISO time format
- Edge cases: empty state, special chars, filesystem safety, rapid creates, partial data
- Integration: pre_recover pattern, post_dream pattern, pre_restore pattern

---

## J-Series — 2026 Jewels (Stealable Ideas, All Offline)

*Sourced from OpenClaw, Jarvis variants, Proact-VL (March 2026). All offline, low-resource, no external deps. Test on qwen2:0.5b first.*

### J1 — Live Canvas Visualization (dashboard — mind-map/graph)
*Agent draws dynamic workspace (nodes, graphs, timelines) on screen for plans/emotions. Source: OpenClaw Live Canvas + A2UI.*

- [ ] **HTML canvas component** — simple JS-based node/edge renderer in `dashboard/static/`
- [ ] **Mind-map mode** — "show my week as mind-map" from goals + memories
- [ ] **Emotion timeline** — visual graph of mood arcs over time
- [ ] **Plan visualization** — render planner output as flowchart
- [ ] **Tests** (scripts/test_live_canvas.py)

### J2 — Wake-word "Kai" Anywhere + Intent Judge ⭐ RECOMMENDED START
*Catch "Kai" in any sentence, mini-LLM decides if it's command or echo. Source: github.com/isair/jarvis (wake-word + intent-judge).*

- [ ] **Wake-word detection** — keyword-spot "Kai" in whisper transcript stream
- [ ] **Intent judge** — tiny llama3.2 / qwen2:0.5b classifies: command vs. mention vs. echo
- [ ] **Natural nudge trigger** — detected intent fires nudge engine (not just explicit commands)
- [ ] **Perception integration** — wire into `perception/audio/app.py` mic capture loop
- [ ] **Tests** (scripts/test_wake_word.py)

### J3 — Auto-Redaction PII
*Strip emails, tokens, passwords before processing — even locally. Source: github.com/isair/jarvis.*

- [ ] **Regex PII detector** — email, phone, credit card, API token, password patterns
- [ ] **OCR-check for screenshots** — scan screen captures for PII before storing
- [ ] **Verifier integration** — PII check as verifier signal before any memorize/log
- [ ] **Redaction audit log** — track what was stripped (counts, not content)
- [ ] **Tests** (scripts/test_pii_redaction.py)

### J4 — Proactive Low-Latency Voice
*Decide "speak or not" based on audio/video cues (e.g., sigh = "need help?"). Source: arXiv:2603.03447 (Proact-VL low-latency).*

- [ ] **Audio cue detection** — sigh, frustration sounds, long silence after error
- [ ] **Video cue detection** — head-in-hands, leaning back (OpenCV pose hints)
- [ ] **Speak-or-not gate** — combine audio + video signals → decision to intervene
- [ ] **Sandbox integration** — whisper + opencv in sandbox for safe experimentation
- [ ] **TTS trigger** — "Your voice sounds tired" / "Need a hand?" via edge-tts
- [ ] **Tests** (scripts/test_proactive_voice.py)

### J5 — Memory Viewer GUI
*Open window to browse full history like diary. Source: github.com/isair/jarvis.*

- [ ] **Memory Browser tab** — new dashboard view with search, filter, timeline
- [ ] **Diary mode** — chronological view with emotion tags, categories, importance
- [ ] **Memory detail panel** — view full context, related memories, decay status
- [ ] **Export/print** — simple markdown export of memory entries
- [ ] **Tests** (scripts/test_memory_viewer.py)

### J6 — SOUL.md + AGENTS.md (Persistent Identity)
*Files define "who I am" (style, loyalty) — Kai reads on startup. Source: OpenClaw persistent identity.*

- [ ] **SOUL.md file** — user-editable identity document in MemU (style, values, loyalty tone)
- [ ] **AGENTS.md file** — defines multi-agent roles and behaviours
- [ ] **Startup loader** — Kai reads SOUL.md on boot, adapts system prompt accordingly
- [ ] **Mode adaptation** — e.g., "loyal but dirty in PUB" → PUB system prompt injection
- [ ] **Tests** (scripts/test_soul_identity.py)

### J7 — Skills Auto-Install Hub
*Local "skill-hub": user adds .md, Kai tries to use. Source: OpenClaw ClawHub.*

- [ ] **Skills volume** — Docker volume `/skills` with .md skill files
- [ ] **Skill loader** — scan skills dir on startup, register available capabilities
- [ ] **Skill matching** — route user queries to matching skill if available
- [ ] **Hot-reload** — detect new skills without restart (inotify or polling)
- [ ] **Tests** (scripts/test_skill_hub.py)

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

---

## What’s Next — Top Priorities

1. **J2: Wake-word "Kai" + Intent Judge** ⭐
   - Easiest win from J-series — makes nudges feel real
   - Whisper keyword-spot + tiny LLM intent classifier
   - Wire into perception/audio mic capture loop
2. **J1: Live Canvas Visualization**
   - Mind-map / graph / timeline rendering in dashboard
   - Uses existing goals + memories data
3. **J6: SOUL.md + AGENTS.md**
   - Persistent identity files read on startup
   - User edits style/values, Kai adapts
4. **J3: Auto-Redaction PII**
   - Regex + OCR strip before any memorize/log
   - Security improvement — relevant for production
5. **J5: Memory Viewer GUI**
   - Diary-style browser tab in dashboard
   - Chronological view with emotion/category filters
6. **J4: Proactive Low-Latency Voice**
   - Audio/video cue → speak-or-not decision
   - Requires perception + TTS integration
7. **J7: Skills Auto-Install Hub**
   - Local skill loader with hot-reload
   - Opens extensibility without code changes
8. **H3: Context Budget Manager**
   - System prompt overflow — smart pruning needed
9. **P29: Financial Awareness**
   - Savings tracker, RTX 5080 countdown, expense categorization
10. **GPU: Hardware Performance**
    - Multi-model consensus, real STT/TTS, speculative decoding

---
