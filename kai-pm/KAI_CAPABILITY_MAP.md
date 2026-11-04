# KAI capability map: KAI today vs KAI intended

**Orion | 2026-11-04 | directed by Dainius and Kai (handoff entries 369–372)**

**What this covers:**
- **Code:** branch `claude/project-rework-plan-pgvp35` at `e32488a`.
- **Features:** the 124-feature build list in `kai-pm/KAI_FEATURE_LIST_README_PLUS_KINGSMAN.md` (104 from the README, K1–K20 from Kingsman).
- **Method:** read only. **Nothing was run.** Every Runtime cell says NOT RUN, because runtime testing is on hold.

This map reports and recommends. It changes nothing in KAI: KAI code changes are still on hold until Dainius releases them.

---

## Update 2026-11-04: Priority 1 wiring fixes applied (handoff entries 375–376)

**Authority.** Dainius released the hold for the 8 wiring fixes only, on this branch, plus offline unit tests. The authority is in entry 375 and is word for word.

**What happened to each item:**

| Item | Result | Population counted first (R4/R6) | Test evidence |
|---|---|---|---|
| P1.1 memory label | **FIXED** | 3 readers of the wrong field: agentic/app.py `_recall_memories`, agentic/planner.py preferences, dashboard/app.py Memory Map | new tests fail on the old code, pass on the new |
| P1.2 missing chat function | **FIXED**: `LLMRouter.chat()` added in common/llm.py, matching the `LLMChatFn` contract | 2 call sites (agentic/app.py:827, :864) | 3 new tests fail on the old code, pass on the new |
| P1.4 memory-write format | **FIXED** | 14 memorize call sites, **7** wrong: 5 in agentic, skill-hunter, house-doctor | by reading only; house-doctor tests fail in this sandbox **before and after** (pydantic/typing load error), so not test-verified |
| P1.5 CPU reads 0 | **FIXED** at the producer: sysmetrics also returns top-level `cpu_percent` | 4 consumers (agentic ×2, cortex, perception spine) all expected it | new assert fails on the old code, passes on the new |
| P1.6 finance | **FIXED**: one correct `_tax_year_start()` for all 3 places; chat context and Finance view read the keys the service returns | 3 tax-year lines; 2 consumers | new boundary test passes (on the old code it fails because the helper is absent; the date bug itself was confirmed by reading) |
| P1.7 verifier score | **FIXED**: the adversary passes memu-core's real `rank_score`, `relevance` and `importance`; chunks with no score stay at 0.5 | 1 site | new test fails on the old code, passes on the new; shows a strong record can now reach 0.60 |
| P1.10 skill loading | **HELD, needs a decision.** Only the hidden crash was fixed (agentic/app.py `s["name"]`). Hunted skills are marked `probationary: true`, and the loader ignores that mark. Making them load would put self-acquired skills into live chat without approval. That is a permission question (K19 earned autonomy), not wiring. The 4 repository skills are knowledge notes with no trigger section, so they need writing, not wiring | — | crash: new test fails on the old code, passes on the new |
| P1.11 wrong ports | **FIXED: 19, not the 6 first listed.** Every hard-coded `http://host:port` default (203 literals, outside data/, kai-pm/, docs/, tests and scripts) compared with the port each compose service really listens on | 188 now match, 0 mismatch; 8 hosts are not compose service names, so they were not checked | calibration: the 5 helper-found ports all appeared in the 19 |

**Not touched (still held):**
- P1.3 trust level;
- P1.8 values recording;
- P1.9 dashboard service token;
- P1.12 flags;
- address problems that are routes, not ports: screen-watcher `/screenshot`, calendar `/summary` fields, the audio host name;
- VAT default £85,000.

**Test run.**
- **Setup:**
  - Python 3.11 in a scratch virtual environment. Installing it needed the network, to PyPI.
  - Tests ran with the **network switched off** (`unshare -n`).
  - Tests ran in a copy of the code made with `git archive` **with `data/SOUL.md` left out**, because importing agentic reads `data/SOUL.md` from the working folder (agentic/app.py:1436).
- **Comparison:** 32 test files, each run on the old code and on the new.
  - 28 unchanged.
  - 4 changed only by the new passing tests.
  - The two files that fail before and after (`test_cognitive_mechanisms`: 34 errors; `test_d89_cognitive_depth`: 7 failed) fail identically, from a pydantic load error in this sandbox.
- **`make policy-check`:** each of its 45 commands run separately on both copies gives identical results, apart from the docs-sync count below.
  - The two Item-8 calibration scripts fail identically on both.
  - `check-docs` and `sync_docs --check` failed on both (README code size was already stale), and the new tests added a stale test count. After `make sync-docs`, which changed only README.md and docs/PROJECT_BACKLOG.md metric lines, both pass.
- **Not run:** any live service, docker or runtime test. Runtime testing stays on hold.

**A pattern worth recording (proposed for Kai, not yet in the failure ledger).** Several tests "passed" while the feature was dead, because each test was written against the consumer's own wrong idea of the data rather than the producer's real output:
- the finance fixtures used `cis_summary`;
- the recall fixture used `content.text`;
- the planner fixtures used `content.text`;
- the test stub returned `list_skills() == []`;
- a mocked `LLMRouter` hid the missing `chat`.

That is 5 instances of one mechanism: a test built beside the code instead of from the producer (CLAUDE.md I-8). Each new test above takes its expected shape from the producer's code.

### Update 2026-11-04 (later): values integrity and VAT (handoff entries 377–379)

Kai reviewed commit `6aaf36f` (entry 377). Dainius then approved a narrow integrity fix (entry 378), with branch-only work and offline tests, no network:

| Item | Result | Test evidence |
|---|---|---|
| P1.8 (1): Kai's reply recorded as Dainius's decision | **FIXED**: the chat path no longer calls `record_decision` | new test fails on the old code, passes on the new |
| P1.8 (2): every chat message learned as a "positive" value | **FIXED**: the chat path no longer posts to `/memory/values/learn`. KAI learns no values from chat until an operator-confirmed design exists | same test |
| P1.8 (3): memu-core treated a missing or unknown outcome as "positive" | **FIXED**: only an explicit `positive` or `negative` teaches a value; anything else returns `not_learned`. The 7 existing tests that taught values without an outcome now send `"outcome": "positive"`, because they represent explicit teaching | 2 new tests fail on the old code, pass on the new |
| VAT threshold £85,000 → **£90,000** | **FIXED in all 8 counted places:** financial-awareness, self_emp_advisor and invoice.py defaults; docker-compose.full.yml and docker-compose.sovereign.yml, which override the code default; .env.example; skills/mtd-vat.md (3 lines) | new default test fails on the old code, passes on the new |

**Not included:**
- cleaning up anything already wrongly learned (it lives under `data/`; a separate decision);
- the "review again near £80,000" guidance text in financial-awareness;
- test fixtures and a classifier example phrase that mention 85,000 as arbitrary numbers.

**The counts in §1 and §3 are unchanged.** They describe the code at `e32488a`, before these fixes. No feature is re-classified as GENUINE on the strength of a fix that has not been run live.

---

## 1. The answer in plain words

Think of KAI as a building at second fix. Most rooms are framed and have cable pulled, but many circuits were never connected at the board. A few are connected to the wrong breaker. Some rooms are only a name on the drawing.

| What the code does today | Features | Site meaning |
|---|---|---|
| **GENUINE**: real, connected, does what it says | **15** | room finished and live |
| **PARTIAL**: real code, but part missing, not connected, or broken by a wiring fault | **86** | cable pulled, not terminated or not tested |
| **SIMULATED**: runs, but gives canned or template answers | **8** | dummy fitting on the wall |
| **NAME_ONLY**: only a name, flag, label or empty function | **11** | room on the drawing only |
| **ABSENT**: nothing found where we looked | **4** | not on the drawing at all |
| **Total** | **124** | |

**The most important finding.** A handful of small wiring faults switch off whole groups of features that are otherwise built. Here are three. Each comes from my own reading of the code, and none has been run:

1. **The proactive observer is probably always switched off.** It asks the trust system for permission under a name, `proactive_observation`, that is not on the trust list. An unknown name is treated as needing the top trust level (GUARDIAN), and a new trust record starts at the bottom level. So unless the stored trust level is GUARDIAN (or the trust system fails to load), every cycle is skipped, and with it anomaly alerts, pattern learning, ritual proposals, scheduling and curiosity. Kai's actual stored trust level is in `data/`, which we do not open, so "probably" is the honest word.
   - **Fixing it is your decision, not a typo fix:** choosing which trust level proactive observation needs decides how much KAI may do on its own.
2. **Memories never reach the chat.** The chat looks for memory text under the label `text`, but the memory service files it under `result`. It is like filing drawings in a cabinet labelled "Rev B" while the site team only ever opens "Rev A". A one-line fix, and probably the single most valuable one.
3. **The "team" features call a function that does not exist.** That covers swarm, teammates and the reasoning state machine. Their code is built, but the call that starts them fails at once.

**Two findings that touch your values directly:**
- **Recording the wrong person's decision.** After every chat, KAI records **its own reply** as if it were **your decision**.
- **Learning values without judgement.** Every message you send is learned as a **"positive"** value, whatever it said.

The Kingsman canon says the opposite: "assistant text never becomes operator truth" (K17). This should be fixed before KAI learns anything more from chat.

**What this does not tell you:** whether anything works when switched on. That needs runtime testing, which is on hold. The map says which parts are worth testing first.

---

## 2. Gap report, in priority order

Every item names the features it affects. Each is marked:
- **[Orion checked]**: I read the code myself (details in §6);
- **[helper reading]**: one reading helper's finding, not yet double-checked.

Nothing here has been run.

### Priority 1: small faults that switch off features already built

These are the cheapest wins. Most are a few lines each. **All are code changes, so all wait for Dainius to release the hold.**

| # | Fault | Features affected | Evidence |
|---|---|---|---|
| P1.1 | Memory recall reads `content.text`, but memories are stored as `content.result` (agentic/app.py:1563; memu-core/app.py:1798) | 18, 27, 37, and the "relevant memories" part of every chat answer | [Orion checked] |
| P1.2 | `LLMRouter` has no `chat` method, but swarm and teammates call `_llm.chat` (agentic/app.py:827, :864; common/llm.py) | 37, 47, 51 | [Orion checked] |
| P1.3 | Trust gate name `proactive_observation` is missing, so it falls back to GUARDIAN (agentic/app.py:2064; trust_core.py:71-98, :195-197). **Needs a ruling on the trust level, not just a code edit** | 20, 22, 25, 26, 56, 57, 58, 59, K20 | [Orion checked] |
| P1.4 | Memory writes from the observer and the Letta sync lack the required `timestamp` and `event_type` fields, so they are probably refused silently (memu-core/app.py:130-141) | 12, 20, 25, 26 | model [Orion checked]; callers [helper reading] |
| P1.5 | CPU always reads 0: sysmetrics sends `cpu.percent`, agentic reads `cpu_percent` (sysmetrics/app.py:60; agentic/app.py:1740, :2140) | 20, 22, 58 | [Orion checked] |
| P1.6 | **Finance:** the screens read field names the service never sends, so you never see the numbers. Also: the tax-year start date is wrong on the 1st–5th of every month from May to December (about 40 days a year): on 1 May it counts from 6 April of the year before, so a whole extra year of income is added (financial-awareness/app.py:81, :95). The VAT threshold default is £85,000 (line 36); current HMRC figure to be confirmed (I believe £90,000, not checked here) | 88, K5 | date bug and default [Orion checked]; field names [helper reading] |
| P1.7 | Adversary sends evidence with a fixed score of 0.5, so the verifier's "strong evidence" test (0.60) can never be met on that path: maximum 0.55 (agentic/adversary.py:189; verifier/app.py:238, :52) | 46, 52 | [Orion checked] |
| P1.8 | **Values integrity:** Kai's reply is recorded as Dainius's decision, and every message is learned as a "positive" value (agentic/app.py:2932, :2940-2943) | 1, 5, K17 | [Orion checked] |
| P1.9 | Dashboard calls other services without the service token, so they refuse: Quick Watch, browser, notifications | 65, 87, 94 | [helper reading] |
| P1.10 | Skills never load (hunted skill files lack a "## Trigger" section; folder mismatch `/data/soul/skills` vs `/skills`). This also hides a crash in the self-capability map (agentic/app.py:757 uses `s.name` on dicts) | 85, 76, K19 | crash [Orion checked]; skill format [helper reading] |
| P1.11 | Addresses that don't match (wrong port or route): verifier 8003 vs 8052; executor 8003 vs 8002; camera to agentic 8000 vs 8007; calendar 8030 vs 8055; backup 8048 vs 8054; screen-watcher `/screenshot` route missing; calendar `/summary` fields missing | 23, 24, 26, 52, 58, 63, 86 | [helper reading] |
| P1.12 | Feature flags with no code behind them: 15 of 53 (G6 count). Switching them changes nothing, but the README implies it does | 100, and the features those 15 flags name | [helper reading] |

### Priority 2: connect what is built (integrate)

These are bigger than Priority 1, and some need a design decision first.

- **Hands are not connected** (62, 63, 66, 67, K7, K12): the Tool Gate decides and the executor executes, but nothing links an approval to an execution. The Unified Hunter registry (34 actuators) is meant to be that link and is not cut over. The governed pipeline in `common/` is not in the live path; its defaults are shadow, advisory or off.
- **Approval is not authenticated** (66, K12): any non-empty name is accepted as the approver (common/policy_bridge/approval.py:47-56), and the risk-tier rule is looked up but not applied (:66) [Orion checked]. **A security design item; it has to be designed before the hands are connected.**
- **Built but never called:**
  - confession (4);
  - conscience check (5);
  - self-reflection schedule (10);
  - struggle detection auto-adapt (60);
  - wake word (61);
  - morning and evening briefings (57);
  - anti-annoyance dismiss and Do Not Disturb (55);
  - consensus `/fuse` (50);
  - temporal projection (40);
  - Socratic questions (38).
- **Built but switched off by default:** self-healing recovery (78).
- **Lost on restart:** these are held in memory only:
  - anti-annoyance counters;
  - ritual proposals;
  - struggle cooldown;
  - pending co-signs;
  - vault note index;
  - knowledge-graph source index;
  - most governed-pipeline state.
- **Backup cannot back up** (86): the image has none of `pg_dump`, `psql` or `redis-cli`, there is no schedule, and the "ledger" backup saves only its statistics.

### Priority 3: design needed (nothing, or only a name, exists)

- **ABSENT (4):**
  - 19 per-teammate memory;
  - 73 no single point of failure;
  - K3 succession;
  - K9 lineage manifest.
- **NAME_ONLY (11):**
  - 34 transitive reasoning;
  - 36 Global Workspace;
  - 41 dialectical synthesis;
  - 42 analogy;
  - 43 concept blending;
  - 44 GPU-era foundations;
  - 48 counterfactual rehearsal;
  - 81 synthetic experience;
  - 98 workspace manager;
  - 103 target hardware;
  - K2 stewardship for your daughter.
- **K2, K3 and K9 are mission items** (MD L21–22, L249–268), not technical extras. They need your direction before anyone designs them.

### Priority 4: replace the dummies (SIMULATED, 8)

These are 8, 39, 49, 59, 68, 93, 95 and 97:
- imagination;
- hypothesis;
- tree search;
- curiosity;
- secret rotation;
- impact oracle;
- operator model;
- shadow branches.

Each runs, but gives template or keyword answers. Two are worth flagging:
- **68:** Vault runs in development mode, and the "rotation" changes no key.
- **93, 95 and 97** are one keyword mechanism in memu-core.

### Security items seen on the way (recorded, not new findings)

- **67, executor:** its allowlist includes python3, pip, docker, curl, git and make, so it can run anything. Already registered as KAI-EXEC-003 CRITICAL.
- **91, Telegram:** with `ALLOWED_CHAT_IDS` empty, anyone can chat with it.
- **62, Tool Gate:** the signature does not cover the request parameters, and `cosign=true` in the body approves itself. Already registered as KAI-AUTH-001/004, per the tool-gate audit; the register was not opened.
- **104, zero telemetry:** the egress enforcement script has no caller.
- **87, broker-bridge:** the helper found GET-only Binance calls and no order-placing calls. This is consistent with the hard constraint, but it is a reading, not a test of key isolation.

---

## 3. Counts by organ

| Organ | Features | GENUINE | PARTIAL | SIMULATED | NAME_ONLY | ABSENT | UNKNOWN |
|---|---|---|---|---|---|---|---|
| O1 Soul / Identity / Inner Life | 11 | 2 | 8 | 1 | 0 | 0 | 0 |
| O2 Memory / Relationship / Continuity | 8 | 1 | 6 | 0 | 0 | 1 | 0 |
| O3 Senses / World Awareness | 7 | 0 | 7 | 0 | 0 | 0 | 0 |
| O4 Intelligence / Reasoning | 7 | 3 | 4 | 0 | 0 | 0 | 0 |
| O5 Cognitive Depth / Specialists | 19 | 3 | 7 | 2 | 7 | 0 | 0 |
| O6 Proactivity / Goals / Attention | 9 | 1 | 7 | 1 | 0 | 0 | 0 |
| O7 Governed Hands / Capabilities | 8 | 0 | 7 | 1 | 0 | 0 | 0 |
| O8 Immune System / Doctor / Resilience | 10 | 1 | 8 | 0 | 0 | 1 | 0 |
| O9 Growth / Dream / Evolution | 6 | 2 | 3 | 0 | 1 | 0 | 0 |
| O10 Continuity / Stewardship / Sustainability | 4 | 0 | 4 | 0 | 0 | 0 | 0 |
| O11 Operator Relationship / Mission Control | 9 | 2 | 3 | 3 | 1 | 0 | 0 |
| ENG Engineering base | 6 | 0 | 5 | 0 | 1 | 0 | 0 |
| Kingsman K1–K20 | 20 | 0 | 17 | 0 | 1 | 2 | 0 |
| **Total** | **124** | **15** | **86** | **8** | **11** | **4** | **0** |

Counts are after Orion's three overrides (§6). Generated from the parsed helper tables by `assemble.py`; the §1 summary figures are asserted equal to these.

---

## 4. Proposed additions from the other Kingsman documents (K21–K46): not yet in the baseline

A reading helper read the 21 other Kingsman documents in full. These 26 rows are proposed: **7 marked NEW, 19 extend an existing entry** (counted from its table; its own prose says 10/16, see §6). Not mapped to code. Full rows with sources and quotes are in Appendix A (RECON).

| # | Proposed feature (first sentence) | NEW or extends |
|---|---|---|
| K21 | Contingency and fail-safe library | NEW (sits beside K15, #77) |
| K22 | Lasting record of goals, obligations, promises, deadlines, watches and timers | Extends #53, #58 |
| K23 | Interruption policy | Extends #55, #94, K20 |
| K24 | UNKNOWN is a real answer, and Kai knows its own blind spots | Extends K13, K16 |
| K25 | Predictions stay separate from facts | Extends #40 (sits beside K17) |
| K26 | Operator-availability states | Extends K3, K4 |
| K27 | Trusted human steward roles | Extends K3 |
| K28 | Data stewardship at succession | Extends K3 |
| K29 | Data labels travel with the data | NEW |
| K30 | Outside content is data, never instructions | NEW (sits beside K17) |
| K31 | Protected approval screen | Extends K12, #62 |
| K32 | Durable action workflows | Extends K12, #67 |
| K33 | Network limits per permission | Extends #67 |
| K34 | Tamper-evident audit log | Extends #64 |
| K35 | Automated restore drills | Extends #86, K9 |
| K36 | Safe archive / read-only preservation mode | NEW |
| K37 | Update protection against rollback and freezing | NEW |
| K38 | Ability to change cryptography over Kai's lifetime | NEW (sits beside #68) |
| K39 | Resource and consequence budgets across all organs | Extends K18, #77 |
| K40 | Clear meaning for each timestamp | NEW |
| K41 | The doctor uses the engineering doctrine as diagnostic rules | Extends K14, #70 |
| K42 | Self-check of lineage and learned state | Extends K14 |
| K43 | Mission Control panels | Extends #92 |
| K44 | Deliberation record | Extends #64, #71 |
| K45 | Paper trading and market/strategy/opportunity intelligence | Extends #87, #88 (may already be in the README text) |
| K46 | Self-maintenance and live organ upgrades | Extends K6, K8, K19 |

---

## 5. Feature index (all 124)

One row per feature. "What the code does" and "Gap and next action" are in each helper's table, Appendix A (column G gives which). ⚑ = class changed by Orion (§6).

| # | Feature | Class | Code location | G |
|---|---|---|---|---|
| | **O1 Soul / Identity / Inner Life** | | | |
| 1 | Ohana Core (moral fingerprint / goals) | PARTIAL | agentic/moral_core.py:167-176, :182-196, :288-294 | G1 |
| 2 | PUB/WORK personality modes | PARTIAL | agentic/app.py:1485-1514, :2625-2628; tool-gate/app.py:853-875 | G1 |
| 3 | Soul console view | GENUINE | dashboard/static/app.html:3918 (refreshEQ), :1636-1690; dashboard/app.py:1419-1446 | G1 |
| 4 | Confession engine | PARTIAL | memu-core/app.py:5495-5557; dashboard/app.py:1604-1608 | G1 |
| 5 | Conscience & values | PARTIAL | memu-core/app.py:6764-6940; agentic/app.py:2456-2472, :2932 | G1 |
| 6 | Emotional memory & predictive empathy | PARTIAL | memu-core/app.py:5117-5208; agentic/app.py:2390-2410, :2929 | G1 |
| 7 | Epistemic humility | GENUINE | memu-core/app.py:5390-5483; agentic/app.py:2687-2688, :2788; agentic/conviction.py:158 | G1 |
| 8 | Imagination engine | SIMULATED | memu-core/app.py:6194-6719; agentic/app.py:2434-2453, :2931 | G1 |
| 9 | Narrative identity | PARTIAL | memu-core/app.py:5689-5843, :5853-5934; agentic/app.py:2413-2431, :2796-2808 | G1 |
| 10 | Self-reflection | PARTIAL | memu-core/app.py:5219-5297; dashboard/app.py:1565-1576 | G1 |
| 11 | Temporal self-model (self-assessment) | PARTIAL | heartbeat/app.py:293-363; dashboard/app.py:972-978 | G1 |
| | **O2 Memory / Relationship / Continuity** | | | |
| 12 | Letta archival memory | PARTIAL | letta-agent/app.py:34-142; agentic/app.py:1592-1630; docker-compose.full.yml:519 | G1 |
| 13 | MARS memory decay / consolidation | PARTIAL | memu-core/app.py:1222-1248, :1361-1380, :3635-3734 | G1 |
| 14 | Obsidian Brain / vault-sync | PARTIAL | vault-sync/app.py:151-311; memu-core/app.py:8292-8386; agentic/app.py:1768-1781 | G1 |
| 15 | Knowledge graph (memu-graph, Cognee/Kuzu) | PARTIAL | memu-graph/app.py:92-195; memu-core/app.py:231-264, :1838-1863; agentic/app.py:1569-1589 | G1 |
| 16 | Memory diary / mind-map views | PARTIAL | dashboard/static/app.html:4574-4720, :4938-5120, :5191-5234; dashboard/app.py:1312-1385 | G1 |
| 17 | Memory stores (Postgres/pgvector/TurboVec/Redis) | GENUINE | memu-core/app.py:293-495, :584-850, :1131-1141 | G1 |
| 18 | memu-core memory service | PARTIAL | memu-core/app.py:1736-1835, :1273-1405; agentic/app.py:1555-1566 | G1 |
| 19 | Per-teammate memory slices | ABSENT | agentic/teammates.py:29-103; agentic/app.py:805-830; agentic/swarm_stages.py:110, :260 | G1 |
| | **O3 Senses / World Awareness** | | | |
| 20 | anomaly detection | PARTIAL | agentic/app.py:1897, agentic/app.py:2148, agentic/app.py:2064 | G2 |
| 21 | cortex (perception spine / NPU inference) | PARTIAL | cortex/app.py:462, cortex/app.py:306, agentic/cortex.py:211 | G2 |
| 22 | cross-sensor correlation | PARTIAL | agentic/app.py:1918 | G2 |
| 23 | multi-modal fusion | PARTIAL | perception/camera/app.py:277, perception/camera/app.py:363, fusion-engine/app.py:90 | G2 |
| 24 | sensors & perception services | PARTIAL | sysmetrics/app.py:55, perception/vision/app.py:36, screen-watcher/app.py:84 | G2 |
| 25 | sensory pattern learning | PARTIAL | agentic/app.py:1997, agentic/app.py:2203 | G2 |
| 26 | world state & provenance | PARTIAL | agentic/app.py:2229, agentic/app.py:1691, heartbeat/app.py:412 | G2 |
| | **O4 Intelligence / Reasoning** | | | |
| 27 | 14-way context gather | GENUINE | agentic/app.py:2660, agentic/planner.py:230 | G2 |
| 28 | agentic reasoning service | PARTIAL | agentic/app.py:2555, agentic/app.py:806, agentic/app.py:843 | G2 |
| 29 | conviction scoring (→ information only) | PARTIAL | agentic/conviction.py:177, agentic/conviction.py:31, agentic/app.py:3142 | G2 |
| 30 | local LLM serving (Ollama) | GENUINE | common/llm.py:265, common/llm.py:370, docker-compose.full.yml:795 | G2 |
| 31 | memory-driven planner | PARTIAL | agentic/planner.py:269, agentic/planner.py:230, agentic/app.py:3077 | G2 |
| 32 | priority queue | PARTIAL | agentic/priority_queue.py:70, agentic/priority_queue.py:124, agentic/app.py:787 | G2 |
| 33 | specialist router | GENUINE | agentic/router.py:124, memu-core/app.py:200, memu-core/app.py:1327 | G2 |
| | **O5 Cognitive Depth / Specialists** | | | |
| 34 | D100 transitive reasoning | NAME_ONLY | memu-graph/transitive.py:74-100, 102-123; common/feature_flags.py:89 | G3 |
| 35 | D101 causal world model | PARTIAL | agentic/causal_world_model.py:100-166, 178-209, 259-294; agentic/policy_memory.py:97-165; agentic/app.py:1964-1974, 2276-2286 | G3 |
| 36 | D102 Global Workspace | NAME_ONLY | agentic/global_workspace.py:116-137, 141-149, 175-211; agentic/app.py:2289-2308 | G3 |
| 37 | D90 swarm assembly | PARTIAL ⚑ (was GENUINE) | agentic/app.py:843-901; agentic/swarm_stages.py:494-515; agentic/swarm.py:162-204 | G3 |
| 38 | D92 Socratic self-questioning | PARTIAL | agentic/questioner.py:54-96; agentic/swarm_stages.py:513-514; agentic/app.py:860-868 | G3 |
| 39 | D93 hypothesis engine | SIMULATED | agentic/hypothesis.py:66-89, 93-169; agentic/curiosity.py:77-92; agentic/app.py:2311-2315 | G3 |
| 40 | D94 temporal projection | PARTIAL | agentic/forecaster.py:110-136 | G3 |
| 41 | D95 dialectical synthesis | NAME_ONLY | agentic/dialectic.py:44-63, 66-70 | G3 |
| 42 | D96 analogical reasoning | NAME_ONLY | agentic/analogy.py:56-78 | G3 |
| 43 | D97 concept blending | NAME_ONLY | agentic/concept_blend.py:50-72 | G3 |
| 44 | GPU-era foundations | NAME_ONLY | agentic/counterfactual.py:16-46; tool-gate/app.py:1021-1055; agentic/app.py:2242-2247 | G3 |
| 45 | SAGE critique / self-review | GENUINE | verifier/app.py:351-411, 499-500; agentic/adversary.py:565-671, 731-732 | G3 |
| 46 | adversary engine | GENUINE | agentic/adversary.py:674-772; agentic/app.py:3117-3130; agentic/swarm_stages.py:426-434 | G3 |
| 47 | cognitive reasoning FSM | PARTIAL ⚑ (was GENUINE) | agentic/cognitive_fsm.py:208-312, 96-134; agentic/app.py:870-883 | G3 |
| 48 | counterfactual rehearsal | NAME_ONLY | agentic/counterfactual.py:16-46; agentic/app.py:29, 765 | G3 |
| 49 | debate / tree search | SIMULATED | agentic/tree_search.py:76-185, 211-296; agentic/app.py:3154-3178 | G3 |
| 50 | multi-model consensus / specialist endpoints | PARTIAL | fusion-engine/app.py:90-133, 247-293; agentic/router.py:524-547 | G3 |
| 51 | persistent teammates and reputation | PARTIAL ⚑ (was GENUINE) | agentic/teammates.py:22, 64-103; agentic/app.py:806-834, 904-907; agentic/swarm.py:71-157 | G3 |
| 52 | verifier (SAGE semantic fact-check) | GENUINE | verifier/app.py:162-251, 475-525, 418-452 | G3 |
| | **O6 Proactivity / Goals / Attention** | | | |
| 53 | Goals view (Ohana goals) | PARTIAL | dashboard/app.py:1215 · memu-core/app.py:4046-4183, 4196 · dashboard/static/app.html:3395-3634 | G4 |
| 54 | Watch rule engine | GENUINE | monitor-service/app.py:220-231, 102-132, 156-173 | G4 |
| 55 | Anti-annoyance | PARTIAL | memu-core/app.py:4468-4580, 4698-4736 · supervisor/app.py:268 | G4 |
| 56 | Emergent ritual discovery | PARTIAL | agentic/app.py:1997-2017, 2020-2043 | G4 |
| 57 | Proactive observer / agent | PARTIAL | agentic/app.py:2046-2338 · memu-core/app.py:7177-7191, 7209-7535 · supervisor/app.py:396-441 | G4 |
| 58 | Proactive scheduling | PARTIAL | agentic/app.py:2161-2187 · calendar-service/app.py:181-196 | G4 |
| 59 | Resource-aware curiosity | SIMULATED | agentic/curiosity.py:42-54, 57-99 · agentic/hypothesis.py:93-140 · agentic/app.py:2314 | G4 |
| 60 | Struggle detection | PARTIAL | memu-core/app.py:4877-4969 · dashboard/app.py:1461 · dashboard/static/app.html:4524-4535 | G4 |
| 61 | Wake word & intent | PARTIAL | perception/wake/app.py:75-99, 102-153, 188-219 · agentic/app.py:2535-2551 | G4 |
| | **O7 Governed Hands / Capabilities** | | | |
| 62 | Tool Gate + HMAC/Ed25519 auth | PARTIAL | tool-gate/app.py:541-572, 763-833 · common/auth.py:52-67, 86-133 | G4 |
| 63 | Unified Hunter (34 actuators, not cut over) | PARTIAL | common/actuator_registry/catalog.py:249-280 · common/autonomy/legacy_bridge.py:37-55 · agentic/trust_integration.py:222-247 | G4 |
| 64 | Action audit ledger | PARTIAL | tool-gate/app.py:351-476, 980-997 · ledger-worker/app.py:86-212 | G4 |
| 65 | Browser agent | PARTIAL | browser-agent/app.py:122-277 · dashboard/app.py:2174-2234, 2464-2475 · dashboard/static/app.html:2041-2077 | G4 |
| 66 | Governed decision path (end-goal) | PARTIAL | agentic/app.py:3182-3228 · tool-gate/app.py:595-670 · agentic/trust_integration.py:165-260 | G4 |
| 67 | Sandboxed executor | PARTIAL | executor/app.py:134-200, 240-295, 416-470 | G4 |
| 68 | Secret management & rotation | SIMULATED | docker-compose.sovereign.yml:86-121 · scripts/vault-init.sh · scripts/vault-rotate.sh:6-9 | G4 |
| 69 | Trust negotiation / autonomy requests | PARTIAL | tool-gate/app.py:1012-1059 · common/feature_flags.py:69 | G4 |
| | **O8 Immune System / Doctor / Resilience** | | | |
| 70 | House Doctor / self-diagnosis | GENUINE | house-doctor/app.py:72, :150, :215; agentic/app.py:2263 | G5 |
| 71 | Checkpoint / time-travel debug | PARTIAL | agentic/kai_config.py:1356, :1428; agentic/app.py:3415 | G5 |
| 72 | Heartbeat / pulse | PARTIAL | heartbeat/app.py:186, :224, :150 | G5 |
| 73 | No single point of failure (end-goal) | ABSENT | docker-compose.full.yml:30 (`restart: unless-stopped`); agentic/model_council.py:381 | G5 |
| 74 | Observability (metrics/alerts) | PARTIAL | metrics-gateway/app.py:108, :159, :221; prometheus.yml:12 | G5 |
| 75 | Security self-hacking | PARTIAL | agentic/security_audit.py:119, :416; agentic/introspect_app.py:169; agentic/adversary.py:501 | G5 |
| 76 | Self-capability map | PARTIAL | agentic/app.py:724–779 | G5 |
| 77 | Structured errors / circuit breakers | PARTIAL | common/errors.py:23; common/resilience.py:35; agentic/app.py:160 | G5 |
| 78 | Supervisor self-healing and recovery log | PARTIAL | supervisor/app.py:188, :159; memu-core/app.py:1510 | G5 |
| 79 | System FSM | PARTIAL | agentic/system_fsm.py:44; agentic/app.py:2650, :2320 | G5 |
| | **O9 Growth / Dream / Evolution** | | | |
| 80 | Agent-Evolver | PARTIAL | agentic/kai_config.py:1130, :1065, :1255 | G5 |
| 81 | D99 synthetic experience | NAME_ONLY | agentic/synthetic_experience.py:52, :62 | G5 |
| 82 | Agentic cold path (dream/evolve/audit) | GENUINE | agentic/introspect_app.py:58, :117, :169; dashboard/app.py:985 | G5 |
| 83 | Capability gap logging | GENUINE | agentic/app.py:2341–2369, :2657, :1891 | G5 |
| 84 | Dream state / dream cycles | PARTIAL | agentic/kai_config.py:877–974; agentic/introspect_app.py:58 | G5 |
| 85 | Skills hub / skill hunter / provenance | PARTIAL | skill-hunter/app.py:157, :211, :314; agentic/router.py:644 | G5 |
| | **O10 Continuity / Stewardship / Sustainability** | | | |
| 86 | backup service | PARTIAL | backup-service/app.py:47-68, :111-138; backup-service/Dockerfile:1-6 | G6 |
| 87 | broker-bridge (Binance, read-only) + Broker view | PARTIAL | broker-bridge/app.py:60-81, :121-220; dashboard/app.py:2536-2569 | G6 |
| 88 | financial awareness / UK self-employment advisor | PARTIAL | financial-awareness/app.py:394-404; agentic/app.py:2718-2740; kai-advisor/app.py:58-70 | G6 |
| 89 | sovereign offline operation (end-goal) | PARTIAL | common/llm.py:64-67; output/tts/app.py:3-6, :104; docker-compose.sovereign.yml:622-628 | G6 |
| | **O11 Operator Relationship / Mission Control** | | | |
| 90 | D98 cognitive fingerprint | PARTIAL | agentic/app.py:2576-2581; agentic/cognitive_fingerprint.py:96-111, :126-150 | G6 |
| 91 | Telegram interface | GENUINE | telegram-bot/app.py:141-167, :203-298, :388-419 | G6 |
| 92 | dashboard / operator console (→ Mission Control) | GENUINE | dashboard/app.py (189 `@app.` routes); dashboard/static/app.html:33-93, :3272-3307 | G6 |
| 93 | impact oracle | SIMULATED | memu-core/app.py:7974-8070 | G6 |
| 94 | notifications | PARTIAL | output/notify/app.py:61-136; dashboard/app.py:2355-2390; common/resilience.py:35-80 | G6 |
| 95 | operator model (P22) | SIMULATED | memu-core/app.py:7645-7720, :5131-5144; agentic/app.py:2497-2531 | G6 |
| 96 | presence: chat / voice / TTS / avatar | PARTIAL | output/tts/app.py:88-121; output/avatar/app.py:29-38; docker-compose.minimal.yml:346 | G6 |
| 97 | shadow branches | SIMULATED | memu-core/app.py:8111-8215 | G6 |
| 98 | workspace manager | NAME_ONLY | workspace-manager/app.py:12-14 | G6 |
| | **ENG Engineering base** | | | |
| 99 | engineering toolchain / CI gates | PARTIAL | Makefile:420-425, :830-857; .github/workflows/core-tests.yml:110-115, :318-352; .pre-commit-config.yaml:7-58 | G6 |
| 100 | feature-flag system (53 flags) | PARTIAL | common/feature_flags.py:30, :102-115; agentic/app.py:759 | G6 |
| 101 | honest status / build &amp; test substrate | PARTIAL | scripts/sync_docs.py:24-80, :297-315; docker-compose.minimal.yml | G6 |
| 102 | private overlay network (tailscale) | PARTIAL | docker-compose.sovereign.yml:52-84 | G6 |
| 103 | target hardware (Flow Z13 / Strix Halo) | NAME_ONLY | common/gpu_utils.py:35-58; common/runtime.py:36-42; agentic/cortex.py:211-223 | G6 |
| 104 | zero telemetry / local core | PARTIAL | memu-core/Dockerfile:70; docker-compose.full.yml:500; scripts/enforce_egress.sh:1-40 | G6 |
| | **Kingsman K1–K20** | | | |
| K1 | Protect and care within earned authority | PARTIAL | agentic/trust_integration.py:165; agentic/moral_core.py:213; agentic/trust_core.py:71 | G7 |
| K2 | Stewardship for Dainius's daughter | NAME_ONLY | agentic/trust_core.py:95; agentic/wisdom_graph.py:50; agentic/wisdom_ingestion.py:69 | G7 |
| K3 | Succession | ABSENT | none | G7 |
| K4 | Keeps running when Dainius is unavailable | PARTIAL | supervisor/app.py:100,159; backup-service/app.py:168,215; common/resilience.py:186 | G7 |
| K5 | Economic self-sufficiency, separate from family assets | PARTIAL | agentic/paper_trader.py:86; financial-awareness/app.py:366; broker-bridge/app.py:121 | G7 |
| K6 | Long-horizon survival | PARTIAL | scripts/auto_rotate_hmac.py:1; scripts/backup_offsite.sh:1; common/llm.py:57 | G7 |
| K7 | One KAI, one governed decision path | PARTIAL | scripts/security/check_architecture_rules.py:40; agentic/trust_integration.py:222; common/autonomy/legacy_bridge.py:140 | G7 |
| K8 | Replaceable organs | PARTIAL | common/llm.py:196,220; common/model_registry.py:1; common/contracts/base.py:88 | G7 |
| K9 | Lineage Manifest / Identity Ledger | ABSENT | none | G7 |
| K10 | Three kinds of state with their own change rules | PARTIAL | scripts/test_invariant_guards.py:1; scripts/security/check_doctrine_integrity.py:1; common/autonomy/evidence_service.py:48 | G7 |
| K11 | Growth without architectural amnesia | PARTIAL | .claude/skills/kai-handoff/handoff.py; scripts/security/check_doctrine_integrity.py:1; agentic/introspect_app.py:58 | G7 |
| K12 | Kingsman control loop | PARTIAL | common/vertical_slice/paper_trade_slice.py:143,283; agentic/app.py:637 | G7 |
| K13 | Evidence Plane | PARTIAL | common/contracts/base.py:70,88; common/world_state/snapshot_store.py:44; trust-ledger/ledger.py:94 | G7 |
| K14 | A4 structural self-diagnosis | PARTIAL | scripts/security/report_runtime_topology.py:1; scripts/security/check_compose_drift.py:1; scripts/security/check_architecture_rules.py:1 | G7 |
| K15 | One recovery loop | PARTIAL | agentic/app.py:2046; house-doctor/app.py:150,215; supervisor/app.py:159 | G7 |
| K16 | Truthful degradation | PARTIAL | common/degraded.py:86,104,214; common/runtime.py (CircuitBreaker); supervisor/app.py:188 | G7 |
| K17 | Values kept separate from facts | PARTIAL | common/policy_bridge/assessment.py:155; common/autonomy/authority.py:62; agentic/app.py:2940 | G7 |
| K18 | Model runtime manager and hardware modes | PARTIAL | agentic/model_selector.py:114; agentic/cortex.py:211; sysmetrics/app.py:142 | G7 |
| K19 | Earned autonomy | PARTIAL | common/autonomy/authority.py:38; common/autonomy/legacy_bridge.py:55,140; skill-hunter/app.py:210 | G7 |
| K20 | Proactive organism loop | PARTIAL | agentic/app.py:2046,2064,2275 | G7 |

---

## 6. How this was made, its limits, and disclosures

**Readers.** Eight reading helpers worked under one brief, banked as CAPMAP-BRIEF (entry 371):
- read only;
- no runtime;
- never touch `data/SOUL.md`;
- bounded negatives.

The work was split this way:
- G1–G6 took the README features by organ.
- G7 took K1–K20.
- One helper (RECON) read the 21 other Kingsman documents for missing features.

G1, G2 and G4 stopped at the 5-hour usage limit. They were resumed with their saved work on 2026-11-04 and finished. All eight outputs are in Appendix A word for word, with a SHA-256 for each.

**Parsing.** A script parsed the eight tables into 124 rows, one per feature, and checked them:
- none missing;
- none extra;
- every class one of the six allowed.

Before my overrides the counts were GENUINE 18, PARTIAL 83, SIMULATED 8, NAME_ONLY 11 and ABSENT 4.

**My overrides (3).** G3 classed 37 (swarm), 47 (reasoning state machine) and 51 (teammates) as GENUINE. I re-read the code and changed all three to PARTIAL:
- the swarm request builds its pipeline with `llm_chat_fn=_llm.chat` (agentic/app.py:864);
- the teammate request calls `_llm.chat` (agentic/app.py:827);
- `LLMRouter` (common/llm.py) has no `chat` method and no `__getattr__`.

So `/chat/swarm` raises an error before the state machine is built (:871), and `/chat/teammate` returns HTTP 500. In `agentic/`, outside tests, `CognitiveFSM(` has no other caller. The code of all three is real; their only entry points fail. Found by reading; not run. Spot-check row 12.

**Spot check.** I checked 12 helper claims myself, choosing the ones that drive Priority 1; this was not a random sample. Results:
- 11 confirm a helper's claim, several with a narrower scope, stated per row;
- 1 is my own finding (the overrides above).

The full table is below, word for word.

**Kingsman additions K21–K46 (§4).**
- **The count.** RECON's own sentence says "10 are new and 16 extend". Its table marks **7 NEW** (K21, K29, K30, K36, K37, K38, K40) and **19 extend**. The table is the evidence, so §4 uses 7 and 19.
- **A limit.** RECON compared against the short feature names, not README.md, which it treated as out of bounds. So some "extend" rows may already be described in the README text; RECON names K45 (paper trading) as the likeliest.
- **Status.** These 26 are **proposed**. They are not added to the 124-feature baseline and not yet mapped to code. Kai's direction was "add only genuine differences", so adding them is for Kai to review and Dainius to decide.

**Search-rule disclosures (helpers).** The brief's rule 1 was mine, and it was internally inconsistent. It said "never run any search over the whole repository", but also that a Grep at the root was acceptable "with the glob `!data/**`". Each helper's own account:
- **G5: broke the rule twice.** It ran `grep -rln … --include=*.py .` from the repository root, and a Grep with `glob: **/*.py` at the root without excluding `data/**`. Both reached `data/`, but matched `.py` files only, so `data/SOUL.md` (a `.md` file) was not read, and no `data/` result was used.
- **G3:** one search over the whole repository with `!data/**` excluded (allowed by the brief's second sentence, against its first).
- **G4:** several root-path searches, each with an exclusion glob that included `data/**`, or with a list of named directories.
- **G1:** `grep -r` over a list of nearly all top-level directories, with `data/` excluded.
- **G7:** two single-file greps on root files (docker-compose.full.yml, Makefile).
- **G2:** greps of the three named root compose files.
- **G6:** no root search.

No helper reports opening, listing or searching `data/SOUL.md`. That rests on their own accounts; I have no independent record of their tool calls. My own checks for this map read named files only.

**Not looked at.**
- `data/` (by rule), so the stored trust level, persona files and trust record were not seen;
- test suites;
- README.md text, for the reconcile helper;
- per-service internals where a helper says so in its notes.

**Runtime.** Nothing was run. Every row's Runtime is NOT RUN; five rows also cite `output/health_scorecard.json` (2026-02-25), which records only `/health` liveness answers.

### Spot-check table (verbatim)

# Orion spot-check of helper references (2026-11-04, branch HEAD e32488a, working tree)

Method: Read/Grep on the named files only (paths given explicitly; no root or data/ search).
Sample: 7 helper claims chosen because they drive the quick-fix list. Not a random sample.

| # | Helper claim | Source | Checked | Result |
|---|---|---|---|---|
| 1 | Tax-year start bug `today.month >= 4 and today.day >= 6` | G6 #88 | financial-awareness/app.py:81, :95 | CONFIRMED. Example: on 1 May, day 1 < 6, so the year start falls back a year and income from the whole previous tax year is counted. (Line 212 not opened.) |
| 2 | VAT threshold default 85000 | G6 #88 | financial-awareness/app.py:36 | CONFIRMED the default is 85000. That the current UK threshold is £90,000 is the helper's and my own knowledge, not checked against an HMRC source here. |
| 3 | Adversary sends rank_score 0.5, so the verifier's strong-chunk score cannot reach 0.60 | G3 #46/#52 | agentic/adversary.py:188-191; verifier/app.py:225-243, :52 | CONFIRMED arithmetic: 0.5*0.6 + overlap<=1*0.2 + relevance 0*0.1 + importance default 0.5*0.1 = at most 0.55 < 0.60. So strong_chunks stays 0 on this path. The helper's "can never PASS" consequence was NOT traced further by me. |
| 4 | `s.name` on dicts in self-capability map | G5 #76 | agentic/app.py:757; agentic/router.py:723-733 | CONFIRMED list_skills() returns dicts. AttributeError fires only if at least one skill is loaded. If G5 #85 is right that no hunter skills load, the two defects mask each other (endpoint returns 0 skills, no crash). |
| 5 | K17: Kai's own reply recorded as operator decision; user message learned as "positive" value | G7 K17 | agentic/app.py:2932, :2940-2943 | CONFIRMED: `decision=response[:300]`; `"outcome": "positive"` for every chat turn. |
| 6 | Approval unauthenticated; auto-approve stamps "dainius" | G7 | common/policy_bridge/approval.py:47-89; common/vertical_slice/paper_trade_slice.py:110 | CONFIRMED: any non-empty approver string is accepted (docstring says "authenticated"); APPROVAL_MATRIX entry fetched at :66 but never used to check scope. Default approver "dainius" at paper_trade_slice.py:110 (a parameter default; whether named "auto_approve" not checked). |
| 7 | `proactive_observation` not in CAPABILITY_GATES → GUARDIAN | G7 K20 | agentic/app.py:2064-2070; agentic/trust_core.py:71-98, :195-197; agentic/trust_integration.py:185-191 | CONFIRMED chain: unknown capability defaults to GUARDIAN; gate_autonomous_action calls can_do; the loop sleeps when not allowed. So the proactive observer is suppressed every cycle unless trust level is GUARDIAN (or the trust core fails to load, which fails open for this capability). Helper had marked the last step "unverified"; now verified by reading. Not run. |

Result: 7 of 7 claims confirmed by reading; 2 carry a narrower scope than the helper stated (#3 consequence untraced, #4 masked). Nothing was run.

## Second batch (after G1/G2 returned)

| # | Helper claim | Source | Checked | Result |
|---|---|---|---|---|
| 8 | LLMRouter has no `chat` method, so /chat/teammate and /chat/swarm break | G2 note 4 | common/llm.py (all `class`/`def` lines, no `__getattr__`); agentic/app.py:827, :1421 | CONFIRMED by reading: `_llm = LLMRouter()`; methods are available, stub_mode, query, query_multi, _live_query, _stub_response, stream. `_llm.chat(...)` at :827 would raise AttributeError. Only line 827 checked, not every call site. Not run. |
| 9 | memu-core MemoryUpdate requires timestamp and event_type | G1 #12, G2 note 2 | memu-core/app.py:130-141 | CONFIRMED the model has required `timestamp: str` and `event_type: str` and no `content` field. That each named caller omits them and gets a 422 is the helpers' reading; I did not open each caller. |
| 10 | CPU always reads 0 | G2 note 3 | sysmetrics/app.py:59-60; agentic/app.py:1740, :2140 | CONFIRMED for agentic: sysmetrics returns `cpu.percent`; agentic reads top-level `cpu_percent` (default 0). RAM read `memory.percent` is consistent. cortex and common consumers not opened. |
| 11 | Recall reads `content.text`/`content.query`, but memu-core stores `content.result`, so memories never reach chat | G1 #18, note 1 | agentic/app.py:1555-1566; memu-core/app.py:1793-1803 | CONFIRMED for the main writer (/memory/memorize, line 1798 `"result"`). G1's other nine writer lines not opened by me. Records written by any other path with a `text` key would still be recalled; I did not search for one. |
| 12 | `/chat/swarm` and the reasoning FSM also fail (extends #8) | Orion, from #8 | agentic/app.py:843-901 (read in full), :864, :871; Grep `CognitiveFSM\(` in agentic/ (non-test): only :871 | FOUND by Orion, not by G3: `llm_chat_fn=_llm.chat` at :864 is evaluated before the FSM is built at :871, so with FF_SWARM on the request raises AttributeError. CognitiveFSM has no other non-test caller in agentic/. Teammate chat (:827) catches the error and returns HTTP 500. This overrides G3's GENUINE for #37, #47, #51. Not run. |

Totals: 12 checks. 11 confirm a helper claim (several with narrower scope, stated per row); 1 (#12) is my own finding that overrides three G3 classifications.

---

## Appendix A: helper outputs, word for word

Each block is the helper's final message as returned, extracted from the session transcript. SHA-256 is of the saved file (UTF-8, trailing newline).

### G1

SHA-256 `9a623130fd1bf683dca5d8eee3c9cfd049f29fccbd75afb0b249eeef030ef91b` · 14255 bytes

<details><summary>show</summary>

Capability map for G1 (O1 Soul/Identity, O2 Memory), features 1–19.

| # | Feature | Code location (file:line, at most 3) | Class | What the code actually does (one line) | Runtime | Gap and next action |
|---|---|---|---|---|---|---|
| 1 | Ohana Core (moral fingerprint / goals) | agentic/moral_core.py:167-176, :182-196, :288-294 | PARTIAL | `inject_into_prompt` returns the prompt unchanged and `can_operate()` always returns False. `record_decision` runs on every chat even with the flag off (agentic/app.py:2940), but it saves Kai's own reply under the key "general", not the operator's decision. `evaluate_action_alignment` is real keyword scoring, called from moral_imagination.py:230 and trust_integration.py:101. | NOT RUN | finish: learning should record operator decisions and real situation types; the injection is a stub; FF_OHANA_CORE only gates the no-op injection |
| 2 | PUB/WORK personality modes | agentic/app.py:1485-1514, :2625-2628; tool-gate/app.py:853-875 | PARTIAL | There are two fixed system prompts, chosen per request. tool-gate has a real schedule (WORK Mon–Fri 08–18). The dashboard always sends `mode` from its browser storage, default 'PUB' (app.html:2097), so the schedule never applies to dashboard chat. Ctrl+Shift+M toggle is real (app.html:3300). | NOT RUN | integrate: the dashboard should leave `mode` empty unless the operator has overridden it |
| 3 | Soul console view | dashboard/static/app.html:3918 (refreshEQ), :1636-1690; dashboard/app.py:1419-1446 | GENUINE | The Ctrl+8 'eq' view (titled "Soul") loads mood, emotion arc, relationship and milestones, domain confidence and reflections from real memu-core endpoints. SOUL.md and AGENTS.md editors read and write through agentic /soul and /agents-registry (agentic/app.py:464-507). | NOT RUN | retain (the agentic prompt only takes the first 2000 characters of SOUL.md, app.py:1523) |
| 4 | Confession engine | memu-core/app.py:5495-5557; dashboard/app.py:1604-1608 | PARTIAL | It finds up to 3 same-category "keeper" memories and returns a canned "I gave advice that might have been off" message, with a 1h cooldown. Nothing calls it: no caller found in the code directories searched (see note 10), and the dashboard UI never fetches /api/confess. It does not check whether the advice was actually wrong. | NOT RUN | integrate: call it when a correction arrives; finish: check the old advice against the correction |
| 5 | Conscience & values | memu-core/app.py:6764-6940; agentic/app.py:2456-2472, :2932 | PARTIAL | Every chat message is learned as a value with `outcome` hard-coded "positive", so negative values are never learned. Values go into the chat prompt. /memory/conscience/check is only reachable through a dashboard proxy the UI never calls, so integrity stays 1.0 and the integrity warning cannot fire. Loyalty and gratitude are dashboard-only. FF_CONSCIENCE_FILTER has no consumer and nothing gates actions. | NOT RUN | finish: real outcome signal, and call the conscience check before actions; integrate the flag |
| 6 | Emotional memory & predictive empathy | memu-core/app.py:5117-5208; agentic/app.py:2390-2410, :2929 | PARTIAL | Keyword counting over 7 emotions plus neutral; no model is used, despite the README's qwen note. The timeline is kept in Redis and the mood/arc goes into chat. Predictive empathy is only the `emotional_context` key at agentic/app.py:2242, and FF_PREDICTIVE_EMPATHY has no consumer. | NOT RUN | finish: model-based detection; predictive empathy is name-only |
| 7 | Epistemic humility | memu-core/app.py:5390-5483; agentic/app.py:2687-2688, :2788; agentic/conviction.py:158 | GENUINE | Per-domain confidence comes from correction and feedback counts. A low-confidence warning is added to the chat prompt and also lowers the conviction gate. The heuristic is limited: domains are only the 8 construction categories plus "general", and the feedback store is in-process (memu-core/app.py:4978). | NOT RUN | retain; finish: persist the feedback store, broaden the domains |
| 8 | Imagination engine | memu-core/app.py:6194-6719; agentic/app.py:2434-2453, :2931 | SIMULATED | Counterfactual and creative-synthesis outputs are template strings (domain pairs chosen with `random`). The "inner monologue" is just the first 200 characters of Kai's reply, typed by keyword. Empathize is a keyword/length heuristic (this one is wired into chat). FF_IMAGINATION_ENGINE has no consumer. | NOT RUN | replace with LLM-backed reasoning, or retain as a labelled heuristic |
| 9 | Narrative identity | memu-core/app.py:5689-5843, :5853-5934; agentic/app.py:2413-2431, :2796-2808 | PARTIAL | The identity narrative is template sentences filled with real counts (days, memories, corrections, top categories). Story arcs are windows labelled by correction rate. Autobiography entries are the *operator's* message, chosen by keyword significance. Legacy and future-self are dashboard-only. FF_NARRATIVE_IDENTITY has no consumer. | NOT RUN | finish: real autobiographical synthesis; integrate the flag |
| 10 | Self-reflection | memu-core/app.py:5219-5297; dashboard/app.py:1565-1576 | PARTIAL | Builds a strengths/weaknesses entry from feedback, emotion and correction counts. It only runs when someone clicks the dashboard button (/api/reflect). memory-compressor calls a different `/memory/reflect` (introspect insights), not this one. | NOT RUN | integrate: schedule it, and feed it into behaviour beyond the identity text |
| 11 | Temporal self-model (self-assessment) | heartbeat/app.py:293-363; dashboard/app.py:972-978 | PARTIAL | Compares the current reading with the *last call's* snapshot stored in /tmp, not with last week. `total_memories` reads keys `total_memories`/`total`, but memu-core stats return `records` (memu-core/app.py:1994), so it is always 0. CPU, error and uptime figures are real. FF_SELF_ASSESSMENT has no consumer. | NOT RUN | finish: fix the key, use a real weekly window, persist the history |
| 12 | Letta archival memory | letta-agent/app.py:34-142; agentic/app.py:1592-1630; docker-compose.full.yml:519 | PARTIAL | A real Letta client wrapper (run, archival export). agentic calls it only when FF_LETTA_TASKS is on (default off), and the service is behind the `introspection` profile. The sync posts `{content, category}` to /memory/memorize, which requires `timestamp` and `event_type` (memu-core/app.py:130-141). Inference: the request will be refused and the status is never checked, so sync fails silently. | NOT RUN | finish: fix the sync payload and check the response; verify letta==0.16.8 still provides `create_client` (unverified, not run) |
| 13 | MARS memory decay / consolidation | memu-core/app.py:1222-1248, :1361-1380, :3635-3734 | PARTIAL | Real R=e^{-τ/S} decay with prune, fade, strengthen and a value-keyword survival filter, run every 24h and on a record-count watermark by memory-compressor (memory-compressor/app.py:76-218). Defects: line 1365 sets `last_accessed` before the interval is computed, so the interval is always 1.0; τ is measured from creation, not last access; the filter reads a `triggers` key that values never have. FF_MARS_CONSOLIDATION has no consumer. memory-compressor is defined only in docker-compose.full.yml (grep of minimal/sovereign compose: no match). | NOT RUN | finish: fix the interval and τ; integrate the flag; add the compressor to the other stacks |
| 14 | Obsidian Brain / vault-sync | vault-sync/app.py:151-311; memu-core/app.py:8292-8386; agentic/app.py:1768-1781 | PARTIAL | The watcher, SHA256 skip and ingest into memu-core memory records are real. "cognee:" ids are labels only (no Cognee call). Wikilinks are stored but never turned into graph links. Vault search runs over an in-process dict that is lost on restart. The 4 templates are not referenced in vault-sync code. agentic's /vault/export proxy sends no auth header to the auth-protected /export, and no caller exports the ≥9.0 insights. FF_VAULT_CONTEXT literally searches for the word "recent". | NOT RUN | finish: persist the note index, add auth to the export, auto-export insights, use the templates |
| 15 | Knowledge graph (memu-graph, Cognee/Kuzu) | memu-graph/app.py:92-195; memu-core/app.py:231-264, :1838-1863; agentic/app.py:1569-1589 | PARTIAL | Real Cognee ingest, query and forget, with fire-and-forget fan-out from memu-core. The service is behind the `introspection` profile. The source index used for forget is in-memory. full.yml defaults FF_GRAPH_INGEST=true for memu-core only (line 140); agentic does not get it, so (inference) chat never queries the graph. The "Memory Map" view does not use memu-graph (see #16). | NOT RUN | integrate: make the flag consistent across services, persist the source index, point Memory Map at the graph |
| 16 | Memory diary / mind-map views | dashboard/static/app.html:4574-4720, :4938-5120, :5191-5234; dashboard/app.py:1312-1385 | PARTIAL | The D3 Canvas and Memory Map are real, but the Map is category hubs built from memu-core /memory/retrieve, not memu-graph. Diary "recent" is a semantic search on a fixed phrase, not a date listing. Emotion badges read `content.emotion`, which memu-core writers never set. | NOT RUN | finish: a real chronological listing; use memu-graph data for the Map |
| 17 | Memory stores (Postgres/pgvector/TurboVec/Redis) | memu-core/app.py:293-495, :584-850, :1131-1141 | GENUINE | PGVectorStore does real cosine search (`<=>`). TurboVecStore is an IdMapIndex plus Postgres metadata. Redis (memu-core/app.py:2056) holds the session buffer and P17–P22 state, with an in-process fallback. With VECTOR_STORE unset the code defaults to the in-memory store; compose sets turbovec (full/minimal) or postgres (sovereign). The lakeFS client is a stub (STUBS S5). | NOT RUN | retain; make the in-memory default fail loudly |
| 18 | memu-core memory service | memu-core/app.py:1736-1835, :1273-1405; agentic/app.py:1555-1566 | PARTIAL | memorize (PII redaction, auto category and importance, embedding) and multi-signal ranked retrieval are real, and the hot/cold split exists (introspect_app.py:85-99). However, agentic `_recall_memories` reads `content.text`/`content.query`. Every memu-core writer content dict I inspected stores `content.result` (see note 1). So the "Relevant memories" block in /chat, and the swarm recall (app.py:861), get nothing. | NOT RUN | integrate: read `content.result` in `_recall_memories` (highest-value single fix) |
| 19 | Per-teammate memory slices | agentic/teammates.py:29-103; agentic/app.py:805-830; agentic/swarm_stages.py:110, :260 | ABSENT | Teammates get a system prompt and world state only. The swarm calls the shared, unscoped `_recall_memories(ctx.query)`. Searched agentic/teammates.py, swarm.py, swarm_stages.py for memory/retrieve/_recall_memories/specialty/slice; memu-core/app.py for "teammate" (0 hits); and agentic, common, memu-core for teammate near memory/retrieve (only the swarm_stages lines above). | NOT RUN | build: specialty-scoped top-k retrieval |

**Notes**
1. **Memories never reach the chat (#18, #19).** memu-core stores the memory text under `result` at every writer content dict I inspected in memu-core/app.py (lines 1798, 2907, 2981, 3048, 3543, 3811, 4067, 5029, 5052, 8309). `agentic/app.py:1563` only accepts `text` or `query`, so `_recall_memories` returns an empty list, for /chat and for the swarm. Not run. The Memory Map snippet has a similar mismatch: it reads `text`/`result_raw` (dashboard/app.py:1358).
2. **Six flags have no consumer**: IMAGINATION_ENGINE, NARRATIVE_IDENTITY, CONSCIENCE_FILTER, SELF_ASSESSMENT, MARS_CONSOLIDATION, PREDICTIVE_EMPATHY. They appear only in common/feature_flags.py and scripts/test_*.py. Universe: every top-level directory except data, kai-pm, docs, output and runtime-logs (`*.py,*.js,*.html,*.yml,*.yaml`), plus a separate grep of the three root compose files. Switching them on or off changes nothing.
3. **One mechanism behind most O1 features.** #4–#10 are all memu-core P17–P20 keyword heuristics with templated text. Several are reachable only through dashboard proxies the UI never calls: /api/confess, /api/conscience/check, /api/loyalty/record.
4. **P17–P20 state is in Redis**, falling back to in-process lists. `retrieve_ranked` reads the in-process `_emotional_timeline` (memu-core/app.py:1299). Inference: when Redis is up that list stays empty, so the emotion boost to recall is 0.
5. **Opt-in services.** memu-graph and letta-agent are behind the `introspection` profile, vault-sync behind `vault` (minimal only). memory-compressor is in full.yml only. Two compose files were checked line by line: memu-core-introspect and vault-sync blocks were grepped/read in minimal; full.yml blocks were read.
6. **Runtime: nothing was re-run.** The only recorded artefact found is output/health_scorecard.json (2026-02-25). It shows memu-core, memory-compressor and heartbeat "UP" on /health only, which is liveness, not feature behaviour.
7. **Leads used.** kai-pm/STUBS_AND_PLACEHOLDERS.md (S5 lakeFS stub, flag table) and kai-pm/CODE_AUDIT_BATCH_LETTA_AGENT.md (grep only, no G1 behaviour finding). Not opened: CODE_AUDIT_BATCH_MEMU_PERSONALITY_AUTONOMY, MEMORY_GRAPH and VAULT_SYNC.
8. **Not verified:** whether letta==0.16.8 still exports `create_client` (would need an import, which is forbidden), and the exact error vault-sync returns when the export arrives without a token (`check_token` was not opened).
9. **data/ was never touched.** data/SOUL.md was not opened, listed or searched. agentic reads it at runtime (agentic/app.py:1426); I took that only from the code.
10. **Searches at the repository root:**
    - `ls -d */` (directory names only).
    - `ls *.yml *.yaml; ls docker-compose*`.
    - `grep` and `sed` on the three named files docker-compose.full.yml, docker-compose.minimal.yml and docker-compose.sovereign.yml.
    - Several `grep -r` runs over a directory list built from `ls -d */` that excluded data/ (and kai-pm, docs, output, runtime-logs; some runs also excluded scripts and memu-core). These covered most top-level directories at once, so they come close to a whole-repository search. data/ was never included in any of them.
    - `ls` of output/ and runtime-logs/, and a read of output/health_scorecard.json.

</details>

### G2

SHA-256 `e18d509d4078e96b0610dfe7a24b3f4e17f52fbd801a46bfcfaa7468c549a55e` · 11211 bytes

<details><summary>show</summary>

| # | Feature | Code location (file:line, at most 3) | Class | What the code actually does (one line) | Runtime | Gap and next action (retain / finish / integrate / replace) |
|---|---|---|---|---|---|---|
| 20 | anomaly detection | agentic/app.py:1897, agentic/app.py:2148, agentic/app.py:2064 | PARTIAL | Real rolling z-score per metric (window 48, warm-up 6, \|z\|>2.0) for cpu, ram, email_unread and docker_unhealthy. It runs inside the proactive observer, but the CPU input is always 0 (note 3), the observer is likely stopped by the trust gate (note 1), and the alert write to memory has the wrong format (note 2). | NOT RUN | finish: fix the trust capability, the memorize payload and the sysmetrics field |
| 21 | cortex (perception spine / NPU inference) | cortex/app.py:462, cortex/app.py:306, agentic/cortex.py:211 | PARTIAL | The cortex service polls 9 sensors and builds Level 2/3 summaries, an intent fan and a topic-shift bridge from regex and template rules. Agentic reads its /state into the world block. `_npu_synthesize` only raises NotImplementedError (FF_CORTEX_NPU default OFF). The perception spine runs in shadow mode. | NOT RUN | retain the rule layer; NPU path is NAME_ONLY; integrate the spine (note 7) |
| 22 | cross-sensor correlation | agentic/app.py:1918 | PARTIAL | Five fixed string-match rules over the observer's text lines (cpu+docker, ram+docker, cpu+ram, git+email, aq+email). Edges go to CausalGraph only if FF_CAUSAL_WORLD_MODEL is on (default OFF). The CPU rules can never fire because CPU reads 0 (note 3). | NOT RUN | finish: same fixes as #20 |
| 23 | multi-modal fusion | perception/camera/app.py:277, perception/camera/app.py:363, fusion-engine/app.py:90 | PARTIAL | The live path, `/proactive/auto` (called by supervisor), uses only the heuristic `_speak_or_not`. The LLM path `interpret_multi` has no caller in the searched dirs and posts to agentic:8000, but agentic listens on 8007. fusion-engine returns a deterministic hash stub when LLM_BACKENDS is empty, and no compose file sets it. | NOT RUN | integrate: fix the URLs and wire the LLM path; fusion-engine is SIMULATED until backends are set |
| 24 | sensors & perception services | sysmetrics/app.py:55, perception/vision/app.py:36, screen-watcher/app.py:84 | PARTIAL | Services use real libraries (psutil, OpenCV/DeepFace, faster-whisper, imaplib, caldav, feedparser, docx/openpyxl/pptx/ezdxf, mss/pytesseract), each with a stub fallback. minimal compose sets WHISPER_BACKEND=stub. screen-watcher POSTs to /screenshot, which screen-capture does not have. news-feed has no /summary. audio-service, vision-service, clipboard-service and files-service directories do not exist; the code is under perception/. | NOT RUN | finish: fix route mismatches (note 5), choose real backends; per-service depth not audited |
| 25 | sensory pattern learning | agentic/app.py:1997, agentic/app.py:2203 | PARTIAL | Keeps the last 10 cycles in memory, classifies lines by string, and flags any type seen in 3 or more cycles. At 7 or more it appends a ritual proposal to /data/RITUALS.md. The sensor_pattern memory write has the wrong format (note 2), and the whole thing sits behind the observer trust gate (note 1). | NOT RUN | finish: fix the payload; persist the history (it resets on restart) |
| 26 | world state & provenance | agentic/app.py:2229, agentic/app.py:1691, heartbeat/app.py:412 | PARTIAL | The observer builds {value, source, timestamp, confidence} per field and posts it to memu-core (wrong format, note 2). `_sense_world` really probes 9 services plus screen, clipboard and cortex with 2s timeouts on each /chat, and names any source that failed. The world anchor is heartbeat `/world` (cached once a day from calendar-sync `/context`), but its default URL uses port 8030 while calendar-sync listens on 8055, and no consumer in agentic, memu-core or cortex. The spine world_state store is fed only in active mode. | NOT RUN | retain the injection; finish persistence; integrate the anchor (port and network) |
| 27 | 14-way context gather | agentic/app.py:2660, agentic/planner.py:230 | GENUINE | `asyncio.gather` of 13 fetchers plus the session fetch. All 17 memu-core paths it calls exist in memu-core/app.py (a text check, not a call). Graph and Letta channels are off by default (FF_GRAPH_INGEST and FF_LETTA_TASKS default False). Each failure falls back to an empty value. | NOT RUN | retain; quality depends on sensors and LLM |
| 28 | agentic reasoning service | agentic/app.py:2555, agentic/app.py:806, agentic/app.py:843 | PARTIAL | `/chat` routes the message, gathers context and streams from the LLM router. `/run` scores a plan template and never calls an LLM. `/chat/teammate` and `/chat/swarm` call `_llm.chat`, which LLMRouter does not have (note 4). | NOT RUN | finish: add or replace `chat` on LLMRouter |
| 29 | conviction scoring (→ information only) | agentic/conviction.py:177, agentic/conviction.py:31, agentic/app.py:3142 | PARTIAL | Six heuristic signals (keyword overlap, step count, word count, rethink count, specialist keywords, domain confidence), plus planner and adversary modifiers. Below 8.0 it loops up to 3 rethinks, then tree search. The rethink rebuilds the same 2-step template, so the score rises with the rethink count rather than with a better plan. | NOT RUN | retain as information only; replace the rethink with real re-planning |
| 30 | local LLM serving (Ollama) | common/llm.py:265, common/llm.py:370, docker-compose.full.yml:795 | GENUINE | A real HTTP client to Ollama (`/v1/chat/completions`) with a model-presence check and Groq/OpenRouter fallbacks. The compose files define the ollama image and ollama-pull. With no URL set it returns a deterministic stub. The default model is qwen2.5:0.5b, a test placeholder. | NOT RUN | retain; upgrade the model when hardware exists |
| 31 | memory-driven planner | agentic/planner.py:269, agentic/planner.py:230, agentic/app.py:3077 | PARTIAL | Wired into `/run`. Matches past episodes by keyword, fetches corrections, nudges and preferences, and computes a conviction modifier. The output is a list of plan steps (metadata only) that no LLM executes. "Gap-aware" is NOT_FOUND_IN agentic/planner.py (grep "gap"). | NOT RUN | finish: have the plan drive execution or an LLM |
| 32 | priority queue | agentic/priority_queue.py:70, agentic/priority_queue.py:124, agentic/app.py:787 | PARTIAL | A real priority queue with an asyncio semaphore and preemption. The only use is `/queue/stats`. No `submit` caller and no read of FF_PRIORITY_QUEUE in the 46 code dirs searched, so the flag is name-only. | NOT RUN | integrate: route LLM calls through `submit`, or drop the flag |
| 33 | specialist router | agentic/router.py:124, memu-core/app.py:200, memu-core/app.py:1327 | GENUINE | Two keyword classifiers. agentic `classify()` sends /chat to 8 routes with zero-LLM dispatch. memu-core `classify_category` tags memories into 8 UK construction categories and boosts same-category records at retrieval. The specialists only get separate models if LLM_DEEPSEEK/KIMI/DOLPHIN_URL are set. | NOT RUN | retain; wire separate model endpoints later |

**Notes**

1. **Trust gate likely stops the observer.** The observer asks permission for "proactive_observation" (agentic/app.py:2064). That name is not in `CAPABILITY_GATES` (agentic/trust_core.py:71), so it is treated as needing GUARDIAN, the top level. A fresh `TrustRecord` starts at DORMANT, the bottom level, so the observer would skip every cycle. This affects features 20, 22, 25 and 26. I found it by reading the code, not running it. I did not look for a saved trust record in `data/`.
2. **Memory writes have the wrong format.** The observer posts `{content, category, user_id}` to memu-core (agentic/app.py:2178, 2195, 2214, 2252; also 1601). memu-core's `MemoryUpdate` requires `timestamp` and `event_type`, and the text field is `result_raw` (memu-core/app.py:130). I expect a 422 rejection, and the callers never check the status, so it would fail silently. Not run.
3. **CPU always reads 0.** sysmetrics `/snapshot` returns `cpu.percent` (sysmetrics/app.py:60). Every consumer reads a top-level `cpu_percent`: agentic/app.py:1740 and 2140, cortex/app.py:209, common/perception_spine/adapters.py:135, common/world_state/reducers.py:206.
4. **LLMRouter has no `chat` method.** In common/llm.py it defines only `query`, `query_multi` and `stream`, with no `__getattr__`. A grep for `def chat` in agentic/ and common/ found only the route handlers. So `/chat/teammate` and `/chat/swarm` break.
5. **Address mismatches between services.** Found by comparing routes and defaults in the code and the three compose files:
   - screen-watcher calls `/screenshot`, but screen-capture only has `/capture` and `/capture/file`.
   - news-feed has no `/summary`. A non-200 reply in `_sense_world` is dropped without being listed as unavailable.
   - camera's LLM path uses `agentic:8000`, but agentic listens on 8007. Its audio default is `perception-audio`, but the compose service is `audio-service`.
   - heartbeat looks for calendar-sync on port 8030, which compose does not override; calendar-sync listens on 8055. The two are also on different networks.
6. **One mechanism, several features.** Features 20, 22, 25 and the provenance part of 26 all live in `_proactive_observer`. The world-context part of 26 and feature 27 share `_sense_world`. "Multi-modal fusion" is two unrelated pieces: the camera's speak-or-not heuristic and the fusion-engine stub. In the dirs I searched, fusion-engine is referenced only by dashboard/supervisor health checks.
7. **Shadow-mode defaults.** `KAI_PERCEPTION_MODE` defaults to shadow and `KAI_CORTEX_SOURCE` defaults to poll (common/perception_spine/shadow.py:47, cortex_source.py:29). Neither is set in the three compose files, so the spine's world_state store stays empty.
8. **Services are split across compose files.** Most sensors and cortex are in docker-compose.minimal.yml, which sets WHISPER_BACKEND=stub. fusion-engine, screen-capture, camera and audio (profile sensors) and parakeet (its own profile) are in docker-compose.full.yml. I did not check how the files are combined at bring-up.
9. **Not examined, and leads used.**
   - Not examined: `data/` (by rule), test suites, the dashboard Body/Feeds panels (README L1203–1204), and the internals of monitor-service, document-parser, email-reader, calendar-service and files beyond import and stub greps.
   - Leads used: kai-pm/STUBS_AND_PLACEHOLDERS.md (S1–S4, T1) and the header of kai-pm/CODE_AUDIT_BATCH_WORLD_ANCHOR.md.
   - runtime-logs/sovereign is empty, so there are no recorded runtime results.
10. **Commands touching the repository root** (none recursive, none into `data/`):
    - `ls -d */`: directory names only.
    - `ls docker-compose*.yml compose*.yml` and `ls docker-compose.yml …`: file names only.
    - `grep -n` on three named files: docker-compose.full.yml, docker-compose.minimal.yml and docker-compose.sovereign.yml. One grep also had `-rn --include=*.py`, which matched nothing in those three files.
    - All other searches named specific code dirs: a list of 46 code dirs, plus agentic, common, cortex, memu-core and similar.

</details>

### G3

SHA-256 `dbf461fca99884c4921dd894db6724a106cde61e9a18dd03f5c61d7b30e8e3ee` · 12598 bytes

<details><summary>show</summary>

I mapped all 19 features by reading the code. Five of them, those marked GENUINE (37, 46, 47, 51, 52), have real code wired to an endpoint or caller. Two that the README calls "✅ Live" (Socratic questioning and temporal projection) are not connected to anything. Nothing was run.

| # | Feature | Code location (file:line, at most 3) | Class | What the code actually does (one line) | Runtime | Gap and next action |
|---|---|---|---|---|---|---|
| 34 | D100 transitive reasoning | memu-graph/transitive.py:74-100, 102-123; common/feature_flags.py:89 | NAME_ONLY | `can_reason()` always returns False, whatever the flag says. `reason()` returns one "[STUB]" insight. PageRank, shortest-path and rule-mining return `[]`. No importer found in agentic, common, memu-core, memu-graph, supervisor, verifier, fusion-engine, cortex, dashboard, executor or tool-gate. | NOT RUN | finish (write the graph algorithms) and integrate (no caller yet) |
| 35 | D101 causal world model | agentic/causal_world_model.py:100-166, 178-209, 259-294; agentic/policy_memory.py:97-165; agentic/app.py:1964-1974, 2276-2286 | PARTIAL | `CausalGraph` keeps edges in memory, but path, effect and prediction queries return empty. The simulator returns `[]` or 0. The surprise check returns None. `PolicyLibrary` really appends to and keyword-searches `/data/policies.jsonl`, but nothing calls it. The app's edge write has the wrong arguments (TypeError) and its surprise call uses a method name that does not exist (AttributeError); both errors are swallowed. All three flags are OFF. | NOT RUN | finish (graph traversal, simulator), fix the two app call bugs, then integrate PolicyLibrary |
| 36 | D102 Global Workspace | agentic/global_workspace.py:116-137, 141-149, 175-211; agentic/app.py:2289-2308 | NAME_ONLY | `submit_bid` only logs, `select_winner` returns None, `broadcast` does nothing, `get_stream` returns `[]`, `can_operate` returns False. The observer's bid leaves out 4 required fields (TypeError, swallowed). Only the cortex bid (cortex.py:310) is built correctly. FF OFF. | NOT RUN | finish (bidding and broadcast loop), fix the observer's bid |
| 37 | D90 swarm assembly | agentic/app.py:843-901; agentic/swarm_stages.py:494-515; agentic/swarm.py:162-204 | GENUINE | `POST /chat/swarm` runs real LLM stage functions (Scout, Sage, Doctor, Oracle) over a shared `SwarmContext`, plus the adversary. The final score is a fixed 5-signal weighted formula using reputation weights. The debate plan is a template, and the specialist is hard-coded to "DeepSeek-V4". | NOT RUN | retain; integrate Socratic and forecast stages (rows 38, 40) |
| 38 | D92 Socratic self-questioning | agentic/questioner.py:54-96; agentic/swarm_stages.py:513-514; agentic/app.py:860-868 | PARTIAL | `SocraticQuestioner` makes a real LLM call with a fixed fallback list. But `/chat/swarm` never passes a questioner, `CognitiveFSM.run` has no questioner stage, and `enriched_query` is never read by any stage. Already found as KAI-AWARE-001 / KAI-SWARM-004 in the audits. | NOT RUN | integrate (pass a questioner, add the FSM stage, have stages read `enriched_query`) |
| 39 | D93 hypothesis engine | agentic/hypothesis.py:66-89, 93-169; agentic/curiosity.py:77-92; agentic/app.py:2311-2315 | SIMULATED | It is wired into the observer tick, but the call passes no LLM and no memory source. So every hypothesis is a fixed "If the pattern … holds" sentence built from the same 5 hard-coded seed questions, always judged INCONCLUSIVE, and appended to `/data/CURIOSITY.md`. | NOT RUN | integrate (pass `_llm.chat` and `_recall_memories`, take seeds from memory, not a static list) |
| 40 | D94 temporal projection | agentic/forecaster.py:110-136 | PARTIAL | `TemporalForecaster.project()` asks the LLM for 4 scenarios, with a fixed fallback. No importer found in the directories listed in row 34, and `agentic/app.py` has no `/forecast` endpoint. | NOT RUN | integrate (into the swarm after the Oracle stage, or behind an endpoint) |
| 41 | D95 dialectical synthesis | agentic/dialectic.py:44-63, 66-70 | NAME_ONLY | `can_synthesize()` always returns False, and `synthesize()` returns a "[STUB]" string with confidence 0. No importer found in the directories listed in row 34. | NOT RUN | finish (the dual-model logic) and integrate |
| 42 | D96 analogical reasoning | agentic/analogy.py:56-78 | NAME_ONLY | `can_find()` always returns False, and `find_analogy()` returns a "[STUB]" solution. No importer found in the directories listed in row 34. | NOT RUN | finish and integrate |
| 43 | D97 concept blending | agentic/concept_blend.py:50-72 | NAME_ONLY | `can_blend()` always returns False, and `blend()` returns a "[STUB] a×b" name with placeholder properties. No importer found in the directories listed in row 34. | NOT RUN | finish and integrate |
| 44 | GPU-era foundations | agentic/counterfactual.py:16-46; tool-gate/app.py:1021-1055; agentic/app.py:2242-2247 | NAME_ONLY | Counterfactual returns a fixed "stub_pending_gpu" result. The trust-negotiation endpoint logs to the ledger and always answers `pending_approval`, and its flag `FF_TRUST_NEGOTIATION` is OFF. The `emotional_context` block in the world model is hard-coded empty. The curiosity slot behaves as in row 39. | NOT RUN | finish each slot separately; the umbrella label claims more than exists |
| 45 | SAGE critique / self-review | verifier/app.py:351-411, 499-500; agentic/adversary.py:565-671, 731-732 | GENUINE | Two fixed-rule checks over the other checks' results (looking for groupthink, thin evidence, contradictions, false consensus). There is no LLM, so it is not "AI arguing with itself". It always runs inside `/verify` and `challenge_plan`. Neither `FF_SAGE_CRITIQUE` nor `FF_SAGE_SELF_REVIEW` is read anywhere in the searched directories. | NOT RUN | retain; either wire the two flags or delete them |
| 46 | adversary engine | agentic/adversary.py:674-772; agentic/app.py:3117-3130; agentic/swarm_stages.py:426-434 | GENUINE | Fixed-rule challenges: history, verifier (HTTP), policy (HTTP), consistency, calibration, security and SAGE self-review. That makes 7 on `/run` but 6 in the swarm, because the swarm passes no `injection_re`. See note 4 for a defect in the verifier challenge. | NOT RUN | retain; fix the fixed `rank_score` |
| 47 | cognitive reasoning FSM | agentic/cognitive_fsm.py:208-312, 96-134; agentic/app.py:870-883 | GENUINE | A real state machine with per-stage timeouts, debate/escalate and rethink caps, HALT, and 4 swarm configs. Gaps: `AgentHandoff` is a plain dataclass that validates nothing, stages parse free-text JSON from the LLM, and RETHINK re-runs the same gate on unchanged context. | NOT RUN | retain; make RETHINK actually re-gather or re-debate |
| 48 | counterfactual rehearsal | agentic/counterfactual.py:16-46; agentic/app.py:29, 765 | NAME_ONLY | `rehearse()` returns a fixed stub result and `can_rehearse()` returns False. `rehearse()` is imported but never called. Only `can_rehearse()` is reported, by `GET /introspect/capabilities`. | NOT RUN | finish and integrate |
| 49 | debate / tree search | agentic/tree_search.py:76-185, 211-296; agentic/app.py:3154-3178 | SIMULATED | Runs on `/run` only after 3 failed rethinks. Each "branch" is the query plus a fixed text suffix, the plan is a template, and the score is the word-matching conviction formula. No LLM generates any reasoning. `debate_branches` and `tree_search_with_debate` are never called. `FF_TREE_SEARCH` is never read. | NOT RUN | replace (branches should be generated by the LLM), then integrate debate |
| 50 | multi-model consensus / specialist endpoints | fusion-engine/app.py:90-133, 247-293; agentic/router.py:524-547 | PARTIAL | `/fuse` really sends the prompt to several backends in parallel and measures agreement (embedding or word-overlap). But with no `LLM_BACKENDS` set, the answers are hash-based fixed text. No caller of `/fuse` found in agentic, common, supervisor, dashboard or executor; the router's `MULTI_SIGNAL` route returns None. `model_council` picks a model; it does not build consensus. | NOT RUN | integrate (route `MULTI_SIGNAL` to `/fuse`) and configure backends |
| 51 | persistent teammates and reputation | agentic/teammates.py:22, 64-103; agentic/app.py:806-834, 904-907; agentic/swarm.py:71-157 | GENUINE | `POST /chat/teammate/{name}` makes a real LLM call with the persona prompt plus a world snapshot. `TeammateRep` counters are saved to and loaded from `/data/teammate_reputation.json`. The personas are read from a `data/teammates/` folder, which I did not inspect (see note 6). | NOT RUN | retain; check the persona path inside the container |
| 52 | verifier (SAGE semantic fact-check) | verifier/app.py:162-251, 475-525, 418-452 | GENUINE | Fixed-weight signals (memu-core ranked retrieval with embeddings, plus word overlap, consistency and material claims) are combined into PASS, REPAIR or FAIL_CLOSED. Called by the adversary, the router's `FACT_CHECK` and fusion-engine. | NOT RUN | retain; fix the adversary evidence pack and the router port (notes 4, 8) |

Notes:
1. **One module with no callers.** `forecaster`, `dialectic`, `analogy`, `concept_blend`, `policy_memory` and `memu-graph/transitive` have no importer in: agentic, common, memu-core, memu-graph, supervisor, verifier, fusion-engine, cortex, dashboard, executor, tool-gate (`.py`, `.js`, `.html`). The only references are tests in `scripts/`.
2. **Gates hard-coded to False.** The D95–D97, D100, D101 and D102 gates all end with `return False`, after a flag check. Turning a flag on cannot activate them; the code itself has to change. They also import `from feature_flags`, not `from common.feature_flags`.
3. **Flags nobody reads.** `FF_TREE_SEARCH`, `FF_SAGE_CRITIQUE` and `FF_SAGE_SELF_REVIEW` are only in the registry. `FF_SOCRATIC` is read only by `can_question()`, which nothing calls. `MODEL_COUNCIL` and `MORAL_IMAGINATION` are not in the registry at all, so they default to OFF unless set in the environment.
4. **Defect in the adversary's verifier challenge (found by reading, not run).** `agentic/adversary.py:188-191` sends every chunk with a fixed `rank_score` of 0.5. In `verifier/app.py:237` the best possible chunk score is then 0.6·0.5 + 0.2·1.0 + 0.1·0 + 0.1·0.5 = 0.55, below the 0.60 "strong chunk" threshold (`verifier/app.py:52`). So `strong_chunks` is always 0, and the verifier can never return PASS on this path. When the chunk list is empty, the verifier falls back to memu-core and this cap does not apply.
5. **Retries that change nothing real.** `score_conviction` gives bonus points for the retry count itself (`conviction.py:118-129`). So debate retries in the swarm, and the `/run` rethink loop, raise the score mechanically. The FSM's RETHINK re-runs the same gate on the same context.
6. **Not inspected:** `data/teammates/`, because `data/` is out of bounds. In the image, `WORKDIR` is `/app` and the build copies `data/` to `/app/data`, but `TEAMMATES_DIR` works out to `/data/teammates`. Whether personas load depends on the `/data` mount, which is defined in the compose files at the repository root. I did not read those.
7. **Bounds breach and audits.** My first search for `TransitiveReasoner` ran over the whole repository with `!data/**` excluded, as the brief allows. Its results included `README.md`, `kai-pm/` and `scripts/`, which goes against the "no repository root" instruction. Every search after that was limited to named code directories. Audits that already record some of these findings: `CODE_AUDIT_BATCH_QUESTIONER_TEAMMATES_WATCHDOG.md` (Socratic), `CODE_AUDIT_BATCH_SWARM_ORCHESTRATION.md` (`enriched_query`) and `CODE_AUDIT_BATCH_COGNITIVE_GOVERNANCE_FOUNDATIONS.md:119,122` (`CausalEdge`, surprise call). I did not find the bid-missing-fields bug in row 36 or the verifier cap in note 4 recorded. That rests on a grep over the `CODE_AUDIT_BATCH_*.md` files, with one hit file read; it is not a full read.
8. **Port mismatch.** `agentic/router.py:446` defaults `VERIFIER_URL` to port 8003, while everywhere else uses 8052. It only matters if the environment leaves `VERIFIER_URL` unset, which I could not check from the root compose files.
9. **Stub register mismatch.** `kai-pm/STUBS_AND_PLACEHOLDERS.md:30,133` says to set `LLM_DEEPSEEK_URL`, `LLM_KIMI_URL` and `LLM_DOLPHIN_URL`. But `fusion-engine/app.py:52` reads only `LLM_BACKENDS`. I did not check whether compose turns those three variables into `LLM_BACKENDS`.
10. **Runtime:** `runtime-logs/sovereign/` is empty, and I found no recorded runtime result for any of these features. Nothing was executed.

</details>

### G4

SHA-256 `cd3eafa405f18564783d00d30c07ebd0e2364be477e2b8089ce422754402c040` · 14438 bytes

<details><summary>show</summary>

| # | Feature | Code location (file:line, at most 3) | Class | What the code actually does (one line) | Runtime | Gap and next action |
|---|---|---|---|---|---|---|
| 53 | Goals view (Ohana goals) | dashboard/app.py:1215 · memu-core/app.py:4046-4183, 4196 · dashboard/static/app.html:3395-3634 | PARTIAL | Creating, listing and drift-checking goals work, and the Goals view also shows reminders, tasks and briefings. Goal data comes from memu-core, not agentic. Goal updates change only the in-process record. `PGVectorStore.update_record` (memu-core/app.py:501-520) saves only metadata fields, so status and progress notes are lost under the postgres store. Goals are read from the newest 10,000 memories only. "Progress bars" are just a count. | NOT RUN | **Finish:** save goal content on update. Query goals by event type, not "newest 10k". |
| 54 | Watch rule engine | monitor-service/app.py:220-231, 102-132, 156-173 | GENUINE | A background loop runs every second and checks each rule on its interval: it reads an HTTP JSON field, applies a gt/lt/contains/changed/pct test with a cooldown, then sends a notify or TTS message and logs an alert. The dashboard can add, delete, enable and disable rules (dashboard/app.py:2395-2435). | NOT RUN | **Retain; fix the "scrape" source.** It posts `{url, selector}` to browser-agent `/scrape`, but `/scrape` takes no body (browser-agent/app.py:181), so it reads whatever page is already open. It also sends no Authorization header (a search of monitor-service/app.py for `Authorization` and `KAI_SERVICE_TOKEN` found nothing). |
| 55 | Anti-annoyance | memu-core/app.py:4468-4580, 4698-4736 · supervisor/app.py:268 | PARTIAL | Each nudge type has its own cooldown, which grows 1.5x per dismissal (capped at 24h). There is a Do Not Disturb (DND) mode with a bypass at urgency 0.9 or higher. The supervisor's Telegram nudges use the filtered endpoint. | NOT RUN | **Integrate.** Nothing calls `/memory/nudge/dismiss` or `/memory/dnd` in agentic, dashboard, supervisor, common, memu-core, telegram-bot or heartbeat. The dashboard's `/api/nudges` reads the unfiltered `/memory/proactive` (dashboard/app.py:264). State lives only in the process (audit KAI-PERSONA-099). |
| 56 | Emergent ritual discovery | agentic/app.py:1997-2017, 2020-2043 | PARTIAL | When a pattern type appears in 7 or more of the last 10 observer cycles, it appends a "pending approval" proposal to `/data/RITUALS.md`. | NOT RUN | **Finish.** The docstring says it notifies the operator, but the code only writes the file and a log line. It also needs `SENSORY_LEARNING` on. The set of already-proposed rituals lives only in memory, so it proposes again after a restart. It runs inside the trust-gated observer (note 1). |
| 57 | Proactive observer / agent | agentic/app.py:2046-2338 · memu-core/app.py:7177-7191, 7209-7535 · supervisor/app.py:396-441 | PARTIAL | The 5-minute loop probes docker, email, air quality, git, sysmetrics, calendar and house-doctor, then writes observation memories. There is **no weather or news probe**. The "13-action registry" is a fixed list of 13 descriptions; nothing executes from it. Scheduled tasks and reminders "fire" only as a Telegram message of their title. Morning and evening briefings run only when requested (dashboard POST); no scheduler calls them in agentic, supervisor, heartbeat, telegram-bot, common or tool-gate. | NOT RUN | **Integrate.** The loop is trust-gated with an unregistered capability name and is probably suppressed on a fresh trust record (note 1). Add weather and news probes, task-type dispatch and a briefing schedule. |
| 58 | Proactive scheduling | agentic/app.py:2161-2187 · calendar-service/app.py:181-196 | PARTIAL | It combines "event within 30 minutes" with air-quality, CPU and git state into a schedule memory. | NOT RUN | **Fix the contract.** Calendar `/summary` returns only `{"summary": str}`. The code needs `next_event` and `minutes_until_next`, which are absent from calendar-service/app.py and calendar-sync/, so this branch never fires. It also sits behind the trust gate (note 1). |
| 59 | Resource-aware curiosity | agentic/curiosity.py:42-54, 57-99 · agentic/hypothesis.py:93-140 · agentic/app.py:2314 | SIMULATED | It is called with no LLM and no memory function. It builds template "hypotheses" from 5 hard-coded seed questions, always marks them INCONCLUSIVE and logs them. The GPU branch returns None. | NOT RUN | **Replace.** Pass in the LLM and memory functions, and mine real open questions instead of the seed list. |
| 60 | Struggle detection | memu-core/app.py:4877-4969 · dashboard/app.py:1461 · dashboard/static/app.html:4524-4535 | PARTIAL | A keyword-based score over the last 15 session messages, using 5 signals (short messages, repeats, keywords, "?" count, rapid-fire). It returns an offer of help, which the dashboard shows as a pop-up. | NOT RUN | **Integrate.** "Auto-adapts" is not wired: agentic/app.py never calls `/memory/struggle` (no match for "struggle"). The UI always asks for `session_id=default`. There is one global cooldown. |
| 61 | Wake word & intent | perception/wake/app.py:75-99, 102-153, 188-219 · agentic/app.py:2535-2551 | PARTIAL | Wake detection is a regex over text (or an optional Whisper transcript) with a fixed 0.85/0.95 confidence. Intent comes from an LLM, falling back to a regex heuristic. | NOT RUN | **Integrate.** `FF_WAKE_INTENT_ROUTING` defaults to OFF (common/feature_flags.py:44). Nothing calls `/wake/detect` or `/wake/process` from perception/ or agentic/. dashboard/static makes no `/api/wake` call, although the proxies exist (dashboard/app.py:1930-1945). |
| 62 | Tool Gate + HMAC/Ed25519 auth | tool-gate/app.py:541-572, 763-833 · common/auth.py:52-67, 86-133 | PARTIAL | HMAC-SHA256 signature, timestamp window, nonce replay check, primary and previous keys with revocation, and the dev secret refused unless `HMAC_ALLOW_DEV_SECRET` is set. agentic signs its calls (agentic/app.py:3192). | NOT RUN | **Finish.** Ed25519 is not used by the gate (note 9). The signature covers only actor\|session\|tool\|nonce\|ts, not the parameters. A body field `cosign=true` approves itself (tool-gate/app.py:630). The tool-gate audit says these are already registered as KAI-AUTH-001 and KAI-AUTH-004; I did not open the register. |
| 63 | Unified Hunter (34 actuators, not cut over) | common/actuator_registry/catalog.py:249-280 · common/autonomy/legacy_bridge.py:37-55 · agentic/trust_integration.py:222-247 | PARTIAL | The catalog has 34 actuators (36 matches of `_reg(` in catalog.py, minus the definition and the build call). All start in the LEGACY state. The bridge only advises (`KAI_AUTONOMY_ENFORCE` defaults to false). agentic exposes only `/uh/status`, `/uh/actuators` and a simulated `/uh/paper-trade` (agentic/app.py:576-644). | NOT RUN | **Integrate (operator decision).** No service routes work through the registry. agentic imports `build_catalog` only for reports, and outside the excluded data, docs, tests, scripts and Markdown nothing else imports it. `mutating_handlers.py:37,159` defaults to `executor:8003`, but the executor listens on 8002 (executor/app.py:522). |
| 64 | Action audit ledger | tool-gate/app.py:351-476, 980-997 · ledger-worker/app.py:86-212 | PARTIAL | tool-gate appends every gate decision to a hash-chained file (`/data/tool-gate/ledger.jsonl`). ledger-worker checks the chain and Merkle root every hour, collects stats, and on request archives the last 10,000 entries to `/tmp/ledger-archives`. | NOT RUN | **Finish.** It records gate decisions, not executions; the executor writes nothing to it. The docstring's "nonce cleanup" has no function in ledger-worker/app.py. Archives go to /tmp, keep only 30 files and hold only 10,000 entries. The worker needs a `BEARER_TOKEN` that tool-gate trusts. |
| 65 | Browser agent | browser-agent/app.py:122-277 · dashboard/app.py:2174-2234, 2464-2475 · dashboard/static/app.html:2041-2077 | PARTIAL | Real Playwright: navigate, click, type, scrape, screenshot, run and DuckDuckGo search, behind `require_service_auth`. `browse:` and `search:` in chat call `/api/browser/run` and `/api/browser/search`. | NOT RUN | **Fix the wiring.** The dashboard proxies send no Authorization header, so with a token set browser-agent answers 401, and with none set it answers 503 (unless `KAI_ALLOW_UNAUTHENTICATED`). A search of dashboard/app.py found no service-token code, and common/http_hygiene.py has no Authorization. The service is defined only in docker-compose.minimal.yml:380 (profile `external-egress`). |
| 66 | Governed decision path (end-goal) | agentic/app.py:3182-3228 · tool-gate/app.py:595-670 · agentic/trust_integration.py:165-260 | PARTIAL | Low conviction produces a "need more data" summary. With a task hint, agentic signs a request to the gate behind a circuit breaker and a breaker for tool errors. The gate returns approve, co-sign or block, and the decision is written into the plan. | NOT RUN | **Integrate.** No execution follows an approval. The executor never checks a gate decision. An operator co-sign creates no grant that can be executed (audit TOOL_GATE_EXTENSION:111). |
| 67 | Sandboxed executor | executor/app.py:134-200, 240-295, 416-470 | PARTIAL | Runs commands in a subprocess with `shell=False`, a base-command allowlist, a regex denylist, AST checks on Python, a malware scan and `require_service_auth`. | NOT RUN | **Replace the sandbox and integrate.** The allowlist includes python3, pip, docker, curl, git and make, so it can run anything (KAI-EXEC-003 CRITICAL, CODE_AUDIT_BATCH_EXECUTOR.md:13). The "restricted builtins" comment is not implemented (lines 273-280). It has no caller (note 6). The gate's tools (shell, qgis, n8n, noop, speak; tool-gate/app.py:593) do not match the executor's (shell, noop, script, python). |
| 68 | Secret management & rotation | docker-compose.sovereign.yml:86-121 · scripts/vault-init.sh · scripts/vault-rotate.sh:6-9 | SIMULATED | Vault runs in `-dev` mode and is seeded with two entries of URLs and settings. The "rotator" only writes a `rotated_at` timestamp every 30 days and changes no key. | NOT RUN | **Replace.** No Vault client or address was found in tool-gate, executor, common, agentic, memu-core or ledger-worker. Secrets come from environment variables or `/run/secrets` (common/auth.py:13-35). `scripts/auto_rotate_hmac.py` writes `output/hmac_rotation_state.json` and revokes the old key at once; its consumers were not traced. |
| 69 | Trust negotiation / autonomy requests | tool-gate/app.py:1012-1059 · common/feature_flags.py:69 | PARTIAL | When the flag is on, it writes a ledger entry with approved=False and returns `pending_approval`. When off, it returns 503. | NOT RUN | **Finish.** The flag defaults to **False**, although README L1150 says ON and the docstring says True. The endpoint has no authentication. No pending item is stored and there is no approve endpoint. Nothing calls it from agentic, dashboard, supervisor, common, memu-core, executor or telegram-bot. |

**Notes**

1. **One mechanism probably blocks features 56–59.** I read this in the code but did not run it. `_proactive_observer` calls `gate_autonomous_action("proactive_observation")` (agentic/app.py:2064). That name is not in `CAPABILITY_GATES` (agentic/trust_core.py:71-98), so `can_do` falls back to the GUARDIAN level (line 197). A new trust record starts at DORMANT (line 129). So on a fresh record every cycle should be suppressed, and ritual discovery, scheduling and curiosity never run. If TrustCore fails to load, the gate fails open instead. I did not inspect the real trust record on disk.
2. **Service-to-service contracts nobody checks.** Five cases: calendar `/summary` fields (58); monitor sends a body that `/scrape` ignores (54); dashboard calls browser-agent without auth (65); actuator registry points at executor port 8003, not 8002 (63); gate and executor tool lists differ (67).
3. **Hands not connected (62, 63, 66, 67).** The gate decides and the executor executes, but nothing joins them. The Unified Hunter registry is meant to be that link and has not been cut over.
4. **Flags that disagree with the README.** `TRUST_NEGOTIATION` defaults to False (README ✓). `WAKE_INTENT_ROUTING` defaults to False, which matches the README.
5. **Stale leads.** STUBS U2 says the Goals backend is agentic; the code proxies to memu-core. CODE_AUDIT_BATCH_BROWSER_AGENT and _EXECUTOR call those endpoints unauthenticated, but the current code has `require_service_auth` on them.
6. **Bounded negative for 67.** I searched agentic, dashboard, supervisor, common, tool-gate, memu-core, telegram-bot and heartbeat for "/execute". The only hits were in common/actuator_registry (catalog, verification, mutating_handlers); no running service calls it.
7. **Lost on restart.** Anti-annoyance counters and DND, the ritual-proposed set, the struggle cooldown, pending co-signs and monitor last values all live only in process memory.
8. **Not looked at:** data/ (by rule), kai-pm/UH_PROGRESS_TRACKER.md, kai-pm/CODE_AUDIT_REGISTER.md, any `TurboVecStore.update_record` override, the trust-record data location, and scripts/test_core_results.json (a recorded result I did not open). No recorded runtime result is cited.
9. **Ed25519.** It lives in common/service_identity.py and common/service_auth.py:223-351. In a search that excluded data, kai-pm, docs, tests, scripts and Markdown, its only consumer is cortex/app.py:569. It is not used by tool-gate. `identity_required` defaults to false.
10. **Searches I ran with the repository root as the path.** Each used either an exclusion glob or a list of named directories; none could reach data/.
    - (a) Exclusion `!{data/**,kai-pm/**,docs/**,*.md,tests/**}` for the anti_annoyance, ritual, struggle and FF_* names.
    - (b) Exclusion `!{data/**,kai-pm/**,docs/**,**/*.md,scripts/**,tests/**}` for `require_service_identity`/`check_identity`, `EXECUTOR_URL`, and the `common.actuator_registry`/`autonomy`/`policy_bridge` imports.
    - (c) Exclusion `!{data/**,kai-pm/**,docs/**,**/*.md,tests/**}` for `/gate/request`.
    - (d) Named-directory globs for: `proactive/filtered`/`nudge/dismiss`/`dnd`; `briefing/morning|evening`; `autonomy/request`; `/execute`.
    - Bash listings at the root by name: `ls -d monitor*`, `ls -d calendar*`, `ls docker-compose*.yml`, `ls -d <named dirs>`.
    - Greps of the three root compose files by name.

</details>

### G5

SHA-256 `f8c892eb9e41ce30f23f6a93133b84b6e7add399f99c0fd85f2714434b9c1b53` · 11889 bytes

<details><summary>show</summary>

Brief followed: read-only, no runtime. Runtime is NOT RUN for all 16 features. `runtime-logs/sovereign/` is empty, so I found no recorded runtime result to cite.

| # | Feature | Code location (file:line, at most 3) | Class | What the code actually does (one line) | Runtime | Gap and next action |
|---|---|---|---|---|---|---|
| 70 | House Doctor / self-diagnosis | house-doctor/app.py:72, :150, :215; agentic/app.py:2263 | GENUINE | 9 rules (D001–D009) applied to tags found by keyword-matching observation strings. Writes a medical_report to memu-core and calls notify on WARNING/CRITICAL. The agentic proactive observer calls it (FF_HOUSE_DOCTOR, on by default). | NOT RUN | **Retain.** `world_state` is accepted but never read (:41; matches KAI-DOCTOR-011 in CODE_AUDIT_BATCH_HOUSE_DOCTOR.md). It is keyword matching, not reasoning. |
| 71 | Checkpoint / time-travel debug | agentic/kai_config.py:1356, :1428; agentic/app.py:3415 | PARTIAL | Saves and lists checkpoints as JSON (breaker, guard, budget and override snapshots), diffs and deletes them. Restore only writes back the breaker state. | NOT RUN | **Finish.** Restore skips guards, budget and overrides, although its docstring says "LangGraph state". Auto-checkpoints on /recover (:441) and after a dream (introspect_app.py:97) always run: FF_CHECKPOINT_AUTO is never read anywhere in agentic/ or common/. |
| 72 | Heartbeat / pulse | heartbeat/app.py:186, :224, :150 | PARTIAL | A passive service. It records time when POST /tick arrives, reports "stale" after 300s, and runs auto-sleep (memu compress and decay) only when /status is polled. /world caches calendar-sync context. | NOT RUN | **Finish.** No background loop. I found no production caller of /tick (searched executor, ledger-worker, dashboard, metrics-gateway and supervisor; only scripts/ci/live_smoke.py and tests call it), so by code reading /health turns degraded about 5 minutes after start. The "Pulse" Ctrl+2 view (dashboard/static/app.html:2709) reads the dashboard's own `/` and `/metrics`, not this service. |
| 73 | No single point of failure (end-goal) | docker-compose.full.yml:30 (`restart: unless-stopped`); agentic/model_council.py:381 | ABSENT | Only container restart policies and a model-council LLM failover exist. No replicated service, store or failover. | NOT RUN | **Design needed.** Searched the 3 docker-compose*.yml files, common/**, supervisor/ and agentic/ for replicas, deploy:, failover, sentinel, standby and SPOF. Hits were the `deploy:` blocks (resource limits and restart policy only) and model_council failover. |
| 74 | Observability (metrics/alerts) | metrics-gateway/app.py:108, :159, :221; prometheus.yml:12 | PARTIAL | metrics-gateway really scrapes 11 services every 30s and exposes Prometheus text at /metrics/text. The Prometheus, Alertmanager and Grafana containers are defined only in docker-compose.sovereign.yml:126–160. | NOT RUN | **Integrate.** prometheus.yml scrapes `/metrics` on each service, and those return JSON (e.g. agentic/app.py:781, supervisor/app.py:532), not Prometheus text. alert.rules.yml uses `http_requests_total`, `sovereign_circuit_breaker_state` and other metrics that no `*.py` file in the repo emits. Point Prometheus at /metrics/text or add an exporter. |
| 75 | Security self-hacking | agentic/security_audit.py:119, :416; agentic/introspect_app.py:169; agentic/adversary.py:501 | PARTIAL | Runs 34 payloads (21 injection, 13 sanitisation) against the in-process INJECTION_RE regex and sanitize_string, plus 4 environment-variable policy checks. Runs on GET /security/audit and as challenge 6 on every /run; SAGE review is challenge 7. | NOT RUN | **Finish.** It never fuzzes any HTTP API. The HMAC category is skipped because neither caller passes sign/verify functions. FF_SECURITY_AUDIT is never read. |
| 76 | Self-capability map | agentic/app.py:724–779 | PARTIAL | Live pings to 10 services, plus flags, baselines, FSM, teammates, gap log top 5 and counterfactual availability. | NOT RUN | **Fix.** Line 757 does `s.name` on dicts returned by `router.list_skills()` (router.py:723). By code reading this raises AttributeError once any skill is loaded. It works only while zero skills load, which is the case for the 4 repo skills/*.md files (see #85). |
| 77 | Structured errors / circuit breakers | common/errors.py:23; common/resilience.py:35; agentic/app.py:160 | PARTIAL | CircuitBreaker and resilient_call (retry, backoff, breaker) are really used in agentic, dashboard, executor and supervisor. ErrorCode/KaiError defines 23 codes, not the 20 the README says. | NOT RUN | **Integrate.** Outside data/, KaiError and ErrorCode appear only in common/errors.py, tests (scripts/test_error_codes.py, test_image_modules.py, test_core_results.json) and one audit file. No service raises them. HealingEngine (resilience.py:266) has no caller in agentic, common, supervisor, dashboard, executor or memu-core. |
| 78 | Supervisor self-healing and recovery log | supervisor/app.py:188, :159; memu-core/app.py:1510 | PARTIAL | A real 15s sweep of /health with per-service breakers and fleet history. It calls `&lt;svc&gt;/recover` only when SUPERVISOR_RECOVERY_ENABLED is true. Only memu-core's own /recover writes a conscience entry. | NOT RUN | **Finish.** Compose sets SUPERVISOR_RECOVERY_ENABLED "false" (full:280, minimal:880), and each service's /recover is also off by default (RECOVERY_ENABLED=false). agentic is not in the supervisor's default service list. Heals done by the supervisor are not logged to conscience. Supervisor runs only under the `recovery` profile. |
| 79 | System FSM | agentic/system_fsm.py:44; agentic/app.py:2650, :2320 | PARTIAL | A real asyncio-locked FSM with 5 states, 9 events and 15 transitions (README says 16). By default only USER_MESSAGE is fired, from /chat. | NOT RUN | **Finish.** SESSION_END, HEAL_STARTED, FOCUS_* and ANOMALY_CRITICAL are never fired, so RECOVERING can never be reached and ACTIVE never returns to IDLE. SERVICE_DOWN/RESTORED need FF_SERVICE_WATCHDOG, which is not in the flag registry (defaults to off) and not set in compose. Curiosity is not gated on IDLE (app.py:2311). |
| 80 | Agent-Evolver | agentic/kai_config.py:1130, :1065, :1255 | PARTIAL | Groups failed episodes by failure_class (3 or more) and emits fix text from 8 fixed per-class templates. Stored to memu and to the evolver JSON file. | NOT RUN | **Finish.** The "insights" are canned templates. Nothing applies them. FF_EVOLVER_ENABLED and FF_DREAM_PHASE_7 are never read, so it always runs as dream phase 7 and on /evolve/analyze. 25+ HIGH/MEDIUM findings (KAI-LEARN-0xx) in CODE_AUDIT_BATCH_DREAM_EVOLVER_CHECKPOINTS.md. |
| 81 | D99 synthetic experience | agentic/synthetic_experience.py:52, :62 | NAME_ONLY | `can_generate()` always returns False. `generate()` returns a "[STUB] … pending GPU" scenario. | NOT RUN | **Retain as a placeholder.** No caller of synthetic_experience in agentic/ or common/. |
| 82 | Agentic cold path (dream/evolve/audit) | agentic/introspect_app.py:58, :117, :169; dashboard/app.py:985 | GENUINE | A separate FastAPI process (port 8023) serving /dream, /evolve/analyze, /evolve/suggestions and /security/audit. The dashboard proxies /api/dream to it. | NOT RUN | **Retain.** Only under the compose `introspection` profile (full:383). The quality of the work it hosts is covered in #75, #80 and #84. |
| 84 | Dream state / dream cycles | agentic/kai_config.py:877–974; agentic/introspect_app.py:58 | PARTIAL | Runs these phases over the last 30 days of episodes: failure clusters, rule dedup, word-overlap patterns, always/never contradiction check, knowledge boundary, evolver. Persists the cycle and up to 5 insights to memu. | NOT RUN | **Finish.** No MARS memory-decay phase (decay is called only from heartbeat sleep, heartbeat/app.py:166). FF_DREAM_ENABLED is never read. Nothing triggers it automatically: the docstring's "heartbeat triggers dream" has no code in heartbeat/, so it runs only manually through the dashboard. memu-core advertises `POST /memory/dream` (app.py:7189), but no such route exists in memu-core/app.py or memu-core/introspect_app.py. |
| 83 | Capability gap logging | agentic/app.py:2341–2369, :2657, :1891 | GENUINE | When no skill matches and router confidence is below 0.4, it increments an in-memory Counter keyed on the first 80 characters of the message. It calls skill-hunter /hunt at 3 or more misses. | NOT RUN | **Retain/adjust.** The count is cumulative, not "consecutive", and keyed on literal text, not intent. It is lost on restart and never reset after a hunt. The downstream acquisition does not work (#85). |
| 85 | Skills hub / skill hunter / provenance | skill-hunter/app.py:157, :211, :314; agentic/router.py:644 | PARTIAL | Hunter: a fixed keyword→package table, a check that the package exists on PyPI, then writes a templated .md with front-matter and a .meta.json. Hub: a loader with red-flag regex scan, TTL prune and unload. | NOT RUN | **Integrate/fix.** The parser drops any file without a `## Trigger` section (router.py:644). Hunted files have none, and none of the 4 repo skills/*.md files do either, so nothing loads. In full compose the hunter writes `/data/soul/skills` while agentic reads `/skills` and `/app/data/skills`. The loader never reads front-matter or `disabled`. No caller of `/skill/{name}/error` in agentic/ or common/. Front-matter key is `package`, not `pypi_package`. 3 CRITICAL findings in CODE_AUDIT_BATCH_SKILL_HUNTER.md. |

Notes:
1. **I broke a search rule twice.** (a) `grep -rln … --include=*.py .` ran from the repository root, and (b) a Grep with `glob: **/*.py` at the root did not exclude `data/**`. Both went into data/ but read only `.py` files, so data/SOUL.md was not opened, and no data/ results were used. Every other search was limited to named directories or files.
2. **Several "default ON" flags are never read** in agentic/ or common/: FF_CHECKPOINT_AUTO, FF_SECURITY_AUDIT, FF_DREAM_PHASE_7. FF_DREAM_ENABLED and FF_EVOLVER_ENABLED are never read either, so the README's "OFF" for them has no effect: dream and evolver run whenever they are called.
3. **FF_SERVICE_WATCHDOG is used but not registered** (`is_enabled` returns False for unknown flags), and it is not set in the 3 compose files. It is the only route to the FSM's DEGRADED state, and it also gates `/watchdog/*`. #79 is therefore effectively single-transition by default.
4. **Self-healing is off everywhere by default.** Compose sets supervisor recovery to "false", and every service's /recover returns "disabled" unless RECOVERY_ENABLED=true. #71's pre-recover checkpoint and #78's conscience log are reachable only after both are turned on.
5. **Skill acquisition is broken at two points.** The hunter's file format and the loader's parser don't match, and in full compose the directory the hunter writes to is not the one agentic reads. Fix format and mount together; the #76 crash would then show up as soon as skills load.
6. **Dream, evolver, gap and diagnosis logic is mostly keyword and word-overlap heuristics.** It runs on real episodes but has no model or verification behind it. CODE_AUDIT_BATCH_DREAM_EVOLVER_CHECKPOINTS.md records the same thing (KAI-LEARN-0xx).
7. **The observability alerts can never fire as written.** I read prometheus.yml, alert.rules.yml, alertmanager.yml and the compose files directly as named files (not searched). The alert metric names appear in no `.py` file in the repo (searched excluding nothing but non-.py files; see note 1).
8. **Recovery log (#78):** the only conscience entry for a heal comes from memu-core's own /recover (memu-core/app.py:1510–1521). It is outside my listed directories, so I opened it only for this feature.
9. **I did not look at** notify-service, telegram-bot or perception-telegram (who receives alerts) or the dashboard's other views beyond the Pulse fetch.

</details>

### G6

SHA-256 `9c7092b983d34d93ab86683d9f9c0c0d4e282886906a8b6b0672c6ce6b337940` · 14040 bytes

<details><summary>show</summary>

I mapped all 19 features against the code. Nothing was run, so every Runtime cell is NOT RUN. All work was read-only, and I stayed out of `data/` and never searched from the repository root.

| # | Feature | Code location (file:line, at most 3) | Class | What the code actually does (one line) | Runtime | Gap and next action |
|---|---|---|---|---|---|---|
| 86 | backup service | backup-service/app.py:47-68, :111-138; backup-service/Dockerfile:1-6 | PARTIAL | Endpoints for pg_dump, Redis BGSAVE, memory export, ledger export, list and restore. The image installs none of `pg_dump`, `psql` or `redis-cli`, so the Postgres, Redis and restore endpoints return 503 (the code audit already records this as KAI-BACKUPX-002). The "ledger" backup saves only `/ledger/stats`, not the ledger entries. | NOT RUN. output/health_scorecard.json (2026-02-25) records only that `/health` answered UP; not re-run | **finish**: install the client tools, back up the real ledger content, and add a schedule. No scheduled caller found in supervisor/, heartbeat/, Makefile or .github/workflows; scripts/backup_offsite.sh is a manual script |
| 87 | broker-bridge (Binance, read-only) + Broker view | broker-bridge/app.py:60-81, :121-220; dashboard/app.py:2536-2569 | PARTIAL | Real HMAC-signed, GET-only Binance client: balance, positions, orders, PnL, depth, trades, plus stocks and forex via yfinance. The whole file has no order-placing calls. The dashboard proxies all of it. The Quick Watch button posts to monitor-service `/rules`, which requires a service token, and the dashboard proxy sends none, so that step is refused | NOT RUN | **retain** the reads. **integrate**: send the service token on Quick Watch. Starts only with the `finance` profile (docker-compose.minimal.yml:518) |
| 88 | financial awareness / UK self-employment advisor | financial-awareness/app.py:394-404; agentic/app.py:2718-2740; kai-advisor/app.py:58-70 | PARTIAL | The service does real CIS, VAT, Income Tax and Class 4 NI arithmetic and returns `{cis, vat, tax}`. Both consumers read keys the service never returns (`cis_summary`, `vat_position`, `tax_estimate`, `gross_ytd`…): the chat context injection and the Finance view (dashboard/static/app.html:1535-1543). So the operator never sees the numbers. kai-advisor is a substring match over docs that otherwise answers "I'm just a simple KAI advisor stub". The router's TAX_ADVISORY path uses common/self_emp_advisor.py (agentic/router.py:414-428), simple keyword rules | NOT RUN | **integrate**: fix the key contract in both consumers. **replace** kai-advisor. The service only runs in docker-compose.full.yml (`finance` profile) |
| 89 | sovereign offline operation (end-goal) | common/llm.py:64-67; output/tts/app.py:3-6, :104; docker-compose.sovereign.yml:622-628 | PARTIAL | A local Ollama is on internal networks in the minimal stack. But speech output needs internet (Microsoft edge-tts), Groq and OpenRouter cloud LLM fallbacks switch on when their keys are set, the sovereign file has Ollama and TTS commented out ("TODO: enable GPU"), and speech-to-text is a stub by default (docker-compose.minimal.yml:346) | NOT RUN | **finish**: offline TTS (Piper is only "planned" in the tts docstring), a real STT backend, and a mode that refuses cloud LLM fallbacks |
| 90 | D98 cognitive fingerprint | agentic/app.py:2576-2581; agentic/cognitive_fingerprint.py:96-111, :126-150 | PARTIAL | Every `/chat` message appends a keyword-heuristic sample to `/data/cognitive_fingerprint.jsonl`. `infer()` returns `"stub_pending_gpu_clustering"`. `progress()`, `infer()` and `can_infer()` have no callers in agentic/*.py or dashboard/app.py | NOT RUN | **finish**: build the inference and wire it to a reader. Weigh whether keyword samples are worth 90 records |
| 91 | Telegram interface | telegram-bot/app.py:141-167, :203-298, :388-419 | GENUINE | Polls the Telegram API, streams agentic `/chat` back as replies, sends voice notes to audio-service for STT and replies with TTS audio, and has a service-authenticated `/alert` push. If `ALLOWED_CHAT_IDS` is empty, anyone can chat with it | NOT RUN | **retain**. Voice depends on STT (stub by default) and cloud TTS. Defined only in docker-compose.full.yml with the `external-egress` profile |
| 92 | dashboard / operator console (→ Mission Control) | dashboard/app.py (189 `@app.` routes); dashboard/static/app.html:33-93, :3272-3307 | GENUINE | FastAPI gateway with authenticated proxies. The UI has 16 views, not the 10 the README says. Ctrl+1–8, Ctrl+K, Escape (close dropdown or stop generation) and Ctrl+Shift+M are all real. The mode toggle is display-only (`/api/mode` returns `local_only`, dashboard/app.py:246-254) | NOT RUN. health_scorecard.json `/health` UP only | **retain**. Many views depend on backends that are not in the default stack |
| 93 | impact oracle | memu-core/app.py:7974-8070 | SIMULATED | Matches words of the action against goal titles, uses "skip" or "do" keywords to pick negative or positive, and fills in fixed consequence sentences. The "emotional forecast" is a count of negative keywords | NOT RUN | **replace** with model-backed prediction, or label it a heuristic. Dashboard reaches it via `/api/oracle/predict` (dashboard/app.py:1898) |
| 94 | notifications | output/notify/app.py:61-136; dashboard/app.py:2355-2390; common/resilience.py:35-80 | PARTIAL | Tries notify-send (libnotify is installed in the image), otherwise keeps an in-memory queue of 100 that the dashboard polls. There is no phone push. `/notify` and the dismiss endpoints require a service token, and the dashboard's proxy calls send none, so send and dismiss from the dashboard are refused | NOT RUN | **integrate**: send the token from the dashboard and persist the queue. Phone push exists only in telegram-bot `/alert` |
| 95 | operator model (P22) | memu-core/app.py:7645-7720, :5131-5144; agentic/app.py:2497-2531 | SIMULATED | Echo, nudge ladder and cross-mode run on every chat (agentic/app.py:2673, :2868). Emotion detection is a keyword count, and the echo lines are fixed templates. The flag `FF_OPERATOR_MODEL` (common/feature_flags.py:38) is checked nowhere, so turning it off changes nothing | NOT RUN | **replace** the keyword detection with real learning, and either wire the flag or remove it |
| 96 | presence: chat / voice / TTS / avatar | output/tts/app.py:88-121; output/avatar/app.py:29-38; docker-compose.minimal.yml:346 | PARTIAL | Chat streams through agentic (LLM may be the stub or the 0.5b placeholder model). TTS really synthesises audio with edge-tts, a cloud service; the 🔊 button calls it (app.html:2324). Speech-to-text is a stub by default. The avatar endpoint returns "avatar request queued" and does nothing else | NOT RUN. health_scorecard.json `/health` UP only | **finish** STT and offline TTS. **replace** the avatar (name only) |
| 97 | shadow branches | memu-core/app.py:8111-8215 | SIMULATED | Stores a decision and an alternative together with keyword-matched memories and goals. The "projection" is a fixed sentence ("would have changed your trajectory"). Kept in Redis, last 100 | NOT RUN | **replace** with real counterfactual reasoning, or relabel it as a decision journal |
| 98 | workspace manager | workspace-manager/app.py:12-14 | NAME_ONLY | Only a `/health` endpoint. No lifecycle logic | NOT RUN. health_scorecard.json `/health` UP only | **finish or remove** |
| 99 | engineering toolchain / CI gates | Makefile:420-425, :830-857; .github/workflows/core-tests.yml:110-115, :318-352; .pre-commit-config.yaml:7-58 | PARTIAL | `go_no_go`, `check-docs`, `coverage-floors` and pre-commit hooks are real and called from CI. Trivy and pip-audit run but do not block: the step is `continue-on-error` and only prints a warning, although the README says Trivy fails the build ("exit 1"). `make dep-audit` cannot fail (`|| echo WARNING`). `chaos_ci.py` kills and restarts three local processes | NOT RUN | **finish**: decide whether Trivy and pip-audit should block, and correct the README to match |
| 100 | feature-flag system (53 flags) | common/feature_flags.py:30, :102-115; agentic/app.py:759 | PARTIAL | A real registry of 53 flags (32 on, 21 off; matches the README) read from `FF_` variables. 15 of the 53 flags are never checked by any non-test code (see note 4) | NOT RUN | **finish**: wire the 15 flags or delete them. "All capabilities toggleable" is not true for those 15 |
| 101 | honest status / build &amp; test substrate | scripts/sync_docs.py:24-80, :297-315; docker-compose.minimal.yml | PARTIAL | The README counts are recomputed from the tree and checked by `check-docs`. The minimal stack has 36 services, 18 behind profiles, so 18 start by default (matches the README). Capability rows are not checked by any machine, and the GPU-era `can_*()` methods return False (agentic/global_workspace.py:25) | NOT RUN. output/health_scorecard.json is the only recorded result: `/health` for 22 services, 2026-02-25, includes an "orchestrator" service | **retain**. Add per-capability runtime evidence once runtime testing resumes |
| 102 | private overlay network (tailscale) | docker-compose.sovereign.yml:52-84 | PARTIAL | Compose configuration only, using the vendor tailscale image with the `external-egress` profile, on `egress-net` only. No KAI service routes through it, the auth key ships as a placeholder (`tskey-example`), and there is no KAI code for it | NOT RUN | **integrate**: decide which services the tailnet should expose |
| 103 | target hardware (Flow Z13 / Strix Halo) | common/gpu_utils.py:35-58; common/runtime.py:36-42; agentic/cortex.py:211-223 | NAME_ONLY | GPU detection only knows NVIDIA (CUDA, nvidia-smi); there is no AMD GPU (ROCm/HIP) path. The NPU path raises NotImplementedError. No GPU device passthrough (`/dev/kfd`, `/dev/dri`, `gpus`) in any of the 3 compose files | NOT RUN | **finish**: AMD GPU detection, device passthrough and NPU qualification once the hardware arrives |
| 104 | zero telemetry / local core | memu-core/Dockerfile:70; docker-compose.full.yml:500; scripts/enforce_egress.sh:1-40 | PARTIAL | Hugging Face telemetry is turned off in memu-core and memu-graph, `TELEMETRY_DISABLED` is set for memu-graph, and five of the minimal stack's seven networks are internal-only. `enforce_egress.sh` has no caller in Makefile, scripts/setup.sh or the workflows. Outbound paths exist: edge-tts, Binance, yfinance, Telegram, and Groq/OpenRouter when keys are set | NOT RUN | **finish**: wire the egress enforcement and make every outbound path explicit and opt-in |

**Notes**

1. **One key mismatch breaks finance end to end.** `/finance/summary` returns `cis`, `vat` and `tax`. The agentic injection and the dashboard Finance view both read `cis_summary`, `vat_position` and `tax_estimate`, with different sub-keys again. So `FF_FINANCIAL_CONTEXT` adds nothing to chat, and the Finance tiles show blanks. This is from reading the code, not from a run.
2. **Two defects in financial-awareness, found while reading.** I fixed neither. The tax-year start test `today.month &gt;= 4 and today.day &gt;= 6` (financial-awareness/app.py:81, :95, :212) picks the previous tax year whenever the day of the month is below 6, including today (3 November), so year-to-date figures include the prior tax year. The VAT threshold defaults to £85,000 (line 36). From my own knowledge, not this repository, the UK threshold has been £90,000 since April 2024.
3. **The missing service token is not limited to one route.** Dashboard proxies (`_proxy_post` → `resilient_call`, plus the direct `pooled_client` deletes) never set an Authorization header: I searched common/resilience.py and dashboard/app.py for bearer, authorization, `KAI_SERVICE_TOKEN` and headers and found nothing. notify-service `/notify`, `/pending` deletes and monitor-service `/rules` all require a token. No compose file sets `KAI_ALLOW_UNAUTHENTICATED`. I checked only those two targets; other proxied write endpoints may have the same gap.
4. **15 of 53 flags have no consumer.** These are DREAM_PHASE_7, CHECKPOINT_AUTO, TREE_SEARCH, PRIORITY_QUEUE, SAGE_CRITIQUE, IMAGINATION_ENGINE, OPERATOR_MODEL, NARRATIVE_IDENTITY, CONSCIENCE_FILTER, MARS_CONSOLIDATION, SELF_ASSESSMENT, SECURITY_AUDIT, EVOLVER_ENABLED, SAGE_SELF_REVIEW and PREDICTIVE_EMPATHY. How I counted: in `.py` files under all 47 top-level code directories (excluding data/, kai-pm/, docs/, runtime-logs/, `test_*`, conftest and feature_flags.py itself), I looked for `is_enabled("NAME")` or the literal `FF_NAME`. A flag name built at runtime would be missed.
5. **Features 93, 95 and 97 are one mechanism.** All are P22 in memu-core/app.py: keyword matching (`_detect_emotion`) plus fixed sentences, stored in capped Redis lists. They should be fixed or relabelled together.
6. **Where the services are defined.** backup-service, telegram-bot, financial-awareness, kai-advisor, workspace-manager and avatar-service are defined only in docker-compose.full.yml; tailscale only in docker-compose.sovereign.yml. So `make core-up` (minimal file) starts none of them. broker-bridge is in the minimal file but behind the `finance` profile.
7. **The only recorded runtime result is a health check.** output/health_scorecard.json (dated 2026-02-25, last committed 2026-08-05) shows `/health` answering for 22 services, not that any capability works. Its source is unknown, and it lists an "orchestrator" service I did not find among the minimal-stack services. I did not re-run it.
8. **One lead in kai-pm/STUBS_AND_PLACEHOLDERS.md is out of date.** It says the Thinking view depends on `agentic-introspect`, but `/api/thinking` actually reads from `agentic` (dashboard/app.py:893-899). I did not check its other rows line by line.
9. **What I did not look at.** I did not read the cross-mode scan and nudge-ladder bodies in full, the monitor-service rule engine, the perception/audio STT code beyond its defaults, or the PROJECT_STATUS.md checklist logic. I did not open the CODE_AUDIT files except to confirm backup KAI-BACKUPX-002.

</details>

### G7

SHA-256 `035544bc580e7ed14a338b50207b65ee2185c96cfffb587a071a01978d521acd` · 13265 bytes

<details><summary>show</summary>

| # | Feature | Code location (file:line, at most 3) | Class | What the code actually does (one line) | Runtime | Gap and next action |
|---|---|---|---|---|---|---|
| K1 | Protect and care within earned authority | agentic/trust_integration.py:165; agentic/moral_core.py:213; agentic/trust_core.py:71 | PARTIAL | Before an autonomous act, a trust-level check runs, then a keyword match against the Ohana values profile (a match on a harm keyword blocks), and the attempt is logged to the trust ledger. Nothing more specific to "care" or "protect" was found. | NOT RUN | Retain the gate. Finish: define "protect/care" as concrete behaviours with checks. |
| K2 | Stewardship for Dainius's daughter | agentic/trust_core.py:95; agentic/wisdom_graph.py:50; agentic/wisdom_ingestion.py:69 | NAME_ONLY | `daughter_profile` is only a capability label at the GUARDIAN level, and no caller asks for it (searched the code folders). Elsewhere "daughter" appears only as regex and seed-graph text. | NOT RUN | Design first, then build: a protected-person model with limited authority. |
| K3 | Succession | none | ABSENT | Searched `*.py` in all top-level code folders (not data/, kai-pm/, docs/) for `successor\|succession` and `beneficiar`: 0 files. | NOT RUN | Design needed: a successor-authority record that keeps lineage and never impersonates Dainius. |
| K4 | Keeps running when Dainius is unavailable | supervisor/app.py:100,159; backup-service/app.py:168,215; common/resilience.py:186 | PARTIAL | Supervisor health-sweeps services, uses circuit breakers and a watchdog on its own loop. Auto-recovery is off unless `SUPERVISOR_RECOVERY_ENABLED=true`. Backup and Postgres restore run only on request; no backup scheduler found in backup-service/ or supervisor/. | NOT RUN | Integrate: scheduled backups, a restore drill, and decide whether recovery should be on by default. |
| K5 | Economic self-sufficiency, separate from family assets | agentic/paper_trader.py:86; financial-awareness/app.py:366; broker-bridge/app.py:121 | PARTIAL | Paper trading only (no real capital), a read-only broker, and UK tax and VAT calculators for Dainius. No income mechanism, no Kai-owned budget, no separation from family assets (searched `segregat\|family.?asset\|protected.?asset`: 0). | NOT RUN | Retain the simulation. Design: a separate Kai ledger and budget with revocable authorisation. |
| K6 | Long-horizon survival | scripts/auto_rotate_hmac.py:1; scripts/backup_offsite.sh:1; common/llm.py:57 | PARTIAL | Manual scripts rotate the HMAC secret and make an encrypted offsite archive. LLM backends come from env settings, with optional Groq/OpenRouter. No monitoring of provider loss, credential expiry or running costs (searched those terms: 0 runtime hits). | NOT RUN | Finish: an inventory of dependencies, expiries and costs, plus a migration runbook. |
| K7 | One KAI, one governed decision path | scripts/security/check_architecture_rules.py:40; agentic/trust_integration.py:222; common/autonomy/legacy_bridge.py:140 | PARTIAL | A CI gate stops decision modules from importing actuators. But several authorities run side by side (old trust scalar, scoped authority in advisory mode, tool-gate, PolicyEngine), and paper trades have two paths: the old `paper_trader` and the `/uh/paper-trade` slice. | NOT RUN | Integrate: move actuators onto one path, then switch `KAI_AUTONOMY_ENFORCE` on. |
| K8 | Replaceable organs | common/llm.py:196,220; common/model_registry.py:1; common/contracts/base.py:88 | PARTIAL | Models are swapped by env settings or the registry. Typed contracts give stable interfaces. With no backend configured, the LLM returns a hashed stub text (S1 in the stubs register). No swap procedure that preserves lineage. | NOT RUN | Retain. Add swap qualification checks (see K9). |
| K9 | Lineage Manifest / Identity Ledger | none | ABSENT | Searched code folders for `identity.?ledger\|lineage.?manifest`: 0. "lineage" appears only for evidence, erasure and world-state supersession. | NOT RUN | Design needed (the source document says it does not authorise building it). |
| K10 | Three kinds of state with their own change rules | scripts/test_invariant_guards.py:1; scripts/security/check_doctrine_integrity.py:1; common/autonomy/evidence_service.py:48 | PARTIAL | Only fragments: source-level CI guards protect governance code; the evidence store grades records and refuses to let self-generated sources qualify, but no production code writes to it. No runtime split into invariants, organs and learned state. | NOT RUN | Design a state taxonomy, then wire evidence grading into memory writes. |
| K11 | Growth without architectural amnesia | .claude/skills/kai-handoff/handoff.py; scripts/security/check_doctrine_integrity.py:1; agentic/introspect_app.py:58 | PARTIAL | Continuity lives in development tooling: the handoff log, append-only decisions, the doctrine drift gate. The dream and evolver endpoints exist. No runtime mechanism carries architecture memory across versions. | NOT RUN | Retain the tooling. Link to K9. |
| K12 | Kingsman control loop | common/vertical_slice/paper_trade_slice.py:143,283; agentic/app.py:637 | PARTIAL | Steps that really run: market event, ingress, world-state snapshot, proposal, policy, approval gate, capability issue and consume. Specialists, deliberation and review steps are absent. The Ohana and safety checks have no handlers ("advisory only"). The actuator receipt and the `CONFIRMED` outcome are fixed values, not real results. With `auto_approve=true` the caller approves as "dainius". No learning step. | NOT RUN | Finish: a real actuator call, an independent verifier, real assessors and authenticated human approval. |
| K13 | Evidence Plane | common/contracts/base.py:70,88; common/world_state/snapshot_store.py:44; trust-ledger/ledger.py:94 | PARTIAL | Real parts: provenance and digest-bound contracts, a world store that keeps conflicts and staleness, a graded evidence store, and a hash-chained trust ledger. The perception spine runs in shadow mode by default (journals only; world state is fed only when `KAI_PERCEPTION_MODE=active`). The evidence store is in memory only and nothing writes to it. | NOT RUN | Integrate: switch the spine to active, persist evidence, and make policy read it. |
| K14 | A4 structural self-diagnosis | scripts/security/report_runtime_topology.py:1; scripts/security/check_compose_drift.py:1; scripts/security/check_architecture_rules.py:1 | PARTIAL | Offline CI and report scripts derive the service topology, check compose drift and check architecture rules. None is a runtime organ. No root-cause reasoning, repair proposals or post-repair verification found in the code folders. | NOT RUN | Design is waiting on Item 8 (ordered before A4). Reuse these scripts as inputs. |
| K15 | One recovery loop | agentic/app.py:2046; house-doctor/app.py:150,215; supervisor/app.py:159 | PARTIAL | Proactive loop sends observations to House Doctor, which tags them by keyword, applies a rule table, files a report and sends a notification. Watchdog events go to the FSM. Supervisor runs separately; its "heal" is a POST to `/recover` (off by default), and HTTP 200 counts as success. `HealingEngine` (common/resilience.py:266) is used only by a test. No approval step and no independent verification. | NOT RUN | Integrate the separate mechanisms into one loop, add an approval step, and verify recovery independently. |
| K16 | Truthful degradation | common/degraded.py:86,104,214; common/runtime.py (CircuitBreaker); supervisor/app.py:188 | PARTIAL | Honest-degradation replies (503, or 200 that names the missing sources) are used in 4 services: dashboard, memu-core, memu-graph, telegram-bot. 37 non-script files record swallowed failures. Circuit breakers contain failures. Recovery is off by default and is not verified. | NOT RUN | Retain. Adopt the degradation replies in other services and add recovery verification. |
| K17 | Values kept separate from facts | common/policy_bridge/assessment.py:155; common/autonomy/authority.py:62; agentic/app.py:2940 | PARTIAL | The assessor registry cannot grant permission, and any block is final. The nonce-bound `confirm_value` exists but nothing calls it. **Contrary path:** every chat turn passes Kai's own reply as `decision=response[:300]` into OhanaCore `record_decision`, and posts the user message to `/memory/values/learn` with `outcome: positive`. | NOT RUN | Replace that write path. Use `confirm_value` for operator values. |
| K18 | Model runtime manager and hardware modes | agentic/model_selector.py:114; agentic/cortex.py:211; sysmetrics/app.py:142 | PARTIAL | Route-based model selection is wired into chat. The NPU path raises `NotImplementedError`. Battery level is read but nothing uses it. No residency profiles or battery/portable/docked/maintenance modes (searched those terms: hits only in cortex docstrings and sysmetrics). | NOT RUN | Build after hardware is confirmed. Hardware modes are in design only. |
| K19 | Earned autonomy | common/autonomy/authority.py:38; common/autonomy/legacy_bridge.py:55,140; skill-hunter/app.py:210 | PARTIAL | Scoped, expiring, revocable grants are real code but held in memory, advisory by default, and no production code issues a grant. The old TrustCore scalar binds instead and can promote itself on scores (trust_core.py:373). "Skill generation" writes a markdown template naming a PyPI package; there is no sandbox test. No agent swarms. | NOT RUN | Finish: persist grants, collect disagreement data, then turn enforcing on. |
| K20 | Proactive organism loop | agentic/app.py:2046,2064,2275 | PARTIAL | Polls 6 sensors, writes observations to memory, bids to the workspace, and calls House Doctor. "Forecast" is a surprise check against the previous snapshot that is only logged. No ignore/watch/propose/notify/act decision step and no learning whether it helped. **Gating defect by reading:** it asks the gate for `proactive_observation`, which is not in `CAPABILITY_GATES`, so `can_do` falls back to the GUARDIAN level (trust_core.py:196) and the loop is skipped unless the stored level is GUARDIAN. | NOT RUN | Fix the capability mapping, then add the decide and learn steps. |

Notes:
1. **Shared wiring gap:** the governed-pipeline packages under `common/` (world_state, policy_bridge, proposal_workspace, actuator_registry, autonomy, erasure) are imported outside `common/` only by `agentic/app.py`, `agentic/trust_integration.py` and `scripts/`. In production they serve only `/uh/status`, `/uh/actuators`, `/uh/paper-trade` and `/uh/erasure`. Outside tests and `scripts/`, nothing calls `ActuatorRegistry.dispatch` or `attach_all_handlers`. K7, K12, K13, K17 and K19 are mostly this one mechanism, built but not yet in the live path.
2. **Default switches keep the new path inactive:** `KAI_PERCEPTION_MODE=shadow`, `KAI_AUTONOMY_ENFORCE=false`, `SUPERVISOR_RECOVERY_ENABLED=false`. With these, the older paths decide.
3. **In-memory only:** EvidenceService, AutonomyAuthority grants, value confirmations and the House Doctor ring buffer have no persistence I found in the files I read. They reset on every restart.
4. **K20 finding needs a runtime check:** it comes from reading the code. The stored trust level lives under `data/trust/`, which I did not open, so whether the observer actually runs today is UNVERIFIED.
5. **Approval is not authenticated:** `ApprovalGate.approve` takes the approver as a plain string (common/policy_bridge/approval.py:47). `/uh/paper-trade` lets any caller holding the `paper_trade_slice` service credential stamp "dainius" by setting `auto_approve=true`.
6. **Missing capability names:** besides `proactive_observation`, the gate calls `web_scout_fetch` / `web_scout_search` (and their `_autonomous` forms) are not in `CAPABILITY_GATES` either, so they fall back to GUARDIAN. I did not check the paper_trader and model_council capability names.
7. **Search universe for the negatives:** `*.py` under every top-level code folder except data/, kai-pm/, docs/, output/ and runtime-logs/. Matches under `scripts/test_*` were removed from the hit lists for the term searches; the K3 and K9 searches matched nothing even before that filter. Non-Python files were not searched, apart from the two below.
8. **Rule disclosure:** I ran two single-file greps on repository-root files: `docker-compose.full.yml` for "backup" and `Makefile` for "auto_rotate". Neither was recursive and neither touched data/. No other root search; `data/SOUL.md` was never touched.
9. **Recorded runtime results:** `runtime-logs/sovereign/` is empty. I did not open `output/health_scorecard.json`. I used the lead `kai-pm/STUBS_AND_PLACEHOLDERS.md`; its S1 item (the LLM stub) bears on K8. I used no CODE_AUDIT_BATCH files.
10. **Behaviour outside the scope of K1–K20:** these are recorded here, not acted on. Backup-service has three different default ports in its callers: 8048 in `mutating_handlers.py`, 8054 in `dashboard/app.py` and `backup_offsite.sh`. The trust ledger counts every chat reply with conviction ≥ 7 as +0.1 "consistency" evidence, which is self-generated. It cannot promote past OBSERVER today, because no production caller records judgment or values evidence (the only `record_evidence` call adds consistency).

</details>

### RECON

SHA-256 `9fd4d2ad9e9931f5636cad44e7e8ba6c3a96b020be0d32643d6510b8168a08ac` · 14006 bytes

<details><summary>show</summary>

I read all 21 files in full and found **26 proposed additions**. 10 are new and 16 extend an existing entry.

One limit on the "already covered" judgement: I compared against the short feature names in the list, not against README.md. README.md is in the repository root, which you put out of bounds. So a row marked "Extends" may already be partly described in the README's own text. That matters most for K45 (paper trading).

## 1. Proposed additions

| Proposed id | Feature (one line) | Source file:line | Extends existing # or NEW | Exact quote (short) |
|---|---|---|---|---|
| K21 | **Contingency and fail-safe library:** versioned, tested playbooks for known failures. Containing a failure, degrading and repairing need separate permission. Retry and recovery attempts have limits, after which Kai escalates instead of looping. Failing skills are quarantined, and playbooks can be combined. A playbook that worked before gains no extra authority. | CONTINGENCY_AND_FAILSAFE_LIBRARY_DESIGN:26, 133–145, 159, 254–265, 267–279 | NEW (sits beside K15, #77) | "KNOWN FAILURE MODES SHOULD HAVE KNOWN, TESTED, GOVERNED RESPONSES." |
| K22 | **Lasting record of goals, obligations, promises, deadlines, watches and timers.** Kai's initiative comes from the gap between what is true now and what should be true. | PROACTIVE_ORGANISM_DOCTRINE:94–115; MASTER_..._V0_2:507–524 | Extends #53, #58 | "Without goals and obligations, proactive behaviour degenerates into notification heuristics." |
| K23 | **Interruption policy:** stays quiet on purpose. Takes account of whether Dainius is busy, focused, asleep or offline. In focus mode only urgent items get through, and a critical alert can break through only the suppression types it is allowed to. Learns what to say now and what can wait for a daily or weekly summary. | PROACTIVE_ORGANISM_DOCTRINE:57–70, 202–203, 271–272 | Extends #55, #94, K20 | "The ability to **stay quiet deliberately** is part of mature proactivity." |
| K24 | **UNKNOWN is a real answer, and Kai knows its own blind spots.** States such as UNOBSERVED, STALE, CONFLICTING and SOURCE_UNAVAILABLE are kept as they are. A sensor that is down is not a "no". When its watchers fail, Kai reports PROACTIVE AWARENESS DEGRADED instead of "nothing wrong". | PROACTIVE_ORGANISM_DOCTRINE:236–241; CANDIDATE_ARCHITECTURE_V0_1:355–365 | Extends K13, K16 | "Kai must know when his ability to notice has failed." |
| K25 | **Predictions stay separate from facts**, even when a prediction triggers proactive action. | PROACTIVE_ORGANISM_DOCTRINE:121–129 | Extends #40 (sits beside K17) | "Kai must not convert predictions into current facts simply because they trigger proactive action." |
| K26 | **Operator-availability states** (present, temporarily unreachable, unavailable for a long time, succession review, succession confirmed). During a temporary absence Kai keeps essentials running, pays only pre-approved essential costs, contacts named trusted people and avoids irreversible actions. A dead-man timer alone never counts as evidence. | LONG_HORIZON_STEWARDSHIP:66–80, 123, 303–313 | Extends K3, K4 | "Temporary unavailability must **not** silently trigger permanent authority transfer." |
| K27 | **Trusted human steward roles:** emergency contact, technical custodian, legal executor or trustee, financial steward, recovery key-holder. Each has narrow, explicit powers. | LONG_HORIZON_STEWARDSHIP:317–330 | Extends K3 | "narrow, explicit powers rather than one universal master key" |
| K28 | **Data stewardship at succession:** decides what data transfers, what stays sealed or is deleted, what the daughter may access, retention periods, encryption-key succession, and the successor's right to remove data. | LONG_HORIZON_STEWARDSHIP:336–352 | Extends K3 | "\"Outlive me\" does not imply \"reveal everything after me.\"" |
| K29 | **Data labels travel with the data:** each record's classification, purpose and origin follow it through retrieval, model context, choice of local or remote model, tool use and anything sent out. Covers retention, and erasure (including cryptographic erasure) set against the permanent audit record. | MATURATION_..._2026-08-24:37, 696–712; MASTER_..._V0_2:1098, 1396 | NEW | "a local-only/private memory cannot be sent to a remote model merely because that model was selected as a specialist" |
| K30 | **Outside content is data, never instructions** (defence against prompt injection and memory poisoning). Web pages, documents, email, screen, clipboard and model output cannot become preferences, trusted memory, goals or instructions. | CANDIDATE_ARCHITECTURE_V0_1:1129; MATURATION_..._2026-08-24:168–172, 280 | NEW (sits beside K17) | "untrusted retrieved/web/document text stays tagged as data, not instructions" |
| K31 | **Protected approval screen:** high-consequence approvals need a second identity check and show the exact action fingerprint (digest). Saying yes in chat is not a high-consequence approval. | CANDIDATE_ARCHITECTURE_V0_1:697–699; MASTER_..._V0_2:679–683 | Extends K12, #62 | "ordinary conversational text should not be treated as cryptographic high-consequence approval by default" |
| K32 | **Durable action workflows:** each action is recorded so it survives a restart, carries a key that prevents doing it twice, and can be cancelled or undone where that is valid. An action whose result is unknown is checked against the real target before any retry. | MATURATION_..._2026-08-24:353–364; CANDIDATE_ARCHITECTURE_V0_1:742–754 | Extends K12, #67 | "absence of a receipt is never permission to retry a consequential operation" |
| K33 | **Network limits per permission:** each permission sets allowed destinations, methods, data classes, upload and download limits and time budget. | CANDIDATE_ARCHITECTURE_V0_1:779–792; MASTER_..._V0_2:750–761 | Extends #67 | "Broad internet access must not become implicit authority." |
| K34 | **Tamper-evident audit log:** signed periodic checkpoints, copied to an independent backup location and checked on restore and startup. | CANDIDATE_ARCHITECTURE_V0_1:1055–1065 | Extends #64 | "A Redis hash chain alone is insufficient as long-horizon tamper evidence" |
| K35 | **Automated restore drills:** isolated test restores, recovery-point and recovery-time targets per data class, at least one copy off the device, and a restore must prove lineage and authority. | MASTER_..._V0_2:909–925; CANDIDATE_ARCHITECTURE_V0_1:1242–1254, 1487–1489 | Extends #86, K9 | "“Containers boot” is not sufficient restore success." |
| K36 | **Safe archive / read-only preservation mode:** Kai stops on purpose instead of carrying on when authority or funding cannot be established. | CANDIDATE_ARCHITECTURE_V0_1:1238–1239, 1254; ROOT_ARCHITECTURE:326; LONG_HORIZON_STEWARDSHIP:456 | NEW | "safe preservation mode exists when authority or funding cannot be established" |
| K37 | **Update protection against rollback and freezing:** refuses older releases that were validly signed but are vulnerable, detects update information that has stopped refreshing, and allows rollback only as an explicit, limited, visible emergency step. | MATURATION_..._2026-08-24:38, 447–462, 760–800 | NEW | "Is this still an allowed release to run, or is somebody replaying a valid but obsolete/vulnerable version?" |
| K38 | **Ability to change cryptography over Kai's lifetime:** signatures and keys carry algorithm and key IDs, so stronger algorithms can be adopted while old records stay verifiable. | MATURATION_..._2026-08-24:39, 464–479 | NEW (sits beside #68) | "design key/algorithm transitions so old data can remain verifiable while new operations move to stronger algorithms" |
| K39 | **Resource and consequence budgets across all organs:** tokens, memory, network bytes, retries, queues, interruptions, financial consequence and storage. Expensive work is admitted only within budget, and degradation is explicit. | MATURATION_..._2026-08-24:36, 662–690 | Extends K18, #77 | "an exhausted budget cannot silently switch to a less-governed implementation" |
| K40 | **Clear meaning for each timestamp:** when an event happened, when Kai observed it, when a decision was made, when something expires, and timed deadlines. A caller's clock is never trusted for authority, and clock-sync problems are made visible. | MATURATION_..._2026-08-24:35, 637–656 | NEW | "caller-supplied wall time never controls authority, retention or lease validity without qualification" |
| K41 | **The doctor uses the engineering doctrine as diagnostic rules:** banked lessons become patterns to detect and seeds for generating tests. | README_ARCHITECTURE_REFRESH_PLAN:155–181 | Extends K14, #70 | "use these not merely as documentation but as diagnostic anti-patterns and test-generation seeds" |
| K42 | **Self-check of lineage and learned state:** spots stale, corrupt or conflicting learned state, an organ that was replaced but still holds authority, and a migration that changed authority. Has contingency families for lineage divergence. | PRIMARY_MISSION_CANON_PROPAGATION_MAP:175–183, 205–213 | Extends K14 | "is a supposedly replaced organ actually still authoritative somewhere?" |
| K43 | **Mission Control panels:** active watches, prepared proposals and suppressed alerts; standing permissions and when they expire; resilience and playbook status with remaining retry budget; long-horizon status (runway, backup age, last restore drill, credential expiry, succession readiness). A green tick disappears when its evidence is withdrawn. | PROACTIVE_ORGANISM_DOCTRINE:247–258; CONTINGENCY_AND_FAILSAFE_LIBRARY_DESIGN:308–318; LONG_HORIZON_STEWARDSHIP:424–436; MASTER_..._V0_2:989–1037 | Extends #92 | "This lets Dainius govern initiative rather than only inspect completed actions." |
| K44 | **Deliberation record:** keeps the exact task, world-state snapshot, evidence, model identities, prompt version, disagreement, assumptions, and budget or stop reason. Always weighs a "do nothing" option. | MASTER_..._V0_2:563, 586–603; CANDIDATE_ARCHITECTURE_V0_1:1762 | Extends #64, #71 | "compare alternatives including no-action" |
| K45 | **Paper trading and market/strategy/opportunity intelligence.** Real money movement is kept separate: it needs a separate permission and an independent check afterwards. | EXISTING_KAI_EVOLUTION_MASTER_PLAN_V0_3:119, 151–153, 781–793 | Extends #87, #88 (may already be in the README text) | "paper-trading vertical slice" |
| K46 | **Self-maintenance and live organ upgrades:** detects ageing parts and proposes or performs updates, patches, certificate renewal and storage cleanup within earned limits. Upgrades go through trial, rollback and post-update checks, without taking the whole of Kai offline. | LONG_HORIZON_STEWARDSHIP:356–376; ORGANIC_RESILIENCE_ARCHITECTURE_DOCTRINE:306–321; CANDIDATE_ARCHITECTURE_V0_1:1300–1319 | Extends K6, K8, K19 | "detect obsolescence and propose/perform maintenance within earned limits" |

File names in the source column drop the `kai-pm/KINGSMAN_` prefix and `.md`. "MASTER_..._V0_2" is `KINGSMAN_MASTER_ARCHITECTURE_AND_PROFESSIONALISATION_CANDIDATE_V0_2.md`, and "MATURATION_..._2026-08-24" is `KINGSMAN_EXISTING_KAI_MATURATION_IMPLEMENTATION_RESEARCH_REISSUE_2026-08-24.md`.

**Considered and left out:**
- **Already covered:** single-use permission checked at the final step, independent outcome checks, no self-verification (K12). Service identity and "membership ≠ identity" (#62). Hardware modes and the model manager (K18). Lineage manifest (K9). Supervisor split (K15). Financial separation (K5). Dependency end-of-life watching (K6). Learning only from checked outcomes (K12).
- **Likely already in the README text:** mood arcs, gratitude/relationship continuity, theory of mind, legacy time capsules and policy memory. They appear only as names in the evolution plan (lines 485–516). I could not check them against README.md, so they are not proposed.

## 2. Files read (all 21, in full)

| File | Result |
|---|---|
| KINGSMAN_PROACTIVE_ORGANISM_DOCTRINE.md | K22, K23, K24, K25, K43 |
| KINGSMAN_ORGANIC_RESILIENCE_ARCHITECTURE_DOCTRINE.md | K46 (rest covered by K8, K11, K14, K16) |
| KINGSMAN_CONTINGENCY_AND_FAILSAFE_LIBRARY_DESIGN.md | K21, K43 |
| KINGSMAN_LONG_HORIZON_STEWARDSHIP_AND_SUCCESSION.md | K26, K27, K28, K36, K43, K46 |
| KINGSMAN_ROOT_ARCHITECTURE_AND_CANON_ALIGNMENT.md | K36 (rest covered) |
| KINGSMAN_PRIMARY_MISSION_CANON_PROPAGATION_MAP.md | K42 |
| KINGSMAN_ARCHITECTURAL_PILLARS_INDEX.md | no new features |
| KINGSMAN_README_ARCHITECTURE_REFRESH_PLAN.md | K41 (otherwise document process) |
| KINGSMAN_MASTER_ARCHITECTURE_AND_PROFESSIONALISATION_CANDIDATE_V0_2.md | K22, K29, K31, K33, K35, K43, K44 |
| KINGSMAN_CANDIDATE_ARCHITECTURE_V0_1_DEEPSEEK_REVIEW.md | K24, K30, K31, K32, K33, K34, K35, K36, K44, K46 |
| KINGSMAN_EXISTING_KAI_EVOLUTION_MASTER_PLAN_V0_3.md | K45 (otherwise migration process) |
| KINGSMAN_EXISTING_KAI_MATURATION_IMPLEMENTATION_RESEARCH_REISSUE_2026-08-24.md | K29, K30, K32, K37, K38, K39, K40 |
| KINGSMAN_ENGINEERING_ARCHITECTURE_DRAWING_SET_V0_1.md | no new features (repeats V0_1 and V0_2) |
| KINGSMAN_ENGINEERING_ARCHITECTURE_VISUAL_STANDARD.md | no new features |
| KINGSMAN_MASTER_CANON_INPUT_MANIFEST_V0_2.md | no new features |
| KINGSMAN_ARCHITECTURE_DOCUMENT_AUTHORITY_INDEX_V0_2.md | no new features |
| KINGSMAN_ARCHITECTURE_DOCUMENT_AUTHORITY_INDEX_V0_3.md | no new features |
| KINGSMAN_ARCHITECTURE_DOCUMENT_AUTHORITY_INDEX_CURRENT.md | no new features (restates K37–K40 obligations) |
| KINGSMAN_CANON_SYNCHRONISATION_REGISTER.md | no new features |
| KINGSMAN_CANON_SYNCHRONISATION_REGISTER_V0_3.md | no new features |
| KINGSMAN_CANON_SYNCHRONISATION_REGISTER_CURRENT.md | no new features (restates K29–K30, K32, K37–K40) |

I skipped the two files K1–K20 came from, as you asked. I did not open data/, SOUL.md or README.md, and made no edits.

</details>

