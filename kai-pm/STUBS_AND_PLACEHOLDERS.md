# Stubs, Placeholders & Shells

**Last updated:** 2026-07-21
**Purpose:** Single reference for every known stub, placeholder, shell UI, hardcoded value, or graceful-degradation path that must eventually be replaced with real implementations. Things in this file are NOT bugs — they are intentional, tracked deferrals.

---

## How to read this file

| Category | Meaning |
|---|---|
| `UI_SHELL` | Dashboard view that renders but shows "unavailable" when its backend isn't in the active compose profile |
| `LLM_STUB` | LLM response mode that fires when no live Ollama/Groq/OpenRouter backend is reachable |
| `FAKE_IMPL` | A working but deliberately simplified implementation (e.g. hash-based embeddings instead of a real model) |
| `HARDCODED_DEFAULT` | A credential, URL, or value with a safe dev default that must be overridden in production |
| `PLACEHOLDER_SECRET` | An env var that ships blank or with a `change-me` value and must be set before any real use |
| `DEFERRED_FEATURE` | A code path, endpoint, or feature that exists in the architecture but isn't wired end-to-end yet |

**Unlock column** — what removes this entry from the list.

---

## UI Shells — Dashboard Views

These views proxy to backend services. They degrade gracefully to `"unavailable"` cards when the service isn't running, but they are NOT broken — they function correctly with `make full-up`.

| View | Key | Backend | Proxy endpoints | Unlock |
|---|---|---|---|---|
| **Thinking** | `Ctrl+3` | `agentic` + `agentic-introspect` | `/api/thinking`, `/api/tempo`, `/api/boundaries`, `/api/silence`, `/api/dream` | `make full-up` or `docker-compose.full.yml up agentic agentic-introspect` |
| **Goals** | `Ctrl+5` | `agentic` | `/api/goals`, `/api/reminders`, `/api/tasks` | `make full-up` |
| **Logs** | `Ctrl+7` | `agentic` | `/api/logs` | `make full-up` |

**Functional since last update (no longer shells):**
- Canvas (D3 v7, J1 — 2026-07-21)
- Soul editor (SOUL.md/AGENTS.md, J6 — 2026-07-21)
- Diary (Memory Diary, J5 — 2026-07-21)

---

## LLM Stub Mode

**File:** `common/llm.py:215` — `stub_mode` property, `_stub_response()` at line 359

**What it does:** When `OLLAMA_URL` is unreachable AND no Groq/OpenRouter keys are set, `LLMRouter` enters stub mode. All specialist queries return deterministic hash-tagged placeholder text like `[specialist stub-a3f2] Wire OLLAMA_URL...`. Responses carry `source="stub"`.

**Detection:** `scripts/behavioral_scoreboard.py` checks for stub markers in LLM output and skips scoring if `router.stub_mode` is true.

**Unlock:** Set `OLLAMA_URL` to a live Ollama instance, OR set `GROQ_API_KEY`, OR set `OPENROUTER_API_KEY` in `.env`.

---

## Fake Embeddings

**File:** `memu-core/app.py:954` — `_ALLOW_FAKE_EMBEDDINGS`, `_embed()` at line ~967

**What it does:** When `sentence-transformers` isn't installed and `MEMU_ALLOW_FAKE_EMBEDDINGS=true`, memu-core uses a deterministic SHA-256 hash-based embedding instead of a real model. Similarity scores become meaningless (all distances are uniform) but the system stays up.

**Active in:** CI (`MEMU_ALLOW_FAKE_EMBEDDINGS=true` in `core-tests.yml`).

**Unlock:** Install `sentence-transformers` (requires ~1 GB model download). Set `MEMU_ALLOW_FAKE_EMBEDDINGS=false` (the default). Real embeddings require `all-MiniLM-L6-v2` to be present at startup.

---

## Hardcoded Dev Defaults

These are legitimate dev defaults that **must be overridden in any non-dev environment**.

| Location | Variable | Dev default | Why dangerous | Override |
|---|---|---|---|---|
| `docker-compose.full.yml:28` | `POSTGRES_PASSWORD` | `localdev` | Readable in compose file; weak password | Set `DB_PASSWORD` in `.env` |
| `docker-compose.full.yml:91,122,712` | `PG_URI` | `postgresql://keeper:localdev@…` | Hardcoded password in URI | Set `DB_PASSWORD`; compose will substitute |
| `docker-compose.minimal.yml:29,121,153` | `POSTGRES_PASSWORD` / `PG_URI` | `localdev` | Same as above | Set `DB_PASSWORD` in `.env` |
| `docker-compose.sovereign.yml:33,151,176,208,281` | `DB_PASSWORD` / `DATABASE_URL` | `localdev` | Same | Set `DB_PASSWORD` in `.env` |

---

## Placeholder Secrets (blank or `change-me` defaults)

These must be set before the relevant feature can work at all.

| File | Variable | Current value | Required for |
|---|---|---|---|
| `.env.example:24` | `INTERSERVICE_HMAC_SECRET` | `change-me-in-production` | Inter-service HMAC auth (all services) |
| `.env.example:15` | `TELEGRAM_BOT_TOKEN` | *(blank)* | Telegram bot integration |
| `.env.example:16` | `TELEGRAM_CHAT_ID` | *(blank)* | Telegram bot integration |
| `.env.example:63` | `GROQ_API_KEY` | *(blank)* | Groq cloud LLM fallback |
| `.env.example:68` | `OPENROUTER_API_KEY` | *(blank)* | OpenRouter cloud LLM fallback |

---

## LLM Model Placeholder

**File:** `common/llm.py:53` — `_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")`

`qwen2.5:0.5b` (~500 M params) is explicitly a **test placeholder** — too small for meaningful reasoning, emotional detection, or structured output. The model registry auto-adapts context budgets and prompt templates to whatever model is set, so switching is three env vars.

**Upgrade path when GPU arrives:**
```
OLLAMA_MODEL=qwen2.5:7b        # minimum for reasoning
# or
OLLAMA_MODEL=qwen2.5:14b       # recommended for emotional intelligence + SAGE critique
```

**Downstream effects of the placeholder model:**
- Thinking/tempo/silence views will show values but quality is unreliable
- Emotional intelligence detection is near-random at 0.5b
- SAGE self-critique produces low-quality feedback
- Multi-model consensus always routes to the same 0.5b endpoint (all "specialists" are the same model)

---

## Deferred / Not-Yet-Wired Features

| Feature | Status | Location | Unlock |
|---|---|---|---|
| Letta archival memory | Feature-flagged off (`FF_LETTA_TASKS=false`) | `agentic/app.py`, `letta-agent/` | Set `FF_LETTA_TASKS=true` after live Ollama verified |
| Graph memory fan-out | Feature-flagged off (`FF_GRAPH_INGEST=false` default) | `memu-core/app.py`, `memu-graph/` | Set `FF_GRAPH_INGEST=true`; requires `memu-graph` service running |
| Real STT (Whisper) | Code present, no audio hardware in CI | `audio-service/` | Provision audio hardware; set `STT_BACKEND=faster-whisper` |
| Real TTS (edge-tts) | Code present, no audio hardware in CI | `tts-service/` | Same as STT |
| Avatar / video | Service exists, GPU-gated | `avatar-service/` | GPU + camera hardware |
| Multi-model consensus | Architecture ready; all 3 specialists route to same endpoint | `common/llm.py`, `fusion-engine/` | Wire separate `OLLAMA_URL_*` per specialist when GPU arrives |
| agentic-introspect in sovereign profile | Present in `full.yml`, missing from `docker-compose.sovereign.yml` | `docker-compose.sovereign.yml` | Add `agentic-introspect` service to sovereign compose (known gap, DECISIONS.md) |

---

## Dashboard Backend Fallbacks ("unavailable" returns)

These are **intentional graceful-degradation paths** in `dashboard/app.py` — NOT bugs. Each one means the relevant backend service is not in the active compose profile. Listed here for completeness.

| Endpoint | Fallback value | Backend service |
|---|---|---|
| `/api/thinking` | `{"status":"unavailable","pathways":[]}` | `agentic-introspect` |
| `/api/tempo` | `{"status":"unavailable","tempo":"unknown"}` | `agentic-introspect` |
| `/api/boundaries` | `{"status":"unavailable","zones":[]}` | `agentic-introspect` |
| `/api/silence` | `{"status":"unavailable","silence_topics":[]}` | `agentic-introspect` |
| `/api/dream` | `{"status":"unavailable","message":"…"}` | `agentic-introspect` |
| `/api/logs` | `{"status":"unavailable","total_entries":0}` | `agentic-introspect` |
| `/api/security-audit` | `{"status":"unavailable","findings":[],…}` | `agentic-introspect` |
| `/api/goals` | `{"status":"unavailable","goals":[]}` | `agentic` |
| `/api/soul` (GET/POST) | `{"status":"unavailable","content":""}` | `agentic` |
| `/api/pii/scan` | `{"status":"unavailable","pii_found":{},"total_pii":0}` | `verifier` |
| `/api/memories/recent` | `{"records":[],"count":0}` | `memu-core` |

---

## One-liner JS `alert()` stub

**File:** `dashboard/static/app.html:2696`

A single `alert(msg)` call remains in the JS (likely a confirmation modal stub). Replace with an inline toast/modal when UX polish is needed.

---

## Coverage Gaps (not stubs, but tracked here for completeness)

- `common/auth.py` — 30% covered (lines 21-133 mostly untested)
- `common/llm.py` — 23% covered (stub + live inference paths)
- `common/policy.py` — 0% covered
- `common/prompt_templates.py` — 0% covered
- `common/rate_limit.py` — 0% covered
- `common/gpu_utils.py` — 0% covered (GPU-dependent)

These are measured by `make coverage`. Gate is set at 65% for the `common/` aggregate.

---

## Maintenance

When a stub is resolved, **delete its row** from this file and add a note to `DECISIONS.md`. Do not leave ~~strikethrough~~ rows — resolved items belong in CHANGELOG or DECISIONS, not here.
