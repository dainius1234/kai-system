# Stubs, Placeholders & Shells

**Last updated:** 2026-07-21 (post-D70)
**Purpose:** Single reference for every known stub, placeholder, shell UI, hardcoded value, or graceful-degradation path that must eventually be replaced with a real implementation. Things in this file are NOT bugs — they are intentional, tracked deferrals.

---

## How to read this file

| Category | Meaning |
|---|---|
| `UI_SHELL` | Dashboard view or button that renders but does nothing / shows "unavailable" when backend is absent |
| `LLM_STUB` | LLM response path that fires canned text when no live Ollama/Groq/OpenRouter backend is reachable |
| `CODE_STUB` | A working but deliberately simplified implementation (e.g. hash-based embeddings, no-op methods) |
| `HARDCODED_VALUE` | A credential, URL, or value that is a literal string where an env var substitution should be used |
| `PLACEHOLDER_DEFAULT` | An env var that ships blank or with a `change-me` / weak value and must be set before any real use |
| `DEFERRED_FEATURE` | A code path, endpoint, or feature that exists in the architecture but is not wired end-to-end yet |
| `TODO_COMMENT` | Code that raises `NotImplementedError` or carries an explicit "TODO: enable when…" comment |

**Unlock column** — what removes this entry from the list.
**Maintenance rule:** When a stub is resolved, delete its row and record the decision in `DECISIONS.md`. Do not leave ~~strikethrough~~ rows — resolved items belong in CHANGELOG or DECISIONS, not here.

---

## Code Stubs

| # | File | Line(s) | Description | Unlock |
|---|------|---------|-------------|--------|
| S1 | `common/llm.py` | 359–398 | `LLMRouter._stub_response()` / `stream()` return `"[specialist stub-{h}] Wire OLLAMA_URL…"` when no backend URL is configured. `stub_mode` property reports this state. | Set `OLLAMA_URL`, `GROQ_API_KEY`, or `OPENROUTER_API_KEY` |
| S2 | `fusion-engine/app.py` | 9–11, 127–132 | Module docstring says _"currently uses stub LLM backends (returns canned responses)"_. `_query_specialist()` falls to a deterministic hash string when `LLM_BACKENDS` is empty — i.e. no `LLM_DEEPSEEK_URL` / `LLM_KIMI_URL` / `LLM_DOLPHIN_URL` set. Conviction and consensus scores are meaningless in this mode. | Wire `LLM_DEEPSEEK_URL`, `LLM_KIMI_URL`, `LLM_DOLPHIN_URL` |
| S3 | `perception/audio/app.py` | 37, 167–168 | When `WHISPER_BACKEND=stub` (or `faster-whisper` not installed), `_transcribe()` returns the literal string `"[transcript: stub mode — set WHISPER_BACKEND=local for real STT]"`. This string propagates into memu-core as a memory record. | Install `faster-whisper`; set `WHISPER_BACKEND=local` |
| S4 | `screen-capture/app.py` | 78 | `_ocr_image_bytes()` returns `"[OCR unavailable — install pytesseract + Pillow + tesseract-ocr]"` when OCR dependencies are missing. Ends up stored in memory if service runs without them. | Install `pytesseract`, `Pillow`, system `tesseract-ocr` |
| S5 | `memu-core/app.py` | 56–84 | When `lakefs_client` pip package is absent, an in-memory `LakeFSClient` is substituted. `revert()` is a no-op (line 77: `# no-op stub`). All version history is lost on process restart. | Install `lakefs_client`; wire a real LakeFS server URL |
| S6 | `orchestrator/app.py` | entire file | Docstring: _"DEPRECATED. Exists only as a placeholder for a potential future 'final risk authority' layer."_ Exposes only `/health`, does nothing. Runs on port 8050. | Either implement the risk-authority gating layer or remove from compose files |
| S7 | `sandboxes/shell/app.py` | 27 | `POST /run` always returns `{"status": "blocked", "message": "sandbox execution disabled in stub"}`. No shell command is ever executed. | Implement subprocess sandbox with timeout, resource limits, and command allowlist |
| S8 | `kai-advisor/app.py` | 53 | When no knowledge chunk matches the query, advisor replies `"I heard: '{q}', but I'm just a simple KAI advisor stub."` — substring matching over markdown docs with no LLM inference. | Wire LLM-backed RAG using `MODEL` env var |

---

## TODO / NotImplementedError

| # | File | Line(s) | Description | Unlock |
|---|------|---------|-------------|--------|
| T1 | `docker-compose.sovereign.yml` | 439–446 | Two blocks commented out with `# TODO: enable GPU when core is stable.` — the `ollama` container and the `tts-service` build. Neither is wired in the sovereign profile. | Uncomment when GPU hardware validated |
| T2 | `verifier/app.py` | 16–17, 283–310 | Strategy 4 (`_keyword_plausibility`) docstring: _"placeholder for multi-LLM consensus when DeepSeek-V4 + Kimi-2.5 are wired"_. Body uses regex absolute-language detection only. | Wire DeepSeek-V4 + Kimi-2.5; implement cross-model agreement scoring |
| T3 | `scripts/hse_rams.py` | 11–13 | `main()` raises `NotImplementedError("RAMS generation not yet implemented. Needs python-docx, data/site_data.csv, and RAMS template.")`. Deliberately tracked in `quality_gate.py`'s `KNOWN_STUBS`. | Implement `python-docx` RAMS/method-statement generator |

---

## UI Shells

Views that render but show "unavailable" when their backend service isn't in the active compose profile. They are NOT broken — they degrade correctly. They become live with `make full-up`.

| # | View / Element | Backend | Proxy endpoints that degrade | Unlock |
|---|---|---|---|---|
| U1 | **Thinking** (`Ctrl+3`) | `agentic-introspect` | `/api/thinking`, `/api/tempo`, `/api/boundaries`, `/api/silence`, `/api/dream` | `make full-up` |
| U2 | **Goals** (`Ctrl+5`) | `agentic` | `/api/goals`, `/api/reminders`, `/api/tasks` | `make full-up` |
| U3 | **Logs** (`Ctrl+7`) | `agentic-introspect` | `/api/logs`, `/api/security-audit` | `make full-up` |
| U4 | **File attachment button** (`dashboard/static/app.html:1767`) | — | `handleFiles()` shows `"File attachments coming soon"` toast; no upload occurs | Implement `/api/upload` endpoint + file-to-chat routing |

**Functional since last update (removed from shell list):**
- Canvas → D3 v7 SVG (J1, 2026-07-21)
- Soul editor → SOUL.md + AGENTS.md live editor (J6, 2026-07-21)
- Diary → full Memory Diary with date groups + rich cards (J5, 2026-07-21)

---

## Hardcoded Values (literal strings that ignore env vars)

These are distinct from placeholder defaults — the env var substitution is **missing**, so setting the variable has no effect.

| # | File | Line(s) | What's hardcoded | Problem | Fix |
|---|------|---------|-----------------|---------|-----|

---

## Placeholder Defaults (env vars with weak/blank defaults)

| # | File | Variable | Current default | Required for | Production fix |
|---|------|---------|----------------|-------------|----------------|
| P1 | `.env.example:24` | `INTERSERVICE_HMAC_SECRET` | `change-me-in-production` | Inter-service HMAC auth (all services) | Generate with `openssl rand -hex 32` |
| P2 | `.env.example:15` | `TELEGRAM_BOT_TOKEN` | *(blank)* | Telegram bot integration | Get from @BotFather |
| P3 | `.env.example:16` | `TELEGRAM_CHAT_ID` | *(blank)* | Telegram alerts | Get from Telegram API |
| P4 | `.env.example:63` | `GROQ_API_KEY` | *(blank)* | Groq cloud LLM fallback | Groq console → API keys |
| P5 | `.env.example:68` | `OPENROUTER_API_KEY` | *(blank)* | OpenRouter cloud LLM fallback | openrouter.ai → API keys |
| P6 | `.env.example:19` | `TS_AUTHKEY` | `tskey-example` | Tailscale mesh networking | Tailscale admin → auth keys |
| P7 | `.env.example:33` | `BRIDGE_SHARED_SECRET` | *(blank)* | Vault/bridge inter-service auth | Generate random secret |
| P8 | `.env.example:34` | `VAULT_ROOT_TOKEN` | *(blank)* | HashiCorp Vault integration | Vault `init` output |
| P9 | `.env.example:37` | `GRAFANA_ADMIN_PASSWORD` | `admin` | Grafana dashboard | Set before first deploy |
| P10 | `.env.example:6` | `DB_PASSWORD` | `change-me` | PostgreSQL for all services | `openssl rand -hex 16` |
| P11 | `docker-compose.sovereign.yml:33,151,176,208,281` | `DB_PASSWORD` | `localdev` fallback | All sovereign DB connections | Set `DB_PASSWORD` in `.env`; consider `${DB_PASSWORD:?…}` mandatory form |
| P12 | `docker-compose.sovereign.yml:133` | `GRAFANA_ADMIN_PASSWORD` | `admin` fallback | Grafana admin login | Set `GRAFANA_ADMIN_PASSWORD`; consider mandatory form |

---

## LLM Model Placeholder

**File:** `common/llm.py:53` — `_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")`

`qwen2.5:0.5b` (~500 M params) is explicitly a **test placeholder** — too small for meaningful reasoning, emotional detection, or structured output. The model registry auto-adapts context budgets and prompt templates to whatever model is set.

**Downstream effects in stub-model mode:**
- Thinking/tempo views show values but quality is near-random
- Emotional intelligence detection unreliable
- SAGE self-critique produces low-quality feedback
- Multi-model consensus always routes to same 0.5b (all specialists are identical)

**Upgrade when GPU arrives:**
```
OLLAMA_MODEL=qwen2.5:7b    # minimum for real reasoning
OLLAMA_MODEL=qwen2.5:14b   # recommended for emotional intelligence + SAGE
```

---

## Fake Embeddings

**File:** `memu-core/app.py:954` — `_ALLOW_FAKE_EMBEDDINGS`, `_embed()` at ~967

When `sentence-transformers` isn't installed and `MEMU_ALLOW_FAKE_EMBEDDINGS=true`, memu-core uses a deterministic SHA-256 hash-based embedding. Semantic similarity becomes meaningless (all distances are uniform).

**Active in CI** (`MEMU_ALLOW_FAKE_EMBEDDINGS=true` in `core-tests.yml`).

**Unlock:** Install `sentence-transformers` (~1 GB model download, `all-MiniLM-L6-v2`). Set `MEMU_ALLOW_FAKE_EMBEDDINGS=false` (already the default — CI overrides it).

---

## Deferred / Not-Yet-Wired Features

| Feature | Gating mechanism | Location | Unlock |
|---|---|---|---|
| Letta archival memory | `FF_LETTA_TASKS=false` | `agentic/app.py`, `letta-agent/` | Set `FF_LETTA_TASKS=true` after live Ollama verified |
| Graph memory fan-out | `FF_GRAPH_INGEST=false` default | `memu-core/app.py`, `memu-graph/` | Set `FF_GRAPH_INGEST=true`; requires `memu-graph` service |
| Real STT (Whisper) | No audio hardware in CI | `perception/audio/` | Audio hardware + `WHISPER_BACKEND=local` |
| Real TTS (edge-tts) | No audio hardware in CI | `tts-service/` | Same as STT |
| Avatar / video | GPU + camera gated | `avatar-service/` | GPU + camera hardware provisioned |
| Multi-model consensus | All 3 specialists route to same endpoint | `common/llm.py`, `fusion-engine/` | Wire `LLM_DEEPSEEK_URL`, `LLM_KIMI_URL`, `LLM_DOLPHIN_URL` |
| `agentic-introspect` in sovereign profile | Missing from `docker-compose.sovereign.yml` | Known gap — documented in DECISIONS.md | Add service to sovereign compose |

---

## Dashboard Backend Fallbacks ("unavailable" returns)

These are **intentional graceful-degradation paths** in `dashboard/app.py` — NOT bugs. Listed here so engineers know which endpoint returning "unavailable" means which backend is missing.

| Endpoint | Fallback | Backend service |
|---|---|---|
| `/api/thinking` | `{"status":"unavailable","pathways":[]}` | `agentic-introspect` |
| `/api/tempo` | `{"status":"unavailable","tempo":"unknown"}` | `agentic-introspect` |
| `/api/boundaries` | `{"status":"unavailable","zones":[]}` | `agentic-introspect` |
| `/api/silence` | `{"status":"unavailable","silence_topics":[]}` | `agentic-introspect` |
| `/api/dream` | `{"status":"unavailable","message":"…"}` | `agentic-introspect` |
| `/api/logs` | `{"status":"unavailable","total_entries":0}` | `agentic-introspect` |
| `/api/security-audit` | `{"status":"unavailable","findings":[],"risk_score":-1}` | `agentic-introspect` |
| `/api/goals` | `{"status":"unavailable","goals":[]}` | `agentic` |
| `/api/soul` (GET/POST) | `{"status":"unavailable","content":""}` | `agentic` |
| `/api/pii/scan` | `{"status":"unavailable","pii_found":{},"total_pii":0}` | `verifier` |
| `/api/memories/recent` | `{"records":[],"count":0}` | `memu-core` |

---

## Minor Cosmetic Stubs

| Item | Location | Description |
|---|---|---|
| `alert()` call | `dashboard/static/app.html:2696` | Single `alert(msg)` — likely a confirmation modal stub; replace with an inline toast/modal |
