# Kai Code Audit — Telegram Bot Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-TG-001 | CRITICAL | Empty allowlist fails open and permits every Telegram chat |
| KAI-TG-002 | CRITICAL | Unauthenticated `/alert` can send arbitrary messages through the bot |
| KAI-TG-003 | HIGH | Bot token is embedded in request URLs and can leak through exception logging |
| KAI-TG-004 | HIGH | Telegram messages directly enter agentic chat without an independent trust boundary |
| KAI-TG-005 | HIGH | Voice messages are fully downloaded before the size limit is checked |
| KAI-TG-006 | HIGH | Poll loop handles updates serially and one slow request blocks all chats |
| KAI-TG-007 | HIGH | Caller-selected alert chat IDs are not checked against the allowlist |
| KAI-TG-008 | MEDIUM | Health and metrics expose chat identifiers, modes and activity without authentication |
| KAI-TG-009 | MEDIUM | Error responses expose internal exception details |
| KAI-TG-010 | MEDIUM | Per-chat modes and counters are process-local and lost on restart |
| KAI-TG-011 | MEDIUM | Telegram API client is recreated for every operation |
| KAI-TG-012 | MEDIUM | Polling offset is non-durable and updates can be replayed after restart |
| KAI-TG-013 | MEDIUM | Configuration values and default mode are not validated |
| KAI-TG-014 | MEDIUM | Markdown fallback can produce duplicate delivery attempts |

---

## Telegram bot: `telegram-bot/app.py`

### KAI-TG-001 — CRITICAL — Empty allowlist fails open
**Issue:** `_is_allowed` returns `True` whenever `ALLOWED_CHAT_IDS` is empty. The documented configuration explicitly says empty means allow all.  
**Risk:** A deployment that omits or misconfigures the allowlist permits any Telegram user or group that discovers/adds the bot to send text and voice into Kai, consume inference resources and receive responses.  
**Recommendation:** Fail closed when no approved chat IDs are configured and require explicit verified enrolment.  
**Status:** OPEN — immediate remediation required

### KAI-TG-002 — CRITICAL — Unauthenticated proactive messaging
**Issue:** `POST /alert` requires no authentication or authorisation. It sends caller-controlled text through the Telegram bot and accepts an optional caller-selected `chat_id`.  
**Risk:** Any network-reachable caller can impersonate Kai, send phishing or false operational messages to Telegram users/channels accessible to the bot and create notification spam.  
**Recommendation:** Require authenticated service identity, destination scopes and provenance-labelled templates.  
**Status:** OPEN — immediate remediation required

### KAI-TG-003 — HIGH — Bot token can leak through logged exceptions
**Issue:** `TG_API` and `TG_FILE` embed `TELEGRAM_BOT_TOKEN` in URLs. Exceptions from Telegram HTTP calls can contain request URLs. Several handlers log raw exception text, including polling, chat, STT/TTS and alert failures.  
**Risk:** The bot token may be written to logs, allowing anyone with log access to control the bot.  
**Recommendation:** Redact credentials from all URLs and exceptions, and use a transport wrapper that never exposes secret-bearing request targets.  
**Status:** OPEN

### KAI-TG-004 — HIGH — External chat is treated as agentic input
**Issue:** Allowed Telegram text is sanitised only as a string and sent directly to `agentic /chat` with a stable session ID. Voice transcripts follow the same path. There is no channel-specific untrusted-input policy, approval boundary or provenance constraint.  
**Risk:** Telegram content can drive the full agentic pipeline, memory and tools with the same apparent operator session continuity, amplifying prompt-injection and account-compromise impact.  
**Recommendation:** Apply channel-specific capability restrictions, provenance and explicit confirmation for consequential actions.  
**Status:** OPEN

### KAI-TG-005 — HIGH — Voice limit is checked after full download
**Issue:** `_download_file` accesses `r.content`, materialising the complete Telegram file before comparing its length with `MAX_VOICE_BYTES`.  
**Risk:** Oversized or concurrent files can exhaust memory despite the stated 10 MB limit.  
**Recommendation:** Stream with a strict byte counter and enforce Telegram metadata limits before download.  
**Status:** OPEN

### KAI-TG-006 — HIGH — One update blocks the entire bot
**Issue:** The long-poll loop processes each update with `await _handle(upd)` serially. A chat response can wait up to 120 seconds, plus STT, TTS and Telegram operations.  
**Risk:** One slow or adversarial message delays every subsequent user and prevents timely processing of other updates, creating a trivial denial of service.  
**Recommendation:** Dispatch updates into a bounded per-chat worker queue with global concurrency and time limits.  
**Status:** OPEN

### KAI-TG-007 — HIGH — Alert destination bypasses allowlist
**Issue:** When `/alert` supplies `chat_id`, the endpoint never calls `_is_allowed`.  
**Risk:** Internal callers or attackers reaching the endpoint can target any chat the bot is capable of messaging, even if it is not approved for interactive access.  
**Recommendation:** Enforce destination allowlists and service-specific routing policies on every outbound message.  
**Status:** OPEN

### KAI-TG-008 — MEDIUM — Operational and chat metadata is public
**Issue:** `/health` and `/metrics` expose bot state, message/voice counts, last activity time, device type and the complete chat-ID-to-mode mapping without authentication.  
**Risk:** Callers can enumerate Telegram chat identifiers and infer operator activity and usage patterns.  
**Recommendation:** Require scoped operational access and remove raw chat IDs from metrics.  
**Status:** OPEN

### KAI-TG-009 — MEDIUM — Internal errors are returned/logged
**Issue:** `/alert` returns truncated raw exception text, while chat failures are converted into user-visible strings containing exception details.  
**Risk:** Network topology, service names, response details and potentially secret-bearing URLs can be disclosed.  
**Recommendation:** Return stable error codes and protected trace identifiers only.  
**Status:** OPEN

### KAI-TG-010 — MEDIUM — Session state is volatile and worker-local
**Issue:** Chat modes, counters, last timestamp and running state are module-level variables.  
**Risk:** Restart loses mode preferences and metrics; multiple workers expose inconsistent state.  
**Recommendation:** Use a shared durable session store or enforce one bot worker explicitly.  
**Status:** OPEN

### KAI-TG-011 — MEDIUM — HTTP clients are repeatedly recreated
**Issue:** `_tg`, downloads, chat, STT and TTS each create new `httpx.AsyncClient` instances. Polling also creates a new client each cycle.  
**Risk:** Connection reuse is lost, TLS and socket churn increase, and resource pressure grows under message volume.  
**Recommendation:** Reuse lifecycle-managed clients with bounded pools.  
**Status:** OPEN

### KAI-TG-012 — MEDIUM — Polling offset is non-durable
**Issue:** `offset` starts at zero on every process start and is stored only in the polling coroutine.  
**Risk:** Depending on Telegram retention and acknowledgement timing, updates may be replayed after restart, causing duplicate responses, memory entries or tool actions.  
**Recommendation:** Persist the last committed update ID transactionally after successful handling.  
**Status:** OPEN

### KAI-TG-013 — MEDIUM — Configuration is weakly validated
**Issue:** Default mode, service URLs, allowlist syntax and booleans are accepted directly. An invalid allowlist can silently produce an empty parsed set while the raw string remains non-empty, denying all chats without readiness failure.  
**Risk:** Misconfiguration produces unsafe fail-open behaviour, unexplained denial of service or routing to unintended services.  
**Recommendation:** Validate typed configuration and fail startup/readiness on invalid security settings.  
**Status:** OPEN

### KAI-TG-014 — MEDIUM — Markdown fallback is not idempotent
**Issue:** If the first `sendMessage` attempt raises after Telegram accepted but before the response was received, `_send_text` retries without Markdown.  
**Risk:** Users can receive duplicate messages and downstream alert counts become unreliable.  
**Recommendation:** Use idempotency-aware delivery tracking or avoid automatic retry after ambiguous outcomes.  
**Status:** OPEN

---

## Batch totals

- Findings: **14**
- Critical: **2**
- High: **5**
- Medium: **7**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **366**
- Critical: **40**
- High: **145**
- Medium: **178**
- Low: **3**

## Files materially reviewed in this batch

`telegram-bot/app.py`.
