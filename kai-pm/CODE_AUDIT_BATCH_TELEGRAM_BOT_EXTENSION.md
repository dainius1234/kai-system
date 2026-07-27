# Kai Code Audit — Telegram Bot Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_TELEGRAM_BOT.md`. The existing 14 findings are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-TGX-001 | CRITICAL | Authorisation uses chat ID rather than sender identity, so every member of an allowed group inherits operator access |
| KAI-TGX-002 | CRITICAL | Telegram callers reach Agentic’s global `keeper` memory, identity, conscience and private context |
| KAI-TGX-003 | HIGH | The Telegram bot token is stored in process environment rather than a mounted secret |
| KAI-TGX-004 | HIGH | `_bot_running` becomes true before token validity or successful polling is established |
| KAI-TGX-005 | HIGH | Invalid or revoked-token polling can fail forever while health reports `status: ok` |
| KAI-TGX-006 | HIGH | Health has no last-successful-poll age, bot identity or committed-update readiness check |
| KAI-TGX-007 | HIGH | Docker health treats the HTTP-200 `no_token` state as healthy |
| KAI-TGX-008 | HIGH | Alert text is unbounded at request intake and is sent as Telegram Markdown |
| KAI-TGX-009 | HIGH | Model output and transcripts are sent as unescaped Markdown and can create deceptive links or mentions |
| KAI-TGX-010 | HIGH | No rate limit, caller quota or message-delivery admission policy protects `/alert` |
| KAI-TGX-011 | HIGH | Poll offset advances before update handling succeeds, permanently dropping failed updates |
| KAI-TGX-012 | HIGH | Multiple workers or replicas can poll the same bot concurrently without leader election |
| KAI-TGX-013 | HIGH | Agentic streamed output is accumulated without a token or response-size limit |
| KAI-TGX-014 | HIGH | Missing `[DONE]` or endless token events can occupy the handler until the broad timeout |
| KAI-TGX-015 | HIGH | Generic sanitisation silently truncates and changes text before Agentic receives it |
| KAI-TGX-016 | HIGH | Voice transcripts bypass the text-path sanitisation and provenance boundary |
| KAI-TGX-017 | HIGH | Telegram update and `getFile` response bodies/JSON are fully materialised without strict schemas or size limits |
| KAI-TGX-018 | HIGH | Every downloaded voice file is labelled `audio/ogg` regardless of actual content |
| KAI-TGX-019 | HIGH | Audio-service transcript output is unbounded and weakly schema-validated |
| KAI-TGX-020 | HIGH | TTS responses are fully materialised without a maximum byte or duration limit |
| KAI-TGX-021 | HIGH | Voice replies are uploaded without a maximum audio size |
| KAI-TGX-022 | HIGH | Agentic responses are sent to TTS without PII or secret redaction |
| KAI-TGX-023 | HIGH | Private transcripts and Agentic responses are transmitted to Telegram without data minimisation |
| KAI-TGX-024 | HIGH | Voice transcription, TTS and Telegram retention have no explicit consent or deletion controls |
| KAI-TGX-025 | HIGH | `/mode work` changes only Agentic prompt mode and does not change Tool Gate enforcement |
| KAI-TGX-026 | HIGH | Session IDs are predictable from chat IDs and group members share one Agentic session |
| KAI-TGX-027 | HIGH | Agentic, audio and TTS calls use no service authentication or response identity verification |
| KAI-TGX-028 | MEDIUM | Internal Agentic, audio and TTS traffic uses plaintext HTTP |
| KAI-TGX-029 | MEDIUM | `/voice` is advertised as a toggle but is not implemented |
| KAI-TGX-030 | MEDIUM | Telegram rate-limit responses and `retry_after` instructions are not handled |
| KAI-TGX-031 | MEDIUM | Poll failures use a fixed five-second retry without jitter or failure-class backoff |
| KAI-TGX-032 | MEDIUM | `/status` exposes global counters and device state to every authorised chat member |
| KAI-TGX-033 | MEDIUM | Alert fallback selects the first configured chat and ordinary active chats are not retained as targets |
| KAI-TGX-034 | MEDIUM | Message and voice counters increment before processing and delivery succeeds |
| KAI-TGX-035 | MEDIUM | Typing-action failures are silently discarded |
| KAI-TGX-036 | MEDIUM | No immutable audit links Telegram sender/update, Agentic session, response and outbound message IDs |
| KAI-TGX-037 | MEDIUM | Outbound Telegram calls have no idempotency key or committed-send reconciliation |

---

## Critical findings

### KAI-TGX-001 — CRITICAL — Group allowlist grants every member operator access
**Issue:** `_is_allowed()` checks only `message.chat.id`. In a Telegram group, every sender shares the group chat ID; `message.from.id` is never authorised.  
**Risk:** Any member of an allowed group can issue Kai commands, submit prompts and receive private responses.  
**Recommendation:** require an approved sender identity and optionally an approved chat, with explicit group policy.  
**Status:** OPEN — immediate remediation required

### KAI-TGX-002 — CRITICAL — Telegram reaches global keeper context
**Issue:** Telegram messages are sent to Agentic `/chat`. Agentic loads and mutates global `keeper` memories, financial, identity, conscience and sensory context independently of the Telegram sender.  
**Risk:** An external Telegram user can extract or poison the operator’s private long-term state.  
**Recommendation:** create a restricted principal-scoped Telegram identity with no implicit keeper/global memory access.  
**Status:** OPEN — immediate remediation required

---

## High-severity findings

### KAI-TGX-003 — HIGH — Token in process environment
Compose supplies `TELEGRAM_BOT_TOKEN` as an environment variable rather than a mounted secret with ownership/rotation controls.

### KAI-TGX-004 — HIGH — Running precedes readiness
The poller sets `_bot_running=True` before its first successful Telegram request.

### KAI-TGX-005 — HIGH — Invalid token stays green
Persistent 401/404 polling errors do not clear `_bot_running`.

### KAI-TGX-006 — HIGH — No poll freshness evidence
Health does not record the last successful Telegram request, bot username, update ID or poll error.

### KAI-TGX-007 — HIGH — Missing-token container is healthy
Docker checks only whether `/health` returns HTTP success; `status:no_token` is therefore accepted.

### KAI-TGX-008 — HIGH — Unbounded Markdown alert intake
`AlertPayload.text` has no request limit; truncation occurs only when building the outbound Telegram payload.

### KAI-TGX-009 — HIGH — Markdown injection/phishing
Alerts, model output and transcripts are sent with `parse_mode: Markdown` without escaping links, mentions or formatting controls.

### KAI-TGX-010 — HIGH — No alert admission control
Anonymous callers can consume Telegram quotas and flood trusted operator channels.

### KAI-TGX-011 — HIGH — Update acknowledged before processing
`offset` is incremented before `_handle()` completes. Handler failure means later polling confirms and loses the update.

### KAI-TGX-012 — HIGH — Concurrent pollers
No distributed lease enforces one `getUpdates` owner per bot token.

### KAI-TGX-013 — HIGH — Unbounded Agentic accumulation
All stream tokens are appended to a list and joined before Telegram’s 4,096-character send truncation.

### KAI-TGX-014 — HIGH — Weak stream completion
Only `[DONE]` terminates normally; a stream can continue until the broad 120-second timeout.

### KAI-TGX-015 — HIGH — Destructive text transformation
`sanitize_string()` strips characters and truncates, changing facts/commands and mapping distinct inputs to the same prompt.

### KAI-TGX-016 — HIGH — Voice bypasses text treatment
The transcript goes directly to Agentic and is echoed to Telegram as Markdown.

### KAI-TGX-017 — HIGH — Weak Telegram response boundaries
Complete `getUpdates`/`getFile` responses are trusted through unbounded nested dictionaries/lists.

### KAI-TGX-018 — HIGH — Incorrect media declaration
Every file is forwarded as OGG regardless of actual bytes or Telegram metadata.

### KAI-TGX-019 — HIGH — Unbounded transcript
Any JSON `transcript` value is accepted without string/length validation.

### KAI-TGX-020 — HIGH — Unbounded TTS body
Complete TTS response bytes are buffered; only a minimum length is checked.

### KAI-TGX-021 — HIGH — Unbounded outbound audio
`_send_voice()` uploads the complete returned byte array without a maximum.

### KAI-TGX-022 — HIGH — Sensitive TTS egress
Up to 2,000 Agentic response characters are sent to TTS without redaction or purpose constraints.

### KAI-TGX-023 — HIGH — Sensitive Telegram egress
Private memory, finance, email, identity or operational content can be transmitted externally as text/voice.

### KAI-TGX-024 — HIGH — Missing consent/retention model
No consent revision, retention period, deletion request or processing-purpose record governs voice/transcript/TTS/Telegram data.

### KAI-TGX-025 — HIGH — False WORK assurance
The command changes the `mode` field forwarded to Agentic only; Tool Gate’s effective mode remains unchanged.

### KAI-TGX-026 — HIGH — Predictable/shared session identity
`session_id = tg-{chat_id}` is not a secret capability; all group senders share the same working history.

### KAI-TGX-027 — HIGH — Unauthenticated downstream calls
Internal service requests carry no HMAC, mTLS, caller delegation or response attestation.

---

## Medium-severity findings

### KAI-TGX-028 — MEDIUM — Plaintext internal traffic
Default Agentic/audio/TTS URLs use HTTP.

### KAI-TGX-029 — MEDIUM — Missing voice-toggle command
Help advertises `/voice`, but unknown commands are ignored and TTS remains globally configured.

### KAI-TGX-030 — MEDIUM — No Telegram quota handling
429 response metadata is not parsed; no queue/cooldown honours `retry_after`.

### KAI-TGX-031 — MEDIUM — Fixed retry loop
Poll failures sleep five seconds with no jitter or permanent/transient classification.

### KAI-TGX-032 — MEDIUM — In-chat operational disclosure
Any allowed group participant can request global message counts, TTS and device state.

### KAI-TGX-033 — MEDIUM — Incorrect proactive target tracking
Fallback chooses the first configured numeric ID. `_chat_modes` records only chats that issued `/mode`, not the actual last active chat.

### KAI-TGX-034 — MEDIUM — Premature counters
Counts increment before download, transcription, Agentic or Telegram delivery completes.

### KAI-TGX-035 — MEDIUM — Silent typing failure
All `sendChatAction` errors are suppressed.

### KAI-TGX-036 — MEDIUM — Missing causal audit
There is no immutable chain from sender/chat/update/file to Agentic request/session and Telegram response IDs.

### KAI-TGX-037 — MEDIUM — Missing send idempotency
Committed-but-timed-out outbound sends cannot be reconciled and may be repeated.

---

## Batch totals

- Findings: **37**
- Critical: **2**
- High: **25**
- Medium: **10**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,130**
- Critical: **189**
- High: **1,058**
- Medium: **880**
- Low: **3**

## Files materially reviewed

`telegram-bot/app.py`, the existing Telegram audit, Telegram deployment and integrations with Agentic, Audio, TTS, Supervisor and Telegram Bot API.
