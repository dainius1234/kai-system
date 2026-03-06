# Kai System — Gaps & Hardening Checklist

> Consolidated gap analysis. Items marked ✅ are already implemented.
> Remaining items are prioritised by impact.
>
> Updated: 5 March 2026

---

## 1. Tool Gate

| Item | Status | Notes |
|---|---|---|
| TPM signature verification (`tpm2_sign` → `tpm2_verify`) | ⏳ Hardware-blocked | Requires `/dev/tpm0` on RTX 5080 laptop. HMAC covers this in codespace. |
| Rate limit (>3 calls/60s → 429) | ✅ Done | `common/rate_limit.py` wired into tool-gate. Configurable window + max calls. |
| Co-sign veto (dashboard rejects → force reject with reason) | ✅ Done | tool-gate co-sign flow with approval/reject. Ledger records reason. |
| HMAC strict key ID mode | ✅ Done | `INTERSERVICE_HMAC_STRICT_KEY_ID` env var. Tested in `scripts/test_auth_hmac.py`. |

## 2. Memory Core (memu-core)

| Item | Status | Notes |
|---|---|---|
| Redis cache for `/route` results (10 min TTL) | 🔲 Open | Low priority — current keyword routing is sub-1ms. Worth adding when LLM-based routing is enabled. |
| Vector cleanup job (delete >90 day embeddings) | ✅ Done | `POST /memory/cleanup?max_age_days=90` in memu-core. Preserves pinned memories. |
| User-ID filter (session_id == keeper) | ✅ Done | Single-operator system. All memories belong to keeper. No multi-tenant risk. |
| Cross-session context | ✅ Done | memu-core pgvector retrieves across sessions by default. |

## 3. Executor

| Item | Status | Notes |
|---|---|---|
| Sandbox timeout (30s max, kill + log) | ✅ Done | `EXECUTION_TIMEOUT` env var in executor. Default 30s. |
| Output sanitization (strip dangerous commands) | ✅ Done | `sanitize_string()` in langgraph/app.py strips injection patterns. |
| Result hashing (SHA256 before ledger store) | ✅ Done | Ledger entries are hash-chained. Episode spool uses SHA256 checksums. |

## 4. Dashboard

| Item | Status | Notes |
|---|---|---|
| Real-time status (Redis pubsub) | ✅ Done | SSE endpoint `GET /api/events` with Redis pubsub subscription. 4 channels: `kai:health`, `kai:episode`, `kai:breaker`, `kai:memory`. 15s keepalive heartbeat. |
| Ledger stats (last write, entry count) | ✅ Done | `GET /api/ledger-stats` in dashboard, proxies to ledger-worker `/stats`. |
| Memory heatmap (top 5 tags, word cloud) | 🔲 Open | Low priority. Nice-to-have visualization. |
| Thinking Pathways (conviction, tempo, boundary, silence) | ✅ Done | P8 complete. `thinking.html` with 6 visualization cards. |

## 5. System-Wide

| Item | Status | Notes |
|---|---|---|
| `/health` on every service | ✅ Done | All 25 services expose `/health` → `{"status": "ok"}`. |
| Auto-restart policy (`unless-stopped`) | ✅ Done | Compose files use `unless-stopped` for persistent services. |
| GPU stub comments | ✅ Done | `common/llm.py` handles GPU/CPU fallback. Ollama abstracts device selection. |
| JSON structured logging | ✅ Done | `setup_json_logger()` in `common/runtime.py` outputs JSON to both file and stdout. All services use it. |
| FAISS-GPU for memory retrieval | ⏳ Hardware-blocked | pgvector handles this for now. FAISS-GPU is only worth it at >1M vectors. |
| One-line rollback in executor | 🔲 Open | Low priority. Executor is stateless — rollback is "restart container". |
| Unit tests in /scripts | ✅ Done | 47 test-core targets, 270 individual tests. Full coverage. |
| Auto-restart script (Ara calls herself) | ✅ Done | Supervisor watchdog + heartbeat auto-sleep already cover this. |

---

## Priority Summary

### Do Now (Codespace)
1. ~~**JSON structured logging**~~ ✅ Done
2. ~~**Ledger stats endpoint**~~ ✅ Done
3. ~~**Vector cleanup job**~~ ✅ Done

### Do on Hardware Arrival
1. **TPM signature verification** — wire `tpm2_sign`/`tpm2_verify` in tool-gate
2. **FAISS-GPU** — if pgvector becomes slow at scale
3. **HP1–HP6** — see Hardware Performance Track in `unfair_advantages.md`

### Park (Nice-to-Have)
- ~~Redis pubsub for real-time dashboard~~ ✅ Done (SSE /api/events)
- Memory heatmap visualization
- Redis cache for routing results
