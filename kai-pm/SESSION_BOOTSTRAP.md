# Session Bootstrap

**Read this first. In 60 seconds you will know where truth lives.**

> 📌 **If you are a new chat / new agent / future-Dainius reading this:**
> Do not re-learn the repo from scratch.
> Open the four files below in order and trust the highest-canonical one when docs disagree.

---

## 1) The 4-file bootstrap order

1. [`CLEANUP_TODO.md`](CLEANUP_TODO.md) — live execution state, checkboxes, active work.
2. [`STATUS.md`](STATUS.md) — short human summary of phase, focus, and next 3 actions.
3. [`NAVIGATION.md`](NAVIGATION.md) — routing map for the rest of the PM docs.
4. [`../SESSION_BACKLOG.md`](../SESSION_BACKLOG.md) — running log of what actually happened.

If there is drift:

- `CLEANUP_TODO.md` wins for live work state.
- `STATUS.md` wins for the short summary.
- `NAVIGATION.md` wins for where to look next.
- `README.md` is public-facing and lower priority than the PM docs above.

---

## 2) Project one-liner

Kai is a self-sovereign, local-first personal AI system built as cooperating services, designed to grow memory, reasoning, and operational reliability without external platform lock-in.

---

## 3) Current reality snapshot

- **Current phase:** Phase 0 — Pre-GPU Hardening
- **Current focus:** Cleanup Sprint Week 2.1 — `agentic/app.py` route split + PM truth reconciliation
- **What changed recently:** the `langgraph/` → `agentic/` rename landed, the `prompts.py` split landed, and the active cleanup PR now trims leftover route-split coupling while PM docs are reconciled to merged/open PR reality.

For the authoritative live details, do **not** trust this file — open:

- [`STATUS.md`](STATUS.md) for the current summary
- [`CLEANUP_TODO.md`](CLEANUP_TODO.md) for exact checklist state
- open PRs: https://github.com/dainius1234/kai-system/pulls

---

## 4) Canonical doc roles

| Need | Canonical file |
|---|---|
| What is in progress right now? | [`CLEANUP_TODO.md`](CLEANUP_TODO.md) |
| What should I do next? | [`STATUS.md`](STATUS.md) |
| Where is the roadmap / risk / decision doc? | [`NAVIGATION.md`](NAVIGATION.md) |
| Why was a decision made? | [`DECISIONS.md`](DECISIONS.md) |
| What changed across sessions? | [`../SESSION_BACKLOG.md`](../SESSION_BACKLOG.md) |
| What changed in the repo publicly? | [`../CHANGELOG.md`](../CHANGELOG.md) |

---

## 5) Next-move ritual

When starting fresh:

1. Open [`CLEANUP_TODO.md`](CLEANUP_TODO.md).
2. Open [`STATUS.md`](STATUS.md).
3. Check open PRs.
4. Open [`NAVIGATION.md`](NAVIGATION.md) only if you need deeper context.
5. Pick exactly one live item and keep the PR scoped to that one concern.

---

## 6) PM operating rules

- Document decisions in the repo, not only in chat.
- One PR = one concern.
- Do not let `STATUS.md`, `CLEANUP_TODO.md`, and shipped code drift apart.
- Treat `DECISIONS.md` as append-only.
- Before claiming a task is active, verify against `README.md`, `CHANGELOG.md`, and open PRs.

---

## 7) How to recover after context loss

1. Open this file.
2. Open [`CLEANUP_TODO.md`](CLEANUP_TODO.md).
3. Open [`STATUS.md`](STATUS.md).
4. Check open PRs.
5. Ask: *"Resume brother — read SESSION_BOOTSTRAP and tell me the next move."*
