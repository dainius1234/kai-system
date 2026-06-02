# Repo Navigation — "You Are Here"

> **If you are lost, start here.** This file tells you which PM doc to open next.
> It is the routing layer, not the status tracker.

**Last updated:** 2026-06-02

---

## Read order for a fresh session

1. [`SESSION_BOOTSTRAP.md`](SESSION_BOOTSTRAP.md) — 60-second reset.
2. [`CLEANUP_TODO.md`](CLEANUP_TODO.md) — live work tracker.
3. [`STATUS.md`](STATUS.md) — short summary and next 3 actions.
4. Use the table below only for deeper context.

---

## "I want to know X" → "Open Y"

| If your question is… | Open this file |
|---|---|
| **What should I work on right now?** | [`CLEANUP_TODO.md`](CLEANUP_TODO.md) |
| What's the short current summary? | [`STATUS.md`](STATUS.md) |
| What's the big-picture roadmap? | [`STRATEGIC_PLAN.md`](STRATEGIC_PLAN.md) |
| What's the current phase order? | [`SEQUENCE.md`](SEQUENCE.md) |
| What metrics are real today? | [`METRICS.md`](METRICS.md) + [`REALITY_CHECK_2026-05-10.md`](REALITY_CHECK_2026-05-10.md) |
| What's risky right now? | [`RISKS.md`](RISKS.md) |
| Should we adopt tech X? | [`TECH_WATCH.md`](TECH_WATCH.md) |
| Why did we decide X? | [`DECISIONS.md`](DECISIONS.md) |
| I'm a new session/agent — bootstrap me | [`SESSION_BOOTSTRAP.md`](SESSION_BOOTSTRAP.md) |
| What does the README *actually* mean? | [`../README.md`](../README.md) + the "Honest Limitations" section |
| What's the running cross-session log? | [`../SESSION_BACKLOG.md`](../SESSION_BACKLOG.md) |
| What's the full historical backlog? | [`../docs/PROJECT_BACKLOG.md`](../docs/PROJECT_BACKLOG.md) |
| What changed when? | [`../CHANGELOG.md`](../CHANGELOG.md) |

---

## Canonical roles

When two docs overlap, use this split:

- **Execution state:** [`CLEANUP_TODO.md`](CLEANUP_TODO.md)
- **Short summary / next actions:** [`STATUS.md`](STATUS.md)
- **Routing / where to read next:** [`NAVIGATION.md`](NAVIGATION.md)
- **Roadmap:** [`STRATEGIC_PLAN.md`](STRATEGIC_PLAN.md)
- **Historical log:** [`../SESSION_BACKLOG.md`](../SESSION_BACKLOG.md)

---

## Doc hierarchy (which file overrides which)

When two docs disagree, the higher one wins:

```
1. CLEANUP_TODO.md            ← live execution state
2. STATUS.md                  ← current summary / next actions
3. NAVIGATION.md              ← routing layer
4. REALITY_CHECK_<date>.md    ← factual correction layer
5. STRATEGIC_PLAN.md          ← canonical roadmap
6. SEQUENCE.md / RISKS.md / METRICS.md / TECH_WATCH.md
7. README.md                  ← public-facing, marketing-adjacent
8. docs/PROJECT_BACKLOG.md    ← historical
```

---

## Three things to know about this repo

1. **Phase 0 (Hardening) is the current phase.** Everything that looks "intelligent" in the README is real *plumbing* but is bottlenecked on the default LLM (`qwen2:0.5b`, ~400M params) being too small to reason.
2. **You can do months of real engineering on CPU.** The GPU unblocks *measurement of intelligence*, not *building of the system*.
3. **The README is honest about what's real vs aspirational.** Capability tables describe code that exists and runs, but output quality is model-bound.

---

## When in doubt

- Open [`CLEANUP_TODO.md`](CLEANUP_TODO.md).
- Open [`STATUS.md`](STATUS.md).
- Run `make sync-docs` if generated status docs drift.
- Run `make check-docs` before landing doc changes.
