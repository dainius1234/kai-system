# Repo Navigation — "You Are Here"

> **If you are lost, start here.** This file tells you which doc to open for which question.
> All other PM docs are *referenced from here*, not the other way around.

**Last updated:** 2026-05-10

---

## "I want to know X" → "Open Y"

| If your question is… | Open this file |
|---|---|
| **What should I work on right now?** | [`PHASE_0_5_BACKLOG.md`](PHASE_0_5_BACKLOG.md) ← **start here 99% of the time** |
| What's the big-picture roadmap? | [`STRATEGIC_PLAN.md`](STRATEGIC_PLAN.md) |
| What's the current phase order? | [`SEQUENCE.md`](SEQUENCE.md) |
| What metrics are real today? | [`METRICS.md`](METRICS.md) + [`REALITY_CHECK_2026-05-10.md`](REALITY_CHECK_2026-05-10.md) |
| What's risky right now? | [`RISKS.md`](RISKS.md) |
| Should we adopt tech X? | [`TECH_WATCH.md`](TECH_WATCH.md) |
| Why did we decide X? | [`DECISIONS.md`](DECISIONS.md) |
| I'm a new session/agent — bootstrap me | [`SESSION_BOOTSTRAP.md`](SESSION_BOOTSTRAP.md) |
| What does the README *actually* mean? | [`../README.md`](../README.md) + the "Honest Limitations" section |
| What's the full historical backlog? | [`../docs/PROJECT_BACKLOG.md`](../docs/PROJECT_BACKLOG.md) |
| What changed when? | [`../CHANGELOG.md`](../CHANGELOG.md) |

---

## Doc hierarchy (which file overrides which)

When two docs disagree, the higher one wins:

```
1. REALITY_CHECK_<date>.md   ← most recent reality always wins
2. PHASE_0_5_BACKLOG.md      ← what to do now
3. STRATEGIC_PLAN.md         ← canonical roadmap
4. SEQUENCE.md / RISKS.md / METRICS.md / TECH_WATCH.md
5. README.md                 ← public-facing, marketing-adjacent
6. docs/PROJECT_BACKLOG.md   ← historical
```

---

## Three things to know about this repo

1. **Phase 0 (Hardening) is the current phase.** Everything that looks "intelligent" in the
   README is real *plumbing* but is bottlenecked on the default LLM (`qwen2:0.5b`, ~400M params)
   being too small to reason. Real intelligence unlocks at **Phase 1 (Local LLM Integration)**,
   which requires the **RTX 5080 GPU** to be physically connected.

2. **You can do months of real engineering on CPU.** See [`PHASE_0_5_BACKLOG.md`](PHASE_0_5_BACKLOG.md).
   GPU unblocks *measurement of intelligence*, not *building of the system*.

3. **The README is honest about what's real vs aspirational** in its "Honest Limitations" and
   "What's Real vs What Needs Hardware" sections. Capability tables describe code that *exists
   and runs*, but the *quality* of the output is model-bound.

---

## When in doubt

- Run `make sync-docs` to refresh metrics.
- Run `make check-docs` to verify nothing has drifted.
- Open `PHASE_0_5_BACKLOG.md` and pick item #1.
