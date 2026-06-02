# Phase 0.5 Backlog — CPU-Safe Work Before GPU Lands

> **Open this file when sitting down to work on Kai.** Pick the top unblocked item.
> All items here are CPU-only — no GPU required. They de-risk Phase 1 and ship value now.

**Status:** Active — 4-week PM-driven sprint
**Last updated:** 2026-05-10
**Owner:** @dainius1234 (PM-assisted, max automation)

---

## The 4-week plan

| Week | Theme | Outcome |
|---|---|---|
| **1 (now)** | Real intelligence backend + automation infrastructure | Real LLM output on real prompts (CPU-only) + self-running maintenance |
| **2** | Domain skills + first real usage | Kai becomes *yours*, not generic |
| **3** | CIS-aware finances + phone access (PWA) | Kai becomes useful on a Tuesday morning |
| **4** | Self-scoring + safety net | Kai self-sustains, you stop babysitting |

End of month definition of done: **You can demo Kai to a friend. You use it daily. It scores itself weekly. It backs itself up.**

---

## The items, in dependency order

### Week 1 — In flight / dispatching now (PM-driven, agent-implemented)

#### 0. ⭐ GitHub Models backend + Phi-4-mini default + behavioral scoreboard + coverage gate
**Status:** 🚧 Coding agent dispatched (PR opening shortly)
**Why first:** unblocks every later item — gives Kai a real LLM today, CPU-only, free.

#### 0a. Automation infrastructure (Friday cleanup, weekly Report Card, off-site backup, demo doc, journal/skills templates)
**Status:** 🚧 Will dispatch immediately after item 0 lands
**Deliverable:**
- `.github/workflows/friday-cleanup.yml` — weekly auto-PR: lint, dep bump, stale-branch list, metric refresh
- `.github/workflows/weekly-report-card.yml` — Mon 09:00 cron runs behavioral scoreboard, posts to tracking issue
- `scripts/backup_offsite.sh` + verification — encrypted nightly backup with restore test
- `docs/DEMO.md` — 5-min "show a friend" walkthrough
- `docs/operator-journal/_template.md` — for Dainius to dump real session notes
- `skills/_template.md` + 3 starter empty skill files (cis-deductions, ladder-safety, mtd-vat) for Dainius to fill in
- `docs/CIS_FINANCE_DESIGN.md` — design doc for CIS-aware P29

#### 0b. Additional remote backends (Groq, HuggingFace, OpenRouter) as fallback chain
**Status:** Queued (depends on 0)
**Why:** Redundancy. Free. Groq runs Llama-3.3-70B at ~500 tok/s — *faster than your future GPU*.

---

### Week 2 — Things only Dainius can do (PM-supported, not dispatchable)

#### 1. Write 10 site-engineer skill files
**Why:** This is the moat. Nobody else can write your domain knowledge. Skills Hub already loads `.md` files hot.
**Templates ready:** `skills/_template.md` + 3 starters scaffolded by item 0a.
**Suggested topics:** CIS deductions, CSCS card requirements, ladder/scaffold safety (Working at Height Regs 2005), CDM 2015 duties, NICEIC checks, BS 7671 basics, asbestos awareness (CAR 2012), site diary template, RAMS template, TBT topics rota.

#### 2. First real operator journal entries
**Why:** Right now Kai is built but nobody drives it. Even one real session a week is the only feedback loop that matters.
**Template ready:** `docs/operator-journal/_template.md` scaffolded by item 0a.
**Cadence:** 1 entry per week minimum. PM (me) will read them and propose follow-up work.

---

### Week 3 — User-facing value (PM-driven, agent-implemented)

#### 3. CIS-aware P29 Financial Awareness
**Status:** Queued behind item 0a (design doc lands in 0a, build in week 3)
**Scope:** NOT generic savings tracker. UK construction-subcontractor reality:
- CIS deductions tracker (20% / 30% / gross)
- Invoice generator with CIS line
- MTD-ready VAT summary
- Class 2/4 NI calculator
- Mileage tracker
**Deliverable:** `financial-awareness/` service + `/finance/cis`, `/finance/invoice`, `/finance/vat`, `/finance/summary` endpoints + dashboard tab.

#### 4. PWA polish for one-tap phone install
**Status:** README says PWA exists — verify and harden.
**Deliverable:** App icon, splash screen, offline shell, voice button on home, install instructions in `docs/PHONE_SETUP.md`.

---

### Week 4 — Self-sustaining (PM-driven, agent-implemented)

#### 5. Wire weekly Report Card to surface trends
**Status:** Cron lands in 0a — week 4 adds trend graph + auto-issue if regression.

#### 6. Verify off-site backup + restore drill
**Status:** Script lands in 0a — week 4 runs first restore drill, documents in runbook.

#### 7. Close lingering non-cleanup PRs (#54)
**Status:** PR #46 already merged on 2026-06-01; only #54 remains in this older Phase 0.5 bucket. Cleanup Sprint draft PRs (#67, #69) are tracked separately in Week 2 PM docs.

---

## What's still NOT in scope (deliberately)

- Anything requiring real LLM quality measurement before item 0 lands
- Multi-specialist *consensus quality* — Phase 2
- Voice/avatar quality — Phase 4
- Custom mobile app (PWA only — no app store accounts)
- Anything that would tempt fake metrics

---

## Done definition for Phase 0.5

End of week 4 → mark Phase 0 truly closed in STRATEGIC_PLAN → ready to flip to
Phase 1 the day RTX 5080 lands. By that point, Kai is already *useful* — GPU
becomes optimisation, not blocker.
