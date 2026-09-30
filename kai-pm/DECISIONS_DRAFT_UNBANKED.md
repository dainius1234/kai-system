# DRAFT — banking candidates for the ⚠ UNBANKED rulings in HANDOFF_LOG entries 1–5

**THIS IS A DRAFT. IT IS NOT `DECISIONS.md` AND BANKS NOTHING.**
No D-number is allocated. Nothing here has been appended to
`kai-pm/DECISIONS.md`, which is append-only and irreversible. Dainius
confirms (and edits) this draft first; the allocator is re-derived
immediately before any authorised append, with the strict grammar
`^## D[0-9]+( +—|$)`.

**Provenance.** Drafted by Orion (session_01AuiBo9KTWtrAa9w5xnZHJH) on
2026-09-30 on Dainius's authorisation ("Yes: A and B"), at the request of
session_01PvwTQHZU2sxi6i3oBmoqoT (row 2, part B).

**How every quote was verified (R16/R17).** Each quote below was found
**character-for-character** in a user message of this session's transcript
(`~/.claude/projects/-home-user-kai-system/84284242-….jsonl`) by a
mechanical substring search. The "received" time is that message's
transcript timestamp (UTC). Kai's rulings reached this session as text
pasted by Dainius; they are attributed to Kai because the pasted text is
Kai's adjudication, and they are kept separate from Dainius's own words.

**Universe.** Every line containing `⚠ UNBANKED` in
`kai-pm/HANDOFF_LOG.md` at `a9b2693`: 21 matching lines, 1 of which is the
header's explanatory prose, leaving 20 ruling lines. Deduplicated
(the repeated "25 Sept admission still UNBANKED" state lines, and the §1/§2
duplicates of entry 1) into **14 distinct rulings**: **12 bankable from
verbatim text below**, **2 cannot bank** (section C). Rulings that are
bankable only in part say so. Reconciliation: sections A and B hold 13
rows for 12 rulings, because B1 and B2 are two quotes of one ruling
(adopting the skill); 12 + 2 = 14.

---

## CORRECTION FOUND WHILE DRAFTING — dates in HANDOFF_LOG entry 1

Entry 1 (written by Orion) tags Kai's rulings on the admission, the
86ebfde withdrawal, K2, the Dropbox store and Q6–Q9 as
`[CONVERSATION 2026-09-30 Kai]`. The transcript timestamps show they were
**received on 2026-09-25** (15:57:02Z and 16:18:17Z). The session's date
later changed to 2026-09-30 and Orion dated them from "today", not from the
source (an R16 error). The dates below are the measured ones. The log is
append-only; the correction goes in a new entry, not an edit.

---

## A. Kai's rulings

### A1. The 25 September admission of four commits — stands

- **Received:** 2026-09-25T16:18:17Z, pasted by Dainius.
- **Kai, verbatim:**
  > "The contradiction is resolved. The 25 September admission ruling stands."
  >
  > "I recovered the earlier ruling from the prior thread. It was explicit and commit-scoped: 8e3ee69, fc1bb9d, 630ceaf, and eb52f73 were accepted; fc1bb9d carried zero authority weight because it is non-authoritative field notes; and eb52f73 was named the last admitted technical state."
- **Limit, stated:** this is Kai's own **restatement** of the original
  25 September ruling. The original wording is not held in this session
  (it was compressed out of context). Bank the restatement, citing it as a
  restatement.
- **Covers log lines:** E1 §1 admission; E1 §2 "four commits admitted";
  the repeated "admission still UNBANKED" lines in E2–E5.

### A2. The D379 closeout and tranche remain rejected — **bankable in part**

- **Received:** 2026-09-25T16:18:17Z, pasted by Dainius.
- **Kai, verbatim:**
  > "D379 tranche: still not closed"
  >
  > "D379 closeout remained rejected but their bounded changes were accepted"
- **Cannot bank from verbatim text:** the list of the **six blockers**
  (Stage-A validation; Pass-A subject binding; Stage-B transport; Q1a-9
  proxy; holdout population binding; D380/D385-compliant interpreter).
  This session holds that list only in a compressed summary, not in Kai's
  words. The list must come from Kai's original text or be restated by Kai.
- **Covers log lines:** E1 §1 "closeout and tranche REJECTED"; E1 §2
  "closeout and tranche rejected (…)".

### A3. The rebuild from 86ebfde is cancelled; the repair base is eb52f73

- **Received:** 2026-09-25T16:18:17Z.
- **Kai, verbatim:**
  > "NO. The rebuild is cancelled."
  >
  > "The new repair branch should therefore start from:"
  > (followed by `eb52f73fa6485534ca7e28a42055861c69e94cc4`)
- **Covers log line:** E1 §2 "the 25 Sept admission stands; the rebuild from
  86ebfde is cancelled; the repair branch will come from eb52f73".

### A4. Kai withdraws his cold-start 86ebfde ruling

- **Received:** 2026-09-25T16:18:17Z.
- **Kai, verbatim:**
  > "My later new-thread ruling to reconstruct from 86ebfde was therefore wrong."
- **Covers log line:** E1 §5 "Kai's cold-start rebuild-from-86ebfde ruling …
  Kai withdrew it".

### A5. K2 — launch-site count is a measurement; coverage is a gate

- **Received:** 2026-09-25T16:18:17Z.
- **Kai, verbatim:**
  > "The number of launch sites is a reported measurement, not a hard-coded gate."
  >
  > "The coverage property is a gate:"
- **Covers log line:** E1 §2 "K2".

### A6. Dropbox is the canonical build-log store

- **Received:** 2026-09-25T16:18:17Z.
- **Kai, verbatim:**
  > "For the durable canonical copy, I choose Dropbox, not a GitHub Actions artifact."
  >
  > "No build starts until the durable destination exists."
- **Covers log line:** E1 §2 "Dropbox is the canonical build-log store".

### A7. Q6–Q9 rulings

- **Received:** 2026-09-25T15:57:02Z.
- **Kai, verbatim** (the question headings as they appear; the full ruling
  table is in that message):
  > "Q6 — F12 B5 blocker?"
- **Limit, stated:** only the Q6 heading was checked by substring search
  here. **Before any append, the full Q6–Q9 table must be copied from that
  message and re-verified** the same way; this draft does not reproduce it.
- **Covers log line:** E1 §2 "Q6–Q9".

### A8. Plan v4 findings KAI-V4-01 … 08

- **Received:** 2026-09-30T17:34:56Z, pasted by Dainius, attributed in
  that message to "Kai's 30 September adjudication".
- **Verbatim (first heading checked):**
  > "KAI‑V4‑01 — BLOCKER"
- **Limit, stated:** as A7 — the eight headlines must be copied from that
  message and re-verified before any append. They are also carried, as
  received, in `kai-pm/D379_PLAN_V4_1.md` Part 0.
- **Covers log line:** E1 §2 "plan v4 findings KAI-V4-01..08".

---

## B. Dainius's rulings (his own words to this session)

| # | received (UTC) | Dainius, verbatim | covers log line |
|---|---|---|---|
| B1 | 2026-09-30T17:42:31Z | "Ok go with B, draft our own" | E1 §2 "adopt and implement this handoff skill" |
| B2 | 2026-09-30T18:08:35Z | "Implement to the highest standard" | E1 §2 (same) |
| B3 | 2026-09-30T18:34:37Z | "add the pointer to CLAUDE.md" | E2 §2 |
| B4 | 2026-09-30T18:44:36Z | "Make it automatic but make sure all checks done to make sure all ok before implementing" | E3 §2 |
| B5 | 2026-09-30T19:19:28Z | "Ok how we finish all test and get real unassumed results to finalize it" | E4 §2 |

**Note on weight.** B1–B5 authorised tooling work (the kai-handoff skill,
the CLAUDE.md pointer, the SessionStart hook, the test run). Whether they
belong in `DECISIONS.md` at all, or only in the handoff log, is Dainius's
call; they change no Stage-A proposition and no D379 state.

---

## C. Cannot bank — no verbatim source held by this session

| # | ruling (as the log summarises it) | why |
|---|---|---|
| C1 | Dainius · 2026-09-30 · "option 2: start sessions on the rework branch, and fix the hook so READ happens automatically" (E5 §2) | Said to session_01PvwTQHZU2sxi6i3oBmoqoT, not to this one. A substring search of this transcript for "Go with option 2" finds **0** matches. Quoting it from that session's log would be cascaded memory (R16). That session, or Dainius, holds the verbatim text. |
| C2 | Dainius · 2026-09-30 · "I authorise" to that session's gap table, rows 1–5 (E5 §2) | Same: given to the other session. This session received a *different* "I authorise" (about the hook reaching `main`), which must not be conflated with it. |

---

## D. Suggested next steps (not taken)

1. Dainius reads this draft and decides, ruling by ruling: bank, merge into
   one continuity entry (see plan v4.1 Part D), or leave in the log only.
2. For A7 and A8, copy the full tables from the source messages and re-run
   the verbatim check before any append.
3. For A2's blocker list and C1–C2, get the verbatim text from its holder
   (Kai; the other session) before banking.
4. Only then: re-derive the allocator, and append with `&&`-chained gates.
