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

**The table above is kept as written (the state at 2817bd8).** Update below.

### C-UPDATE (2026-09-30, after HANDOFF_LOG entry 7) — C1 and C2 now drafted, as RELAYED quotes

The other session (session_01PvwTQHZU2sxi6i3oBmoqoT) copied Dainius's
messages to it word for word into HANDOFF_LOG entry 7 §2, and Dainius
pasted the same list into this session (received 2026-09-30T20:46:45Z).
A substring search finds each quote in both places.

**Evidence class, stated.** These are **relayed** quotes: both copies were
produced by that session from its own transcript, which this session
cannot open. That is one source, not two independent ones (R13). Dainius
is the ruling-maker and pasted them himself, which is why they are drafted
here. Exact send times are not held; the date 2026-09-30 is.

**C1 → drafted.** Dainius, verbatim (relayed):
> "Go with option 2, start sessions on the rework branch and fix hook auto restart"

**C2 → drafted.** Dainius, verbatim (relayed), to that session's gap
table (rows 1–5, as HANDOFF_LOG entry 5 §2 lists them):
> "I authorise"

Caution: in this session's own transcript, "I authorise" also occurs
four times with other meanings. The row list it approved is taken from
entry 5, not from the quote itself.

**Also relayed in entry 7, not rulings:** "New season" (the session's
name); "So  is it best you can do and is it all avenues explored to make
sure hook works and you got long term memory" (a question, double space
as sent); "Yes, send Orion the row 2 request" (an instruction to relay,
not a programme ruling).

**New in entry 7 §2, attribution noted.** The ruling "Orion continues all
other work; one session per branch at a time" was **worded by Orion**, in
a message Orion drafted for Dainius to paste. It is Dainius's because he
sent it. If banked, it should say so.

**Result.** Section C now has **0** rulings that cannot be banked from a
verbatim or relayed-verbatim source. The six-blocker list (A2) and the
full A7/A8 tables are still open, as stated above.

---

## D. Suggested next steps (not taken)

1. Dainius reads this draft and decides, ruling by ruling: bank, merge into
   one continuity entry (see plan v4.1 Part D), or leave in the log only.
2. For A7 and A8, copy the full tables from the source messages and re-run
   the verbatim check before any append.
3. For A2's blocker list and C1–C2, get the verbatim text from its holder
   (Kai; the other session) before banking.
4. Only then: re-derive the allocator, and append with `&&`-chained gates.

---

## E. FINAL TEXT FOR CONFIRMATION — Dainius's choice of 2026-09-30

**Dainius, verbatim:** "go with your recommendation". That means: fold A1–A6
into one continuity entry; bank A7 and A8, each as its own entry, with their
full text copied word for word; keep B1–B5 and C1–C2 in the log only.

**Still NOT appended.** The three entries below come back to Dainius for
confirmation first. The D-numbers stay placeholders (`D<a>`, `D<b>`,
`D<c>`). At the moment of an authorised append, the allocator is
re-derived with `^## D[0-9]+( +—|$)`, and the entries take the next free
numbers in this order. `<UTC>` is the measured time of the append.

**How the quotes were produced.** Each quoted block below was CUT by a
program from the source message text in this session's transcript, never
retyped. After assembly, every block was checked as a substring of its
source (see E.4). Tabs in the Q6–Q9 table are kept inside a code block.

### E.1 — Continuity record (folds A1–A6)

```
## D<a> — <UTC> — CONTINUITY RECORD: THE 25 SEPTEMBER 2026 ADMISSION OF 8e3ee69 · fc1bb9d · 630ceaf · eb52f73, AND RELATED RULINGS. GOVERNANCE ONLY — NO IMPLEMENTATION, CANDIDATE, STAGE-A OR HOLDOUT AUTHORITY.
```

**Authority.** Kai's rulings received by Orion's session on
2026-09-25T16:18:17Z, relayed by Dainius; banked at Dainius's instruction
of 2026-09-30 ("go with your recommendation"). The record's content is
the specification Kai himself wrote for it:

> who: Dainius as final authority, following Kai’s commit-by-commit adjudication
> when: 25 September 2026
> why: the four commits were individually adjudicated; D379 closeout remained rejected but their bounded changes were accepted
> state: eb52f73 is the admitted technical restart point; fc1bb9d carries zero authority weight; no candidate/downstream authority was granted
> correction: the later reconstruction-from-86ebfde ruling resulted from incomplete cold-start recovery and does not supersede the 25 September admission

**The ruling, in Kai's words:**

> The contradiction is resolved. The 25 September admission ruling stands.
>
> I recovered the earlier ruling from the prior thread. It was explicit and commit-scoped: 8e3ee69, fc1bb9d, 630ceaf, and eb52f73 were accepted; fc1bb9d carried zero authority weight because it is non-authoritative field notes; and eb52f73 was named the last admitted technical state. You then explicitly accepted that position and said future repair work starts from eb52f73.

**Programme state as Kai recorded it:**

```
Physical HEAD: 0af5d32072bcf1d09c09e5c83c9b5b71b5560676
Physical tree: 4c05a750e1aed738879c140b51782de83bdf26e3
Admitted technical state: eb52f73fa6485534ca7e28a42055861c69e94cc4
D379 tranche: still not closed
Candidate / real Stage A / production holdout: still not authorised
PR #122: remains DO NOT MERGE
```

**Correction, in Kai's words:**

> My later new-thread ruling to reconstruct from 86ebfde was therefore wrong. It was made with incomplete continuity: I had repository evidence but was missing an already-made consequential admission decision. That is exactly the kind of cold-start failure our doctrine is supposed to prevent.

**Related rulings in the same message (A3, A5, A6):**

- Rebuild from 86ebfde (K1):
> NO. The rebuild is cancelled.
  The repair base is `eb52f73fa6485534ca7e28a42055861c69e94cc4`.
- K2:
> The number of launch sites is a reported measurement, not a hard-coded gate. We must never encode “there shall be nine” because the legitimate denominator can change.
>
> The coverage property is a gate:
>
> Every Python child-process launch in the governed harness population must pass through the governed launcher and therefore receive the required startup condition; any bypassing Python launch site is a hard failure.
- Build-log store:
> For the durable canonical copy, I choose Dropbox, not a GitHub Actions artifact.
>
> No build starts until the durable destination exists.

**Not banked here, stated.** The list of the six blockers behind the
rejected closeout is not held in Kai's words in this session, so it is not
put in Kai's mouth. Orion's working list is in `kai-pm/D379_PLAN_V4_1.md`
§B1 (non-authoritative).

**Rule carried.** Repository silence cannot erase a consequential decision
that was explicitly made and accepted; recover the authority history first,
then adjudicate. Kai, verbatim, on one line:

> current repository state can invalidate an old factual claim, but repository silence cannot erase a prior consequential decision that was explicitly made and accepted.

### E.2 — Q6–Q9 rulings (A7)

```
## D<b> — <UTC> — KAI RULINGS Q6–Q9 ON THE D379 REPAIR PLAN (F12, F13, plan_selection, BUILD-LOG AUTHORITY). GOVERNANCE ONLY — BANKING IS NOT EXECUTION.
```

**Authority.** Kai's ruling received 2026-09-25T15:57:02Z, relayed by
Dainius. Table verbatim, tabs preserved:

```
Q6–Q9 — KAI RULING

Question	Ruling	Reason	Confidence
Q6 — F12 B5 blocker?	YES — BLOCKER	D379 explicitly requires I1B-4 duplicate output → REFUSE BEFORE SELECTION. Current reconcile() accepts identical duplicate multisets.	1.00
Q7 — F13 harness blocker?	YES — BLOCKER	I independently counted 9 [sys.executable, …] launch sites. Python flags do not inherit from the parent merely because sys.executable is reused.	0.99
Q8 — plan_selection acceptable?	YES, with strict boundary conditions	A single real decision function inside holdout.py is preferable to reproducing holdout semantics inside the harness.	0.96
Q9 — commit build log if ≤5 MB?	NO, not as stated	File size does not create repository authority. R10 says full evidence survives; D379 says no arbitrary new tracked paths.	0.99
```

**Scope, stated.** The same message also ruled on the F12 class repair, the
F13 launcher location, Q1a-9's semantic invariant, the interpreter build,
and the repair branch base. Its branch-base ruling (rebuild from `86ebfde`)
was later withdrawn by Kai (see D<a>). Those parts are not banked by this
entry.

### E.3 — Plan v4 findings (A8)

```
## D<c> — <UTC> — KAI FINDINGS KAI-V4-01 … 08 ON D379 REPAIR PLAN v4. PLAN RETURNED FOR REVISION. GOVERNANCE ONLY — BANKING IS NOT EXECUTION.
```

**Authority.** Received 2026-09-30T17:34:56Z from Dainius, reproducing the
findings of "Kai's 30 September adjudication". Findings verbatim:

> 1. KAI‑V4‑01 — BLOCKER: Stage‑B result, provenance and binding can be changed together and self-certify.
> 2. KAI‑V4‑02 — BLOCKER: Repaired controls can generate their own acceptance evidence without a pre-capture fixity gate.
> 3. KAI‑V4‑03 — BLOCKER: F13’s child-launch population is not closed across aliases, wrappers and alternate launch mechanisms.
> 4. KAI‑V4‑04 — BLOCKER: E7’s full-log round-trip cannot occur before the build that creates the log.
> 5. KAI‑V4‑05 — MAJOR: Interpreter dependency and signature closure are incomplete.
> 6. KAI‑V4‑06 — MAJOR: The NFC/NFD refusal lacks an explicit canonicalisation rule.
> 7. KAI‑V4‑07 — MAJOR: A Part D commit placed only on the old branch would not be ancestral to the repair branch.
> 8. KAI‑V4‑08 — MAJOR: Authority calibration proves mapping completeness, but not that each authority mapping is correct.

**Consequence.** Plan v4.1 (`kai-pm/D379_PLAN_V4_1.md`) was written in
response. Kai has not yet checked v4.1, and DeepSeek has not attacked it.
D379 execution remains stopped.

### E.4 — Verification record

Measured on 2026-09-30 after assembly, against the three source messages
(transcript timestamps 2026-09-25T15:57:02Z, 2026-09-25T16:18:17Z,
2026-09-30T17:34:56Z):

- blockquote paragraphs and code-block spans checked: **14**, found verbatim
  in a source: **14**, not found: **0**. That is 13 spans from the first run,
  plus the Kai "Rule carried" quote, which the first run missed because it
  was line-wrapped inside quotation marks; it now stands on one line and
  matches.
- Checker calibration: one character changed in a verified span
  ("self-certify" → "self certify") → NOT found (known-positive); the
  unchanged span → found (known-negative).
