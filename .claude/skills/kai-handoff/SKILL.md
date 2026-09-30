---
name: kai-handoff
description: KAI project continuity handoff. Use READ at the start of every session in this repository, before any other work. Use WRITE before a session ends, before context is compressed, after any consequential ruling (admission, rejection, grant, hold), or whenever the operator says "handoff". The log is kai-pm/HANDOFF_LOG.md; the tool is .claude/skills/kai-handoff/handoff.py.
---

# kai-handoff

**A handoff is WORKING MEMORY WITH SOURCES.** It creates no programme
state, grants no permission, admits nothing and closes nothing. Authority
lives in `kai-pm/DECISIONS.md`, `kai-pm/FAILURE_PATTERN_LEDGER.md` and Git
objects. A handoff only points at them, and says plainly where something
is not yet in any of them.

It exists because of one incident. On 2026-09-25 Kai admitted four
commits in conversation. The decision was never banked. A later cold
thread read the repository, found silence, and reversed it. **Repository
silence cannot erase a consequential decision, and conversation memory
cannot establish one.** The handoff carries such decisions forward,
visibly marked, until they are banked.

## Hard rules

1. **Every claim carries a source tag.** In sections 0–7 every claim is
   a bullet (`- `), and every bullet carries at least one of:
   - `[GIT <sha>]`
   - `[D<n>]`
   - `[LEDGER INC-YYYY-MM-DD-NN]`
   - `[FILE <path>[:<line>]]`
   - ``[CMD `<command>` → <result>]``
   - `[CONVERSATION <YYYY-MM-DD> <who, what>]`

   A claim with no source is not written. Continuation lines are
   indented two spaces.
2. **Conversation-only rulings are marked `⚠ UNBANKED`** and must carry
   their `[CONVERSATION …]` tag. In section 2 every ruling is either
   `[D<n>]` or `⚠ UNBANKED`. There is no third state. Never drop one,
   never state one as settled fact.
3. **Append only.** Never edit or delete an earlier entry. A correction
   is a new entry that names what it corrects. `handoff.py check`
   enforces this against the committed log.
4. **Section 0 is measured, never typed.** Paste `handoff.py measure`
   output verbatim. Memory locates; it never supplies a value (R16).
5. **Say what you don't know.** Section 7 lists everything believed but
   not verified. An empty section 7 must say so with a source, not by
   omission.
6. **R1 on your own action.** Do not say "saved", "recorded" or
   "handed off" until `handoff.py check` exits 0 and the entry is
   committed and pushed.
7. **A handoff is not permission to commit it.** Committing follows the
   session's current authority like any other change. If committing is
   not authorised, say that the handoff exists only in the working tree.

## READ mode — session start, before any other work

```bash
python3 -B .claude/skills/kai-handoff/handoff.py verify
```

1. If the log is absent, say so. Do not reconstruct one from memory.
2. `verify` re-measures and compares with the last entry's section 0.
   - **MATCH** carries forward.
   - **DIFFERS** is reported old → new, and **the repository wins for
     facts**.
   - It lists the commits since the recorded HEAD, and warns if that HEAD
     is no longer an ancestor, which means history diverged. In that
     case, stop and report.
3. Read the last entry in full: sections 1–7. For every `⚠ UNBANKED`
   ruling, raise it with the operator before acting on anything that
   depends on it. It is neither established nor erased.
4. Report to the operator before starting work:
   - the verified state and any differences;
   - the unbanked rulings;
   - the open questions and their owners;
   - the **next authorised step, quoted with its source**.
5. Do nothing that section 3 of the last entry, or a newer instruction,
   does not authorise. Where they conflict, the newer operator
   instruction governs, subject to R14 (reconcile against the governing
   contract first).

## WRITE mode

1. `python3 -B .claude/skills/kai-handoff/handoff.py measure` and paste
   the output into section 0.
2. Fill sections 1–8 from sources you **open now** (R16). Use the template
   below.
3. Append the entry to the end of `kai-pm/HANDOFF_LOG.md`.
4. `python3 -B .claude/skills/kai-handoff/handoff.py check` must exit 0.
5. Commit and push only if authorised (rule 7), chained with `&&` (R3).
   Then report which it was.

## Entry template

```markdown
## HANDOFF <YYYY-MM-DDTHH:MM:SSZ> — <session or thread id> — by <producer>

### 0. Measured state

<verbatim output of `handoff.py measure`>

### 1. The four states

- physical: … [GIT …]
- authorised: … [D… | CONVERSATION …]
- evidence: … [FILE …]
- admission: … [D… | CONVERSATION … ⚠ UNBANKED]

### 2. Rulings since the last handoff

- <who> · <when> · <what> [D<n>] | ⚠ UNBANKED [CONVERSATION …]

### 3. Authorised / Held / Forbidden

- AUTHORISED: … [source]
- HELD: … [source]
- FORBIDDEN: … [source]

### 4. Open questions

- <question> — owner: <Dainius | Kai | Orion | DeepSeek> [source]

### 5. Incidents and corrections

- <what was wrong, what corrected it> [LEDGER …] or "not recorded — <why>" [source]

### 6. Next authorised step

- <exact quote> [source]   — or —   - None recorded [source]

### 7. What I am unsure of

- <belief not verified, and what would verify it> [source of the belief]

### 8. Reader's verification

python3 -B .claude/skills/kai-handoff/handoff.py verify
```

## Calibration

`python3 -B .claude/skills/kai-handoff/handoff.py selftest` proves that
every `check` rule fires on a known-positive and stays silent on the
known-negative, and that `verify` reads the **last** entry. The expected
answers come from the synthetic constructions, not from the checker.
Run it after any change to `handoff.py`.

## Limits, stated plainly

- Nothing forces WRITE to run. It depends on the producer running it at
  the right moments listed in the description, unless a SessionStart or
  PreCompact hook is added later.
- `check` proves that the entry has the right form and carries sources.
  It does not prove that a source says what the line claims. READ mode
  re-verifies section 0 mechanically. Sections 1–7 are checked by
  opening their sources.
- `verify` compares measured facts only. A misunderstood ruling, written
  down faithfully, stays misunderstood. The remedy is banking it.
