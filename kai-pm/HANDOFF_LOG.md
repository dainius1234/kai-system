# HANDOFF LOG — KAI project continuity

**NON-AUTHORITATIVE WORKING MEMORY WITH SOURCES. APPEND-ONLY.**

This log creates no programme state, grants no permission, admits nothing
and closes nothing. Authority lives in `kai-pm/DECISIONS.md`,
`kai-pm/FAILURE_PATTERN_LEDGER.md` and Git objects. Every entry points at
those, and marks `⚠ UNBANKED` any ruling that exists only in conversation.

- Read: `python3 -B .claude/skills/kai-handoff/handoff.py verify`
- Check: `python3 -B .claude/skills/kai-handoff/handoff.py check`
- Rules: `.claude/skills/kai-handoff/SKILL.md`

Never edit or delete an entry. A correction is a new entry.

---

## HANDOFF 2026-09-30T18:18:10Z — session_01AuiBo9KTWtrAa9w5xnZHJH — by Orion

### 0. Measured state

- utc: 2026-09-30T18:18:10Z  [CMD `date -u +%FT%TZ` → 2026-09-30T18:18:10Z]
- branch: claude/project-rework-plan-pgvp35  [CMD `git rev-parse --abbrev-ref HEAD` → claude/project-rework-plan-pgvp35]
- head: 0af5d32072bcf1d09c09e5c83c9b5b71b5560676  [CMD `git rev-parse HEAD` → 0af5d32072bcf1d09c09e5c83c9b5b71b5560676]
- tree: 4c05a750e1aed738879c140b51782de83bdf26e3  [CMD `git rev-parse HEAD^{tree}` → 4c05a750e1aed738879c140b51782de83bdf26e3]
- uncommitted_paths: 1  [CMD `git status --porcelain | count lines` → 1]
- remote:claude/project-rework-plan-pgvp35: 0af5d32072bcf1d09c09e5c83c9b5b71b5560676  [CMD `git ls-remote --heads origin` → 0af5d32072bcf1d09c09e5c83c9b5b71b5560676]
- remote:main: 194db0a0c13b4d5b322997fc1ceb33bdd21a77bc  [CMD `git ls-remote --heads origin` → 194db0a0c13b4d5b322997fc1ceb33bdd21a77bc]
- remote:claude/cai-v1-bootstrap: 3f2dad036823ab95f1a469bf0a882316be86ce24  [CMD `git ls-remote --heads origin` → 3f2dad036823ab95f1a469bf0a882316be86ce24]
- remote:feat/d87-cognitive-architecture: b95f0d6bc6f4bd67b2e1a934e9dd05aa487d4fd4  [CMD `git ls-remote --heads origin` → b95f0d6bc6f4bd67b2e1a934e9dd05aa487d4fd4]
- decisions_headings: 369  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → 369]
- decisions_distinct: 369  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → 369]
- decisions_duplicates: none  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → none]
- decisions_highest: D386  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → D386]
- ledger_incident_headings: 38  [CMD `grep -E '^### \x60(INC-\d{4}-\d{2}-\d{2}-(\d+))\x60' kai-pm/FAILURE_PATTERN_LEDGER.md` → 38]
- ledger_incident_distinct: 38  [CMD `grep -E '^### \x60(INC-\d{4}-\d{2}-\d{2}-(\d+))\x60' kai-pm/FAILURE_PATTERN_LEDGER.md` → 38]
- ledger_highest: INC-2026-09-19-38  [CMD `grep -E '^### \x60(INC-\d{4}-\d{2}-\d{2}-(\d+))\x60' kai-pm/FAILURE_PATTERN_LEDGER.md` → INC-2026-09-19-38]
- handoff_entries: 0 (no log)  [CMD `grep -c '^## HANDOFF ' kai-pm/HANDOFF_LOG.md` → 0 (no log)]

### 1. The four states

- physical: HEAD `0af5d32` on `claude/project-rework-plan-pgvp35` before this entry's commit; it differs from `eb52f73` only in `kai-pm/ORION_FIELD_NOTES.md` (+99)  [GIT 0af5d32] [CMD `git diff --stat eb52f73 0af5d32` → 1 file, 99 insertions]
- authorised: D379 execution — NONE  [CONVERSATION 2026-09-30 Dainius, "Execution remains stopped"]
- authorised: this handoff skill and log only, as two new tracked locations (`.claude/skills/kai-handoff/`, `kai-pm/HANDOFF_LOG.md`)  [CONVERSATION 2026-09-30 Dainius, "Ok go with B, draft our own" then "Implement to the highest standard"]
- evidence: latest D379 capture is at `eb52f73`: 40 cases, PASS 16, FAIL 0, HELD 24  [FILE kai-pm/house_in_order_h2_v13/build_evidence/D379_CLOSEOUT.txt:26]
- admission: `eb52f73` is the admitted technical state; `8e3ee69`, `fc1bb9d` (zero authority weight), `630ceaf`, `eb52f73` accepted individually ⚠ UNBANKED  [CONVERSATION 2026-09-25 Kai, commit-by-commit adjudication; reaffirmed CONVERSATION 2026-09-30 Kai]
- admission: the D379 closeout and tranche are REJECTED, six blockers ⚠ UNBANKED  [CONVERSATION 2026-09-25 Kai]

### 2. Rulings since the last handoff

- Latest banked ruling is D386; nothing later is banked. This is the first handoff, so the load-bearing unbanked rulings are listed below  [D386] [CMD `handoff.py measure` → decisions_highest D386]
- Kai · 2026-09-25 · four commits admitted; `eb52f73` is the admitted technical restart state ⚠ UNBANKED  [CONVERSATION 2026-09-25 Kai]
- Kai · 2026-09-25 · D379 closeout and tranche rejected (Stage-A validation; Pass-A subject binding; Stage-B transport; Q1a-9 proxy; holdout population binding; D380/D385-compliant interpreter) ⚠ UNBANKED  [CONVERSATION 2026-09-25 Kai]
- Kai · 2026-09-30 · the 25 Sept admission stands; the rebuild from `86ebfde` is cancelled; the repair branch will come from `eb52f73` ⚠ UNBANKED  [CONVERSATION 2026-09-30 Kai]
- Kai · 2026-09-30 · K2: the launch-site count is a measurement; coverage (`bypass=0`) is a gate ⚠ UNBANKED  [CONVERSATION 2026-09-30 Kai]
- Kai · 2026-09-30 · Dropbox is the canonical build-log store; no build before the destination is proven ⚠ UNBANKED  [CONVERSATION 2026-09-30 Kai]
- Kai · 2026-09-30 · Q6–Q9: F12 and F13 are blockers; `plan_selection` accepted with boundary conditions; no size-based authority for committing logs ⚠ UNBANKED  [CONVERSATION 2026-09-30 Kai]
- Kai · 2026-09-30 · plan v4 findings KAI-V4-01..08 (01–04 BLOCKER, 05–08 MAJOR) ⚠ UNBANKED  [CONVERSATION 2026-09-30 Kai]
- Dainius · 2026-09-30 · adopt and implement this handoff skill ⚠ UNBANKED  [CONVERSATION 2026-09-30 Dainius]

### 3. Authorised / Held / Forbidden

- AUTHORISED: the handoff skill, this log, and committing/pushing them to `claude/project-rework-plan-pgvp35`  [CONVERSATION 2026-09-30 Dainius, "Implement to the highest standard"]
- HELD: D379 implementation; branch creation; the CPython download and build; any Dropbox operation (E7a); the Part D append to `DECISIONS.md`; any capture  [CONVERSATION 2026-09-30 Dainius, list of unauthorised actions]
- FORBIDDEN: `BINANCE_API_KEY`/`BINANCE_API_SECRET` leaving broker-bridge; a push to `main` without authorisation; editing `DECISIONS.md` (append-only); destructive git; opening a PR unless asked  [FILE CLAUDE.md:318-330]
- FORBIDDEN: merging PR #122 (DO NOT MERGE)  [FILE kai-pm/DECISIONS.md:39202]
- FORBIDDEN: production Stage A, candidate, production Pass A/classification, holdout, blind 40, D387 allocation without authority  [CONVERSATION 2026-09-30 Kai, "Unauthorised" list]

### 4. Open questions

- Kai checks plan v4.1 against the repository, and confirms each KAI-V4-01..08 remedy answers the finding (Orion had headlines only) — owner: Kai  [CONVERSATION 2026-09-30 Orion, plan v4.1]
- Is D380's missing native-dependency identity (for example libssl/libz under /usr/lib) accepted risk, or does it need later authority? — owner: Kai  [CONVERSATION 2026-09-30 Orion, plan v4.1 E3]
- What CPython tag-signature state is acceptable before a build? — owner: Kai  [CONVERSATION 2026-09-30 Orion, plan v4.1 E1]
- After review, DeepSeek attacks v4.1 on the eight surfaces — owner: DeepSeek via Kai  [CONVERSATION 2026-09-30 Kai]
- Part D continuity-record append; the E7a Dropbox trial; the implementation grant — owner: Dainius  [CONVERSATION 2026-09-30 Orion, plan v4.1 open items]
- Delete the parked CAI branch `claude/cai-v1-bootstrap` (still on the remote; Orion's delete was refused with HTTP 403 earlier) — owner: Dainius  [CMD `git ls-remote --heads origin` → claude/cai-v1-bootstrap 3f2dad03]
- Whether to add a pointer to this skill in `CLAUDE.md`, so READ mode is prompted in every session — owner: Dainius  [CONVERSATION 2026-09-30 Orion, not implemented]

### 5. Incidents and corrections

- Orion's plan v2 proposed three tracked paths outside D379 §2 ("NO OTHER TRACKED PATH"); withdrawn in v3. Not recorded — ledger allocation is not authorised under the hold  [FILE kai-pm/DECISIONS.md:35106]
- Kai's cold-start rebuild-from-`86ebfde` ruling came from incomplete continuity; Kai withdrew it on 2026-09-30. Not recorded; it is the motivating case for this skill ⚠ UNBANKED  [CONVERSATION 2026-09-30 Kai]
- `ORION_FIELD_NOTES.md` says "370 entries"; the strict grammar measures 369. The notes are frozen, so this stays uncorrected  [FILE kai-pm/ORION_FIELD_NOTES.md:37] [CMD `handoff.py measure` → decisions_headings 369]
- Pre-existing gate failure at `0af5d32`: `make policy-check` fails only at `check-docs` ("README.md is STALE"). It is not caused by this change and not fixed here  [CMD `make policy-check` → rc=2, check-docs]
- Highest recorded ledger incident  [LEDGER INC-2026-09-19-38]

### 6. Next authorised step

- D379: "Execution remains stopped." The next gate is Kai's check of plan v4.1  [CONVERSATION 2026-09-30 Dainius]

### 7. What I am unsure of

- Plan v4.1 exists only in conversation; its full text is in no repository file. A lost thread loses it  [CONVERSATION 2026-09-30 Orion]
- The CPython `v3.11.15` source tree `8c6959bc…` and the valid tag signature are Kai-verified only; Orion cannot re-derive them without a download  [CONVERSATION 2026-09-30 Kai]
- The 369/369/D386 allocator is Orion-measured; Kai could not count the file independently  [CONVERSATION 2026-09-30 Kai]
- The Dropbox connector's size limit, overwrite behaviour and byte fidelity are unmeasured  [CONVERSATION 2026-09-30 Orion, plan v4.1 A4]
- Committing this log moves `claude/project-rework-plan-pgvp35` off `0af5d32`. Plan v4.1 B0 says that branch "stays at `0af5d32` unless the Part D append is authorised", so that sentence becomes inaccurate. The D379 repair base (`eb52f73`) is unaffected. Kai to note  [GIT 0af5d32]
- `scripts/security/hygiene_survey.py` counts `.claude/skills/kai-handoff` as a "service" (56 of 60 → 57 of 61), because its exclusion list names `scripts`, `tests` and `kai-pm` but not `.claude`. It is a survey, not a gate, and its finding total is unchanged at 7. The gate was not edited  [FILE scripts/security/hygiene_survey.py:74-76]
- The remote carries `feat/d87-cognitive-architecture` (`b95f0d6b`), which was not in Kai's 2026-09-30 branch list; its status is not examined  [CMD `git ls-remote --heads origin` → feat/d87-cognitive-architecture b95f0d6b]

### 8. Reader's verification

python3 -B .claude/skills/kai-handoff/handoff.py verify
python3 -B .claude/skills/kai-handoff/handoff.py check

## HANDOFF 2026-09-30T18:41:45Z — session_01AuiBo9KTWtrAa9w5xnZHJH — by Orion

### 0. Measured state

- utc: 2026-09-30T18:41:45Z  [CMD `date -u +%FT%TZ` → 2026-09-30T18:41:45Z]
- branch: claude/project-rework-plan-pgvp35  [CMD `git rev-parse --abbrev-ref HEAD` → claude/project-rework-plan-pgvp35]
- head: 7123ec2211ec2c70e531c070351bc1cfde05e1ff  [CMD `git rev-parse HEAD` → 7123ec2211ec2c70e531c070351bc1cfde05e1ff]
- tree: 5a92a278e507e688bff3fd4699b00066f9aa7a7e  [CMD `git rev-parse HEAD^{tree}` → 5a92a278e507e688bff3fd4699b00066f9aa7a7e]
- uncommitted_paths: 1  [CMD `git status --porcelain | count lines` → 1]
- remote:main: 194db0a0c13b4d5b322997fc1ceb33bdd21a77bc  [CMD `git ls-remote --heads origin` → 194db0a0c13b4d5b322997fc1ceb33bdd21a77bc]
- remote:claude/cai-v1-bootstrap: 3f2dad036823ab95f1a469bf0a882316be86ce24  [CMD `git ls-remote --heads origin` → 3f2dad036823ab95f1a469bf0a882316be86ce24]
- remote:claude/project-rework-plan-pgvp35: 7123ec2211ec2c70e531c070351bc1cfde05e1ff  [CMD `git ls-remote --heads origin` → 7123ec2211ec2c70e531c070351bc1cfde05e1ff]
- remote:feat/d87-cognitive-architecture: b95f0d6bc6f4bd67b2e1a934e9dd05aa487d4fd4  [CMD `git ls-remote --heads origin` → b95f0d6bc6f4bd67b2e1a934e9dd05aa487d4fd4]
- decisions_headings: 369  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → 369]
- decisions_distinct: 369  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → 369]
- decisions_duplicates: none  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → none]
- decisions_highest: D386  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → D386]
- ledger_incident_headings: 38  [CMD `grep -E '^### \x60(INC-\d{4}-\d{2}-\d{2}-(\d+))\x60' kai-pm/FAILURE_PATTERN_LEDGER.md` → 38]
- ledger_incident_distinct: 38  [CMD `grep -E '^### \x60(INC-\d{4}-\d{2}-\d{2}-(\d+))\x60' kai-pm/FAILURE_PATTERN_LEDGER.md` → 38]
- ledger_highest: INC-2026-09-19-38  [CMD `grep -E '^### \x60(INC-\d{4}-\d{2}-\d{2}-(\d+))\x60' kai-pm/FAILURE_PATTERN_LEDGER.md` → INC-2026-09-19-38]
- handoff_entries: 1  [CMD `grep -c '^## HANDOFF ' kai-pm/HANDOFF_LOG.md` → 1]

### 1. The four states

- physical: HEAD `7123ec2` (the kai-handoff commit), with the CLAUDE.md pointer uncommitted at measurement time  [GIT 7123ec2]
- authorised: D379 execution — NONE, unchanged since the previous entry  [CONVERSATION 2026-09-30 Dainius, "Execution remains stopped"]
- evidence: unchanged since the previous entry  [FILE kai-pm/house_in_order_h2_v13/build_evidence/D379_CLOSEOUT.txt:26]
- admission: unchanged since the previous entry; the 25 Sept admission is still ⚠ UNBANKED  [CONVERSATION 2026-09-25 Kai]

### 2. Rulings since the last handoff

- Dainius · 2026-09-30 · add the kai-handoff pointer to CLAUDE.md ⚠ UNBANKED  [CONVERSATION 2026-09-30 Dainius, "add the pointer to CLAUDE.md"]

### 3. Authorised / Held / Forbidden

- AUTHORISED: the CLAUDE.md pointer (a session-start line and one row in "Where things are"), committed to `claude/project-rework-plan-pgvp35`  [CONVERSATION 2026-09-30 Dainius]
- HELD and FORBIDDEN: unchanged from the previous entry  [CONVERSATION 2026-09-30 Dainius]

### 4. Open questions

- RESOLVED from the previous entry: "add a CLAUDE.md pointer" — now done  [FILE CLAUDE.md:8-18]
- All other open questions carry forward unchanged from the previous entry — owners as listed there  [CONVERSATION 2026-09-30 Orion]

### 5. Incidents and corrections

- The 7123ec2 commit message first said "14 known-positives". The measured count is 16 (12 check-rule cases plus 4 malformed tags); corrected before the push, with the tree hash unchanged  [GIT 7123ec2] [CMD `handoff.py selftest | count POS` → 16]

### 6. Next authorised step

- D379: "Execution remains stopped." The next gate is still Kai's check of plan v4.1  [CONVERSATION 2026-09-30 Dainius]

### 7. What I am unsure of

- Automatic discovery: during this session the harness listed `kai-handoff` among the available skills after 7123ec2. That shows discovery in a live session; a brand-new session has not yet been observed  [CONVERSATION 2026-09-30 harness skill list]
- The CLAUDE.md pointer tells a session to run READ mode. Nothing enforces it; only a SessionStart hook would, and none is configured  [FILE CLAUDE.md:8-18]

### 8. Reader's verification

python3 -B .claude/skills/kai-handoff/handoff.py verify
python3 -B .claude/skills/kai-handoff/handoff.py check
