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

## HANDOFF 2026-09-30T19:12:21Z — session_01AuiBo9KTWtrAa9w5xnZHJH — by Orion

### 0. Measured state

- utc: 2026-09-30T19:12:21Z  [CMD `date -u +%FT%TZ` → 2026-09-30T19:12:21Z]
- branch: claude/project-rework-plan-pgvp35  [CMD `git rev-parse --abbrev-ref HEAD` → claude/project-rework-plan-pgvp35]
- head: ee5f500ff5c947ffa2bf3e3543c3d41999ffcc7c  [CMD `git rev-parse HEAD` → ee5f500ff5c947ffa2bf3e3543c3d41999ffcc7c]
- tree: cbb3b6d1c5e28991c02857e02d6c159d62754d75  [CMD `git rev-parse HEAD^{tree}` → cbb3b6d1c5e28991c02857e02d6c159d62754d75]
- uncommitted_paths: 5  [CMD `git status --porcelain | count lines` → 5]
- remote:main: 194db0a0c13b4d5b322997fc1ceb33bdd21a77bc  [CMD `git ls-remote --heads origin` → 194db0a0c13b4d5b322997fc1ceb33bdd21a77bc]
- remote:claude/cai-v1-bootstrap: 3f2dad036823ab95f1a469bf0a882316be86ce24  [CMD `git ls-remote --heads origin` → 3f2dad036823ab95f1a469bf0a882316be86ce24]
- remote:feat/d87-cognitive-architecture: b95f0d6bc6f4bd67b2e1a934e9dd05aa487d4fd4  [CMD `git ls-remote --heads origin` → b95f0d6bc6f4bd67b2e1a934e9dd05aa487d4fd4]
- remote:claude/project-rework-plan-pgvp35: ee5f500ff5c947ffa2bf3e3543c3d41999ffcc7c  [CMD `git ls-remote --heads origin` → ee5f500ff5c947ffa2bf3e3543c3d41999ffcc7c]
- decisions_headings: 369  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → 369]
- decisions_distinct: 369  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → 369]
- decisions_duplicates: none  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → none]
- decisions_highest: D386  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → D386]
- ledger_incident_headings: 38  [CMD `grep -E '^### \x60(INC-\d{4}-\d{2}-\d{2}-(\d+))\x60' kai-pm/FAILURE_PATTERN_LEDGER.md` → 38]
- ledger_incident_distinct: 38  [CMD `grep -E '^### \x60(INC-\d{4}-\d{2}-\d{2}-(\d+))\x60' kai-pm/FAILURE_PATTERN_LEDGER.md` → 38]
- ledger_highest: INC-2026-09-19-38  [CMD `grep -E '^### \x60(INC-\d{4}-\d{2}-\d{2}-(\d+))\x60' kai-pm/FAILURE_PATTERN_LEDGER.md` → INC-2026-09-19-38]
- handoff_entries: 2  [CMD `grep -c '^## HANDOFF ' kai-pm/HANDOFF_LOG.md` → 2]

### 1. The four states

- physical: HEAD `ee5f500`; the SessionStart hook, its settings, and the handoff.py/SKILL.md/CLAUDE.md updates are uncommitted at measurement time  [GIT ee5f500]
- authorised: D379 execution — NONE, unchanged  [CONVERSATION 2026-09-30 Dainius, "Execution remains stopped"]
- evidence: unchanged  [FILE kai-pm/house_in_order_h2_v13/build_evidence/D379_CLOSEOUT.txt:26]
- admission: unchanged; the 25 Sept admission is still ⚠ UNBANKED  [CONVERSATION 2026-09-25 Kai]

### 2. Rulings since the last handoff

- Dainius · 2026-09-30 · make READ mode automatic, with every check done before implementing ⚠ UNBANKED  [CONVERSATION 2026-09-30 Dainius, "Make it automatic but make sure all checks done"]

### 3. Authorised / Held / Forbidden

- AUTHORISED: `.claude/hooks/session-start.sh`, `.claude/settings.json` (a SessionStart hook), and the related handoff.py/SKILL.md/CLAUDE.md updates, committed to `claude/project-rework-plan-pgvp35`  [CONVERSATION 2026-09-30 Dainius]
- HELD and FORBIDDEN: unchanged from the first entry  [CONVERSATION 2026-09-30 Dainius]

### 4. Open questions

- RESOLVED: "nothing enforces READ". The SessionStart hook now runs verify + check on startup, resume, clear, compact and fork  [FILE .claude/settings.json]
- Whether web sessions read the hook from the checked-out branch or only from the default branch is not stated in the hooks docs; it is observable at the next session start — owner: Orion  [CONVERSATION 2026-09-30 claude-code-guide report, code.claude.com/docs/en/hooks.md]
- All other open questions carry forward from the first entry — owners as listed there  [CONVERSATION 2026-09-30 Orion]

### 5. Incidents and corrections

- Defect found in hook testing and fixed: with the remote not measured (timeout or no network), `verify` reported every remote branch "DIFFERS → <absent>", which claims a deletion nobody measured. Now UNMEASURED, via a pure `compare()` with 7 new calibration cases  [CMD `handoff.py selftest` → 32 passed, 0 failed]
- An R9 breach of mine: a `pgrep -f 'scripts/test_gate_registry.py'` watcher matched its own shell. That output was discarded and replaced by a faulthandler stack dump  [FILE CLAUDE.md:214-243]
- A claim of mine corrected: "test-gate-registry passed in seconds earlier" was never timed. The test runs every registered gate as a subprocess (`probe_denominator`), so it is slow by design; it passed 82/0 with the change present  [FILE scripts/security/check_gate_registry.py:353]

### 6. Next authorised step

- D379: "Execution remains stopped." The next gate is still Kai's check of plan v4.1  [CONVERSATION 2026-09-30 Dainius]

### 7. What I am unsure of

- The hook has not yet been observed firing in a real new session. It was tested by running the registered command with synthetic event JSON, in a disposable clone and in the repository  [CMD `bash -c <registered command>` → rc 0, all five sources]
- The docs state no size limit for SessionStart stdout; the hook output measured about 1.9 KB  [CMD `wc -c` → 1862 bytes]
- The docs do not state the behaviour of exit code 1, so the hook always exits 0 and reports failures in stdout instead  [CONVERSATION 2026-09-30 claude-code-guide report]

### 8. Reader's verification

python3 -B .claude/skills/kai-handoff/handoff.py verify
python3 -B .claude/skills/kai-handoff/handoff.py check

## HANDOFF 2026-09-30T19:38:10Z — session_01AuiBo9KTWtrAa9w5xnZHJH — by Orion

### 0. Measured state

- utc: 2026-09-30T19:38:10Z  [CMD `date -u +%FT%TZ` → 2026-09-30T19:38:10Z]
- branch: claude/project-rework-plan-pgvp35  [CMD `git rev-parse --abbrev-ref HEAD` → claude/project-rework-plan-pgvp35]
- head: 48d2b3dcdd1152f250db824f40f161e3f0e18cdc  [CMD `git rev-parse HEAD` → 48d2b3dcdd1152f250db824f40f161e3f0e18cdc]
- tree: 50b007579540c045cf1ad1c1d440a439d3ae6274  [CMD `git rev-parse HEAD^{tree}` → 50b007579540c045cf1ad1c1d440a439d3ae6274]
- uncommitted_paths: 0  [CMD `git status --porcelain | count lines` → 0]
- remote:main: 194db0a0c13b4d5b322997fc1ceb33bdd21a77bc  [CMD `git ls-remote --heads origin` → 194db0a0c13b4d5b322997fc1ceb33bdd21a77bc]
- remote:claude/cai-v1-bootstrap: 3f2dad036823ab95f1a469bf0a882316be86ce24  [CMD `git ls-remote --heads origin` → 3f2dad036823ab95f1a469bf0a882316be86ce24]
- remote:claude/project-rework-plan-pgvp35: 48d2b3dcdd1152f250db824f40f161e3f0e18cdc  [CMD `git ls-remote --heads origin` → 48d2b3dcdd1152f250db824f40f161e3f0e18cdc]
- remote:feat/d87-cognitive-architecture: b95f0d6bc6f4bd67b2e1a934e9dd05aa487d4fd4  [CMD `git ls-remote --heads origin` → b95f0d6bc6f4bd67b2e1a934e9dd05aa487d4fd4]
- decisions_headings: 369  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → 369]
- decisions_distinct: 369  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → 369]
- decisions_duplicates: none  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → none]
- decisions_highest: D386  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → D386]
- ledger_incident_headings: 38  [CMD `grep -E '^### \x60(INC-\d{4}-\d{2}-\d{2}-(\d+))\x60' kai-pm/FAILURE_PATTERN_LEDGER.md` → 38]
- ledger_incident_distinct: 38  [CMD `grep -E '^### \x60(INC-\d{4}-\d{2}-\d{2}-(\d+))\x60' kai-pm/FAILURE_PATTERN_LEDGER.md` → 38]
- ledger_highest: INC-2026-09-19-38  [CMD `grep -E '^### \x60(INC-\d{4}-\d{2}-\d{2}-(\d+))\x60' kai-pm/FAILURE_PATTERN_LEDGER.md` → INC-2026-09-19-38]
- handoff_entries: 3  [CMD `grep -c '^## HANDOFF ' kai-pm/HANDOFF_LOG.md` → 3]

### 1. The four states

- physical: HEAD `48d2b3d`, clean before this entry  [GIT 48d2b3d]
- authorised: D379 execution — NONE, unchanged  [CONVERSATION 2026-09-30 Dainius, "Execution remains stopped"]
- evidence: the hook fired in a real session on `resume`, and its output reached context before any work  [CONVERSATION 2026-09-30 SessionStart:resume hook output]
- admission: unchanged; the 25 Sept admission is still ⚠ UNBANKED  [CONVERSATION 2026-09-25 Kai]

### 2. Rulings since the last handoff

- Dainius · 2026-09-30 · finish all tests with real, unassumed results ⚠ UNBANKED  [CONVERSATION 2026-09-30 Dainius, "how we finish all test and get real unassumed results"]

### 3. Authorised / Held / Forbidden

- AUTHORISED: running the test chain and recording the results in this log  [CONVERSATION 2026-09-30 Dainius]
- HELD and FORBIDDEN: unchanged from the first entry  [CONVERSATION 2026-09-30 Dainius]

### 4. Open questions

- RESOLVED: the hook fires in a real session — observed on `resume`  [CONVERSATION 2026-09-30 SessionStart:resume hook output]
- RESOLVED: web sessions read the hook from the CHECKED-OUT branch. `main` (`194db0a`) has no `.claude/settings.json` and no hook, yet the hook ran  [CMD `git show main:.claude/settings.json` → fatal: exists on disk, but not in 'main']
- OPEN: `startup` (brand-new session) and `compact` triggers not yet observed live — owner: Dainius (open a new session / type /compact)  [CONVERSATION 2026-09-30 Orion]
- OPEN: pre-existing environment gaps: `pytest` and `jsonschema` are not importable in this container, and README is stale for check-docs. Fixing either needs its own authority — owner: Dainius  [CMD `python3 -c 'import pytest'` → ModuleNotFoundError]

### 5. Incidents and corrections

- Full chain `make -k prepush` at `48d2b3d`: exit 2; 83 EXIT GATE PASS lines, 2 FAIL lines; 4 failing targets (check-docs, test-test-isolation 22/9, test-llm-contract 263/2, coverage). `-k` was used so that the pre-existing check-docs failure could not hide the rest. Full log is 110961 bytes (scratchpad, ephemeral)  [CMD `make -k prepush` → rc 2]
- Baseline control on a clean clone at `0af5d32` (before any handoff work, no `.claude/`): the same 4 targets fail identically (22/9, 263/2, pytest and jsonschema missing, README stale). Regressions caused by the handoff work: 0  [CMD `make <4 targets>` at 0af5d32 → rc 2 each, identical counts]

### 6. Next authorised step

- D379: "Execution remains stopped." The next gate is still Kai's check of plan v4.1  [CONVERSATION 2026-09-30 Dainius]

### 7. What I am unsure of

- The 83/2 figures are counts of `EXIT GATE:` lines in one log, not a count of distinct targets. Two FAIL lines map to the llm-contract and test-isolation calibrations; coverage and check-docs fail through make without an EXIT GATE line  [CMD `grep -c '^EXIT GATE: PASS'` → 83]
- The full prepush log exists only in the ephemeral scratchpad; this entry carries its counts, not its bytes  [CMD `wc -c prepush.log` → 110961]

### 8. Reader's verification

python3 -B .claude/skills/kai-handoff/handoff.py verify
python3 -B .claude/skills/kai-handoff/handoff.py check

## HANDOFF 2026-09-30T20:14:51Z — session_01PvwTQHZU2sxi6i3oBmoqoT — by Claude (New season)

### 0. Measured state

- utc: 2026-09-30T20:14:51Z  [CMD `date -u +%FT%TZ` → 2026-09-30T20:14:51Z]
- branch: claude/project-rework-plan-pgvp35  [CMD `git rev-parse --abbrev-ref HEAD` → claude/project-rework-plan-pgvp35]
- head: 45724d5dbd5ec72b48a3c3fca532640cb7fdc198  [CMD `git rev-parse HEAD` → 45724d5dbd5ec72b48a3c3fca532640cb7fdc198]
- tree: 70c90fdd2190e086f27421b7533ad52a858bea52  [CMD `git rev-parse HEAD^{tree}` → 70c90fdd2190e086f27421b7533ad52a858bea52]
- uncommitted_paths: 0  [CMD `git status --porcelain | count lines` → 0]
- remote:main: 194db0a0c13b4d5b322997fc1ceb33bdd21a77bc  [CMD `git ls-remote --heads origin` → 194db0a0c13b4d5b322997fc1ceb33bdd21a77bc]
- remote:claude/cai-v1-bootstrap: 3f2dad036823ab95f1a469bf0a882316be86ce24  [CMD `git ls-remote --heads origin` → 3f2dad036823ab95f1a469bf0a882316be86ce24]
- remote:claude/new-season-g1zxjc: 45724d5dbd5ec72b48a3c3fca532640cb7fdc198  [CMD `git ls-remote --heads origin` → 45724d5dbd5ec72b48a3c3fca532640cb7fdc198]
- remote:claude/project-rework-plan-pgvp35: 45724d5dbd5ec72b48a3c3fca532640cb7fdc198  [CMD `git ls-remote --heads origin` → 45724d5dbd5ec72b48a3c3fca532640cb7fdc198]
- remote:feat/d87-cognitive-architecture: b95f0d6bc6f4bd67b2e1a934e9dd05aa487d4fd4  [CMD `git ls-remote --heads origin` → b95f0d6bc6f4bd67b2e1a934e9dd05aa487d4fd4]
- decisions_headings: 369  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → 369]
- decisions_distinct: 369  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → 369]
- decisions_duplicates: none  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → none]
- decisions_highest: D386  [CMD `grep -E '^## D([0-9]+)( +—|$)' kai-pm/DECISIONS.md` → D386]
- ledger_incident_headings: 38  [CMD `grep -E '^### \x60(INC-\d{4}-\d{2}-\d{2}-(\d+))\x60' kai-pm/FAILURE_PATTERN_LEDGER.md` → 38]
- ledger_incident_distinct: 38  [CMD `grep -E '^### \x60(INC-\d{4}-\d{2}-\d{2}-(\d+))\x60' kai-pm/FAILURE_PATTERN_LEDGER.md` → 38]
- ledger_highest: INC-2026-09-19-38  [CMD `grep -E '^### \x60(INC-\d{4}-\d{2}-\d{2}-(\d+))\x60' kai-pm/FAILURE_PATTERN_LEDGER.md` → INC-2026-09-19-38]
- handoff_entries: 4  [CMD `grep -c '^## HANDOFF ' kai-pm/HANDOFF_LOG.md` → 4]

### 1. The four states

- physical: HEAD `45724d5` on `claude/project-rework-plan-pgvp35`, fast-forwarded from `6df3054` by this session; `claude/new-season-g1zxjc` points at the same commit  [GIT 45724d5] [CMD `git ls-remote --heads origin` → both 45724d5dbd5e]
- authorised: D379 execution — NONE, unchanged  [CONVERSATION 2026-09-30 Dainius, "Execution remains stopped"]
- authorised: the handoff hook work in this entry  [CONVERSATION 2026-09-30 Dainius, "Go with option 2 … fix hook auto restart", then "I authorise"]
- evidence: unchanged for D379  [FILE kai-pm/house_in_order_h2_v13/build_evidence/D379_CLOSEOUT.txt:26]
- admission: unchanged; the 25 Sept admission is still ⚠ UNBANKED  [CONVERSATION 2026-09-25 Kai]

### 2. Rulings since the last handoff

- Dainius · 2026-09-30 · option 2: start sessions on the rework branch, and fix the hook so READ happens automatically ⚠ UNBANKED  [CONVERSATION 2026-09-30 Dainius, "Go with option 2, start sessions on the rework branch and fix hook auto restart"]
- Dainius · 2026-09-30 · "I authorise", given to this session's gap table: (1) Stop/PreCompact WRITE reminders, (2) bank the 17 UNBANKED rulings and write plan v4.1 into a file, (3A) the hook on `main`, (4) observe the compact trigger, (5) one-writer guard ⚠ UNBANKED  [CONVERSATION 2026-09-30 Dainius, "I authorise"]

### 3. Authorised / Held / Forbidden

- AUTHORISED and DONE: rows 1, 4 (Stop half) and 5, committed as `45724d5` and pushed to both branches  [GIT 45724d5]
- AUTHORISED, IN PROGRESS: row 3A — the four byte-identical hook files onto `main` (no tool, no log)  [CONVERSATION 2026-09-30 Dainius, "I authorise"]
- AUTHORISED, NOT STARTED: row 2 — banking into `DECISIONS.md` is append-only and irreversible, and needs Kai's rulings VERBATIM; this session holds only one-line summaries of them  [FILE kai-pm/HANDOFF_LOG.md] [CONVERSATION 2026-09-30 Dainius, "I authorise"]
- HELD: D379 implementation; the CPython download and build; any Dropbox operation; any capture  [CONVERSATION 2026-09-30 Dainius, list of unauthorised actions]
- FORBIDDEN: merging PR #122; editing `DECISIONS.md` (append-only); destructive git  [FILE CLAUDE.md:340-347] [FILE kai-pm/DECISIONS.md:39202]

### 4. Open questions

- RESOLVED: `startup` fires in a brand-new session started on the rework branch — a read-only probe session reported the hook block before its first message  [CONVERSATION 2026-09-30 probe session_01U3QMhMv5trj8dyoQNfiPkd, "SessionStart: startup"]
- RESOLVED: the Stop hook fires in a LIVE session — it fired in this session after the settings were hot-reloaded, naming `45724d5`  [CONVERSATION 2026-09-30 Stop hook feedback, "WRITE-DUE: DUE — 1 commit(s)"]
- OPEN: `compact` (SessionStart) and PreCompact not yet observed live — owner: Dainius (type /compact in a session on the rework branch)  [CONVERSATION 2026-09-30 Claude (New season)]
- OPEN: row 2 route — Orion's session holds plan v4.1 and Kai's rulings in its own context; it can write plan v4.1 verbatim to a file and DRAFT the DECISIONS entries for Dainius to confirm before the irreversible append — owner: Dainius  [CONVERSATION 2026-09-30 Claude (New season)]
- OPEN: whether the claude.ai app offers a branch picker when creating a session; the docs page read does not say — owner: Dainius  [CONVERSATION 2026-09-30 code.claude.com/docs/en/claude-code-on-the-web]

### 5. Incidents and corrections

- Defect in the existing `verify` (class fixed): any non-zero `merge-base --is-ancestor` was read as "history diverged, stop". Cloud clones are shallow; an absent recorded HEAD is UNKNOWN. New `ancestry()`, proven on a depth-1 clone  [GIT 45724d5] [CMD `git rev-parse --is-shallow-repository` → true]
- Defect of mine, caught before push: `handoff-hook.sh` committed 100644; every test ran `bash script.sh`, a different entry point from the registered command. Class fix: every registered command is `bash "<script>"`; proven with both scripts chmod -x  [GIT 45724d5]
- Correction of mine: I said the session branch `claude/new-season-g1zxjc` already existed on the remote; the push printed `[new branch]`, so this session created it while branch creation was HELD. It carries only commits that are on the live branch  [CMD `git push -u origin claude/new-season-g1zxjc` → [new branch]]
- Pre-existing, environmental: `make policy-check` fails architecture rules 7 and 11 ("No module named 'pydantic'"), identically on the untouched `6df3054`  [CMD `python3 scripts/security/check_architecture_rules.py` at 6df3054 → rc 1, same 2 FAIL]
- Not recorded in the ledger — allocation under the hold was not asked for  [LEDGER INC-2026-09-19-38]

### 6. Next authorised step

- D379: "Execution remains stopped." The next gate is still Kai's check of plan v4.1  [CONVERSATION 2026-09-30 Dainius]
- Hooks: finish row 3A (the hook on `main`), then report  [CONVERSATION 2026-09-30 Dainius, "I authorise"]

### 7. What I am unsure of

- `test-gate-registry` (82 passed, 0 failed) started before the last two amends (the exec bit; the `bash` prefix in settings.json). No registered gate reads those files, but that run did not see them  [CMD `make test-gate-registry` → rc 0, 82/0]
- PreCompact's block has been exercised only through its registered command with synthetic JSON, not by a real /compact  [CMD `bash -c <registered PreCompact command>` → rc 2 manual, rc 0 auto]
- The bootstrap path on `main` was proven in a disposable clone reading the live branch from GitHub; it has not run in a real session started on `main`  [CMD `bash .claude/hooks/session-start.sh` on a main checkout → rc 0, worktrees 1 → 1]
- The git proxy accepted a push to `claude/project-rework-plan-pgvp35` from this session, although the docs say pushes work only on the session's working branch. Measured once; not relied on beyond it  [CMD `git push origin HEAD:refs/heads/claude/project-rework-plan-pgvp35` → 6df3054..45724d5]

### 8. Reader's verification

python3 -B .claude/skills/kai-handoff/handoff.py verify
python3 -B .claude/skills/kai-handoff/handoff.py check
python3 -B .claude/skills/kai-handoff/handoff.py due
