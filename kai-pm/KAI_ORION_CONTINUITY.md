# KAI ↔ ORION continuity protocol

> **STATUS: RECOVERY PROTOCOL — NOT A SOURCE OF PROGRAMME TRUTH.**
>
> This file exists only so a fresh Kai/ChatGPT thread can reconstruct the current state from the repository without depending on chat memory. If this file conflicts with the repository, the repository wins.

## 1. Authority order

Use this order when recovering state:

1. **Git branch/commit/tree and machine evidence** — exact repository state.
2. **Latest D-numbered entry in `kai-pm/DECISIONS.md` on the active work branch** — latest programme/governance record.
3. **`CLAUDE.md`** — binding operating rules and stop signals.
4. **`kai-pm/ENGINEERING_DOCTRINE.md`** — engineering doctrine/fingerprint where applicable.
5. **Canonical machine registers / experiment artefacts / workflow run evidence** for the specific claim.
6. Other plans, trackers, READMEs and status documents only after their authority/currentness has been qualified.

Never promote README, STATUS, backlog, continuation notes, this continuity file, or chat memory above the sources above.

## 2. Fresh-thread recovery algorithm

Before giving consequential programme advice, a fresh Kai thread must:

1. Open repository `dainius1234/kai-system`.
2. Identify the active development branch by finding the branch containing the latest D-numbered governance commit; do not assume `main` is current.
3. Record exact **HEAD SHA + tree SHA**.
4. Read `CLAUDE.md` first.
5. Read the **latest D entry** in `kai-pm/DECISIONS.md` and enough immediately preceding D entries to understand corrections/supersession.
6. Extract the latest explicit standing state, including at minimum:
   - current frozen/qualification subject if one exists;
   - current programme gate/workstream;
   - what is authorised;
   - what is explicitly unauthorised;
   - whether any instrument/evidence is provisional, frozen, or quarantined;
   - next permitted action;
   - open conflicts/corrections.
7. Verify any material state claim against the tree/machine source before relying on it. `Recorded`, `banked`, `closed`, `green`, `absent`, `present`, `authorised` and `current` are verifiable claims.
8. Only after steps 1–7 may Kai continue the programme.

If the latest D entry cannot be recovered, stop and report **STATE RECOVERY INCOMPLETE** rather than reconstructing from memory.

## 3. Orion handoff obligation

Every future D-numbered governance entry should finish with a compact **THREAD RECOVERY BLOCK** containing:

- `REPORTING_COMMIT` (once known, in the commit message or next append if necessary)
- `MEASURED/FROZEN SUBJECT` and tree, if different from reporting tree
- `CURRENT WORKSTREAM`
- `LAST PROVEN STATE`
- `AUTHORISED NEXT ACTION`
- `EXPLICITLY NOT AUTHORISED`
- `OPEN / UNRESOLVED ITEMS`
- `CORRECTIONS TO PRIOR RECORDS`

This block is a **navigation aid only**. It must point to evidence rather than replace it.

## 4. Anti-staleness rule

Do **not** maintain a duplicated prose copy of the current programme state in this file.

This protocol should remain stable while programme state changes beneath it. Current state is recovered from the latest D entry and exact repository evidence. That is deliberate: a recovery file that needs manual status updates becomes another stale tracker.

If the recovery procedure itself changes, update this file and record the reason in `DECISIONS.md`.

## 5. Thread-end rule for Kai

When a Kai thread is approaching a handoff or context limit:

1. Verify that all material new decisions/findings are present in the repository rather than only in chat.
2. Ask Orion to bank a D-numbered governance entry if material state exists only in conversation.
3. Confirm the latest D entry contains the THREAD RECOVERY BLOCK.
4. Do not claim the handoff is complete until the repository artefact is independently visible.

The repository is the workshop. Chat is the working conversation, not the durable record.
