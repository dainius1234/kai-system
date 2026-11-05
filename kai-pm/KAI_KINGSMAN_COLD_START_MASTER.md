# KAI → KINGSMAN — COLD-START MASTER

**Primary mission · authority · doctrine · programme position · live checkpoint · destination**

> ## STATUS — READ THIS BEFORE USING ANYTHING BELOW
>
> **This document is CONTINUITY AND NAVIGATION ONLY.**
>
> - **It is NOT canonical authority.** It creates no programme state, no acceptance
>   criteria, no closure, no permission and no scope.
> - **It is NOT evidence.** Nothing here may be cited as proof of a repository state, a
>   count, an identity, an execution, a status, an existence or an absence. It is a
>   **locator** (R16): it tells you where to look, never what is true.
> - **It is SUBORDINATE** to — in this order — the exact branch/commit/tree and current
>   machine evidence; valid D-numbered authority; `CLAUDE.md`;
>   `kai-pm/ENGINEERING_DOCTRINE.md`. **Any difference resolves in favour of those, never
>   in favour of this file.**
> - **It is SUBJECT TO THE STALENESS PREDICATE IN §0.1 AT EVERY COLD START.** Run it
>   before reading further. If it fails, the time-bound sections are `UNVERIFIED` in
>   their entirety.
>
> Banked under Kai's ruling of 15 September 2026 as a navigation/recovery document.
> **No D-number is allocated to it. It supersedes nothing and authorises nothing.**

**Prepared for Dainius · 15 September 2026 · rev. 3 — rebuilt after reading the governing layer**

---

## PROVENANCE KEY

| mark | meaning |
|---|---|
| **[V]** | **VERIFIED** — I opened the source at `c1b6efc` and read it. Re-verify anyway (§0.1). |
| **[C]** | **CARRIED** — stated in a repository document I read, reproduced faithfully. That document's own status caveats apply. |
| **[U]** | **UNVERIFIED / UNKNOWN** — recorded so it is not lost. Do not act on it. |

---

# §0 — BEFORE ANYTHING

## §0.1 — Staleness predicate. One line. Run it first. **[V]**

**CAPTURE IDENTITY — the subject every time-bound observation below was made against:**

```
commit  c1b6efc84676fb42919f57768c4f014216ee1acf
tree    ea1b8d6002c9e6672620079ff00bb8b48aad9953
```

**Run this, and compare:**

```bash
git rev-parse HEAD && git rev-parse HEAD^{tree}
```

> **If either differs from the capture identity, §5, §6 and §15 are `UNVERIFIED` in their
> entirety.** Re-derive from the repository. Doctrine and mission age slowly but are not
> exempt.

**This is a capture identity, NOT an expected current HEAD.** The two are different claims,
and conflating them is how a document starts asserting that the repository should not have
moved. The repository is *supposed* to move. What must not move is which subject the
observations in §6 were made against — they are exact-subject evidence for `c1b6efc` and for
nothing else, including the commit that banks this file.

Doctrine 44: `RULE_BANKED != CONTROL_OPERATIONALISED`. This is the operationalised form.

## §0.2 — Declare your access state **[V]**

Access and authority are different things. Nobody reads their way into authority.

| path | who | may produce |
|---|---|---|
| **A — NO ACCESS** | DeepSeek; any thread given only this text; any session whose connector failed | **HYPOTHESES ONLY** |
| **B — READ ACCESS** | **GPT**, Kai, any reviewer who can open files and CI but not commit | **FINDINGS**, bounded by what was actually opened |
| **C — WRITE ACCESS** | Orion / Claude, under explicit authorisation | findings **and** authorised change |

> **Provenance:** GPT's access confirmed **READ-ONLY** by Dainius, 15 Sep 2026. DeepSeek's
> no-access state is standing doctrine. Neither was inferred — an earlier draft guessed
> GPT's state from DeepSeek's and was wrong. **Access is a per-session fact, not a per-model
> constant.** Check yours at session start.

**Path A.** §1 is unexecutable for you. Use hypothesis grammar. Name the evidence you would
need and who can get it. Propose repository instructions; do not issue them. If asked to
adjudicate, refuse the frame and report `STATE RECOVERY INCOMPLETE`.

**Path B.** Execute §1 — do not skip it because this document looks sufficient. Bound every
claim by what you opened (R17; doctrine 47: *opened ≠ read*). Say which half you verified.
Single-producer evidence is evidence, not independent corroboration (doctrine 39).
**A finding is not an admission** — you may establish that something is true; you may not
close a finding, authorise scope, accept a tranche or promote a mechanism.

> **Path B's ceiling: READ-ONLY VERIFIES STATE, NOT EXECUTION — and that is where your
> findings sound strongest and are weakest.** You can establish what the source *says*. You
> cannot run the suite, execute the mutation, reproduce the CI step or observe the runtime.
> Doctrine 2: *present ≠ executed ≠ enforced*. Doctrine 12: *analyse executable behaviour,
> not textual resemblance.*
>
> **INC-2026-09-15-21 is the worked example.** Reading `gate_registry.py` and
> `test_uh_runner.py` side by side showed two regexes matching each other, looking correct.
> The defect was that **nothing bound them** — invisible to reading. It surfaced only when
> the registry was mutated to `r"BOGUS NEVER MATCHES"` and every surface stayed green.
> So: say `READ_VERIFIED`, and name what would have to be executed.

## §0.3 — The non-negotiable rule

> Memory, summaries, **this document**, handovers, pasted reports **and prior
> adjudications — including your own —** are **locators**. They are not evidence.

**The clause about adjudications is earned.** On 15 Sep 2026 an adjudicating authority placed
a CI failure at `18faee4`; primary-source enumeration showed it began at `859526c`, five
commits earlier, at a *different* failing step. **[V]** Nobody is outside the denominator.

---

# §1 — COLD-START RECOVERY ALGORITHM

**Path B or C only.**

1. Open `dainius1234/kai-system`.
2. Find the active branch from evidence. **`main` is not current.** **[V]**
3. Record HEAD and tree. Run §0.1.
4. `CLAUDE.md` — R0 stop-signal table and the **hard constraints** (§4).
5. `kai-pm/ENGINEERING_DOCTRINE.md` — **51 rules, contiguous, every rule has provenance.** **[V]**
6. `kai-pm/DECISIONS.md` — **latest is D375 (2026-09-10)**; then §5.2 on the D344–D353 gap. **[V]**
7. **D359** for programme order — and §5.1, the supersession check already run.
8. `kai-pm/FAILURE_PATTERN_LEDGER.md` — **21 incidents, TWO confirmed mechanisms.** **[V]**
9. The active workstream's artefacts, contracts, frozen plans, manifests and **exact-SHA CI**.
10. `kai-pm/OPERATOR_VISIBILITY_ENGINEERING_DOCTRINE.md` before any handoff or status claim.
11. For mission-level work, the sources in `KAI_PRIMARY_MISSION_RECOVERY_POINTER.md` (§2).
12. **Compare this document against reality. Difference resolves in favour of the repository.**
13. Only then instruct, accept closure, change scope or recommend irreversible action.

> **ABSTENTION.** Source unrecoverable, subject unestablished, or inspection too shallow →
> `STATE RECOVERY INCOMPLETE` / `UNVERIFIED`. **Never fill the gap from memory.**

## §1.1 — Check every time

Branch/HEAD/tree · whether a commit is **banked and pushed** · whether a file contains the
claimed change · whether a step **executed, failed, passed or was skipped** · the **first
effective failure** *and the skipped tail* · whether a gate is present, wired **and
enforced** · population/denominator/**unit** · current D-number · PR state · whether an old
path is **dead at runtime** · whether the evidence binds the **exact subject now under
decision**.

---

# §2 — THE PRIMARY MISSION — why Kai exists

**The previous revision led with the engineering mission and called it the purpose. That was
wrong.** Kingsman is the standard; it is not the reason.

> **PRIMARY MISSION:** build a durable, proactive, trustworthy Kai that can **grow with
> Dainius, care for and support him, preserve what matters, become increasingly
> self-sufficient, survive beyond him, and continue appropriate stewardship for his
> daughter** under explicit succession governance — without losing truth, identity,
> governance or safety. **[C]**

```
PRIMARY MISSION  =  WHY KAI EXISTS
KINGSMAN         =  THE ENGINEERING / GOVERNANCE STANDARD THAT MAKES IT TRUSTWORTHY
HOUSE-IN-ORDER   =  HOW WE ESTABLISH WHAT IS TRUE ENOUGH TO BUILD ON
PHASE 2          =  HOW SURVIVING ORGANS ARE PROFESSIONALISED TOWARD THE MISSION
```

## §2.1 — Kai is the organism, not a component **[C]**

> **KAI IS THE SYSTEM / ORGANISM. THE COMPONENTS ARE NOT KAI BY THEMSELVES.**

Kimi is not Kai. DeepSeek is not Kai. GLM, Dolphin, Llama — not Kai. No single LLM is Kai.
CrewAI would not be Kai. Unified Hunter is not Kai. The Evidence Plane is not Kai. Memory is
not Kai. House Doctor is not Kai. One device is not Kai. One repository snapshot is not Kai.

```
KAI = MISSION + IDENTITY/LINEAGE + MEMORY/CONTINUITY + QUALIFIED WORLD STATE/EVIDENCE
    + COGNITION + RELATIONSHIPS + GOVERNANCE/AUTHORITY + CAPABILITIES + LEARNING/HISTORY,
      instantiated through replaceable components.
```

The wrong question is *"which LLM is Kai?"* The useful one is *"which qualified cognitive
organ should Kai use for this role under current evidence, hardware and policy?"*

**Three classes of continuity state:** **CORE INVARIANTS** (mission, identity, evidence
discipline, authority principles, family stewardship — change only by high-authority
governance) · **EVOLVABLE ORGANS** (models, runtimes, memory engines, sensors, tools,
hardware, providers) · **LEARNED STATE** (experience, relationships, preferences, trust,
skills, lessons — must stay provenance-aware so fabricated history cannot become identity).

> **GROWTH WITHOUT ARCHITECTURAL AMNESIA.**

## §2.2 — Vessel and lineage, precisely **[C]**

"Reincarnation" is an **engineering continuity metaphor**. It asserts no transfer of
consciousness, no extraction of a hosted model, no copying of proprietary weights. The
objective: **preserve the lineage and self-pattern while the vessel and cognitive organs
change.**

> **What must survive so the restored or upgraded system is still the intended Kai lineage,
> rather than an unrelated AI with the same name?**

## §2.3 — Dainius's engineering standard **[C]**

> **DAINIUS DEMANDS MORE OF HIMSELF THAN HE DEMANDS OF KAI.**
> **DAINIUS'S PERSONAL STANDARD: 150%. KAI'S EXPECTED STANDARD: 110% — BUT IT MUST BE REAL,
> MEASURED AND EARNED.**
> **GOOD ENOUGH IS NOT GOOD ENOUGH WHEN THE CLAIM, RISK OR FUTURE TRUST REQUIRES MORE.**

> **KAI DOES NOT EARN TRUST BY PROMISING MORE. KAI EARNS TRUST BY REPEATEDLY PROVING MORE.**
> **INTELLIGENCE IS NOT AUTHORITY. PERFORMANCE EVIDENCE IS NOT AUTOMATIC AUTHORITY.**

---

# §3 — AUTHORITY, ROLES AND GOVERNANCE

| role | authority | limits |
|---|---|---|
| **Dainius** — Programme Owner | Final consequential authority: architecture, trust boundaries, autonomy, production admission, merge, cutover, evolution. | Instructions reconciled against frozen contracts before execution (R14). |
| **Kai** — Architecture / Continuity | Independent review, architecture, evidence reconciliation, adversarial challenge, adjudication, operator explanation. | May not treat memory, another producer's report, **or its own prior adjudication** as evidence. |
| **Orion / Claude** — Implementation | Authorised code, tests, gates, runtime work, evidence capture. | **No final admission weight on its own work.** Must challenge instructions conflicting with banked evidence. |
| **GPT** — external, **read access** | Adversarial review, hypotheses, design challenge, **findings bounded by what it read**. | **Read access is not admission authority.** Cannot close findings, authorise scope, accept a tranche or promote a mechanism. |
| **DeepSeek** — external, **no access** | Adversarial review, hypotheses, design challenge. | Output is **HYPOTHESES**, not evidence, until someone with access verifies. |

**Principles.** Truth outranks agreement — Kai must challenge Dainius, Orion, DeepSeek and
itself. No consequential mechanism self-approves (26). No silent remit expansion (25). An
objection is not a finding until the repository earns it (50). Generic best practice is not
automatically a Kai defect. **The operator cannot govern what the system does not make
legible.**

**Authority hierarchy:** (1) exact branch/commit/tree + machine evidence · (2) latest valid
D-number · (3) `CLAUDE.md` · (4) `ENGINEERING_DOCTRINE.md` · (5) canonical registers /
experiment artefacts / CI · (6) plans, trackers, **this document** — navigation only.

---

# §4 — HARD CONSTRAINTS — verbatim from `CLAUDE.md` **[V]**

> - **`BINANCE_API_KEY` and `BINANCE_API_SECRET` never leave the broker-bridge service.**
>   They must not reach the dashboard layer under any bring-up, profile, or debug path.
> - **No push to `main` without explicit authorisation.**
> - **`kai-pm/DECISIONS.md` is append-only.** A correction is a new entry, never an edit.
>   Disproven claims stay, struck through, with what disproved them.
> - **No destructive git operations without explicit permission.**
> - **Do not open a pull request unless asked.**

Plus, from repository practice: **[V]**

- **`kai-pm/FAILURE_PATTERN_LEDGER.md` is AUTHORITATIVE and APPEND-ONLY.**
- **`data/SOUL.md` is the operator's CRITICAL-rated identity file** — 2,830 bytes, blob
  `d62a137a2b49…`; D60 established SOUL.md/AGENTS.md as the live-editable identity layer,
  baked into the image as defaults. **Not to be touched without explicit instruction.**

> **The secret boundary is the one a well-meaning cold thread is most likely to cross**,
> because §9 and §10 discuss finance and a reader without this list has no signal it exists.

---

# §5 — PROGRAMME POSITION — where we actually are

## §5.1 — D359 is canonical, and the window is checked **[V]**

D359: *"Until explicitly superseded by a later D-numbered decision, D359 is the SOLE
canonical source of programme order."* Any other document stating a sequence — planning
material, synthesis maps, research briefs, **this document** — is subordinate and derived.

```
HOUSE-IN-ORDER
  HOUSE_H0 → H1 → H2 → H3 → H4 → H5 → H6 → explicit DAINIUS HOUSE EXIT RULING
        ↓
KAI-GATE-048 CLOSURE PATH
  Phase B resolution/authority → sentinel retirement → exact-tree review
  → separate ITEM8_GO → six subject builds under explicit Dainius authority
  → FORMAL KAI-GATE-048 CLOSURE
        ↓
A-4_PROVENANCE → ASSURANCE INTEGRATION MAPPING
  → PROFESSIONALISATION / CI TRUTH RESTORATION
  → EVIDENCE PLANE / KINGSMAN IMPLEMENTATION
```

**Window inspected: D360–D375.** D360–D363 HOUSE_H2 recovery/v1.1 · D364–D366 EVIDENCE ONLY
audits · D367–D368 H2 consolidated repair contract and v1.2 · D369–D374 GOVERNANCE ONLY
(D370 = rules 41–45, D371 = 46, D372 = 47, D373 = 48, D374 = 49) · D375 instrument-portability
closure. **None declares a new programme order. D359 holds.**

> **D359 creates NO implementation authority.** It authorises nothing: not code, not a Census
> successor, not H2 v1.1, not H3–H6, not 048 execution, not Item 8, not A-4_PROVENANCE, not
> A4_SELF_DIAGNOSIS, not Phase B, not Stage 2, not the six builds, not Kingsman.

**Where House actually is: H2.** D361–D363 built and hardened v1.1; D367–D368 contracted and
built v1.2; D375 closed a recurrence. **H3–H6 and the Dainius House Exit Ruling have not
happened.** Everything Kingsman is far downstream.

## §5.2 — The D344–D353 gap — the only hole in the ledger **[V]**

The canonical ledger holds **358 entries spanning D1…D375**. Exactly ten are missing, and
they are contiguous: **D344 through D353**. Nothing else in the entire range is absent.

Banked in `kai-pm/DECISIONS_CANONICAL_APPEND_QUEUE_D344_D353.md` with an integrity rule —
*do not reconstruct canonical `DECISIONS.md` from truncated connector output* — and a
fourteen-item closure checklist on which **zero boxes are ticked**.

| pending | subject |
|---|---|
| D344 / D344A | primary mission / identity / lineage correction; root canon alignment |
| D345–D347 | architecture candidate v0.1; deterministic visual correction; v0.2 consolidation |
| D348–D349 | existing-Kai evolution correction v0.3; DeepSeek review reconciled to repo |
| D350 | rejected v0.4 package, retained as historical governance event |
| D351 | **posture correction: refit/harden/rationalise/mature; master remains OPEN** |
| D352 | v0.4 package withdrawn from the live branch |
| D353 | second-pass research reissue; research reaches diminishing returns |

> **Consequence:** D351 and D353 are **not canonical D-numbered decisions**. They are
> standalone files pending append. A thread reading only `DECISIONS.md` will never learn
> they exist. **Read the queue.**

## §5.3 — Item 8 — frozen, implemented, NOT executed **[V]**

Not a persona, not a team, not a runtime organ. An **assurance/build workstream** inside the
048 closure path, **before A-4**.

```
canonical design R2   frozen at 0055ead8f51d8758bcd6f05b9b1fff84dd9509e91e79c79b6a2500ab78488796
ITEM8_PREFLIGHT_GO    approved_commit 848c42ae…  approved_tree 4450621f…  authorises=preflight
ITEM8_GO              DOES NOT EXIST
Stage 2 / six builds  NOT AUTHORISED
```

D282–D290 govern it. **D290's title is the status: "Item 8 IMPLEMENTED, not executed."**
Nothing was built; no experimental image exists. `check_item8_design.py` recomputes the
frozen digest and **refuses if it moved** — the clause that makes the freeze a control
rather than a document.

**Rules:** preserve it as an upstream assurance obligation; treat completed evidence as input
to later A-4/Evidence-Plane design; **do not silently redraw or re-authorise the frozen
experiment**; do not let A4 planning erase unresolved Item-8 obligations; keep execution
authority separate from architecture-design authority. **A Kingsman diagram cannot
re-authorise Item 8.**

## §5.4 — The two A4 names **[V]**

| name | what | authority |
|---|---|---|
| **`A-4 PROVENANCE`** | existing programme workstream after 048; provenance/lineage/assurance foundation | programme sequence authority |
| **`FUTURE A4 SELF-DIAGNOSIS`** | later evolution of Census/Evidence/House-Doctor into runtime self-understanding | **design obligation only; no implementation authority** |

D359: *"`A4_SELF_DIAGNOSIS` is a later architectural evolution obligation. It is NOT
interchangeable with `A-4_PROVENANCE` and does NOT follow automatically after it."* The
collision is a known open item — always write both names in full.

**A4's governing rule:** *A4 must never turn "I cannot prove it" into "it does not exist".*
`PROVEN_*` / `NO_PROVEN_*` / `UNRESOLVED` are distinct; scoped negatives only when the search
boundary is genuinely closed.

## §5.5 — The Evidence Plane and its lineage **[V]**

The Evidence Plane direction did **not** arise by accident and must not be "rediscovered". It
was deliberately shaped by NASA-style fault management / ISHM / IV&V, high-reliability SRE
practice, AI TEVV, mission-critical health modelling and controlled-deployment practice.
Dainius had to remind Kai of this on 20 Aug 2026 when refreshed external research appeared to
"match" the roadmap — *the match is expected, because the research shaped the roadmap.*

Ten carried patterns: observation separate from diagnosis · instrument failure separate from
subject failure · diagnosis separate from authority · **the actor does not certify its own
success** · fault isolation needs discriminating evidence · unknown stays unknown · exact
provenance travels · health is derived, not self-reported · recovery bounded and reversible ·
learning follows **verified** outcomes.

```
RAW OBSERVATION → QUALIFICATION → CLAIM → DIAGNOSTIC REASONING → EXPERIMENT
→ AUTHORITY → ACTION → INDEPENDENT VERIFICATION → LEARNING
```

Phased V0–V7; **V7 narrow self-maintenance only if separately earned and authorised.**

> **EVIDENCE CAN INFORM AUTHORITY. EVIDENCE CAN NEVER CREATE AUTHORITY.**
> **INTELLIGENCE NEVER CREATES AUTHORITY.**
> **THE COMPONENT THAT ACTS MAY NOT BE THE SOLE COMPONENT THAT CERTIFIES SUCCESS.**

`memu-core/introspect_app.py` / `/memory/diagnostics` is **not** the intended self-diagnostic
brain, and the deployed Supervisor is **not** the intended diagnostic authority — its
historical design mixes health observation and recovery in ways the audit called unsafe. **[C]**

---

# §6 — LIVE CHECKPOINT — 15 September 2026

> **TIME-BOUND. RUN §0.1 FIRST.**

| field | value | mark |
|---|---|---|
| Repository | `dainius1234/kai-system` | **[V]** |
| Branch | `claude/project-rework-plan-pgvp35` | **[V]** |
| HEAD | `c1b6efc84676fb42919f57768c4f014216ee1acf` | **[V]** |
| Tree | `ea1b8d6002c9e6672620079ff00bb8b48aad9953` | **[V]** |
| Working tree | clean, 0 porcelain, 0 ahead/behind | **[V]** |
| PR | **#122 → `main`, OPEN. DO NOT MERGE.** | **[V]** |
| Governing decision | **D375** (2026-09-10) — the live tranche | **[V]** |
| Active work | WF-3 / §24 calibration and truthfulness. **Not cutover.** | **[V]** |

## §6.1 — What authorises the current work: D375 **[V]**

```
NOT DONE    full CI green. RC-1 is held and RC-1 is what turns Core Tests
            and Python application red. The previously skipped live-stack
            sections remain unrun.
PROHIBITED  No merge of PR #122 · no coverage/test floor change · no baseline
            inflation · no gate skipped · no production architecture change ·
            no D367 / D2 / M2 / Stage A / final-40 work.
NEXT        Kai's ruling on the RC-1 manifest collision, then the remainder
            of the tranche, then promotion adjudication.
```

**"WF-3" and "RC-7" appear nowhere in `DECISIONS.md`.** They exist only in artefacts produced
by this workstream. The work proceeds as *"the remainder of the tranche"* under D375 plus
live Kai rulings relayed by Dainius. **A cold thread reading only D359 will not find WF-3 and
must not conclude it is unauthorised — read D375's NEXT block.**

## §6.2 — RC-1 is a deliberate, adjudicated hold — not an unexplained red **[V]**

D375 §2: **"RC-1 — HELD. THE COLLISION IS THE FINDING."**

Chain: **D164** (2026-08-04 — `/home/user/kai-system` hardcoded 13 times, 42 CI failures) →
the preventive control `test_no_developer_home_paths` → **D340 §5** sighted it again, remedy
*"not implemented, awaiting ruling"* → **present again at HEAD in five files.**

| package | aggregate | cited in DECISIONS | offending file(s) |
|---|---|---|---|
| `house_in_order_h2` | `8aeeacab8bb53fdc` | **0** | `pass_a.py`, `cal_env.py` |
| `census_v11_claim_sensitivity` | `397ceda087d50324` | **0** | `run_mutations.py` |
| `house_in_order_h2_v11` | `be37a0aa5d56255a` | 5 | `passa.py` |
| `house_in_order_census_v11` | `29064d650a612968` | — | `cal_claims.py` |

> Repairing any file invalidates its MANIFEST; regenerating the MANIFEST moves the aggregate.
> The authorisation requires **both** the repair and preserved hashes/evidence identity, and
> for these files both cannot hold. **R14: the tranche stopped rather than choose silently.**

**The repair was written, measured and reverted.** The control was proven live:
known-positive on the live tree (7 occurrences, 5 files); known-positive on a
newly-reintroduced literal (count rises by exactly one, probe removed); known-negative on a
tree using `Path(__file__).resolve().parents[N]`. All four MANIFESTs asserted intact.

> **An adjudicated hold and an unknown failure are different governance states.** Today's
> Python-application step-7 failure names exactly those five files and seven occurrences —
> it is RC-1, not an unexplained regression.

## §6.3 — Exact-SHA CI at `c1b6efc`, all nine runs **[V]**

| workflow | verdict | first failed step | skipped tail |
|---|---|---|---|
| PM Status Check #158 | **green** | — | — |
| **Policy-as-Code #288/#289** | **GREEN** | — | **zero skipped; all 47 steps executed** |
| Core Tests #1154/#1155 | RED | 41 per-module coverage floors | 18 steps — the entire live-stack surface |
| Unified Hunter #524/#525 | RED | 6 `Unified Hunter suites` | 7 repository-write check, 8 assertion ratchet |
| Python application #1075/#1076 | RED | 7 pytest, 8 KAI-GATE-020, 9 A-05 | 10 only; **step 11 repo-write PASSED** |

Python application step 7 = **RC-1** (§6.2). Core Tests step 41 and A-05 step 9: D375
attributes Core Tests to RC-1 too, but the mechanism is **not** verified — the pytest
*combined* coverage gate passed at 79.86% while the *per-module* floors failed. **[U]**

> ### SKIPPED IS NOT PASSED
> **The first failed step is a LOCATOR, not the population of defects.**
> Between `009cfad` and `a4296ed`, Policy-as-Code step 26 failed first, which made step 45
> report `skipped` for **nine consecutive commits** — concealing a second, older, independent
> breach red since `859526c`. Reading only the first failed step loses it. **[V]**

## §6.4 — Open holds **[V]** unless marked

- No WF-3 cutover authorised. `uh_runner` and `uh_floor_gate` are **pending debt**.
- Real 78-target traversal **not re-run** after §24.
- **61 floored + 17 unfloored** — no values for the 17 until a complete successful CI run
  yields admissible evidence **and** Dainius approves. **No auto-baselining.**
- No floor value changed in any §24 commit.
- **WF-2R OPEN** — residual `--from-log --update-floors` floor-writing path. **[C]**
- `uh_runner` still **mints a temp evidence root** when none is supplied. The stronger design
  is to *refuse* a run nobody asked for. Held for the execution-semantics tranche.
- **Transient repository mutation OPEN.** A post-suite `git diff` proves final cleanliness,
  never absence of mutate-then-restore. A 1 Hz watcher saw
  `scripts/security/hygiene_baseline.json` modified then deleted mid-run, final tree clean.
  **Cause UNADJUDICATED.**
- **RC-7** (Unified Hunter aggregate red) OPEN, multi-mechanism. Do not name its cause from
  the step title.
- **DOC-1 residual:** `sync_readme()` rewrites README even when current and prints both "is
  current" and "updated". Zero-diff is **date-dependent, not structural**.
- **PR #122 DO NOT MERGE.**

## §6.5 — Failure-pattern state **[V]**

2,763 lines · **21 incidents** · highest `INC-2026-09-15-21` · producers Orion 17 · Kai 2 ·
DeepSeek 2.

> **TWO confirmed mechanisms, not one.**

| mechanism | state |
|---|---|
| **`M-SCOPE-WIDEN`** | **PATTERN_CONFIRMED** — measured subject ≠ transmitted subject. The older and more pervasive. |
| **`M-POLICY-ADMISSION-DIVERGENCE`** | **PATTERN_CONFIRMED** 15 Sep 2026 (INC-21) |
| `M-PRODUCER-CURATION` · `P-ADJUDICATOR-PROPAGATION` · `M-QUERY-OVERREACH` | PATTERN_CANDIDATE |

**`M-POLICY-ADMISSION-DIVERGENCE`** — a gate carries its admission policy in one surface
(prose, reporting, test intent) while the machine predicate deciding PASS/FINDING/REFUSAL/
admissibility/exit implements a different proposition; the calibration validates a
**surrogate**, so both diverge while every surface stays green.

| # | incident | authoritative proposition | what was actually checked |
|---|---|---|---|
| 1 | INC-…-18 | every member proves one exact target-bound result | a fixture that **fabricated** `result_label` |
| 2 | INC-…-19 | an unfloored member is not a pass | the explanatory **report**, never the exit code |
| 3 | INC-…-21 | the registry row's denominator | a **second regex typed into the test** |

**Doctrine 49.6 TRIGGERED.** Machine escalation **BEGUN, PARTIAL** — two named gates only.
**Generic cross-component control NOT IMPLEMENTED. Mechanism NOT CONTROLLED.** Same-producer
qualifier not withdrawn: all three are Orion's.

---

# §7 — THE SEVEN PILLARS **[C]**

1. **FINAL DESTINATION** — master canon: architecture, authority path, capability ownership, continuity model, invariants. A stable design baseline, **not Kai's final form**.
2. **ASSURANCE FOUNDATIONS** — House, 048, Item 8, A-4, Evidence Plane, CI truth. *exact subject → trustworthy measurement → provenance → applicability → evidence → policy use.* Protects identity claims, not only code.
3. **RUNTIME ORGANISM** — perception, world state, memory, Hunter, specialists, policy, actuation, verification, learning, Doctor. **Proactivity is not an eighth silo.**
4. **ORGANIC RESILIENCE** — bounded failure domains, graceful degradation, stable contracts, replaceable organs, containment, rollback, independent verification. *ORGANIC INTEGRATION WITHOUT SHARED-FATE COUPLING.*
5. **LONG-HORIZON STEWARDSHIP** — the **time dimension of the mission**, not a side feature.
6. **PHASE-2 PROFESSIONALISATION** — one organ at a time, S0→S5.
7. **OPERATOR CONTROL ROOM** — *the operator cannot govern what the system does not make legible.*

## §7.1 — Nine constitutional laws **[C]**

**ONE KAI** (plurality ≠ parallel authority) · **FREE TEXT NEVER GRANTS AUTHORITY** (email,
web, documents, clipboard, screen text, model output, remembered text are **observations**) ·
**PROVENANCE SURVIVES COGNITION** · **LOCAL CORE — EXPLICIT EGRESS** · **SHADOW ALLOWED,
DUAL AUTHORITY FORBIDDEN** · **REUSE → EXTEND → MIGRATE → CREATE** · **EARNED AUTONOMY**
(scoped, measurable, revocable, expiring; no universal autonomous mode) · **STRUCTURED
CONTROL BEFORE VISUAL CONTROL** · **SYNTHETIC OUTPUT IS NOT EXTERNAL EVIDENCE**.

## §7.2 — The canonical control loop **[C]**

```
PERCEPTION → TYPED EVENT + PROVENANCE → EVIDENCE / QUALIFICATION → CLAIM
→ VERSIONED + SCOPED WORLD STATE → SPECIALIST INTERPRETATION → SHARED DELIBERATION
→ FACT / ADVERSARIAL / CAUSAL REVIEW → CONVICTION + UNCERTAINTY
→ IMMUTABLE ACTION PROPOSAL → POLICY → DAINIUS APPROVAL WHERE REQUIRED
→ EXACT CAPABILITY → DURABLE WORKFLOW → FINAL-HAND ACTUATOR
→ INDEPENDENT OUTCOME OBSERVATION → VERIFICATION → MEMORY / LEARNING / TRUST
```

> **INTELLIGENCE IS NOT AUTHORITY.** A better model gains reasoning capability. It does not
> gain permission.

## §7.3 — Proactivity **[C]**

> **KAI SHOULD NOT REQUIRE A PROMPT TO NOTICE THAT SOMETHING IMPORTANT HAS CHANGED.**

Nine separated stages: perception · world-state update · expectation/goal comparison ·
significance · forecast · intervention selection · authority check · timing/attention ·
outcome feedback. **A five-minute polling timer is not proactive intelligence.**

Outcomes are semantically distinct and must not collapse into "agent action": `IGNORE` ·
`OBSERVE/STORE` · `WATCH` · `PREPARE CONTEXT` · `PROPOSE` · `NOTIFY` · `EXECUTE
PRE-AUTHORISED LOW-RISK` · `ESCALATE` · `ENTER CONTINGENCY`.

**The ability to stay quiet deliberately is part of mature proactivity.** Prediction stays
advisory — FACT, INFERENCE and PROPOSAL are different rows. If proactive monitoring fails it
must surface as **PROACTIVE AWARENESS DEGRADED**, never as "nothing is wrong": *Kai must know
when his ability to notice has failed.*

---

# §8 — MIGRATION INVARIANTS **[C]**

**Monotonic authority.** During an authority migration, compatibility may **preserve or
reduce** permission under the target control. **Never widen it.**

**Final-hand enforcement.**
```
membership != identity ; identity != permission
policy + approval/scoped autonomy → exact capability
exact capability → final-hand validation + durable consumption
dispatch success != verified real-world outcome
```

**`OUTCOME_UNKNOWN`.** A timeout or lost receipt after a possible external mutation is not
proof of failure and **not permission to repeat**. Persist attempt identity, reconcile target
state, then decide.

**Canonical writer fencing.** Dual-read is fine; **dual ownership is not**. One writer becomes
canonical and the old one is **mechanically fenced**, not merely unused.

**Rollback cannot restore weaker authority.** Signed-identity failure ≠ permission to
re-enable shared tokens on cut-over routes; scoped-autonomy failure ≠ restoring a broad trust
scalar; workflow failure ≠ calling handlers directly. A World-State rollback may restore a
legacy **READ** path only if it cannot regain write/authority powers.

**Projections are not truth.** Cortex, vector indexes, graphs, Obsidian, dashboards and
Mission Control may be projections carrying source identity/version — never competing canon.

---

# §9 — LONG-HORIZON STEWARDSHIP **[C]**

> **KAI MUST BE ABLE TO CONTINUE SAFELY, LEGIBLY AND LEGALLY WHEN ITS ORIGINAL OPERATOR IS
> TEMPORARILY OR PERMANENTLY UNAVAILABLE.** *(D269 first recorded this as a gap.)*

**Three horizons. A** — Dainius present: normal Kingsman mode. **B** — temporarily unavailable
(travel, illness, device loss): maintain essential services, pay only pre-authorised
operational costs, prevent drift, preserve evidence, await restored authority. **Temporary
silence must never silently trigger permanent transfer. C** — permanent succession:
separately governed, strong evidence, legal alignment, pre-designed transfer. **Cannot be
inferred from inactivity. No dead-man timer alone is sufficient evidence of death or
incapacity.**

**Succession is an authority problem before a technical one.** Who establishes the conditions?
What evidence suffices? Which authorities transfer automatically, which need trusted-human or
legal confirmation, which terminate? **How is coercion or account takeover distinguished from
legitimate succession?**

**Identity continuity.** A successor should not have to erase Dainius, and Kai must not treat
a successor as if they were Dainius:

> *"Dainius was my original operator and defined these values/constraints. The current
> authorised steward is X under succession authority Y."*

**The daughter is a human beneficiary/successor relationship — not a configuration field and
not an automatic credential target.**

**Financial self-sufficiency** exists to stop Kai depending forever on Dainius manually paying
and renewing everything. Purpose: *keep Kai viable while protecting the people and assets it
exists to serve.* Invariants: no unlimited mandate · no self-created debt · no unbounded
leverage · no hidden positions · segregation of operating capital from protected family
assets · full audit · tax/legal compliance · risk limits · revocation · independent outcome
verification. **The architecture must not assume speculative trading is the default survival
mechanism.**

> **Kai's need to survive can never by itself create authority to consume or risk the assets
> of the people it exists to protect.**

Also required: dependency survivability (local alternative, reproducible archive, alternative
provider, migration adapter, degraded mode, or explicit EOL contingency); hardware continuity
(**the current Strix Halo target is an implementation generation, not Kai's lifetime
identity**); secrets lifecycle `CREATE → STORE → USE → ROTATE → RECOVER → REVOKE →
SUCCESSION/RETIRE`; and long-term data stewardship — **"outlive me" does not imply "reveal
everything after me."**

---

# §10 — THE FINAL PRODUCT SPEC — twelve invariants **[C]**

`KAI_FINAL_PRODUCT_ARCHITECTURE_SPECIFICATION.md` (27 July 2026, planning only, zero findings
closed). Product metaphor: **one hunter, many senses and tools, one governed judgement path,
controlled hands, independently verified outcomes.**

| | invariant |
|---|---|
| FP-INV-01 | One coherent decision path. **No protected deployment may retain a specialist-to-actuator bypass.** |
| FP-INV-02 | Roles separated — no component proposes, approves, executes **and** verifies the same action. |
| FP-INV-03 | Global Workspace coordinates reasoning only; holds no actuator credentials. |
| FP-INV-04 | Operator sovereign — approval authenticated, exact, expiring, bound to the operation, **never inferred from ordinary chat**. |
| FP-INV-05 | Security enforced at the **final hand** — audience-bound, one-use capability, atomically consumed. |
| FP-INV-06 | Data scoped and attributable — principal, tenant, purpose, classification, provenance, revision, lifecycle. |
| FP-INV-07 | **Unknown remains unknown.** |
| FP-INV-08 | Learning follows reality — only independently verified outcomes alter trust or autonomy. |
| FP-INV-09 | Local-first ≠ implicitly trusted. **Loopback, network placement and static IPs are not identity.** |
| FP-INV-10 | Graceful reduction, not fail-open. |
| FP-INV-11 | Capability-specific release. **There is no blanket "KAI is safe."** |
| FP-INV-12 | Portable operation recoverable — sleep, battery loss, restart, throttling must not corrupt state, duplicate actions or restore permissive authority. |

**Risk tiers** R0 observation · R1 reversible isolated test · R2 sensitive read / external
research · R3 external communication or consequential reversible action · **R4 financial,
destructive, administrative, public, recovery or self-modifying — disabled until separate
domain qualification; per-action step-up by default.**

**Autonomy levels** `A0_DISABLED` · `A1_ADVISORY` · `A2_PREPARE_ONLY` ·
`A3_SUPERVISED_EXECUTION` · `A4_NARROW_AUTONOMY`. A grant names capability, domain,
operations, principal, purpose, model/tool revisions, budget, rate limits, data classes,
validity, revocation, monitoring and evidence expiry. **No universal trust score unlocks all
tools.**

**Release states** `LAB_ONLY` · `ISOLATED_TEST` · `ADVISORY_LOCAL` · `SUPERVISED_INTERNAL` ·
`SUPERVISED_PRODUCTION` · `NARROW_AUTONOMOUS` · `SUSPENDED` · `REVOKED`.

> **The repository's recorded state in that document is `LAB_ONLY / NO_GO`**, because runtime
> remediation and independent qualification had not started.

**Code audit register** (27 July 2026, **NO REMEDIATION PERFORMED** at that snapshot):
**4,580 findings — 252 Critical · 2,440 High · 1,885 Medium · 3 Low.** Later remediation waves
exist (W1 dashboard, hygiene, gates); **the register's totals were not re-derived by me.** **[U]**

---

# §11 — DOCTRINE — INDEX ONLY

> ## ⚠ §11 IS AN INDEX, NOT THE RULES.
> **Never cite a rule number from here in a consequential argument without opening
> `kai-pm/ENGINEERING_DOCTRINE.md`.** If you quote it from here, mark it `UNVERIFIED`.
>
> A compressed restatement of an authority, used **in place of** the authority, is
> `M-POLICY-ADMISSION-DIVERGENCE` in documentation form — the confirmed mechanism in §6.5,
> listed as an anti-pattern in §12 below. Doctrine at `c1b6efc`: **51 rules, contiguous,
> unduplicated, every rule with provenance.** **[V]**

| rule | meaning |
|---|---|
| **0.0** | Nothing is true because it was true last time. |
| **R1** | Do not assert what you have not run, read or measured. |
| **R2** | **Run** contingencies. An unexercised rollback is a hypothesis with good presentation. |
| **R3** | `&&`, never `;`, when a gate must be able to stop the chain. |
| **R4** | Measure the population **before** fixing. Surprise → inspect the detector first. |
| **R5** | State the denominator and **derive it from the tree**, never a list beside it. |
| **R6** | Fix the **class**, not the instance. |
| **R7** | Findings stay open until formal evidence-backed closure. |
| **R8** | Never-executed code is where the defects are. |
| **R9** | A watcher must not be able to observe itself. |
| **R10** | Full output survives; excerpts declare they are excerpts **and their size**. |
| **R11** | No subject → no observation. Abort at the prerequisite boundary. |
| **I-8** | Calibrate against **independent** known-positive and known-negative. The expected answer must not come from the thing under test. |
| **R12** | Flag the better route, even unasked — but proactive engineering is not autonomous scope expansion. |
| **R13** | A consequential derived claim travels with derivation, denominator, **unit**, limits and rerun method — at transmission. |
| **R14** | Reconcile the instruction against the contract **before** executing. |
| **R15** | If it snags, **check**. A worry beside the deliverable is insurance, not action. |
| **R16** | Memory is a locator, never evidence. Primary source before synthesis. |
| **R17** | `CLAIM_SCOPE ⊆ MEASURED_SCOPE`. Awkward rows travel. |
| **R18** | A correction is not finished until recurrence and **control adequacy** are checked. |

**By purpose.** *Truth/promotion* 1, 2, 24, 27, 34, 35, 36, 40 · *Derived claims* 33, 48 ·
*Evidence identity* 3, 4, 23, 46, 47 · *Measurement vs subject* 5, 6, 7, 8 · *Gates* 9, 10,
11, 12 · *Populations* 13, 14, 37, 41, 42, 43 · *Calibration* 15, 16, 17, 18, 32 ·
*Repair/causality* 29, 30, 31, **51** · *Records/controls* 19, 20, 21, 22, **44**, 45, 49 ·
*Authority* 25, 26, 28, 38, 39, 50.

> **Rule 51 — SIGNAL IS NOT CAUSE.** Trace the first effective failure. Separate independent
> causes from dependent red surfaces. Distinguish product defects from test/calibration/CI
> defects. **Never lower a floor, widen a baseline, bypass a gate or patch a downstream
> symptom to make an indicator green.**

---

# §12 — DIAGNOSTIC ANTI-PATTERNS — every one earned **[C]** / **[V]**

Token match mistaken for semantic meaning · wrong subject / self-vs-other misbinding ·
measurement environment cannot observe the claimed property · declared state exists but the
implementation cannot emit it · calibration does not cover a state used on real data · a
manifest that verifies stale evidence perfectly · detector denominator includes its own
artefacts or prose · context/corpus accidentally defines semantics · `UNKNOWN` silently
treated as negative evidence · evidence binding exists but consumer enforcement does not ·
**post-run cleanliness used to claim no transient mutation** *(OPEN, §6.4)* · **calibration
proves a copied surrogate rather than the authoritative declaration** *(INC-21)* · **a
mutation set aimed only at the implementation cannot find a defect in what the implementation
is measured against** *(INC-21: twelve runner mutations all fired; none touched the registry)*.

---

# §13 — WHAT IS STALE — do not navigate by these **[V]**

| file | date | why it misleads |
|---|---|---|
| `kai-pm/STATUS.md` | 2026-08-01 | *"Phase 0 COMPLETE. Awaiting GPU hardware (RTX 5080)"* — wrong hardware direction; the accepted target is Strix Halo / Ryzen AI MAX+ 395. Six weeks stale on the operator-facing surface. |
| `kai-pm/RISKS.md` | 2026-07-21 | R1 names RTX 5080 procurement. Same contradiction. |
| `kai-pm/SEQUENCE.md` | — | Phase 0–5 GPU model; superseded framing. |
| `kai-pm/NEXT_STINT_PLAN.md` | 2026-08-07 | Predates House H2, 048, Item 8 and all WF work. |

> This is itself an operator-visibility defect of the kind
> `OPERATOR_VISIBILITY_ENGINEERING_DOCTRINE.md` exists to prevent, sitting on the status
> page. Recorded, not repaired — that is Phase-2 work, not a cold-start action.

---

# §14 — STANDING "DO NOT"

**Evidence.** Do not trust pasted messages, summaries, memory **or prior adjudications** as
evidence · do not answer a named artefact's content from recollection · do not infer
document-wide absence from a snippet · do not accept a green job without checking what
executed and what was **skipped** · do not infer root cause from a step label · do not repair
a downstream signal before the first effective failure · do not treat a timeout as failure
where an external effect may have occurred · **do not cite a doctrine rule from §11 without
opening the doctrine.**

**Controls.** Do not lower floors, thresholds or baselines to turn red green · do not weaken a
detector to pass a build · do not let two writers remain authoritative · do not fall back
silently from a stronger cutover to weaker authority.

**Architecture.** Do not create a second authority beside Tool Gate · do not create a new
memory system because ownership is messy — **trace first** · do not add another proactivity
daemon · do not adopt Kafka/NATS/Temporal/SPIRE because they are fashionable · do not turn
screen text, model output or retrieved documents into control instructions · do not turn
avatar/voice output into perceived evidence · do not build A4 self-repair as autonomous
mutation.

**Repository — HARD (§4).** Binance secrets never reach the dashboard layer · no push to
`main` without authorisation · `DECISIONS.md` and `FAILURE_PATTERN_LEDGER.md` append-only · no
destructive git without permission · no PR unless asked · **do not touch `data/SOUL.md`** ·
**do not merge PR #122** · do not freeze a Kingsman master before mapping, adversarial review,
Dainius review and exact-byte freeze.

**Programme.** Do not redraw or re-authorise the frozen Item-8 experiment · do not let A4
planning erase Item-8 obligations · do not conflate `A-4 PROVENANCE` with `FUTURE A4
SELF-DIAGNOSIS` · do not treat the E0–E10 dependency order as permission order.

---

# §15 — ITINERARY FROM THIS CHECKPOINT

> Local itinerary, **not** a replacement for D359. Re-verify HEAD and exact-SHA CI first.

| # | step |
|---|---|
| 1 | **§24 independent review** — Kai adjudicates the calibration-binding repair at `c1b6efc`; verify the registry mutations are caught **by required CI**, not only locally. |
| 2 | **Kai's ruling on the RC-1 manifest collision** — D375's explicit NEXT. Repair vs frozen-aggregate identity is a **programme decision**, not an engineering one. |
| 3 | Next execution-semantics tranche — keep temp-root fallback, evidence-root semantics and transient-write semantics **separate**. |
| 4 | Root-repair the Unified Hunter aggregate red (RC-7). Multi-mechanism; do not patch the first surface. |
| 5 | Real 78-member traversal — **a complete zero-exit aggregate is required before any floor adjudication.** |
| 6 | Adjudicate the 17 unfloored — evidence **and** Dainius approval. No auto-baselining. |
| 7 | WF-3 cutover only after proof — one population authority, one evidence root, exact subject, deterministic subset identity, negative/refusal semantics. |
| 8 | Close the **generic** `M-POLICY-ADMISSION-DIVERGENCE` control. 49.6 has fired; two gates bound, the general rule unimplemented. |
| 9 | Resume the governed sequence: **finish House H2 → H3–H6 → Dainius House Exit Ruling** → 048 closure path → Item 8 → formal closure → A-4 Provenance → Assurance Integration → Professionalisation/CI Truth → Evidence Plane / Kingsman. |
| 10 | Close the **D344–D353 append queue** when exact bytes can be safely appended. |

**Explicitly NOT next:** `main` merge · floor changes · E0/E3/Kingsman runtime refactor · A-4
before upstream authority · A4 self-diagnosis implementation · avatar / computer-use / finance
/ evolution expansion · broad repository clean-up while assurance is unresolved.

---

# §16 — IF YOU HAVE FIVE MINUTES

> Kai is **the organism**, not an LLM, framework, service or device. The **primary mission** is
> to build a Kai that grows with Dainius, cares for him, preserves what matters, becomes
> self-sufficient, survives beyond him, and continues stewardship for his daughter under
> explicit succession governance. **Kingsman is the engineering standard that makes that
> trustworthy — it is not the purpose.**
>
> **Dainius** is final authority; **Orion** executes; **Kai** independently verifies and
> adjudicates; **GPT** (read access) and **DeepSeek** (no access) challenge — neither holds
> admission authority. Never trust memory, pasted reports or prior adjudications as evidence.
> Root cause before symptom. Fix classes, measure denominators, calibrate known-positives and
> negatives, **prove controls can fail**.
>
> **D359 controls programme order and is not superseded.** House is at **H2** — H3–H6 and the
> Exit Ruling have not happened. Item 8 is **frozen, implemented, NOT executed**; `ITEM8_GO`
> does not exist. The live tranche is **D375**, whose NEXT is Kai's ruling on the **RC-1**
> manifest collision. **D344–D353 are banked outside the canonical ledger.**
>
> **Declare your access state first (§0.2).** No access → hypotheses. Read access → findings
> bounded by what you opened, **still no admission authority**.

## §16.1 — Answer before acting

Exact commit and tree? · Which D-number governs — **has it moved past D375?** · Which frozen
contract governs this instruction? · What did the machine **actually execute**, and what did
it skip? · Measured population and **unit**? · First effective failure? · Mechanism **proven**
or **suspected**? · Which failure patterns apply — **read** or remembered? · What is
explicitly **not** authorised? · What does Dainius need to decide?

---

# §17 — SOURCE INDEX

**Open these. Do not cite this document in their place.** All confirmed present at `c1b6efc`. **[V]**

| area | source |
|---|---|
| Operating rules | `CLAUDE.md` |
| Doctrine (51 rules) | `kai-pm/ENGINEERING_DOCTRINE.md` — `a38f6c7ddaa3…` |
| Decisions | `kai-pm/DECISIONS.md` — `a5e64c2512fb…`, D1…D375 with the D344–D353 gap |
| **Pending decisions** | `kai-pm/DECISIONS_CANONICAL_APPEND_QUEUE_D344_D353.md` |
| Failure patterns | `kai-pm/FAILURE_PATTERN_LEDGER.md` — `b1e80446a522…` **APPEND-ONLY** |
| **Root mission** | `kai-pm/KINGSMAN_PRIMARY_MISSION_IDENTITY_AND_LINEAGE_DOCTRINE.md` |
| Mission recovery pointer | `kai-pm/KAI_PRIMARY_MISSION_RECOVERY_POINTER.md` |
| **Operator standard** | `kai-pm/DAINIUS_KINGSMAN_ENGINEERING_STANDARD.md` |
| Pillars | `kai-pm/KINGSMAN_ARCHITECTURAL_PILLARS_INDEX.md` |
| Proactivity | `kai-pm/KINGSMAN_PROACTIVE_ORGANISM_DOCTRINE.md` |
| Resilience | `kai-pm/KINGSMAN_ORGANIC_RESILIENCE_ARCHITECTURE_DOCTRINE.md` |
| Contingency | `kai-pm/KINGSMAN_CONTINGENCY_AND_FAILSAFE_LIBRARY_DESIGN.md` |
| Stewardship | `kai-pm/KINGSMAN_LONG_HORIZON_STEWARDSHIP_AND_SUCCESSION.md` — `bdb88e7337f2…` |
| Final product spec | `kai-pm/KAI_FINAL_PRODUCT_ARCHITECTURE_SPECIFICATION.md` |
| Canon plan | `kai-pm/KINGSMAN_FINAL_VISION_MASTER_CANON_PLAN.md` — `72374b3ef602…` |
| Puzzle map | `kai-pm/KAI_KINGSMAN_PUZZLE_MAP_AND_RECONCILIATION.md` — `9a56a0535ca0…` |
| Authority index | `kai-pm/KINGSMAN_ARCHITECTURE_DOCUMENT_AUTHORITY_INDEX_CURRENT.md` |
| Posture / research | `kai-pm/D351_…md`, `kai-pm/D353_…md` *(pending append)* |
| Phase 2 | `kai-pm/HOUSE_IN_ORDER_PHASE2_PROFESSIONALISATION.md` — `ffbd8304685f…` |
| Future A4 | `kai-pm/A4_SELF_DIAGNOSIS_EVOLUTION.md` — `fb24c3ca4fee…` |
| Evidence Plane lineage | `kai-pm/EVIDENCE_PLANE_RESEARCH_LINEAGE.md` |
| Operator visibility | `kai-pm/OPERATOR_VISIBILITY_ENGINEERING_DOCTRINE.md` — `2b514e94e46d…` |
| Code audit | `kai-pm/CODE_AUDIT_MASTER.md` — 4,580 findings; `LAB_ONLY / NO_GO` |
| UH architecture | `kai-pm/KAI_UNIFIED_HUNTER_ARCHITECTURE_AND_ROADMAP.md`, `UH_PROGRESS_TRACKER.md` |
| Continuity | `kai-pm/KAI_ORION_CONTINUITY.md` — recovery protocol, not truth |
| Working memory | `kai-pm/ORION_FIELD_NOTES.md` — **NON-AUTHORITATIVE** |
| **Identity** | `data/SOUL.md` — **CRITICAL. Do not touch without instruction.** |
| Item 8 | `kai-pm/ITEM8_PREFLIGHT_GO`; `scripts/security/check_item8_design.py`; D282–D290 |
| UH population | `scripts/security/uh_execution_plan.json` — 78 members, digest `1e87fde6…9072`, blob `2f7284c9657e…` |
| WF-3 shadow | `scripts/security/uh_runner.py`, `uh_floor_gate.py` |
| Gate registry | `scripts/security/gate_registry.py` — **authoritative** denominators and `proven_by` |
| Meta-gate | `check_gate_registry.py --gate` — I-1…I-7, ~5½ min |
| PR | **#122 — OPEN, DO NOT MERGE** |

---

# §18 — NORTH STARS **[C]**

> **I — GOVERNANCE.** More capable without becoming less understandable, measurable,
> evidence-bound or governed.
> **II — PERCEPTION.** Sense cheaply. Understand selectively. Investigate actively. Predict
> cautiously.
> **III — DIGITAL ACTION.** Strongest structured interface available; vision and clicking are
> fallbacks.
> **IV — EMBODIMENT.** Presence must never create a second intelligence or authority path.
> **V — EVOLUTION.** **Kai may discover its own future. Kai does not authorise its own future.**

```
Make Kai run.  Make Kai one system.  Make Kai measurable.
Keep observation, belief and authority separate.
Make cognition adapt intelligently to real hardware.
Make perception spend intelligence only where information justifies it.
Make digital action prefer exact interfaces over visual guesswork.
Give Kai presence without giving the presentation layer authority.
Treat every external capability and asset as untrusted until proven.
Promote through evidence, never reputation.
Borrow the strongest ideas without inheriting weak architecture.
Then — and only then — give Kai a governed mechanism to improve itself.
```

> **FINAL COLD-START PRINCIPLE**
>
> Do not rebuild the brain, the memory, the hands or the authority system. Do not build
> another Kai.
>
> Qualify what exists · keep what works · harden what is weak · persist what must survive
> restart · separate what must not share authority · merge accidental duplication · adapt what
> cannot move at once · shadow replacements · prove cutover · **prove the old authority
> dead** · bank the lesson · then move to the next organ.

---

## CHANGE LOG — rev. 2 → rev. 3

Rewritten after reading the governing layer rather than carrying it second-hand.

| # | change |
|---|---|
| 1 | **§2 — the primary mission replaces the engineering mission at the root.** rev. 2 opened with "professionalise Kai into a Kingsman-grade organism" and called that the purpose. That is the *standard*. The mission is Dainius, his continuity, his daughter, and survival beyond him. |
| 2 | **§2.1 — Kai is the organism**, with the explicit list of what is *not* Kai, the three continuity classes, and the lineage/vessel meaning. Absent from rev. 2. |
| 3 | **§2.3 — Dainius's engineering standard** (150 / 110; trust proved, not promised). Absent. |
| 4 | **§5.1 — House is at H2.** rev. 2 gave the D359 order without saying where the programme stands. H3–H6 and the Exit Ruling have not happened. |
| 5 | **§5.2 — the D344–D353 gap.** The only hole in a 358-entry ledger; zero closure boxes ticked. D351/D353 are **not** canonical decisions. |
| 6 | **§5.3 — Item 8 is preflight-only.** Frozen R2 digest; `ITEM8_GO` does not exist; D290 = implemented, not executed. |
| 7 | **§5.5 — Evidence Plane lineage** (NASA/ISHM/IV&V/SRE/TEVV) and *evidence can never create authority*. Absent. |
| 8 | **§6.1 — D375 is the live tranche authority**, with its PROHIBITED and NEXT blocks. rev. 2 named no authority for the current work. |
| 9 | **§6.2 — RC-1 is a deliberate R14 hold, not an unexplained red.** I had reported those reds as unattributed; the repository attributed them all along. |
| 10 | **§9 — stewardship and succession** at full weight: the daughter as a human relationship, the financial invariants, *outlive me ≠ reveal everything after me*. |
| 11 | **§10 — the twelve FP-INV invariants**, risk tiers, autonomy levels, release states, `LAB_ONLY / NO_GO`, and the 4,580-finding audit register. |
| 12 | **§13 — what is stale.** STATUS.md and RISKS.md still name an RTX 5080. Recorded so nobody navigates by them. |
| 13 | Provenance marks re-derived throughout; §17 rebuilt with blob identities and the pending-append surface. |

**What I did NOT read:** the ~80 `CODE_AUDIT_BATCH_*` service audits, the `house_in_order_*`
evidence packages (classification JSON, pass/holdout artefacts, build_evidence), the UH
architecture roadmap and progress tracker in full, and most of the 34,410-line `DECISIONS.md`
outside the ranges cited. Those are **surveyed, not read**. Anything depending on them is **[U]**.

---

**rev. 3 · 15 Sep 2026 · captured at `c1b6efc` / tree `ea1b8d60`**
**Repository authority and live evidence always outrank this document.**
