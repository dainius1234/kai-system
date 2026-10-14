KAI — FINAL SOURCE-BOUND RECONCILIATION OF D379 REPAIR PLAN v4.5

Date: 30 September 2026
Reviewed subject: D379 Repair Plan v4.5
DeepSeek result: NO DESIGN BLOCKER FOUND
Authority of this file: NON-AUTHORITATIVE REVIEW/PLANNING RECORD
Programme authority: NONE BY THIS FILE
Implementation authority: NONE BY THIS FILE
Capture authority: NONE
Candidate/Stage-A/holdout/merge authority: NONE

This record preserves Kai’s final source-bound reconciliation after DeepSeek’s v4.5 closure review.

It does not replace D379/D380/D381/D385 or D387–D389.

⸻

1. FINAL ADJUDICATION

D379 repair plan v4.5 is:

ACCEPTABLE FOR AN IMPLEMENTATION-AUTHORITY DECISION.

That conclusion means:

* no unresolved design blocker remains in the plan;
* all three DeepSeek v4.5 MAJOR findings were independently checked against primary repository/banked evidence;
* the remaining questions/minors were either source-closed or converted into explicit implementation conditions;
* exact historical v4.1 was subsequently recovered and used for a preservation audit;
* implementation still requires Dainius’s explicit separate grant.

It does NOT mean implementation automatically starts.

⸻

2. GOVERNANCE COMMIT G — VERIFIED AFTER PLAN REVIEW

The continuity banking was subsequently completed.

Governance commit G:

77fdc37426a2a60804b52f918d5f2f16c1e2bd2b

Parent:

f140419370b792ec8ad26db58b1f742b5668d626

G changes exactly:

kai-pm/DECISIONS.md

with:

+113 / -0

and no other tracked path.

The three banked entries are:

* D387 — 25 September admission continuity;
* D388 — Kai Q6–Q9 rulings;
* D389 — KAI-V4-01…08 historical findings.

A later, separate handoff-only commit is:

36ad03491c9987e8a3aa55921e5400a4cb963c4c

which changes only:

kai-pm/HANDOFF_LOG.md

and is NOT part of G.

The current rework/handoff branch must not be merged into the repair branch.

⸻

3. DS-V4.5-01 — NETWORK ISOLATION

DeepSeek finding:

environment fingerprint equality alone cannot prove no network-derived build input entered either build.

Ruling:

UPHELD. CLOSED BY IMPLEMENTATION SPECIFICATION.

The two CPython reproducibility builds must run with outbound network mechanically disabled.

The build environment must not merely record network state.

It must prevent dependency acquisition over the network during both builds.

Acceptable implementation must establish no outbound dependency channel for the build commands.

If network isolation cannot be demonstrated:

STOP.

Environment fingerprint E remains required independently.

⸻

4. DS-V4.5-02 — STDLIB REFUSAL PROPAGATION

DeepSeek finding:

a low-level StageIdentityError could theoretically be swallowed by a caller.

Ruling:

UPHELD AS A REQUIRED CHECK; CURRENT PRODUCTION PATHS SOURCE-CLOSED.

Repository inspection at eb52f73 found:

stage_identity.py

* build_stdlib_identity()
* _stdlib_identity()
* build_runtime()
* verify_runtime_identity()

Pass A:

passa.py

calls:

SI.verify_runtime_identity(desc)

before production and again when creating producer provenance.

Classification:

run_h2_v12.py

calls:

SI.verify_runtime_identity(desc).

The inspected production chains do not contain a caller that catches a stdlib/runtime StageIdentityError and continues with a stale or prior identity.

Pass A’s _check_population() catches StageIdentityError only to convert it into SystemExit, i.e. refusal.

The repaired qualifier must use the same fail-closed runtime path.

Mandatory hostile control remains:

make a governed stdlib member unreadable/disappearing;

confirm not merely that build_stdlib_identity() raises, but that the actual producer/qualifier pipeline refuses.

No successful smaller identity is permitted.

⸻

5. DS-V4.5-03 — classify_origin() LOCATION

DeepSeek questioned whether repairing classify_origin() could require changing a file outside B4.

Ruling:

DISPROVED.

At eb52f73:

def classify_origin(...)

is in:

kai-pm/house_in_order_h2_v13/stage_identity.py

approximately line 525.

stage_identity.py is already inside the authorised B4 repair surface.

No B4 expansion is required.

⸻

6. DS-V4.5-04 — cal_fixtures.py LAUNCH REPORTING

The v4.5 wording could be read as though the F13 static analyser covers both:

* d379_controls.py;
* cal_fixtures.py.

That is not the intended claim.

Final interpretation:

F13’s governed static/runtime denominator is the process-launch surface implemented by d379_controls.py.

cal_fixtures.py is a separate historical fixture subject.

Its internal Python launch is reported by a separate non-F13 observation/measurement.

It is not counted as part of F13’s D379-control denominator and is not silently omitted.

No common-source independence claim is made.

⸻

7. DS-V4.5-05 — D385 KNOWN-POSITIVE INTERPRETER

Banked D385 was retrieved and independently checked.

D385 does NOT say that stock CPython 3.11.15 is automatically known-positive.

D385 defines a known-positive as an actually measured:

D380-COMPLIANT INTERPRETER

which satisfies existing D380 rules with no exception class.

Required properties include:

* stdlib/platstdlib mechanically identified;
* external package roots excluded;
* every governed symlink’s final target remains within governed roots;
* no dangling links;
* single ownership;
* no duplicate canonical member;
* loaded filesystem-backed stdlib represented.

The current distribution interpreter is banked as a known-negative.

The v3.11.15 build is therefore a candidate calibration runtime only.

If the built v3.11.15 runtime does not actually satisfy D380/D385:

STOP.

Do NOT weaken D380/D385.

Do NOT add distro-specific exceptions.

⸻

8. DS-V4.5-06 — EARLY CENSUS IMPORT

Repository inspection across the ten H2 source files found direct governed Census imports only in:

passa.py

inside the Pass-A build path:

import docgraph as G, opscan as O, claims as C

No current top-level H2 import of those governed Census modules was found before that point.

Therefore S19 is a valid fail-closed precondition for the current architecture.

The repaired verified-byte Census loader must establish its boundary before that first governed Census import.

⸻

9. DS-V4.5-07 — FAILED CENSUS EXECUTION

Required implementation clarification:

when the verified-byte loader pre-registers a Census module in sys.modules and compile/exec fails:

* remove the partially initialised governed module entry from sys.modules;
* then propagate the failure.

A partially initialised governed Census module may not survive and later satisfy an import.

⸻

10. DS-V4.5-08 — STAGE-A IDENTITY AUTHORITY

The wording:

“recompute from supplied canonical Stage-A descriptor”

must not be implemented as self-consistency.

Final interpretation:

derive the complete canonical Stage-A descriptor independently under v4.5 §9 from actual instrument/subject/history/Census/runtime contexts.

Then derive the expected Stage-A identity from THAT independently rederived descriptor.

Compare:

* independently rederived identity;
* identity of supplied descriptor.

The supplied descriptor is commitment/input.

It is not its own authority.

⸻

11. DS-V4.5-09 — AUDIT-HOOK INSTALLATION

Each d379_controls.py process that may create a governed child must install its launch observer at process entry before execution reaches any function capable of launching a child.

It is insufficient to install the observer lazily after some controls have already executed.

This applies independently to:

* capture parent mode;
* control child mode;
* ordinary control execution mode

where that process has governed launch capability.

⸻

12. DS-V4.5-10 — BUILD TRANSCRIPT

The E7 build transcript is not a manually curated summary.

For each Build A / Build B, preserve the complete raw driver evidence for the governed build commands:

* exact command invocation;
* stdout bytes;
* stderr bytes;
* actual return code;
* command order;
* environment-fingerprint reference.

Any additional structured summary is convenience only.

It cannot substitute for the raw combined transcript.

⸻

13. DS-V4.5-11 — FROZEN NON-CODE INPUTS

Every explicitly frozen non-code input admitted by B0 dependency closure is also a member of the F fixity/dependency population.

No B0 dependency may disappear from F merely because it is not Python source.

If B0 finds no additional non-code input, record zero.

If it finds one, it must be enumerated and hashed at F.

⸻

14. DS-V4.5-12 — TEMPORARY STATE FILE

The temporary state-exchange file is process scratch state only.

Implementation requirement:

use a managed temporary directory and cleanup in finally.

Delete scratch state on both:

* normal completion;
* failure/exception.

Any adjudication-relevant fact must already have been emitted to durable captured evidence.

No residual temp file is evidence.

⸻

15. EXACT HISTORICAL v4.1 RECOVERED

After the v4.5 review, the repository acquired:

kai-pm/D379_PLAN_V4_1.md

at the rework branch.

Its header states that everything below its verbatim marker is the complete historical v4.1 held by Orion’s session, with no reconstruction gaps.

Kai read that exact recovered record and performed a preservation audit against v4.5.

This audit does NOT make v4.1 programme authority.

Its purpose is to ensure the later plan did not accidentally lose an original required repair/control.

⸻

16. v4.1 PRESERVATION AUDIT — REQUIRED DEFECT FAMILIES

The final implementation matrix must retain:

B1–B6

The six primary D379/H2 repair blockers.

F7 — bytecode-disabled from startup

Late mutation of sys.dont_write_bytecode is not adequate.

Known positive must obtain the required state from process startup.

F8 — D380 §7.11 loaded-stdlib membership

Every loaded filesystem-backed stdlib module must be represented in the governed stdlib snapshot, unless it is legitimately built-in/frozen under D380.

A sourceless/unrepresented filesystem-backed stdlib member must refuse.

F9 — unreadable governed member

Unreadable/disappearing governed member must refuse.

No silent continue producing a smaller identity.

F10 — symlink cycle distinction

A symlink loop must be diagnosed/refused as a cycle.

A dangling symlink must be diagnosed/refused as dangling.

A valid in-root symlink is accepted.

Do not collapse cycle into “dangling”.

F12 — duplicate on both sides

Duplicate tree population:

REFUSE.

Duplicate output population:

REFUSE.

Same duplicate present in both populations:

still REFUSE before reconciliation/selection.

F13 — child launch closure

Launch count is a measurement, not a fixed number.

Coverage is the gate.

Governed D379 Python child launches must route through the governed mechanism and satisfy required startup/runtime conditions.

⸻

17. STAGE-B ANCHOR — NON-CIRCULAR CALIBRATION INTERPRETATION

Historical v4.1 correctly requires more than a separate binding file:

rows + provenance + binding rewritten together must not self-certify.

For D379 hostile calibration, the non-circular sequence is:

producer exits

→ separate binder creates binding B over clean final bytes

→ D379 control parent independently computes/retains sha256(B) in parent state

→ hostile mutation is constructed where applicable

→ consumer receives:

* artifact;
* original binding B;
* expected sha256(B) retained by parent.

The expected binding digest does not come from:

* the producer artifact;
* the binding file being verified;
* the artifact directory.

For Q1a-7/coordinated-rewrite hostile cases, the original parent-held anchor remains unchanged.

The D379 parent/control implementation is later frozen by fixity F before the consequential capture.

No second Unix/GitHub principal is invented.

The same-principal malicious-repository-owner actor remains outside D379’s banked threat model.

⸻

18. REPAIR-BRANCH LINEAGE AFTER G

Governance G now exists on the rework/handoff branch.

The repair branch must NOT be created from the current rework HEAD.

The intended lineage is:

eb52f73

→ first repair-branch commit R

→ technical v4.5 repair commits.

R replays ONLY G’s DECISIONS.md additions.

No handoff/hook/documentation commits from the current rework branch are merged into the repair branch.

Mechanical replay gate:

the G decision diff and R decision diff must be byte-equivalent.

Orion recorded the current G fingerprints as:

sha256:

ba175463b2e18644c7eafa898c34294bdc59b6cb2a26b8f7ccea12c879c70a75

patch-id:

149f2dbd97a0a878af0a1c6f8d6239f2ae162c00

Those values are producer-supplied continuity evidence and must be reverified when R is created.

⸻

19. BUILD NETWORK CONDITION

DeepSeek DS-V4.5-01 exposed that equality of E does not prove no network input entered the build.

Therefore before implementation the build requirement is sharpened:

both reproducibility builds must run with outbound network mechanically disabled.

E remains required.

Network isolation and E equality prove different properties.

No dependency download is authorised during configure/make/install.

⸻

20. IMPLEMENTATION-AUTHORITY STATE

At the time this reconciliation record is preserved:

continuity D387–D389 is banked.

v4.5 has passed DeepSeek with:

NO DESIGN BLOCKER FOUND.

Kai has independently reconciled the final findings.

The plan is ready for Dainius to make an implementation-authority decision.

However:

NO IMPLEMENTATION AUTHORITY IS CREATED BY THIS FILE.

Until Dainius explicitly grants the bounded implementation tranche:

* do not create R;
* do not create the repair branch;
* do not run E7a;
* do not build CPython;
* do not mutate B4;
* do not produce F;
* do not capture;
* do not create production Stage A;
* do not run candidate/holdout/blind 40;
* do not merge PR #122.

⸻

21. CAPTURE REMAINS A SEPARATE FUTURE AUTHORITY

Even after implementation is separately authorised:

the repair stops at fixity F.

Kai independently reviews F.

Then Dainius must separately authorise exactly one consequential D379 capture.

Implementation authority does NOT include capture authority.

⸻

FINAL KAI POSITION

D379 v4.5 design status: no unresolved design blocker.

DeepSeek planning gate: complete.

Kai reconciliation gate: complete.

Continuity banking: complete at G / D387–D389.

Historical v4.1 preservation audit: complete.

Implementation: NOT YET AUTHORISED BY THIS RECORD.

Capture: NOT AUTHORISED.

Production Stage A / candidate / holdout / blind 40 / merge: NOT AUTHORISED.

END OF KAI FINAL SOURCE-BOUND RECONCILIATION OF D379 v4.5