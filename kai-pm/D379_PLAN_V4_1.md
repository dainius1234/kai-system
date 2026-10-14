# D379 PLAN v4.1 — PRESERVED TEXT (non-authoritative record)

**Provenance.** Written by Orion (session_01AuiBo9KTWtrAa9w5xnZHJH) on
2026-09-30 on Dainius's authorisation ("Yes: A and B"), at the request of
session_01PvwTQHZU2sxi6i3oBmoqoT (row 2, part A). Everything below the
line `=== VERBATIM BEGINS ===` is plan v4.1 exactly as Orion transmitted it
in conversation on 2026-09-30, from the text held in this session's
context. No part was reconstructed from memory. Gaps: **none** — the full
text was held.

**Status.** Plan v4.1 is a PLANNING TEXT. It creates no programme state,
grants no permission and is not adopted. Kai had not yet checked it
against the repository when it was written; DeepSeek has not attacked it.
D379 execution remains stopped.

**Known to be out of date since transmission (not edited below):**
- A2/B0 describe `claude/project-rework-plan-pgvp35` at `0af5d32`. The
  branch has since moved (kai-handoff commits, starting at `7123ec2`), so
  B0's "stays at `0af5d32` unless the Part D append is authorised" is no
  longer accurate. The D379 repair base `eb52f73` is unaffected.
- The allocator snapshot (369/369, highest D386) was measured at
  2026-09-30T17:29:38Z and must be re-derived before any append.

=== VERBATIM BEGINS ===

# D379 repair plan v4.1: complete text

**From:** Orion, 2026-09-30. Text only. No branch, commit, Dropbox call, download, build or capture has been made, and none is authorised.

**How the findings reached me.** Kai's eight findings, KAI-V4-01 to 08, arrived as one-line headlines. Each remedy below is my design against that headline. Kai should confirm that each remedy answers the finding he meant.

---

## Part 0: Change register (v4 → v4.1)

| id | source | change | section changed |
|---|---|---|---|
| ER-1 | v4 erratum 1 | F13 "reconstructed tree" becomes "repaired tree" | B1 |
| ER-2 | v4 erratum 2 | a round-trip failure stops under **S9**, not S7 | A4, E7 |
| ER-3 | v4 erratum 3 | the "Dainius uploads it himself" fallback is **deleted**; failure means S9, and the build is not evidenced | A4, E7 |
| ER-4 | v4 erratum 4 | the old branch stays at `0af5d32` unless the Part D append is authorised | B0 |
| ER-5 | v4 erratum 5 | the Part D append is stated as a governance action outside the repair surface | B4, B7, Part D |
| ER-6 | v4 erratum 6 | all measurements re-dated to 2026-09-30 | A2 |
| KAI-V4-01 | BLOCKER | Stage-B gets an **independent anchor**: a separate binder process, plus an expected-anchor input the consumer checks | C3, B5, S10 |
| KAI-V4-02 | BLOCKER | **pre-capture fixity gate**: the capture runs only from a Kai-reviewed, frozen commit, with a clean tree | new B8, S11, B7 |
| KAI-V4-03 | BLOCKER | F13 is enforced at **runtime by an audit hook** built from process-creation events taken from the CPython source; static counting stays as a measurement | E4, B5 |
| KAI-V4-04 | BLOCKER | E7 is split: **E7a** proves the store before the build with a disposable file (needs authority); **E7b** round-trips the real log after the build and before the capture | E7, B7, S9 |
| KAI-V4-05 | MAJOR | native-dependency closure is recorded and a **governance gap** is raised; the signer key needs two independent sources | E1, E2, E3 |
| KAI-V4-06 | MAJOR | explicit path canonicalisation rule: **NFC required, never normalised**, with a fixed order of checks | C1, C5, B5 |
| KAI-V4-07 | MAJOR | Part D is recorded on the old branch **and replayed** as the repair branch's first commit, with a proof that the content is identical | B0, Part D |
| KAI-V4-08 | MAJOR | **per-slot differential controls** prove each authority mapping is *correct*, not just present | C4, B5 |

---

## Part A: State, keeping the two verifiers separate

### A1. Rulings in force (unchanged from v4)

- The 25 September admission stands. `eb52f73` is the admitted technical state. `fc1bb9d` is admitted and carries zero authority weight.
- The rebuild from `86ebfde` is cancelled.
- K2: the launch-site count is a measurement; coverage is a gate.
- Dropbox is the canonical store for the full build log.

### A2. Repository facts, 2026-09-30

| fact | Orion-measured (local clone, 2026-09-30T17:29:38Z) | Kai-independent (GitHub, 2026-09-30) |
|---|---|---|
| `claude/project-rework-plan-pgvp35` | `0af5d32072bcf1d09c09e5c83c9b5b71b5560676` (`git ls-remote`) | **same** |
| `main` | `194db0a0c13b4d5b322997fc1ceb33bdd21a77bc` | **same** |
| `claude/cai-v1-bootstrap` | `3f2dad036823ab95f1a469bf0a882316be86ce24` | **same** |
| `claude/d379-repair-eb52f73` | absent | **absent** |
| `eb52f73…0af5d32` | one file, `ORION_FIELD_NOTES.md`, +99 lines (`git diff --stat`) | **same**: one commit, +99/−0 |
| working tree clean | `git status --porcelain` printed 0 lines | not verified |
| tree of `0af5d32` | `4c05a750e1aed738879c140b51782de83bdf26e3` | not verified |
| tree of `eb52f73` | `63108560782eb62fdb0ecf66ffbc48d1477a221b` | not verified |
| decision headings, grammar `^## D[0-9]+( +—|$)` over `kai-pm/DECISIONS.md` | 369 headings, 369 distinct, 0 duplicates, highest D386 | **not verified** (the connector did not return the contents) |
| `DECISIONS.md` matches for `2026-09-2[5-9]`, `2026-09-30`, `25 September` | 0 | not verified |

**Preserved discrepancy.**
- `ORION_FIELD_NOTES.md` §0, line 37, says "370 entries". Under the strict grammar above, I measure 369.
- The notes are frozen under the hold, so the "370" stands uncorrected in them.
- The allocator figure is **Orion-measured only**. It is not Kai-verified until Kai counts the file independently.
- The allocator must be re-derived immediately before any authorised append.

### A3. Mutation-surface correction (from v3, still in force)

- v2's three new tracked paths stay withdrawn.
- The environment driver is a `--environment` mode of `d379_controls.py`.
- Environment evidence goes into `D379_CONTROLS.txt` and `D379_CLOSEOUT.txt`.

### A4. Dropbox: capability is unmeasured (ER-2, ER-3, KAI-V4-04)

**What is unmeasured.**
- The connector documents `create_file` for text files, and lists binary upload as **not supported**.
- Size limit, overwrite behaviour and byte fidelity are all **unmeasured**.
- The log exists only inside the build container.

**What follows.**
- There is **no out-of-band fallback**; the v4 "Dainius uploads it himself" fallback is deleted (ER-3).
- If the store cannot hold the exact bytes, the result is **S9**: the build is not evidenced, and it stops there.
- The capability test is **E7a** (see B3). It is a Dropbox write, so it is **not authorised**, and the build cannot start without it.

---

## Part B: Plan v4.1

### B0. Base and lineage (ER-4, KAI-V4-07)

- **Branch.** `claude/d379-repair-eb52f73`, from `eb52f73fa6485534ca7e28a42055861c69e94cc4`. It is created only on Dainius's grant of branch creation and execution.
- **Old branch.** `claude/project-rework-plan-pgvp35` stays at `0af5d32` **unless the Part D append is authorised**. In that case it gains exactly one governance commit. It is never rewritten, deleted or force-pushed.
- **Gates.**
  - Creation gate: `test "$(git rev-parse HEAD)" = eb52f73fa6485534ca7e28a42055861c69e94cc4`.
  - Per-commit gate: `git merge-base --is-ancestor eb52f73 HEAD`.
- **Part D replay (KAI-V4-07), if Part D is authorised.**
  - The **first** commit on the repair branch replays the Part D governance commit, so the record is an ancestor of every repair commit.
  - Proof: the diff of the replay commit must equal the diff of the old-branch governance commit, byte for byte. That is, `git diff G^ G` equals `git diff R^ R`, where G is the governance commit and R is the replay.
  - This is not a merge: a merge would bring in `0af5d32`'s field-notes lines, which lie outside the surface.
  - Measured fact that makes the replay clean: `DECISIONS.md` is identical at `eb52f73` and `0af5d32`, because only the notes differ (A2).
- **Diff reporting.** Every repair diff is reported against `eb52f73`. Fail-old controls run on `eb52f73` bytes.
- **Push authority.** Pushing the new branch needs the same explicit grant.

### B1. Blockers and findings (ER-1)

Line references are at `eb52f73`, read on 2026-09-25. The package code is byte-identical at `0af5d32`.

| id | defect | source | severity |
|---|---|---|---|
| B1 | Stage-A validation incomplete (the contents of five blocks, sort order, formats); an unsorted `h2_sources` gives a second identity | `stage_identity.py:374-431`, `:138-144` | BLOCKER |
| B2 | Pass A is not bound to subject, tree, population, Census or history; provenance copies expected values; Census bytes are unchecked | `passa.py:1178-1215`, `:1146-1150`; `stage_identity.py:676-678` | BLOCKER |
| B3 | No production Stage-B transport; `json.load(open())`; no `--stage-b` | `qualify.py:307-318` | BLOCKER |
| B4 | Q1a-9 tests a substitute by text match; legal digest slots have no value authority | `d379_controls.py:1697-1760`; `verify_provenance` skips non-H2 members with `continue` | BLOCKER |
| B5 | Holdout tree is unbound (strip, blank-skip); result is unbound | `holdout.py:115-121`, `:130` | BLOCKER |
| B6 | No D380/D385-compliant interpreter | D385; INC-34 | BLOCKER |
| F7 | Setting `sys.dont_write_bytecode` late passes | `stage_identity.py:330` | BLOCKER |
| F8 | D380 §7.11 is unimplemented | `stage_identity.py:586-587` | BLOCKER |
| F9 | Unreadable entries are silently skipped | `stage_identity.py:249-250`, `:288-289` | BLOCKER |
| F10 | A symlink loop is reported as dangling | `stage_identity.py:265-268` | MAJOR |
| F12 | A duplicate present on both sides is accepted | `holdout.py:85-87` | BLOCKER |
| F13 | Children are launched without startup flags. `grep` found 9 `sys.executable` sites at `eb52f73`: lines 1949, 2401, 3561, 3564, 3747, 3762, 3763, 3911, 3919. This is a grep figure, **not a closed population** (KAI-V4-03). It is re-measured on the **repaired** tree and never hard-coded | `d379_controls.py` | BLOCKER |

### B2. WP-C: code repairs

All within the D379 §2 surface. No new H2 module; `h2_sources` stays exactly ten.

**C1 · Stage-A validation by mode** (`stage_identity.py`)

- **Every mode:**
  - exact nested shapes for all ten fields;
  - no unknown fields;
  - no duplicate JSON keys, enforced with `object_pairs_hook`. Measured on 2026-09-25: `json.loads('{"a":1,"a":2}')` returns `{'a': 2}`;
  - canonical ordering, checked but never re-sorted;
  - Git-ID and digest formats, a non-bool `int`, and ISO dates;
  - cross-field consistency.
- **Path canonicalisation rule (KAI-V4-06).** Normative, and applied in exactly this order to every path in every population:
  1. valid UTF-8, or REFUSE;
  2. **the string must already equal its NFC form, or REFUSE "not NFC"**. Paths are **never normalised** into acceptance: normalising would make two different byte strings one identity;
  3. D380 §6.10 structure: relative, `/`-separated, no `.` or `..` segments, no backslash, or REFUSE;
  4. duplicate check on the exact validated strings, or REFUSE "duplicate".
  - Therefore an NFD path is refused at step 2, and it never reaches duplicate comparison.
  - The existing `_norm_path` already refuses non-NFC (line 175). v4.1 makes the order normative and applies it to the tree, output and h2_sources populations alike.
- **PRODUCTION** additionally requires the frozen D380/D381 values, plus the re-derivation in C2.
- **CALIBRATION** accepts synthetic values that pass every all-mode rule, with zero production, admission or holdout weight.
- **Constants** are cross-checked against the D380 text, parsed independently from `DECISIONS.md`.

**C2 · Subject and dependency binding** (`passa.py`, `stage_identity.py`, `run_h2_v12.py`)

- **Before `build()`, REFUSE unless each of these equals the descriptor:**
  - `--subject`;
  - the resolved tree;
  - the §6.6 bytes re-derived from `git ls-tree`, with every path passing the C1 order;
  - `sha256(MANIFEST.sha256)`. Measured at `eb52f73` on 2026-09-25: `29064d65…`;
  - the §6.8 history derivation.
- **Provenance.** It records observed values only.
- **Census reconciliation.** Every loaded CENSUS member must be in the bound manifest, with an equal digest.
- **`run_h2_v12.py`.** It verifies the Pass-A subject and the Pass-A Stage-B record.

**C3 · Stage-B with an independent anchor** (`stage_identity.py`, `passa.py`, `run_h2_v12.py`, `qualify.py`, `holdout.py`). This is KAI-V4-01.

- **Problem accepted.** In v4 the producer wrote the result, its provenance and its binding. Anyone able to rewrite all three consistently passes every consumer check. Provenance fields are Stage-A-bound (C4), but the result **rows** are bound only by the binding. So a coordinated rewrite of rows plus a new binding self-certifies.
- **Remedy, two layers.**
  1. **Separate binder.** The producer never writes a binding.
     - After the producer process has **exited**, a separate process (`stage_identity.py bind <artefact>`) reads the final bytes from disk and writes `<artefact>.stage_b.json`.
     - The binder records the producer's exit status, and refuses to bind if it is non-zero.
  2. **Independent anchor.** Every governed consumer (`run_h2_v12.py` for Pass A, `qualify.py`, `holdout.py`) takes a **required** `--expected-binding-sha256 <hex>`.
     - It REFUSES unless `sha256(binding bytes) == <hex>` and the binding verifies against the artefact.
     - The expected value comes from a record the producer cannot rewrite. In this tranche that is the committed capture evidence, `D379_CONTROLS.txt`, at a pushed commit.
     - The consumer never reads the anchor from the directory that holds the artefact.
- **One-read consumption (unchanged).** The artefact and the binding are each read **once** into an immutable buffer. The hash and the parse both come from that buffer.
- **Residual, stated.** An actor who controls both the artefact directory **and** the pushed evidence commit defeats this. That is the evidence-authority boundary, and it goes to DeepSeek surface 7 rather than being claimed closed.

**C4 · Q1a-9: semantic no-self-authority invariant** (`stage_identity.py`, `d379_controls.py`)

- **Invariant.** No in-band provenance value may derive its authority from the containing artefact's own final bytes. The final-byte digest lives only in the external Stage-B binding.
- **Schema limb.** This is a regression guard, not a repair. The old code already rejects extra fields at all three levels.
- **Value-authority limb.**
  - Digest-bearing slots are derived mechanically from the schema.
  - Each slot has a named authority from a closed set.
  - Unknown authority → REFUSE.
  - Coverage gate: `slots=N · authority-named=N · unknown=0`.
- **The mapping as read at `eb52f73`.** The implementation derives the slot list; this table is not copied into code.

  | slot | authority |
  |---|---|
  | `stage_a_identity`, `stage_a_descriptor_digest` | `STAGE_A` |
  | `runtime_identity.executable_sha256`, `.stdlib_identity` | `STAGE_A` + `OBSERVED_RUNTIME` |
  | `subject_*`, `tree_paths_identity`, `census_identity`, `history_source_identity` | `STAGE_A` + `OBSERVED_SUBJECT` |
  | `producer_population[H2].sha256` | `GOVERNED_SOURCE_BYTES` |
  | `producer_population[CENSUS].sha256` | `GOVERNED_CENSUS_MANIFEST` |
  | `producer_population[STDLIB/BUILTIN/FROZEN].sha256` | `EMPTY_BY_RULE` (must be exactly `""`) |
  | `input_binding.pass_a_artifact_sha256`, `.pass_a_producer_provenance_digest` | `CONSUMED_PASS_A_BYTES` |
  | `input_binding.pass_a_stage_a_identity` | `STAGE_A` |
  | the artefact's own final bytes | `EXTERNAL_STAGE_B` only |

- **Mapping correctness (KAI-V4-08).** Completeness is not correctness: a slot mapped to the wrong authority still counts as "named". For **every** derived slot S with named authority A, a differential control runs in a child:
  - (i) **S wrong, A unchanged** → REFUSE, naming S. This proves S is checked at all.
  - (ii) **A changed consistently, S unchanged.** For example, a synthetic CALIBRATION descriptor or manifest in which A's value differs → REFUSE on S. This proves S is compared *against A*.
  - (iii) **A and S changed together, consistently** → PASS. This proves A is the authority, not a hard-coded constant or a different source.
  - (iv) **A different authority B ≠ A changed, S and A unchanged** → S's verdict does not change. This proves S is not bound to the wrong authority. It is run wherever B can be varied independently; where it cannot, it is recorded as "not independently variable" and never silently skipped.
  - For `EMPTY_BY_RULE` slots, (ii) and (iii) do not apply. Instead: any non-empty value → REFUSE, and the STDLIB member must be absent from the snapshot → REFUSE (F8).
  - Expected verdicts come from how the case is constructed, not from the mapping code (I-8).
- **Unverified slots.** Governed consumers REFUSE when any slot is left unverified.
- **Removed:** the text search and any fixed-point construction.
- **Positive limb:** finalise, then the separate binder hashes the file, then the anchor is recorded, then the consumer verifies against the exact buffer it read.

**C5 · Holdout** (`holdout.py`)

- **`main()`, in this order:**
  1. validate the PRODUCTION Stage-A descriptor (this gate is unchanged);
  2. read the binding bytes once and check them against `--expected-binding-sha256`;
  3. read the result bytes once and verify them against the binding;
  4. parse that same buffer;
  5. read the exact `--tree-paths` bytes: no strip, no blank-skip, §6.6 form, every path through the C1 order, identity and count equal to the descriptor;
  6. call **one** `plan_selection()`;
  7. write the output.
- **`plan_selection(tree_paths, output_paths, aggregate)`:**
  - independently refuses duplicates on the tree side and on the output side, **before** any multiset comparison;
  - then exact reconciliation, then deterministic `select`;
  - it never derives an aggregate itself, and there is no production bypass;
  - controls import this function and carry no copy of its logic.

### B3. WP-E: environment

**E1 · Source and signature closure** (KAI-V4-05, signature part)

- **Fetch location.** Tag `v3.11.15` is fetched into the scratchpad only.
- **REFUSE unless all three match:**
  - tag object `2323bfc729b041c43b1e5e4c5f18c548fc345323`: Orion-measured (`git ls-remote`, 2026-09-25) and Kai-verified;
  - peeled commit `2340a037f7450e70fccfe411e6531afb4d57a312`: Orion-measured and Kai-verified;
  - source tree `8c6959bc70b201b477138f00c432a3bb2f1caddd`: **Kai-verified only**, and re-derived here.
- **Signature closure.**
  - GitHub's "verified" badge is **one authority**: GitHub's.
  - To verify the tag here, the signer's public key must come from **two independent authorities**, with matching fingerprints before `git verify-tag`. Candidates are the signer's GitHub key endpoint and a public keyserver; which of these is reachable through this proxy is **unmeasured**. python.org is blocked (measured 2026-09-25).
  - Outcomes are recorded as exactly one of:
    - `VERIFIED-TWO-SOURCE`;
    - `VERIFIED-SINGLE-SOURCE (<which>)`;
    - `UNVERIFIED-HERE; KAI-VERIFIED-ON-GITHUB`.
  - Nothing is assumed. Kai decides whether single-source is acceptable **before** the build.
- **Recorded:** `sha256(git archive <commit>)` and `git --version`.

**E2 · Build independence and toolchain closure** (KAI-V4-05, toolchain part)

- **Independence.** Two clean source checkouts, two clean build directories and two clean staging roots, with the same logical `--prefix=<P>`. Each build installs with `make install DESTDIR=<stage_n>`, and build 2 inherits nothing from build 1.
- **Build commands.** `./configure --prefix=<P> --without-ensurepip`, then `make`, then `make install DESTDIR=…`. No `--enable-shared` and no PGO.
- **Environment.** `LC_ALL=C`, a fixed `SOURCE_DATE_EPOCH`, `PYTHON*` unset. The full environment is recorded.
- **Toolchain closure.**
  - The package name, version and sha256 of each toolchain binary actually invoked (`gcc`, `cc1`, `as`, `ld`, `ar`), resolved via `realpath` and `dpkg -S` / `dpkg -s`.
  - The libc version.
  - `pyconfig.h`, `config.log` and the `Makefile` digests.
  - The missing-optional-modules line.
- **Comparison.** Both builds' `executable_sha256` and `stdlib_identity` must be equal. If they differ, report the **first differing derived artefact**, and do not weaken the comparison.
- **Unmeasured expectations:** that `getpath` derives the prefix from the binary's location, and that `--prefix` is compiled into the binary.

**E3 · Identity and native-dependency closure** (KAI-V4-05, dependency part)

- **Identity.** `executable_sha256`, `implementation_name`, `cache_tag`, `version`, and `stdlib_identity` from the repaired §7 builder.
- **Native dependencies.**
  - `lib-dynload/*.so` files are inside platstdlib, so D380 §7 hashes them.
  - **Their dynamic dependencies are not covered** (for example `libssl`, `libz` and `libffi` under `/usr/lib/x86_64-linux-gnu`), and neither is the interpreter's own libc/libm.
  - v4.1 **records** them as runtime evidence only:
    - `ldd` of the executable and of every `lib-dynload` object;
    - at capture time, the set of shared objects actually mapped (`/proc/self/maps`), with the realpath and sha256 of each.
  - They are **not** added to Stage A: D385 forbids a V2 bump and new fields.
- **Governance gap, raised and not repaired.** D380's runtime identity does not bind native shared-library dependencies. Kai to rule whether that is accepted risk, or needs later authority.

**E4 · Startup conditions and launch closure** (KAI-V4-03)

- **Startup flags.**
  - `-B` is the D380 §6.9 requirement. The evidence is `sys.flags.dont_write_bytecode == 1` and `sys.dont_write_bytecode is True`. F7 REFUSES without the startup flag.
  - `-E` and `-s` are controlled environment conditions, not D380 requirements.
  - `-S` and `-I` are not adopted.
- **Launch closure (runtime enforcement).** A static census of `sys.executable` sites cannot be closed over aliasing (`exe = sys.executable`), wrappers (`_run_child`, `run_governed_child`) or alternative mechanisms (`os.system`, `os.exec*`, `os.spawn*`, `os.posix_spawn*`, `pty.spawn`, `multiprocessing`, `asyncio` subprocesses, `shell=True`). v4.1 therefore enforces at runtime:
  1. **Event population, derived and not listed.** Process-creation audit events are extracted mechanically from the **fetched CPython `v3.11.15` source** (E1): every `PySys_Audit` / `sys.audit` call whose event name is in the `os.*` process group or the `subprocess.*` group. The derived set and its count are printed. This makes the population a property of the interpreter actually built, not a list kept alongside it.
  2. **Audit hook.** `d379_controls.py` installs `sys.addaudithook` at start, before any launch. For every derived event it applies these rules:
     - a Python interpreter launch must be exactly the product of `governed_argv()`: the built interpreter plus `-B -E -s`;
     - a non-Python executable must be in a closed, printed allowlist (`git`, `setpriv`);
     - `setpriv` must wrap a `governed_argv()` command;
     - any `shell=True`, or any launch not matching these rules → the hook raises, and the capture FAILS.
  3. **Children.** Governed children are either product scripts, whose only grandchildren are `git` (to be measured), or snippet children, which install the same hook as their first statement. Each Python child records its observed `sys.flags` and the hook's installed state.
  4. **Static measurement (K2), unchanged in role.** The AST census over `d379_controls.py` and the ten H2 sources prints `population=N · governed=N · bypass=M`, and `bypass != 0` is a FAIL.
     - It is now **supplementary**, because the runtime hook is the closure.
     - Known-positives cover each alternative mechanism: an alias, a wrapper, `os.system`, `multiprocessing`, and `shell=True`. **Each must be caught by the hook**, and where the static census can see one, by the census too.
     - The known-negative is a governed launch, which passes.
  - **Residual, stated.** Audit events cover launches that go through CPython's instrumented paths. A C extension calling `fork`/`exec` directly would bypass them. EC-8 plus the dependency record in E3 are the evidence that no such extension is loaded. This goes to DeepSeek.

**E5 · Compliance criteria (unchanged)**

| id | criterion | rule |
|---|---|---|
| EC-1 | stdlib roots and CASE recorded | D380 §7.1 |
| EC-2 | external roots excluded first | §7.3 |
| EC-3 | every symlink stays inside the roots; none dangling; no cycles | §7.4, §7.6 |
| EC-4 | each file has a single owner | §7.7 |
| EC-5 | no file lost to a swallowed error (F9) | §7.4 |
| EC-6 | every loaded stdlib file is represented (F8) | §7.11 |
| EC-7 | `-B` in effect from startup (F7) | §6.9 |
| EC-8 | no `sitecustomize`/`usercustomize`; no package or `.pth` in `site-packages`; every `sys.path` entry governed | D385 §B, D379 §6 |
| EC-9 | a relocated copy gives identical identities | D380 §9 |

**E6 · Capture boundary**

- The driver is `d379_controls.py --environment` under the distribution interpreter, which is not a governed measurement.
- The one capture is `d379_controls.py` under the built interpreter via `governed_argv()`, from the B8 fixity commit.
- `D380-STDLIB-NEG-1` is re-measured in the same container.
- The interpreter stays in the scratchpad, for calibration only (D385 §D).
- Build logs go to Dropbox, canonical:
  - path `KAI/D379/interpreter/v3.11.15/<build-id>/<sha256>.log`;
  - `D379_CONTROLS.txt` records sha256, byte count, build/run identity, UTC timestamp and path;
  - an Actions copy is convenience only.

**E7 · Store proof, re-sequenced** (KAI-V4-04, ER-2, ER-3)

- **E7a: capability, before any build.** This is **not authorised**; it needs Dainius's explicit grant.
  - Write one disposable synthetic text file, at least as large as the expected log and containing LF, CRLF and non-ASCII bytes, to a scratch Dropbox path.
  - Record whether an existing path is refused or overwritten.
  - Retrieve it through a separate read path, and recompute sha256 and byte count.
  - If byte fidelity or overwrite refusal is not demonstrated → **S9**, and no build.
- **E7b: the real log, after the build and before the capture.**
  1. Confirm the target path is absent.
  2. Write both full logs.
  3. Retrieve through a separate path and recompute sha256 and byte count; both must be equal.
  4. Only then do the builds count as evidenced, and only then may B8 and the capture proceed.
  - The container must not be released between the build and E7b.
  - Any failure → **S9**.

### B4. Mutation surface (ER-5)

**Repair surface, a subset of D379 §2:**
- `stage_identity.py`: C1, C2 (Census), C3 binder and loader, C4, F7, F8, F9, F10;
- `passa.py`: C2, C3;
- `run_h2_v12.py`: C2, C3;
- `qualify.py`: C3;
- `holdout.py`: C1 paths, C5, F12;
- `build_evidence/d379_controls.py`: controls, `governed_argv()`, the audit hook and census, `--environment`;
- `D379_CONTROLS.txt` and `D379_CLOSEOUT.txt`: regenerated once, by the capture.

**No other tracked path in the repair tranche.**

**Separate governance action.** The Part D append to `kai-pm/DECISIONS.md` is authorised by D379 §2 as "BANKING RECORD ONLY". It is outside the repair surface and needs its own grant from Dainius.

**Excluded:**
- `FAILURE_PATTERN_LEDGER.md`;
- `ORION_FIELD_NOTES.md`, frozen with its "370" left as is;
- `CLAUDE.md` and `ENGINEERING_DOCTRINE.md`;
- `cal_fixtures.py`;
- `classify.py`, `envelope.py`, `ontology.py` and `subjectbind.py`;
- the Census package, `house_in_order_h2/` and `data/SOUL.md`;
- production Stage A, the candidate, production Pass A and classification, holdout execution and the blind 40;
- PR #122, merges, `main` and the CAI branch;
- interpreter binaries, source and logs.

### B5. Controls

**Conventions.**
- **Fail-old** runs on `eb52f73` bytes and must show the defect.
- **Pass-new** must give the intended verdict for the intended reason.
- **Opposite side** means the correct case does not refuse.
- Every verdict comes from a child's exit code and output, under `governed_argv()` with the audit hook installed.
- "—" in the fail-old column means a new case; it is not a claim that old code passes.

| case | fail-old | pass-new | opposite side |
|---|---|---|---|
| C1-a..e, one byte in each of contract/subject/tree_paths/census/history (PRODUCTION, in-memory) | accepted | REFUSE, naming the block | frozen values accepted |
| C1-f, unsorted `h2_sources` | two identities | REFUSE | one identity |
| C1-g, duplicate JSON key | last key wins | REFUSE | — |
| C1-h, inconsistent synthetic CALIBRATION | accepted | REFUSE | consistent synthetic accepted |
| C1-i, constants vs D380 text | — | equal | mutated parsed copy is unequal |
| **C1-j, path order (KAI-V4-06):** NFD-only path; NFC+NFD twin; invalid UTF-8; `./a`; `a/../b` | per case (the old `_norm_path` refuses non-NFC; the tree and output populations were never checked) | each REFUSES at its **first-effective** step (twin → "not NFC" on the NFD member, never "duplicate") | NFC unique paths accepted |
| C2-a..f, wrong commit / tree / path ± / manifest byte / Census byte / Census module missing from manifest | not refused | REFUSE | correct synthetic subject proceeds |
| C2-g, observed ≠ descriptor | copies descriptor | REFUSE before writing | equal proceeds |
| C3-a..f, result byte / identity swap / provenance-digest swap / Q1a-7 deletion / cross-run binding / `--stage-b` absent | no such argument | REFUSE | clean accepted |
| C3-g, one read | — | hash and parse from one buffer | — |
| **C3-h, coordinated rewrite (KAI-V4-01):** rows, provenance and binding all rewritten consistently, with the anchor unchanged | no anchor exists: accepted | REFUSE on anchor mismatch | untampered with correct anchor passes |
| **C3-i, binder separation:** producer attempts to write its own binding; producer exits non-zero | — | binding refused / binder refuses | normal exit leads to binding |
| C4-schema | already refuses (regression guard) | REFUSE | clean passes |
| C4-value, every slot × {digest of pre-plant bytes, stripped digest, unrelated valid digest} | non-H2 slots accepted | REFUSE, naming the slot | clean passes |
| **C4-diff (KAI-V4-08), every slot:** (i) S wrong; (ii) A changed, S not; (iii) A and S both changed; (iv) B ≠ A changed | non-H2 slots: (i) accepted | (i) REFUSE; (ii) REFUSE; (iii) PASS; (iv) unchanged, or "not independently variable" recorded | — |
| C4 authority coverage | no authority for non-H2 slots | `unknown=0` | an extra unmapped slot → FAIL |
| C5-a..f, tree bytes: swap / extra / missing / trailing space / CRLF / no final LF | accepted | REFUSE | exact bytes accepted |
| C5-g, result without or with a mismatching binding or anchor | accepted | REFUSE | verified binding accepted |
| F12 population class (child → `plan_selection`): unique; output-only duplicate; tree-only duplicate; **duplicate on both sides**; missing; extra | both-sides duplicate accepted (measured 2026-09-25) | every non-clean case REFUSES before selection | unique set reconciles |
| I1B-1..4, I1A-3, I1-gate | as ruled on 2026-09-25 | as ruled | — |
| F7 | late assignment accepted | REFUSE | `-B` accepted |
| F8 | sourceless `.pyc` classed as STDLIB | REFUSE | represented `.py` accepted |
| F9, via `setpriv` to uid 65534 | identity equals "file removed" | REFUSE | readable tree accepted |
| F9 launcher proof | — | the child's observed `os.getuid()` and `setpriv`'s exit code are recorded; if either is wrong → FAIL (CONTROL FAILURE) | a uid-0 run of the same file reads it |
| F10 | loop reported as "dangling" | loop → cycle; dangling → dangling | in-root link accepted |
| **F13 closure (KAI-V4-03):** alias, wrapper, `os.system`, `multiprocessing`, `shell=True`, `setpriv` wrapping a raw argv | not intercepted | the hook raises and the capture FAILS, one case per mechanism | governed launch passes |
| F13 static census | detector on `eb52f73` must report `bypass>0`; its exact value is whatever it reports (9 is a grep figure, not a prediction) | `bypass=0` | a synthetic direct site → FAIL |
| F13 audit event population | — | the derived event set is printed from the fetched source | an event removed from the derived set in a copy → a known launch goes unintercepted → the calibration FAILS |
| **B8 fixity (KAI-V4-02):** dirty tree; HEAD ≠ fixity commit; a governed file byte changed after fixity | — | the capture refuses to start | clean tree at the fixity commit starts |
| E | distribution interpreter: NEG-1 REFUSE | built interpreter meets EC-1..9 | injected faults in a copy each REFUSE |

### B6. Stop conditions

- **S1.** E1–E5 cannot be established → stop and return evidence. D379 stays HELD; no acceptance capture is claimed.
- **S2.** No change to D380 or D385 of any kind.
- **S3.** A fail-old control does not show its defect → stop.
- **S4.** A file outside B4 is needed → stop and ask.
- **S5.** A mostly-HELD capture is never called a closeout.
- **S6.** Exactly one capture, from the B8 fixity commit, with no manual alteration → stop for Kai.
- **S7.** No durable store has been chosen → no build.
- **S8.** The creation, ancestry or Part D replay-equality gate fails → stop.
- **S9.** E7a or E7b does not reproduce sha256 and byte count exactly, or the target path already exists → stop. The build is not evidenced, and there is **no** out-of-band fallback.
- **S10 (KAI-V4-01).** A consumer is invoked without an anchor obtained from the committed evidence → REFUSE. Any path by which the anchor could be read from the artefact directory → stop.
- **S11 (KAI-V4-02).** The capture cannot run from the reviewed fixity commit on a clean tree → stop.
- **S12 (KAI-V4-05).** Tag-signature state is below what Kai accepted before the build → stop.

### B7. Order (ER-5, KAI-V4-02, KAI-V4-04)

1. Kai checks v4.1 against the repository.
2. DeepSeek attack.
3. Kai reconciles every finding against primary evidence.
4. Dainius decides:
   - the Part D append (a governance action);
   - E7a;
   - implementation authority.
5. If Part D is authorised: allocator re-derived, append, commit on the old branch.
6. Branch from `eb52f73`. If Part D was authorised, its replay is the first commit.
7. E7a, if authorised. E1 fetch. Kai rules on the E1 signature state.
8. WP-C repairs and controls, committed.
9. E2–E5 builds, then E7b.
10. **B8 fixity:**
    - commit the exact capture code;
    - record the SHA and the sha256 of every B4 file;
    - Kai reviews that SHA;
    - Dainius authorises the capture against that SHA.
11. The one capture. It records the fixity SHA, the file digests and a clean-tree check in its header. The capture commit's diff must be exactly `D379_CONTROLS.txt` and `D379_CLOSEOUT.txt`.
12. Stop. Kai's independent review, then tranche accept or reject. Later, and separately, candidate authority.

### B8. Pre-capture fixity gate (new, KAI-V4-02)

- **Problem accepted.** Repaired controls produce their own acceptance evidence. Without fixity, the code that decides PASS could change between review and capture.
- **The fixity commit F.**
  - It contains every B4 source file, including all expected verdicts: each case's expected class and reason is fixed in code at F.
  - Kai reviews F by SHA **before** the capture.
- **The capture refuses to start unless:**
  - `git rev-parse HEAD == F`;
  - `git status --porcelain` is empty;
  - the sha256 of every B4 source file equals the value recorded for F.
- **Capture header.** It records F, those digests, the interpreter identity (E3) and the anchor values (C3).
- **Post-capture proof.** `git diff --stat F C` must list exactly the two `.txt` files; anything else fails the tranche.
- **Residual, stated.** F is only as trustworthy as Kai's review of F. The gate prevents drift after review; it does not replace the review.

---

## Part C: DeepSeek attack packet (Kai to forward)

> **Role.** Adversarial reviewer. You create no programme state; Kai reconciles every finding against primary source.
>
> **Subject.** Plan v4.1 Part B, for repository `dainius1234/kai-system`:
> - physical HEAD `0af5d32` on `claude/project-rework-plan-pgvp35` (notes-only commit on top);
> - admitted state and proposed base `eb52f73`.
>
> **Return format.** For each finding give:
> - a severity: BLOCKER / MAJOR / MINOR / QUESTION;
> - the section attacked;
> - a concrete failure scenario;
> - the exact repository evidence that would establish or disprove it (file:line, command or commit).
>
> No finding without a falsifiable evidence request.
>
> **Attack all eight surfaces:**
>
> 1. **Admission lineage (B0, Part D).**
>    - Does the Part D replay make the admission ancestral to every repair commit?
>    - Can replay equality pass while content differs?
>    - Can the creation and ancestry gates pass with a wrong base?
> 2. **Stage-B independence (C3, S10).**
>    - After the separate binder and the committed anchor, can result, provenance, binding and anchor still be changed together?
>    - Is there any consumer path that reads the anchor from a location the producer controls?
>    - Is one-buffer reading enforced at every consumer?
> 3. **Q1a-9 (C4).**
>    - Do the C4-diff controls (i)–(iv) actually prove each mapping correct?
>    - Which slots are "not independently variable", and does that leave a hole?
>    - Does `EMPTY_BY_RULE` open one?
> 4. **Interpreter trust (E1–E3).**
>    - Two-source key provenance.
>    - Build independence.
>    - Toolchain closure.
>    - Native dependencies outside D380's identity: is recording them enough for a calibration runtime?
>    - `-B -E -s` versus `-S`/`-I`.
>    - The calibration/production boundary.
> 5. **Harness equivalence (B5, E4).**
>    - Is the audit-hook event population truly derived from the built source?
>    - Can any launch mechanism escape it (C extensions, `ctypes`, `os.fork` without exec)?
>    - Can `setpriv` failure masquerade as a permission result?
> 6. **Population integrity.**
>    - The C1 path order (NFC, then structure, then duplicates).
>    - Duplicates on either side or both.
>    - Denominator drift.
>    - Are the F13 census and the C4 slot derivation closed over their own populations?
> 7. **Evidence authority (B8, C3, E7).**
>    - Can the fixity commit F, the capture, the anchors, the Stage-B files or the Dropbox logs self-certify, or be regenerated without detection?
>    - What binds the Dropbox log to F and to the capture?
> 8. **Bypass and degraded paths.**
>    - The old `reconcile()` reachable?
>    - Helper calls that skip the mode gate.
>    - Stale fixtures, fallback runtimes.
>    - Any `except`/`continue` that turns REFUSE or UNKNOWN into PASS.

---

## Part D: Continuity record — DRAFT ONLY, NO D-NUMBER (ER-5, KAI-V4-07)

**Status.**
- Not allocated, not appended, not committed.
- It is a **governance action** under D379 §2, "BANKING RECORD ONLY". It is outside the repair surface and needs Dainius's explicit grant.
- The allocator is re-derived immediately before any append, using the grammar `^## D[0-9]+( +—|$)`.
- Orion-measured on 2026-09-30: 369/369, highest D386. Not Kai-verified. The field notes' "370" is a recorded discrepancy.

```
## D<n> — <measured UTC date of append> — CONTINUITY RECORD: 25 SEPTEMBER 2026
## ADMISSION OF 8e3ee69 · fc1bb9d · 630ceaf · eb52f73. GOVERNANCE ONLY —
## NO IMPLEMENTATION, CANDIDATE, STAGE-A OR HOLDOUT AUTHORITY.

WHO        Dainius, final authority, following Kai's commit-by-commit
           adjudication. Recorded by Orion at Dainius's instruction.
WHEN       Admission made 2026-09-25 (conversation). Recorded here on the
           measured date above, because it was not made durable at the time.
WHY        The four commits made after authority was consumed at the return
           of 86ebfde were individually adjudicated. Their bounded changes
           were accepted. The D379 closeout and tranche remained REJECTED
           (six blockers: Stage-A validation; Pass-A subject binding;
           Stage-B transport; Q1a-9 proxy; holdout population binding;
           D380/D385-compliant interpreter).

STATE      8e3ee69   ACCEPTED   technical
           fc1bb9d   ACCEPTED   ZERO authority weight (non-authoritative
                                field notes)
           630ceaf   ACCEPTED   technical
           eb52f73   ACCEPTED   technical — THE ADMITTED TECHNICAL RESTART
                                STATE (eb52f73fa6485534ca7e28a42055861c69e94cc4)
           0af5d32   physical HEAD of claude/project-rework-plan-pgvp35;
                     notes only; grants no execution authority.
           No candidate, real Stage A, production Pass A/classification,
           holdout, blind 40, admission, merge or D379 closure is granted.

CORRECTION A later cold-start ruling (a new Kai thread) directed
           reconstruction from 86ebfde. It was made from repository evidence
           that lacked this admission, and IT DOES NOT SUPERSEDE IT. It is
           withdrawn by Kai. The mechanism: a consequential decision made
           only in conversation, and therefore invisible to repository-first
           recovery.

RULE CARRIED
           Repository state can invalidate an old factual claim. Repository
           SILENCE cannot erase a prior consequential decision that was
           explicitly made and accepted. Recover the authority history first,
           then adjudicate.
```

**Append mechanics, if authorised.** Chained with `&&`:
1. Re-derive the allocator and print it.
2. Confirm D<n> is absent.
3. Append at EOF.
4. `git diff --stat` shows `kai-pm/DECISIONS.md` only, with insertions only.
5. Commit G on `claude/project-rework-plan-pgvp35`, then push.

**Ancestry (KAI-V4-07).**
- G on the old branch alone would **not** be an ancestor of a repair branch created from `eb52f73`.
- So if the repair branch is created, its first commit R replays G, and the diff of R must equal the diff of G (S8).
- G makes the record durable whether or not a repair grant ever exists. R makes it ancestral to all repair work.
- No merge, because a merge would import `0af5d32`'s notes lines.

---

**Open for Kai.**
- Review v4.1 against the repository.
- Confirm that each KAI-V4-01..08 remedy answers the finding as intended: I had the headlines only.
- Rule on the E3 native-dependency governance gap.
- Before the build, rule on the acceptable E1 signature state.

**Open for Dainius.**
- Part D append.
- E7a.
- The implementation grant.

**Status.** This is the final and only part of v4.1. Nothing was mutated; execution remains stopped.
