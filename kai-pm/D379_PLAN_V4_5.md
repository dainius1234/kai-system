D379 REPAIR PLAN v4.5

FINAL CLOSURE CANDIDATE

Date: 30 September 2026
Status: PLAN ONLY — FROZEN FOR FINAL ADVERSARIAL CLOSURE REVIEW
Implementation authority: NONE
Repository mutation authorised by this document: NONE
Production Stage A: NOT AUTHORISED
Real candidate: NOT AUTHORISED
Pass A / classification / qualification / holdout: NOT AUTHORISED
Blind 40: NOT AUTHORISED
PR #122 merge: NOT AUTHORISED

This is the COMPLETE review subject.

It supersedes v4.2, v4.3 and v4.4 as planning text.

Banked D379/D380/D381/D385 authority outranks this plan.

⸻

1. PURPOSE

v4.5 is the bounded repair plan for the D379/H2 pre-candidate assurance instrument.

Its function is:

BROKEN / HELD ASSURANCE MACHINERY

→ repair exact identity/provenance/runtime/holdout controls

→ hostile-calibrate those controls

→ establish a D380-compliant known-positive interpreter

→ freeze the exact repaired instrument

→ execute one governed evidence capture

→ stop for independent Kai adjudication.

It is not a general KAI architecture project.

It does NOT authorise the real candidate.

⸻

2. VERIFIED PHYSICAL STATE

Repository:

dainius1234/kai-system

main:

194db0a0c13b4d5b322997fc1ceb33bdd21a77bc

Current D379 branch:

claude/project-rework-plan-pgvp35

Physical HEAD at plan freeze:

7123ec2211ec2c70e531c070351bc1cfde05e1ff

PR #122:

OPEN
UNMERGED
DO NOT MERGE.

Proposed technical restart:

eb52f73fa6485534ca7e28a42055861c69e94cc4

Latest producer capture at that technical state:

* cases 40;
* PASS 16;
* FAIL 0;
* HELD 24;
* checks passed 204;
* failed 33;
* control process non-zero.

This is producer evidence.

D379 remains OPEN.

⸻

3. BANKED AUTHORITY

D379:

608d706d8452b8e578a484f7b75331a5cb9c28d9

D380:

c50989779baf0485e2a4a5ceb2113093441691e3

D381:

838b7637058c5ba3b8f3b6c5430ebdc324249b96

D385 remains governing for the known-negative / known-positive interpreter boundary.

No new D-number is created by this plan.

⸻

4. D379 STAGE-B CONTRACT — EXACT SCOPE

This section resolves DS-V4.3-01 and DS-V4.4-06 from the actual banked D379 text.

Banked D379 §4 states:

EXTERNAL STAGE-B BINDING — owned by stage_identity.py, computed on FINAL bytes

Its prescribed sequence is:

finalise artefact → hash exact final artefact bytes → binding entry

The binding contains:

* artifact path;
* artifact SHA-256;
* artifact kind;
* Stage-A identity;
* producer component;
* producer-provenance digest.

Banked D379 further states that the Stage-B aggregate is derived externally and is never placed inside the byte population from which it is derived.

Banked D379 §5 states that for durable production the external Stage-B binding carries:

producer_provenance_digest

alongside:

artifact_sha256

and that this digest is the independent anchor used to prevent a provenance list defining its own completeness.

Most importantly, banked Q1a-7 defines its concrete hostile subject:

one governed member deleted from OUTPUT provenance after production, independently captured runtime binding unchanged → REFUSE.

Therefore the actual D379 threat case is mechanically defined:

* clean producer result exists;
* external binding is captured;
* result provenance is subsequently altered;
* original external binding remains unchanged;
* consumer must refuse.

D379 contains NO requirement for:

* a second Unix identity;
* a second GitHub account;
* another human;
* another credential;
* malicious repository-owner resistance.

v4.5 therefore establishes the banked D379 structural invariant.

It expressly does NOT claim malicious same-principal rewrite resistance.

A later programme decision may add such a threat model.

It is not silently invented here.

⸻

5. D380 NATIVE-DEPENDENCY SCOPE — EXACT AUTHORITY

This resolves DS-V4.4-05.

Banked D380 §3 states:

SCOPE IS DELIBERATELY LIMITED TO PYTHON MODULE / IMPORT ORIGINS

and expressly states that D380:

does NOT claim to cryptographically attest every native process or operating-system library dependency

It then explicitly names:

* libc;
* dynamic loader;
* system TLS;
* kernel interfaces;
* transitively loaded native libraries

as outside the contract, with no whole-OS attestation claimed or implied.

Therefore:

native dependencies remain diagnostic evidence only.

They are not Stage-A identity fields.

S13 remains deleted.

No weakening occurs because this is the exact banked scope.

⸻

6. PART D — TECHNICAL RESTART MUST BE RESOLVED FIRST

Before a repair branch exists, Dainius must separately resolve:

eb52f73 IS the accepted D379 technical restart

or:

eb52f73 IS NOT the accepted D379 technical restart.

Pending does not permit implementation.

If accepted:

record the continuity/accountability decision as governance commit G.

That record remains simple:

WHO
WHEN
WHY
WHAT / COMMIT
RESULT.

No CAI machinery.

⸻

7. B0 — COMPLETE TECHNICAL BASELINE CLOSURE

If Part D accepts eb52f73, derive the complete D379 implementation closure at:

* eb52f73;
* G.

The corresponding populations must be exact-byte identical.

7.1 H2 instrument population

Derive directly from:

stage_identity.H2_SOURCES.

Expected population:

Do not duplicate it in another hand-maintained list.

7.2 Census package

Include:

house_in_order_census_v11/MANIFEST.sha256

and every exact member declared by that manifest.

Expected governed Census aggregate:

29064d650a61296806df3c3bcab3322f7364da7df674ac93e79d0671475d757a

7.3 frozen contract

Include:

kai-pm/H2_REPAIR_CONTRACT_D367.md

Expected SHA-256 is fixed by banked D380 §6.3:

0ce5792ed72e6e7051ecc050664490899a847d01de2f62cff564f460d46800bb

The expected value is therefore GOVERNING AUTHORITY from D380.

It is not an arbitrary current-code constant.

7.4 D379 control implementation

Include:

build_evidence/d379_controls.py

7.5 local dependency closure

Statically derive local imports/implementation dependencies starting from all executable B4 members.

Every dependency must resolve into:

* Stage-A H2 source;
* governed Census source;
* D367 contract;
* D379 controls;
* or an explicitly frozen non-code input.

Unresolved executable dependency:

STOP.

The exact claim is:

the declared D379 execution closure is byte-equivalent at eb52f73 and G.

The whole repository is NOT claimed byte-identical.

⸻

8. FOUR CONTEXTS — NEVER COLLAPSE

8.1 instrument source root

This is the repaired H2 implementation actually executing.

It is not caller-supplied.

Derivation rule:

1. take the actual loaded stage_identity.py filesystem source;
2. require filesystem-backed source;
3. resolve its real path;
4. require its basename/location correspond to the governed Stage-A member:
    kai-pm/house_in_order_h2_v13/stage_identity.py;
5. derive instrument repository root as the canonical ancestor corresponding to the repo-relative H2_SOURCES paths;
6. require every loaded CLASS_H2 module to resolve beneath that same canonical root;
7. require each resulting repo-relative path to be a member of H2_SOURCES.

No:

* sys.argv[0];
* environment variable;
* --subject-repo;
* shell CWD

may define instrument root.

A symlinked stage_identity source resolves to its canonical real location before root derivation.

Can-fail:

instrument at A;

subject repo at B;

B contains same-named H2 files.

Expected:

H2 authority remains A.

This resolves DS-V4.4-02.

⸻

8.2 subject repository

The D380 frozen document subject.

Required commit:

d8aac4d49e6ba997e3eb38062c0917186ee3f197

Required tree:

3abc9e9d8ca11966a6f996d5f0af68072ee5b117

Required tracked .md population:

Subject repo authority comes from:

Git commit/tree/content.

Not local pathname.

⸻

8.3 history source

Separate logical role.

Must be non-shallow and satisfy the exact D380 §6.8 history identity.

Its local path is not Stage-A authority.

⸻

8.4 Census package

Separate logical role.

Its authority is its exact governed CONTENT identity.

Its pathname is deliberately not Stage-A authority.

Therefore Census is NOT required to sit outside:

* subject repo;
* history repo;
* instrument checkout.

Location independence is a D380 property.

What matters is that the exact consumed Census bytes satisfy the governed aggregate.

⸻

9. COMPLETE STAGE-A PRODUCTION CONSTRUCTION

Current source has no production build_stage_a().

v4.5 requires one implementation in:

stage_identity.py.

Shape validation does not establish semantics.

For PRODUCTION every semantic field below is independently rederived.

9.1 schema

Derived:

H2_STAGE_A_V2

9.2 mode

Derived:

PRODUCTION

9.3 h2_sources

Derive from exact ten H2_SOURCES members at instrument root.

Read exact source bytes.

Hash exact bytes.

No caller-supplied source digest.

9.4 contract

Read exact D367 contract bytes.

Compute SHA-256.

Compare with banked D380 §6.3 digest.

9.5 governance

Construct exact closed V2 governance:

D379 bank commit
D380 bank commit
D381 bank commit.

No arbitrary caller governance.

9.6 subject

From actual subject repo derive:

* HEAD;
* tree;
* tracked .md population.

Require exact D380 values.

9.7 tree_paths

Derive D380 canonical tree-path population.

Path rules:

* relative;
* POSIX;
* NFC;
* no backslash;
* no absolute path;
* no empty component;
* no .;
* no ...

Non-NFC:

REFUSE.

No silent normalisation.

9.8 Census

Derive from the exact same verified Census byte objects actually executed under §10.

9.9 history

Re-derive the complete D380 §6.8 history object.

No descriptor value serves as its own proof.

9.10 runtime

Call:

build_runtime()

against the current producer process.

Derive:

* executable SHA;
* implementation name;
* cache tag;
* version;
* stdlib identity;
* bytecode state.

9.11 supplied descriptor equality

Pass A receives materialised --stage-a.

Before any output:

construct the full Stage-A descriptor independently from §§9.1–9.10.

Require canonical supplied bytes == canonical independently rederived bytes.

Then require identity equality.

Every nested production field is semantically rederived.

There is no shape-only semantic production field.

Mismatch:

REFUSE BEFORE OUTPUT.

⸻

10. CENSUS EXACT-BYTE EXECUTION — CLOSED ORDERING

This section closes DS-V4.4-01.

Current source evidence:

* passa.py imports no Census module at file/module startup;
* top-level local H2 import is envelope.Witness;
* stage_identity.py imports no Census module;
* current Census modules are first imported inside passa.build().

Therefore the final repair can establish the boundary before first Census import.

10.1 no preloaded governed Census module

Before any Census source is executed, require:

docgraph not in sys.modules

opscan not in sys.modules

claims not in sys.modules

If any governed Census module required by this Pass-A path is already present:

REFUSE.

Do NOT silently replace it.

This catches an ancestor/wrapper that imported a same-named module first.

10.2 derive required Census execution closure

For current Pass A the governed execution roots are:

* docgraph.py
* opscan.py
* claims.py

Before execution, statically inspect their local Census imports.

Any additional local Census dependency becomes part of the same verified-byte execution closure.

Current source inspection finds no local cross-import between those three, but this is re-derived at implementation time.

10.3 read once

Read each required Census source once into immutable bytes.

For each:

* validate manifest membership;
* SHA-256 exact bytes;
* compare against manifest;
* bind to governed Census aggregate.

10.4 install before any ordinary Census import

For each governed Census module:

1. create module/spec under its exact module name;
2. register the module object in sys.modules BEFORE executing its code;
3. compile the already-verified byte object;
4. execute those SAME compiled bytes in that registered module object;
5. record SHA-256 of the byte object actually supplied to compile/exec.

This pre-registration handles import-cycle semantics without permitting ordinary path resolution to substitute another module.

10.5 Pass A uses the installed objects

Pass A receives/uses those installed module objects directly.

No later:

sys.path.insert(...)

followed by a fresh ordinary Census source read is authoritative.

Any ordinary import docgraph/opscan/claims after installation must resolve to the exact object already held in sys.modules.

Assert object identity.

10.6 can-fail controls

Census preload attack

Preload alternate docgraph into sys.modules.

Then invoke governed Pass A loader.

Expected:

REFUSE before measurement.

alternate path attack

Place alternate same-named Census files earlier on sys.path.

Governed modules not preloaded.

Install verified Census objects.

Then import names.

Expected:

sys.modules[name] remains the governed verified object.

byte mutation attack

Verify bytes A.

Change filesystem bytes to B before module execution.

Expected:

executed module remains compiled from verified A.

dependency expansion attack

Make a governed Census module require an unaccounted local Census module.

Expected:

execution closure changes / STOP until that module is manifest-verified and included.

S18 is satisfied only if the exact bytes verified are the exact bytes compiled/executed.

⸻

11. PRODUCER POPULATION ROOT REPAIR

Current classify_origin() derives:

H2 root

and Census root

from caller-supplied repo_root.

That is a verified current defect.

Repair the interface.

11.1 H2 classification

Uses canonical instrument root from §8.1.

11.2 Census classification

Uses the governed Census module identity/verified-byte registry from §10.

11.3 subject repo

Never supplies H2/Census code authority.

11.4 history source

Never supplies H2/Census code authority.

Context-collapse control:

subject repo contains byte-identical or modified same-named H2 modules.

Expected:

never classified as instrument modules merely because subject repo was supplied.

⸻

12. PRODUCER AND QUALIFIER RUNTIME — DISTINCT CONTROLS

Banked D380 §5 explicitly states:

Q1a asks:

WHO PRODUCED THE RESULT?

§8(6) asks:

ARE THE QUALIFIER’S OWN EXECUTING BYTES THE GOVERNED BYTES?

It says the two controls must never be collapsed.

12.1 current runtime verifier semantics

Current:

build_runtime()

observes the calling process by reading:

* os.path.realpath(sys.executable);
* exact executable bytes;
* sys.implementation.name;
* sys.implementation.cache_tag;
* sys.version;
* independently derived stdlib identity;
* sys.dont_write_bytecode.

verify_runtime_identity() compares that observed block with Stage A.

12.2 producer

Producer verifies its own executing runtime.

Records observed runtime in provenance.

Stage-B later binds exact provenance digest.

12.3 qualifier

Qualifier separately verifies ITS OWN current executing runtime.

It does not pretend to reobserve the terminated producer.

Current qualify.py lacks this common verifier call.

Repair it.

12.4 known negative

Producer provenance valid and bound.

Qualifier executes under runtime B while Stage A expects A.

Expected:

qualifier refuses on its own runtime mismatch.

⸻

13. EXTERNAL STAGE-B EXACT-BYTE CHAIN

13.1 regular artifact requirement

For governed Stage-B production/consumption artifacts:

lstat the supplied artifact path.

Require:

regular file.

Symlink artifact path:

REFUSE.

This removes path/symlink ambiguity rather than widening artifact identity.

No FIFO/device/directory.

13.2 producer finalisation

Producer closes output and terminates.

13.3 binder

Separate governed invocation of stage_identity.py:

* opens final regular artifact;
* reads bytes once;
* derives SHA-256;
* parses those SAME bytes where provenance extraction is required;
* derives canonical producer-provenance digest;
* creates external binding.

Producer does not provide expected:

* artifact SHA;
* provenance digest;
* binding digest.

13.4 consumer

Consumer likewise:

* refuses symlink/non-regular artifact;
* reads bytes once;
* hashes those SAME bytes;
* parses those SAME bytes;
* compares to original Stage-B binding.

A different inode containing identical bytes is equivalent under D379’s content identity.

A symlink is refused before that question arises.

⸻

14. Q1a-7 — ORIGINAL BINDING MUST REMAIN ORIGINAL

Hostile sequence:

1. clean CLASSIFICATION artifact finalised;
2. original provenance P0;
3. binder produces Stage-B B0 over clean final artifact;
4. record:
    provenance_digest(P0);
5. create tampered P1 by removing exactly one governed producer-population member;
6. decrement denominator consistently;
7. assert:
    population(P1) is strict subset of population(P0);
8. assert:
    denominator(P1) == len(population(P1));
9. assert:
    provenance_digest(P0) != provenance_digest(P1);
10. assert:
    in-object verification alone ACCEPTS P1;
11. retain B0 unchanged;
12. run real qualifier with tampered artifact + original B0;
13. expected:
    REFUSE.

Current source already performs the critical digest inequality and strict-subset assertions.

They remain mandatory.

A hostile construction that fails to mutate the provenance is itself:

CONTROL FAILURE.

It may not count as Q1a-7 PASS.

⸻

15. PASS-A → CLASSIFICATION → QUALIFICATION

15.1 Pass A external binding

After Pass A finalises:

Stage-B records:

* exact Pass-A artifact SHA;
* Stage-A identity;
* producer component;
* Pass-A producer-provenance digest.

15.2 classification

Classification:

* reads exact Pass-A bytes once;
* hashes those bytes;
* parses same bytes;
* validates Pass-A binding;
* validates Stage-A identity;
* validates producer provenance.

Then executes.

15.3 classification Stage-B

Created after classification finalises.

15.4 qualifier

Qualification receives:

* Stage A;
* exact classification bytes;
* original classification Stage-B binding;
* referenced Pass-A binding.

It must mechanically close:

input_binding.pass_a_artifact_sha256

and:

input_binding.pass_a_producer_provenance_digest.

These currently remain explicitly unverified by shipped qualifier code.

⸻

16. COMPLETE PROVENANCE/AUTHORITY MAPPING

This resolves DS-V4.4-03.

Authority sources are classified as:

OBSERVED_CURRENT
derived from current process/repository/filesystem.

PRECOMMITTED_STAGE_A
value already frozen in the independently validated pre-execution Stage-A commitment.

EXACT_ARTIFACT_BYTES
derived from the exact bytes the current consumer read.

BANKED_CONSTANT
fixed by D379/D380/D381.

No slot’s recorded provenance value is permitted to define its own expected value.

16.1 PASS_A / CLASSIFICATION common fields

stage_a_identity

Authority:

PRECOMMITTED_STAGE_A.

Mechanism:

recompute from supplied canonical Stage-A descriptor.

stage_a_descriptor_digest

Authority:

PRECOMMITTED_STAGE_A.

Mechanism:

SHA-256 exact canonical descriptor bytes.

producer_component

Authority:

BANKED_CONSTANT + executable role.

Pass-A process expects:

PASS_A.

Classification expects:

CLASSIFICATION.

producer_population

Authority at production:

OBSERVED_CURRENT.

Mechanism:

actual loaded module/import-origin observation classified against Stage A.

Authority at later verification:

the exact bound producer provenance block plus Stage-A member semantics.

producer_denominator

Authority at production:

len(canonical observed producer population).

Authority later:

exact Stage-B-bound provenance plus structural equality:

recorded denominator == len(recorded population).

runtime_identity

Authority at production:

OBSERVED_CURRENT via verify_runtime_identity().

Expected state:

PRECOMMITTED_STAGE_A runtime.

At later qualification:

producer’s exact runtime record is protected by Stage-B provenance digest and checked against Stage A.

subject_commit

Authority:

PRECOMMITTED_STAGE_A originating from OBSERVED subject HEAD at Stage-A construction.

Pass A re-observes exact subject checkout.

subject_tree

Authority:

Git tree of exact subject commit.

tree_paths_identity

Authority:

D380 canonical derivation from exact subject tree path population.

16.2 PASS_A-only fields

census_identity

Authority:

exact verified/consumed Census byte population from §10.

Compared against PRECOMMITTED_STAGE_A.

history_source_identity

Authority:

D380 §6.8 derivation from actual non-shallow history source.

Compared against PRECOMMITTED_STAGE_A.

16.3 CLASSIFICATION input_binding

pass_a_artifact_sha256

Authority:

EXACT_ARTIFACT_BYTES consumed by classification plus Pass-A Stage-B.

pass_a_stage_a_identity

Authority:

validated Stage A plus exact Pass-A provenance/binding.

pass_a_producer_provenance_digest

Authority:

canonical provenance block parsed from the SAME exact Pass-A bytes and compared with original external Pass-A Stage-B.

16.4 Stage-B fields

artifact_sha256

Authority:

EXACT final regular-file bytes read by binder.

artifact_kind

Authority:

BANKED producer/output role.

stage_a_identity

Authority:

validated precommit Stage A.

producer_component

Authority:

BANKED producer role.

producer_provenance_digest

Authority:

canonical provenance block parsed from the exact artifact bytes read by binder.

16.5 H2 source digest slots

Authority:

exact instrument source bytes at governed instrument root.

16.6 D367 contract digest

Authority:

exact contract bytes compared with BANKED_CONSTANT from D380 §6.3.

16.7 Census aggregate

Authority:

exact governed Census manifest/member construction.

16.8 runtime executable SHA

Authority:

exact current sys.executable bytes.

16.9 stdlib identity

Authority:

independently enumerated H2_PY_STDLIB_V1 object under D380.

Every mapping receives a differential hostile control:

correct source → PASS;

valid wrong source/digest → REFUSE;

neighbouring valid digest substituted → REFUSE;

stale value → REFUSE.

⸻

17. HOLDOUT POPULATION

Before selection:

1. derive frozen subject tree paths;
2. validate canonical path form;
3. reject NFD/non-NFC;
4. require tree uniqueness;
5. validate output paths;
6. require output uniqueness;
7. compare exact sets;
8. compare exact cardinalities;
9. only then select.

Same duplicate on both sides:

REFUSE.

Candidate output determines only whether reconciliation succeeds.

It never defines selection population or seed.

⸻

18. CPYTHON 3.11.15 SOURCE IDENTITY

Target:

v3.11.15

Annotated tag object:

2323bfc729b041c43b1e5e4c5f18c548fc345323

Target commit:

2340a037f7450e70fccfe411e6531afb4d57a312

Target tree:

8c6959bc70b201b477138f00c432a3bb2f1caddd

Expected signer:

Pablo Galindo Salgado

Expected fingerprint:

A035 C8C1 9219 BA82 1ECE A86B 64E6 28F8 D684 696D

Reverify immediately before build.

Mismatch:

STOP.

No automatic version substitution.

⸻

19. CONTROLLED BUILD ENVIRONMENT

This closes DS-V4.4-08.

Before build 1 derive immutable environment fingerprint E.

At minimum E records:

* operating environment/container/rootfs identity where available;
* exact compiler executable SHA;
* compiler version;
* compiler helper SHA;
* assembler SHA/version;
* linker SHA/version;
* ar SHA/version;
* libc identity/version;
* PATH;
* relevant build environment variables;
* configure arguments;
* verified CPython source identity;
* logical install prefix.

No package installation/update or toolchain mutation is permitted between the two builds.

No network-derived dependency may silently enter either build.

Immediately before build 2 independently rederive E.

Require:

E(build1) == E(build2).

Difference:

STOP.

This demonstrates two isolated clean builds under one unchanged controlled environment.

It does not claim environmental independence.

⸻

20. TWO REPRODUCIBLE BUILDS

Separate:

* source worktree;
* build directory;
* staging root.

Same frozen environment E.

Build form remains the recovered bounded form:

./configure --prefix=<P> --without-ensurepip

make

make install DESTDIR=<staging-root>

No PGO unless separately justified/authorised.

Compare:

* executable SHA-256;
* H2_PY_STDLIB_V1.

Mismatch:

STOP and report first divergence.

⸻

21. NATIVE DEPENDENCY RESIDUAL

Native OS dependencies may be recorded for diagnostics.

They are not D380 identity.

Residual boundary remains explicitly recorded:

D380 is Python module/runtime identity,

not whole-OS attestation.

⸻

22. STDLIB FAIL-CLOSED ENUMERATION

Remove silent omission paths.

Governed identity enumeration encountering:

* PermissionError;
* FileNotFoundError;
* disappearing member;
* unreadable member

must REFUSE.

No successful smaller identity.

Can-fail propagates through:

build_stdlib_identity

→ _stdlib_identity

→ build_runtime

→ verify_runtime_identity

→ actual producer/qualifier.

⸻

23. D379 CHILD-LAUNCH TOPOLOGY

The earlier “outer parent observer” statement was too narrow.

Current topology includes:

capture process

→ D379 control child (d379_controls.py --child)

→ individual H2 test subprocesses.

Separately:

capture process

→ cal_fixtures.py

→ its historical fail-old subprocess.

v4.5 distinguishes these.

23.1 governed F13 population

F13 governs Python subprocess launches implemented by:

d379_controls.py

in every mode in which that file executes:

* capture parent;
* --child control process;
* ordinary control execution.

Each d379_controls.py process installs its runtime observer before its first governed child launch.

Therefore the matrix child’s own H2 test launches are observed by the matrix child.

23.2 cal_fixtures subtree

cal_fixtures.py remains a separate historical fixture subject.

Its internal pre-repair subprocess is recorded separately and is NOT represented as part of F13’s D379-control-launch denominator.

It is not silently omitted:

the closeout records cal_fixtures as a separate subject, and the static census separately reports its internal Python launch.

No common-source corroboration claim is made.

⸻

24. STATIC LAUNCH GRAMMAR

The analyser is intentionally bounded to D379 control source.

Recognised launch forms:

* direct subprocess.run;
* subprocess.Popen;
* statically resolvable imported aliases;
* named governed launcher;
* direct os.exec*;
* os.spawn*;
* os.posix_spawn*;
* os.system;
* multiprocessing process construction;
* direct sys.executable;
* literal env/python;
* literal hard-coded Python.

Potentially launch-capable ambiguous forms include:

* eval;
* exec;
* dynamic importlib;
* dynamic getattr on launch-capable modules/objects;
* dynamic callable indirection whose target cannot be resolved;
* generated shell/Python command whose executable cannot be statically established.

Presence of such a construct in the D379 control harness:

UNRESOLVED.

Gate refuses completeness.

24.1 false-positive handling

A refusal is NOT overridden ad hoc.

Kai inspects the exact construct.

Resolution requires one of:

1. refactor harness within authorised B4 so the launch relation is statically explicit; or
2. formally extend the declared grammar and hostile controls.

No whitelist-by-convenience.

This resolves DS-V4.4-10.

⸻

25. RUNTIME LAUNCH OBSERVATION

Each relevant d379_controls.py process installs Python audit observation before any governed child launch.

Observation claim is limited to direct process-creation events emitted from that process.

Record:

* executable;
* argv;
* cwd where available;
* mapping to static launch site.

Runtime-observed direct launch absent from static population:

REFUSE.

Static launch site exercised with unexpected command topology:

REFUSE.

No claim is made about arbitrary C code launching processes outside Python’s audit surface.

No claim is made that the outer capture process alone observes grandchildren.

This resolves DS-V4.4-09 by matching observation to the actual process topology.

⸻

26. E7 EXTERNAL BUILD-EVIDENCE POPULATION

E7 concerns the external durable storage of the TWO interpreter build transcripts.

Therefore precommitted build-log object population is:

N = 2

Specifically:

1. Build A complete combined transcript.
2. Build B complete combined transcript.

Each transcript contains the complete governed driver record for that build including:

* environment fingerprint reference;
* configure command/output;
* make command/output;
* install command/output;
* stdout;
* stderr;
* actual command return codes.

Other D379 artifacts are not falsely counted inside E7’s build-log capacity claim.

⸻

27. E7A CAPACITY

Before either build freeze:

per-object demonstrated capacity:

T

aggregate demonstrated capacity:

U

Require:

U >= 2 × T.

Write two independent synthetic objects each of exactly T through the actual selected storage/transport mechanism.

Synthetic corpus includes:

* LF;
* CRLF;
* non-ASCII;
* binary-safe envelope edge cases as applicable.

Destinations are unique content-addressed/build-addressed paths.

Use atomic create-if-absent where supported.

If the storage mechanism cannot provide atomic create-if-absent, prove an equivalent write-once protocol before build.

Independently read both objects back.

Exact byte count and SHA must match.

Failure:

NO BUILD.

⸻

28. E7B ACTUAL LOGS

For Build A and Build B:

* one complete combined transcript each;
* size <= T;
* combined size <= U;
* unique destination;
* no overwrite;
* independently retrieved;
* exact SHA;
* exact byte count.

Failure:

S9.

No manual fallback.

No late upload.

⸻

29. PRE-CAPTURE FIXITY F

Current hand-maintained subject_digests() is abolished as the assurance denominator.

Its current nine-file list is known incomplete.

29.1 derivation-rule location

The complete freeze-population derivation mechanism lives inside:

build_evidence/d379_controls.py.

No external config controls the population.

This answers DS-V4.4-12.

29.2 derivation inputs

The F-version of the rule consumes:

* stage_identity.H2_SOURCES;
* the exact governed Census manifest;
* manifest-declared Census member population;
* fixed D367 contract path + BANKED D380 expected digest;
* local dependency closure derivation;
* D379 control source itself.

29.3 self-inclusion

d379_controls.py is itself a population member.

This is not hash self-reference.

Its source does not contain its own computed digest.

At F its bytes are immutable Git content.

The F-version algorithm derives a set that includes its own source path and hashes those already-fixed source bytes.

29.4 no mutable external path list

There is no sidecar config containing the population.

Any future external population config would itself become an F dependency and would require explicit plan revision.

⸻

30. CURRENT CAPTURE OUTPUT POPULATION

Current capture writes:

tracked durable output 1:

D379_CONTROLS.txt

tracked durable output 2:

D379_CLOSEOUT.txt

The state exchange file created under a temporary directory is ephemeral process state and not a tracked evidence output.

After capture:

tracked F→capture diff must equal exactly those two files.

Anything else:

STOP.

⸻

31. D367 CONTRACT EXPECTED DIGEST SOURCE

The expected digest:

0ce5792ed72e6e7051ecc050664490899a847d01de2f62cff564f460d46800bb

comes from banked D380 §6.3.

D380 fixes:

path:

kai-pm/H2_REPAIR_CONTRACT_D367.md

and that exact SHA-256.

Implementation may contain the constant required to enforce the banked value.

The constant is implementation, not authority.

At fixity review Kai checks the implementation constant against banked D380.

Changing code + constant together cannot redefine the contract because D380’s banked bytes remain the authority.

This resolves DS-V4.4-11.

⸻

32. HOSTILE CONTROL STANDARD

Every material repair receives:

* exact subject proof;
* known positive;
* known negative;
* boundary condition;
* expected predicate;
* real process return code where process behaviour is claimed;
* population denominator;
* mutation/can-fail test.

A red process is insufficient unless the intended predicate caused the refusal.

Required families include:

* Stage-A nested/full rederivation;
* instrument-vs-subject root substitution;
* Census preload attack;
* Census same-byte mutation attack;
* Census alternate-sys.path attack;
* history identity mismatch;
* producer runtime mismatch;
* qualifier runtime mismatch;
* Q1a-7 unchanged original binding;
* Q1a-9 semantic authority;
* per-slot wrong-authority substitution;
* F12 same-duplicate population;
* NFD path rejection;
* stdlib unreadable-member failure;
* D379 launch grammar;
* runtime unmapped launch;
* fixity denominator;
* capture-diff closure.

⸻

33. REPAIR MUTATION SURFACE

No new tracked source file.

Authorised source surface, if Dainius later grants implementation:

* stage_identity.py
* passa.py
* run_h2_v12.py
* qualify.py
* holdout.py
* build_evidence/d379_controls.py

Final capture outputs only:

* D379_CONTROLS.txt
* D379_CLOSEOUT.txt

Outside surface includes:

* cal_fixtures.py;
* classify.py;
* envelope.py;
* ontology.py;
* subjectbind.py;
* Census package;
* D367 contract;
* governance records except separately authorised Part D.

Need to modify one:

STOP and return for authority.

⸻

34. EXECUTION SEQUENCE

1. Freeze v4.5.
2. DeepSeek reviews THIS COMPLETE v4.5.
3. Kai independently reconciles.
4. No unresolved design blocker permitted.
5. Dainius accepts/refuses plan.
6. Dainius separately resolves Part D.
7. If accepted, commit G.
8. Derive B0 closure at eb52f73 and G.
9. Require exact equality.
10. Create repair branch from G.
11. Confirm lineage and clean tree.
12. Freeze build environment fingerprint E.
13. E7a prove N=2, T and U storage capability.
14. Reverify CPython signed source identity.
15. Build A.
16. Reverify environment E unchanged.
17. Build B.
18. E7b preserve and re-read both complete transcripts.
19. Compare executable/stdlib reproducibility.
20. Measure resulting interpreter against D380/D385.
21. Require actual D380-compliant known-positive runtime.
22. Begin D379 source repair.
23. Implement complete Stage-A production construction.
24. Implement full production Stage-A rederivation.
25. Separate instrument/subject/history/Census contexts.
26. Implement exact-byte Census loader before first Census import.
27. Repair producer population root derivation.
28. Complete external Stage-B transport.
29. Complete Pass-A→classification→qualification binding.
30. Add qualifier own-runtime verification.
31. Repair complete per-slot authority controls.
32. Repair holdout independent uniqueness/canonicality.
33. Repair stdlib fail-closed enumeration.
34. Repair bounded launch grammar/launcher/observation.
35. Execute full hostile matrix.
36. DO NOT CAPTURE.
37. Produce fixity commit F.
38. Kai independently reviews exact F, freeze population and expected predicates.
39. Dainius separately authorises exactly one capture against F.
40. Require HEAD == F.
41. Require clean tree.
42. Require exact F-derived governed identities.
43. Run one capture.
44. Require tracked F→capture diff exactly:
    D379_CONTROLS.txt
    D379_CLOSEOUT.txt
45. STOP FOR KAI.

No automatic real Stage A.

No candidate.

No holdout.

⸻

35. STOP CONDITIONS

S1
Interpreter prerequisite fails → STOP.

S2
Repair weakens D380/D385 → STOP.

S3
Fail-old test fails to construct its named historical subject → STOP.

S4
Need mutation outside authorised surface → STOP.

S5
HELD-heavy matrix is not closeout → STOP.

S6
One authorised capture only; then STOP.

S7
External build evidence store unavailable → NO BUILD.

S8
Baseline closure or lineage equivalence fails → STOP.

S9
E7 capacity/fidelity/write-once/read-back fails → STOP.

S10
Required original Stage-B binding unavailable, stale in the wrong way, or regenerated after hostile tampering → REFUSE.

S11
Capture subject is not exact reviewed F → STOP.

S12
CPython source/signature identity fails → STOP.

S14
Unresolved local dependency closure → STOP.

S15
Instrument/subject/history/Census context collapse → REFUSE.

S16
Supplied production Stage A != independently rederived Stage A → REFUSE.

S17
D379 harness contains unresolved launch-capable construction → REFUSE completeness claim.

S18
Pass A cannot prove the Census module objects it USED executed the exact verified Census bytes → REFUSE.

S19
A governed Census module is present in sys.modules before the verified-byte Census loader establishes the boundary → REFUSE.

S20
Build environment fingerprint differs between reproducibility builds → STOP.

⸻

36. EXPLICIT RESIDUAL BOUNDARIES

R1 same-principal malicious rewrite

Outside banked D379’s structural Q1a/Q1a-7 predicate.

No second principal is claimed.

R2 native OS layer

Explicitly excluded from D380’s Python-module identity contract.

R3 launch observation

Bounded to declared D379 control-process source grammar and direct launches from each relevant D379-control process.

Not universal host monitoring.

R4 historical producer

Qualifier cannot reobserve a terminated producer.

Producer observes itself at production.

Stage-B binds that exact provenance.

Qualifier checks bound producer evidence and independently verifies its own runtime.

R5 cal_fixtures historical subprocess

Separate subject.

Not silently treated as part of F13’s D379-control-launch denominator.

Its internal subprocess is reported separately.

⸻

37. FINAL DEEPSEEK REVIEW MANDATE

Review this COMPLETE v4.5.

Do not restart from v4.2 assumptions already disproved by banked authority.

The remaining question is:

Can any mechanical predicate in v4.5 pass while the exact invariant it claims to establish is false?

Mandatory focus:

1. Census preload/order/same-byte execution.
2. canonical instrument-root derivation.
3. per-slot authority-source enumeration.
4. Stage-B regular-file/exact-byte chain.
5. D379 structural threat-model compliance.
6. D380 native boundary compliance.
7. N=2 / T / U build-log storage semantics.
8. frozen build environment E.
9. actual process-tree launch observation.
10. fixity derivation-rule self-inclusion.
11. D367 contract expected-value authority.
12. Q1a-7 subject construction.

Do not demand a stronger adversary than the banked contract unless you quote the banked clause requiring it.

For each remaining finding provide:

ID

Classification:
BLOCKER / MAJOR / MINOR / QUESTION

Evidence status:
PLAN-INTERNAL / SUPPLIED-EVIDENCE / UNVERIFIED / OUT-OF-SCOPE

Exact claim

Concrete counterexample

Can predicate pass while invariant false?

Minimum correction

At end state exactly:

NO DESIGN BLOCKER FOUND

or:

DESIGN BLOCKER REMAINS: <IDs>

No code.

No implementation authority.

No D-number.

No merge recommendation.

END OF D379 REPAIR PLAN v4.5