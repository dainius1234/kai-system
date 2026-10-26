DEEPSEEK — FINAL ADVERSARIAL CLOSURE REVIEW OF D379 v4.5

Baseline: eb52f73fa6485534ca7e28a42055861c69e94cc4
Plan: v4.5, complete text as supplied.
Authority: none. Findings are hypothesis input; Kai reconciles.

⸻

EXECUTIVE ADVISORY VERDICT

NO DESIGN BLOCKER FOUND.

Every DS-V4.4 finding (01–13) is materially closed by v4.5. The four-context separation, the Census same-byte loading with pre-registration, the instrument-root derivation rule, the per-slot authority enumeration, the lstat/regular-file rule, the quoted D379/D380 clauses, the E7 N=2/T/U storage semantics, the environment fingerprint E, and the cal_fixtures topology distinction all respond to the corresponding v4.4 findings with mechanical rules rather than restated principles.

Three MAJORs and six MINORs remain. None is a design blocker. Two of the MAJORs require a repository evidence request to confirm the classification (they may become BLOCKERs only if the evidence shows what they hypothesise).

⸻

FINDING REGISTER

DS-V4.5-01 — Build isolation does not enforce absence of network

Classification: MAJOR
Evidence status: PLAN-INTERNAL
Exact claim attacked: §19: “No network-derived dependency may silently enter either build.”

Counterexample: E is captured before build 1 and re-derived before build 2. E records compiler/toolchain/libc SHAs. If a build step fetches a header or object from network during build 1, and E does not capture that fetch, the build succeeds; build 2 fetches the same, E still matches, and no test detects the network input. If between the two builds the network-fetched object changes but its hash is not in E, E remains equal.

Can predicate pass while invariant false: Yes — the predicate is “E(build1) == E(build2)”, which is a claim about the captured fields, not about the absence of network inputs. Silent network input is invisible to E unless E’s field population covers every fetched object.

Minimum correction: State one of:

1. Network is disabled during both builds (mechanism: no-network container, no DNS, no outbound socket).
2. Network activity is recorded as part of the build evidence and checked for changes between builds.
3. E’s field population is extended to include every object the build could fetch.

If the CPython build is self-contained (no network fetch is required by configure/make), state that explicitly. Otherwise the isolation claim is not established.

⸻

DS-V4.5-02 — §22 stdlib fail-closed propagation is not verified at callers

Classification: MAJOR
Evidence status: PLAN-INTERNAL
Exact claim attacked: §22: “must REFUSE. No successful smaller identity. Can-fail propagates through…”

Counterexample: build_stdlib_identity is repaired to refuse on PermissionError. But a caller wraps the call in a try/except that logs and continues with the last-known-good identity. The repair is inert: verify_runtime_identity receives a valid-looking identity block and passes.

The plan lists the call chain build_stdlib_identity → _stdlib_identity → build_runtime → verify_runtime_identity → producer/qualifier but does not state that every call site of these functions has been inspected for swallowing.

Can predicate pass while invariant false: Yes — the refusal fires, is caught by a caller, and the pipeline continues with stale state.

Minimum correction: Enumerate every call site of build_stdlib_identity, _stdlib_identity, build_runtime, and verify_runtime_identity in the six B4 files. For each, state whether a surrounding try/except swallows the refusal. Any swallowing caller must be removed or must re-raise. Add a can-fail control that plants an unreadable member and confirms the pipeline refuses — not just that the lowest-level function raises.

⸻

DS-V4.5-03 — §11 producer-root repair location unspecified; potential B4 scope violation

Classification: MAJOR
Evidence status: UNVERIFIED
Exact claim attacked: §11: “Current classify_origin() derives: H2 root and Census root from caller-supplied repo_root. That is a verified current defect. Repair the interface.”

Counterexample: classify_origin() is named in §11 but its file location is not stated. If classify_origin() lives in classify.py, and classify.py is not in B4 (§33 excludes it), then the §11 repair cannot be implemented within the authorised mutation surface. The plan then has an internal contradiction: it authorises a repair it cannot carry out.

If classify_origin() is in stage_identity.py or passa.py, B4 is sufficient.

Can predicate pass while invariant false: N/A — this is a scope sufficiency question.

Minimum correction: State the file and function containing classify_origin(). If it is in a non-B4 file, either:

· add that file to B4 with justification, or
· state that the repair is entirely at the caller boundary (passa.py/runs that pass repo_root) without modifying the derivation logic in the callee.

Falsifiable evidence request: git grep -n “def classify_origin” eb52f73 – kai-pm/house_in_order_h2_v13/. State the file.

⸻

DS-V4.5-04 — §23.2 cal_fixtures static census mechanism is inconsistent with §24

Classification: QUESTION
Evidence status: PLAN-INTERNAL
Exact claim attacked: §23.2: “the static census separately reports its internal Python launch.” §24: “The analyser is intentionally bounded to D379 control source.”

Counterexample: §24 restricts the static analyser to d379_controls.py. §23.2 claims the static census reports a launch inside cal_fixtures.py. If the analyser is bounded to d379_controls.py, how is cal_fixtures.py’s launch enumerated?

Minimum correction: State the mechanism. Either §24’s boundary is extended to include cal_fixtures.py’s launch surface (with an explicit statement of what changes), or §23.2’s claim is qualified as “reported by a separate, non-F13 mechanism.”

⸻

DS-V4.5-05 — §34 step 21 assumes v3.11.15 will satisfy D385 known-positive

Classification: QUESTION
Evidence status: UNVERIFIED
Exact claim attacked: §1 and §34 step 21: “establish a D380-compliant known-positive interpreter.”

Counterexample: The plan builds CPython 3.11.15 and requires the result be known-positive per D385. If D385’s known-positive criteria are not satisfied by a stock 3.11.15 build — e.g., because they require a specific build configuration or patch set — the plan has no fallback path.

Minimum correction: State the exact D385 known-positive criteria and confirm whether stock v3.11.15 satisfies them. If not, state what additional configuration is required.

Falsifiable evidence request: Quote D385’s known-positive clause from kai-pm/DECISIONS.md at its bank commit.

⸻

DS-V4.5-06 — §35 S19 may over-refuse on legitimate H2 module-import chains

Classification: QUESTION
Evidence status: PLAN-INTERNAL
Exact claim attacked: §35 S19: “A governed Census module is present in sys.modules before the verified-byte Census loader establishes the boundary → REFUSE.”

Counterexample: If passa.py at module import time transitively imports something that itself imports a governed Census module — for example, an H2 module that itself imports docgraph for unrelated reasons — then docgraph is present in sys.modules before the loader runs. S19 refuses a legitimate flow.

Minimum correction: Either:

· state that no H2 module transitively imports a governed Census module (verified by import-graph inspection), so the precondition always holds; or
· qualify S19: only refuse if the pre-existing entry was NOT installed by the loader in this same process.

⸻

DS-V4.5-07 — §10.4 module state after exec failure unspecified

Classification: MINOR
Evidence status: PLAN-INTERNAL
Exact claim attacked: §10.4: pre-register module, then compile, then exec.

Counterexample: If the exec raises, sys.modules holds a partially-initialized module. A subsequent import of that name returns the partial module.

Minimum correction: On exec failure, remove the module from sys.modules before propagating the exception. State this as part of §10.4.

⸻

DS-V4.5-08 — §16.1 stage_a_identity wording is ambiguous

Classification: MINOR
Evidence status: PLAN-INTERNAL
Exact claim attacked: §16.1: stage_a_identity authority: PRECOMMITTED_STAGE_A; mechanism: “recompute from supplied canonical Stage-A descriptor.”

Counterexample: “Supplied canonical Stage-A descriptor” reads as the descriptor being verified. If the expected value is recomputed from the descriptor being verified, the check is internal consistency only.

Minimum correction: Reword to: “recompute from the independently rederived canonical Stage-A descriptor produced by §9; compare to the supplied descriptor’s identity.” The independent authority is §9’s rederivation.

⸻

DS-V4.5-09 — §23.1 observer installation timing unspecified

Classification: MINOR
Evidence status: PLAN-INTERNAL
Exact claim attacked: §23.1: “Each d379_controls.py process installs its runtime observer before its first governed child launch.”

Counterexample: If the observer is installed inside a function that is called after some children launch, the requirement is nominally satisfied but practically violated.

Minimum correction: State that the observer is installed at module import time, before any function that can launch a child is callable.

⸻

DS-V4.5-10 — §26 “complete transcript” curation unspecified

Classification: MINOR
Evidence status: PLAN-INTERNAL
Exact claim attacked: §26: “Each transcript contains the complete governed driver record for that build.”

Counterexample: “Complete” is a claim about bytes. If the driver writes a curated summary rather than bytes-as-produced, the transcript omits fields (timestamps, transient state) that a raw transcript would contain.

Minimum correction: State whether the transcript is bytes-as-produced by the driver, or a curated record. If curated, state what is included and what is omitted.

⸻

DS-V4.5-11 — §29.2 frozen non-code inputs are not listed in F’s population

Classification: MINOR
Evidence status: PLAN-INTERNAL
Exact claim attacked: §7.5 lists “explicitly frozen non-code input” as a valid B0 dependency; §29.2 does not include such inputs in F’s population.

Counterexample: If a B0 closure depends on a frozen non-code input (e.g., a data file), F does not verify that input’s bytes remain unchanged between F and capture.

Minimum correction: Add “every explicitly frozen non-code input from §7.5” to §29.2’s population list, or state that no such inputs were found during B0.

⸻

DS-V4.5-12 — §30 temporary state-exchange file persistence unspecified

Classification: MINOR
Evidence status: PLAN-INTERNAL
Exact claim attacked: §30: “The state exchange file created under a temporary directory is ephemeral process state and not a tracked evidence output.”

Counterexample: If the temp file persists after capture, it is a non-tracked side effect. A future reviewer cannot tell whether it existed because the plan doesn’t state whether the capture deletes it.

Minimum correction: State whether the temp file is deleted on capture success and on failure; whether any diagnostic residue remains; and whether its absence is checked.

⸻

DEPENDENCY-ORDER ATTACK

The §34 sequence is internally consistent. No cycle detected.

Two clarifications worth noting:

· Step 21 (D385 known-positive interpreter) is currently the terminal gate on the build phase; if v3.11.15 does not satisfy D385 (DS-V4.5-05), the plan has no path forward without reopening the interpreter decision.
· Step 43 (one capture) is correctly terminal; S6 and S11 correctly guard it.

⸻

PROXY-SUBSTITUTION REGISTER

Claim Proxy Actual invariant
§19 no-network E equality No network input entered either build
§22 fail-closed Lowest-level raise Whole pipeline refuses
§11 root repair “Interface” File and function explicitly in B4
§23.2 cal_fixtures launch “Reported separately” Analyser scope explicitly covers it
§26 complete transcript Curated record Bytes-as-produced
§29.2 F population Non-code inputs unlisted All B0 dependencies in F

Each should be stated in the plan as “the mechanism establishes X; it does not establish Y” where X ≠ Y.

⸻

CLOSING

NO DESIGN BLOCKER FOUND.

The plan can advance to an implementation-authority decision once DS-V4.5-03 is resolved by naming the file containing classify_origin() and confirming it is within B4 (or adding it). DS-V4.5-01, DS-V4.5-02 and DS-V4.5-04 through DS-V4.5-12 can be closed either by specification or by explicit residual-risk register entries.

Blocker IDs (none confirmed): —

MAJOR IDs requiring closure before implementation: DS-V4.5-01, DS-V4.5-02, DS-V4.5-03.

No code. No implementation authority. No D-number. No merge recommendation.