"""What each check in `scripts/security/` is, and what it claims.

Adding a check today means remembering three places — the `policy-check`
target, a CI workflow, and a suite proving it can fail — with nothing
cross-checking them. Two of twelve were already inconsistent when this
file was written. That is the `PUBLIC_ROUTES` problem: `dashboard/app.py`
solved it by making every route declare its own scope, so a new
unauthenticated route is *visibly* unsafe rather than invisibly so.

This is the same declaration for the watching layer. A check that is not
declared here fails `check_gate_registry.py`, so a new one is visibly
unregistered rather than quietly unwatched.

**There is deliberately no free-text `exception` field.** The operator's
rubric governs this file:

> If it can't be encoded so the system enforces it, it's not an
> exception — it's debt.

So every departure from the norm is a typed field the meta-check reads
and reports on, never a note explaining why a rule does not apply:

  - `kind=REPORT` says *this is not a gate* — the reason it is absent
    from `policy-check` is recorded, not inferred from its absence.
  - `optional_inputs` says *this input may legitimately be missing* —
    per path, rather than a blanket `continue` that makes every input
    optional.
  - `probe=False` requires `probe_skip_reason`, so a check that is too
    expensive to run says so out loud instead of being silently skipped.
  - `pending_wiring` names the step that will wire a gate up. A gate that
    is not yet enforced is reported on every run until it is.

Each of those is visible in the meta-check's output every time it runs.
None of them can be satisfied by someone knowing why.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

# A gate fails the build when it finds a violation. A report informs and
# must *not* be wired as a gate — a report that gates makes the build
# depend on information nobody promised was actionable.
GATE = "gate"
REPORT = "report"

# Trigger classes -- see Gate.trigger_class.
CONTINUOUS = "CONTINUOUS"
SENTINEL_AUTHORISED = "SENTINEL_AUTHORISED"

COMPOSE_FILES = (
    "docker-compose.full.yml",
    "docker-compose.minimal.yml",
    "docker-compose.sovereign.yml",
)


@dataclass(frozen=True)
class Gate:
    module: str
    kind: str
    summary: str

    # I-1 — what this check reads. A required input that is missing means
    # the check cannot answer its question, which is a failure, not a
    # pass. Optional inputs are declared per path so that "this one may
    # be absent" never generalises into "absence is fine".
    inputs: Tuple[str, ...] = ()
    optional_inputs: Tuple[str, ...] = ()

    # I-2 — a regex its output must match, naming how much it inspected.
    denominator: Optional[str] = None
    probe: bool = True
    probe_skip_reason: Optional[str] = None

    # I-3 — the suite that injects a violation and asserts it fires.
    proven_by: Optional[str] = None

    # I-7 — a ratchet must prove its instrument still measures.
    #
    # A gate that bounds a MAXIMUM ("this count may not rise") is
    # satisfied by zero, and zero is exactly what a detector that has
    # stopped detecting reports. The bound is enforced correctly and the
    # silence is the danger. On 2026-08-05 a tokenising bug took the
    # hygiene survey's `clients` from 16 to 0 and its adoption count from
    # 149 to 0, and the gate passed — it was doing precisely what a
    # ratchet does.
    #
    # `ratchet=True` means the verdict rests on a stored baseline, and
    # `calibrated_by` must then name the suite that points the detector
    # at input whose answer is known *before* pointing it at the
    # repository. A historical baseline says "this is what we saw last
    # time"; it cannot say whether last time's instrument was working.
    #
    # Gates bounding a MINIMUM do not need this — a blinded detector
    # reports zero, zero is below the floor, and the gate fires. Recorded
    # anyway where it applies, because "safe by direction" is a property
    # worth stating rather than rediscovering.
    ratchet: bool = False
    calibrated_by: Optional[str] = None

    # I-4 — where it is enforced. Discovered independently and compared.
    in_policy_check: bool = False
    in_workflows: Tuple[str, ...] = ()
    pending_wiring: Optional[str] = None

    # HOW THE WORKFLOWS IN `in_workflows` ARE TRIGGERED, machine-readably.
    #
    # Rule 9 says a gate's trigger conditions are part of the gate. The
    # obvious reading -- "every input must appear in the workflow's
    # paths: filter" -- is right for a CONTINUOUS gate and WRONG for a
    # one-shot authorised experiment, where it would mean that editing an
    # analyser DISPATCHES the experiment. That is a manufactured evidence
    # run, and rule 10 already says evidence-admission rules do not
    # authorise evidence production.
    #
    # Two classes, so a future paths-coverage detector (finding #50) can
    # tell them apart FROM THE REGISTRY instead of inferring an exception
    # from a comment it cannot parse:
    #
    #   CONTINUOUS          every enforcement input must trigger it.
    #   SENTINEL_AUTHORISED input changes must NOT execute it. An explicit
    #                       sentinel triggers it, and the run revalidates
    #                       its frozen inputs and calibration before
    #                       reaching the subject.
    #
    # Default CONTINUOUS: the safe direction is over-triggering a check,
    # never silently exempting one. (D280)
    trigger_class: str = "CONTINUOUS"

    # The register entry this check's own defects are tracked under.
    findings: Tuple[str, ...] = field(default_factory=tuple)


# All six now fail closed on a missing input and report the same
# denominator, via `gate_inputs.require` / `count_services`. The unit is
# service definitions rather than compose files: "3 files" is 3 whatever
# happens, and a denominator that cannot move cannot reveal a scanner
# that has gone blind.
_COMPOSE_GATE = dict(
    kind=GATE,
    inputs=COMPOSE_FILES,
    denominator=r"inspected: \d+ service definitions",
    in_policy_check=True,
    in_workflows=("policy-checks.yml",),
    findings=("KAI-GATE-001", "KAI-GATE-002", "KAI-GATE-003"),
)

REGISTRY: Tuple[Gate, ...] = (
    # ── The eight pre-incident compose gates ─────────────────────────
    # None reports a denominator; none has ever been observed failing.
    # All eight skip absent inputs. They are the retrofit backlog.
    # Long-form port syntax and malformed shapes both handled. GATE-010.
    Gate(module="check_port_bindings",
         summary="only the dashboard may publish a port, on 127.0.0.1",
         proven_by="scripts/test_compose_gates.py",
         **_COMPOSE_GATE),
    Gate(module="check_default_profiles",
         proven_by="scripts/test_compose_gates.py",
         summary="no dangerous service in the default profile",
         **_COMPOSE_GATE),
    # Rewritten from a denylist of nine guessable words into a rule:
    # a secret may be referenced, never valued. KAI-GATE-007.
    # Found auditing G-07's closure: the record said the token was
    # "wired into 8 service blocks across all three compose profiles";
    # it was 8 in total, split 3/1/4, and `executor` — which runs
    # POST /execute — had none in `full` or `sovereign`. Fail-closed, so
    # not an open endpoint: a stack whose tool execution answers 503 to
    # every call, with the symptom appearing at the caller.
    Gate(module="check_service_tokens",
         kind=GATE,
         summary="every auth-enforcing service is given KAI_SERVICE_TOKEN",
         inputs=("docker-compose.full.yml", "docker-compose.sovereign.yml",
                 "docker-compose.minimal.yml"),
         denominator=r"inspected: \d+ auth-enforcing service definitions",
         proven_by="scripts/test_service_tokens.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-024",)),

    Gate(module="check_secret_fallbacks",
         kind=GATE,
         summary="a secret may be referenced, never given a value here",
         inputs=COMPOSE_FILES,
         denominator=r"inspected: \d+ environment values",
         proven_by="scripts/test_secret_gates.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-007",)),
    # The "explicit networks assignment" rule was `pass`. GATE-012.
    Gate(module="check_network_zones",
         summary="network zone segmentation holds",
         proven_by="scripts/test_compose_gates.py",
         **_COMPOSE_GATE),
    Gate(module="check_turbovec_writers",
         proven_by="scripts/test_compose_gates.py",
         summary="TurboVec has a single writer",
         **_COMPOSE_GATE),
    # `ALLOWED_RESTART` was declared and never referenced. KAI-GATE-008.
    Gate(module="check_restart_recovery",
         summary="restart and recovery stay contained",
         proven_by="scripts/test_secret_gates.py",
         **_COMPOSE_GATE),
    # Denylist of four words became a rule: versioned or digest. GATE-011.
    # `--verify-exists` runs in core-tests.yml as well, because a pinned
    # tag and an *existing* tag are different properties and only the
    # second one needs a network to check. `ollama/ollama:0.6` satisfied
    # this gate for months and had been withdrawn from Docker Hub.
    Gate(module="check_image_tags",
         summary="every image tag is versioned or a digest, and resolves",
         proven_by="scripts/test_compose_gates.py",
         **{**_COMPOSE_GATE,
            "in_workflows": ("policy-checks.yml", "core-tests.yml"),
            "findings": _COMPOSE_GATE.get("findings", ()) + ("KAI-GATE-028",)}),
    # Rewritten directionally and given a denominator, a failure suite
    # and fail-closed inputs — the first of the eight to be retrofitted.
    Gate(module="check_compose_drift",
         kind=GATE,
         summary="hardening may differ between profiles, but only upward",
         inputs=COMPOSE_FILES,
         denominator=r"inspected: \d+ service definitions",
         proven_by="scripts/test_compose_drift.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-002", "KAI-GATE-003")),

    # ── The four built or repaired after an incident ─────────────────
    Gate(module="check_architecture_rules",
         kind=GATE,
         summary="the 15 §15 architecture rules (A-01)",
         inputs=("common/contracts", "common/policy_bridge"),
         denominator=r"rules accounted for:\s*\d+/\d+",
         proven_by="scripts/test_architecture_rules.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-001",)),

    Gate(module="hygiene_survey",
         kind=GATE,
         summary="repo-wide HTTP/time hygiene ratchet (H-5)",
         inputs=("scripts/security/hygiene_baseline.json",),
         denominator=r"\d+ of \d+ services carry none of these",
         proven_by="scripts/test_hygiene_gate.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         # KAI-GATE-022 opened 2026-08-05: widening the survey from
         # service entry points to every first-party module made 16
         # per-request httpx clients and 1 unbounded body visible in
         # library code. Same class as H-2/H-3, which were closed at the
         # narrower scope — so those closures were true of what was
         # measured and not of the repository. Registered rather than
         # absorbed into the baseline silently (Rule 7).
         ratchet=True,
         calibrated_by=(
             "scripts/test_hygiene_gate.py — every detector fires on a known positive and ignores prose describing the same pattern; the denominator is DETECTORS, so a new detector without a sample fails"),
         findings=("KAI-GATE-001", "KAI-GATE-022")),

    Gate(module="check_dashboard_findings",
         kind=REPORT,
         summary="revalidates all 96 KAI-DASH findings — a status report",
         inputs=("dashboard/app.py", "common/dashboard_auth.py"),
         denominator=r"Coverage: all \d+ findings accounted for",
         proven_by="scripts/test_dashboard_findings.py",
         in_policy_check=False,
         in_workflows=(),
         findings=("KAI-GATE-001", "KAI-GATE-005")),

    Gate(module="check_assertion_floors",
         kind=GATE,
         summary="assertion-count ratchet (A-02)",
         inputs=("scripts/security/assertion_floors.json",),
         denominator=r"\d+ suites, \d+ assertions",
         probe=False,
         probe_skip_reason="runs `make test-uh` — minutes, not seconds; "
                           "probed by scripts/test_assertion_floors.py "
                           "against synthetic logs instead",
         proven_by="scripts/test_assertion_floors.py",
         # Bounds a MINIMUM, so a blinded counter reports zero, zero is
         # below every floor, and the gate fires. Declared anyway: the
         # calibration proves a suite producing no count is reported as
         # vanished rather than silently dropped.
         ratchet=True,
         calibrated_by=(
             "scripts/test_assertion_floors.py — a suite that produces no "
             "count is reported as vanished rather than passing, and an "
             "emptied floors file leaves every suite unrecorded"),
         in_policy_check=False,
         in_workflows=("unified-hunter.yml",)),

    # A-04c: every CI step that can pass without doing its job must be
    # declared with a reason, an owner and a review date. Also refuses a
    # workflow that does not parse — running nothing is indistinguishable
    # from having no failures.
    Gate(module="check_ci_tolerations",
         kind=GATE,
         summary="no undeclared CI suppression; every workflow parses",
         inputs=(".github/workflows",),
         denominator=r"inspected: \d+ workflow lines",
         proven_by="scripts/test_ci_tolerations.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-015", "KAI-GATE-016")),

    # A test that is never called asserts nothing, and nothing fails to
    # draw attention to it. Calibrated against known-good suites before
    # it reports — three earlier versions of this detector produced
    # 1,555, 1,813 and 54 confident wrong answers.
    # A workflow can be valid YAML and valid bash and still die on a
    # string only jq ever parses. `drift-detector.yml` failed all 15 of
    # its scheduled runs for three and a half months on one, and no gate
    # here could see it. KAI-GATE-032.
    Gate(module="check_workflow_filters",
         kind=GATE,
         summary="every jq filter embedded in a workflow compiles",
         inputs=(".github/workflows",),
         denominator=r"inspected: \d+ embedded jq filter",
         proven_by="scripts/test_workflow_filters.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-032",)),
    # Unbounded content, constant delimiter. `friday-cleanup.yml` failed
    # on it, and both YAML and bash accept it. KAI-GATE-034.
    Gate(module="check_workflow_outputs",
         kind=GATE,
         summary="no $GITHUB_OUTPUT heredoc is bounded by a constant",
         inputs=(".github/workflows",),
         denominator=r"inspected: \d+ \$GITHUB_OUTPUT",
         proven_by="scripts/test_workflow_outputs.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-034",)),
    # One character stopped every image build and thirteen steps after
    # it, in a Dockerfile nothing had ever parsed. KAI-GATE-035.
    Gate(module="check_dockerfile_flags",
         kind=GATE,
         summary="Dockerfile instruction flags are hyphenated, not underscored",
         inputs=("document-parser/Dockerfile",),
         denominator=r"inspected: \d+ instruction flag",
         proven_by="scripts/test_dockerfile_flags.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-035",)),
    # 22 of 52 Dockerfiles were never built by CI, and one held a parse
    # error that stopped thirteen steps. KAI-GATE-036.
    Gate(module="check_dockerfile_coverage",
         kind=GATE,
         summary="every Dockerfile is built by a profile or declared unbuilt",
         inputs=COMPOSE_FILES,
         denominator=r"inspected: \d+ Dockerfile",
         proven_by="scripts/test_dockerfile_coverage.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-036",)),
    # The sovereign profile could not build a single one of its nine
    # services, and looked healthy because its boot step reused images
    # another profile had built. KAI-GATE-037.
    Gate(module="check_dockerfile_context",
         kind=GATE,
         summary="every COPY source resolves in the context its profile declares",
         inputs=COMPOSE_FILES,
         denominator=r"inspected: \d+ COPY source",
         proven_by="scripts/test_dockerfile_context.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-037",)),
    # postgres refused to start with a blank password, and compose said
    # so in a warning printed on every invocation, all day. KAI-GATE-040.
    Gate(module="check_compose_env",
         kind=GATE,
         summary="every compose bring-up step supplies the variables it needs",
         inputs=(".github/workflows",),
         denominator=r"inspected: \d+ compose bring-up step",
         proven_by="scripts/test_compose_env.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-040",)),
    # The executor's retry-with-backoff ran its body zero times for its
    # whole life: `for d in $BACKOFF_SCHEDULE` was interpolated by
    # compose, not the shell, and both names became the empty string.
    # KAI-GATE-041. Found a third instance on its first real run —
    # `tailscale up --hostname=${TS_HOSTNAME}` had never named the node.
    Gate(module="check_compose_interpolation",
         kind=GATE,
         summary="every `$` in a compose command reaches whoever wrote it",
         inputs=COMPOSE_FILES,
         denominator=r"inspected: \d+ variable reference",
         proven_by="scripts/test_compose_interpolation.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-041",)),
    # memu-core's writer branch imported `socket` inside an `if` that
    # always raises and used it after — so the branch had never once
    # completed, and `.writer.lock` had never been written in any
    # deployment. KAI-GATE-042.
    Gate(module="check_unreachable_bindings",
         kind=GATE,
         summary="every import binds on a path that reaches its uses",
         inputs=(".",),
         denominator=r"inspected: \d+ function",
         proven_by="scripts/test_unreachable_bindings.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-042",)),
    # document-parser and the dashboard both used `UploadFile` without
    # listing python-multipart, so both raised at import and neither
    # container had ever started. Invisible to every unit test, because
    # CI installs all requirements into one environment. KAI-GATE-043.
    Gate(module="check_implicit_deps",
         kind=GATE,
         summary="every implicit dependency is declared where it is used",
         inputs=(".",),
         denominator=r"inspected: \d+ service director",
         proven_by="scripts/test_implicit_deps.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-043",)),
    # agentic/Dockerfile named ten files by hand against a directory of
    # thirty-seven; app.py imported twenty-seven of the ones it missed,
    # so the container died at import on every boot it ever had.
    # KAI-GATE-044 — the list-beside-the-thing pattern, 14th venue.
    Gate(module="check_image_modules",
         kind=GATE,
         summary="every image contains the modules its entry point imports",
         inputs=(".",),
         denominator=r"inspected: \d+ Python service image",
         proven_by="scripts/test_image_modules.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-044",)),
    # security_fuzz_upload said `environ.setdefault("KAI_DASHBOARD_ROLE",
    # "keeper")` and ran as whatever the environment said. When CI gained
    # an operator token, eight of its fourteen tests silently changed
    # from checking upload validation to checking authorisation.
    # KAI-GATE-045.
    Gate(module="check_shipped_package_deps",
         kind=GATE,
         summary="an image that ships a first-party package installs the "
                 "unguarded imports of the parts it reaches",
         inputs=(),
         denominator=r"inspected: \d+ first-party package copy\(ies\)",
         proven_by="scripts/test_shipped_package_deps.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",)),
    Gate(module="check_healthcheck_runnable",
         kind=GATE,
         summary="a healthcheck invokes only binaries its image provides",
         inputs=COMPOSE_FILES,
         denominator=r"inspected: \d+ healthcheck\(s\)",
         proven_by="scripts/test_healthcheck_runnable.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",)),
    Gate(module="check_depends_on_readiness",
         kind=GATE,
         summary="every depends_on states what it waits for",
         inputs=COMPOSE_FILES,
         denominator=r"inspected: \d+ depends_on edge\(s\)",
         proven_by="scripts/test_depends_on_readiness.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",)),
    Gate(module="check_test_identity",
         kind=GATE,
         summary="every test pins the identity it claims to run as",
         inputs=(".",),
         denominator=r"inspected: \d+ environ.setdefault call",
         proven_by="scripts/test_test_identity.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-045",)),
    # Every one of the ten defects found on 2026-08-06 lived in code
    # that had never executed. This measures that surface: 34 of 49
    # services with a Dockerfile have never been started by CI, 8 of
    # them boot by default in a shipped profile.
    #
    # A REPORT, not a gate. How much coverage is enough is a decision
    # about CI time, and a gate red on a number nobody chose is a gate
    # people learn to ignore.
    Gate(module="report_execution_coverage",
         kind=REPORT,
         summary="which services CI has never started, derived from its "
                 "own `up -d` lines",
         inputs=COMPOSE_FILES + (".github/workflows/core-tests.yml",),
         denominator=r"inspected: \d+ service\(s\) with a Dockerfile",
         proven_by="scripts/test_execution_coverage.py",
         in_policy_check=False,
         in_workflows=()),
    Gate(module="check_test_wiring",
         kind=GATE,
         summary="every test in a self-run suite is dispatched",
         inputs=("Makefile", "scripts"),
         denominator=r"inspected: \d+ self-run suites",
         proven_by="scripts/test_test_wiring.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-018",)),

    # A-05: no test file may change the interpreter for the files after
    # it. Found because the repo-wide pytest had been executing zero
    # tests for a week: it aborted at collection, and the six errors all
    # named files that passed when run alone. `replaced` is at zero and
    # enforced; `added`/`env_set` ratchet down from a declared baseline.
    Gate(module="check_test_isolation",
         kind=GATE,
         summary="no test leaves a real module replaced for the next file",
         inputs=("scripts/security/isolation_plugin.py",
                 "scripts/security/test_isolation_baseline.json"),
         denominator=r"inspected: \d+ test files that alter global state",
         probe=False,
         probe_skip_reason="runs the full pytest suite to observe the real "
                           "session — minutes, not seconds; probed by "
                           "scripts/test_test_isolation.py against synthetic "
                           "reports instead",
         proven_by="scripts/test_test_isolation.py",
         in_policy_check=False,
         in_workflows=("python-app.yml",),
         ratchet=True,
         calibrated_by=(
             "scripts/test_test_isolation.py — one fixture exercising all five reported categories (replaced, added, env_set, env_changed, path_added); the denominator is the plugin's own finding keys"),
         findings=("KAI-GATE-019",)),

    # KAI-GATE-020: the repo-wide result itself, as a ratchet. Recorded at
    # 4,208 passed / 0 failed / 0 errors — from zero tests running at all.
    # The pass count is a floor as well, because otherwise deleting a test
    # would be a way to satisfy this gate.
    Gate(module="check_suite_floor",
         kind=GATE,
         summary="the repo-wide pytest result may not regress (KAI-GATE-020)",
         inputs=("scripts/security/suite_floor.json",),
         denominator=r"inspected: \d+ tests passed",
         probe=False,
         probe_skip_reason="reads a captured run; producing one takes minutes "
                           "and CI already has it. Probed by "
                           "scripts/test_suite_floor.py against synthetic logs",
         proven_by="scripts/test_suite_floor.py",
         in_policy_check=False,
         in_workflows=("python-app.yml",),
         ratchet=True,
         calibrated_by=(
             "scripts/test_suite_floor.py — the parser reads 8 failed and 3 errors from a known summary, and a log with no summary returns None rather than zero failures"),
         findings=("KAI-GATE-020",)),

    # ── The meta-check, bound by the rules it enforces ───────────────
    # It appears in its own registry on purpose. The recursion is
    # depth-one and closed: it declares a denominator, fails closed on an
    # unreadable registry, and is proven by a synthetic-registry suite.
    # A-04e landed as *partial* enforcement: I-4 is at zero and enforced,
    # I-1/I-2/I-3 are reported while the retrofit proceeds. A big-bang
    # flip would have meant a permanently red gate, which is an ignored
    # gate — defect 9 wearing a fix's clothes.
    Gate(module="check_gate_registry",
         kind=GATE,
         summary="the four instrumentation invariants (A-04)",
         inputs=("scripts/security/gate_registry.py", "Makefile"),
         denominator=r"\d+ declared, \d+ found on disk",
         proven_by="scripts/test_gate_registry.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",)),

    # ── The eight instruments the denominator could not see ──────────
    #
    # This registry's scope was `scripts/security/*.py` — a *directory*,
    # which is where the checks happened to be put, not what makes
    # something an instrument. What makes it one is that CI runs it and
    # a non-zero exit stops the build.
    #
    # Measured 2026-08-06: 30 modules in that directory and eight
    # outside it that can fail the build, none registered, none held to
    # I-1..I-7 — with the meta-check printing `GATE PASSED` over all of
    # it. The seventeenth venue of this programme's one finding, in the
    # file whose whole job is to catch it.
    #
    # The widening repaid itself on the first module looked at:
    # `go_no_go_check` opened with `except Exception: SystemExit(0)` for
    # "dashboard not running", so `make go_no_go` passed on every run
    # where nothing was listening. It could not tell a GO decision from
    # no decision at all. Fixed; absence is now declared by the caller.
    Gate(module="ci/live_smoke",
         kind=GATE,
         summary="every service with a healthcheck is healthy, and the "
                 "exercised endpoints answer",
         inputs=("docker-compose.minimal.yml",),
         denominator=r"inspected: \d+ service\(s\) with a declared health port",
         probe=False,
         probe_skip_reason="talks to a running stack over `docker compose "
                           "exec`; probed by scripts/test_live_smoke.py "
                           "against an injected runner instead",
         proven_by="scripts/test_live_smoke.py",
         in_workflows=("core-tests.yml",)),
    Gate(module="ci/compose_probe",
         kind=GATE,
         summary="waits on Docker's own verdict of each healthcheck, and "
                 "reports a dead container as dead rather than uncheckable",
         inputs=("docker-compose.minimal.yml",),
         denominator=r"waited on: \d+ service\(s\)",
         probe=False,
         probe_skip_reason="needs a running daemon; probed by "
                           "scripts/test_compose_probe.py against an "
                           "injected runner instead",
         proven_by="scripts/test_compose_probe.py",
         in_workflows=("core-tests.yml", "stage1-replay.yml")),
    Gate(module="ci/kill_isolation",
         kind=GATE,
         summary="memu-core stays healthy AND writable with "
                 "memu-core-introspect stopped",
         inputs=("docker-compose.minimal.yml",),
         denominator=r"probing: \S+ on port \d+",
         probe=False,
         probe_skip_reason="needs the minimal stack up; probed by "
                           "scripts/test_ci_scripts.py against injected "
                           "exec_http/load_ports instead",
         proven_by="scripts/test_ci_scripts.py",
         in_workflows=("core-tests.yml",)),
    Gate(module="report_service_identity",
         kind=REPORT,
         summary="which shared-token endpoints need verified caller "
                 "identity (B) and which are fine on membership (A)",
         inputs=(),
         denominator=r"inspected: \d+ endpoint\(s\)",
         probe=False,
         probe_skip_reason="scans first-party service source; the A/B "
                           "split is declared judgement, so it is "
                           "reviewed rather than probed",
         proven_by="scripts/test_service_identity.py",
         pending_wiring="reported while the identity architecture is "
                        "decided; it would gate at B == 0 or every B "
                        "endpoint using verified identity",
         in_workflows=()),
    Gate(module="report_perception_intake",
         kind=REPORT,
         summary="the full UH-2 perception intake surface, with a verdict "
                 "per source — WORKING only when the whole path is proven",
         inputs=COMPOSE_FILES,
         denominator=r"inspected: \d+ perception source\(s\)",
         probe=False,
         probe_skip_reason="imports the live UH-2 registry and reducer map; "
                           "probed by scripts/test_perception_intake.py "
                           "against synthetic compose trees instead",
         proven_by="scripts/test_perception_intake.py",
         pending_wiring="reported while the intake rebuild is decided; it "
                        "would gate at WORKING == denominator, which is "
                        "currently 2 of 44",
         in_workflows=()),
    Gate(module="check_service_reachability",
         kind=REPORT,
         summary="a service that both depends_on a peer and holds its URL "
                 "must share a network with it — Docker DNS only resolves "
                 "names on a joined network",
         inputs=COMPOSE_FILES,
         denominator=r"inspected: \d+ edge\(s\)",
         proven_by="scripts/test_service_reachability.py",
         calibrated_by="scripts/test_service_reachability.py",
         pending_wiring="reported until the network-topology decision on "
                        "its 5 findings is taken; widening a service's "
                        "networks to silence it would change the security "
                        "topology to quieten a check",
         in_workflows=("policy-checks.yml",)),
    # #41's denominator, derived rather than remembered. The task carried
    # the figure 26 for weeks; the tree says 34. Its six columns exist
    # because every adjacent pair has been conflated at least once:
    # defined / profile-gated / profile-set-enabled / individually
    # startable / runtime-proven / expected by a live caller.
    # #41 defect class B. Profiles-off is the INTENDED posture, so what
    # the live core does about an absent gated dependency is a question
    # about the system as it is meant to run.
    Gate(module="report_degradation_tolerance",
         kind=REPORT,
         summary="what the default core does when a profile-gated "
                 "dependency is absent; 41 caller->dependency edges "
                 "reduce to 4 call mechanisms, and the dangerous class "
                 "lives at the call sites, not in the mechanisms",
         inputs=COMPOSE_FILES,
         denominator=r"inspected: \d+ live caller -> absent-dependency edge",
         proven_by="scripts/test_degradation_tolerance.py",
         calibrated_by="scripts/test_degradation_tolerance.py",
         in_policy_check=False,
         findings=("KAI-GATE-047",)),
    # The A/B/C denominator for the offline-startup invariant. Three
    # populations kept apart because merging any adjacent pair picks the
    # architecture: source reachability is not deployment applicability,
    # and neither is runtime evidence. Its own first version reported all
    # four traced services as loading at IMPORT — an `ast.walk` that
    # descends into function bodies — which would have argued for baking
    # a model into every image on evidence that did not exist.
    Gate(module="report_model_load_denominator",
         kind=REPORT,
         summary="A/B/C: which runnable container paths can reach a model "
                 "load, which of those are deployed without egress, and "
                 "which have a CITED runtime observation; A is source "
                 "reachability and is NOT a count of affected services",
         inputs=COMPOSE_FILES,
         denominator=r"inspected: \d+ service definition\(s\)",
         proven_by="scripts/test_model_load_denominator.py",
         calibrated_by="scripts/test_model_load_denominator.py",
         in_policy_check=False,
         in_workflows=(),
         findings=("KAI-GATE-048",)),
    # KAI-GATE-048, the CALIBRATION half. The workflow runs it as its own
    # step BEFORE the measurement, so a non-zero exit stops the job — an
    # uncalibrated classifier must not get to decorate a verdict. That is
    # what makes this an instrument rather than a test, and why I-4
    # discovered it as one the moment it was wired.
    Gate(module="test_model_startup_classifier",
         kind=GATE,
         summary="the four known model-startup shapes — memu-core, "
                 "memu-core-introspect, ollama-pull and a lazy "
                 "memu-graph — must produce four DIFFERENT verdicts, and "
                 "`classify()` must not be able to see a service name",
         inputs=(),
         denominator=r"Model Startup Classifier Calibration: \d+ passed",
         proven_by="scripts/test_model_startup_classifier.py",
         calibrated_by="scripts/test_model_startup_classifier.py",
         in_policy_check=False,
         in_workflows=("memu-graph-startup-proof.yml",),
         findings=("KAI-GATE-048",)),
    Gate(module="summarise_memu_graph_startup",
         kind=REPORT,
         summary="reads the KAI-GATE-048 stage logs into an observation "
                 "record; a missing or unparseable log stays NOT "
                 "MEASURED and never becomes a proven absence",
         inputs=(),
         denominator=r"inspected: \d+ of \d+ expected stage log",
         probe=False,
         probe_skip_reason="requires a stage-log directory produced by a "
                           "deployed collector; both parser directions "
                           "are asserted on synthetic stage-log trees in "
                           "scripts/test_model_startup_classifier.py",
         proven_by="scripts/test_model_startup_classifier.py",
         calibrated_by="scripts/test_model_startup_classifier.py",
         in_policy_check=False,
         # Invoked by collect_memu_graph_startup.sh, not by a workflow
         # `run:` step. I-4 compares the declaration against what the
         # parse can see, and the parse can only see direct invocations —
         # so the honest declaration is (), with the real caller named.
         in_workflows=(),
         pending_wiring="run by scripts/security/collect_memu_graph_startup.sh, "
                        "which memu-graph-startup-proof.yml invokes; a "
                        "workflow-level declaration would claim a wiring "
                        "the workflow parse cannot confirm",
         findings=("KAI-GATE-048",)),
    # D189's authorised definition unit. Its stages C and D are
    # known-negatives that MUST fail, so it is a REPORT: a gate whose
    # correct result includes failures is a gate people learn to ignore.
    Gate(module="summarise_asset_contract",
         kind=REPORT,
         summary="answers KAI-GATE-048's five asset-contract questions "
                 "from measured stage logs; the contract is PROVEN only "
                 "when the network-removed stage succeeds on the asset "
                 "set the fetch stage produced",
         inputs=(),
         denominator=r"inspected: \d+ of \d+ expected stage log",
         probe=False,
         probe_skip_reason="requires a stage-log directory produced by "
                           "four throwaway containers with and without "
                           "network; the parsers are asserted on "
                           "synthetic stage-log trees in "
                           "scripts/test_asset_contract.py",
         proven_by="scripts/test_asset_contract.py",
         calibrated_by="scripts/test_asset_contract.py",
         in_policy_check=False,
         in_workflows=(),
         pending_wiring="run by "
                        "scripts/security/define_memu_graph_asset_contract.sh, "
                        "which memu-graph-startup-proof.yml invokes; a "
                        "workflow-level declaration would claim a wiring "
                        "the workflow parse cannot confirm",
         findings=("KAI-GATE-048",)),
    # KAI-GATE-048 Phase 1's verdict. A GATE, not a report: unlike the
    # measurement jobs, its whole purpose is to refuse. It exits non-zero
    # when acceptance is not met AND when the evidence is merely missing,
    # because "we did not measure" must not read as "it works".
    # Both calibrations are invoked as their own workflow `run:` steps,
    # BEFORE the measurements they judge — so a broken instrument stops
    # the job instead of decorating its output. That makes them
    # enforcing, and I-4 discovered them as such the moment they were
    # wired.
    Gate(module="test_asset_contract",
         kind=GATE,
         summary="the asset-contract summariser may print CONTRACT PROVEN "
                 "only when the network-removed stage SUCCEEDED on the "
                 "asset set the fetch stage produced; proven / not-proven "
                 "/ disproven / ambiguous stay four distinct findings",
         inputs=(),
         denominator=r"Asset Contract Summariser Calibration: \d+ passed",
         proven_by="scripts/test_asset_contract.py",
         calibrated_by="scripts/test_asset_contract.py",
         in_policy_check=False,
         in_workflows=("memu-graph-startup-proof.yml",),
         findings=("KAI-GATE-048",)),
    Gate(module="test_memu_graph_acceptance",
         kind=GATE,
         summary="the Phase 1 verdict's can-fail stage is INVERTED (a "
                 "non-zero exit is its PASS), and blocking ALL networking "
                 "must NOT satisfy the capability check — memu-graph "
                 "delegates embedding work to an internal peer",
         inputs=(),
         denominator=r"memu-graph Acceptance Calibration: \d+ passed",
         proven_by="scripts/test_memu_graph_acceptance.py",
         calibrated_by="scripts/test_memu_graph_acceptance.py",
         in_policy_check=False,
         in_workflows=("memu-graph-startup-proof.yml",),
         findings=("KAI-GATE-048",)),
    Gate(module="summarise_memu_graph_acceptance",
         kind=GATE,
         summary="KAI-GATE-048 Phase 1 acceptance: the asset loads from "
                 "the shipped image with no network, readiness is still "
                 "reached without loading it, and the real capability "
                 "works under the INTENDED topology with the internal "
                 "delegate present and external registry egress absent",
         inputs=(),
         denominator=r"inspected: \d+ of \d+ expected stage log",
         probe=False,
         probe_skip_reason="requires a stage-log directory produced by a "
                           "remediated image on a live stack; both "
                           "directions of every check are asserted on "
                           "synthetic stage-log trees in "
                           "scripts/test_memu_graph_acceptance.py",
         proven_by="scripts/test_memu_graph_acceptance.py",
         calibrated_by="scripts/test_memu_graph_acceptance.py",
         in_policy_check=False,
         in_workflows=("memu-graph-startup-proof.yml",),
         findings=("KAI-GATE-048",)),
    # KAI-GATE-049's calibration, run as its own workflow step BEFORE
    # the diagnostic — an uncalibrated analyser must not decorate a
    # verdict about which stage owns a stall.
    Gate(module="test_graph_stall",
         kind=GATE,
         summary="the four stall states — slow LLM work, waiting on the "
                 "delegate, stuck elsewhere, local compute — must stay "
                 "four distinct verdicts; a non-return must never read as "
                 "a proven hang; and a container replaced mid-observation "
                 "must yield UNKNOWN rather than an execution state, "
                 "including when the identity fields are absent entirely",
         inputs=(),
         denominator=r"Graph Stall Analyser Calibration: \d+ passed",
         proven_by="scripts/test_graph_stall.py",
         calibrated_by="scripts/test_graph_stall.py",
         in_policy_check=False,
         in_workflows=("memu-graph-startup-proof.yml",),
         findings=("KAI-GATE-049",)),
    # A GATE, and it was a REPORT until run 8 showed why. What it
    # ENFORCES is that the diagnostic run obtained an observation at all
    # — `ingest.log` must carry the probe's pre-request `ENTERED` marker.
    # It does NOT gate on the stall verdict: which stage owns the silence
    # and whether the process was computing stay informational, because
    # nobody has promised those are actionable yet.
    #
    # Run 8 fired no request (the probe was invoked without its
    # subcommand), and the three sections each reported their own absence
    # correctly — "no markers", "1 sample", "outcome not established" —
    # while the module exited 0 and the job went green. Three true
    # statements summed to a diagnostic run that diagnosed nothing.
    Gate(module="summarise_graph_stall",
         kind=GATE,
         summary="fails closed when the run asked the service nothing, so "
                 "an unmeasured diagnostic cannot go green; when a request "
                 "WAS sent, names the cognee task entered without "
                 "returning and whether the process was computing or "
                 "blocked — but only after ADJACENT-PAIR continuity proves "
                 "the samples describe one execution instance, since a "
                 "replaced container reads as flat CPU to a first-vs-last "
                 "difference; authorises no remedy — slow work, a blocked "
                 "wait and a deadlock have three different owners",
         inputs=(),
         denominator=r"inspected: \d+ of \d+ expected stage log",
         probe=False,
         probe_skip_reason="requires a stage-log directory produced by a "
                           "live stack observed past its own client "
                           "budget; every branch is asserted on synthetic "
                           "stage-log trees in scripts/test_graph_stall.py",
         proven_by="scripts/test_graph_stall.py",
         calibrated_by="scripts/test_graph_stall.py",
         in_policy_check=False,
         in_workflows=("memu-graph-startup-proof.yml",),
         findings=("KAI-GATE-049",)),
    Gate(module="probe_graph_stall",
         kind=REPORT,
         summary="runs INSIDE the image: POSTs /graph/ingest without the "
                 "300s budget under investigation, and samples pid-1 CPU "
                 "ticks, open sockets to the delegate, and the process "
                 "identity (container_id, /proc/1/stat field 22) that lets "
                 "a reader tell one execution instance from its "
                 "replacement; its argv contract is a pure function so a "
                 "caller's command line can be validated before any stack "
                 "exists",
         inputs=(),
         denominator=r"inspected: \d+ connection\(s\), \d+ cognee log line",
         probe=False,
         probe_skip_reason="stdlib-only probe that must execute inside a "
                           "running memu-graph container; reading "
                           "/proc/1/stat and /proc/net/tcp on the host "
                           "would measure the wrong process entirely",
         proven_by="scripts/test_graph_stall.py",
         in_policy_check=False,
         in_workflows=(),
         pending_wiring="invoked by "
                        "scripts/security/diagnose_graph_stall.sh via "
                        "`docker compose exec`, which the workflow parse "
                        "cannot see as an invocation of this module",
         findings=("KAI-GATE-049",)),
    # KAI-GATE-050. Opened 2026-08-13 from run 31733359906: cognee's
    # pipeline failed 422 and `/graph/ingest` answered 200
    # {"status":"ingested"}. Established from source, not inferred —
    # cognee raises PipelineRunFailedError (run_tasks.py:147) then
    # deliberately does NOT re-raise it (:185-187, intent in a comment),
    # so the failure travels as a return value; memu-graph/app.py:96
    # discards that return value and its only predicate is
    # `except Exception`, which cannot fire.
    Gate(module="test_ingest_contract",
         kind=GATE,
         summary="HTTP 200 must never be the success predicate — the 200 "
                 "is the thing under suspicion — and a pipeline with no "
                 "terminal marker must not read as a completed one, "
                 "because cognee swallows PipelineRunFailedError without "
                 "re-raising it",
         inputs=(),
         denominator=r"Ingest Contract Analyser Calibration: \d+ passed",
         proven_by="scripts/test_ingest_contract.py",
         calibrated_by="scripts/test_ingest_contract.py",
         in_policy_check=False,
         in_workflows=("memu-graph-startup-proof.yml",),
         findings=("KAI-GATE-050",)),
    # KAI-GATE-050 REMEDIATION. The predicate itself lives in
    # memu-graph/cognify_result.py -- production code, not a check -- so
    # what is registered here is the suite that proves it fires. Without
    # this entry the suite would be an enforced test nobody declared.
    Gate(module="test_cognify_result",
         kind=GATE,
         summary="did cognee return a TERMINAL SUCCESSFUL pipeline "
                 "result? Both directions, plus a status the predicate "
                 "has never been taught -- the rule is a class, not the "
                 "one failure observed; hard-coding PipelineRunFailedError "
                 "would have fixed run 9 and stayed blind to the next mode",
         inputs=(),
         denominator=r"inspected: \d+ terminal-success status\(es\) accepted",
         proven_by="scripts/test_cognify_result.py",
         calibrated_by="scripts/test_cognify_result.py",
         in_policy_check=False,
         in_workflows=("memu-graph-startup-proof.yml",),
         findings=("KAI-GATE-050",)),
    Gate(module="summarise_ingest_contract",
         kind=GATE,
         summary="correlates cognee's OWN terminal pipeline status with "
                 "the HTTP response `/graph/ingest` returned; fails on a "
                 "2xx over a pipeline that did not complete, and equally "
                 "on an observation that established neither side — "
                 "unmeasured is not clean",
         inputs=(),
         denominator=r"inspected: \d+ clean-stack observation",
         probe=False,
         probe_skip_reason="requires a stage-log directory produced by "
                           "two clean stacks each running a full "
                           "~400s ingest; all four correlation cells and "
                           "both failure-to-measure paths are asserted on "
                           "synthetic stage-log trees in "
                           "scripts/test_ingest_contract.py",
         proven_by="scripts/test_ingest_contract.py",
         calibrated_by="scripts/test_ingest_contract.py",
         in_policy_check=False,
         in_workflows=("memu-graph-startup-proof.yml",),
         findings=("KAI-GATE-050",)),
    Gate(module="probe_ingest_contract",
         kind=REPORT,
         summary="runs INSIDE the image: POSTs /graph/ingest recording "
                 "status AND body, and dumps cognee's own log file IN "
                 "FULL after the request returns — the terminal pipeline "
                 "marker run 9 sampled past and never captured",
         inputs=(),
         denominator=r"inspected: \d+ cognee log file, \d+ line",
         probe=False,
         probe_skip_reason="stdlib-only probe that must execute inside a "
                           "running memu-graph container; on the host "
                           "there is no cognee log directory and no "
                           "endpoint to call",
         proven_by="scripts/test_ingest_contract.py",
         in_policy_check=False,
         in_workflows=(),
         pending_wiring="invoked by "
                        "scripts/security/measure_ingest_contract.sh via "
                        "`docker compose exec`, which the workflow parse "
                        "cannot see as an invocation of this module",
         findings=("KAI-GATE-050",)),
    # KAI-GATE-048 C, Q1/Q2/Q6 capture. Observation only -- it changes
    # no mode, model, timeout, retry, schema, validation or topology.
    Gate(module="test_llm_contract",
         kind=GATE,
         summary="a schema DEFINITION and an INSTANCE of it must never "
                 "collapse to one verdict -- the observed failure is the "
                 "first wearing the shape of the second -- and a schema "
                 "echo, a wrong-key object and no response must stay three "
                 "verdicts, because they have three different owners; "
                 "REQUIRED FIELDS PRESENT must never be promoted to VALID "
                 "INSTANCE, because a top-level key check is not JSON "
                 "Schema validation; and each attempt's contract must be "
                 "recovered from THAT attempt, never from the outer caller",
         inputs=(),
         denominator=r"inspected: \d+ response verdict\(s\) discriminated",
         proven_by="scripts/test_llm_contract.py",
         calibrated_by="scripts/test_llm_contract.py",
         in_policy_check=False,
         in_workflows=("memu-graph-startup-proof.yml", "stage1-replay.yml"),
         findings=("KAI-GATE-048",)),
    Gate(module="summarise_llm_contract",
         kind=REPORT,
         summary="the per-attempt table -- effective structured-output "
                 "mode read at runtime rather than inferred from config, "
                 "plus prompt/schema/response hashes so reproducibility is "
                 "measured not eyeballed; assigns NO ownership between "
                 "prompt construction, adapter mode, model compliance and "
                 "the validator",
         inputs=(),
         denominator=r"inspected: \d+ model call\(s\)",
         probe=False,
         probe_skip_reason="requires a capture file produced by driving "
                           "cognee in-process inside the memu-graph "
                           "image; every branch is asserted on synthetic "
                           "capture files in scripts/test_llm_contract.py",
         proven_by="scripts/test_llm_contract.py",
         calibrated_by="scripts/test_llm_contract.py",
         in_policy_check=False,
         in_workflows=("memu-graph-startup-proof.yml",),
         findings=("KAI-GATE-048",)),
    Gate(module="probe_llm_contract",
         kind=REPORT,
         summary="runs INSIDE the image: wraps the adapter's own client "
                 "method with a strict pass-through, recording every "
                 "attempt's request and raw response plus the RESOLVED "
                 "instructor mode; alters no argument and returns the "
                 "original object",
         inputs=(),
         denominator=r"inspected: \d+ model call\(s\) captured",
         probe=False,
         probe_skip_reason="imports cognee and drives a real pipeline "
                           "inside a running memu-graph container; on the "
                           "host there is no cognee, no adapter and no "
                           "delegate to observe",
         proven_by="scripts/test_llm_contract.py",
         in_policy_check=False,
         in_workflows=(),
         pending_wiring="invoked by "
                        "scripts/security/capture_llm_contract.sh via "
                        "`docker compose exec`, which the workflow parse "
                        "cannot see as an invocation of this module",
         findings=("KAI-GATE-048",)),
    # D248's P1 prerequisite, and the reason it is a separate suite: a
    # shared file would have put a static census inside the LLM capture
    # workflow's paths filter, so editing an analyser would have fired a
    # live model capture nobody authorised.
    # D251. Answers "changed paths ∩ live-capture trigger paths = ∅"
    # BEFORE a push, because "I did not touch the probe" never proved it.
    Gate(module="check_capture_trigger_paths",
         kind=REPORT,
         summary="which workflows can call a real model, which of those "
                 "also WRITE a capture that could become evidence, and "
                 "whether any changed path falls inside their trigger "
                 "filters; both halves derived by walking the tree, and a "
                 "workflow with no paths filter is reported as firing on "
                 "every push rather than as having no triggers",
         inputs=(),
         denominator=r"inspected: \d+ changed path\(s\) against \d+ "
                     r"live-capture workflow\(s\)",
         probe=False,
         probe_skip_reason="needs a working tree with changes to inspect; "
                           "the matcher, the filter reader, the "
                           "intersection and the two-class separation are "
                           "each asserted with a known-positive and a "
                           "known-negative in "
                           "scripts/test_capture_trigger_paths.py",
         proven_by="scripts/test_capture_trigger_paths.py",
         calibrated_by="scripts/test_capture_trigger_paths.py",
         in_policy_check=False,
         in_workflows=(),
         pending_wiring="invoked by the author before a push, and by "
                        "`make trigger-check`; wiring it into a workflow "
                        "needs a base ref to diff against, which a push "
                        "event does not carry unambiguously",
         findings=("KAI-GATE-048",)),
    Gate(module="test_capture_trigger_paths",
         kind=GATE,
         summary="the trigger check must not answer its negative claim in "
                 "the reassuring direction: a workflow with no paths "
                 "filter reads as ABSENT rather than empty, `*` does not "
                 "span a separator, a prefix is not a match, and "
                 "live-model must stay separable from capture-writing",
         inputs=(),
         denominator=r"inspected: \d+ live-capture workflow\(s\)",
         proven_by="scripts/test_capture_trigger_paths.py",
         calibrated_by="scripts/test_capture_trigger_paths.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-048",)),
    # D251. Five causes of a missing artifact, five states.
    Gate(module="classify_artifact_fetch",
         kind=REPORT,
         summary="why an artifact did not arrive, as one of seven "
                 "distinguishable states rather than one 'not performed': "
                 "a run still going, a permissions failure HERE, a "
                 "network failure, an artifact absent after a COMPLETED "
                 "run, an expired one, a malformed one, and present; only "
                 "the last licenses a measurement",
         inputs=(),
         denominator=r"inspected: \d+ artifact fetch across \d+ "
                     r"distinguishable state\(s\)",
         probe=False,
         probe_skip_reason="the classification is a pure function of facts "
                           "the caller gathered from the API, so it is "
                           "calibrated without a network in "
                           "scripts/test_artifact_fetch_states.py; only "
                           "the gathering needs one",
         proven_by="scripts/test_artifact_fetch_states.py",
         calibrated_by="scripts/test_artifact_fetch_states.py",
         in_policy_check=False,
         in_workflows=("p1-replay-completeness.yml",),
         findings=("KAI-GATE-048",)),
    Gate(module="test_artifact_fetch_states",
         kind=GATE,
         summary="the five failure causes must stay distinguishable: a run "
                 "in progress is not an absent artifact, a permissions "
                 "failure here is not the subject having produced "
                 "nothing, expired is not absent, and transport is asked "
                 "before a run status we may not have been able to fetch",
         inputs=(),
         denominator=r"inspected: \d+ fetch state\(s\) discriminated",
         proven_by="scripts/test_artifact_fetch_states.py",
         calibrated_by="scripts/test_artifact_fetch_states.py",
         in_policy_check=False,
         in_workflows=("p1-replay-completeness.yml",),
         findings=("KAI-GATE-048",)),
    # S1 (D255-D257). Selection is the step where a post-result choice
    # would be easiest to make and hardest to notice.
    Gate(module="stage1_replay",
         kind=REPORT,
         summary="D247's Stage-1 experiment and nothing else: re-select "
                 "under S1, REFUSE unless the frozen seq/prompt/contract "
                 "identity reproduces, rebuild response_format with "
                 "ast.literal_eval and assert the exact typed value, then "
                 "replay N1=10 times with no Instructor, no validation and "
                 "no retry; a transport error is one execution and is not "
                 "replaced, and the original captured response is never "
                 "read",
         inputs=(),
         denominator=r"inspected: \d+ replay execution\(s\) of \d+ "
                     r"precommitted",
         probe=False,
         probe_skip_reason="needs a production capture and a live model "
                           "endpoint; reconstruction, identity refusal, the "
                           "response boundary and the fixed denominator are "
                           "all asserted offline in "
                           "scripts/test_stage1_replay.py",
         proven_by="scripts/test_stage1_replay.py",
         calibrated_by="scripts/test_stage1_replay.py",
         in_policy_check=False,
         in_workflows=("stage1-replay.yml",),
         findings=("KAI-GATE-048",)),
    Gate(module="test_stage1_replay",
         kind=GATE,
         summary="ast.literal_eval and never eval; a repr that parses to "
                 "the wrong typed value REFUSES; a changed prompt, "
                 "contract or seq REFUSES; a key recorded ABSENT is "
                 "omitted rather than sent as null; and the original "
                 "response is unreadable from the frozen manifest, "
                 "asserted against a sentinel",
         inputs=(),
         denominator=r"inspected: \d+ precommitted replay execution\(s\)",
         proven_by="scripts/test_stage1_replay.py",
         calibrated_by="scripts/test_stage1_replay.py",
         in_policy_check=False,
         in_workflows=("stage1-replay.yml",),
         findings=("KAI-GATE-048",)),
    # D262's repair. "I only touched the plumbing" is an assertion
    # until something computes it.
    Gate(module="check_invocation_identity",
         kind=REPORT,
         summary="whether a repair changed what the model is actually "
                 "asked: the transitive closure of module-level names "
                 "reachable from the definitions that BUILD and SEND the "
                 "request, digested per definition OLD vs NEW, with every "
                 "reached repo module required unchanged in full and "
                 "every out-of-surface change reported but not failed",
         inputs=(),
         denominator=r"inspected: \d+ top-level definition\(s\), \d+ in "
                     r"the model-facing surface",
         probe=False,
         probe_skip_reason="it compares two git revisions, so a probe "
                           "would need a repository state to compare "
                           "against; both directions are calibrated by "
                           "injected mutation in "
                           "scripts/test_invocation_identity.py",
         proven_by="scripts/test_invocation_identity.py",
         calibrated_by="scripts/test_invocation_identity.py",
         in_policy_check=False,
         # Run BY HAND at repair time, against two git revisions. It is
         # not wired into the Stage-1 workflow: making it a gate there
         # means pinning a baseline commit into the experiment, which is
         # a change to the experiment and needs its own authorisation.
         in_workflows=(),
         findings=("KAI-GATE-048",)),
    Gate(module="test_invocation_identity",
         kind=GATE,
         summary="the identity check must be able to say BOTH things: a "
                 "mutation inside the derived surface breaches, one "
                 "outside it is reported and does not, a removed "
                 "in-surface definition cannot take its own scope with "
                 "it, and an aliased repo module resolves to its file "
                 "rather than to its alias",
         inputs=(),
         denominator=r"inspected: \d+ top-level definition\(s\), \d+ in "
                     r"the model-facing surface",
         proven_by="scripts/test_invocation_identity.py",
         calibrated_by="scripts/test_invocation_identity.py",
         in_policy_check=False,
         in_workflows=("stage1-replay.yml",),
         findings=("KAI-GATE-048",)),
    # D265. Attempt 2's defect: server health read as model readiness.
    Gate(module="check_model_ready",
         kind=REPORT,
         summary="whether the EXACT model the replay will send is "
                 "present, asked of the server's own inventory and "
                 "matched exactly against runtime.model from the frozen "
                 "manifest rather than a literal; a prefix is not a "
                 "match, a pulled identity that disagrees with the one "
                 "to be requested refuses, an unreadable inventory "
                 "refuses, and /v1/models corroborates without a veto",
         inputs=(),
         denominator=r"inspected: \d+ model identities across \d+ server",
         probe=False,
         probe_skip_reason="it needs a running ollama on an internal "
                           "network; both directions are calibrated "
                           "against a real HTTP server serving fixture "
                           "inventories in scripts/test_model_ready.py",
         proven_by="scripts/test_model_ready.py",
         calibrated_by="scripts/test_model_ready.py",
         in_policy_check=False,
         in_workflows=("stage1-replay.yml",),
         findings=("KAI-GATE-048",)),
    Gate(module="test_model_ready",
         kind=GATE,
         summary="the readiness gate must say BOTH things: a healthy "
                 "server holding nothing refuses, a different tag of the "
                 "same family refuses, the exact model permits the "
                 "replay, every refusal returns a verdict rather than a "
                 "traceback, and the pull is a foreground gate ordered "
                 "before the probe and the replay",
         inputs=(),
         denominator=r"inspected: \d+ model-readiness scenario\(s\) "
                     r"across \d+ gate",
         proven_by="scripts/test_model_ready.py",
         calibrated_by="scripts/test_model_ready.py",
         in_policy_check=False,
         in_workflows=("stage1-replay.yml",),
         findings=("KAI-GATE-048",)),
    # D268. The doctrine's spine, in its one decidable form. Earned by
    # run 31906667051: the condition was declared and bypassed.
    Gate(module="check_declared_prerequisites",
         kind=REPORT,
         summary="a depends_on condition must be IN FORCE where the "
                 "service is started, not merely declared: every "
                 "--no-deps site is resolved against the compose "
                 "declarations it skips, an undeclared bypass fails, a "
                 "compose file that cannot be resolved is UNRESOLVED "
                 "rather than clean, and a service with no conditions is "
                 "trivially clean rather than unresolvable",
         inputs=(),
         denominator=r"inspected: \d+ bypass site\(s\) against \d+ "
                     r"declared condition\(s\)",
         probe=False,
         probe_skip_reason="it reads compose files and execution sites "
                           "from the tree, so a probe would need a second "
                           "tree to read; both directions are calibrated "
                           "against fixture repositories built on disk in "
                           "scripts/test_declared_prerequisites.py",
         proven_by="scripts/test_declared_prerequisites.py",
         calibrated_by="scripts/test_declared_prerequisites.py",
         # REPORT, not GATE, and deliberately: it currently reports 3
         # undeclared bypasses in verify_identity_in_containers.sh, which
         # are a finding awaiting the operator's judgement (Programme
         # Rule 7). Wiring a red gate into policy-check would force them
         # to be declared away rather than decided. It is promoted to a
         # GATE, in_policy_check, the moment they are judged. Until then
         # calling it enforcing would be the claim this file exists to
         # catch: a declaration that is not in force.
         # NOT enforcing yet: it currently reports 3 undeclared bypasses
         # in verify_identity_in_containers.sh, which are a finding
         # awaiting the operator's judgement (Programme Rule 7). Wiring a
         # red gate into policy-check would force them to be declared
         # away rather than decided. It becomes enforcing when they are.
         in_policy_check=False,
         in_workflows=(),
         findings=("KAI-GATE-048",)),
    Gate(module="test_declared_prerequisites",
         kind=GATE,
         summary="the gate must be neither too narrow, too generous nor "
                 "too wide: a service with no conditions is clean rather "
                 "than unresolvable, an unresolvable compose file is "
                 "UNRESOLVED rather than clean, a declaration for the "
                 "wrong dependency does not cover a bypass, and "
                 "--no-deps is never banned outright",
         inputs=(),
         denominator=r"inspected: \d+ declared-prerequisite scenario\(s\) "
                     r"across \d+ gate",
         proven_by="scripts/test_declared_prerequisites.py",
         calibrated_by="scripts/test_declared_prerequisites.py",
         # The CALIBRATION enforces even while the report it proves does
         # not. An instrument with open findings is exactly the one whose
         # ability to fail must not quietly lapse.
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-048",)),
    # D273 / rule 28. Two records of the doctrine diverged and the
    # casualty was rule 4 -- the anti-drift rule itself.
    Gate(module="check_doctrine_integrity",
         kind=GATE,
         summary="the doctrine must stay mechanically comparable: rules "
                 "numbered contiguously with no gap and no duplicate, "
                 "every rule carrying provenance in the earned-by table, "
                 "the population scoped to the rules section so a bold "
                 "numbered item elsewhere is not counted, and a "
                 "fingerprint published that any external copy must "
                 "reproduce",
         inputs=(),
         denominator=r"inspected: \d+ rule\(s\) across \d+ provenance "
                     r"entry\(s\)",
         proven_by="scripts/test_doctrine_integrity.py",
         calibrated_by="scripts/test_doctrine_integrity.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-048",)),
    Gate(module="test_doctrine_integrity",
         kind=GATE,
         summary="the integrity gate must catch a dropped rule, a split "
                 "rule and a missing rules section, must NOT count "
                 "section 0's bold numbered step as a rule, must find a "
                 "rule whose bold statement wraps, and its fingerprint "
                 "must move on a reworded, dropped or renumbered rule "
                 "while staying still for prose outside them",
         inputs=(),
         denominator=r"inspected: \d+ doctrine-integrity scenario\(s\) "
                     r"across \d+ gate",
         proven_by="scripts/test_doctrine_integrity.py",
         calibrated_by="scripts/test_doctrine_integrity.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-048",)),
    # D278 / D247 §6 item 10. The bar demanded tree, IMAGE and run id;
    # nothing in the tree recorded an image identity (D277).
    Gate(module="collect_image_identity",
         kind=REPORT,
         summary="which image actually executed, bound to the tree and "
                 "run that produced it -- named DOCKER_LOCAL_IMAGE_ID "
                 "rather than a digest, because a built-in-job image is "
                 "never pushed and has no registry digest; RepoDigests "
                 "kept as ABSENT/NULL/VALUE, and a failed, empty or "
                 "Id-less inspect recorded as UNRECORDED rather than as "
                 "an empty identity field",
         inputs=COMPOSE_FILES,
         denominator=r"inspected: \d+ service\(s\), \d+ recorded, "
                     r"\d+ UNRECORDED",
         probe=False,
         probe_skip_reason="needs a Docker daemon and a built image; this "
                           "host has neither. Every path -- the recorded "
                           "case, all three RepoDigests states and six "
                           "refusals -- is asserted against the shipped "
                           "CLI with an injected docker in "
                           "scripts/test_image_identity.py",
         proven_by="scripts/test_image_identity.py",
         calibrated_by="scripts/test_image_identity.py",
         in_policy_check=False,
         in_workflows=("stage1-replay.yml",),
         trigger_class=SENTINEL_AUTHORISED,
         findings=("KAI-GATE-048",)),
    Gate(module="test_image_identity",
         kind=GATE,
         summary="the collector must record an identity when one exists "
                 "and REFUSE when one does not: a failed inspect, an "
                 "inspect exiting 0 with no payload, a payload carrying "
                 "no Id, a service resolving to two images and a compose "
                 "resolution that names nothing must all read UNRECORDED "
                 "with the prerequisite named, never an empty string; "
                 "and ABSENT must not collapse into NULL",
         inputs=(),
         denominator=r"inspected: \d+ image-identity scenario\(s\) across "
                     r"\d+ collector",
         proven_by="scripts/test_image_identity.py",
         calibrated_by="scripts/test_image_identity.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",),
         findings=("KAI-GATE-048",)),
    # D285/D288. Item 8's frozen experiment and its three instruments.
    Gate(module="check_item8_design",
         kind=GATE,
         summary="the frozen Item-8 canonical design must be byte-identical "
                 "to what D288 froze before any build runs; a moved design, "
                 "the superseded R1 digest, an unreadable decisions file or "
                 "an ambiguous region all REFUSE, because an amended "
                 "experiment running under a frozen design's authority "
                 "would be invisible without this",
         inputs=("kai-pm/DECISIONS.md",),
         denominator=r"inspected: \d+ canonical region across \d+ frozen "
                     r"design",
         proven_by="scripts/test_item8_instruments.py",
         calibrated_by="scripts/test_item8_instruments.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml", "item8-network-contingency.yml"),
         findings=("KAI-GATE-048",)),
    Gate(module="derive_item8_dockerfile",
         kind=REPORT,
         summary="Item 8's experimental Dockerfiles, derived mechanically "
                 "from the shipped ones with treatment-mutation cardinality "
                 "asserted (B1=0, B2=1, B3=1) and the pinned frontend added "
                 "as scaffolding; refuses rather than silently emitting the "
                 "source when its anchor no longer matches, and refuses to "
                 "write over the shipped Dockerfile",
         inputs=("memu-core/Dockerfile", "memu-graph/Dockerfile"),
         denominator=r"inspected: \d+ shipped Dockerfile, \d+ treatment "
                     r"mutation\(s\) of \d+ required",
         probe=False,
         probe_skip_reason="writes a derived file, so probing it in the "
                           "meta-check would create artifacts as a side "
                           "effect of measuring; all six derivations and "
                           "every refusal are asserted in "
                           "scripts/test_item8_instruments.py",
         proven_by="scripts/test_item8_instruments.py",
         calibrated_by="scripts/test_item8_instruments.py",
         in_policy_check=False,
         in_workflows=("item8-network-contingency.yml",),
         trigger_class=SENTINEL_AUTHORISED,
         findings=("KAI-GATE-048",)),
    Gate(module="collect_explicit_image_identity",
         kind=REPORT,
         summary="identity for an image named directly rather than resolved "
                 "through Compose, emitting the same JSONL contract as "
                 "collect_image_identity by importing its primitives rather "
                 "than copying them; a failed or Id-less inspect records "
                 "UNRECORDED, never an empty identity field",
         inputs=(),
         denominator=r"inspected: \d+ explicit image reference, \d+ "
                     r"recorded, \d+ UNRECORDED",
         probe=False,
         probe_skip_reason="needs a Docker daemon and a built experimental "
                           "image; this host has neither. The recorded case "
                           "and every refusal are asserted against the "
                           "shipped CLI with an injected docker in "
                           "scripts/test_item8_instruments.py",
         proven_by="scripts/test_item8_instruments.py",
         calibrated_by="scripts/test_item8_instruments.py",
         in_policy_check=False,
         in_workflows=("item8-network-contingency.yml",),
         trigger_class=SENTINEL_AUTHORISED,
         findings=("KAI-GATE-048",)),
    Gate(module="summarise_item8",
         kind=REPORT,
         summary="Item 8's six verdicts on two axes that may not launder "
                 "one another -- the contingency, and the collectors' first "
                 "live-daemon qualification -- refusing to compose a verdict "
                 "from fewer than the six precommitted branches",
         inputs=(),
         denominator=r"inspected: \d+ branch result\(s\) of \d+ "
                     r"precommitted",
         probe=False,
         probe_skip_reason="needs a results file produced by the six frozen "
                           "builds; its absent-input and short-input "
                           "refusals are asserted in "
                           "scripts/test_item8_instruments.py",
         proven_by="scripts/test_item8_instruments.py",
         calibrated_by="scripts/test_item8_instruments.py",
         in_policy_check=False,
         in_workflows=("item8-network-contingency.yml",),
         trigger_class=SENTINEL_AUTHORISED,
         findings=("KAI-GATE-048",)),
    Gate(module="test_item8_instruments",
         kind=GATE,
         summary="Item 8's three instruments must each refuse: a moved "
                 "frozen design (including one differing by a single "
                 "character) stops the build, a derivation whose anchor "
                 "moved refuses instead of emitting the source, B3 denies "
                 "network to exactly the HF instruction and leaves pip "
                 "alone, and an explicit-image record must be readable by "
                 "the unchanged sibling collector",
         inputs=(),
         denominator=r"inspected: \d+ Item-8 instrument scenario\(s\) "
                     r"across \d+ instruments",
         proven_by="scripts/test_item8_instruments.py",
         calibrated_by="scripts/test_item8_instruments.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml", "item8-network-contingency.yml"),
         findings=("KAI-GATE-048",)),
    Gate(module="select_replay_subject",
         kind=REPORT,
         summary="which captured request becomes the Stage-1 replay "
                 "subject -- the lowest-seq production row, with five "
                 "preconditions that REFUSE rather than fall through to "
                 "another row -- published as an allow-list request-side "
                 "projection so that no response field, no timing and no "
                 "hash of the response-bearing row can bias or be inferred "
                 "from the choice",
         inputs=(),
         denominator=r"inspected: \d+ production request row\(s\) across "
                     r"\d+ S1 precondition\(s\)",
         probe=False,
         probe_skip_reason="needs a production capture, which exists only "
                           "as a CI artifact; every precondition and the "
                           "response boundary are asserted on synthetic "
                           "rows in scripts/test_replay_subject_selection.py",
         proven_by="scripts/test_replay_subject_selection.py",
         calibrated_by="scripts/test_replay_subject_selection.py",
         in_policy_check=False,
         in_workflows=("p1-replay-completeness.yml",),
         findings=("KAI-GATE-048",)),
    Gate(module="test_replay_subject_selection",
         kind=GATE,
         summary="each of S1's five preconditions must refuse, and NO "
                 "response-bearing value may reach the published "
                 "projection -- asserted against rows whose response "
                 "fields carry a sentinel, so a leak is detected rather "
                 "than an absence merely observed",
         inputs=(),
         denominator=r"inspected: \d+ request-side field\(s\) allowed",
         proven_by="scripts/test_replay_subject_selection.py",
         calibrated_by="scripts/test_replay_subject_selection.py",
         in_policy_check=False,
         in_workflows=("p1-replay-completeness.yml", "stage1-replay.yml"),
         findings=("KAI-GATE-048",)),
    Gate(module="test_p1_replay_completeness",
         kind=GATE,
         summary="the two replay-completeness axes must NEVER substitute "
                 "for one another -- a spotless capture with no call-path "
                 "source must read as REQUEST_INCOMPLETE_POSITIONAL, not "
                 "as replayable, because the probe records positional "
                 "arguments nowhere; and the both-defects case must never "
                 "collapse into either single verdict, because kwargs "
                 "completeness and positional completeness have different "
                 "repairs",
         inputs=(),
         denominator=r"inspected: \d+ P1 verdict\(s\) discriminated",
         proven_by="scripts/test_p1_replay_completeness.py",
         calibrated_by="scripts/test_p1_replay_completeness.py",
         in_policy_check=False,
         in_workflows=("p1-replay-completeness.yml",),
         findings=("KAI-GATE-048",)),
    Gate(module="p1_replay_completeness",
         kind=REPORT,
         summary="whether a captured request can be replayed faithfully, "
                 "on two axes that may never substitute for one another: "
                 "keyword completeness measured from the run's own "
                 "artifact, and positional completeness established from "
                 "the call path's source inside the image that ran it; "
                 "emits one of five verdicts and never 'probably "
                 "complete'",
         inputs=(),
         denominator=r"inspected: \d+ production request row\(s\)",
         probe=False,
         probe_skip_reason="needs a capture file produced by driving "
                           "cognee in-process inside the memu-graph image "
                           "AND that image's site-packages; on the host "
                           "neither exists. Every verdict, both axes and "
                           "each refusal path are asserted on synthetic "
                           "captures and synthetic source trees in "
                           "scripts/test_p1_replay_completeness.py",
         proven_by="scripts/test_p1_replay_completeness.py",
         calibrated_by="scripts/test_p1_replay_completeness.py",
         in_policy_check=False,
         in_workflows=("p1-replay-completeness.yml",),
         findings=("KAI-GATE-048",)),
    Gate(module="report_runtime_topology",
         kind=REPORT,
         summary="what the tree DEFINES, what it GATES, and what a "
                 "repo-defined path actually STARTS; a never-started "
                 "service is usually CORRECT, because the P0 containment "
                 "model requires consequential services to be gated",
         inputs=COMPOSE_FILES,
         denominator=r"inspected: \d+ service definition\(s\)",
         proven_by="scripts/test_runtime_topology.py",
         calibrated_by="scripts/test_runtime_topology.py",
         in_policy_check=False,
         findings=("KAI-GATE-046",)),
    Gate(module="report_embedding_backends",
         kind=REPORT,
         summary="every service choosing between a semantic backend and a "
                 "fallback; 2 of 3 degrade SILENTLY, so 'service started' "
                 "is not evidence the semantic backend started",
         inputs=COMPOSE_FILES,
         denominator=r"inspected: \d+ service\(s\) choosing between",
         proven_by="scripts/test_embedding_backends.py",
         calibrated_by="scripts/test_embedding_backends.py",
         in_policy_check=False,
         in_workflows=()),
    Gate(module="probe_embedding_backend",
         kind=REPORT,
         summary="runs INSIDE a built image and reports whether the "
                 "semantic operation actually executed; the verdict is the "
                 "exit code, never a grep over its output",
         inputs=(),
         denominator=r"inspected: 3 stage\(s\) of .*semantic path",
         proven_by="scripts/test_embedding_backends.py",
         calibrated_by="scripts/test_embedding_backends.py",
         probe=False,
         probe_skip_reason="it is designed to run inside a service "
                           "container, where the library and baked model "
                           "exist. Probing it on the developer host would "
                           "measure the host, and reporting that as the "
                           "image's state is the exact confusion this "
                           "instrument exists to prevent.",
         in_policy_check=False,
         in_workflows=()),
    Gate(module="generate_service_keys",
         kind=REPORT,
         summary="generates one ed25519 keypair per service and the trusted "
                 "receiver key map; NOT a gate — it writes deployment key "
                 "material and must never run in policy-check",
         inputs=(),
         denominator=r"inspected: \d+ service key\(s\) generated",
         proven_by="scripts/test_service_identity_wiring.py",
         calibrated_by="scripts/test_service_identity_wiring.py",
         probe=False,
         probe_skip_reason="probing it would WRITE PRIVATE KEY MATERIAL. A "
                           "meta-check must not create secrets as a side "
                           "effect of measuring, and a generator run with no "
                           "arguments correctly refuses rather than emitting "
                           "an empty key map. The denominator is exercised in "
                           "scripts/test_service_identity_wiring.py against a "
                           "temporary directory instead.",
         in_policy_check=False,
         in_workflows=()),
    Gate(module="check_service_identity_wiring",
         kind=GATE,
         summary="a private signing key must be mounted into exactly ONE "
                 "service — two services sharing a key are one principal, "
                 "which is the measured defect this mechanism removes",
         inputs=COMPOSE_FILES,
         denominator=r"inspected: \d+ service\(s\) that sign or verify",
         proven_by="scripts/test_service_identity_wiring.py",
         calibrated_by="scripts/test_service_identity_wiring.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",)),
    Gate(module="check_bind_mount_portability",
         kind=GATE,
         summary="a bind mount must not name a path that exists on one "
                 "machine — Docker creates a missing source as an EMPTY "
                 "directory, so the service boots healthy and reads nothing",
         inputs=COMPOSE_FILES,
         denominator=r"inspected: \d+ bind mount\(s\)",
         proven_by="scripts/test_bind_mount_portability.py",
         calibrated_by="scripts/test_bind_mount_portability.py",
         in_policy_check=True,
         in_workflows=("policy-checks.yml",)),
    Gate(module="ci/post_mortem",
         kind=REPORT,
         summary="print the captured step logs that have content, and "
                 "name the empty ones in one line instead of a section "
                 "each — the noise is what evicted the real output",
         inputs=(),
         denominator=r"inspected: \d+ captured step log\(s\)",
         probe=False,
         probe_skip_reason="reads the log files a failed run left in "
                           "/tmp; probed by scripts/test_post_mortem.py "
                           "against fixtures shaped like run 712, where "
                           "one section had output and twelve did not",
         proven_by="scripts/test_post_mortem.py",
         in_workflows=("core-tests.yml",)),
    Gate(module="ci/assert_clean_bringup",
         kind=GATE,
         summary="a bring-up that warned did not fully succeed",
         inputs=(),
         denominator=r"inspected: \d+ line\(s\) of bring-up output",
         probe=False,
         probe_skip_reason="takes the log a bring-up wrote; probed by "
                           "scripts/test_bringup_guards.py against "
                           "synthetic logs instead",
         proven_by="scripts/test_bringup_guards.py",
         in_workflows=("core-tests.yml",)),
    Gate(module="ci/make_dev_secrets",
         kind=GATE,
         summary="every file-backed secret the profile declares exists "
                 "before the bring-up needs it",
         inputs=("docker-compose.full.yml",),
         denominator=r"inspected: \d+ file-backed secret\(s\)",
         proven_by="scripts/test_bringup_guards.py",
         in_workflows=("core-tests.yml",)),
    Gate(module="test_restart_persistence",
         kind=GATE,
         summary="a memory written before a memu-core restart is "
                 "retrievable after it",
         inputs=("docker-compose.minimal.yml",),
         denominator=r"\[4/4\]",
         probe=False,
         probe_skip_reason="writes to a live memu-core and restarts its "
                           "container; probed by scripts/test_ci_scripts.py "
                           "against an injected caller instead",
         proven_by="scripts/test_ci_scripts.py",
         in_workflows=("core-tests.yml",)),
    Gate(module="sync_docs",
         kind=GATE,
         summary="README and backlog metrics match the codebase",
         inputs=("README.md", "docs/PROJECT_BACKLOG.md"),
         denominator=r"Tests:\s+[\d,]+ functions in \d+ files",
         probe=False,
         probe_skip_reason="`--check` is the gating form and exits 1 on "
                           "drift, so probing it here would fail the "
                           "meta-check whenever docs are stale rather than "
                           "reporting a denominator; probed by "
                           "scripts/test_ci_scripts.py against its counters",
         proven_by="scripts/test_ci_scripts.py",
         # Two callers with different intent: core-tests gates on it via
         # `make check-docs`, friday-cleanup reports it advisorily. Both
         # are real and both are declared.
         in_workflows=("core-tests.yml", "friday-cleanup.yml")),
    Gate(module="go_no_go_check",
         kind=GATE,
         summary="the dashboard's go/no-go decision is GO — and an "
                 "unreachable dashboard is not a GO",
         inputs=(),
         denominator=r"go_no_go: (PASS|FAIL|SKIPPED)",
         probe=False,
         probe_skip_reason="polls a dashboard that is not up here, and "
                           "failing on that is now the point; probed by "
                           "scripts/test_ci_scripts.py in all four "
                           "directions instead",
         proven_by="scripts/test_ci_scripts.py",
         # Reached through `make go_no_go`, not named in a step. The
         # meta-check follows make targets now, so this is the answer it
         # gets rather than the one I guessed.
         in_workflows=("core-tests.yml",)),
)

BY_MODULE = {gate.module: gate for gate in REGISTRY}
