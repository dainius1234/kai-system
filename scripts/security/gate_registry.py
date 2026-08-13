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
         in_workflows=("core-tests.yml",)),
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
