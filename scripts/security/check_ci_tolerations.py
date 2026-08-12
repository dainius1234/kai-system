#!/usr/bin/env python3
"""Every CI step that can pass without doing its job must say so.

The operator's rule, which this file encodes:

> **Zero tolerance for silent failure. High tolerance for documented
> skips with a reason and an owner.** If a CI step can't run reliably, it
> should say so loudly and point to who owns making it reliable. What it
> can't do is say "passed" when it did nothing.

A survey of the six workflows found 24 tolerant patterns. Most are not
defects, and saying so matters as much as the findings:

  - **6 are icon ternaries** — ``[ "$CODE" = "0" ] && echo "✅" || echo
    "⚠️"``. Report formatting, not error suppression.
  - **8 are install tolerance** — ``pip install psutil || true``. Checked
    empirically: none of the five suites behind those installs skips on a
    missing import, so the dep going absent makes the *test* fail. The
    tolerance is on the install; the test is still the gate.
  - **3 are legitimate absence handling** — ``grep ... || true`` where no
    match is a valid outcome, and a GNU/BSD ``date`` fallback.

That leaves the ones that genuinely let a check pass while doing nothing.
They are declared below with a reason, an owner and a review date, and
this gate fails on any tolerant pattern in a workflow that is **not**
declared — so a tenth added next month is visibly undeclared rather than
quietly absorbed.

**A warning is a suppression too.** Added 2026-08-05, after
`The "DB_PASSWORD" variable is not set` printed on every compose
invocation in every log for a day and was read past — by me, in three
reports, aloud — while postgres refused to start because of it.

A step that prints `::warning::` and carries on has decided that
something is not worth failing over. That is the same decision as
swallowing an exit code, and it needs the same three things: a reason, an
owner, and a date. Otherwise it recurs forever, and a signal that recurs
forever is one nobody reads.

The operator's rule, which this encodes:

> **Nothing repeats unexplained.** A recurring signal is fixed, made to
> fail, or declared — with a name against it and a date on it. The middle
> ground, printing forever, is what teaches everyone to ignore it.

Three buckets, per the operator:

  ``DOCUMENTED_SKIP``  best-effort by design. Must print an explicit
                       SKIPPED line, not a warning that reads like a pass.
  ``NEEDS_OWNER``      should enforce, blocked on something external.
                       Carries an owner and a review date.
  ``DEFECT``           should enforce and nothing blocks it. Fix it.

Exit 0 = every toleration is declared.  Exit 1 = one is not.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
WORKFLOWS = REPO / ".github" / "workflows"

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import inspected, require  # noqa: E402

DOCUMENTED_SKIP = "documented-skip"
NEEDS_OWNER = "needs-owner"
DEFECT = "defect"


@dataclass(frozen=True)
class Toleration:
    workflow: str
    step: str            # substring of the step name
    bucket: str
    reason: str
    owner: str
    review_by: str       # ISO date; a skip without an expiry is forever


DECLARED: Tuple[Toleration, ...] = (
    # #41-B deployed. The dashboard is the SUBJECT here, so the
    # collector records its self-report last and labelled, and takes
    # ground truth from a socket probe and `docker inspect` instead.
    Toleration(
        workflow="degradation-deployed-proof.yml",
        step="Deployed degradation evidence",
        bucket=DOCUMENTED_SKIP,
        reason="The collector's `stage()` records every command's exit "
               "status as evidence and continues, because a probe that "
               "proves a dependency ABSENT is a successful measurement of "
               "an intended state, not a build to break. Profiles-off is "
               "the correct posture, so nearly every command here is "
               "expected to fail in a specific way, and the specific way "
               "IS the result. It exits 2 only when the measurement "
               "cannot be trusted at all — a profile leaking in. Retire "
               "when #53's disposition is decided and this becomes a "
               "regression gate.",
        owner="orion",
        review_by="2026-11-01",
    ),
    Toleration(
        workflow="degradation-deployed-proof.yml",
        step="Evidence summary",
        bucket=DOCUMENTED_SKIP,
        reason="`cat ... || echo (not executed = UNKNOWN)` — a missing "
               "measurement must read as UNKNOWN, never as silence and "
               "never as success. It does not judge completeness: the "
               "caller-logic denominator is already banked, and this job "
               "deliberately exercises a subset, so a completeness "
               "verdict here would imply a coverage claim the job does "
               "not make.",
        owner="orion",
        review_by="2026-11-01",
    ),
    # One declaration per service, because the match is on the step name
    # and there is one INDEPENDENT collector step per service. They
    # replace two earlier declarations — a separate "Diagnose image
    # resolution" step, and a single Claim A step that looped over all
    # three services. The loop was the defect: run #2 failed that one
    # step when its evidence was incomplete, and GitHub then SKIPPED
    # Claim B. One measurement suppressed an independent one.
    Toleration(
        workflow="embedding-backend-proof.yml",
        step="Claim A — memu-core",
        bucket=DOCUMENTED_SKIP,
        reason="The collector exits 0 for every DEFINED probe verdict and "
               "non-zero ONLY on instrument malfunction, so this step "
               "cannot fail on a finding. That is deliberate: a probe "
               "proving the semantic backend absent is a SUCCESSFUL "
               "MEASUREMENT of a FAILED capability, and failing here "
               "would let one service's negative result skip the "
               "independent measurements after it. Completeness is judged "
               "once, last, by `Evidence summary`. Retire when the "
               "class-level remediation lands and this becomes a gate.",
        owner="orion",
        review_by="2026-11-01",
    ),
    Toleration(
        workflow="embedding-backend-proof.yml",
        step="Claim A — agentic",
        bucket=DOCUMENTED_SKIP,
        reason="Same collector, same argument as memu-core, declared "
               "separately because the match is on the step name and each "
               "differently-named step is its own decision. agentic is "
               "the service whose Claim A was UNRESOLVED in run #2 with "
               "the failing stage unknowable, so this row existing at all "
               "is the point of the restructure.",
        owner="orion",
        review_by="2026-11-01",
    ),
    Toleration(
        workflow="embedding-backend-proof.yml",
        step="Claim A — fusion-engine",
        bucket=DOCUMENTED_SKIP,
        reason="Same collector, same argument, declared separately for "
               "the same reason. fusion-engine built successfully in run "
               "#2 and `config --images` still exposed no image name for "
               "it; the cause is NOT yet measured, so nothing here names "
               "one. The collector records each resolution stage's "
               "command, stdout, stderr and exit status so the next run "
               "answers that from evidence.",
        owner="orion",
        review_by="2026-11-01",
    ),
    Toleration(
        workflow="embedding-backend-proof.yml",
        step="Claim B — does memu-core reach that backend on its production default?",
        bucket=DOCUMENTED_SKIP,
        reason="Two suppressions. `printenv || echo (unset)` is "
               "CORROBORATION ONLY — an absent variable is the expected "
               "and correct state for the production default, so it must "
               "not fail; the authoritative result is the vector width, "
               "captured separately with its own exit status. `down -v || "
               "true` is teardown after the measurement is already "
               "recorded. Retire with the rest of this job.",
        owner="orion",
        review_by="2026-11-01",
    ),
    Toleration(
        workflow="embedding-backend-proof.yml",
        step="Evidence summary",
        bucket=DOCUMENTED_SKIP,
        reason="The declared item is the two `cat ... || echo` fallbacks, "
               "which print an explicit `(not executed = UNKNOWN)` when a "
               "result file is absent. That is the I-1 behaviour this "
               "programme requires: a missing measurement must read as "
               "UNKNOWN, never as silence and never as success. The step "
               "itself is NOT tolerant — it is the job's only "
               "completeness judgement and exits 1 when fewer than three "
               "Claim-A measurements or no Claim-B measurement exist. It "
               "can do that safely only because it is last and the "
               "artifact upload after it runs `if: always()`, so failing "
               "here suppresses nothing. The verdict is about whether the "
               "EVIDENCE SET is complete, never about what the evidence "
               "says: a FAKE result passes it, an unresolved image does "
               "not.",
        owner="orion",
        review_by="2026-11-01",
    ),
    Toleration(
        workflow="core-tests.yml",
        step="Measured — baked model revision and embedding load time",
        bucket=DOCUMENTED_SKIP,
        reason="Pure instrumentation: it reads the model revision baked "
               "into the image and the measured embedding load time so "
               "start-period can be tuned from a number rather than a "
               "guess. It must never fail a build, because a missing "
               "measurement is not a broken stack — the steps that judge "
               "the stack are the bring-up and the live smoke either side "
               "of it. Both fallbacks print a specific reason rather than "
               "an empty line, so an absent value is visible as an absent "
               "value. Retire this once the revision is pinned and "
               "start-period is set from a measured figure.",
        owner="orion",
        review_by="2026-09-15",
    ),
    Toleration(
        workflow="core-tests.yml",
        step="GitHub Models CI backend smoke test",
        bucket=DOCUMENTED_SKIP,
        reason="Free-tier GitHub Models is rate-limited by design, so a "
               "failure here is usually throttling rather than a defect. "
               "It must print SKIPPED with the reason, not a warning that "
               "reads like a pass.",
        owner="operator",
        review_by="2026-11-01",
    ),
    Toleration(
        workflow="core-tests.yml",
        step="Run memu-graph live ingest/query/forget cycle",
        bucket=NEEDS_OWNER,
        reason="This is live verification — the thing this programme is "
               "shortest of — and it is currently allowed to fail every "
               "run forever. It depends on Ollama model quality and an "
               "extension.kuzudb.com download, neither of which is under "
               "our control today. It should enforce once those are "
               "pinned; until then the skip is explicit and dated.",
        owner="operator",
        review_by="2026-10-01",
    ),
    Toleration(
        workflow="core-tests.yml",
        step="Dependency vulnerability scan",
        bucket=NEEDS_OWNER,
        reason="pip-audit finds real CVEs (urllib3 PYSEC-2026-141/142, "
               "wheel CVE-2026-24049) in transitive dependencies we do "
               "not pin. Enforcing today would make CI permanently red, "
               "which is an ignored gate. Needs an ignore-file with "
               "per-CVE expiry before it can enforce. `2>/dev/null` is "
               "removed: it was hiding that --strict exits 1 for an "
               "unauditable *system* package, so the warning named the "
               "wrong cause.",
        owner="operator",
        review_by="2026-10-01",
    ),
    Toleration(
        workflow="core-tests.yml",
        step="Scan container images for vulnerabilities",
        bucket=NEEDS_OWNER,
        reason="Same argument as pip-audit, one layer down: trivy reports "
               "CRITICAL/HIGH CVEs in the base images and their distro "
               "packages, which we do not control and cannot patch by "
               "pinning our own requirements. Enforcing today makes CI "
               "permanently red, and a permanently red gate is an ignored "
               "gate. The *findings* are advisory; a scanner that could "
               "not execute is not — the follow-on step exits 1 on any "
               "outcome other than success or failure, so a missing "
               "binary, a rate-limited DB download or a syntax error "
               "still breaks the build. Tracked for remediation as "
               "KAI-GATE-026.",
        owner="operator",
        review_by="2026-11-01",
    ),
    Toleration(
        workflow="python-app.yml",
        step="Dependency vulnerability scan",
        bucket=NEEDS_OWNER,
        reason="Same scan, same blocker as core-tests.yml. Declared "
               "separately so removing one does not silently leave the "
               "other.",
        owner="operator",
        review_by="2026-10-01",
    ),
    Toleration(
        workflow="core-tests.yml",
        step="Vulnerability findings are advisory",
        bucket=NEEDS_OWNER,
        reason="The reporting half of the trivy toleration: it turns the "
               "scan's outcome into a visible line and warns when "
               "CRITICAL/HIGH findings exist. It emits `::warning::` on "
               "every run where they do, which under the rule above needs "
               "an owner and a date of its own — the scan step's "
               "declaration does not cover a differently-named step. "
               "Found by the warning rule on its first run, on a step "
               "written earlier the same day.",
        owner="operator",
        review_by="2026-11-01",
    ),
    Toleration(
        workflow="core-tests.yml",
        step="Dump container logs on failure",
        bucket=DOCUMENTED_SKIP,
        reason="Runs only `if: failure()`, so the job is already red and "
               "nothing this step does can turn it green — it has no "
               "enforcement to skip. Every command tolerates its own exit "
               "code deliberately: the step exists to describe a failure, "
               "and a diagnostic that aborts before printing the next "
               "fact is worse than no diagnostic. Added on 2026-08-05 "
               "because the bring-up above it failed in one second on the "
               "first run that ever reached it, and the log API serves "
               "only a tail — so the questions have to be asked after the "
               "failure, where the answers land in readable range.",
        owner="orion",
        review_by="2027-01-01",
    ),
    Toleration(
        workflow="core-tests.yml",
        step="Dump full-profile logs on failure",
        bucket=DOCUMENTED_SKIP,
        reason="The same argument as the minimal-profile dump step, "
               "declared separately because the match is on the step name "
               "and 'Dump container logs on failure' does not cover a "
               "differently-named step — the gate caught exactly that on "
               "the run these two lines were added. Runs only "
               "`if: failure()`, so the job is already red and nothing "
               "here can turn it green; it has no enforcement to skip. "
               "Both `|| true`s are deliberate: a diagnostic that aborts "
               "before printing the next fact is worse than no diagnostic, "
               "and replacing the original failure with a dump failure "
               "hides the thing the step exists to explain.",
        owner="orion",
        review_by="2027-01-01",
    ),
    Toleration(
        workflow="core-tests.yml",
        step="Dump sovereign logs on failure",
        bucket=DOCUMENTED_SKIP,
        reason="The third of three profile dump steps, declared "
               "separately for the same reason as the second: the match "
               "is on the step name, and each differently-named step is "
               "its own decision needing its own owner. `if: failure()` "
               "only, so the job is already red and this has no "
               "enforcement to skip. Its `|| true`s are deliberate — a "
               "diagnostic that aborts before printing the next fact is "
               "worse than no diagnostic, and replacing the real failure "
               "with a dump failure destroys what the step exists to "
               "produce. Gained them on 2026-08-06 when it was narrowed "
               "to `dump_unhealthy`: it had named three services by hand, "
               "which is a hand-written list of the one thing not "
               "knowable in advance — which container broke.",
        owner="orion",
        review_by="2027-01-01",
    ),
    Toleration(
        workflow="core-tests.yml",
        step="Post-mortem",
        bucket=DOCUMENTED_SKIP,
        reason="Same argument as the dump step, and it exists because that "
               "one was unreadable: the log API serves a fixed-size tail "
               "and the eleven teardown steps after it push forty warning "
               "lines each into the window. This one is last in the file, "
               "so nothing can displace it. Runs only `if: failure()`; it "
               "has no enforcement to skip, and every command tolerates "
               "its own exit code so one missing tool cannot stop the "
               "next fact from printing.",
        owner="orion",
        review_by="2027-01-01",
    ),
    Toleration(
        workflow="friday-cleanup.yml",
        step="Post Friday cleanup issue",
        bucket=DOCUMENTED_SKIP,
        reason="Issue creation can fail because the label does not exist "
               "yet. The report is the product; failing the whole weekly "
               "job because a label is missing would be worse. Prints "
               "SKIPPED so a silently absent report is distinguishable "
               "from one nobody read.",
        owner="operator",
        review_by="2026-11-01",
    ),
    Toleration(
        workflow="weekly-report-card.yml",
        step="Post Weekly Report Card issue",
        bucket=DOCUMENTED_SKIP,
        reason="Same as the Friday cleanup issue step.",
        owner="operator",
        review_by="2026-11-01",
    ),
    Toleration(
        workflow="unified-hunter.yml",
        step="Report the surface that was exercised",
        bucket=DOCUMENTED_SKIP,
        reason="Purely informational and runs `if: always()`. The gate is "
               "the preceding assertion-ratchet step, which does not "
               "tolerate anything. This one only prints what was counted.",
        owner="orion",
        review_by="2027-01-01",
    ),
)

# Patterns that suppress a non-zero exit. Install tolerance, absence
# handling and icon ternaries are excluded by shape, not by exception:
# each is a different construct, not the same construct forgiven.
# Only the unambiguous swallows. `if cmd; then ... else ... fi` also
# suppresses, and adding it here matched every ordinary shell
# conditional in the repository — a false-positive machine, which is the
# defect this file is meant to avoid producing.
#
# Detecting that shape structurally is guesswork, so it is not guessed.
# Every tolerated step carries an explicit `# ci-toleration:` marker
# instead, cross-checked against the declarations below in both
# directions. A step that swallows an exit code without a marker, or a
# marker with no declaration, or a declaration with no marker, all fail.
# That is the same shape as I-4, and it does not depend on recognising a
# shell idiom.
_SUPPRESSION = re.compile(r"\|\|\s*(echo|true)\b|continue-on-error:\s*true")
_MARKER = re.compile(r"#\s*ci-toleration:\s*(\S+)")
_INSTALL = re.compile(r"pip install|-f requirements\.txt")
_ICON_TERNARY = re.compile(r'&&\s*echo\s*"[^"]*"\s*\|\|\s*echo')
_ABSENCE = re.compile(r"grep .*\|\| true|date -u .*\|\| true|--jq .*\|\| true")


def _step_name(lines: List[str], index: int) -> str:
    for i in range(index, -1, -1):
        match = re.search(r"^\s*-\s*name:\s*(.+)$", lines[i])
        if match:
            return match.group(1).strip()
    return "<unnamed>"


def survey() -> Tuple[List[Tuple[str, int, str, str]], int]:
    """Return (suppressions, lines_scanned)."""
    found: List[Tuple[str, int, str, str]] = []
    scanned = 0
    for path in sorted(WORKFLOWS.glob("*.yml")):
        lines = path.read_text(encoding="utf-8").splitlines()
        scanned += len(lines)
        for i, line in enumerate(lines):
            # A comment cannot swallow an exit code. This scanned raw
            # lines, so a comment *explaining* a `|| true` was reported
            # as a second `|| true` — which trains people to describe
            # their code less precisely to keep a gate quiet, the exact
            # opposite of what this file is for. Caught when a step whose
            # comment said "`|| true` twice, deliberately" was reported
            # three times.
            if line.lstrip().startswith("#"):
                continue
            if not _SUPPRESSION.search(line):
                continue
            if (_INSTALL.search(line) or _ICON_TERNARY.search(line)
                    or _ABSENCE.search(line)):
                continue
            found.append((path.name, i + 1, _step_name(lines, i), line.strip()))
    return found, scanned


def markers() -> List[Tuple[str, str]]:
    """(workflow, bucket) for every `# ci-toleration:` marker."""
    out = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            match = _MARKER.search(line)
            if match:
                out.append((path.name, match.group(1)))
    return out


_WARNING = re.compile(r"::warning::")


def warning_emitters() -> List[Tuple[str, str]]:
    """(workflow, step name) for every step that prints `::warning::`.

    Read from the parsed workflow rather than by line, because a warning
    belongs to the step that emits it and only the parse knows which step
    a line is in.
    """
    import yaml
    out: List[Tuple[str, str]] = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue            # `unparseable()` owns that finding
        for job in (doc.get("jobs") or {}).values():
            for step in (job.get("steps") or []):
                if _WARNING.search(step.get("run") or ""):
                    out.append((path.name, step.get("name") or "<unnamed>"))
    return out


def undeclared_warnings() -> List[str]:
    """Steps that warn on every run without an owner or a date."""
    out = []
    for workflow, step in warning_emitters():
        if not any(d.workflow == workflow and d.step in step
                   for d in DECLARED):
            out.append(f"{workflow}: '{step}' emits ::warning:: but is not "
                       f"declared. A warning that recurs with nobody "
                       f"answerable for it is a signal nobody reads.")
    return out


def orphan_declarations() -> List[str]:
    """Declarations naming a step that exists in no workflow.

    The drift check below it compares `(workflow, bucket)` pairs, so a
    declaration could name a step that had been renamed or deleted and
    still be counted as matched by some *other* marker in the same file
    carrying the same bucket. That is R5 in its usual shape: a check
    whose scope is smaller than its name implies. "The record and the
    file have drifted apart" is a claim about STEPS, and it was being
    tested at the granularity of BUCKETS.

    Calibrated when this was written: restructuring
    `embedding-backend-proof.yml` into one collector per service left
    exactly two declarations pointing at steps that no longer existed —
    a deleted `Diagnose image resolution` and a `Claim A` step that had
    looped over all three services. This function reported those two and
    none of the other sixteen. Known-positive and known-negative from the
    same run, and the expected answer came from the workflow files rather
    than from this list.
    """
    import yaml
    live: dict = {}
    for path in sorted(WORKFLOWS.glob("*.yml")):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue            # `unparseable()` owns that finding
        names = []
        for job in (doc.get("jobs") or {}).values():
            for step in ((job or {}).get("steps") or []):
                names.append((step or {}).get("name") or "")
        live[path.name] = names
    out = []
    for d in DECLARED:
        if not any(d.step in name for name in live.get(d.workflow, [])):
            out.append(f"{d.workflow}: '{d.step}' is declared but no step "
                       f"of that name exists. A toleration for a step that "
                       f"is gone reads as coverage nobody has.")
    return out


def unparseable() -> List[str]:
    """Workflows a YAML parser rejects.

    A workflow that does not parse runs nothing, and "runs nothing" is
    indistinguishable from "has no failures" in exactly the way this
    whole programme is about. Found while adding the toleration gate:
    three files terminate a `run: |` block early because an embedded
    `python3 -c "` heredoc starts at column 0.
    """
    import yaml
    broken = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            first = str(exc).splitlines()[0]
            broken.append(f"{path.name}: {first}")
            continue
        broken.extend(_schema_violations(path.name, doc))
    return broken


#: Keys GitHub accepts on a step. Anything else is a typo that GitHub
#: rejects, and a rejected workflow schedules nothing.
_STEP_KEYS = {"name", "id", "if", "uses", "run", "with", "env", "shell",
              "working-directory", "continue-on-error", "timeout-minutes"}


def _schema_violations(name: str, doc: dict) -> List[str]:
    """Shapes that parse as YAML but that GitHub refuses to run.

    **Parseable is not runnable, and this file assumed they were the
    same.** `unparseable()` was written because "a workflow that does not
    parse runs nothing, and running nothing is indistinguishable from
    having no failures". That reasoning is right and its scope was one
    member of the class: it asked whether *Python* could read the file,
    not whether *GitHub* would run it.

    Found on 2026-08-06. `policy-checks.yml` had

        - name: Test wiring — no test defined and never called
        - name: Every jq filter embedded in a workflow compiles
          run: python scripts/security/check_workflow_filters.py

    — a step with a name and no body. It is valid YAML and invalid
    GitHub. Every run of that workflow completed as `failure` with
    **zero jobs scheduled**, having executed none of its 30 steps, on
    every push for at least a day. Roughly fifteen gates declare
    `in_workflows=("policy-checks.yml",)`; none of them had run in CI.

    It survived because a startup failure looks like any other red
    workflow from the outside, and because the only workflow anyone was
    watching was `core-tests.yml`.

    Deliberately narrow: a step must have `run` or `uses`, a job must
    have `runs-on` or `uses`, and step keys must be ones GitHub knows.
    Those are unambiguous. GitHub's full schema is larger, and guessing
    at the rest would produce findings against workflows that run fine —
    the inverse defect this programme keeps re-learning.
    """
    out: List[str] = []
    jobs = doc.get("jobs") or {}
    if not jobs:
        out.append(f"{name}: declares no jobs, so it runs nothing")
    for job_name, job in jobs.items():
        job = job or {}
        if not job.get("runs-on") and not job.get("uses"):
            out.append(f"{name}: job '{job_name}' has no `runs-on` — "
                       f"GitHub cannot schedule it")
        for index, step in enumerate(job.get("steps") or []):
            step = step or {}
            label = step.get("name") or f"step {index}"
            if "run" not in step and "uses" not in step:
                out.append(
                    f"{name}: job '{job_name}' step {index} "
                    f"({label!r}) has neither `run` nor `uses`. GitHub "
                    f"rejects the whole file, schedules no jobs, and the "
                    f"run completes as a failure having executed nothing "
                    f"— which reads exactly like an ordinary red build.")
            unknown = sorted(set(step) - _STEP_KEYS)
            if unknown:
                out.append(
                    f"{name}: job '{job_name}' step {index} ({label!r}) "
                    f"has key(s) GitHub does not accept: {unknown}")
    return out


def main() -> int:
    require((".github/workflows",))
    suppressions, scanned = survey()
    broken = unparseable()

    undeclared: List[str] = []
    for workflow, lineno, step, text in suppressions:
        if not any(d.workflow == workflow and d.step in step
                   for d in DECLARED):
            undeclared.append(f"{workflow}:{lineno} [{step}] {text[:60]}")

    # Both directions, so neither a marker nor a declaration can drift
    # away from the other unnoticed.
    unowned = undeclared_warnings()
    marked = markers()
    unmarked = [f"{d.workflow}: {d.step}" for d in DECLARED
                if (d.workflow, d.bucket) not in marked]
    declared_pairs = {(d.workflow, d.bucket) for d in DECLARED}
    orphan_markers = [f"{w}: {b}" for w, b in marked
                      if (w, b) not in declared_pairs]
    orphan_steps = orphan_declarations()
    stale = unmarked

    print(inspected(scanned, "workflow lines",
                    f"across {len(list(WORKFLOWS.glob('*.yml')))} workflows"))
    print(f"\n  {len(suppressions)} suppression(s), {len(DECLARED)} declared")
    for bucket in (DEFECT, NEEDS_OWNER, DOCUMENTED_SKIP):
        entries = [d for d in DECLARED if d.bucket == bucket]
        if entries:
            print(f"\n  {bucket} ({len(entries)}):")
            for d in entries:
                print(f"    - {d.workflow}: {d.step}")
                print(f"        owner={d.owner}  review by {d.review_by}")

    if broken:
        print(f"\nFAIL: {len(broken)} workflow(s) do not parse as YAML:\n")
        for line in broken:
            print(f"  - {line}")
        print("\n  A workflow that does not parse runs nothing, and running "
              "nothing is\n  indistinguishable from having no failures.")
        return 1

    if undeclared:
        print(f"\nFAIL: {len(undeclared)} undeclared suppression(s):\n")
        for line in undeclared:
            print(f"  - {line}")
        print("\n  A CI step that can pass without doing its job must be "
              "declared\n  with a reason, an owner and a review date. Silence "
              "is the defect.")
        return 1

    if unowned:
        print(f"\nFAIL: {len(unowned)} step(s) warn on every run without an "
              f"owner:\n")
        for line in unowned:
            print(f"  - {line}")
        print("\n  Nothing repeats unexplained. A recurring signal is "
              "fixed, made to\n  fail, or declared with a name against it "
              "and a date on it.")
        return 1

    if orphan_markers:
        print(f"\nFAIL: {len(orphan_markers)} marker(s) with no "
              f"declaration:\n")
        for line in orphan_markers:
            print(f"  - {line}")
        return 1

    if orphan_steps:
        print(f"\nFAIL: {len(orphan_steps)} declaration(s) name a step that "
              f"does not exist:\n")
        for line in orphan_steps:
            print(f"  - {line}")
        print("\n  Drift is a claim about steps. Checking it per bucket "
              "would let a\n  renamed or deleted step keep its toleration "
              "alive behind another one.")
        return 1

    if stale:
        print(f"\nFAIL: {len(stale)} declaration(s) carry no "
              f"`# ci-toleration:` marker in the workflow — the record and "
              f"the file have drifted apart:\n")
        for line in stale:
            print(f"  - {line}")
        return 1

    print("\nPASS: every CI suppression is declared, owned and dated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
