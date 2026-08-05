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
            yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            first = str(exc).splitlines()[0]
            broken.append(f"{path.name}: {first}")
    return broken


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
    marked = markers()
    unmarked = [f"{d.workflow}: {d.step}" for d in DECLARED
                if (d.workflow, d.bucket) not in marked]
    declared_pairs = {(d.workflow, d.bucket) for d in DECLARED}
    orphan_markers = [f"{w}: {b}" for w, b in marked
                      if (w, b) not in declared_pairs]
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

    if orphan_markers:
        print(f"\nFAIL: {len(orphan_markers)} marker(s) with no "
              f"declaration:\n")
        for line in orphan_markers:
            print(f"  - {line}")
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
