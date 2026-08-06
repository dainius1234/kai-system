#!/usr/bin/env python3
"""A compose variable with no default must be supplied by the step that uses it.

`docker-compose.minimal.yml` declares

    POSTGRES_PASSWORD: ${DB_PASSWORD}

with no `:-` default, and that is correct — a default password in a
shipped compose file is precisely what `check_secret_fallbacks` forbids.
The consequence is that whoever brings the stack up must supply one.

On 2026-08-05 the bring-up step in `core-tests.yml` did not. The only
step in the workflow that set `DB_PASSWORD` was the sovereign boot, 150
lines further down. postgres refuses to initialise with an empty
superuser password, so the container went unhealthy and every dependent
failed with `dependency failed to start`.

**Compose said so, every single time:**

    The "DB_PASSWORD" variable is not set. Defaulting to a blank string.

That line appeared in every compose invocation in the workflow, in every
log read that day, and was filtered as noise. The evidence was never
missing. This gate exists because a warning nobody reads is worth the
same as no warning at all — the H-6 lesson, arriving from the outside.

**Scope is per step, not per file.** A variable set on one step is not
set on another; they are separate processes. Checking the file as a
whole would have reported this repository clean while it was broken,
which is the denominator error this programme keeps finding.

Only variables **without** a `:-` default are in scope. A default is a
deliberate statement that blank is acceptable; this gate is about the
ones that say the opposite.

**Scope is also per service.** `docker compose up -d ollama memu-graph`
starts two services, not the profile. The first version of this gate
ignored the service list and reported six findings, five of which were
Grafana, Tailscale and Vault variables belonging to services that step
never starts. A gate with false positives gets ignored, so the service
list is honoured — together with the `depends_on` closure, because
compose starts those too.

Exit 0 = every step supplies what it needs.  Exit 1 = one does not.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
WORKFLOWS = REPO / ".github" / "workflows"

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import inspected, require  # noqa: E402

#: `${VAR}` with no `:-` default.
_NO_DEFAULT = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")

#: `docker compose -f <file> up [flags] [services...]`
_COMPOSE_UP = re.compile(
    r"docker\s+compose\s+-f\s+(\S+\.yml)\s+up([^\n|&;]*)")

#: Flags that take no value, so anything after them may be a service name.
_UP_FLAGS = {"-d", "--detach", "--build", "--no-build", "--force-recreate",
             "--no-deps", "--wait", "--remove-orphans", "--abort-on-"
             "container-exit", "--pull", "--quiet-pull"}


def named_services(tail: str, defined: Set[str]) -> List[str]:
    """Service names given to `up`, or [] meaning 'the whole profile'.

    Only tokens the profile actually defines count. Everything else on
    that line — flags, a trailing `\\` continuation, a redirect, a pipe
    into `tee` — is shell, not a service.

    **This fails closed on purpose.** The first version treated any
    non-flag token as a service name, so the bring-up step's

        docker compose -f docker-compose.minimal.yml up -d --build \\

    yielded the service list `['\\']`, which matched nothing, which
    scoped the check to no services, which required no variables — and
    the gate passed the exact defect it was written for. Its own
    calibration caught that. Recognising nothing now means "the whole
    profile", which is the safe direction: it can over-report, never
    under-report.
    """
    names = [t for t in tail.split() if t in defined]
    return names


def closure(doc: dict, services: List[str]) -> Set[str]:
    """`services` plus everything they depend on — compose starts those too."""
    defined = doc.get("services") or {}
    seen: Set[str] = set()
    queue = list(services)
    while queue:
        name = queue.pop()
        if name in seen or name not in defined:
            continue
        seen.add(name)
        deps = (defined.get(name) or {}).get("depends_on")
        queue.extend(list(deps) if isinstance(deps, (dict, list)) else [])
    return seen


def defined_services(compose_file: Path) -> Set[str]:
    import yaml
    if not compose_file.exists():
        return set()
    doc = yaml.safe_load(compose_file.read_text(encoding="utf-8")) or {}
    return set(doc.get("services") or {})


def required_by(compose_file: Path, services: List[str]) -> Set[str]:
    """Variables the named services need someone else to provide.

    An empty `services` list means the whole profile, which is what
    `up` with no names does.
    """
    import yaml
    if not compose_file.exists():
        return set()
    text = compose_file.read_text(encoding="utf-8")
    if not services:
        return set(_NO_DEFAULT.findall(text))
    doc = yaml.safe_load(text) or {}
    wanted = closure(doc, services)
    defined = doc.get("services") or {}
    found: Set[str] = set()
    for name in sorted(wanted):
        found |= set(_NO_DEFAULT.findall(yaml.safe_dump(defined.get(name) or {})))
    return found


def steps_bringing_up(doc: dict):
    """(step name, compose file, env available, services named, step env).

    The fifth element is the step's **own** `env:`, kept apart from the
    job-scope env it is merged into. The two answer different questions:
    what is *available* to compose, and what this step *chose to set*.
    Only the second can be a mistake — see `unread_by_compose`.
    """
    out = []
    for job in (doc.get("jobs") or {}).values():
        job_env = {str(k): str(v) for k, v in (job.get("env") or {}).items()}
        for step in (job.get("steps") or []):
            run = step.get("run") or ""
            if not run:
                continue
            step_env = {str(k): str(v)
                        for k, v in (step.get("env") or {}).items()}
            env = dict(job_env)
            env.update(step_env)
            for compose_file, tail in _COMPOSE_UP.findall(run):
                defined = defined_services(REPO / compose_file)
                out.append((step.get("name") or "<unnamed>", compose_file,
                            env, named_services(tail, defined), step_env))
    return out


def unread_by_compose(compose_file: Path, step_env: dict) -> List[str]:
    """Variables the step sets for a bring-up that the profile never reads.

    The other direction of this gate, and the one that was missing.

    Setting a variable compose does not reference is not a harmless
    extra: it *looks* like configuration and is not. On 2026-08-06 the
    full profile's bring-up carried

        env:
          MEMU_ALLOW_FAKE_EMBEDDINGS: "true"

    and `docker-compose.full.yml` never named it. Compose passes a
    variable into a container only when the service asks for it, and
    only `minimal` asked. memu-core raises at import if
    sentence-transformers cannot load and that flag is not true
    (memu-core/app.py:1042), so the container died before it listened
    and compose reported the one fact it had: `is unhealthy`.

    Four CI steps had set that variable, for months, believing they were
    configuring something.

    **Conservative on purpose.** A variable is reported only when the
    compose file does not mention it *anywhere* — not merely when the
    services in scope do not. Anything narrower would report on
    configuration that works, and a scope larger than reality sends
    people to break working code and buries the true finding. This one
    is unambiguous: a name absent from the file can have no effect
    through it.
    """
    if not compose_file.exists():
        # I-1: unreadable input is not a clean bill of health.
        return [f"{compose_file.name}: named by a bring-up step but not in "
                f"the tree — nothing here could be checked"]
    text = compose_file.read_text(encoding="utf-8")
    return sorted(name for name in step_env
                  if f"${{{name}" not in text and f"${name}" not in text)


def audit() -> Tuple[List[str], int, int]:
    """Return (findings, bring-up steps inspected, workflows read)."""
    import yaml

    findings: List[str] = []
    paths = sorted(WORKFLOWS.glob("*.yml"))
    inspected_steps = 0
    for path in paths:
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            # I-1: a workflow that will not parse is a workflow whose
            # steps were never examined, not a workflow with no problems.
            findings.append(f"{path.name}: unreadable ({exc})")
            continue
        for (step_name, compose_file, env, services,
             step_env) in steps_bringing_up(doc):
            inspected_steps += 1
            needed = required_by(REPO / compose_file, services)
            missing = sorted(v for v in needed if v not in env)
            for variable in missing:
                findings.append(
                    f"{path.name}: step '{step_name}' brings up "
                    f"{compose_file}, which needs ${{{variable}}} and "
                    f"declares no default — but the step does not set it. "
                    f"Compose will substitute a blank string and say so in "
                    f"a warning nobody reads.")
            # The other direction: set, but never read.
            for variable in unread_by_compose(REPO / compose_file, step_env):
                findings.append(
                    f"{path.name}: step '{step_name}' sets {variable} for a "
                    f"bring-up of {compose_file}, which never references it. "
                    f"Compose passes a variable into a container only when "
                    f"the service names it, so this configures nothing while "
                    f"looking like it does.")
    return findings, inspected_steps, len(paths)


def main() -> int:
    require((".github/workflows",))
    findings, steps, workflows = audit()

    print(inspected(steps, "compose bring-up step(s)",
                    f"across {workflows} workflows"))
    print()
    if findings:
        print(f"FAIL: {len(findings)} variable(s) that do not connect:\n")
        for line in findings:
            print(f"  - {line}")
        print("\n  A variable set on one step is not set on another — they "
              "are separate\n  processes. Checking the file as a whole would "
              "have called this\n  repository clean while postgres was "
              "refusing to start.\n\n  And a variable the profile never "
              "references configures nothing while\n  looking like it does: "
              "MEMU_ALLOW_FAKE_EMBEDDINGS was set by four CI\n  steps for "
              "months and reached the container in none of them.")
        return 1
    if steps == 0:
        print("PASS: no compose bring-up steps found — nothing to check.")
        return 0
    # Both directions in the summary. A PASS that names one of the two
    # checks it ran is the same defect this gate is about: a statement
    # whose scope is smaller than the work behind it.
    print(f"PASS: across {steps} bring-up step(s) — every variable the "
          f"profile needs is\n      supplied, and every variable the step "
          f"sets is one the profile reads.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
