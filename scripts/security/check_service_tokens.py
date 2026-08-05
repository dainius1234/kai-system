"""Every service that enforces service-auth must be given its token.

`common/service_auth.require_service_auth` is **fail closed**: a service
with no `KAI_SERVICE_TOKEN` returns 503 on every protected route rather
than serving them unauthenticated. That is the right default and it is
why G-03 is safe.

It also means a service that enforces auth and is never given a token is
**dead in that profile** — every protected call answers 503 — and the
symptom appears at the caller, which reads it as the callee being broken
rather than unconfigured.

Found on 2026-08-05 auditing G-07's closure. The record said the token
was "wired into 8 service blocks across all three compose profiles". It
is wired into 8 blocks *in total*, split 3/1/4, and two services that
enforce auth were missed:

  - `executor` in `full` and `sovereign` — `POST /execute`
    (`tool_execute`) and `POST /recover`. The executor is what actually
    runs tools, so both profiles ship a stack whose tool execution
    refuses every call.
  - `vault-sync` in `minimal`.

Nothing was watching for it, because the closure was a count of blocks
edited rather than a rule about which services need one.

The rule this enforces: **a service whose code calls
`Depends(require_service_auth(...))` must declare `KAI_SERVICE_TOKEN` in
its environment in every compose profile that runs it.**

Fails closed. An unreadable compose file, or a build context that names
a directory that is not there, is a finding rather than a service
quietly skipped — the shape that let `check_port_bindings` pass over
absent files before A-04.

Exit codes:
  0  every auth-enforcing service has its token in every profile
  1  at least one does not, or an input could not be read
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
COMPOSE_FILES = (
    "docker-compose.full.yml",
    "docker-compose.sovereign.yml",
    "docker-compose.minimal.yml",
)
TOKEN = "KAI_SERVICE_TOKEN"
MARKER = "Depends(require_service_auth("

_EXCLUDED = {"scripts", "tests", ".venv", "venv", "_archive", "node_modules",
             "__pycache__", ".git", "kai-pm", "site-packages"}


def _entrypoint(dockerfile: Path) -> Path | None:
    """The .py file a Dockerfile actually runs, resolved beside it.

    Granularity is the whole point. The first version asked "does any
    file under this service's top-level directory enforce auth", which
    produced **4 false findings out of 8**: `agentic-introspect` runs
    `introspect_app.py` (no protected routes) while `agentic/app.py` has
    them, and `avatar-service` / `tts-service` build from
    `output/avatar` and `output/tts` while the only file in `output/`
    with a protected route is `output/notify/app.py`.

    A gate that flags three innocent services teaches people to ignore
    it, so it resolves the entry point: `CMD ["python", "app.py"]` and
    `CMD ["uvicorn", "app:app", ...]` both name a module beside the
    Dockerfile.
    """
    try:
        lines = dockerfile.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    directive = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("CMD", "ENTRYPOINT")):
            directive = stripped
    if not directive:
        return None
    for token in directive.replace("[", " ").replace("]", " ") \
                          .replace(",", " ").replace('"', " ") \
                          .replace("'", " ").split():
        if token.endswith(".py"):
            candidate = dockerfile.parent / token
            return candidate if candidate.exists() else None
        if ":" in token and not token.startswith("-"):
            module = token.split(":")[0]
            candidate = dockerfile.parent / f"{module}.py"
            if candidate.exists():
                return candidate
    return None


def _dockerfile_of(cfg: dict) -> Path | None:
    """Where this service's Dockerfile lives, however the profile spells it.

        build: ./executor                                -> executor/Dockerfile
        build: {context: ./executor}                     -> executor/Dockerfile
        build: {context: ., dockerfile: x/y/Dockerfile}  -> x/y/Dockerfile

    The third is what `full.yml` uses; understanding only the first two
    made this check inspect 2 service definitions instead of 16 and
    report PASS for the profile holding the defect. Caught by its own
    denominator (I-2), which is the entire argument for printing one.
    """
    build = cfg.get("build")
    if isinstance(build, str):
        return REPO / build.strip("./") / "Dockerfile"
    if not isinstance(build, dict):
        return None
    dockerfile = str(build.get("dockerfile") or "").strip("./")
    if dockerfile:
        return REPO / dockerfile
    context = str(build.get("context") or "").strip("./")
    return (REPO / context / "Dockerfile") if context else None


class _Undecidable:
    """Third answer, spelled out rather than smuggled through `None`.

    "Could not tell" is not "no", and a bare `return None` beside a
    `.exists()` check is exactly the shape that lets the two collapse
    into each other — the meta-check's I-1 scanner flagged this function
    for it, correctly in spirit even though the caller did report the
    absence. A named third value cannot be misread by the next person,
    and cannot be silently treated as False by a later edit.
    """

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "UNDECIDABLE"


UNDECIDABLE = _Undecidable()


def enforces_auth(cfg: dict):
    """True, False, or UNDECIDABLE — never None-as-no.

    UNDECIDABLE is returned when the Dockerfile or its entry point
    cannot be resolved, and the caller reports it as a finding. Absence
    of evidence has been read as evidence of absence too many times in
    this repository for that to be left implicit.
    """
    dockerfile = _dockerfile_of(cfg)
    resolvable = dockerfile is not None and dockerfile.exists()
    if not resolvable:
        return UNDECIDABLE
    entry = _entrypoint(dockerfile)
    if entry is None:
        return UNDECIDABLE
    try:
        return MARKER in entry.read_text(encoding="utf-8")
    except OSError:
        return UNDECIDABLE


def audit() -> Tuple[List[str], int, int]:
    """Return (findings, services inspected, profiles inspected)."""
    import yaml

    findings: List[str] = []
    undecidable: List[str] = []
    inspected = 0
    profiles = 0

    for name in COMPOSE_FILES:
        path = REPO / name
        if not path.exists():
            # I-1: an absent input is a failure, not a pass. A profile
            # that vanished is exactly when this check is most needed.
            findings.append(f"{name}: missing — cannot verify tokens")
            continue
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:
            findings.append(f"{name}: unreadable ({exc})")
            continue
        profiles += 1
        for service, cfg in sorted((doc.get("services") or {}).items()):
            cfg = cfg or {}
            if cfg.get("build") is None:
                continue        # not built from this tree
            verdict = enforces_auth(cfg)
            if verdict is UNDECIDABLE:
                undecidable.append(f"{name}: '{service}'")
                continue
            if not verdict:
                continue
            inspected += 1
            env = cfg.get("environment") or {}
            names = env.keys() if isinstance(env, dict) else {
                str(e).split("=")[0] for e in env}
            if TOKEN not in names:
                entry = _entrypoint(_dockerfile_of(cfg))
                where = entry.relative_to(REPO) if entry else "?"
                findings.append(
                    f"{name}: service '{service}' runs {where}, which "
                    f"enforces require_service_auth, but declares no "
                    f"{TOKEN} — its protected routes answer 503 in this "
                    f"profile")
    if undecidable:
        # Reported, not swallowed: a service whose entry point cannot be
        # resolved is unknown, and unknown is not the same as safe.
        findings.append(
            f"entry point unresolvable for {len(undecidable)} service(s), "
            f"so their auth requirement is unknown: "
            f"{', '.join(sorted(undecidable)[:6])}"
            + (" ..." if len(undecidable) > 6 else ""))
    return findings, inspected, profiles


def main() -> int:
    findings, inspected, profiles = audit()
    print(f"  inspected: {inspected} auth-enforcing service definitions "
          f"(across {profiles} compose files)")
    print()
    if findings:
        print(f"FAIL: {len(findings)} auth-enforcing service(s) without a token:")
        print()
        for line in findings:
            print(f"  - {line}")
        print()
        print("  require_service_auth fails closed, so this is not an open")
        print("  endpoint — it is a service that answers 503 to every")
        print("  protected call. The symptom shows up at the caller, which")
        print("  reads it as the callee being broken rather than")
        print(f"  unconfigured. Add {TOKEN}: \"${{{TOKEN}:-}}\" to the service's")
        print("  environment; empty still means fail-closed.")
        return 1
    print(f"PASS: every auth-enforcing service declares {TOKEN} in every "
          f"profile that runs it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
