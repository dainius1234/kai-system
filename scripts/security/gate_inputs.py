"""Fail closed on a missing input, and say how much was inspected.

Two of the four instrumentation invariants (I-1, I-2), in one place, so
that adopting them is an import rather than twelve independent edits.

**I-1 — boundary blindness.** The operator's name for the defect this
prevents: *a check that cannot distinguish "the system is correct" from
"the system is absent".* Eleven of twelve checks had the shape

    for name in COMPOSE_FILES:
        path = repo_root / name
        if not path.exists():
            continue                 # ← a renamed file is not a violation

which makes a rename indistinguishable from a clean bill of health.
Proven on `check_port_bindings`: pointed at filenames that do not exist
it printed `PASS: No disallowed port bindings found.` and exited 0 —
byte-identical to a real pass.

Their diagnosis is the precise one, and it generalises past file paths:
the script answers *"of the things I looked at, were any wrong?"* while
claiming to answer *"are the things correct?"*. "I looked at nothing" is
a valid answer to the first and a silent failure on the second.

**I-2 — the denominator.** A `PASS` that does not say what it examined is
unfalsifiable; it reads identically whether it inspected fifty services
or zero. `inspected()` is the one-line form of what
`check_architecture_rules` already prints (`15/15 — 12 enforced, 3
declared uncheckable`), which exists because that gate once silently
omitted 6 of its 15 rules.

Usage::

    from scripts.security.gate_inputs import require, inspected

    def main() -> int:
        files = require(COMPOSE_FILES)        # exits 1 if any is missing
        ...
        print(inspected(len(files), "compose files"))

On optionality: `require()` takes an explicit `optional=` list rather
than treating absence as acceptable by default. As measured when this was
written, **no check has any optional input** — all three compose files
are git-tracked, so "absent in some checkout" cannot happen. The
parameter exists because a future input might genuinely be optional, and
per-path is the only form that does not let one real exception
generalise into "absence is fine everywhere".
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, List, Sequence

REPO = Path(__file__).resolve().parent.parent.parent


class MissingInputs(Exception):
    """Raised instead of exiting, so callers and tests can both use this."""

    def __init__(self, missing: Sequence[str]):
        self.missing = list(missing)
        super().__init__(", ".join(self.missing))


def resolve(paths: Iterable[str], optional: Iterable[str] = (),
            root: Path = REPO) -> List[Path]:
    """Return the paths that exist, raising if a required one does not."""
    optional = set(optional)
    found: List[Path] = []
    missing: List[str] = []
    for rel in paths:
        path = root / rel
        if path.exists():
            found.append(path)
        elif rel not in optional:
            missing.append(rel)
    if missing:
        raise MissingInputs(missing)
    return found


def require(paths: Iterable[str], optional: Iterable[str] = (),
            root: Path = REPO) -> List[Path]:
    """`resolve`, but exits 1 with an explanation — the CLI entry form.

    Refusing is the whole point. A check that cannot find what it audits
    has not audited it, and certifying it anyway is the defect.
    """
    try:
        return resolve(paths, optional, root)
    except MissingInputs as exc:
        print("REFUSED: this check cannot find what it audits.\n")
        for rel in exc.missing:
            print(f"  - missing: {rel}")
        print("\nA missing input is not a passing check. Either the file "
              "moved —\nin which case this check has been inspecting "
              "nothing — or it was\nremoved deliberately, in which case "
              "say so by declaring it optional.")
        raise SystemExit(1)


def count_services(paths: Iterable[Path]) -> int:
    """Total service definitions across the given compose files.

    A denominator of "3 compose files" is nearly useless — it is 3
    whatever happens. The number that moves, and that reveals a scanner
    which has gone blind, is how many services were actually examined.
    """
    import yaml
    total = 0
    for path in paths:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        total += len(data.get("services") or {})
    return total


def compose_files(root: Path = REPO) -> List[Path]:
    """Every root compose profile, from a glob rather than a list.

    Lived in three files independently — `check_compose_interpolation`,
    `report_execution_coverage` and, as a hand-written pair of names, in
    `test_docker_e2e`. Three copies of a denominator is three chances for
    one of them to fall behind a fourth profile, which is the
    list-beside-the-thing pattern applied to the lists themselves.
    """
    return sorted(p for p in root.glob("docker-compose*.y*ml") if p.is_file())


def built_services(root: Path = REPO) -> "dict":
    """`service name -> Dockerfile path` for every image this repo builds.

    The denominator for any question of the form *"do all our services
    do X?"*. Derived from the `build:` stanzas, so a service added to a
    profile is in scope the moment it is added — and a service using
    somebody else's image (`redis:7-alpine`) is not, because that is not
    our code to hold to our rules.

    A service defined in more than one profile maps to the Dockerfile of
    whichever profile names it; they agree today, and `check_image_modules`
    is the check that would notice if they stopped.
    """
    import yaml
    out = {}
    for path in compose_files(root):
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        for name, cfg in (doc.get("services") or {}).items():
            build = (cfg or {}).get("build")
            if not build:
                continue
            if isinstance(build, dict):
                rel = build.get("dockerfile") or f"{build.get('context','.')}/Dockerfile"
            else:
                rel = f"{build}/Dockerfile"
            out.setdefault(name, root / rel)
    return out


def inspected(count: int, unit: str, extra: str = "") -> str:
    """The denominator line. Say what was examined, every time.

    A count of zero is reported as such rather than smoothed over: it is
    the single most useful thing to see when a check has gone blind.
    """
    line = f"  inspected: {count} {unit}"
    if extra:
        line += f" ({extra})"
    if count == 0:
        line += "\n  WARNING: zero inputs inspected — this is not a pass."
    return line
