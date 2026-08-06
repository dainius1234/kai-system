#!/usr/bin/env python3
"""Print the logs of the containers that are actually wrong, and only those.

Why this exists
---------------

The GitHub Actions log API serves a **fixed-size tail** — measured, not
assumed: asking for 130 lines and asking for 255 returned the same
15,780 characters. That is why the post-mortem runs last and why every
live step tees to a file it prints.

On 2026-08-06 the full profile failed with

    Container kai-system-memu-core-1  Error
    dependency failed to start: container kai-system-memu-core-1 is unhealthy

and the container's own log — the only thing that says *why* — was
written by the dump step, which sits far enough back to be outside that
window. So the remedy the post-mortem exists to apply had been applied
to the bring-up logs and not to the container logs. The third time that
shape has appeared today: a fix applied to some of its subject.

`docker compose logs --tail 40` across twenty services is eight hundred
lines, which does not fit in the window either, and burying the answer
in noise is the same failure as truncating it. So this prints the
containers that are not healthy, derived from `docker compose ps` rather
than named in a list — because which container broke is not knowable in
advance, which is precisely what a hand-written list assumes.

A diagnostic never fails the build
----------------------------------

This exits 0 whatever happens. It runs after something has already gone
wrong, and replacing the real failure with a diagnostic one destroys the
information the step exists to produce. That decision is declared in
`check_ci_tolerations`, not left implicit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.ci.compose_probe import compose_health, run  # noqa: E402

#: States that mean "this container is fine". Everything else is worth
#: a log — including `no-healthcheck`, because a container nobody is
#: checking is a container nobody is checking.
_OK = {"healthy", "running"}


def suspects(compose_file: str) -> List[str]:
    """Services whose state is not plainly healthy, from Docker's verdict."""
    try:
        health = compose_health(compose_file)
    except Exception as exc:                # noqa: BLE001 — see module docstring
        print(f"  (could not read container state: {exc})")
        return []
    return sorted(name for name, state in health.items()
                  if not any(state.startswith(ok) for ok in _OK))


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compose-file", required=True)
    parser.add_argument("--tail", type=int, default=60)
    args = parser.parse_args(argv)

    names = suspects(args.compose_file)
    print(f"  inspected: container state for {args.compose_file}; "
          f"{len(names)} service(s) not plainly healthy")

    if not names:
        # Not silence: a failure with every container healthy is itself
        # a fact, and a useful one — it says the failure was not a
        # container dying.
        print("  Every container reports healthy or running. Whatever "
              "failed, it was not\n  a container falling over — look at the "
              "step's own output instead.")
        return 0

    for name in names:
        print(f"\n──────── {name} (last {args.tail} lines) ────────")
        code, out, err = run(["docker", "compose", "-f", args.compose_file,
                              "logs", "--no-color", "--tail", str(args.tail),
                              name])
        body = (out or "").strip() or (err or "").strip()
        print(body if body else "  (the container produced no output at all)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
