"""CLI go/no-go check: poll the dashboard's go-no-go endpoint.

Exits non-zero on a non-GO decision — and, unless told otherwise, on
*not being able to ask*.

The absence rule
----------------

This opened with:

    except Exception:
        # Non-running environment fallback; static success for CI stage.
        print("go_no_go: dashboard not running; static checks only")
        raise SystemExit(0)

which is I-1 exactly: the script could not distinguish **the decision is
GO** from **there was nothing to ask**, and gave the same exit code for
both. `make go_no_go` gates CI, so on every run where no dashboard was
listening — which is every run — this printed a line and passed.

Found on 2026-08-06 by widening the instrumentation registry's
denominator past `scripts/security/`. Eight build-failing instruments
lived outside it; this was the first one looked at.

The fix is not to make absence fatal everywhere: the compile stage
genuinely has no dashboard and genuinely should still run. It is to make
absence a **declared** choice at the call site rather than a silent one
inside the script. `--allow-absent` says *I know nothing is listening
and I am asking anyway*. Without it, an unreachable dashboard is a
failure — so a live invocation that quietly stops reaching the service
says so instead of going green.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Sequence

URL = "http://localhost:8080/go-no-go"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=URL,
                        help="the go-no-go endpoint to poll")
    parser.add_argument(
        "--allow-absent", action="store_true",
        help="treat an unreachable dashboard as a pass. For stages with "
             "no dashboard by design — the caller declares that, rather "
             "than this script assuming it on everybody's behalf.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        with urllib.request.urlopen(args.url, timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        if args.allow_absent:
            print(f"go_no_go: SKIPPED — {args.url} unreachable ({exc}); the "
                  f"caller passed --allow-absent, so this stage has no "
                  f"dashboard by design. No decision was obtained.")
            return 0
        print(f"go_no_go: FAIL — could not reach {args.url} ({exc}). An "
              f"unanswered question is not a GO. Pass --allow-absent if "
              f"this stage is not meant to have a dashboard.")
        return 1

    decision = payload.get("decision")
    if decision != "GO":
        print(f"go_no_go: FAIL — decision={decision!r}: {payload}")
        return 1

    print(f"go_no_go: PASS — decision=GO, from {len(payload)} field(s) in "
          f"the response")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
