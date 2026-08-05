"""The hot path must survive the cold-maintenance process being killed.

`memu-core-introspect` is the store-maintenance process split off from
`memu-core` (DECISIONS.md D21). If stopping it takes the hot path down
with it, the split bought nothing. This asserts it does not.

Written as a file rather than a heredoc inside the workflow. An embedded
`python3 -c "` heredoc has already terminated a `run: |` block early in
three workflows in this repository — `check_ci_tolerations.unparseable()`
exists because of it — and a workflow that does not parse runs nothing,
which is indistinguishable from having no failures.

Every call goes through `docker compose exec`, because `memu-core` is on
`agent-net` and `data-net`, both declared `internal: true`. The previous
version was three `curl http://localhost:8001|8009` probes against host
ports that have not existed since `e4655bc`.

Exit 0 = the hot path stayed up and writable.
Exit 1 = it did not, or the probe could not run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.ci.compose_probe import exec_http, load_ports  # noqa: E402

HOT = "memu-core"
COLD = "memu-core-introspect"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose-file", default="docker-compose.minimal.yml")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        ports = load_ports(args.compose_file)
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1
    if HOT not in ports:
        # I-1. No address means the check cannot run, and a check that
        # cannot run has not passed.
        print(f"FAIL: {HOT} declares no health port in {args.compose_file}, "
              f"so there is nothing to probe.")
        return 1

    port = ports[HOT]
    print(f"  probing: {HOT} on port {port} (from its healthcheck), "
          f"inside {args.compose_file}")

    ok, detail, _ = exec_http(args.compose_file, HOT, port, "GET", "/health")
    if not ok:
        print(f"FAIL: {HOT} unhealthy after {COLD} was stopped: {detail}")
        return 1
    print(f"     health ok: {detail}")

    ok, detail, body = exec_http(
        args.compose_file, HOT, port, "POST", "/memory/memorize",
        {"timestamp": "2026-01-01T00:00:00Z",
         "event_type": "ci-kill-isolation",
         "result_raw": "kill isolation smoke test",
         "user_id": "ci"})
    if not ok:
        print(f"FAIL: hot memorize write failed while {COLD} was down: {detail}")
        return 1
    print(f"     memorize ok: {detail} {body.strip()[:200]}")

    print(f"PASS: {HOT} stayed healthy and writable with {COLD} stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
