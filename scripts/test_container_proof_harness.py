#!/usr/bin/env python3
"""Prove the container-proof harness before spending the real Docker run.

`verify_identity_in_containers.sh` is the single experiment that resolves
the last UNKNOWN. It had been checked with `bash -n` (shell syntax) and
by compiling its embedded snippets. Neither of those exercises the
**control flow**, and neither can show that a non-execution is incapable
of reading as a pass — which is the property that actually matters, and
the one this repository keeps getting wrong.

So this drives the real script with a **stub `docker` on PATH**. The
stub does not fake the Python: it *executes* every `python -c` program
the script builds, so the shell/Python boundary — heredoc quoting,
`$PAYLOAD` interpolation, snippet insertion — is crossed for real. Only
the container and the network are simulated, and the network is
simulated by running Kai's **actual** `check_identity` over the request
the snippet produced.

The order the operator specified, and the order these run in:

    shell syntax -> embedded Python -> HARNESS CONTROL FLOW (here)
    -> real Docker execution (elsewhere, still UNKNOWN)

I-8: every scenario has a known-positive and a known-negative, and the
negative cases all assert the same thing from the other side — that the
word PROVEN never appears unless the work was actually done.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SCRIPT = REPO / "scripts/security/verify_identity_in_containers.sh"

PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ok    {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


# ── the stub receiver: real verification, simulated transport ───────────

RECEIVER = r'''
"""Injected via sitecustomize: urlopen answered by Kai's real auth code."""
import io, json, os, sys, urllib.request, urllib.error

sys.path.insert(0, os.environ["KAI_REPO"])


class _RawResp(io.BytesIO):
    def __init__(self, status, raw):
        super().__init__(raw)
        self.status = status
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _Resp(io.BytesIO):
    def __init__(self, status, payload):
        super().__init__(json.dumps(payload).encode())
        self.status = status
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _fake_urlopen(req, timeout=None):
    if os.environ.get("STUB_BAD_RESPONSE"):
        # A container that answers with something that is not the
        # contract. The harness must FAIL, not shrug.
        return _RawResp(200, b"<html>proxy error</html>")
    import common.service_auth as sa
    sa.reset_identity_context()
    headers = {k: v for k, v in req.header_items()}
    principal, status, detail = sa.check_identity(
        headers, "cortex_observe_turn",
        destination=os.environ["KAI_SERVICE_NAME"],
        method=req.get_method(),
        path=urllib.parse.urlparse(req.full_url).path,
        body=req.data or b"", require_grant=True)
    if os.environ.get("STUB_ACCEPT_ALL"):
        # A BROKEN RECEIVER that accepts everything. Every refusal case
        # must then fail the harness -- otherwise the harness would
        # report PASS against a system with no verification at all.
        return _Resp(200, {"turn_source": "agentic"})
    if principal is None:
        raise urllib.error.HTTPError(req.full_url, status, detail, {}, None)
    source = principal.identity if principal.usable_for_provenance else None
    return _Resp(200, {"bridge_active": False, "turn_source": source})


import urllib.parse  # noqa: E402
urllib.request.urlopen = _fake_urlopen
'''

STUB_DOCKER = r'''#!/usr/bin/env bash
# Stub `docker`. Simulates the daemon and the containers; the PYTHON IS
# REAL — every `python -c` program the harness builds is executed.
set -uo pipefail

case "$1" in
  info)
    [ "${STUB_DAEMON:-up}" = "up" ] && exit 0 || exit 1 ;;
  inspect)
    echo "${STUB_HEALTH:-healthy}"; exit 0 ;;
  compose) ;;
  *) exit 0 ;;
esac

# find the subcommand after the compose flags
sub=""
for arg in "$@"; do
  case "$arg" in
    build|run|up|restart|stop) sub="$arg"; break ;;
  esac
done

case "$sub" in
  build)   exit "${STUB_BUILD:-0}" ;;
  up|restart|stop) exit 0 ;;
  run)
    # everything after `-c` is the program
    prog=""
    prev=""
    for arg in "$@"; do
      [ "$prev" = "-c" ] && prog="$arg"
      prev="$arg"
    done
    [ -z "$prog" ] && exit 0
    # Run as an unprivileged user, like a container should. Under root a
    # 0444 file is still writable, so the read-only key map check would
    # prove nothing -- and the harness's own read-only assertion is one
    # of the things being tested here.
    if [ -n "${STUB_UNPRIV:-}" ]; then
      PYTHONPATH="$STUB_SITE:$KAI_REPO" setpriv --reuid=65534 --regid=65534 \
        --clear-groups python3 -c "$prog"
    else
      PYTHONPATH="$STUB_SITE:$KAI_REPO" python3 -c "$prog"
    fi
    exit $?
    ;;
esac
exit 0
'''


def _environment(tmp: Path, **overrides) -> dict:
    """A stub PATH plus the key material the snippets expect."""
    bindir = tmp / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    docker = bindir / "docker"
    docker.write_text(STUB_DOCKER, encoding="utf-8")
    docker.chmod(0o755)

    site = tmp / "site"
    site.mkdir(exist_ok=True)
    (site / "sitecustomize.py").write_text(RECEIVER, encoding="utf-8")

    # Real key material, generated fresh — the snippets sign for real.
    from scripts.security.generate_service_keys import generate
    import contextlib
    import io as _io
    keys = tmp / "keys"
    with contextlib.redirect_stdout(_io.StringIO()):
        generate(["agentic", "executor"], {"cortex_observe_turn": ["agentic"]},
                 keys)
    keymap = keys / "keymap.json"
    keymap.chmod(0o444)          # read-only, as the container mounts it

    # The stub runs the programs as nobody, so the key material must be
    # readable by that uid while staying unreadable to group/other --
    # which is what the harness asserts about the private key.
    for path in (keys, keys / "private", keys / "private" / "agentic.key",
                 keys / "private" / "executor.key", keymap, tmp):
        try:
            os.chown(path, 65534, 65534)
        except (PermissionError, FileNotFoundError):
            pass

    env = dict(os.environ)
    env.update({
        "PATH": f"{bindir}:{env['PATH']}",
        "KAI_REPO": str(REPO),
        "STUB_SITE": str(site),
        "KAI_SERVICE_KEYMAP": str(keymap),
        "KAI_SERVICE_PRIVATE_KEY": str(keys / "private" / "agentic.key"),
        "KAI_SERVICE_KEY_ID": "agentic-v1",
        "KAI_SERVICE_NAME": "cortex",
        "KAI_SERVICE_TOKEN": "a-valid-shared-token",
        "KAI_NONCE_CACHE_PATH": str(tmp / "nonces.json"),
        "HEALTH_TRIES": "2",
        "HEALTH_SLEEP": "0",
        "STUB_UNPRIV": "1",
    })
    env.update(overrides)
    return env


def run_harness(tmp: Path, script: Path = None, **overrides):
    """(returncode, output) from the real script under a stub docker."""
    result = subprocess.run(
        ["bash", str(script or SCRIPT)], cwd=str(REPO),
        env=_environment(tmp, **overrides),
        capture_output=True, text=True, timeout=300)
    return result.returncode, result.stdout + result.stderr


def main() -> int:  # noqa: C901 - a list of scenarios
    check("the harness exists", SCRIPT.is_file())
    syntax = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True)
    check("shell syntax is valid", syntax.returncode == 0,
          syntax.stderr.decode()[:200])

    # ── 1. THE HAPPY PATH: stub docker succeeds, every case must pass ──
    with tempfile.TemporaryDirectory() as d:
        code, out = run_harness(Path(d))
        check("a fully working stub drives the harness to PASS",
              code == 0, out[-1500:])
        check("and it says the status may be upgraded",
              "EXIT GATE: PASS" in out and "now PROVEN" in out)
        check("zero failures reported", " FAIL  " not in out, out[-1200:])

        # The embedded Python really ran, and the request really verified.
        check("the signed request was ACCEPTED as agentic — the shell/Python "
              "boundary was crossed for real",
              "a correctly signed, granted request is ACCEPTED, as agentic"
              in out and "ok    a correctly signed" in out)
        check("forged identity headers on a valid signature changed nothing",
              "ok    forged identity headers on a VALID signature" in out)
        check("the replay case ran and refused the second use",
              "ok    the same signed request succeeds ONCE" in out)
        happy_ok_count = out.count("  ok    ")
        check("every case reported, not a subset", happy_ok_count >= 15,
              f"{happy_ok_count} ok lines")

    # ── 2. NO DAEMON must refuse, and must never read as proven ──
    with tempfile.TemporaryDirectory() as d:
        code, out = run_harness(Path(d), STUB_DAEMON="down")
        check("no daemon exits 2 — neither pass nor fail", code == 2)
        check("it says the status remains UNKNOWN", "UNKNOWN" in out)
        check("NOT EXECUTED NEVER READS AS PROVEN",
              "PROVEN" not in out and "EXIT GATE: PASS" not in out, out[:400])

    # ── 3. a docker that is not installed at all ──
    with tempfile.TemporaryDirectory() as d:
        env = _environment(Path(d))
        (Path(d) / "bin" / "docker").unlink()
        result = subprocess.run(["bash", str(SCRIPT)], cwd=str(REPO), env=env,
                                capture_output=True, text=True, timeout=120)
        out = result.stdout + result.stderr
        check("an absent docker binary also exits 2", result.returncode == 2)
        check("and still claims nothing",
              "PROVEN" not in out and "EXIT GATE: PASS" not in out)

    # ── 4. a BUILD failure must fail, not pass ──
    with tempfile.TemporaryDirectory() as d:
        code, out = run_harness(Path(d), STUB_BUILD="1")
        check("a failing image build exits non-zero", code == 1)
        check("and names the build as the failure",
              "FAIL  cortex and agentic images build" in out)
        check("and refuses to upgrade the claim",
              "remains UNKNOWN" in out and "now PROVEN" not in out)

    # ── 5. an UNHEALTHY service must fail ──
    with tempfile.TemporaryDirectory() as d:
        code, out = run_harness(Path(d), STUB_HEALTH="unhealthy")
        check("a service that never becomes healthy fails", code == 1)
        check("and says which check failed",
              "FAIL  cortex reaches healthy" in out)
        check("and does not claim proof",
              "now PROVEN" not in out)

    # ── 6. MALFORMED embedded Python must fail, and be visible ──
    #
    # The failure mode this guards: a broken snippet produces no output,
    # the comparison sees an empty string, and it reads as a wrong status
    # rather than as a crash.
    with tempfile.TemporaryDirectory() as d:
        broken = Path(d) / "broken.sh"
        text = SCRIPT.read_text(encoding="utf-8")
        text = text.replace(
            "raw, headers = signed_json_request(destination='cortex', "
            "method='POST',\n                                   "
            "path='/observe_turn', payload=$PAYLOAD)\n\" \"200 agentic\"",
            "raw, headers = this_function_does_not_exist($PAYLOAD)\n"
            "\" \"200 agentic\"", 1)
        broken.write_text(text, encoding="utf-8")
        check("the broken copy differs from the original",
              broken.read_text() != SCRIPT.read_text())
        code, out = run_harness(Path(d), script=broken)
        check("a malformed embedded snippet FAILS the harness", code == 1,
              out[-800:])
        check("and the container output is printed, so a crash is not "
              "mistaken for a wrong status code",
              "NameError" in out or "Traceback" in out, out[-800:])

    # ── 7. AN INVALID RESPONSE must fail, not be shrugged off ──
    with tempfile.TemporaryDirectory() as d:
        code, out = run_harness(Path(d), STUB_BAD_RESPONSE="1")
        check("a container answering non-contract output FAILS the harness",
              code == 1, out[-600:])
        check("and does not claim proof", "now PROVEN" not in out)

    # ── 8. A BROKEN RECEIVER that accepts everything must fail ──
    #
    # The scenario that matters most: if verification silently stopped
    # working, the harness must NOT report success. Every refusal case
    # turns into an acceptance, and the harness has to notice.
    with tempfile.TemporaryDirectory() as d:
        code, out = run_harness(Path(d), STUB_ACCEPT_ALL="1")
        check("a receiver that verifies NOTHING fails the harness", code == 1)
        check("and the refusal cases are the ones that report it",
              "FAIL  an unsigned request with only the shared token is "
              "REFUSED" in out, out[-900:])
        check("and it refuses to upgrade the claim", "now PROVEN" not in out)

    # ── 9. the timeout override changes duration, never semantics ──
    text = SCRIPT.read_text(encoding="utf-8")
    check("HEALTH_TRIES defaults to the real value",
          'HEALTH_TRIES="${HEALTH_TRIES:-30}"' in text)
    check("HEALTH_SLEEP defaults to the real value",
          'HEALTH_SLEEP="${HEALTH_SLEEP:-2}"' in text)
    uses = [ln.strip() for ln in text.splitlines()
            if "HEALTH_TRIES" in ln or "HEALTH_SLEEP" in ln]
    # Two defaults, and two wait loops — the initial start and the one
    # after the restart — each contributing a `seq` and a `sleep`. The
    # first version of this assertion said four and was simply wrong
    # about the script, not about the property.
    check("the overrides appear ONLY in the defaults and the wait loops",
          len(uses) == 6
          and sum(("seq 1" in u or u.startswith("sleep ")) for u in uses) == 4,
          str(uses))
    check("no check, expectation or exit path reads them",
          not any("check " in u or "exit" in u or "PASS" in u for u in uses),
          str(uses))

    # ── 10. the exit code is a function of the failures, not of luck ──
    with tempfile.TemporaryDirectory() as d:
        code, out = run_harness(Path(d))
        tally = [ln for ln in out.splitlines() if "Container proof:" in ln]
        check("the harness prints a tally", len(tally) == 1, str(tally))
        check("a green tally accompanies exit 0",
              code == 0 and "0 failed" in tally[0], str(tally))
    with tempfile.TemporaryDirectory() as d:
        code, out = run_harness(Path(d), STUB_BUILD="1")
        tally = [ln for ln in out.splitlines() if "Container proof:" in ln]
        check("a non-zero failure count accompanies exit 1",
              code == 1 and "0 failed" not in tally[0], str(tally))

    print("=" * 66)
    print(f"Container proof harness tests: {PASSED} passed, {FAILED} failed")
    print(f"EXIT GATE: {'PASS' if FAILED == 0 else 'FAIL'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    print("Container-proof harness — driven by a stub docker")
    print("=" * 66)
    sys.exit(main())
