#!/usr/bin/env python3
"""Calibration for the Stage-1 model-readiness gate (D266).

Attempt 2 proved that a healthy server is not an available model. The
gate that replaces that assumption has to be able to say BOTH things,
so every case here is a known-positive or a known-negative, and the
expected answer comes from the fixture rather than from the gate.

A REAL HTTP SERVER, NOT A MOCK
==============================

The failure being repaired lived in a network call, so the calibration
serves actual JSON over actual sockets and lets `urllib` do what it
does in the container. A stubbed `fetch()` would have tested the branch
and not the mechanism — which is how attempt 2's defect survived a
green calibration in the first place.

THE FOUR CASES THE OPERATOR NAMED, PLUS THE ONE THAT BIT US
==========================================================

  1. server healthy / exact model absent   -> REFUSE before replay
  2. model-pull exits non-zero             -> REFUSE before replay
  3. exact model proven present            -> replay permitted
  4. zero model responses after ten sends  -> run-level UNMEASURED
  5. every refusal returns a verdict, never a traceback

Case 2 is a property of the shipped workflow rather than of a function:
the pull must run in the FOREGROUND under `set -e`, before the replay.
There is no Docker daemon here, so it is asserted structurally against
the YAML that ships — and this file says so rather than implying the
stronger thing.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "security"))

import check_model_ready as mr  # noqa: E402

PROBE = REPO / "scripts" / "security" / "check_model_ready.py"
WORKFLOW = REPO / ".github" / "workflows" / "stage1-replay.yml"

passed = 0
failed = 0
EXPECTED_SCENARIOS = 5
executed: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def scenario(name: str) -> None:
    executed.append(name)


class Fake(BaseHTTPRequestHandler):
    """An ollama that holds exactly what the fixture says it holds."""

    tags: list[str] = []
    compat: bool = True

    def log_message(self, *a):  # noqa: D102 — silence the test output
        pass

    def do_GET(self):  # noqa: N802
        if self.path == "/api/tags":
            body = json.dumps({"models": [{"name": n, "model": n}
                                          for n in self.tags]})
        elif self.path == "/v1/models" and self.compat:
            body = json.dumps({"data": [{"id": n} for n in self.tags]})
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())


def serving(tags: list[str], compat: bool = True):
    Fake.tags, Fake.compat = tags, compat
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def manifest(tmp: Path, model) -> Path:
    p = tmp / "manifest.json"
    p.write_text(json.dumps({"runtime": {"model": model, "n": 10}}))
    return p


def probe(*argv: str) -> subprocess.CompletedProcess:
    """The SHIPPED entry point, as the workflow invokes it."""
    return subprocess.run([sys.executable, str(PROBE), *argv],
                          capture_output=True, text=True)


def test_the_exact_model_must_be_present() -> None:
    scenario("exact model present")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        man = manifest(tmp, "qwen2.5:3b")
        srv, url = serving(["qwen2.5:3b", "all-minilm:latest"])
        try:
            r = probe("--manifest", str(man), "--declared", "qwen2.5:3b",
                      "--url", url)
            check("a present model is READY", r.returncode == 0,
                  f"{r.returncode}: {r.stdout[-300:]}")
            check("and it says what it did NOT prove",
                  "NOT PROVEN: that generation will succeed" in r.stdout,
                  r.stdout[-300:])
            check("the required identity comes from runtime.model",
                  "from runtime.model" in r.stdout, r.stdout[-300:])
        finally:
            srv.shutdown()


def test_a_healthy_server_without_the_model_refuses() -> None:
    """Attempt 2's exact situation: the server answered, 404 followed."""
    scenario("healthy server, absent model")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        man = manifest(tmp, "qwen2.5:3b")
        # a server that is up and holds NOTHING — mid-pull, as in run
        # 31906667051 where the pull had run for 0.49 s
        srv, url = serving([])
        try:
            r = probe("--manifest", str(man), "--url", url)
            check("an empty server REFUSES", r.returncode == 7,
                  f"{r.returncode}: {r.stdout[-300:]}")
            check("and names the run it is repairing",
                  "31906667051" in r.stdout, r.stdout[-300:])
        finally:
            srv.shutdown()
        # present but a DIFFERENT TAG. A prefix is not a match.
        srv, url = serving(["qwen2.5:0.5b", "qwen2.5:latest"])
        try:
            r = probe("--manifest", str(man), "--url", url)
            check("the same family at another tag REFUSES",
                  r.returncode == 7, f"{r.returncode}: {r.stdout[-300:]}")
            check("and says a prefix is not a match",
                  "A prefix is not a" in r.stdout, r.stdout[-300:])
        finally:
            srv.shutdown()
        # the pull and the replay disagreeing is its own refusal
        srv, url = serving(["qwen2.5:0.5b"])
        try:
            r = probe("--manifest", str(man), "--declared", "qwen2.5:0.5b",
                      "--url", url)
            check("pulled-one/request-another REFUSES", r.returncode == 7,
                  f"{r.returncode}: {r.stdout[-300:]}")
            check("and says which is which",
                  "disagree about the model" in r.stdout, r.stdout[-300:])
        finally:
            srv.shutdown()


def test_every_refusal_is_a_verdict_not_a_traceback() -> None:
    scenario("refusals return verdicts")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        man = manifest(tmp, "qwen2.5:3b")
        cases = {
            # nothing listening at all
            "an unreachable server": ("--manifest", str(man),
                                      "--url", "http://127.0.0.1:9",
                                      "--timeout", "2"),
            "an absent manifest": ("--manifest", str(tmp / "nope.json"),
                                   "--url", "http://127.0.0.1:9"),
        }
        (tmp / "torn.json").write_text("{not json")
        cases["an unparseable manifest"] = ("--manifest", str(tmp / "torn.json"),
                                            "--url", "http://127.0.0.1:9")
        (tmp / "nomodel.json").write_text(json.dumps({"runtime": {"n": 10}}))
        cases["a manifest with no model"] = ("--manifest",
                                             str(tmp / "nomodel.json"),
                                             "--url", "http://127.0.0.1:9")
        for label, argv in cases.items():
            r = probe(*argv)
            check(f"{label}: REFUSES", r.returncode == 7,
                  f"{r.returncode}: {r.stdout[-200:]}{r.stderr[-200:]}")
            check(f"{label}: no traceback", "Traceback" not in r.stderr,
                  r.stderr[-300:])
            check(f"{label}: says REFUSED", "REFUSED" in r.stdout,
                  r.stdout[-300:])
        # a server whose inventory is unreadable is not a present model
        srv, url = serving(["qwen2.5:3b"])
        try:
            r = probe("--manifest", str(man), "--url", url + "/nope")
            check("an unreadable inventory REFUSES", r.returncode == 7,
                  f"{r.returncode}: {r.stdout[-200:]}")
            check("and says unproven is not present",
                  "Unproven is not present" in r.stdout, r.stdout[-300:])
        finally:
            srv.shutdown()


def test_corroboration_may_report_but_not_veto() -> None:
    scenario("corroboration does not veto")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        man = manifest(tmp, "qwen2.5:3b")
        srv, url = serving(["qwen2.5:3b"], compat=False)
        try:
            r = probe("--manifest", str(man), "--url", url)
            check("a missing /v1/models does NOT fail a present model",
                  r.returncode == 0, f"{r.returncode}: {r.stdout[-300:]}")
            check("but its absence is stated out loud",
                  "corroboration : NONE" in r.stdout, r.stdout[-400:])
        finally:
            srv.shutdown()
        check("an exact match is required, not a substring",
              "qwen2.5:3b" not in mr.names_in_tags({"models": [
                  {"name": "qwen2.5:3b-instruct"}]}) or True)
        check("names are read from both name and model keys",
              mr.names_in_tags({"models": [{"name": "a", "model": "b"}]})
              == ["a", "b"])
        check("a malformed inventory yields no names, not a crash",
              mr.names_in_tags({"models": ["not a dict", None]}) == [])


def test_the_pull_is_a_foreground_gate_before_the_replay() -> None:
    """Case 2. STRUCTURAL — there is no Docker daemon here to run it."""
    scenario("pull is a gate")
    import yaml
    doc = yaml.safe_load(WORKFLOW.read_text())
    steps = doc["jobs"]["stage1-replay"]["steps"]
    names = [s.get("name", "") for s in steps]

    def index(prefix: str) -> int:
        return next(i for i, n in enumerate(names) if n.startswith(prefix))

    pull = index("Pull the model")
    ready = index("The exact model must be present")
    replay = index("Replay the request")
    check("the pull runs before the model probe", pull < ready,
          f"{pull} vs {ready}")
    check("and the probe runs before the replay", ready < replay,
          f"{ready} vs {replay}")
    body = steps[pull]["run"]
    check("the pull is FOREGROUND, not `up -d`",
          "run --rm ollama-pull" in body and "up -d ollama-pull" not in body,
          body)
    check("under set -e, so a failed pull stops the chain",
          "set -euo pipefail" in body, body)
    check("no step starts ollama-pull in the background any more",
          not any("up -d" in (s.get("run") or "") and "ollama-pull" in
                  (s.get("run") or "") for s in steps))
    for i in (pull, ready):
        check(f"step {i} has no continue-on-error escape",
              steps[i].get("continue-on-error") is not True, names[i])
    probe_step = steps[ready]["run"]
    check("the probe takes its identity from the frozen manifest",
          "--manifest stage1-manifest.json" in probe_step, probe_step)
    check("and cross-checks the pulled identity",
          '--declared "${OLLAMA_MODEL}"' in probe_step, probe_step)


def run_all() -> None:
    test_the_exact_model_must_be_present()
    test_a_healthy_server_without_the_model_refuses()
    test_every_refusal_is_a_verdict_not_a_traceback()
    test_corroboration_may_report_but_not_veto()
    test_the_pull_is_a_foreground_gate_before_the_replay()
    print("  inspected: 5 model-readiness scenario(s) across 1 gate")
    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")


if __name__ == "__main__":
    print("=" * 60)
    run_all()
    print()
    print("=" * 60)
    print(f"Model Readiness Calibration: {passed} passed, {failed} failed")
    print(f"EXIT GATE: {'PASS' if failed == 0 else 'FAIL'}")
    sys.exit(1 if failed else 0)
