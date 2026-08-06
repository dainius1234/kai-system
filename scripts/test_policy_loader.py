"""Tests for `common/policy.py` — the file nothing had ever read.

`security/policy.yml` calls itself, in its own header, *"the single
source of truth — every runtime decision (tool-gate gating, verifier
verdicts, circuit breaker thresholds, quarantine triggers) reads from
this file."* On 2026-08-06 the sovereign profile started tool-gate for
the first time and printed:

    JSONDecodeError: Expecting value: line 14 column 1 (char 13)
    POLICY FILE CORRUPT OR UNREADABLE — failing closed.

The file was not corrupt. pyyaml was not installed, so a fallback
loader ran `json.loads` over a YAML document — which could only ever
return `{}`. Thirty-five service images ship `common/` and none declared
pyyaml, so this had been true in every container since the loader was
written.

The design question this file settles
-------------------------------------

`check_shipped_package_deps` cannot catch this. The import was guarded,
and a static rule broad enough to flag it also flags `torch` against
`weather-service`, whose guarded import has a perfectly correct fallback
(a CUDA probe returning False). What distinguished pyyaml is that its
*fallback could not work* — a property of one bug, not a rule.

DeepSeek's answer, asked for a second opinion: a gate cannot decide
whether a guarded fallback works, but the **service can decide that its
own output is nonsense**. Undecidable statically, decidable at runtime.

So the loader now refuses to start on a policy file that exists, has
bytes, and parses to nothing — and these tests exercise that path with
pyyaml genuinely unimportable, against the real `security/policy.yml`.
Knowing what a *correct* parse looks like is the hard problem; knowing
that an *empty* one is wrong is not.
"""
from __future__ import annotations

import builtins
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPO = Path(__file__).resolve().parents[1]
POLICY_FILE = REPO / "security" / "policy.yml"

passed = 0
failed = 0
EXPECTED_SCENARIOS = 6
executed: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def scenario(name: str) -> None:
    executed.append(name)


class _NoYaml:
    """Make `import yaml` fail, exactly as it does in an image without it."""

    def __enter__(self):
        self._real = builtins.__import__
        self._saved = sys.modules.pop("yaml", None)

        def fake(name, *args, **kwargs):
            if name == "yaml" or name.startswith("yaml."):
                raise ImportError("No module named 'yaml'")
            return self._real(name, *args, **kwargs)

        builtins.__import__ = fake
        return self

    def __exit__(self, *exc):
        builtins.__import__ = self._real
        if self._saved is not None:
            sys.modules["yaml"] = self._saved


def _reload(env: dict = None):
    """Re-import common.policy under a given environment."""
    saved = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    sys.modules.pop("common.policy", None)
    try:
        return importlib.import_module("common.policy"), None
    except Exception as exc:                      # noqa: BLE001 — the subject
        return None, exc
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── the real defect ──────────────────────────────────────────────────

def test_the_real_policy_file_loads_when_pyyaml_is_present() -> None:
    """The baseline. Without this, every assertion below could be
    passing because the file is unreadable for some other reason."""
    scenario("loads with pyyaml")
    module, exc = _reload()
    check("it imported", exc is None, str(exc))
    check("and the policy is not empty", module and bool(module.POLICY),
          str(module and len(module.POLICY)))
    check("with the sections services actually read",
          module and {"verifier", "evidence", "circuit_breakers"} <=
          set(module.POLICY), str(module and sorted(module.POLICY)))


def test_without_pyyaml_it_refuses_to_start() -> None:
    """The fix. Fed the *real* policy.yml with pyyaml unimportable —
    exactly the state all 35 images were in."""
    scenario("refuses without pyyaml")
    with _NoYaml():
        module, exc = _reload()
    check("it raised rather than loading an empty policy",
          module is None and exc is not None, f"{module}, {exc}")
    check("and the message names the dependency, not the file",
          exc and "pyyaml" in str(exc), str(exc)[:200])
    check("and says it is refusing to start",
          exc and "Refusing to start" in str(exc), str(exc)[:200])


def test_the_old_behaviour_is_reachable_only_by_declaring_it() -> None:
    """A deployment that genuinely wants hardcoded defaults can still
    have them — but it has to say so, and it is told what it gave up.
    Same shape as `go_no_go_check --allow-absent`: absence becomes a
    choice at the call site instead of an assumption in the library."""
    scenario("escape hatch is explicit")
    with _NoYaml():
        module, exc = _reload({"KAI_POLICY_ALLOW_EMPTY": "1"})
    check("it imports", exc is None, str(exc))
    check("with an empty policy", module is not None and module.POLICY == {},
          str(module and module.POLICY))


def test_accessors_still_return_safe_defaults_on_an_empty_policy() -> None:
    """Refusing to start is the new behaviour, not the only protection.
    If a deployment opts out, the hardcoded defaults must still be the
    restrictive ones — fail-closed was always the right direction."""
    scenario("defaults remain restrictive")
    with _NoYaml():
        module, _ = _reload({"KAI_POLICY_ALLOW_EMPTY": "1"})
    weights = module.evidence_weights()
    check("evidence weights fall back to a full set",
          abs(sum(weights.values()) - 1.0) < 0.01, str(weights))
    breaker = module.circuit_breaker_defaults()
    check("and the breaker has a real threshold",
          breaker.get("failure_threshold", 0) > 0, str(breaker))


def test_an_absent_policy_file_is_a_different_case() -> None:
    """A file that is not there is a deployment choice. A file that is
    there and yields nothing is an impossible output. Only the second
    refuses."""
    scenario("absent file allowed")
    module, exc = _reload({"SOVEREIGN_POLICY_PATH": "/nonexistent/policy.yml"})
    check("it imports", exc is None, str(exc))
    check("with an empty policy", module is not None and module.POLICY == {},
          str(module and module.POLICY))


def test_the_policy_hash_changes_with_the_file() -> None:
    """The hash is shown on the dashboard and logged at startup. A
    constant hash would make a swapped policy invisible."""
    scenario("hash tracks the file")
    module, _ = _reload()
    first = module.policy_hash
    module2, _ = _reload({"SOVEREIGN_POLICY_PATH": "/nonexistent/policy.yml"})
    check("a different file gives a different hash",
          module2.policy_hash != first, f"{first} vs {module2.policy_hash}")
    check("and the real one is not the empty-input hash",
          len(first) == 16 and first != "e3b0c44298fc1c14", first)


def run_all() -> None:
    test_the_real_policy_file_loads_when_pyyaml_is_present()
    test_without_pyyaml_it_refuses_to_start()
    test_the_old_behaviour_is_reachable_only_by_declaring_it()
    test_accessors_still_return_safe_defaults_on_an_empty_policy()
    test_an_absent_policy_file_is_a_different_case()
    test_the_policy_hash_changes_with_the_file()
    _reload()          # leave the module in its normal state

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Policy Loader Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
