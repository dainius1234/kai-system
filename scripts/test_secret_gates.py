"""Secret and restart gate tests — two of the eight that could never fail.

Both gates here were in the `KAI-GATE-003` backlog: never observed
failing, so possibly vacuous. Reading their semantics first — rather than
retrofitting a denominator onto them — found a real defect in each.

`check_secret_fallbacks` matched a **denylist of nine guessable words**,
so `${DB_PASSWORD:-hunter2}` and a hardcoded `BINANCE_API_SECRET` both
passed. The danger is not that a default is weak; it is that a default
exists. Its docstring also advertised a hardcoded-secret scan that had no
implementing pattern.

`check_restart_recovery` declared `ALLOWED_RESTART` and never referenced
it, denying exactly one string instead, so `restart: nonsense-value`
passed.

Two cases here guard against false positives rather than misses, because
a gate that flags working configuration gets someone to "fix" working
configuration:

  - `HUGGINGFACE_TOKENIZER` contains "TOKEN" and is a model name.
  - `/run/secrets/hmac_secret` is a path to where a secret lives.

And one guards the case that motivated the rewrite: the single dangerous
default in this repository hid under a **non-secret key**,
`GATE_SESSION_ID: "${CAMERA_GATE_TOKEN:-camera-gate-token-1}"`. A rule
that inspected only the key would have missed it.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_restart_recovery as restart  # noqa: E402
from scripts.security import check_secret_fallbacks as secrets  # noqa: E402
from scripts.security.gate_inputs import MissingInputs, resolve  # noqa: E402

passed = 0
failed = 0

EXPECTED_SCENARIOS = 19
executed: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        msg = f"  FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def scenario(name: str) -> None:
    executed.append(name)


def compose(env_lines: str, service: str = "svc") -> Path:
    """A synthetic compose file with one service and a given environment."""
    tmp = Path(tempfile.mkdtemp()) / "c.yml"
    tmp.write_text(
        "services:\n"
        f"  {service}:\n"
        "    image: x\n"
        "    environment:\n" + env_lines
    )
    return tmp


def secret_violations(env_lines: str, service: str = "svc") -> list:
    v, _, _ = secrets.check_file(compose(env_lines, service))
    return v


# ── The rule: referenced or empty, never valued ──────────────────────

def test_a_bare_reference_is_allowed():
    scenario("bare-ref")
    check("${TOKEN} passes", not secret_violations('      API_TOKEN: "${API_TOKEN}"\n'))


def test_an_explicitly_empty_default_is_allowed():
    scenario("empty-default")
    check("${TOKEN:-} passes — it fails closed",
          not secret_violations('      API_TOKEN: "${API_TOKEN:-}"\n'))


def test_a_weak_default_fails():
    scenario("weak-default")
    v = secret_violations('      DB_PASSWORD: "${DB_PASSWORD:-localdev}"\n')
    check("the old denylist case still fails", v, str(v))


def test_a_strong_looking_default_fails():
    """The whole reason for the rewrite: the old gate passed this."""
    scenario("strong-default")
    v = secret_violations('      DB_PASSWORD: "${DB_PASSWORD:-hunter2}"\n')
    check("a default that is not a guessable word still fails", v, str(v))
    v2 = secret_violations('      JWT_SECRET: "${JWT_SECRET:-a8f3c9d1e7b2}"\n')
    check("a random-looking default fails too", v2, str(v2))


def test_a_hardcoded_secret_fails():
    """The docstring claim that had no implementing pattern."""
    scenario("hardcoded")
    v = secret_violations('      BINANCE_API_SECRET: "sk_live_abc123def456"\n')
    check("a literal secret fails", v, str(v))
    check("the reason names the exposure",
          v and "anyone can read" in v[0], str(v))


def test_a_secret_hiding_under_a_non_secret_key_fails():
    """`GATE_SESSION_ID` is not secret-shaped; `CAMERA_GATE_TOKEN` is."""
    scenario("hidden-key")
    v = secret_violations(
        '      GATE_SESSION_ID: "${CAMERA_GATE_TOKEN:-camera-gate-token-1}"\n')
    check("the interpolated variable is inspected, not only the key",
          v, str(v))
    check("the variable is named", v and "CAMERA_GATE_TOKEN" in v[0], str(v))


# ── False positives are defects too ──────────────────────────────────

def test_a_tokenizer_is_not_a_token():
    scenario("tokenizer")
    v = secret_violations('      HUGGINGFACE_TOKENIZER: bert-base-uncased\n')
    check("whole-word matching spares TOKENIZER", not v, str(v))


def test_a_path_to_a_secret_is_not_a_secret():
    scenario("secret-path")
    v = secret_violations('      INTERSERVICE_HMAC_SECRET: /run/secrets/hmac\n')
    check("a filesystem path passes", not v, str(v))
    v2 = secret_violations('      TRUSTED_TOKENS_PATH: /config/tokens.txt\n')
    check("a _PATH suffix passes", not v2, str(v2))


def test_a_switch_that_mentions_a_secret_is_not_a_secret():
    scenario("switch")
    v = secret_violations('      TOKEN_HEADER: X-Kai-Token\n')
    check("a header name is not a credential", not v, str(v))


def test_a_directory_is_not_a_secret():
    scenario("secrets-dir")
    v = secret_violations('      SECRETS_DIR: "${SECRETS_DIR:-./runtime}"\n')
    check("SECRETS_DIR is a location, not a value", not v, str(v))


# ── The remaining explicit rule ──────────────────────────────────────

def test_dev_hmac_in_a_deployment_fails():
    scenario("hmac-dev")
    v = secret_violations('      HMAC_ALLOW_DEV_SECRET: "true"\n')
    check("a dev HMAC switch fails", v, str(v))


def test_a_declared_exception_is_exempt_and_reported():
    """Encoded, per (service, key), and printed — not a silent pass."""
    scenario("declared")
    v, declared, _ = secrets.check_file(
        compose('      LLM_API_KEY: ollama-local-no-key-required\n',
                service="memu-graph"))
    check("a declared non-secret does not fail", not v, str(v))
    check("but it is reported every run", declared, str(declared))


def test_the_exception_list_is_narrow():
    """Per (service, key), so one real exception cannot generalise."""
    scenario("exception-narrow")
    v = secret_violations('      LLM_API_KEY: ollama-local-no-key-required\n',
                          service="some-other-service")
    check("the same key elsewhere still fails", v, str(v))


def test_the_denominator_counts_values_not_files():
    scenario("denominator")
    _, _, n = secrets.check_file(
        compose('      A_TOKEN: "${A_TOKEN:-}"\n      B: c\n'))
    check("every environment value is counted", n == 2, str(n))


def test_a_missing_compose_file_is_refused():
    scenario("fail-closed")
    with tempfile.TemporaryDirectory() as tmp:
        raised = False
        try:
            resolve(secrets.COMPOSE_FILES, root=Path(tmp))
        except MissingInputs:
            raised = True
    check("a missing input is refused, not skipped", raised)


# ── check_restart_recovery: the allowlist is now the rule ────────────

def restart_violations(value: str) -> list:
    tmp = Path(tempfile.mkdtemp()) / "c.yml"
    tmp.write_text(
        "x-d: &d\n  logging:\n    options:\n      max-size: 10m\n"
        '      max-file: "3"\n'
        f"services:\n  s:\n    image: x\n    restart: {value}\n")
    return [v for v in restart.check_file(tmp) if "restart" in v]


def test_restart_always_still_fails():
    scenario("restart-always")
    check("restart: always fails", restart_violations("always"))


def test_allowed_restart_values_pass():
    scenario("restart-allowed")
    for value in ("unless-stopped", "on-failure"):
        check(f"restart: {value} passes", not restart_violations(value),
              value)


def test_an_unrecognised_restart_value_now_fails():
    """`ALLOWED_RESTART` was dead code; a typo used to pass silently."""
    scenario("restart-typo")
    v = restart_violations("nonsense-value")
    check("an invalid restart policy fails", v, str(v))
    check("the allowlist is quoted back",
          v and "unless-stopped" in v[0], str(v))


def test_the_allowlist_constant_is_actually_used():
    """A declared-but-unwired constant is an implementation simpler than
    it reads — the same shape as a dead `if ...: pass` branch."""
    scenario("allowlist-wired")
    import inspect
    source = inspect.getsource(restart.check_file)
    check("ALLOWED_RESTART is referenced by the check",
          "ALLOWED_RESTART" in source, "still dead code")


def run_all() -> None:
    """Run every test_* in this module, DERIVED, not listed.

    This was a hand-written list of calls, and it behaved exactly as R5
    predicts a list kept beside the thing behaves: five tests for the
    key-identifier exception were added below and none of them ran. The
    suite reported 28 passed and had not executed a single new
    assertion — a test that runs nothing reads precisely like a test
    that passes.

    The count check below stays, and is now the independent evidence
    (I-8): every test must call `scenario()` exactly once, so the number
    of scenarios executed must equal the number of test functions found.
    Discovery and the tally come from different places, so neither can
    quietly excuse the other.
    """
    tests = sorted((name, fn) for name, fn in globals().items()
                   if name.startswith("test_") and callable(fn))
    for _, fn in tests:
        fn()

    check("every discovered test ran exactly one scenario",
          len(executed) == len(tests),
          f"{len(executed)} scenario(s) from {len(tests)} test(s): {executed}")
    check(f"at least the {EXPECTED_SCENARIOS} historical scenarios remain — "
          f"a ratchet, so deleting tests cannot make this pass",
          len(executed) >= EXPECTED_SCENARIOS, str(len(executed)))
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


# ── Public key IDENTIFIER vs secret key MATERIAL ─────────────────────
#
# The distinction this section exists to encode:
#
#   KAI_SERVICE_KEY_ID is an identiFIER. It travels in the
#   X-Kai-Signature header of every signed request and is listed in the
#   receiver's key map, which looks the identity up BY it. It is public
#   by construction.
#
#   The private key is the secret, and it is mounted from a file.
#
# The risk in encoding that is obvious and is what these tests guard:
# a detector taught to ignore "KEY_ID" would stop seeing real key
# material. So each of these asserts the exception is narrow AND that
# sensitivity is unchanged, which is the property that matters.

def test_the_declared_key_id_is_allowed_for_its_declared_service():
    scenario("key-id-allowed")
    violations, declared, _ = secrets.check_file(compose(
        '      KAI_SERVICE_KEY_ID: agentic-v1\n', service="agentic"))
    check("KAI_SERVICE_KEY_ID=agentic-v1 does not trigger the gate",
          not violations)
    check("and it is REPORTED as a declared exception, not hidden",
          any("KAI_SERVICE_KEY_ID" in d for d in declared))


def test_the_exception_is_scoped_to_one_service():
    scenario("key-id-scoped")
    check("the SAME variable in another service still triggers the gate",
          secret_violations('      KAI_SERVICE_KEY_ID: agentic-v1\n',
                            service="not-agentic"))


def test_no_broad_key_id_heuristic_was_created():
    scenario("no-key-id-heuristic")
    # If a "KEY_ID means safe" rule had been invented, these would pass
    # silently. They must not: the exception is a typed (service, name)
    # pair, not a word the detector now trusts everywhere.
    for name in ("SIGNING_KEY_ID", "PRIVATE_KEY_ID", "API_KEY_ID",
                 "AWS_SECRET_KEY_ID"):
        check(f"{name} with a literal value is still a violation",
              secret_violations(f'      {name}: some-literal-value\n'))


def test_real_key_material_is_still_caught():
    scenario("key-material-still-caught")
    # Sensitivity, asserted rather than assumed. Every one of these is a
    # secret carrying a value, and each must still be refused — in the
    # SAME service the narrow exception applies to.
    material = (
        ("KAI_SERVICE_PRIVATE_KEY", "ed25519:" + "ab" * 32),
        ("PRIVATE_KEY", "ed25519:" + "cd" * 32),
        ("KAI_SERVICE_TOKEN", "77c50e67f9f0e144a373f262da548d38"),
        ("INTERSERVICE_HMAC_SECRET", "not-a-reference"),
        ("API_KEY", "sk-literal"),
        ("DB_PASSWORD", "hunter2"),
    )
    for name, value in material:
        check(f"{name} carrying a value is STILL refused",
              secret_violations(f'      {name}: "{value}"\n',
                                service="agentic"))


def test_the_declared_exception_is_a_typed_pair():
    scenario("exception-is-typed")
    # Encoded in the gate, not in someone's head — and narrow enough to
    # name both halves.
    check("the exception is keyed by (service, variable)",
          ("agentic", "KAI_SERVICE_KEY_ID") in secrets.DECLARED_NON_SECRETS)
    reason = secrets.DECLARED_NON_SECRETS[("agentic", "KAI_SERVICE_KEY_ID")]
    check("and it records WHY it is public — the header and the key map, "
          "not merely an assurance that it is fine",
          "X-Kai-Signature" in reason and "key map" in reason)
    check("and it names where the actual secret lives",
          "private key" in reason.lower())
    check("the exception list has not grown into a habit",
          len(secrets.DECLARED_NON_SECRETS) <= 2)


if __name__ == "__main__":
    run_all()
    print(f"\n{'='*60}")
    print(f"Secret & Restart Gate Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
