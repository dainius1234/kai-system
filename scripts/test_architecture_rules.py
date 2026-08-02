"""Architecture dependency gate tests — roadmap §15.

The gate itself is only worth having if it can fail. These tests inject
real violations into a temporary copy of the tree and assert each rule
catches its own, because a check that silently passes everything is worse
than no check: it manufactures confidence.

That is not hypothetical here — the first negative test written for this
gate appeared to pass while the injected violation had never actually
been written to the file. The test was checking nothing.
"""
from __future__ import annotations

import ast
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_architecture_rules as arch

passed = 0
failed = 0


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


REPO = Path(__file__).resolve().parent.parent


class _Injected:
    """Temporarily append text to a source file, then restore it.

    Verifies the injection actually parsed, so a negative test cannot
    quietly become a no-op.
    """

    def __init__(self, rel_path: str, snippet: str) -> None:
        self.path = REPO / rel_path
        self.snippet = snippet
        self.original: str | None = None

    def __enter__(self):
        self.original = self.path.read_text(encoding="utf-8")
        self.path.write_text(self.original + self.snippet, encoding="utf-8")
        # Prove the injection is real and parseable.
        ast.parse(self.path.read_text(encoding="utf-8"))
        return self

    def __exit__(self, *exc):
        if self.original is not None:
            self.path.write_text(self.original, encoding="utf-8")
        return False


# ═══════════════════════════════════════════════════════════════════
# 1. The gate passes on the current tree
# ═══════════════════════════════════════════════════════════════════

def test_current_tree_is_clean():
    for label, rule in arch.CHECKS:
        violations = rule()
        check(f"clean_{label.split()[0]}", not violations,
              "; ".join(str(v).strip() for v in violations[:3]))


# ═══════════════════════════════════════════════════════════════════
# 2. Each rule can actually fail
# ═══════════════════════════════════════════════════════════════════

def test_rule_1_detects_provider_importing_actuator():
    snippet = "\n\nfrom paper_trader import get_paper_trader  # test injection\n"
    with _Injected("agentic/alpha_signals.py", snippet):
        violations = arch.rule_1_provider_imports_actuator()
        check("rule1_detects", len(violations) == 1,
              f"got {len(violations)}")
        if violations:
            check("rule1_names_module",
                  "alpha_signals" in violations[0].module)
            check("rule1_names_actuator",
                  "paper_trader" in violations[0].message)

    check("rule1_clean_after_restore",
          not arch.rule_1_provider_imports_actuator())


def test_rule_1_detects_transformer_importing_actuator():
    snippet = "\n\nimport swarm  # test injection\n"
    with _Injected("agentic/causal_world_model.py", snippet):
        violations = arch.rule_1_provider_imports_actuator()
        check("rule1_transformer_detects", len(violations) == 1)


def test_rule_2_detects_specialist_importing_actuator():
    snippet = "\n\nfrom paper_trader import get_paper_trader  # test injection\n"
    with _Injected("agentic/hypothesis.py", snippet):
        violations = arch.rule_2_specialist_side_effects()
        check("rule2_import_detects", len(violations) >= 1)
        if violations:
            check("rule2_import_message",
                  "imports actuator" in violations[0].message)


def test_rule_2_detects_specialist_posting_to_side_effect_target():
    snippet = (
        "\n\nasync def _test_injection(client):\n"
        "    return await client.post('http://notify-service:8031/notify', json={})\n"
    )
    with _Injected("agentic/dialectic.py", snippet):
        violations = arch.rule_2_specialist_side_effects()
        check("rule2_sideeffect_detects", len(violations) >= 1,
              f"got {len(violations)}")


def test_rule_2_ignores_read_only_post():
    """A POST to a verifier is a request body, not a side effect.

    Flagging it would train people to ignore the gate.
    """
    snippet = (
        "\n\nasync def _test_readonly(client):\n"
        "    return await client.post('http://verifier:8009/verify', json={})\n"
    )
    with _Injected("agentic/analogy.py", snippet):
        violations = arch.rule_2_specialist_side_effects()
        check("rule2_ignores_verifier", not violations,
              "; ".join(str(v).strip() for v in violations))


def test_rule_3_detects_d102_credentials():
    snippet = "\n\n_TEST_KEY = 'BINANCE_API_SECRET'  # test injection\n"
    with _Injected("agentic/global_workspace.py", snippet):
        violations = arch.rule_3_d102_credentials()
        check("rule3_detects_credential", len(violations) >= 1)
        if violations:
            check("rule3_names_credential",
                  "BINANCE_API_SECRET" in violations[0].message)


def test_rule_3_detects_d102_actuator_import():
    snippet = "\n\nimport teammates  # test injection\n"
    with _Injected("agentic/global_workspace.py", snippet):
        violations = arch.rule_3_d102_credentials()
        check("rule3_detects_actuator",
              any("imports actuator" in v.message for v in violations))


def test_rule_14_detects_fail_open():
    snippet = (
        "\n\ndef _test_fail_open():\n"
        "    try:\n"
        "        pass\n"
        "    except Exception:\n"
        "        pass\n"
    )
    with _Injected("common/policy_bridge/capability.py", snippet):
        violations = arch.rule_14_no_fail_open()
        check("rule14_detects", len(violations) >= 1)
        if violations:
            check("rule14_names_path",
                  "policy_bridge" in violations[0].module)


# ═══════════════════════════════════════════════════════════════════
# 3. Role taxonomy integrity
# ═══════════════════════════════════════════════════════════════════

def test_roles_are_disjoint():
    """A module in two roles is exactly the dual-role bug UH-0 flagged."""
    groups = {
        "providers": arch.PERCEPTION_PROVIDERS,
        "transformers": arch.TRANSFORMERS,
        "specialists": arch.PROPOSAL_SPECIALISTS,
        "authorities": arch.POLICY_AUTHORITIES,
        "actuators": arch.ACTUATORS,
    }
    names = list(groups)
    overlaps = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = groups[a] & groups[b]
            if shared:
                overlaps.append(f"{a}∩{b}={sorted(shared)}")
    check("roles_disjoint", not overlaps, "; ".join(overlaps))


def test_every_classified_module_exists():
    missing = []
    for group in (arch.PERCEPTION_PROVIDERS, arch.TRANSFORMERS,
                  arch.PROPOSAL_SPECIALISTS, arch.POLICY_AUTHORITIES,
                  arch.ACTUATORS):
        for module in group:
            if not (REPO / "agentic" / f"{module}.py").exists():
                missing.append(module)
    check("classified_modules_exist", not missing, str(missing))


def test_uncheckable_rules_are_declared():
    """Rules that cannot be checked must say so, not be silently absent."""
    check("uncheckable_declared", len(arch.NOT_STATICALLY_CHECKABLE) == 3)
    for rule in (9, 13, 15):
        check(f"rule_{rule}_declared_uncheckable",
              rule in arch.NOT_STATICALLY_CHECKABLE)


def test_all_checks_return_lists():
    for label, rule in arch.CHECKS:
        result = rule()
        check(f"returns_list_{label.split()[0]}", isinstance(result, list))


# ═══════════════════════════════════════════════════════════════════
# 4. Every §15 rule is accounted for
# ═══════════════════════════════════════════════════════════════════

def test_all_fifteen_rules_accounted_for():
    """No rule may be silently absent.

    This gate shipped once with rules 5, 6, 7, 8, 10 and 11 neither
    enforced nor declared uncheckable — invisible, which is the exact
    failure the gate exists to prevent. This test makes that
    impossible to repeat.
    """
    unaccounted = sorted(arch.ALL_RULES - arch.accounted_rules())
    check("all_15_rules_accounted", not unaccounted,
          f"silently missing: {unaccounted}")
    check("accounted_is_15", len(arch.accounted_rules()) == 15,
          str(len(arch.accounted_rules())))


def test_enforced_and_declared_are_disjoint():
    """A rule cannot be both enforced and declared uncheckable."""
    import re
    enforced = {int(re.match(r"(\d+)", label).group(1))
                for label, _ in arch.CHECKS}
    overlap = enforced & set(arch.NOT_STATICALLY_CHECKABLE)
    check("no_rule_both_ways", not overlap, str(sorted(overlap)))


# ═══════════════════════════════════════════════════════════════════
# 5. The six previously-missing rules can each fail
# ═══════════════════════════════════════════════════════════════════

def test_rule_6_detects_unprotected_side_effect_route():
    snippet = (
        '\n\n@app.post("/uh-test-unprotected")\n'
        "async def _test_unprotected():\n"
        "    return {}\n"
    )
    with _Injected("output/notify/app.py", snippet):
        violations = arch.rule_6_action_routes_registered()
        check("rule6_detects", len(violations) >= 1)
        if violations:
            check("rule6_names_route",
                  any("uh-test-unprotected" in v.message for v in violations))


def test_rule_6_ignores_health_routes():
    snippet = (
        '\n\n@app.post("/health")\n'
        "async def _test_health():\n"
        "    return {}\n"
    )
    with _Injected("vault-sync/app.py", snippet):
        violations = arch.rule_6_action_routes_registered()
        check("rule6_ignores_health",
              not any("/health" in v.message for v in violations))


def test_rule_7_reports_open_legacy_path():
    """Rule 7 reads the legacy verifier, so it tracks real closure state."""
    violations = arch.rule_7_legacy_paths_closed()
    check("rule7_clean_now", not violations,
          "; ".join(v.message for v in violations[:2]))

    import common.actuator_registry.legacy_verification as lv
    original = lv.open_legacy_paths
    lv.open_legacy_paths = lambda: {"fake-actuator": "still unauthenticated"}
    try:
        import importlib
        importlib.reload(arch)
        found = arch.rule_7_legacy_paths_closed()
        check("rule7_detects_open", len(found) >= 1)
    finally:
        lv.open_legacy_paths = original
        importlib.reload(arch)


def test_rule_8_detects_success_shaped_dict():
    snippet = (
        "\n\ndef _test_success_shape():\n"
        "    return {'success': True}\n"
    )
    with _Injected("common/erasure/coordinator.py", snippet):
        violations = arch.rule_8_typed_operation_state()
        check("rule8_detects", len(violations) >= 1)
        if violations:
            check("rule8_explains",
                  "typed operation state" in violations[0].message)


def test_rule_8_ignores_real_payloads():
    """A dict built from real values is a payload, not a success shape."""
    snippet = (
        "\n\ndef _test_real_payload(count):\n"
        "    return {'status': count, 'extra': count}\n"
    )
    with _Injected("common/erasure/coordinator.py", snippet):
        violations = arch.rule_8_typed_operation_state()
        check("rule8_ignores_payload",
              not any("_test_real_payload" in str(v.line) for v in violations))


def test_rule_10_requires_context_fields():
    violations = arch.rule_10_records_carry_context()
    check("rule10_clean_now", not violations,
          "; ".join(v.message for v in violations[:3]))


def test_rule_10_detects_bare_basemodel_contract():
    snippet = (
        "\n\nclass _TestBareContract(BaseModel):\n"
        "    value: str\n"
    )
    with _Injected("common/contracts/action.py", snippet):
        violations = arch.rule_10_records_carry_context()
        check("rule10_detects_bare",
              any("_TestBareContract" in v.message for v in violations))


def test_rule_11_model_output_cannot_grant_trust():
    violations = arch.rule_11_model_output_labelled()
    check("rule11_clean_now", not violations,
          "; ".join(v.message for v in violations[:2]))

    from common.contracts.autonomy import EvidenceGrade
    check("rule11_model_generated_disqualified",
          not EvidenceGrade.MODEL_GENERATED.qualifies())
    check("rule11_simulated_disqualified",
          not EvidenceGrade.SIMULATED.qualifies())


def test_rule_5_requires_short_circuit():
    violations = arch.rule_5_trust_cannot_bypass()
    check("rule5_clean_now", not violations,
          "; ".join(v.message for v in violations[:2]))


# ── Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_current_tree_is_clean()
    test_rule_1_detects_provider_importing_actuator()
    test_rule_1_detects_transformer_importing_actuator()
    test_rule_2_detects_specialist_importing_actuator()
    test_rule_2_detects_specialist_posting_to_side_effect_target()
    test_rule_2_ignores_read_only_post()
    test_rule_3_detects_d102_credentials()
    test_rule_3_detects_d102_actuator_import()
    test_rule_14_detects_fail_open()
    test_roles_are_disjoint()
    test_every_classified_module_exists()
    test_uncheckable_rules_are_declared()
    test_all_checks_return_lists()
    test_all_fifteen_rules_accounted_for()
    test_enforced_and_declared_are_disjoint()
    test_rule_6_detects_unprotected_side_effect_route()
    test_rule_6_ignores_health_routes()
    test_rule_7_reports_open_legacy_path()
    test_rule_8_detects_success_shaped_dict()
    test_rule_8_ignores_real_payloads()
    test_rule_10_requires_context_fields()
    test_rule_10_detects_bare_basemodel_contract()
    test_rule_11_model_output_cannot_grant_trust()
    test_rule_5_requires_short_circuit()

    print(f"\n{'='*60}")
    print(f"Architecture Rule Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
