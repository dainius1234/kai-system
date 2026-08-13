#!/usr/bin/env python3
"""Anti-collapse calibration for the KAI-GATE-048 classifier.

The requirement this file exists to satisfy, verbatim:

    The instrument must be able to distinguish memu-graph from the
    already-known cases. ... If the detector classifies based only on
    syntax such as: package import, --model, "pull", model-looking
    string, internal network — then stop and fix the instrument before
    trusting the result.

The strongest form of that guarantee is structural: `classify()` takes no
service name, no source text and no image tag, so it *cannot* be
recognising one. What is left to prove is that the four known shapes
produce four different verdicts, and that each one moves when the
observation moves rather than staying put.

The four shapes are fed as observation bundles. `memu-graph`'s bundle is
the one under measurement, so it is asserted here only in the forms the
collector can actually return — the point is that each possible return
lands somewhere distinct, not that we already know which.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.security.classify_model_startup import (  # noqa: E402
    LAZY, NO_LOAD, Observations, PRE_DELEGATED, PRE_EXTERNAL, PRE_LOCAL,
    RUNTIME, UNKNOWN, VERDICTS, classify,
)

passed = 0
failed = 0
EXPECTED_SCENARIOS = 16
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


# ── the four known shapes, as observations ───────────────────────────

MEMU_CORE = Observations(          # complete local/offline contract
    reached_ready=True, loaded_before_ready=True, loaded_at_all=True,
    external_resolution_attempted=False, asset_present_locally=True,
    egress_available=False, evidence_level=RUNTIME,
)
INTROSPECT = Observations(         # pre-readiness external, broken
    reached_ready=False, loaded_before_ready=True, loaded_at_all=True,
    external_resolution_attempted=True, asset_present_locally=False,
    egress_available=False, evidence_level=RUNTIME,
)
OLLAMA_PULL = Observations(        # fetch-looking, delegated, correct
    reached_ready=True, loaded_before_ready=True, loaded_at_all=True,
    delegated_to="ollama", delegate_has_egress=True,
    egress_available=False, evidence_level=RUNTIME,
)
GRAPH_LAZY = Observations(         # one possible memu-graph outcome
    reached_ready=True, loaded_before_ready=False, loaded_at_all=True,
    asset_present_locally=False, egress_available=False,
    evidence_level=RUNTIME,
)


def test_the_four_known_shapes_do_not_collapse() -> None:
    scenario("four shapes stay distinct")
    got = {
        "memu-core": classify(MEMU_CORE)[0],
        "memu-core-introspect": classify(INTROSPECT)[0],
        "ollama-pull": classify(OLLAMA_PULL)[0],
        "memu-graph(lazy)": classify(GRAPH_LAZY)[0],
    }
    check("four inputs give four distinct verdicts",
          len(set(got.values())) == 4, str(got))
    check("memu-core is PRE-READINESS LOCAL", got["memu-core"] == PRE_LOCAL,
          got["memu-core"])
    check("introspect is PRE-READINESS EXTERNAL",
          got["memu-core-introspect"] == PRE_EXTERNAL,
          got["memu-core-introspect"])
    check("ollama-pull is PRE-READINESS DELEGATED",
          got["ollama-pull"] == PRE_DELEGATED, got["ollama-pull"])
    check("a lazy graph load is REQUEST-TIME / LAZY",
          got["memu-graph(lazy)"] == LAZY, got["memu-graph(lazy)"])
    check("every verdict is in the declared vocabulary",
          set(got.values()) <= VERDICTS, str(got))


def test_delegation_is_tested_before_egress() -> None:
    """`ollama-pull` has NO egress and runs a fetch. Checking egress
    first turns a correct design into a defect — the inverted scope."""
    scenario("delegation beats egress")
    verdict, why = classify(OLLAMA_PULL)
    check("not called external", verdict != PRE_EXTERNAL, verdict)
    check("the delegate is named in the reason", "ollama" in why, why)
    check("egress absence alone did not decide it",
          OLLAMA_PULL.egress_available is False and verdict == PRE_DELEGATED,
          f"{OLLAMA_PULL.egress_available} {verdict}")


def test_a_delegate_without_egress_is_not_laundered() -> None:
    """The other direction: delegation must not become a blanket excuse.
    If the peer also has no egress, the constraint is unresolved and the
    verdict must say so."""
    scenario("delegate without egress")
    obs = Observations(**{**OLLAMA_PULL.__dict__, "delegate_has_egress": False,
                          "notes": []})
    verdict, why = classify(obs)
    check("still delegated", verdict == PRE_DELEGATED, verdict)
    check("but the reason flags the unresolved constraint",
          "NO EGRESS" in why.upper(), why)


def test_timing_is_tested_before_the_asset_contract() -> None:
    """Same missing asset, two timings, two verdicts. If these collapsed,
    a first-request failure and a readiness failure would share a remedy
    and one of them would get the wrong one."""
    scenario("timing before contract")
    lazy = classify(GRAPH_LAZY)[0]
    eager = classify(Observations(
        reached_ready=False, loaded_before_ready=True, loaded_at_all=True,
        external_resolution_attempted=True, asset_present_locally=False,
        egress_available=False, evidence_level=RUNTIME))[0]
    check("lazy is LAZY", lazy == LAZY, lazy)
    check("eager is EXTERNAL", eager == PRE_EXTERNAL, eager)
    check("they differ", lazy != eager, f"{lazy} == {eager}")


def test_a_lazy_path_with_no_local_asset_says_so() -> None:
    """Option 3′'s failure mode, in the reason string: deferring the load
    does not make the asset appear."""
    scenario("lazy without asset")
    _, why = classify(GRAPH_LAZY)
    check("names the first-request resolution", "first request" in why, why)
    check("names the egress constraint", "no proven egress" in why, why)
    withasset = classify(Observations(
        reached_ready=True, loaded_before_ready=False, loaded_at_all=True,
        asset_present_locally=True, egress_available=False,
        evidence_level=RUNTIME))
    check("and a lazy path WITH the asset does not say it",
          "first request" not in withasset[1], withasset[1])


def test_not_measured_propagates_to_unknown() -> None:
    """`None` is not `False`. "We did not look" must never read as
    "we looked and it was absent"."""
    scenario("None is not False")
    check("no observations at all -> UNKNOWN",
          classify(Observations())[0] == UNKNOWN, "")
    check("a load with unknown timing -> UNKNOWN",
          classify(Observations(reached_ready=True, loaded_at_all=True))[0]
          == UNKNOWN, "")
    check("delegation with an unmeasured delegate -> UNKNOWN",
          classify(Observations(
              reached_ready=True, loaded_at_all=True,
              loaded_before_ready=True, delegated_to="ollama"))[0] == UNKNOWN,
          "")
    check("a lazy path with unmeasured asset still classifies as LAZY",
          classify(Observations(reached_ready=True, loaded_at_all=True,
                                loaded_before_ready=False))[0] == LAZY, "")
    _, why = classify(Observations(reached_ready=True, loaded_at_all=True,
                                   loaded_before_ready=False))
    check("...and says the asset was not measured", "NOT MEASURED" in why, why)


def test_a_failed_bring_up_does_not_produce_a_healthy_verdict() -> None:
    """R11. No subject, no observation — with the one exception that is a
    stronger finding, not a weaker one: if the load is what stopped it."""
    scenario("R11 boundary")
    verdict, why = classify(Observations(reached_ready=False,
                                         evidence_level=RUNTIME))
    check("UNKNOWN, not NO_LOAD", verdict == UNKNOWN, verdict)
    check("says what was not measured", "NOT MEASURED" in why, why)
    check("but a load that CAUSED the failure is still reported",
          classify(INTROSPECT)[0] == PRE_EXTERNAL, "")


def test_no_load_is_distinguishable_from_not_measured() -> None:
    scenario("no-load vs not-measured")
    absent = classify(Observations(reached_ready=True, loaded_at_all=False,
                                   evidence_level=RUNTIME))
    unseen = classify(Observations(reached_ready=True, evidence_level=RUNTIME))
    check("an observed absence is NO MODEL LOAD OBSERVED",
          absent[0] == NO_LOAD, absent[0])
    check("an absent observation is UNKNOWN", unseen[0] == UNKNOWN, unseen[0])
    check("they differ", absent[0] != unseen[0], "")


def test_the_classifier_takes_no_name() -> None:
    """The structural guarantee. If a service name cannot enter the
    function, the function cannot be recognising one."""
    scenario("no name is an input")
    import inspect
    from scripts.security import classify_model_startup as mod
    params = set(inspect.signature(classify).parameters)
    check("classify takes exactly one argument", params == {"obs"}, str(params))
    fields = set(Observations().__dict__)
    for banned in ("name", "service", "image", "source", "network", "command"):
        check(f"no `{banned}` field in Observations", banned not in fields,
              str(sorted(fields)))
    # Comments are stripped first. This check fired on the explanatory
    # comment naming `ollama-pull` as the reason delegation is tested
    # before egress — prose that cannot influence a verdict. Scanning
    # executable text keeps the assertion about behaviour; deleting the
    # comment to satisfy a literal match would have removed the argument
    # and kept the check happy, which is the wrong trade.
    src = "\n".join(line.split("#")[0]
                    for line in inspect.getsource(mod.classify).splitlines())
    for literal in ("memu", "ollama", "cognee", "transformers",
                    "internal", "--model"):
        check(f"`{literal}` does not appear in classify()'s executable text",
              literal not in src, "the classifier is matching a name")


def test_identical_observations_give_identical_verdicts() -> None:
    """The same evidence about two different services must produce the
    same verdict. Anything else means something other than the evidence
    is deciding."""
    scenario("evidence alone decides")
    a = classify(Observations(**{**GRAPH_LAZY.__dict__, "notes": []}))
    b = classify(Observations(**{**GRAPH_LAZY.__dict__, "notes": []}))
    check("deterministic", a == b, f"{a} vs {b}")
    core_shape = classify(Observations(**{**MEMU_CORE.__dict__, "notes": []}))
    check("a service with memu-core's evidence gets memu-core's verdict",
          core_shape[0] == PRE_LOCAL, core_shape[0])


def test_every_declared_verdict_is_reachable() -> None:
    """A vocabulary entry no input can produce is an inert rule (I-5)."""
    scenario("all verdicts reachable")
    produced = {
        classify(MEMU_CORE)[0], classify(INTROSPECT)[0],
        classify(OLLAMA_PULL)[0], classify(GRAPH_LAZY)[0],
        classify(Observations(reached_ready=True, loaded_at_all=False))[0],
        classify(Observations())[0],
    }
    missing = VERDICTS - produced
    check("every verdict in the vocabulary is reachable", not missing,
          f"unreachable: {sorted(missing)}")


# ══ the PARSING half — where a wrong bundle would come from ═════════
#
# `classify()` is only as good as the Observations handed to it. The
# collector's stage logs are parsed by `summarise_memu_graph_startup`,
# and a parser that read "0 files" as "not measured" (or the reverse)
# would produce a confident wrong verdict from correct evidence. Both
# directions are asserted on synthetic stage-log directories.

def _stage_dir(root, files: dict):
    from pathlib import Path as P
    d = P(root)
    for name, body in files.items():
        (d / name).write_text(textwrap.dedent(body))
    return d


def test_the_cache_snapshot_parser_reads_counts_and_absence() -> None:
    scenario("cache count parser")
    import tempfile
    from scripts.security import summarise_memu_graph_startup as s
    with tempfile.TemporaryDirectory() as tmp:
        d = _stage_dir(tmp, {
            "empty.log": """
                HF_HOME=/data/hf_cache
                --- file count ---
                0
                """,
            "full.log": """
                /data/hf_cache/models--bert-base-uncased/config.json
                --- file count ---
                7
                """,
            "garbage.log": "the container exploded\n",
        })
        check("zero parses as 0, not None",
              s.file_count(s.read(d / "empty.log")) == 0,
              str(s.file_count(s.read(d / "empty.log"))))
        check("seven parses as 7",
              s.file_count(s.read(d / "full.log")) == 7, "")
        check("an unparseable log is None, NOT zero",
              s.file_count(s.read(d / "garbage.log")) is None,
              "a missing count must not read as an empty cache")
        check("a missing file is None",
              s.file_count(s.read(d / "nope.log")) is None, "")


def test_the_maps_parser_separates_absent_from_unmeasured() -> None:
    scenario("maps parser")
    import tempfile
    from scripts.security import summarise_memu_graph_startup as s
    with tempfile.TemporaryDirectory() as tmp:
        d = _stage_dir(tmp, {
            "loaded.log": "tokenizers: 4\ntorch: 0\nsafetensors: 0\n",
            "notloaded.log": "tokenizers: 0\ntorch: 0\nsafetensors: 0\n",
            "nothing.log": "exec failed: container not running\n",
        })
        check("a mapped extension is YES",
              s.maps_loaded(s.read(d / "loaded.log")) is True, "")
        check("all zeros is NO", s.maps_loaded(s.read(d / "notloaded.log")) is False, "")
        check("an exec failure is NOT MEASURED, not NO",
              s.maps_loaded(s.read(d / "nothing.log")) is None,
              "a failed probe must not read as a proven absence")


def test_external_resolution_is_matched_on_transport_not_on_words() -> None:
    """The hazard rule again, in the parser: the word "model" must not
    be evidence of a registry round-trip."""
    scenario("external attempt parser")
    from scripts.security import summarise_memu_graph_startup as s
    check("a registry host counts",
          s.external_attempt("OSError: couldn't connect to 'https://huggingface.co'")
          is True, "")
    check("the hub retry line counts",
          s.external_attempt("Retrying in 2s [Retry 2/5]") is True, "")
    check("the word 'model' alone does NOT",
          s.external_attempt("loading model from local cache") is False,
          "a construct was matched instead of a transport event")
    check("a model-shaped name alone does NOT",
          s.external_attempt("HUGGINGFACE_TOKENIZER=bert-base-uncased") is False,
          "an env value was read as a network attempt")
    check("no logs at all is NOT MEASURED",
          s.external_attempt(None, None) is None, "")


def test_the_bundle_builder_detects_request_time_growth() -> None:
    """The end-to-end shape this whole unit exists to distinguish: the
    cache is empty in the image, still empty at readiness, and grows
    across the first request."""
    scenario("request-time growth end to end")
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        _stage_dir(tmp, {
            "image-cache.log": "--- file count ---\n0\n",
            "cache-ready.log": "--- file count ---\n0\n",
            "cache-after.log": "--- file count ---\n11\n",
            "maps-ready.log": "tokenizers: 0\ntorch: 0\nsafetensors: 0\n",
            "maps-after.log": "tokenizers: 3\ntorch: 0\nsafetensors: 0\n",
            "chronology.log": "FIRST PASSING HEALTH PROBE at +12.4s from StartedAt\n",
            "egress-probe.log": "huggingface.co:443 FAILED (...) -> no egress on this path\n",
            "live-cycle.log": "OSError: couldn't connect to 'https://huggingface.co'\n",
            "service-logs.log": "",
        })
        out = subprocess.run(
            [sys.executable, "scripts/security/summarise_memu_graph_startup.py",
             "--stage-logs", tmp, "--probe-rc", "0", "--live-rc", "1"],
            cwd=REPO, capture_output=True, text=True, timeout=60).stdout
        check("verdict is REQUEST-TIME / LAZY", f"VERDICT: {LAZY}" in out, out[-400:])
        check("it reports the growth", "cache GREW by 11" in out, out[-400:])
        check("readiness was reached", "reached readiness      YES" in out, out[-400:])
        check("loaded before ready is NO", "loaded before ready    NO" in out, out[-400:])
        check("egress proven absent", "egress proven          NO" in out, out[-400:])


def test_the_bundle_builder_refuses_when_nothing_was_collected() -> None:
    """An empty stage directory must not produce a verdict. This is the
    run-1 failure in miniature: a table of correct-looking rows built on
    a prerequisite that never held."""
    scenario("empty stage dir refuses")
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = subprocess.run(
            [sys.executable, "scripts/security/summarise_memu_graph_startup.py",
             "--stage-logs", tmp],
            cwd=REPO, capture_output=True, text=True, timeout=60).stdout
        check("verdict is UNKNOWN", f"VERDICT: {UNKNOWN}" in out, out[-400:])
        check("and every field says NOT MEASURED",
              out.count("NOT MEASURED") >= 5, out[-600:])


def run_all() -> None:
    test_the_four_known_shapes_do_not_collapse()
    test_delegation_is_tested_before_egress()
    test_a_delegate_without_egress_is_not_laundered()
    test_timing_is_tested_before_the_asset_contract()
    test_a_lazy_path_with_no_local_asset_says_so()
    test_not_measured_propagates_to_unknown()
    test_a_failed_bring_up_does_not_produce_a_healthy_verdict()
    test_no_load_is_distinguishable_from_not_measured()
    test_the_classifier_takes_no_name()
    test_identical_observations_give_identical_verdicts()
    test_every_declared_verdict_is_reachable()
    test_the_cache_snapshot_parser_reads_counts_and_absence()
    test_the_maps_parser_separates_absent_from_unmeasured()
    test_external_resolution_is_matched_on_transport_not_on_words()
    test_the_bundle_builder_detects_request_time_growth()
    test_the_bundle_builder_refuses_when_nothing_was_collected()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Model Startup Classifier Calibration: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
