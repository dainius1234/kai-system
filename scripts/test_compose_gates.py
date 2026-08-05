"""Port, network-zone and image-tag gate tests — three more that could not fail.

All three were in the `KAI-GATE-003` backlog. Read before hardening, per
the operator's ruling, and each had a defect the retrofit alone would
have preserved under a confident denominator.

The two `check_port_bindings` cases are the failure mode the operator
predicted: not a miss, but a **misleading message**. A gate that says
"port 443 is exposed" when the scanner could not parse the file sends you
to check the port when the parser is what needs checking.

  - Compose's long-form syntax (`host_ip: 127.0.0.1`) is correct, and the
    gate reported it as a violation while telling the operator to bind to
    127.0.0.1 — which they already had.
  - A `ports:` value that is a string rather than a list was iterated
    character by character, producing nine violations about ports named
    '8', '0' and ':'.

`check_network_zones` claimed "every service has an explicit networks
assignment" and implemented it as `if svc_nets is None: pass`.

`check_image_tags` used a denylist of four words, so `myimg:main` passed.
Measured first: all 18 image tags in this repository contain a digit, so
a rule costs nothing and catches every unversioned name.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_image_tags as tags  # noqa: E402
from scripts.security import check_network_zones as zones  # noqa: E402
from scripts.security import check_default_profiles as profiles  # noqa: E402
from scripts.security import check_port_bindings as ports  # noqa: E402
from scripts.security import check_turbovec_writers as turbovec  # noqa: E402

passed = 0
failed = 0

EXPECTED_SCENARIOS = 28
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


def yml(body: str) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "c.yml"
    tmp.write_text(body)
    return tmp


# ── check_port_bindings ──────────────────────────────────────────────

def test_a_non_dashboard_publisher_fails():
    scenario("port-nondash")
    v = ports.check_file(yml('services:\n  agentic:\n    image: x\n'
                             '    ports:\n      - "8007:8007"\n'))
    check("only the dashboard may publish", v, str(v))


def test_the_dashboard_may_publish_on_loopback():
    scenario("port-loopback")
    v = ports.check_file(yml('services:\n  dashboard:\n    image: x\n'
                             '    ports:\n      - "127.0.0.1:8080:8080"\n'))
    check("short-form loopback passes", not v, str(v))


def test_long_form_loopback_is_not_a_violation():
    """The false positive: correct config reported as wrong."""
    scenario("port-longform-ok")
    v = ports.check_file(yml(
        'services:\n  dashboard:\n    image: x\n    ports:\n'
        '      - target: 8080\n        host_ip: 127.0.0.1\n'
        '        published: 8080\n'))
    check("long-form loopback passes", not v, str(v))


def test_long_form_without_loopback_still_fails():
    scenario("port-longform-bad")
    v = ports.check_file(yml(
        'services:\n  dashboard:\n    image: x\n    ports:\n'
        '      - target: 8080\n        published: 8080\n'))
    check("long-form on all interfaces fails", v, str(v))
    check("the message names the host_ip",
          v and "host_ip" in v[0], str(v))


def test_a_malformed_ports_value_says_so():
    """The misleading-message case: don't report ports named '8' and ':'."""
    scenario("port-malformed")
    v = ports.check_file(yml('services:\n  dashboard:\n    image: x\n'
                             '    ports: "8080:8080"\n'))
    check("exactly one violation, not one per character", len(v) == 1, str(v))
    check("it names the shape, not a port", "malformed" in v[0], str(v))
    check("it says the check could not run",
          "could not be checked" in v[0], str(v))


# ── check_network_zones ──────────────────────────────────────────────

def test_a_service_with_no_networks_fails():
    """`if svc_nets is None: pass` — the rule existed in prose only."""
    scenario("zone-no-networks")
    v = zones.check_file(yml('services:\n  x:\n    image: y\n'
                             'networks:\n  agent-net:\n    internal: true\n'))
    check("a service outside every zone is caught", v, str(v))
    check("the consequence is named",
          v and "default bridge" in v[0], str(v))


def test_a_service_with_networks_passes():
    scenario("zone-assigned")
    v = zones.check_file(yml('services:\n  x:\n    image: y\n'
                             '    networks:\n      - agent-net\n'
                             'networks:\n  agent-net:\n    internal: true\n'))
    check("an assigned service passes", not v, str(v))


def test_a_static_ip_fails():
    scenario("zone-static-ip")
    v = zones.check_file(yml(
        'services:\n  x:\n    image: y\n    networks:\n      agent-net:\n'
        '        ipv4_address: 10.0.0.5\n'
        'networks:\n  agent-net:\n    internal: true\n'))
    check("a static IP is caught", v, str(v))


def test_the_legacy_flat_network_fails():
    scenario("zone-legacy")
    v = zones.check_file(yml('services:\n  x:\n    image: y\n'
                             '    networks:\n      - sovereign-net\n'
                             'networks:\n  sovereign-net: {}\n'))
    check("the old flat network is caught", v, str(v))


# ── check_image_tags ─────────────────────────────────────────────────

def test_known_mutable_tags_fail():
    scenario("tag-mutable")
    for image in ("postgres:latest", "myimg:main", "x:master", "y:dev"):
        v = tags.check_file(yml(f'services:\n  s:\n    image: {image}\n'))
        check(f"{image} is rejected", v, image)


def test_an_unversioned_tag_fails_even_if_unlisted():
    """The rule, not the list: `alpine` is on no denylist and moves."""
    scenario("tag-unversioned")
    v = tags.check_file(yml('services:\n  s:\n    image: node:alpine\n'))
    check("an unversioned tag fails", v, str(v))
    check("the reason is that it carries no version",
          v and "no version" in v[0], str(v))


def test_versioned_tags_pass():
    scenario("tag-versioned")
    for image in ("redis:7-alpine", "pgvector/pgvector:pg15",
                  "tailscale/tailscale:v1.78", "python:3.11-slim"):
        v = tags.check_file(yml(f'services:\n  s:\n    image: {image}\n'))
        check(f"{image} passes", not v, f"{image}: {v}")


def test_a_digest_passes():
    scenario("tag-digest")
    v = tags.check_file(yml('services:\n  s:\n    image: app@sha256:abc123\n'))
    check("a digest is the strongest pin", not v, str(v))


def test_an_image_with_no_tag_fails():
    scenario("tag-none")
    v = tags.check_file(yml('services:\n  s:\n    image: postgres\n'))
    check("an untagged image is implicitly latest", v, str(v))


# ── check_default_profiles ───────────────────────────────────────────
#
# This gate and the next were found *correct* on every probe. They are
# here because "correct today" and "proven able to fail" are different
# claims, and only the second survives someone editing it later.

def test_a_dangerous_service_with_no_profile_fails():
    scenario("profile-unprofiled")
    v = profiles.check_file(yml('services:\n  executor:\n    image: x\n'))
    check("an unprofiled dangerous service is caught", v, str(v))
    check("the consequence is named",
          v and "default" in str(v[0]), str(v))


def test_a_correctly_profiled_service_passes():
    scenario("profile-ok")
    v = profiles.check_file(yml('services:\n  executor:\n    image: x\n'
                                '    profiles: ["execution"]\n'))
    check("a profiled dangerous service passes", not v, str(v))


def test_an_empty_profile_list_fails():
    """`profiles: []` reads as configured and behaves as unconfigured."""
    scenario("profile-empty")
    v = profiles.check_file(yml('services:\n  executor:\n    image: x\n'
                                '    profiles: []\n'))
    check("an empty profile list is caught", v, str(v))


# ── check_turbovec_writers ───────────────────────────────────────────

def test_a_second_turbovec_writer_fails():
    scenario("turbovec-second-writer")
    v = turbovec.check_file(yml('services:\n  other:\n    image: x\n'
                                '    environment:\n'
                                '      VECTOR_STORE: turbovec\n'))
    check("a second writer is caught", v, str(v))
    check("the primary is named",
          v and turbovec.PRIMARY_WRITER in str(v[0]), str(v))


def test_a_reader_mounting_read_write_fails():
    scenario("turbovec-rw-mount")
    v = turbovec.check_file(yml(
        'services:\n  reader:\n    image: x\n    environment:\n'
        '      VECTOR_STORE: turbovec\n      TURBOVEC_READ_ONLY: "true"\n'
        '    volumes:\n      - turbovec:/data\n'))
    check("a read-write mount by a reader is caught", v, str(v))


def test_the_primary_writer_may_not_be_read_only():
    """If the only writer is read-only, nothing can write at all."""
    scenario("turbovec-primary-ro")
    v = turbovec.check_file(yml(
        f'services:\n  {turbovec.PRIMARY_WRITER}:\n    image: x\n'
        '    environment:\n      VECTOR_STORE: turbovec\n'
        '      TURBOVEC_READ_ONLY: "true"\n'))
    check("a read-only primary writer is caught", v, str(v))


# ── check_image_tags --verify-exists ─────────────────────────────────
# A tag can be perfectly pinned and simply not be there. On 2026-08-05
# `ollama/ollama:0.6` passed every rule above and had been withdrawn from
# Docker Hub, so every `docker compose up` died in under a second and
# thirteen CI steps never ran. Three more were found the moment the gate
# learned to ask: `prom/prometheus:v3.2`, `prom/alertmanager:v0.28`,
# `ghcr.io/mudler/parakeet.cpp-server:v0.1.0`. All four were
# minor-*series* tags, which upstreams retarget and eventually drop; the
# patch releases underneath them were still there.
#
# These assertions are offline. `tag_resolves` talks to a registry, and a
# test that needs the network is a test that fails on a train.


def test_a_reference_is_split_the_way_a_registry_reads_it():
    scenario("tag-split-reference")
    cases = {
        "redis:7-alpine": ("", "library/redis", "7-alpine"),
        "ollama/ollama:0.6.8": ("", "ollama/ollama", "0.6.8"),
        "ghcr.io/mudler/parakeet.cpp-server:v0.5.0":
            ("ghcr.io", "mudler/parakeet.cpp-server", "v0.5.0"),
        "python": ("", "library/python", "latest"),
    }
    for image, expected in cases.items():
        check(f"{image} splits correctly",
              tags.split_reference(image) == expected,
              f"{tags.split_reference(image)} != {expected}")


def test_an_official_image_gets_the_library_prefix():
    """`redis` is `library/redis` to the registry. Getting this wrong
    makes every official image look withdrawn — four false findings."""
    scenario("tag-library-prefix")
    _, repo, _ = tags.split_reference("redis:7-alpine")
    check("official images resolve under library/", repo == "library/redis", repo)


def test_every_distinct_image_is_collected_once():
    """The denominator. `ollama/ollama` appears twice in the minimal
    profile and four times across all three; it is one question."""
    scenario("tag-images-deduped")
    a = yml('services:\n  x:\n    image: redis:7\n  y:\n    image: redis:7\n'
            '  z:\n    image: nginx:1.2\n  w:\n    build: ./b\n')
    found = tags.images_in([a])
    check("duplicates collapse", len(found) == 2, str(found))
    check("built services are not registry questions",
          all("build" not in i[2] for i in found), str(found))


def test_verification_failure_is_a_finding_not_a_pass():
    """I-1. \"Could not ask the registry\" is not \"the image is there\"."""
    scenario("tag-unreachable-fails")
    original = tags.tag_resolves
    try:
        tags.tag_resolves = lambda image, timeout=20.0: (
            False, "could not verify (URLError: offline)")
        a = yml('services:\n  s:\n    image: redis:7-alpine\n')
        violations, checked = tags.verify_existence([a])
        check("an unverifiable tag is reported", len(violations) == 1,
              str(violations))
        check("and the denominator still counts it", checked == 1, str(checked))
        check("the message says it could not verify",
              violations and "could not verify" in violations[0],
              str(violations))
    finally:
        tags.tag_resolves = original


def test_a_withdrawn_tag_is_reported_with_its_service():
    scenario("tag-withdrawn-named")
    original = tags.tag_resolves
    try:
        tags.tag_resolves = lambda image, timeout=20.0: (
            False, "not found in the registry (HTTP 404)")
        a = yml('services:\n  ollama:\n    image: ollama/ollama:0.6\n')
        violations, _ = tags.verify_existence([a])
        check("it fails", len(violations) == 1, str(violations))
        check("and names the service", "ollama" in violations[0], str(violations))
        check("and names the tag", "0.6" in violations[0], str(violations))
    finally:
        tags.tag_resolves = original


def test_a_resolvable_tag_is_not_reported():
    scenario("tag-resolvable-passes")
    original = tags.tag_resolves
    try:
        tags.tag_resolves = lambda image, timeout=20.0: (True, "resolves")
        a = yml('services:\n  s:\n    image: redis:7-alpine\n')
        violations, checked = tags.verify_existence([a])
        check("no finding", violations == [], str(violations))
        check("one image checked", checked == 1, str(checked))
    finally:
        tags.tag_resolves = original


def test_tag_resolves_itself_reports_an_error_as_absence():
    """Calibration for the check above, which did not have it.

    The first version of these tests stubbed `tag_resolves` wholesale, so
    flipping its own `except` clause from False to True — the exact I-1
    collapse of "could not ask" into "it is fine" — changed nothing and
    50 assertions still passed. A test that cannot see the defect it was
    written for is the thing this programme keeps finding, so the real
    function is driven here with the network taken away from it.
    """
    scenario("tag-resolves-error-path")
    original = tags.urllib.request.urlopen

    def explode(*a, **k):
        raise OSError("network is unreachable")

    try:
        tags.urllib.request.urlopen = explode
        ok, detail = tags.tag_resolves("redis:7-alpine", timeout=1)
        check("an unreachable registry is not a resolved tag", ok is False,
              f"{ok} / {detail}")
        check("and the detail says why", "could not verify" in detail, detail)
    finally:
        tags.urllib.request.urlopen = original


def test_tag_resolves_reports_a_404_as_absence():
    scenario("tag-resolves-404")
    original = tags.urllib.request.urlopen

    def not_found(*a, **k):
        raise tags.urllib.error.HTTPError(
            "http://x", 404, "Not Found", {}, None)

    try:
        tags.urllib.request.urlopen = not_found
        ok, detail = tags.tag_resolves("ollama/ollama:0.6", timeout=1)
        check("a 404 is absence", ok is False, f"{ok} / {detail}")
        check("and says so", "not found" in detail, detail)
    finally:
        tags.urllib.request.urlopen = original


def run_all() -> None:
    test_a_non_dashboard_publisher_fails()
    test_the_dashboard_may_publish_on_loopback()
    test_long_form_loopback_is_not_a_violation()
    test_long_form_without_loopback_still_fails()
    test_a_malformed_ports_value_says_so()
    test_a_service_with_no_networks_fails()
    test_a_service_with_networks_passes()
    test_a_static_ip_fails()
    test_the_legacy_flat_network_fails()
    test_known_mutable_tags_fail()
    test_an_unversioned_tag_fails_even_if_unlisted()
    test_versioned_tags_pass()
    test_a_digest_passes()
    test_an_image_with_no_tag_fails()
    test_a_dangerous_service_with_no_profile_fails()
    test_a_correctly_profiled_service_passes()
    test_an_empty_profile_list_fails()
    test_a_second_turbovec_writer_fails()
    test_a_reader_mounting_read_write_fails()
    test_the_primary_writer_may_not_be_read_only()
    test_a_reference_is_split_the_way_a_registry_reads_it()
    test_an_official_image_gets_the_library_prefix()
    test_every_distinct_image_is_collected_once()
    test_verification_failure_is_a_finding_not_a_pass()
    test_a_withdrawn_tag_is_reported_with_its_service()
    test_a_resolvable_tag_is_not_reported()
    test_tag_resolves_itself_reports_an_error_as_absence()
    test_tag_resolves_reports_a_404_as_absence()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'='*60}")
    print(f"Compose Gate Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
