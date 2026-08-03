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
from scripts.security import check_port_bindings as ports  # noqa: E402

passed = 0
failed = 0

EXPECTED_SCENARIOS = 14
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
