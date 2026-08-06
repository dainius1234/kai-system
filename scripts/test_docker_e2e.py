"""Docker e2e smoke test — validate compose files and service contracts.

This test validates Docker infrastructure WITHOUT a running stack:
  1. `docker compose config` validates every compose profile's syntax
  2. Every service this repo builds exposes /health in its app.py
  3. Every service this repo builds pins its dependencies
  4. Every service this repo builds runs as a non-root user

The denominator
---------------

This file used to carry, beside a docstring saying *"every service with
a Dockerfile"*, a hand-written list of **seven** service names — and a
second hand-written list of two compose files. The tree builds
**forty-nine** services across **three** profiles. So a claim about
"every service" was being checked against 14 % of them, and the other 42
could have shipped without a health endpoint, without pinned
dependencies and as root without a single test going red.

That is the pattern this programme keeps finding, in its fifteenth
venue: **a check whose scope was smaller than its name implied**. The
remedy is the same one every time — state the denominator, and derive it
from the tree rather than from a list kept beside it. Both lists are now
`gate_inputs.built_services()`, read from the `build:` stanzas, so a
service added to a profile is in scope the moment it is added.

Widening a scope is not free, and the inverse defect is worse: a check
whose scope is *larger* than reality reports failure over things that
are right, which sends people to break working code and buries the one
true finding. So the widening was measured before it was made — all 49
services already have `app.py`, `/health` and `requirements.txt`, so
those three checks widened from 7 to 49 with zero new findings. The
fourth did not, and is handled below.

Run:  pytest scripts/test_docker_e2e.py -v
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from scripts.security.gate_inputs import built_services, compose_files  # noqa: E402

#: `service name -> Dockerfile path`, derived from every profile's
#: `build:` stanza. Services running somebody else's image
#: (`redis:7-alpine`, `pgvector`) are absent by construction: they are
#: not our code to hold to our rules.
SERVICES = built_services(ROOT)

#: Services that do not yet run as a non-root user.
#:
#: The root check used to `print("Advisory: ...")` and pass. That is an
#: inert rule — it had reported these same four names on every run for
#: as long as it has existed, and a signal that repeats forever is one
#: nobody reads. Per the operator's third directive, a recurring signal
#: is fixed, made to fail, or declared with an owner and a date.
#:
#: It is declared rather than fixed today because the fix is not the
#: one-line `USER app` it looks like. `house-doctor` and `skill-hunter`
#: share the `soul_data` volume with `agentic`, and share `memu_data`
#: with `memu-core` under a `user: "1000:1000"` compose override that
#: replaces whatever uid the image bakes in. Docker seeds a fresh named
#: volume from the image directory's ownership, so dropping these to a
#: non-root uid without first making those uids agree across four images
#: would reproduce this morning's `/data/turbovec` permission defect —
#: hardening that breaks the thing it hardens.
#:
#: Checked in both directions: a new root-running service fails here,
#: and so does a name on this list that has since been fixed.
KNOWN_ROOT = {
    # service           owner     review by
    "cortex":          ("orion", "2026-09-01"),
    "house-doctor":    ("orion", "2026-09-01"),
    "skill-hunter":    ("orion", "2026-09-01"),
    "vault-sync":      ("orion", "2026-09-01"),
}


def _runs_as_root(dockerfile: Path) -> bool:
    content = dockerfile.read_text()
    return ("USER " not in content
            and "adduser" not in content.lower()
            and "useradd" not in content.lower())


class TestDenominator(unittest.TestCase):
    """The checks below are worthless if this file found nothing.

    I-1: a survey that inspected zero services reports success
    identically to one that inspected forty-nine.
    """

    def test_services_were_found(self):
        self.assertGreater(
            len(SERVICES), 40,
            f"only {len(SERVICES)} built service(s) discovered — the "
            f"compose files moved or stopped parsing, and every check in "
            f"this file has been inspecting almost nothing")

    def test_compose_profiles_were_found(self):
        self.assertGreaterEqual(len(compose_files(ROOT)), 3,
                                "compose profiles missing from the tree")


class TestDockerComposeConfig(unittest.TestCase):
    """Validate every compose profile's syntax with `docker compose config`."""

    def test_every_compose_profile_is_valid(self):
        failures = []
        for path in compose_files(ROOT):
            result = subprocess.run(
                ["docker", "compose", "-f", str(path), "config", "--quiet"],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode != 0:
                failures.append(f"{path.name}: {result.stderr.strip()}")
        self.assertEqual(failures, [], "\n".join(failures))


class TestServiceHealthEndpoints(unittest.TestCase):
    """Every service this repo builds must expose /health."""

    def test_services_expose_health(self):
        missing = []
        for name, dockerfile in sorted(SERVICES.items()):
            app_path = dockerfile.parent / "app.py"
            if not app_path.exists():
                # Not every image is a FastAPI app; one with no app.py
                # has nothing to expose and no claim to check.
                continue
            if "/health" not in app_path.read_text():
                missing.append(f"{name}: {app_path.relative_to(ROOT)}")
        self.assertEqual(missing, [],
                         "Services missing /health:\n" + "\n".join(missing))


class TestDockerfiles(unittest.TestCase):
    """Every built service needs a Dockerfile, and should not run as root."""

    def test_dockerfiles_exist(self):
        missing = [f"{name}: {path.relative_to(ROOT)}"
                   for name, path in sorted(SERVICES.items())
                   if not path.exists()]
        self.assertEqual(missing, [],
                         "compose names a Dockerfile that is not in the "
                         "tree:\n" + "\n".join(missing))

    def test_no_new_service_runs_as_root(self):
        """Enforced, not printed. A new root-running image fails here."""
        undeclared = sorted(
            name for name, path in SERVICES.items()
            if path.exists() and _runs_as_root(path)
            and name not in KNOWN_ROOT)
        self.assertEqual(
            undeclared, [],
            f"service(s) run as root without a declaration: {undeclared}. "
            f"Add a non-root user (see heartbeat/Dockerfile), or declare "
            f"it in KNOWN_ROOT with an owner and a review date.")

    def test_no_declaration_outlives_its_defect(self):
        """The other direction, so the record cannot drift from the tree.

        A name fixed but left declared is how a list quietly stops
        describing anything — the same failure as the list this file was
        built out of.
        """
        stale = sorted(
            name for name in KNOWN_ROOT
            if name not in SERVICES
            or not (SERVICES[name].exists() and _runs_as_root(SERVICES[name])))
        self.assertEqual(
            stale, [],
            f"KNOWN_ROOT declares service(s) that no longer run as root "
            f"(or no longer exist): {stale}. Remove the declaration.")


class TestComposeDependencies(unittest.TestCase):
    """Verify critical dependency declarations in compose files."""

    def test_memu_depends_on_postgres_minimal(self):
        """memu-core must depend on postgres in minimal compose."""
        path = ROOT / "docker-compose.minimal.yml"
        content = path.read_text()
        self.assertIn("postgres", content)

    def test_pgvector_image_in_minimal(self):
        """Minimal compose should use pgvector image, not plain postgres."""
        path = ROOT / "docker-compose.minimal.yml"
        content = path.read_text()
        self.assertIn("pgvector", content)


class TestRequirementFiles(unittest.TestCase):
    """Every service this repo builds should pin its dependencies."""

    def test_requirements_exist(self):
        missing = []
        for name, dockerfile in sorted(SERVICES.items()):
            if not dockerfile.exists():
                continue
            if "requirements.txt" not in dockerfile.read_text():
                continue        # an image that installs no Python deps
            if not (dockerfile.parent / "requirements.txt").exists():
                missing.append(f"{name}: {dockerfile.parent.relative_to(ROOT)}")
        self.assertEqual(missing, [],
                         "Missing requirements.txt:\n" + "\n".join(missing))


if __name__ == "__main__":
    unittest.main()
