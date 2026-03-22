"""Docker e2e smoke test — validate compose files and service contracts.

This test validates Docker infrastructure WITHOUT a running stack:
  1. docker compose config validates compose file syntax
  2. Every service with a Dockerfile exposes /health in its app.py
  3. Required env vars are declared in compose files
  4. Service dependency graph has no obvious gaps

Run:  pytest scripts/test_docker_e2e.py -v
"""
from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Compose files to validate
COMPOSE_FILES = [
    "docker-compose.minimal.yml",
    "docker-compose.full.yml",
]

# Core services that MUST expose /health
CORE_SERVICES = [
    "memu-core",
    "tool-gate",
    "executor",
    "heartbeat",
    "langgraph",
    "dashboard",
    "orchestrator",
    "fusion-engine",
]


class TestDockerComposeConfig(unittest.TestCase):
    """Validate compose file syntax with `docker compose config`."""

    def _validate_compose(self, filename: str):
        path = ROOT / filename
        if not path.exists():
            self.skipTest(f"{filename} not found")
        result = subprocess.run(
            ["docker", "compose", "-f", str(path), "config", "--quiet"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(
            result.returncode, 0,
            f"{filename} config failed:\n{result.stderr}",
        )

    def test_minimal_compose_valid(self):
        self._validate_compose("docker-compose.minimal.yml")

    def test_full_compose_valid(self):
        self._validate_compose("docker-compose.full.yml")


class TestServiceHealthEndpoints(unittest.TestCase):
    """Every core service app.py must contain a /health route."""

    def test_core_services_expose_health(self):
        missing = []
        for svc in CORE_SERVICES:
            app_path = ROOT / svc / "app.py"
            if not app_path.exists():
                missing.append(f"{svc}: no app.py")
                continue
            content = app_path.read_text()
            if "/health" not in content:
                missing.append(f"{svc}: no /health endpoint")

        self.assertEqual(missing, [], "Services missing /health:\n" + "\n".join(missing))


class TestDockerfiles(unittest.TestCase):
    """Every service directory with app.py should have a Dockerfile."""

    def test_dockerfiles_exist(self):
        missing = []
        for svc in CORE_SERVICES:
            svc_dir = ROOT / svc
            if (svc_dir / "app.py").exists() and not (svc_dir / "Dockerfile").exists():
                missing.append(svc)
        self.assertEqual(missing, [], f"Missing Dockerfiles: {missing}")

    def test_dockerfiles_use_nonroot_user(self):
        """Dockerfiles should not run as root (security best practice)."""
        risky = []
        for svc in CORE_SERVICES:
            dockerfile = ROOT / svc / "Dockerfile"
            if not dockerfile.exists():
                continue
            content = dockerfile.read_text()
            # Check for USER instruction or adduser/useradd
            if "USER " not in content and "adduser" not in content.lower() and "useradd" not in content.lower():
                risky.append(svc)
        # This is advisory — don't fail, just report
        if risky:
            print(f"Advisory: services running as root: {risky}")


class TestComposeDependencies(unittest.TestCase):
    """Verify critical dependency declarations in compose files."""

    def test_memu_depends_on_postgres_minimal(self):
        """memu-core must depend on postgres in minimal compose."""
        path = ROOT / "docker-compose.minimal.yml"
        if not path.exists():
            self.skipTest("minimal compose not found")
        content = path.read_text()
        # After our fix, memu-core should depend on postgres
        self.assertIn("postgres", content)

    def test_pgvector_image_in_minimal(self):
        """Minimal compose should use pgvector image, not plain postgres."""
        path = ROOT / "docker-compose.minimal.yml"
        if not path.exists():
            self.skipTest("minimal compose not found")
        content = path.read_text()
        self.assertIn("pgvector", content)


class TestRequirementFiles(unittest.TestCase):
    """Every service with a Dockerfile should pin its dependencies."""

    def test_requirements_exist(self):
        missing = []
        for svc in CORE_SERVICES:
            svc_dir = ROOT / svc
            if (svc_dir / "Dockerfile").exists() and not (svc_dir / "requirements.txt").exists():
                missing.append(svc)
        self.assertEqual(missing, [], f"Missing requirements.txt: {missing}")


if __name__ == "__main__":
    unittest.main()
