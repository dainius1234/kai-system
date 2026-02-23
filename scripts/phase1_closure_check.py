from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.sovereign.yml"


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def service_block(raw: str, service: str) -> str:
    pattern = rf"(?ms)^  {re.escape(service)}:\n(.*?)(?=^  [a-zA-Z0-9_-]+:\n|^volumes:|^networks:|\Z)"
    m = re.search(pattern, raw)
    return m.group(1) if m else ""


def has_block_dependency(raw: str, service: str, expected: list[str]) -> bool:
    block = service_block(raw, service)
    if not block:
        return False
    for dep in expected:
        if f"- {dep}" not in block:
            return False
    return True


def has_healthcheck(raw: str, service: str) -> bool:
    block = service_block(raw, service)
    return "healthcheck:" in block


def main() -> None:
    raw = COMPOSE.read_text(encoding="utf-8")

    require(has_block_dependency(raw, "tool-gate", ["postgres"]), "tool-gate must depend on postgres")
    require(has_block_dependency(raw, "memu-core", ["postgres", "tool-gate"]), "memu-core must depend on postgres+tool-gate")
    require(has_block_dependency(raw, "executor", ["tool-gate", "memu-core"]), "executor must depend on tool-gate+memu-core")

    for core in ("tool-gate", "memu-core", "executor", "dashboard"):
        require(has_healthcheck(raw, core), f"{core} missing healthcheck")

    require("# TODO: enable GPU when core is stable." in raw, "missing GPU TODO comments")

    for script in ("health_sweep.sh", "contract_smoke.sh"):
        require((ROOT / "scripts" / script).exists(), f"missing scripts/{script}")

    report = {
        "status": "ok",
        "phase1": {
            "patch_set_A": "closed",
            "patch_set_B": "closed",
            "patch_set_C": "closed",
            "patch_set_D": "closed",
            "patch_set_E": "closed (static checks)",
            "patch_set_F": "closed",
        },
        "notes": [
            "Runtime compose-up health still requires Docker-capable host for full dynamic validation."
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
