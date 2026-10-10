#!/usr/bin/env python3
"""P0-PR-05 CI gate: a secret must have no value in the compose file at all.

The rule, stated once: **a secret-shaped variable may be referenced, but
never given a value here.** `${TOKEN}` and `${TOKEN:-}` are both fine —
one is unset, the other explicitly empty, and both make the service fail
closed. Anything else is a value living in a file anyone can read.

This gate previously matched a **denylist of nine words**
(`localdev|changeme|password|...`). Measured against synthetic inputs, it
caught `${DB_PASSWORD:-localdev}` and missed every one of these:

    ${DB_PASSWORD:-hunter2}
    ${JWT_SECRET:-a8f3c9d1e7b2}
    BINANCE_API_SECRET: "sk_live_abc123def456"
    POSTGRES_PASSWORD: SuperSecret99

A denylist of guessable words is the wrong shape, because the danger is
not that a default is *weak* — it is that a default *exists*. This
programme's own principle is *missing secret → 503, never open*; any
default defeats it, and a strong-looking one defeats it while looking
responsible.

The docstring also advertised three scans and implemented two: "hardcoded
passwords/tokens in environment blocks" had no implementing pattern. That
is an architecture gate silently omitting part of itself, inside the gate
that guards secrets. It is implemented here.

**What the repository actually looked like** when this was rewritten: 111
`${VAR:-default}` occurrences, 26 with secret-shaped names, of which
**22 already defaulted to empty**. The convention was right almost
everywhere and nothing enforced it. Exactly one carried a real value.

Exit 0 = clean.  Exit 1 = violations found, or an input is missing.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.security.gate_inputs import inspected, require  # noqa: E402

COMPOSE_FILES = (
    "docker-compose.full.yml",
    "docker-compose.minimal.yml",
    "docker-compose.sovereign.yml",
)

# Matched on whole underscore-separated words, never as substrings.
# `HUGGINGFACE_TOKENIZER` contains "TOKEN" and is a model name; flagging
# it would be a false positive, and a survey with false positives invites
# someone to "fix" working configuration.
SECRET_WORDS = frozenset({
    "SECRET", "SECRETS", "PASSWORD", "PASSWD", "TOKEN", "TOKENS", "KEY",
    "KEYS", "CREDENTIAL", "CREDENTIALS", "PASS", "HMAC", "SALT",
    "APIKEY", "PRIVATEKEY", "AUTHKEY",
})

# A name ending in one of these refers to *where a secret lives*, not to
# the secret. `/run/secrets/hmac_secret` is a path and belongs in the file.
LOCATION_SUFFIXES = ("_PATH", "_DIR", "_FILE", "_URL", "_LOCATION")

# Config that merely mentions a secret. Switches, not values.
CONFIG_WORDS = frozenset({
    "ALLOW", "ENABLE", "ENABLED", "DISABLE", "REQUIRE", "REQUIRED",
    "ROTATE", "ROTATION", "MODE", "TTL", "EXPIRY", "ALGO", "ALGORITHM",
    "HEADER", "PREFIX",
})

# Encoded exceptions, per (service, key), each with a stated reason, and
# printed on every run. An exception nobody can see is debt.
DECLARED_NON_SECRETS: Dict[Tuple[str, str], str] = {
    ("memu-graph", "LLM_API_KEY"):
        "Ollama runs locally and needs no key, but the OpenAI-compatible "
        "client requires a non-empty string. The value is a sentence "
        "saying so, not a credential.",
    ("agentic", "KAI_SERVICE_KEY_ID"):
        "A key IDENTIFIER, not key material. It travels in the "
        "X-Kai-Signature header of every signed request and is listed in "
        "the receiver's key map, so it is public by construction — the "
        "receiver looks the identity up BY it. The secret is the private "
        "key, which is mounted from a file and never appears here. "
        "Hiding a public identifier would make rotation unreadable "
        "without hiding anything.",
}

BANNED = (
    (re.compile(r"HMAC_ALLOW_DEV_SECRET", re.I), re.compile(r"true", re.I),
     "HMAC_ALLOW_DEV_SECRET must not be true in a deployment definition"),
)

_INTERPOLATION = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _words(name: str) -> List[str]:
    return [w for w in re.split(r"[_\-]", name.upper()) if w]


def is_secret_name(name: str) -> bool:
    """True when this names a secret, rather than a switch or a location."""
    if name.upper().endswith(LOCATION_SUFFIXES):
        return False
    words = set(_words(name))
    if not words & SECRET_WORDS:
        return False
    return not (words & CONFIG_WORDS)


def _environment(cfg: dict) -> Dict[str, str]:
    env = (cfg or {}).get("environment") or {}
    if isinstance(env, list):
        pairs = (e.split("=", 1) for e in env if isinstance(e, str) and "=" in e)
        return dict(pairs)
    return {str(k): "" if v is None else str(v) for k, v in env.items()}


def judge(key: str, value: str) -> str:
    """Return "" when acceptable, else the reason it is not.

    Both the key *and* any variable it interpolates are considered. The
    one dangerous default in this repository hid under a non-secret key::

        GATE_SESSION_ID: "${CAMERA_GATE_TOKEN:-camera-gate-token-1}"

    Checking only the key would have missed the finding that motivated
    this rewrite.
    """
    refs = _INTERPOLATION.findall(value)
    if not (is_secret_name(key) or any(is_secret_name(v) for v, _ in refs)):
        return ""

    for var, default in refs:
        if default:
            return (f"'{var}' defaults to {default!r} — a secret with a "
                    f"default is the value it falls back to")

    if refs or not value:
        return ""
    if value.startswith(("/", "./")):
        return ""              # a path to where the secret lives
    return (f"literal value {value[:24]!r} — a secret in a file anyone can "
            f"read is not a secret")


def check_file(path: Path) -> Tuple[List[str], List[str], int]:
    """Return (violations, declared_exceptions, values_examined)."""
    violations: List[str] = []
    declared: List[str] = []
    examined = 0
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return [f"{path.name}: failed to parse: {exc}"], [], 0

    for svc, cfg in sorted((data.get("services") or {}).items()):
        for key, value in sorted(_environment(cfg).items()):
            examined += 1

            for name_pat, value_pat, message in BANNED:
                if name_pat.search(key) and value_pat.search(value):
                    violations.append(f"{path.name}: {svc}.{key}: {message}")

            reason = judge(key, value)
            if not reason:
                continue
            if (svc, key) in DECLARED_NON_SECRETS:
                declared.append(f"{path.name}: {svc}.{key} — "
                                f"{DECLARED_NON_SECRETS[(svc, key)]}")
                continue
            violations.append(f"{path.name}: {svc}.{key}: {reason}")

    return violations, declared, examined


def main() -> int:
    paths = require(COMPOSE_FILES)
    violations: List[str] = []
    declared: List[str] = []
    examined = 0
    for path in paths:
        found, exempt, count = check_file(path)
        violations += found
        declared += exempt
        examined += count

    print(inspected(examined, "environment values",
                    f"across {len(paths)} compose files"))

    if declared:
        print(f"\n  Declared non-secrets ({len(declared)}) — encoded in the "
              f"gate, not in someone's head:")
        for line in declared:
            print(f"    ~ {line}")

    if violations:
        print(f"\nFAIL: {len(violations)} secret(s) carrying a value:\n")
        for v in violations:
            print(f"  - {v}")
        print("\n  A secret may be referenced (${VAR}) or explicitly empty")
        print("  (${VAR:-}). It must never carry a value here.")
        return 1

    print("\nPASS: every secret is referenced, none is valued.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
