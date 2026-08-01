"""Verify that a legacy path is genuinely closed.

``disable_legacy_path()`` on its own is bookkeeping: it sets a flag and
believes whoever called it.  That is exactly the weakness the roadmap
warns about — "retaining old and new action paths temporarily" is an
explicitly rejected anti-pattern, and a flag is easy to set optimistically.

This module ties the flag to a checkable fact.  Every legacy path in the
catalogue is expressed as a condition that can be evaluated against the
current source tree, so marking a path closed requires it to *be* closed.

Most legacy paths recorded in the audit were the **unauthenticated**
versions of endpoints that still exist. For those, "closed" means the
endpoint now requires authentication — not that the route is gone.
Deleting a route the dashboard depends on would be a regression, not
progress; requiring a capability token is the actual fix.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent.parent


class LegacyVerificationError(Exception):
    pass


def _route_is_authenticated(rel_path: str, method: str, route: str) -> Tuple[bool, str]:
    """Whether a route's decorator declares service authentication."""
    source = REPO / rel_path
    if not source.exists():
        return False, f"{rel_path} not found"

    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return False, f"{rel_path} unparseable: {exc}"

    for node in ast.walk(tree):
        for dec in getattr(node, "decorator_list", []):
            if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)):
                continue
            if dec.func.attr != method or not dec.args:
                continue
            first = dec.args[0]
            if not isinstance(first, ast.Constant) or first.value != route:
                continue
            if "require_service_auth" in ast.unparse(dec):
                return True, f"{method.upper()} {route} requires service auth"
            return False, f"{method.upper()} {route} is unauthenticated"

    return False, f"{method.upper()} {route} not found in {rel_path}"


def _symbol_absent(rel_path: str, symbol: str) -> Tuple[bool, str]:
    """Whether a function/method no longer exists in a module."""
    source = REPO / rel_path
    if not source.exists():
        return True, f"{rel_path} no longer exists"

    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return False, f"{rel_path} unparseable: {exc}"

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == symbol:
                return False, f"{symbol}() still defined in {rel_path}"
    return True, f"{symbol}() removed from {rel_path}"


# actuator identity → check returning (closed, explanation)
LEGACY_CHECKS: Dict[str, Callable[[], Tuple[bool, str]]] = {
    "browser-reader": lambda: _route_is_authenticated(
        "browser-agent/app.py", "post", "/navigate"),
    "browser-actor": lambda: _route_is_authenticated(
        "browser-agent/app.py", "post", "/click"),
    "notify-service": lambda: _route_is_authenticated(
        "output/notify/app.py", "post", "/notify"),
    "monitor-service": lambda: _route_is_authenticated(
        "monitor-service/app.py", "post", "/rules"),
    "checkpoint-manager": lambda: _route_is_authenticated(
        "agentic/app.py", "post", "/checkpoint/{checkpoint_id}/restore"),
    "telegram-bot": lambda: _route_is_authenticated(
        "telegram-bot/app.py", "post", "/alert"),
    "backup-service": lambda: _route_is_authenticated(
        "backup-service/app.py", "post", "/backup"),
    "db-restore": lambda: _route_is_authenticated(
        "backup-service/app.py", "post", "/restore/postgres"),
    "vault-sync": lambda: _route_is_authenticated(
        "vault-sync/app.py", "post", "/export"),
    "executor-shell": lambda: _route_is_authenticated(
        "executor/app.py", "post", "/execute"),
    # auto_trade() was removed outright in the P0 remediation, so for the
    # paper trader "closed" genuinely means the symbol is gone.
    "paper-trader": lambda: _symbol_absent(
        "agentic/strategy_engine.py", "auto_trade"),
}


def verify_legacy_closed(actuator_identity: str) -> Tuple[bool, str]:
    """Whether this actuator's legacy path is genuinely closed.

    An actuator with no recorded legacy path has nothing to close and
    passes trivially.
    """
    check = LEGACY_CHECKS.get(actuator_identity)
    if check is None:
        return True, "no legacy path recorded"
    try:
        return check()
    except Exception as exc:
        return False, f"verification failed: {type(exc).__name__}: {exc}"


def verify_all() -> Dict[str, Tuple[bool, str]]:
    return {name: verify_legacy_closed(name) for name in sorted(LEGACY_CHECKS)}


def open_legacy_paths() -> Dict[str, str]:
    """Actuators whose legacy path is still open, with the reason."""
    return {
        name: reason
        for name, (closed, reason) in verify_all().items()
        if not closed
    }
