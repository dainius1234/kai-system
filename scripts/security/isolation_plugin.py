"""A pytest plugin that watches `sys.modules` across file boundaries.

The question "does this test file change the interpreter for the files
after it?" is decided by pytest, at run time, in one process. Nothing
static answers it: the first version of this detector imported each file
in a subprocess and reported seven offenders, and it was wrong — not
because the seven were innocent, but because the worst one
(`test_cognitive_mechanisms.py`, which replaced `fastapi`) does its
damage from `setup_method`, long after import. An import-time probe
cannot see a run-time edit.

So this hooks the thing that actually decides: `pytest_runtest_protocol`,
which fires per test. It records the `sys.modules` state when a new file
starts and compares it when that file's last test has finished.

Reported per file:

  replaced   a name that pointed at a real module and now points at
             something else. `fastapi` becoming a two-attribute stub is
             this, and it is the most damaging: the next file to
             `from fastapi import Depends` fails, and the failure names
             *that* file.
  added      a stub left behind under a name that was previously free.
             Harmless until something else wants the name for real.
  env_set    an environment variable this file introduced. `os.environ
             .setdefault("VECTOR_STORE", "memory")` at module scope is
             the same defect as `sys.modules.setdefault`: the first file
             to run wins and every file after it silently inherits a
             value it did not choose.
  env_changed  an existing variable given a different value.
  path_added   a directory added to `sys.path`, which changes what a
             bare `import name` resolves to for everything after.

Not reported: names that merely became *imported*. Importing a module is
what tests are for.
"""
from __future__ import annotations

import json
import os
import sys
import types
from typing import Any, Dict, List, Optional

# stdlib deprecation shims that appear and vanish on their own.
_IGNORED = frozenset({"typing.io", "typing.re"})


def _fingerprint(module: Any) -> str:
    """Enough of a module's identity to tell a swap from a no-op."""
    if not isinstance(module, types.ModuleType):
        return f"mock:{type(module).__name__}"
    origin = getattr(getattr(module, "__spec__", None), "origin", None)
    return f"mod:{origin or getattr(module, '__file__', None) or 'none'}"


class IsolationPlugin:
    def __init__(self) -> None:
        self._current: Optional[str] = None
        self._before: Dict[str, str] = {}
        self._env: Dict[str, str] = {}
        self._path: List[str] = []
        self.findings: Dict[str, Dict[str, List[str]]] = {}

    def _snapshot(self) -> Dict[str, str]:
        return {name: _fingerprint(mod) for name, mod in list(sys.modules.items())}

    def _close(self) -> None:
        if self._current is None:
            return
        after = self._snapshot()
        env_now = dict(os.environ)
        env_set = sorted(k for k in env_now if k not in self._env)
        env_changed = sorted(f"{k} ({self._env[k]!r} -> {env_now[k]!r})"
                             for k in env_now
                             if k in self._env and self._env[k] != env_now[k])
        path_added = [p for p in sys.path if p not in self._path]
        replaced, added = [], []
        for name, mark in sorted(after.items()):
            if name in _IGNORED:
                continue
            was = self._before.get(name)
            if was is None:
                if not mark.startswith("mod:") or mark == "mod:none":
                    added.append(name)
            elif was != mark and was.startswith("mod:") and was != "mod:none":
                replaced.append(f"{name} ({was} -> {mark})")
        if replaced or added or env_set or env_changed or path_added:
            self.findings[self._current] = {
                "replaced": replaced, "added": added,
                "env_set": env_set, "env_changed": env_changed,
                "path_added": path_added,
            }
        self._current = None

    def pytest_runtest_protocol(self, item, nextitem):  # noqa: D401
        path = str(getattr(item, "path", item.fspath))
        if path != self._current:
            self._close()
            self._current = path
            self._before = self._snapshot()
            self._env = dict(os.environ)
            self._path = list(sys.path)
        return None

    def pytest_sessionfinish(self, session, exitstatus):  # noqa: D401
        self._close()
        out = os.environ.get("KAI_ISOLATION_REPORT")
        if out:
            with open(out, "w", encoding="utf-8") as handle:
                json.dump(self.findings, handle, indent=2, sort_keys=True)


def pytest_configure(config):
    config.pluginmanager.register(IsolationPlugin(), "kai-isolation")
