"""Install `sys.modules` stubs for the length of an import, then put it back.

`sys.modules` is process-global. A test module that stubs `common.runtime`
so it can import a service without the real dependency, and then leaves
the stub in place, has not stubbed a module — it has *edited the
interpreter* for everything collected after it.

That is not hypothetical. On 2026-08-04 the repo-wide pytest in
`python-app.yml` was aborting at collection with six errors, so **not one
of its 4,187 tests ran**, and had been doing so on every run since at
least 27 July. Five of the six errors came from a single line:

    sys.modules["common"] = types.ModuleType("common")     # test_cortex.py

`test_cortex` sorts before `test_erasure`, `test_error_codes`,
`test_feature_flags`, `test_flags_enabled` and `test_migration`, and all
five import `common.<something>` at module scope. They failed with
``No module named 'common.world_state'; 'common' is not a package``, and
every one of them passed when run alone. The sixth error was the same
shape via `common.runtime` left as a `MagicMock`.

The failure mode is worth naming because it is the opposite of the ones
this programme has been chasing. A self-consuming guard stops checking
and *looks* like a pass. This stops checking and looks like a *failure* —
but a failure in a file that is not the culprit, on a workflow that does
not run on the branch anyone was working on. Nobody was lying to; nobody
was looking.

So: stubs are scoped, and the scope is enforced by `finally`.

    with stubbed({"common.runtime": fake, "common": fake_pkg}):
        spec.loader.exec_module(mod)

The module under test keeps its own reference to the stub — that is what
makes this safe. Only `sys.modules` is restored, and it is restored
exactly: names that were absent before are removed rather than set to
`None`, because a `None` entry in `sys.modules` is a cached *import
failure* and would break the next importer just as thoroughly.

`scripts/security/check_test_isolation.py` fails the build on any test
module that leaves `sys.modules` altered, so a new one cannot reintroduce
this without saying so.
"""
from __future__ import annotations

import contextlib
import sys
from typing import Any, Dict, Iterable, Iterator, Mapping
from unittest.mock import MagicMock

__all__ = ["stubbed", "absent_stubs", "AGENTIC_HEAVY_DEPS"]

# Third-party and agentic-local modules that `agentic/app.py` and its
# neighbours import but that are not installed in the test environment.
# This list was copy-pasted verbatim into five suites (test_p16..test_p20)
# before it was declared here once. A shared list is not tidiness: when
# `agentic` gained a dependency, five files had to be found and edited,
# and the one that was missed failed in a way that looked like a bug in
# whatever ran after it.
AGENTIC_HEAVY_DEPS = (
    "sentence_transformers", "psutil", "redis", "redis.asyncio",
    "psycopg2", "psycopg2.extras", "psycopg2.pool", "lakefs_client",
    "kai_config", "conviction", "router", "planner", "adversary",
    "security_audit", "tree_search", "priority_queue", "model_selector",
    "aioredis",
)


def fake_redis() -> Any:
    """A `redis` module stub — the same one four suites each defined.

    `redis` is an optional dependency of `agentic/kai_config.py`. Four
    suites carried a byte-identical `_FakeRedis` to get past that import.
    """
    import types

    module = types.ModuleType("redis")

    class _FakeRedis:
        @classmethod
        def from_url(cls, *a, **kw):
            return cls()

        def ping(self):
            return True

    module.Redis = _FakeRedis
    return module


def absent_stubs(names: Iterable[str], factory=MagicMock) -> Dict[str, Any]:
    """A stub per name that is not already importable.

    Preserves the `setdefault` semantics the call sites had: a dependency
    that really is installed is used for real, and only what is missing is
    faked. What changes is that the result is handed to `stubbed()` and so
    has an end.
    """
    return {name: factory() for name in names if name not in sys.modules}


@contextlib.contextmanager
def stubbed(modules: Mapping[str, Any]) -> Iterator[None]:
    """Install `modules` into `sys.modules`, restoring it exactly on exit.

    Order matters on the way in: a parent package must be present before
    a submodule is imported, and callers already pass both. Order does
    not matter on the way out, because restoration is by name.
    """
    saved: Dict[str, Any] = {}
    absent = []
    for name in modules:
        if name in sys.modules:
            saved[name] = sys.modules[name]
        else:
            absent.append(name)

    sys.modules.update(modules)
    try:
        yield
    finally:
        for name in absent:
            # `del`, not `= None`. A None entry is a cached import
            # failure, which fails the next importer instead of letting
            # it find the real module.
            sys.modules.pop(name, None)
        sys.modules.update(saved)
