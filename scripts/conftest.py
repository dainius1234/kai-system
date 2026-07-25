"""scripts/ conftest — stubs missing optional deps so all test files collect.

redis==5.0.8 is a declared service dep (dashboard, agentic, memu-core) but
may not be installed in offline/CI environments that skip service requirements.
Tests that exercise actual redis connectivity are skipped at runtime; the stub
here prevents the entire test collection from failing on import.

The from_url() stub raises ConnectionError on ping() so that services that test
their redis-unavailable fallback path (e.g. build_saver() in kai_config.py)
still trigger their fallback to the in-memory/spool backend.
"""

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# ── Vault-sync: load service modules under importable aliases ─────────────────
# pytest_configure defined inside a test file is not called as a plugin hook;
# this conftest.py is the correct place for session-wide path/module setup.
_ROOT = Path(__file__).parent.parent
for _sub in ("vault-sync", "common", "memu-core"):
    _p = str(_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

_VAULT_ALIASES = {
    "vault_sync_parser": _ROOT / "vault-sync" / "parser.py",
    "vault_sync_mapper": _ROOT / "vault-sync" / "mapper.py",
    "vault_sync_watcher": _ROOT / "vault-sync" / "watcher.py",
    "vault_sync_app": _ROOT / "vault-sync" / "app.py",
}
for _alias, _src in _VAULT_ALIASES.items():
    if _alias not in sys.modules:
        _spec = importlib.util.spec_from_file_location(_alias, _src)
        if _spec:
            _mod = importlib.util.module_from_spec(_spec)
            sys.modules[_alias] = _mod
            try:
                _spec.loader.exec_module(_mod)
            except Exception:
                pass

# Allow offline tests that don't depend on embedding quality to run without
# sentence-transformers installed (mirrors CI's MEMU_ALLOW_FAKE_EMBEDDINGS).
os.environ.setdefault("MEMU_ALLOW_FAKE_EMBEDDINGS", "true")

if "redis" not in sys.modules:
    _ping_mock = MagicMock(side_effect=ConnectionError("redis stub — no real redis"))
    _client_mock = MagicMock()
    _client_mock.ping = _ping_mock
    _from_url_mock = MagicMock(return_value=_client_mock)

    _redis_stub = MagicMock()
    _redis_stub.from_url = _from_url_mock
    _redis_stub.asyncio = MagicMock()
    _redis_stub.asyncio.from_url = MagicMock(return_value=MagicMock())

    sys.modules["redis"] = _redis_stub
    sys.modules["redis.asyncio"] = _redis_stub.asyncio
