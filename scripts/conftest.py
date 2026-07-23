"""scripts/ conftest — stubs missing optional deps so all test files collect.

redis==5.0.8 is a declared service dep (dashboard, agentic, memu-core) but
may not be installed in offline/CI environments that skip service requirements.
Tests that exercise actual redis connectivity are skipped at runtime; the stub
here prevents the entire test collection from failing on import.

The from_url() stub raises ConnectionError on ping() so that services that test
their redis-unavailable fallback path (e.g. build_saver() in kai_config.py)
still trigger their fallback to the in-memory/spool backend.
"""

import os
import sys
from unittest.mock import MagicMock

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
