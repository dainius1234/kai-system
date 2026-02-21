from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'memu-core'))


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


memu = load('memu-core/app.py', 'memu_app')
update = memu.MemoryUpdate(timestamp='2026-01-01T00:00:00', event_type='evt', result_raw='x', user_id='keeper', relevance=0.1, pin=True)
result = asyncio.run(memu.memorize_event(update))
assert result['status'] == 'appended'
latest = memu.store.search(top_k=1)[0]
assert latest.pinned is True
assert latest.relevance == 1.0, 'keeper pinned records must force relevance=1.0'

before = memu.store.count()
memu.store.compress()
after = memu.store.count()
assert after >= 1 and after == before, 'keeper-pinned vectors must not be deleted'

runtime = load('common/runtime.py', 'runtime_mod')
stream = runtime.AuditStream('test-service', redis_url='')
assert stream.verify_or_halt() is True

executor = load('executor/app.py', 'executor_app')
assert callable(executor.alive)
print('hardening smoke checks passed')
