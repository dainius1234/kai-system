from __future__ import annotations

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
record = memu.MemoryRecord(id='r1', timestamp='2026-01-01T00:00:00', event_type='evt', content={'user_id': 'keeper'}, embedding=[0.2] * 8, relevance=0.1, pinned=True)
commit = memu.store.insert(record)
assert memu.store.count() >= 1
memu.store.compress()
assert memu.store.count() >= 1, 'keeper-pinned vectors must not be deleted'
memu.store.revert(commit.commit_id)
assert memu.store.count() >= 1

runtime = load('common/runtime.py', 'runtime_mod')
stream = runtime.AuditStream('test-service', redis_url='')
assert stream.verify_or_halt() is True

executor = load('executor/app.py', 'executor_app')
assert callable(executor.alive)
print('hardening smoke checks passed')
