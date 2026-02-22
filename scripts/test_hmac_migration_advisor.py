#!/usr/bin/env python3
import os
import subprocess
import sys


def run(env):
    base = os.environ.copy()
    base.update(env)
    cp = subprocess.run(
        [sys.executable, 'scripts/hmac_migration_advisor.py'],
        check=True,
        text=True,
        capture_output=True,
        env=base,
    )
    return cp.stdout


def main() -> int:
    out = run({
        'AUTH_SERVICES': '3',
        'AUTH_TEAMS': '1',
        'HMAC_ROTATIONS_PER_QUARTER': '1',
        'HMAC_INCIDENTS_90D': '0',
        'EXTERNAL_VERIFIER_DEPENDENCIES': '0',
        'ZERO_TRUST_TARGET': 'false',
        'AUDITABILITY_SCORE': '0.4',
    })
    assert 'STAY ON HMAC' in out

    out = run({
        'AUTH_SERVICES': '9',
        'AUTH_TEAMS': '3',
        'HMAC_ROTATIONS_PER_QUARTER': '4',
    })
    assert 'MIGRATE NEXT PHASE' in out

    print('test_hmac_migration_advisor: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
