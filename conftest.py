"""Global test configuration — runs before any test imports."""

import os

# Allow dev HMAC secret in test environment to avoid RuntimeError
# in common.auth._secret() when INTERSERVICE_HMAC_SECRET is not set.
os.environ.setdefault("HMAC_ALLOW_DEV_SECRET", "true")
