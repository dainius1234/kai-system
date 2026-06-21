"""CI-only sanity check for the GitHub Models test backend (D36).

Skips everywhere except a real GitHub Actions run with the `models: read`
workflow permission — GITHUB_TOKEN is absent locally and on PRs from forks,
and models.github.ai isn't reachable from a typical dev sandbox's network
policy either. This is a smoke test for the CI backend itself (proving the
endpoint, auth, and model ID actually work end-to-end), not a check on any
production code path.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from github_models_client import query, is_available  # noqa: E402


class TestGitHubModelsBackend(unittest.TestCase):
    @unittest.skipUnless(
        is_available(),
        "GITHUB_TOKEN not set — only runs in GitHub Actions with models: read",
    )
    def test_live_query_returns_real_response(self):
        result = query("Reply with exactly the word: pong")
        self.assertEqual(result.source, "live", f"expected live response, got: {result}")
        self.assertTrue(result.text.strip())


if __name__ == "__main__":
    unittest.main()
