"""Runtime chassis tests — C2 (stream heartbeat), C5 (model tags), C9 (warmup).

These tests exercise the three chassis-polish features added in the C2/C5/C9
PR and are designed to run without a live Ollama or GPU.  All external calls
are mocked.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── helpers ──────────────────────────────────────────────────────────

def _run(coro):
    """Run an async coroutine in a new event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _collect_stream(gen):
    """Exhaust an async generator and return a list of yielded items."""
    async def _drain():
        items = []
        async for item in gen:
            items.append(item)
        return items
    return _run(_drain())


# ── C2: Streaming heartbeat / stall detection ────────────────────────

class _SlowLineResponse:
    """Fake httpx streaming response: emits *fast_lines* immediately, then stalls."""

    def __init__(self, fast_lines, stall_seconds=10):
        self._fast_lines = fast_lines
        self._stall_seconds = stall_seconds

    def raise_for_status(self):
        pass

    def aiter_lines(self):
        return self._aiter()

    async def _aiter(self):
        for line in self._fast_lines:
            yield line
        await asyncio.sleep(self._stall_seconds)
        yield "data: [DONE]"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class _FakeHttpxClient:
    def __init__(self, response, *args, **kwargs):
        self._response = response

    def stream(self, method, url, **kwargs):
        return self._response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class TestStreamHeartbeat(unittest.TestCase):
    """C2 — per-token heartbeat / stall detection in LLMRouter.stream()."""

    def setUp(self):
        # Short heartbeat so tests finish fast
        os.environ["STREAM_HEARTBEAT_TIMEOUT"] = "1"

    def tearDown(self):
        os.environ.pop("STREAM_HEARTBEAT_TIMEOUT", None)

    def _make_router(self):
        from common.llm import LLMRouter
        return LLMRouter(backends={"Ollama": "http://ollama:11434"})

    def test_stall_message_emitted_within_deadline(self):
        """Stall detection fires before STREAM_HEARTBEAT_TIMEOUT + 1 s slack."""
        fast_line = json.dumps({"choices": [{"delta": {"content": "hello"}}]})
        response = _SlowLineResponse(
            fast_lines=[f"data: {fast_line}"],
            stall_seconds=10,  # much longer than heartbeat
        )

        router = self._make_router()
        fake_client = _FakeHttpxClient(response)

        deadline = float(os.environ["STREAM_HEARTBEAT_TIMEOUT"]) + 2.0  # 1 s HB + 2 s slack
        start = time.monotonic()

        with patch("httpx.AsyncClient", return_value=fake_client):
            tokens = _collect_stream(
                router.stream("Ollama", [{"role": "user", "content": "hi"}])
            )

        elapsed = time.monotonic() - start

        # Must finish within the deadline
        self.assertLess(elapsed, deadline, f"stream took {elapsed:.2f}s > {deadline}s deadline")

        # First token "hello" should arrive, then the stall message
        self.assertIn("hello", tokens)
        stall_tokens = [t for t in tokens if "stalled" in t]
        self.assertTrue(stall_tokens, f"Expected a stall message token, got: {tokens}")

    def test_no_stall_when_tokens_arrive_quickly(self):
        """No stall message when tokens arrive before the heartbeat window."""
        lines = [
            f"data: {json.dumps({'choices': [{'delta': {'content': 'hi'}}]})}",
            f"data: {json.dumps({'choices': [{'delta': {'content': ' there'}}]})}",
            "data: [DONE]",
        ]

        class _FastResponse:
            def raise_for_status(self): pass

            def aiter_lines(self):
                return self._aiter()

            async def _aiter(self):
                for line in lines:
                    yield line

            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass

        router = self._make_router()
        fake_client = _FakeHttpxClient(_FastResponse())

        with patch("httpx.AsyncClient", return_value=fake_client):
            tokens = _collect_stream(
                router.stream("Ollama", [{"role": "user", "content": "hi"}])
            )

        self.assertIn("hi", tokens)
        self.assertIn(" there", tokens)
        stall_tokens = [t for t in tokens if "stalled" in t]
        self.assertEqual(stall_tokens, [], f"Unexpected stall messages: {stall_tokens}")

    def test_stub_mode_no_heartbeat_needed(self):
        """No backend configured → stub response, no asyncio.wait_for involved."""
        from common.llm import LLMRouter
        # Pass a non-empty dict with a blank URL so the constructor produces a
        # no-backend router (empty dict {} is falsy, falling back to _DEFAULT_URLS)
        router = LLMRouter(backends={"Ollama": ""})
        tokens = _collect_stream(
            router.stream("Ollama", [{"role": "user", "content": "test"}])
        )
        self.assertEqual(len(tokens), 1)
        self.assertIn("stub", tokens[0])


# ── C5: Ollama /api/tags pre-flight ──────────────────────────────────

class TestEnsureModelAvailable(unittest.TestCase):
    """C5 — verify model is loaded via Ollama /api/tags before routing."""

    def setUp(self):
        # Bust the in-process cache before each test
        import common.llm as _llm_mod
        _llm_mod._model_tags_cache = None
        _llm_mod._model_tags_cache_ts = 0.0

    def tearDown(self):
        import common.llm as _llm_mod
        _llm_mod._model_tags_cache = None
        _llm_mod._model_tags_cache_ts = 0.0

    def _fake_tags_client(self, model_names):
        """Build a fake httpx.AsyncClient that returns a /api/tags payload."""
        models_payload = {"models": [{"name": n} for n in model_names]}

        class _FakeResp:
            def raise_for_status(self): pass
            def json(self): return models_payload

        class _FakeClient:
            async def get(self, url): return _FakeResp()
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass

        return _FakeClient()

    def test_model_present_exact_match(self):
        from common.llm import ensure_model_available
        fake = self._fake_tags_client(["qwen2:0.5b", "llama3:8b"])
        with patch("httpx.AsyncClient", return_value=fake):
            result = _run(ensure_model_available("qwen2:0.5b"))
        self.assertTrue(result)

    def test_model_absent_returns_false(self):
        from common.llm import ensure_model_available
        fake = self._fake_tags_client(["llama3:8b"])
        with patch("httpx.AsyncClient", return_value=fake):
            result = _run(ensure_model_available("qwen2:0.5b"))
        self.assertFalse(result)

    def test_prefix_match_returns_true(self):
        """qwen2:0.5b should match qwen2:0.5b-instruct-q4_0."""
        from common.llm import ensure_model_available
        fake = self._fake_tags_client(["qwen2:0.5b-instruct-q4_0"])
        with patch("httpx.AsyncClient", return_value=fake):
            result = _run(ensure_model_available("qwen2:0.5b"))
        self.assertTrue(result)

    def test_ollama_unreachable_returns_true(self):
        """Fail-open: if Ollama is unreachable, don't block routing."""
        from common.llm import ensure_model_available
        import httpx

        class _BrokenClient:
            async def get(self, url): raise httpx.ConnectError("refused")
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass

        with patch("httpx.AsyncClient", return_value=_BrokenClient()):
            result = _run(ensure_model_available("qwen2:0.5b"))
        self.assertTrue(result)

    def test_cache_hit_skips_second_request(self):
        """Second call within TTL must not re-query Ollama."""
        from common.llm import ensure_model_available

        call_count = 0

        class _CountingResp:
            def raise_for_status(self): pass
            def json(self): return {"models": [{"name": "qwen2:0.5b"}]}

        class _CountingClient:
            async def get(self, url):
                nonlocal call_count
                call_count += 1
                return _CountingResp()

            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass

        # Override TTL to something long so the cache doesn't expire
        with patch.dict(os.environ, {"MODEL_TAGS_CACHE_TTL": "60"}):
            with patch("httpx.AsyncClient", return_value=_CountingClient()):
                _run(ensure_model_available("qwen2:0.5b"))  # cache miss → hits Ollama
                _run(ensure_model_available("qwen2:0.5b"))  # cache hit → no HTTP call

        self.assertEqual(call_count, 1, "Expected exactly 1 HTTP call (cache should serve the second)")

    def test_model_fallback_in_live_query(self):
        """_live_query falls back to OLLAMA_MODEL when requested model absent."""
        from common.llm import LLMRouter, _OLLAMA_MODEL
        import common.llm as _llm_mod

        router = LLMRouter(backends={"Ollama": "http://ollama:11434"})

        # Tags say only the default model is available
        tags_payload = {"models": [{"name": _OLLAMA_MODEL}]}

        posted_models = []

        class _FakeTagsResp:
            def raise_for_status(self): pass
            def json(self): return tags_payload

        class _FakeChatResp:
            def raise_for_status(self): pass

            def json(self):
                return {"choices": [{"message": {"content": "ok"}}], "model": _OLLAMA_MODEL, "usage": {}}

        class _FakeClient:
            async def get(self, url): return _FakeTagsResp()

            async def post(self, url, **kwargs):
                posted_models.append(kwargs.get("json", {}).get("model"))
                return _FakeChatResp()

            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass

        # Bust cache
        _llm_mod._model_tags_cache = None
        _llm_mod._model_tags_cache_ts = 0.0

        with patch("httpx.AsyncClient", return_value=_FakeClient()):
            _run(router.query("Ollama", "hello"))

        # The model sent in the POST should be the fallback default
        self.assertEqual(posted_models, [_OLLAMA_MODEL])


# ── C9: Model warm-up / pre-load ────────────────────────────────────

class TestLLMWarmup(unittest.TestCase):
    """C9 — warm-up fires once on startup, respects env flags."""

    def setUp(self):
        # Bust the tags cache before each test
        import common.llm as _llm_mod
        _llm_mod._model_tags_cache = None
        _llm_mod._model_tags_cache_ts = 0.0

    def tearDown(self):
        import common.llm as _llm_mod
        _llm_mod._model_tags_cache = None
        _llm_mod._model_tags_cache_ts = 0.0
        # Clean up env knobs
        for key in ("LLM_WARMUP_ENABLED", "OLLAMA_AUTO_PULL"):
            os.environ.pop(key, None)

    def test_warmup_calls_router_query(self):
        """Warm-up fires a query on the router when model is available."""
        from common.llm import LLMRouter, llm_warmup, LLMResponse

        router = LLMRouter(backends={})  # stub mode — query always returns a stub
        called = []

        async def _fake_query(specialist, prompt, **kwargs):
            called.append({"specialist": specialist, "prompt": prompt})
            return LLMResponse(specialist=specialist, text="ok", latency_ms=1.0, source="stub")

        with patch.dict(os.environ, {"LLM_WARMUP_ENABLED": "true"}):
            with patch.object(router, "query", side_effect=_fake_query):
                # ensure_model_available must return True (fail-open for unreachable)
                with patch("common.llm.ensure_model_available", new=AsyncMock(return_value=True)):
                    _run(llm_warmup(router=router, specialist="Ollama"))

        self.assertEqual(len(called), 1)
        self.assertEqual(called[0]["prompt"], "warmup")

    def test_warmup_disabled_skips_query(self):
        """Setting LLM_WARMUP_ENABLED=false skips the warm-up entirely."""
        from common.llm import LLMRouter, llm_warmup

        router = LLMRouter(backends={})
        called = []

        async def _fake_query(*args, **kwargs):
            called.append(True)

        with patch.dict(os.environ, {"LLM_WARMUP_ENABLED": "false"}):
            with patch.object(router, "query", side_effect=_fake_query):
                _run(llm_warmup(router=router))

        self.assertEqual(called, [], "query should NOT be called when warmup is disabled")

    def test_warmup_skips_when_model_absent_no_autopull(self):
        """Model absent + OLLAMA_AUTO_PULL=false → warm prompt skipped."""
        from common.llm import LLMRouter, llm_warmup

        router = LLMRouter(backends={})
        query_called = []

        async def _fake_query(*args, **kwargs):
            query_called.append(True)

        with patch.dict(os.environ, {"LLM_WARMUP_ENABLED": "true", "OLLAMA_AUTO_PULL": "false"}):
            with patch("common.llm.ensure_model_available", new=AsyncMock(return_value=False)):
                with patch.object(router, "query", side_effect=_fake_query):
                    _run(llm_warmup(router=router, specialist="Ollama"))

        self.assertEqual(query_called, [])

    def test_warmup_pulls_and_queries_when_autopull_enabled(self):
        """Model absent + OLLAMA_AUTO_PULL=true → pull called, then warm prompt."""
        from common.llm import LLMRouter, llm_warmup, LLMResponse

        router = LLMRouter(backends={})
        query_called = []
        pull_called = []

        async def _fake_query(specialist, prompt, **kwargs):
            query_called.append(prompt)
            return LLMResponse(specialist=specialist, text="ok", latency_ms=1.0, source="stub")

        async def _fake_pull(model, base_url):
            pull_called.append(model)

        with patch.dict(os.environ, {"LLM_WARMUP_ENABLED": "true", "OLLAMA_AUTO_PULL": "true"}):
            with patch("common.llm.ensure_model_available", new=AsyncMock(return_value=False)):
                with patch("common.llm._pull_model", side_effect=_fake_pull):
                    with patch.object(router, "query", side_effect=_fake_query):
                        _run(llm_warmup(router=router, specialist="Ollama"))

        self.assertEqual(len(pull_called), 1, "pull should be called once")
        self.assertEqual(query_called, ["warmup"])

    def test_autopull_not_called_when_model_available(self):
        """OLLAMA_AUTO_PULL=true but model is available → no pull."""
        from common.llm import LLMRouter, llm_warmup, LLMResponse

        router = LLMRouter(backends={})
        pull_called = []

        async def _fake_pull(model, base_url):
            pull_called.append(model)

        async def _fake_query(specialist, prompt, **kwargs):
            return LLMResponse(specialist=specialist, text="ok", latency_ms=1.0, source="stub")

        with patch.dict(os.environ, {"LLM_WARMUP_ENABLED": "true", "OLLAMA_AUTO_PULL": "true"}):
            with patch("common.llm.ensure_model_available", new=AsyncMock(return_value=True)):
                with patch("common.llm._pull_model", side_effect=_fake_pull):
                    with patch.object(router, "query", side_effect=_fake_query):
                        _run(llm_warmup(router=router, specialist="Ollama"))

        self.assertEqual(pull_called, [], "pull must NOT be called when model is already available")


if __name__ == "__main__":
    unittest.main()
