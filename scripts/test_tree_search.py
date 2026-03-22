"""Tests for HP4 CoT Tree Search + Conviction Pruning."""
import os
import sys
import asyncio
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "langgraph"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tree_search import tree_search, Branch, TreeSearchResult, _generate_variations, _branch_id


def _mock_build_plan(prompt, specialist, chunks):
    """Deterministic plan builder for testing."""
    return {
        "summary": f"Plan for: {prompt[:30]}",
        "steps": [{"action": "test"}],
        "specialist": specialist,
    }


def _mock_score_fn(prompt, plan, chunks, rethink_count):
    """Score that improves with depth/variation."""
    base = 5.0
    # Variations with "risks" keyword score higher
    if "risk" in prompt.lower():
        base += 2.0
    # Refinement prompts score higher
    if "Improve" in prompt:
        base += 2.0
    # More chunks = more context = higher score
    base += min(len(chunks) * 0.1, 1.0)
    return min(base, 10.0)


def _mock_high_score(prompt, plan, chunks, rethink_count):
    """Always returns high conviction."""
    return 9.5


def _mock_low_score(prompt, plan, chunks, rethink_count):
    """Always returns low conviction."""
    return 2.0


class TestBranch(unittest.TestCase):
    def test_branch_creation(self):
        b = Branch(id="b1", plan={"s": "test"}, prompt="hello", conviction=7.5, depth=0)
        self.assertEqual(b.id, "b1")
        self.assertEqual(b.conviction, 7.5)
        self.assertFalse(b.pruned)

    def test_branch_id_deterministic(self):
        id1 = _branch_id("test prompt", 0)
        id2 = _branch_id("test prompt", 0)
        self.assertEqual(id1, id2)
        id3 = _branch_id("test prompt", 1)
        self.assertNotEqual(id1, id3)


class TestVariations(unittest.TestCase):
    def test_generates_correct_count(self):
        v = _generate_variations("hello", 3)
        self.assertEqual(len(v), 3)

    def test_baseline_is_unmodified(self):
        v = _generate_variations("hello", 1)
        self.assertEqual(v[0], "hello")

    def test_variations_differ(self):
        v = _generate_variations("hello", 4)
        self.assertEqual(len(set(v)), 4)


class TestTreeSearch(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_finds_best_branch(self):
        result = self._run(tree_search(
            "analyse the risks of this approach",
            "Ollama", [{"content": "chunk1"}],
            _mock_build_plan, _mock_score_fn,
            n_branches=3, max_depth=2, prune_threshold=4.0, min_conviction=8.0,
        ))
        self.assertIsInstance(result, TreeSearchResult)
        self.assertGreater(result.best_branch.conviction, 0)
        self.assertGreater(result.total_branches, 1)

    def test_early_exit_on_high_conviction(self):
        result = self._run(tree_search(
            "hello", "Ollama", [],
            _mock_build_plan, _mock_high_score,
            n_branches=3, max_depth=2, min_conviction=8.0,
        ))
        # Should exit on first branch since score is 9.5
        self.assertEqual(result.total_branches, 1)
        self.assertGreaterEqual(result.best_branch.conviction, 9.0)

    def test_prune_threshold_works(self):
        result = self._run(tree_search(
            "simple query", "Ollama", [],
            _mock_build_plan, _mock_low_score,
            n_branches=3, max_depth=2, prune_threshold=5.0, min_conviction=8.0,
        ))
        self.assertGreater(result.pruned_branches, 0)

    def test_single_branch_single_depth(self):
        result = self._run(tree_search(
            "hello", "Ollama", [],
            _mock_build_plan, _mock_score_fn,
            n_branches=1, max_depth=1,
        ))
        self.assertEqual(result.total_branches, 1)
        self.assertEqual(result.max_depth, 0)

    def test_search_time_tracked(self):
        result = self._run(tree_search(
            "test", "Ollama", [],
            _mock_build_plan, _mock_score_fn,
            n_branches=2, max_depth=1,
        ))
        self.assertGreaterEqual(result.search_time_ms, 0)

    def test_improvement_calculated(self):
        result = self._run(tree_search(
            "analyse risks carefully", "Ollama", [{"content": "c1"}, {"content": "c2"}],
            _mock_build_plan, _mock_score_fn,
            n_branches=4, max_depth=1, prune_threshold=1.0,
        ))
        self.assertEqual(len(result.all_scores), 4)
        self.assertIsInstance(result.improvement, float)

    def test_clamped_parameters(self):
        """n_branches and max_depth are clamped to sane ranges."""
        result = self._run(tree_search(
            "test", "Ollama", [],
            _mock_build_plan, _mock_score_fn,
            n_branches=100, max_depth=100,  # should be clamped to 8 and 4
        ))
        self.assertLessEqual(result.total_branches, 8 * 4 + 8)

    def test_with_fetch_chunks(self):
        async def mock_fetch(prompt):
            return [{"content": "extra context"}]

        result = self._run(tree_search(
            "test with fetch", "Ollama", [],
            _mock_build_plan, _mock_score_fn,
            fetch_chunks_fn=mock_fetch,
            n_branches=2, max_depth=2,
        ))
        self.assertGreater(result.total_branches, 1)


class TestTreeSearchResult(unittest.TestCase):
    def test_improvement_empty_scores(self):
        r = TreeSearchResult(
            best_branch=Branch(id="b1", plan={}, prompt="", conviction=5.0),
            total_branches=1, pruned_branches=0, max_depth=0,
            search_time_ms=1.0, all_scores=[],
        )
        self.assertEqual(r.improvement, 0.0)


if __name__ == "__main__":
    unittest.main()
