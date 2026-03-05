"""HP4 — CoT Tree Search with Conviction Pruning.

Generates multiple reasoning branches, scores each via conviction,
prunes low-conviction paths, and returns the best surviving branch.
Designed to replace the linear rethink loop in /run.

Usage:
    from tree_search import tree_search
    best = await tree_search(user_input, specialist, chunk_dicts, ...)
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Branch:
    """A single reasoning path through the tree."""
    id: str
    plan: Dict[str, Any]
    prompt: str
    conviction: float = 0.0
    depth: int = 0
    pruned: bool = False
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TreeSearchResult:
    """Output of a tree search run."""
    best_branch: Branch
    total_branches: int
    pruned_branches: int
    max_depth: int
    search_time_ms: float
    all_scores: List[float] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        """Score improvement from worst to best branch."""
        if not self.all_scores:
            return 0.0
        return max(self.all_scores) - min(self.all_scores)


# ── Prompt variation strategies ──────────────────────────────────────

_VARIATION_PROMPTS = [
    "",  # baseline — no modification
    "\n\nApproach this step-by-step, considering edge cases first.",
    "\n\nThink about what could go wrong. Start with the risks.",
    "\n\nWhat would an expert in this domain prioritise?",
]


def _generate_variations(base_prompt: str, n_branches: int) -> List[str]:
    """Create N prompt variations for branching."""
    variations = []
    for i in range(n_branches):
        suffix = _VARIATION_PROMPTS[i % len(_VARIATION_PROMPTS)]
        variations.append(base_prompt + suffix)
    return variations


def _branch_id(prompt: str, depth: int) -> str:
    """Deterministic branch ID from prompt content."""
    h = hashlib.sha256(f"{prompt}:{depth}".encode()).hexdigest()[:12]
    return f"branch-{depth}-{h}"


async def tree_search(
    user_input: str,
    specialist: str,
    chunk_dicts: List[Dict[str, Any]],
    build_plan_fn: Callable,
    score_fn: Callable,
    fetch_chunks_fn: Optional[Callable] = None,
    n_branches: int = 3,
    max_depth: int = 2,
    prune_threshold: float = 5.0,
    min_conviction: float = 8.0,
) -> TreeSearchResult:
    """Run tree search over reasoning paths.

    Args:
        user_input: Original user query.
        specialist: Selected specialist/model name.
        chunk_dicts: Memory context chunks.
        build_plan_fn: Callable(prompt, specialist, chunks) -> plan dict.
        score_fn: Callable(prompt, plan, chunks, rethink_count) -> float.
        fetch_chunks_fn: Optional async callable(prompt) -> extra chunks.
        n_branches: Number of branches per level.
        max_depth: Maximum tree depth (1 = single level, 2 = one refinement).
        prune_threshold: Conviction below this = pruned.
        min_conviction: Target conviction (early exit if reached).

    Returns:
        TreeSearchResult with the best branch and search metadata.
    """
    start = time.monotonic()

    # Clamp parameters to sane ranges
    n_branches = max(1, min(n_branches, 8))
    max_depth = max(1, min(max_depth, 4))

    all_branches: List[Branch] = []
    active_prompts = _generate_variations(user_input, n_branches)

    for depth in range(max_depth):
        new_branches: List[Branch] = []

        for prompt in active_prompts:
            plan = build_plan_fn(prompt, specialist, chunk_dicts)
            conv = score_fn(prompt, plan, chunk_dicts, depth)
            branch = Branch(
                id=_branch_id(prompt, depth),
                plan=plan,
                prompt=prompt,
                conviction=conv,
                depth=depth,
            )
            new_branches.append(branch)
            all_branches.append(branch)

            # Early exit if we've hit target conviction
            if conv >= min_conviction:
                elapsed = (time.monotonic() - start) * 1000
                return TreeSearchResult(
                    best_branch=branch,
                    total_branches=len(all_branches),
                    pruned_branches=sum(1 for b in all_branches if b.pruned),
                    max_depth=depth,
                    search_time_ms=round(elapsed, 1),
                    all_scores=[b.conviction for b in all_branches],
                )

        # Prune low-conviction branches
        for b in new_branches:
            if b.conviction < prune_threshold:
                b.pruned = True

        survivors = [b for b in new_branches if not b.pruned]
        if not survivors:
            # All pruned — keep the best one anyway
            survivors = sorted(new_branches, key=lambda b: b.conviction, reverse=True)[:1]

        # For next depth: refine surviving prompts with feedback
        if depth < max_depth - 1:
            next_prompts = []
            for b in survivors:
                refined = f"{b.prompt}\n\nPrevious attempt scored {b.conviction:.1f}/10. Improve the reasoning."
                next_prompts.append(refined)

                # Optionally fetch more context for refinement
                if fetch_chunks_fn:
                    try:
                        extra = await fetch_chunks_fn(b.prompt)
                        if extra:
                            chunk_dicts = chunk_dicts + [
                                {"content": c} if isinstance(c, str) else c
                                for c in extra
                            ]
                    except Exception:
                        pass

            active_prompts = next_prompts

    # Select the best branch across all depths
    best = max(all_branches, key=lambda b: b.conviction)
    elapsed = (time.monotonic() - start) * 1000

    return TreeSearchResult(
        best_branch=best,
        total_branches=len(all_branches),
        pruned_branches=sum(1 for b in all_branches if b.pruned),
        max_depth=max(b.depth for b in all_branches),
        search_time_ms=round(elapsed, 1),
        all_scores=[b.conviction for b in all_branches],
    )
