# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the worker-side df2t async tree-carry fail-closed guard.

``_validate_df2t_async_tree_carry`` refuses to stage a df2t batch whose
active scheduling mode dropped the stashed tree (async without the carry):
df2t would otherwise silently verify as a chain. A tree-less step with only
``-1`` padding draft placeholders is legitimate.
"""

import pytest

from vllm.v1.outputs import StashedTree
from vllm.v1.worker.gpu.model_runner import _validate_df2t_async_tree_carry


def _carried() -> dict[str, StashedTree]:
    tree = StashedTree(
        token_ids=[7, 8, 9],
        depths=[1, 2, 2],
        parents=[-1, 0, 1],
        arena=[0, 1, 2],
        words=[1, 3, 7],
        num_nodes=3,
    )
    return {"req0": tree}


def test_real_drafts_without_tree_raise():
    with pytest.raises(RuntimeError, match="dropped the tree"):
        _validate_df2t_async_tree_carry({"req0": [7, 8, 9]}, {})


def test_real_drafts_with_carried_tree_pass():
    _validate_df2t_async_tree_carry({"req0": [7, 8, 9]}, _carried())


def test_padding_only_drafts_without_tree_pass():
    # A first decode (or a padding step) schedules -1 placeholders with no
    # tree; that is legitimate, not a dropped carry.
    _validate_df2t_async_tree_carry({"req0": [-1, -1, -1]}, {})


def test_real_drafts_missing_tree_raise_even_in_mixed_batch():
    tree = _carried()["req0"]
    with pytest.raises(RuntimeError, match="dropped the tree"):
        _validate_df2t_async_tree_carry(
            {"req0": [-1, -1, -1], "req1": [5, 6]}, {"req0": tree}
        )


def test_mixed_batch_with_all_real_drafts_carried_passes():
    tree = _carried()["req0"]
    _validate_df2t_async_tree_carry(
        {"req0": [-1, -1, -1], "req1": [7, 8, 9]}, {"req1": tree}
    )


def test_df2t_dispatch_guard_drafts_without_trees_raise():
    """A df2t decode batch scheduling drafts but carrying no stashed tree must
    raise at the sampler dispatch (the sync-path safety net): df2t never
    populates draft_logits, so a chain rejection at T>0 would be rejection
    against a meaningless distribution."""
    from vllm.v1.worker.gpu.model_runner import _df2t_ensure_trees_staged

    with pytest.raises(RuntimeError, match="stashed tree"):
        _df2t_ensure_trees_staged(use_df2t=True, num_draft_tokens=4, trees_staged=False)


def test_df2t_dispatch_guard_trees_staged_passes():
    from vllm.v1.worker.gpu.model_runner import _df2t_ensure_trees_staged

    _df2t_ensure_trees_staged(use_df2t=True, num_draft_tokens=4, trees_staged=True)
    # No drafts (the treeless/prefill path) never triggers the guard.
    _df2t_ensure_trees_staged(use_df2t=True, num_draft_tokens=0, trees_staged=False)


def test_df2t_dispatch_guard_only_fires_for_df2t():
    from vllm.v1.worker.gpu.model_runner import _df2t_ensure_trees_staged

    # Chain methods (eagle/dflash) reject without draft_logits by design; the
    # df2t-only guard must not constrain them.
    _df2t_ensure_trees_staged(use_df2t=False, num_draft_tokens=4, trees_staged=False)


def test_df2t_dispatch_guard_silent_during_kernel_warmup():
    """The kernel-warmup mock decode steps schedule draft tokens without any
    stashed tree, so the guard must stay silent while ``in_kernel_warmup`` is
    set (their sampled output is discarded, and the tree was never expected)."""
    from vllm.v1.worker.gpu.model_runner import _df2t_ensure_trees_staged

    _df2t_ensure_trees_staged(
        use_df2t=True, num_draft_tokens=4, trees_staged=False, in_kernel_warmup=True
    )
    with pytest.raises(RuntimeError, match="stashed tree"):
        _df2t_ensure_trees_staged(
            use_df2t=True, num_draft_tokens=4, trees_staged=False
        )
