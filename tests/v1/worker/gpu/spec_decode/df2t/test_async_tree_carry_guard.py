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
