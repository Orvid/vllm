# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Round-trip tests for carrying a per-request ``TreeDraft`` beside draft tokens.

The df2t verify step (Task 6) needs the tree structure one engine step after it
is proposed. It rides the same hop as ``DraftTokenIds`` (worker -> scheduler ->
scheduled spec tokens) as plain python lists, so these tests pin the
list-ification and the set -> get -> recover seam.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.df2t.tree_metadata import (
    TreeDraft,
    build_tree_naive,
)
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler


def _tree(num_nodes: int = 4) -> TreeDraft:
    """A TreeDraft in real df2t shape: 1D int64 tensors plus num_nodes."""
    torch.manual_seed(0)
    scores = torch.randn(4, 4, 4)
    cand = torch.arange(4, dtype=torch.long).repeat(4, 1)
    tree = build_tree_naive(scores, cand, num_nodes, L=4)
    tree.num_nodes = torch.tensor([num_nodes], dtype=torch.int32)
    return tree


def _tree_on(device: str) -> TreeDraft:
    tree = _tree()
    return TreeDraft(
        token_ids=tree.token_ids.to(device),
        depths=tree.depths.to(device),
        parents=tree.parents.to(device),
        arena=tree.arena.to(device),
        words=tree.words.to(device),
        num_nodes=tree.num_nodes.to(device),
    )


def _assert_tree_equal(a: TreeDraft, b: TreeDraft) -> None:
    assert torch.equal(a.token_ids, b.token_ids)
    assert torch.equal(a.depths, b.depths)
    assert torch.equal(a.parents, b.parents)
    assert torch.equal(a.arena, b.arena)
    assert torch.equal(a.words, b.words)
    assert torch.equal(a.num_nodes, b.num_nodes)


def test_tree_draft_lists_roundtrip():
    tree = _tree()
    back = TreeDraft.from_lists(tree.to_lists(), device="cpu")
    _assert_tree_equal(back, tree)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tree_draft_carry_through_handler():
    h = DraftTokensHandler()
    tree = _tree_on("cuda")
    batch = SimpleNamespace(req_ids=["0"], has_structured_output_reqs=False)
    h.set_draft_tokens(
        batch, torch.zeros(1, 4, device="cuda", dtype=torch.long), trees=[tree]
    )

    res = h.get_draft_tokens()
    assert res is not None
    assert res.req_ids == ["0"]
    assert res.spec_tree_draft is not None
    (stashed,) = res.spec_tree_draft
    back = TreeDraft.from_lists(stashed, device="cuda")
    _assert_tree_equal(back, tree)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_no_tree_path_returns_placeholders_unchanged():
    h = DraftTokensHandler()
    batch = SimpleNamespace(req_ids=["0", "1"], has_structured_output_reqs=False)
    h.set_draft_tokens(batch, torch.zeros(2, 3, device="cuda", dtype=torch.long))

    res = h.get_draft_tokens()
    assert res is not None
    assert res.req_ids == ["0", "1"]
    assert res.draft_token_ids == [[-1, -1, -1], [-1, -1, -1]]
    assert res.spec_tree_draft is None
