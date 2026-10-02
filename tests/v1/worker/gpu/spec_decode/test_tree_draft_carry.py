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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tree_length_mismatch_raises():
    """A trees list whose length differs from the batch request count must
    raise, never silently mismatched to the wrong request."""
    h = DraftTokensHandler()
    batch = SimpleNamespace(req_ids=["0", "1"], has_structured_output_reqs=False)
    with pytest.raises(ValueError, match="trees length"):
        h.set_draft_tokens(
            batch,
            torch.zeros(2, 4, device="cuda", dtype=torch.long),
            trees=[_tree()],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tree_draft_carry_through_handler_batched():
    """The batched tree-array carry (worker -> scheduler) must produce the
    exact same ``StashedTree`` host lists as per-request ``to_lists``, and the
    batched ``from_lists_batched`` rebuild must recover identical tensors."""
    h = DraftTokensHandler()
    trees = [_tree_on("cuda"), _tree_on("cuda")]
    arrays = (
        torch.stack([t.token_ids for t in trees], dim=0),
        torch.stack([t.depths for t in trees], dim=0),
        torch.stack([t.parents for t in trees], dim=0),
        torch.stack([t.arena for t in trees], dim=0),
        torch.stack([t.words for t in trees], dim=0),
        torch.stack([t.num_nodes for t in trees], dim=0).reshape(-1),
    )
    batch = SimpleNamespace(req_ids=["0", "1"], has_structured_output_reqs=False)
    h.set_draft_tokens(
        batch, torch.zeros(2, 4, device="cuda", dtype=torch.long), tree_arrays=arrays
    )

    res = h.get_draft_tokens()
    assert res is not None
    assert res.req_ids == ["0", "1"]
    assert res.spec_tree_draft is not None
    expected = [t.to_lists() for t in trees]
    for got, exp in zip(res.spec_tree_draft, expected):
        assert got.token_ids == exp.token_ids
        assert got.depths == exp.depths
        assert got.parents == exp.parents
        assert got.arena == exp.arena
        assert got.words == exp.words
        assert got.num_nodes == exp.num_nodes

    # Batched rebuild, variable widths + a treeless request in the middle.
    stashed = res.spec_tree_draft
    stashed[1] = None
    back = TreeDraft.from_lists_batched(stashed, device="cuda")
    assert back[1] is None
    tree0 = trees[0]
    _assert_tree_equal(back[0], tree0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_from_lists_batched_variable_widths():
    """Per-request widths survive the batched rebuild (a scheduler-trimmed
    tree keeps its truncated width and num_nodes, so downstream shape checks
    like tree.token_ids.shape[0] == scheduled count still hold)."""
    s0 = _tree(4).to_lists()
    s2 = _tree(6).to_lists()
    s2.num_nodes = 6
    stashed = [s0, None, s2]
    back = TreeDraft.from_lists_batched(stashed, device="cuda")
    assert back[0].token_ids.shape[0] == 4
    assert back[2].token_ids.shape[0] == 6
    assert int(back[0].num_nodes.reshape(-1)[0]) == 4
    assert int(back[2].num_nodes.reshape(-1)[0]) == 6
    for r in (0, 2):
        assert back[r].token_ids.tolist() == stashed[r].token_ids
        assert back[r].depths.tolist() == stashed[r].depths
        assert back[r].parents.tolist() == stashed[r].parents
        assert back[r].arena.tolist() == stashed[r].arena
        assert back[r].words.tolist() == stashed[r].words
