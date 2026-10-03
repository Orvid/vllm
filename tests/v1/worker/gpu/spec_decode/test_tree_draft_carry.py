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


def _tree_with_token(tok: int, device: str) -> TreeDraft:
    """A one-node tree whose single token is ``tok`` (for stash tests)."""
    return TreeDraft(
        token_ids=torch.tensor([tok], dtype=torch.long, device=device),
        depths=torch.tensor([1], dtype=torch.long, device=device),
        parents=torch.tensor([-1], dtype=torch.long, device=device),
        arena=torch.tensor([0], dtype=torch.long, device=device),
        words=torch.tensor([1], dtype=torch.long, device=device),
        num_nodes=torch.tensor([1], dtype=torch.int32, device=device),
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

    # The stash is consumed by the take: a second take returns no trees.
    res2 = h.get_draft_tokens()
    assert res2 is not None and res2.spec_tree_draft is None

    # Rebuilt trees, variable widths + a treeless request in the middle.
    stashed = res.spec_tree_draft
    stashed[1] = None
    back = TreeDraft.from_lists_batched(stashed, device="cuda")
    assert back[1] is None
    tree0 = trees[0]
    _assert_tree_equal(back[0], tree0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tree_draft_stash_overwrite_before_take_raises():
    """Stashing a second step's trees before the previous step's stash was
    taken must raise: the single pinned buffer is reused, so silently
    overwriting it would hand step N+1's trees to step N. The staged->taken
    ordering is the engine's graft contract, and this keeps it fail-closed."""
    h = DraftTokensHandler()
    batch = SimpleNamespace(req_ids=["a"], has_structured_output_reqs=False)
    nt = torch.zeros(1, 1, device="cuda", dtype=torch.long)

    def _arrays(tok: int):
        t = _tree_with_token(tok, "cuda")
        return (
            torch.stack([t.token_ids], dim=0),
            torch.stack([t.depths], dim=0),
            torch.stack([t.parents], dim=0),
            torch.stack([t.arena], dim=0),
            torch.stack([t.words], dim=0),
            torch.stack([t.num_nodes], dim=0).reshape(-1),
        )

    h.set_draft_tokens(batch, nt, tree_arrays=_arrays(7))
    with pytest.raises(RuntimeError, match="never taken"):
        h.set_draft_tokens(batch, nt, tree_arrays=_arrays(9))
    # The first stash is still intact; consuming it succeeds normally.
    res = h.get_draft_tokens()
    assert res is not None
    assert res.spec_tree_draft[0].token_ids == [7]  # type: ignore[index]
    # And the normal set -> take -> set -> take cycle stays legal.
    h.set_draft_tokens(batch, nt, tree_arrays=_arrays(11))
    res = h.get_draft_tokens()
    assert res is not None
    assert res.spec_tree_draft[0].token_ids == [11]  # type: ignore[index]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tree_draft_throwaway_warmup_stash_is_abandoned():
    """Warmup/graph-capture stages trees step after step with no take (its
    drafts are discarded). The leftover throwaway stash must be abandoned by
    the first real serving step — not tripping the take-ordering guard and
    never served stale — while a real-serving take-skip still raises."""
    h = DraftTokensHandler()
    batch = SimpleNamespace(req_ids=["a"], has_structured_output_reqs=False)
    nt = torch.zeros(1, 1, device="cuda", dtype=torch.long)

    def _arrays(tok: int):
        t = _tree_with_token(tok, "cuda")
        return (
            torch.stack([t.token_ids], dim=0),
            torch.stack([t.depths], dim=0),
            torch.stack([t.parents], dim=0),
            torch.stack([t.arena], dim=0),
            torch.stack([t.words], dim=0),
            torch.stack([t.num_nodes], dim=0).reshape(-1),
        )

    # Two warmup stages (no take in between) — legal.
    h.set_draft_tokens(batch, nt, tree_arrays=_arrays(1), allow_pending_overwrite=True)
    h.set_draft_tokens(batch, nt, tree_arrays=_arrays(2), allow_pending_overwrite=True)
    # First real serving step: the leftover throwaway stash is abandoned, and
    # the real stash is served instead.
    h.set_draft_tokens(batch, nt, tree_arrays=_arrays(3))
    res = h.get_draft_tokens()
    assert res is not None
    assert res.spec_tree_draft[0].token_ids == [3]  # type: ignore[index]
    # A real-serving re-stash before the take still fails closed.
    h.set_draft_tokens(batch, nt, tree_arrays=_arrays(4))
    with pytest.raises(RuntimeError, match="never taken"):
        h.set_draft_tokens(batch, nt, tree_arrays=_arrays(5))


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
