# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the triton tree builder vs the torch reference builder.

The reference (``build_tree_naive``) re-orders every successor row best-first
internally, exactly like the kernel, so both are fed the same raw rows whose
per-predecessor orders differ. ``torch.equal`` is the bit-identical gate.
"""

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.tree.selection.df2t import TreeBuilder
from vllm.v1.worker.gpu.spec_decode.tree.tree_metadata import build_tree_naive


def _raw_inputs(R, L, K, B, seed):
    """Raw unsorted successor rows with a distinct ordering per predecessor.

    Every (request, depth, predecessor) row carries an independent permutation
    of K well-separated values (gaps of 1 >> perturbation < 0.45), so no two
    predecessors share a successor order and the best-first sort must resolve a
    genuinely different sequence per row. ``cand`` are arbitrary token ids.
    """
    torch.manual_seed(seed)
    cand = torch.randint(0, 256, (R, L, K), device="cuda")
    order = torch.argsort(torch.randn(R, L, K, K, device="cuda"), dim=-1)
    vals = torch.arange(K, 0, -1, device="cuda").float().view(1, 1, 1, K)
    pert = (torch.rand(R, L, K, K, device="cuda") - 0.5) * 0.9
    scores = torch.gather(vals.expand(R, L, K, K).contiguous(), -1, order) + pert
    return scores, cand


def _assert_matches(out, scores, cand, B, L):
    for r in range(scores.shape[0]):
        ref = build_tree_naive(scores[r].cpu(), cand[r].cpu(), B, L)
        n = out.num_nodes[r].item()
        assert n == len(ref.token_ids)
        assert torch.equal(out.token_ids[r, :n].cpu(), ref.token_ids)
        assert torch.equal(out.depths[r, :n].cpu(), ref.depths)
        assert torch.equal(out.parents[r, :n].cpu(), ref.parents)
        assert torch.equal(out.words[r, :n].cpu(), ref.words)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_builder_matches_naive_in_bulk():
    R, L, K, B = 8, 8, 8, 12
    scores, cand = _raw_inputs(R, L, K, B, seed=1)
    tb = TreeBuilder(R, L, K, B)
    out = tb.build(scores, cand)
    _assert_matches(out, scores, cand, B, L)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_builder_matches_naive_varying_predecessors():
    # Per-predecessor orders differ here, so a non-self-sorting reference that
    # reads ``cand`` by raw rank would diverge from the kernel's per-row
    # permuted tokens; both must agree bit-for-bit.
    R, L, K, B = 4, 6, 5, 10
    scores, cand = _raw_inputs(R, L, K, B, seed=3)
    tb = TreeBuilder(R, L, K, B)
    out = tb.build(scores, cand)
    _assert_matches(out, scores, cand, B, L)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_builder_clamps_when_fewer_prefixes():
    R, L, K, B = 3, 3, 2, 30
    scores, cand = _raw_inputs(R, L, K, B, seed=2)
    tb = TreeBuilder(R, L, K, B)
    out = tb.build(scores, cand)
    n_expect = sum(K**d for d in range(1, L + 1))
    for r in range(R):
        n = out.num_nodes[r].item()
        assert n == n_expect
    _assert_matches(out, scores, cand, B, L)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_builder_matches_naive_at_depth_sixteen():
    """The Stage-0 L=16 depth cap packs: a 16-deep path can set the int64 sign
    bit, so the heap tie-break must compare paths unsignedly; on real (rarely
    tying) scores the emitted tree still matches the torch reference."""
    R, L, K, B = 2, 16, 4, 12
    scores, cand = _raw_inputs(R, L, K, B, seed=11)
    tb = TreeBuilder(R, L, K, B)
    out = tb.build(scores, cand)
    _assert_matches(out, scores, cand, B, L)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_builder_depth_sixteen_is_deterministic_across_builds():
    """L=16 must not flip order between builds from sign-wrapped paths: the
    same input always emits the same tree (the heap path comparison is
    unsigned-consistent)."""
    R, L, K, B = 2, 16, 5, 16
    scores, cand = _raw_inputs(R, L, K, B, seed=13)
    tb = TreeBuilder(R, L, K, B)
    a = tb.build(scores, cand)
    b = tb.build(scores, cand)
    assert torch.equal(a.token_ids, b.token_ids)
    assert torch.equal(a.num_nodes, b.num_nodes)
    assert torch.equal(a.words, b.words)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_builder_strides_by_drafter_steps():
    """The drafter emits ``num_speculative_steps`` score rows per request while
    the tree depth cap L can be smaller (default ``min(num_spec,
    block_size-1)``). The kernel must read every request's rows at stride
    ``num_steps``, not ``L``, or requests r > 0 read another request's scores —
    which silently collapses acceptance on the first non-first decode.

    With num_steps=12 > L=5 over R=3 requests, the kernel reads only the first
    L rows of each request; the reference matches on the same first-L rows.
    """
    R, S, K, B, L = 3, 12, 4, 10, 5
    scores, cand = _raw_inputs(R, S, K, B, seed=7)
    tb = TreeBuilder(R, L, K, B, num_steps=S)
    out = tb.build(scores, cand)
    for r in range(R):
        ref = build_tree_naive(scores[r].cpu()[:L], cand[r].cpu()[:L], B, L)
        n = out.num_nodes[r].item()
        assert n == len(ref.token_ids)
        assert torch.equal(out.token_ids[r, :n].cpu(), ref.token_ids)
        assert torch.equal(out.depths[r, :n].cpu(), ref.depths)
        assert torch.equal(out.parents[r, :n].cpu(), ref.parents)
        assert torch.equal(out.words[r, :n].cpu(), ref.words)


def test_triton_builder_never_emits_impossible_candidates():
    """Candidates scored -inf or NaN (a top-k short of valid entries returns
    sentinel ids there) are never emitted: the tree just gets smaller."""
    R, L, K, B = 1, 3, 4, 32
    sentinel = 2**31 - 1
    scores, cand = _raw_inputs(R, L, K, B, seed=7)
    # Two real candidates per position; the others are impossible.
    scores[..., 2] = float("-inf")
    scores[..., 3] = float("nan")
    cand[..., 2:] = sentinel
    out = TreeBuilder(R, L, K, B, "cuda").build(scores, cand)
    n = int(out.num_nodes[0])
    assert n == 2 + 4 + 8  # a full binary tree of depth 3
    assert (out.token_ids[0, :n] != sentinel).all()


def test_children_follow_their_parents_candidate_row():
    """Transition rows are indexed by the predecessor's candidate index, not
    its best-first rank: the best path must be the greedy chain through the
    transitions, which here prefer the other candidate after each one."""
    L, K = 4, 2
    cand = torch.tensor([[10, 11], [20, 21], [30, 31], [40, 41]], device="cuda")
    scores = torch.zeros(1, L, K, K, device="cuda")
    scores[0, 0, :, 1] = 3.0  # the anchor prefers candidate 1
    for d in range(1, L):
        scores[0, d, 0, 1] = 3.0  # after candidate 0 prefer candidate 1
        scores[0, d, 1, 0] = 3.0  # after candidate 1 prefer candidate 0
    greedy = [11, 20, 31, 40]

    out = TreeBuilder(1, L, K, 8, "cuda").build(scores, cand[None])
    ref = build_tree_naive(scores[0].cpu(), cand.cpu(), 8, L)
    for tokens, depths in (
        (out.token_ids[0].cpu(), out.depths[0].cpu()),
        (ref.token_ids, ref.depths),
    ):
        # Best-first: the greedy chain's nodes are emitted first, in order.
        assert depths[: len(greedy)].tolist() == [1, 2, 3, 4]
        assert tokens[: len(greedy)].tolist() == greedy


def _routing(token_experts: dict[int, list[list[int]]], num_experts: int = 8):
    """A routing memory with two MoE layers (capturer layers 0 and 2) whose
    table holds ``token_experts[token] = [layer0 experts, layer1 experts]``,
    each expert costing 1 ms."""
    from vllm.v1.worker.gpu.spec_decode.tree.routing import MoERoutingMemory

    tokens = list(token_experts)
    routes = torch.full((len(tokens), 3, 2), -1, dtype=torch.int32, device="cuda")
    for i, tok in enumerate(tokens):
        routes[i, 0] = torch.tensor(token_experts[tok][0])
        routes[i, 2] = torch.tensor(token_experts[tok][1])
    mem = MoERoutingMemory(
        routes, [0, 2], num_experts, torch.ones(2), torch.device("cuda")
    )
    mem.record(torch.tensor(tokens, device="cuda"), len(tokens))
    return mem


def test_routing_memory_records_each_moe_layer_by_token_bucket():
    from vllm.v1.worker.gpu.spec_decode.tree.routing import _HASH_MULT

    mem = _routing({7: [[1, 2], [3, 4]], 9: [[5, 6], [0, 7]]})
    for tok, (l0, l1) in {7: ([1, 2], [3, 4]), 9: ([5, 6], [0, 7])}.items():
        b = (tok * _HASH_MULT) & (mem.buckets - 1)
        assert mem.table[0, b].tolist() == l0
        assert mem.table[1, b].tolist() == l1  # capturer layer 2


def _two_way_choice():
    """Under the anchor, candidate 10 (mass 0.55) needs four new experts and
    candidate 11 (mass 0.45) only experts the anchor (token 5) already uses."""
    L, K = 1, 2
    cand = torch.tensor([[[10, 11]]], device="cuda")
    scores = torch.zeros(1, L, K, K, device="cuda")
    scores[0, 0, :, 0] = torch.log(torch.tensor(0.55))
    scores[0, 0, :, 1] = torch.log(torch.tensor(0.45))
    routing = _routing(
        {5: [[0, 1], [0, 1]], 10: [[2, 3], [2, 3]], 11: [[0, 1], [0, 1]]}
    )
    return scores, cand, routing


@pytest.mark.parametrize("lam, first", [(0.0, 10), (1.0, 11)])
def test_expert_aware_selection_trades_mass_for_resident_experts(lam, first):
    """With lam = 0 the likelier node comes first; once a ms is worth a token,
    the node reusing the anchor's experts does."""
    scores, cand, routing = _two_way_choice()
    tb = TreeBuilder(1, 1, 2, 2, "cuda")
    tb.bind_routing(routing, torch.tensor([lam], device="cuda"))
    out = tb.build(scores, cand, torch.tensor([5], device="cuda"))
    assert out.token_ids[0].tolist()[0] == first
    assert sorted(out.token_ids[0].tolist()) == [10, 11]


def test_expert_aware_selection_without_cost_is_the_mass_order():
    """With costs priced at zero, expert-aware selection builds the same tree
    as ranking by mass alone."""
    R, L, K, B = 3, 4, 4, 24
    scores, cand = _raw_inputs(R, L, K, B, seed=3)
    plain = TreeBuilder(R, L, K, B, "cuda").build(scores, cand)
    aware = TreeBuilder(R, L, K, B, "cuda")
    aware.bind_routing(_routing({1: [[0, 1], [2, 3]]}), torch.zeros(1, device="cuda"))
    out = aware.build(scores, cand, torch.tensor([1, 1, 1], device="cuda"))
    assert torch.equal(out.token_ids, plain.token_ids)
    assert torch.equal(out.parents, plain.parents)
