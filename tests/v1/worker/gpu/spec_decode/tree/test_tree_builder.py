# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the triton tree builder vs the torch reference builder.

The reference (``build_tree_naive``) re-orders every successor row best-first
internally, exactly like the kernel, so both are fed the same raw rows whose
per-predecessor orders differ. ``torch.equal`` is the bit-identical gate.
"""

from types import SimpleNamespace

import numpy as np
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


def _routes(row_experts: list[list[list[int]]]) -> torch.Tensor:
    """A capturer buffer with two MoE layers (capturer layers 0 and 2): row i
    routes to ``row_experts[i] = [layer0 experts, layer1 experts]``."""
    routes = torch.full((len(row_experts), 3, 2), -1, dtype=torch.int32, device="cuda")
    for i, (l0, l1) in enumerate(row_experts):
        routes[i, 0] = torch.tensor(l0)
        routes[i, 2] = torch.tensor(l1)
    return routes


def _routing(token_experts: dict[int, list[list[int]]], num_experts: int = 8):
    """A predictor that names ``token_experts[token]`` (experts 0 and 1 for
    any other token), each expert costing 1 ms."""
    from vllm.v1.worker.gpu.spec_decode.tree.routing import MoERoutingPredictor

    class FixedRouting(MoERoutingPredictor):
        def _predict_into(self, ids, hidden, pred):
            pred[:, : ids.numel()] = lookup[ids].permute(1, 0, 2)

    lookup = torch.tensor([0, 1], device="cuda").repeat(512, 2, 1)
    for tok, experts in token_experts.items():
        lookup[tok] = torch.tensor(experts)
    return FixedRouting(
        _routes(list(token_experts.values())),
        [0, 2],
        num_experts,
        torch.ones(2),
        torch.device("cuda"),
    )


def _decode_batch(
    tokens: list[int], qsl: list[int], positions: list[int] | None = None
) -> SimpleNamespace:
    """A forward of decode requests, request r spanning rows qsl[r]:qsl[r+1]
    (consecutive positions unless given)."""
    if positions is None:
        positions = list(range(len(tokens)))
    return SimpleNamespace(
        num_reqs=len(qsl) - 1,
        num_tokens=len(tokens),
        query_start_loc_np=np.array(qsl),
        query_start_loc=torch.tensor(qsl, device="cuda"),
        is_prefilling_np=np.zeros(len(qsl) - 1, dtype=bool),
        idx_mapping=torch.arange(len(qsl) - 1, device="cuda"),
        input_ids=torch.tensor(tokens, device="cuda"),
        positions=torch.tensor(positions, device="cuda"),
    )


def test_top_experts_kernel_matches_topk_into_a_slot_table():
    """The fused top-k names each (row, layer)'s best experts, best first,
    and writes them straight into a [layers, slots, k] table."""
    from vllm.v1.worker.gpu.spec_decode.tree.routing import _top_experts

    logits = torch.randn(5, 3, 200, device="cuda")
    ref = logits.topk(8, dim=-1).indices
    assert torch.equal(_top_experts(logits, 8), ref)
    table = torch.full((3, 9, 8), -1, dtype=torch.int16, device="cuda")
    _top_experts(logits, 8, out=table)
    assert torch.equal(table[:, :5].long(), ref.permute(1, 0, 2))
    assert (table[:, 5:] == -1).all()


def test_routing_meter_totals_each_forwards_distinct_experts():
    """Tokens 7 and 9 share expert 2 of layer 0: the forward reaches 7
    distinct experts at 1 ms each, and a forward of token 7 alone adds 4."""
    mem = _routing({7: [[1, 2], [3, 4]], 9: [[2, 6], [0, 7]]})
    mem.observe_forward(_decode_batch([7, 9], [0, 2]))
    assert mem.forward_cost_total.item() == 7.0
    mem.observe_forward(_decode_batch([7], [0, 1]))
    assert mem.forward_cost_total.item() == 11.0


# Experts of anchor 5 and of token 9 drafted after it at depths 1 and 2.
_CHAIN_ROUTES = ([[1, 2], [3, 4]], [[6, 7], [0, 8]], [[9, 10], [11, 12]])


def _chain_predictor():
    """An online predictor over the verify of anchor 5 and its chain 9 -> 9
    (positions anchor + depth), the draft's candidates and hidden states."""
    from vllm.v1.worker.gpu.spec_decode.tree.routing import OnlineRoutingPredictor

    torch.manual_seed(0)
    pred = OnlineRoutingPredictor(
        embed=torch.nn.Embedding(64, 32).cuda(),
        hidden_size=32,
        draft_hidden_size=16,
        max_num_reqs=1,
        routes=_routes(list(_CHAIN_ROUTES)),
        moe_layer_ids=[0, 2],
        num_experts=16,
        expert_cost_ms=torch.ones(2),
        device=torch.device("cuda"),
    )
    batch = _decode_batch([5, 9, 9], [0, 3], positions=[10, 11, 12])
    cand = torch.tensor([[[9], [9]]], device="cuda")
    return pred, batch, cand, torch.randn(1, 2, 16, device="cuda")


def test_online_predictor_batches_forwards_into_one_update(monkeypatch):
    """With a 6-row batch, the 3-row forwards are buffered and every second
    one takes a single optimizer step over both."""
    from vllm.v1.worker.gpu.spec_decode.tree import routing

    monkeypatch.setattr(routing, "_TRAIN_BATCH_ROWS", 6)
    pred, batch, cand, draft_hidden = _chain_predictor()
    graded = []  # predicted experts graded so far (rows x layers x top-k)
    with torch.inference_mode():
        for _ in range(4):
            pred.observe_forward(batch)
            pred.predict(cand, batch.input_ids[:1], draft_hidden)
            graded.append(pred.hit_stats[1].item() - routing._HIT_PRIOR)
    assert graded == [0.0, 24.0, 24.0, 48.0]


def test_online_predictor_tells_draft_positions_apart(monkeypatch):
    """Token 9 drafted at depths 1 and 2 routes to different experts there:
    trained on decode forwards, the predictor names each from the drafter's
    hidden state at its position (and the anchor's from its token), and its
    measured hit rate follows."""
    from vllm.v1.worker.gpu.spec_decode.tree import routing

    # An optimizer step every forward, to learn within the test.
    monkeypatch.setattr(routing, "_TRAIN_MAX_STALE", 1)
    anchor, depth1, depth2 = _CHAIN_ROUTES
    pred, batch, cand, draft_hidden = _chain_predictor()
    with torch.inference_mode():
        for _ in range(300):
            pred.observe_forward(batch)
            pred.predict(cand, batch.input_ids[:1], draft_hidden)
            # Paced like decode steps, so the pinned staging slots are free.
            torch.accelerator.synchronize()
        # Both run every decode step: neither may wait on the device.
        torch.cuda.set_sync_debug_mode("error")
        try:
            pred.observe_forward(batch)
            out = pred.predict(cand, batch.input_ids[:1], draft_hidden)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    # Slots: (depth 1, candidate 0), (depth 2, candidate 0), then the anchor.
    for slot, truth in enumerate([depth1, depth2, anchor]):
        assert sorted(out[0, slot].tolist()) == truth[0]
        assert sorted(out[1, slot].tolist()) == truth[1]
    hits, total = pred.hit_stats.tolist()
    assert hits / total > 0.7


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


def test_expert_aware_selection_shares_experts_across_the_batch():
    """Request 0's tree activates experts 2 and 3; request 1 then prefers its
    slightly less likely node that routes there over one needing new
    experts."""
    L, K = 1, 2
    cand = torch.tensor([[[20, 21]], [[30, 31]]], device="cuda")
    scores = torch.zeros(2, L, K, K, device="cuda")
    scores[:, 0, :, 0] = torch.log(torch.tensor(0.55))
    scores[:, 0, :, 1] = torch.log(torch.tensor(0.45))
    routing = _routing(
        {
            5: [[0, 1], [0, 1]],  # both anchors
            20: [[2, 3], [2, 3]],
            21: [[6, 7], [6, 7]],
            30: [[4, 5], [4, 5]],  # new experts for request 1
            31: [[2, 3], [2, 3]],  # what request 0 already activates
        }
    )
    tb = TreeBuilder(2, 1, 2, 1, "cuda")
    tb.bind_routing(routing, torch.tensor([1.0], device="cuda"))
    out = tb.build(scores, cand, torch.tensor([5, 5], device="cuda"))
    assert out.token_ids[:, 0].tolist() == [20, 31]


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
