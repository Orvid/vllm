# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the triton tree builder vs the torch reference builder."""

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.df2t.tree_builder import TreeBuilder
from vllm.v1.worker.gpu.spec_decode.df2t.tree_metadata import build_tree_naive


def _aligned_inputs(R, L, K, B, seed):
    """Best-first transition rows plus their rank-aligned candidate tokens.

    The lazy-k-best reference prices the next sibling in rank order, so each
    (depth, predecessor) row must be descending and ``cand[d, r]`` must hold the
    r-th best successor token. Column ``r`` carries value ``K - r`` with a
    sub-0.5 per-(row, predecessor) perturbation: the order is fixed (gaps of 1
    exceed the perturbation range) while masses still vary across predecessors.
    """
    torch.manual_seed(seed)
    cand = torch.randint(0, 256, (R, L, K), device="cuda")
    rvals = torch.arange(K, 0, -1, device="cuda").float().view(1, 1, 1, K)
    pert = (torch.rand(R, L, K, K, device="cuda") - 0.5) * 0.9
    scores = rvals.expand(R, L, K, K) + pert
    return scores, cand


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_builder_matches_naive_in_bulk():
    R, L, K, B = 8, 8, 8, 12
    scores, cand = _aligned_inputs(R, L, K, B, seed=1)
    tb = TreeBuilder(R, L, K, B)
    out = tb.build(scores, cand)
    for r in range(R):
        ref = build_tree_naive(scores[r].cpu(), cand[r].cpu(), B, L)
        n = out.num_nodes[r].item()
        assert n == len(ref.token_ids)
        assert torch.equal(out.token_ids[r, :n].cpu(), ref.token_ids)
        assert torch.equal(out.depths[r, :n].cpu(), ref.depths)
        assert torch.equal(out.parents[r, :n].cpu(), ref.parents)
        assert torch.equal(out.words[r, :n].cpu(), ref.words)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_builder_sorts_unsorted_rows():
    # DFlash2 _score_edges emits successors out of mass order; the builder must
    # re-rank each row itself. Shuffle the columns of an aligned setup and hand
    # the shuffled pair to the kernel, while the reference sees the best-first
    # pair; both must still select the same exact top-B tree.
    R, L, K, B = 4, 6, 5, 10
    scores, cand = _aligned_inputs(R, L, K, B, seed=3)
    shuf = torch.argsort(torch.randn(R, L, K, device="cuda"), dim=-1, descending=True)
    scores_shuf = torch.gather(scores, -1, shuf.unsqueeze(-2).expand(R, L, K, K))
    cand_shuf = torch.gather(cand, -1, shuf)
    tb = TreeBuilder(R, L, K, B)
    out = tb.build(scores_shuf, cand_shuf)
    for r in range(R):
        ref = build_tree_naive(scores[r].cpu(), cand[r].cpu(), B, L)
        n = out.num_nodes[r].item()
        assert n == len(ref.token_ids)
        assert torch.equal(out.token_ids[r, :n].cpu(), ref.token_ids)
        assert torch.equal(out.words[r, :n].cpu(), ref.words)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triton_builder_clamps_when_fewer_prefixes():
    R, L, K, B = 3, 3, 2, 30
    scores, cand = _aligned_inputs(R, L, K, B, seed=2)
    tb = TreeBuilder(R, L, K, B)
    out = tb.build(scores, cand)
    n_expect = sum(K**d for d in range(1, L + 1))
    for r in range(R):
        ref = build_tree_naive(scores[r].cpu(), cand[r].cpu(), B, L)
        n = out.num_nodes[r].item()
        assert n == len(ref.token_ids) == n_expect
        assert torch.equal(out.token_ids[r, :n].cpu(), ref.token_ids)
        assert torch.equal(out.words[r, :n].cpu(), ref.words)
