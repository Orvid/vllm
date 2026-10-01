# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity tests for the triton tree builder vs the torch reference builder.

The reference (``build_tree_naive``) re-orders every successor row best-first
internally, exactly like the kernel, so both are fed the same raw rows whose
per-predecessor orders differ. ``torch.equal`` is the bit-identical gate.
"""

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.df2t.tree_builder import TreeBuilder
from vllm.v1.worker.gpu.spec_decode.df2t.tree_metadata import build_tree_naive


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
