# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the brute-force SpecInfer multi-round rejection reference (the
oracle Task 10's GPU ``TreeRejectionSampler`` must match)."""

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.df2t.reference_specinfer import (
    _residual,
    specinfer_reference,
)
from vllm.v1.worker.gpu.spec_decode.df2t.tree_metadata import (
    TreeDraft,
    _best_first_perm,
    build_tree_naive,
    pack_ancestor_words,
)


def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=-1)


def _tree(nodes: list[tuple[int, int, int]], device: str = "cpu") -> TreeDraft:
    """Build a TreeDraft from ``(depth, token_id, parent_arena)`` tuples."""
    token_ids = torch.tensor([t for _, t, _ in nodes], dtype=torch.long, device=device)
    depths = torch.tensor([d for d, _, _ in nodes], dtype=torch.long, device=device)
    parents = torch.tensor([p for _, _, p in nodes], dtype=torch.long, device=device)
    arena = torch.arange(len(nodes), dtype=torch.long, device=device)
    words = pack_ancestor_words(parents, arena, len(nodes))
    return TreeDraft(token_ids, depths, parents, arena, words)


def _row_logits(argmax_token: int, vocab: int) -> torch.Tensor:
    logits = torch.zeros(vocab)
    logits[argmax_token] = 10.0
    return logits


def _candidate_q(
    cands: list[int], q_logp_row: list[float], vocab: int, rows: int, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows of an unmodified [rows, K, K] proposal; only the given (0, 0) row
    carries the candidates, so the anchor's rank-0 row is exercised."""
    q_cand = torch.zeros(rows, k, k, dtype=torch.long)
    q_logp = torch.zeros(rows, k, k)
    q_cand[0, 0] = torch.tensor(cands)
    q_logp[0, 0] = torch.tensor(q_logp_row)
    return q_cand, q_logp


def test_reference_greedy_accepts_chain_then_rejects():
    """A 2-chain accepted then a depth-3 reject: num_accepted == 2 and the
    recovered token is the argmax of the truncation-corrected residual (the
    true target token here)."""
    vocab = 256
    target_logits = torch.stack(
        [
            _row_logits(5, vocab),
            _row_logits(9, vocab),
            _row_logits(250, vocab),
            torch.zeros(vocab),
        ]
    )
    # depth-1 child 5, depth-2 child 9, depth-3 child 100 (not the greedy token)
    tree = _tree([(1, 5, -1), (2, 9, 0), (3, 100, 1)])
    q_cand, q_logp = _candidate_q([5, 7], [0.0, 0.0], vocab, rows=3, k=2)
    q_cand[1, 0] = torch.tensor([9, 6])
    q_cand[2, 0] = torch.tensor([100, 200])

    out, num_accepted = specinfer_reference(target_logits, tree, q_cand, q_logp)

    assert num_accepted == 2
    assert out == [5, 9, 250]
    p = _softmax(target_logits[2])
    q_full = torch.zeros_like(p)
    q_full[[100, 200]] = _softmax(q_logp[2, 0])
    resid = torch.clamp(p - q_full, min=0.0)
    resid /= resid.sum()
    assert out[-1] == int(torch.argmax(resid))


def test_reference_sampling_output_matches_target():
    """Statistical losslessness smoke: for a tree where the proposal dominates
    the target on the expanded candidates, the first emitted token's empirical
    distribution matches the target (generous tolerance)."""
    vocab = 4
    p = torch.tensor([0.4, 0.4, 0.1, 0.1])
    target_logits = torch.stack([torch.log(p), torch.log(p), torch.log(p)])
    # both depth-1 siblings (tokens 0 and 1) expanded; q uniform over {0, 1}
    tree = _tree([(1, 0, -1), (1, 1, -1)])
    q_cand, q_logp = _candidate_q([0, 1], [0.0, 0.0], vocab, rows=1, k=2)

    n = 6000
    first = torch.zeros(vocab)
    for seed in range(n):
        out, _ = specinfer_reference(
            target_logits, tree, q_cand, q_logp, greedy=False, seed=seed
        )
        first[out[0]] += 1
    freq = first / n

    assert torch.allclose(freq, p, atol=0.03)


def test_reference_residual_recovered_from_q_correction():
    """Residual correctness: when the greedy token lies inside the candidate set
    but on an unexpanded slot, the recovered token comes from
    ``max(p - q, 0) / Z`` (and differs from the greedy argmax), because the
    proposal mass on the unexpanded candidate is subtracted."""
    vocab = 32
    a, b, c, e = 3, 10, 7, 20
    target_logits = torch.stack(
        [
            _row_logits(a, vocab),
            torch.zeros(vocab),
            torch.zeros(vocab),
        ]
    )
    # target row 1: softmax = (0.6 on b, 0.2 on c, 0.2 on e)
    row1 = torch.full((vocab,), -30.0)
    row1[b] = torch.log(torch.tensor(0.6))
    row1[c] = torch.log(torch.tensor(0.2))
    row1[e] = torch.log(torch.tensor(0.2))
    target_logits[1] = row1
    # depth-1 child `a` accepted; the depth-2 child is `c` (expanded), while
    # the proposal also carries `b` on an unexpanded slot.
    tree = _tree([(1, a, -1), (2, c, 0)])
    q_cand, q_logp = _candidate_q([a, 99], [0.0, 0.0], vocab, rows=2, k=2)
    q_cand[1, 0] = torch.tensor([b, c])
    q_logp[1, 0] = torch.log(torch.tensor([0.9, 0.1]))

    out, num_accepted = specinfer_reference(target_logits, tree, q_cand, q_logp)

    p = _softmax(target_logits[1])
    q_full = torch.zeros_like(p)
    q_full[[b, c]] = _softmax(q_logp[1, 0])
    resid = torch.clamp(p - q_full, min=0.0)
    resid /= resid.sum()
    assert num_accepted == 1
    assert int(torch.argmax(p)) == b  # greedy token is on an unexpanded slot
    assert out == [a, int(torch.argmax(resid))]
    assert out[-1] == e


def test_reference_residual_building_block():
    """The truncated-support residual is nonnegative, sums to one, and its
    denominator stays positive when the target has mass outside the candidate
    support."""
    cases = [
        (torch.tensor([0.4, 0.4, 0.2]), torch.tensor([0.5, 0.5, 0.0])),
        (torch.tensor([0.6, 0.2, 0.2]), torch.tensor([0.4, 0.6, 0.0])),
    ]
    for p, q in cases:
        r = _residual(p, q)
        assert torch.all(r >= 0)
        assert torch.allclose(r.sum(), torch.tensor(1.0))
    # target mass strictly outside the support keeps Z > 0
    p = torch.tensor([0.5, 0.3, 0.2])
    q = torch.tensor([0.9, 0.1, 0.0])
    r = _residual(p, q)
    assert torch.any(r > 0)


def test_reference_full_depth_accept_emits_one_bonus():
    """A fully-accepted chain exhausts the depth budget and appends exactly one
    bonus target draw from the last accepted node's row."""
    vocab = 128
    target_logits = torch.stack(
        [
            _row_logits(5, vocab),
            _row_logits(9, vocab),
            _row_logits(77, vocab),
        ]
    )
    tree = _tree([(1, 5, -1), (2, 9, 0)])
    q_cand, q_logp = _candidate_q([5, 7], [0.0, 0.0], vocab, rows=2, k=2)
    q_cand[1, 0] = torch.tensor([9, 6])

    out, num_accepted = specinfer_reference(target_logits, tree, q_cand, q_logp)

    assert num_accepted == 2
    assert out == [5, 9, 77]
    assert len(out) == num_accepted + 1


def test_reference_deterministic_same_seed():
    """The same seed yields byte-identical outputs across calls."""
    vocab = 4
    p = torch.tensor([0.4, 0.4, 0.1, 0.1])
    target_logits = torch.stack([torch.log(p), torch.log(p), torch.log(p)])
    tree = _tree([(1, 0, -1), (1, 1, -1)])
    q_cand, q_logp = _candidate_q([0, 1], [0.0, 0.0], vocab, rows=1, k=2)

    runs = [
        specinfer_reference(target_logits, tree, q_cand, q_logp, greedy=False, seed=123)
        for _ in range(4)
    ]
    for out, num_accepted in runs[1:]:
        assert out == runs[0][0]
        assert num_accepted == runs[0][1]


@pytest.mark.parametrize("greedy", [True, False])
def test_reference_real_tree_walk_invariants(greedy):
    """On a tree from the real lazy-k-best builder plus its aligned proposal
    rows, the walk completes with consistent shapes and deterministic output."""
    torch.manual_seed(0)
    L, K, B, vocab = 4, 3, 6, 64
    scores = torch.randn(L, K, K)
    cand = torch.randint(0, vocab, (L, K))
    tree = build_tree_naive(scores, cand, B, L)
    logp = torch.log_softmax(scores.float(), dim=-1)
    perm = _best_first_perm(logp)
    q_logp = torch.gather(logp, -1, perm)
    q_cand = torch.gather(cand[:, None, :].expand(L, K, K).contiguous(), -1, perm)
    target_logits = torch.randn(1 + B, vocab)

    kw = dict(
        temperature=0.9 if not greedy else 1.0,
        top_k=None,
        top_p=None,
        seed=7 if not greedy else None,
    )
    out1, num1 = specinfer_reference(
        target_logits, tree, q_cand, q_logp, greedy=greedy, **kw
    )
    out2, num2 = specinfer_reference(
        target_logits, tree, q_cand, q_logp, greedy=greedy, **kw
    )

    assert num1 == num2
    max_depth = int(tree.depths.max())
    assert 0 <= num1 <= max_depth
    assert len(out1) == num1 + 1
    assert all(isinstance(t, int) and 0 <= t < vocab for t in out1)
