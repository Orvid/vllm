# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the brute-force traversal verifier reference.

Traversal draws one token per depth from the fully processed target
distribution at the current row, descends iff it is a child of the current
node, and otherwise emits it and stops (leaf/bonus included). These tests pin
greedy behaviour, the T=0 argmax-everywhere invariant, the
over-proposed-unexpanded sampling regime, the
leaf bonus path, and determinism.
"""

import importlib.util
from typing import TypedDict

import torch

from tests.v1.worker.gpu.spec_decode.tree.reference_verifier import (
    tree_verify_reference,
)
from vllm.v1.worker.gpu.spec_decode.tree.tree_metadata import build_tree_naive


def test_reference_verifier_not_in_runtime_package():
    """The brute-force oracle stays TEST material: no runtime module
    named ``vllm...df2t.reference_verifier`` ships, and the test-tree import
    above is what resolves it."""
    assert (
        importlib.util.find_spec(
            "vllm.v1.worker.gpu.spec_decode.tree.reference_verifier"
        )
        is None
    )


def _row_logits(argmax_token: int, vocab: int) -> torch.Tensor:
    """One-hot-ish row whose raw (and processed) argmax is ``argmax_token``."""
    logits = torch.zeros(vocab)
    logits[argmax_token] = 10.0
    return logits


def _logits_from_probs(probs: list[float]) -> torch.Tensor:
    return torch.log(torch.tensor(probs, dtype=torch.float))


def _child_index(
    token_ids: torch.Tensor, parents: torch.Tensor, u: int, x: int
) -> int | None:
    """Test-side child lookup (arena-index parent convention)."""
    for i in range(parents.shape[0]):
        if int(parents[i]) == u and int(token_ids[i]) == x:
            return i
    return None


def _real_tree(b: int, L: int, k: int, vocab: int, seed: int):
    """A tree from the real lazy-k-best builder, as plain token/parent lists."""
    torch.manual_seed(seed)
    scores = torch.randn(L, k, k)
    cand = torch.randint(0, vocab, (L, k))
    draft = build_tree_naive(scores, cand, b, L)
    return draft.token_ids, draft.parents


class _SamplingParams(TypedDict, total=False):
    """Sampling parameters a case passes to the reference."""

    temperature: float
    top_k: int
    top_p: float


def test_traversal_greedy_chain_then_mismatch():
    """Anchor -> token5 -> token9, then the argmax is not a child: exactly two
    tokens accepted, the mismatch token (== argmax at T=0) is emitted."""
    vocab = 256
    target_logits = torch.stack(
        [
            _row_logits(5, vocab),
            _row_logits(9, vocab),
            _row_logits(42, vocab),
            torch.zeros(vocab),
        ]
    )
    token_ids = torch.tensor([5, 9])
    parents = torch.tensor([-1, 0])

    out, num_accepted = tree_verify_reference(target_logits, token_ids, parents)

    assert num_accepted == 2
    assert out == [5, 9, 42]


def test_traversal_t0_argmax_everywhere():
    """Invariant 4: at T=0 every emitted token (accepted, mismatch, and the
    leaf bonus on a fully accepted path) equals the raw argmax of the
    processed distribution at its row, exactly."""
    vocab = 64
    cases: list[tuple[tuple[torch.Tensor, torch.Tensor], _SamplingParams]] = [
        (_real_tree(7, 4, 3, vocab, seed=1), {}),
        (_real_tree(12, 6, 4, vocab, seed=2), {"top_k": 3, "top_p": 0.95}),
    ]
    for (token_ids, parents), sampling in cases:
        b = token_ids.shape[0]
        target_logits = torch.randn(1 + b, vocab)
        out, num_accepted = tree_verify_reference(
            target_logits, token_ids, parents, **sampling
        )
        # replay the greedy walk to recover each emitted token's row
        u = -1
        rows = [0]
        while True:
            child = _child_index(
                token_ids, parents, u, int(torch.argmax(target_logits[rows[-1]]))
            )
            if child is None:
                break
            rows.append(1 + child)
            u = child
        assert num_accepted == len(rows) - 1
        assert len(out) == len(rows)
        for token, row in zip(out, rows):
            expected = int(torch.argmax(target_logits[row]))
            assert token == expected


def test_traversal_sampling_over_proposed_unexpanded():
    """The regime that broke the v5-prequel residual rule: a candidate the
    drafter would over-propose (second-highest transition mass, so it is in the
    served candidate set) is left unexpanded by a B=1 tree; traversal must
    still emit it with the target's probability for that token."""
    torch.manual_seed(3)
    vocab, k, L, b = 8, 4, 1, 1
    # candidate ranks 0..3 at (0, 0): token 0 is best (expanded), token 1 has
    # high transition mass but is NOT expanded (B=1 keeps only the best child).
    scores = torch.tensor([[[1.0, 0.5, -5.0, -5.0]] * k], dtype=torch.float)
    cand = torch.arange(k, dtype=torch.long).repeat(L, 1)
    draft = build_tree_naive(scores, cand, b, L)
    token_ids, parents = draft.token_ids, draft.parents
    assert token_ids.tolist() == [0]

    q = 1
    prob = {0: 0.5, q: 0.25, 2: 0.15, 3: 0.10}
    p = [prob.get(t, 0.0) for t in range(vocab)]
    target_logits = torch.stack(
        [
            _logits_from_probs([max(v, 1e-9) for v in p]),
            _row_logits(2, vocab),  # bonus row: never emits q
        ]
    )

    n = 6000
    freq_q = 0.0
    for seed in range(n):
        out, _ = tree_verify_reference(
            target_logits, token_ids, parents, greedy=False, seed=seed
        )
        freq_q += 1.0 if out[-1] == q else 0.0
    assert abs(freq_q / n - prob[q]) < 0.025


def test_traversal_bonus_via_leaf():
    """A fully accepted chain exhausts the tree at a leaf; the same step emits
    the bonus token (== argmax of the leaf's processed row), no separate bonus
    path and no extra count."""
    vocab = 128
    target_logits = torch.stack(
        [
            _row_logits(5, vocab),
            _row_logits(9, vocab),
            _row_logits(13, vocab),
            _row_logits(88, vocab),
        ]
    )
    token_ids = torch.tensor([5, 9, 13])
    parents = torch.tensor([-1, 0, 1])

    out, num_accepted = tree_verify_reference(target_logits, token_ids, parents)

    assert num_accepted == 3
    assert out == [5, 9, 13, 88]


def test_traversal_anchor_only_emits_single():
    """An empty tree (no expanded children anywhere) emits one token at the
    anchor row with zero accepted."""
    vocab = 32
    target_logits = torch.stack([_row_logits(3, vocab)])
    out, num_accepted = tree_verify_reference(
        target_logits,
        torch.tensor([], dtype=torch.long),
        torch.tensor([], dtype=torch.long),
    )
    assert num_accepted == 0
    assert out == [3]


def test_traversal_deterministic_same_seed():
    """The same seed yields identical outputs across calls."""
    vocab = 64
    token_ids, parents = _real_tree(8, 5, 3, vocab, seed=5)
    target_logits = torch.randn(1 + token_ids.shape[0], vocab)
    runs = [
        tree_verify_reference(
            target_logits,
            token_ids,
            parents,
            greedy=False,
            temperature=0.9,
            top_p=0.9,
            seed=123,
        )
        for _ in range(4)
    ]
    for out, num_accepted in runs[1:]:
        assert out == runs[0][0]
        assert num_accepted == runs[0][1]
