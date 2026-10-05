# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TreeDraft metadata and the lazy-k-best torch reference builder."""

from itertools import product

import torch

from vllm.v1.worker.gpu.spec_decode.tree.tree_metadata import (
    TreeDraft,
    build_tree_naive,
)


def _all_prefixes(K: int, L: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for d in range(1, L + 1):
        out.extend(product(range(K), repeat=d))
    return out


def _path_mass(logp: torch.Tensor, path: tuple[int, ...]) -> float:
    mass = 0.0
    for d, rank in enumerate(path, start=1):
        prank = path[d - 2] if d >= 2 else 0
        mass += float(logp[d - 1, prank, rank])
    return mass


def _exact_top_b(
    paths: list[tuple[int, ...]], logp: torch.Tensor, B: int
) -> list[tuple[int, ...]]:
    return sorted(
        paths,
        key=lambda p: (-_path_mass(logp, p), -_path_mass(logp, p[:-1]), p),
    )[:B]


def _paths_from_draft(draft: TreeDraft) -> list[tuple[int, ...]]:
    paths: list[tuple[int, ...]] = []
    for i in range(len(draft.token_ids)):
        parent_path = paths[draft.parents[i]] if draft.parents[i] >= 0 else ()
        paths.append(parent_path + (int(draft.token_ids[i]),))
    return paths


def test_build_tree_naive_prefix_and_words():
    torch.manual_seed(0)
    L, K, B = 4, 4, 6
    scores = torch.randn(L, K, K)
    cand = torch.zeros(L, K, dtype=torch.long)
    draft = build_tree_naive(scores, cand, B, L)
    assert draft.token_ids.shape == (B,)
    # prefix-closed: every non-root node's parent is present
    for i in range(1, B):
        assert draft.parents[i].item() < i
    # ancestor word: bit of parent set in child (roots have no parent)
    for i in range(1, B):
        p = draft.parents[i].item()
        if p >= 0:
            assert (draft.words[i] >> draft.arena[p]) & 1


def test_build_tree_naive_matches_exact_topB():
    torch.manual_seed(0)
    L, K, B = 4, 4, 8
    # Lazy k-best prices the next sibling by rank order, so successors must be
    # best-first: sort the transition scores descending along the candidate axis.
    scores = torch.sort(torch.randn(L, K, K), dim=-1, descending=True).values
    cand = torch.arange(K, dtype=torch.long).repeat(L, 1)
    logp = torch.log_softmax(scores.float(), dim=-1)
    draft = build_tree_naive(scores, cand, B, L)
    got = _paths_from_draft(draft)
    expected = _exact_top_b(_all_prefixes(K, L), logp, B)
    assert got == expected


def test_build_tree_naive_clamps_at_all_prefixes():
    torch.manual_seed(0)
    L, K, B = 3, 2, 30
    scores = torch.randn(L, K, K)
    cand = torch.arange(K, dtype=torch.long).repeat(L, 1)
    draft = build_tree_naive(scores, cand, B, L)
    assert draft.token_ids.shape == (14,)
    assert set(draft.depths.tolist()) == {1, 2, 3}
    assert sorted(_paths_from_draft(draft)) == sorted(_all_prefixes(K, L))
