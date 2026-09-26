# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TreeDraft: per-request tree node SoA plus a torch reference builder."""

import heapq
from dataclasses import dataclass

import torch


@dataclass
class TreeDraft:
    # All shape (B,), int64, cuda.
    token_ids: torch.Tensor
    depths: torch.Tensor
    parents: torch.Tensor  # parent arena index; root's parent = -1
    arena: torch.Tensor  # == arange(B); kept for clarity
    words: torch.Tensor  # uint64 ancestor-or-self bitmask over arena


def pack_ancestor_words(
    parents: torch.Tensor, arena: torch.Tensor, n: int
) -> torch.Tensor:
    words = torch.zeros(n, dtype=torch.int64)
    for i in range(n):
        if parents[i].item() >= 0:
            words[i] = words[parents[i].item()] | (1 << arena[i].item())
        else:
            words[i] = 1 << arena[i].item()
    return words


def _paths_to_draft(
    paths: list[tuple[int, ...]], logp: torch.Tensor, cand: torch.Tensor
) -> TreeDraft:
    n = len(paths)
    token_ids = torch.zeros(n, dtype=torch.long)
    depths = torch.zeros(n, dtype=torch.long)
    parents = torch.full((n,), -1, dtype=torch.long)
    arena = torch.arange(n)
    index: dict[tuple[int, ...], int] = {(): -1}
    for i, path in enumerate(paths):
        depth = len(path)
        token_ids[i] = cand[depth - 1, path[-1]]
        depths[i] = depth
        parents[i] = index[path[:-1]]
        index[path] = i
    words = pack_ancestor_words(parents, arena, n)
    return TreeDraft(token_ids, depths, parents, arena, words)


def build_tree_naive(
    scores: torch.Tensor, cand: torch.Tensor, B: int, L: int
) -> TreeDraft:
    """Return the top-B tree prefixes ordered by transition mass.

    Args:
        scores: [L, K, K] successor logits for (depth, predecessor rank,
            candidate rank). Candidates must be ordered best-first along the
            last axis so the lazy next-sibling advance stays monotone in mass.
        cand: [L, K] successor token ids per depth; rank columns of
            ``scores`` index into these rows.
        B: number of emitted nodes (the implicit depth-0 anchor is excluded).
        L: depth cap; downgoing nodes past ``L`` are never expanded.

    Returns:
        A TreeDraft in emit order with all depths >= 1, parents[i] < i, and
        ancestor words packed over the arena.

    """
    K = scores.shape[1]
    logp = torch.log_softmax(scores.float(), dim=-1)
    # Heap of (-mass, -parent_mass, path); parent_mass is the mass of the node
    # above ``path`` (the anchor has mass 0.0), used to price the next sibling.
    heap: list[tuple[float, float, tuple[int, ...]]] = [(0.0, 0.0, ())]
    emitted: list[tuple[int, ...]] = []
    while len(emitted) < B and heap:
        neg, neg_pm, path = heapq.heappop(heap)
        mass, parent_mass = -neg, -neg_pm
        if path:
            emitted.append(path)
        depth = len(path)
        if path:
            parent_path = path[:-1]
            prank = parent_path[-1] if parent_path else 0
            rank = path[-1] + 1
            if rank < K:
                sib_mass = parent_mass + float(logp[depth - 1, prank, rank])
                heapq.heappush(heap, (-sib_mass, -parent_mass, parent_path + (rank,)))
        if depth >= L:
            continue
        # best child of this node
        c_rank = 0
        child_mass = mass + float(logp[depth, path[-1] if path else 0, c_rank])
        heapq.heappush(heap, (-child_mass, -mass, path + (c_rank,)))
    return _paths_to_draft(emitted, logp, cand)
