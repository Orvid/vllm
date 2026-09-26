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
    # Only populated by the batched triton builder: shape (R,) int32 cuda.
    num_nodes: torch.Tensor | None = None


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


def _best_first_perm(logp: torch.Tensor) -> torch.Tensor:
    """Per-row best-first successor order: ``perm[d, p, r]`` is the original
    candidate index whose transition log-probability is the r-th largest in row
    (d, p).

    Builds the order by removing the row argmax K times (left-most ties),
    mirroring the masked-argmax pass in ``_tree_builder_kernel`` so the two
    implementations cannot re-order the same row differently.
    """
    remaining = logp.float().clone()
    perm = torch.empty_like(logp, dtype=torch.long)
    for r in range(logp.shape[-1]):
        best = torch.argmax(remaining, dim=-1, keepdim=True)
        perm[..., r : r + 1] = best
        remaining.scatter_(-1, best, float("-inf"))
    return perm


def _paths_to_draft(paths: list[tuple[int, ...]], tokens: torch.Tensor) -> TreeDraft:
    n = len(paths)
    token_ids = torch.zeros(n, dtype=torch.long)
    depths = torch.zeros(n, dtype=torch.long)
    parents = torch.full((n,), -1, dtype=torch.long)
    arena = torch.arange(n)
    index: dict[tuple[int, ...], int] = {(): -1}
    for i, path in enumerate(paths):
        depth = len(path)
        prank = path[-2] if depth >= 2 else 0
        token_ids[i] = tokens[depth - 1, prank, path[-1]]
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
            candidate rank) in any order. Each row is re-ordered best-first
            internally so the lazy next-sibling advance stays monotone in mass.
        cand: [L, K] successor token ids per depth; the k-th candidate of a
            (depth, predecessor) row is ``cand[depth, k]``.
        B: number of emitted nodes (the implicit depth-0 anchor is excluded).
        L: depth cap; downgoing nodes past ``L`` are never expanded.

    Returns:
        A TreeDraft in emit order with all depths >= 1, parents[i] < i, and
        ancestor words packed over the arena. Tokens are read back through the
        per-row best-first permutation, so ``cand`` need not be pre-sorted.
        On already-best-first rows the permutation is the identity and the
        result matches the historical behaviour.

    """
    K = scores.shape[1]
    logp = torch.log_softmax(scores.float(), dim=-1)
    perm = _best_first_perm(logp)
    # Heap works over sorted positions; tokens are the original-indexed cands.
    sorted_logp = torch.gather(logp, -1, perm)
    tokens = torch.gather(
        cand[:, None, :].expand(logp.shape[0], K, K).contiguous(), -1, perm
    )
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
                sib_mass = parent_mass + float(sorted_logp[depth - 1, prank, rank])
                heapq.heappush(heap, (-sib_mass, -parent_mass, parent_path + (rank,)))
        if depth >= L:
            continue
        # best child of this node
        c_rank = 0
        child_mass = mass + float(sorted_logp[depth, path[-1] if path else 0, c_rank])
        heapq.heappush(heap, (-child_mass, -mass, path + (c_rank,)))
    return _paths_to_draft(emitted, tokens)
