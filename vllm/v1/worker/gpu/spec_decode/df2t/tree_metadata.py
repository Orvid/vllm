# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TreeDraft: per-request tree node SoA plus a torch reference builder."""

import heapq
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from vllm.v1.outputs import StashedTree


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

    def to_lists(self) -> StashedTree:
        """Portable python-list form for the scheduler hop (not GPU tensors)."""
        num_nodes = int(self.num_nodes.item()) if self.num_nodes is not None else None
        return StashedTree(
            token_ids=self.token_ids.tolist(),
            depths=self.depths.tolist(),
            parents=self.parents.tolist(),
            arena=self.arena.tolist(),
            words=self.words.tolist(),
            num_nodes=num_nodes,
        )

    @classmethod
    def from_lists(
        cls, stashed: StashedTree, device: str | torch.device
    ) -> "TreeDraft":
        """Re-materialize the tensors back on ``device`` (int64, num_nodes i32)."""
        num_nodes = (
            torch.tensor([stashed.num_nodes], dtype=torch.int32, device=device)
            if stashed.num_nodes is not None
            else None
        )
        return cls(
            token_ids=torch.tensor(stashed.token_ids, dtype=torch.long, device=device),
            depths=torch.tensor(stashed.depths, dtype=torch.long, device=device),
            parents=torch.tensor(stashed.parents, dtype=torch.long, device=device),
            arena=torch.tensor(stashed.arena, dtype=torch.long, device=device),
            words=torch.tensor(stashed.words, dtype=torch.long, device=device),
            num_nodes=num_nodes,
        )

    @classmethod
    def from_lists_batched(
        cls,
        stashed: "Sequence[StashedTree | None]",
        device: str | torch.device,
    ) -> "list[TreeDraft | None]":
        """Rebuild a batch of stashed trees with one batched host->device copy.

        Replaces a loop of per-request :meth:`from_lists` (which lands ``R * 6``
        tiny ``torch.tensor(list, device=device)`` H2D copies) with a single
        ``[R, W, 5]`` int64 transfer plus one ``[R]`` int32 ``num_nodes``
        transfer, then per-request strided views sliced to each tree's own live
        width. The rebuilt tensors are bit-identical to the per-request loop
        (same values, dtypes, and per-request ``token_ids.shape[0]`` == the
        stashed width); requests without a tree stay ``None``, as do ``None``
        ``num_nodes`` values.

        Args:
            stashed: ``[R]`` per-request ``StashedTree`` in batch order.
            device: Target device of the batched copy.

        """
        R = len(stashed)
        widths = [len(s.token_ids) if s is not None else 0 for s in stashed]
        W = max(widths) if widths else 0
        if W == 0:
            return [None] * R
        has_nn = [bool(s is not None and s.num_nodes is not None) for s in stashed]
        nn = np.zeros(R, dtype=np.int32)
        arr = np.zeros((R, W, 5), dtype=np.int64)
        for r, s in enumerate(stashed):
            if s is None:
                continue
            n = widths[r]
            arr[r, :n, 0] = s.token_ids
            arr[r, :n, 1] = s.depths
            arr[r, :n, 2] = s.parents
            arr[r, :n, 3] = s.arena
            arr[r, :n, 4] = s.words
            nn[r] = int(s.num_nodes) if has_nn[r] else n
        arr_t = torch.from_numpy(arr).to(device)
        nn_t = torch.from_numpy(nn).to(device)
        out: list[TreeDraft | None] = []
        for r in range(R):
            n = widths[r]
            if n == 0:
                out.append(None)
                continue
            out.append(
                cls(
                    token_ids=arr_t[r, :n, 0],
                    depths=arr_t[r, :n, 1],
                    parents=arr_t[r, :n, 2],
                    arena=arr_t[r, :n, 3],
                    words=arr_t[r, :n, 4],
                    num_nodes=nn_t[r : r + 1] if has_nn[r] else None,
                )
            )
        return out


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
