# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TreeDraft: per-request tree node SoA plus a torch reference builder."""

import heapq
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.outputs import StashedTree


@dataclass
class TreeDraft:
    # All shape (B,), int64, cuda.
    token_ids: torch.Tensor
    depths: torch.Tensor
    parents: torch.Tensor  # parent arena index; root's parent = -1
    arena: torch.Tensor  # == arange(B); kept for clarity
    words: torch.Tensor  # uint64 ancestor-or-self bitmask over arena
    # Live node count, shape (R,) integer cuda (batched builder / carry only).
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
        """Rebuild a batch of stashed trees as per-request views of one
        batched host->device copy (see :meth:`TreeBatch.from_stashed`).

        Requests without a tree stay ``None``, as do ``None`` ``num_nodes``.
        """
        batch = TreeBatch.from_stashed(stashed, device)
        return batch.drafts if batch is not None else [None] * len(stashed)


@dataclass
class TreeBatch:
    """A step's df2t trees as batched ``[R, W]`` device tensors.

    One non-blocking host->device upload carries every request's tree, so the
    per-step consumers (verify-row rewrite, FA4 mask staging, verifier, KV
    compaction) work on whole-batch tensors instead of per-request slices.
    Rows of requests without a tree are inert: zero width, zero ``num_nodes``,
    parents ``-2`` and zero words.
    """

    token_ids: torch.Tensor  # [R, W] int64
    depths: torch.Tensor  # [R, W] int64
    parents: torch.Tensor  # [R, W] int64
    arena: torch.Tensor  # [R, W] int64
    words: torch.Tensor  # [R, W] int64
    # [R] int64 live node count (clamped to the width; 0 = no tree).
    num_nodes: torch.Tensor
    # [R] int64 scheduled tree width (0 = no tree), and its host mirror.
    widths: torch.Tensor
    widths_np: np.ndarray
    # Per-request views (``None`` = no tree), for the non-batched consumers.
    drafts: "list[TreeDraft | None]"

    @property
    def uniform_width(self) -> int | None:
        """The shared width when every request carries a tree of the same
        width (the uniform ``1 + B`` decode), else ``None``."""
        w = self.widths_np
        if w.size and w[0] > 0 and bool((w == w[0]).all()):
            return int(w[0])
        return None

    @classmethod
    def from_store(
        cls,
        store: torch.Tensor,
        idx_mapping: torch.Tensor,
        widths: np.ndarray,
    ) -> "TreeBatch":
        """Gather a batch from the device tree store.

        Args:
            store: ``[max_num_reqs, 4 * B + 1]`` packed rows per request state
                (tokens | depths | parents | words | num_nodes).
            idx_mapping: ``[R]`` batch -> request-state index.
            widths: ``[R]`` scheduled tree width per request (0 = no tree);
                a width below the drafted node count keeps the tree's first
                ``width`` nodes.

        """
        B = (store.shape[1] - 1) // 4
        rows = store.index_select(0, idx_mapping)
        widths = widths.astype(np.int64)
        W = int(widths.max())
        widths_t = async_tensor_h2d(widths, device=store.device)
        num_nodes = torch.minimum(rows[:, 4 * B], widths_t)
        token_ids = rows[:, :W]
        depths = rows[:, B : B + W]
        parents = rows[:, 2 * B : 2 * B + W]
        words = rows[:, 3 * B : 3 * B + W]
        arena = torch.arange(W, device=store.device).expand(rows.shape[0], W)
        drafts: list[TreeDraft | None] = []
        for r, n in enumerate(widths.tolist()):
            if n == 0:
                drafts.append(None)
                continue
            drafts.append(
                TreeDraft(
                    token_ids=token_ids[r, :n],
                    depths=depths[r, :n],
                    parents=parents[r, :n],
                    arena=arena[r, :n],
                    words=words[r, :n],
                    num_nodes=num_nodes[r : r + 1],
                )
            )
        return cls(
            token_ids=token_ids,
            depths=depths,
            parents=parents,
            arena=arena,
            words=words,
            num_nodes=num_nodes,
            widths=widths_t,
            widths_np=widths,
            drafts=drafts,
        )

    @classmethod
    def from_stashed(
        cls,
        stashed: "Sequence[StashedTree | None]",
        device: str | torch.device,
    ) -> "TreeBatch | None":
        """Upload ``[R]`` per-request ``StashedTree`` (batch order), or
        ``None`` when no request carries a tree."""
        R = len(stashed)
        widths = np.fromiter(
            (len(s.token_ids) if s is not None else 0 for s in stashed),
            dtype=np.int64,
            count=R,
        )
        W = int(widths.max()) if R else 0
        if W == 0:
            return None
        # One flat int64 upload: [5, R, W] fields, then [R] num_nodes and
        # [R] widths.
        flat = np.zeros(5 * R * W + 2 * R, dtype=np.int64)
        fields = flat[: 5 * R * W].reshape(5, R, W)
        fields[2] = -2
        nn = flat[5 * R * W : 5 * R * W + R]
        flat[5 * R * W + R :] = widths
        has_nn = np.zeros(R, dtype=bool)
        for r, s in enumerate(stashed):
            if s is None:
                continue
            n = widths[r]
            fields[0, r, :n] = s.token_ids
            fields[1, r, :n] = s.depths
            fields[2, r, :n] = s.parents
            fields[3, r, :n] = s.arena
            fields[4, r, :n] = s.words
            if s.num_nodes is not None:
                has_nn[r] = True
                nn[r] = min(max(int(s.num_nodes), 0), n)
            else:
                nn[r] = n
        flat_t = async_tensor_h2d(flat, device=device)
        fields_t = flat_t[: 5 * R * W].view(5, R, W)
        nn_t = flat_t[5 * R * W : 5 * R * W + R]
        nn32 = nn_t.to(torch.int32)
        drafts: list[TreeDraft | None] = []
        for r in range(R):
            n = int(widths[r])
            if n == 0:
                drafts.append(None)
                continue
            drafts.append(
                TreeDraft(
                    token_ids=fields_t[0, r, :n],
                    depths=fields_t[1, r, :n],
                    parents=fields_t[2, r, :n],
                    arena=fields_t[3, r, :n],
                    words=fields_t[4, r, :n],
                    num_nodes=nn32[r : r + 1] if has_nn[r] else None,
                )
            )
        return cls(
            token_ids=fields_t[0],
            depths=fields_t[1],
            parents=fields_t[2],
            arena=fields_t[3],
            words=fields_t[4],
            num_nodes=nn_t,
            widths=flat_t[5 * R * W + R :],
            widths_np=widths,
            drafts=drafts,
        )


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


def _candidate_index(path: tuple[int, ...], perm: torch.Tensor) -> int:
    """Candidate index of the node at ``path`` (best-first ranks per depth):
    each rank is resolved in the row of its parent's candidate index, the
    anchor's row being 0."""
    idx = 0
    for depth, rank in enumerate(path):
        idx = int(perm[depth, idx, rank])
    return idx


def _paths_to_draft(
    paths: list[tuple[int, ...]], tokens: torch.Tensor, perm: torch.Tensor
) -> TreeDraft:
    n = len(paths)
    token_ids = torch.zeros(n, dtype=torch.long)
    depths = torch.zeros(n, dtype=torch.long)
    parents = torch.full((n,), -1, dtype=torch.long)
    arena = torch.arange(n)
    index: dict[tuple[int, ...], int] = {(): -1}
    for i, path in enumerate(paths):
        depth = len(path)
        row = _candidate_index(path[:-1], perm)
        token_ids[i] = tokens[depth - 1, row, path[-1]]
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
        scores: [L, K, K] successor logits for (depth, predecessor candidate
            index, candidate index); the predecessor of depth 0 is the anchor
            (row 0). Each row is re-ordered best-first internally so the lazy
            next-sibling advance stays monotone in mass.
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
            prow = _candidate_index(parent_path, perm)
            rank = path[-1] + 1
            if rank < K:
                sib_mass = parent_mass + float(sorted_logp[depth - 1, prow, rank])
                heapq.heappush(heap, (-sib_mass, -parent_mass, parent_path + (rank,)))
        if depth >= L:
            continue
        # best child of this node
        c_rank = 0
        row = _candidate_index(path, perm)
        child_mass = mass + float(sorted_logp[depth, row, c_rank])
        heapq.heappush(heap, (-child_mass, -mass, path + (c_rank,)))
    return _paths_to_draft(emitted, tokens, perm)
