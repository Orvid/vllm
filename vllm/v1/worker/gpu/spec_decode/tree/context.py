# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-agnostic tree structure of a verify batch, and of what it commits.

Every mixer that supports chain speculative decoding is causal over a prefix;
tree verification only replaces "the previous row" with "the parent row". So
the structure a layer needs is one parent pointer per batch row:

- ``row_parent[i]`` is the batch row whose state row ``i`` continues, or -1
  when it continues the request's committed state. Chain rows (prefill chunks,
  plain decodes, a tree's anchor) point at the previous row of their request;
  tree node rows point at their parent node's row (or the anchor's).
- ``tree_width[r]`` is request ``r``'s verify width (0 = no tree). A request
  with a tree keeps its committed state untouched during the forward; its
  accepted path is committed after the walk (:class:`TreeCommit`).

The buffers are persistent so FULL CUDA graphs read the current step's values
on replay. They are published to layers as ``ForwardContext.tree_verify``.
"""

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fill_row_parents_kernel(
    row_parent_ptr,  # [max_tokens] int32
    tree_width_ptr,  # [max_num_reqs] int32
    query_start_loc_ptr,  # [R + 1] int32
    parents_ptr,  # [R, W] int64 parent arena per node (-1 = the anchor)
    parents_stride,
    widths_ptr,  # [R] int64 verify width per request (0 = no tree)
    BLOCK: tl.constexpr,
    MAX_NODE: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    qs = tl.load(query_start_loc_ptr + r)
    qe = tl.load(query_start_loc_ptr + r + 1)
    for off in range(qs, qe, BLOCK):
        rows = off + tl.arange(0, BLOCK)
        parent = tl.where(rows == qs, -1, rows - 1)
        tl.store(row_parent_ptr + rows, parent.to(tl.int32), mask=rows < qe)
    n = tl.load(widths_ptr + r).to(tl.int32)
    tl.store(tree_width_ptr + r, n)
    if n > 0:
        # The verify block is the request's last 1 + n rows: anchor, nodes.
        block_start = qe - (n + 1)
        cols = tl.arange(0, MAX_NODE)
        live = cols < n
        p = tl.load(parents_ptr + r * parents_stride + cols, mask=live, other=-1)
        parent = tl.where(p < 0, block_start, block_start + 1 + p)
        tl.store(
            row_parent_ptr + block_start + 1 + cols, parent.to(tl.int32), mask=live
        )


@triton.jit
def _build_commit_kernel(
    rows_ptr,  # [R * (1 + L)] int64 committed rows, compacted
    row_pos_ptr,  # [R * (1 + L)] committed rows' sequence positions
    cu_rows_ptr,  # [R + 1] int32
    num_rows_ptr,  # [R] int32
    arenas_ptr,  # [*, arena_stride] int64 accepted arena per depth (col 0 unused)
    arena_stride,
    query_start_loc_ptr,  # [R + 1] int32
    positions_ptr,  # [num_tokens] batch rows' sequence positions
    tree_width_ptr,  # [R] int32
    num_reqs,
    L: tl.constexpr,  # max accepted depth
    BLOCK_R: tl.constexpr,
    LP: tl.constexpr,  # next power of two >= 1 + L
):
    r = tl.arange(0, BLOCK_R)
    rm = r < num_reqs
    width = tl.load(tree_width_ptr + r, mask=rm, other=0)
    qe = tl.load(query_start_loc_ptr + r + 1, mask=rm, other=0)
    block_start = qe - (width + 1)
    j = tl.arange(0, LP)
    arena = tl.load(
        arenas_ptr + r[:, None] * arena_stride + j[None, :],
        mask=rm[:, None]
        & (j[None, :] >= 1)
        & (j[None, :] <= L)
        & (j[None, :] < arena_stride),
        other=-1,
    )
    depth = tl.sum((arena >= 0).to(tl.int32), axis=1)
    n = tl.where(rm & (width > 0), 1 + depth, 0)
    cu = tl.cumsum(n, axis=0)
    start = cu - n
    tl.store(cu_rows_ptr + 1 + r, cu, mask=rm)
    tl.store(cu_rows_ptr + r, 0, mask=r == 0)
    tl.store(num_rows_ptr + r, n, mask=rm)
    row = tl.where(
        j[None, :] == 0, block_start[:, None], block_start[:, None] + 1 + arena
    )
    mask = rm[:, None] & (j[None, :] < n[:, None])
    tl.store(rows_ptr + start[:, None] + j[None, :], row.to(tl.int64), mask=mask)
    anchor_pos = tl.load(positions_ptr + block_start, mask=rm & (width > 0), other=0)
    tl.store(
        row_pos_ptr + start[:, None] + j[None, :],
        anchor_pos[:, None] + j[None, :],
        mask=mask,
    )


@dataclass
class TreeCommit:
    """The committed path of every tree request, after the walk.

    ``rows[cu_rows[r] : cu_rows[r + 1]]`` are request ``r``'s batch rows that
    become part of its sequence this step, in order: the anchor, then the
    accepted nodes from shallowest to deepest. Requests without a tree commit
    nothing here (their layers committed in the forward). Sizes are device
    values; ``rows`` is sized for the worst case, and its entries past this
    step's rows are 0 (a valid row), so any prefix of it can be gathered.
    """

    rows: torch.Tensor
    # Sequence position of each committed row (the anchor's is the request's
    # committed length C; an accepted node's is C + its depth).
    positions: torch.Tensor
    cu_rows: torch.Tensor
    num_rows: torch.Tensor
    num_reqs: int
    max_rows_per_req: int


class TreeVerifyContext:
    """Persistent per-row tree structure of the current verify batch.

    Args:
        max_num_tokens: Rows a batch can hold.
        max_num_reqs: Requests a batch can hold.
        max_depth: Deepest node a tree can have (bounds a committed path).
        device: Device of the buffers.
        max_width: Most nodes a request's tree can verify (its verify block
            is the anchor plus these); defaults to the rows a batch holds.

    """

    def __init__(
        self,
        max_num_tokens: int,
        max_num_reqs: int,
        max_depth: int,
        device: torch.device,
        max_width: int | None = None,
    ):
        self.max_num_reqs = max_num_reqs
        self.max_width = max_num_tokens - 1 if max_width is None else max_width
        # A path has at most one node per depth, so a tree deeper than it is
        # wide (a drafter depth above the node budget) is bounded by its width.
        self.max_depth = min(max_depth, self.max_width)
        self.row_parent = torch.full(
            (max_num_tokens,), -1, dtype=torch.int32, device=device
        )
        self.tree_width = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.num_reqs = 0
        # Commit buffers, written after each walk.
        per_req = 1 + self.max_depth
        self._commit_rows = torch.zeros(
            max_num_reqs * per_req, dtype=torch.int64, device=device
        )
        self._commit_positions = torch.zeros(
            max_num_reqs * per_req, dtype=torch.int64, device=device
        )
        self._cu_rows = torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device)
        self._num_rows = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    def fill(
        self,
        query_start_loc: torch.Tensor,
        num_reqs: int,
        parents: torch.Tensor,
        widths: torch.Tensor,
    ) -> None:
        """Write this step's row parents and per-request tree widths.

        Args:
            query_start_loc: ``[num_reqs + 1]`` row offsets per request.
            num_reqs: Requests in the batch.
            parents: ``[num_reqs, W]`` parent arena per node (unit column
                stride; -1 = the anchor).
            widths: ``[num_reqs]`` verify width per request (0 = no tree).

        """
        self.num_reqs = num_reqs
        if num_reqs == 0:
            return
        assert parents.stride(1) == 1
        _fill_row_parents_kernel[(num_reqs,)](
            self.row_parent,
            self.tree_width,
            query_start_loc,
            parents,
            parents.stride(0),
            widths,
            BLOCK=128,
            MAX_NODE=triton.next_power_of_2(max(parents.shape[1], 1)),
        )

    def build_commit(
        self,
        accepted_arenas: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
    ) -> TreeCommit:
        """The committed path per tree request from the walk's accepted
        arenas (``[num_reqs, 1 + B]``, column ``d`` = the node accepted at
        depth ``d``, -1 = none) and the batch rows' sequence ``positions``."""
        num_reqs = self.num_reqs
        if num_reqs:
            # Past this step's rows the buffer would keep earlier steps' rows,
            # which can lie outside a smaller batch; consumers gather a
            # worst-case prefix, so make every entry a valid row.
            self._commit_rows.zero_()
            _build_commit_kernel[(1,)](
                self._commit_rows,
                self._commit_positions,
                self._cu_rows,
                self._num_rows,
                accepted_arenas,
                accepted_arenas.stride(0),
                query_start_loc,
                positions,
                self.tree_width,
                num_reqs,
                L=self.max_depth,
                BLOCK_R=triton.next_power_of_2(num_reqs),
                LP=triton.next_power_of_2(1 + self.max_depth),
            )
        return TreeCommit(
            rows=self._commit_rows,
            positions=self._commit_positions,
            cu_rows=self._cu_rows[: num_reqs + 1],
            num_rows=self._num_rows[:num_reqs],
            num_reqs=num_reqs,
            max_rows_per_req=1 + self.max_depth,
        )
