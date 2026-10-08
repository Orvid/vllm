# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tree verification for attention that reads explicit KV index lists.

Sparse and sliding-window attention backends (DeepSeek-V4's SWA, sparse MLA)
take, per query row, a list of KV slots to attend to. A tree row's list must
hold its context and its own ancestors, never its siblings: a row at depth
``d`` under an anchor at sequence position ``C`` sits at position ``C + d``,
and its causal window over positions ``[C + d - W + 1, C + d]`` resolves the
context part (positions ``<= C``) through the request's block table and the
in-tree part (``C + 1 .. C + d``) through the rows of its path, whose KV went
to their own linear slots (``slot_mapping``) during the verify forward.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _tree_window_indices_kernel(
    indices_ptr,  # [num_rows, width] int32 KV slot per entry, -1 past the length
    indices_stride,
    lens_ptr,  # [num_rows] int32
    row_depth_ptr,  # [max_tokens] int32 (0 = not a tree node)
    row_anchor_ptr,  # [max_tokens] int32
    path_rows_ptr,  # [max_tokens, D] int32
    slot_mapping_ptr,  # [num_tokens] the rows' own KV slots
    query_start_loc_ptr,  # [R + 1] int32
    seq_lens_ptr,  # [R] int32
    token_to_req_ptr,  # [num_tokens] int32
    block_table_ptr,  # [R, max_blocks] int32
    block_table_stride,
    block_size,
    window,
    replay_start_ptr,  # [R] int32 lowest readable position (0 = none)
    width,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    DP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    depth = tl.load(row_depth_ptr + row)
    if depth == 0:
        return
    anchor = tl.load(row_anchor_ptr + row).to(tl.int64)
    req = tl.load(token_to_req_ptr + row).to(tl.int64)
    qs = tl.load(query_start_loc_ptr + req)
    qe = tl.load(query_start_loc_ptr + req + 1)
    seq_len = tl.load(seq_lens_ptr + req)
    # The anchor's (linear) sequence position: the request's committed length.
    c = seq_len - (qe - qs) + (anchor - qs)
    pos = c + depth
    start = tl.maximum(pos - window + 1, 0)
    start = tl.maximum(start, tl.load(replay_start_ptr + req))
    n_ctx = tl.maximum(c - start + 1, 0)
    # Context entries, positions start .. c, through the block table.
    for off in range(0, width, BLOCK):
        i = off + tl.arange(0, BLOCK)
        p = start + i
        live = i < n_ctx
        blk = tl.load(
            block_table_ptr + req * block_table_stride + p // block_size, mask=live
        )
        slot = tl.where(live, blk * block_size + p % block_size, -1)
        tl.store(indices_ptr + row * indices_stride + i, slot, mask=i < width)
    # The path entries overwrite padding the loop above just wrote.
    tl.debug_barrier()
    # In-tree entries: the path rows' own slots (depths 1 .. depth).
    k = tl.arange(0, DP)
    on_path = k < depth
    path = tl.load(path_rows_ptr + row * D + k, mask=on_path, other=0)
    slots = tl.load(slot_mapping_ptr + path, mask=on_path, other=-1)
    dst = n_ctx + k
    tl.store(
        indices_ptr + row * indices_stride + dst,
        slots.to(tl.int32),
        mask=on_path & (dst < width),
    )
    tl.store(lens_ptr + row, tl.minimum(n_ctx + depth, width).to(tl.int32))


def fill_tree_window_indices(
    indices: torch.Tensor,
    lens: torch.Tensor,
    tree_verify,
    num_rows: int,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    window: int,
    replay_start: torch.Tensor,
) -> None:
    """Rewrite the KV index lists of the tree node rows among the first
    ``num_rows`` rows (others keep their causal lists) to their sliding
    ``window`` over context and path.

    Args:
        indices: ``[num_rows, width]`` (any trailing unit dims flattened) KV
            slot lists, -1 past each row's length.
        lens: ``[num_rows]`` list lengths.
        tree_verify: The batch's ``TreeVerifyContext``.
        num_rows: Leading rows to rewrite (the decode rows).
        slot_mapping: The rows' own KV slots in this cache.
        query_start_loc: ``[R + 1]`` row offsets per request.
        seq_lens: ``[R]`` sequence length per request, this step's rows
            included.
        token_to_req: Each row's request.
        block_table: This cache's block table.
        block_size: This cache's block size.
        window: Positions each row sees, itself included.
        replay_start: ``[R]`` lowest readable position per request.

    """
    if num_rows == 0:
        return
    flat = indices.view(indices.shape[0], -1)
    d = tree_verify.path_rows.shape[1]
    _tree_window_indices_kernel[(num_rows,)](
        flat,
        flat.stride(0),
        lens,
        tree_verify.row_depth,
        tree_verify.row_anchor,
        tree_verify.path_rows,
        slot_mapping,
        query_start_loc,
        seq_lens,
        token_to_req,
        block_table,
        block_table.stride(0),
        block_size,
        window,
        replay_start,
        flat.shape[1],
        D=d,
        BLOCK=triton.next_power_of_2(min(flat.shape[1], 1024)),
        DP=triton.next_power_of_2(d),
    )
