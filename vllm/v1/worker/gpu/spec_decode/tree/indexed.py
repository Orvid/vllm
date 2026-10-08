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

Chunk-compressed caches (one entry per ``R`` positions, completed by the row
at position ``R * b + R - 1``) need one more rule: a tree completes an entry
at each boundary depth on every branch, so each node at a boundary keeps its
entry in a per-node **scratch entry** past anything the step otherwise
touches (:func:`tree_scratch_entry`), and a row's in-tree entries are those
of its path. Compression ratios too large to give every node its own entry
are handled by stopping a request's walk short of the depth completing one
of their entries (:func:`tree_accept_depth_cap`).
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def tree_scratch_entry(c, arena, max_nodes, R: tl.constexpr):
    """Scratch compressed-entry index of node ``arena`` of a tree whose
    anchor sits at position ``c`` (entries are per ``R`` positions, the tree
    has at most ``max_nodes`` nodes): past every entry the step's own rows
    (linear positions up to ``c + max_nodes``) map to."""
    return (c + 1 + max_nodes) // R + 1 + arena


def tree_accept_depth_cap(
    anchors: torch.Tensor,
    compress_ratios: tuple[int, ...],
    scratch_ratios: tuple[int, ...],
    max_nodes: int,
    max_model_len: int,
) -> torch.Tensor | None:
    """Deepest depth each request's tree walk may accept (``[R]``), or None
    without compressed caches.

    A tree completing an entry of a ratio without scratch would share it
    between siblings, and so would one whose scratch entries run past the
    request's blocks (``max_model_len``): the walk then stops short of the
    first depth completing such an entry. The rows past it compute but are
    never accepted, and the next step's anchor writes the entry for real.

    Args:
        anchors: ``[R]`` anchor positions.
        compress_ratios: The target's compression ratios.
        scratch_ratios: Those keeping in-tree entries in scratch.
        max_nodes: Most nodes a tree can have (fixes the scratch base).
        max_model_len: The model's context length.

    """
    cap = None
    c = anchors.to(torch.int64)
    for r in compress_ratios:
        first = torch.remainder(-(c + 1), r)
        first = torch.where(first == 0, r, first)
        limit = first - 1
        if r in scratch_ratios:
            last_entry = (c + 1 + max_nodes) // r + 1 + max_nodes
            fits = (last_entry + 1) * r <= max_model_len
            limit = torch.where(fits, torch.iinfo(torch.int64).max, limit)
        cap = limit if cap is None else torch.minimum(cap, limit)
    return cap


@triton.jit
def _tree_compressed_slots_kernel(
    slots_ptr,  # [num_tokens] int64 compressed slot each row writes (-1 = none)
    positions_ptr,  # [num_tokens] int64 rows' sequence positions
    row_depth_ptr,
    row_anchor_ptr,
    token_to_req_ptr,
    block_table_ptr,  # [R, max_blocks] int32 (blocks of entries)
    block_table_stride,
    block_size,  # entries per block
    max_blocks,
    max_nodes,
    RATIO: tl.constexpr,
    SCRATCH: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    depth = tl.load(row_depth_ptr + row)
    if depth == 0:
        return
    anchor = tl.load(row_anchor_ptr + row).to(tl.int64)
    pos = tl.load(positions_ptr + row)
    slot = tl.full((), -1, tl.int64)
    if SCRATCH:  # noqa: SIM102 (a constexpr branch)
        if (pos + 1) % RATIO == 0:
            c = tl.load(positions_ptr + anchor)
            entry = tree_scratch_entry(c, row - anchor - 1, max_nodes, RATIO)
            # Scratch past the request's blocks (near max_model_len) is not
            # written; the walk never accepts this row then.
            if entry // block_size < max_blocks:
                req = tl.load(token_to_req_ptr + row).to(tl.int64)
                blk = tl.load(
                    block_table_ptr + req * block_table_stride + entry // block_size
                )
                slot = blk.to(tl.int64) * block_size + entry % block_size
    tl.store(slots_ptr + row, slot)


def apply_tree_compressed_slots(
    slots: torch.Tensor,
    tree_verify,
    num_rows: int,
    positions: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    ratio: int,
    max_nodes: int,
    scratch: bool,
) -> None:
    """Point the compressed-entry writes of the tree rows among the first
    ``num_rows`` rows at their nodes' scratch entries (``scratch``), or else
    at nothing (no accepted tree row completes an entry of a ratio without
    scratch: the walk stops short of it, and the next step's anchor writes
    it). Other rows keep their positional slots.

    Args:
        slots: ``[num_tokens]`` compressed slot per row, -1 = no write.
        tree_verify: The batch's ``TreeVerifyContext``.
        num_rows: Leading rows to consider.
        positions: Rows' sequence positions.
        token_to_req: Each row's request.
        block_table: The compressed cache's block table.
        block_size: Entries per block.
        ratio: Positions per entry.
        max_nodes: Most nodes a tree can have (fixes the scratch base).
        scratch: Whether this ratio keeps in-tree entries in scratch.

    """
    if num_rows == 0:
        return
    _tree_compressed_slots_kernel[(num_rows,)](
        slots,
        positions,
        tree_verify.row_depth,
        tree_verify.row_anchor,
        token_to_req,
        block_table,
        block_table.stride(0),
        block_size,
        block_table.shape[1],
        max_nodes,
        RATIO=ratio,
        SCRATCH=scratch,
    )


@triton.jit
def tree_path_entry_slot(
    k,  # [N] which in-tree entry on the row's path (0 = shallowest)
    valid,  # [N] bool
    row,
    anchor,
    c,  # anchor position
    req,
    path_rows_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    max_blocks,
    max_nodes,
    D: tl.constexpr,
    RATIO: tl.constexpr,
):
    """Slot of the ``k``-th compressed entry a tree row's path completes
    inside the tree: the entry of its node at boundary depth ``first + k *
    RATIO`` (``first`` the shallowest depth whose position ends an entry),
    kept in that node's scratch entry. Scratch past the request's blocks
    (rows the walk never accepts) reads slot 0."""
    first = (RATIO - (c + 1) % RATIO) % RATIO
    first = tl.where(first == 0, RATIO, first)
    depth_k = first + k * RATIO
    node = tl.load(
        path_rows_ptr + row * D + depth_k - 1, mask=valid & (depth_k <= D), other=0
    ).to(tl.int64)
    entry = tree_scratch_entry(c, node - anchor - 1, max_nodes, RATIO)
    blk = tl.load(
        block_table_ptr + req * block_table_stride + entry // block_size,
        mask=valid & (entry // block_size < max_blocks),
        other=0,
    ).to(tl.int64)
    return blk * block_size + entry % block_size


@triton.jit
def _tree_remap_compressed_kernel(
    global_ptr,  # [num_rows, width] int32 global slots (rewritten in place)
    global_stride,
    local_ptr,  # [num_rows, width] int32 local compressed indices
    local_stride,
    width,
    positions_ptr,
    row_depth_ptr,
    row_anchor_ptr,
    path_rows_ptr,
    token_to_req_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    max_blocks,
    max_nodes,
    D: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    depth = tl.load(row_depth_ptr + row)
    if depth == 0:
        return
    anchor = tl.load(row_anchor_ptr + row).to(tl.int64)
    c = tl.load(positions_ptr + anchor)
    n_ctx = (c + 1) // RATIO
    req = tl.load(token_to_req_ptr + row).to(tl.int64)
    for off in range(0, width, BLOCK):
        i = off + tl.arange(0, BLOCK)
        live = i < width
        local = tl.load(local_ptr + row * local_stride + i, mask=live, other=-1)
        in_tree = live & (local >= n_ctx)
        slot = tree_path_entry_slot(
            local - n_ctx,
            in_tree,
            row,
            anchor,
            c,
            req,
            path_rows_ptr,
            block_table_ptr,
            block_table_stride,
            block_size,
            max_blocks,
            max_nodes,
            D,
            RATIO,
        )
        tl.store(global_ptr + row * global_stride + i, slot.to(tl.int32), mask=in_tree)


def remap_tree_compressed_indices(
    global_indices: torch.Tensor,
    local_indices: torch.Tensor,
    tree_verify,
    num_rows: int,
    positions: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    ratio: int,
    max_nodes: int,
) -> None:
    """Point the tree rows' selected in-tree compressed entries (local index
    ``>= (C + 1) // ratio``, the ``k``-th entry its path completes inside the
    tree) at their nodes' scratch slots, after the local indices were mapped
    to global slots positionally.

    Args:
        global_indices: ``[num_rows, width]`` global slots, rewritten.
        local_indices: ``[num_rows, width]`` local indices they came from.
        tree_verify: The batch's ``TreeVerifyContext``.
        num_rows: Leading rows to consider.
        positions: Rows' sequence positions.
        token_to_req: Each row's request.
        block_table: The compressed cache's block table.
        block_size: Entries per block.
        ratio: Positions per entry.
        max_nodes: Most nodes a tree can have (fixes the scratch base).

    """
    if num_rows == 0:
        return
    g = global_indices.view(global_indices.shape[0], -1)
    loc = local_indices.view(local_indices.shape[0], -1)
    width = g.shape[1]
    d = tree_verify.path_rows.shape[1]
    _tree_remap_compressed_kernel[(num_rows,)](
        g,
        g.stride(0),
        loc,
        loc.stride(0),
        width,
        positions,
        tree_verify.row_depth,
        tree_verify.row_anchor,
        tree_verify.path_rows,
        token_to_req,
        block_table,
        block_table.stride(0),
        block_size,
        block_table.shape[1],
        max_nodes,
        D=d,
        RATIO=ratio,
        BLOCK=triton.next_power_of_2(min(width, 1024)),
    )


@triton.jit
def _tree_row_seq_lens_kernel(
    seq_lens_ptr, positions_ptr, row_depth_ptr, num_rows, BLOCK: tl.constexpr
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = rows < num_rows
    depth = tl.load(row_depth_ptr + rows, mask=live, other=0)
    pos = tl.load(positions_ptr + rows, mask=live, other=0)
    tl.store(seq_lens_ptr + rows, (pos + 1).to(tl.int32), mask=live & (depth > 0))


def apply_tree_row_seq_lens(
    seq_lens: torch.Tensor, tree_verify, positions: torch.Tensor, num_rows: int
) -> None:
    """Set the per-row causal lengths (``[num_rows]``, row order) of the tree
    rows among the first ``num_rows`` rows to their true position + 1: a
    row's length counts its tree depth, not its place in the verify block,
    so position-derived counts (e.g. completed compressed entries) cover the
    context plus exactly the entries on its own path."""
    if num_rows == 0:
        return
    _tree_row_seq_lens_kernel[(triton.cdiv(num_rows, 256),)](
        seq_lens, positions, tree_verify.row_depth, num_rows, BLOCK=256
    )


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


@triton.jit
def _tree_indexer_logits_kernel(
    logits_ptr,  # [num_rows, width] fp32
    logits_stride,
    q_ptr,  # [num_tokens, H, HEAD] fp8
    q_stride0,
    q_stride1,
    weights_ptr,  # [num_tokens, H] fp32
    weights_stride,
    k_cache_ptr,  # uint8 blocks: block_size * HEAD fp8 values, then fp32 scales
    kv_block_stride,
    kv_block_size,
    positions_ptr,
    row_depth_ptr,
    row_anchor_ptr,
    path_rows_ptr,
    token_to_req_ptr,
    block_table_ptr,
    block_table_stride,
    max_blocks,
    max_nodes,
    H: tl.constexpr,
    HP: tl.constexpr,
    HEAD: tl.constexpr,
    D: tl.constexpr,
    RATIO: tl.constexpr,
    MAX_K: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    depth = tl.load(row_depth_ptr + row)
    if depth == 0:
        return
    anchor = tl.load(row_anchor_ptr + row).to(tl.int64)
    c = tl.load(positions_ptr + anchor)
    n_ctx = (c + 1) // RATIO
    n_path = (tl.load(positions_ptr + row) + 1) // RATIO - n_ctx
    req = tl.load(token_to_req_ptr + row).to(tl.int64)
    h = tl.arange(0, HP)
    d = tl.arange(0, HEAD)
    q = tl.load(
        q_ptr + row * q_stride0 + h[:, None] * q_stride1 + d[None, :],
        mask=(h < H)[:, None],
        other=0.0,
    ).to(tl.float32)
    w = tl.load(weights_ptr + row * weights_stride + h, mask=h < H, other=0.0)
    for k in tl.static_range(MAX_K):
        if k < n_path:
            slot = tree_path_entry_slot(
                k,
                k < n_path,
                row,
                anchor,
                c,
                req,
                path_rows_ptr,
                block_table_ptr,
                block_table_stride,
                kv_block_size,
                max_blocks,
                max_nodes,
                D,
                RATIO,
            )
            base = k_cache_ptr + (slot // kv_block_size) * kv_block_stride
            pos_in_block = slot % kv_block_size
            kv = tl.load(base + pos_in_block * HEAD + d)
            kv = kv.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scale_ptr = base + kv_block_size * HEAD + pos_in_block * 4
            scale = tl.load(scale_ptr.to(tl.pointer_type(tl.float32)))
            score = tl.sum(q * kv[None, :], axis=1)
            logit = tl.sum(tl.maximum(score, 0.0) * w, axis=0) * scale
            tl.store(logits_ptr + row * logits_stride + n_ctx + k, logit)


def tree_indexer_logits(
    logits: torch.Tensor,
    q: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    tree_verify,
    num_rows: int,
    positions: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    compress_ratio: int,
    max_nodes: int,
) -> None:
    """Rescore the tree rows' in-tree compressed entries in the indexer's
    decode logits: column ``n_ctx + k`` of a tree row (``n_ctx`` its
    context's entries) is the ``k``-th entry its own path completes inside
    the tree, kept in that node's scratch entry, while the paged logits read
    whatever sits at that index linearly. Same score as the paged kernel:
    ``sum_h w_h * relu(q_h . k) * k_scale`` over the FP8 indexer cache.

    Args:
        logits: ``[num_rows, width]`` decode logits, one row per token.
        q: ``[num_tokens, H, head_dim]`` FP8 indexer queries.
        weights: ``[num_tokens, H]`` head weights (q scale folded in).
        kv_cache: FP8 indexer cache, ``[num_blocks, block_size, ...]``.
        tree_verify: The batch's ``TreeVerifyContext``.
        num_rows: Leading (decode) rows to consider.
        positions: Rows' sequence positions.
        token_to_req: Each row's request.
        block_table: The indexer cache's per-request block table.
        compress_ratio: Positions per entry.
        max_nodes: Most nodes a tree can have (fixes the scratch base).

    """
    if num_rows == 0:
        return
    assert q.dtype == torch.float8_e4m3fn and kv_cache.dtype == torch.uint8
    num_heads, head_dim = q.shape[1], q.shape[2]
    d = tree_verify.path_rows.shape[1]
    _tree_indexer_logits_kernel[(num_rows,)](
        logits,
        logits.stride(0),
        q,
        q.stride(0),
        q.stride(1),
        weights,
        weights.stride(0),
        kv_cache,
        kv_cache.stride(0),
        kv_cache.shape[1],
        positions,
        tree_verify.row_depth,
        tree_verify.row_anchor,
        tree_verify.path_rows,
        token_to_req,
        block_table,
        block_table.stride(0),
        block_table.shape[1],
        max_nodes,
        H=num_heads,
        HP=triton.next_power_of_2(num_heads),
        HEAD=head_dim,
        D=d,
        RATIO=compress_ratio,
        MAX_K=d // compress_ratio + 1,
    )
