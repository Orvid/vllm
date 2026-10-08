# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 compressor entries completed inside a speculative tree.

The compressor pools the per-token states of the ``(1 + overlap) * ratio``
positions ending at a boundary into one entry, gathering them by position.
A tree row at position ``C + d`` (anchor at ``C``) must pool its own path,
not whichever rows sit at those positions linearly: positions ``<= C`` come
from the committed state through the block table, positions ``C + 1 .. C +
d`` from its path rows' states (written to their own linear slots by the
verify forward). This rewrites the entries of the tree boundary rows after
the regular compressor ran; their compressed slots point at per-node scratch
entries (``apply_tree_compressed_slots``), so nothing else is touched. The
pooled entry is normalized, rotated, quantized and stored exactly as the
regular Triton kernels do (shared device functions).
"""

import torch

from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    _indexer_fp8_norm_rope_store,
    _indexer_mxfp4_norm_rope_store,
    _sparse_attn_norm_rope_store,
)
from vllm.triton_utils import tl, triton

_VARIANT_SPARSE_ATTN = 0
_VARIANT_INDEXER_FP8 = 1
_VARIANT_INDEXER_MXFP4 = 2


@triton.jit
def _tree_compress_kernel(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    positions_ptr,
    state_slot_mapping_ptr,  # [num_tokens] rows' own state slots
    token_to_req_ptr,
    block_table_ptr,  # state cache block table
    block_table_stride,
    block_size,
    row_depth_ptr,
    row_anchor_ptr,
    path_rows_ptr,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,  # [num_tokens] compressed slot per row (scratch)
    kv_cache_block_size,
    D: tl.constexpr,
    VARIANT: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    depth = tl.load(row_depth_ptr + row)
    if depth == 0:
        return
    if tl.load(state_slot_mapping_ptr + row) < 0:
        return
    position = tl.load(positions_ptr + row)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    kv_slot_idx = tl.load(kv_slot_mapping_ptr + row)
    if kv_slot_idx < 0:
        return
    anchor = tl.load(row_anchor_ptr + row).to(tl.int64)
    c = tl.load(positions_ptr + anchor)
    req = tl.load(token_to_req_ptr + row).to(tl.int64)

    # ── Gather the window along the row's path ─────────────────────────
    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0
    in_tree = pos > c
    # Context positions through the block table.
    ctx_blocks = tl.load(
        block_table_ptr + req * block_table_stride + pos // block_size,
        mask=mask_pos & ~in_tree,
        other=0,
    ).to(tl.int64)
    # In-tree positions c + k (k = 1 .. depth) through path row k - 1.
    k = tl.where(in_tree, pos - c - 1, 0)
    path_row = tl.load(path_rows_ptr + row * D + k, mask=in_tree, other=0)
    path_slot = tl.load(
        state_slot_mapping_ptr + path_row.to(tl.int64), mask=in_tree, other=0
    ).to(tl.int64)
    block_numbers = tl.where(in_tree, path_slot // block_size, ctx_blocks)
    block_offsets = tl.where(in_tree, path_slot % block_size, pos % block_size)
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    row_base = (
        state_cache_ptr
        + block_numbers * state_cache_stride0
        + block_offsets * state_cache_stride1
        + head_offset
    )
    combined_mask = mask_pos[:, None] & mask[None, :]
    score = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=combined_mask,
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)
    kv = tl.load(row_base[:, None] + block[None, :], mask=combined_mask, other=0.0)
    compressed_kv = tl.sum(kv * score, axis=0)

    if VARIANT == 0:
        _sparse_attn_norm_rope_store(
            compressed_kv,
            position,
            kv_slot_idx,
            rms_norm_weight_ptr,
            rms_norm_eps,
            cos_sin_cache_ptr,
            cos_sin_stride,
            k_cache_ptr,
            kv_cache_block_size,
            HEAD_SIZE,
            TRITON_BLOCK_SIZE,
            COMPRESS_RATIO,
            ROPE_HEAD_DIM,
            FP8_MAX,
            QUANT_BLOCK,
            TOKEN_STRIDE,
            SCALE_DIM,
            KV_BLOCK_STRIDE,
            False,
        )
    elif VARIANT == 1:
        _indexer_fp8_norm_rope_store(
            compressed_kv,
            position,
            kv_slot_idx,
            rms_norm_weight_ptr,
            rms_norm_eps,
            cos_sin_cache_ptr,
            cos_sin_stride,
            k_cache_ptr,
            kv_cache_block_size,
            HEAD_SIZE,
            TRITON_BLOCK_SIZE,
            COMPRESS_RATIO,
            ROPE_HEAD_DIM,
            FP8_MAX,
            QUANT_BLOCK,
            TOKEN_STRIDE,
            SCALE_DIM,
            KV_BLOCK_STRIDE,
        )
    else:
        _indexer_mxfp4_norm_rope_store(
            compressed_kv,
            position,
            kv_slot_idx,
            rms_norm_weight_ptr,
            rms_norm_eps,
            cos_sin_cache_ptr,
            cos_sin_stride,
            k_cache_ptr,
            kv_cache_block_size,
            HEAD_SIZE,
            TRITON_BLOCK_SIZE,
            COMPRESS_RATIO,
            ROPE_HEAD_DIM,
            QUANT_BLOCK,
            TOKEN_STRIDE,
            SCALE_DIM,
            KV_BLOCK_STRIDE,
        )


def tree_compress_norm_rope_store(
    tree_verify,
    num_rows: int,
    state_cache: torch.Tensor,
    positions: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    use_fp4_cache: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    """Recompute the compressed entries of the tree boundary rows among the
    first ``num_rows`` rows from their paths (see the module docstring); same
    arguments as ``compress_norm_rope_store_triton`` plus the batch's
    ``TreeVerifyContext``.

    Args:
        tree_verify: The batch's ``TreeVerifyContext``.
        num_rows: Leading rows to consider.
        state_cache: Compressor state cache.
        positions: Rows' sequence positions.
        state_slot_mapping: Rows' own state slots.
        token_to_req: Each row's request.
        block_table: State cache block table.
        block_size: State cache block size.
        state_width: Width of a state's kv half (scores follow).
        cos_sin_cache: RoPE cache.
        kv_cache: Compressed KV cache written.
        kv_slot_mapping: Compressed slot per row (scratch for tree rows).
        head_dim: Entry head size.
        rope_head_dim: Rotated part of the head.
        compress_ratio: Positions per entry.
        overlap: Whether windows overlap the previous entry.
        use_fp4_cache: MXFP4 indexer cache.
        rms_norm_weight: Entry RMSNorm weight.
        rms_norm_eps: Entry RMSNorm epsilon.
        quant_block: Quantization block.
        token_stride: Bytes per entry's values.
        scale_dim: Bytes per entry's scales.

    """
    if num_rows == 0:
        return
    if head_dim == 512:
        variant = _VARIANT_SPARSE_ATTN
    else:
        variant = _VARIANT_INDEXER_MXFP4 if use_fp4_cache else _VARIANT_INDEXER_FP8
    d = tree_verify.path_rows.shape[1]
    _tree_compress_kernel[(num_rows,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        positions,
        state_slot_mapping,
        token_to_req,
        block_table,
        block_table.stride(0),
        block_size,
        tree_verify.row_depth,
        tree_verify.row_anchor,
        tree_verify.path_rows,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        kv_cache,
        kv_slot_mapping,
        kv_cache.shape[1],
        D=d,
        VARIANT=variant,
        HEAD_SIZE=head_dim,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(head_dim),
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=rope_head_dim,
        FP8_MAX=448.0,
        QUANT_BLOCK=quant_block,
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=scale_dim,
        KV_BLOCK_STRIDE=kv_cache.stride(0),
        num_warps=4 if head_dim == 512 else 1,
    )
