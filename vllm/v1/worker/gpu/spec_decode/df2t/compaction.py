# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV compaction of the accepted tree chain.

After the tree verify accepts a chain, each accepted node's K/V still lives at
its scattered tree-block slot ``C + 1 + arena`` while inference reads context
by ``position -> slot`` (slot ``C + depth``). ``compact_accepted_chain`` copies
the accepted chain's K/V to those canonical slots and zeroes the stale
scattered slots so rejected sibling K/V is never readable.
"""

from collections.abc import Sequence

from torch import Tensor

from vllm.triton_utils import tl, triton

_CopyBlock = 256


@triton.jit
def _compact_kernel(
    src_off_ptr,  # [a] int64 flat source element offsets
    dst_off_ptr,  # [a] int64 flat destination element offsets
    copy_mask_ptr,  # [a] int1: 0 when a node already lives at its canonical slot
    kv_ptr,  # flat per-layer cache [num_slots * NUM_EL]
    a,
    NUM_EL: tl.constexpr,  # elements per slot (num_heads * content)
    BLOCK: tl.constexpr,  # copy granularity
):
    node = tl.program_id(0).to(tl.int64)
    do_copy = tl.load(copy_mask_ptr + node)
    base_src = tl.load(src_off_ptr + node)
    base_dst = tl.load(dst_off_ptr + node)
    offs = tl.arange(0, BLOCK)
    for start in tl.range(0, NUM_EL, BLOCK):
        inc = start + offs
        valid = (inc < NUM_EL) & do_copy
        val = tl.load(kv_ptr + base_src + inc, mask=valid, other=0)
        tl.store(kv_ptr + base_dst + inc, val, mask=valid)


@triton.jit
def _zero_kernel(
    off_ptr,  # [a] int64 flat element offsets of the stale source slots
    copy_mask_ptr,  # [a] int1
    kv_ptr,  # flat per-layer cache [num_slots * NUM_EL]
    a,
    NUM_EL: tl.constexpr,  # elements per slot (num_heads * content)
    BLOCK: tl.constexpr,  # copy granularity
):
    node = tl.program_id(0).to(tl.int64)
    do_zero = tl.load(copy_mask_ptr + node)
    base = tl.load(off_ptr + node)
    offs = tl.arange(0, BLOCK)
    for start in tl.range(0, NUM_EL, BLOCK):
        inc = start + offs
        tl.store(kv_ptr + base + inc, 0, mask=(inc < NUM_EL) & do_zero)


def compact_accepted_chain(
    accept_arenas: Tensor,
    accept_depths: Tensor,
    slot_mapping: Tensor,
    kv_caches: Sequence[Tensor],
    C: int,
    num_accepted: int | None = None,
) -> None:
    """Relocate accepted-tree KV to canonical slots, zeroing the stale source.

    Args:
        accept_arenas: [a] int64 cuda arena indices of the accepted nodes, in
            chain order (depth 1 first).
        accept_depths: [a] int64 cuda depths of the accepted nodes.
        slot_mapping: int64 block-slot -> physical KV slot for the request's
            verify block, covering ``C + 1 + max(arena)`` and ``C + max(depth)``.
        kv_caches: per-layer KV caches, each the logical ``[num_blocks,
            num_heads, tokens_per_block, content]`` view (as
            ``create_kv_cache_views`` returns, ``content`` channels per head)
            in the LBNHC physical layout, i.e. strided ``(tokens_per_block *
            num_heads * content, content, num_heads * content, 1)``. A physical
            slot holds one contiguous ``num_heads * content`` run per layer.
        C: anchor block-slot of the request's tree verify block.
        num_accepted: number of accepted nodes; defaults to
            ``accept_arenas.shape[0]`` so padded buffers can be sliced.

    Accepted node ``j`` whose KV lives at scattered block-slot
    ``C + 1 + arena`` (Task 6 tree-verify layout) is copied to block-slot
    ``C + depth`` and its scattered slot is zeroed. A node with
    ``depth == 1 + arena`` already lives at its canonical slot and is left
    untouched. Nothing is launched (or written) when ``num_accepted == 0``.

    """
    a = int(num_accepted) if num_accepted is not None else int(accept_arenas.shape[0])
    if a == 0 or not kv_caches:
        return
    arenas = accept_arenas[:a]
    depths = accept_depths[:a]
    assert arenas.shape == depths.shape == (a,)

    c = int(C)
    src_block_slot = c + 1 + arenas
    dst_block_slot = c + depths
    src_phys = slot_mapping.gather(0, src_block_slot)
    dst_phys = slot_mapping.gather(0, dst_block_slot)
    copy_mask = src_phys != dst_phys

    for layer in kv_caches:
        if layer.ndim != 4:
            raise ValueError(
                "df2t compaction needs 4D [B, H, N, C] per-layer caches; "
                f"got shape {tuple(layer.shape)}"
            )
        _, num_heads, tokens_per_block, content = layer.shape
        elems = num_heads * content
        # Flat ``physical_slot * elems`` addressing over the raw storage is
        # valid only when a slot's KV is one contiguous elems-sized run and
        # slots sit back-to-back: the LBNHC logical-view stride pattern. Any
        # other physical layout (e.g. LBHNC, where heads are strided by N*C)
        # would read/write head-strided positions instead, so reject it.
        if layer.stride() != (tokens_per_block * elems, content, elems, 1):
            raise ValueError(
                "df2t compaction requires per-slot-contiguous (LBNHC) layer "
                f"caches; got strides {layer.stride()} for shape "
                f"{tuple(layer.shape)}"
            )
        src_off = src_phys * elems
        dst_off = dst_phys * elems
        grid = (a,)
        _compact_kernel[grid](
            src_off,
            dst_off,
            copy_mask,
            layer,
            a,
            NUM_EL=elems,
            BLOCK=_CopyBlock,
        )
        _zero_kernel[grid](
            src_off,
            copy_mask,
            layer,
            a,
            NUM_EL=elems,
            BLOCK=_CopyBlock,
        )
