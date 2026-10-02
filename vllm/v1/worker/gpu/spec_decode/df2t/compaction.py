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
from typing import Any

import torch
from torch import Tensor

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import get_dtype_size

_CopyBlock = 256


@triton.jit
def _gather_kernel(
    src_off_ptr,  # [a] int64 flat source element offsets
    scratch_ptr,  # [a, NUM_EL]
    kv_ptr,  # flat per-layer cache [num_slots * NUM_EL]
    a,
    NUM_EL: tl.constexpr,  # elements per slot (num_heads * content)
    BLOCK: tl.constexpr,  # copy granularity
):
    node = tl.program_id(0).to(tl.int64)
    base_src = tl.load(src_off_ptr + node)
    offs = tl.arange(0, BLOCK)
    for start in tl.range(0, NUM_EL, BLOCK):
        inc = start + offs
        valid = inc < NUM_EL
        val = tl.load(kv_ptr + base_src + inc, mask=valid, other=0)
        tl.store(scratch_ptr + node * NUM_EL + inc, val, mask=valid)


@triton.jit
def _scatter_kernel(
    dst_off_ptr,  # [a] int64 flat destination element offsets
    scratch_ptr,  # [a, NUM_EL]
    kv_ptr,  # flat per-layer cache [num_slots * NUM_EL]
    a,
    NUM_EL: tl.constexpr,  # elements per slot (num_heads * content)
    BLOCK: tl.constexpr,  # copy granularity
):
    node = tl.program_id(0).to(tl.int64)
    base_dst = tl.load(dst_off_ptr + node)
    offs = tl.arange(0, BLOCK)
    for start in tl.range(0, NUM_EL, BLOCK):
        inc = start + offs
        valid = inc < NUM_EL
        val = tl.load(scratch_ptr + node * NUM_EL + inc, mask=valid, other=0)
        tl.store(kv_ptr + base_dst + inc, val, mask=valid)


@triton.jit
def _zero_kernel(
    off_ptr,  # [a] int64 flat element offsets of stale source slots
    stale_ptr,  # [a] int1: 1 when the source is stale and must be zeroed
    kv_ptr,  # flat per-layer cache [num_slots * NUM_EL]
    a,
    NUM_EL: tl.constexpr,  # elements per slot (num_heads * content)
    BLOCK: tl.constexpr,  # copy granularity
):
    node = tl.program_id(0).to(tl.int64)
    do_zero = tl.load(stale_ptr + node)
    base = tl.load(off_ptr + node)
    offs = tl.arange(0, BLOCK)
    for start in tl.range(0, NUM_EL, BLOCK):
        inc = start + offs
        tl.store(kv_ptr + base + inc, 0, mask=(inc < NUM_EL) & do_zero)


@triton.jit
def _plan_group_kernel(
    num_sampled_ptr,  # [R] int32 per-request emitted-token count
    has_tree_ptr,  # [R] int8 1 when the request carries a tree
    arenas_ptr,  # [R, 1 + W] int64 accepted arena per chain depth (-1 = none)
    depth_ptr,  # [R, W] int64 per-arena tree node depth (0 = padding)
    c_ptr,  # [R] int64 anchor block-slot C per request
    bt_ptr,  # [R, nb] int64 per-group block table
    out_src,  # [G, max_a] int64 source phys slot per entry
    out_dst,  # [G, max_a] int64 destination phys slot per entry
    out_enter,  # [G, max_a] int8 1 when the entry must move
    out_stale,  # [G, max_a] int8 1 when the source is stale and must be zeroed
    a,  # R * W live entry count
    W,  # accepted-chain width (plan columns per request)
    nb,  # blocks per request row
    bs,  # tokens per block
    g,  # this kernel's kv-cache group index
    astride,  # row stride of accepted_arenas (1 + W)
    max_a,  # persistent stride of the [G, max_a] plan buffers
):
    i = tl.program_id(0).to(tl.int64)
    r = i // W
    j = i - r * W
    n_acc = tl.load(num_sampled_ptr + r) - 1
    ht = tl.load(has_tree_ptr + r)
    valid = (j < n_acc) & (ht != 0)
    arena = tl.load(arenas_ptr + r * astride + 1 + j)
    ar = tl.maximum(arena, 0)
    depth = tl.load(depth_ptr + r * W + ar)
    c = tl.load(c_ptr + r)
    s_block = c + 1 + arena
    d_block = c + depth
    s_bidx = tl.minimum(s_block // bs, nb - 1)
    d_bidx = tl.minimum(d_block // bs, nb - 1)
    src_phys = tl.load(bt_ptr + r * nb + s_bidx) * bs + s_block % bs
    dst_phys = tl.load(bt_ptr + r * nb + d_bidx) * bs + d_block % bs
    move = src_phys != dst_phys
    stale = move & valid
    # Aliasing protection, exactly the reference's per-request semantics: a
    # source is stale only when it is not the destination of any valid
    # accepted node of the SAME request (``torch.isin(src, dst)`` over the
    # request's own accepted nodes). Cross-request physical slots cannot
    # coincide in the engine (each request owns disjoint KV blocks), so the
    # scan is bounded to this request's ``W`` columns and masked by the
    # per-column validity. O(R * W^2) per group for the whole batch.
    for k in tl.range(0, W):
        vk = (k < n_acc) & (ht != 0)
        ark = tl.maximum(tl.load(arenas_ptr + r * astride + 1 + k), 0)
        dk = tl.load(depth_ptr + r * W + ark)
        dbk = c + dk
        dbidx = tl.minimum(dbk // bs, nb - 1)
        dpk = tl.load(bt_ptr + r * nb + dbidx) * bs + dbk % bs
        stale = stale & ((src_phys != dpk) | (vk == 0))
    base = g * max_a
    tl.store(out_src + base + i, src_phys)
    tl.store(out_dst + base + i, dst_phys)
    tl.store(out_enter + base + i, (valid & move).to(tl.int8))
    tl.store(out_stale + base + i, stale.to(tl.int8))


@triton.jit
def _gather_all_kernel(
    kv_base,  # first layer's cache view (base pointer of the shared backing)
    layer_off_ptr,  # [L] int64 element offset of each layer from kv_base
    elems_ptr,  # [L] int32 elements per slot per layer (num_heads * content)
    group_of_ptr,  # [L] int32 kv-cache group each layer belongs to
    src_ptr,  # [G, a] int64 source phys slot per (group, entry)
    enter_ptr,  # [G, a] int8: 1 when the entry must move
    scratch_ptr,  # [SLAB, L, MAX_EL] per-(entry, layer) scratch
    slab_off,  # global entry offset of this slab launch
    a,  # live entry count reaching into this slab
    g_stride,  # row stride of the [G, a] plan buffers
    L,  # number of layer views
    MAX_EL: tl.constexpr,  # scratch element capacity per (entry, layer)
    BLOCK: tl.constexpr,  # copy granularity
):
    i = tl.program_id(0).to(tl.int64)
    layer = tl.program_id(1)
    g = tl.load(group_of_ptr + layer)
    el = tl.load(elems_ptr + layer).to(tl.int64)
    entry = slab_off + i
    # ``a`` is this slab's live entry count, so ``i < a`` is the correct
    # in-slab / in-batch guard (``entry < a`` would compare a global entry
    # against a per-slab count and erase every slab after the first). It also
    # keeps the plan loads inside [G, max_a]: ``entry < slab_off + a <= a_batch
    # <= g_stride``.
    in_plan = i < a
    do = tl.load(enter_ptr + g * g_stride + entry, mask=in_plan, other=0)
    phys = tl.load(src_ptr + g * g_stride + entry, mask=in_plan, other=0)
    base_off = tl.load(layer_off_ptr + layer) + phys * el
    offs = tl.arange(0, BLOCK)
    for start in tl.range(0, el, BLOCK):
        inc = start + offs
        valid = inc < el
        val = tl.load(kv_base + base_off + inc, mask=valid, other=0)
        tl.store(
            scratch_ptr + (i * L + layer) * MAX_EL + inc,
            val,
            mask=valid & (do != 0) & in_plan,
        )


@triton.jit
def _scatter_zero_all_kernel(
    kv_base,  # first layer's cache view (base pointer of the shared backing)
    layer_off_ptr,  # [L] int64 element offset of each layer from kv_base
    elems_ptr,  # [L] int32 elements per slot per layer
    group_of_ptr,  # [L] int32 kv-cache group each layer belongs to
    dst_ptr,  # [G, a] int64 destination phys slot per (group, entry)
    src_ptr,  # [G, a] int64 source phys slot per (group, entry)
    stale_ptr,  # [G, a] int8: 1 when the source is stale and must be zeroed
    enter_ptr,  # [G, a] int8: 1 when the entry must move
    scratch_ptr,  # [SLAB, L, MAX_EL] per-(entry, layer) scratch
    slab_off,  # global entry offset of this slab launch
    a,  # live entry count reaching into this slab
    g_stride,  # row stride of the [G, a] plan buffers
    L,  # number of layer views
    MAX_EL: tl.constexpr,  # scratch element capacity per (entry, layer)
    BLOCK: tl.constexpr,  # copy granularity
):
    i = tl.program_id(0).to(tl.int64)
    layer = tl.program_id(1)
    g = tl.load(group_of_ptr + layer)
    el = tl.load(elems_ptr + layer).to(tl.int64)
    entry = slab_off + i
    idx = g * g_stride + entry
    # Same in-slab guard as the gather kernel (``a`` is this slab's live
    # count): never touch entries of later slabs or read the plan buffers past
    # the batch on the final partial slab.
    in_plan = i < a
    do = tl.load(enter_ptr + idx, mask=in_plan, other=0)
    stale = tl.load(stale_ptr + idx, mask=in_plan, other=0)
    base_dst = (
        tl.load(layer_off_ptr + layer)
        + tl.load(dst_ptr + idx, mask=in_plan, other=0) * el
    )
    base_src = (
        tl.load(layer_off_ptr + layer)
        + tl.load(src_ptr + idx, mask=in_plan, other=0) * el
    )
    offs = tl.arange(0, BLOCK)
    for start in tl.range(0, el, BLOCK):
        inc = start + offs
        valid = (inc < el) & in_plan
        val = tl.load(scratch_ptr + (i * L + layer) * MAX_EL + inc, mask=valid, other=0)
        tl.store(kv_base + base_dst + inc, val, mask=valid & (do != 0))
        tl.store(kv_base + base_src + inc, 0, mask=valid & (stale != 0))


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

    Scattered and canonical slots of different accepted nodes may coincide (a
    node whose canonical slot ``C + depth`` is another node's scattered slot
    ``C + 1 + arena``), so the copies go through a scratch buffer (all sources
    are read before any destination is written) and only sources that are not
    any destination are zeroed — otherwise the stale-slot zero would clobber an
    accepted node's relocated KV.

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
    move = src_phys != dst_phys
    if not bool(move.any().item()):
        # Every accepted node already lives at its canonical slot.
        return
    stale = move & ~torch.isin(src_phys, dst_phys)

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
        # Read every source before any destination write so overlapping
        # scattered/canonical slots cannot read a half-written slot.
        scratch = torch.empty(a, elems, dtype=layer.dtype, device=layer.device)
        grid = (a,)
        _gather_kernel[grid](
            src_off,
            scratch,
            layer,
            a,
            NUM_EL=elems,
            BLOCK=_CopyBlock,
        )
        _scatter_kernel[grid](
            dst_off,
            scratch,
            layer,
            a,
            NUM_EL=elems,
            BLOCK=_CopyBlock,
        )
        _zero_kernel[grid](
            src_off,
            stale,
            layer,
            a,
            NUM_EL=elems,
            BLOCK=_CopyBlock,
        )


_ScratchBudgetBytes = 256 * 1024 * 1024


def _align8(nbytes: int) -> int:
    return (nbytes + 7) & ~7


def _buffer_layout(
    *,
    G: int,
    max_num_reqs: int,
    max_a: int,
    W: int,
    L: int,
    max_el: int,
    itemsize: int,
    slab: int,
) -> tuple[dict[str, tuple[int, int]], int]:
    """Byte offsets/sizes of every persistent compactor buffer, 8-byte aligned.

    Returns a ``{name: (byte_offset, byte_size)}`` map and the total byte size.
    The layout is the single source of truth shared by the profiling-phase
    reservation (:meth:`Df2tKVCompaction.reserved_bytes`), the fresh
    allocations, and the carved-into-storage construction, so the reserved
    size and the runtime buffers can never disagree.
    """
    layout: dict[str, tuple[int, int]] = {}
    off = 0
    for name, numel, size in (
        ("group_of", L, 4),
        ("elems", L, 4),
        ("layer_off", L, 8),
        ("src", G * max_a, 8),
        ("dst", G * max_a, 8),
        ("enter", G * max_a, 1),
        ("stale", G * max_a, 1),
        ("tree_depth", max_num_reqs * W, 8),
        ("has_tree", max_num_reqs, 1),
        ("scratch", slab * L * max_el, itemsize),
    ):
        off = _align8(off)
        nbytes = numel * size
        layout[name] = (off, nbytes)
        off += nbytes
    return layout, off


class Df2tKVCompaction:
    """Batched, host-sync-free compaction of one step's accepted chains.

    Vectorizes what ``compact_accepted_chain`` does per (request, group,
    layer) into a fixed-shape per-step plan (built by one small triton kernel
    per kv-cache group) and two fixed-shape triton launches (one gather, one
    fused scatter + stale-zero) that cover every layer of every group in a
    single grid. All per-(request, group) host syncs and the many-small-launch
    fan-out of the correctness-first wiring are gone; the whole step is
    device-only and every buffer below is persistent (refreshed in place,
    never allocated on the hot path) — the pattern the CUDA-graph
    infrastructure primed.

    Per step, :meth:`compact` is called with the verifier's accepted-chain
    output, per-request tree depths/block tables, and builds the group-major
    plan ``[G, R * W]`` (src/dst slots, enter/stale masks), then launches
    ``_gather_all_kernel`` and ``_scatter_zero_all_kernel`` over a ``(slab,
    L)`` grid. Every accepted node's KV is relocated from its scattered
    ``C + 1 + arena`` slot to the canonical ``C + depth`` slot and the stale
    (non-aliased) sources are zeroed, with the same aliasing protection the
    reference primitive guarantees: within a slab every source is read before
    any destination is written, and a source that is some same-request
    destination is never zeroed. A node whose canonical slot already equals
    its scattered slot is left byte-for-byte untouched.

    Slabs tile at whole-request boundaries (a slab is a multiple of the chain
    width ``W``), so a request's aliased src/dst pairs always stay inside one
    gather -> scatter slab. The stale scan is per-request (matching the
    reference's per-request ``torch.isin`` exactly): a source is considered
    stale when it is not the destination of any valid accepted node of the
    same request. Cross-request physical slots cannot coincide in the engine
    (each request is allocated its own KV blocks), so no cross-request
    aliasing can arise; if that invariant ever broke, the move would be
    handled per-slab anyway (both requests share a slab only when their
    entries do, and slabs hold whole requests).

    Layer addressing uses per-layer element offsets measured from one base
    view's ``data_ptr`` at init; it does not assume the layers share a single
    backing allocation, only per-slot-contiguous (LBNHC) layers, which the
    stride guard below enforces.
    """

    _ScratchBudgetBytes = _ScratchBudgetBytes

    def __init__(
        self,
        max_num_reqs: int,
        W: int,
        views_by_name: dict[str, Tensor],
        kv_cache_groups: Sequence[Any],
        device: torch.device,
        storage: Tensor | None = None,
    ):
        """Persist the layer metadata and plan/scratch buffers.

        Args:
            max_num_reqs: Upper bound on requests in a step; sizes the runtime
                plan rows (``max_num_reqs * W``).
            W: Accepted-chain width (``TreeVerifier.accepted_arenas`` has
                ``1 + W`` columns); one plan column per chain depth.
            views_by_name: The runner's per-layer KV cache views
                (``_kv_cache_views_by_name``), ``[B, H, N, C]`` in the LBNHC
                layout.
            kv_cache_groups: The runner's ``kv_cache_config.kv_cache_groups``;
                ``group.layer_names`` lists the group's layers and
                ``group.kv_cache_spec.block_size`` its block size.
            device: The device the views live on.
            storage: Optional flat ``uint8`` GPU buffer to carve every
                persistent plan/scratch/metadata tensor out of. The memory
                profiling phase allocates exactly :meth:`reserved_bytes` and
                hands it over here, so the compactor's buffers are part of the
                GPU reservation instead of being lazily allocated after the KV
                cache filled the budget (a post-profile OOM). When ``None`` the
                buffers are allocated fresh (unit tests).

        """
        self._W = W
        self._max_a = max_num_reqs * W
        self._device = device
        self.block_sizes: list[int] = []
        layers: list[tuple[int, Tensor]] = []
        for g, group in enumerate(kv_cache_groups):
            bs = getattr(group.kv_cache_spec, "block_size", None)
            if bs is None:
                raise ValueError(
                    "df2t compaction needs a uniform block_size on every kv-cache group"
                )
            if not isinstance(bs, int):
                # UniformTypeKVCacheSpecs resolves to an int block_size.
                bs = int(bs)
            self.block_sizes.append(bs)
            for name in group.layer_names:
                layer = views_by_name.get(name)
                if layer is None:
                    continue
                if layer.ndim != 4:
                    raise ValueError(
                        "df2t compaction needs 4D [B, H, N, C] per-layer "
                        f"caches; got shape {tuple(layer.shape)}"
                    )
                _, num_heads, tokens_per_block, content = layer.shape
                elems = num_heads * content
                if layer.stride() != (
                    tokens_per_block * elems,
                    content,
                    elems,
                    1,
                ):
                    raise ValueError(
                        "df2t compaction requires per-slot-contiguous "
                        f"(LBNHC) layer caches; got strides {layer.stride()}"
                    )
                layers.append((g, layer))

        self._G = len(kv_cache_groups)
        self._L = len(layers)
        if self._L == 0:
            self._empty = True
            return
        self._empty = False
        if any(layer.device.type != device.type for _, layer in layers):
            raise ValueError("df2t compaction views must live on one device")
        dtype = layers[0][1].dtype
        if any(layer.dtype != dtype for _, layer in layers):
            raise ValueError(
                "df2t compaction requires one dtype across the layer caches; "
                f"got {dtype} and mixed dtypes"
            )

        base_layer = min(layers, key=lambda gl: gl[1].data_ptr())[1]
        itemsize = base_layer.element_size()
        layer_info = []
        for idx, (g, layer) in enumerate(layers):
            _, num_heads, _, content = layer.shape
            layer_info.append((g, num_heads, content, layer))
        self._max_el = max(
            num_heads * content for _, num_heads, content, _ in layer_info
        )
        self._kv_base = base_layer
        self._base_ptr = base_layer.data_ptr()

        # Persistent plan + scratch. Rows [G, max_a]; details via g_stride.
        max_a = self._max_a
        row_bytes = self._L * self._max_el * itemsize
        # Precise scratch sizing: by default the scratch covers the whole
        # ``max_num_reqs`` batch in ONE slab (a single ``(slab, L)``-grid
        # gather -> scatter pair per step), sized from the model's real
        # ``L * MAX_EL * itemsize`` per-slot bytes and the ``max_num_reqs``-
        # scaled plan -- never a blanket constant. ``_ScratchBudgetBytes`` only
        # caps the scratch on unusually large (rows x per-slot bytes) models;
        # the capped slab still tiles at whole-request boundaries (a
        # W-multiple) so a request's aliased src/dst pairs never straddle two
        # gather->scatter slabs.
        scratch_slots = max_a
        if max_a * row_bytes > self._ScratchBudgetBytes:
            raw_slab = max(1, self._ScratchBudgetBytes // row_bytes)
            scratch_slots = min((raw_slab // W) * W, max_a)
            if scratch_slots < W:
                scratch_slots = W
        self._slab = scratch_slots
        self._layout, self._reserved_bytes = _buffer_layout(
            G=self._G,
            max_num_reqs=max_num_reqs,
            max_a=max_a,
            W=W,
            L=self._L,
            max_el=self._max_el,
            itemsize=itemsize,
            slab=scratch_slots,
        )
        self._storage: Tensor | None
        if storage is not None:
            if storage.numel() < self._reserved_bytes:
                raise ValueError(
                    "df2t compaction storage reservation is too small: "
                    f"{storage.numel()} bytes available, "
                    f"{self._reserved_bytes} required. Rebuild the memory-"
                    "profiling reservation from the real KV cache layout."
                )
            if storage.device.type != device.type or (
                device.index is not None and storage.device.index != device.index
            ):
                raise ValueError(
                    "df2t compaction storage must live on the compactor's "
                    f"device {device}, got {storage.device}"
                )
            self._storage = storage
        else:
            self._storage = None

        def carve(name: str, shape: tuple[int, ...], buf_dtype: torch.dtype) -> Tensor:
            off, nbytes = self._layout[name]
            if self._storage is None:
                return torch.zeros(shape, dtype=buf_dtype, device=device)
            raw = self._storage[off : off + nbytes].view(buf_dtype)
            return raw.view(shape).zero_()

        # carves mirror torch.zeros exactly (fully initialized persistent
        # buffers), so carved storage is bit-identical to fresh allocations.
        self._group_of = carve("group_of", (self._L,), torch.int32)
        self._elems = carve("elems", (self._L,), torch.int32)
        self._layer_off = carve("layer_off", (self._L,), torch.int64)
        for idx, (g, num_heads, content, _) in enumerate(layer_info):
            self._group_of[idx] = g
            self._elems[idx] = num_heads * content
            self._layer_off[idx] = (
                layer_info[idx][3].data_ptr() - base_layer.data_ptr()
            ) // itemsize
        self._src = carve("src", (self._G, max_a), torch.int64)
        self._dst = carve("dst", (self._G, max_a), torch.int64)
        self._enter = carve("enter", (self._G, max_a), torch.int8)
        self._stale = carve("stale", (self._G, max_a), torch.int8)
        self._tree_depth = carve("tree_depth", (max_num_reqs, W), torch.int64)
        self._has_tree = carve("has_tree", (max_num_reqs,), torch.bool)
        self._scratch = carve(
            "scratch",
            (scratch_slots * self._L * self._max_el,),
            base_layer.dtype,
        )
        self._g_stride = max_a

    @staticmethod
    def _buffer_layout_bytes(
        *,
        G: int,
        max_num_reqs: int,
        max_a: int,
        W: int,
        L: int,
        max_el: int,
        itemsize: int,
        slab: int,
    ) -> int:
        """Total bytes of every persistent compactor buffer, in ``_layout``
        order. Shared by the profiling reservation (via :meth:`reserved_bytes`)
        and :meth:`__init__`, so the two can never disagree."""
        return _buffer_layout(
            G=G,
            max_num_reqs=max_num_reqs,
            max_a=max_a,
            W=W,
            L=L,
            max_el=max_el,
            itemsize=itemsize,
            slab=slab,
        )[1]

    @classmethod
    def reserved_bytes(
        cls,
        *,
        max_num_reqs: int,
        nodes_max: int,
        kv_cache_groups: Sequence[Any],
        scratch_budget_bytes: int | None = None,
    ) -> int:
        """Precise GPU bytes the compactor will hold, from KV cache group specs.

        Mirrors :meth:`__init__`'s sizing from the model's real per-slot bytes
        (``num_heads * content`` per layer, the itemsize, and the layer count)
        and the ``max_num_reqs``-scaled plan, with no KV cache tensors needed,
        so the memory-profiling phase can reserve the exact amount before the
        KV cache is allocated. Attention-like layers contribute their real
        ``elems = num_heads * state_content_bytes // itemsize``; layers without
        a derivable layout contribute zero, and :meth:`__init__` fails closed
        if a real view would need more than the reservation.

        Args:
            max_num_reqs: Upper bound on requests in a step.
            nodes_max: The tree node budget ``W`` (one plan column per accepted
                chain depth; the TreeVerifier's ``B``).
            kv_cache_groups: The engine's ``kv_cache_groups``
                (``KVCacheGroupSpec`` list); each ``group.layer_names`` counts
                toward ``L`` and ``group.kv_cache_spec`` provides the per-layer
                element count and dtype.
            scratch_budget_bytes: Upper cap on the scratch; defaults to
                :attr:`_ScratchBudgetBytes`.

        """
        budget = (
            scratch_budget_bytes
            if scratch_budget_bytes is not None
            else cls._ScratchBudgetBytes
        )
        L = 0
        max_el = 0
        itemsize = 0
        for group in kv_cache_groups:
            spec = group.kv_cache_spec
            num_heads = getattr(spec, "num_heads", None)
            content_bytes = getattr(spec, "state_content_size_bytes", None)
            dtype = getattr(spec, "dtype", None)
            if num_heads is None or content_bytes is None or dtype is None:
                # Not an attention-like layer the compactor can relocate.
                continue
            el = num_heads * (content_bytes // get_dtype_size(dtype))
            if el <= 0:
                continue
            L += len(group.layer_names)
            max_el = max(max_el, el)
            itemsize = get_dtype_size(dtype)
        if L == 0:
            return 0
        max_a = max_num_reqs * nodes_max
        row_bytes = L * max_el * itemsize
        slab = max_a
        if max_a * row_bytes > budget:
            raw_slab = max(1, budget // row_bytes)
            slab = min((raw_slab // nodes_max) * nodes_max, max_a)
            if slab < nodes_max:
                slab = nodes_max
        return cls._buffer_layout_bytes(
            G=len(kv_cache_groups),
            max_num_reqs=max_num_reqs,
            max_a=max_a,
            W=nodes_max,
            L=L,
            max_el=max_el,
            itemsize=itemsize,
            slab=slab,
        )

    def _build_tree_depths(self, trees: Sequence[Any], R: int) -> Tensor:
        td = self._tree_depth[:R].zero_()
        for r, tree in enumerate(trees):
            if tree is None:
                continue
            d = tree.depths
            n = min(d.numel(), self._W)
            if n:
                td[r, :n].copy_(d[:n])
        return td

    def compact(
        self,
        num_sampled: Tensor,
        accepted_arenas: Tensor,
        trees: Sequence[Any],
        num_computed: Tensor,
        block_tables: Sequence[Tensor],
        R: int,
    ) -> None:
        """Relocate one step's accepted chains for ``R`` requests.

        No host syncs and no allocation happen here; inactive entries are
        masked out in the kernels. The per-group phys slots are derived from
        the block tables with the LBNHC
        ``phys = block * block_size + slot % block_size`` mapping, mirrored
        from the wiring.

        Args:
            num_sampled: ``[R]`` int32 emitted-token count per request (the
                verifier's ``num_sampled``).
            accepted_arenas: ``[max_reqs, 1 + W]`` int64 accepted arena per
                chain depth per request (``-1`` = none).
            trees: Per-request ``TreeDraft`` (or ``None``); provides each
                request's node depths indexed by arena.
            num_computed: ``[R]`` int32 anchor block-slot ``C`` per request
                (already gathered by the caller through ``idx_mapping``).
            block_tables: Per-group block tables ``[R, num_blocks]``.
            R: Number of requests.

        """
        if self._empty or R == 0:
            return
        if self._kv_base.data_ptr() != self._base_ptr:
            raise RuntimeError(
                "df2t compaction backing KV cache moved after the compactor "
                "was built; re-create the compactor"
            )
        W = self._W
        a = R * W
        has_tree = self._has_tree[:R].zero_()
        for r, tree in enumerate(trees[:R]):
            if tree is not None:
                has_tree[r] = True
        depth = self._build_tree_depths(trees, R)
        astride = accepted_arenas.shape[1]
        for g, bt in enumerate(block_tables):
            _plan_group_kernel[(a,)](
                num_sampled,
                has_tree,
                accepted_arenas,
                depth,
                num_computed,
                bt,
                self._src,
                self._dst,
                self._enter,
                self._stale,
                a,
                W,
                bt.shape[1],
                self.block_sizes[g],
                g,
                astride,
                self._g_stride,
            )

        g_stride = self._g_stride
        L = self._L
        for so in range(0, a, self._slab):
            a_so = min(a - so, self._slab)
            grid = (self._slab, L)
            _gather_all_kernel[grid](
                self._kv_base,
                self._layer_off,
                self._elems,
                self._group_of,
                self._src,
                self._enter,
                self._scratch,
                so,
                a_so,
                g_stride,
                L,
                MAX_EL=self._max_el,
                BLOCK=_CopyBlock,
            )
            _scatter_zero_all_kernel[grid](
                self._kv_base,
                self._layer_off,
                self._elems,
                self._group_of,
                self._dst,
                self._src,
                self._stale,
                self._enter,
                self._scratch,
                so,
                a_so,
                g_stride,
                L,
                MAX_EL=self._max_el,
                BLOCK=_CopyBlock,
            )
