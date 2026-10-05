# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlexAttention tree-apportionment mask for the df2t verify forward.

The tree verify block occupies logical slots ``C .. C + B`` of a request
(context ``< C``, anchor at ``C``, node with arena ``a`` at ``C + 1 + a``), and
the Flex logical q/kv indices equal those block slots. ``build_flex_tree_mask``
returns a ``logical_mask_mod`` closure that attends only to context + anchor +
ancestor-or-self nodes, plus a ``BlockSparsityHint`` pruning the node-region KV
blocks that a given query can never reach.
"""

from collections.abc import Callable
from functools import cache
from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.flex_attention import (
    BlockSparsityHint,
    FlexAttentionMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.df2t.tree_metadata import (
        TreeBatch,
        TreeDraft,
    )


def pack_tree_mask_inputs(
    trees: "list[TreeDraft | None] | None",
    num_reqs: int,
    B: int,
    device: torch.device,
) -> torch.Tensor:
    """Pack per-request ancestor-or-self words into a ``[num_reqs, B]`` tensor.

    Args:
        trees: [num_reqs] per-request ``TreeDraft`` (from the scheduler carry),
            or None for requests without a tree.
        num_reqs: Batch size.
        B: Node-count ceiling (``max_tree_budget``).
        device: Target device; tree words already live there.

    Returns:
        ``words`` [num_reqs, B] int64, zero for requests with no tree / fewer
        nodes (padding bits never match, so they attach no nodes).

    """
    words = torch.zeros(num_reqs, B, dtype=torch.long, device=device)
    if trees is None:
        return words
    for r, tree in enumerate(trees):
        if tree is not None:
            words[r, : tree.words.shape[0]] = tree.words
    return words


@triton.jit
def _fill_tree_mask_kernel(
    words_out_ptr,  # [max_num_reqs, OUT_W] int64
    c_out_ptr,  # [max_num_reqs] int32
    depths_out_ptr,  # [max_num_reqs, OUT_W] int32
    words_ptr,  # [num_reqs, >= width] int64, row stride ``src_stride``
    depths_ptr,  # [num_reqs, >= width] int64, row stride ``src_stride``
    src_stride,
    widths_ptr,  # [num_reqs] tree width per request (0 = no tree)
    c_ptr,  # anchor context length per request state
    idx_ptr,  # [num_reqs] batch -> c_ptr row
    num_reqs,
    OUT_W,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    active = r < num_reqs
    w = tl.load(widths_ptr + r, mask=active, other=0)
    live = cols < w
    words = tl.load(words_ptr + r * src_stride + cols, mask=live, other=0)
    depths = tl.load(depths_ptr + r * src_stride + cols, mask=live, other=0)
    state = tl.load(idx_ptr + r, mask=active, other=0)
    c = tl.load(c_ptr + state, mask=active, other=0)
    out = cols < OUT_W
    tl.store(words_out_ptr + r * OUT_W + cols, words, mask=out)
    tl.store(depths_out_ptr + r * OUT_W + cols, depths.to(tl.int32), mask=out)
    tl.store(c_out_ptr + r, c.to(tl.int32))


def fill_tree_mask_inputs_batched(
    words_out: torch.Tensor,
    c_per_req_out: torch.Tensor,
    depths_out: torch.Tensor,
    tree_batch: "TreeBatch",
    num_computed: torch.Tensor,
    idx_mapping: torch.Tensor,
) -> None:
    """One-launch :func:`fill_tree_mask_inputs` for a ``TreeBatch``: rows
    ``< num_reqs`` get each request's words/depths (zero past its width) and
    its anchor length ``num_computed[idx_mapping[r]]``; every other row of the
    persistent buffers is zeroed."""
    num_reqs = idx_mapping.shape[0]
    out_w = words_out.shape[1]
    assert words_out.is_contiguous() and depths_out.is_contiguous()
    assert depths_out.shape == words_out.shape
    assert tree_batch.words.stride(0) == tree_batch.depths.stride(0)
    _fill_tree_mask_kernel[(words_out.shape[0],)](
        words_out,
        c_per_req_out,
        depths_out,
        tree_batch.words,
        tree_batch.depths,
        tree_batch.words.stride(0),
        tree_batch.widths,
        num_computed,
        idx_mapping,
        num_reqs,
        out_w,
        BLOCK=triton.next_power_of_2(out_w),
    )


def fill_tree_mask_inputs(
    words_out: torch.Tensor,
    c_per_req_out: torch.Tensor,
    trees: "list[TreeDraft | None] | None",
    num_reqs: int,
    B: int,
    c_per_req: torch.Tensor,
    depths_out: torch.Tensor | None = None,
) -> None:
    """Fill a persistent ``[>= num_reqs, B]`` ancestor-words buffer and its
    per-request anchor-length buffer for this step, in place.

    The FA4 tree ``mask_mod`` reads its aux tensors by scalar index, so under
    CUDA graphs those aux tensors must be stable graph-input buffers whose
    contents are refreshed between replays (same addresses, new values). This
    helper writes into pre-allocated persistent buffers (``words_out`` /
    ``c_per_req_out``) instead of allocating per step.

    Args:
        words_out: [max_num_reqs, B] int64, zeroed then filled for the first
            ``num_reqs`` rows with each request's ancestor-or-self words;
            requests without a tree (or below the tree's node count) keep zero
            rows, whose bits never match.
        c_per_req_out: [max_num_reqs] int32, filled for the first
            ``num_reqs`` rows with each request's anchor context length ``C``.
        trees: [num_reqs] per-request ``TreeDraft`` or None for requests
            without a tree.
        num_reqs: Batch size (rows populated; the rest are zeroed).
        B: Node-count ceiling (``max_tree_budget``).
        c_per_req: [num_reqs] anchor context lengths (int64 or int32).
        depths_out: optional [max_num_reqs, B] node-depth buffer filled for the
            first ``num_reqs`` rows with each tree's per-arena depths (the
            windowed FA4 mask's positional-window aux); zeroed when None-able
            rows run past a tree's node count.

    """
    words_out.zero_()
    # The runner's source (req_states.num_computed_tokens.gpu) is int32, so the
    # dtype cast is a no-op self-return in the hot path and allocates only if a
    # caller ever feeds a different dtype.
    c_per_req_out[:num_reqs].copy_(c_per_req.to(c_per_req_out.dtype))
    if c_per_req_out.numel() > num_reqs:
        c_per_req_out[num_reqs:].zero_()
    if depths_out is not None:
        depths_out.zero_()
        if depths_out.numel() > num_reqs * depths_out.shape[1]:
            depths_out[num_reqs:].zero_()
    if trees is None:
        return
    for r, tree in enumerate(trees[:num_reqs]):
        if tree is not None:
            words_out[r, : tree.words.shape[0]] = tree.words
            if depths_out is not None:
                depths_out[r, : tree.depths.shape[0]] = tree.depths


def build_flex_tree_mask(
    num_reqs: int,
    C_per_req: torch.Tensor,
    words: torch.Tensor,
    B: int,
    request_lookup: torch.Tensor,
    window: int | None = None,
    depths: torch.Tensor | None = None,
) -> tuple[Callable, BlockSparsityHint]:
    """Build the df2t tree verify ``logical_mask_mod`` and block-sparsity hint.

    Args:
        num_reqs: Number of requests in the batch.
        C_per_req: [num_reqs] int anchor context length (block-slot ``C``) per
            request (the request's decode ``num_computed_tokens``).
        words: [num_reqs, B] int64 ancestor-or-self bitmask over arena indices:
            bit ``j`` of ``words[r, q]`` is set iff node ``j`` is an ancestor
            or itself of query node ``q`` of request ``r``.
        B: Node-count ceiling (the width of ``words``). Requests with fewer
            nodes keep zero padding bits, which never match.
        request_lookup: [num_tokens] int32 packed query index -> request index
            (the Flex metadata's ``doc_ids``), used to fetch per-request data.
        window: The target's sliding window; ``None`` (default) keeps the
            window-less ancestor-or-self mask (fully-causal target). A window
            composes the positional ``q_pos - k_pos < window`` term into the
            mask (positions: context ``k`` at ``k``, anchor at ``C``, node
            ``j`` at ``C + depths[j]``), so a windowed target's tree rows
            respect the target's window instead of widening to full context.
        depths: [num_reqs, B] per-request node depths, required iff ``window``
            is set.

    Returns:
        A ``(logical_mask_mod, block_sparsity_hint)`` pair for
        ``FlexAttentionMetadata``. ``logical_mask_mod`` follows the flex
        ``(b, h, q_idx, kv_idx) -> bool`` convention where ``b`` is the packed
        query index (vLLM's ``get_paged_mask_mod`` passes the raw query index,
        not the batch index) and ``q_idx``/``kv_idx`` are *logical*
        (block-slot) indices. A key is allowed iff it is context (``< C``), the
        anchor (``== C``), or a node (``C + 1 + j``) whose bit ``j`` is set in
        the query node's word -- each additionally subject to the positional
        window when one is set. A request without a tree this step (a prefill
        chunk, fresh or continuing after cached context, co-scheduled with a
        tree decode) falls back to per-request (windowed) causal attention over
        its context and its own rows. Tree presence is ``words[r, 0] != 0``:
        node 0's ancestor-or-self word always contains itself. All inputs
        broadcast elementwise, matching the other flex mask mods
        (``causal_mask_mod`` et al).

    """
    assert C_per_req.shape == (num_reqs,)
    assert words.shape == (num_reqs, B)

    def logical_mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Elementwise tree mask; ``b`` is the packed query index."""
        del h
        r = request_lookup[b]
        C = C_per_req[r]
        in_context = kv_idx < C
        is_anchor = kv_idx == C
        node = kv_idx - (C + 1)
        is_node_key = (node >= 0) & (node < B)
        node_j = torch.clamp(node, 0, 63)
        arena = torch.clamp(q_idx - (C + 1), 0, B - 1)
        is_node_query = (q_idx >= C + 1) & (q_idx < C + 1 + B)
        bit = ((words[r, arena] >> node_j) & 1) != 0
        node_allowed = is_node_query & is_node_key & bit
        # A treeless request (a prefill chunk co-scheduled with a tree decode)
        # attends causally over its context and its own rows.
        causal = (words[r, 0] == 0) & (kv_idx <= q_idx)
        keep = in_context | is_anchor | node_allowed | causal
        if window is not None:
            # Positional window term. Positions equal the block slot for
            # context/anchor/treeless rows and ``C + depth`` for tree nodes:
            # a pure slot-distance window would mis-window the depth-decoupled
            # node rows (siblings at mirrored depths).
            assert depths is not None, "a windowed flex tree mask requires depths"
            is_treeless = words[r, 0] == 0
            q_pos = torch.where(
                is_treeless,
                q_idx,
                torch.where(is_node_query, C + depths[r, arena], q_idx),
            )
            k_arena = torch.clamp(node, 0, B - 1)
            k_pos = torch.where(
                is_treeless,
                kv_idx,
                torch.where(is_node_key, C + depths[r, k_arena], kv_idx),
            )
            keep = keep & ((q_pos - k_pos) < window)
        return keep

    def hint_fn(
        q_idx: torch.Tensor,
        logical_q_idx: torch.Tensor,
        kv_block_idx: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Prune KV blocks holding no key the query could attend.

        A kv block (logical block ``K``, logical keys ``[K*block_size,
        (K+1)*block_size)``) stays visible for a query iff it carries a context
        or anchor key, or the query is a node query whose ancestor-or-self word
        has a bit set among the block's node columns. Exact: a block containing
        an allowed key is never pruned.
        """
        r = request_lookup[q_idx.long()]
        C = C_per_req[r]
        block_start = kv_block_idx * block_size
        context_or_anchor = block_start <= C[:, None]
        # Treeless requests attend causally; keep every block at or before the
        # query's block (the causal keys live there), mirroring
        # logical_mask_mod.
        treeless = (words[r, 0] == 0)[:, None] & (block_start <= logical_q_idx[:, None])
        arena = torch.clamp(logical_q_idx - (C + 1), 0, B - 1)
        is_node_query = (logical_q_idx >= C + 1) & (logical_q_idx < C + 1 + B)
        word_scalar = words[r, arena]
        node_lo = torch.clamp(block_start - (C + 1)[:, None], 0, B)
        node_hi = torch.clamp(block_start + block_size - (C + 1)[:, None], 0, B)
        # Node columns j in [node_lo, node_hi) of the block; when the whole 64
        # columns fit use all-ones so bit 63 (the int64 sign bit) stays covered.
        span = (node_hi - node_lo).clamp(0, 64)
        node_bits = torch.where(span < 64, (1 << span) - 1, -span.new_full((), 1))
        node_bits = node_bits << node_lo
        node_visible = is_node_query[:, None] & (
            (word_scalar[:, None] & node_bits) != 0
        )
        return context_or_anchor | treeless | node_visible

    return logical_mask_mod, BlockSparsityHint(hint_fn)


@cache
def _make_fa4_tree_mask_mod(B: int, window: int | None):
    """Build the cached FA4 CuTE-DSL ``mask_mod`` for the node budget ``B``.

    Mirrors ``_make_mm_prefix_mask_mod`` (``flash_attn.py``): FA4 reads aux
    tensors by scalar index only under cutlass-dsl 4.7.1 (2-D tensors without
    ``aux_tensor_leading_dims``; naive tensor indexing and 1-D leading-dims
    fail), so the ancestor word ``words[r, arena]`` and per-request anchor
    length ``C_per_req[r]`` are loaded through ``ssa_to_scalar``-clamped scalar
    indices, exactly like ``_load_q_range``. Cached by ``(B, window)``: FA4's
    ``hash_callable`` mixes the closure-cell repr into the compile key, so an
    uncached closure would poke a fresh function address every forward and
    force a full JIT recompile.

    ``kv_idx`` is the ABSOLUTE logical block-slot (context ``< C``, anchor
    ``== C``, node ``C + 1 + j``) for arbitrarily long contexts: the fork's
    Sm80-base paged forward now threads the page table into ``fastdiv_mods``
    (mirroring ``flash_fwd_sm90.py``/``flash_fwd_sm100.py``), so the paged
    mask index no longer wraps at page boundaries. ``q_idx`` is the local
    verify-block query index (0 = anchor, ``1 + a`` = node arena ``a``), so
    the block-slot of the query is ``C + q_idx``. A key is allowed iff it is
    context, the anchor, or a node ``j`` whose bit is set in the query node's
    ancestor-or-self word.

    ``window`` is the target layer's sliding window (``None`` = no window, the
    fully-causal target). With a window the mask additionally ANDs the
    *positional* sliding-window term ``q_pos - k_pos < window`` using the
    arena-decoupled positions (see ``build_flex_tree_mask``): a plain FA
    ``window_size`` would window by block-slot, which equals position only for
    context/anchor rows, not for tree nodes (node ``j`` sits at block-slot
    ``C + 1 + j`` but position ``C + depths[j]``). ``aux_tensors[2]`` holds
    ``depths`` ``[>= num_reqs, B]`` only when ``window`` is set.
    """
    import cutlass
    import cutlass.cute as cute

    from vllm.vllm_flash_attn.cute.utils import (  # type: ignore[import-untyped]
        scalar_to_ssa,
        ssa_to_scalar,
    )

    # The unwindowed mask keeps the tree mask's ancestor-or-self semantics on
    # scalar booleans exactly as before (the fully-causal target).
    @cute.jit
    def fa4_tree_mask_mod(
        batch_idx: cute.TensorSSA,
        head_idx: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors=None,
    ):
        del head_idx
        words = aux_tensors[0]
        c_per_req = aux_tensors[1]
        c = c_per_req[batch_idx[0]]
        q = ssa_to_scalar(q_idx)
        k = ssa_to_scalar(kv_idx)
        in_context = k < c
        is_anchor = k == c
        node = k - (c + 1)
        is_node_key = (node >= 0) & (node < B)
        node_j = cutlass.min(cutlass.max(node, 0), 63)
        arena = cutlass.max(q - 1, 0)
        is_node_query = q >= 1
        bit = ((words[batch_idx[0], arena] >> node_j) & 1) != 0
        node_allowed = is_node_query & is_node_key & bit
        # Treeless requests (node 0's self bit unset) attend causally over
        # their context and their own rows (query block-slot ``c + q``).
        causal = (words[batch_idx[0], 0] == 0) & (k <= c + q)
        keep = in_context | is_anchor | node_allowed | causal
        return scalar_to_ssa(keep, cutlass.Int32)

    # The windowed mask additionally ANDs the positional sliding-window term,
    # mirroring the Flex path. Positions are the block slot for
    # context/anchor/treeless rows and ``c + depth`` for tree nodes
    # (depth-decoupled from arena block slots), so the mask is built per
    # ``(B, window)`` with ``window`` baked as the compile-time bound.
    @cute.jit
    def fa4_tree_mask_mod_windowed(
        batch_idx: cute.TensorSSA,
        head_idx: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors=None,
    ):
        del head_idx
        words = aux_tensors[0]
        c_per_req = aux_tensors[1]
        depths = aux_tensors[2]
        c = c_per_req[batch_idx[0]]
        q = ssa_to_scalar(q_idx)
        k = ssa_to_scalar(kv_idx)
        in_context = k < c
        is_anchor = k == c
        node = k - (c + 1)
        is_node_key = (node >= 0) & (node < B)
        node_j = cutlass.min(cutlass.max(node, 0), 63)
        arena = cutlass.max(q - 1, 0)
        is_node_query = q >= 1
        bit = ((words[batch_idx[0], arena] >> node_j) & 1) != 0
        node_allowed = is_node_query & is_node_key & bit
        causal = (words[batch_idx[0], 0] == 0) & (k <= c + q)
        keep = in_context | is_anchor | node_allowed | causal
        # Positional window term (select-free through cute.where on tensor
        # conditions). ``is_treeless`` chooses the consecutive-positions row
        # offset (c + q); tree rows use the depth offset (c + depth).
        is_node_q_t = scalar_to_ssa(is_node_query, cutlass.Int32) != 0
        is_treeless_t = scalar_to_ssa(words[batch_idx[0], 0] == 0, cutlass.Int32) != 0
        q_pos = cute.where(
            is_treeless_t,
            scalar_to_ssa(c + q, cutlass.Int32),
            cute.where(
                is_node_q_t,
                scalar_to_ssa(c + depths[batch_idx[0], arena], cutlass.Int32),
                scalar_to_ssa(c, cutlass.Int32),
            ),
        )
        is_node_k_t = scalar_to_ssa(is_node_key, cutlass.Int32) != 0
        k_arena = cutlass.min(cutlass.max(node, 0), B - 1)
        k_pos = cute.where(
            is_treeless_t,
            scalar_to_ssa(k, cutlass.Int32),
            cute.where(
                is_node_k_t,
                scalar_to_ssa(c + depths[batch_idx[0], k_arena], cutlass.Int32),
                scalar_to_ssa(k, cutlass.Int32),
            ),
        )
        keep = (scalar_to_ssa(keep, cutlass.Int32) != 0) & ((q_pos - k_pos) < window)
        # Materialize the bool tensor as an Int32 tensor (0/1) for the kernel.
        return cute.where(
            keep,
            scalar_to_ssa(cutlass.Int32(1), cutlass.Int32),
            scalar_to_ssa(cutlass.Int32(0), cutlass.Int32),
        )

    mask_mod = fa4_tree_mask_mod if window is None else fa4_tree_mask_mod_windowed
    mask_mod.use_fast_sampling = True
    mask_mod.__vec_size__ = 1  # scalar reads take lane 0 of q/kv_idx
    return mask_mod


def build_fa4_tree_mask(
    num_reqs: int,
    C_per_req: torch.Tensor,
    words: torch.Tensor,
    B: int,
    window: int | None = None,
    depths: torch.Tensor | None = None,
) -> tuple[Callable, list[torch.Tensor]]:
    """Build the df2t tree verify FA4 ``mask_mod`` and its aux tensors.

    Args:
        num_reqs: Number of requests in the batch. ``C_per_req``/``words``/
            ``depths`` may be larger (persistent CUDA-graph aux pools sized
            ``max_num_reqs``); only the first ``num_reqs`` rows carry live
            data.
        C_per_req: [>= num_reqs] int32 anchor context length (block-slot ``C``)
            per request, read by scalar index in the mask_mod.
        words: [>= num_reqs, B] int64 ancestor-or-self bitmask over arena
            indices (bit ``j`` set iff node ``j`` is an ancestor or itself of
            the query node), read by scalar index in the mask_mod.
        B: Node-count ceiling (the width of ``words``).
        window: The target's sliding window; ``None`` (default) keeps the
            window-less ancestor-or-self mask (fully-causal target). A window
            composes the positional ``q_pos - k_pos < window`` term into the
            mask, so a windowed target no longer loses its window on tree rows.
        depths: [>= num_reqs, B] per-request node depths (position ``C +
            depths[r, arena]``), required iff ``window`` is set.

    Returns:
        A ``(mask_mod, aux_tensors)`` pair for the FA4 varlen paged forward.
        ``aux_tensors`` is ``[words, C_per_req]`` when ``window`` is None and
        ``[words, C_per_req, depths]`` when windowed, matching the mask_mod's
        aux reads.

    """
    assert C_per_req.shape[0] >= num_reqs, (C_per_req.shape, num_reqs)
    assert words.shape[0] >= num_reqs and words.shape[1] == B, (
        words.shape,
        num_reqs,
        B,
    )
    if window is None:
        return _make_fa4_tree_mask_mod(B, None), [words, C_per_req]
    assert depths is not None, "a windowed FA4 tree mask requires depths"
    assert depths.shape[0] >= num_reqs and depths.shape[1] == B, (
        depths.shape,
        num_reqs,
        B,
    )
    return _make_fa4_tree_mask_mod(B, window), [words, C_per_req, depths]


def attach_tree_flex_mask(
    metadata: FlexAttentionMetadata,
    num_reqs: int,
    C_per_req: torch.Tensor,
    words: torch.Tensor,
    B: int,
    window: int | None = None,
    depths: torch.Tensor | None = None,
) -> None:
    """Attach the df2t tree mask to a built ``FlexAttentionMetadata``.

    Sets ``logical_mask_mod`` and ``block_sparsity_hint`` from
    ``build_flex_tree_mask`` (using the metadata's own ``doc_ids`` as the
    request lookup) and rebuilds ``mask_mod`` and the block mask so the next
    forward runs tree-apportioned attention.

    When ``window`` is set the tree mask already ANDs the positional window, so
    the metadata's built-in *index-based* sliding-window term (``|q_idx -
    kv_idx| < sliding_window``, valid only where block-slot == position) is
    neutralized: composing both would double-window and mis-window the
    depth-decoupled node rows.

    Args:
        metadata: A fully built ``FlexAttentionMetadata`` whose ``doc_ids``
            host the packed query index -> request lookup.
        num_reqs: Number of requests in the batch.
        C_per_req: [num_reqs] per-request anchor context length.
        words: [num_reqs, B] per-request ancestor-or-self words.
        B: Node-count ceiling (width of ``words``).
        window: The target's sliding window (see ``build_flex_tree_mask``);
            ``None`` keeps the window-less tree mask.
        depths: [num_reqs, B] per-request node depths, required iff ``window``
            is set.

    """
    mask_mod, hint = build_flex_tree_mask(
        num_reqs,
        C_per_req,
        words,
        B,
        metadata.doc_ids,
        window=window,
        depths=depths,
    )
    metadata.logical_mask_mod = mask_mod
    metadata.block_sparsity_hint = hint
    if window is not None:
        # The tree mask carries the (positional) window itself.
        metadata.sliding_window = None
    metadata.mask_mod = metadata.get_mask_mod()
    if metadata.direct_build:
        metadata.block_mask = metadata._build_block_mask_direct()
    else:
        metadata.block_mask = metadata.build_block_mask()
