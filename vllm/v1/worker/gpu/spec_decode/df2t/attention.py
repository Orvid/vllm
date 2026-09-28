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

from vllm.v1.attention.backends.flex_attention import (
    BlockSparsityHint,
    FlexAttentionMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.spec_decode.df2t.tree_metadata import TreeDraft


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


def build_flex_tree_mask(
    num_reqs: int,
    C_per_req: torch.Tensor,
    words: torch.Tensor,
    B: int,
    request_lookup: torch.Tensor,
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

    Returns:
        A ``(logical_mask_mod, block_sparsity_hint)`` pair for
        ``FlexAttentionMetadata``. ``logical_mask_mod`` follows the flex
        ``(b, h, q_idx, kv_idx) -> bool`` convention where ``b`` is the packed
        query index (vLLM's ``get_paged_mask_mod`` passes the raw query index,
        not the batch index) and ``q_idx``/``kv_idx`` are *logical*
        (block-slot) indices. A key is allowed iff it is context (``< C``), the
        anchor (``== C``), or a node (``C + 1 + j``) whose bit ``j`` is set in
        the query node's word. All inputs broadcast elementwise, matching the
        other flex mask mods (``causal_mask_mod`` et al).

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
        return in_context | is_anchor | node_allowed

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
        return context_or_anchor | node_visible

    return logical_mask_mod, BlockSparsityHint(hint_fn)


@cache
def _make_fa4_tree_mask_mod(B: int):
    """Build the cached FA4 CuTE-DSL ``mask_mod`` for the node budget ``B``.

    Mirrors ``_make_mm_prefix_mask_mod`` (``flash_attn.py``): FA4 reads aux
    tensors by scalar index only under cutlass-dsl 4.7.1 (2-D tensors without
    ``aux_tensor_leading_dims``; naive tensor indexing and 1-D leading-dims
    fail), so the ancestor word ``words[r, arena]`` and per-request anchor
    length ``C_per_req[r]`` are loaded through ``ssa_to_scalar``-clamped scalar
    indices, exactly like ``_load_q_range``. Cached by ``B``: FA4's
    ``hash_callable`` mixes the closure-cell repr into the compile key, so an
    uncached closure would poke a fresh function address every forward and
    force a full JIT recompile.

    ``kv_idx`` is the local KV position within the sequence, which equals the
    logical block-slot (context ``< C``, anchor ``== C``, node ``C + 1 + j``)
    while the request's KV fits within a single KV page (fastdiv divmod in the
    paged ``apply_mask`` wraps ``kv_idx`` by the page size beyond one page --
    see the task 14 report; the runner fails closed in that regime), and
    ``q_idx`` is the local verify-block query index (0 = anchor, ``1 + a``
    = node arena ``a``), so the block-slot of the query is ``C + q_idx``. A
    key is allowed iff it is context, the anchor, or a node ``j`` whose bit is
    set in the query node's ancestor-or-self word.
    """
    import cutlass
    import cutlass.cute as cute

    from vllm.vllm_flash_attn.cute.utils import (  # type: ignore[import-untyped]
        scalar_to_ssa,
        ssa_to_scalar,
    )

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
        return scalar_to_ssa(in_context | is_anchor | node_allowed, cutlass.Int32)

    fa4_tree_mask_mod.use_fast_sampling = True
    fa4_tree_mask_mod.__vec_size__ = 1  # scalar reads take lane 0 of q/kv_idx
    return fa4_tree_mask_mod


def build_fa4_tree_mask(
    num_reqs: int,
    C_per_req: torch.Tensor,
    words: torch.Tensor,
    B: int,
) -> tuple[Callable, list[torch.Tensor]]:
    """Build the df2t tree verify FA4 ``mask_mod`` and its aux tensors.

    Args:
        num_reqs: Number of requests in the batch.
        C_per_req: [num_reqs] int32 anchor context length (block-slot ``C``)
            per request, read by scalar index in the mask_mod.
        words: [num_reqs, B] int64 ancestor-or-self bitmask over arena
            indices (bit ``j`` set iff node ``j`` is an ancestor or itself of
            the query node), read by scalar index in the mask_mod.
        B: Node-count ceiling (the width of ``words``).

    Returns:
        A ``(mask_mod, aux_tensors)`` pair for the FA4 varlen paged forward.
        ``mask_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info,
        aux_tensors)`` applies the same ancestor-or-self semantics as
        ``build_flex_tree_mask`` under the FA4 coordinate conventions (see
        ``_make_fa4_tree_mask_mod``): ``batch_idx`` is the request index,
        ``q_idx`` is the local verify-block query index, ``kv_idx`` is the
        logical block-slot (exact only while each request's KV fits within one
        KV page; the runner fails closed beyond that, see
        ``GPUModelRunner._attach_tree_fa4_mask``). ``aux_tensors[0]`` is
        ``words`` and ``aux_tensors[1]`` is ``C_per_req``.

    """
    assert C_per_req.shape == (num_reqs,)
    assert words.shape == (num_reqs, B)
    return _make_fa4_tree_mask_mod(B), [words, C_per_req]


def attach_tree_flex_mask(
    metadata: FlexAttentionMetadata,
    num_reqs: int,
    C_per_req: torch.Tensor,
    words: torch.Tensor,
    B: int,
) -> None:
    """Attach the df2t tree mask to a built ``FlexAttentionMetadata``.

    Sets ``logical_mask_mod`` and ``block_sparsity_hint`` from
    ``build_flex_tree_mask`` (using the metadata's own ``doc_ids`` as the
    request lookup) and rebuilds ``mask_mod`` and the block mask so the next
    forward runs tree-apportioned attention.

    Args:
        metadata: A fully built ``FlexAttentionMetadata`` whose ``doc_ids``
            host the packed query index -> request lookup.
        num_reqs: Number of requests in the batch.
        C_per_req: [num_reqs] per-request anchor context length.
        words: [num_reqs, B] per-request ancestor-or-self words.
        B: Node-count ceiling (width of ``words``).

    """
    mask_mod, hint = build_flex_tree_mask(
        num_reqs, C_per_req, words, B, metadata.doc_ids
    )
    metadata.logical_mask_mod = mask_mod
    metadata.block_sparsity_hint = hint
    metadata.mask_mod = metadata.get_mask_mod()
    if metadata.direct_build:
        metadata.block_mask = metadata._build_block_mask_direct()
    else:
        metadata.block_mask = metadata.build_block_mask()
