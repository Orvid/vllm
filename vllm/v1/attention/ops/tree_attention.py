# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative tree verification for attention backends without a mask hook.

Every backend reads one per-row structure, ``TreeVerifyContext.row_mask``:
bit ``d`` of a row's mask is set iff the row attends the key ``d`` slots
before its own slot, and keys 64 or more slots back are always visible. A
tree row sits at its arena slot but at its anchor's position plus its depth;
``row_pos_delta`` is the difference, for position-keyed sliding windows.

Backends use it in one of three ways:

- a kernel that masks per key (Triton unified attention) tests the bits;
- a kernel with a relative attention bias (b12x) gets a ``-inf``-like bias on
  the hidden distances (:func:`fill_tree_relative_bias`);
- a kernel that can only run causal or full attention over a paged cache
  (FlashAttention 2/3) splits the verify of each tree request: its own kernel
  attends every row to the request's context with no mask, and
  :func:`tree_local_attention` attends each row to the request's own rows
  under the mask and merges the two by their log-sum-exp. Requests without
  a tree run the backend's causal kernel.
"""

import torch

from vllm.triton_utils import tl, triton

# Keys a row mask covers: a tree holds at most 64 nodes.
TREE_MASK_SPAN = 64
# Query rows a request may have for the split path to verify it: its own rows
# are attended in one tile. A tree verify block is at most 65 rows.
TREE_LOCAL_ROWS = 128
# The masked bias: far below any real logit, finite so a split-KV chunk that
# only holds hidden keys merges with zero weight instead of NaN.
TREE_HIDDEN_BIAS = -30000.0


def check_tree_attention_layer(
    sliding_window: tuple[int, int] | None,
    alibi_slopes: torch.Tensor | None,
    *,
    windowed: bool = True,
) -> None:
    """Fail closed on a layer whose attention a tree verify would get wrong.

    Args:
        sliding_window: The layer's ``(left, right)`` window, ``(-1, -1)`` or
            None for full attention.
        alibi_slopes: ALiBi slopes, which are keyed by slot, not position.
        windowed: Whether the backend applies a positional window to tree
            rows. A tree row's window must also reach every ancestor, so the
            window has to exceed the tree span.

    """
    if alibi_slopes is not None:
        raise NotImplementedError(
            "speculative tree verification does not support ALiBi: its bias "
            "is keyed by slot, and a tree row's slot is not its position."
        )
    if sliding_window is None or sliding_window[0] < 0:
        return
    if not windowed:
        raise NotImplementedError(
            "this attention backend cannot apply a sliding window to "
            "speculative tree rows (their window is keyed by position, not "
            "slot); use FlashAttention-4, Triton or Flex attention."
        )
    if sliding_window[0] + 1 <= TREE_MASK_SPAN:
        raise NotImplementedError(
            f"speculative tree verification needs a sliding window above "
            f"{TREE_MASK_SPAN} tokens, got {sliding_window[0] + 1}."
        )


@triton.jit
def _fill_tree_relative_bias_kernel(
    bias_ptr,  # [rows, H, E]
    row_mask_ptr,  # [>= rows] int64
    num_heads,
    E: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    word = tl.load(row_mask_ptr + row)
    d = tl.arange(0, E)
    visible = (d >= 64) | (((word >> tl.minimum(d, 63).to(tl.int64)) & 1) != 0)
    values = tl.where(visible, 0.0, HIDDEN)
    for h0 in range(0, num_heads, BLOCK_H):
        h = h0 + tl.arange(0, BLOCK_H)
        ptrs = bias_ptr + (row * num_heads + h[:, None]) * E + d[None, :]
        tl.store(
            ptrs,
            tl.broadcast_to(values[None, :], (BLOCK_H, E)).to(
                bias_ptr.dtype.element_ty
            ),
            mask=h[:, None] < num_heads,
        )


def fill_tree_relative_bias(bias: torch.Tensor, row_mask: torch.Tensor) -> None:
    """Write a relative attention bias for ``bias.shape[0]`` rows.

    Args:
        bias: ``[rows, num_heads, extent]`` contiguous, indexed by how many
            slots a key lies before the row's own slot; zero where the row
            sees the key, :data:`TREE_HIDDEN_BIAS` where it does not.
        row_mask: ``TreeVerifyContext.row_mask``.

    """
    rows, num_heads, extent = bias.shape
    assert bias.is_contiguous() and extent == triton.next_power_of_2(extent)
    if rows == 0:
        return
    _fill_tree_relative_bias_kernel[(rows,)](
        bias,
        row_mask,
        num_heads,
        E=extent,
        HIDDEN=TREE_HIDDEN_BIAS,
        BLOCK_H=min(triton.next_power_of_2(num_heads), 16),
    )


@triton.jit
def _tree_local_context_lens_kernel(
    out_ptr,  # [num_reqs] int32
    query_start_loc_ptr,
    seq_lens_ptr,
    tree_width_ptr,
    num_reqs,
    LOCAL_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = r < num_reqs
    qs = tl.load(query_start_loc_ptr + r, mask=m, other=0)
    qe = tl.load(query_start_loc_ptr + r + 1, mask=m, other=0)
    seq_len = tl.load(seq_lens_ptr + r, mask=m, other=0)
    width = tl.load(tree_width_ptr + r, mask=m, other=0)
    query_len = qe - qs
    split = (width > 0) & (query_len <= LOCAL_ROWS)
    # Rows outside the split are discarded, but a paged kernel may fault on
    # a request with rows and no keys: give it its first key.
    ctx = tl.where(split, seq_len - query_len, tl.minimum(query_len, 1))
    tl.store(out_ptr + r, ctx.to(tl.int32), mask=m)


def tree_local_context_lens(
    out: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    tree_width: torch.Tensor,
) -> torch.Tensor:
    """Per request, the keys its rows attend without a mask in the split
    path: everything before its first row for a request verifying a tree.

    Only tree requests are split (:func:`tree_local_attention`); the others
    (prefill chunks, fresh prompts, plain decodes) run the backend's causal
    kernel. A tree always has context: its anchor follows a prompt.
    """
    num_reqs = seq_lens.shape[0]
    out = out[:num_reqs]
    if num_reqs:
        _tree_local_context_lens_kernel[(triton.cdiv(num_reqs, 256),)](
            out,
            query_start_loc,
            seq_lens,
            tree_width,
            num_reqs,
            TREE_LOCAL_ROWS,
            256,
        )
    return out


@triton.jit
def _tree_local_attention_kernel(
    out_ptr,
    out_s0,
    out_s1,
    q_ptr,
    q_s0,
    q_s1,
    k_ptr,
    k_s0,
    k_s1,
    v_ptr,
    v_s0,
    v_s1,
    ctx_out_ptr,
    ctx_out_s0,
    ctx_out_s1,
    ctx_lse_ptr,
    ctx_lse_s_head,
    ctx_lse_s_tok,
    row_mask_ptr,
    query_start_loc_ptr,
    tree_width_ptr,
    num_reqs,
    q_scale_ptr,
    scale,
    softcap,
    G: tl.constexpr,
    G_PAD: tl.constexpr,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    LOCAL_ROWS: tl.constexpr,
    REQS_PAD: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_Q_SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    # The row's request: the last start at or before it.
    reqs = tl.arange(0, REQS_PAD)
    starts = tl.load(
        query_start_loc_ptr + reqs, mask=reqs <= num_reqs, other=2147483647
    )
    req = tl.sum((starts <= row).to(tl.int32), axis=0) - 1
    if (req < 0) | (req >= num_reqs):
        return
    qs = tl.load(query_start_loc_ptr + req)
    qe = tl.load(query_start_loc_ptr + req + 1)
    if (row >= qe) | (qe - qs > LOCAL_ROWS) | (tl.load(tree_width_ptr + req) == 0):
        return

    g = tl.arange(0, G_PAD)
    dd = tl.arange(0, D_PAD)
    heads = kv_head * G + g
    gm = g < G
    dm = dd < D
    q = tl.load(
        q_ptr + row.to(tl.int64) * q_s0 + heads[:, None] * q_s1 + dd[None, :],
        mask=gm[:, None] & dm[None, :],
        other=0.0,
    )
    if USE_Q_SCALE:
        q = (q.to(tl.float32) * tl.load(q_scale_ptr)).to(k_ptr.dtype.element_ty)

    back = tl.arange(0, LOCAL_ROWS)
    key_rows = row - back
    word = tl.load(row_mask_ptr + row)
    bit = (word >> tl.minimum(back, 63).to(tl.int64)) & 1
    visible = (key_rows >= qs) & ((back >= 64) | (bit != 0))
    kv_rows = tl.maximum(key_rows, 0).to(tl.int64)
    k = tl.load(
        k_ptr + kv_rows[:, None] * k_s0 + kv_head * k_s1 + dd[None, :],
        mask=visible[:, None] & dm[None, :],
        other=0.0,
    )
    v = tl.load(
        v_ptr + kv_rows[:, None] * v_s0 + kv_head * v_s1 + dd[None, :],
        mask=visible[:, None] & dm[None, :],
        other=0.0,
    )

    s = tl.dot(q, tl.trans(k)) * scale
    if USE_SOFTCAP:
        s = softcap * (2.0 / (1.0 + tl.exp(-2.0 * s / softcap)) - 1.0)
    s = tl.where(visible[None, :], s, float("-inf"))
    m = tl.max(s, axis=1)
    p = tl.exp(s - m[:, None])
    l_local = tl.sum(p, axis=1)
    acc = tl.dot(p.to(v.dtype), v)

    lse = tl.load(
        ctx_lse_ptr + heads * ctx_lse_s_head + row.to(tl.int64) * ctx_lse_s_tok,
        mask=gm,
        other=float("-inf"),
    )
    # An empty context reports +inf (FlashAttention) or NaN.
    lse = tl.where((lse == float("inf")) | (lse != lse), float("-inf"), lse)
    ctx_out = tl.load(
        ctx_out_ptr
        + row.to(tl.int64) * ctx_out_s0
        + heads[:, None] * ctx_out_s1
        + dd[None, :],
        mask=gm[:, None] & dm[None, :],
        other=0.0,
    ).to(tl.float32)
    top = tl.maximum(m, lse)
    w_ctx = tl.where(lse == float("-inf"), 0.0, tl.exp(lse - top))
    w_local = tl.exp(m - top)
    merged = (ctx_out * w_ctx[:, None] + acc * w_local[:, None]) / (
        w_ctx + l_local * w_local
    )[:, None]
    tl.store(
        out_ptr + row.to(tl.int64) * out_s0 + heads[:, None] * out_s1 + dd[None, :],
        merged.to(out_ptr.dtype.element_ty),
        mask=gm[:, None] & dm[None, :],
    )


def tree_local_attention(
    out: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    ctx_out: torch.Tensor,
    ctx_lse: torch.Tensor,
    row_mask: torch.Tensor,
    query_start_loc: torch.Tensor,
    tree_width: torch.Tensor,
    scale: float,
    softcap: float = 0.0,
    q_scale: torch.Tensor | None = None,
) -> None:
    """Attend each row of a tree request to its request's own rows under
    the tree mask, and merge with its attention to the context.

    Rows of requests without a tree are left untouched in ``out``.

    Args:
        out: ``[num_tokens, num_heads, head_size]``, written per local row.
        query: ``[num_tokens, num_heads, head_size]``.
        key: ``[num_tokens, num_kv_heads, head_size]``, this step's keys.
        value: ``[num_tokens, num_kv_heads, head_size]``, this step's values.
        ctx_out: ``[num_tokens, num_heads, head_size]``, each row's attention
            to its request's context (:func:`tree_local_context_lens`).
        ctx_lse: ``[num_heads, num_tokens]`` log-sum-exp of ``ctx_out``,
            natural log; ``-inf`` or ``+inf`` for an empty context.
        row_mask: ``TreeVerifyContext.row_mask``.
        query_start_loc: ``[num_reqs + 1]`` row offsets.
        tree_width: ``TreeVerifyContext.tree_width`` (0 = no tree).
        scale: Softmax scale.
        softcap: Logit soft cap, 0 for none.
        q_scale: Dequantization scale of an FP8 ``query``.

    """
    num_tokens, num_heads, head_size = query.shape
    num_kv_heads = key.shape[1]
    group = num_heads // num_kv_heads
    num_reqs = query_start_loc.shape[0] - 1
    if num_tokens == 0 or num_reqs == 0:
        return
    assert ctx_lse.shape[0] == num_heads and ctx_lse.shape[1] >= num_tokens
    _tree_local_attention_kernel[(num_tokens, num_kv_heads)](
        out,
        out.stride(0),
        out.stride(1),
        query,
        query.stride(0),
        query.stride(1),
        key,
        key.stride(0),
        key.stride(1),
        value,
        value.stride(0),
        value.stride(1),
        ctx_out,
        ctx_out.stride(0),
        ctx_out.stride(1),
        ctx_lse,
        ctx_lse.stride(0),
        ctx_lse.stride(1),
        row_mask,
        query_start_loc,
        tree_width,
        num_reqs,
        q_scale,
        scale,
        softcap,
        G=group,
        G_PAD=max(16, triton.next_power_of_2(group)),
        D=head_size,
        D_PAD=triton.next_power_of_2(head_size),
        LOCAL_ROWS=TREE_LOCAL_ROWS,
        REQS_PAD=triton.next_power_of_2(num_reqs + 1),
        USE_SOFTCAP=softcap > 0,
        USE_Q_SCALE=q_scale is not None,
    )
