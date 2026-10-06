# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tree-verify kernels for recurrent mixers.

A tree verify batch gives every row a parent row (``row_parent``, see
``TreeVerifyContext``): a row continues the state its parent ended in, or the
request's committed state when the parent is -1 or outside the request. These
kernels compute each row as such a continuation while leaving the committed
state untouched; the accepted path is committed after the walk with the
ordinary chain kernels.

Rows are batch rows; request ``n`` owns rows
``[query_start_loc[n], query_start_loc[n + 1])`` and parents always precede
their children within it.
"""

import torch

from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _row_request(query_start_loc_ptr, row, num_seqs, BLOCK_N: tl.constexpr):
    """Index of the request owning ``row`` (``num_seqs`` for padding rows)."""
    ns = tl.arange(0, BLOCK_N)
    ends = tl.load(query_start_loc_ptr + 1 + ns, mask=ns < num_seqs, other=2**30)
    return tl.sum((ends <= row).to(tl.int32), axis=0)


@triton.jit
def _tree_conv1d_kernel(
    x_ptr,
    x_stride,
    out_ptr,
    out_stride,
    state_ptr,
    state_stride_slot,
    state_stride_dim,
    state_stride_col,
    w_ptr,
    w_stride_dim,
    bias_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    row_parent_ptr,
    num_seqs,
    dim,
    KW: tl.constexpr,
    STATE_LEN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    n = _row_request(query_start_loc_ptr, row, num_seqs, BLOCK_N)
    if n >= num_seqs:
        return
    bos = tl.load(query_start_loc_ptr + n).to(tl.int64)
    slot = tl.load(state_indices_ptr + n).to(tl.int64)
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    dm = d < dim
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + d, mask=dm, other=0.0).to(tl.float32)
    # Tap t multiplies weight column KW - 1 - t: t = 0 is the row itself, t > 0
    # walks t steps up its parent chain, then into the committed conv state
    # (newest input in the last state column).
    cur = row
    in_batch = row >= 0
    col = STATE_LEN
    for t in tl.static_range(KW):
        if t > 0:
            p = tl.load(row_parent_ptr + cur).to(tl.int64)
            up = in_batch & (p >= bos) & (p < cur)
            cur = tl.where(up, p, cur)
            col = tl.where(up, col, col - 1)
            in_batch = up
        from_state = (~in_batch) & (col >= 0) & (slot > 0)
        tap_x = tl.load(x_ptr + cur * x_stride + d, mask=dm & in_batch, other=0.0)
        tap_s = tl.load(
            state_ptr
            + slot * state_stride_slot
            + d * state_stride_dim
            + col * state_stride_col,
            mask=dm & from_state,
            other=0.0,
        )
        tap = tl.where(in_batch, tap_x.to(tl.float32), tap_s.to(tl.float32))
        w = tl.load(w_ptr + d * w_stride_dim + (KW - 1 - t), mask=dm, other=0.0)
        acc += w.to(tl.float32) * tap
    if SILU:
        acc = acc * tl.sigmoid(acc)
    tl.store(out_ptr + row * out_stride + d, acc.to(out_ptr.dtype.element_ty), mask=dm)


def tree_causal_conv1d(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_seqs: int,
    row_parent: torch.Tensor,
) -> torch.Tensor:
    """Causal depthwise conv over each row's ancestor chain.

    Args:
        x: ``[T, dim]`` conv inputs of the batch rows.
        conv_state: ``(slots, dim, state_len)``; the last ``width - 1``
            columns hold the request's newest committed inputs. Read only.
        weight: ``(dim, width)``.
        bias: ``(dim,)`` or None.
        activation: ``"silu"``/``"swish"`` or None.
        state_indices: ``[num_seqs]`` conv state slot per request.
        query_start_loc: ``[num_seqs + 1]`` row offsets per request.
        num_seqs: Requests.
        row_parent: ``[>= T]`` parent row per row.

    Returns:
        ``[T, dim]`` conv outputs (rows past the last request untouched).

    """
    num_rows, dim = x.shape
    width = weight.shape[1]
    out = torch.empty_like(x)
    if num_rows == 0 or num_seqs == 0:
        return out
    assert x.stride(1) == 1 and weight.stride(1) == 1
    block_d = 256
    _tree_conv1d_kernel[(num_rows, triton.cdiv(dim, block_d))](
        x,
        x.stride(0),
        out,
        out.stride(0),
        conv_state,
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        weight,
        weight.stride(0),
        bias,
        state_indices,
        query_start_loc,
        row_parent,
        num_seqs,
        dim,
        KW=width,
        STATE_LEN=conv_state.shape[2],
        HAS_BIAS=bias is not None,
        SILU=activation in ("silu", "swish"),
        BLOCK_D=block_d,
        BLOCK_N=triton.next_power_of_2(num_seqs),
    )
    return out


@triton.jit
def _tree_gated_delta_rule_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    o_ptr,
    h0_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    row_parent_ptr,
    scale,
    stride_state_slot,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    MAXD: tl.constexpr,
):
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    n = i_nh // HV
    hv = i_nh % HV
    h = hv // (HV // H)
    bos = tl.load(query_start_loc_ptr + n).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + n + 1).to(tl.int64)
    slot = tl.load(state_indices_ptr + n).to(tl.int64)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mk = o_k < K
    mv = o_v < V
    mh = mv[:, None] & mk[None, :]
    h0 = tl.load(
        h0_ptr
        + slot * stride_state_slot
        + hv * V * K
        + o_v[:, None] * K
        + o_k[None, :],
        mask=mh & (slot > 0),
        other=0.0,
    ).to(tl.float32)
    dpos = tl.arange(0, MAXD)
    for row in range(bos, eos):
        # Ancestor path of ``row`` within the request, deepest first.
        path = tl.full([MAXD], -1, tl.int64)
        cur = row
        length = 0
        for s in tl.static_range(MAXD):
            valid = cur >= bos
            path = tl.where((dpos == s) & valid, cur, path)
            length += valid.to(tl.int32)
            p = tl.load(row_parent_ptr + tl.maximum(cur, 0)).to(tl.int64)
            cur = tl.where(valid & (p >= bos) & (p < cur), p, -1)
        # Replay the path from the committed state, oldest row first.
        b_h = h0
        for s in range(0, length):
            r = tl.sum(tl.where(dpos == length - 1 - s, path, 0), axis=0)
            b_k = tl.load(k_ptr + (r * H + h) * K + o_k, mask=mk, other=0.0).to(
                tl.float32
            )
            b_v = tl.load(v_ptr + (r * HV + hv) * V + o_v, mask=mv, other=0.0).to(
                tl.float32
            )
            b_g = tl.load(g_ptr + r * HV + hv).to(tl.float32)
            b_beta = tl.load(beta_ptr + r * HV + hv).to(tl.float32)
            b_h *= tl.exp(b_g)
            b_v -= tl.sum(b_h * b_k[None, :], 1)
            b_v *= b_beta
            b_h += b_v[:, None] * b_k[None, :]
        b_q = tl.load(q_ptr + (row * H + h) * K + o_k, mask=mk, other=0.0).to(
            tl.float32
        )
        b_o = tl.sum(b_h * (b_q * scale)[None, :], 1)
        tl.store(
            o_ptr + (row * HV + hv) * V + o_v,
            b_o.to(o_ptr.dtype.element_ty),
            mask=mv,
        )


def tree_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_seqs: int,
    row_parent: torch.Tensor,
    max_depth: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Gated delta rule where each row continues its parent row's state.

    Args:
        q: ``[T, H, K]`` queries (already l2-normalized).
        k: ``[T, H, K]`` keys (already l2-normalized).
        v: ``[T, HV, V]``.
        g: ``[T, HV]`` log decay.
        beta: ``[T, HV]``.
        ssm_state: ``(slots, HV, V, K)`` committed states. Read only.
        state_indices: ``[num_seqs]`` state slot per request.
        query_start_loc: ``[num_seqs + 1]`` row offsets per request.
        num_seqs: Requests.
        row_parent: ``[>= T]`` parent row per row.
        max_depth: Longest ancestor path (rows) within a request.
        scale: Query scale (default ``K ** -0.5``).

    Returns:
        ``[T, HV, V]`` outputs.

    """
    num_rows, H, K = k.shape
    HV, V = v.shape[1], v.shape[2]
    o = torch.empty_like(v)
    if num_rows == 0 or num_seqs == 0:
        return o
    if scale is None:
        scale = K**-0.5
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert ssm_state.stride(3) == 1 and ssm_state.stride(2) == K
    BV = 32
    _tree_gated_delta_rule_kernel[(triton.cdiv(V, BV), num_seqs * HV)](
        q,
        k,
        v,
        g,
        beta,
        o,
        ssm_state,
        state_indices,
        query_start_loc,
        row_parent,
        scale,
        ssm_state.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=BV,
        MAXD=triton.next_power_of_2(max_depth),
    )
    return o


@triton.jit
def _tree_gated_delta_rule_chunk_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    o_ptr,
    h0_ptr,
    state_indices_ptr,
    query_start_loc_ptr,
    row_parent_ptr,
    scale,
    stride_state_slot,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    MAXD: tl.constexpr,
    LOG2_MAXD: tl.constexpr,
):
    """One request's verify rows as a single chunk of the gated delta rule,
    with the causal mask replaced by the ancestor relation.

    A row's transformed value depends only on its ancestors, so the chunk's
    lower-triangular system (ancestors precede descendants) yields every row's
    output from the committed state at once:
      A = beta_t exp(G_t - G_s) k_t.k_s for s a strict ancestor of t,
      T = (I + A)^-1, w = T (beta exp(G) k), u = T (beta v),
      v' = u - w S0^T, o = exp(G) q S0^T + (q k^T exp(G_t - G_s))_anc v',
    with G_t the decay summed along t's path.
    """
    i_v = tl.program_id(0)
    i_nh = tl.program_id(1)
    n = i_nh // HV
    hv = i_nh % HV
    h = hv // (HV // H)
    bos = tl.load(query_start_loc_ptr + n).to(tl.int64)
    eos = tl.load(query_start_loc_ptr + n + 1).to(tl.int64)
    t = tl.arange(0, BT)
    rows = bos + t
    live = rows < eos
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    mv = o_v < V

    # Ancestor-or-self relation between the request's rows.
    anc = t[:, None] == t[None, :]
    cur = tl.where(live, rows, -1)
    for _ in tl.static_range(MAXD):
        p = tl.load(row_parent_ptr + tl.maximum(cur, 0), mask=cur >= 0, other=-1)
        p = p.to(tl.int64)
        cur = tl.where((cur >= 0) & (p >= bos) & (p < cur), p, -1)
        anc = anc | ((cur - bos)[:, None] == t[None, :])
    anc = anc & live[:, None] & live[None, :]

    b_g = tl.load(g_ptr + rows * HV + hv, mask=live, other=0.0).to(tl.float32)
    b_beta = tl.load(beta_ptr + rows * HV + hv, mask=live, other=0.0).to(tl.float32)
    big_g = tl.sum(tl.where(anc, b_g[None, :], 0.0), axis=1)
    decay = tl.where(anc, tl.exp(big_g[:, None] - big_g[None, :]), 0.0)

    b_k = tl.load(
        k_ptr + (rows[:, None] * H + h) * K + o_k[None, :],
        mask=live[:, None],
        other=0.0,
    ).to(tl.float32)
    b_q = tl.load(
        q_ptr + (rows[:, None] * H + h) * K + o_k[None, :],
        mask=live[:, None],
        other=0.0,
    ).to(tl.float32)
    b_q = b_q * scale
    b_v = tl.load(
        v_ptr + (rows[:, None] * HV + hv) * V + o_v[None, :],
        mask=live[:, None] & mv[None, :],
        other=0.0,
    ).to(tl.float32)

    kk = tl.dot(b_k, tl.trans(b_k), input_precision="ieee")
    strict = anc & (t[:, None] != t[None, :])
    a = tl.where(strict, b_beta[:, None] * decay * kk, 0.0)
    # T = (I + A)^-1. A links strict ancestors only, so A^j vanishes once j
    # exceeds the longest path (MAXD rows): T = prod_j (I + (-A)^(2^j)).
    eye = (t[:, None] == t[None, :]).to(tl.float32)
    m = -a
    tinv = eye + m
    for _ in tl.static_range(1, LOG2_MAXD):
        m = tl.dot(m, m, input_precision="ieee")
        tinv = tl.dot(tinv, eye + m, input_precision="ieee")

    w = tl.dot(
        tinv, b_k * (b_beta * tl.exp(big_g))[:, None], input_precision="ieee"
    )
    u = tl.dot(tinv, b_v * b_beta[:, None], input_precision="ieee")

    slot = tl.load(state_indices_ptr + n).to(tl.int64)
    s0 = tl.load(
        h0_ptr
        + slot * stride_state_slot
        + hv * V * K
        + o_v[:, None] * K
        + o_k[None, :],
        mask=mv[:, None] & (slot > 0),
        other=0.0,
    ).to(tl.float32)
    s0_t = tl.trans(s0)  # [K, BV]
    v_new = u - tl.dot(w, s0_t, input_precision="ieee")
    qk = tl.dot(b_q, tl.trans(b_k), input_precision="ieee")
    o = tl.exp(big_g)[:, None] * tl.dot(b_q, s0_t, input_precision="ieee")
    o += tl.dot(qk * decay, v_new, input_precision="ieee")
    tl.store(
        o_ptr + (rows[:, None] * HV + hv) * V + o_v[None, :],
        o.to(o_ptr.dtype.element_ty),
        mask=live[:, None] & mv[None, :],
    )


def tree_gated_delta_rule_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_seqs: int,
    row_parent: torch.Tensor,
    max_depth: int,
    max_rows: int,
    scale: float | None = None,
) -> torch.Tensor:
    """:func:`tree_gated_delta_rule` as one masked chunk per request
    (tensor-core dots); requests may hold at most ``max_rows`` rows."""
    num_rows, H, K = k.shape
    HV, V = v.shape[1], v.shape[2]
    o = torch.empty_like(v)
    if num_rows == 0 or num_seqs == 0:
        return o
    if scale is None:
        scale = K**-0.5
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert ssm_state.stride(3) == 1 and ssm_state.stride(2) == K
    assert K == triton.next_power_of_2(K)
    BV = 32
    _tree_gated_delta_rule_chunk_kernel[(triton.cdiv(V, BV), num_seqs * HV)](
        q,
        k,
        v,
        g,
        beta,
        o,
        ssm_state,
        state_indices,
        query_start_loc,
        row_parent,
        scale,
        ssm_state.stride(0),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=max(16, triton.next_power_of_2(max_rows)),
        BV=BV,
        MAXD=triton.next_power_of_2(max_depth),
        LOG2_MAXD=max(1, (triton.next_power_of_2(max_depth) - 1).bit_length()),
        num_warps=4,
    )
    return o


@triton.jit
def _commit_conv_state_kernel(
    x_ptr,
    x_stride,
    state_ptr,
    state_stride_slot,
    state_stride_dim,
    state_stride_col,
    state_indices_ptr,
    cu_rows_ptr,
    dim,
    STATE_LEN: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Shift each request's committed rows into its conv state, keeping the
    newest ``STATE_LEN`` inputs of ``[old state, rows]``."""
    n = tl.program_id(0)
    lo = tl.load(cu_rows_ptr + n).to(tl.int64)
    m = tl.load(cu_rows_ptr + n + 1).to(tl.int64) - lo
    slot = tl.load(state_indices_ptr + n).to(tl.int64)
    if (m == 0) | (slot <= 0):
        return
    d = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    dm = d < dim
    base = state_ptr + slot * state_stride_slot + d * state_stride_dim
    # New column j is element m + j of [old state (STATE_LEN), rows (m)]. With
    # m >= 1 it reads column m + j > j, so writing columns in increasing order
    # never reads one already overwritten.
    for j in tl.static_range(STATE_LEN):
        src = m + j
        old = tl.load(base + src * state_stride_col, mask=dm & (src < STATE_LEN))
        new = tl.load(
            x_ptr + (lo + src - STATE_LEN) * x_stride + d,
            mask=dm & (src >= STATE_LEN),
        )
        val = tl.where(src < STATE_LEN, old, new.to(old.dtype))
        tl.store(base + j * state_stride_col, val, mask=dm)


def commit_gated_delta_rule_paths(
    rows: torch.Tensor,
    cu_rows: torch.Tensor,
    num_seqs: int,
    max_rows: int,
    state_indices: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    activation: str | None,
    ssm_state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> None:
    """Advance each request's committed conv and recurrent state over its
    committed rows (the chain recurrence kernel, fed by a chain-shaped tree
    conv).

    Args:
        rows: ``[>= cu_rows[-1], conv_dim + 2 * HV]`` the committed rows' raw
            inputs (``mixed_qkv | b | a``), packed per request.
        cu_rows: ``[num_seqs + 1]`` row offsets per request (a request with no
            rows keeps its state).
        num_seqs: Requests.
        max_rows: Most rows a request commits.
        state_indices: ``[num_seqs]`` state slot per request.
        conv_state: ``(slots, dim, width - 1)``, updated in place.
        conv_weight: ``(dim, width)``.
        conv_bias: ``(dim,)`` or None.
        activation: Conv activation.
        ssm_state: ``(slots, HV, V, K)``, updated in place.
        A_log: ``[HV]`` log decay parameter.
        dt_bias: ``[HV]`` decay bias.
        num_k_heads: Key heads.
        head_k_dim: Key head size.
        head_v_dim: Value head size.

    """
    if num_seqs == 0:
        return
    num_v_heads = A_log.shape[0]
    q_dim = num_k_heads * head_k_dim
    conv_dim = 2 * q_dim + num_v_heads * head_v_dim
    raw_qkv = rows[:, :conv_dim].contiguous()
    b = rows[:, conv_dim : conv_dim + num_v_heads].contiguous()
    a = rows[:, conv_dim + num_v_heads :].contiguous()
    num_rows = raw_qkv.shape[0]
    # A committed path is a chain: each row continues the previous one.
    # One spare entry absorbs the start offset of trailing empty requests.
    chain_parent = torch.arange(-1, num_rows, dtype=torch.int32, device=rows.device)
    chain_parent[cu_rows[:-1].long()] = -1
    mixed_qkv = tree_causal_conv1d(
        raw_qkv,
        conv_state,
        conv_weight,
        conv_bias,
        activation,
        state_indices,
        cu_rows,
        num_seqs,
        chain_parent,
    )
    block_d = 256
    _commit_conv_state_kernel[(num_seqs, triton.cdiv(conv_dim, block_d))](
        raw_qkv,
        raw_qkv.stride(0),
        conv_state,
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        state_indices,
        cu_rows,
        conv_dim,
        STATE_LEN=conv_state.shape[2],
        BLOCK_D=block_d,
    )
    q, k, v = torch.split(mixed_qkv, [q_dim, q_dim, conv_dim - 2 * q_dim], dim=-1)
    # Every token of a request writes the same slot, so the last one leaves
    # the state after the whole committed path.
    slots = state_indices[:, None].expand(num_seqs, max_rows).contiguous()
    fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q.contiguous().view(1, num_rows, num_k_heads, head_k_dim),
        k=k.contiguous().view(1, num_rows, num_k_heads, head_k_dim),
        v=v.contiguous().view(1, num_rows, num_v_heads, head_v_dim),
        initial_state=ssm_state,
        inplace_final_state=True,
        cu_seqlens=cu_rows,
        ssm_state_indices=slots,
        use_qk_l2norm_in_kernel=True,
    )
