# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tree verification on attention kernels without a mask hook.

Every kernel here reads ``TreeVerifyContext.row_mask``; each test runs a
batch of a tree request, a chain (prefill chunk) request and a 64-node tree
through the kernel and compares it to SDPA under the mask derived directly
from the trees' parents and positions.
"""

from dataclasses import dataclass

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.ops.tree_attention import (
    TREE_HIDDEN_BIAS,
    fill_tree_relative_bias,
    tree_local_attention,
    tree_local_context_lens,
)
from vllm.v1.worker.gpu.spec_decode.tree.context import TreeVerifyContext

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="needs a GPU"
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
TOL = 2e-2


@dataclass
class Req:
    context: int  # committed tokens before the request's first row
    rows: int  # query rows (anchor + nodes for a tree)
    parents: list[int] | None = None  # parent arena per node, -1 = anchor


def _requests() -> list[Req]:
    return [
        # n0, n2 under the anchor; n1 under n0; n3 under n1; n4 under n2.
        Req(context=37, rows=6, parents=[-1, 0, -1, 1, 2]),
        Req(context=20, rows=5),
        # 64 siblings: node 63 sits 64 slots past its anchor.
        Req(context=150, rows=65, parents=[-1] * 64),
        # A deep chain of 12 nodes.
        Req(context=300, rows=13, parents=[-1] + list(range(11))),
        # A fresh prompt: no context at all.
        Req(context=0, rows=7),
    ]


@dataclass
class Batch:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    ctx: TreeVerifyContext
    # Per row: visible key slots and the positions of the row and every key.
    visible: list[torch.Tensor]
    positions: list[torch.Tensor]


def _build(reqs: list[Req], window: int | None = None) -> Batch:
    """The batch's tree context and its reference per-row masks."""
    qsl = [0]
    for r in reqs:
        qsl.append(qsl[-1] + r.rows)
    query_start_loc = torch.tensor(qsl, dtype=torch.int32, device=DEVICE)
    max_w = max((len(r.parents) for r in reqs if r.parents), default=1)
    parents = torch.full((len(reqs), max_w), -1, dtype=torch.int64, device=DEVICE)
    widths = torch.zeros(len(reqs), dtype=torch.int64, device=DEVICE)
    for i, r in enumerate(reqs):
        if r.parents:
            parents[i, : len(r.parents)] = torch.tensor(r.parents)
            widths[i] = len(r.parents)
    ctx = TreeVerifyContext(qsl[-1] + 8, len(reqs), max_depth=64, device=DEVICE)
    ctx.fill(query_start_loc, len(reqs), parents, widths, widths.cpu().numpy())

    visible, positions = [], []
    for r in reqs:
        seq_len = r.context + r.rows
        pos = torch.arange(seq_len)
        depth = [0] * (r.rows - 1)
        if r.parents:
            for a, p in enumerate(r.parents):
                depth[a] = 1 if p < 0 else depth[p] + 1
                pos[r.context + 1 + a] = r.context + depth[a]
        for i in range(r.rows):
            slot = r.context + i
            mask = torch.zeros(seq_len, dtype=torch.bool)
            if r.parents and i > 0:
                mask[: r.context + 1] = True
                a = i - 1
                while a >= 0:
                    mask[r.context + 1 + a] = True
                    a = r.parents[a]
            else:
                mask[: slot + 1] = True
            if window is not None:
                mask &= (pos[slot] - pos) < window
            visible.append(mask)
            positions.append(pos)
    seq_lens = torch.tensor(
        [r.context + r.rows for r in reqs], dtype=torch.int32, device=DEVICE
    )
    return Batch(query_start_loc, seq_lens, ctx, visible, positions)


def _paged(reqs: list[Req], block_size: int, hkv: int, d: int):
    """Random K/V per request, written to a paged cache with shuffled pages."""
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    max_blocks = max(-(-(r.context + r.rows) // block_size) for r in reqs)
    num_blocks = len(reqs) * max_blocks + 3
    perm = torch.randperm(num_blocks, device=DEVICE, generator=gen)
    block_table = perm[: len(reqs) * max_blocks].view(len(reqs), max_blocks)
    block_table = block_table.to(torch.int32)
    k_cache = torch.zeros(num_blocks, block_size, hkv, d, dtype=DTYPE, device=DEVICE)
    v_cache = torch.zeros_like(k_cache)
    ks, vs = [], []
    for i, r in enumerate(reqs):
        n = r.context + r.rows
        k = torch.randn(n, hkv, d, dtype=DTYPE, device=DEVICE, generator=gen)
        v = torch.randn(n, hkv, d, dtype=DTYPE, device=DEVICE, generator=gen)
        for s in range(n):
            blk = block_table[i, s // block_size]
            k_cache[blk, s % block_size] = k[s]
            v_cache[blk, s % block_size] = v[s]
        ks.append(k)
        vs.append(v)
    return k_cache, v_cache, block_table, ks, vs


def _reference(reqs, batch, q, ks, vs, scale, softcap=0.0):
    out = torch.empty_like(q, dtype=torch.float32)
    hq, hkv = q.shape[1], ks[0].shape[1]
    row = 0
    for i, r in enumerate(reqs):
        k = ks[i].float().repeat_interleave(hq // hkv, dim=1)
        v = vs[i].float().repeat_interleave(hq // hkv, dim=1)
        for _ in range(r.rows):
            s = torch.einsum("hd,khd->hk", q[row].float(), k) * scale
            if softcap:
                s = softcap * torch.tanh(s / softcap)
            s = s.masked_fill(~batch.visible[row].to(DEVICE)[None], float("-inf"))
            out[row] = torch.einsum("hk,khd->hd", s.softmax(-1), v)
            row += 1
    return out


@pytest.mark.parametrize("window", [None, 100])
def test_triton_unified_attention_tree_mask(window):
    """The Triton unified kernel's tree mask keeps each tree row to its
    context and ancestors (positionally windowed), and chain rows causal."""
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    reqs = _requests()
    batch = _build(reqs, window)
    hq, hkv, d, block_size = 8, 2, 128, 16
    k_cache, v_cache, block_table, ks, vs = _paged(reqs, block_size, hkv, d)
    q = torch.randn(int(batch.query_start_loc[-1]), hq, d, dtype=DTYPE, device=DEVICE)
    out = torch.empty_like(q)
    scale = d**-0.5
    ones = torch.ones(len(reqs), hkv, device=DEVICE)
    unified_attention(
        q=q,
        k=k_cache,
        v=v_cache,
        out=out,
        cu_seqlens_q=batch.query_start_loc,
        max_seqlen_q=max(r.rows for r in reqs),
        seqused_k=batch.seq_lens,
        max_seqlen_k=int(batch.seq_lens.max()),
        softmax_scale=scale,
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        block_table=block_table,
        softcap=0,
        q_descale=None,
        k_descale=ones,
        v_descale=ones,
        tree_mask=batch.ctx.row_mask,
        tree_pos_delta=batch.ctx.row_pos_delta,
    )
    ref = _reference(reqs, batch, q, ks, vs, scale)
    torch.testing.assert_close(out.float(), ref, atol=TOL, rtol=TOL)


@pytest.mark.parametrize("mode", ["extend", "verify"])
def test_b12x_relative_bias_tree_mask(mode):
    """b12x paged attention under the tree relative bias matches the tree
    mask, on mixed batches (extend plans) and on uniform verify batches
    (verify plans, which split KV over pages)."""
    b12x = pytest.importorskip("b12x.attention.paged")
    if not b12x.is_supported():
        pytest.skip("b12x paged attention needs SM12x")
    from b12x.attention.paged._forward import paged_attention_forward
    from b12x.attention.paged._scratch import (
        B12XPagedAttentionScratchCaps,
        plan_paged_attention_scratch,
    )
    from b12x.attention.paged.planner import create_paged_plan

    split_kv = mode == "verify"
    if split_kv:
        reqs = [
            Req(context=100, rows=13, parents=[-1, 0, -1, 1, 2] + list(range(4, 11))),
            Req(context=250, rows=13, parents=[-1] * 12),
            Req(context=191, rows=13, parents=[-1] + list(range(11))),
        ]
    else:
        reqs = _requests()
    batch = _build(reqs)
    hq, hkv, d, page = 8, 2, 128, 64
    k_cache, v_cache, block_table, ks, vs = _paged(reqs, page, hkv, d)
    num_tokens = int(batch.query_start_loc[-1])
    q = torch.randn(num_tokens, hq, d, dtype=DTYPE, device=DEVICE)
    kwargs = dict(fixed_split_size=1) if split_kv else {}
    plan = create_paged_plan(
        q,
        k_cache,
        v_cache,
        block_table,
        batch.seq_lens,
        batch.query_start_loc,
        mode=mode,
        disable_split_kv=not split_kv,
        **kwargs,
    )
    scratch_plan = plan_paged_attention_scratch(
        B12XPagedAttentionScratchCaps(
            device=DEVICE,
            mode=mode,
            dtype=DTYPE,
            kv_dtype=DTYPE,
            num_q_heads=hq,
            num_kv_heads=hkv,
            head_dim_qk=d,
            head_dim_vo=d,
            page_size=page,
            max_total_q=plan.total_q,
            max_batch=len(reqs),
            max_page_table_width=block_table.shape[1],
            max_work_items=max(plan.new_batch_size, 1),
            max_partial_rows=plan.total_num_partial_rows,
            num_cache_pages=k_cache.shape[0],
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=DEVICE)
        for shape, dtype in scratch_plan.shapes_and_dtypes()
    )
    bias = torch.empty(num_tokens, hq, 64, dtype=DTYPE, device=DEVICE)
    fill_tree_relative_bias(bias, batch.ctx.row_mask)
    out = torch.empty_like(q)
    binding = scratch_plan.bind(
        scratch=scratch,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        output=out,
        page_table=block_table,
        cache_seqlens=batch.seq_lens,
        cu_seqlens_q=batch.query_start_loc,
        active_total_q=num_tokens,
        relative_attention_bias=bias,
        disable_split_kv=not split_kv,
        **kwargs,
    )
    assert binding.scratch.plan.split_kv == split_kv
    paged_attention_forward(binding=binding)
    ref = _reference(reqs, batch, q, ks, vs, d**-0.5)
    torch.testing.assert_close(out.float(), ref, atol=TOL, rtol=TOL)


def test_relative_bias_hides_exactly_the_masked_distances():
    """The relative bias is zero at every distance a row sees and the hidden
    bias elsewhere, on every head, and zero past the mask's span."""
    reqs = _requests()
    batch = _build(reqs)
    rows = int(batch.query_start_loc[-1])
    bias = torch.full((rows, 3, 128), 7.0, dtype=DTYPE, device=DEVICE)
    fill_tree_relative_bias(bias, batch.ctx.row_mask)
    words = batch.ctx.row_mask[:rows].tolist()
    expected = torch.zeros(rows, 128)
    for i, w in enumerate(words):
        for dist in range(64):
            if not (w >> dist) & 1:
                expected[i, dist] = TREE_HIDDEN_BIAS
    for h in range(3):
        torch.testing.assert_close(bias[:, h].float().cpu(), expected.to(DTYPE).float())


@pytest.mark.parametrize("softcap", [0.0, 30.0])
def test_tree_local_attention_merges_context(softcap):
    """For tree requests, context attention (no mask, every row) merged with
    the in-request masked attention equals the tree-masked attention; rows
    of requests without a tree are left to the causal kernel."""
    reqs = _requests() + [Req(context=10, rows=200)]
    batch = _build(reqs)
    hq, hkv, d = 8, 2, 64
    gen = torch.Generator(device=DEVICE).manual_seed(1)
    ks = [
        torch.randn(
            r.context + r.rows, hkv, d, dtype=DTYPE, device=DEVICE, generator=gen
        )
        for r in reqs
    ]
    vs = [torch.randn_like(k) for k in ks]
    num_tokens = int(batch.query_start_loc[-1])
    q = torch.randn(num_tokens, hq, d, dtype=DTYPE, device=DEVICE, generator=gen)
    key = torch.cat([k[r.context :] for k, r in zip(ks, reqs)])
    value = torch.cat([v[r.context :] for v, r in zip(vs, reqs)])
    scale = d**-0.5

    ctx_lens = tree_local_context_lens(
        torch.empty(len(reqs), dtype=torch.int32, device=DEVICE),
        batch.query_start_loc,
        batch.seq_lens,
        batch.ctx.tree_width,
    )
    # Requests without a tree are not split; a dummy key keeps the paged
    # kernel off zero-length requests.
    assert ctx_lens.tolist() == [r.context if r.parents else 1 for r in reqs]
    # The context pass: every row of a request sees its first ctx_len keys.
    ctx_out = torch.zeros(num_tokens, hq, d, dtype=DTYPE, device=DEVICE)
    ctx_lse = torch.full((hq, num_tokens), float("inf"), device=DEVICE)
    row = 0
    for i, r in enumerate(reqs):
        n = int(ctx_lens[i])
        k = ks[i][:n].float().repeat_interleave(hq // hkv, dim=1)
        v = vs[i][:n].float().repeat_interleave(hq // hkv, dim=1)
        for _ in range(r.rows):
            if n:
                s = torch.einsum("hd,khd->hk", q[row].float(), k) * scale
                if softcap:
                    s = softcap * torch.tanh(s / softcap)
                ctx_out[row] = torch.einsum("hk,khd->hd", s.softmax(-1), v).to(DTYPE)
                ctx_lse[:, row] = s.logsumexp(-1)
            row += 1

    sentinel = 5.0
    out = torch.full_like(q, sentinel)
    tree_local_attention(
        out,
        q,
        key,
        value,
        ctx_out,
        ctx_lse,
        batch.ctx.row_mask,
        batch.query_start_loc,
        batch.ctx.tree_width,
        scale,
        softcap,
    )
    ref = _reference(reqs, batch, q, ks, vs, scale, softcap)
    row = 0
    for r in reqs:
        rows = slice(row, row + r.rows)
        if r.parents:
            torch.testing.assert_close(out[rows].float(), ref[rows], atol=TOL, rtol=TOL)
        else:
            assert (out[rows] == sentinel).all()
        row += r.rows
