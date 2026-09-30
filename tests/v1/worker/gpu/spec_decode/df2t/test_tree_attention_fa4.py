# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the df2t FA4 tree mask_mod (committed kernel vehicle).

Task 14 swaps the df2t tree-verify mask onto the FlashAttention-4 CuTe-DSL
``mask_mod`` vehicle behind ``tree_attention_backend == "fa4"`` (the default),
leaving the Flex path behind ``== "flex"`` for A/B.

Coordinate conventions (validated here as an explicit sub-gate, see
``test_fa4_tree_mask_kv_idx_is_logical_blockslot`` and the ``_manual_matrix``
reference): in the FA4 paged varlen forward the mask_mod sees
``q_idx`` as the *local verify-block query index* (0 = anchor, 1+a = node
arena ``a``, so block-slot ``= C + q_idx``) and ``kv_idx`` as the *local KV
position within the sequence*, which equals the logical block-slot (context
``< C``, anchor ``== C``, node ``C + 1 + j``). This matches the Flex path's
block-slot convention (Task 8), so the same ancestor-or-self semantics hold on
both vehicles. ``batch_idx`` is the request index, so per-request ``C`` and
ancestor words are read from aux tensors keyed by it.

These tests must stay FULLY OFFLINE: they construct a real `VllmConfig` via
`create_vllm_config`, which resolves the model config through vLLM's loader.
The `qwen3_config_path` fixture (see conftest.py) provides a locally-written
Qwen3 config directory so no HuggingFace access ever happens.
"""

from typing import TYPE_CHECKING

import pytest
import torch

from tests.v1.attention.test_attention_backends import (
    MockAttentionLayer,
    create_and_prepopulate_kv_cache,
)
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType
from vllm.v1.worker.gpu.spec_decode.df2t.attention import (
    build_fa4_tree_mask,
    build_flex_tree_mask,
    pack_tree_mask_inputs,
)
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

if TYPE_CHECKING:
    pass


def _sm120() -> bool:
    capability = current_platform.get_device_capability()
    return (
        current_platform.is_cuda() and capability is not None and capability.major == 12
    )


def _fa4_available() -> bool:
    if not current_platform.is_cuda():
        return False
    from vllm.v1.attention.backends.fa_utils import is_fa_version_supported

    return is_fa_version_supported(4)


requires_sm120_fa4 = pytest.mark.skipif(
    not (_sm120() and _fa4_available()),
    reason="FA4 df2t tree mask requires an SM120 device with FA4 available",
)

if _sm120() and _fa4_available():  # pragma: no branch
    DEVICE = torch.device("cuda:0")
    DTYPE = torch.bfloat16
    TOL = 3e-2


def _manual_matrix(C: int, words: list[int], B: int) -> torch.Tensor:
    """Manual ``[C + 1 + B, C + 1 + B]`` boolean matrix over block-slot (q, k).

    Identical to the Flex-path reference (Task 8 tests): key ``k`` is allowed iff
    ``k < C`` (context), ``k == C`` (anchor), or ``k == C + 1 + j`` with bit ``j``
    set in the query node's ancestor-or-self word. Rows below ``C`` are context
    positions that never query in the verify layout and are left False.
    """
    n = C + 1 + B
    matrix = torch.zeros(n, n, dtype=torch.bool)
    words_t = torch.tensor(words, dtype=torch.long)
    for arena in range(B):
        q = C + 1 + arena
        matrix[q, : C + 1] = True
        for j in range(B):
            k = C + 1 + j
            matrix[q, k] = bool((words_t[arena] >> j) & 1)
    matrix[C, : C + 1] = True
    return matrix


def _build_flex_reference_mask(specs, query_start_loc, max_b):
    """The ``build_flex_tree_mask`` closure over the full tree batch.

    Evaluated on the block-slot grid, it yields the per-request boolean matrix
    the Flex-pathed reference would apply; the FA4 forward must reproduce it.
    Returns ``(mask_mod, request_lookup, c_per_req, words)`` on CPU.
    """
    num_reqs = len(specs)
    request_lookup = torch.zeros(query_start_loc[-1], dtype=torch.int32)
    for r, (lo, hi) in enumerate(zip(query_start_loc, query_start_loc[1:])):
        request_lookup[lo:hi] = r
    c_per_req = torch.tensor([c for c, _, _ in specs], dtype=torch.long)
    words = torch.zeros(num_reqs, max_b, dtype=torch.long)
    for r, (_, w, _) in enumerate(specs):
        words[r, : len(w)] = torch.tensor(w, dtype=torch.long)
    mask_mod, _ = build_flex_tree_mask(
        num_reqs, c_per_req, words, max_b, request_lookup
    )
    return mask_mod, request_lookup, c_per_req, words


def _flex_matrix(
    mask_mod, request_lookup, c_per_req, words, max_b, query_start_loc, r
) -> torch.Tensor:
    """Evaluate the prebuilt flex closure on request ``r``'s block-slot grid."""
    C = c_per_req[r].item()
    # The request's node count comes from its query span (B_r + 1 queries).
    B_r = (query_start_loc[r + 1] - query_start_loc[r]) - 1
    n = C + 1 + B_r
    start = query_start_loc[r]
    b_for_q = torch.full((n,), start, dtype=torch.long)
    b_for_q[C + 1 :] = start + 1 + torch.arange(B_r, dtype=torch.long)
    q = torch.arange(n, dtype=torch.long)[:, None].expand(n, n)
    k = torch.arange(n, dtype=torch.long)[None, :].expand(n, n)
    h = torch.zeros(1, dtype=torch.long)
    b = b_for_q[:, None].expand(n, n)
    return mask_mod(b, h, q, k)


def _tree_specs() -> list[tuple[int, list[int], int]]:
    return [(4, [0b1, 0b11, 0b101], 3), (2, [0b1, 0b11], 2)]


def _df2t_spec_config():
    import types

    return types.SimpleNamespace(
        method="df2t",
        use_df2t=lambda: True,
        max_tree_budget=64,
        tree_attention_backend="fa4",
        num_speculative_tokens=5,
        enable_adaptive_verification=False,
    )


def test_build_fa4_tree_mask_shapes_and_cache():
    """build_fa4_tree_mask returns (mask_mod, [words, c_per_req]) of the right
    shapes, reuses the cached closure for the same B, and re-compiles per B."""
    device = torch.device("cpu")
    num_reqs, B = 2, 3
    c_per_req = torch.tensor([4, 2], dtype=torch.int32, device=device)
    words = torch.tensor(
        [[0b1, 0b11, 0b101], [0b1, 0b11, 0]], dtype=torch.int64, device=device
    )

    mm1, aux1 = build_fa4_tree_mask(num_reqs, c_per_req, words, B)
    mm2, aux2 = build_fa4_tree_mask(num_reqs, c_per_req, words, B)
    assert mm1 is mm2  # same B -> same cached closure -> same compile key
    assert aux1[0] is words and aux1[1] is c_per_req
    assert aux2[0] is words

    mm3, _ = build_fa4_tree_mask(
        num_reqs,
        c_per_req,
        torch.zeros(num_reqs, B + 1, dtype=torch.int64, device=device),
        B + 1,
    )
    assert mm3 is not mm1  # different B is a different mask_mod

    assert callable(mm1)

    mm_packed, aux_packed = build_fa4_tree_mask(
        num_reqs,
        c_per_req,
        pack_tree_mask_inputs(None, num_reqs, B, device),
        B,
    )
    assert aux_packed[0].shape == (num_reqs, B)


@requires_sm120_fa4
def test_fa4_tree_mask_matches_manual_and_flex_reference():
    """FA4 paged varlen with build_fa4_tree_mask matches the manual block-slot
    SDPA reference AND the Flex-path mask matrix reference (same tree), so the
    swapped-in committed kernel vehicle reproduces the always-correct flex
    vehicle's ancestor-or-self semantics within fp tolerance."""
    specs = _tree_specs()
    num_reqs = len(specs)
    query_lens = [b + 1 for _, _, b in specs]
    seq_lens = [c + 1 + b for c, _, b in specs]
    block_size = 16
    max_b = max(b for _, _, b in specs)
    max_seqlen = max(seq_lens)

    torch.manual_seed(0)
    cu = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    cu[1:] = torch.tensor(query_lens, dtype=torch.int32, device=DEVICE).cumsum(0)
    seqused_k = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    max_blocks = (max_seqlen + block_size - 1) // block_size
    # Distinct physical page per request: if kv_idx were physical/absolute (or
    # requests shared a page) this would fail (see the dedicated kv_idx test).
    physical = torch.arange(
        num_reqs * max_blocks, dtype=torch.int32, device=DEVICE
    ).view(num_reqs, max_blocks)

    Hq, Hkv = 16, 8
    num_tokens = sum(query_lens)
    q = torch.randn(num_tokens, Hq, 128, dtype=DTYPE, device=DEVICE)
    k_cache = torch.randn(
        num_reqs * max_blocks, block_size, Hkv, 128, dtype=DTYPE, device=DEVICE
    )
    v_cache = torch.randn_like(k_cache)

    # Fill each request's full sequence (context + anchor + nodes) at
    # block-slots 0..seq_len-1 and stash the per-request K/V for references.
    kfulls, vfulls = [], []
    pos = 0
    for r, (c, w, b_nodes) in enumerate(specs):
        seq_len = seq_lens[r]
        kf = torch.randn(seq_len, Hkv, 128, dtype=DTYPE, device=DEVICE)
        vf = torch.randn_like(kf)
        kfulls.append(kf)
        vfulls.append(vf)
        blk_pages = physical[r].tolist()
        for i in range(seq_len):
            k_cache[blk_pages[i // block_size], i % block_size] = kf[i]
            v_cache[blk_pages[i // block_size], i % block_size] = vf[i]
        pos += seq_len

    words = torch.zeros(num_reqs, max_b, dtype=torch.int64, device=DEVICE)
    for r, (_, w, _) in enumerate(specs):
        words[r, : len(w)] = torch.tensor(w, dtype=torch.int64, device=DEVICE)
    c_per_req = torch.tensor([c for c, _, _ in specs], dtype=torch.int32, device=DEVICE)
    scale = 128**-0.5

    mask_mod, aux = build_fa4_tree_mask(num_reqs, c_per_req, words, max_b)
    out = flash_attn_varlen_func(
        q,
        k_cache,
        v_cache,
        max_seqlen_q=max(query_lens),
        cu_seqlens_q=cu,
        max_seqlen_k=max_seqlen,
        cu_seqlens_k=None,
        seqused_k=seqused_k,
        block_table=physical,
        softmax_scale=scale,
        fa_version=4,
        num_splits=1,
        causal=False,
        window_size=(-1, -1),
        mask_mod=mask_mod,
        aux_tensors=aux,
    )

    # SDPA references: both the manual matrix and the Flex closure's matrix
    # over the verify-block rows (block-slots C..C+B).
    query_start_loc = [0]
    for r, (c, w, b_nodes) in enumerate(specs):
        query_start_loc.append(query_start_loc[-1] + b_nodes + 1)
    flex_mod, flex_lookup, flex_c, flex_words = _build_flex_reference_mask(
        specs, query_start_loc, max_b
    )
    manual_out = torch.zeros_like(q, dtype=torch.float32)
    flex_out = torch.zeros_like(q, dtype=torch.float32)
    st = 0
    for r, (c, w, b_nodes) in enumerate(specs):
        qr = q[st : st + b_nodes + 1].float()
        kk = kfulls[r].transpose(0, 1)[None].float().repeat_interleave(Hq // Hkv, dim=1)
        vv = vfulls[r].transpose(0, 1)[None].float().repeat_interleave(Hq // Hkv, dim=1)
        m_manual = (
            _manual_matrix(c, w, b_nodes)[c:].unsqueeze(0).unsqueeze(0).to(DEVICE)
        )
        m_flex = (
            _flex_matrix(
                flex_mod, flex_lookup, flex_c, flex_words, max_b, query_start_loc, r
            )[c:]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )
        manual_out[st : st + b_nodes + 1] = (
            torch.nn.functional.scaled_dot_product_attention(
                qr.unsqueeze(0).transpose(1, 2), kk, vv, attn_mask=m_manual, scale=scale
            )
            .transpose(1, 2)
            .squeeze(0)
        )
        flex_out[st : st + b_nodes + 1] = (
            torch.nn.functional.scaled_dot_product_attention(
                qr.unsqueeze(0).transpose(1, 2), kk, vv, attn_mask=m_flex, scale=scale
            )
            .transpose(1, 2)
            .squeeze(0)
        )
        st += b_nodes + 1

    # Manual and flex matrices are identical on this tree.
    assert torch.equal(
        _manual_matrix(4, [0b1, 0b11, 0b101], 3)[4:],
        _flex_matrix(
            flex_mod, flex_lookup, flex_c, flex_words, max_b, query_start_loc, 0
        )[4:],
    )
    err_manual = (out.float() - manual_out).abs().max().item()
    err_flex = (out.float() - flex_out).abs().max().item()
    assert err_manual < TOL, (
        f"FA4 tree mask diverged from manual reference: {err_manual}"
    )
    assert err_flex < TOL, f"FA4 tree mask diverged from flex reference: {err_flex}"


@requires_sm120_fa4
def test_fa4_tree_mask_treeless_request_with_context_is_causal():
    """A request with context (C > 0) but no tree this step (a chunked-prefill
    continuation or prefix-cache hit co-scheduled with a tree decode) attends
    causally over its context and its own rows, while the co-batched tree
    request keeps its ancestor-or-self mask."""
    tree_c, tree_words, tree_b = 4, [0b1, 0b11, 0b101], 3
    chunk_c, chunk_rows = 20, 5  # spans the 16-slot page boundary
    query_lens = [tree_b + 1, chunk_rows]
    seq_lens = [tree_c + 1 + tree_b, chunk_c + chunk_rows]
    block_size = 16
    max_b = tree_b
    max_blocks = (max(seq_lens) + block_size - 1) // block_size
    physical = torch.arange(2 * max_blocks, dtype=torch.int32, device=DEVICE).view(
        2, max_blocks
    )

    torch.manual_seed(5)
    Hq, Hkv = 16, 8
    q = torch.randn(sum(query_lens), Hq, 128, dtype=DTYPE, device=DEVICE)
    k_cache = torch.randn(
        2 * max_blocks, block_size, Hkv, 128, dtype=DTYPE, device=DEVICE
    )
    v_cache = torch.randn_like(k_cache)
    kfulls, vfulls = [], []
    for r, seq_len in enumerate(seq_lens):
        kf = torch.randn(seq_len, Hkv, 128, dtype=DTYPE, device=DEVICE)
        vf = torch.randn_like(kf)
        for i in range(seq_len):
            page = physical[r, i // block_size]
            k_cache[page, i % block_size] = kf[i]
            v_cache[page, i % block_size] = vf[i]
        kfulls.append(kf)
        vfulls.append(vf)

    words = torch.zeros(2, max_b, dtype=torch.int64, device=DEVICE)
    words[0] = torch.tensor(tree_words, dtype=torch.int64, device=DEVICE)
    c_per_req = torch.tensor([tree_c, chunk_c], dtype=torch.int32, device=DEVICE)
    cu = torch.tensor(
        [0, query_lens[0], sum(query_lens)], dtype=torch.int32, device=DEVICE
    )
    scale = 128**-0.5
    mask_mod, aux = build_fa4_tree_mask(2, c_per_req, words, max_b)
    out = flash_attn_varlen_func(
        q,
        k_cache,
        v_cache,
        max_seqlen_q=max(query_lens),
        cu_seqlens_q=cu,
        max_seqlen_k=max(seq_lens),
        cu_seqlens_k=None,
        seqused_k=torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE),
        block_table=physical,
        softmax_scale=scale,
        fa_version=4,
        num_splits=1,
        causal=False,
        window_size=(-1, -1),
        mask_mod=mask_mod,
        aux_tensors=aux,
    )

    n = seq_lens[1]
    rows = torch.arange(chunk_c, n)[:, None]
    masks = [
        _manual_matrix(tree_c, tree_words, tree_b)[tree_c:],
        torch.arange(n)[None, :] <= rows,
    ]
    st = 0
    for r, m in enumerate(masks):
        qr = q[st : st + query_lens[r]].float().unsqueeze(0).transpose(1, 2)
        kk = kfulls[r].transpose(0, 1)[None].float().repeat_interleave(Hq // Hkv, 1)
        vv = vfulls[r].transpose(0, 1)[None].float().repeat_interleave(Hq // Hkv, 1)
        ref = torch.nn.functional.scaled_dot_product_attention(
            qr, kk, vv, attn_mask=m[None, None].to(DEVICE), scale=scale
        )
        ref = ref.transpose(1, 2).squeeze(0)
        err = (out[st : st + query_lens[r]].float() - ref).abs().max().item()
        assert err < TOL, f"request {r} diverged from its reference: {err}"
        st += query_lens[r]


@requires_sm120_fa4
def test_fa4_tree_mask_kv_idx_is_logical_blockslot():
    """kv_idx in the FA4 paged mask is the logical block-slot, not an absolute
    physical KV slot: a query told to keep exactly key ``C + 2`` attends the
    values stored at block-slot C+2 even when the sequence's physical pages
    start at a non-zero base, and its output equals those values exactly."""
    C, B = 3, 2
    block_size = 16
    seq_len = C + 1 + B
    seq_len_q = B + 1
    cu = torch.tensor([0, seq_len_q], dtype=torch.int32, device=DEVICE)
    seqused_k = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)
    physical = torch.tensor([[40, 41]], dtype=torch.int32, device=DEVICE)
    num_pages = int(physical.max()) + 1
    torch.manual_seed(3)
    Hq, Hkv = 16, 8
    q = torch.randn(seq_len_q, Hq, 128, dtype=DTYPE, device=DEVICE)
    k_cache = torch.randn(num_pages, block_size, Hkv, 128, dtype=DTYPE, device=DEVICE)
    v_cache = torch.randn_like(k_cache)
    kfull = torch.randn(seq_len, Hkv, 128, dtype=DTYPE, device=DEVICE)
    vfull = torch.zeros(seq_len, Hkv, 128, dtype=DTYPE, device=DEVICE)
    for s in range(seq_len):
        vfull[s, :, s % 128] = 1.0  # distinct one-hot per block-slot
    for i in range(seq_len):
        k_cache[physical[0, i // block_size], i % block_size] = kfull[i]
        v_cache[physical[0, i // block_size], i % block_size] = vfull[i]
    c_per_req = torch.tensor([C], dtype=torch.int32, device=DEVICE)
    words = torch.tensor([[0b1, 0b11]], dtype=torch.int64, device=DEVICE)

    import cutlass.cute as cute
    from cutlass import Int32  # type: ignore[attr-defined]

    from vllm.vllm_flash_attn.cute.utils import scalar_to_ssa, ssa_to_scalar

    @cute.jit
    def keep_one_key(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors=None):
        b = batch_idx[0]
        c_r = aux_tensors[1][b]
        q_s = ssa_to_scalar(q_idx)
        k_s = ssa_to_scalar(kv_idx)
        anchor_slot = aux_tensors[2][0]
        r = (q_s == 0) & (k_s == anchor_slot)
        r = r | ((q_s >= 1) & (k_s == c_r + 1))
        return scalar_to_ssa(r, Int32)

    keep_one_key.use_fast_sampling = True
    keep_one_key.__vec_size__ = 1
    anchor_slot = torch.tensor([C + 2], dtype=torch.int32, device=DEVICE)
    out = flash_attn_varlen_func(
        q,
        k_cache,
        v_cache,
        max_seqlen_q=seq_len_q,
        cu_seqlens_q=cu,
        max_seqlen_k=seq_len,
        cu_seqlens_k=None,
        seqused_k=seqused_k,
        block_table=physical,
        softmax_scale=128**-0.5,
        fa_version=4,
        num_splits=1,
        causal=False,
        window_size=(-1, -1),
        mask_mod=keep_one_key,
        aux_tensors=[words, c_per_req, anchor_slot],
    )
    # Anchor (q_idx=0) attends only block-slot C+2; nodes attend block-slot C+1.
    assert out[0, 0].argmax().item() == C + 2
    assert out[1, 0].argmax().item() == C + 1
    mask = torch.zeros(seq_len_q, seq_len, dtype=torch.bool, device=DEVICE)
    mask[0, C + 2] = True
    mask[1:, C + 1] = True
    kk = kfull.transpose(0, 1)[None].repeat_interleave(Hq // Hkv, dim=1)
    vv = vfull.transpose(0, 1)[None].repeat_interleave(Hq // Hkv, dim=1)
    ref = (
        torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(0).transpose(1, 2).float(),
            kk.float(),
            vv.float(),
            attn_mask=mask[None, None],
            scale=128**-0.5,
        )
        .transpose(1, 2)
        .squeeze(0)
    )
    assert (out.float() - ref).abs().max().item() < TOL


@requires_sm120_fa4
def test_fa4_tree_mask_multi_page_kv_idx_is_absolute():
    """kv_idx is the ABSOLUTE logical block-slot across KV pages (fork fix
    threads the page table into compute_fastdiv_mods, mirroring sm100): a mask
    told to keep exactly block-slot 200 of a 300-token sequence (2nd/3rd page)
    attends the one-hot V stored at slot 200, not any wrapped in-page offset.

    The one-hot discriminators identify slots uniquely across heads (slot s
    lives in kv head s // 128, feature s % 128), and the read below uses the
    GQA-corrected query head (query head 2*kh reads kv head kh)."""
    block_size = 128
    seq_len = 300
    seq_len_q = 1
    cu = torch.tensor([0, seq_len_q], dtype=torch.int32, device=DEVICE)
    seqused_k = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)
    physical = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=DEVICE)
    num_pages = 3
    torch.manual_seed(7)
    Hq, Hkv = 16, 8
    q = torch.randn(seq_len_q, Hq, 128, dtype=DTYPE, device=DEVICE)
    k_cache = torch.randn(num_pages, block_size, Hkv, 128, dtype=DTYPE, device=DEVICE)
    v_cache = torch.zeros(num_pages, block_size, Hkv, 128, dtype=DTYPE, device=DEVICE)
    for s in range(seq_len):
        # slot s -> kv head s // 128 (= its page/block), feature s % 128.
        v_cache[s // block_size, s % block_size, s // 128, s % 128] = 1.0

    import cutlass.cute as cute
    from cutlass import Int32  # type: ignore[attr-defined]

    from vllm.vllm_flash_attn.cute.utils import scalar_to_ssa, ssa_to_scalar

    def make_keep(target):
        @cute.jit
        def keep_slot(
            batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors=None
        ):
            k_s = ssa_to_scalar(kv_idx)
            return scalar_to_ssa(k_s == target, Int32)

        keep_slot.use_fast_sampling = True
        keep_slot.__vec_size__ = 1
        return keep_slot

    def run(keep_slot):
        return flash_attn_varlen_func(
            q,
            k_cache,
            v_cache,
            max_seqlen_q=1,
            cu_seqlens_q=cu,
            max_seqlen_k=seq_len,
            cu_seqlens_k=None,
            seqused_k=seqused_k,
            block_table=physical,
            softmax_scale=128**-0.5,
            fa_version=4,
            num_splits=1,
            causal=False,
            window_size=(-1, -1),
            mask_mod=keep_slot,
            aux_tensors=[
                torch.zeros(1, 64, dtype=torch.int64, device=DEVICE),
                torch.zeros(1, dtype=torch.int32, device=DEVICE),
            ],
        )

    for target in (200, 128, 256, 72):
        out = run(make_keep(target))
        kh, feat = target // 128, target % 128
        hit = int(out[0, 2 * kh % Hq, feat].item() > 0.25)
        assert hit == 1, f"slot {target} not attended as itself (kv_idx wrapped?)"
    # A slot that does not exist (2049) is never attended.
    out = run(make_keep(2049))
    assert out.abs().max().item() < TOL


@requires_sm120_fa4
def test_fa4_tree_mask_multi_page_matches_manual_and_flex_reference():
    """Multi-page parity: with contexts spanning >= 2 KV pages (block 128), the
    FA4 tree mask reproduces the manual block-slot reference AND the Flex
    closure's matrix exactly, so the per-key tree mask is honored across page
    boundaries (kv_idx absolute)."""
    block_size = 128
    specs = [(200, [0b1, 0b11, 0b101, 0b1111], 4), (130, [0b1, 0b11], 2)]
    num_reqs = len(specs)
    query_lens = [b + 1 for _, _, b in specs]
    seq_lens = [c + 1 + b for c, _, b in specs]
    max_b = max(b for _, _, b in specs)
    max_seqlen = max(seq_lens)

    torch.manual_seed(0)
    cu = torch.zeros(num_reqs + 1, dtype=torch.int32, device=DEVICE)
    cu[1:] = torch.tensor(query_lens, dtype=torch.int32, device=DEVICE).cumsum(0)
    seqused_k = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    max_blocks = (max_seqlen + block_size - 1) // block_size
    physical = torch.arange(
        num_reqs * max_blocks, dtype=torch.int32, device=DEVICE
    ).view(num_reqs, max_blocks)

    Hq, Hkv = 16, 8
    total_q = sum(query_lens)
    q = torch.randn(total_q, Hq, 128, dtype=DTYPE, device=DEVICE)
    num_pages = num_reqs * max_blocks
    k_cache = torch.randn(num_pages, block_size, Hkv, 128, dtype=DTYPE, device=DEVICE)
    v_cache = torch.randn_like(k_cache)
    kfulls, vfulls = [], []
    for r, (c, w, b_nodes) in enumerate(specs):
        seq_len = seq_lens[r]
        kf = torch.randn(seq_len, Hkv, 128, dtype=DTYPE, device=DEVICE)
        vf = torch.randn_like(kf)
        kfulls.append(kf)
        vfulls.append(vf)
        for i in range(seq_len):
            k_cache[physical[r, i // block_size], i % block_size] = kf[i]
            v_cache[physical[r, i // block_size], i % block_size] = vf[i]

    words = torch.zeros(num_reqs, max_b, dtype=torch.int64, device=DEVICE)
    for r, (_, w, _) in enumerate(specs):
        words[r, : len(w)] = torch.tensor(w, dtype=torch.int64, device=DEVICE)
    c_per_req = torch.tensor([c for c, _, _ in specs], dtype=torch.int32, device=DEVICE)
    scale = 128**-0.5

    mask_mod, aux = build_fa4_tree_mask(num_reqs, c_per_req, words, max_b)
    out = flash_attn_varlen_func(
        q,
        k_cache,
        v_cache,
        max_seqlen_q=max(query_lens),
        cu_seqlens_q=cu,
        max_seqlen_k=max_seqlen,
        cu_seqlens_k=None,
        seqused_k=seqused_k,
        block_table=physical,
        softmax_scale=scale,
        fa_version=4,
        num_splits=1,
        causal=False,
        window_size=(-1, -1),
        mask_mod=mask_mod,
        aux_tensors=aux,
    )

    manual_out = torch.zeros_like(q, dtype=torch.float32)
    flex_out = torch.zeros_like(q, dtype=torch.float32)
    query_start_loc = [0]
    for r, (c, w, b_nodes) in enumerate(specs):
        query_start_loc.append(query_start_loc[-1] + b_nodes + 1)
    flex_mod, flex_lookup, flex_c, flex_words = _build_flex_reference_mask(
        specs, query_start_loc, max_b
    )
    st = 0
    for r, (c, w, b_nodes) in enumerate(specs):
        qr = q[st : st + b_nodes + 1].float()
        kk = kfulls[r].transpose(0, 1)[None].float().repeat_interleave(Hq // Hkv, dim=1)
        vv = vfulls[r].transpose(0, 1)[None].float().repeat_interleave(Hq // Hkv, dim=1)
        m = _manual_matrix(c, w, b_nodes)[c:].unsqueeze(0).unsqueeze(0).to(DEVICE)
        m_flex = (
            _flex_matrix(
                flex_mod, flex_lookup, flex_c, flex_words, max_b, query_start_loc, r
            )[c:]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )
        manual_out[st : st + b_nodes + 1] = (
            torch.nn.functional.scaled_dot_product_attention(
                qr.unsqueeze(0).transpose(1, 2), kk, vv, attn_mask=m, scale=scale
            )
            .transpose(1, 2)
            .squeeze(0)
        )
        flex_out[st : st + b_nodes + 1] = (
            torch.nn.functional.scaled_dot_product_attention(
                qr.unsqueeze(0).transpose(1, 2), kk, vv, attn_mask=m_flex, scale=scale
            )
            .transpose(1, 2)
            .squeeze(0)
        )
        st += b_nodes + 1
    err_manual = (out.float() - manual_out).abs().max().item()
    err_flex = (out.float() - flex_out).abs().max().item()
    assert err_manual < TOL, f"multi-page tree mask diverged from manual: {err_manual}"
    assert err_flex < TOL, (
        f"multi-page tree mask diverged from flex reference: {err_flex}"
    )
    assert torch.equal(
        _manual_matrix(200, [0b1, 0b11, 0b101, 0b1111], 4)[200:],
        _flex_matrix(
            flex_mod, flex_lookup, flex_c, flex_words, max_b, query_start_loc, 0
        )[200:],
    )


@requires_sm120_fa4
def test_fa4_tree_mask_sign_bit_node_63():
    """B=64 edge case: node arena 63 (bit 63, the int64 sign bit) is matched,
    mirroring the Flex mask's node-63 support, and the node-region bound
    ``j < B`` stays exact."""
    C, B = 2, 64
    block_size = 128
    seq_len = C + 1 + B
    seq_len_q = B + 1
    max_blocks = (seq_len + block_size - 1) // block_size
    physical = torch.zeros(1, max_blocks, dtype=torch.int32, device=DEVICE)
    torch.manual_seed(1)
    Hq, Hkv = 16, 8
    k_cache = torch.randn(max_blocks, block_size, Hkv, 128, dtype=DTYPE, device=DEVICE)
    v_cache = torch.randn_like(k_cache)
    q = torch.randn(seq_len_q, Hq, 128, dtype=DTYPE, device=DEVICE)
    kfull = torch.randn(seq_len, Hkv, 128, dtype=DTYPE, device=DEVICE)
    vfull = torch.randn_like(kfull)
    for i in range(seq_len):
        k_cache[i // block_size, i % block_size] = kfull[i]
        v_cache[i // block_size, i % block_size] = vfull[i]

    words = torch.zeros(1, B, dtype=torch.int64, device=DEVICE)
    words[0, 0] = 1
    words[0, 63] = -(1 << 63) + 1  # node63: self (bit 63) + node0 (bit 0)
    c_per_req = torch.tensor([C], dtype=torch.int32, device=DEVICE)
    cu = torch.tensor([0, seq_len_q], dtype=torch.int32, device=DEVICE)
    seqused_k = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)

    mask_mod, aux = build_fa4_tree_mask(1, c_per_req, words, B)
    out = flash_attn_varlen_func(
        q,
        k_cache,
        v_cache,
        max_seqlen_q=seq_len_q,
        cu_seqlens_q=cu,
        max_seqlen_k=seq_len,
        cu_seqlens_k=None,
        seqused_k=seqused_k,
        block_table=physical,
        softmax_scale=128**-0.5,
        fa_version=4,
        num_splits=1,
        causal=False,
        window_size=(-1, -1),
        mask_mod=mask_mod,
        aux_tensors=aux,
    )
    matrix = torch.zeros(seq_len_q, seq_len, dtype=torch.bool, device=DEVICE)
    words_l = [int(words[0, a].item()) for a in range(B)]
    for a in range(B):
        matrix[1 + a, : C + 1] = True
        for j in range(B):
            matrix[1 + a, C + 1 + j] = bool((words_l[a] >> j) & 1)
    matrix[0, : C + 1] = True
    kk = kfull.transpose(0, 1)[None].repeat_interleave(Hq // Hkv, dim=1)
    vv = vfull.transpose(0, 1)[None].repeat_interleave(Hq // Hkv, dim=1)
    ref = (
        torch.nn.functional.scaled_dot_product_attention(
            q.unsqueeze(0).transpose(1, 2).float(),
            kk.float(),
            vv.float(),
            attn_mask=matrix[None, None],
            scale=128**-0.5,
        )
        .transpose(1, 2)
        .squeeze(0)
    )
    assert (out.float() - ref).abs().max().item() < TOL


@requires_sm120_fa4
def test_fa4_tree_mask_gates_attention(qwen3_config_path):
    """The df2t verify forward on the real FlashAttentionImpl+metadata honors
    the attached FA4 tree mask_mod: attaching build_fa4_tree_mask to the
    FlashAttentionMetadata and running the FA4 impl matches the manual
    block-slot SDPA reference, so sibling nodes do not leak."""
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionImpl,
        FlashAttentionMetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import KVCacheLayout

    vllm_config = create_vllm_config(
        model_name=qwen3_config_path,
        max_model_len=64,
        block_size=16,
        num_gpu_blocks=1024,
        max_num_seqs=8,
        max_num_batched_tokens=256,
    )
    vllm_config.speculative_config = _df2t_spec_config()
    head_size = vllm_config.model_config.get_head_size()
    num_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config
    )
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(
        vllm_config.parallel_config
    )
    scale = 1.0 / (head_size**0.5)

    specs = _tree_specs()
    num_reqs = len(specs)
    query_lens = [b + 1 for _, _, b in specs]
    seq_lens = [c + 1 + b for c, _, b in specs]
    max_b = max(b for _, _, b in specs)

    batch_spec = BatchSpec(seq_lens=seq_lens, query_lens=query_lens, name="df2t_tree")
    common = create_common_attn_metadata(
        batch_spec, vllm_config.cache_config.block_size, DEVICE
    )

    torch.manual_seed(0)
    k_full, v_full, k_contexts, v_contexts = [], [], [], []
    for c, _, b_nodes in specs:
        seq_len = c + 1 + b_nodes
        k_full.append(
            torch.randn(seq_len, num_kv_heads, head_size, device=DEVICE, dtype=DTYPE)
        )
        v_full.append(
            torch.randn(seq_len, num_kv_heads, head_size, device=DEVICE, dtype=DTYPE)
        )
        k_contexts.append(k_full[-1][:c])
        v_contexts.append(v_full[-1][:c])

    kv_cache = create_and_prepopulate_kv_cache(
        k_contexts=k_contexts,
        v_contexts=v_contexts,
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=DTYPE,
        device=DEVICE,
        num_blocks=vllm_config.cache_config.num_gpu_blocks or 1024,
        common_attn_metadata=common,
        layout=KVCacheLayout.LBNHC,
    )

    # Manual SDPA reference over the verify-block query rows (block-slots
    # C..C+B) against each request's full sequence keys. Run in fp32.
    q_rows, k_rows, v_rows, sdpa_rows = [], [], [], []
    for r, (c, w, b_nodes) in enumerate(specs):
        q_r = torch.randn(b_nodes + 1, num_heads, head_size, device=DEVICE, dtype=DTYPE)
        kfull = k_full[r].transpose(0, 1)[None]
        vfull = v_full[r].transpose(0, 1)[None]
        if num_heads != num_kv_heads:
            kfull = kfull.repeat_interleave(num_heads // num_kv_heads, dim=1)
            vfull = vfull.repeat_interleave(num_heads // num_kv_heads, dim=1)
        mask = _manual_matrix(c, w, b_nodes)[c:].unsqueeze(0).unsqueeze(0).to(DEVICE)
        out = (
            torch.nn.functional.scaled_dot_product_attention(
                q_r.unsqueeze(0).transpose(1, 2).float(),
                kfull.float(),
                vfull.float(),
                attn_mask=mask,
                scale=scale,
            )
            .transpose(1, 2)
            .squeeze(0)
            .to(DTYPE)
        )
        q_rows.append(q_r)
        k_rows.append(k_full[r][c:])
        v_rows.append(v_full[r][c:])
        sdpa_rows.append(out)

    query = torch.cat(q_rows, dim=0)
    key = torch.cat(k_rows, dim=0)
    value = torch.cat(v_rows, dim=0)
    sdpa_output = torch.cat(sdpa_rows, dim=0)

    c_per_req = torch.tensor([c for c, _, _ in specs], dtype=torch.int32, device=DEVICE)
    words = torch.zeros(num_reqs, max_b, dtype=torch.int64, device=DEVICE)
    for r, (_, w, _) in enumerate(specs):
        words[r, : len(w)] = torch.tensor(w, dtype=torch.int64, device=DEVICE)

    kv_spec = create_standard_kv_cache_spec(vllm_config)
    with set_current_vllm_config(vllm_config):
        builder = FlashAttentionMetadataBuilder(
            kv_spec, ["placeholder"], vllm_config, DEVICE
        )
        attn_metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)
        mask_mod, aux = build_fa4_tree_mask(num_reqs, c_per_req, words, max_b)
        attn_metadata.df2t_tree_mask_mod = mask_mod
        attn_metadata.df2t_tree_aux_tensors = aux
        impl = FlashAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            attn_type=AttentionType.DECODER,
            kv_cache_dtype="auto",
        )

    layer = MockAttentionLayer(DEVICE)
    output = torch.empty_like(query)
    impl.do_kv_cache_update(layer, key, value, kv_cache, attn_metadata.slot_mapping)
    backend_output = impl.forward(
        layer, query, key, value, kv_cache, attn_metadata, output=output
    )

    assert backend_output.shape == sdpa_output.shape
    err = (backend_output.float() - sdpa_output.float()).abs().max().item()
    assert err < 2e-2, f"FA4 impl tree mask diverged from manual reference: {err}"


def _fake_fa4_metadata():
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    meta = object.__new__(FlashAttentionMetadata)
    meta.df2t_tree_mask_mod = None
    meta.df2t_tree_aux_tensors = None
    return meta


class _FakeBackend:
    def __init__(self, name: str):
        self._name = name

    def get_name(self):
        return self._name


def _fake_tree(words, device) -> "object":
    return type(
        "Tree", (), {"words": torch.tensor(words, dtype=torch.long, device=device)}
    )()


def _fake_fa4_kv_cache_config(block_size, layer_names=("target.attn", "target.attn2")):
    import types

    spec = types.SimpleNamespace(block_size=block_size)
    group = types.SimpleNamespace(layer_names=list(layer_names), kv_cache_spec=spec)
    return types.SimpleNamespace(kv_cache_groups=[group])


def _fake_runner(spec, req_states, device, attn_groups, kv_cache_config=None):
    import types

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    if kv_cache_config is None:
        kv_cache_config = _fake_fa4_kv_cache_config(block_size=16)
    runner = types.SimpleNamespace(
        speculative_config=spec,
        req_states=req_states,
        device=device,
        _pending_tree_flex_mask=None,
        _pending_tree_fa4_mask=None,
        attn_groups=attn_groups,
        kv_cache_config=kv_cache_config,
    )
    for name in (
        "_stage_tree_fa4_mask",
        "_fa4_tree_verify_layer_names",
        "_fa4_tree_page_sizes",
        "_attach_tree_fa4_mask",
    ):
        setattr(runner, name, types.MethodType(getattr(GPUModelRunner, name), runner))
    return runner


def test_runner_stage_attach_fa4_tree_mask():
    """The runner's fa4 stage -> attach path wires build_fa4_tree_mask onto the
    target FlashAttention verify group only, keyed by per-request C and words,
    leaving a non-FlashAttention metadata (another group) untouched."""
    device = torch.device("cpu")
    spec = type(
        "Spec",
        (),
        {"tree_attention_backend": "fa4", "max_tree_budget": 64},
    )()
    num_computed = torch.tensor([4, 2, 0, 0], dtype=torch.int32)
    req_states = type(
        "ReqStates", (), {"num_computed_tokens": type("T", (), {"gpu": num_computed})()}
    )()
    trees = [_fake_tree([0b1, 0b11, 0b101], device), _fake_tree([0b1, 0b11], device)]
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    attn_groups = [
        [
            type(
                "G",
                (),
                {
                    "backend": _FakeBackend("FLASH_ATTN"),
                    "layer_names": ["target.attn", "target.attn2"],
                },
            )(),
            type(
                "G",
                (),
                {
                    "backend": _FakeBackend("FLEX_ATTENTION"),
                    "layer_names": ["other.attn"],
                },
            )(),
        ]
    ]
    runner = _fake_runner(spec, req_states, device, attn_groups)

    runner._stage_tree_fa4_mask(True, trees, 2, idx_mapping)
    assert runner._pending_tree_fa4_mask is not None

    target_meta = _fake_fa4_metadata()
    other_meta = _fake_fa4_metadata()
    runner._attach_tree_fa4_mask({"target.attn": target_meta, "other.attn": other_meta})
    assert runner._pending_tree_fa4_mask is None
    assert target_meta.df2t_tree_mask_mod is not None
    assert target_meta.df2t_tree_aux_tensors is not None
    assert target_meta.df2t_tree_aux_tensors[0].shape == (2, 64)
    assert target_meta.df2t_tree_aux_tensors[1].tolist() == [4, 2]
    assert other_meta.df2t_tree_mask_mod is None
    assert other_meta.df2t_tree_aux_tensors is None


def test_runner_stage_fa4_scopes_zero_context():
    """A zero-context (fresh-prompt prefill, C == 0) request staged in an fa4
    tree batch is scoped per-request (causal over its own rows: see
    ``_make_fa4_tree_mask_mod``) instead of rejected, so it can verify its
    first token alongside a tree decode."""
    device = torch.device("cpu")
    spec = type("Spec", (), {"tree_attention_backend": "fa4", "max_tree_budget": 64})()
    num_computed = torch.tensor([4, 0], dtype=torch.int32)
    req_states = type(
        "ReqStates", (), {"num_computed_tokens": type("T", (), {"gpu": num_computed})()}
    )()
    trees = [_fake_tree([0b1, 0b11], device)]
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    runner = _fake_runner(
        spec,
        req_states,
        device,
        [
            [
                type(
                    "G",
                    (),
                    {
                        "backend": _FakeBackend("FLASH_ATTN"),
                        "layer_names": ["target.attn"],
                    },
                )()
            ]
        ],
    )
    # No assert: the contextless request is staged with a zero anchor length.
    runner._stage_tree_fa4_mask(True, trees, 2, idx_mapping)
    pending = runner._pending_tree_fa4_mask
    assert pending is not None
    _, c_per_req, words, _ = pending
    assert c_per_req.tolist() == [4, 0]
    assert bool((words[1] == 0).all().item())


def test_runner_attach_fa4_fails_closed_when_page_size_undeterminable():
    """Attach fails closed if the tree group's KV page size is undeterminable
    (no kv_cache group lists the tree layers): never attach a mask whose
    kv_idx semantics depend on an unknown page size."""
    device = torch.device("cpu")
    spec = type("Spec", (), {"tree_attention_backend": "fa4", "max_tree_budget": 64})()
    num_computed = torch.tensor([200, 4, 0, 0], dtype=torch.int32)
    req_states = type(
        "ReqStates", (), {"num_computed_tokens": type("T", (), {"gpu": num_computed})()}
    )()
    trees = [_fake_tree([0b1, 0b11], device), _fake_tree([0b1], device)]
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    # A kv_cache_config whose groups do not list the tree layers.
    kv_cache_config = _fake_fa4_kv_cache_config(
        block_size=16, layer_names=("other.attn",)
    )
    runner = _fake_runner(
        spec,
        req_states,
        device,
        [
            [
                type(
                    "G",
                    (),
                    {
                        "backend": _FakeBackend("FLASH_ATTN"),
                        "layer_names": ["target.attn"],
                    },
                )()
            ]
        ],
        kv_cache_config=kv_cache_config,
    )
    runner._stage_tree_fa4_mask(True, trees, 2, idx_mapping)
    assert runner._pending_tree_fa4_mask is not None
    with pytest.raises(RuntimeError, match="page size"):
        runner._attach_tree_fa4_mask({"target.attn": _fake_fa4_metadata()})
    # The pending field is cleared even on failure.
    assert runner._pending_tree_fa4_mask is None


def test_runner_attach_fa4_fails_closed_when_staged_mask_dropped():
    """A staged fa4 tree mask must never silently vanish: if no attention
    metadata is produced (attn_metadata is None) on a df2t fa4-tree batch, the
    attach raises instead of dropping the mask and running the verify forward
    without it."""
    device = torch.device("cpu")
    spec = type("Spec", (), {"tree_attention_backend": "fa4", "max_tree_budget": 64})()
    num_computed = torch.tensor([4, 2, 0, 0], dtype=torch.int32)
    req_states = type(
        "ReqStates", (), {"num_computed_tokens": type("T", (), {"gpu": num_computed})()}
    )()
    trees = [_fake_tree([0b1, 0b11], device), _fake_tree([0b1], device)]
    runner = _fake_runner(
        spec,
        req_states,
        device,
        [
            [
                type(
                    "G",
                    (),
                    {
                        "backend": _FakeBackend("FLASH_ATTN"),
                        "layer_names": ["target.attn"],
                    },
                )()
            ]
        ],
    )
    runner._stage_tree_fa4_mask(True, trees, 2, torch.tensor([0, 1], dtype=torch.int32))
    assert runner._pending_tree_fa4_mask is not None
    with pytest.raises(RuntimeError, match="dropped"):
        runner._attach_tree_fa4_mask(None)
    assert runner._pending_tree_fa4_mask is None


def test_runner_attach_fa4_fails_closed_without_fa4_metadata():
    """If the staged fa4 tree mask finds no FlashAttentionMetadata on the
    target layers (e.g. a non-FlashAttention vehicle was actually chosen), the
    attach fails closed with a clear error instead of silently dropping the
    mask."""
    device = torch.device("cpu")
    spec = type("Spec", (), {"tree_attention_backend": "fa4", "max_tree_budget": 64})()
    num_computed = torch.tensor([4, 2, 0, 0], dtype=torch.int32)
    req_states = type(
        "ReqStates", (), {"num_computed_tokens": type("T", (), {"gpu": num_computed})()}
    )()
    trees = [_fake_tree([0b1, 0b11, 0b101], device), _fake_tree([0b1, 0b11], device)]
    runner = _fake_runner(
        spec,
        req_states,
        device,
        [
            [
                type(
                    "G",
                    (),
                    {
                        "backend": _FakeBackend("FLEX_ATTENTION"),
                        "layer_names": ["target.attn"],
                    },
                )()
            ]
        ],
    )
    runner._stage_tree_fa4_mask(True, trees, 2, torch.tensor([0, 1], dtype=torch.int32))
    with pytest.raises(ValueError, match="fa4"):
        runner._attach_tree_fa4_mask({"target.attn": _fake_fa4_metadata()})
