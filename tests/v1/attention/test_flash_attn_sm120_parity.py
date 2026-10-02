# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FA4 parity and selection on SM120 (RTX 5090, compute capability 12.0).

FA4 is available on consumer Blackwell, but `get_flash_attn_version` only
selects it under df2t speculative decoding; without a df2t speculative config
SM120 keeps resolving to FA2 exactly as before. These tests pin both sides of
that gate and run the adopted FA4 CuTe-DSL forward against FA2 and a dense
float32 SDPA reference.

All configs are built offline from a locally-written minimal Qwen3-0.6B
config.json (no HuggingFace access), mirroring the df2t spec-decode fixtures.
"""

import json

import pytest
import torch

from vllm.platforms import current_platform

# --------------------------------------------------------------------------- #
# Promote definite analysis over uncertain assertions: this suite runs against
# the real kernel, so the SKIP guard must be airtight.
# --------------------------------------------------------------------------- #


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
    reason="FA4 on SM120 parity requires an SM120 device with FA4 available",
)

# Imported conditionally because these pull in CuTe / FA4 build artifacts that
# are absent wherever requires_sm120_fa4 skips.
if _sm120() and _fa4_available():  # pragma: no branch
    from tests.v1.attention.utils import (
        create_standard_kv_cache_spec,
        create_vllm_config,
    )
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backends.fa_utils import (
        get_flash_attn_version,
        is_fa_version_supported,
    )
    from vllm.v1.attention.backends.flash_attn import FA4DenseAttentionKernel
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    DEVICE = torch.device("cuda:0")
    DTYPE = torch.bfloat16
    HEAD_DIM = 128
    NUM_HEADS = 16
    NUM_KV_HEADS = 8
    BLOCK_SIZE = 128
    TOL = 3e-2

    _MINIMAL_QWEN3_CONFIG = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": NUM_HEADS,
        "num_key_value_heads": NUM_KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": 151936,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "tie_word_embeddings": True,
        "sliding_window": None,
        "use_sliding_window": False,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
    }


@pytest.fixture(scope="module")
def qwen3_config_path(tmp_path_factory):
    """A local directory holding a minimal Qwen3-0.6B config.json (offline)."""
    root = tmp_path_factory.mktemp("qwen3-0.6b") / "config.json"
    root.write_text(json.dumps(_MINIMAL_QWEN3_CONFIG))
    return str(root.parent)


# --------------------------------------------------------------------------- #
# Selection gate
# --------------------------------------------------------------------------- #


def _fake_spec_config(df2t: bool, flex: bool = False):
    import types

    return types.SimpleNamespace(
        method="df2t" if df2t else "dflash",
        use_df2t=lambda: df2t,
        use_df2t_tree=lambda: df2t,
        num_speculative_tokens=5,
        tree_attention_backend="flex" if flex else "fa4",
    )


def _make_cfg(qwen3_config_path, df2t, flex=False, flash_attn_version=None):
    cfg = create_vllm_config(
        model_name=qwen3_config_path,
        max_model_len=1024,
        block_size=BLOCK_SIZE,
        num_gpu_blocks=64,
    )
    cfg.speculative_config = _fake_spec_config(df2t=df2t, flex=flex)
    if flash_attn_version is not None:
        cfg.attention_config.flash_attn_version = flash_attn_version
    return cfg


def _real_df2t_vllm_config(qwen3_config_path, flash_attn_version=None):
    """A VllmConfig built with a REAL df2t SpeculativeConfig.

    Passing a real SpeculativeConfig means ``VllmConfig.__post_init__`` runs
    the df2t attention cross-validation (``_validate_df2t_attention``) at
    construction time, exercising the wiring rather than an isolated method.
    """
    from types import SimpleNamespace

    from vllm.config import VllmConfig
    from vllm.config.speculative import SpeculativeConfig

    base = create_vllm_config(
        model_name=qwen3_config_path,
        max_model_len=1024,
        block_size=BLOCK_SIZE,
        num_gpu_blocks=64,
    )
    if flash_attn_version is not None:
        base.attention_config.flash_attn_version = flash_attn_version
    # df2t is incompatible with async scheduling; the base config resolved it
    # to True without a speculative config, so reset it for the real df2t run.
    base.scheduler_config.async_scheduling = False
    draft_model_config = SimpleNamespace(
        architectures=["DFlash2DraftModel"],
        hf_config=SimpleNamespace(
            num_hidden_layers=2,
            block_size=16,
            dflash_config={
                "block_size": 16,
                "selector_top_k": 3,
                "mask_token_id": 151643,
            },
        ),
        model="draft",
        max_model_len=128,
        get_hidden_size=lambda: 64,
        get_vocab_size=lambda: 4096,
        verify_with_parallel_config=lambda *_: None,
        compute_hash=lambda: "draft-hash",
    )
    speculative_config = SpeculativeConfig(
        method="df2t",
        num_speculative_tokens=15,
        draft_model_config=draft_model_config,
        tree_budget=None,
    )
    return VllmConfig(
        model_config=base.model_config,
        cache_config=base.cache_config,
        parallel_config=base.parallel_config,
        scheduler_config=base.scheduler_config,
        device_config=base.device_config,
        load_config=base.load_config,
        compilation_config=base.compilation_config,
        attention_config=base.attention_config,
        speculative_config=speculative_config,
    )


@requires_sm120_fa4
def test_get_flash_attn_version_gates_fa4_on_df2t(qwen3_config_path):
    """SM120 resolves FA2 by default and only upgrades to FA4 under df2t.

    The gate is the active speculative config (via `set_current_vllm_config`),
    so a non-df2t speculative decoder must NOT pull in FA4.
    """
    assert (
        get_flash_attn_version(head_size=HEAD_DIM, kv_cache_block_size=BLOCK_SIZE) == 2
    )

    with set_current_vllm_config(_make_cfg(qwen3_config_path, df2t=False)):
        assert (
            get_flash_attn_version(head_size=HEAD_DIM, kv_cache_block_size=BLOCK_SIZE)
            == 2
        )

    with set_current_vllm_config(_make_cfg(qwen3_config_path, df2t=True)):
        assert (
            get_flash_attn_version(head_size=HEAD_DIM, kv_cache_block_size=BLOCK_SIZE)
            == 4
        )

    assert is_fa_version_supported(4)


@requires_sm120_fa4
def test_df2t_forced_fa2_override_raises(qwen3_config_path):
    """Explicitly forcing FA2 under df2t must error, never silently fall back.

    The allowed set is {fa4, flex}: an explicit ``flash_attn_version=2``
    override is rejected both at VllmConfig construction (real df2t
    SpeculativeConfig, so ``__post_init__`` runs the cross-validation) and at
    runtime selection.
    """
    with pytest.raises(ValueError, match="df2t"):
        _real_df2t_vllm_config(qwen3_config_path, flash_attn_version=2)

    cfg = _make_cfg(qwen3_config_path, df2t=True, flash_attn_version=2)
    with set_current_vllm_config(cfg), pytest.raises(RuntimeError, match="df2t"):
        get_flash_attn_version(head_size=HEAD_DIM, kv_cache_block_size=BLOCK_SIZE)


def _minimal_df2t_validate_config(*, rswa_window=None, is_mm_prefix_lm=False):
    """A lightweight VllmConfig exercising ``_validate_df2t_attention`` alone.

    ``_validate_df2t_attention`` only reads the speculative config's
    ``use_df2t()`` (confirmed df2t) plus the attention/model surface relevant
    to the df2t mask vehicles. This avoids rebuilding a full model just to
    reject an R-SWA or prefix-LM target.
    """
    from types import SimpleNamespace

    from vllm.config import VllmConfig

    cfg = object.__new__(VllmConfig)
    cfg.speculative_config = _fake_spec_config(df2t=True)
    cfg.attention_config = SimpleNamespace(
        flash_attn_version=None, backend=None, backend_per_kind={}
    )
    cfg.model_config = SimpleNamespace(
        rswa_window=rswa_window, is_mm_prefix_lm=is_mm_prefix_lm
    )
    return cfg


@requires_sm120_fa4
def test_df2t_validate_rejects_rswa_target(qwen3_config_path):
    """df2t + an R-SWA target must fail closed at config time: the FA4 commit
    kernel dispatches ``rswa_mask_mod_fn or mm_mask_mod or df2t_tree_mask_mod``,
    so an R-SWA target (causal already False on R-SWA layers) would silently
    drop the tree mask and let tree siblings attend each other."""
    cfg = _minimal_df2t_validate_config(rswa_window=128)
    with pytest.raises(ValueError, match="R-SWA"):
        cfg._validate_df2t_attention()


@requires_sm120_fa4
def test_df2t_validate_rejects_mm_prefix_target(qwen3_config_path):
    """df2t + a prefix-LM (mm-prefix) target must fail closed at config time:
    the mm-prefix ``mask_mod`` precedes the tree mask in the commit kernel
    dispatch, so an mm-prefix target would silently drop the tree mask."""
    cfg = _minimal_df2t_validate_config(is_mm_prefix_lm=True)
    with pytest.raises(ValueError, match="mm-prefix"):
        cfg._validate_df2t_attention()


@requires_sm120_fa4
def test_df2t_validate_accepts_plain_target(qwen3_config_path):
    """A plain (non-R-SWA, non-mm-prefix) df2t target passes the df2t attention
    validation unchanged."""
    cfg = _minimal_df2t_validate_config()
    cfg._validate_df2t_attention()  # no raise

    # FA4 is accepted by both seams.
    _real_df2t_vllm_config(qwen3_config_path, flash_attn_version=4)
    cfg = _make_cfg(qwen3_config_path, df2t=True, flash_attn_version=4)
    with set_current_vllm_config(cfg):
        assert (
            get_flash_attn_version(head_size=HEAD_DIM, kv_cache_block_size=BLOCK_SIZE)
            == 4
        )


@requires_sm120_fa4
def test_df2t_forced_fa3_override_raises_on_major10(monkeypatch, qwen3_config_path):
    """df2t + an explicit fa3 override on a Blackwell major raises like fa2.

    The pre-existing major>=10 fallback normalizes an explicit fa3 to FA4 for
    non-df2t runs; under df2t (non-flex) that silent normalization must not
    apply — the override raises at runtime too. Non-df2t behavior is
    unchanged.
    """
    from vllm.platforms.interface import DeviceCapability
    from vllm.vllm_flash_attn import flash_attn_interface as fai

    platform_cls = current_platform.__class__
    monkeypatch.setattr(
        platform_cls,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(major=10, minor=0)),
    )
    monkeypatch.setattr(fai, "_is_fa4_supported", lambda: (True, None))

    with pytest.raises(ValueError, match="df2t"):
        _real_df2t_vllm_config(qwen3_config_path, flash_attn_version=3)

    with (
        set_current_vllm_config(
            _make_cfg(qwen3_config_path, df2t=True, flash_attn_version=3)
        ),
        pytest.raises(RuntimeError, match="df2t"),
    ):
        get_flash_attn_version(head_size=HEAD_DIM, kv_cache_block_size=BLOCK_SIZE)

    # Non-df2t + explicit fa3 on major>=10 still normalizes to FA4.
    with set_current_vllm_config(
        _make_cfg(qwen3_config_path, df2t=False, flash_attn_version=3)
    ):
        assert (
            get_flash_attn_version(head_size=HEAD_DIM, kv_cache_block_size=BLOCK_SIZE)
            == 4
        )


@requires_sm120_fa4
def test_df2t_force_fa4_spans_other_majors(monkeypatch, qwen3_config_path):
    """df2t forces FA4 on every architecture, not just SM120.

    The real device is SM120; we mock the platform capability to a non-sm120
    major and the FA4-support probe so the all-arch rule (df2t overrides the
    SM90 FA3 default) and the flex escape hatch are exercised offline.
    """
    from vllm.platforms.interface import DeviceCapability
    from vllm.vllm_flash_attn import flash_attn_interface as fai

    platform_cls = current_platform.__class__

    monkeypatch.setattr(
        platform_cls,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(major=9, minor=0)),
    )
    monkeypatch.setattr(fai, "_is_fa4_supported", lambda: (True, None))

    # fa4 vehicle on a simulated SM90 major: FA4 wins over the FA3 default.
    with set_current_vllm_config(_make_cfg(qwen3_config_path, df2t=True)):
        assert (
            get_flash_attn_version(head_size=HEAD_DIM, kv_cache_block_size=BLOCK_SIZE)
            == 4
        )
    # flex vehicle is accepted too (no raise), FA4 still resolves on sm90.
    with set_current_vllm_config(_make_cfg(qwen3_config_path, df2t=True, flex=True)):
        assert (
            get_flash_attn_version(head_size=HEAD_DIM, kv_cache_block_size=BLOCK_SIZE)
            == 4
        )
    # fa4 unsupported + non-flex vehicle (simulated SM80) must raise...
    monkeypatch.setattr(
        platform_cls,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(major=8, minor=0)),
    )
    monkeypatch.setattr(fai, "_is_fa4_supported", lambda: (False, "not supported"))
    with (
        set_current_vllm_config(_make_cfg(qwen3_config_path, df2t=True)),
        pytest.raises(RuntimeError, match="df2t"),
    ):
        get_flash_attn_version(head_size=HEAD_DIM, kv_cache_block_size=BLOCK_SIZE)
    # ...while the flex vehicle stays a valid fa4-less handle.
    with set_current_vllm_config(_make_cfg(qwen3_config_path, df2t=True, flex=True)):
        get_flash_attn_version(head_size=HEAD_DIM, kv_cache_block_size=BLOCK_SIZE)


# --------------------------------------------------------------------------- #
# Parity
# --------------------------------------------------------------------------- #


def _cu_seqlens(lens):
    out = torch.zeros(len(lens) + 1, dtype=torch.int32)
    out[1:] = torch.tensor(lens, dtype=torch.int32).cumsum(0)
    return out.to(DEVICE)


def _build_paged(seq_lens):
    """Random Q/K/V plus a paged cache with one contiguous page run per seq."""
    torch.manual_seed(0)
    total = sum(seq_lens)
    q = torch.randn(total, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)
    max_seqlen = max(seq_lens)
    num_blocks = (max_seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE
    physical = torch.arange(
        len(seq_lens) * num_blocks, dtype=torch.int32, device=DEVICE
    ).view(len(seq_lens), num_blocks)
    k_cache = torch.randn(
        len(seq_lens) * num_blocks,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=DTYPE,
        device=DEVICE,
    )
    v_cache = torch.randn_like(k_cache)
    seqused_k = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    cu = _cu_seqlens(seq_lens)
    return q, k_cache, v_cache, physical, seqused_k, cu, max_seqlen


def _gather_paged_reference(
    q, k_cache, v_cache, physical, seqused_k, cu, scale, causal, window_size=(-1, -1)
):
    """Dense float32 (varlen) reference over the packed rows of the cache."""
    ref = torch.zeros_like(q, dtype=torch.float32)
    st = 0
    for i, slen in enumerate(seqused_k.tolist()):
        rows = []
        for page in physical[i].tolist():
            rows.append(k_cache[page])
        kk = (
            torch.cat(rows, 0)[:slen]
            .float()
            .repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
        )
        rows_v = []
        for page in physical[i].tolist():
            rows_v.append(v_cache[page])
        vv = (
            torch.cat(rows_v, 0)[:slen]
            .float()
            .repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
        )
        qi = q[st : st + slen].float()
        # (head, seq_q, head_dim) x (head, head_dim, seq_k) -> (head, seq_q, seq_k)
        attn = torch.matmul(qi.permute(1, 0, 2), kk.permute(1, 2, 0)) * scale
        r = torch.arange(slen, device=DEVICE)
        keep = torch.ones(slen, slen, dtype=torch.bool, device=DEVICE)
        if causal:
            keep &= r[:, None] >= r[None, :]
        if window_size != (-1, -1):
            left, right = window_size
            wl = left if left >= 0 else slen
            wr = right if right >= 0 else slen
            keep &= (r[None, :] >= r[:, None] - wr) & (r[None, :] <= r[:, None] + wl)
        attn = attn.masked_fill(~keep[None, :, :], float("-inf"))
        prob = torch.softmax(attn, dim=-1)
        # (head, seq_q, seq_k) x (head, seq_k, head_dim) -> (seq_q, head, head_dim)
        ref[st : st + slen] = torch.matmul(prob, vv.permute(1, 0, 2)).permute(1, 0, 2)
        st += slen
    return ref


def _run_paged_fa(
    q,
    k_cache,
    v_cache,
    physical,
    seqused_k,
    cu,
    max_seqlen,
    scale,
    fa_version,
    causal,
    window_size=(-1, -1),
):
    return flash_attn_varlen_func(
        q,
        k_cache,
        v_cache,
        max_seqlen_q=max_seqlen,
        cu_seqlens_q=cu,
        max_seqlen_k=max_seqlen,
        cu_seqlens_k=None,
        seqused_k=seqused_k,
        block_table=physical,
        softmax_scale=scale,
        fa_version=fa_version,
        causal=causal,
        window_size=window_size,
        num_splits=1,
    )


SEQLENS = [96, 256, 40]


@requires_sm120_fa4
def test_varlen_paged_causal_matches_fa2_and_sdpa():
    """FA4 varlen paged causal forward matches FA2 and a dense SDPA reference."""
    q, k_cache, v_cache, physical, seqused_k, cu, max_seqlen = _build_paged(SEQLENS)
    scale = HEAD_DIM**-0.5

    out4 = _run_paged_fa(
        q, k_cache, v_cache, physical, seqused_k, cu, max_seqlen, scale, 4, causal=True
    )
    out2 = _run_paged_fa(
        q, k_cache, v_cache, physical, seqused_k, cu, max_seqlen, scale, 2, causal=True
    )
    reference = _gather_paged_reference(
        q, k_cache, v_cache, physical, seqused_k, cu, scale, causal=True
    )

    assert torch.isclose(out4.float(), out2.float(), atol=TOL).all(), (
        "FA4 paged causal diverged from FA2"
    )
    assert (out4.float() - reference).abs().max() < TOL, (
        "FA4 paged causal diverged from the SDPA reference"
    )


@requires_sm120_fa4
def test_varlen_paged_noncausal_window_matches_sdpa():
    """A non-causal symmetric window on paged KV matches the SDPA reference."""
    q, k_cache, v_cache, physical, seqused_k, cu, max_seqlen = _build_paged(SEQLENS)
    scale = HEAD_DIM**-0.5
    window = (64, 64)

    out4 = _run_paged_fa(
        q,
        k_cache,
        v_cache,
        physical,
        seqused_k,
        cu,
        max_seqlen,
        scale,
        4,
        causal=False,
        window_size=window,
    )
    reference = _gather_paged_reference(
        q,
        k_cache,
        v_cache,
        physical,
        seqused_k,
        cu,
        scale,
        causal=False,
        window_size=window,
    )

    assert (out4.float() - reference).abs().max() < TOL, (
        "FA4 paged non-causal window diverged from the SDPA reference"
    )


@requires_sm120_fa4
def test_fa4_dense_warmup_keys_cover_decode_and_extend(qwen3_config_path):
    """The SM120 warmup emits the decode/extend keys needed by the vLLM path.

    WARMUP slots in as if a production startup ran `compile_flash_attn_varlen_*
    from_specs` for a target-decode attention layer: both q_stages (decode and
    extend) and both single/non-single-batch variants are emitted, and the
    decode single-batch key actually compiles on the SM120 device.
    """
    cfg = create_vllm_config(
        model_name=qwen3_config_path,
        max_model_len=2048,
        block_size=BLOCK_SIZE,
        num_gpu_blocks=64,
    )
    kv_cache_spec = create_standard_kv_cache_spec(cfg)

    keys = FA4DenseAttentionKernel().get_warmup_keys(
        vllm_config=cfg,
        kv_cache_spec=kv_cache_spec,
        num_qo_heads=NUM_HEADS,
    )
    q_stages = {k.q_stage for k in keys}
    assert q_stages == {1, 2}, f"expected decode+extend warmup keys, got {q_stages}"
    assert all(k.num_splits == 1 for k in keys)
    assert all(k.is_single_batch in (True, False) for k in keys)
    assert len(keys) == 4

    decode_key = next(k for k in keys if k.q_stage == 1 and k.is_single_batch)
    FA4DenseAttentionKernel().compile(decode_key)
