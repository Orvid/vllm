# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tree verification through attention backends' own builders and impls.

Each backend builds its metadata from ``CommonAttentionMetadata.tree_verify``
(``supports_tree_verify``); the forward must match SDPA under the mask
derived from the trees' parents, eagerly and replayed from a CUDA graph.
"""

import types

import pytest
import torch

from tests.v1.attention.test_attention_backends import (
    create_and_prepopulate_kv_cache,
    run_attention_backend,
)
from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from tests.v1.worker.gpu.spec_decode.tree.test_tree_attention_kernels import (
    Req,
    _build,
    _reference,
    _requests,
)
from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheLayout

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="needs CUDA")

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
BLOCK_SIZE = 64


def _spec_config():
    return types.SimpleNamespace(
        method="dflash",
        use_tree=lambda: True,
        use_df2t=lambda: True,
        max_tree_budget=64,
        num_draft_rows_per_req=64,
        num_speculative_tokens=5,
        tree_attention_backend="fa4",
        enable_adaptive_verification=False,
    )


def _supported(backend: AttentionBackendEnum) -> bool:
    if backend == AttentionBackendEnum.B12X:
        from vllm.utils.b12x import get_b12x_paged_attention

        paged = get_b12x_paged_attention()
        return paged is not None and paged.is_supported()
    if backend == AttentionBackendEnum.FLASH_ATTN:
        from vllm.v1.attention.backends.fa_utils import is_fa_version_supported

        return is_fa_version_supported(2)
    return True


def _packed_b12x_cache(kv_cache: torch.Tensor, head_size: int) -> torch.Tensor:
    key = kv_cache[..., :head_size].permute(0, 2, 1, 3)
    value = kv_cache[..., head_size:].permute(0, 2, 1, 3)
    packed = torch.stack((key, value), dim=1).flatten(-2)
    out = torch.zeros(
        (packed.shape[0], packed.shape[2], packed.shape[1], packed.shape[3]),
        dtype=packed.dtype,
        device=packed.device,
    ).permute(0, 2, 1, 3)
    out.copy_(packed)
    return out


@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize(
    "backend",
    [
        AttentionBackendEnum.TRITON_ATTN,
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.B12X,
    ],
)
def test_backend_verifies_trees(
    backend, use_cuda_graph, qwen3_config_path, workspace_init
):
    if not _supported(backend):
        pytest.skip(f"{backend.name} is not available here")
    reqs: list[Req] = _requests()
    if backend == AttentionBackendEnum.B12X:
        # b12x verifies trees through relative bias but cannot window them;
        # a chunk too long for one extend plan would go to the next bucket.
        reqs = reqs[:3]
    vllm_config = create_vllm_config(
        model_name=qwen3_config_path,
        max_model_len=1024,
        block_size=BLOCK_SIZE,
        num_gpu_blocks=64,
        max_num_seqs=8,
        max_num_batched_tokens=512,
    )
    vllm_config.speculative_config = _spec_config()
    # FA2 splits the verify; FA4 (the default on SM120) masks it instead.
    vllm_config.attention_config.flash_attn_version = 2
    num_heads = vllm_config.model_config.get_num_attention_heads(
        vllm_config.parallel_config
    )
    num_kv_heads = vllm_config.model_config.get_num_kv_heads(
        vllm_config.parallel_config
    )
    head_size = vllm_config.model_config.get_head_size()

    batch = _build(reqs)
    common = create_common_attn_metadata(
        BatchSpec(
            seq_lens=[r.context + r.rows for r in reqs],
            query_lens=[r.rows for r in reqs],
        ),
        BLOCK_SIZE,
        DEVICE,
    )
    assert torch.equal(common.query_start_loc, batch.query_start_loc)
    common.tree_verify = batch.ctx

    gen = torch.Generator(device=DEVICE).manual_seed(0)
    ks = [
        torch.randn(
            r.context + r.rows,
            num_kv_heads,
            head_size,
            dtype=DTYPE,
            device=DEVICE,
            generator=gen,
        )
        for r in reqs
    ]
    vs = [torch.randn_like(k) for k in ks]
    num_tokens = int(batch.query_start_loc[-1])
    query = torch.randn(
        num_tokens, num_heads, head_size, dtype=DTYPE, device=DEVICE, generator=gen
    )
    key = torch.cat([k[r.context :] for k, r in zip(ks, reqs)])
    value = torch.cat([v[r.context :] for v, r in zip(vs, reqs)])

    kv_cache = create_and_prepopulate_kv_cache(
        k_contexts=[k[: r.context] for k, r in zip(ks, reqs)],
        v_contexts=[v[: r.context] for v, r in zip(vs, reqs)],
        block_size=BLOCK_SIZE,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=DTYPE,
        device=DEVICE,
        num_blocks=64,
        common_attn_metadata=common,
        layout=KVCacheLayout.LBNHC,
    )
    if backend == AttentionBackendEnum.B12X:
        kv_cache = _packed_b12x_cache(kv_cache, head_size)

    with set_current_vllm_config(vllm_config):
        output = run_attention_backend(
            backend,
            create_standard_kv_cache_spec(vllm_config),
            ["placeholder"],
            vllm_config,
            DEVICE,
            common,
            query,
            key,
            value,
            kv_cache,
            use_cuda_graph=use_cuda_graph,
        )
    ref = _reference(reqs, batch, query, ks, vs, head_size**-0.5)
    torch.testing.assert_close(output.float(), ref, atol=2e-2, rtol=2e-2)
