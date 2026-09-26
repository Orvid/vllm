# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DF2T speculator dispatch in ``init_speculator``."""

from types import SimpleNamespace
from typing import Any

import torch

from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.registry import ModelRegistry
from vllm.v1.worker.gpu.spec_decode import init_speculator

ARCH = "DFlash2DraftModel"


def _fake_draft_model_config() -> SimpleNamespace:
    """A lightweight draft ``ModelConfig`` stand-in exposing the surface the
    DFlash2 speculator ``__init__`` reads (block_size, selector_top_k,
    mask_token_id)."""
    return SimpleNamespace(
        architectures=[ARCH],
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


def _fake_vllm_config() -> Any:
    """Minimal config that lets the DF2T speculator ``__init__`` run on CPU."""
    speculative_config = SpeculativeConfig(
        method="df2t",
        num_speculative_tokens=15,
        draft_model_config=_fake_draft_model_config(),
        tree_budget=None,
    )
    model_config = SimpleNamespace(
        model="some/model",
        max_model_len=128,
        dtype=torch.float32,
        use_fp64_gumbel=False,
        convert_type="none",
        runner_type="generate",
        trust_remote_code=False,
        model_impl="vllm",
        registry=ModelRegistry,
        _get_transformers_backend_cls=lambda: "TransformersForCausalLM",
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
    )
    return SimpleNamespace(
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=512),
        model_config=model_config,
        parallel_config=SimpleNamespace(
            data_parallel_size=1, data_parallel_rank=0, prefill_context_parallel_size=1
        ),
    )


def test_init_speculator_routes_df2t() -> None:
    from vllm.v1.worker.gpu.spec_decode.df2t.speculator import DF2TSpeculator

    spec = init_speculator(_fake_vllm_config(), torch.device("cpu"))

    assert isinstance(spec, DF2TSpeculator)
    assert spec.num_query_per_req == 1 + 15
