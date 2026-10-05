# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic-head tests for the df2t confidence-head model.

``DF2TQwen3ForCausalLM`` extends ``DFlash2Qwen3ForCausalLM`` with a
DSpark-style ``nn.Linear(hidden_size, 1)`` acceptance-confidence head named
``df2t_conf_head``, gated on ``dflash_config.enable_confidence_head``. No real
head-bearing checkpoint exists yet (it will come from a training run), so these
tests build the module from a tiny local Qwen3 config (fully offline) and
validate head construction, the ``compute_confidence`` contract, the
load-time guards, and the registry entry.
"""

import json
import logging

import pytest
import torch

from tests.v1.attention.utils import create_vllm_config
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.qwen3_df2t import DF2TQwen3ForCausalLM
from vllm.model_executor.models.registry import ModelRegistry

_TINY_QWEN3_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 64,
    "intermediate_size": 192,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 16,
    "vocab_size": 2048,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
    "sliding_window": None,
    "use_sliding_window": False,
    "bos_token_id": 0,
    "eos_token_id": 1,
}


def _write_local_config(tmp_path) -> str:
    root = tmp_path / "qwen3-df2t-tiny"
    root.mkdir(exist_ok=True)
    (root / "config.json").write_text(json.dumps(_TINY_QWEN3_CONFIG))
    return str(root)


def _make_vllm_config(model_path: str, enable_confidence_head: bool) -> VllmConfig:
    """A real offline VllmConfig with a df2t draft (DFlash2 backbone).

    The draft and target share one small Qwen3 config, as in the df2t suite's
    conftest pattern; ``hf_config`` is amended with the drafted
    ``dflash_config`` so the DF2T model source lives in the test, not in remote
    checkpoints.
    """
    vllm_config = create_vllm_config(
        model_name=model_path,
        dtype=torch.float32,
        max_model_len=64,
        block_size=16,
        num_gpu_blocks=64,
        max_num_seqs=8,
        max_num_batched_tokens=256,
    )
    vllm_config.model_config.hf_config.update(
        {
            "architectures": ["DFlash2DraftModel"],
            "dflash_config": {
                "block_size": 16,
                "conv_kernel_size": 3,
                "conv_group_size": 4,
                "selector_rank": 4,
                "selector_top_k": 3,
                "enable_confidence_head": enable_confidence_head,
            },
        }
    )
    vllm_config.model_config.model_arch_config.architectures = ["DFlash2DraftModel"]
    vllm_config.speculative_config = SpeculativeConfig(
        method="dflash",
        tree_selection="df2t",
        num_speculative_tokens=15,
        draft_model_config=vllm_config.model_config,
        draft_parallel_config=vllm_config.parallel_config,
        tree_budget=None,
    )
    return vllm_config


def _build_model(tmp_path, enable_confidence_head: bool):
    vllm_config = _make_vllm_config(
        _write_local_config(tmp_path), enable_confidence_head
    )
    with set_current_vllm_config(vllm_config):
        return DF2TQwen3ForCausalLM(vllm_config=vllm_config)


def test_confidence_head_is_gated_on_config(tmp_path, dist_init):
    enabled = _build_model(tmp_path, enable_confidence_head=True)
    assert isinstance(enabled.df2t_conf_head, torch.nn.Linear)
    assert (
        enabled.df2t_conf_head.in_features,
        enabled.df2t_conf_head.out_features,
    ) == (64, 1)

    disabled = _build_model(tmp_path, enable_confidence_head=False)
    assert disabled.df2t_conf_head is None


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_compute_confidence_returns_per_token_float32(tmp_path, dist_init, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = _build_model(tmp_path, enable_confidence_head=True)
    model.df2t_conf_head = model.df2t_conf_head.to(device)
    hidden_states = torch.randn(2, 64, device=device)

    confidence = model.compute_confidence(hidden_states)

    assert confidence.shape == (2,)
    assert confidence.dtype == torch.float32


def test_confidence_head_loads_synthetic_weights(tmp_path, dist_init):
    model = _build_model(tmp_path, enable_confidence_head=True)
    head_weight = torch.randn(1, 64)
    head_bias = torch.randn(1)
    weights = [
        ("df2t_conf_head.weight", head_weight),
        ("df2t_conf_head.bias", head_bias),
    ]

    model.load_weights(weights)

    torch.testing.assert_close(model.df2t_conf_head.weight, head_weight)
    torch.testing.assert_close(model.df2t_conf_head.bias, head_bias)


def test_confidence_head_warns_on_orphan_weights(tmp_path, dist_init, caplog):
    model = _build_model(tmp_path, enable_confidence_head=False)

    with caplog.at_level(logging.WARNING):
        model.load_weights([("df2t_conf_head.weight", torch.randn(1, 64))])

    assert model.df2t_conf_head is None
    assert any("df2t_conf_head" in record.message for record in caplog.records)


def test_confidence_head_raises_when_enabled_without_weights(tmp_path, dist_init):
    model = _build_model(tmp_path, enable_confidence_head=True)

    with pytest.raises(RuntimeError, match="df2t_conf_head"):
        model.load_weights([])


def test_confidence_head_rejects_incomplete_weights(tmp_path, dist_init):
    """An enabled head must fail loudly, not with a raw KeyError or a silently
    random bias, when the checkpoint supplies partial head weights."""
    bias_only = _build_model(tmp_path, enable_confidence_head=True)
    with pytest.raises(RuntimeError, match="df2t_conf_head.*incomplete"):
        bias_only.load_weights([("df2t_conf_head.bias", torch.randn(1))])

    weight_only = _build_model(tmp_path, enable_confidence_head=True)
    with pytest.raises(RuntimeError, match="df2t_conf_head.*incomplete"):
        weight_only.load_weights([("df2t_conf_head.weight", torch.randn(1, 64))])


def test_compute_confidence_raises_without_head(tmp_path, dist_init):
    model = _build_model(tmp_path, enable_confidence_head=False)
    with pytest.raises(RuntimeError, match="enable_confidence_head"):
        model.compute_confidence(torch.randn(2, 64))


def test_registry_resolves_df2t_draft_model():
    assert (
        ModelRegistry.models["DF2TDraftModel"].load_model_cls() is DF2TQwen3ForCausalLM
    )
