# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fixtures for the tree verify spec-decode tests."""

import json

import pytest

_MINIMAL_QWEN3_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
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


@pytest.fixture
def qwen3_config_path(tmp_path):
    """A local directory holding a minimal Qwen3-0.6B config.json.

    The tree verify tests build real VllmConfigs to exercise attention/metadata
    machinery that needs model shape fields (head_size/heads/etc.). Resolving
    a remote HuggingFace id would require network access, so these tests use
    a locally-written config directory instead — fully offline and portable.
    """
    root = tmp_path / "qwen3-0.6b"
    root.mkdir()
    (root / "config.json").write_text(json.dumps(_MINIMAL_QWEN3_CONFIG))
    return str(root)
