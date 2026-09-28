# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DF2T tree speculative decoding config surface."""

from types import SimpleNamespace
from typing import Any, get_args

import pytest

from vllm.config.speculative import SpeculativeConfig

ARCH = "DFlash2DraftModel"


def _draft_cfg(arch: str = ARCH) -> SimpleNamespace:
    """A lightweight stand-in for a draft ``ModelConfig`` whose ``hf_config``
    exposes ``dflash_config.block_size`` (16), the DFlash2 draft-block size."""
    return SimpleNamespace(
        architectures=[arch],
        hf_config=SimpleNamespace(
            block_size=16,
            dflash_config=SimpleNamespace(block_size=16),
        ),
        model="draft",
        max_model_len=128,
        verify_with_parallel_config=lambda *_: None,
        compute_hash=lambda: "draft-hash",
    )


def _build(arch: str = ARCH, **over: Any) -> SpeculativeConfig:
    kwargs: dict[str, Any] = dict(
        method="df2t",
        num_speculative_tokens=15,
        draft_model_config=_draft_cfg(arch),
        tree_budget=None,
    )
    kwargs.update(over)
    return SpeculativeConfig(**kwargs)


def test_df2t_method_registered():
    import vllm.config.speculative as m

    assert "df2t" in get_args(m.SpeculativeMethod)


def test_use_df2t():
    cfg = _build()
    assert cfg.use_df2t()
    cfg.method = "dflash"
    assert not cfg.use_df2t()


def test_df2t_tree_resolution():
    cfg = _build()
    cfg._validate_df2t()
    assert cfg.tree_budget == cfg.max_tree_budget == 64
    assert cfg.max_tree_depth == 15
    assert cfg.parallel_drafting


def test_df2t_rejects_wrong_arch():
    with pytest.raises(ValueError, match="df2t"):
        _build(arch="DFlashDraftModel")


def test_df2t_rejects_tree_budget_over_ceiling():
    with pytest.raises(ValueError, match="tree_budget"):
        _build(tree_budget=128)


def test_df2t_rejects_max_tree_depth_over_tokens():
    with pytest.raises(ValueError, match="max_tree_depth"):
        _build(max_tree_depth=20)


def test_df2t_rejects_missing_num_speculative_tokens():
    cfg = _build()
    cfg.num_speculative_tokens = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        cfg._validate_df2t()


def test_df2t_rejects_draft_without_hf_config():
    """A dummy draft with the right architecture but no ``hf_config`` must raise
    a clean error, not an AttributeError."""
    draft = _draft_cfg()
    draft.hf_config = None  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="HF config"):
        _build(draft_model_config=draft)


def test_df2t_get_num_draft_tokens():
    cfg = _build()
    cfg.tree_budget = 32
    assert cfg.get_num_draft_tokens() == 32
    cfg.tree_s = 0.5
    assert cfg.get_num_draft_tokens() == 16
    cfg.tree_budget = None
    assert cfg.get_num_draft_tokens() == 15


def test_df2t_nodes_ceiling_must_not_exceed_max_tree_budget():
    with pytest.raises(ValueError, match="nodes_ceiling"):
        _build(nodes_ceiling=65, max_tree_budget=64)


def test_df2t_tree_knobs_change_hash():
    base = _build()
    modified = _build(tree_budget=32)
    assert base.compute_hash() != modified.compute_hash()


def test_df2t_hash_covers_aux_hidden_state_layers():
    """df2t drafting consumes aux hidden states (DFlash2 flow, like dflash/
    dspark); the compute-graph hash must reflect the draft's
    ``eagle_aux_hidden_state_layer_ids``."""
    draft = _draft_cfg()
    draft.hf_config.eagle_aux_hidden_state_layer_ids = [1, 2]  # type: ignore[attr-defined]
    modified = _build(draft_model_config=draft)
    assert _build().compute_hash() != modified.compute_hash()
