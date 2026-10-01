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


def test_df2t_tree_lookahead_is_budget_ceiling():
    """The df2t scheduler reservation / KV-arena ceiling is ``1 +
    max_tree_budget`` (the tree lookahead), never derived from
    ``num_speculative_tokens``."""
    cfg = _build()
    cfg._validate_df2t()
    assert cfg.max_tree_budget == 64
    assert cfg.tree_lookahead == 1 + cfg.max_tree_budget
    assert cfg.tree_decode_width == 1 + cfg.max_tree_budget
    assert cfg.tree_lookahead == cfg.tree_decode_width
    assert cfg.tree_lookahead != 1 + cfg.num_speculative_tokens


def test_df2t_max_num_new_slots_for_drafting_is_tree_budget():
    """df2t reserves the tree budget ``max_tree_budget`` draft slots per
    request, so a tree wider than the drafter's block depth (B > L) is
    expressible. The dflash path keeps reserving ``num_speculative_tokens``."""
    cfg = _build()
    cfg._validate_df2t()
    assert cfg.max_num_new_slots_for_drafting == cfg.max_tree_budget
    assert cfg.max_num_new_slots_for_drafting != cfg.num_speculative_tokens
    cfg.method = "dflash"
    assert cfg.max_num_new_slots_for_drafting == cfg.num_speculative_tokens


def test_df2t_num_draft_rows_per_req_is_tree_budget():
    """The scheduler's per-request draft-row count decouples from
    ``num_speculative_tokens`` for df2t; other speculative methods keep their
    own draft positions."""
    cfg = _build()
    cfg._validate_df2t()
    assert cfg.num_draft_rows_per_req == cfg.max_tree_budget
    cfg.method = "dflash"
    assert cfg.num_draft_rows_per_req == cfg.num_speculative_tokens


def test_df2t_accepts_tree_depth_sixteen():
    """Stage-0 target L=16 must configure: the depth cap is validated only
    against the drafter block depth, so a drafter with block_size >= 17 admits
    max_tree_depth == 16 (the builder widens its packed path accordingly)."""
    cfg = _build(num_speculative_tokens=16, max_tree_depth=16)
    draft = _draft_cfg()
    draft.hf_config.dflash_config.block_size = 17  # type: ignore[attr-defined]
    cfg.draft_model_config = draft  # type: ignore[assignment]
    cfg._validate_df2t()
    assert cfg.max_tree_depth == 16
    assert cfg.max_tree_depth <= cfg.num_speculative_tokens


def test_df2t_hash_covers_aux_hidden_state_layers():
    """df2t drafting consumes aux hidden states (DFlash2 flow, like dflash/
    dspark); the compute-graph hash must reflect the draft's
    ``eagle_aux_hidden_state_layer_ids``."""
    draft = _draft_cfg()
    draft.hf_config.eagle_aux_hidden_state_layer_ids = [1, 2]  # type: ignore[attr-defined]
    modified = _build(draft_model_config=draft)
    assert _build().compute_hash() != modified.compute_hash()
