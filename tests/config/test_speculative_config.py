# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the tree speculative verification config surface."""

from types import SimpleNamespace
from typing import Any

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
        method="dflash",
        tree_selection="df2t",
        num_speculative_tokens=15,
        draft_model_config=_draft_cfg(arch),
        tree_budget=None,
    )
    kwargs.update(over)
    return SpeculativeConfig(**kwargs)


def test_df2t_method_is_a_deprecated_alias_for_dflash_tree_selection():
    """df2t is a tree selection method, not a drafter: ``method="df2t"``
    resolves to the dflash drafter with ``tree_selection="df2t"``."""
    cfg = _build(method="df2t", tree_selection=None)
    assert cfg.method == "dflash"
    assert cfg.tree_selection == "df2t"
    assert cfg.use_tree()


def test_chain_verification_keeps_the_drafter_reservations():
    """Without a tree selection method the same drafter verifies its chain:
    every tree-only reservation falls back to the drafter's own values."""
    chain = _build()
    chain.tree_selection = None
    K = chain.num_speculative_tokens
    assert not chain.use_tree()
    assert chain.max_num_new_slots_for_drafting == K
    assert chain.num_draft_rows_per_req == K
    assert chain.get_num_draft_tokens() == K
    assert chain.tree_decode_width == 0
    tree = _build()
    assert tree.max_num_new_slots_for_drafting == tree.max_tree_budget != K


def test_tree_resolution():
    cfg = _build()
    assert cfg.tree_budget == cfg.max_tree_budget == 64
    assert cfg.max_tree_depth == 15
    assert cfg.parallel_drafting


def test_tree_accepts_drafter_without_candidate_head():
    """A plain DFlash drafter drives trees from each position's top-k."""
    cfg = _build(arch="DFlashDraftModel")
    assert cfg.use_tree()
    assert cfg.tree_candidate_top_k == 8


def test_expert_aware_selection_needs_adaptive_width():
    """df2t_moe prices expert cost in tokens at the adaptive controller's
    throughput estimate, so it fails closed without it."""
    with pytest.raises(ValueError, match="adaptive_tree_width"):
        _build(tree_selection="df2t_moe")
    assert _build(tree_selection="df2t_moe", adaptive_tree_width=True).use_tree()


def test_tree_rejects_sequential_drafters():
    with pytest.raises(ValueError, match="in parallel"):
        _build(method="ngram", draft_model_config=None)


def test_tree_rejects_tree_budget_over_ceiling():
    with pytest.raises(ValueError, match="tree_budget"):
        _build(tree_budget=128)


def test_tree_rejects_max_tree_budget_over_64():
    """The ancestor word is a single int64 per node (one bit per arena), so a
    ``max_tree_budget`` above 64 is rejected at validation instead of
    overflowing the word at runtime."""
    with pytest.raises(ValueError, match="max_tree_budget"):
        _build(max_tree_budget=65)
    with pytest.raises(ValueError, match="<= 64"):
        _build(max_tree_budget=128)
    assert _build().max_tree_budget == 64


def test_tree_rejects_max_tree_depth_over_tokens():
    with pytest.raises(ValueError, match="max_tree_depth"):
        _build(max_tree_depth=20)


def test_tree_rejects_missing_num_speculative_tokens():
    cfg = _build()
    cfg.num_speculative_tokens = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        cfg._validate_tree()


def test_tree_rejects_draft_without_hf_config():
    """A draft with the right architecture but no ``hf_config`` raises a clean
    error, not an AttributeError."""
    draft = _draft_cfg()
    draft.hf_config = None  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="HF"):
        _build(draft_model_config=draft)


def test_tree_get_num_draft_tokens():
    cfg = _build()
    cfg.tree_budget = 32
    assert cfg.get_num_draft_tokens() == 32
    cfg.tree_s = 0.5
    assert cfg.get_num_draft_tokens() == 16
    cfg.tree_budget = None
    assert cfg.get_num_draft_tokens() == 15


def test_tree_nodes_ceiling_must_not_exceed_max_tree_budget():
    with pytest.raises(ValueError, match="nodes_ceiling"):
        _build(nodes_ceiling=65, max_tree_budget=64)


def test_tree_knobs_change_hash():
    base = _build()
    assert base.compute_hash() != _build(tree_budget=32).compute_hash()
    chain = _build()
    chain.tree_selection = None
    assert base.compute_hash() != chain.compute_hash()


def test_tree_lookahead_is_budget_ceiling():
    """The scheduler reservation / KV-arena ceiling is ``1 +
    max_tree_budget``, never derived from ``num_speculative_tokens``."""
    cfg = _build()
    assert cfg.tree_lookahead == 1 + cfg.max_tree_budget == 65
    assert cfg.tree_lookahead != 1 + cfg.num_speculative_tokens


def test_tree_decode_width_is_static_budget_not_ceiling():
    """The uniform decode-graph width is ``1 + nodes_max``; with a smaller
    tree_budget the decode width follows the budget while the reservation
    keeps the ceiling."""
    cfg = _build(tree_budget=16, max_tree_budget=64)
    assert cfg.tree_decode_width == 17
    assert cfg.tree_lookahead == 65
    assert cfg.num_draft_rows_per_req == 16
    assert cfg.max_num_new_slots_for_drafting == 64


def test_tree_accepts_tree_depth_sixteen():
    """The depth cap is validated only against the drafter block depth, so a
    drafter with block_size >= 17 admits max_tree_depth == 16."""
    cfg = _build(num_speculative_tokens=16, max_tree_depth=16)
    draft = _draft_cfg()
    draft.hf_config.dflash_config.block_size = 17  # type: ignore[attr-defined]
    cfg.draft_model_config = draft  # type: ignore[assignment]
    cfg._validate_tree()
    assert cfg.max_tree_depth == 16


def test_tree_hash_covers_aux_hidden_state_layers():
    """The drafter consumes aux hidden states; the compute-graph hash must
    reflect the draft's ``eagle_aux_hidden_state_layer_ids``."""
    draft = _draft_cfg()
    draft.hf_config.eagle_aux_hidden_state_layer_ids = [1, 2]  # type: ignore[attr-defined]
    assert _build().compute_hash() != _build(draft_model_config=draft).compute_hash()
