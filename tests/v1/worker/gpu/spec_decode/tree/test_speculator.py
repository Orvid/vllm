#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for tree drafting composed onto a drafter in ``init_speculator``."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.config.speculative import SpeculativeConfig
from vllm.model_executor.models.registry import ModelRegistry
from vllm.v1.worker.gpu.spec_decode import init_speculator

ARCH = "DFlash2DraftModel"


def _req_states() -> Any:
    return SimpleNamespace()


def _fake_draft_model_config() -> SimpleNamespace:
    """A lightweight draft ``ModelConfig`` stand-in exposing the surface the
    DFlash2 speculator ``__init__`` reads (block_size, selector_top_k,
    mask_token_id)."""
    return SimpleNamespace(
        architectures=[ARCH],
        use_mla=False,
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


def _fake_vllm_config(tree_budget: int | None = None, tree_s: float = 1.0) -> Any:
    """Minimal config that lets the tree speculator ``__init__`` run on CPU."""
    from vllm.config import ParallelConfig

    speculative_config = SpeculativeConfig(
        method="dflash",
        tree_selection="df2t",
        num_speculative_tokens=15,
        draft_model_config=_fake_draft_model_config(),
        tree_budget=tree_budget,
        tree_s=tree_s,
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
    compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.NONE,
    )
    compilation_config.decode_mode = lambda: compilation_config.cudagraph_mode
    return SimpleNamespace(
        speculative_config=speculative_config,
        scheduler_config=SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=512),
        model_config=model_config,
        parallel_config=ParallelConfig(),
        compilation_config=compilation_config,
        watermark_config=None,
    )


def test_init_speculator_composes_tree_onto_drafter() -> None:
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
        DFlash2Speculator,
    )
    from vllm.v1.worker.gpu.spec_decode.tree.speculator import TreeSpeculatorMixin

    spec = init_speculator(_fake_vllm_config(), torch.device("cpu"), _req_states())

    assert isinstance(spec, TreeSpeculatorMixin)
    assert isinstance(spec, DFlash2Speculator)
    assert hasattr(spec, "_draft_forward")
    assert not hasattr(spec, "_draft_proposal")
    assert spec.num_query_per_req == 1 + 15
    # The drafter forward shape stays num_spec (the block depth)...
    assert spec.num_speculative_steps == 15
    # ...but the draft-carry width is the tree budget, so a tree wider than the
    # drafter block depth is expressible.
    assert spec.nodes_max == 64
    assert spec.tree_decode_width == 65
    assert spec.draft_tokens.shape[1] == 64


def test_plain_dflash_drafts_trees_from_its_position_distributions() -> None:
    """A drafter without a candidate head composes too, with K from the
    config's top-k fallback."""
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
        DFlash2Speculator,
    )

    vllm_config = _fake_vllm_config()
    vllm_config.speculative_config.draft_model_config.architectures = [
        "DFlashDraftModel"
    ]
    spec = init_speculator(vllm_config, torch.device("cpu"), _req_states())

    assert isinstance(spec, DFlashSpeculator)
    assert not isinstance(spec, DFlash2Speculator)
    assert spec.top_k == vllm_config.speculative_config.tree_candidate_top_k


def test_independent_proposal_offers_top_k_and_no_edge_term() -> None:
    from vllm.v1.worker.gpu.spec_decode.tree.proposal import DraftProposal

    logits = torch.tensor([[[0.0, 3.0, 1.0, 2.0], [5.0, 4.0, 0.0, 1.0]]])
    proposal = DraftProposal(logits)
    ids, unary = proposal.candidates(2)

    assert ids.tolist() == [[[1, 3], [0, 1]]]
    assert unary.tolist() == [[[3.0, 2.0], [5.0, 4.0]]]
    assert proposal.conditional(ids) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_selector_scores_independent_positions_by_their_own_logits() -> None:
    """With no edge term, every edge into a candidate carries the
    candidate's own logit."""
    from vllm.v1.worker.gpu.spec_decode.tree.proposal import DraftProposal
    from vllm.v1.worker.gpu.spec_decode.tree.selection.df2t import TreeBuilder

    torch.manual_seed(0)
    R, S, K, V = 2, 4, 3, 50
    logits = torch.randn(R, S, V, device="cuda")
    a = TreeBuilder(R, S, K, 12, "cuda")
    b = TreeBuilder(R, S, K, 12, "cuda")
    a.select(DraftProposal(logits))
    values, ids = logits.topk(K, dim=-1)
    b.build(values.unsqueeze(-2).expand(R, S, K, K), ids)
    assert torch.equal(a.packed, b.packed)


class _MarkovModel:
    """A DSpark stand-in: target vocab 6, draft vocab 3 holding target tokens
    1, 3 and 4, and a Markov bias of ``prev * w[draft id]``."""

    d2t = torch.tensor([1, 2, 2])  # draft id + offset = target id
    t2d = torch.tensor([-1, 0, -1, 1, 2, -1])
    w = torch.tensor([0.0, 1.0, -1.0])

    def map_draft_to_target(self, ids):
        return ids + self.d2t[ids]

    def markov_embed(self, token_ids):
        return token_ids.float()

    def markov_bias(self, embed):
        return embed[:, None] * self.w

    def markov_transition_bias(self, prev, index):
        return prev.float().unsqueeze(-1) * self.w[index].unsqueeze(-2)


def test_dspark_candidates_follow_the_greedy_chain() -> None:
    """Candidates are ranked under the Markov head given the greedy previous
    token, so the chain sampler's draft is always the first candidate."""
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkProposal

    m = _MarkovModel()
    base = torch.tensor([[[0.5, 2.0, 1.0], [3.0, 0.5, 1.5]]])  # [R=1, S=2, 3]
    proposal = DSparkProposal(m, base, torch.tensor([5]), m.t2d)
    ids, values = proposal.candidates(2)
    # Position 0 after the anchor 5: [0.5, 7.0, -4.0] -> draft 1, 0.
    # Position 1 after token 3: [3.0, 3.5, -1.5] -> draft 1, 0.
    assert ids.tolist() == [[[3, 1], [3, 1]]]
    assert values.tolist() == [[[7.0, 0.5], [3.5, 3.0]]]


def test_dspark_proposal_scores_edges_with_the_markov_head() -> None:
    """Edges carry the backbone logit plus the Markov bias from the
    predecessor token (the anchor at the first position), in draft vocab."""
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkProposal

    m = _MarkovModel()
    base = torch.tensor([[[0.5, 2.0, 1.0], [3.0, 0.5, 1.5]]])
    proposal = DSparkProposal(m, base, torch.tensor([5]), m.t2d)
    scores = proposal.conditional(torch.tensor([[[3, 1], [3, 0]]]))
    # Position 0 follows the anchor (5).
    assert scores[0, 0].tolist() == [[7.0, 0.5], [7.0, 0.5]]
    # Position 1 follows 3 or 1; target 0 is outside the draft vocab.
    assert scores[0, 1, 0, 0].item() == 0.5 + 3.0
    assert scores[0, 1, 1, 0].item() == 0.5 + 1.0
    assert scores[0, 1, :, 1].tolist() == [float("-inf"), float("-inf")]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_generate_draft_builds_tree_and_draft_tokens() -> None:
    """DF2T ``_generate_draft`` must build the per-request tree over the same
    candidate transition scores and fill ``draft_tokens`` from the tree node
    token ids (emit order, padded to the per-step width).

    The fake ``_score_candidates`` returns the PRODUCTION [R, L, K, K]
    predecessor->successor score shape (``_score_edges`` yields ``blpr,blcr->
    blpc``), so the real TreeBuilder consumption path is exercised; each
    request's tree must then match the torch ``build_tree_naive`` oracle
    bit-for-bit on the same scores/candidates.
    """
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.spec_decode.tree.speculator import TreeSpeculatorMixin
    from vllm.v1.worker.gpu.spec_decode.tree.tree_metadata import build_tree_naive

    torch.manual_seed(0)
    spec = init_speculator(_fake_vllm_config(), torch.device("cuda"), _req_states())
    assert isinstance(spec, TreeSpeculatorMixin)
    num_reqs, steps = 3, spec.num_speculative_steps
    K = spec.top_k
    hidden = 64
    captured: dict[str, torch.Tensor] = {}

    def fake_run_model(*_a, **_k):
        return torch.randn(num_reqs * steps, hidden, device="cuda")

    def fake_compute_candidates(h):
        cand = torch.randint(0, 512, (h.shape[0], K), device="cuda")
        unary = torch.randn(h.shape[0], K, device="cuda")
        captured["cand_flat"] = cand
        return cand, unary

    def fake_score_candidates(cand, unary, hidden):
        # Production shape: [num_reqs, L, K, K], not the old [R, K, K] fake.
        scores = torch.randn(cand.shape[0], cand.shape[1], K, K, device="cuda")
        captured["scores"] = scores
        return scores

    spec._run_model = fake_run_model
    spec.model = SimpleNamespace(compute_candidates=fake_compute_candidates)
    spec._score_candidates = fake_score_candidates
    spec.sample_indices[: num_reqs * steps] = torch.arange(
        num_reqs * steps, device="cuda"
    )

    spec._generate_draft(num_reqs, num_reqs * steps, {}, {}, None)

    # latest_tree_drafts exposes one TreeDraft per request with real nodes.
    drafts = spec.latest_tree_drafts
    assert drafts is not None and len(drafts) == num_reqs
    assert all(d is not None for d in drafts)

    # The serve-time tree must be the same top-B prefix tree the torch oracle
    # would produce from the same candidate transition scores: node count,
    # token ids, depths and parents all agree bit-for-bit.
    cand_full = captured["cand_flat"].view(num_reqs, steps, K)
    scores = captured["scores"]
    assert scores.shape == (num_reqs, steps, K, K)
    for r in range(num_reqs):
        ref = build_tree_naive(
            scores[r].cpu(), cand_full[r].cpu(), spec.nodes_max, spec.max_depth
        )
        n = int(drafts[r].num_nodes.item())
        assert n == len(ref.token_ids)
        assert torch.equal(drafts[r].token_ids[:n].cpu(), ref.token_ids)
        assert torch.equal(drafts[r].depths[:n].cpu(), ref.depths)
        assert torch.equal(drafts[r].parents[:n].cpu(), ref.parents)

    # draft_tokens width == the scheduler's per-request carry width (the tree
    # budget nodes_max, NOT the drafter block depth); each request's draft row
    # is its tree's highest-priority node token ids (emit order). Padding only
    # kicks in when the tree emits fewer nodes than the width (never here).
    width = spec.nodes_max
    assert spec.draft_tokens.shape[1] == width
    for r in range(num_reqs):
        n = int(drafts[r].num_nodes.item())
        assert n >= width, "the tree must fill the per-step draft width"
        assert torch.equal(spec.draft_tokens[r], drafts[r].token_ids[:width])


def test_chain_config_builds_the_plain_drafter() -> None:
    """Without tree selection the same drafter verifies its chain: no tree
    machinery is composed in and the draft carry is the drafter depth."""
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
        DFlash2Speculator,
    )
    from vllm.v1.worker.gpu.spec_decode.tree.speculator import TreeSpeculatorMixin

    cfg = _fake_vllm_config()
    cfg.speculative_config.tree_selection = None
    spec = init_speculator(cfg, torch.device("cpu"), _req_states())
    assert type(spec) is DFlash2Speculator
    assert not isinstance(spec, TreeSpeculatorMixin)
    assert spec.draft_tokens.shape[1] == 15


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_generate_draft_budget_below_num_spec_degrades_not_crashes() -> None:
    """WS-A finding [7]: tree_budget x tree_s below num_speculative_tokens must
    hit the v5 §3.4 degradation order (below L, shorten to a chain of
    ``nodes_max``), not slice ``tree.token_ids`` past its ``nodes_max``-wide
    buffer. With tree_budget=16, tree_s=0.5 -> nodes_max=8 < num_spec=15, the
    build writes an 8-wide chain and the draft carry pads it to the tree-budget
    width."""
    from vllm.v1.worker.gpu.spec_decode.tree.speculator import TreeSpeculatorMixin

    torch.manual_seed(0)
    spec = init_speculator(
        _fake_vllm_config(tree_budget=16, tree_s=0.5),
        torch.device("cuda"),
        _req_states(),
    )
    assert isinstance(spec, TreeSpeculatorMixin)
    assert spec.nodes_max == 8
    assert spec.num_speculative_steps == 15
    assert spec.draft_tokens.shape[1] == spec.nodes_max
    num_reqs, steps = 3, spec.num_speculative_steps
    K = spec.top_k

    spec._run_model = lambda *_a, **_k: torch.randn(num_reqs * steps, 64, device="cuda")
    spec.model = SimpleNamespace(
        compute_candidates=lambda h: (
            torch.randint(0, 512, (h.shape[0], K), device="cuda"),
            torch.randn(h.shape[0], K, device="cuda"),
        )
    )
    spec._score_candidates = lambda cand, unary, hidden: torch.randn(
        cand.shape[0], cand.shape[1], K, K, device="cuda"
    )
    spec.sample_indices[: num_reqs * steps] = torch.arange(
        num_reqs * steps, device="cuda"
    )

    spec._generate_draft(num_reqs, num_reqs * steps, {}, {}, None)

    drafts = spec.latest_tree_drafts
    assert drafts is not None and len(drafts) == num_reqs
    # The degraded shape (L, B) = (nodes_max, nodes_max) caps the tree depth at
    # the budget; each request emits exactly nodes_max nodes with depth <= B.
    for r in range(num_reqs):
        assert drafts[r].num_nodes is not None
        n = int(drafts[r].num_nodes.item())
        assert n == spec.nodes_max
        depths = drafts[r].depths[:n].cpu().tolist()
        assert max(depths) <= spec.nodes_max, "degraded tree is depth-capped at B"
        # The carried draft row is the tree token ids, no out-of-bounds slice.
        assert torch.equal(spec.draft_tokens[r][:n], drafts[r].token_ids[:n])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_get_latest_tree_drafts_fails_closed_on_over_read() -> None:
    """Requesting more trees than the last build wrote must fail closed.

    ``get_latest_tree_drafts(num_reqs)`` reads the persistent TreeBuilder
    buffers; rows past the last build's width hold stale trees from an older
    batch, so an over-read must raise instead of handing out garbage. Covers
    both the eager build (``_generate_draft``) and a replayed drafter graph
    (``_mark_draft_graph_built``) width sources.
    """
    from vllm.v1.worker.gpu.spec_decode.tree.speculator import TreeSpeculatorMixin

    torch.manual_seed(0)
    spec = init_speculator(_fake_vllm_config(), torch.device("cuda"), _req_states())
    assert isinstance(spec, TreeSpeculatorMixin)
    num_reqs, steps = 2, spec.num_speculative_steps
    K = spec.top_k

    spec._run_model = lambda *_a, **_k: torch.randn(num_reqs * steps, 64, device="cuda")
    spec.model = SimpleNamespace(
        compute_candidates=lambda h: (
            torch.randint(0, 512, (h.shape[0], K), device="cuda"),
            torch.randn(h.shape[0], K, device="cuda"),
        )
    )
    spec._score_candidates = lambda cand, unary, hidden: torch.randn(
        cand.shape[0], cand.shape[1], K, K, device="cuda"
    )
    spec.sample_indices[: num_reqs * steps] = torch.arange(
        num_reqs * steps, device="cuda"
    )

    # No build yet: nothing to read, None (not an over-read).
    assert spec.latest_tree_drafts is None

    spec._generate_draft(num_reqs, num_reqs * steps, {}, {}, None)
    assert spec._last_built_rows == num_reqs
    assert spec.get_latest_tree_drafts(num_reqs) is not None
    with pytest.raises(RuntimeError, match="over-read"):
        spec.get_latest_tree_drafts(num_reqs + 1)

    # A replayed drafter graph reports its own width.
    spec._mark_draft_graph_built(1)
    assert spec._last_built_rows == 1
    with pytest.raises(RuntimeError, match="over-read"):
        spec.get_latest_tree_drafts(num_reqs)


def _tree_speculator_with_seams() -> Any:
    """A composed tree speculator with the drafter forward fixed to a tiny
    deterministic proposal and the sample tail spied."""
    from vllm.v1.worker.gpu.spec_decode.tree.speculator import TreeSpeculatorMixin

    torch.manual_seed(0)
    spec = init_speculator(_fake_vllm_config(), torch.device("cuda"), _req_states())
    assert isinstance(spec, TreeSpeculatorMixin)
    num_reqs, steps = 2, spec.num_speculative_steps
    K = spec.top_k
    spec._run_model = lambda *_a, **_k: torch.randn(num_reqs * steps, 64, device="cuda")
    spec.model = SimpleNamespace(
        compute_candidates=lambda h: (
            torch.randint(0, 512, (h.shape[0], K), device="cuda"),
            torch.randn(h.shape[0], K, device="cuda"),
        )
    )
    spec._score_candidates = lambda cand, unary, hidden: torch.randn(
        cand.shape[0], cand.shape[1], K, K, device="cuda"
    )
    spec.sample_indices[: num_reqs * steps] = torch.arange(
        num_reqs * steps, device="cuda"
    )
    return spec


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tree_speculator_generate_draft_never_runs_the_chain_sampler() -> None:
    """The mixin drives selection from the drafter's forward and writes the
    tree node tokens; the chain-sampling tail must never run."""
    from unittest import mock

    from vllm.v1.worker.gpu.spec_decode.tree.speculator import TreeSpeculatorMixin

    spec = _tree_speculator_with_seams()
    assert isinstance(spec, TreeSpeculatorMixin)
    with (
        mock.patch.object(spec, "_draft_sample", wraps=spec._draft_sample) as smp,
        mock.patch.object(
            spec.tree_selector, "select", wraps=spec.tree_selector.select
        ) as select,
    ):
        spec._generate_draft(2, 30, {}, {}, None)
    smp.assert_not_called()
    assert select.call_count == 1
    assert spec.draft_tokens[:2].shape == (2, spec.draft_tokens.shape[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tree_speculator_dummy_path_runs_the_plain_drafter() -> None:
    """A dummy/profile run (no attention metadata) must take the plain
    drafter's chain path, bypassing the tree entirely."""
    spec = _tree_speculator_with_seams()
    spec._generate_draft(2, 30, None, None, None, CUDAGraphMode.NONE)
    # The chain tail ran (its draft-token write is the plain path's), and the
    # tree read should see no build.
    assert spec._last_built_rows == 0
    assert spec.latest_tree_drafts is None
