# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the DF2T speculator dispatch in ``init_speculator``."""

from types import SimpleNamespace
from typing import Any

import pytest
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


def _fake_vllm_config(tree_budget: int | None = None, tree_s: float = 1.0) -> Any:
    """Minimal config that lets the DF2T speculator ``__init__`` run on CPU."""
    speculative_config = SpeculativeConfig(
        method="df2t",
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
    # The drafter forward shape stays num_spec (the block depth)...
    assert spec.drafter_block_depth == 15
    # ...but the draft-carry width is the tree budget, so a tree wider than the
    # drafter block depth is expressible.
    assert spec.nodes_max == 64
    assert spec.tree_decode_width == 65
    assert spec.draft_tokens.shape[1] == 64


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

    from vllm.v1.worker.gpu.spec_decode.df2t.speculator import DF2TSpeculator
    from vllm.v1.worker.gpu.spec_decode.df2t.tree_metadata import build_tree_naive

    torch.manual_seed(0)
    spec = init_speculator(_fake_vllm_config(), torch.device("cuda"))
    assert isinstance(spec, DF2TSpeculator)
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

    spec._generate_draft(num_reqs, num_reqs * steps, None, None, None)

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_generate_draft_budget_below_num_spec_degrades_not_crashes() -> None:
    """WS-A finding [7]: tree_budget x tree_s below num_speculative_tokens must
    hit the v5 §3.4 degradation order (below L, shorten to a chain of
    ``nodes_max``), not slice ``tree.token_ids`` past its ``nodes_max``-wide
    buffer. With tree_budget=16, tree_s=0.5 -> nodes_max=8 < num_spec=15, the
    build writes an 8-wide chain and the draft carry pads it to the tree-budget
    width."""
    from vllm.v1.worker.gpu.spec_decode.df2t.speculator import DF2TSpeculator

    torch.manual_seed(0)
    spec = init_speculator(
        _fake_vllm_config(tree_budget=16, tree_s=0.5), torch.device("cuda")
    )
    assert isinstance(spec, DF2TSpeculator)
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

    spec._generate_draft(num_reqs, num_reqs * steps, None, None, None)

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
    from vllm.v1.worker.gpu.spec_decode.df2t.speculator import DF2TSpeculator

    torch.manual_seed(0)
    spec = init_speculator(_fake_vllm_config(), torch.device("cuda"))
    assert isinstance(spec, DF2TSpeculator)
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

    spec._generate_draft(num_reqs, num_reqs * steps, None, None, None)
    assert spec._last_built_rows == num_reqs
    assert spec.get_latest_tree_drafts(num_reqs) is not None
    with pytest.raises(RuntimeError, match="over-read"):
        spec.get_latest_tree_drafts(num_reqs + 1)

    # A replayed drafter graph reports its own width.
    spec._mark_draft_graph_built(1)
    assert spec._last_built_rows == 1
    with pytest.raises(RuntimeError, match="over-read"):
        spec.get_latest_tree_drafts(num_reqs)
