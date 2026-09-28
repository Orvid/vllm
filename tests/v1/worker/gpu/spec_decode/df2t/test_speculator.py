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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_generate_draft_builds_tree_and_draft_tokens() -> None:
    """DF2T ``_generate_draft`` must build the per-request tree over the same
    candidate transition scores and fill ``draft_tokens`` from the tree node
    token ids (emit order, padded to the per-step width)."""
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.spec_decode.df2t.speculator import DF2TSpeculator

    torch.manual_seed(0)
    spec = init_speculator(_fake_vllm_config(), torch.device("cuda"))
    assert isinstance(spec, DF2TSpeculator)
    num_reqs, steps = 3, spec.num_speculative_steps
    K = spec.top_k
    hidden = 64

    def fake_run_model(*_a, **_k):
        return torch.randn(num_reqs * steps, hidden, device="cuda")

    def fake_compute_candidates(h):
        cand = torch.randint(0, 512, (h.shape[0], K), device="cuda")
        unary = torch.randn(h.shape[0], K, device="cuda")
        return cand, unary

    def fake_score_candidates(cand, unary, hidden):
        R = hidden.shape[0]
        return torch.randn(R, K, K, device="cuda")

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
    for d in drafts:
        assert d.token_ids.shape[0] == d.num_nodes.item() > 0
        assert (d.arena == torch.arange(d.token_ids.shape[0], device="cuda")).all()

    # draft_tokens width == the scheduler's per-step width; each request's draft
    # row is its tree's highest-priority node token ids (emit order). Padding
    # only kicks in when the tree emits fewer nodes than the width (never here).
    width = spec.num_speculative_steps
    for r in range(num_reqs):
        n = int(drafts[r].num_nodes.item())
        assert n >= width, "the tree must fill the per-step draft width"
        assert torch.equal(spec.draft_tokens[r], drafts[r].token_ids[:width])
