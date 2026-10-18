# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DFlash2 draft step splits into _draft_forward (the candidate head, as
a CandidateHeadProposal) and _draft_sample (the candidate walk), so tree
verification can drive selection without the chain walker."""

from types import SimpleNamespace
from unittest import mock

import torch

from tests.v1.worker.gpu.spec_decode.tree.draft_util import fake_dflash


def fake_dflash2(num_reqs: int, steps: int, vocab: int, hidden: int, top_k: int = 3):
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
        CandidateSampler,
        DFlash2Speculator,
    )

    drafter = fake_dflash(num_reqs=num_reqs, steps=steps, vocab=vocab, hidden=hidden)
    drafter.__class__ = DFlash2Speculator
    drafter.top_k = top_k
    drafter.candidate_sampler = CandidateSampler(num_reqs, steps, top_k, "cuda")
    drafter.num_query_per_req = steps + 1
    drafter.enable_adaptive_verification = False
    drafter.acceptance_estimator = None
    drafter.use_acceptance_estimator = False

    class FakeDFlash2Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(candidate_selector=self._candidate_selector)

        def compute_candidates(self, hidden_states):
            num_sample = hidden_states.shape[0]
            ids = torch.arange(top_k, device="cuda").expand(num_sample, top_k)
            unary = torch.ones_like(ids, dtype=torch.float32)
            return ids, unary

        def _candidate_selector(
            self, candidate_ids, unary_logits, hidden_states, anchor_token_ids
        ):
            num_reqs, num_steps, k = candidate_ids.shape
            return (
                torch.arange(k, device="cuda")
                .float()
                .expand(num_reqs, num_steps, k, k)
                .clone()
            )

    drafter.model = FakeDFlash2Model().to("cuda")
    drafter.input_buffers = SimpleNamespace(
        input_ids=torch.arange(num_reqs * (steps + 1), dtype=torch.long, device="cuda")
    )
    return drafter


def test_draft_forward_returns_a_candidate_head_proposal():
    drafter = fake_dflash2(num_reqs=3, steps=4, vocab=16, hidden=8, top_k=3)
    proposal = drafter._draft_forward(3, 12, None, None, None, mock.ANY)
    ids, scores = proposal._ids, proposal._scores
    assert ids.shape == (3, 4, 3)
    assert scores.shape == (3, 4, 3, 3)
    assert proposal.num_positions == 4
    assert ids.is_cuda


def test_generate_draft_orders_forward_then_sample():
    drafter = fake_dflash2(num_reqs=2, steps=2, vocab=16, hidden=8, top_k=3)
    with (
        mock.patch.object(
            drafter, "_draft_forward", wraps=drafter._draft_forward
        ) as fwd,
        mock.patch.object(drafter, "_draft_sample", wraps=drafter._draft_sample) as smp,
    ):
        drafter._generate_draft(2, 8, None, None, None, None)
    fwd.assert_called_once()
    smp.assert_called_once()
    assert drafter.draft_tokens[:2].shape == (2, 2)


def test_draft_sample_runs_the_candidate_walk_and_optional_verification_tail():
    drafter = fake_dflash2(num_reqs=2, steps=2, vocab=16, hidden=8, top_k=3)
    drafter.enable_adaptive_verification = True
    with mock.patch.object(
        drafter.candidate_sampler, "sample", wraps=drafter.candidate_sampler.sample
    ) as walk:
        proposal = drafter._draft_forward(2, 8, None, None, None, None)
        drafter._draft_sample(proposal)
    walk.assert_called_once()
    assert walk.call_args.args[0] is proposal._ids
    assert walk.call_args.args[1] is proposal._scores
    assert drafter.draft_tokens[:2].shape == (2, 2)
