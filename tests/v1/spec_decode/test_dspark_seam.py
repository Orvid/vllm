# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DSpark draft step splits into _draft_forward (the parallel backbone,
as a DSparkProposal) and _draft_sample (the sequential Markov chain tail), so
tree verification can drive selection without the chain sampler."""

from types import SimpleNamespace
from unittest import mock

import torch

from tests.v1.worker.gpu.spec_decode.tree.draft_util import fake_dflash


def fake_dspark(num_reqs: int, steps: int, vocab: int, hidden: int):
    from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

    drafter = fake_dflash(num_reqs=num_reqs, steps=steps, vocab=vocab, hidden=hidden)
    drafter.__class__ = DSparkSpeculator
    drafter._anchor_idx = (
        torch.arange(num_reqs, dtype=torch.int64, device="cuda") * steps
    )
    drafter._step_cols = torch.arange(steps, dtype=torch.int32, device="cuda")
    drafter._target_to_draft = None
    drafter._draft_topk = None
    drafter._d2t_scatter_index = None
    drafter._draft_scatter_buf = None
    drafter.use_confidence_head = False
    drafter.sample_from_anchor = True
    drafter.num_query_per_req = steps

    class FakeModel(torch.nn.Module):
        def compute_draft_logits(self, hidden_states):
            return hidden_states.sum(dim=-1).unsqueeze(-1).expand(-1, vocab).clone()

        def markov_embed(self, prev):
            return prev

        def markov_bias(self, embed):
            return torch.zeros(embed.shape[0], vocab, device="cuda")

        def map_draft_to_target(self, draft_ids):
            return draft_ids

        def markov_transition_bias(self, prev, draft_ids):
            return torch.zeros(
                draft_ids.shape[0],
                draft_ids.shape[1],
                draft_ids.shape[2],
                vocab,
                device=draft_ids.device,
            )

    drafter.model = FakeModel().to("cuda")
    drafter.input_buffers = SimpleNamespace(
        input_ids=torch.zeros(num_reqs * steps, dtype=torch.long, device="cuda")
    )
    return drafter


def test_draft_forward_returns_a_dspark_proposal():
    drafter = fake_dspark(num_reqs=3, steps=4, vocab=16, hidden=8)
    proposal = drafter._draft_forward(3, 12, None, None, None, mock.ANY)
    assert proposal.marginal_logits().shape == (3, 4, 16)
    assert proposal.position_hidden().shape == (3, 4, 8)
    assert proposal.marginal_logits().is_cuda


def test_generate_draft_orders_forward_then_sample():
    drafter = fake_dspark(num_reqs=2, steps=2, vocab=16, hidden=8)
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


def test_draft_sample_replays_the_chain_from_forward():
    drafter = fake_dspark(num_reqs=2, steps=3, vocab=16, hidden=8)
    drafter._generate_draft(2, 12, None, None, None, None)
    expected = drafter.draft_tokens[:2].clone()

    drafter.draft_tokens.zero_()
    proposal = drafter._draft_forward(2, 12, None, None, None, None)
    drafter._draft_sample(proposal)
    assert torch.equal(drafter.draft_tokens[:2], expected)
