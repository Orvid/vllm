# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The DFlash draft step splits into _draft_forward (distributions, as a
DraftProposal) and _draft_sample (the chain tail), so tree verification can
skip the sampler without the drafter knowing trees exist."""

from unittest import mock

import torch

from tests.v1.worker.gpu.spec_decode.tree.draft_util import fake_dflash


def test_draft_forward_returns_the_position_distributions():
    drafter = fake_dflash(num_reqs=3, steps=4, vocab=16, hidden=8)
    proposal = drafter._draft_forward(3, 12, None, None, None, mock.ANY)
    assert proposal.marginal_logits().shape == (3, 4, 16)
    assert proposal.position_hidden().shape == (3, 4, 8)
    assert proposal.marginal_logits().is_cuda


def test_generate_draft_skips_nothing_when_plain_and_orders_forward_then_sample():
    drafter = fake_dflash(num_reqs=2, steps=2, vocab=16, hidden=8)
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


def test_draft_sample_writes_the_chain_from_draft_logits():
    drafter = fake_dflash(num_reqs=2, steps=2, vocab=16, hidden=8)
    proposal = drafter._draft_forward(2, 8, None, None, None, None)
    with mock.patch.object(
        drafter,
        "sample_draft",
        return_value=torch.arange(4, device="cuda").view(2, 2),
    ) as sd:
        drafter._draft_sample(proposal)
    sd.assert_called_once()
    assert drafter.draft_tokens[:2].tolist() == [[0, 1], [2, 3]]
