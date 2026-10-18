# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a real drafter object without loading a checkpoint: patch the
model forward with a tiny deterministic fake."""

import torch


def fake_model(vocab: int, hidden: int):
    class FakeModel(torch.nn.Module):
        def compute_logits(self, hidden_states, *, out=None):
            # Deterministic, hidable: logits row = sum of the hidden row.
            logits = hidden_states.sum(dim=-1).unsqueeze(-1).expand(-1, vocab)
            if out is not None:
                out[: logits.numel()].view_as(logits).copy_(logits)
                return out[: logits.numel()].view_as(logits)
            return logits

        # dflash/dspark surface used under test
        def compute_draft_logits(self, hidden_states):
            return self.compute_logits(hidden_states)

    return FakeModel()


def fake_run_model(num_tokens: int, hidden: int) -> torch.Tensor:
    """The drafter forward's output: one hidden row per token."""
    return (
        torch.arange(num_tokens * hidden, device="cuda")
        .view(num_tokens, hidden)
        .float()
    )


def fake_dflash(num_reqs: int, steps: int, vocab: int, hidden: int):
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

    drafter = object.__new__(DFlashSpeculator)
    drafter.num_speculative_steps = steps
    drafter.max_num_reqs = num_reqs
    drafter.draft_tokens = torch.zeros(num_reqs, steps, dtype=torch.long, device="cuda")
    drafter.draft_logits = torch.zeros(
        num_reqs, steps, vocab, dtype=torch.float32, device="cuda"
    )
    drafter.sample_indices = torch.arange(num_reqs * steps, device="cuda")
    drafter.sample_pos = torch.zeros(num_reqs * steps, dtype=torch.long, device="cuda")
    drafter.sample_idx_mapping = torch.arange(
        num_reqs, device="cuda"
    ).repeat_interleave(steps)
    drafter.sample_col = torch.arange(steps, dtype=torch.int32, device="cuda").repeat(
        num_reqs
    )
    drafter.temperature = torch.ones(num_reqs, dtype=torch.float32, device="cuda")
    drafter.seeds = torch.zeros(num_reqs, dtype=torch.int64, device="cuda")
    drafter.draft_max_seq_len = steps + 1
    drafter.use_local_argmax_reduction = False
    drafter.draft_watermarker = None
    drafter.use_fp64_gumbel = False
    drafter.acceptance_estimator = None
    drafter.model = fake_model(vocab, hidden).to("cuda")
    drafter._run_model = lambda num_tokens, *args, **kwargs: fake_run_model(
        num_tokens, hidden
    )
    # Bind the drafter's real chain tail so the seam tests exercise it; the
    # fake surface above (sample positions, temperatures, watermarker) is what
    # it needs to run without real attention metadata.
    drafter.sample_draft = DFlashSpeculator.sample_draft.__get__(drafter)
    return drafter
