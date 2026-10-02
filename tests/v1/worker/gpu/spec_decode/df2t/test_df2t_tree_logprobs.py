# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""df2t real-logprobs layout and value drills (WS-B-2).

The df2t verify block emits, per request, one token per chain depth (the
accepted path + the terminal/resampled token). Before this task a ``logprobs``
request raised NotImplementedError inside the worker on the first verify step
(killing the engine); these tests pin the replacement: per-emitted-token
logprobs gathered from the fully processed distribution at the row each token
was actually drawn from.

A chain-depth-d token is drawn from verify row ``0`` at depth 0 and from row
``accepted_arenas[r, d] + 1`` for ``d >= 1`` (the last accepted node's target
row, whose distribution conditions on exactly the accepted-chain prefix of
length ``d``). The tests run the REAL ``TreeVerifier`` on a synthetic block,
then check that ``_df2t_tree_logprobs`` reproduces an independent reference:
the log-softmax of the processed rows at those draw rows, gathered at the
emitted token, with per-request CSR boundaries in batch order.
"""

import numpy as np
import pytest
import torch

from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.model_runner import _df2t_tree_logprobs
from vllm.v1.worker.gpu.spec_decode.df2t.tree_verifier import TreeVerifier

_DEVICES = ["cuda"]


def _chain_tree(
    R: int, B: int, device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A pure chain tree (node ``a`` is the child of node ``a - 1``)."""
    parents = torch.stack(
        [torch.arange(-1, B - 1, dtype=torch.long, device=device)] * R
    )
    token_ids = torch.randint(0, 5, (R, B), device=device)
    num_nodes = torch.full((R,), B, dtype=torch.long, device=device)
    return token_ids, parents, num_nodes


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_df2t_tree_logprobs_matches_reference(device):
    R, B, V = 3, 6, 32
    g = torch.Generator(device=device).manual_seed(3)
    target_logits = torch.randn(R, 1 + B, V, generator=g, device=device) * 0.9
    token_ids, parents, num_nodes = _chain_tree(R, B, device)
    verifier = TreeVerifier(R, B, V, device)
    sampled, num_sampled = verifier(
        target_logits, token_ids, parents, num_nodes, greedy=True
    )
    acc = verifier.accepted_arenas

    processed = target_logits.reshape(-1, V).clone()
    cu_np = np.cumsum([0] + [1 + B] * R)
    cu = [int(x) for x in cu_np.tolist()]
    ns = num_sampled.tolist()

    # Independent reference: draw row per chain depth, log_softmax gather.
    rows_ref = []
    tokens_ref = []
    for r in range(R):
        for d in range(ns[r]):
            row = cu[r] + (int(acc[r, d].item()) + 1 if acc[r, d].item() >= 0 else 0)
            rows_ref.append(row)
            tokens_ref.append(int(sampled[r, d].item()))
    rows_t = torch.tensor(rows_ref, device=device)
    ref = torch.nn.functional.log_softmax(processed[rows_t], dim=-1).gather(
        -1, torch.tensor(tokens_ref, device=device).unsqueeze(-1)
    )[:, 0]

    state_np = np.zeros(R, dtype=np.intp)
    lp = _df2t_tree_logprobs(
        processed,
        sampled,
        num_sampled,
        acc,
        cu_np,
        state_np,
        5,
        "raw_logprobs",
        None,
        0,
    )
    assert lp is not None
    assert isinstance(lp, LogprobsTensors)
    assert lp.logprob_token_ids.shape[0] == sum(ns)
    # Column 0 holds the emitted token; its logprob is the draw-row gather.
    assert lp.logprob_token_ids[:, 0].tolist() == tokens_ref
    assert torch.allclose(lp.logprobs[:, 0], ref, atol=1e-6)
    # Per-request CSR boundaries in batch order.
    want_cu = [0]
    for v in ns:
        want_cu.append(want_cu[-1] + v)
    assert lp.cu_num_generated_tokens == want_cu
    # Top-k alternatives are the same softmax, ordered by value.
    topk = torch.topk(torch.nn.functional.log_softmax(processed[rows_t], dim=-1), 5)
    assert torch.allclose(
        torch.sort(lp.logprobs, dim=-1, descending=True).values[:, 1 : 5 + 1],
        torch.sort(topk.values, dim=-1, descending=True).values[:, :5],
        atol=1e-6,
    )


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_df2t_tree_logprobs_none_when_unrequested(device):
    R, B, V = 2, 4, 16
    target_logits = torch.randn(R, 1 + B, V, device=device)
    token_ids, parents, num_nodes = _chain_tree(R, B, device)
    verifier = TreeVerifier(R, B, V, device)
    sampled, num_sampled = verifier(
        target_logits, token_ids, parents, num_nodes, greedy=True
    )
    processed = target_logits.reshape(-1, V)
    acc = verifier.accepted_arenas
    cu_np = np.cumsum([0] + [1 + B] * R)
    assert (
        _df2t_tree_logprobs(
            processed,
            sampled,
            num_sampled,
            acc,
            cu_np,
            np.zeros(R, dtype=np.intp),
            -1,
            "raw_logprobs",
            None,
            0,
        )
        is None
    )
