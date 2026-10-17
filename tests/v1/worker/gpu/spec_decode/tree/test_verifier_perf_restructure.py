# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the per-step verifier perf restructure (fast path).

Two guarantees, both bit-identity discriminations on the PRODUCTION device loop
(``_verify_device``, the no-params/seeds path the runner takes every step):

1. ``all_greedy`` (T=0) fast path: pure argmax over the gathered row, no
   per-round softmax / batched multinomial. The multinomial spy is the
   discriminator: the slow path calls it every round even on an all-greedy
   batch (discarded by the greedy mask); the fast path must never call it.
   Outputs must stay bit-identical to the slow path AND match the reference.

2. The allocation rewrite (full-batch masked writes, no data-dependent
   ``nonzero`` compaction — the capture prerequisite) must not change the
   OBSERVABLE contract: every read column still equals the reference oracle,
   and an early-stop request's don't-care columns keep their init values
   (``sampled`` zero, ``accepted_arenas`` -1), exactly as the pre-rewrite
   device loop left them by only writing active rows.
"""

import pytest
import torch

from tests.v1.worker.gpu.spec_decode.tree.reference_verifier import (
    tree_verify_reference,
)
from vllm.v1.worker.gpu.spec_decode.tree.tree_verifier import TreeVerifier


def _random_trees(num_reqs, B, vocab, seed):
    rng = torch.Generator().manual_seed(seed)
    token_ids = torch.zeros(num_reqs, B, dtype=torch.long)
    parents = torch.full((num_reqs, B), -2, dtype=torch.long)
    num_nodes = torch.zeros(num_reqs, dtype=torch.long)
    for r in range(num_reqs):
        b = max(1, int(torch.randint(1, B + 1, (1,), generator=rng).item()))
        used: dict[int, set[int]] = {p: set() for p in range(-1, b)}
        for i in range(b):
            p = -1 if i == 0 else int(torch.randint(-1, i, (1,), generator=rng).item())
            tok = int(torch.randint(0, vocab, (1,), generator=rng).item())
            while tok in used[p]:
                tok = int(torch.randint(0, vocab, (1,), generator=rng).item())
            used[p].add(tok)
            token_ids[r, i] = tok
            parents[r, i] = p
        num_nodes[r] = b
    return token_ids, parents, num_nodes


def _check_against_reference(
    sampled, num_sampled, accepted_arenas, target, token_ids, parents, num_nodes
):
    """Assert every READ column matches the greedy reference oracle and every
    don't-care column keeps its init value (sampled 0, accepted_arenas -1)."""
    R, nbuf, V = target.shape  # nbuf = 1 + B
    B = nbuf - 1
    for r in range(R):
        b = int(num_nodes[r])
        out_ref, nacc_ref = tree_verify_reference(
            target[r].cpu(), token_ids[r, :b], parents[r, :b], greedy=True
        )
        n = int(num_sampled[r].item())
        assert n - 1 == nacc_ref
        # Read columns: the emitted token stream.
        assert sampled[r, :n].tolist() == out_ref
        # Accepted-chain arenas for the accepted depths must be real arenas.
        for d in range(1, n):
            assert accepted_arenas[r, d].item() != -1
        # Don't-care columns (beyond the emitted stream) keep init values.
        for d in range(n, B + 1):
            assert sampled[r, d].item() == 0
            assert accepted_arenas[r, d].item() == -1
    return True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_all_greedy_fast_path_bit_identical_and_multinomial_free(monkeypatch):
    torch.manual_seed(7)
    R, B, V = 5, 6, 320
    device = torch.device("cuda")
    token_ids, parents, num_nodes = _random_trees(R, B, V, seed=51)
    target = torch.randn(R, 1 + B, V, device=device)

    calls = [0]
    _orig_multinomial = torch.multinomial

    def _spy_multinomial(*args, **kwargs):
        calls[0] += 1
        return _orig_multinomial(*args, **kwargs)

    def run(all_greedy):
        tv = TreeVerifier(R, B, V, device, num_rounds=1 + B)
        gm = torch.full((R,), True, dtype=torch.bool, device=device)
        s, n = tv(
            target.clone(),
            token_ids.to(device),
            parents.to(device),
            num_nodes.to(device),
            greedy=gm.clone(),
            all_greedy=all_greedy,
        )
        return s.clone(), n.clone(), tv.accepted_arenas[:R].clone()

    calls[0] = 0
    monkeypatch.setattr(torch, "multinomial", _spy_multinomial)
    s_slow, n_slow, a_slow = run(False)
    slow_calls = calls[0]
    assert slow_calls > 0, "slow path must sample every round"

    calls[0] = 0
    s_fast, n_fast, a_fast = run(True)
    assert calls[0] == 0, "all_greedy path must never call torch.multinomial"

    assert torch.equal(s_slow, s_fast)
    assert torch.equal(n_slow, n_fast)
    assert torch.equal(a_slow, a_fast)
    _check_against_reference(
        s_fast, n_fast, a_fast, target, token_ids, parents, num_nodes
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_greedy_fused_walk_bit_identical_to_torch_loop():
    """The fused triton T=0 tree-walk kernel must be bit-identical to the
    torch device loop (the reference) on sampled/num_sampled/accepted_arenas
    across varied trees (early-stop, full chain, multi-request, small B) and
    must reproduce the reference oracle's emitted stream on every column it
    reads."""
    from vllm.v1.worker.gpu.spec_decode.tree.tree_verifier import (
        TreeVerifier as TV,
    )

    torch.manual_seed(4)
    for R, B, V in ((1, 4, 96), (8, 6, 256), (6, 8, 512)):
        device = torch.device("cuda")
        token_ids, parents, num_nodes = _random_trees(R, B, V, seed=R * 101 + B)
        target = torch.randn(R, 1 + B, V, device=device)
        token_ids_d = token_ids.to(device)
        parents_d = parents.to(device)
        num_nodes_d = num_nodes.to(device)
        gm = torch.full((R,), True, dtype=torch.bool, device=device)

        # unfused torch loop (reference)
        tv_u = TV(R, B, V, device, num_rounds=1 + B)
        s_u, n_u = tv_u._verify_device(
            target.clone(),
            token_ids_d.clone(),
            parents_d.clone(),
            num_nodes_d.clone(),
            gm.clone(),
            all_greedy=True,
            fused=False,
        )
        a_u = tv_u.accepted_arenas[:R].clone()
        torch.cuda.synchronize()

        # fused kernel (production)
        tv_f = TV(R, B, V, device, num_rounds=1 + B)
        s_f, n_f = tv_f._verify_device(
            target.clone(),
            token_ids_d.clone(),
            parents_d.clone(),
            num_nodes_d.clone(),
            gm.clone(),
            all_greedy=True,
            fused=True,
        )
        a_f = tv_f.accepted_arenas[:R].clone()
        torch.cuda.synchronize()

        assert torch.equal(s_f, s_u), (R, B, "sampled")
        assert torch.equal(n_f, n_u), (R, B, "num_sampled")
        assert torch.equal(a_f, a_u), (R, B, "accepted_arenas")

        # and the emitted stream matches the oracle on every read column
        for r in range(R):
            b = int(num_nodes[r])
            out_ref, nacc_ref = tree_verify_reference(
                target[r].cpu(), token_ids[r, :b], parents[r, :b], greedy=True
            )
            n = int(n_f[r].item())
            assert n - 1 == nacc_ref
            assert s_f[r, :n].tolist() == out_ref
    # exercise __call__ routing (fused used under the production shape)
    R, B, V = 5, 6, 320
    token_ids, parents, num_nodes = _random_trees(R, B, V, seed=71)
    target = torch.randn(R, 1 + B, V, device=device)
    tv = TV(R, B, V, device, num_rounds=1 + B)
    gm = torch.full((R,), True, dtype=torch.bool, device=device)
    s, n = tv(
        target.clone(),
        token_ids.to(device),
        parents.to(device),
        num_nodes.to(device),
        greedy=gm.clone(),
        all_greedy=True,
    )
    assert s.shape == (R, 1 + B) and n.shape == (R,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_alloc_rewrite_preserves_observable_contract():
    """The masked full-batch-write rewrite must keep the exact observable
    contract of the device loop: read columns match the oracle and don't-care
    columns of early-stop rows stay at init values (0 for sampled, -1 for
    accepted_arenas), across batches with early-stop AND full-chain requests."""
    torch.manual_seed(3)
    # One request stops at depth 0 (its anchor matches nothing), another runs
    # to the full chain; the rest land in between.
    R, B, V = 4, 6, 96
    device = torch.device("cuda")
    token_ids, parents, num_nodes = _random_trees(R, B, V, seed=61)
    target = torch.randn(R, 1 + B, V, device=device)
    # Force request 0's anchor argmax outside every drafted child so it stops
    # at depth 0 (num_sampled == 1).
    target[0, 0, :].fill_(-100.0)
    target[0, 0, V - 1] = 5.0

    tv = TreeVerifier(R, B, V, device, num_rounds=1 + B)
    gm = torch.full((R,), True, dtype=torch.bool, device=device)
    sampled, num_sampled = tv(
        target.clone(),
        token_ids.to(device),
        parents.to(device),
        num_nodes.to(device),
        greedy=gm.clone(),
        all_greedy=True,
    )
    assert int(num_sampled[0].item()) == 1
    _check_against_reference(
        sampled,
        num_sampled,
        tv.accepted_arenas[:R],
        target,
        token_ids,
        parents,
        num_nodes,
    )
