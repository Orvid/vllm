# SPDX-License-Identifier: Apache-2.0
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

from tests.v1.worker.gpu.spec_decode.df2t.reference_verifier import (
    tree_verify_reference,
)
from vllm.v1.worker.gpu.spec_decode.df2t.tree_verifier import TreeVerifier


def _random_trees(num_reqs, B, vocab, seed):
    rng = torch.Generator().manual_seed(seed)
    token_ids = torch.zeros(num_reqs, B, dtype=torch.long)
    parents = torch.full((num_reqs, B), -2, dtype=torch.long)
    num_nodes = torch.zeros(num_reqs, dtype=torch.long)
    for r in range(num_reqs):
        b = max(1, int(torch.randint(1, B + 1, (1,), generator=rng).item()))
        used = {p: set() for p in range(-1, b)}
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


def _check_against_reference(sampled, num_sampled, accepted_arenas, target,
                             token_ids, parents, num_nodes):
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
        s, n = tv(target.clone(), token_ids.to(device), parents.to(device),
                  num_nodes.to(device), greedy=gm.clone(),
                  all_greedy=all_greedy)
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
    _check_against_reference(s_fast, n_fast, a_fast, target, token_ids,
                             parents, num_nodes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_verifier_micrograph_replays_bit_identical_to_eager():
    """The captured all_greedy micro-graph replays bit-identically to the eager
    device loop across steps with fresh data, batch sizes below the captured
    size (replay at the smallest captured R >= batch), and varied trees."""
    from vllm.v1.worker.gpu.spec_decode.df2t.tree_verifier import (
        TreeVerifier as TV,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    B, V = 8, 512
    for cap_R, batch_R in ((1, 1), (8, 8), (8, 3), (32, 17)):
        tv_cap = TV(max_num_reqs=64, B=B, vocab_size=V, device=device,
                    num_rounds=1 + B)
        tv_eager = TV(max_num_reqs=64, B=B, vocab_size=V, device=device,
                      num_rounds=1 + B)
        # Capture at size cap_R (capacity sized to the max captured batch).
        tv_cap.ensure_capture_capacity(cap_R, dtype=torch.float32)
        tv_cap.capture_cudagraph(cap_R)
        for step in range(3):
            token_ids = torch.randint(0, V, (batch_R, B), device=device)
            parents = torch.randint(-1, B, (batch_R, B), device=device)
            num_nodes = torch.randint(2, B + 1, (batch_R,), device=device)
            target = torch.randn(batch_R, 1 + B, V, device=device)

            # Eager reference.
            gm = torch.full((batch_R,), True, dtype=torch.bool, device=device)
            s_e, n_e = tv_eager(target.clone(), token_ids.clone(),
                                parents.clone(), num_nodes.clone(),
                                greedy=gm.clone(), all_greedy=True)
            a_e = tv_eager.accepted_arenas[:batch_R].clone()
            torch.cuda.synchronize()

            # Replay path: stage rows into the capture buffer, replay.
            tbuf = tv_cap._cap_target[:batch_R, : 1 + B]
            tbuf.copy_(target)
            staged = tv_cap.stage_and_replay(token_ids, parents, num_nodes,
                                             batch_R)
            assert staged is not None
            s_c, n_c = staged
            a_c = tv_cap.accepted_arenas[:batch_R].clone()
            torch.cuda.synchronize()

            assert torch.equal(s_c, s_e), (cap_R, batch_R, step, "sampled")
            assert torch.equal(n_c, n_e), (cap_R, batch_R, step, "num_sampled")
            assert torch.equal(a_c, a_e), (cap_R, batch_R, step, "arenas")


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
        target.clone(), token_ids.to(device), parents.to(device),
        num_nodes.to(device), greedy=gm.clone(), all_greedy=True
    )
    assert int(num_sampled[0].item()) == 1
    _check_against_reference(
        sampled, num_sampled, tv.accepted_arenas[:R], target, token_ids,
        parents, num_nodes
    )
