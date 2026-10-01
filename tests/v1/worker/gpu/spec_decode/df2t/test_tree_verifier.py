# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the torch TreeVerifier (v5 spec §3.6 TRAVERSAL).

The verifier must reproduce ``tree_verify_reference`` exactly but vectorized
across requests: at each depth draw ONE token from the fully processed target
distribution at the current row, descend iff it is a child of the current node,
else emit it and stop (the same rule emits the leaf bonus on a fully accepted
path). Requests that stop early are masked out of later depths. The output
contract matches the rejection samplers: ``sampled [num_reqs, 1 + B]`` and
``num_sampled [num_reqs]`` (per-request accepted-emitted length).
"""

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.df2t.reference_verifier import (
    _processed as _ref_processed,
)
from vllm.v1.worker.gpu.spec_decode.df2t.reference_verifier import (
    tree_verify_reference,
)
from vllm.v1.worker.gpu.spec_decode.df2t.tree_verifier import (
    TreeVerifier,
)

_DEVICES = ["cpu", "cuda"]


def _make_slot_layer(num_slots, num_heads, head_dim, device="cuda"):
    """Engine-style LBNHC per-layer view: (logical view, physical storage).

    Physical storage is the contiguous ``[1, num_slots, num_heads, head_dim]``
    array; the view is the non-contiguous transpose ``[1, num_heads, num_slots,
    head_dim]`` (strides ``(num_slots * num_heads * head_dim, head_dim,
    num_heads * head_dim, 1)``), exactly what ``create_kv_cache_views`` yields
    under LBNHC. ``compact_accepted_chain`` addresses slot ``phys`` via the
    flat ``phys * elems`` run, so tests seed/read through ``storage[0, phys]``
    and hand ``view`` to the primitive.
    """
    storage = torch.zeros(
        1, num_slots, num_heads, head_dim, dtype=torch.float32, device=device
    )
    return storage.transpose(1, 2), storage


class _FakeBlockTables:
    """Minimal stand-in for ``BlockTables.gather_block_tables``.

    Holds one row-major int64 block table per KV group, indexed by the padded
    request id (the mapping is the identity on patch rows, which is not what is
    under test here).
    """

    def __init__(self, tables):
        self._tables = [t for t in tables]

    def gather_block_tables(self, idx_mapping, num_reqs_padded=0):
        del idx_mapping  # identity gather on request rows
        return tuple(t[:num_reqs_padded] for t in self._tables)


def _compute_tree_depths(parents: torch.Tensor) -> torch.Tensor:
    """A valid depth per arena (parents precede children; ``-1`` = anchor)."""
    b = parents.shape[0]
    depths = torch.empty(b, dtype=torch.long)
    for i in range(b):
        p = int(parents[i])
        depths[i] = depths[p] + 1 if p >= 0 else 1
    return depths


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_df2t_verify_rejects_logprobs_requests():
    """A df2t request that asks for logprobs must fail closed, never silently
    return empty logprobs (the tree-verify path does not emit them yet)."""
    import numpy as np

    from vllm.v1.worker.gpu.model_runner import _df2t_verify_rejects_logprobs
    from vllm.v1.worker.gpu.sample.states import SamplingStates

    ss = SamplingStates(4, 128)
    # No request asks for logprobs -> allowed.
    _df2t_verify_rejects_logprobs(ss, np.arange(4))
    ss.num_logprobs[1] = 1
    with pytest.raises(NotImplementedError, match="df2t"):
        _df2t_verify_rejects_logprobs(ss, np.arange(4))


def _random_trees(
    num_reqs: int, B: int, vocab: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-request random valid trees with variable node counts.

    Node ``i`` gets a parent in ``[-1, i-1]`` (arena-order parents before
    children, ``-1`` = anchor child) and a token unique among its siblings.
    Unused arena rows carry the ``-2`` padding sentinel so they can never match
    a live node during traversal.
    """
    rng = torch.Generator().manual_seed(seed)
    token_ids = torch.zeros(num_reqs, B, dtype=torch.long)
    parents = torch.full((num_reqs, B), -2, dtype=torch.long)
    num_nodes = torch.zeros(num_reqs, dtype=torch.long)
    for r in range(num_reqs):
        b = int(torch.randint(1, B + 1, (1,), generator=rng).item())
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


def _reference_runs(
    target_logits: torch.Tensor,
    token_ids: torch.Tensor,
    parents: torch.Tensor,
    num_nodes: torch.Tensor,
    greedy: bool,
    seed: int | None = None,
    **sampling,
) -> tuple[list[list[int]], list[int]]:
    """Run ``tree_verify_reference`` per request on the unpadded rows."""
    out_lists: list[list[int]] = []
    num_accepted: list[int] = []
    for r in range(target_logits.shape[0]):
        b = int(num_nodes[r].item())
        out, nacc = tree_verify_reference(
            target_logits[r],
            token_ids[r, :b],
            parents[r, :b],
            greedy=greedy,
            seed=seed + r if seed is not None else None,
            **sampling,
        )
        out_lists.append(out)
        num_accepted.append(nacc)
    return out_lists, num_accepted


@pytest.mark.parametrize("device", _DEVICES)
def test_tree_verifier_matches_reference_greedy(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    num_reqs, B, vocab = 7, 8, 96
    token_ids, parents, num_nodes = _random_trees(num_reqs, B, vocab, seed=11)
    target_logits = torch.randn(num_reqs, 1 + B, vocab, device=device)
    token_ids = token_ids.to(device)
    parents = parents.to(device)
    num_nodes = num_nodes.to(device)

    tv = TreeVerifier(num_reqs, B, vocab, device)
    sampled, num_sampled = tv(target_logits, token_ids, parents, num_nodes, greedy=True)

    out_lists, num_accepted = _reference_runs(
        target_logits, token_ids, parents, num_nodes, greedy=True
    )
    for r in range(num_reqs):
        assert int(num_sampled[r].item()) - 1 == num_accepted[r]
        assert sampled[r, : int(num_sampled[r].item())].tolist() == out_lists[r]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_builder_partial_tree_wide_buffer_verifies_losslessly():
    """A tree that exhausts its prefix space emits fewer live nodes than its
    buffer width; the STASHED 30-wide row must still verify losslessly because
    the verifier delimits by the live ``num_nodes`` (the inert tail is never
    matched). This is the per-request-width-below-ceiling decoupling: a
    request whose B_i is below the step's graph width lays out fewer nodes and
    the padding is inert."""
    from vllm.v1.worker.gpu.spec_decode.df2t.tree_builder import TreeBuilder

    torch.manual_seed(0)
    R, L, K, B, vocab = 2, 2, 2, 30, 128
    # K=2, L=2 -> 2 + 4 = 6 live prefixes, far below the B=30 buffer.
    scores = torch.randn(R, L, K, K, device="cuda")
    cand = torch.randint(0, vocab, (R, L, K), device="cuda")
    tb = TreeBuilder(R, L, K, B)
    tree = tb.build(scores, cand)
    assert tree.num_nodes is not None
    assert tree.token_ids.shape == (R, B)
    for r in range(R):
        assert int(tree.num_nodes[r].item()) <= 6 < B
        # Sanitize exactly as get_latest_tree_drafts does (inert tail).
        live = int(tree.num_nodes[r].item())
        tree.token_ids[r, live:].fill_(0)
        tree.depths[r, live:].zero_()
        tree.parents[r, live:].fill_(-2)
        tree.words[r, live:].zero_()

        target_logits = torch.randn(1, 1 + B, vocab, device="cuda")
        tv = TreeVerifier(1, B, vocab, torch.device("cuda"))
        sampled, num_sampled = tv(
            target_logits,
            tree.token_ids[r : r + 1],
            tree.parents[r : r + 1],
            tree.num_nodes[r : r + 1],
            greedy=True,
        )
        # Lossless: the reference on the live rows matches the tree verify.
        out_ref, nacc_ref = tree_verify_reference(
            target_logits[0],
            tree.token_ids[r, :live].cpu(),
            tree.parents[r, :live].cpu(),
            greedy=True,
        )
        n = int(num_sampled[0].item())
        assert n - 1 == nacc_ref
        assert sampled[0, :n].tolist() == out_ref


@pytest.mark.parametrize("device", _DEVICES)
def test_tree_verifier_matches_reference_sampling(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(1)
    # top-k boundary ties and top-p-after-top-k: the processed convention must
    # match the oracle bit-for-bit, so flip exact top-k boundary values in the
    # logits to exercise the >= boundary-tie keep.
    num_reqs, B, vocab = 7, 8, 64
    token_ids, parents, num_nodes = _random_trees(num_reqs, B, vocab, seed=22)
    target_logits = (torch.randn(num_reqs, 1 + B, vocab, device=device) * 0.5).round(
        decimals=2
    )
    # Make the k-th boundary tie common so topk ordering is stressed.
    thr = torch.topk(target_logits, 4, dim=-1).values[:, :, -1:]
    target_logits = torch.where(
        torch.rand_like(target_logits[..., :1]) < 0.5,
        torch.where(target_logits < thr, thr, target_logits),
        target_logits,
    )
    token_ids = token_ids.to(device)
    parents = parents.to(device)
    num_nodes = num_nodes.to(device)

    sampling = dict(temperature=0.8, top_k=4, top_p=0.9)
    seeds = torch.arange(num_reqs, dtype=torch.long, device=device) + 4000

    tv = TreeVerifier(num_reqs, B, vocab, device)
    sampled, num_sampled = tv(
        target_logits,
        token_ids,
        parents,
        num_nodes,
        greedy=False,
        temperature=sampling["temperature"],
        top_k=sampling["top_k"],
        top_p=sampling["top_p"],
        seeds=seeds,
    )

    # The reference draws per request from a sequential generator seeded per
    # request; the verifier must reproduce the same streams exactly.
    out_lists, num_accepted = _reference_runs(
        target_logits,
        token_ids,
        parents,
        num_nodes,
        greedy=False,
        seed=4000,
        **sampling,
    )
    for r in range(num_reqs):
        assert int(num_sampled[r].item()) - 1 == num_accepted[r]
        assert sampled[r, : int(num_sampled[r].item())].tolist() == out_lists[r]


@pytest.mark.parametrize("device", _DEVICES)
def test_tree_verifier_t0_argmax_invariant(device):
    """v5 §3.7 invariant 4: at T=0 every emitted token (accepted, mismatch,
    and the leaf bonus on a fully accepted path) equals the argmax of the
    processed distribution at its row. The verifier must not introduce a
    second draw or a separate bonus rule."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(2)
    num_reqs, B, vocab = 5, 6, 50
    token_ids, parents, num_nodes = _random_trees(num_reqs, B, vocab, seed=33)
    target_logits = torch.randn(num_reqs, 1 + B, vocab, device=device)
    token_ids = token_ids.to(device)
    parents = parents.to(device)
    num_nodes = num_nodes.to(device)

    cases = [dict(), dict(top_k=5, top_p=0.92)]
    tv = TreeVerifier(num_reqs, B, vocab, device)
    for sampling in cases:
        sampled, num_sampled = tv(
            target_logits,
            token_ids,
            parents,
            num_nodes,
            greedy=True,
            **sampling,
        )
        for r in range(num_reqs):
            # Replay the deterministic T=0 walk to recover each emitted row.
            b = int(num_nodes[r].item())
            u = -1
            rows: list[int] = []
            for d in range(B + 1):
                row = 0 if u == -1 else 1 + u
                rows.append(row)
                a = int(
                    torch.argmax(
                        _ref_processed(
                            target_logits[r, row],
                            sampling.get("temperature"),
                            sampling.get("top_k"),
                            sampling.get("top_p"),
                        )
                    )
                )
                match = None
                for i in range(b):
                    if int(parents[r, i]) == u and int(token_ids[r, i]) == a:
                        match = i
                        break
                if match is None:
                    break
                u = match
            n = int(num_sampled[r].item())
            assert n == len(rows)
            for d in range(n):
                expected = int(
                    torch.argmax(
                        _ref_processed(
                            target_logits[r, rows[d]],
                            sampling.get("temperature"),
                            sampling.get("top_k"),
                            sampling.get("top_p"),
                        )
                    )
                )
                assert sampled[r, d].item() == expected


@pytest.mark.parametrize("device", _DEVICES)
def test_tree_verifier_chain_equals_greedy_walk_t0(device):
    """A chain-shaped tree (B=L, node i's parent is i-1) at T=0 reproduces the
    DFlash2 greedy-walk accepted sequence: walk positions, accept while the
    target argmax equals the drafted token, else emit the argmax and stop."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(3)
    num_reqs, B, vocab = 4, 7, 80
    # Chain draft tokens (distinct) in arena order, token of node i.
    chain_tokens = torch.randint(0, vocab, (num_reqs, B), device=device)
    token_ids = chain_tokens.clone()
    parents = torch.stack(
        [torch.arange(-1, B - 1, dtype=torch.long, device=device)] * num_reqs
    )
    num_nodes = torch.full((num_reqs,), B, dtype=torch.long, device=device)
    target_logits = torch.randn(num_reqs, 1 + B, vocab, device=device)

    tv = TreeVerifier(num_reqs, B, vocab, device)
    sampled, num_sampled = tv(target_logits, token_ids, parents, num_nodes, greedy=True)

    for r in range(num_reqs):
        # DFlash2 greedy walk, re-derived without any tree machinery.
        expected: list[int] = []
        u = -1
        for d in range(B + 1):
            row = 0 if u == -1 else 1 + u
            a = int(torch.argmax(target_logits[r, row]).item())
            expected.append(a)
            if u + 1 < B and chain_tokens[r, u + 1].item() == a:
                u += 1
                continue
            break
        assert int(num_sampled[r].item()) == len(expected)
        assert sampled[r, : len(expected)].tolist() == expected


@pytest.mark.parametrize("device", _DEVICES)
def test_tree_verifier_stops_and_masks_early_done_requests(device):
    """A request whose anchor draw matches no tree child emits exactly one
    token (the anchor row draw) and is masked out of every later depth, so
    num_sampled == 1 and only the anchor column of its sampled row is written."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # One request has an anchor whose argmax never matches any drafted token:
    # it emits exactly one token and num_sampled == 1.
    num_reqs, B, vocab = 2, 4, 32
    token_ids = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]], device=device)
    parents = torch.tensor([[-1, 0, 1, 2], [-1, 0, 1, 2]], device=device)
    num_nodes = torch.full((num_reqs,), B, dtype=torch.long, device=device)
    # Request 0: anchor logits argmax = 7 which is not in the tree at all.
    logits0 = torch.randn(1 + B, vocab, device=device)
    logits0[0, :] = -100.0
    logits0[0, 7] = 10.0
    # Request 1: argmax at every row is a tree token, so a long acceptance.
    logits1 = torch.randn(1 + B, vocab, device=device)
    for i, t in enumerate([1, 2, 3, 4]):
        logits1[i, :] = -100.0
        logits1[i, t] = 10.0
    logits1[1 + B - 1, 5] = 30.0  # terminal bonus
    target_logits = torch.stack([logits0, logits1], dim=0)

    tv = TreeVerifier(num_reqs, B, vocab, device)
    sampled, num_sampled = tv(target_logits, token_ids, parents, num_nodes, greedy=True)
    assert int(num_sampled[0].item()) == 1
    assert sampled[0, 0].item() == 7
    # Request 0 stops at depth 0: no later column of its row is written.
    assert sampled[0, 1:].tolist() == [0] * B
    assert int(num_sampled[1].item()) == B + 1
    assert sampled[1, : B + 1].tolist() == [1, 2, 3, 4, 5]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_model_runner_df2t_tree_verify_output_contract():
    """The runner wiring (_df2t_tree_verify) gathers each request's target
    rows, verifies with the TreeVerifier, and emits the exact rejection-sampler
    contract: sampled [R, 1+B] int64 + num_sampled [R] int32, with
    num_rejected == num_logits - num_sampled so the existing postprocess
    plumbing consumes it unchanged."""
    from types import MethodType, SimpleNamespace

    import numpy as np

    from vllm.v1.worker.gpu.input_batch import (
        InputBatch,
        InputBuffers,
    )
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.sample.states import SamplingStates

    device = torch.device("cuda")
    num_reqs, B, vocab = 4, 6, 64
    torch.manual_seed(9)
    token_ids, parents, num_nodes = _random_trees(num_reqs, B, vocab, seed=5)
    logits_all = torch.randn(num_reqs, 1 + B, vocab, device=device)

    # Flatten into the runner's [num_logits, vocab] layout with per-request
    # rows [cu[r]:cu[r+1]] == request r's 1 + num_nodes[r] target rows.
    cu_np = np.zeros(num_reqs + 1, dtype=np.int64)
    num_nodes_cpu = num_nodes.numpy()
    cu_np[1:] = np.cumsum(1 + num_nodes_cpu)
    logits = torch.cat(
        [logits_all[r, : 1 + int(num_nodes[r])] for r in range(num_reqs)], dim=0
    ).to(device)
    cu = torch.tensor(cu_np, dtype=torch.int32, device=device)
    assert logits.shape[0] == int(cu_np[-1])

    batches = InputBuffers(num_reqs, logits.shape[0], device)
    ib = InputBatch.make_dummy(num_reqs, logits.shape[0], batches, is_padding=True)
    ib.cu_num_logits = cu
    ib.seq_lens = torch.ones(num_reqs, dtype=torch.int32, device=device) * 50
    ib.trees = [
        SimpleNamespace(
            token_ids=token_ids[r, : int(num_nodes[r])].to(device),
            parents=parents[r, : int(num_nodes[r])].to(device),
            depths=_compute_tree_depths(parents[r, : int(num_nodes[r])]).to(device),
        )
        for r in range(num_reqs)
    ]

    tree_verifier = TreeVerifier(num_reqs, B, vocab, device)
    ss = SamplingStates(num_reqs, vocab)
    # temperature 0.0 is the vLLM greedy convention; the wiring maps it to the
    # verifier's greedy flag, so compare against the greedy reference.
    ss.temperature.np[:] = 0.0
    ss.temperature.copy_to_uva()
    # top_k/top_p default to "keep all"; no request is seeded.
    prefill_len = SimpleNamespace(
        gpu=torch.zeros(num_reqs, dtype=torch.int32, device=device)
    )
    pipeline_calls: list[int] = []

    def apply_sampling_params(logits, *args, **kwargs):
        # All-default params: the real pipeline would return the logits
        # unchanged; record that the wiring routed through it.
        pipeline_calls.append(1)
        return logits

    # Real accepted-KV compaction wiring (not a stub): a single LBNHC KV group
    # whose block-slot -> physical mapping is a constant offset per slot. C == 0
    # keeps every accepted slot inside block 0, so compaction always stays
    # within the view but still moves accepted KV to the canonical slot.
    kv_view, kv_storage = _make_slot_layer(16, 2, 4, device)
    block_table = torch.zeros(num_reqs, 1, dtype=torch.long, device=device)

    fake_self = SimpleNamespace(
        tree_verifier=tree_verifier,
        sampler=SimpleNamespace(
            sampling_states=ss,
            req_states=SimpleNamespace(prefill_len=prefill_len),
            compute_nans=False,
            apply_sampling_params=apply_sampling_params,
        ),
        device=device,
        _kv_cache_views_by_name={"layer.0": kv_view},
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["layer.0"],
                    kv_cache_spec=SimpleNamespace(block_size=16),
                )
            ]
        ),
        block_tables=_FakeBlockTables([block_table]),
        req_states=SimpleNamespace(
            num_computed_tokens=SimpleNamespace(
                gpu=torch.zeros(num_reqs, dtype=torch.int32, device=device)
            )
        ),
    )
    fake_self._compact_df2t_accepted_kv = MethodType(
        GPUModelRunner._compact_df2t_accepted_kv, fake_self
    )

    out = GPUModelRunner._df2t_tree_verify(fake_self, logits, ib)
    sampled, num_sampled = out.sampled_token_ids, out.num_sampled

    assert pipeline_calls, "df2t must route logits through the sampling pipeline"

    assert sampled.dtype == torch.int64 and sampled.shape == (num_reqs, B + 1)
    assert num_sampled.dtype == torch.int32
    assert out.num_rejected is not None
    for r in range(num_reqs):
        b = int(num_nodes[r].item())
        out_ref, nacc_ref = tree_verify_reference(
            logits_all[r], token_ids[r, :b], parents[r, :b], greedy=True
        )
        n = int(num_sampled[r].item())
        assert n - 1 == nacc_ref
        assert sampled[r, :n].tolist() == out_ref
        assert int(out.num_rejected[r].item()) == int(cu_np[r + 1] - cu_np[r]) - n


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_model_runner_df2t_tree_verify_mixed_batch():
    """A request without a scheduled tree in a df2t step must not crash the
    worker: it is verified as an empty tree (a plain single-token draw from its
    anchor row, num_sampled == 1), while the tree-carrying requests in the same
    step keep their full tree verification untouched."""
    from types import MethodType, SimpleNamespace

    import numpy as np

    from vllm.v1.worker.gpu.input_batch import (
        InputBatch,
        InputBuffers,
    )
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.sample.states import SamplingStates

    device = torch.device("cuda")
    vocab = 48
    B = 5
    # Request 0 carries a tree (chain of 2 nodes), request 1 has no tree.
    num_reqs = 2
    torch.manual_seed(10)
    token_ids = torch.tensor([[7, 3], [0, 0]], device=device)
    parents = torch.tensor([[-1, 0], [-2, -2]], device=device)
    logits_all = torch.randn(num_reqs, 1 + B, vocab, device=device)

    # Flatten to the runner layout: request 0 -> 1 + 2 rows, request 1 -> 1 row.
    cu_np = np.array([0, 1 + 2, 1 + 2 + 1], dtype=np.int64)
    logits = torch.cat([logits_all[0, :3], logits_all[1, :1]], dim=0).to(device)
    cu = torch.tensor(cu_np, dtype=torch.int32, device=device)

    batches = InputBuffers(num_reqs, logits.shape[0], device)
    ib = InputBatch.make_dummy(num_reqs, logits.shape[0], batches, is_padding=True)
    ib.cu_num_logits = cu
    ib.seq_lens = torch.ones(num_reqs, dtype=torch.int32, device=device) * 50
    ib.trees = [
        SimpleNamespace(
            token_ids=token_ids[0, :2],
            parents=parents[0, :2],
            depths=_compute_tree_depths(parents[0, :2]).to(device),
        ),
        None,
    ]

    ss = SamplingStates(num_reqs, vocab)
    ss.temperature.np[:] = 0.0
    ss.temperature.copy_to_uva()
    prefill_len = SimpleNamespace(
        gpu=torch.zeros(num_reqs, dtype=torch.int32, device=device)
    )

    def apply_sampling_params(logits, *args, **kwargs):
        return logits

    # Real accepted-KV compaction wiring (not a stub), same minimal single-
    # group setup as the output-contract test.
    kv_view, _kv_storage = _make_slot_layer(16, 2, 4, device)
    block_table = torch.zeros(num_reqs, 1, dtype=torch.long, device=device)

    fake_self = SimpleNamespace(
        tree_verifier=TreeVerifier(num_reqs, B, vocab, device),
        sampler=SimpleNamespace(
            sampling_states=ss,
            req_states=SimpleNamespace(prefill_len=prefill_len),
            compute_nans=False,
            apply_sampling_params=apply_sampling_params,
        ),
        device=device,
        _kv_cache_views_by_name={"layer.0": kv_view},
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["layer.0"],
                    kv_cache_spec=SimpleNamespace(block_size=16),
                )
            ]
        ),
        block_tables=_FakeBlockTables([block_table]),
        req_states=SimpleNamespace(
            num_computed_tokens=SimpleNamespace(
                gpu=torch.zeros(num_reqs, dtype=torch.int32, device=device)
            )
        ),
    )
    fake_self._compact_df2t_accepted_kv = MethodType(
        GPUModelRunner._compact_df2t_accepted_kv, fake_self
    )

    out = GPUModelRunner._df2t_tree_verify(fake_self, logits, ib)
    sampled, num_sampled = out.sampled_token_ids, out.num_sampled

    assert sampled.shape == (num_reqs, 1 + 2)  # 1 + batch b_max (request 0)
    # Plain request: exactly one token from its anchor row (the argmax at T=0).
    assert int(num_sampled[1].item()) == 1
    assert sampled[1, 0].item() == int(torch.argmax(logits_all[1, 0]).item())
    assert int(out.num_rejected[1].item()) == 0
    # Tree request: unchanged full traversal, matching the greedy reference.
    out_ref, nacc_ref = tree_verify_reference(
        logits_all[0], token_ids[0, :2], parents[0, :2], greedy=True
    )
    n = int(num_sampled[0].item())
    assert n - 1 == nacc_ref
    assert sampled[0, :n].tolist() == out_ref
    assert int(out.num_rejected[0].item()) == int(cu_np[1] - cu_np[0]) - n


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_df2t_production_pipeline_matches_reference():
    """Pins the PRODUCTION df2t verify pipeline against the oracle.

    Real serving routes the verify block through ``SamplingStates`` kernels
    (apply_temperature, apply_min_p, apply_top_k_top_p) before the TreeVerifier
    softmaxes every row. This test drives exactly those kernels with NON-default
    per-request temperature/top-k/top-p/min_p (incl. an exact top-k boundary
    tie) and compares the resulting per-token distribution to the oracle's
    processed convention: a dropped or wrongly ordered temperature/top-k/top-p
    stage in the kernel path (which the runner parity tests' keep-all default
    would mask) fails the support or argmax equality here.
    """
    from vllm.v1.worker.gpu.sample.states import SamplingStates

    device = torch.device("cuda")
    num_reqs, vocab = 8, 128
    params: list[dict[str, float | int | None]] = [
        dict(temperature=0.8, top_k=4, top_p=0.9, min_p=0.0),
        dict(temperature=1.0, top_k=6, top_p=0.8, min_p=0.0),
        dict(temperature=0.7, top_k=3, top_p=None, min_p=0.0),
        dict(temperature=1.1, top_k=None, top_p=0.95, min_p=0.0),
        dict(temperature=0.9, top_k=2, top_p=0.9, min_p=0.05),
        dict(temperature=1.0, top_k=None, top_p=None, min_p=0.0),
        dict(temperature=0.5, top_k=8, top_p=0.6, min_p=0.1),
        dict(temperature=1.0, top_k=5, top_p=0.9, min_p=0.0),
    ]
    torch.manual_seed(20260928)
    raw = torch.randn(num_reqs, vocab, device=device)
    # Row 0: exact tie at the top-k=4 boundary (two tokens at rank 4/5), so the
    # kernel must keep BOTH boundary tokens or the support check fails.
    row0 = torch.full((vocab,), -5.0, device=device)
    row0[0], row0[1], row0[2] = 8.0, 7.0, 6.0
    row0[3], row0[4] = 5.0, 5.0
    raw[0] = row0

    ss = SamplingStates(num_reqs, vocab)
    for r, p in enumerate(params):
        ss.temperature.np[r] = p["temperature"]
        ss.top_k.np[r] = p["top_k"] if p["top_k"] else vocab
        ss.top_p.np[r] = p["top_p"] if p["top_p"] is not None else 1.0
        ss.min_p.np[r] = p["min_p"]
        ss.seeds_set[r] = False
    ss.temperature.copy_to_uva()
    ss.top_k.copy_to_uva()
    ss.top_p.copy_to_uva()
    ss.min_p.copy_to_uva()

    logits = raw.clone()
    exp_map = torch.arange(num_reqs, dtype=torch.int32, device=device)
    idx_np = torch.arange(num_reqs).numpy()
    ss.apply_temperature(logits, exp_map, idx_np)
    ss.apply_min_p(logits, exp_map, idx_np)
    processed = ss.apply_top_k_top_p(logits, exp_map, idx_np)

    def _expected_minp(row, temperature, top_k, top_p, min_p):
        # Production order (temperature -> min_p -> top-k -> top-p), using the
        # oracle convention for each stage. With min_p == 0 this reduces to the
        # oracle's processed distribution.
        temperature = float(temperature) if temperature is not None else None
        top_k = int(top_k) if top_k is not None else None
        top_p = float(top_p) if top_p is not None else None
        p = _ref_processed(row, temperature, top_k, top_p)
        if min_p and 0.0 < float(min_p) <= 1.0:
            keep = p >= float(min_p) * float(p.max())
            p = torch.where(keep, p, torch.zeros_like(p))
            if bool(keep.any()):
                p = p / p.sum()
        return p

    for r, p in enumerate(params):
        expected = _expected_minp(
            raw[r],
            p["temperature"],
            p["top_k"],
            p["top_p"],
            p["min_p"],
        ).to(torch.float64)
        actual = torch.softmax(processed[r].to(torch.float64), dim=-1)
        support_ok = bool(((actual > 1e-9) == (expected > 1e-9)).all().item())
        assert support_ok, f"row {r}: kept set diverges from the reference"
        assert int(torch.argmax(actual)) == int(torch.argmax(expected)), (
            f"row {r}: argmax diverges from the reference"
        )
        max_diff = float((actual - expected).abs().max())
        assert max_diff < 1e-5, f"row {r}: probability diff {max_diff}"


def _df2t_compaction_ib(num_reqs, trees):
    """An InputBatch-shaped stand-in exposing the fields the compaction wiring
    reads (num_reqs, padding, trees, idx_mapping, idx_mapping_np)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        num_reqs=num_reqs,
        num_reqs_after_padding=num_reqs,
        trees=trees,
        idx_mapping=torch.arange(num_reqs, dtype=torch.int32, device="cuda"),
        idx_mapping_np=np.arange(num_reqs),
    )


def _df2t_compaction_phys(req_bt, c, max_delta, block_size):
    """The runner's block-slot -> physical mapping: ``phys[i]`` is the physical
    slot for block-slot ``c + i``."""
    slots = c + torch.arange(max_delta + 1, device="cuda")
    return req_bt[slots // block_size] * block_size + slots % block_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_model_runner_df2t_compaction_wiring_moves_per_group():
    """``_compact_df2t_accepted_kv`` (the real runner method) relocates each
    accepted node's KV from its scattered ``C + 1 + arena`` block-slot to the
    canonical ``C + depth`` block-slot, through the per-group block tables,
    and zeroes the stale scattered slots.

    Covers: block-slot math over ``num_computed_tokens``, per-group ``phys``
    construction, the ``C + 1 + arena -> C + depth`` mapping, the already
    canonical (no-op) skip, and the ``UniformTypeKVCacheSpecs`` group spec
    (block_size access on a uniform KV group)."""
    from types import SimpleNamespace

    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        UniformTypeKVCacheSpecs,
    )
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    device = torch.device("cuda")
    num_reqs = 3
    block_size = 4

    # Request 0: C=6, accepts a chain of 2 nodes (arenas 1, 2 at depths 1, 4),
    # so both relocate to non-aliased canonical slots (scatter/dst disjoint).
    # Request 1: C=2, single accepted node, arena 2 depth 3 (canonical no-op).
    # Request 2: num_sampled == 1 -> skipped by the wiring.
    num_sampled = torch.tensor([3, 2, 1], dtype=torch.int32, device=device)
    accepted_arenas = torch.full((num_reqs, 8), -1, dtype=torch.long, device=device)
    accepted_arenas[0, 1] = 1
    accepted_arenas[0, 2] = 2
    accepted_arenas[1, 1] = 2
    trees = [
        SimpleNamespace(
            depths=torch.tensor([0, 1, 4, 9, 9], dtype=torch.long, device=device)
        ),
        SimpleNamespace(
            depths=torch.tensor([0, 5, 3, 9, 9], dtype=torch.long, device=device)
        ),
        None,
    ]
    num_computed = torch.tensor([6, 2, 0], dtype=torch.int32, device=device)

    # Group 0 uses a real UniformTypeKVCacheSpecs (R3: block_size must read
    # off a uniform-type group without unwrapping); group 1 a plain spec.
    uniform0 = UniformTypeKVCacheSpecs.from_specs(
        {
            "l.0": FullAttentionSpec(
                block_size=block_size, num_kv_heads=2, head_size=3, dtype="float32"
            ),
            "l.1": FullAttentionSpec(
                block_size=block_size, num_kv_heads=2, head_size=3, dtype="float32"
            ),
        }
    )
    assert uniform0 is not None
    groups = [
        SimpleNamespace(
            layer_names=["l.0", "l.1"],
            kv_cache_spec=uniform0,
        ),
        SimpleNamespace(
            layer_names=["l.20"],
            kv_cache_spec=SimpleNamespace(block_size=block_size),
        ),
    ]
    layer_specs = [
        [("l.0", 2, 3), ("l.1", 2, 3)],
        [("l.20", 1, 5)],
    ]
    views_by_name = {}
    storages = {}
    for _, specs in enumerate(layer_specs):
        for name, h, d in specs:
            views_by_name[name], storages[name] = _make_slot_layer(256, h, d, device)

    block_tables = [
        torch.tensor(
            [[10, 30, 40], [20, 50, 0], [1, 1, 1]], dtype=torch.long, device=device
        ),
        torch.tensor(
            [[1, 2, 3], [11, 12, 0], [1, 1, 1]], dtype=torch.long, device=device
        ),
    ]

    def _fill(distinct, h, d):
        return torch.full((h, d), distinct, dtype=torch.float32, device=device)

    # Request 0 scattered sources (per group) hold the accepted nodes' KV; the
    # request 1 canonical slot holds KV that must be left untouched.
    r0_g0 = _df2t_compaction_phys(block_tables[0][0], 6, 4, block_size)
    # slots 6..10 -> phys [122, 123, 160, 161, 162]
    assert r0_g0.tolist() == [122, 123, 160, 161, 162]
    r0_g1 = _df2t_compaction_phys(block_tables[1][0], 6, 4, block_size)
    assert r0_g1.tolist() == [10, 11, 12, 13, 14]
    r1_g0 = _df2t_compaction_phys(block_tables[0][1], 2, 3, block_size)
    r1_g1 = _df2t_compaction_phys(block_tables[1][1], 2, 3, block_size)
    # Request 1 canonical slot is the phys index for depth 3 (index 3).
    canon0 = int(r1_g0[3].item())
    canon1 = int(r1_g1[3].item())
    assert canon0 == 201 and canon1 == 49

    for name, h, d in [("l.0", 2, 3), ("l.1", 2, 3)]:
        storages[name][0, int(r0_g0[2])] = _fill(1, h, d)  # node1 scatter
        storages[name][0, int(r0_g0[3])] = _fill(2, h, d)  # node2 scatter
        storages[name][0, canon0] = _fill(9, h, d)  # canonical no-op request
    storages["l.20"][0, int(r0_g1[2])] = _fill(1, 1, 5)
    storages["l.20"][0, int(r0_g1[3])] = _fill(2, 1, 5)
    storages["l.20"][0, canon1] = _fill(9, 1, 5)

    fake_self = SimpleNamespace(
        tree_verifier=SimpleNamespace(
            num_sampled=num_sampled, accepted_arenas=accepted_arenas
        ),
        _kv_cache_views_by_name=views_by_name,
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        block_tables=_FakeBlockTables(block_tables),
        req_states=SimpleNamespace(
            num_computed_tokens=SimpleNamespace(gpu=num_computed)
        ),
        device=device,
    )
    ib = _df2t_compaction_ib(num_reqs, trees)

    GPUModelRunner._compact_df2t_accepted_kv(fake_self, ib)

    # Request 0 / group 0: node1 scatter 160 -> canonical 123 and node2 scatter
    # 161 -> canonical 162; both scattered slots come back zeroed.
    for name in ("l.0", "l.1"):
        assert torch.equal(
            storages[name][0, int(r0_g0[1])], _fill(1, 2, 3)
        )  # dst depth1
        assert torch.equal(
            storages[name][0, int(r0_g0[4])], _fill(2, 2, 3)
        )  # dst depth4
        assert torch.equal(
            storages[name][0, int(r0_g0[2])], torch.zeros((2, 3), device=device)
        )
        assert torch.equal(
            storages[name][0, int(r0_g0[3])], torch.zeros((2, 3), device=device)
        )

    # Request 0 / group 1 uses its OWN block table: node1 12 -> 11, node2
    # 13 -> 14.
    assert torch.equal(storages["l.20"][0, int(r0_g1[1])], _fill(1, 1, 5))
    assert torch.equal(storages["l.20"][0, int(r0_g1[4])], _fill(2, 1, 5))
    assert torch.equal(
        storages["l.20"][0, int(r0_g1[2])], torch.zeros((1, 5), device=device)
    )
    assert torch.equal(
        storages["l.20"][0, int(r0_g1[3])], torch.zeros((1, 5), device=device)
    )

    # Request 1: already canonical (canonical slot == scattered slot), so the
    # KV there is left byte-for-byte untouched.
    for name in ("l.0", "l.1"):
        assert torch.equal(storages[name][0, canon0], _fill(9, 2, 3))
    assert torch.equal(storages["l.20"][0, canon1], _fill(9, 1, 5))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_model_runner_df2t_compaction_wiring_noop_cases():
    """The wiring is a no-op (no KV written) when no request accepted > 1
    node, when the batch has no trees, or when the runner has no KV views."""
    from types import SimpleNamespace

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    device = torch.device("cuda")
    num_reqs = 2
    view, storage = _make_slot_layer(16, 2, 4, device)
    storage.fill_(3.5)
    snapshot = storage.clone()
    block_table = torch.zeros(num_reqs, 1, dtype=torch.long, device=device)

    def _run(verifier, trees):
        fake_self = SimpleNamespace(
            tree_verifier=verifier,
            _kv_cache_views_by_name={"layer.0": view},
            kv_cache_config=SimpleNamespace(
                kv_cache_groups=[
                    SimpleNamespace(
                        layer_names=["layer.0"],
                        kv_cache_spec=SimpleNamespace(block_size=16),
                    )
                ]
            ),
            block_tables=_FakeBlockTables([block_table]),
            req_states=SimpleNamespace(
                num_computed_tokens=SimpleNamespace(
                    gpu=torch.zeros(num_reqs, dtype=torch.int32, device=device)
                )
            ),
            device=device,
        )
        ib = _df2t_compaction_ib(num_reqs, trees)
        GPUModelRunner._compact_df2t_accepted_kv(fake_self, ib)

    # Zero-accepted: num_sampled all <= 1 -> nothing written.
    _run(
        SimpleNamespace(
            num_sampled=torch.tensor([1, 1], dtype=torch.int32, device=device),
            accepted_arenas=torch.zeros(num_reqs, 8, dtype=torch.long, device=device),
        ),
        [SimpleNamespace(depths=torch.zeros(4, dtype=torch.long, device=device))]
        * num_reqs,
    )
    assert torch.equal(storage, snapshot)

    # Accepted nodes but every tree is None -> no KV group touched.
    _run(
        SimpleNamespace(
            num_sampled=torch.tensor([3, 1], dtype=torch.int32, device=device),
            accepted_arenas=torch.zeros(num_reqs, 8, dtype=torch.long, device=device),
        ),
        [None, None],
    )
    assert torch.equal(storage, snapshot)

    # No KV views at all -> the runner returns before touching anything.
    fake_self = SimpleNamespace(
        tree_verifier=SimpleNamespace(
            num_sampled=torch.tensor([3, 3], dtype=torch.int32, device=device),
            accepted_arenas=torch.zeros(num_reqs, 8, dtype=torch.long, device=device),
        ),
        _kv_cache_views_by_name=None,
        block_tables=_FakeBlockTables([block_table]),
        req_states=SimpleNamespace(
            num_computed_tokens=SimpleNamespace(
                gpu=torch.zeros(num_reqs, dtype=torch.int32, device=device)
            )
        ),
        device=device,
    )
    ib_no_attrs = _df2t_compaction_ib(num_reqs, [None, None])
    ib_no_attrs.trees = [
        SimpleNamespace(depths=torch.zeros(4, dtype=torch.long, device=device))
    ] * num_reqs
    GPUModelRunner._compact_df2t_accepted_kv(fake_self, ib_no_attrs)
    assert torch.equal(storage, snapshot)
