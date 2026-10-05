# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed regression: the df2t per-step verify path must be host-sync-free.

M1's per-step cost decomposition is only trustworthy if verifying a tree (the
worker's per-step verify path: rebuilding the tree avoidably H2D, laying out the
verify block, running the tree verifier and routing the accepted KV through the
compactor) never forces a device->host scalar round-trip. Those round-trips
(torch ``.item()``/``.tolist()`` on CUDA tensors and explicit synchronize calls)
erased the async overlap. This test instruments those exact call sites and runs
the per-step path end-to-end (layout rewrite -> verifier -> verified-contract
assembly) asserting zero such reads appear.

The one remaining boundary cost is the worker->scheduler draft-tree carry,
which the scheduler boundary fundamentally requires in host form: it is
excluded here (it lives in ``DraftTokensHandler.set_draft_tokens`` /
``get_draft_tokens``, not the per-step verify path this file drives).
"""

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    prepare_tree_inputs,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.states import SamplingStates
from vllm.v1.worker.gpu.spec_decode.df2t.tree_metadata import TreeBatch, TreeDraft
from vllm.v1.worker.gpu.spec_decode.df2t.tree_verifier import TreeVerifier


def _tree_on(tokens, parents, depths, device) -> TreeDraft:
    b = len(tokens)
    return TreeDraft(
        token_ids=torch.tensor(tokens, dtype=torch.long, device=device),
        depths=torch.tensor(depths, dtype=torch.long, device=device),
        parents=torch.tensor(parents, dtype=torch.long, device=device),
        arena=torch.arange(b, dtype=torch.long, device=device),
        words=torch.zeros(b, dtype=torch.long, device=device),
        num_nodes=torch.tensor([b], dtype=torch.int32, device=device),
    )


class _CountingSync:
    """Context manager counting CUDA-aware host round-trips and sync calls: any
    ``.item()``/``.tolist()``/``.cpu()``/``.numpy()`` invoked on (or returning
    from) a CUDA tensor, or an explicit
    ``torch.cuda.synchronize``/``torch.accelerator.synchronize``, increments a
    counter so the per-step path can fail closed if one reappears. CPU tensors
    are not counted (their reads are free host memory)."""

    def __enter__(self):
        self.orig_item = torch.Tensor.item
        self.orig_tolist = torch.Tensor.tolist
        self.orig_cpu = torch.Tensor.cpu
        self.orig_numpy = torch.Tensor.numpy
        self.orig_sync = torch.cuda.synchronize
        self.orig_accel_sync = torch.accelerator.synchronize
        orig_item, orig_tolist = self.orig_item, self.orig_tolist
        orig_cpu, orig_numpy = self.orig_cpu, self.orig_numpy
        orig_sync, orig_accel_sync = self.orig_sync, self.orig_accel_sync
        counts = {"item": 0, "tolist": 0, "cpu": 0, "numpy": 0, "synchronize": 0}
        self.counts = counts

        def item(self, *a, **k):
            if self.is_cuda:
                counts["item"] += 1
            return orig_item(self, *a, **k)

        def tolist(self, *a, **k):
            if self.is_cuda:
                counts["tolist"] += 1
            return orig_tolist(self, *a, **k)

        def cpu(self, *a, **k):
            if self.is_cuda:
                counts["cpu"] += 1
            return orig_cpu(self, *a, **k)

        def numpy(self, *a, **k):
            if self.is_cuda:
                counts["numpy"] += 1
            return orig_numpy(self, *a, **k)

        def sync(*a, **k):
            counts["synchronize"] += 1
            return orig_sync(*a, **k)

        def accel_sync(*a, **k):
            counts["synchronize"] += 1
            return orig_accel_sync(*a, **k)

        torch.Tensor.item = item
        torch.Tensor.tolist = tolist
        torch.Tensor.cpu = cpu
        torch.Tensor.numpy = numpy
        torch.cuda.synchronize = sync
        torch.accelerator.synchronize = accel_sync
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.Tensor.item = self.orig_item
        torch.Tensor.tolist = self.orig_tolist
        torch.Tensor.cpu = self.orig_cpu
        torch.Tensor.numpy = self.orig_numpy
        torch.cuda.synchronize = self.orig_sync
        torch.accelerator.synchronize = self.orig_accel_sync
        return False


def _make_slot_layer(num_slots, num_heads, head_dim, device="cuda"):
    """Engine-style LBNHC per-layer view: ``(view, storage)``.

    The view is the non-contiguous transpose ``[1, num_heads, num_slots,
    head_dim]`` (strides ``(num_slots * num_heads * head_dim, head_dim,
    num_heads * head_dim, 1)``), exactly what ``create_kv_cache_views`` yields
    under LBNHC (the compactor requires per-slot-contiguous layers).
    """
    storage = torch.zeros(
        1, num_slots, num_heads, head_dim, dtype=torch.float32, device=device
    )
    return storage.transpose(1, 2), storage


class _FakeBlockTables:
    """Stand-in for ``BlockTables.gather_block_tables`` (one group)."""

    def __init__(self, table):
        self._table = table

    def gather_block_tables(self, idx_mapping, num_reqs_padded=0):
        del idx_mapping
        return [self._table[:num_reqs_padded]]


def _fake_runner(device, tree_verifier, batch_size, vocab):
    """Minimal runner surface driving ``_df2t_tree_verify`` AND the real
    accepted-KV compactor (single LBNHC group) so both device paths run under
    the host-sync counter."""
    ss = SamplingStates(batch_size, vocab)
    ss.temperature.np[:] = 0.0
    ss.temperature.copy_to_uva()

    def apply_sampling_params(logits, *args, **kwargs):
        return logits

    kv_view, _kv_storage = _make_slot_layer(32, 2, 4, device)
    block_sizes = 16
    num_blocks = 8
    block_table = torch.zeros(batch_size, num_blocks, dtype=torch.long, device=device)
    fake = SimpleNamespace(
        tree_verifier=tree_verifier,
        sampler=SimpleNamespace(
            sampling_states=ss,
            req_states=SimpleNamespace(
                prefill_len=SimpleNamespace(
                    gpu=torch.zeros(batch_size, dtype=torch.int32, device=device)
                )
            ),
            compute_nans=False,
            apply_sampling_params=apply_sampling_params,
        ),
        device=device,
        max_num_reqs=batch_size,
        _kv_cache_views_by_name={"layer.0": kv_view},
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["layer.0"],
                    kv_cache_spec=SimpleNamespace(block_size=block_sizes),
                )
            ]
        ),
        block_tables=_FakeBlockTables(block_table),
        req_states=SimpleNamespace(
            num_computed_tokens=SimpleNamespace(
                gpu=torch.zeros(batch_size, dtype=torch.int32, device=device)
            )
        ),
    )
    fake._compact_df2t_accepted_kv = MethodType(
        GPUModelRunner._compact_df2t_accepted_kv, fake
    )
    fake._df2t_chain_width = MethodType(GPUModelRunner._df2t_chain_width, fake)
    fake._df2t_finish_verify = MethodType(GPUModelRunner._df2t_finish_verify, fake)
    return fake


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("uniform", [False, True])
def test_df2t_per_step_verify_path_is_host_sync_free(uniform):
    """The per-step verify path (tree rebuild -> layout rewrite -> verifier ->
    verifier-output assembly) makes no CUDA scalar reads or synchronize calls.

    The runtime path this exercises is exactly what a decode step runs
    (`make_input_batch`'s ``prepare_tree_inputs`` and ``_df2t_tree_verify``);
    the instrumentation fail-closes if any ``.item()``/``.tolist()`` on a CUDA
    tensor, an explicit synchronize, or any other synchronizing CUDA op (e.g. a
    blocking scalar host->device write) reappears on that path. ``uniform``
    covers the same-width (uniform ``1 + B`` decode) batch, whose verify reads
    the logits and the batched trees in place.
    """
    device = torch.device("cuda")
    vocab = 64
    torch.manual_seed(3)

    # Request 0 a 2-deep tree, request 1 a chain, so the verifier processes a
    # real depth mix.
    second = (
        _tree_on([20, 21, 22], [-1, 0, 0], [1, 2, 2], device)
        if uniform
        else _tree_on([20, 21], [-1, 0], [1, 2], device)
    )
    trees = [_tree_on([10, 11, 12], [-1, 0, 1], [1, 2, 3], device), second]
    widths = [t.token_ids.shape[0] for t in trees]
    b_max = max(widths)
    num_reqs = len(trees)
    num_logits_per_req = [w + 1 for w in widths]
    total_logits = sum(num_logits_per_req)

    cu_np = np.zeros(num_reqs + 1, dtype=np.int32)
    cu_np[1:] = np.cumsum(num_logits_per_req)
    logits = torch.randn(total_logits, vocab, device=device)
    query_start = np.concatenate([[0], np.cumsum(num_logits_per_req)])
    qsl = torch.tensor(query_start, dtype=torch.int32, device=device)
    positions = torch.arange(total_logits, dtype=torch.int64, device=device)
    slot_positions = positions.clone()
    input_ids = torch.zeros(total_logits, dtype=torch.int32, device=device)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=device)
    last_sampled = torch.full((num_reqs,), 7, dtype=torch.int32, device=device)
    seq_lens = torch.ones(num_reqs, dtype=torch.int32, device=device) * 100
    prefill_len = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    cu = torch.tensor(cu_np, dtype=torch.int32, device=device)
    logits_indices = torch.zeros(total_logits, dtype=torch.int64, device=device)

    cu_np_ib = cu_np

    # The device tree store as the drafter leaves it: one packed row per
    # request (tokens | depths | parents | words | num_nodes), inert tails.
    store = torch.zeros(num_reqs, 4 * b_max + 1, dtype=torch.long, device=device)
    store[:, 2 * b_max : 3 * b_max] = -2
    for r, t in enumerate(trees):
        n = widths[r]
        store[r, :n] = t.token_ids
        store[r, b_max : b_max + n] = t.depths
        store[r, 2 * b_max : 2 * b_max + n] = t.parents
        store[r, 3 * b_max : 3 * b_max + n] = t.words
        store[r, 4 * b_max] = n
    widths_np = np.asarray(widths)
    tv = TreeVerifier(64, b_max, vocab, device)
    fake = _fake_runner(device, tv, num_reqs, vocab)
    batches = InputBuffers(num_reqs, total_logits, device)
    ib = InputBatch.make_dummy(num_reqs, total_logits, batches, is_padding=True)
    ib.cu_num_logits = cu
    ib.cu_num_logits_np = cu_np_ib
    ib.seq_lens = seq_lens

    def step():
        tree_batch = TreeBatch.from_store(store, idx_mapping, widths_np)
        assert (tree_batch.uniform_width is not None) == uniform
        prepare_tree_inputs(
            input_ids,
            positions,
            slot_positions,
            idx_mapping,
            last_sampled,
            qsl,
            seq_lens,
            prefill_len,
            cu,
            logits_indices,
            tree_batch.drafts,
            tree_batch=(tree_batch.token_ids, tree_batch.depths, tree_batch.widths),
        )
        ib.trees = tree_batch.drafts
        ib.tree_batch = tree_batch
        return GPUModelRunner._df2t_tree_verify(fake, logits, ib)

    # One warm-up step builds the compactor (a one-time init), so the counted
    # step below is the steady-state per-step path.
    step()
    torch.accelerator.synchronize()
    with _CountingSync() as cnt:
        torch.cuda.set_sync_debug_mode("error")
        try:
            out = step()
        finally:
            torch.cuda.set_sync_debug_mode("default")
        assert out.sampled_token_ids.shape == (num_reqs, b_max + 1)

    assert cnt.counts == {
        "item": 0,
        "tolist": 0,
        "cpu": 0,
        "numpy": 0,
        "synchronize": 0,
    }, (
        "the df2t per-step verify path must make no CUDA scalar reads, "
        "device->host copies or synchronize calls; got {cnt.counts}"
    )
    # Test-side reads after the counting region; requires a live verification.
    for r in range(num_reqs):
        assert int(out.num_sampled[r].item()) >= 1
    # The accepted-KV compactor's device path actually ran under the counter
    # (built by the fallback; not skipped for lack of KV views), so its
    # host-sync-freeness is part of this contract.
    assert getattr(fake, "_df2t_compaction", None) is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_df2t_verify_layout_rewrite_is_host_sync_free():
    """``prepare_tree_inputs`` (the GPU path) makes no CUDA scalar reads."""
    device = torch.device("cuda")
    trees = [
        _tree_on([10, 11, 12], [-1, 0, 1], [1, 2, 3], device),
        None,
        _tree_on([20], [-1], [1], device),
    ]
    num_reqs = len(trees)
    num_logits = [t.token_ids.shape[0] + 1 if t is not None else 1 for t in trees]
    total = sum(num_logits)
    qsl = torch.tensor(
        [0] + list(np.cumsum(num_logits)), dtype=torch.int32, device=device
    )
    cum = np.cumsum([n - 1 for n in num_logits])
    cu = torch.tensor([0] + list(cum), dtype=torch.int32, device=device)
    positions = torch.arange(total, dtype=torch.int64, device=device)
    slot = positions.clone()
    input_ids = torch.zeros(total, dtype=torch.int32, device=device)
    idx = torch.arange(num_reqs, dtype=torch.int32, device=device)
    last = torch.full((num_reqs,), 3, dtype=torch.int32, device=device)
    seq_lens = torch.ones(num_reqs, dtype=torch.int32, device=device) * 50
    prefill = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    li = torch.zeros(total, dtype=torch.int64, device=device)
    with _CountingSync() as cnt:
        prepare_tree_inputs(
            input_ids,
            positions,
            slot,
            idx,
            last,
            qsl,
            seq_lens,
            prefill,
            cu,
            li,
            trees,
        )
    assert cnt.counts == {
        "item": 0,
        "tolist": 0,
        "cpu": 0,
        "numpy": 0,
        "synchronize": 0,
    }, f"prepare_tree_inputs made CUDA host reads: {cnt.counts}"
