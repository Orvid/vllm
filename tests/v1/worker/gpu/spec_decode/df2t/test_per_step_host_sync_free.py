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
from vllm.v1.worker.gpu.spec_decode.df2t.tree_metadata import TreeDraft
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


def _counting_sync():
    """Return a context manager counting CUDA scalar reads + sync calls."""


class _CountingSync:
    """Context manager counting CUDA-aware scalar reads and sync calls: any
    ``.item()``/``.tolist()`` invoked on a CUDA tensor, or an explicit
    ``torch.cuda.synchronize``/``torch.accelerator.synchronize``, increments a
    counter so the per-step path can fail closed if one reappears."""

    def __enter__(self):
        self.orig_item = torch.Tensor.item
        self.orig_tolist = torch.Tensor.tolist
        self.orig_sync = torch.cuda.synchronize
        self.orig_accel_sync = torch.accelerator.synchronize
        orig_item, orig_tolist = self.orig_item, self.orig_tolist
        orig_sync, orig_accel_sync = self.orig_sync, self.orig_accel_sync
        counts = {"item": 0, "tolist": 0, "synchronize": 0}
        self.counts = counts

        def item(self, *a, **k):
            if self.is_cuda:
                counts["item"] += 1
            return orig_item(self, *a, **k)

        def tolist(self, *a, **k):
            if self.is_cuda:
                counts["tolist"] += 1
            return orig_tolist(self, *a, **k)

        def sync(*a, **k):
            counts["synchronize"] += 1
            return orig_sync(*a, **k)

        def accel_sync(*a, **k):
            counts["synchronize"] += 1
            return orig_accel_sync(*a, **k)

        torch.Tensor.item = item
        torch.Tensor.tolist = tolist
        torch.cuda.synchronize = sync
        torch.accelerator.synchronize = accel_sync
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.Tensor.item = self.orig_item
        torch.Tensor.tolist = self.orig_tolist
        torch.cuda.synchronize = self.orig_sync
        torch.accelerator.synchronize = self.orig_accel_sync
        return False


def _fake_runner(device, tree_verifier, batch_size, vocab):
    """Minimal runner surface for :meth:`GPUModelRunner._df2t_tree_verify`."""
    ss = SamplingStates(batch_size, vocab)
    ss.temperature.np[:] = 0.0
    ss.temperature.copy_to_uva()

    def apply_sampling_params(logits, *args, **kwargs):
        return logits

    prefill_len = SimpleNamespace(
        gpu=torch.zeros(batch_size, dtype=torch.int32, device=device)
    )
    fake = SimpleNamespace(
        tree_verifier=tree_verifier,
        sampler=SimpleNamespace(
            sampling_states=ss,
            req_states=SimpleNamespace(prefill_len=prefill_len),
            compute_nans=False,
            apply_sampling_params=apply_sampling_params,
        ),
        device=device,
        # No KV views: the accepted-KV compactor no-ops its (host-sync-free)
        # gate early; compaction's device-only path is pinned by the compaction
        # tests instead.
        _kv_cache_views_by_name=None,
    )
    fake._compact_df2t_accepted_kv = MethodType(
        GPUModelRunner._compact_df2t_accepted_kv, fake
    )
    return fake


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_df2t_per_step_verify_path_is_host_sync_free():
    """The per-step verify path (tree rebuild -> layout rewrite -> verifier ->
    verifier-output assembly) makes no CUDA scalar reads or synchronize calls.

    The runtime path this exercises is exactly what a decode step runs
    (`make_input_batch`'s ``prepare_tree_inputs`` and ``_df2t_tree_verify``);
    the instrumentation fail-closes if any ``.item()``/``.tolist()`` on a CUDA
    tensor or an explicit synchronize reappears on that path.
    """
    device = torch.device("cuda")
    vocab = 64
    torch.manual_seed(3)

    # Two requests: request 0 a 2-deep tree, request 1 a 3-node chain, so the
    # verifier processes a real depth mix.
    trees = [
        _tree_on([10, 11, 12], [-1, 0, 1], [1, 2, 3], device),
        _tree_on([20, 21], [-1, 0], [1, 2], device),
    ]
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
    seq_lens = (torch.ones(num_reqs, dtype=torch.int32, device=device) * 100)
    prefill_len = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    cu = torch.tensor(cu_np, dtype=torch.int32, device=device)
    logits_indices = torch.zeros(total_logits, dtype=torch.int64, device=device)

    cu_np_ib = cu_np

    # Rebuilt trees exactly as the scheduler carry round-trip produces.
    stashed = [t.to_lists() for t in trees]
    rebuilt = TreeDraft.from_lists_batched(stashed, device)

    with _CountingSync() as cnt:
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
            rebuilt,
        )
        tv = TreeVerifier(64, b_max, vocab, device)
        fake = _fake_runner(device, tv, num_reqs, vocab)
        batches = InputBuffers(num_reqs, total_logits, device)
        ib = InputBatch.make_dummy(
            num_reqs, total_logits, batches, is_padding=True
        )
        ib.cu_num_logits = cu
        ib.cu_num_logits_np = cu_np_ib
        ib.seq_lens = seq_lens
        ib.trees = rebuilt
        out = GPUModelRunner._df2t_tree_verify(fake, logits, ib)
        assert out.sampled_token_ids.shape == (num_reqs, b_max + 1)

    assert cnt.counts == {"item": 0, "tolist": 0, "synchronize": 0}, (
        "the df2t per-step verify path must make no CUDA scalar reads or "
        f"synchronize calls; got {cnt.counts}"
    )
    # Test-side reads after the counting region; requires a live verification.
    for r in range(num_reqs):
        assert int(out.num_sampled[r].item()) >= 1


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
    seq_lens = (torch.ones(num_reqs, dtype=torch.int32, device=device) * 50)
    prefill = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    li = torch.zeros(total, dtype=torch.int64, device=device)
    with _CountingSync() as cnt:
        prepare_tree_inputs(
            input_ids, positions, slot, idx, last, qsl, seq_lens, prefill, cu, li,
            trees,
        )
    assert cnt.counts == {"item": 0, "tolist": 0, "synchronize": 0}, (
        f"prepare_tree_inputs made CUDA scalar reads: {cnt.counts}"
    )
