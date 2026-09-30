# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for compacting the accepted chain's KV into canonical slots.

Task 7 ships the KV-compaction primitive: after the tree verify accepts a
chain, accepted nodes' K/V still live at their scattered tree-block slots
``C + 1 + arena`` while inference reads context by ``position -> slot`` (slot
``C + depth``). These tests pin, on GPU with fake per-layer caches mirroring
the engine's logical ``[B, H, N, C]`` LBNHC view (physical storage
``[B, N, H, C]``), that compaction moves each accepted node's K/V to the
canonical slot, zeroes the stale scattered slots, handles multi-layer /
multi-node batches, treats ``num_accepted == 0`` as a no-op, and rejects
layouts where the flat slot-addressing would corrupt (e.g. LBHNC).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.df2t.compaction import compact_accepted_chain


def _make_layer(num_slots, num_heads, head_dim):
    """Engine-style LBNHC layer: (logical view, physical storage).

    The physical cache is laid out ``[1, num_slots, num_heads, head_dim]``
    (contiguous, ``[B, N, H, C]``); the per-layer view is the non-contiguous
    transpose ``[1, num_heads, num_slots, head_dim]`` (logical ``[B, H, N, C]``
    with strides ``(num_slots * num_heads * head_dim, head_dim,
    num_heads * head_dim, 1)``), exactly what ``create_kv_cache_views`` yields
    under LBNHC. Tests seed/read through ``storage`` and pass ``view`` to the
    primitive.
    """
    storage = torch.zeros(
        1, num_slots, num_heads, head_dim, dtype=torch.float32, device="cuda"
    )
    return storage.transpose(1, 2), storage


def _phys(slot_mapping, block_slots):
    return [int(slot_mapping[s].item()) for s in block_slots]


def _scatter_and_canon_slots(C, arenas, depths):
    scatter = [C + 1 + int(a) for a in arenas]
    canon = [C + int(d) for d in depths]
    return scatter, canon


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compaction_moves_accepted_chain_to_canonical_slots():
    """The key/value that was at the accepted scattered slots now lives at the
    canonical ``C + depth`` slots, through the block-slot -> physical mapping.
    """
    num_slots, H, D = 16, 4, 8
    C = 4
    arenas = [1, 5]
    depths = [1, 3]
    # A non-trivial (cyclic-shifted) mapping proves translation through
    # ``slot_mapping`` and not a direct block-slot index into the cache.
    slot_mapping = torch.roll(
        torch.arange(num_slots, dtype=torch.long, device="cuda"), 3
    )
    scatter_slot, canon_slot = _scatter_and_canon_slots(C, arenas, depths)
    scatter_phys = _phys(slot_mapping, scatter_slot)
    canon_phys = _phys(slot_mapping, canon_slot)
    assert len(set(scatter_phys + canon_phys)) == 2 * len(arenas)

    view, storage = _make_layer(num_slots, H, D)
    expected = [torch.full((H, D), float(i + 1), device="cuda") for i in arenas]
    for phys, val in zip(scatter_phys, expected):
        storage[0, phys] = val

    compact_accepted_chain(
        torch.tensor(arenas, dtype=torch.long, device="cuda"),
        torch.tensor(depths, dtype=torch.long, device="cuda"),
        slot_mapping,
        [view],
        C,
        num_accepted=len(arenas),
    )

    for phys, val in zip(canon_phys, expected):
        assert torch.equal(storage[0, phys], val)
    for phys in scatter_phys:
        assert torch.equal(storage[0, phys], torch.zeros((H, D), device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compaction_zeroes_stale_scattered_slots():
    """After compaction the accepted nodes' old scattered slots hold only
    zeros, so rejected sibling K/V is never readable through them."""
    num_slots, H, D = 12, 3, 6
    C = 2
    # Every accepted node actually moves (depth != 1 + arena) so each scattered
    # slot is a stale source that must come back zeroed. Depths are chosen so
    # no scattered slot coincides with another node's canonical slot (an
    # aliased source would be preserved, not zeroed -- see the aliasing tests).
    arenas = [0, 3, 6]
    depths = [2, 6, 8]
    slot_mapping = torch.roll(
        torch.arange(num_slots, dtype=torch.long, device="cuda"), 3
    )
    scatter_slot, _ = _scatter_and_canon_slots(C, arenas, depths)
    scatter_phys = _phys(slot_mapping, scatter_slot)
    _, canon_slot = _scatter_and_canon_slots(C, arenas, depths)
    canon_phys = _phys(slot_mapping, canon_slot)
    assert len(set(scatter_phys) & set(canon_phys)) == 0

    view, storage = _make_layer(num_slots, H, D)
    for phys in scatter_phys:
        storage[0, phys] = torch.full((H, D), 7.0, device="cuda")

    compact_accepted_chain(
        torch.tensor(arenas, dtype=torch.long, device="cuda"),
        torch.tensor(depths, dtype=torch.long, device="cuda"),
        slot_mapping,
        [view],
        C,
    )

    for phys in scatter_phys:
        assert torch.equal(storage[0, phys], torch.zeros((H, D), device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compaction_multi_layer_multi_node():
    """Copying happens per accepted node across every layer, with each layer
    carrying its own distinctive content."""
    num_slots, H = 16, 4
    C = 5
    arenas = [2, 7]
    depths = [1, 2]
    slot_mapping = torch.roll(
        torch.arange(num_slots, dtype=torch.long, device="cuda"), 3
    )
    scatter_slot, canon_slot = _scatter_and_canon_slots(C, arenas, depths)
    scatter_phys = _phys(slot_mapping, scatter_slot)
    canon_phys = _phys(slot_mapping, canon_slot)

    layers = [_make_layer(num_slots, H, D) for D in (8, 16)]
    for layer_idx, (_, storage) in enumerate(layers):
        fill = torch.full((H, storage.shape[-1]), float(layer_idx + 1), device="cuda")
        for phys in scatter_phys:
            storage[0, phys] = fill.clone()

    compact_accepted_chain(
        torch.tensor(arenas, dtype=torch.long, device="cuda"),
        torch.tensor(depths, dtype=torch.long, device="cuda"),
        slot_mapping,
        [view for view, _ in layers],
        C,
    )

    for layer_idx, (_, storage) in enumerate(layers):
        fill = torch.full((H, storage.shape[-1]), float(layer_idx + 1), device="cuda")
        for phys in canon_phys:
            assert torch.equal(storage[0, phys], fill)
        for phys in scatter_phys:
            assert torch.equal(storage[0, phys], torch.zeros_like(fill))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compaction_num_accepted_zero_is_noop():
    """With ``num_accepted == 0`` compaction returns cleanly and leaves the
    cache byte-for-byte unchanged."""
    num_slots, H, D = 8, 4, 8
    slot_mapping = torch.arange(num_slots, dtype=torch.long, device="cuda")
    view, storage = _make_layer(num_slots, H, D)
    storage.fill_(3.5)
    snapshot = storage.clone()

    compact_accepted_chain(
        torch.tensor([], dtype=torch.long, device="cuda"),
        torch.tensor([], dtype=torch.long, device="cuda"),
        slot_mapping,
        [view],
        C=1,
        num_accepted=0,
    )

    assert torch.equal(storage, snapshot)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compaction_preserves_node_already_at_canonical_slot():
    """An accepted node whose canonical slot equals its scattered slot (depth
    ``1 + arena``) must be left untouched, not zeroed."""
    num_slots, H, D = 10, 4, 8
    C = 5
    arenas = [2]
    depths = [3]  # C + 1 + 2 == C + 3: scattered and canonical coincide.
    slot_mapping = torch.arange(num_slots, dtype=torch.long, device="cuda")
    view, storage = _make_layer(num_slots, H, D)
    storage[0, C + 1 + arenas[0]] = torch.full((H, D), 9.0, device="cuda")

    compact_accepted_chain(
        torch.tensor(arenas, dtype=torch.long, device="cuda"),
        torch.tensor(depths, dtype=torch.long, device="cuda"),
        slot_mapping,
        [view],
        C,
    )

    assert torch.equal(
        storage[0, C + 1 + arenas[0]],
        torch.full((H, D), 9.0, device="cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compaction_slot_mapping_translation_is_used():
    """Physical and block slots differ under a shifted mapping, so a compaction
    that ignored ``slot_mapping`` would seed/read the wrong cache rows."""
    num_slots, H, D = 16, 2, 4
    C = 3
    arenas = [1]
    depths = [1]
    slot_mapping = torch.roll(
        torch.arange(num_slots, dtype=torch.long, device="cuda"), 3
    )
    scatter_slot, canon_slot = _scatter_and_canon_slots(C, arenas, depths)
    assert _phys(slot_mapping, scatter_slot) != scatter_slot
    assert _phys(slot_mapping, canon_slot) != canon_slot

    view, storage = _make_layer(num_slots, H, D)
    storage[0, _phys(slot_mapping, scatter_slot)[0]] = torch.full(
        (H, D), 5.0, device="cuda"
    )

    compact_accepted_chain(
        torch.tensor(arenas, dtype=torch.long, device="cuda"),
        torch.tensor(depths, dtype=torch.long, device="cuda"),
        slot_mapping,
        [view],
        C,
    )

    assert torch.equal(
        storage[0, _phys(slot_mapping, canon_slot)[0]],
        torch.full((H, D), 5.0, device="cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compaction_stride_guard_accepts_lbnc_rejects_lbhnc():
    """The stride guard accepts the engine's true LBNHC per-layer view and
    rejects a contiguous (LBHNC-order) view whose flat addressing would
    corrupt, instead of silently copying the wrong head-strided positions."""
    num_slots, H, D = 16, 4, 8
    C = 3
    arenas = [1]
    depths = [1]
    slot_mapping = torch.arange(num_slots, dtype=torch.long, device="cuda")
    scatter_slot, canon_slot = _scatter_and_canon_slots(C, arenas, depths)
    scatter_phys = _phys(slot_mapping, scatter_slot)
    canon_phys = _phys(slot_mapping, canon_slot)

    # A contiguous [1, H, N, C] tensor is the LBHNC memory order: head stride
    # is N*C, so slots are head-strided and flat slot addressing is invalid.
    lbhnc_view = torch.zeros(1, H, num_slots, D, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="per-slot-contiguous"):
        compact_accepted_chain(
            torch.tensor(arenas, dtype=torch.long, device="cuda"),
            torch.tensor(depths, dtype=torch.long, device="cuda"),
            slot_mapping,
            [lbhnc_view],
            C,
        )

    # The true LBNHC logical view (non-contiguous transpose) is accepted and
    # compacts correctly through its raw storage.
    view, storage = _make_layer(num_slots, H, D)
    storage[0, scatter_phys[0]] = torch.full((H, D), 6.0, device="cuda")

    compact_accepted_chain(
        torch.tensor(arenas, dtype=torch.long, device="cuda"),
        torch.tensor(depths, dtype=torch.long, device="cuda"),
        slot_mapping,
        [view],
        C,
    )

    assert torch.equal(
        storage[0, canon_phys[0]], torch.full((H, D), 6.0, device="cuda")
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compaction_aliased_source_is_not_zeroed():
    """An accepted node's scattered slot ``C + 1 + arena`` can equal another
    node's canonical slot ``C + depth`` (e.g. arenas [1, 2] / depths [1, 2]).
    Compaction must still move every node's KV to its canonical slot and only
    zero sources that are not any destination: zeroing this node's source would
    clobber the other node's relocated KV."""
    num_slots, H, D = 16, 2, 3
    C = 0
    arenas = [1, 2]
    depths = [1, 2]
    slot_mapping = torch.arange(num_slots, dtype=torch.long, device="cuda")
    scatter_slot, canon_slot = _scatter_and_canon_slots(C, arenas, depths)
    # scatter == [2, 3], canon == [1, 2]; node 2's canonical slot (2) is node
    # 1's scattered slot, so a naive "zero every source" would lose node 2.
    assert scatter_slot == [2, 3] and canon_slot == [1, 2]

    view, storage = _make_layer(num_slots, H, D)
    for i in range(2):
        storage[0, scatter_slot[i]] = torch.full((H, D), float(i + 1), device="cuda")

    compact_accepted_chain(
        torch.tensor(arenas, dtype=torch.long, device="cuda"),
        torch.tensor(depths, dtype=torch.long, device="cuda"),
        slot_mapping,
        [view],
        C,
    )

    # Both nodes live at their canonical slots; only the truly stale source
    # (slot 3) is zeroed, slot 2 is node 2's canonical KV, not a zero.
    assert torch.equal(storage[0, 1], torch.full((H, D), 1.0, device="cuda"))
    assert torch.equal(storage[0, 2], torch.full((H, D), 2.0, device="cuda"))
    assert torch.equal(storage[0, 3], torch.zeros((H, D), device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compaction_aliased_source_mapping_shifted():
    """The aliasing hazard under a NON-identity block-slot -> physical mapping:
    relocation stays correct through ``slot_mapping`` and the colliding source
    slot is preserved, not zeroed."""
    num_slots, H, D, C = 16, 2, 3, 2
    arenas = [1, 2]
    depths = [1, 2]
    slot_mapping = torch.roll(
        torch.arange(num_slots, dtype=torch.long, device="cuda"), 3
    )
    scatter_slot, canon_slot = _scatter_and_canon_slots(C, arenas, depths)
    scatter_phys = _phys(slot_mapping, scatter_slot)
    canon_phys = _phys(slot_mapping, canon_slot)
    assert len(set(scatter_phys + canon_phys)) == 3  # one aliased pair

    view, storage = _make_layer(num_slots, H, D)
    for i in range(2):
        storage[0, scatter_phys[i]] = torch.full((H, D), float(3 + i), device="cuda")

    compact_accepted_chain(
        torch.tensor(arenas, dtype=torch.long, device="cuda"),
        torch.tensor(depths, dtype=torch.long, device="cuda"),
        slot_mapping,
        [view],
        C,
    )

    for i in range(2):
        assert torch.equal(
            storage[0, canon_phys[i]],
            torch.full((H, D), float(3 + i), device="cuda"),
        )
    stale = [p for p in scatter_phys if p not in canon_phys]
    for p in stale:
        assert torch.equal(storage[0, p], torch.zeros((H, D), device="cuda"))


def _make_batched_cache(group_specs, block_size, num_blocks, dtype=torch.float32):
    """Multi-group / multi-layer LBNHC cache sharing one raw backing.

    Each layer's view is ``as_strided`` over one ``buf`` exactly like
    ``create_kv_cache_views``: logical ``[num_blocks, heads, tokens_per_block,
    content]`` with per-slot-contiguous LBNHC strides. Returns ``(views,
    buf)`` where a physical slot ``s`` of a layer sits at flat element offset
    ``s * elems`` from the layer's base.
    """
    layers = [(name, h, d) for grp in group_specs for name, h, d in grp]
    total = sum(h * d for _, h, d in layers) * block_size * num_blocks
    buf = torch.zeros(total, dtype=dtype, device="cuda")
    views = {}
    offset = 0
    for name, h, d in layers:
        elems = h * d
        views[name] = torch.as_strided(
            buf,
            (num_blocks, h, block_size, d),
            (block_size * elems, d, elems, 1),
            storage_offset=offset,
        )
        offset += elems * block_size * num_blocks
    return views, buf


def _seed_kv(views, group_specs, block_size, num_slots):
    """Fill each layer's phys slots [0, num_slots) with a layer/slot-unique
    nonzero value, so a misplaced relocation (a moved slot's payload landing
    on the wrong physical slot) is detectable by ``torch.equal``."""
    layer_idx = 0
    for grp in group_specs:
        for name, h, d in grp:
            if name not in views:
                continue
            v = views[name]
            base = float(layer_idx * 100000)
            for s in range(num_slots):
                v[s // block_size, :, s % block_size, :].fill_(base + s + 1)
            layer_idx += 1


def _rand_chain_batch(R, W, num_blocks, seed):
    """Random realistic accepted chains.

    The verifier's accepted nodes are one tree path: distinct arenas with
    strictly increasing tree depths, so canonical slots ``C + depth`` are
    distinct and the gather->scatter result is deterministic (no two nodes
    write the same canonical slot). Existing non-path depth collisions (two
    accepted nodes sharing a depth) cannot come out of the verifier and would
    make "bit-identical to the reference" a racy, ill-defined assertion.
    """
    torch.manual_seed(seed)
    num_sampled = torch.randint(
        1, W + 1, (R,), dtype=torch.int32, device="cuda"
    )
    accepted_arenas = torch.full((R, W + 1), -1, dtype=torch.long, device="cuda")
    num_computed = torch.randint(0, num_blocks - 2, (R,), dtype=torch.int32,
                                 device="cuda")
    block_tables = torch.randint(
        0, num_blocks, (R, 6), dtype=torch.long, device="cuda"
    )
    trees = []
    for r in range(R):
        n = int(num_sampled[r].item()) - 1
        depths = torch.zeros(W, dtype=torch.long, device="cuda")
        arena_order = torch.randperm(W)
        for j in range(1, n + 1):
            arena = int(arena_order[j - 1].item())
            accepted_arenas[r, j] = arena
            depths[arena] = j
        trees.append(SimpleNamespace(depths=depths))
    return num_sampled, accepted_arenas, num_computed, block_tables, trees


def _old_compact_reference(
    views_by_name, group_specs, block_size, num_sampled, accepted_arenas,
    num_computed, block_tables, trees, R,
):
    for r in range(R):
        n_sampled = int(num_sampled[r].item())
        if n_sampled <= 1:
            continue
        n_acc = n_sampled - 1
        arenas = accepted_arenas[r, 1 : 1 + n_acc]
        depths = trees[r].depths[arenas.clamp(min=0)]
        c = int(num_computed[r].item())
        max_delta = max(
            int(1 + arenas.max().item()), int(depths.max().item())
        )
        slots = c + torch.arange(max_delta + 1, device="cuda")
        for grp in group_specs:
            layer_views = [views_by_name[n] for n, _, _ in grp if n in views_by_name]
            if not layer_views:
                continue
            req_bt = block_tables[r]
            phys = req_bt[slots // block_size] * block_size + slots % block_size
            compact_accepted_chain(
                arenas, depths, phys, layer_views, 0, num_accepted=n_acc
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_compaction_matches_reference_bit_identical():
    """``Df2tKVCompaction.compact`` (two fixed-shape all-layer launches) is
    bit-identical to the per-(request, group, layer) ``compact_accepted_chain``
    reference on a multi-group / multi-layer LBNHC cache, across randomized
    batches that exercise relocation, the already-canonical skip, and the
    aliasing protection."""
    from vllm.v1.worker.gpu.spec_decode.df2t.compaction import (
        Df2tKVCompaction,
    )

    R, W = 5, 7
    block_size, num_blocks = 16, 64
    group_specs = [
        [("l.2", 2, 3), ("l.8", 2, 3), ("l.14", 2, 3)],
        [("l.4", 1, 5), ("l.10", 1, 5)],
        [("l.1", 4, 4)],
    ]
    for seed in range(6):
        views, buf = _make_batched_cache(group_specs, block_size, num_blocks)
        num_sampled, accepted_arenas, num_computed, block_tables, trees = (
            _rand_chain_batch(R, W, num_blocks, seed)
        )
        groups = [
            SimpleNamespace(
                layer_names=[n for n, _, _ in grp],
                kv_cache_spec=SimpleNamespace(block_size=block_size),
            )
            for grp in group_specs
        ]
        num_slots = int(num_computed.max().item()) + W + 2
        _seed_kv(views, group_specs, block_size, num_slots)
        torch.cuda.synchronize()
        initial = buf.clone()
        _old_compact_reference(
            views,
            group_specs,
            block_size,
            num_sampled,
            accepted_arenas,
            num_computed,
            block_tables,
            trees,
            R,
        )
        torch.cuda.synchronize()
        reference = buf.clone()
        buf.copy_(initial)
        torch.cuda.synchronize()
        compactor = Df2tKVCompaction(
            R, W, views, groups, torch.device("cuda")
        )
        compactor.compact(
            num_sampled,
            accepted_arenas,
            trees,
            num_computed,
            [block_tables] * len(groups),
            R,
        )
        torch.cuda.synchronize()
        assert torch.equal(buf, reference), f"seed {seed} diverged"
        # A single (slab, L)-grid pair covers the whole batch, and slabs tile
        # at whole-request boundaries (a W-multiple).
        assert compactor._slab >= R * W
        assert compactor._slab % W == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_compaction_treeless_requests_untouched():
    """Requests without a tree (or with no accepted chain) must leave their KV
    byte-for-byte untouched even when the batch otherwise compacts."""
    from vllm.v1.worker.gpu.spec_decode.df2t.compaction import (
        Df2tKVCompaction,
    )

    R, W = 4, 6
    block_size, num_blocks = 16, 64
    group_specs = [[("l.0", 2, 3)], [("l.1", 1, 4)]]
    views, buf = _make_batched_cache(group_specs, block_size, num_blocks)
    groups = [
        SimpleNamespace(
            layer_names=[n for n, _, _ in grp],
            kv_cache_spec=SimpleNamespace(block_size=block_size),
        )
        for grp in group_specs
    ]
    buf.fill_(3.5)
    snapshot = buf.clone()
    torch.manual_seed(9)
    accepted_arenas = torch.full((R, W + 1), -1, dtype=torch.long, device="cuda")
    num_computed = torch.zeros(R, dtype=torch.int32, device="cuda")
    block_tables = torch.zeros(R, 4, dtype=torch.long, device="cuda")
    num_sampled = torch.tensor([1, 3, 1, 1], dtype=torch.int32, device="cuda")
    trees = [
        None,
        SimpleNamespace(depths=torch.arange(W, device="cuda")),
        None,
        None,
    ]
    compactor = Df2tKVCompaction(R, W, views, groups, torch.device("cuda"))
    compactor.compact(
        num_sampled,
        accepted_arenas,
        trees,
        num_computed,
        [block_tables] * len(groups),
        R,
    )
    torch.cuda.synchronize()
    assert torch.equal(buf, snapshot)


def _force_slab_budget(budget: int):
    """Context manager overriding Df2tKVCompaction's scratch-budget class attr."""
    from contextlib import contextmanager

    from vllm.v1.worker.gpu.spec_decode.df2t.compaction import (
        Df2tKVCompaction,
    )

    @contextmanager
    def _cm():
        prev = Df2tKVCompaction._ScratchBudgetBytes
        Df2tKVCompaction._ScratchBudgetBytes = budget
        try:
            yield
        finally:
            Df2tKVCompaction._ScratchBudgetBytes = prev

    return _cm()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_compaction_multi_slab_aliasing_at_boundary():
    """Critical regression: a batch spanning several slabs must keep every
    intra-request aliased pair exact. Node A's canonical slot equals node B's
    scattered slot; with non-request-aligned slab tiling the earlier slab's
    scatter overwrites the later slab's source before its gather (moves
    already-moved bytes). Slabs now tile at whole-request (W-multiple)
    boundaries, so each aliased pair stays in one slab and the gather ->
    scatter ordering holds; the result must be byte-for-byte the per-request
    reference."""
    from vllm.v1.worker.gpu.spec_decode.df2t.compaction import (
        Df2tKVCompaction,
    )

    R, W = 24, 4
    block_size, num_blocks = 16, 256
    nb = 16  # C up to 23*6 + 6 -> slot < 152 -> block index < 10.
    # max_el = 4 (2x2 / 1x4), L = 2 -> row_bytes = 2 * 4 * 4 = 32.
    group_specs = [[("l.0", 2, 2)], [("l.1", 1, 4)]]
    views, buf = _make_batched_cache(group_specs, block_size, num_blocks)
    groups = [
        SimpleNamespace(
            layer_names=[n for n, _, _ in grp],
            kv_cache_spec=SimpleNamespace(block_size=block_size),
        )
        for grp in group_specs
    ]
    # Distinct C per request keeps every request's slots disjoint.
    num_computed = torch.arange(R, dtype=torch.int32, device="cuda") * 6
    block_tables = (
        torch.arange(0, nb * block_size, block_size, device="cuda")
        .unsqueeze(0)
        .expand(R, nb)
        .contiguous()
    )
    num_sampled = torch.full((R,), 4, dtype=torch.int32, device="cuda")
    accepted_arenas = torch.full((R, W + 1), -1, dtype=torch.long, device="cuda")
    trees = []
    for r in range(R):
        depths = torch.zeros(W, dtype=torch.long, device="cuda")
        if r in (6, 13, 19):
            # Requests straddling the old non-aligned boundaries (26/52/78 in
            # the pre-fix tiling). A realistic chain (distinct arenas,
            # strictly increasing depths) where BOTH nodes are real movers and
            # A's canonical slot equals B's scattered slot: A (entry 24/52/76,
            # pre-fix earlier slab) scatters C+1 over B's source, then B (entry
            # 26/54/78, pre-fix later slab) gathers C+1 -- so the pre-fix
            # tiling reads A's payload for B. Under request-aligned slabs the
            # pair stays in one gather->scatter slab and both relocations are
            # exact.
            accepted_arenas[r, 1] = 1
            accepted_arenas[r, 2] = 2
            accepted_arenas[r, 3] = 0  # B: src C+1 == A's dst C+1
            depths[1] = 1
            depths[2] = 2
            depths[0] = 3
        else:
            accepted_arenas[r, 1] = 0
            accepted_arenas[r, 2] = 1
            depths[0] = 1
            depths[1] = 2
        trees.append(SimpleNamespace(depths=depths))

    num_slots = int(num_computed.max().item()) + W + 2
    _seed_kv(views, group_specs, block_size, num_slots)
    torch.cuda.synchronize()
    initial = buf.clone()
    _old_compact_reference(
        views, group_specs, block_size, num_sampled, accepted_arenas,
        num_computed, block_tables, trees, R,
    )
    torch.cuda.synchronize()
    reference = buf.clone()
    buf.copy_(initial)
    torch.cuda.synchronize()

    # L=2, max_el=4, itemsize=4 -> row_bytes=32; budget=26*32 -> raw slab 26
    # (< a=96, multi-slab) and NOT a W-multiple pre-fix.
    with _force_slab_budget(26 * 32):
        compactor = Df2tKVCompaction(R, W, views, groups, torch.device("cuda"))
        assert compactor._slab < R * W  # the batch really spans slabs
        assert compactor._slab % W == 0  # and tiles at request boundaries
        compactor.compact(
            num_sampled, accepted_arenas, trees, num_computed,
            [block_tables] * len(groups), R,
        )
    torch.cuda.synchronize()
    assert torch.equal(buf, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_batched_compaction_multi_slab_random_stress():
    """Randomized multi-slab stress: a batch spanning several request-aligned
    slabs must stay byte-for-byte identical to the per-request reference
    across seeds (relocation, already-canonical, and same-request aliasing
    included)."""
    from vllm.v1.worker.gpu.spec_decode.df2t.compaction import (
        Df2tKVCompaction,
    )

    R, W = 40, 5
    block_size, num_blocks = 16, 512
    nb = 24  # C up to 39*8 + 6 -> slot < 320 -> block index < 20.
    # max_el = 4 (2x2 / 1x4), L = 2 -> row_bytes = 32.
    group_specs = [[("l.0", 2, 2)], [("l.1", 1, 4)]]
    views, buf = _make_batched_cache(group_specs, block_size, num_blocks)
    groups = [
        SimpleNamespace(
            layer_names=[n for n, _, _ in grp],
            kv_cache_spec=SimpleNamespace(block_size=block_size),
        )
        for grp in group_specs
    ]
    block_tables = (
        torch.arange(0, nb * block_size, block_size, device="cuda")
        .unsqueeze(0)
        .expand(R, nb)
        .contiguous()
    )
    for seed in range(4):
        torch.manual_seed(seed)
        num_computed = torch.arange(R, dtype=torch.int32, device="cuda") * 8
        num_sampled = torch.randint(2, W + 1, (R,), dtype=torch.int32,
                                    device="cuda")
        accepted_arenas = torch.full((R, W + 1), -1, dtype=torch.long,
                                     device="cuda")
        trees = []
        for r in range(R):
            n = int(num_sampled[r].item()) - 1
            depths = torch.zeros(W, dtype=torch.long, device="cuda")
            # Realistic chain: distinct arenas, strictly increasing depths.
            arena_order = torch.randperm(W)
            for j in range(1, n + 1):
                arena = int(arena_order[j - 1].item())
                accepted_arenas[r, j] = arena
                depths[arena] = j
            trees.append(SimpleNamespace(depths=depths))
        _seed_kv(
            views, group_specs, block_size,
            int(num_computed.max().item()) + W + 2,
        )
        torch.cuda.synchronize()
        initial = buf.clone()
        _old_compact_reference(
            views, group_specs, block_size, num_sampled, accepted_arenas,
            num_computed, block_tables, trees, R,
        )
        torch.cuda.synchronize()
        reference = buf.clone()
        buf.copy_(initial)
        torch.cuda.synchronize()
        # L=2, max_el=4, itemsize=4 -> row_bytes=32; budget=17*32=544 ->
        # raw slab 17, W-aligned to 15 (multiple of 5) < a=200 (multi-slab).
        with _force_slab_budget(17 * 32):
            compactor = Df2tKVCompaction(
                R, W, views, groups, torch.device("cuda")
            )
            assert compactor._slab < R * W
            assert compactor._slab % W == 0
            compactor.compact(
                num_sampled, accepted_arenas, trees, num_computed,
                [block_tables] * len(groups), R,
            )
        torch.cuda.synchronize()
        assert torch.equal(buf, reference), f"seed {seed} diverged"
