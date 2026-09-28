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
