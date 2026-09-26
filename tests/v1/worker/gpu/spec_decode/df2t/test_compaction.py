# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for compacting the accepted chain's KV into canonical slots.

Task 7 ships the KV-compaction primitive: after the tree verify accepts a
chain, accepted nodes' K/V still live at their scattered tree-block slots
``C + 1 + arena`` while inference reads context by ``position -> slot`` (slot
``C + depth``). These tests pin, on GPU with fake per-layer caches shaped
``[1, num_slots, num_heads, head_dim]``, that compaction moves each accepted
node's K/V to the canonical slot, zeroes the stale scattered slots, handles
multi-layer/multi-node batches, and treats ``num_accepted == 0`` as a no-op.
"""

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.df2t.compaction import compact_accepted_chain


def _cache(num_slots, num_heads, head_dim):
    return torch.zeros(
        1, num_slots, num_heads, head_dim, dtype=torch.float32, device="cuda"
    )


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

    cache = _cache(num_slots, H, D)
    expected = [torch.full((H, D), float(i + 1), device="cuda") for i in arenas]
    for phys, val in zip(scatter_phys, expected):
        cache[0, phys] = val

    compact_accepted_chain(
        torch.tensor(arenas, dtype=torch.long, device="cuda"),
        torch.tensor(depths, dtype=torch.long, device="cuda"),
        slot_mapping,
        [cache],
        C,
        num_accepted=len(arenas),
    )

    for phys, val in zip(canon_phys, expected):
        assert torch.equal(cache[0, phys], val)
    for phys in scatter_phys:
        assert torch.equal(cache[0, phys], torch.zeros((H, D), device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compaction_zeroes_stale_scattered_slots():
    """After compaction the accepted nodes' old scattered slots hold only
    zeros, so rejected sibling K/V is never readable through them."""
    num_slots, H, D = 12, 3, 6
    C = 2
    # Every accepted node actually moves (depth != 1 + arena) so each scattered
    # slot is a stale source that must come back zeroed.
    arenas = [0, 3, 6]
    depths = [2, 3, 4]
    slot_mapping = torch.roll(
        torch.arange(num_slots, dtype=torch.long, device="cuda"), 3
    )
    scatter_slot, _ = _scatter_and_canon_slots(C, arenas, depths)
    scatter_phys = _phys(slot_mapping, scatter_slot)

    cache = _cache(num_slots, H, D)
    for phys in scatter_phys:
        cache[0, phys] = torch.full((H, D), 7.0, device="cuda")

    compact_accepted_chain(
        torch.tensor(arenas, dtype=torch.long, device="cuda"),
        torch.tensor(depths, dtype=torch.long, device="cuda"),
        slot_mapping,
        [cache],
        C,
    )

    for phys in scatter_phys:
        assert torch.equal(cache[0, phys], torch.zeros((H, D), device="cuda"))


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

    caches = [_cache(num_slots, H, D) for D in (8, 16)]
    for layer, cache in enumerate(caches):
        for phys in scatter_phys:
            cache[0, phys] = torch.full(
                (H, cache.shape[-1]), float(layer + 1), device="cuda"
            )

    compact_accepted_chain(
        torch.tensor(arenas, dtype=torch.long, device="cuda"),
        torch.tensor(depths, dtype=torch.long, device="cuda"),
        slot_mapping,
        caches,
        C,
    )

    for layer, cache in enumerate(caches):
        fill = torch.full((H, cache.shape[-1]), float(layer + 1), device="cuda")
        for phys in canon_phys:
            assert torch.equal(cache[0, phys], fill)
        for phys in scatter_phys:
            assert torch.equal(cache[0, phys], torch.zeros_like(fill))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compaction_num_accepted_zero_is_noop():
    """With ``num_accepted == 0`` compaction returns cleanly and leaves the
    cache byte-for-byte unchanged."""
    num_slots, H, D = 8, 4, 8
    slot_mapping = torch.arange(num_slots, dtype=torch.long, device="cuda")
    cache = _cache(num_slots, H, D)
    cache.fill_(3.5)
    snapshot = cache.clone()

    compact_accepted_chain(
        torch.tensor([], dtype=torch.long, device="cuda"),
        torch.tensor([], dtype=torch.long, device="cuda"),
        slot_mapping,
        [cache],
        C=1,
        num_accepted=0,
    )

    assert torch.equal(cache, snapshot)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compaction_preserves_node_already_at_canonical_slot():
    """An accepted node whose canonical slot equals its scattered slot (depth
    ``1 + arena``) must be left untouched, not zeroed."""
    num_slots, H, D = 10, 4, 8
    C = 5
    arenas = [2]
    depths = [3]  # C + 1 + 2 == C + 3: scattered and canonical coincide.
    slot_mapping = torch.arange(num_slots, dtype=torch.long, device="cuda")
    cache = _cache(num_slots, H, D)
    cache[0, C + 1 + arenas[0]] = torch.full((H, D), 9.0, device="cuda")

    compact_accepted_chain(
        torch.tensor(arenas, dtype=torch.long, device="cuda"),
        torch.tensor(depths, dtype=torch.long, device="cuda"),
        slot_mapping,
        [cache],
        C,
    )

    assert torch.equal(
        cache[0, C + 1 + arenas[0]], torch.full((H, D), 9.0, device="cuda")
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

    cache = _cache(num_slots, H, D)
    cache[0, _phys(slot_mapping, scatter_slot)[0]] = torch.full(
        (H, D), 5.0, device="cuda"
    )

    compact_accepted_chain(
        torch.tensor(arenas, dtype=torch.long, device="cuda"),
        torch.tensor(depths, dtype=torch.long, device="cuda"),
        slot_mapping,
        [cache],
        C,
    )

    assert torch.equal(
        cache[0, _phys(slot_mapping, canon_slot)[0]],
        torch.full((H, D), 5.0, device="cuda"),
    )
