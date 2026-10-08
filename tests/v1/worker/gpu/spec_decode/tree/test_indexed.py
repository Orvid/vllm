# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tree rows' KV index lists for index-list (sparse / sliding-window)
attention."""

import pytest
import torch

from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    _fused_kv_compress_norm_rope_insert_indexer_attn,
)
from vllm.models.deepseek_v4.common.ops.tree_compress import (
    tree_compress_norm_rope_store,
)
from vllm.v1.worker.gpu.spec_decode.tree.context import TreeVerifyContext
from vllm.v1.worker.gpu.spec_decode.tree.indexed import (
    apply_tree_compressed_slots,
    fill_tree_window_indices,
    remap_tree_compressed_indices,
    tree_indexer_logits,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEVICE = torch.device("cuda")


def test_tree_rows_see_their_context_window_and_path_not_siblings():
    """Request 1 verifies a tree under an anchor at position 10: node 0 and
    node 2 under the anchor, node 1 under node 0, node 3 under node 1. With a
    window of 4 positions, node 3 (position 13) sees positions 10 (the anchor,
    context) and its path 11, 12, 13; node 2 (position 11) sees 8, 9, 10 and
    itself, never node 0, its sibling at the same position."""
    ctx = TreeVerifyContext(16, 4, max_depth=3, device=DEVICE)
    qsl = torch.tensor([0, 2, 7], dtype=torch.int32, device=DEVICE)
    parents = torch.tensor(
        [[-2, -2, -2, -2], [-1, 0, -1, 1]], dtype=torch.int64, device=DEVICE
    )
    ctx.fill(qsl, 2, parents, torch.tensor([0, 4], device=DEVICE))
    # Request 0: 2 decode rows after 6 committed tokens; request 1: anchor at
    # linear position 10, nodes at linear slots 11..14.
    seq_lens = torch.tensor([8, 15], dtype=torch.int32, device=DEVICE)
    token_to_req = torch.tensor([0, 0, 1, 1, 1, 1, 1], dtype=torch.int32, device=DEVICE)
    block_size = 4
    # Request 1's blocks 0..3 are physical blocks 7, 3, 9, 5.
    block_table = torch.tensor(
        [[0, 1, 2, 0], [7, 3, 9, 5]], dtype=torch.int32, device=DEVICE
    )

    def slot(req, p):
        return int(block_table[req, p // block_size]) * block_size + p % block_size

    slot_mapping = torch.tensor(
        [slot(0, 6), slot(0, 7)] + [slot(1, 10 + j) for j in range(5)],
        dtype=torch.int64,
        device=DEVICE,
    )
    window = 4
    indices = torch.full((7, 1, window), -7, dtype=torch.int32, device=DEVICE)
    lens = torch.full((7,), -7, dtype=torch.int32, device=DEVICE)
    fill_tree_window_indices(
        indices,
        lens,
        ctx,
        7,
        slot_mapping,
        qsl,
        seq_lens,
        token_to_req,
        block_table,
        block_size,
        window,
        torch.zeros(2, dtype=torch.int32, device=DEVICE),
    )
    # Chain rows and the anchor (row 2) are left as the causal kernel wrote
    # them.
    assert lens[:3].tolist() == [-7, -7, -7]
    # Node rows 3..6 hold nodes 0..3 at linear slot positions 11..14.
    expected = {
        3: [8, 9, 10, 11],  # node 0, depth 1 (position 11)
        4: [9, 10, 11, 12],  # node 1, depth 2: context 9, 10 + node 0, itself
        5: [8, 9, 10, 13],  # node 2, depth 1: never node 0 (position 11)
        6: [10, 11, 12, 14],  # node 3, depth 3: anchor + nodes 0, 1, itself
    }
    for row, positions in expected.items():
        assert lens[row].item() == len(positions)
        got = indices[row, 0, : len(positions)].tolist()
        assert got == [slot(1, p) for p in positions], row


# Compressed entries (ratio 2) of request 1's tree above: anchor at position
# 10, node 0 (row 3, position 11) and node 2 (row 5, position 11) complete an
# entry at depth 1, node 3 (row 6, position 13) at depth 3; node 1 (row 4,
# position 12) completes none. With at most 4 nodes the scratch entries start
# at (10 + 1 + 4) // 2 + 1 = 8, one per node.
RATIO = 2
MAX_NODES = 4
ENTRY_BLOCK = 4


def _ratio2_tree():
    ctx = TreeVerifyContext(16, 4, max_depth=3, device=DEVICE)
    qsl = torch.tensor([0, 2, 7], dtype=torch.int32, device=DEVICE)
    parents = torch.tensor(
        [[-2, -2, -2, -2], [-1, 0, -1, 1]], dtype=torch.int64, device=DEVICE
    )
    ctx.fill(qsl, 2, parents, torch.tensor([0, 4], device=DEVICE))
    positions = torch.tensor([6, 7, 10, 11, 12, 11, 13], device=DEVICE)
    token_to_req = torch.tensor([0, 0, 1, 1, 1, 1, 1], dtype=torch.int32, device=DEVICE)
    # Request 1's entry blocks 0..3 are physical blocks 7, 3, 9, 5.
    entry_table = torch.tensor(
        [[0, 1, 2, 0], [7, 3, 9, 5]], dtype=torch.int32, device=DEVICE
    )
    return ctx, positions, token_to_req, entry_table


def _entry_slot(entry_table, req, entry):
    block = int(entry_table[req, entry // ENTRY_BLOCK])
    return block * ENTRY_BLOCK + entry % ENTRY_BLOCK


@pytest.mark.parametrize("scratch", [True, False])
def test_tree_rows_write_compressed_entries_only_where_unshared(scratch):
    """Siblings completing an entry at the same position must not share it.
    With scratch, each boundary node writes its own scratch entry and other
    tree rows none. Without (a ratio verified as a chain whenever an entry
    would complete inside a tree), only rows at their linear position (every
    row of a chain) write, positionally. Chain rows always keep theirs."""
    ctx, positions, token_to_req, entry_table = _ratio2_tree()
    slots = torch.arange(100, 107, device=DEVICE)
    apply_tree_compressed_slots(
        slots,
        ctx,
        7,
        positions,
        token_to_req,
        entry_table,
        ENTRY_BLOCK,
        RATIO,
        MAX_NODES,
        scratch,
    )
    if scratch:
        e = [_entry_slot(entry_table, 1, n) for n in (8, 10, 11)]
        assert slots.tolist() == [100, 101, 102, e[0], -1, e[1], e[2]]
    else:
        # Nodes 0 and 1 sit at the arena slots their depths give; 2, 3 not.
        assert slots.tolist() == [100, 101, 102, 103, 104, -1, -1]


def test_tree_rows_select_their_paths_in_tree_entries():
    """Local index n_ctx + k of a tree row (n_ctx = 11 // 2 = 5 context
    entries) is the k-th entry completed along its own path."""
    ctx, positions, token_to_req, entry_table = _ratio2_tree()
    local = torch.tensor(
        [
            [0, 3, -1],
            [0, 1, 2],
            [0, 4, -1],
            [5, 0, -1],
            [5, 4, 0],
            [5, 2, -1],
            [6, 5, 4],
        ],
        dtype=torch.int32,
        device=DEVICE,
    )
    global_indices = local + 1000
    remap_tree_compressed_indices(
        global_indices,
        local,
        ctx,
        7,
        positions,
        token_to_req,
        entry_table,
        ENTRY_BLOCK,
        RATIO,
        MAX_NODES,
    )
    node0, node2, node3 = (_entry_slot(entry_table, 1, n) for n in (8, 10, 11))
    expected = (local + 1000).tolist()
    expected[3][0] = node0  # node 0: its own entry
    expected[4][0] = node0  # node 1: node 0's
    expected[5][0] = node2  # node 2: its own, not node 0's
    expected[6][:2] = [node3, node0]  # node 3: k=1 its own, k=0 node 0's
    assert global_indices.tolist() == expected


def test_tree_indexer_scores_path_scratch_entries():
    """The in-tree logit columns of a tree row score its path's scratch
    entries with the paged kernel's formula; other columns stay."""
    ctx, positions, token_to_req, entry_table = _ratio2_tree()
    heads, head_dim, num_blocks = 8, 128, 10
    torch.manual_seed(0)
    values = (torch.randn(num_blocks, ENTRY_BLOCK, head_dim, device=DEVICE) * 4).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand(num_blocks, ENTRY_BLOCK, device=DEVICE) + 0.5
    # Per block: all entries' FP8 values, then their fp32 scales.
    kv_cache = torch.empty(
        num_blocks, ENTRY_BLOCK, head_dim + 4, dtype=torch.uint8, device=DEVICE
    )
    flat = kv_cache.view(num_blocks, -1)
    flat[:, : ENTRY_BLOCK * head_dim] = values.view(torch.uint8).flatten(1)
    flat[:, ENTRY_BLOCK * head_dim :] = scales.view(torch.uint8)
    q = (torch.randn(7, heads, head_dim, device=DEVICE) * 2).to(torch.float8_e4m3fn)
    weights = torch.randn(7, heads, device=DEVICE)
    logits = torch.zeros(7, 16, device=DEVICE)
    tree_indexer_logits(
        logits,
        q,
        weights,
        kv_cache,
        ctx,
        7,
        positions,
        token_to_req,
        entry_table,
        RATIO,
        MAX_NODES,
    )

    def score(row, entry):
        slot = _entry_slot(entry_table, 1, entry)
        k = values.flatten(0, 1)[slot].float()
        s = (q[row].float() @ k).relu()
        return float((s * weights[row]).sum() * scales.flatten()[slot])

    expected = torch.zeros(7, 16)
    expected[3, 5] = score(3, 8)
    expected[4, 5] = score(4, 8)
    expected[5, 5] = score(5, 10)
    expected[6, 5] = score(6, 8)
    expected[6, 6] = score(6, 11)
    torch.testing.assert_close(logits.cpu(), expected, rtol=1e-4, atol=1e-3)


def test_tree_compressor_pools_each_boundary_rows_own_path():
    """A tree boundary row's compressed entry is the one a chain made of its
    path gets: ratio 4 with overlap pools positions 4..11 for both branch ends
    at position 11 (anchor at 9) from the context plus their own path's
    states, never the sibling branch's."""
    head, rope, ratio, state_block = 128, 64, 4, 4
    ctx = TreeVerifyContext(8, 1, max_depth=2, device=DEVICE)
    qsl = torch.tensor([0, 5], dtype=torch.int32, device=DEVICE)
    # Branches node 0 -> node 1 and node 2 -> node 3 under the anchor.
    parents = torch.tensor([[-1, 0, -1, 2]], dtype=torch.int64, device=DEVICE)
    ctx.fill(qsl, 1, parents, torch.tensor([4], device=DEVICE))
    positions = torch.tensor([9, 10, 11, 10, 11], device=DEVICE)
    token_to_req = torch.zeros(5, dtype=torch.int32, device=DEVICE)
    # Linear state slots: anchor at 9, nodes at 10..13; blocks permuted.
    state_table = torch.tensor([[3, 0, 2, 1]], dtype=torch.int32, device=DEVICE)

    def state_slot(p):
        return int(state_table[0, p // state_block]) * state_block + p % state_block

    state_slots = torch.tensor([state_slot(9 + i) for i in range(5)], device=DEVICE)
    torch.manual_seed(0)
    state_cache = torch.randn(4, state_block, 4 * head, device=DEVICE).bfloat16()
    rms_weight = torch.randn(head, device=DEVICE).bfloat16()
    cos_sin = torch.randn(16, rope, device=DEVICE)
    kv_cache = torch.zeros(2, 4, head + 4, dtype=torch.uint8, device=DEVICE)
    # Nodes 1 and 3 write entries 0 and 1 of block 1.
    kv_slots = torch.tensor([-1, -1, 4, -1, 5], device=DEVICE)
    tree_compress_norm_rope_store(
        ctx,
        5,
        state_cache,
        positions,
        state_slots,
        token_to_req,
        state_table,
        state_block,
        2 * head,
        cos_sin,
        kv_cache,
        kv_slots,
        head_dim=head,
        rope_head_dim=rope,
        compress_ratio=ratio,
        overlap=True,
        use_fp4_cache=False,
        rms_norm_weight=rms_weight,
        rms_norm_eps=1e-6,
        quant_block=head,
        token_stride=head,
        scale_dim=4,
    )

    def entry(cache, i):
        block, j = cache[i // 4].flatten(), i % 4
        values = block[j * head : (j + 1) * head]
        scale = block[4 * head + 4 * j : 4 * head + 4 * (j + 1)]
        return torch.cat([values, scale])

    def chain_entry(path_slots):
        """The regular kernel's entry for position 11 of a chain whose
        positions 10, 11 hold the states at ``path_slots``."""
        chain = state_cache.clone()
        rows, src = chain.view(-1, 4 * head), state_cache.view(-1, 4 * head)
        for p, s in zip((10, 11), path_slots):
            rows[state_slot(p)] = src[s]
        out = torch.zeros_like(kv_cache)
        zero = torch.zeros(1, dtype=torch.int64, device=DEVICE)
        _fused_kv_compress_norm_rope_insert_indexer_attn[(1,)](
            chain,
            chain.stride(0),
            chain.stride(1),
            zero.int(),
            torch.tensor([11], device=DEVICE),
            zero,
            state_table,
            state_table.stride(0),
            state_block,
            rms_weight,
            1e-6,
            cos_sin,
            cos_sin.stride(0),
            out,
            zero,
            4,
            HEAD_SIZE=head,
            TRITON_BLOCK_SIZE=head,
            STATE_WIDTH=2 * head,
            COMPRESS_RATIO=ratio,
            OVERLAP=1,
            ROPE_HEAD_DIM=rope,
            FP8_MAX=448.0,
            QUANT_BLOCK=head,
            TOKEN_STRIDE=head,
            SCALE_DIM=4,
            KV_BLOCK_STRIDE=out.stride(0),
            num_warps=1,
        )
        return entry(out, 0)

    branch_a = chain_entry([state_slot(10), state_slot(11)])
    branch_b = chain_entry([state_slot(12), state_slot(13)])
    assert not torch.equal(branch_a, branch_b)
    assert torch.equal(entry(kv_cache, 4), branch_a)
    assert torch.equal(entry(kv_cache, 5), branch_b)
