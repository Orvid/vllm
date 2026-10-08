# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tree rows' KV index lists for index-list (sparse / sliding-window)
attention."""

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.tree.context import TreeVerifyContext
from vllm.v1.worker.gpu.spec_decode.tree.indexed import fill_tree_window_indices

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
