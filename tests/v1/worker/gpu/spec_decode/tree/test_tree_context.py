# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model-agnostic tree verify context: row parents and committed paths."""

import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.tree.context import TreeVerifyContext
from vllm.v1.worker.gpu.spec_decode.tree.mixers import TreeRecurrentLayers

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEVICE = torch.device("cuda")


def _batch(ctx: TreeVerifyContext) -> torch.Tensor:
    """Request 0: a 3-row prefill chunk. Request 1: a tree of 4 nodes, node 0
    and 2 under the anchor, node 1 under 0, node 3 under 1."""
    qsl = torch.tensor([0, 3, 8], dtype=torch.int32, device=DEVICE)
    parents = torch.tensor(
        [[-2, -2, -2, -2], [-1, 0, -1, 1]], dtype=torch.int64, device=DEVICE
    )
    widths = torch.tensor([0, 4], dtype=torch.int64, device=DEVICE)
    ctx.fill(qsl, 2, parents, widths)
    return qsl


def test_row_parents_chain_rows_and_tree_rows():
    """Chain rows continue the previous row (the first continues the
    committed state); tree nodes continue their parent node's row, and the
    anchor's children continue the anchor row."""
    ctx = TreeVerifyContext(16, 4, max_depth=3, device=DEVICE)
    _batch(ctx)
    #  rows:            0   1  2 | 3(anchor) 4(n0) 5(n1) 6(n2) 7(n3)
    expected = [-1, 0, 1, -1, 3, 4, 3, 5]
    assert ctx.row_parent[:8].tolist() == expected
    assert ctx.tree_width[:2].tolist() == [0, 4]


def test_commit_lists_anchor_then_accepted_nodes():
    """The committed rows of a tree request are its anchor then the accepted
    nodes shallowest first; a request without a tree commits nothing."""
    ctx = TreeVerifyContext(16, 4, max_depth=3, device=DEVICE)
    qsl = _batch(ctx)
    accepted = torch.full((2, 5), -1, dtype=torch.int64, device=DEVICE)
    accepted[1, 1] = 0  # node 0 at depth 1
    accepted[1, 2] = 1  # node 1 at depth 2
    commit = ctx.build_commit(accepted, qsl)
    assert commit.num_rows.tolist() == [0, 3]
    assert commit.cu_rows.tolist() == [0, 0, 3]
    assert commit.rows[:3].tolist() == [3, 4, 5]


class _Recurrent:
    def __init__(self):
        self.stash = None
        self.commits = 0

    def tree_stash_spec(self):
        return 8, torch.bfloat16

    def bind_tree_stash(self, stash):
        self.stash = stash

    def tree_commit(self, commit, attn_metadata):
        self.commits += 1


def test_recurrent_layers_get_stashes_and_commits():
    layer = _Recurrent()
    layers = TreeRecurrentLayers({"l0": layer}, max_rows=32, device=DEVICE)
    assert layer.stash.shape == (32, 8) and layer.stash.dtype == torch.bfloat16
    layers.commit(None, None)  # type: ignore[arg-type]
    assert layer.commits == 1


def test_stateful_layer_without_tree_support_fails_closed():
    """A stateful mixer that cannot verify a tree would scan sibling branches
    as one sequence; refuse to serve rather than return wrong tokens."""
    with pytest.raises(ValueError, match="TreeRecurrentLayer"):
        TreeRecurrentLayers({"l0": object()}, max_rows=4, device=DEVICE)
