# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tree selection methods: a drafter's proposal -> best-first prefix tree.

A selector queries the drafter's :class:`DraftProposal` for the candidates and
scores it needs, ranks them, and writes each request's tree into persistent
buffers: one packed ``[R, 4 * B + 1]`` row (tokens | depths | parents | words |
num_nodes, inert past the node count) with named strided views, plus each
node's path log-probability (``logp``) for adaptive verify width. Nodes are
emitted best-first (parents before children), so every prefix of a tree is a
tree.
"""

import torch

from vllm.config import SpeculativeConfig
from vllm.v1.worker.gpu.spec_decode.tree.selection.df2t import TreeBuilder


def make_tree_selector(
    config: SpeculativeConfig,
    *,
    max_num_reqs: int,
    num_candidates: int,
    num_steps: int,
    nodes_max: int,
    pad_token_id: int,
    device: torch.device,
) -> TreeBuilder:
    if config.tree_selection == "df2t":
        assert config.max_tree_depth is not None
        return TreeBuilder(
            max_num_reqs,
            config.max_tree_depth,
            num_candidates,
            nodes_max,
            device,
            num_steps=num_steps,
            num_warps=config.tree_builder_num_warps,
            pad_token_id=pad_token_id,
        )
    raise ValueError(f"unknown tree selection method {config.tree_selection!r}")
