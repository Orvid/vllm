# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DF2T tree speculator: builds a top-B prefix tree instead of a walk."""

from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
    DFlash2Speculator,
)


class DF2TSpeculator(DFlash2Speculator):
    _speculator_name = "DF2T"

    def __init__(self, vllm_config, device):
        super().__init__(vllm_config, device)
        d = vllm_config.speculative_config
        self.max_depth = d.max_tree_depth
        self.tree_budget = d.tree_budget
        # Static budget this cut: nodes_max == the s=1.0 policy value.
        # Task 12 replaces this with resolve_nodes_max/ BudgetPolicy.
        self.nodes_max = d.get_num_draft_tokens()
