# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DF2T tree speculator: builds a top-B prefix tree instead of a walk."""

from typing import Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.df2t.budget import BudgetPolicy
from vllm.v1.worker.gpu.spec_decode.df2t.tree_builder import TreeBuilder
from vllm.v1.worker.gpu.spec_decode.df2t.tree_metadata import TreeDraft
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
        self.budget = BudgetPolicy.from_config(d)
        # Static budget this cut: nodes_max == the s=1.0 policy value, so the
        # no-head path matches d.get_num_draft_tokens() exactly.
        self.nodes_max = self.budget.resolve_nodes_max()
        self._last_tree: TreeDraft | None = None
        self.tree_builder = TreeBuilder(
            self.max_num_reqs, self.max_depth, self.top_k, self.nodes_max, device
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        """Build the per-request top-B prefix tree instead of the greedy walk.

        Reuses the DFlash2 draft forward and the candidate transition scores
        (``_score_candidates`` returns [R, L, K, K] predecessor->successor
        scores, exactly the TreeBuilder input), then materializes the tree's
        node token ids into ``draft_tokens`` (emit/arena order, padded to the
        per-step draft width) and stashes the batched tree for the scheduler
        carry. The tree is re-built every step from the fresh draft scores, so
        the DFlash2 chain path (``method == "dflash"``) is untouched.
        """
        last_hidden_states = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        num_sample = num_reqs * self.num_speculative_steps
        hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(
            num_reqs, self.num_speculative_steps, -1
        )
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden_states.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(
            num_reqs, self.num_speculative_steps, self.top_k
        )
        unary_logits = unary_logits.view_as(candidate_ids)
        scores = self._score_candidates(candidate_ids, unary_logits, hidden_states)
        tree = self.tree_builder.build(scores, candidate_ids)
        assert tree.num_nodes is not None
        self._last_tree = tree

        # The scheduler reserves num_speculative_steps draft slots per request.
        # Fill them with the highest-priority tree node token ids (emit order)
        # and pad with the parallel-drafting mask token where a request's tree
        # is shorter than the slot width.
        width = self.num_speculative_steps
        draft = tree.token_ids[:num_reqs, :width].clone()
        pad = torch.arange(width, dtype=torch.int64, device=self.device).unsqueeze(
            0
        ) >= tree.num_nodes[:num_reqs].to(torch.int64).unsqueeze(-1)
        draft.masked_fill_(pad, self.parallel_drafting_token_id)
        self.draft_tokens[:num_reqs] = draft

    @property
    def latest_tree_drafts(self) -> "list[TreeDraft | None] | None":
        """Per-request ``TreeDraft`` (order-parallel to the batch's ``req_ids``)
        from the most recent ``_generate_draft``, or None before the first step.
        Only df2t speculators expose this; other speculators keep
        ``set_draft_tokens`` on the plain draft-token path via None."""
        if self._last_tree is None:
            return None
        t = self._last_tree
        assert t.num_nodes is not None
        return [
            TreeDraft(
                token_ids=t.token_ids[i],
                depths=t.depths[i],
                parents=t.parents[i],
                arena=t.arena[i],
                words=t.words[i],
                num_nodes=t.num_nodes[i : i + 1],
            )
            for i in range(t.token_ids.shape[0])
        ]
