# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DF2T tree speculator: builds a top-B prefix tree instead of a walk."""

from typing import Any

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.input_batch import InputBatch
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
        # Row count the most recent tree build wrote into the persistent
        # TreeBuilder buffers (eager build or a replayed drafter graph). Used
        # to fail closed if a consumer ever asks for more rows than were built.
        self._last_built_rows = 0
        self.tree_builder = TreeBuilder(
            self.max_num_reqs, self.max_depth, self.top_k, self.nodes_max, device
        )
        # Arena -> accepted-chain depth table for the drafter's next-step
        # context (row ``r`` column ``arena`` = chain depth of the accepted
        # node at that arena, 0 when not accepted). Filled by
        # ``_df2t_accepted_arena_depth`` from the runner's TreeVerifier.
        self._df2t_depth = torch.zeros(
            self.max_num_reqs, self.nodes_max, dtype=torch.int64, device=device
        )

    def _df2t_accepted_arena_depth(
        self, input_batch: InputBatch, num_sampled: torch.Tensor
    ) -> torch.Tensor | None:
        """Accepted-chain arena->depth context for the next draft step.

        The tree verify places node ``arena`` at target row ``ctx_start + 1 +
        arena``, and the accepted chain is a scattered set of arenas
        (``TreeVerifier.accepted_arenas[r, d]``, one per chain depth ``d``).
        The drafter's next-step context must store those nodes at their
        canonical chain positions ``C + depth`` — what the KV compaction
        already relocated the accepted KV to — so ``accepted_depth[r, arena]``
        maps an arena to its chain depth. A request with no accepted chain
        leaves its row all-zero, and the prep kernel falls back to the plain
        (linear) context those requests use.
        """
        tv = self.tree_verifier
        if tv is None:
            return None
        num_reqs = input_batch.num_reqs
        w = self.nodes_max
        acc = tv.accepted_arenas[:num_reqs, 1 : 1 + w]
        # Fixed-shape arena->depth build: column ``d`` of ``accepted_arenas``
        # holds the arena accepted at chain depth ``d + 1`` (-1 = none), so
        # ``depth_vals`` is the chain depth and the -1 paddings reduce to 0 as
        # the max over their column, writing nothing except the all-zero guard
        # row. Fixed shapes keep this a device-local op (no data-dependent
        # ``nonzero``/host sync) on the per-step hot path.
        valid = acc >= 0
        depth_vals = torch.arange(w, device=self.device) + 1
        self._df2t_depth.zero_()
        self._df2t_depth[:num_reqs].scatter_reduce_(
            1, acc.clamp(min=0), torch.where(valid, depth_vals, 0), reduce="amax"
        )
        return self._df2t_depth

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
        self._last_built_rows = num_reqs

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

    def get_latest_tree_drafts(
        self, num_reqs: int
    ) -> "list[TreeDraft | None] | None":
        """Per-request ``TreeDraft`` for the current batch, or None pre-first-step.

        Reads the tree straight from the persistent ``TreeBuilder`` buffers,
        which every step's tree build overwrites — whether that build ran
        eagerly or inside the captured drafter CUDA graph on replay. The
        runner scouts the list to ``num_reqs`` so it stays order-parallel to
        the batch's ``req_ids`` even when a captured drafter graph was sized
        for more (padded) requests; ignoring ``self._last_tree``'s own width
        also avoids returning stale per-descriptor slices after capture. Only
        df2t speculators expose this; other speculators keep
        ``set_draft_tokens`` on the plain draft-token path via None.

        Args:
            num_reqs: Number of requests in the batch being scheduled. Must be
                <= the number of rows the tree buffers were last written for.

        Raises:
            RuntimeError: if ``num_reqs`` exceeds the rows the most recent tree
                build wrote (a stale-buffer over-read that would hand out
                leftover trees from an older batch).

        """
        if self._last_tree is None:
            return None
        if num_reqs > self._last_built_rows:
            raise RuntimeError(
                "df2t tree read would over-read the last tree build: "
                f"{num_reqs} requests but only {self._last_built_rows} rows "
                "were built. The static path builds exactly one tree per "
                "batch; a mismatch means draft/tree timing diverged."
            )
        b = self.tree_builder
        return [
            TreeDraft(
                token_ids=b.token_ids[i],
                depths=b.depths[i],
                parents=b.parents[i],
                arena=b.arena[i],
                words=b.words[i],
                num_nodes=b.num_nodes[i : i + 1],
            )
            for i in range(num_reqs)
        ]

    def _mark_draft_graph_built(self, rows: int) -> None:
        """Record the row count a replayed drafter graph just wrote."""
        self._last_built_rows = rows

    @property
    def latest_tree_drafts(self) -> "list[TreeDraft | None] | None":
        """Back-compat alias over the last drafted row count (tests / eager).

        The runtime seam calls :meth:`get_latest_tree_drafts` with the current
        batch's ``num_reqs``; this property is kept for callers without batch
        context (unit tests), scoped to the last build's own width.
        """
        if self._last_tree is None:
            return None
        return self.get_latest_tree_drafts(self._last_tree.token_ids.shape[0])
