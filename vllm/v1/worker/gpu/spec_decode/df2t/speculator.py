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
        # Naming scheme (WS-A): ``num_speculative_tokens`` is the drafter block
        # depth (the number of POSITIONS the drafter emits per step); the tree
        # budget quantities below are the ONLY drivers of the verify width /
        # decode shape / reservation, never ``num_speculative_tokens``.
        self.drafter_block_depth = self.num_speculative_steps
        self.tree_decode_width = d.tree_decode_width
        self.tree_lookahead = d.tree_lookahead
        # Static budget this cut: nodes_max == the s=1.0 policy value, so the
        # no-head path matches d.get_num_draft_tokens() exactly.
        self.nodes_max = self.budget.resolve_nodes_max()
        self._last_tree: TreeDraft | None = None
        # Row count the most recent tree build wrote into the persistent
        # TreeBuilder buffers (eager build or a replayed drafter graph). Used
        # to fail closed if a consumer ever asks for more rows than were built.
        self._last_built_rows = 0
        self._tree_enabled = d.use_df2t_tree()
        if not self._tree_enabled:
            # ``enable_tree=False``: serve the DFlash2 greedy CHAIN from the
            # SAME drafter (``super()._generate_draft``), reproducing the dflash
            # chain path exactly. The base DFlash2 buffers (draft_tokens sized
            # to ``num_speculative_steps``, the candidate walk) stay untouched;
            # the tree-only machinery (TreeBuilder, the tree-width draft carry,
            # the arena->depth context) is skipped.
            self.tree_builder = None
            return
        self.tree_builder = TreeBuilder(
            self.max_num_reqs,
            self.max_depth,
            self.top_k,
            self.nodes_max,
            device,
            num_steps=self.num_speculative_steps,
            num_warps=d.tree_builder_num_warps,
            pad_token_id=self.parallel_drafting_token_id,
        )
        # The draft-token carry (scheduler drain slots) is the static tree
        # width wide (``tree_decode_width - 1 == nodes_max``), so a tree wider
        # than the drafter block depth (B > L) has slots to schedule; the base
        # class sized it to the drafter depth.
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.tree_decode_width - 1,
            dtype=torch.int64,
            device=device,
        )

    def _df2t_accepted_arenas(self) -> torch.Tensor | None:
        """Accepted-chain context for the next draft step.

        The tree verify places node ``arena`` at target row ``ctx_start + 1 +
        arena``, and the accepted chain is a scattered set of arenas
        (``TreeVerifier.accepted_arenas[r, d]``, one per chain depth ``d``).
        The drafter's next-step context must store those nodes at their
        canonical chain positions ``C + depth`` — what the KV compaction
        already relocated the accepted KV to — so the prep kernel maps each
        arena to its chain depth from this table. Returns ``None`` (the linear
        dflash prep) when the tree is disabled or no verifier is attached.
        """
        if not self._tree_enabled or self.tree_verifier is None:
            return None
        return self.tree_verifier.accepted_arenas

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
        the DFlash2 chain path (``method == "dflash"``, or
        ``method == "df2t"`` with ``enable_tree=False``) is untouched: with
        the tree disabled this delegates to the DFlash2 greedy CHAIN
        (``CandidateSampler`` walk) from the same drafter.
        """
        if not self._tree_enabled:
            return super()._generate_draft(
                num_reqs,
                num_tokens_padded,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp,
                cudagraph_runtime_mode,
            )
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
        assert self.tree_builder is not None
        tree = self.tree_builder.build(scores, candidate_ids)
        assert tree.num_nodes is not None
        self._last_tree = tree
        self._last_built_rows = num_reqs

        # The scheduler reserves tree_decode_width - 1 draft slots per request
        # (the static tree width). Fill them with the tree node token ids (emit
        # order); the builder already pads columns past a tree's node count
        # with the parallel-drafting mask token. The per-request width B_i
        # comes from the budget policy (no confidence head this cut: c_t =
        # None -> B_i = nodes_max).
        width = self.budget.shape(None)[1]
        # The scheduler drains num_draft_rows_per_req == nodes_max slots
        # unconditionally, so a per-request width below the carry width would
        # leave the previous step's branches stale in the un-overwritten
        # columns — fail loudly now, before the confidence-head per-request
        # width can hit it. The write below overwrites the whole carry row.
        assert width == self.draft_tokens.shape[1], (
            "df2t per-request verify width must fill the full draft-carry "
            f"width (got {width} != {self.draft_tokens.shape[1]})"
        )
        self.draft_tokens[:num_reqs] = tree.token_ids[:num_reqs, :width]

    def get_latest_tree_drafts(self, num_reqs: int) -> "list[TreeDraft | None] | None":
        """Per-request ``TreeDraft`` for the current batch, or None pre-first-step.

        Reads the tree straight from the persistent ``TreeBuilder`` buffers,
        which every step's tree build overwrites — whether that build ran
        eagerly or inside the captured drafter CUDA graph on replay. The
        runner scouts the list to ``num_reqs`` so it stays order-parallel to
        the batch's ``req_ids`` even when a captured drafter graph was sized
        for more (padded) requests; ignoring ``self._last_tree``'s own width
        also avoids returning stale per-descriptor slices after capture. Only
        df2t speculators expose this; other speculators keep
        ``set_draft_tokens`` on the plain draft-token path via None. Always
        ``None`` when ``enable_tree=False``: the chain carries plain draft
        tokens, so the runner takes the plain rejection-sampler path.

        Args:
            num_reqs: Number of requests in the batch being scheduled. Must be
                <= the number of rows the tree buffers were last written for.

        Raises:
            RuntimeError: if ``num_reqs`` exceeds the rows the most recent tree
                build wrote (a stale-buffer over-read that would hand out
                leftover trees from an older batch).

        """
        if not self._tree_enabled or self._last_tree is None:
            return None
        buffers = self._latest_tree_buffers(num_reqs)
        if buffers is None:
            return None
        token_ids, depths, parents, arena, words, num_nodes = buffers
        return [
            TreeDraft(
                token_ids=token_ids[i],
                depths=depths[i],
                parents=parents[i],
                arena=arena[i],
                words=words[i],
                num_nodes=num_nodes[i : i + 1],
            )
            for i in range(num_reqs)
        ]

    def _latest_tree_buffers(self, num_reqs: int):
        """Guarded builder views for the last ``num_reqs`` rows.

        Shared by :meth:`get_latest_tree_drafts` (per-request ``TreeDraft``)
        and :meth:`get_latest_tree_packed` (the packed batch), so both see the
        same fail-closed over-read guard. The builder writes every column past
        a tree's live node count inert (mask token, depth 0, parent -2, zero
        ancestor words), so no stale node from an earlier, wider build leaks.
        """
        assert self.tree_builder is not None
        if num_reqs > self._last_built_rows:
            raise RuntimeError(
                "df2t tree read would over-read the last tree build: "
                f"{num_reqs} requests but only {self._last_built_rows} rows "
                "were built. The static path builds exactly one tree per "
                "batch; a mismatch means draft/tree timing diverged."
            )
        b = self.tree_builder
        return (
            b.token_ids[:num_reqs],
            b.depths[:num_reqs],
            b.parents[:num_reqs],
            b.arena[:num_reqs],
            b.words[:num_reqs],
            b.num_nodes[:num_reqs],
        )

    def get_latest_tree_packed(self, num_reqs: int) -> torch.Tensor | None:
        """The batch's trees as packed ``[num_reqs, 4 * nodes_max + 1]`` rows
        (tokens | depths | parents | words | num_nodes), or ``None``
        pre-first-build or with the tree disabled."""
        if not self._tree_enabled or self._last_tree is None:
            return None
        self._latest_tree_buffers(num_reqs)
        assert self.tree_builder is not None
        return self.tree_builder.packed[:num_reqs]

    def _mark_draft_graph_built(self, rows: int) -> None:
        """Record the row count a replayed drafter graph just wrote."""
        if not self._tree_enabled:
            # Tree-off: the chain never supplies trees; leave the build-wide
            # bookkeeping inert so ``get_latest_tree_drafts`` stays None.
            return
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
